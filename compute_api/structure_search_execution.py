"""Structure-search attempts: managed, idempotent local preparation (USPEX step 4B).

``PUT /api/v1/structure-search-attempts/{attempt_id}`` prepares the frozen
search plan of a New structure request into a private directory under the
machine API's state root, and ``GET`` reads the attempt back. It is the
structure-search counterpart of calculation attempts
(``compute_api.execution``), in the same service, with the same tokens,
principal binding, ledger mechanisms and lock.

Layout under ``$BMD_API_STATE_DIR`` (all directories mode 0700)::

    attempts/<id>.json                      calculation attempts (unchanged, v1.2.0 layout)
    ledger.lock                             shared by both kinds
    structure_search/attempts/<id>.json     search attempt records (own schema)
    structure_search/preparations/<id>/preparation-<n>/   prepared inputs

Search records never enter ``attempts/``, so calculation scans, including
those of v1.2.0 after a rollback, see only calculation records; no migration
is needed in either direction. Attempt IDs are unique across both kinds:
each kind refuses, under the shared lock, an ID the other kind uses (409
``attempt_kind_mismatch``).

``PUT`` sequence:

1. An attempt already ``PREPARED_LOCALLY`` for the same principal and
   request is returned as stored (after checking its files), without
   re-planning, even if policy has changed since.
2. Otherwise (a new attempt, or one left ``PREPARING``) the request is
   planned with today's policy through the injected search planner
   (``backend.structure_search.planning.plan_structure_search``); a digest
   other than ``expected_plan_digest`` gives 409 ``plan_changed`` before
   anything is written.
3. Under the lock the record is created (state ``PREPARING``), or the
   existing record's principal and request are checked.
4. ``search_plan.prepare_search_directory`` writes the inputs, which are
   verified; the record becomes ``PREPARED_LOCALLY`` with the plan digest
   and the files' SHA-256.

A different request under the same ID gets 409 ``attempt_request_mismatch``,
another principal 403. A record left ``PREPARING`` is completed by the next
``PUT`` (after step 2): a preparation directory that verifies is adopted, a
missing one is prepared, and one that exists but does not verify is kept,
recorded as abandoned, and preparation continues in ``preparation-<n+1>``.
A damaged prepared attempt gives 409 ``preparation_damaged``. Nothing is
ever overwritten or deleted, and there is no automatic clean-up.

Local preparation does not know where the launch bridge will be on POWER:
``INPUT.txt`` names ``search_plan.UNBOUND_BRIDGE_PATH`` and the manifest
records the binding as unbound. Remote staging (step 4C), SLURM submission
and monitoring are not implemented. Nothing here opens SSH or contacts
POWER. Responses never include local paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from compute_api.canonical import canonical_json, sha256_hex
from compute_api.execution import ExecutionError, _authorize_record
from compute_api.ledger import (
    ATTEMPTS_DIRNAME,
    STRUCTURE_SEARCH_DIRNAME,
    AttemptLedger,
    LedgerUnavailable,
    isoformat,
    utc_now,
)


SEARCH_ATTEMPT_KIND = "structure_search"
SEARCH_RECORD_SCHEMA = "bmd_compute.api_structure_search_attempt_record"
SEARCH_RECORD_SCHEMA_VERSION = 1
STATE_PREPARING = "PREPARING"
STATE_PREPARED_LOCALLY = "PREPARED_LOCALLY"
STATE_STAGING = "STAGING"
STATE_STAGED = "STAGED"
STATE_PREPARED = "PREPARED"  # submission-ready (step 4D-2): last verified on POWER
REMOTE_STATE_STAGING = "STAGING"
REMOTE_STATE_STAGED = "STAGED"
STAGED_REMOTE_INTEGRITY = "last_verified"
# Finalization verifies the runtime package, the POTCAR identities and every
# staged input on the POWER login node in one command, and prints only when it
# is done. Ordinary remote commands keep the runner's default timeout (60 s).
FINALIZE_TIMEOUT_S = 300
TIMINGS_MARKER = "BMD_SEARCH_STAGING_TIMINGS="
LOGGER = logging.getLogger(__name__)  # same as backend.structure_search.staging
PREPARATIONS_DIRNAME = "preparations"
PREPARATION_PREFIX = "preparation-"
RESPONSE_SCHEMA = "bmd_compute.api_structure_search_attempt"
RESPONSE_SCHEMA_VERSION = 1
MAX_PREPARATIONS = 16


class SearchAttemptLedger(AttemptLedger):
    """The structure-search attempt ledger: ``<state root>/structure_search/attempts/``.

    It reuses the calculation ledger's validated reads, atomic create/replace
    and size limits, with its own record schema and directory, so calculation
    scans (including those of older releases) never see a search record. It
    takes the state root's ``ledger.lock``, shared with calculation attempts,
    so cross-kind ID checks and creation are atomic together.
    """

    RECORD_SCHEMA = SEARCH_RECORD_SCHEMA
    RECORD_SCHEMA_VERSION = SEARCH_RECORD_SCHEMA_VERSION

    def __init__(self, state_root: Path):
        super().__init__(state_root / STRUCTURE_SEARCH_DIRNAME)
        self.state_root = state_root
        self.calculations = AttemptLedger(state_root)  # read and lock only; never created here
        self.preparations_dir = self.root / PREPARATIONS_DIRNAME

    @classmethod
    def from_environment_read_only(cls) -> "SearchAttemptLedger":
        """Open without creating anything (reads of a missing directory find nothing)."""

        try:
            return cls(AttemptLedger.state_root_from_environment())
        except LedgerUnavailable:
            raise ExecutionError(503, "execution_not_configured",
                                 "Machine execution is not configured on this service.") from None

    def read(self, attempt_id: str):
        if not self.attempts_dir.is_dir():
            return None
        return super().read(attempt_id)

    def calculation_exists(self, attempt_id: str) -> bool:
        return os.path.lexists(self.calculations.attempts_dir / f"{attempt_id}.json")

    def locked(self):
        return self.calculations.locked()

    def ensure_directories(self) -> None:
        for path in (self.root, self.attempts_dir, self.preparations_dir):
            _private_directory(path)

    def _fsync_directory(self) -> None:  # the search attempts directory, not the calculation one
        if os.name != "posix":
            return
        try:
            fd = os.open(self.attempts_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)


def search_request_sha256(shape) -> str:
    """Identity of a structure-search attempt request (key order does not matter)."""

    return sha256_hex(
        canonical_json(
            {
                "attempt_kind": SEARCH_ATTEMPT_KIND,
                "formula": shape.formula,
                "search": dict(shape.search),
                "expected_plan_digest": shape.expected_plan_digest,
            }
        )
    )


def _private_directory(path: Path) -> None:
    """Create ``path`` (mode 0700) if missing; refuse a symlink or a shared directory."""

    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ExecutionError(503, "state_unsafe", "The structure-search state directory is not usable.")
    if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise ExecutionError(503, "state_unsafe", "The structure-search state directory is not private.")


def _attempt_root(ledger: SearchAttemptLedger, attempt_id: str) -> Path:
    root = ledger.preparations_dir / attempt_id
    _private_directory(root)
    return root


def _preparation_dir(ledger: SearchAttemptLedger, record) -> Path:
    return ledger.preparations_dir / record["attempt_id"] / record["preparation"]["directory"]


def _new_record(*, attempt_id, principal, request_digest, plan_digest) -> dict[str, Any]:
    now = isoformat(utc_now())
    return {
        "schema": SEARCH_RECORD_SCHEMA,
        "schema_version": SEARCH_RECORD_SCHEMA_VERSION,
        "attempt_kind": SEARCH_ATTEMPT_KIND,
        "attempt_id": attempt_id,
        "principal": principal,
        "created_at": now,
        "updated_at": now,
        "request_sha256": request_digest,
        "plan_digest": plan_digest,
        "state": STATE_PREPARING,
        "preparation": {
            "directory": f"{PREPARATION_PREFIX}1",
            "abandoned": [],
            "bridge_path_bound": False,
            "manifest_sha256": None,
            "files": None,
            "prepared_at": None,
        },
        "submission": {"status": "not_available"},
    }


def _calculation_id_conflict() -> ExecutionError:
    return ExecutionError(
        409, "attempt_kind_mismatch", "This attempt ID belongs to a different kind of attempt (a calculation)."
    )


def _check_binding(record: Mapping[str, Any], principal: str, request_digest: str) -> None:
    _authorize_record(record, principal)
    if record["request_sha256"] != request_digest:
        raise ExecutionError(409, "attempt_request_mismatch", "This attempt ID is already bound to a different request.")


def _verified(directory: Path, plan_digest: str) -> dict[str, Any] | None:
    """The manifest of a complete preparation of ``plan_digest`` in ``directory``, else None."""

    from backend.structure_search.search_plan import SearchPlanError, verify_prepared_directory

    if not os.path.lexists(directory):
        return None
    try:
        manifest = verify_prepared_directory(directory, require_current=False)
    except (SearchPlanError, OSError, ValueError):
        return None
    return manifest if manifest.get("plan_digest") == plan_digest else None


def _manifest_sha256(directory: Path) -> str:
    from backend.structure_search.search_plan import PREPARED_FILE

    return hashlib.sha256((directory / PREPARED_FILE).read_bytes()).hexdigest()


def _complete(ledger, record: dict[str, Any], plan) -> dict[str, Any]:
    """Finish a PREPARING attempt: adopt, prepare, or start a new preparation directory."""

    from backend.structure_search.search_plan import UNBOUND_BRIDGE_PATH, prepare_search_directory

    attempt_id = record["attempt_id"]
    ledger.ensure_directories()
    root = _attempt_root(ledger, attempt_id)
    preparation = dict(record["preparation"])
    abandoned = list(preparation.get("abandoned") or [])
    name = preparation["directory"]
    directory = root / name

    manifest = _verified(directory, record["plan_digest"])
    if manifest is None and os.path.lexists(directory):
        # Interrupted or damaged: keep it, never reuse it.
        abandoned.append(name)
        number = int(name.removeprefix(PREPARATION_PREFIX)) + 1
        if number > MAX_PREPARATIONS:
            raise ExecutionError(409, "preparation_failed", "This attempt could not be prepared; use a new attempt ID.")
        name = f"{PREPARATION_PREFIX}{number}"
        directory = root / name
        if os.path.lexists(directory):
            raise ExecutionError(409, "preparation_conflict", "The attempt's preparation state is inconsistent.")
        record["preparation"] = {**preparation, "directory": name, "abandoned": abandoned}
        record["updated_at"] = isoformat(utc_now())
        ledger.replace(record)
    if manifest is None:
        manifest = prepare_search_directory(plan, directory, bridge_path=UNBOUND_BRIDGE_PATH)

    record["state"] = STATE_PREPARED_LOCALLY
    record["preparation"] = {
        "directory": name,
        "abandoned": abandoned,
        "bridge_path_bound": bool((manifest.get("bindings") or {}).get("bound")),
        "manifest_sha256": _manifest_sha256(directory),
        "files": dict(manifest["files"]),
        "prepared_at": isoformat(utc_now()),
    }
    record["updated_at"] = record["preparation"]["prepared_at"]
    ledger.replace(record)
    return record


def search_attempt_response(record: Mapping[str, Any], *, integrity: str | None = None,
                            resources: Mapping[str, Any] | None = None) -> dict[str, Any]:
    preparation = record["preparation"]
    return {
        "schema": RESPONSE_SCHEMA,
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "attempt_id": record["attempt_id"],
        "attempt_kind": SEARCH_ATTEMPT_KIND,
        "plan_digest": record["plan_digest"],
        "state": record["state"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "preparation": {
            "location": "local: BMD Compute service state (not on POWER)",
            "files": preparation.get("files"),
            "manifest_sha256": preparation.get("manifest_sha256"),
            "bridge_path_bound": preparation.get("bridge_path_bound", False),
            "contains_potcar_contents": False,
            "interrupted_preparations": len(preparation.get("abandoned") or []),
            "integrity": integrity,
        },
        "remote_staging_available": record["state"] in {STATE_PREPARED_LOCALLY, STATE_STAGING},
        "staging": _staging_response(record.get("staging")),
        "resources": dict(resources) if resources is not None else None,
        "promotion_available": record["state"] == STATE_STAGED,
        "promotion": _promotion_response(record.get("promotion"), state=record["state"]),
        # PREPARED: verified and ready when last checked on POWER (step 4D-2).
        "submission_ready": record["state"] == STATE_PREPARED,
        "submission_available": record["state"] == STATE_PREPARED,
        "submission": _submission_response(record.get("submission"), state=record["state"],
                                           monitoring=record.get("monitoring")),
        # Step 4D-4: observations (POST …/observations) and cancellation (PUT …/cancellation).
        "observation_available": record["state"] in _OBSERVABLE_STATES and bool(
            (record.get("submission") or {}).get("job_id")),
        "monitoring": _monitoring_response(record.get("monitoring")),
        "cancellation_available": record["state"] in ("SUBMITTED", "CANCEL_REQUESTED") and not (
            record.get("submission") or {}).get("remote_record_pending"),
        "cancellation": _cancellation_response(record.get("cancellation")),
    }


_TERMINAL = ("SEARCH_COMPLETED", "SEARCH_FAILED", "SEARCH_CANCELLED", "SEARCH_TIMED_OUT")
_OBSERVABLE_STATES = ("SUBMITTED", "CANCEL_REQUESTED", *_TERMINAL)


def _monitoring_response(monitoring: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Observations of the job (step 4D-4): the last one, the recorded terminal state, contradictions."""

    if not monitoring:
        return None
    return {
        "observation_count": monitoring.get("observation_count"),
        "last": monitoring.get("last"),
        "terminal": monitoring.get("terminal"),
        "contradiction_count": monitoring.get("contradiction_count", 0),
        "contradictions": monitoring.get("contradictions") or [],
        "operator_review": bool(monitoring.get("contradiction_count")),
    }


def _cancellation_response(cancellation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Requested, command attempted and observed are reported separately (step 4D-4)."""

    if not cancellation:
        return None
    commands = list(cancellation.get("commands") or [])
    return {
        "status": cancellation.get("status"),
        # Repeat the request to resume (after a fresh identity check on POWER); nothing resumes by itself.
        "resume_by_repeating": cancellation.get("status") in (
            "remote_request_uncertain", "remote_recorded_command_pending", "command_outcome_unknown",
            "command_failed"),
        "job_id": cancellation.get("job_id"),
        "requested": {"at": cancellation.get("requested_at"), "by": cancellation.get("requested_by"),
                      "adopted_from_power": bool(cancellation.get("adopted_from_power"))},
        "command": {"count": cancellation.get("command_count", 0), "last": commands[-1] if commands else None,
                    "attempts": commands},
        "observed": cancellation.get("observed"),
        "last_check": cancellation.get("last_check"),
        "last_refusal": cancellation.get("last_refusal"),
    }


def _submission_response(submission: Mapping[str, Any] | None, *, state: str,
                         monitoring: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The SLURM submission (step 4D-3), with scheduler, execution and scientific validity kept apart."""

    if not submission:
        return None
    submitted = state in _OBSERVABLE_STATES
    last = (monitoring or {}).get("last")
    return {
        "status": submission.get("status"),
        "job_id": submission.get("job_id"),
        "job_name": submission.get("job_name"),
        "attempt_comment": submission.get("comment"),
        "requested_at": submission.get("requested_at"),
        "submitted_at": submission.get("submitted_at"),
        "reconciled": bool(submission.get("reconciled")),
        "adopted": bool(submission.get("adopted")),
        "remote_record_pending": bool(submission.get("remote_record_pending")),
        # PUT …/submission/remote-record completes it (verified against SLURM; never runs sbatch).
        "remote_record_recovery_available": submitted and bool(submission.get("remote_record_pending")),
        "remote_record_recovery": submission.get("remote_record_recovery"),
        "uncertain": state == "SUBMISSION_UNCERTAIN",
        "uncertain_reason": submission.get("uncertain_reason"),
        "last_error": submission.get("last_error"),
        "reconciliation": submission.get("reconciliation"),
        # Three separate answers, none inferred from another; reads never refresh them (observations do).
        "scheduler": (None if not submitted else {"state": None, "refreshed": False} if not last else {
            "state": last["scheduler"]["state"], "source": last["scheduler"]["source"], "refreshed": True,
            "observed_at": last["observed_at"], "observation_sequence": last["sequence"],
            "settled": last["scheduler"].get("settled")}),
        "execution": (None if not submitted else {"outcome": None, "read": False} if not last else {
            "outcome": last["execution"]["outcome"], "source": "search_status.json", "read": True,
            "sha256": last["execution"]["sha256"], "observed_at": last["observed_at"],
            "observation_sequence": last["sequence"]}),
        "scientific_validity": {"status": "not_assessed"},
        "finalists": {"status": "not_started"},
    }


def _promotion_response(promotion: Mapping[str, Any] | None, *, state: str) -> dict[str, Any] | None:
    if not promotion:
        return None
    acknowledgement = promotion["acknowledgement"]
    return {
        "remote_state": (promotion.get("result") or {}).get("remote_state"),
        "started_at": promotion.get("started_at"),
        "verified_at": promotion.get("verified_at"),
        # Like STAGED: what POWER last verified, not current integrity.
        "remote_integrity": STAGED_REMOTE_INTEGRITY if promotion.get("verified_at") else None,
        "job_script": dict(promotion["job_script"]),
        "acknowledgement": {
            "acknowledgement_sha256": acknowledgement["acknowledgement_sha256"],
            "resources_sha256": promotion["resources_sha256"],
            "principal": acknowledgement["principal"],
            "acknowledged_at": acknowledgement["acknowledged_at"],
        },
        "modules": list(promotion["modules"]),
        "runtime_check": (promotion.get("result") or {}).get("runtime_check"),
        "in_progress": state == STATE_STAGED and not promotion.get("verified_at"),
    }


def search_resources(ledger, record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The provisional resource request a user acknowledges, from the local frozen plan (local only)."""

    from backend.structure_search.job import resource_request_sha256, resource_request_view
    from backend.structure_search.search_plan import PLAN_FILE, load_plan_text

    if record["state"] not in (STATE_STAGED, STATE_PREPARED):
        return None
    plan = load_plan_text((_preparation_dir(ledger, record) / PLAN_FILE).read_text(encoding="utf-8"))
    return {**resource_request_view(plan), "resources_sha256": resource_request_sha256(plan)}


def _staging_response(staging: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not staging:
        return None
    result = staging.get("result") or {}
    return {
        "location": "POWER (search workspace)",
        "run_name": staging.get("run_name"),
        "remote_state": result.get("remote_state") or REMOTE_STATE_STAGING,
        "started_at": staging.get("started_at"),
        "staged_at": staging.get("staged_at"),
        # STAGED is what POWER last verified (at staged_at), not current integrity;
        # reading it back never contacts POWER.
        "remote_integrity": STAGED_REMOTE_INTEGRITY if staging.get("staged_at") else None,
        "last_verified_at": staging.get("staged_at"),
        "bridge_path_bound": bool(result.get("bridge_path_bound")),
        "bridge_sha256": staging.get("bridge_sha256"),
        "runtime_package_manifest_sha256": staging.get("runtime_manifest_digest"),
        "files": result.get("files"),
        "potcars": result.get("potcars"),
        "contains_potcar_contents": False,
    }


def _stored(ledger: SearchAttemptLedger, record: Mapping[str, Any]) -> dict[str, Any]:
    """The stored result of a prepared attempt, after checking its files are intact."""

    directory = _preparation_dir(ledger, record)
    intact = _verified(directory, record["plan_digest"]) is not None and (
        _manifest_sha256(directory) == record["preparation"]["manifest_sha256"]
    )
    if not intact:
        raise ExecutionError(
            409, "preparation_damaged",
            "This attempt's prepared files no longer match its record; they are not replaced. "
            "Use a new attempt ID.",
        )
    return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))


def prepare_search_attempt(*, search_planner, principal: str, attempt_id: str, shape) -> dict[str, Any]:
    """Prepare (or return the stored preparation of) one structure-search attempt.

    An attempt already ``PREPARED_LOCALLY`` for the same principal and request
    is returned as stored, without re-planning, even if policy has changed
    since. A new attempt, or one left ``PREPARING``, is planned with today's
    policy first, and refused if that no longer gives the approved digest;
    nothing is written before that check.
    """

    from backend.structure_search.planning import StructureSearchPlanRequest

    request_digest = search_request_sha256(shape)
    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is not None:
        _check_binding(record, principal, request_digest)
        if record["state"] != STATE_PREPARING:
            return _stored(ledger, record)

    result = search_planner(
        StructureSearchPlanRequest(
            formula=shape.formula,
            expected_plan_digest=shape.expected_plan_digest,
            **shape.search,
        )
    )  # a stale plan is refused here, before anything is written

    with ledger.locked():
        record = ledger.read(attempt_id)
        if record is None:
            if ledger.calculation_exists(attempt_id):
                raise _calculation_id_conflict()
            ledger.ensure_directories()
            record = _new_record(
                attempt_id=attempt_id,
                principal=principal,
                request_digest=request_digest,
                plan_digest=result.digest,
            )
            if not ledger.create(record):  # pragma: no cover - the lock excludes this
                raise ExecutionError(409, "attempt_conflict", "This attempt was created concurrently; retry.")
        else:
            _check_binding(record, principal, request_digest)
            if record["state"] != STATE_PREPARING:
                return _stored(ledger, record)
        if record["plan_digest"] != result.digest:  # pragma: no cover - implied by the request hash
            raise ExecutionError(409, "plan_changed", "This attempt was bound to a different plan.")
        record = _complete(ledger, record, result.plan)
    return search_attempt_response(record, integrity="verified")


def lookup_search_attempt(*, principal: str, attempt_id: str) -> dict[str, Any]:
    """Read a structure-search attempt back (no planning, no writing)."""

    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No structure-search attempt with this ID exists.")
    _authorize_record(record, principal)
    if record["state"] == STATE_PREPARING:
        return search_attempt_response(record, integrity="incomplete")
    directory = _preparation_dir(ledger, record)
    intact = _verified(directory, record["plan_digest"]) is not None and (
        _manifest_sha256(directory) == record["preparation"]["manifest_sha256"]
    )
    resources = search_resources(ledger, record) if intact else None
    return search_attempt_response(record, integrity="verified" if intact else "damaged", resources=resources)


# ---------------------------------------------------------------- remote staging (4C)


def _staging_failed(code: str, message: str, status: int = 409) -> ExecutionError:
    return ExecutionError(status, code, message)


def _uploads(spec: Mapping[str, Any], staged_files: Mapping[str, bytes], sources: Mapping[str, str]):
    """Every file staging uploads, as ``(run-dir-relative path, text)``, in upload order."""

    from backend.structure_search.staging import SEARCH_SUBDIR

    package = posixpath.basename(spec["runner"]["backend_package_dir"].rstrip("/"))
    # Sorted keys: the text must be reproducible from the spec stored in the ledger on resume.
    uploads = [("submission.json", json.dumps(spec, indent=2, sort_keys=True))]
    uploads += [(posixpath.join(package, name), text) for name, text in sorted(sources.items())]
    uploads += [
        (posixpath.join(SEARCH_SUBDIR, name), data.decode("utf-8"))
        for name, data in sorted(staged_files.items())
    ]
    return uploads


# -- finalization timing (diagnostics only) ---------------------------------------------
#
# The finalize script is built by backend.structure_search.staging, which is part
# of the runtime package uploaded with the attempt; changing that file would change
# the package digest an interrupted attempt is bound to. The service therefore adds
# its timing around the script when sending it: the wrappers below time functions of
# the already verified package in the remote process and call them unchanged.

_TIMED_FUNCTIONS = (
    ("_bmd_guard_module", "bmd_runtime_package_problems", "package_recheck"),
    ("_bmd_search_plan", "load_plan_text", "plan_load"),
    ("_bmd_search_plan", "default_potcar_identity", "potcar_identity"),
    ("_bmd_search_plan", "potcar_identity_from_text", "potcar_parse"),
    ("_bmd_search_plan", "potcar_identity_problems", "potcar_compare"),
    ("_bmd_search_plan", "verify_execution_binding", "verify_staged_inputs"),
    ("_bmd_staging", "_copy_exclusive", "potcar_copy"),
    ("_bmd_staging", "finalize_staged_search", "finalize"),
)
_INSTALL_LINE = "bmd_install_verified_runtime_package(REQUEST['package_dir'], REQUEST['manifest'])"
_MAIN_LINE = "finalize_staged_search_main(REQUEST)"


def instrument_finalize_script(script: str) -> str:
    """The finalize script with phase timings printed on a separate line. Same result line."""

    lines = script.split("\n")
    if lines.count(_INSTALL_LINE) != 1 or lines.count(_MAIN_LINE) != 1 or lines.index(_MAIN_LINE) < lines.index(
            _INSTALL_LINE):
        raise RuntimeError("The finalize script does not have the expected shape; it is not instrumented.")
    timed = "\n".join(f"_bmd_timed({module}, {name!r}, {key!r})" for module, name, key in _TIMED_FUNCTIONS)
    block = f"""_BMD_TIMINGS = {{"guard_verify_and_install_s": round(_bmd_time.monotonic() - _BMD_T0, 3)}}
_bmd_t = _bmd_time.monotonic()
import backend.runtime_package_guard as _bmd_guard_module
import backend.structure_search.search_plan as _bmd_search_plan
import backend.structure_search.staging as _bmd_staging
_BMD_TIMINGS["imports_s"] = round(_bmd_time.monotonic() - _bmd_t, 3)
def _bmd_timed(module, name, key):
    original = getattr(module, name, None)
    if original is None:  # not in the package this attempt was staged with
        return
    def timed(*args, **kwargs):
        start = _bmd_time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            _BMD_TIMINGS[key + "_s"] = round(_BMD_TIMINGS.get(key + "_s", 0.0) + _bmd_time.monotonic() - start, 3)
            _BMD_TIMINGS[key + "_calls"] = _BMD_TIMINGS.get(key + "_calls", 0) + 1
    setattr(module, name, timed)
{timed}
try:
    {_MAIN_LINE}
finally:
    _BMD_TIMINGS["total_s"] = round(_bmd_time.monotonic() - _BMD_T0, 3)
    print({TIMINGS_MARKER!r} + json.dumps(_BMD_TIMINGS, sort_keys=True), flush=True)"""
    index = lines.index(_MAIN_LINE)
    lines[index] = block
    return "import time as _bmd_time\n_BMD_T0 = _bmd_time.monotonic()\n" + "\n".join(lines)


def parse_finalize_timings(output: str) -> dict[str, Any] | None:
    for line in (output or "").splitlines():
        if line.startswith(TIMINGS_MARKER):
            try:
                timings = json.loads(line[len(TIMINGS_MARKER):])
            except ValueError:
                return None
            return timings if isinstance(timings, dict) else None
    return None


class _PhaseTimer:
    """Wall-clock time of each staging phase, logged per attempt (diagnostics only)."""

    def __init__(self, attempt_id: str):
        self.attempt_id = attempt_id
        self.started = time.monotonic()
        self.phases: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        start = time.monotonic()
        status = "failed"
        try:
            yield
            status = "done"
        finally:
            elapsed = time.monotonic() - start
            self.phases[name] = round(elapsed, 3)
            LOGGER.info("structure-search staging %s: %s %s after %.1f s", self.attempt_id, name, status, elapsed)

    def summary(self, outcome: str) -> None:
        LOGGER.info("structure-search staging %s: %s after %.1f s; phases %s", self.attempt_id, outcome,
                    time.monotonic() - self.started, json.dumps(self.phases, sort_keys=True))


class _RemoteStaging:
    """One attempt's staging on POWER: what it uploads and how POWER verifies it.

    Every method that touches POWER must run while the attempt's remote claim
    (``submission_attempts/<id>.lock``) is held.
    """

    def __init__(self, *, spec, plan, local_manifest, sources, manifest, extra_files=None):
        from backend.config import NOTEBOOK_DEFAULTS
        from backend.structure_search.potcar import POTCAR_FUNCTIONAL_DIR
        from backend.structure_search.runtime import SearchRuntimeSettings
        from backend.structure_search.staging import SEARCH_SUBDIR, staged_search_files, staging_layout

        self.spec = spec
        self.plan = plan
        self.sources = sources
        self.attempt_id = spec["submission"]["attempt_id"]
        paths = spec["paths"]
        self.layout = staging_layout(spec["run_name"], workspace_dir=paths["flows_dir"], logs_dir=paths["logs_dir"])
        staged_files = staged_search_files(plan, local_manifest, bridge_path=spec["structure_search"]["bridge_path"])
        self.uploads = _uploads(spec, staged_files, sources)
        # Files promotion adds at the top of the run directory (job script, acknowledgement).
        self.uploads += sorted((dict(extra_files or {})).items())
        self.expected = {path: sha256_hex(text) for path, text in self.uploads}
        self.potcars = {
            posixpath.join(SEARCH_SUBDIR, row["staged_at_execution_as"]): row["dataset_sha256"][:16]
            for row in plan.record["potcar"]["elements"]
        }
        self.expected_files = json.loads(staged_files["prepared_inputs.json"])["files"]
        self.request = {
            "attempt_id": self.attempt_id,
            "run_dir": self.layout["run_dir"],
            "package_dir": self.layout["package_dir"],
            "search_dir": self.layout["search_dir"],
            "bridge_path": self.layout["bridge_path"],
            "manifest": dict(manifest),
            "plan_digest": plan.digest,
            "potcar_library": NOTEBOOK_DEFAULTS["PMG_VASP_PSP_DIR"],
            "functional_dir": POTCAR_FUNCTIONAL_DIR,
            "cache_ready_marker": str(SearchRuntimeSettings.from_config().ready_marker),
            "copy_potcars": True,
        }

    def _python(self) -> str:
        import shlex

        return shlex.quote(str(self.spec["runner"]["python"]))

    def inventory(self, runner) -> dict[str, Any]:
        from backend.structure_search.staging import build_inventory_script, parse_inventory_output

        script = build_inventory_script(self.layout["run_dir"])
        output = runner.run(f"{self._python()} - <<'PY'\n{script}PY", check=False, modules=False, export_env=False)
        return parse_inventory_output(output.stdout)

    def reconcile(self, runner, *, complete: bool = False) -> dict[str, Any]:
        from backend.structure_search.staging import reconcile_staging

        return reconcile_staging(self.inventory(runner), self.expected, attempt_id=self.attempt_id,
                                 potcars=self.potcars, complete=complete)

    def upload_missing(self, runner, plan: Mapping[str, Any]) -> None:
        """Upload what is missing; complete or supersede this attempt's own unfinished uploads."""

        from backend.structure_search.staging import UPLOAD_MODE, staging_temporary_path

        missing = plan["missing"]
        run_dir = self.layout["run_dir"]
        for parent in sorted({posixpath.dirname(path) for path in missing}):
            runner.ensure_directory(posixpath.join(run_dir, parent) if parent else run_dir)
        for relative, text in self.uploads:
            if relative not in missing:
                continue
            target = posixpath.join(run_dir, relative)
            temporary = staging_temporary_path(target, self.attempt_id, self.expected[relative])
            if not missing[relative]["temporary_complete"]:
                runner.put_text(temporary, text, mode=UPLOAD_MODE)  # creates, or supersedes our own partial copy
            runner._replace_file(temporary, target)  # atomic; the target is absent under the claim

    def finalize(self, runner, *, copy_potcars: bool, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from backend.structure_search.staging import build_finalize_script, parse_finalize_output
        from backend.submission import RUNTIME_PACKAGE_GUARD_MODULE
        import shlex

        script = instrument_finalize_script(build_finalize_script(
            {**self.request, **dict(options or {}), "copy_potcars": copy_potcars},
            runtime_sources=self.sources, guard_module=RUNTIME_PACKAGE_GUARD_MODULE,
        ))
        command = f"cd {shlex.quote(self.layout['run_dir'])}\n{self._python()} - <<'PY'\n{script}PY"
        started = time.monotonic()
        try:
            output = runner.run(command, check=False, modules=False, export_env=False,
                                timeout_s=FINALIZE_TIMEOUT_S)
        except Exception as exc:
            LOGGER.warning("structure-search staging %s: finalize (copy_potcars=%s) gave no answer: %s after "
                           "%.1f s (timeout %d s)", self.attempt_id, copy_potcars, type(exc).__name__,
                           time.monotonic() - started, FINALIZE_TIMEOUT_S)
            raise
        timings = parse_finalize_timings(output.stdout)
        LOGGER.info("structure-search staging %s: finalize (copy_potcars=%s) answered after %.1f s "
                    "(timeout %d s); on POWER %s", self.attempt_id, copy_potcars, time.monotonic() - started,
                    FINALIZE_TIMEOUT_S, json.dumps(timings, sort_keys=True) if timings else "no timings")
        self.last_finalize_timings = timings
        result = parse_finalize_output(output.stdout)
        if not result.get("ok"):
            error = _staging_failed(str(result.get("code") or "staging_failed"),
                                    str(result.get("message") or "Staging on POWER failed."))
            if result.get("problems"):
                error.details["problems"] = [str(problem)[:500] for problem in result["problems"][:20]]
            raise error
        bindings = result.get("bindings") or {}
        if (
            result.get("plan_digest") != self.plan.digest
            or result.get("files") != self.expected_files
            or bindings.get("bridge_path") != self.layout["bridge_path"]
            or bindings.get("bound") is not True
            or result.get("contains_potcar_contents") is not False
        ):
            raise _staging_failed("staging_unconfirmed", "POWER did not confirm the staged files.")
        return result

    def verify(self, runner, *, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Read-only verification of the complete staged run directory. Writes nothing."""

        result = self.finalize(runner, copy_potcars=False, options=options)
        self.reconcile(runner, complete=True)
        return result

    def reserve_run_directory(self, runner) -> None:
        """A new attempt: refuse an existing run directory, record STAGING, then create the directory.

        The remote STAGING state, written first, reserves the run name for this
        attempt: a later request that finds the state but not the directory
        creates it, and one that finds both adopts the directory only if
        everything in it is attributable to this attempt (``reconcile``).
        """

        run_dir = self.layout["run_dir"]
        if runner.is_dir(run_dir) or runner.is_file(run_dir):
            raise _staging_failed("run_directory_exists",
                                  "The run directory already exists; it is never reused or replaced.")
        _write_search_state(runner, self.spec, REMOTE_STATE_STAGING, output="")
        self.create_run_directory(runner)

    def create_run_directory(self, runner) -> None:
        import shlex

        created = runner.run(f"mkdir {shlex.quote(self.layout['run_dir'])}", check=False,
                             modules=False, export_env=False)
        if not created.ok:
            raise _staging_failed("run_directory_exists",
                                  "The run directory already exists; it is never reused or replaced.")


def search_state_payload(spec: Mapping[str, Any], state: str, *, output: str = "") -> dict[str, Any]:
    """A remote search attempt state, in the calculation attempt-state format.

    The same fields as ``ParamikoRemoteRunner._write_submission_attempt_state``
    writes, so its reader and fingerprint check accept it; only the state names
    differ (``backend.structure_search.lifecycle``).
    """

    from backend.submission import (
        SUBMISSION_ATTEMPT_STATE_VERSION,
        submission_attempt_comment,
        submission_attempt_fingerprint,
        submission_attempt_id,
        submission_attempt_metadata,
    )

    from copy import deepcopy

    return {
        "version": SUBMISSION_ATTEMPT_STATE_VERSION,
        "attempt_id": submission_attempt_id(dict(spec)),
        "state": state,
        "fingerprint": submission_attempt_fingerprint(dict(spec)),
        "metadata": submission_attempt_metadata(dict(spec)),
        "attempt_comment": submission_attempt_comment(dict(spec)),
        "run_name": spec.get("run_name"),
        "run_dir": (spec.get("paths") or {}).get("run_dir"),
        "remote_script": (spec.get("paths") or {}).get("remote_script"),
        "updated_at": isoformat(utc_now()),
        "output": output,
        "provenance": deepcopy(spec.get("provenance")),
    }


def _write_search_state(runner, spec: Mapping[str, Any], state: str, *, output: str = "") -> None:
    """Replace the remote search attempt state atomically.

    The new state is written to an attempt-bound temporary name beside the
    state file (``.<id>.json.bmdstage-<id>-<sha16>.part``) and renamed over it,
    so a reader sees the old state or the new one, never a partial file. An
    interruption leaves the old state in place; a leftover temporary belongs to
    this attempt and is superseded by the next write of the same content.
    Calculation attempt states are not written this way and are unchanged.
    """

    from backend.structure_search.staging import UPLOAD_MODE, staging_temporary_path

    path = posixpath.join(spec["paths"]["submission_attempts_dir"], f"{spec['submission']['attempt_id']}.json")
    text = json.dumps(search_state_payload(spec, state, output=output), indent=2, sort_keys=True)
    temporary = staging_temporary_path(path, spec["submission"]["attempt_id"], sha256_hex(text))
    runner.put_text(temporary, text, mode=UPLOAD_MODE)
    runner._replace_file(temporary, path)


@contextmanager
def _claimed_runner(spec, *, busy_code: str = "staging_in_progress"):
    from backend.remote_runtime import connected_remote_runner, connection_profile_from_cluster
    from compute_api import execution

    with connected_remote_runner(
        profile=connection_profile_from_cluster(spec["cluster"]),
        runner_factory=execution.RUNNER_FACTORY,
    ) as runner:
        runner.ensure_directory(spec["paths"]["submission_attempts_dir"])
        if not runner._claim_submission_attempt(spec):
            raise _staging_failed(busy_code, "Another request is working on this attempt on POWER; retry.")
        try:
            yield runner
        finally:
            runner._release_submission_attempt_claim(spec)


def _stage_remote(*, spec, plan, local_manifest, sources, manifest) -> dict[str, Any]:
    """Stage one search on POWER through BMD's remote runner. Never runs sbatch, USPEX or VASP."""

    staging = _RemoteStaging(spec=spec, plan=plan, local_manifest=local_manifest, sources=sources,
                             manifest=manifest)
    phase = _PhaseTimer(staging.attempt_id)
    outcome = "failed"
    try:
        with _claimed_runner(spec) as runner:
            phase.phases["connect_and_claim"] = round(time.monotonic() - phase.started, 3)
            state = runner._read_submission_attempt_state(spec)
            if state:
                runner._ensure_submission_attempt_matches(spec, state)
                if state.get("state") == REMOTE_STATE_STAGED:
                    with phase("reverify"):
                        result = staging.verify(runner)  # adopted only after a read-only re-verification
                    outcome = "adopted after re-verification"
                    return {**result, "remote_state": REMOTE_STATE_STAGED}
                if state.get("state") != REMOTE_STATE_STAGING:
                    raise _staging_failed("remote_state_invalid",
                                          f"The remote attempt is in state {state.get('state')!r}.")
                if not runner.is_dir(staging.layout["run_dir"]):
                    staging.create_run_directory(runner)  # interrupted after reserving the run name
            else:
                staging.reserve_run_directory(runner)

            with phase("inventory"):
                uploads = staging.reconcile(runner)
            with phase(f"uploads ({len(uploads['missing'])} file(s))"):
                staging.upload_missing(runner, uploads)
            with phase("finalize"):
                result = staging.finalize(runner, copy_potcars=True)
            with phase("final_inventory"):
                staging.reconcile(runner, complete=True)
            summary = {
                "plan_digest": result["plan_digest"],
                "files": result["files"],
                "potcars": result["potcars"],
                "bridge_path_bound": True,
            }
            _write_search_state(runner, spec, REMOTE_STATE_STAGED, output=json.dumps(summary, sort_keys=True))
            outcome = "staged"
            return {**result, "remote_state": REMOTE_STATE_STAGED}
    finally:
        phase.summary(outcome)


def _staging_inputs(ledger, record):
    """The frozen plan, its local manifest and today's runtime package, checked for staging.

    Refuses a plan BMD Compute would now make differently (``plan_changed``)
    and a runtime package whose bridge is not the one the plan was made for
    (``bridge_changed``). Contacts nothing.
    """

    from backend.runtime_package import runtime_package_manifest_digest, runtime_package_manifest_for_sources
    from backend.structure_search.search_plan import (
        PLAN_FILE,
        PREPARED_FILE,
        SearchPlanError,
        load_plan_text,
        require_current_plan,
    )
    from backend.structure_search.staging import BRIDGE_RELATIVE_PATH
    from backend.submission import build_backend_module_sources

    directory = _preparation_dir(ledger, record)
    plan = load_plan_text((directory / PLAN_FILE).read_text(encoding="utf-8"))
    local_manifest = json.loads((directory / PREPARED_FILE).read_text(encoding="utf-8"))
    try:
        require_current_plan(plan)
    except SearchPlanError as exc:
        raise ExecutionError(409, "plan_changed", "BMD Compute would now plan this search differently; "
                             "it is not staged. Prepare a new attempt.",
                             changed=(exc.diagnostic or {}).get("changed")) from None
    sources = build_backend_module_sources()
    manifest = runtime_package_manifest_for_sources(sources)
    if manifest.get(BRIDGE_RELATIVE_PATH) != plan.record["execution"]["bridge"]["sha256"]:
        raise ExecutionError(409, "bridge_changed", "The launch bridge differs from the one the plan was made for.")
    return plan, local_manifest, sources, manifest, runtime_package_manifest_digest(manifest)


def reverify_staged_search(*, principal: str, attempt_id: str) -> dict[str, Any]:
    """Freshly re-verify a STAGED search on POWER, read-only. Changes nothing anywhere.

    ``STAGED`` records the state POWER last verified, not a guarantee that the
    run directory is still intact. Step 4D must call this, inside the same
    remote claim as any submission, before an attempt may become
    submission-ready; ordinary reads (GET) never contact POWER.

    Checks, against the stored submission spec: the local preparation, that
    the plan is still current, that today's runtime package is the one that
    was uploaded, the remote attempt state (``STAGED``, same fingerprint), and
    on POWER the runtime package (guard), plan digest, bound bridge, POTCAR
    identities and the complete run directory (no file missing, changed,
    added or left half-written). Returns the verification; raises
    :class:`ExecutionError` (409 with the problem's code) otherwise.
    """

    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No structure-search attempt with this ID exists.")
    _authorize_record(record, principal)
    if record["state"] != STATE_STAGED:
        raise ExecutionError(409, "attempt_not_staged", "This attempt has not been staged.")
    _stored(ledger, record)
    plan, local_manifest, sources, manifest, manifest_digest = _staging_inputs(ledger, record)
    staging = record["staging"]
    if manifest_digest != staging["runtime_manifest_digest"]:
        raise ExecutionError(409, "runtime_package_changed",
                             "BMD Compute's runtime package changed since this attempt was staged.")
    spec = staging["spec"]
    remote = _RemoteStaging(spec=spec, plan=plan, local_manifest=local_manifest, sources=sources,
                            manifest=manifest)
    with _claimed_runner(spec) as runner:
        state = runner._read_submission_attempt_state(spec)
        if not state:
            raise _staging_failed("remote_state_missing", "POWER holds no state for this attempt.")
        runner._ensure_submission_attempt_matches(spec, state)
        if state.get("state") != REMOTE_STATE_STAGED:
            raise _staging_failed("remote_state_invalid", f"The remote attempt is in state {state.get('state')!r}.")
        result = remote.verify(runner)
    return {
        "attempt_id": attempt_id,
        "plan_digest": result["plan_digest"],
        "verified_at": isoformat(utc_now()),
        "remote_state": REMOTE_STATE_STAGED,
        "runtime_package_manifest_sha256": manifest_digest,
        "files": result["files"],
        "potcars": result["potcars"],
        "contains_potcar_contents": False,
    }


def stage_search_attempt(*, principal: str, attempt_id: str, shape) -> dict[str, Any]:
    """Stage a PREPARED_LOCALLY search attempt on POWER (idempotent; ends at STAGED)."""

    from backend.structure_search.staging import build_search_submission_spec
    from backend.submission import canonical_run_timestamp

    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No structure-search attempt with this ID exists.")
    _authorize_record(record, principal)
    if record["plan_digest"] != shape.expected_plan_digest:
        raise ExecutionError(409, "plan_digest_mismatch", "This attempt was prepared for a different plan digest.")
    if record["state"] == STATE_PREPARING:
        raise ExecutionError(409, "attempt_not_prepared", "This attempt has not been prepared locally yet.")
    stored = _stored(ledger, record)  # the local preparation must still be intact
    if record["state"] in (STATE_STAGED, STATE_PREPARED):
        return stored

    plan, local_manifest, sources, manifest, manifest_digest = _staging_inputs(ledger, record)
    bridge_sha256 = plan.record["execution"]["bridge"]["sha256"]

    with ledger.locked():
        record = ledger.read(attempt_id)
        _authorize_record(record, principal)
        if record["state"] in (STATE_STAGED, STATE_PREPARED):
            return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))
        if record["state"] == STATE_PREPARED_LOCALLY:
            started = isoformat(utc_now())
            spec = build_search_submission_spec(
                plan=plan,
                attempt_id=attempt_id,
                run_timestamp=canonical_run_timestamp(None),
                created_at=started,
                runtime_manifest=manifest,
            )
            record["state"] = STATE_STAGING
            record["staging"] = {
                "run_name": spec["run_name"],
                "spec": spec,
                "runtime_manifest_digest": manifest_digest,
                "bridge_sha256": bridge_sha256,
                "started_at": started,
                "staged_at": None,
                "result": None,
            }
            record["updated_at"] = started
            ledger.replace(record)
        elif record["state"] == STATE_STAGING:
            if record["staging"]["runtime_manifest_digest"] != manifest_digest:
                raise ExecutionError(
                    409, "runtime_package_changed",
                    "BMD Compute's runtime package changed since staging started; the partial staging "
                    "is kept and not completed. Use a new attempt ID.",
                )
            spec = record["staging"]["spec"]
        else:  # pragma: no cover - states are exhaustive
            raise ExecutionError(409, "remote_state_invalid", "The attempt is in an unexpected state.")

    result = _stage_remote(spec=spec, plan=plan, local_manifest=local_manifest, sources=sources, manifest=manifest)

    with ledger.locked():
        record = ledger.read(attempt_id)
        if record["state"] != STATE_STAGED:
            staged_at = isoformat(utc_now())
            record["state"] = STATE_STAGED
            record["staging"] = {
                **record["staging"],
                "staged_at": staged_at,
                "result": {
                    "remote_state": REMOTE_STATE_STAGED,
                    "bridge_path_bound": True,
                    "files": result["files"],
                    "potcars": result["potcars"],
                },
            }
            record["updated_at"] = staged_at
            ledger.replace(record)
    return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))


__all__ = [
    "SEARCH_ATTEMPT_KIND",
    "SEARCH_RECORD_SCHEMA",
    "SearchAttemptLedger",
    "STATE_PREPARED_LOCALLY",
    "STATE_PREPARING",
    "lookup_search_attempt",
    "stage_search_attempt",
    "prepare_search_attempt",
    "search_attempt_response",
    "search_request_sha256",
]
