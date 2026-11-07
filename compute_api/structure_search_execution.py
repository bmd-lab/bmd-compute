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
import os
import stat
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


def search_attempt_response(record: Mapping[str, Any], *, integrity: str | None = None) -> dict[str, Any]:
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
        "remote_staging_available": False,
        "submission_available": False,
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
    return search_attempt_response(record, integrity="verified")


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
        if record["state"] == STATE_PREPARED_LOCALLY:
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
            if record["state"] == STATE_PREPARED_LOCALLY:
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
    if record["state"] != STATE_PREPARED_LOCALLY:
        return search_attempt_response(record, integrity="incomplete")
    directory = _preparation_dir(ledger, record)
    intact = _verified(directory, record["plan_digest"]) is not None and (
        _manifest_sha256(directory) == record["preparation"]["manifest_sha256"]
    )
    return search_attempt_response(record, integrity="verified" if intact else "damaged")


__all__ = [
    "SEARCH_ATTEMPT_KIND",
    "SEARCH_RECORD_SCHEMA",
    "SearchAttemptLedger",
    "STATE_PREPARED_LOCALLY",
    "STATE_PREPARING",
    "lookup_search_attempt",
    "prepare_search_attempt",
    "search_attempt_response",
    "search_request_sha256",
]
