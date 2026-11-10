"""Exactly-once SLURM submission of a submission-ready structure search (USPEX step 4D-3).

``PUT /api/v1/structure-search-attempts/{id}/submission`` (scope ``submit``,
the attempt's owner) submits a ``PREPARED`` search: locally ``PREPARED →
SUBMITTING → SUBMITTED``, on POWER ``SEARCH_PREPARED → SEARCH_SUBMITTING →
SEARCH_SUBMITTED``. It runs ``sbatch`` at most once per attempt and never
cancels, monitors or interprets anything.

Sequence:

1. Local checks: owner, plan digest, the frozen job script's SHA-256 and the
   acknowledged resource request echoed by the client, an intact local
   preparation, a current plan, the staged runtime package, and a job script
   that the frozen inputs still reproduce byte for byte. The record becomes
   ``SUBMITTING`` before POWER is contacted; from then on only evidence from
   POWER moves it.
2. On POWER, under the attempt's claim and the shared operation limiter, with
   the fingerprint and the promotion identities (job script, acknowledgement)
   checked:

   * ``SEARCH_PREPARED``: a fresh read-only verification of the complete run
     directory (runtime package, frozen plan, bridge, POTCAR identities,
     staged inputs, job script, acknowledgement, login-node readiness). Then
     ``SEARCH_SUBMITTING`` is written atomically, and only then
     ``sbatch --parsable bmd_search_job.sbatch`` runs in the run directory,
     once, with no command-line options: every resource comes from the
     script's frozen ``#SBATCH`` lines, which are the acknowledged request.

     - a job ID: recorded locally at once (``SUBMITTED``, remote record
       pending), then ``SEARCH_SUBMITTED`` (atomically) and the pending flag
       cleared. If that remote write fails, the job ID is kept and
       ``PUT …/submission/remote-record`` completes it (below);
     - a definite refusal (``sbatch: error`` or ``sbatch`` not found, no job
       ID): back to ``SEARCH_PREPARED`` and ``PREPARED`` with the error;
     - anything else (lost connection, timeout, unreadable output): the
       outcome is uncertain. POWER keeps ``SEARCH_SUBMITTING``, the record
       becomes ``SUBMISSION_UNCERTAIN``, and ``sbatch`` is never run again
       for this attempt.

   * ``SEARCH_SUBMITTING`` (an earlier request's outcome is unknown):
     read-only reconciliation, never ``sbatch``. ``squeue`` and ``sacct``
     (this account, the last 7 days) are searched for jobs carrying this
     attempt's identity: the SLURM comment ``bmd_attempt:<id>`` is the
     primary identity, the job name ``bmds-<id[:8]>`` and the working
     directory (the run directory) further checks. Only exactly one job
     reporting all three, with no other candidate and no contradiction
     anywhere, is adopted (``SEARCH_SUBMITTED``). None (accounting may lag),
     several, a contradiction, or a candidate whose comment SLURM does not
     report leave the attempt uncertain: an empty comment proves nothing.
     There is no automatic reset or retry; see ``docs/machine_api.md`` for the
     future operator procedure.
   * ``SEARCH_SUBMITTED``: adopted as recorded.

3. Remote-record recovery (``PUT …/submission/remote-record``): for a local
   ``SUBMITTED`` attempt whose ``SEARCH_SUBMITTED`` write was lost, POWER's
   frozen ``SEARCH_SUBMITTING`` record and SLURM's record of the job are
   checked against the job ID sbatch reported, and only then is
   ``SEARCH_SUBMITTED`` written, keeping that job ID and the original
   evidence. Never ``sbatch``; idempotent.

The response keeps three outcomes apart: the scheduler's state (not
refreshed here), the job's execution outcome (not read here), and scientific
validity (``not_assessed``); finalists are ``not_started``.

Everything here is service-side code; the runtime package is unchanged, so a
search promoted by the previous release can be submitted.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any, Mapping

from compute_api.execution import ExecutionError, _authorize_record
from compute_api.ledger import isoformat, utc_now
from compute_api.structure_search_execution import (
    LOGGER,
    STATE_PREPARED,
    SearchAttemptLedger,
    _calculation_id_conflict,
    _claimed_runner,
    _PhaseTimer,
    _RemoteStaging,
    _staging_failed,
    _staging_inputs,
    _stored,
    _write_search_state,
    search_attempt_response,
    search_resources,
)
from compute_api.structure_search_promotion import ACKNOWLEDGEMENT_FILE, _job_script, _remote_record, \
    acknowledgement_text

STATE_SUBMITTING = "SUBMITTING"
STATE_SUBMISSION_UNCERTAIN = "SUBMISSION_UNCERTAIN"
STATE_SUBMITTED = "SUBMITTED"
SUBMISSION_RECORD_SCHEMA = "bmd_compute.structure_search_submission_record"
SUBMISSION_RECORD_SCHEMA_VERSION = 1
RECONCILIATION_WINDOW = "now-7days"
_JOB_ID = re.compile(r"^([0-9]+)(?:;[A-Za-z0-9_.-]+)?$")


class _NotSubmitted(Exception):
    """A failure before sbatch could have run: POWER still holds SEARCH_PREPARED."""

    def __init__(self, cause: BaseException):
        super().__init__(str(cause))
        self.cause = cause


# -- identities -------------------------------------------------------------------------------


def job_identity(attempt_id: str, run_dir: str) -> dict[str, str]:
    from backend.structure_search.job import JOB_NAME_PREFIX
    from backend.submission import SUBMISSION_ATTEMPT_COMMENT_PREFIX

    return {
        "comment": f"{SUBMISSION_ATTEMPT_COMMENT_PREFIX}{attempt_id}",
        "job_name": f"{JOB_NAME_PREFIX}{attempt_id[:8]}",
        "workdir": run_dir.rstrip("/"),
    }


def parse_parsable_job_id(stdout: str) -> str | None:
    """The job ID ``sbatch --parsable`` printed (``123`` or ``123;cluster``), or None."""

    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    match = _JOB_ID.match(lines[0])
    return match.group(1) if match else None


def definite_rejection(returncode: int, stdout: str, stderr: str) -> bool:
    """sbatch ran (or could not be found) and created no job: safe to return to SEARCH_PREPARED."""

    if returncode == 0 or parse_parsable_job_id(stdout) is not None:
        return False
    return returncode == 127 or "sbatch: error" in (stderr or "")


def scheduler_records(squeue_output: str, sacct_output: str) -> list[dict[str, str]]:
    """Jobs from ``squeue -o '%i|%j|%k|%Z|%T'`` and ``sacct -P -o JobID,JobName,Comment,WorkDir,State``."""

    records = []
    for source, text in (("squeue", squeue_output), ("sacct", sacct_output)):
        for line in (text or "").splitlines():
            fields = line.split("|")
            if len(fields) != 5 or not fields[0].strip():
                continue
            job_id = fields[0].strip()
            # Job steps (``123.batch``) are not jobs. Any other ID is kept whole: an array task or range or a
            # heterogeneous component (``123_4``, ``123_[1-5]``, ``123+0``) is its own candidate, never merged
            # into ``123`` and never dropped.
            if "." in job_id or not job_id[0].isdigit():
                continue
            comment = fields[2].strip()
            records.append({
                "source": source,
                "job_id": job_id,
                "job_name": fields[1].strip(),
                "comment": "" if comment in ("(null)", "None") else comment,
                "workdir": fields[3].strip().rstrip("/"),
                "state": fields[4].strip(),
            })
    return records


def reconcile_jobs(records: list[Mapping[str, str]], identity: Mapping[str, str]) -> dict[str, Any]:
    """Match scheduler records against one attempt's identity. Read-only; decides nothing by itself.

    Records are grouped by job ID (``squeue`` and ``sacct`` may both report a
    job, each with some fields empty); a field counts as reported where any
    source gives it. A job concerns this attempt if it reports the attempt's
    comment, the run directory, or the job name with no directory. Then:

    * **verified**: comment, name and directory all reported and all the
      attempt's. The comment (``bmd_attempt:<uuid>``) is the primary identity;
      name and directory are further checks, never a substitute for it;
    * **conflict**: it contradicts the identity anywhere: two different values
      for one field, the attempt's comment with another name or directory, or
      the run directory or name under another comment;
    * **unverified**: nothing contradicts it but a field (usually the comment,
      which accounting may not keep) is not reported. It is never adopted:
      an empty comment proves nothing either way.

    Outcome: ``conflict`` if any job conflicts; ``matched`` for exactly one
    verified job and no other candidate; ``ambiguous`` for several
    candidates with at least one verified; ``unverified`` for unverified
    candidates only; ``none`` when no job concerns this attempt (which, given
    accounting delays, is not proof that none was submitted).
    """

    jobs: dict[str, dict[str, set[str]]] = {}
    for record in records:
        job = jobs.setdefault(record["job_id"], {"comment": set(), "job_name": set(), "workdir": set(),
                                                 "state": set(), "source": set()})
        for field in ("comment", "job_name", "workdir", "state", "source"):
            if record.get(field):
                job[field].add(record[field])
    verified, unverified, conflicts = [], [], []
    for job_id, job in sorted(jobs.items()):
        ours = {field: identity[field] in job[field] for field in ("comment", "job_name", "workdir")}
        if not (ours["comment"] or ours["workdir"] or (ours["job_name"] and not job["workdir"])):
            continue  # another job (a name prefix shared by another attempt runs elsewhere)
        summary = {"job_id": job_id, **{field: sorted(values) for field, values in job.items()}}
        contradicts = any(len(job[field]) > 1 or (job[field] and not ours[field])
                          for field in ("comment", "job_name", "workdir"))
        if contradicts:
            conflicts.append(summary)
        elif all(ours.values()):
            verified.append(summary)
        else:
            unverified.append(summary)
    if conflicts:
        outcome = "conflict"
    elif len(verified) == 1 and not unverified:
        outcome = "matched"
    elif verified:
        outcome = "ambiguous"
    elif unverified:
        outcome = "unverified"
    else:
        outcome = "none"
    return {
        "outcome": outcome,
        "job_id": verified[0]["job_id"] if outcome == "matched" else None,
        "matches": [job["job_id"] for job in verified],
        "unverified": unverified,
        "conflicts": conflicts,
    }


def _query_scheduler(runner) -> tuple[str, str] | None:
    """squeue and sacct for this account; None when either did not answer (read-only)."""

    squeue = runner.run("squeue -h -u \"$(id -un)\" -o '%i|%j|%k|%Z|%T'", check=False, modules=False,
                        export_env=False)
    sacct = runner.run(f"sacct -X -n -P -u \"$(id -un)\" -S {RECONCILIATION_WINDOW} "
                       "-o JobID,JobName,Comment,WorkDir,State", check=False, modules=False, export_env=False)
    if not squeue.ok or not sacct.ok:
        return None
    return squeue.stdout or "", sacct.stdout or ""


# -- the remote operation -----------------------------------------------------------------------


def _submission_block(record) -> dict[str, Any]:
    spec = record["staging"]["spec"]
    return {"attempt_id": record["attempt_id"], **job_identity(record["attempt_id"], spec["paths"]["run_dir"]),
            "job_script_sha256": record["promotion"]["job_script"]["sha256"]}


_IDENTITY_KEYS = ("attempt_id", "comment", "job_name", "workdir", "job_script_sha256")


def _identity_matches(remote_submission: Mapping[str, Any] | None, record) -> bool:
    """The remote submission record names this attempt's frozen job identity, field for field."""

    expected = _submission_block(record)
    remote_submission = remote_submission or {}
    return all(remote_submission.get(key) == expected[key] for key in _IDENTITY_KEYS)


def _promotion_matches(output: Mapping[str, Any] | None, promotion: Mapping[str, Any]) -> bool:
    output = output or {}
    return ((output.get("job_script") or {}).get("sha256") == promotion["job_script"]["sha256"]
            and output.get("acknowledgement_sha256") == promotion["acknowledgement"]["acknowledgement_sha256"])


def _submit_remote(*, record, inputs, allow_sbatch: bool, on_job_id=None) -> dict[str, Any]:
    """Submit (at most once) or reconcile on POWER. Returns the outcome; see the module docstring."""

    from backend.structure_search import lifecycle
    from backend.structure_search.job import JOB_SCRIPT_NAME

    spec = record["staging"]["spec"]
    promotion = record["promotion"]
    identity = job_identity(record["attempt_id"], spec["paths"]["run_dir"])
    phase = _PhaseTimer(record["attempt_id"])
    outcome = "failed"
    observed_prepared = False  # POWER confirmed SEARCH_PREPARED in this request
    in_sbatch_window = False
    try:
        with _claimed_runner(spec, busy_code="submission_in_progress") as runner:
            phase.phases["connect_and_claim"] = round(time.monotonic() - phase.started, 3)
            state = runner._read_submission_attempt_state(spec)
            if not state:
                raise _staging_failed("remote_state_missing", "POWER holds no state for this attempt.")
            runner._ensure_submission_attempt_matches(spec, state)
            current = state.get("state")
            output = _remote_record(state.get("output")) or {}

            if current in (lifecycle.REMOTE_SEARCH_SUBMITTED, lifecycle.REMOTE_SEARCH_SUBMITTING):
                if not _promotion_matches(output.get("promotion"), promotion):
                    raise _staging_failed("promotion_mismatch",
                                          "POWER records a submission of another promotion of this attempt.")
                submission = dict(output.get("submission") or {})
                if not _identity_matches(submission, record):
                    raise _staging_failed("submission_conflict",
                                          "POWER records a submission with another job identity for this attempt.")
                if current == lifecycle.REMOTE_SEARCH_SUBMITTED:
                    outcome = "adopted"
                    return {"outcome": "submitted", "adopted": True, **submission}
                with phase("reconciliation"):
                    return _reconcile(runner, spec, output, identity)

            if current != lifecycle.REMOTE_SEARCH_PREPARED:
                raise lifecycle.LifecycleError(
                    "transition_not_allowed", f"Submission starts from SEARCH_PREPARED; POWER holds {current}.")
            if not _promotion_matches(output, promotion):
                raise _staging_failed("promotion_mismatch", "POWER holds a different promotion of this attempt.")
            observed_prepared = True
            if not allow_sbatch:
                # The record says an earlier submission is uncertain, yet POWER shows SEARCH_PREPARED: only an
                # operator could have reset it. Never submit automatically on that evidence.
                raise _staging_failed("submission_uncertain",
                                      "This attempt's earlier submission is uncertain; it is not submitted again.")
            if isinstance(inputs, Exception):
                raise inputs  # a stale plan or changed package: refused before sbatch, as at promotion
            plan, local_manifest, sources, manifest, script = inputs
            remote = _RemoteStaging(
                spec=spec, plan=plan, local_manifest=local_manifest, sources=sources, manifest=manifest,
                extra_files={JOB_SCRIPT_NAME: script,
                             ACKNOWLEDGEMENT_FILE: acknowledgement_text(promotion["acknowledgement"])},
            )
            with phase("fresh_verification"):
                remote.verify(runner, options={"job_script": {"name": JOB_SCRIPT_NAME,
                                                              "sha256": promotion["job_script"]["sha256"]},
                                               "runtime_check": True})

            lifecycle.require_remote_transition(current, lifecycle.REMOTE_SEARCH_SUBMITTING)
            promotion_record = {key: value for key, value in output.items() if key != "last_submission_error"}
            submission = {**_submission_block(record), "requested_at": isoformat(utc_now()),
                          "verified_at": isoformat(utc_now())}
            in_sbatch_window = True  # from here on POWER may hold SEARCH_SUBMITTING
            _write_search_state(runner, spec, lifecycle.REMOTE_SEARCH_SUBMITTING,
                                output=json.dumps({"schema": SUBMISSION_RECORD_SCHEMA,
                                                   "schema_version": SUBMISSION_RECORD_SCHEMA_VERSION,
                                                   "promotion": promotion_record, "submission": submission},
                                                  sort_keys=True))
            command = (f"cd {shlex.quote(spec['paths']['run_dir'])} && "
                       f"sbatch --parsable {shlex.quote(JOB_SCRIPT_NAME)}")
            try:
                result = runner.run(command, check=False, modules=False, export_env=False)
            except Exception as exc:  # noqa: BLE001 - the outcome is unknown: never repeat
                LOGGER.warning("structure-search %s: sbatch outcome unknown (%s)", record["attempt_id"],
                               type(exc).__name__)
                outcome = "uncertain"
                return {"outcome": "uncertain", "reason": f"no answer from sbatch ({type(exc).__name__})"}
            job_id = parse_parsable_job_id(result.stdout)
            if result.ok and job_id is not None:
                submitted = {**submission, "job_id": job_id, "submitted_at": isoformat(utc_now()),
                             "reconciled": False}
                if on_job_id is not None:
                    try:  # the job ID is kept locally before anything else can fail
                        on_job_id(submitted)
                    except Exception:  # noqa: BLE001 - the remote record below and the final write still follow
                        LOGGER.error("structure-search %s: could not record job %s locally yet",
                                     record["attempt_id"], job_id, exc_info=True)
                pending = False
                try:
                    lifecycle.require_remote_transition(lifecycle.REMOTE_SEARCH_SUBMITTING,
                                                        lifecycle.REMOTE_SEARCH_SUBMITTED)
                    _write_search_state(runner, spec, lifecycle.REMOTE_SEARCH_SUBMITTED,
                                        output=json.dumps({"schema": SUBMISSION_RECORD_SCHEMA,
                                                           "schema_version": SUBMISSION_RECORD_SCHEMA_VERSION,
                                                           "promotion": promotion_record,
                                                           "submission": submitted}, sort_keys=True))
                except Exception:  # noqa: BLE001 - the job exists; POWER keeps SEARCH_SUBMITTING
                    pending = True
                outcome = "submitted"
                return {"outcome": "submitted", **submitted, "remote_record_pending": pending}
            if definite_rejection(result.returncode, result.stdout, result.stderr):
                error = (result.stderr or result.stdout or "").strip()[-500:]
                lifecycle.require_remote_transition(lifecycle.REMOTE_SEARCH_SUBMITTING,
                                                    lifecycle.REMOTE_SEARCH_PREPARED)
                _write_search_state(runner, spec, lifecycle.REMOTE_SEARCH_PREPARED,
                                    output=json.dumps({**promotion_record, "last_submission_error": error},
                                                      sort_keys=True))
                outcome = "rejected"
                return {"outcome": "rejected", "error": error, "returncode": result.returncode}
            outcome = "uncertain"
            return {"outcome": "uncertain",
                    "reason": f"sbatch exited {result.returncode} without a readable job ID"}
    except Exception as exc:
        if observed_prepared and not in_sbatch_window and allow_sbatch:
            raise _NotSubmitted(exc) from exc  # sbatch did not run; POWER still holds SEARCH_PREPARED
        raise
    finally:
        phase.summary(outcome)


def _summary(found: Mapping[str, Any]) -> dict[str, Any]:
    return {"outcome": found["outcome"], "matches": list(found["matches"]),
            "unverified": list(found["unverified"]), "conflicts": list(found["conflicts"]),
            "checked_at": isoformat(utc_now())}


def _reconcile(runner, spec, output: Mapping[str, Any], identity: Mapping[str, str]) -> dict[str, Any]:
    from backend.structure_search import lifecycle

    answered = _query_scheduler(runner)
    if answered is None:
        return {"outcome": "uncertain", "reason": "squeue or sacct did not answer; nothing was adopted"}
    found = reconcile_jobs(scheduler_records(*answered), identity)
    summary = _summary(found)
    if found["outcome"] != "matched":
        reason = {"none": "no job with this attempt's identity is visible to SLURM (accounting may lag)",
                  "unverified": "a job may be this attempt's, but SLURM does not report its full identity"}
        return {"outcome": "uncertain" if found["outcome"] == "none" else found["outcome"],
                "reason": reason.get(found["outcome"]), "reconciliation": summary}
    submission = {**dict(output.get("submission") or {}), "job_id": found["job_id"],
                  "submitted_at": None, "reconciled": True, "reconciliation": summary}
    lifecycle.require_remote_transition(lifecycle.REMOTE_SEARCH_SUBMITTING, lifecycle.REMOTE_SEARCH_SUBMITTED)
    _write_search_state(runner, spec, lifecycle.REMOTE_SEARCH_SUBMITTED,
                        output=json.dumps({**dict(output), "submission": submission}, sort_keys=True))
    return {"outcome": "submitted", **submission}


# -- the request ----------------------------------------------------------------------------------


def _local_transition(record: dict, target: str) -> None:
    from backend.structure_search import lifecycle

    lifecycle.require_local_transition(record["state"], target)
    record["state"] = target
    record["updated_at"] = isoformat(utc_now())


def _inputs(ledger, record):
    """Plan, manifest, sources and the frozen job script, or the refusal (local only)."""

    from backend.structure_search.job import JobDefinitionError, job_script_sha256

    try:
        plan, local_manifest, sources, manifest, manifest_digest = _staging_inputs(ledger, record)
    except ExecutionError as exc:
        return exc
    if manifest_digest != record["staging"]["runtime_manifest_digest"]:
        return ExecutionError(409, "runtime_package_changed",
                              "BMD Compute's runtime package changed since this attempt was staged. "
                              "Use a new attempt ID.")
    try:
        script = _job_script(record, plan, sources, manifest)
    except JobDefinitionError as exc:
        return ExecutionError(409, exc.code, exc.message)
    if job_script_sha256(script) != record["promotion"]["job_script"]["sha256"]:
        return ExecutionError(409, "job_script_changed",
                              "The job script this attempt froze can no longer be reproduced; use a new attempt ID.")
    return plan, local_manifest, sources, manifest, script


def submit_search_attempt(*, principal: str, attempt_id: str, shape) -> dict[str, Any]:
    """Submit a PREPARED search attempt exactly once, or reconcile an uncertain submission."""

    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No structure-search attempt with this ID exists.")
    _authorize_record(record, principal)
    if record["plan_digest"] != shape.expected_plan_digest:
        raise ExecutionError(409, "plan_digest_mismatch", "This attempt was prepared for a different plan digest.")
    if record["state"] not in (STATE_PREPARED, STATE_SUBMITTING, STATE_SUBMISSION_UNCERTAIN, STATE_SUBMITTED):
        raise ExecutionError(409, "attempt_not_prepared", "Only a PREPARED (submission-ready) attempt can be submitted.")
    promotion = record["promotion"]
    if shape.job_script_sha256 != promotion["job_script"]["sha256"]:
        raise ExecutionError(409, "job_script_mismatch", "This is not the job script the attempt was promoted with.",
                             job_script_sha256=promotion["job_script"]["sha256"])
    if shape.resources_sha256 != promotion["resources_sha256"]:
        raise ExecutionError(409, "acknowledgement_mismatch",
                             "This attempt was promoted for a different resource request.")
    stored = _stored(ledger, record)
    if record["state"] == STATE_SUBMITTED:
        return stored

    inputs = _inputs(ledger, record) if record["state"] == STATE_PREPARED else None
    if isinstance(inputs, ExecutionError):
        raise inputs  # refused locally, before POWER is contacted
    with ledger.locked():
        record = ledger.read(attempt_id)
        _authorize_record(record, principal)
        if record["state"] == STATE_SUBMITTED:
            return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))
        if record["state"] == STATE_PREPARED:
            _local_transition(record, STATE_SUBMITTING)
            record["submission"] = {**_submission_block(record), "requested_at": isoformat(utc_now()),
                                    "requested_by": principal, "status": "submitting", "job_id": None,
                                    "submitted_at": None, "reconciled": False, "last_error": None,
                                    "remote_record_pending": False, "uncertain_reason": None}
            ledger.replace(record)
    if inputs is None:  # resuming SUBMITTING or reconciling: inputs only matter if POWER is still prepared
        inputs = _inputs(ledger, record)

    def record_job(submitted: Mapping[str, Any]) -> None:
        # sbatch answered with a job ID: SUBMITTED locally at once, with the remote record still pending, so the
        # job ID survives anything that fails from here on (the remote write, or this process).
        with ledger.locked():
            current = ledger.read(attempt_id)
            if current["state"] != STATE_SUBMITTING:
                return
            _local_transition(current, STATE_SUBMITTED)
            current["submission"] = {**current["submission"], "status": "submitted",
                                     "job_id": submitted["job_id"], "submitted_at": submitted["submitted_at"],
                                     "reconciled": False, "remote_record_pending": True,
                                     "remote_submission": dict(submitted), "uncertain_reason": None}
            ledger.replace(current)

    try:
        result = _submit_remote(record=record, inputs=inputs, allow_sbatch=record["state"] == STATE_SUBMITTING,
                                on_job_id=record_job)
    except _NotSubmitted as failure:
        # sbatch did not run and POWER still holds SEARCH_PREPARED: the attempt is submission-ready again.
        with ledger.locked():
            current = ledger.read(attempt_id)
            if current["state"] == STATE_SUBMITTING:
                _local_transition(current, STATE_PREPARED)
                current["submission"] = {**current["submission"], "status": "not_submitted",
                                         "last_error": str(failure.cause)[:500]}
                ledger.replace(current)
        raise failure.cause from None

    with ledger.locked():
        record = ledger.read(attempt_id)
        kind = result["outcome"]
        if kind == "submitted":
            if record["state"] == STATE_SUBMITTED and record["submission"].get("job_id") not in (None,
                                                                                               result["job_id"]):
                raise ExecutionError(409, "submission_conflict",
                                     "POWER and the local record name different jobs for this attempt.",
                                     job_ids=[record["submission"]["job_id"], result["job_id"]])
            if record["state"] == STATE_PREPARED:
                _local_transition(record, STATE_SUBMITTING)
            if record["state"] != STATE_SUBMITTED:
                _local_transition(record, STATE_SUBMITTED)
            remote_submission = {key: value for key, value in result.items()
                                 if key not in ("outcome", "adopted", "remote_record_pending")}
            record["submission"] = {**record.get("submission", {}), "status": "submitted",
                                    "remote_submission": remote_submission,
                                    "job_id": result["job_id"], "submitted_at": result.get("submitted_at"),
                                    "reconciled": bool(result.get("reconciled")),
                                    "adopted": bool(result.get("adopted")),
                                    "remote_record_pending": bool(result.get("remote_record_pending")),
                                    "reconciliation": result.get("reconciliation"), "uncertain_reason": None}
            ledger.replace(record)
            LOGGER.info("structure-search %s submitted as SLURM job %s", attempt_id, result["job_id"])
            return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))
        if kind == "rejected":
            _local_transition(record, STATE_PREPARED)
            record["submission"] = {**record["submission"], "status": "rejected", "last_error": result["error"]}
            ledger.replace(record)
            raise ExecutionError(502, "submit_rejected",
                                 "SLURM refused the job; no job exists and the attempt remains submission-ready.",
                                 sbatch_error=result["error"][:300])
        # uncertain, ambiguous or conflict: never submitted again automatically
        if record["state"] == STATE_SUBMITTING:
            _local_transition(record, STATE_SUBMISSION_UNCERTAIN)
        record["submission"] = {**record["submission"], "status": "uncertain",
                                "uncertain_reason": result.get("reason") or kind,
                                "reconciliation": result.get("reconciliation")}
        ledger.replace(record)
    code = {"ambiguous": "submission_ambiguous", "conflict": "submission_conflict",
            "unverified": "submission_unverified"}.get(kind, "submission_uncertain")
    raise ExecutionError(409, code,
                         "The SLURM submission of this attempt is uncertain. It is never submitted again "
                         "automatically; repeating this request only looks for its job.",
                         reconciliation=result.get("reconciliation"))


# -- completing a missing remote record (local SUBMITTED, remote SEARCH_SUBMITTING) -------------------


def _recover_remote(*, record, principal: str) -> dict[str, Any]:
    """Write the missing SEARCH_SUBMITTED for a job sbatch reported. Never runs sbatch; read-only otherwise."""

    from backend.structure_search import lifecycle

    spec = record["staging"]["spec"]
    local = record["submission"]
    job_id = local["job_id"]
    identity = job_identity(record["attempt_id"], spec["paths"]["run_dir"])
    phase = _PhaseTimer(record["attempt_id"])
    outcome = "failed"
    try:
        with _claimed_runner(spec, busy_code="submission_in_progress") as runner:
            state = runner._read_submission_attempt_state(spec)
            if not state:
                raise _staging_failed("remote_state_missing", "POWER holds no state for this attempt.")
            runner._ensure_submission_attempt_matches(spec, state)
            current = state.get("state")
            output = _remote_record(state.get("output")) or {}
            if current not in (lifecycle.REMOTE_SEARCH_SUBMITTING, lifecycle.REMOTE_SEARCH_SUBMITTED):
                raise lifecycle.LifecycleError(
                    "transition_not_allowed",
                    f"The local record holds SLURM job {job_id}, but POWER holds {current}; nothing was changed. "
                    "This needs an operator.")
            if not _promotion_matches(output.get("promotion"), record["promotion"]):
                raise _staging_failed("promotion_mismatch", "POWER records another promotion of this attempt.")
            remote_submission = dict(output.get("submission") or {})
            if not _identity_matches(remote_submission, record):
                raise _staging_failed("submission_conflict",
                                      "POWER records a submission with another job identity for this attempt.")
            if remote_submission.get("job_id") not in (None, job_id):
                raise ExecutionError(409, "submission_conflict",
                                     "POWER and the local record name different jobs for this attempt.",
                                     job_ids=sorted({job_id, str(remote_submission["job_id"])}))
            if current == lifecycle.REMOTE_SEARCH_SUBMITTED:
                outcome = "already_recorded"  # an earlier recovery finished on POWER
                return {"outcome": "recorded", "already": True, "submission": remote_submission}

            with phase("scheduler_check"):
                answered = _query_scheduler(runner)
            if answered is None:
                raise ExecutionError(503, "scheduler_unavailable",
                                     "squeue or sacct did not answer; nothing was changed. Retry later.")
            found = reconcile_jobs(scheduler_records(*answered), identity)
            summary = _summary(found)
            if found["outcome"] == "matched" and found["job_id"] == job_id:
                pass
            elif found["outcome"] == "none":
                raise ExecutionError(409, "job_not_visible",
                                     f"SLURM does not (yet) report job {job_id}; accounting may lag. Nothing was "
                                     "changed; retry later.", reconciliation=summary)
            elif found["outcome"] == "unverified" and [job["job_id"] for job in found["unverified"]] == [job_id]:
                raise ExecutionError(409, "job_identity_unverified",
                                     f"SLURM reports job {job_id} without its full identity (comment, name and "
                                     "working directory). Nothing was changed.", reconciliation=summary)
            elif found["outcome"] in ("ambiguous", "unverified") and job_id in (
                    found["matches"] + [job["job_id"] for job in found["unverified"]]):
                raise ExecutionError(409, "submission_ambiguous",
                                     "SLURM reports more than one job that may be this attempt's. Nothing was "
                                     "changed.", reconciliation=summary)
            else:
                raise ExecutionError(409, "submission_conflict",
                                     f"SLURM's records contradict job {job_id} for this attempt. Nothing was "
                                     "changed.", reconciliation=summary)

            recovered = {**remote_submission, "job_id": job_id, "submitted_at": local.get("submitted_at"),
                         "reconciled": False,
                         "remote_record_recovery": {"recovered_at": isoformat(utc_now()), "recovered_by": principal,
                                                    "scheduler": summary}}
            lifecycle.require_remote_transition(lifecycle.REMOTE_SEARCH_SUBMITTING, lifecycle.REMOTE_SEARCH_SUBMITTED)
            _write_search_state(runner, spec, lifecycle.REMOTE_SEARCH_SUBMITTED,
                                output=json.dumps({**output, "submission": recovered}, sort_keys=True))
            outcome = "recovered"
            return {"outcome": "recorded", "already": False, "submission": recovered}
    finally:
        phase.summary(outcome)


def recover_remote_record(*, principal: str, attempt_id: str, shape) -> dict[str, Any]:
    """Complete the remote SEARCH_SUBMITTED of a locally SUBMITTED attempt whose remote record is pending.

    Idempotent: an attempt with nothing pending returns its stored response
    without contacting POWER. The job ID is the one sbatch reported and the
    local record keeps; it must be echoed (``job_id``), is verified against
    POWER's frozen submission record and SLURM's own record of the job, and
    is never replaced. ``sbatch`` is never run.
    """

    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No structure-search attempt with this ID exists.")
    _authorize_record(record, principal)
    if record["plan_digest"] != shape.expected_plan_digest:
        raise ExecutionError(409, "plan_digest_mismatch", "This attempt was prepared for a different plan digest.")
    if record["state"] != STATE_SUBMITTED:
        raise ExecutionError(409, "attempt_not_submitted",
                             "Only a SUBMITTED attempt (with a job ID from sbatch) has a remote record to complete.")
    submission = record.get("submission") or {}
    if shape.job_id != submission.get("job_id"):
        raise ExecutionError(409, "job_id_mismatch", "This is not the job ID recorded for this attempt.",
                             job_id=submission.get("job_id"))
    if not submission.get("remote_record_pending"):
        return _stored(ledger, record)

    result = _recover_remote(record=record, principal=principal)
    with ledger.locked():
        record = ledger.read(attempt_id)
        if record["state"] != STATE_SUBMITTED or record["submission"].get("job_id") != shape.job_id:
            raise ExecutionError(409, "submission_conflict", "The local record changed during recovery.")
        record["submission"] = {**record["submission"], "remote_record_pending": False,
                                "remote_record_recovered_at": isoformat(utc_now()),
                                "remote_record_recovery": result["submission"].get("remote_record_recovery")}
        ledger.replace(record)
    LOGGER.info("structure-search %s: remote record for job %s completed", attempt_id, shape.job_id)
    return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))


__all__ = [
    "definite_rejection",
    "job_identity",
    "parse_parsable_job_id",
    "reconcile_jobs",
    "recover_remote_record",
    "scheduler_records",
    "submit_search_attempt",
]
