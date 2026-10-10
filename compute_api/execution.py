"""Authenticated, idempotent machine execution of one calculation.

``PUT /api/v1/attempts/{attempt_id}`` prepares, or prepares and submits, one
calculation through BMD Compute's existing machinery:

* scientific resolution, admission and the submission specification come from
  the injected planner (``main.plan_calculation_request``), exactly as for the
  browser and ``POST /api/v1/plans``;
* remote preparation is ``backend.remote_preparation.prepare_remote_submission``;
* submission is ``backend.remote_submission.submit_remote_workflow``;
* the remote attempt-state file, its fingerprint check, its ``mkdir`` lock and
  its PREPARED -> SUBMITTING -> SUBMITTED transitions are the existing
  ``ParamikoRemoteRunner`` contract and remain the authority on POWER.

What this module adds is the binding the remote record cannot hold (see
``compute_api.ledger``): which principal owns a client-chosen attempt UUID,
which request and plan digest it is bound to, which run timestamp rebuilds its
submission specification, and the durable per-principal submission caps.

No caller-supplied SSH target, remote path or job ID is ever used. Remote
connections use the server's connection profiles; remote paths come from the
server-built submission specification.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Mapping

from backend.monitoring import monitor_submitted_job
from backend.remote import REMOTE_OPERATION_BUSY_MESSAGE, RemoteOperationBusy
from backend.remote_preparation import prepare_remote_submission
from backend.remote_runtime import connected_remote_runner, default_connection_profile
from backend.remote_submission import submit_remote_workflow
from backend.runtime_package import build_runtime_package_manifest, runtime_package_manifest_digest
from backend.submission import canonical_run_timestamp

from compute_api import API_VERSION
from compute_api.canonical import canonical_json, sha256_hex
from compute_api.ledger import (
    ATTEMPT_RECORD_SCHEMA,
    ATTEMPT_RECORD_SCHEMA_VERSION,
    AttemptLedger,
    LedgerUnavailable,
    isoformat,
    parse_isoformat,
    utc_now,
)
from compute_api.plan_digest import compute_plan_digest
from compute_api.projection import plan_stage_previews
from compute_api.schemas import ExecutionRequest


# Replaced only by tests; production uses Compute's default Paramiko runner.
RUNNER_FACTORY: Callable[[], Any] | None = None

REMOTE_STATE_MAX_BYTES = 1024 * 1024
ATTEMPT_RESPONSE_SCHEMA = "bmd_compute.api.attempt"
ATTEMPT_RESPONSE_SCHEMA_VERSION = 1

STATE_REGISTERED = "registered"
STATE_PREPARED = "prepared"
STATE_SUBMITTED = "submitted"
STATE_UNCERTAIN = "submission_uncertain"

# Ledger submission status.
SUBMISSION_NOT_REQUESTED = "not_requested"
SUBMISSION_RESERVED = "reserved"
SUBMISSION_SUBMITTED = "submitted"
SUBMISSION_UNCERTAIN = "uncertain"
SUBMISSION_RELEASED = "released"

_SLURM_STATES = frozenset(
    {
        "PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "COMPLETED", "FAILED",
        "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
        "BOOT_FAIL", "DEADLINE", "SUSPENDED", "REQUEUED", "REQUEUE_HOLD",
        "REQUEUE_FED", "RESIZING", "SIGNALING", "STAGE_OUT", "SPECIAL_EXIT",
        "STOPPED", "REVOKED", "RESV_DEL_HOLD",
    }
)
_TERMINAL_SUMMARIES = frozenset({"SUCCESS", "FAILURE"})
_SUMMARIES = frozenset({"PENDING", "RUNNING", "SUCCESS", "FAILURE", "UNKNOWN"})
_SACCT_TIME = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")
_ELAPSED = re.compile(r"^(?:[0-9]{1,4}-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$")
_EXIT_CODE = re.compile(r"^[0-9]{1,3}:[0-9]{1,3}$")
_LOCAL_TIME = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$")
_JOB_ID = re.compile(r"^[0-9]{1,20}(?:_[0-9]{1,10})?$")


@dataclass(frozen=True)
class ExecutionPolicy:
    """Conservative initial API execution limits (not the POWER allocation limits)."""

    max_active_jobs: int = 2
    max_submissions_per_window: int = 5
    max_new_attempts_per_window: int = 20
    window: timedelta = timedelta(hours=24)
    max_cpus: int = 96
    max_memory_gb: int = 128
    max_walltime_seconds: int = 72 * 3600
    max_nodes: int = 1


POLICY = ExecutionPolicy()


class ExecutionError(Exception):
    def __init__(self, status_code: int, code: str, message: str, **details: Any):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


# ---------------------------------------------------------------- helpers


def _ledger() -> AttemptLedger:
    try:
        return AttemptLedger.from_environment()
    except LedgerUnavailable:
        raise ExecutionError(
            503,
            "execution_not_configured",
            "Machine execution is not configured on this service.",
        ) from None


def request_sha256(request: ExecutionRequest) -> str:
    plan = request.plan
    return sha256_hex(
        canonical_json(
            {
                "structure": {"format": plan.structure_format, "text": plan.structure_text},
                "desired_output": plan.desired_output,
                "custom_workflow": plan.custom_workflow,
                "resources": plan.resources,
                "labels": request.labels,
            }
        )
    )


def _walltime_seconds(value: str) -> int | None:
    match = re.fullmatch(r"([0-9]{1,3}):([0-5][0-9]):([0-5][0-9])", str(value or ""))
    if not match:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def enforce_resource_policy(submission_spec: Mapping[str, Any], policy: ExecutionPolicy = None) -> None:
    policy = policy or POLICY
    resources = submission_spec["resources"]
    problems = []
    if int(resources["nodes"]) > policy.max_nodes:
        problems.append({"field": "resources.nodes", "problem": "exceeds_api_limit"})
    if int(resources["ntasks"]) > policy.max_cpus:
        problems.append({"field": "resources.cpus", "problem": "exceeds_api_limit"})
    if int(resources["mem_gb"]) > policy.max_memory_gb:
        problems.append({"field": "resources.memory_gb", "problem": "exceeds_api_limit"})
    walltime = _walltime_seconds(resources["walltime"])
    if walltime is None or walltime > policy.max_walltime_seconds:
        problems.append({"field": "resources.walltime", "problem": "exceeds_api_limit"})
    if problems:
        raise ExecutionError(
            422,
            "resource_limit_exceeded",
            "The effective resources exceed the machine API execution limits.",
            fields=problems,
            limits={
                "nodes": policy.max_nodes,
                "cpus": policy.max_cpus,
                "memory_gb": policy.max_memory_gb,
                "walltime_hours": policy.max_walltime_seconds // 3600,
            },
        )


def current_runtime_digest() -> str:
    return runtime_package_manifest_digest(build_runtime_package_manifest())


def _prepared_runtime_digest(state: Mapping[str, Any]) -> str | None:
    provenance = state.get("provenance") or {}
    runtime_source = ((provenance.get("bmd_compute") or {}).get("runtime_source")) or {}
    manifest = runtime_source.get("manifest")
    if not isinstance(manifest, dict) or not manifest:
        return None
    try:
        return runtime_package_manifest_digest(manifest)
    except Exception:  # noqa: BLE001 - an unreadable manifest is treated as a mismatch
        return None


def _effective_resources(submission_spec: Mapping[str, Any]) -> dict[str, Any]:
    resources = submission_spec["resources"]
    cluster = submission_spec["cluster"]
    return {
        "nodes": int(resources["nodes"]),
        "cpus": int(resources["ntasks"]),
        "memory_gb": int(resources["mem_gb"]),
        "walltime": str(resources["walltime"]),
        "partition": str(cluster["partition"]),
        "account": str(cluster["account"]),
    }


def _remote_unavailable(exc: Exception | None = None) -> ExecutionError:
    if isinstance(exc, RemoteOperationBusy):
        return ExecutionError(503, "remote_busy", "BMD Compute is busy with other remote operations; retry shortly.")
    return ExecutionError(
        503,
        "remote_unavailable",
        "BMD Compute could not reach POWER to read the attempt state; retry later.",
    )


def read_remote_attempt_state(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the authoritative remote attempt state through a server-owned profile."""

    path = str((record.get("remote") or {}).get("attempt_state") or "")
    if not path:
        raise ExecutionError(500, "ledger_inconsistent", "The attempt record is incomplete.")
    try:
        with connected_remote_runner(
            profile=default_connection_profile(),
            runner_factory=RUNNER_FACTORY,
        ) as runner:
            if not runner.is_file(path):
                return None
            text = runner.read_text(path, max_bytes=REMOTE_STATE_MAX_BYTES)
    except Exception as exc:  # noqa: BLE001 - classified, never echoed
        raise _remote_unavailable(exc) from None
    try:
        state = json.loads(text)
    except (TypeError, ValueError):
        raise ExecutionError(409, "remote_state_invalid", "The remote attempt state is not readable.") from None
    if not isinstance(state, dict):
        raise ExecutionError(409, "remote_state_invalid", "The remote attempt state is not readable.")
    return state


def _require_state_matches(state: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    if str(state.get("attempt_id") or "") != record["attempt_id"]:
        raise ExecutionError(409, "attempt_fingerprint_mismatch", "The remote attempt state belongs to another attempt.")
    if str(state.get("fingerprint") or "") != record["attempt_fingerprint"]:
        raise ExecutionError(
            409,
            "attempt_fingerprint_mismatch",
            "The remote attempt was prepared with different calculation metadata.",
        )


def _remote_job_id(state: Mapping[str, Any]) -> str | None:
    job_id = str(state.get("job_id") or "")
    return job_id if _JOB_ID.fullmatch(job_id) else None


def _remote_submitted_at(state: Mapping[str, Any]) -> str | None:
    job_record = state.get("job_record") or {}
    value = str(job_record.get("submitted_at") or "")
    return value if _LOCAL_TIME.fullmatch(value) else None


# --------------------------------------------------------- ledger updates


def _update_record(ledger: AttemptLedger, attempt_id: str, mutate: Callable[[dict], None]) -> dict:
    with ledger.locked():
        record = ledger.read(attempt_id)
        if record is None:
            raise ExecutionError(500, "ledger_inconsistent", "The attempt record disappeared.")
        mutate(record)
        record["updated_at"] = isoformat(utc_now())
        ledger.replace(record)
        return record


def _record_remote_state(ledger: AttemptLedger, attempt_id: str, state: Mapping[str, Any] | None) -> dict:
    remote = (state or {}).get("state")

    def mutate(record: dict) -> None:
        submission = record["submission"]
        if remote == "PREPARED":
            record["state"] = STATE_PREPARED
        elif remote == "SUBMITTING":
            record["state"] = STATE_UNCERTAIN
            if submission["status"] in {SUBMISSION_RESERVED, SUBMISSION_NOT_REQUESTED}:
                submission["status"] = SUBMISSION_UNCERTAIN
        elif remote == "SUBMITTED":
            record["state"] = STATE_SUBMITTED
            submission["status"] = SUBMISSION_SUBMITTED
            submission["job_id"] = _remote_job_id(state)
            submission["submitted_at"] = _remote_submitted_at(state)
            if not submission.get("reserved_at"):
                # Submitted by an earlier request whose reservation was lost; count it.
                submission["reserved_at"] = isoformat(utc_now())

    return _update_record(ledger, attempt_id, mutate)


def _is_active(record: Mapping[str, Any]) -> bool:
    submission = record["submission"]
    if submission["status"] not in {SUBMISSION_RESERVED, SUBMISSION_SUBMITTED, SUBMISSION_UNCERTAIN}:
        return False
    return not (record.get("scheduler") or {}).get("terminal", False)


def _counts_toward_window(record: Mapping[str, Any], since) -> bool:
    submission = record["submission"]
    if submission["status"] == SUBMISSION_RELEASED:
        return False
    reserved_at = parse_isoformat(submission.get("reserved_at"))
    return reserved_at is not None and reserved_at >= since


def _refresh_active_records(ledger: AttemptLedger, principal: str, exclude: str) -> None:
    """Update scheduler/terminal state of the principal's other active attempts."""

    for record in ledger.records():
        if record["principal"] != principal or record["attempt_id"] == exclude or not _is_active(record):
            continue
        submission = record["submission"]
        try:
            if submission["status"] in {SUBMISSION_RESERVED, SUBMISSION_UNCERTAIN}:
                state = read_remote_attempt_state(record)
                if state is not None and state.get("state") == "SUBMITTED":
                    _require_state_matches(state, record)
                    record = _record_remote_state(ledger, record["attempt_id"], state)
            if record["submission"].get("job_id"):
                _refresh_scheduler(ledger, record)
        except ExecutionError:
            # An unverifiable attempt stays counted as active (fail closed).
            continue


def _reserve_submission(ledger: AttemptLedger, attempt_id: str, principal: str, policy: ExecutionPolicy) -> None:
    def over_caps() -> tuple[bool, bool]:
        records = [record for record in ledger.records() if record["principal"] == principal]
        since = utc_now() - policy.window
        window_count = sum(1 for record in records if _counts_toward_window(record, since))
        active_count = sum(1 for record in records if _is_active(record))
        return window_count >= policy.max_submissions_per_window, active_count >= policy.max_active_jobs

    with ledger.locked():
        record = ledger.read(attempt_id)
        if record["submission"]["status"] in {SUBMISSION_RESERVED, SUBMISSION_SUBMITTED, SUBMISSION_UNCERTAIN}:
            return
        window_full, active_full = over_caps()
    if active_full and not window_full:
        _refresh_active_records(ledger, principal, exclude=attempt_id)

    with ledger.locked():
        record = ledger.read(attempt_id)
        if record["submission"]["status"] in {SUBMISSION_RESERVED, SUBMISSION_SUBMITTED, SUBMISSION_UNCERTAIN}:
            return
        window_full, active_full = over_caps()
        if window_full:
            raise ExecutionError(
                429,
                "submission_cap_exceeded",
                f"This principal has reached {policy.max_submissions_per_window} submissions in "
                f"{int(policy.window.total_seconds() // 3600)} hours.",
            )
        if active_full:
            raise ExecutionError(
                429,
                "active_job_cap_exceeded",
                f"This principal already has {policy.max_active_jobs} active jobs.",
            )
        record["submission"]["status"] = SUBMISSION_RESERVED
        record["submission"]["reserved_at"] = isoformat(utc_now())
        record["updated_at"] = isoformat(utc_now())
        ledger.replace(record)


def _release_reservation(ledger: AttemptLedger, attempt_id: str) -> None:
    def mutate(record: dict) -> None:
        if record["submission"]["status"] == SUBMISSION_RESERVED:
            record["submission"]["status"] = SUBMISSION_RELEASED
        record["state"] = STATE_PREPARED

    _update_record(ledger, attempt_id, mutate)


# -------------------------------------------------------------- scheduler


def _normalized_slurm_state(value: Any) -> str:
    token = str(value or "").strip().split(" ", 1)[0].upper().rstrip("+")
    return token if token in _SLURM_STATES else "UNKNOWN"


def _sacct_brief_fields(brief: str, job_id: str) -> dict[str, str | None]:
    for line in str(brief or "").splitlines()[:8]:
        parts = line.split("|")
        if len(parts) >= 7 and parts[0].strip() == job_id:
            elapsed, start, end = parts[3].strip(), parts[4].strip(), parts[5].strip()
            return {
                "elapsed": elapsed if _ELAPSED.fullmatch(elapsed) else None,
                "started_at": start if _SACCT_TIME.fullmatch(start) else None,
                "ended_at": end if _SACCT_TIME.fullmatch(end) else None,
            }
    return {"elapsed": None, "started_at": None, "ended_at": None}


def _refresh_scheduler(ledger: AttemptLedger, record: Mapping[str, Any]) -> dict:
    job_id = record["submission"].get("job_id")
    if not job_id:
        return dict(record)
    result, _job_record = monitor_submitted_job(
        job_id,
        authenticated_attempt_id=record["attempt_id"],
        runner_factory=RUNNER_FACTORY,
    )
    checked_at = isoformat(utc_now())
    if result.get("status") != "success":
        scheduler = {
            "available": False,
            "summary": "UNKNOWN",
            "state": "UNKNOWN",
            "exit_code": None,
            "elapsed": None,
            "started_at": None,
            "ended_at": None,
            "terminal": bool((record.get("scheduler") or {}).get("terminal", False)),
            "checked_at": checked_at,
        }
    else:
        summary = str(result.get("summary") or "UNKNOWN")
        exit_code = str(result.get("exit_code") or "")
        scheduler = {
            "available": True,
            "summary": summary if summary in _SUMMARIES else "UNKNOWN",
            "state": _normalized_slurm_state(result.get("slurm_state")),
            "exit_code": exit_code if _EXIT_CODE.fullmatch(exit_code) else None,
            **_sacct_brief_fields(result.get("brief") or "", job_id),
            "terminal": summary in _TERMINAL_SUMMARIES,
            "checked_at": checked_at,
        }

    def mutate(stored: dict) -> None:
        stored["scheduler"] = scheduler

    return _update_record(ledger, record["attempt_id"], mutate)


# ------------------------------------------------------------- projection


def attempt_response(record: Mapping[str, Any]) -> dict[str, Any]:
    submission = record["submission"]
    scheduler = record.get("scheduler")
    return {
        "schema": ATTEMPT_RESPONSE_SCHEMA,
        "schema_version": ATTEMPT_RESPONSE_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "attempt_id": record["attempt_id"],
        "plan_digest": record["plan_digest"],
        "state": record["state"],
        "labels": dict(record.get("labels") or {}),
        "created_at": record["created_at"],
        "resources": dict(record["resources"]),
        "submission": {
            "requested": submission["status"] != SUBMISSION_NOT_REQUESTED,
            "job_id": submission.get("job_id"),
            "submitted_at_local": submission.get("submitted_at"),
        },
        "scheduler": (
            {
                key: scheduler.get(key)
                for key in ("available", "summary", "state", "exit_code", "elapsed", "started_at", "ended_at", "terminal", "checked_at")
            }
            if scheduler
            else None
        ),
    }


# ------------------------------------------------------------ operations


def _new_record(*, attempt_id, principal, request, request_digest, plan_digest, run_timestamp, submission_spec) -> dict:
    now = isoformat(utc_now())
    submission = submission_spec["submission"]
    return {
        "schema": ATTEMPT_RECORD_SCHEMA,
        "schema_version": ATTEMPT_RECORD_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "principal": principal,
        "created_at": now,
        "updated_at": now,
        "run_timestamp": run_timestamp,
        "request_sha256": request_digest,
        "plan_digest": plan_digest,
        "attempt_fingerprint": submission["attempt_fingerprint"],
        "labels": dict(request.labels),
        "resources": _effective_resources(submission_spec),
        "state": STATE_REGISTERED,
        "remote": {
            "attempt_state": submission["attempt_state"],
            "run_name": submission_spec["run_name"],
        },
        "submission": {
            "status": SUBMISSION_NOT_REQUESTED,
            "reserved_at": None,
            "job_id": None,
            "submitted_at": None,
        },
        "scheduler": None,
    }


def _authorize_record(record: Mapping[str, Any], principal: str) -> None:
    if record["principal"] != principal:
        raise ExecutionError(403, "attempt_forbidden", "This attempt belongs to another principal.")


def _bind_attempt(ledger, planner, attempt_id, principal, request, policy):
    """Return ``(record, plan)`` for an attempt bound to this principal and request."""

    request_digest = request_sha256(request)
    for _ in range(5):
        record = ledger.read(attempt_id)
        if record is not None:
            _authorize_record(record, principal)
            if record["request_sha256"] != request_digest:
                raise ExecutionError(
                    409,
                    "attempt_request_mismatch",
                    "This attempt ID is already bound to a different request.",
                )
            timestamp = record["run_timestamp"]
        else:
            timestamp = canonical_run_timestamp(None)

        plan = planner(
            structure_text=request.plan.structure_text,
            fmt=request.plan.structure_format,
            desired_output=request.plan.desired_output,
            custom_workflow=request.plan.custom_workflow,
            resources=request.plan.resources,
            timestamp=timestamp,
            submission_attempt_id=attempt_id,
        )
        spec = plan.submission_spec
        plan_digest = compute_plan_digest(
            structure=plan.structure,
            submission_spec=spec,
            stage_previews=plan_stage_previews(plan),
        )
        if plan_digest != request.expected_plan_digest:
            raise ExecutionError(
                409,
                "plan_digest_mismatch",
                "The resolved plan does not match expected_plan_digest.",
                plan_digest=plan_digest,
            )
        enforce_resource_policy(spec, policy)

        if record is not None:
            if record["plan_digest"] != plan_digest:
                raise ExecutionError(
                    409,
                    "plan_changed",
                    "The plan bound to this attempt no longer resolves identically on this service.",
                )
            if record["attempt_fingerprint"] != spec["submission"]["attempt_fingerprint"]:
                raise ExecutionError(
                    409,
                    "attempt_fingerprint_mismatch",
                    "This attempt no longer rebuilds the identical submission specification.",
                )
            return record, plan

        with ledger.locked():
            if ledger.read(attempt_id) is not None:
                continue
            records = ledger.records()
            if any(existing["run_timestamp"] == timestamp for existing in records):
                collision = True
            else:
                collision = False
                since = utc_now() - policy.window
                created = sum(
                    1
                    for existing in records
                    if existing["principal"] == principal
                    and (parse_isoformat(existing["created_at"]) or since) >= since
                )
                if created >= policy.max_new_attempts_per_window:
                    raise ExecutionError(
                        429,
                        "attempt_cap_exceeded",
                        f"This principal has created {policy.max_new_attempts_per_window} attempts in "
                        f"{int(policy.window.total_seconds() // 3600)} hours.",
                    )
                new_record = _new_record(
                    attempt_id=attempt_id,
                    principal=principal,
                    request=request,
                    request_digest=request_digest,
                    plan_digest=plan_digest,
                    run_timestamp=timestamp,
                    submission_spec=spec,
                )
                if ledger.create(new_record):
                    return new_record, plan
        if collision:
            # Run directories are named by timestamp; never share one between attempts.
            time.sleep(1.05)
    raise ExecutionError(409, "attempt_busy", "The attempt could not be registered; retry.")


def _prepare_failure(result: Mapping[str, Any]) -> ExecutionError:
    stage = str(result.get("stage") or "")
    if stage == "Remote Capacity":
        return ExecutionError(503, "remote_busy", "BMD Compute is busy with other remote operations; retry shortly.")
    if stage in {"SSH Connection", "SSH Authentication", "SSH Host Verification", "SSH Client Setup"}:
        return ExecutionError(503, "remote_unavailable", "BMD Compute could not reach POWER; retry later.")
    if stage == "Submission Attempt":
        return ExecutionError(409, "attempt_in_progress", "This attempt is being prepared or submitted by another request.")
    return ExecutionError(502, "prepare_failed", "Remote preparation failed on POWER.")


def execute_attempt(*, planner, principal: str, attempt_id: str, request: ExecutionRequest) -> dict[str, Any]:
    policy = POLICY
    ledger = _ledger()
    record, plan = _bind_attempt(ledger, planner, attempt_id, principal, request, policy)
    spec = plan.submission_spec

    state = read_remote_attempt_state(record)
    if state is None:
        result = prepare_remote_submission(spec, runner_factory=RUNNER_FACTORY)
        state = read_remote_attempt_state(record)
        if state is None:
            if result.get("status") == "success":
                raise ExecutionError(503, "remote_state_unconfirmed", "Preparation could not be confirmed; retry.")
            raise _prepare_failure(result)
    _require_state_matches(state, record)

    remote = state.get("state")
    if remote == "PREPARED":
        if _prepared_runtime_digest(state) != current_runtime_digest():
            raise ExecutionError(
                409,
                "runtime_package_changed",
                "This attempt was prepared by a different BMD Compute runtime package; "
                "it is not replaced. Use a new attempt ID.",
            )
        record = _record_remote_state(ledger, attempt_id, state)
        if not request.submit:
            return attempt_response(record)

        _reserve_submission(ledger, attempt_id, principal, policy)
        result = submit_remote_workflow(spec, remote_prepared=True, runner_factory=RUNNER_FACTORY)
        try:
            state = read_remote_attempt_state(record)
        except ExecutionError:
            raise ExecutionError(
                503,
                "submission_outcome_unconfirmed",
                "The submission outcome could not be confirmed; query the attempt before retrying.",
            ) from None
        if state is None:
            raise ExecutionError(409, "remote_state_invalid", "The remote attempt state disappeared.")
        _require_state_matches(state, record)
        remote = state.get("state")
        if remote == "PREPARED":
            if "exit_code" in result:
                # sbatch itself ran and refused the job: a definitive failure. The
                # attempt stays PREPARED and the reservation no longer counts.
                _release_reservation(ledger, attempt_id)
                raise ExecutionError(502, "submit_failed", "SLURM did not accept the submission; the attempt remains prepared.")
            # sbatch did not run (busy service, lost connection, or another
            # request holding the attempt lock). Keep the reservation: the
            # concurrent request, or a retry of this one, resolves it.
            if result.get("reason") == REMOTE_OPERATION_BUSY_MESSAGE:
                raise ExecutionError(503, "remote_busy", "BMD Compute is busy with other remote operations; retry shortly.")
            raise ExecutionError(
                409,
                "submission_not_started",
                "The submission did not start (another request may hold this attempt); query the attempt or retry.",
            )

    record = _record_remote_state(ledger, attempt_id, state)
    if remote == "SUBMITTED":
        return attempt_response(record)
    if remote == "SUBMITTING":
        raise ExecutionError(
            409,
            "submission_uncertain",
            "The SLURM submission outcome for this attempt is uncertain. It will not be resubmitted automatically.",
            attempt=attempt_response(record),
        )
    raise ExecutionError(409, "remote_state_invalid", "The remote attempt is in an unexpected state.")


def lookup_attempt(*, principal: str, attempt_id: str) -> dict[str, Any]:
    ledger = _ledger()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No machine attempt with this ID exists.")
    _authorize_record(record, principal)

    state = read_remote_attempt_state(record)
    if state is not None:
        _require_state_matches(state, record)
        if state.get("state") in {"PREPARED", "SUBMITTING", "SUBMITTED"}:
            record = _record_remote_state(ledger, attempt_id, state)
    if record["submission"].get("job_id"):
        record = _refresh_scheduler(ledger, record)
    return attempt_response(record)
