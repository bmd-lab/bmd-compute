from __future__ import annotations

import json
import logging
import posixpath
import re
import time
import traceback
from typing import Callable

from backend.remote import (
    REMOTE_OPERATION_BUSY_MESSAGE,
    RemoteExecutionError,
    RemoteJobStatus,
    RemoteOperationBusy,
    RemoteRunner,
)
from backend.config import DEFAULT_LOGS_DIR, bmd_debug_enabled
from backend.remote_runtime import (
    connected_remote_runner,
    connection_profile_from_submission_spec,
    default_connection_profile,
)
from backend.run_records import (
    RunRecordContractError,
    validate_job_record_v1,
    validate_submission_record_v1,
)


LOGGER = logging.getLogger(__name__)
MONITOR_JOB_RECORD_MAX_BYTES = 64 * 1024 * 1024


class MonitoringStageError(RuntimeError):
    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        command: str = "",
        stdout: str = "",
        stderr: str = "",
        exception: Exception | None = None,
    ):
        self.stage = stage
        self.reason = reason
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.exception = exception
        self.exception_text = str(exception) if exception else ""
        super().__init__(reason)


PENDING_STATES = {
    "PENDING",
    "CONFIGURING",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
    "STAGE_OUT",
    "RESV_DEL_HOLD",
}
RUNNING_STATES = {
    "RUNNING",
    "COMPLETING",
    "SIGNALING",
}
FAILURE_PREFIXES = (
    "BOOT_FAIL",
    "FAILED",
    "CANCELLED",
    "DEADLINE",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
)


def strip_job_id(job_id: str | None) -> str:
    return re.sub(r"\.(batch|extern)$", "", str(job_id or "").strip())


def is_valid_slurm_job_id(job_id: str | None) -> bool:
    stripped_job_id = strip_job_id(job_id)
    return bool(re.fullmatch(r"[0-9]+(?:_[0-9]+)?", stripped_job_id))


def classify_slurm_state(state: str | None, exit_code: str | None = None) -> str:
    state_upper = (state or "").upper()
    exit_text = str(exit_code or "")

    if state_upper in PENDING_STATES:
        return "PENDING"
    if state_upper in RUNNING_STATES:
        return "RUNNING"
    if state_upper.startswith("COMPLETED") and (not exit_text or exit_text.startswith("0:0")):
        return "SUCCESS"
    if state_upper.startswith(FAILURE_PREFIXES):
        return "FAILURE"
    if exit_text and not exit_text.startswith("0:0"):
        return "FAILURE"
    if state_upper:
        return "UNKNOWN"
    return "UNKNOWN"


def parse_scontrol_output(output: str) -> dict:
    def pick(pattern: str) -> str:
        match = re.search(pattern, output or "")
        return match.group(1) if match else ""

    return {
        "state": pick(r"JobState=([^ \n]+)"),
        "stdout": pick(r"StdOut=([^ \n]+)"),
        "stderr": pick(r"StdErr=([^ \n]+)"),
        "workdir": pick(r"WorkDir=([^ \n]+)"),
        "jobname": pick(r"JobName=([^ \n]+)"),
    }


def parse_sacct_row(job_id: str, output: str) -> dict:
    stripped_job_id = strip_job_id(job_id)
    wanted = {
        stripped_job_id,
        f"{stripped_job_id}.batch",
        f"{stripped_job_id}.extern",
    }

    for line in (output or "").splitlines()[:6]:
        parts = line.split("|")
        if parts and parts[0] in wanted and len(parts) >= 7:
            return {
                "state": parts[1],
                "exit": parts[2],
                "jobname": parts[3],
                "stdout": parts[4],
                "stderr": parts[5],
                "workdir": parts[6],
            }
        if parts and parts[0] in wanted and len(parts) >= 6:
            return {
                "state": parts[1],
                "exit": parts[2],
                "jobname": parts[3],
                "stdout": parts[4],
                "workdir": parts[5],
            }

    return {}


def remote_job_status_from_slurm_outputs(
    job_id: str,
    *,
    squeue_output: str = "",
    scontrol_output: str = "",
    sacct_output: str = "",
    sacct_brief_output: str = "",
    squeue_stderr: str = "",
    scontrol_stderr: str = "",
    sacct_stderr: str = "",
    sacct_brief_stderr: str = "",
    command: str = "",
) -> RemoteJobStatus:
    stripped_job_id = strip_job_id(job_id)
    try:
        scontrol_info = parse_scontrol_output(scontrol_output)
        squeue_state = _first_line(squeue_output)
        sacct_info = parse_sacct_row(stripped_job_id, sacct_output)
    except Exception as exc:
        raise MonitoringStageError(
            "parsing",
            "Failed to parse SLURM monitoring output.",
            command=command,
            stdout=_combined_output(
                squeue=squeue_output,
                scontrol=scontrol_output,
                sacct=sacct_output,
                sacct_brief=sacct_brief_output,
            ),
            stderr=_combined_output(
                squeue=squeue_stderr,
                scontrol=scontrol_stderr,
                sacct=sacct_stderr,
                sacct_brief=sacct_brief_stderr,
            ),
            exception=exc,
        ) from exc

    state = squeue_state or scontrol_info.get("state") or sacct_info.get("state") or "UNKNOWN"
    exit_code = sacct_info.get("exit") or None
    stdout_path = sacct_info.get("stdout") or scontrol_info.get("stdout") or None
    stderr_path = sacct_info.get("stderr") or scontrol_info.get("stderr") or None
    workdir = sacct_info.get("workdir") or scontrol_info.get("workdir") or None
    job_name = sacct_info.get("jobname") or scontrol_info.get("jobname") or None
    try:
        summary = classify_slurm_state(state, exit_code)
    except Exception as exc:
        raise MonitoringStageError(
            "state classification",
            "Failed to classify SLURM job state.",
            command=command,
            stdout=_combined_output(
                squeue=squeue_output,
                scontrol=scontrol_output,
                sacct=sacct_output,
                sacct_brief=sacct_brief_output,
            ),
            stderr=_combined_output(
                squeue=squeue_stderr,
                scontrol=scontrol_stderr,
                sacct=sacct_stderr,
                sacct_brief=sacct_brief_stderr,
            ),
            exception=exc,
        ) from exc
    brief = sacct_brief_output.strip() or f"{stripped_job_id}|{state}"

    return RemoteJobStatus(
        job_id=stripped_job_id,
        state=state,
        exit_code=exit_code,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        workdir=workdir,
        job_name=job_name,
        raw={
            "summary": summary,
            "brief": brief,
            "squeue": (squeue_output or "").strip(),
            "scontrol": (scontrol_output or "").strip(),
            "sacct": (sacct_output or "").strip(),
            "sacct_brief": (sacct_brief_output or "").strip(),
            "command": command,
            "squeue_stderr": (squeue_stderr or "").strip(),
            "scontrol_stderr": (scontrol_stderr or "").strip(),
            "sacct_stderr": (sacct_stderr or "").strip(),
            "sacct_brief_stderr": (sacct_brief_stderr or "").strip(),
            "updated_at": time.time(),
        },
    )


def monitor_job(
    job_id: str,
    *,
    submission_spec: dict | None = None,
    runner_factory: Callable[[], RemoteRunner] | None = None,
) -> dict:
    stripped_job_id = strip_job_id(job_id)
    if not stripped_job_id:
        return _failure_result(
            "Job ID",
            "No SLURM job ID is available for monitoring.",
            "Submit the calculation successfully before checking queue status.",
        )

    is_resume = submission_spec is None
    if not is_valid_slurm_job_id(stripped_job_id):
        return _failure_result(
            "Job ID",
            "Malformed SLURM job ID.",
            "Enter the numeric SLURM job ID returned by sbatch.",
            exception_text=str(job_id or ""),
        )

    profile = (
        connection_profile_from_submission_spec(submission_spec)
        if submission_spec is not None
        else default_connection_profile()
    )

    try:
        with connected_remote_runner(
            profile=profile,
            runner_factory=runner_factory,
        ) as runner:
            status = runner.query_job(stripped_job_id)
    except Exception as exc:
        stage = "SSH Connection" if _looks_like_connection_failure(exc) else "Monitoring"
        LOGGER.exception("Remote monitoring failed at %s.", stage)
        result = _exception_result(exc, default_stage=stage)
        if not isinstance(exc, RemoteOperationBusy):
            result["exception_debug"] = _exception_debug(exc)
        return result

    if is_resume and _is_unknown_job(status):
        return _unknown_job_result(status)

    return _success_result(status)


def monitor_submitted_job(
    job_id: str,
    *,
    authenticated_attempt_id: str,
    runner_factory: Callable[[], RemoteRunner] | None = None,
) -> tuple[dict, dict | None]:
    """Monitor a submitted job after binding it to trusted remote state.

    Caller-carried submission state is intentionally not accepted here. The
    connection profile and job-record path are both derived from server policy.
    """

    stripped_job_id = strip_job_id(job_id)
    if not is_valid_slurm_job_id(stripped_job_id):
        return (
            _failure_result(
                "Job ID",
                "Malformed SLURM job ID.",
                "Use the numeric SLURM job ID returned by sbatch.",
                exception_text=str(job_id or ""),
            ),
            None,
        )

    authoritative_record = None
    try:
        with connected_remote_runner(
            profile=default_connection_profile(),
            runner_factory=runner_factory,
        ) as runner:
            authoritative_record = _authenticated_job_record(
                runner,
                stripped_job_id,
                authenticated_attempt_id,
            )
            status = runner.query_job(stripped_job_id)
    except Exception as exc:
        stage = "SSH Connection" if _looks_like_connection_failure(exc) else "Monitoring"
        LOGGER.exception("Authenticated remote monitoring failed at %s.", stage)
        result = _exception_result(exc, default_stage=stage)
        result["job_id"] = stripped_job_id
        result["identity_retry_allowed"] = (
            authoritative_record is None
            and (
                isinstance(exc, RemoteOperationBusy)
                or _looks_like_connection_failure(exc)
            )
        )
        if not isinstance(exc, RemoteOperationBusy):
            result["exception_debug"] = _exception_debug(exc)
        return result, authoritative_record

    return _success_result(status), authoritative_record


def _authenticated_job_record(
    runner: RemoteRunner,
    job_id: str,
    authenticated_attempt_id: str,
) -> dict:
    state_path = posixpath.join(DEFAULT_LOGS_DIR, f"job_{job_id}.json")
    if not runner.is_file(state_path):
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The authoritative BMD job record is missing.",
        )

    try:
        payload = json.loads(
            runner.read_text(state_path, max_bytes=MONITOR_JOB_RECORD_MAX_BYTES)
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The authoritative BMD job record is malformed.",
            exception=exc,
        ) from exc

    try:
        validate_job_record_v1(payload)
    except RunRecordContractError as exc:
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The authoritative BMD job record failed contract validation.",
            exception=exc,
        ) from exc

    if strip_job_id(payload["job_id"]) != job_id:
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The authoritative BMD job record does not match the requested job.",
        )
    if payload["attempt_id"] != authenticated_attempt_id:
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The submitted job does not belong to the authenticated submission attempt.",
        )

    submission_spec = payload.get("submission_spec")
    try:
        validate_submission_record_v1(submission_spec)
    except RunRecordContractError as exc:
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The authoritative submission record failed contract validation.",
            exception=exc,
        ) from exc

    submission = submission_spec["submission"]
    if submission["attempt_id"] != authenticated_attempt_id:
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The authoritative submission record has a different attempt identity.",
        )
    if submission_spec["paths"]["run_dir"] != payload["run_dir"]:
        raise MonitoringStageError(
            "Monitoring Authorization",
            "The authoritative job and submission records disagree on the run directory.",
        )

    return payload


def _success_result(status: RemoteJobStatus) -> dict:
    summary = status.raw.get("summary") or classify_slurm_state(status.state, status.exit_code)
    return {
        "status": "success",
        "title": "Monitoring",
        "stage": "Monitoring",
        "job_id": status.job_id,
        "slurm_state": status.state,
        "summary": summary,
        "exit_code": status.exit_code,
        "stdout_path": status.stdout_path,
        "stderr_path": status.stderr_path,
        "workdir": status.workdir,
        "job_name": status.job_name,
        "brief": status.raw.get("brief", ""),
        "updated_at": status.raw.get("updated_at"),
        "job_status": status,
    }


def _exception_result(exc: Exception, *, default_stage: str = "Monitoring") -> dict:
    if isinstance(exc, RemoteOperationBusy):
        return _failure_result(
            "Remote Capacity",
            REMOTE_OPERATION_BUSY_MESSAGE,
            "Please try again in a few seconds.",
            exception_text=str(exc),
        )

    if isinstance(exc, MonitoringStageError):
        return _failure_result(
            exc.stage,
            _clean_message(exc.reason),
            _suggestion_for_stage(exc.stage),
            command=exc.command,
            stdout=exc.stdout,
            stderr=exc.stderr,
            exception_text=exc.exception_text,
        )

    if isinstance(exc, RemoteExecutionError):
        output = "\n".join(
            item
            for item in (
                exc.result.stdout or "",
                exc.result.stderr or "",
                str(exc),
            )
            if item
        )
        reason = (
            _clean_message(output)
            if bmd_debug_enabled()
            else "BMD Compute could not check the queue status."
        )
        return _failure_result(
            default_stage,
            reason,
            "Check SLURM command availability and the remote cluster connection.",
            command=exc.result.command,
            stdout=exc.result.stdout,
            stderr=exc.result.stderr,
            exception_text=str(exc),
        )

    class_name = exc.__class__.__name__
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) == "paramiko":
        return _failure_result(
            "SSH Client Setup",
            "Paramiko is not installed in this Python environment.",
            "Install the project environment dependencies and try again.",
            exception_text=str(exc),
        )
    if "Authentication" in class_name or class_name in {"BadAuthenticationType", "PasswordRequiredException"}:
        return _failure_result(
            "SSH Authentication",
            "SSH authentication failed.",
            "Check SSH key or agent access and try again.",
            exception_text=str(exc),
        )
    if default_stage == "SSH Connection" and _looks_like_connection_failure(exc):
        return _failure_result(
            "SSH Connection",
            "Unable to connect to the remote cluster.",
            "Connect to the TAU VPN and try again.",
            exception_text=str(exc),
        )

    return _failure_result(
        default_stage,
        (
            _clean_message(str(exc) or class_name)
            if bmd_debug_enabled()
            else "BMD Compute could not check the queue status."
        ),
        "Check the remote cluster environment and try again.",
        exception_text=str(exc) or class_name,
    )


def _failure_result(
    stage: str,
    reason: str,
    suggestion: str,
    *,
    command: str = "",
    stdout: str = "",
    stderr: str = "",
    exception_text: str = "",
) -> dict:
    return {
        "status": "failed",
        "title": "Monitoring Failed",
        "stage": stage,
        "reason": reason,
        "suggestion": suggestion,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "exception_text": exception_text,
    }


def _exception_debug(exc: Exception) -> dict:
    return {
        "type": str(type(exc)),
        "module": exc.__class__.__module__,
        "class_name": exc.__class__.__name__,
        "repr": repr(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _is_unknown_job(status: RemoteJobStatus) -> bool:
    summary = str(status.raw.get("summary") or "").upper()
    state = str(status.state or "").upper()
    return summary == "UNKNOWN" or state == "UNKNOWN"


def _unknown_job_result(status: RemoteJobStatus) -> dict:
    return _failure_result(
        "Job lookup",
        f"No SLURM record was found for job ID {status.job_id}.",
        "Check the job ID and try again. Completed jobs may disappear from accounting after the cluster retention window.",
        command=status.raw.get("command", ""),
        stdout=_combined_output(
            squeue=status.raw.get("squeue", ""),
            scontrol=status.raw.get("scontrol", ""),
            sacct=status.raw.get("sacct", ""),
            sacct_brief=status.raw.get("sacct_brief", ""),
        ),
        stderr=_combined_output(
            squeue=status.raw.get("squeue_stderr", ""),
            scontrol=status.raw.get("scontrol_stderr", ""),
            sacct=status.raw.get("sacct_stderr", ""),
            sacct_brief=status.raw.get("sacct_brief_stderr", ""),
        ),
    )


def _first_line(output: str) -> str:
    for line in (output or "").splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def _clean_message(message: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(message or "").strip())
    return (cleaned or "Monitoring failed.")[:360]


def _combined_output(**items: str) -> str:
    return "\n".join(
        f"{name}:\n{value}"
        for name, value in items.items()
        if value
    )


def _suggestion_for_stage(stage: str) -> str:
    stage_lower = str(stage or "").lower()
    if "ssh" in stage_lower:
        return "Check SSH key, agent, VPN, and cluster login access."
    if stage_lower in {"squeue", "scontrol", "sacct"}:
        return "Inspect the command output and verify SLURM command availability for the submitted job."
    if stage_lower == "parsing":
        return "Inspect the raw scheduler output; the monitor could not parse it."
    if stage_lower == "state classification":
        return "Inspect the raw scheduler state and update the classifier if SLURM returned a new state."
    return "Check the remote cluster environment and try again."


def _looks_like_connection_failure(exc: Exception) -> bool:
    class_name = exc.__class__.__name__
    if class_name in {"NoValidConnectionsError", "TimeoutError", "gaierror"}:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    message = _clean_message(exc).lower()
    return any(
        fragment in message
        for fragment in (
            "ssh transport",
            "remote client is not connected",
            "timed out",
            "connection refused",
            "connection reset",
            "name or service not known",
            "nodename nor servname",
            "network is unreachable",
            "no route to host",
            "unable to connect",
        )
    )


__all__ = [
    "classify_slurm_state",
    "is_valid_slurm_job_id",
    "monitor_job",
    "monitor_submitted_job",
    "MonitoringStageError",
    "parse_sacct_row",
    "parse_scontrol_output",
    "remote_job_status_from_slurm_outputs",
    "strip_job_id",
]
