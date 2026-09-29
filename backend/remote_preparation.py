from __future__ import annotations

import re
import time
from typing import Callable

from backend.submission import REMOTE_RUNTIME_PREFLIGHT_STEP
from backend.remote import (
    JobRecord,
    REMOTE_OPERATION_BUSY_MESSAGE,
    RemoteExecutionError,
    RemoteOperationBusy,
    RemoteRunner,
    SubmissionAttemptError,
)
from backend.remote_runtime import (
    connected_remote_runner,
    connection_profile_from_cluster,
    connection_profile_from_submission_spec,
    default_connection_profile,
)


SUCCESS_STEPS = [
    "Remote connection established",
    "Remote preflight checks completed",
    "Remote directories prepared",
    "Working directory created",
    "submission.json uploaded",
    "Execution module uploaded",
    "run_job.py uploaded",
    REMOTE_RUNTIME_PREFLIGHT_STEP,
    "POTCAR links prepared",
    "Submission script written",
    "Ready for submission",
]

REMOTE_STATE_STEPS = [
    "Remote directories prepared",
    "Working directory created",
    "submission.json uploaded",
    "Execution module uploaded",
    "run_job.py uploaded",
    REMOTE_RUNTIME_PREFLIGHT_STEP,
    "POTCAR links prepared",
    "Submission script written",
    "Ready for submission",
]


def prepare_remote_submission(
    submission_spec: dict,
    *,
    runner_factory: Callable[[], RemoteRunner] | None = None,
    initial_diagnostics: dict[str, int | float] | None = None,
) -> dict:
    """
    Execute the existing remote dry-run path and return template-ready status.

    This performs no real submission. It connects to the remote host, calls the
    RemoteRunner dry-run submission path, and reports failures without exposing
    Python tracebacks to the browser.
    """

    profile = connection_profile_from_submission_spec(submission_spec)
    stage = "SSH Connection"
    started = time.perf_counter()
    diagnostics = dict(initial_diagnostics or {})
    runner = None

    try:
        with connected_remote_runner(
            profile=profile,
            runner_factory=runner_factory,
            metrics=diagnostics,
        ) as runner:
            stage = "Remote Preparation"
            record = runner.submit(submission_spec, dry_run=True)
    except Exception as exc:
        _merge_runner_diagnostics(diagnostics, runner)
        _finalize_diagnostics(diagnostics, started)
        return _failure_result(exc, stage, submission_spec, diagnostics=diagnostics)

    _merge_runner_diagnostics(diagnostics, runner)
    _finalize_diagnostics(diagnostics, started)
    return _success_result(record, submission_spec, diagnostics=diagnostics)


def remembered_successful_preparation(submission_spec: dict) -> dict:
    """
    Recreate the template context for a remote preparation that already
    completed successfully in the progressive browser flow.
    """

    paths = submission_spec["paths"]
    record = JobRecord(
        job_id=None,
        run_name=submission_spec["run_name"],
        run_dir=paths["run_dir"],
        remote_script=paths["remote_script"],
        log_paths={
            "stdout": paths["log_out"],
            "stderr": paths["log_err"],
            "slurm_out": paths["slurm_out"],
            "slurm_err": paths["slurm_err"],
        },
        cluster=dict(submission_spec["cluster"]),
        resources=dict(submission_spec["resources"]),
        submitted_at="",
        raw_output="\n".join(
            f"PREP_OK={step}"
            for step in _required_remote_state_steps(submission_spec)
        ) + "\n",
        status="dry_run",
    )
    return _success_result(record, submission_spec)


def _success_result(
    record: JobRecord,
    submission_spec: dict,
    *,
    diagnostics: dict[str, int | float] | None = None,
) -> dict:
    verified = _verified_steps(record.raw_output)
    success_steps = _success_steps(submission_spec)
    missing = [
        step
        for step in _required_remote_state_steps(submission_spec)
        if step not in verified
    ]

    if missing:
        result = {
            "status": "failed",
            "title": "Remote Preparation Failed",
            "stage": missing[0],
            "reason": f"Remote dry run completed without verifying: {missing[0]}.",
            "suggestion": "Check the remote preparation output and verify the configured paths.",
            "steps": _failure_steps(missing[0], submission_spec),
            "ready_for_submission": False,
        }
        if diagnostics:
            result["diagnostics"] = diagnostics
        return result

    result = {
        "status": "success",
        "title": "Remote Preparation Complete",
        "steps": [
            {
                "label": label,
                "state": "complete",
            }
            for label in success_steps
        ],
        "job_record": record.to_dict(),
        "ready_for_submission": True,
    }
    if diagnostics:
        result["diagnostics"] = diagnostics
    return result


def _failure_result(
    exc: Exception,
    stage: str,
    submission_spec: dict,
    *,
    diagnostics: dict[str, int | float] | None = None,
) -> dict:
    resolved_stage, reason, suggestion = _classify_failure(exc, stage, submission_spec)
    result = {
        "status": "failed",
        "title": "Remote Preparation Failed",
        "stage": resolved_stage,
        "reason": reason,
        "suggestion": suggestion,
        "steps": _failure_steps(resolved_stage, submission_spec),
        "ready_for_submission": False,
    }
    if diagnostics:
        result["diagnostics"] = diagnostics
    return result


def _merge_runner_diagnostics(diagnostics: dict, runner: RemoteRunner | None) -> None:
    runner_metrics = getattr(runner, "preparation_metrics", None)
    if isinstance(runner_metrics, dict):
        diagnostics.update(runner_metrics)


def _finalize_diagnostics(diagnostics: dict, started: float) -> None:
    diagnostics["remote_service_s"] = time.perf_counter() - started
    local_reconstruction_s = float(diagnostics.get("local_reconstruction_s", 0.0))
    diagnostics["local_reconstruction_package_s"] = (
        local_reconstruction_s
        + float(diagnostics.get("runtime_package_build_s", 0.0))
    )
    diagnostics["total_preparation_s"] = (
        local_reconstruction_s + diagnostics["remote_service_s"]
    )


def _required_remote_state_steps(submission_spec: dict) -> list[str]:
    steps = [
        "Remote directories prepared",
        "Working directory created",
        "submission.json uploaded",
        "Execution module uploaded",
        "run_job.py uploaded",
        REMOTE_RUNTIME_PREFLIGHT_STEP,
    ]

    if submission_spec.get("potcar", {}).get("symlink_targets"):
        steps.append("POTCAR links prepared")

    steps.extend([
        "Submission script written",
        "Ready for submission",
    ])
    return steps


def _success_steps(submission_spec: dict) -> list[str]:
    if submission_spec.get("potcar", {}).get("symlink_targets"):
        return list(SUCCESS_STEPS)

    return [
        step
        for step in SUCCESS_STEPS
        if step != "POTCAR links prepared"
    ]


def _verified_steps(output: str) -> set[str]:
    verified = set()
    for line in (output or "").splitlines():
        if line.startswith("PREP_OK="):
            verified.add(line.split("=", 1)[1].strip())
    return verified


def _failure_steps(failed_stage: str, submission_spec: dict) -> list[dict]:
    steps = []
    failed_step = _display_step_for_stage(failed_stage)
    failed_seen = False

    for label in _success_steps(submission_spec):
        if label == failed_step:
            steps.append({"label": label, "state": "failed"})
            failed_seen = True
            break

        steps.append({"label": label, "state": "complete"})

    if not failed_seen:
        steps.append({"label": failed_stage, "state": "failed"})

    return steps


def _display_step_for_stage(stage: str) -> str:
    if stage in {"SSH Client Setup", "SSH Connection", "SSH Authentication"}:
        return "Remote connection established"
    if stage in {"Remote Preflight", "Submission Attempt"}:
        return "Remote preflight checks completed"
    if stage == "Remote Preparation":
        return "Remote directories prepared"
    return stage


def _classify_failure(
    exc: Exception,
    stage: str,
    submission_spec: dict,
) -> tuple[str, str, str]:
    host = submission_spec.get("cluster", {}).get("remote_host", "the remote cluster")
    username = submission_spec.get("cluster", {}).get("username", "the configured user")
    class_name = exc.__class__.__name__

    if isinstance(exc, RemoteOperationBusy):
        return (
            "Remote Capacity",
            REMOTE_OPERATION_BUSY_MESSAGE,
            "Please try again in a few seconds.",
        )

    if isinstance(exc, SubmissionAttemptError):
        return (
            "Submission Attempt",
            _clean_message(exc),
            exc.suggestion,
        )

    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) == "paramiko":
        return (
            "SSH Client Setup",
            "Paramiko is not installed in this Python environment.",
            "Install the project environment dependencies and try again.",
        )

    if _looks_like_host_key_failure(exc):
        return (
            "SSH Host Verification",
            "SSH host identity verification failed.",
            "Verify the deployment known-hosts entry for the configured cluster host.",
        )

    if "Authentication" in class_name or class_name in {"BadAuthenticationType", "PasswordRequiredException"}:
        return (
            "SSH Authentication",
            "SSH authentication failed.",
            f"Check SSH key or agent access for {username} and try again.",
        )

    if _looks_like_connection_failure(exc):
        return (
            "SSH Connection",
            f"Unable to connect to {host}.",
            "Connect to the TAU VPN and try again.",
        )

    if isinstance(exc, FileNotFoundError):
        return (
            "Remote Preflight",
            _clean_message(exc),
            "Check that the referenced remote path exists and try again.",
        )

    if isinstance(exc, RemoteExecutionError):
        return _remote_execution_failure(exc)

    return (
        stage,
        _clean_message(exc),
        "Check the remote environment and try again.",
    )


def _remote_execution_failure(exc: RemoteExecutionError) -> tuple[str, str, str]:
    combined = _combined_remote_output(exc)
    marker_stage = _marker_value(combined, "PREP_FAILED_STAGE")
    marker_reason = _marker_value(combined, "PREP_FAILED_REASON")

    if marker_stage:
        return (
            marker_stage,
            marker_reason or "Remote preparation verification failed.",
            _suggestion_for_stage(marker_stage),
        )

    message = _clean_message(exc)
    lowered = message.lower()

    if "permission denied" in lowered:
        return (
            "Remote Preparation",
            "Remote directory or file creation was denied.",
            "Check permissions for the configured flows, logs, and POTCAR directories.",
        )

    if "no such file" in lowered or "not found" in lowered:
        return (
            "Remote Preparation",
            "A required remote path was not found.",
            "Check the configured remote environment and POTCAR paths.",
        )

    if "quota" in lowered or "no space left" in lowered:
        return (
            "Remote Preparation",
            "The remote filesystem could not accept the prepared files.",
            "Check quota or available space on the remote filesystem.",
        )

    return (
        "Remote Preparation",
        message,
        "Check the remote directory configuration and try again.",
    )


def _combined_remote_output(exc: RemoteExecutionError) -> str:
    result = exc.result
    return "\n".join(
        item
        for item in (
            result.stdout or "",
            result.stderr or "",
            str(exc),
        )
        if item
    )


def _marker_value(output: str, marker: str) -> str | None:
    prefix = f"{marker}="
    for line in (output or "").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def _suggestion_for_stage(stage: str) -> str:
    if stage in {"Remote directories prepared", "Working directory created"}:
        return "Check permissions and available space for the configured remote working directories."
    if stage in {"submission.json uploaded", "Execution module uploaded", "run_job.py uploaded"}:
        return "Check write permissions for the remote run directory."
    if stage == REMOTE_RUNTIME_PREFLIGHT_STEP:
        return "Check the uploaded BMD Compute runtime package and the configured remote Python environment."
    if stage == "POTCAR links prepared":
        return "Check the configured POTCAR directory and whether existing paths can be replaced by symlinks."
    if stage == "Submission script written":
        return "Check write permissions for the configured flows directory."
    return "Check the remote environment and try again."


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
        )
    )


def _looks_like_host_key_failure(exc: Exception) -> bool:
    class_name = exc.__class__.__name__
    message = str(exc).lower()
    return class_name in {"BadHostKeyException", "SshHostKeyTrustError"} or any(
        fragment in message
        for fragment in (
            "not found in known_hosts",
            "known-hosts file",
            "host key for server",
            "host key does not match",
        )
    )


def _clean_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    message = re.sub(r"\s+", " ", message)
    return message[:320]


__all__ = [
    "REMOTE_STATE_STEPS",
    "SUCCESS_STEPS",
    "connection_profile_from_cluster",
    "connection_profile_from_submission_spec",
    "default_connection_profile",
    "remembered_successful_preparation",
    "prepare_remote_submission",
]
