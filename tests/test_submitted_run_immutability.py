"""Submitted runs are immutable: Prepare may never rewrite their run artifacts.

Every rejection below is checked against a byte-for-byte snapshot of the whole
simulated remote host (all files, including submission.json, the uploaded
backend runtime package, run_job.py, the sbatch script and the attempt state,
plus the directory set), not just the attempt-state value.
"""

from __future__ import annotations

import hashlib
import json

import pytest

import backend.paramiko_remote as paramiko_remote
from backend.remote import (
    RemoteCommandResult,
    RemoteExecutionError,
    SubmissionAttemptAlreadySubmitted,
    SubmissionAttemptInProgress,
    SubmissionAttemptMismatch,
)
from backend.remote_preparation import prepare_remote_submission
from test_submission_idempotency import (
    IdempotencyRunner,
    SharedRemoteState,
    _attempt_state,
    _prepare,
    _submission_spec,
)


class ConnectedIdempotencyRunner(IdempotencyRunner):
    """Idempotency runner usable through the route-level connection wrapper."""

    def connect(self, profile):
        return None

    def close(self):
        return None


def _snapshot(shared: SharedRemoteState) -> dict:
    with shared.lock:
        return {
            "files": {
                path: hashlib.sha256(text.encode("utf-8")).hexdigest()
                for path, text in shared.files.items()
            },
            "directories": sorted(shared.directories),
        }


def _run_artifacts(shared: SharedRemoteState, spec: dict) -> dict:
    run_dir = spec["paths"]["run_dir"].rstrip("/") + "/"
    return {
        path: text
        for path, text in shared.files.items()
        if path.startswith(run_dir) or path == spec["paths"]["remote_script"]
    }


def _submitted(shared: SharedRemoteState, **kwargs) -> tuple[dict, object]:
    spec = _submission_spec(**kwargs)
    spec["provenance"]["bmd_compute"]["source"]["git_commit"] = "original"
    _prepare(shared, spec)
    record = IdempotencyRunner(shared).submit(spec)
    return spec, record


def _same_attempt(spec: dict, **kwargs) -> dict:
    later = _submission_spec(attempt_id=spec["submission"]["attempt_id"], **kwargs)
    # A later Prepare from an updated server would carry different provenance.
    later["provenance"]["bmd_compute"]["source"]["git_commit"] = "newer"
    return later


def test_run_artifacts_that_must_stay_immutable_are_all_uploaded():
    shared = SharedRemoteState()
    spec, _ = _submitted(shared)
    artifacts = _run_artifacts(shared, spec)
    run_dir = spec["paths"]["run_dir"]

    assert f"{run_dir}/submission.json" in artifacts
    assert f"{run_dir}/run_job.py" in artifacts
    assert f"{run_dir}/backend/__init__.py" in artifacts
    assert f"{run_dir}/backend/execution.py" in artifacts
    assert f"{run_dir}/backend/workflows.py" in artifacts
    assert spec["paths"]["remote_script"] in artifacts


def test_prepare_after_submission_is_rejected_before_any_byte_changes():
    shared = SharedRemoteState()
    spec, record = _submitted(shared)
    before = _snapshot(shared)

    with pytest.raises(SubmissionAttemptAlreadySubmitted) as excinfo:
        IdempotencyRunner(shared).submit(_same_attempt(spec), dry_run=True)

    assert excinfo.value.job_id == record.job_id
    assert record.job_id in str(excinfo.value)
    assert _snapshot(shared) == before
    uploaded = json.loads(shared.files[f"{spec['paths']['run_dir']}/submission.json"])
    assert uploaded["provenance"]["bmd_compute"]["source"]["git_commit"] == "original"
    assert _attempt_state(shared, spec)["state"] == "SUBMITTED"
    assert shared.sbatch_calls == 1


def test_identical_prepare_after_submission_is_also_rejected():
    shared = SharedRemoteState()
    spec, _ = _submitted(shared)
    before = _snapshot(shared)

    with pytest.raises(SubmissionAttemptAlreadySubmitted):
        IdempotencyRunner(shared).submit(spec, dry_run=True)

    assert _snapshot(shared) == before


def test_prepare_route_reports_the_rejection_and_changes_nothing():
    shared = SharedRemoteState()
    spec, record = _submitted(shared)
    before = _snapshot(shared)

    result = prepare_remote_submission(
        _same_attempt(spec),
        runner_factory=lambda: ConnectedIdempotencyRunner(shared),
    )

    assert result["status"] == "failed"
    assert result["ready_for_submission"] is False
    assert result["stage"] == "Submission Attempt"
    assert record.job_id in result["reason"]
    assert "new submission attempt" in result["suggestion"]
    # Steps after the attempt check are not reported as completed.
    assert all(
        step["state"] != "complete"
        for step in result["steps"]
        if step["label"] in {"submission.json uploaded", "Ready for submission"}
    )
    assert _snapshot(shared) == before


def test_duplicate_submit_after_rejected_prepare_reuses_the_job():
    shared = SharedRemoteState()
    spec, record = _submitted(shared)
    with pytest.raises(SubmissionAttemptAlreadySubmitted):
        IdempotencyRunner(shared).submit(_same_attempt(spec), dry_run=True)

    duplicate = IdempotencyRunner(shared).submit(spec)

    assert duplicate.job_id == record.job_id
    assert "BMD_ALREADY_SUBMITTED=1" in duplicate.raw_output
    assert shared.sbatch_calls == 1


def test_prepare_of_an_ambiguous_submitting_attempt_is_rejected(monkeypatch):
    monkeypatch.setattr(paramiko_remote, "SUBMISSION_ATTEMPT_STATE_WAIT_S", 0.05)
    monkeypatch.setattr(paramiko_remote, "SUBMISSION_ATTEMPT_STATE_POLL_S", 0.01)
    shared = SharedRemoteState()
    shared.submit_exception = TimeoutError("connection lost after sbatch")
    spec = _submission_spec()
    _prepare(shared, spec)
    with pytest.raises(SubmissionAttemptInProgress):
        IdempotencyRunner(shared).submit(spec)
    assert _attempt_state(shared, spec)["state"] == "SUBMITTING"
    before = _snapshot(shared)

    with pytest.raises(SubmissionAttemptInProgress):
        IdempotencyRunner(shared).submit(spec, dry_run=True)

    assert _snapshot(shared) == before
    # Ambiguous-submission safety is unchanged: still no second sbatch.
    with pytest.raises(SubmissionAttemptInProgress):
        IdempotencyRunner(shared).submit(spec)
    assert shared.sbatch_calls == 1


def test_mismatched_reprepare_of_a_prepared_attempt_is_rejected_unchanged():
    shared = SharedRemoteState()
    spec = _submission_spec(ntasks=24)
    _prepare(shared, spec)
    before = _snapshot(shared)

    with pytest.raises(SubmissionAttemptMismatch):
        IdempotencyRunner(shared).submit(
            _submission_spec(attempt_id=spec["submission"]["attempt_id"], ntasks=48),
            dry_run=True,
        )

    assert _snapshot(shared) == before


def test_prepare_while_another_request_holds_the_attempt_is_rejected_unchanged():
    shared = SharedRemoteState()
    spec = _submission_spec()
    _prepare(shared, spec)
    shared.directories.add(spec["paths"]["submission_attempt_lock"])
    before = _snapshot(shared)

    with pytest.raises(SubmissionAttemptInProgress):
        IdempotencyRunner(shared).submit(spec, dry_run=True)

    assert _snapshot(shared) == before


def test_another_attempt_cannot_overwrite_a_submitted_run_directory():
    shared = SharedRemoteState()
    spec, _ = _submitted(shared)
    # Different attempt that happens to resolve to the same run directory.
    colliding = _submission_spec()
    assert colliding["submission"]["attempt_id"] != spec["submission"]["attempt_id"]
    assert colliding["paths"]["run_dir"] == spec["paths"]["run_dir"]
    before = _snapshot(shared)

    with pytest.raises(SubmissionAttemptMismatch):
        IdempotencyRunner(shared).submit(colliding, dry_run=True)

    assert _snapshot(shared) == before


# --- Legitimate re-preparation and retries remain possible ----------------------


def test_prepared_but_unsubmitted_attempt_may_be_prepared_again():
    shared = SharedRemoteState()
    spec = _submission_spec()
    spec["provenance"]["bmd_compute"]["source"]["git_commit"] = "original"
    _prepare(shared, spec)

    later = _same_attempt(spec)
    record = IdempotencyRunner(shared).submit(later, dry_run=True)

    assert record.status == "dry_run"
    uploaded = json.loads(shared.files[f"{spec['paths']['run_dir']}/submission.json"])
    assert uploaded["provenance"]["bmd_compute"]["source"]["git_commit"] == "newer"
    assert _attempt_state(shared, spec)["state"] == "PREPARED"
    assert spec["paths"]["submission_attempt_lock"] not in shared.directories

    submitted = IdempotencyRunner(shared).submit(later)
    assert submitted.job_id
    assert shared.sbatch_calls == 1


def test_attempt_rejected_by_sbatch_can_be_prepared_again_and_submitted():
    shared = SharedRemoteState()
    spec = _submission_spec()
    _prepare(shared, spec)
    shared.submit_error_result = RemoteCommandResult(
        command="sbatch --parsable run.sh",
        returncode=1,
        stderr="sbatch: error: invalid account",
    )
    with pytest.raises(RemoteExecutionError):
        IdempotencyRunner(shared).submit(spec)
    assert _attempt_state(shared, spec)["state"] == "PREPARED"

    assert IdempotencyRunner(shared).submit(spec, dry_run=True).status == "dry_run"
    retried = IdempotencyRunner(shared).submit(spec)

    assert retried.job_id
    assert shared.sbatch_calls == 2


def test_new_attempt_after_a_submitted_run_prepares_and_submits_separately():
    shared = SharedRemoteState()
    first_spec, first = _submitted(shared)
    first_artifacts = _run_artifacts(shared, first_spec)

    retry_spec = _submission_spec(timestamp="20260629-130000")
    assert retry_spec["paths"]["run_dir"] != first_spec["paths"]["run_dir"]
    _prepare(shared, retry_spec)
    retry = IdempotencyRunner(shared).submit(retry_spec)

    assert retry.job_id != first.job_id
    assert shared.sbatch_calls == 2
    assert _run_artifacts(shared, first_spec) == first_artifacts


def test_failed_upload_releases_the_attempt_claim_and_allows_a_retry():
    shared = SharedRemoteState()
    spec = _submission_spec()

    class FailingUploadRunner(IdempotencyRunner):
        def put_text(self, remote_path, text, *, mode=0o640):
            if remote_path.endswith("/run_job.py"):
                raise OSError("upload interrupted")
            return super().put_text(remote_path, text, mode=mode)

    with pytest.raises(Exception):
        FailingUploadRunner(shared).submit(spec, dry_run=True)
    assert spec["paths"]["submission_attempt_lock"] not in shared.directories
    assert spec["paths"]["submission_attempt_state"] not in shared.files

    _prepare(shared, spec)
    assert IdempotencyRunner(shared).submit(spec).job_id
