from __future__ import annotations

import json
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import backend.paramiko_remote as paramiko_remote
from backend.paramiko_remote import ParamikoRemoteRunner
from backend.remote import (
    BatchSubmissionResult,
    RemoteCommandResult,
    RemoteExecutionError,
    RemotePathInfo,
    SubmissionAttemptInProgress,
    SubmissionAttemptMismatch,
)
from backend.submission import create_submission_spec


FLOW_SPEC = {
    "workflow": "static",
    "potcar_functional": "PBE_64",
    "kpoints": None,
    "incar": {},
    "structure": {
        "type": "pasted_text",
        "format": "poscar",
        "text": "placeholder",
    },
}


class SharedRemoteState:
    def __init__(self):
        self.lock = threading.Lock()
        self.files = {}
        self.directories = set()
        self.sbatch_calls = 0
        self.next_job_id = 920000
        self.submit_started = threading.Event()
        self.allow_submit = threading.Event()
        self.block_submit = False
        self.submit_exception = None
        self.submit_error_result = None
        self.fail_next_submitting_state_write = False


class IdempotencyRunner(ParamikoRemoteRunner):
    def __init__(self, shared: SharedRemoteState):
        super().__init__(client=object())
        self.shared = shared
        self.commands = []

    def ensure_available(self):
        return None

    def run(
        self,
        command,
        *,
        check=False,
        modules=False,
        export_env=False,
        timeout_s=None,
    ):
        self.commands.append(command)
        parts = shlex.split(command)
        if len(parts) == 2 and parts[0] == "mkdir":
            path = parts[1]
            with self.shared.lock:
                if path in self.shared.directories:
                    return RemoteCommandResult(command=command, returncode=1)
                self.shared.directories.add(path)
            return RemoteCommandResult(command=command, returncode=0)

        if len(parts) == 2 and parts[0] == "rmdir":
            path = parts[1]
            with self.shared.lock:
                if path not in self.shared.directories:
                    return RemoteCommandResult(command=command, returncode=1)
                self.shared.directories.remove(path)
            return RemoteCommandResult(command=command, returncode=0)

        return RemoteCommandResult(command=command, returncode=0)

    def ensure_directory(self, remote_path):
        with self.shared.lock:
            self.shared.directories.add(remote_path)
        return RemotePathInfo(path=remote_path, exists=True, kind="dir")

    def is_dir(self, remote_path):
        with self.shared.lock:
            return remote_path in self.shared.directories

    def is_file(self, remote_path):
        with self.shared.lock:
            return remote_path in self.shared.files

    def put_text(self, remote_path, text, *, mode=0o640):
        if (
            self.shared.fail_next_submitting_state_write
            and remote_path.endswith(".json")
            and '"state": "SUBMITTING"' in text
        ):
            self.shared.fail_next_submitting_state_write = False
            raise OSError("state write failed before sbatch")

        with self.shared.lock:
            self.shared.files[remote_path] = text

    def _replace_file(self, source, destination):
        with self.shared.lock:
            self.shared.files[destination] = self.shared.files.pop(source)

    def read_text(self, remote_path, *, max_bytes=None):
        with self.shared.lock:
            return self.shared.files[remote_path]

    def submit_batch(self, request):
        with self.shared.lock:
            self.shared.sbatch_calls += 1
            call_index = self.shared.sbatch_calls

        self.shared.submit_started.set()
        if self.shared.block_submit:
            assert self.shared.allow_submit.wait(timeout=2)

        if self.shared.submit_error_result is not None:
            result = self.shared.submit_error_result
            self.shared.submit_error_result = None
            raise RemoteExecutionError(result)

        if self.shared.submit_exception is not None:
            exc = self.shared.submit_exception
            self.shared.submit_exception = None
            raise exc

        job_id = str(self.shared.next_job_id + call_index)
        return BatchSubmissionResult(
            job_id=job_id,
            raw_output=f"{job_id}\n",
            command=f"sbatch --parsable {request.script_path}",
        )


def _submission_spec(*, attempt_id: str | None = None, ntasks: int = 24) -> dict:
    return create_submission_spec(
        FLOW_SPEC,
        label="Si static",
        timestamp="20260629-120000",
        ntasks=ntasks,
        env={},
        submission_attempt_id=attempt_id,
    )


def _attempt_state(shared: SharedRemoteState, spec: dict) -> dict:
    return json.loads(shared.files[spec["submission"]["attempt_state"]])


def _prepare(shared: SharedRemoteState, spec: dict):
    record = IdempotencyRunner(shared).submit(spec, dry_run=True)
    assert record.status == "dry_run"
    assert _attempt_state(shared, spec)["state"] == "PREPARED"


def test_sequential_duplicate_submit_reuses_submitted_job_id():
    shared = SharedRemoteState()
    spec = _submission_spec()
    _prepare(shared, spec)

    first = IdempotencyRunner(shared).submit(spec)
    duplicate = IdempotencyRunner(shared).submit(spec)

    assert first.job_id == duplicate.job_id
    assert shared.sbatch_calls == 1
    assert "BMD_ALREADY_SUBMITTED=1" in duplicate.raw_output
    assert _attempt_state(shared, spec)["state"] == "SUBMITTED"


def test_concurrent_duplicate_submit_calls_sbatch_once():
    shared = SharedRemoteState()
    shared.block_submit = True
    spec = _submission_spec()
    _prepare(shared, spec)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(IdempotencyRunner(shared).submit, spec)
        assert shared.submit_started.wait(timeout=2)
        second_future = executor.submit(IdempotencyRunner(shared).submit, spec)
        shared.allow_submit.set()
        first = first_future.result(timeout=3)
        second = second_future.result(timeout=3)

    assert first.job_id == second.job_id
    assert shared.sbatch_calls == 1
    assert _attempt_state(shared, spec)["state"] == "SUBMITTED"


def test_identical_science_with_new_attempts_submits_separately():
    shared = SharedRemoteState()
    first_spec = _submission_spec()
    second_spec = _submission_spec()
    assert first_spec["submission"]["attempt_id"] != second_spec["submission"]["attempt_id"]

    _prepare(shared, first_spec)
    _prepare(shared, second_spec)
    first = IdempotencyRunner(shared).submit(first_spec)
    second = IdempotencyRunner(shared).submit(second_spec)

    assert first.job_id != second.job_id
    assert shared.sbatch_calls == 2


def test_submitted_attempt_recovers_after_runner_restart():
    shared = SharedRemoteState()
    spec = _submission_spec()
    _prepare(shared, spec)
    first = IdempotencyRunner(shared).submit(spec)

    restarted_runner_record = IdempotencyRunner(shared).submit(spec)

    assert restarted_runner_record.job_id == first.job_id
    assert shared.sbatch_calls == 1
    assert "BMD_ALREADY_SUBMITTED=1" in restarted_runner_record.raw_output


def test_ambiguous_submitting_attempt_is_not_resubmitted(monkeypatch):
    monkeypatch.setattr(paramiko_remote, "SUBMISSION_ATTEMPT_STATE_WAIT_S", 0.05)
    monkeypatch.setattr(paramiko_remote, "SUBMISSION_ATTEMPT_STATE_POLL_S", 0.01)
    shared = SharedRemoteState()
    shared.submit_exception = TimeoutError("connection lost after sbatch")
    spec = _submission_spec()
    _prepare(shared, spec)

    with pytest.raises(SubmissionAttemptInProgress):
        IdempotencyRunner(shared).submit(spec)
    assert shared.sbatch_calls == 1
    assert _attempt_state(shared, spec)["state"] == "SUBMITTING"

    with pytest.raises(SubmissionAttemptInProgress):
        IdempotencyRunner(shared).submit(spec)
    assert shared.sbatch_calls == 1


def test_failure_before_sbatch_is_retryable():
    shared = SharedRemoteState()
    shared.fail_next_submitting_state_write = True
    spec = _submission_spec()
    _prepare(shared, spec)

    with pytest.raises(OSError):
        IdempotencyRunner(shared).submit(spec)
    assert shared.sbatch_calls == 0
    assert _attempt_state(shared, spec)["state"] == "PREPARED"

    retried = IdempotencyRunner(shared).submit(spec)
    assert retried.job_id
    assert shared.sbatch_calls == 1


def test_sbatch_rejection_returns_to_prepared_for_safe_retry():
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

    retried = IdempotencyRunner(shared).submit(spec)
    assert retried.job_id
    assert shared.sbatch_calls == 2


def test_same_attempt_id_with_different_resources_is_rejected():
    shared = SharedRemoteState()
    prepared_spec = _submission_spec(ntasks=24)
    _prepare(shared, prepared_spec)
    tampered_spec = _submission_spec(
        attempt_id=prepared_spec["submission"]["attempt_id"],
        ntasks=48,
    )

    with pytest.raises(SubmissionAttemptMismatch):
        IdempotencyRunner(shared).submit(tampered_spec)

    assert shared.sbatch_calls == 0


def test_duplicate_submit_does_not_rewrite_prepared_provenance_snapshot():
    shared = SharedRemoteState()
    prepared_spec = _submission_spec()
    prepared_spec["provenance"]["bmd_compute"]["source"]["git_commit"] = "original"
    _prepare(shared, prepared_spec)
    first = IdempotencyRunner(shared).submit(prepared_spec)

    later_spec = _submission_spec(attempt_id=prepared_spec["submission"]["attempt_id"])
    later_spec["provenance"]["bmd_compute"]["source"]["git_commit"] = "newer"
    duplicate = IdempotencyRunner(shared).submit(later_spec)

    uploaded_submission = json.loads(
        shared.files[f"{prepared_spec['paths']['run_dir']}/submission.json"]
    )
    attempt_state = _attempt_state(shared, prepared_spec)

    assert duplicate.job_id == first.job_id
    assert shared.sbatch_calls == 1
    assert uploaded_submission["provenance"]["bmd_compute"]["source"]["git_commit"] == "original"
    assert attempt_state["provenance"]["bmd_compute"]["source"]["git_commit"] == "original"
    assert (
        attempt_state["job_record"]["submission_spec"]["provenance"]["bmd_compute"]["source"]["git_commit"]
        == "original"
    )
