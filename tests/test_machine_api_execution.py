"""Machine API Phase 1B: authenticated, idempotent execution and attempt lookup.

Every test uses an in-memory fake POWER (remote files, mkdir locks, sbatch and
SLURM state). Nothing contacts POWER or submits a job.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from starlette.testclient import TestClient

import backend.paramiko_remote as paramiko_remote
import main
import test_input_reference_contract as reference_cases
from backend.monitoring import classify_slurm_state
from backend.remote import RemoteCommandResult, RemoteJobStatus
from compute_api import auth as api_auth
from compute_api import execution
from compute_api import ledger as ledger_module
from compute_api.schemas import parse_execution_request
from test_submission_idempotency import IdempotencyRunner, SharedRemoteState


SI_POSCAR = reference_cases.SI_POSCAR
LOOPBACK = ("127.0.0.1", 50000)


# ------------------------------------------------------------------ fake POWER


class FakePower(SharedRemoteState):
    def __init__(self):
        super().__init__()
        self.connections = 0
        self.profiles = []
        self.job_states = {}
        self.lock_claims = 0
        self.hold_first_submitting_write = False
        self.claims_before_release = 0

    def factory(self):
        return FakePowerRunner(self)

    def run_dirs(self):
        return sorted({path.rsplit("/", 1)[0] for path in self.files if path.endswith("/submission.json")})


class FakePowerRunner(IdempotencyRunner):
    def run(self, command, **kwargs):
        if command.startswith("mkdir ") and command.rstrip("'").endswith(".lock"):
            with self.shared.lock:
                self.shared.lock_claims += 1
        return super().run(command, **kwargs)

    def put_text(self, remote_path, text, *, mode=0o640):
        if self.shared.hold_first_submitting_write and '"state": "SUBMITTING"' in text:
            self.shared.hold_first_submitting_write = False
            # Hold the claimant between "claimed" and "marked SUBMITTING" until a
            # concurrent duplicate has tried to claim the same attempt.
            deadline = threading.Event()
            for _ in range(200):
                with self.shared.lock:
                    if self.shared.lock_claims >= self.shared.claims_before_release:
                        break
                deadline.wait(0.01)
        return super().put_text(remote_path, text, mode=mode)

    def connect(self, profile):
        with self.shared.lock:
            self.shared.connections += 1
            self.shared.profiles.append(profile)

    def close(self):
        return None

    def query_job(self, job_id):
        state = self.shared.job_states.get(job_id, "PENDING")
        exit_code = "0:0" if state == "COMPLETED" else ("1:0" if state == "FAILED" else None)
        return RemoteJobStatus(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            raw={
                "summary": classify_slurm_state(state, exit_code),
                "brief": f"{job_id}|vasp_run|{state}|00:10:00|2026-10-10T10:00:00|Unknown|leeburton-pool",
            },
        )


@pytest.fixture
def power(monkeypatch):
    fake = FakePower()
    monkeypatch.setattr(execution, "RUNNER_FACTORY", fake.factory)
    monkeypatch.setattr(paramiko_remote, "SUBMISSION_ATTEMPT_STATE_WAIT_S", 0.05)
    monkeypatch.setattr(paramiko_remote, "SUBMISSION_ATTEMPT_STATE_POLL_S", 0.01)
    return fake


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    path = tmp_path / "api_state"
    path.mkdir(mode=0o700)
    monkeypatch.setenv(ledger_module.STATE_DIR_ENV, str(path))
    return path


SCOPES = {
    "runner": ["plan", "read", "prepare", "submit"],
    "preparer": ["plan", "read", "prepare"],
    "planner": ["plan", "read"],
    "submit_only": ["submit"],
    "other": ["plan", "read", "prepare", "submit"],
}


@pytest.fixture
def tokens(tmp_path, monkeypatch, state_dir, power):
    entries, issued = [], {}
    for principal, scopes in SCOPES.items():
        token, token_id, verifier = api_auth.generate_token()
        issued[principal] = token
        entries.append({"principal": principal, "token_id": token_id, "verifier": verifier, "scopes": scopes, "enabled": True})
    path = tmp_path / "api_tokens.json"
    path.write_text(json.dumps({"schema": "bmd_compute.api_tokens", "schema_version": 1, "principals": entries}), encoding="utf-8")
    os.chmod(path, 0o600)
    monkeypatch.setenv(api_auth.TOKENS_FILE_ENV, str(path))
    return issued


@pytest.fixture
def client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TestClient(main.app, client=LOOPBACK)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def plan_body(structure=SI_POSCAR, *, desired_output="energy_only", custom=None, resources=None):
    workflow = {"custom": custom} if custom is not None else {"desired_output": desired_output}
    body = {"structure": {"format": "poscar", "text": structure}, "workflow": workflow}
    if resources is not None:
        body["resources"] = resources
    return body


def digest_for(client, tokens, body):
    response = client.post("/api/v1/plans", headers=bearer(tokens["planner"]), json=body)
    assert response.status_code == 200, response.text
    return response.json()["plan_digest"]


def attempt_body(client, tokens, *, submit, labels=None, **plan_kwargs):
    body = plan_body(**plan_kwargs)
    body["expected_plan_digest"] = digest_for(client, tokens, body)
    body["submit"] = submit
    if labels is not None:
        body["labels"] = labels
    return body


def put(client, token, attempt_id, body):
    return client.put(f"/api/v1/attempts/{attempt_id}", headers=bearer(token), json=body)


def get(client, token, attempt_id):
    return client.get(f"/api/v1/attempts/{attempt_id}", headers=bearer(token))


def new_id():
    return str(uuid.uuid4())


def ledger_record(state_dir, attempt_id):
    return json.loads((state_dir / "attempts" / f"{attempt_id}.json").read_text(encoding="utf-8"))


def write_ledger_record(state_dir, record):
    (state_dir / "attempts" / f"{record['attempt_id']}.json").write_text(json.dumps(record), encoding="utf-8")


def no_remote_activity(power):
    return power.connections == 0 and power.sbatch_calls == 0 and not power.files


# ------------------------------------------------- authentication and scopes


def test_unauthenticated_requests_are_rejected_without_remote_activity(client, tokens, power):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    assert client.put(f"/api/v1/attempts/{attempt_id}", json=body).status_code == 401
    assert client.get(f"/api/v1/attempts/{attempt_id}").status_code == 401
    bad = "bmdc1.0000000000000000." + "A" * 43
    assert put(client, bad, attempt_id, body).status_code == 401
    assert no_remote_activity(power)


@pytest.mark.parametrize("submit", [False, True])
def test_planning_only_credential_cannot_execute(client, tokens, power, state_dir, submit):
    attempt_id = new_id()
    response = put(client, tokens["planner"], attempt_id, attempt_body(client, tokens, submit=submit))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"
    assert no_remote_activity(power)
    assert not (state_dir / "attempts" / f"{attempt_id}.json").exists()


def test_prepare_scope_cannot_submit(client, tokens, power):
    attempt_id = new_id()
    response = put(client, tokens["preparer"], attempt_id, attempt_body(client, tokens, submit=True))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"
    assert no_remote_activity(power)

    prepared = put(client, tokens["preparer"], attempt_id, attempt_body(client, tokens, submit=False))
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["state"] == "prepared"
    assert power.sbatch_calls == 0


def test_submit_scope_cannot_prepare_only_and_lookup_needs_read(client, tokens, power):
    attempt_id = new_id()
    assert put(client, tokens["submit_only"], attempt_id, attempt_body(client, tokens, submit=False)).status_code == 403
    assert get(client, tokens["submit_only"], attempt_id).status_code == 403
    assert no_remote_activity(power)


def test_other_principals_cannot_reuse_or_read_an_attempt(client, tokens, power):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    assert put(client, tokens["runner"], attempt_id, body).status_code == 200
    connections, sbatch = power.connections, power.sbatch_calls

    stolen = put(client, tokens["other"], attempt_id, body)
    assert stolen.status_code == 403
    assert stolen.json()["error"]["code"] == "attempt_forbidden"
    read = get(client, tokens["other"], attempt_id)
    assert read.status_code == 403
    assert (power.connections, power.sbatch_calls) == (connections, sbatch)


def test_unknown_attempt_lookup_is_404_without_remote_activity(client, tokens, power):
    response = get(client, tokens["runner"], new_id())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "attempt_not_found"
    assert no_remote_activity(power)


def test_non_loopback_execution_is_refused(tokens, power):
    remote = TestClient(main.app, client=("132.66.1.2", 40000))
    assert remote.put(f"/api/v1/attempts/{new_id()}", headers=bearer(tokens["runner"]), json={}).status_code == 403
    assert no_remote_activity(power)


# ----------------------------------------------------------- request shape


@pytest.mark.parametrize(
    "attempt_id",
    [
        "NOT-A-UUID",
        "12345678123456781234567812345678",
        "{12345678-1234-5678-9234-567812345678}",
        "12345678-1234-5678-9234-56781234567G",
        "abcdef12-1234-5678-9234-567812345678".upper(),
        "00000000-0000-0000-0000-000000000000",
        "12345678-1234-5678-1234-567812345678",
    ],
)
def test_malformed_attempt_ids_are_rejected_before_any_work(client, tokens, power, attempt_id):
    response = put(client, tokens["runner"], attempt_id, {})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_attempt_id"
    assert get(client, tokens["runner"], attempt_id).status_code == 422
    assert no_remote_activity(power)


@pytest.mark.parametrize(
    "mutate, field, problem",
    [
        (lambda body: body.update(cluster={"remote_host": "evil"}), "cluster", "unknown_field"),
        (lambda body: body.update(paths={"run_dir": "/tmp/x"}), "paths", "unknown_field"),
        (lambda body: body.update(job_id="123"), "job_id", "unknown_field"),
        (lambda body: body.update(ssh_config_host="evil"), "ssh_config_host", "unknown_field"),
        (lambda body: body.update(run_timestamp="20260101-000000"), "run_timestamp", "unknown_field"),
        (lambda body: body["resources"].update(account="other"), "resources.account", "unknown_field"),
        (lambda body: body["resources"].update(partition="other"), "resources.partition", "unknown_field"),
        (lambda body: body["resources"].update(nodes=2), "resources.nodes", "unknown_field"),
        (lambda body: body.pop("expected_plan_digest"), "expected_plan_digest", "required"),
        (lambda body: body.update(expected_plan_digest="abc"), "expected_plan_digest", "must_be_sha256_digest"),
        (lambda body: body.pop("submit"), "submit", "required"),
        (lambda body: body.update(submit="yes"), "submit", "must_be_boolean"),
        (lambda body: body.update(labels={"campaign": "x y"}), "labels.campaign", "must_be_label"),
        (lambda body: body.update(labels={"owner": "x"}), "labels.owner", "unknown_field"),
    ],
)
def test_forbidden_authority_fields_and_bad_shapes_are_rejected(client, tokens, power, mutate, field, problem):
    body = attempt_body(client, tokens, submit=True, resources={"cpus": 24})
    mutate(body)
    response = put(client, tokens["runner"], new_id(), body)
    assert response.status_code == 422
    assert {"field": field, "problem": problem} in response.json()["error"]["fields"]
    assert "evil" not in response.text
    assert no_remote_activity(power)


def test_plan_digest_mismatch_is_rejected_before_remote_activity(client, tokens, power, state_dir):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    body["expected_plan_digest"] = "sha256:" + "0" * 64
    response = put(client, tokens["runner"], attempt_id, body)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "plan_digest_mismatch"
    assert no_remote_activity(power)
    assert not (state_dir / "attempts" / f"{attempt_id}.json").exists()


def test_hse06_soc_remains_rejected(client, tokens, power):
    body = plan_body(custom={"stages": [{"stage_type": "static", "theory": "hse06", "modifiers": ["soc"]}]})
    body.update(expected_plan_digest="sha256:" + "1" * 64, submit=True)
    response = put(client, tokens["runner"], new_id(), body)
    assert response.status_code == 422
    assert response.json()["error"]["diagnostic_code"] == "hse06_soc_not_supported_for_new_calculations"
    assert no_remote_activity(power)


@pytest.mark.parametrize(
    "resources, field",
    [
        ({"cpus": 120}, "resources.cpus"),
        ({"memory_gb": 160}, "resources.memory_gb"),
        ({"walltime": "72:00:01"}, "resources.walltime"),
        ({"walltime": "96:00:00"}, "resources.walltime"),
    ],
)
def test_api_resource_limits_are_enforced(client, tokens, power, resources, field):
    body = attempt_body(client, tokens, submit=False, resources=resources)
    response = put(client, tokens["runner"], new_id(), body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "resource_limit_exceeded"
    assert {"field": field, "problem": "exceeds_api_limit"} in error["fields"]
    assert no_remote_activity(power)


def test_resources_at_the_api_limits_are_accepted(client, tokens, power):
    body = attempt_body(client, tokens, submit=False, resources={"cpus": 96, "memory_gb": 128, "walltime": "72:00:00"})
    response = put(client, tokens["runner"], new_id(), body)
    assert response.status_code == 200, response.text
    assert response.json()["resources"] == {
        "nodes": 1,
        "cpus": 96,
        "memory_gb": 128,
        "walltime": "72:00:00",
        "partition": "leeburton-pool",
        "account": "power-leeburton-users_v2",
    }


def test_execution_without_a_configured_ledger_fails_closed(client, tokens, power, monkeypatch):
    monkeypatch.delenv(ledger_module.STATE_DIR_ENV)
    response = put(client, tokens["runner"], new_id(), attempt_body(client, tokens, submit=True))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "execution_not_configured"
    assert no_remote_activity(power)


def test_unsafe_ledger_directory_fails_closed(client, tokens, power, state_dir):
    os.chmod(state_dir, 0o750)
    response = put(client, tokens["runner"], new_id(), attempt_body(client, tokens, submit=True))
    assert response.status_code == 503
    assert no_remote_activity(power)


# ------------------------------------------------------------ idempotency


def test_identical_retries_never_submit_twice(client, tokens, power, state_dir):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True, labels={"campaign": "bench-01", "cell": "si-24"})
    first = put(client, tokens["runner"], attempt_id, body)
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["state"] == "submitted"
    job_id = payload["submission"]["job_id"]
    assert job_id
    timestamp = ledger_record(state_dir, attempt_id)["run_timestamp"]
    run_dirs = power.run_dirs()

    for _ in range(3):
        retry = put(client, tokens["runner"], attempt_id, body)
        assert retry.status_code == 200
        assert retry.json()["submission"]["job_id"] == job_id
        assert retry.json()["state"] == "submitted"
    assert power.sbatch_calls == 1
    assert power.run_dirs() == run_dirs and len(run_dirs) == 1
    assert ledger_record(state_dir, attempt_id)["run_timestamp"] == timestamp
    assert payload["labels"] == {"campaign": "bench-01", "cell": "si-24"}

    prepare_only = put(client, tokens["runner"], attempt_id, {**body, "submit": False})
    assert prepare_only.status_code == 200
    assert prepare_only.json()["state"] == "submitted"
    assert power.sbatch_calls == 1


def test_prepare_then_submit_reuses_the_prepared_attempt(client, tokens, power, state_dir):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=False)
    assert put(client, tokens["runner"], attempt_id, body).json()["state"] == "prepared"
    prepared_files = dict(power.files)
    timestamp = ledger_record(state_dir, attempt_id)["run_timestamp"]

    again = put(client, tokens["runner"], attempt_id, body)
    assert again.json()["state"] == "prepared"
    submitted = put(client, tokens["runner"], attempt_id, {**body, "submit": True})
    assert submitted.status_code == 200
    assert submitted.json()["state"] == "submitted"
    assert power.sbatch_calls == 1
    assert ledger_record(state_dir, attempt_id)["run_timestamp"] == timestamp
    # The prepared run directory was not rebuilt: every prepared file is unchanged.
    for path, text in prepared_files.items():
        if "/submission_attempts/" not in path:
            assert power.files[path] == text, path


def test_reusing_an_attempt_id_with_a_different_request_is_rejected(client, tokens, power):
    attempt_id = new_id()
    assert put(client, tokens["runner"], attempt_id, attempt_body(client, tokens, submit=False)).status_code == 200
    connections = power.connections
    changed = attempt_body(client, tokens, submit=False, resources={"cpus": 48})
    response = put(client, tokens["runner"], attempt_id, changed)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "attempt_request_mismatch"
    relabelled = attempt_body(client, tokens, submit=False, labels={"cell": "other"})
    assert put(client, tokens["runner"], attempt_id, relabelled).json()["error"]["code"] == "attempt_request_mismatch"
    assert power.connections == connections


def test_existing_remote_attempt_with_a_different_fingerprint_is_not_touched(client, tokens, power):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    state_path = f"/bmd-db/guest/logs/submission_attempts/{attempt_id}.json"
    foreign = json.dumps({"attempt_id": attempt_id, "state": "PREPARED", "fingerprint": "f" * 64})
    power.files[state_path] = foreign

    response = put(client, tokens["runner"], attempt_id, body)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "attempt_fingerprint_mismatch"
    assert power.files == {state_path: foreign}
    assert power.sbatch_calls == 0


def test_concurrent_duplicate_puts_submit_exactly_once(client, tokens, power, state_dir):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    request = parse_execution_request(body)
    power.block_submit = True

    def call():
        try:
            return execution.execute_attempt(
                planner=main.plan_calculation_request,
                principal="runner",
                attempt_id=attempt_id,
                request=request,
            )["state"]
        except execution.ExecutionError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(call) for _ in range(4)]
        assert power.submit_started.wait(timeout=30)
        power.allow_submit.set()
        outcomes = [future.result(timeout=60) for future in futures]

    assert power.sbatch_calls == 1
    assert "submitted" in outcomes
    assert set(outcomes) <= {"submitted", "attempt_in_progress", "submission_uncertain", "submission_not_started"}
    final = get(client, tokens["runner"], attempt_id).json()
    assert final["state"] == "submitted"
    assert len(power.run_dirs()) == 1
    reserved = ledger_record(state_dir, attempt_id)["submission"]
    assert reserved["status"] == "submitted"


def test_duplicate_racing_between_claim_and_submitting_mark_is_blocked_by_the_remote_lock(client, tokens, power):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=False)
    assert put(client, tokens["runner"], attempt_id, body).json()["state"] == "prepared"
    request = parse_execution_request({**body, "submit": True})
    power.lock_claims = 0
    power.hold_first_submitting_write = True
    power.claims_before_release = 2

    def call():
        try:
            return execution.execute_attempt(
                planner=main.plan_calculation_request,
                principal="runner",
                attempt_id=attempt_id,
                request=request,
            )["state"]
        except execution.ExecutionError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [future.result(timeout=60) for future in [pool.submit(call) for _ in range(2)]]
    assert power.lock_claims >= 2, "the duplicate must have raced the claimant"
    assert power.sbatch_calls == 1
    assert "submitted" in outcomes


def test_a_retry_is_not_blocked_by_its_own_reservation(client, tokens, power, monkeypatch):
    monkeypatch.setattr(execution, "POLICY", dataclasses.replace(execution.POLICY, max_active_jobs=1, max_submissions_per_window=1))
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    power.fail_next_submitting_state_write = True
    assert put(client, tokens["runner"], attempt_id, body).json()["error"]["code"] == "submission_not_started"
    retried = put(client, tokens["runner"], attempt_id, body)
    assert retried.status_code == 200, retried.text
    assert retried.json()["state"] == "submitted"
    _, another = _submit_new(client, tokens, body)
    assert another.status_code == 429


def test_crash_after_prepare_recovers_with_the_same_run_and_identity(client, tokens, power, state_dir, monkeypatch):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    original = execution._record_remote_state
    calls = {"n": 0}

    def crash_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("process killed after remote PREPARED")
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_record_remote_state", crash_once)
    crashed = put(client, tokens["runner"], attempt_id, body)
    assert crashed.status_code == 500
    assert power.sbatch_calls == 0
    run_dirs = power.run_dirs()
    timestamp = ledger_record(state_dir, attempt_id)["run_timestamp"]

    recovered = put(client, tokens["runner"], attempt_id, body)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["state"] == "submitted"
    assert power.sbatch_calls == 1
    assert power.run_dirs() == run_dirs
    assert ledger_record(state_dir, attempt_id)["run_timestamp"] == timestamp


def test_lost_response_after_sbatch_returns_the_existing_job(client, tokens, power, monkeypatch):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    original = execution.read_remote_attempt_state
    calls = {"raised": False}

    def unreachable_after_sbatch(record):
        if power.sbatch_calls and not calls["raised"]:
            calls["raised"] = True
            raise execution.ExecutionError(503, "remote_unavailable", "lost")
        return original(record)

    monkeypatch.setattr(execution, "read_remote_attempt_state", unreachable_after_sbatch)
    first = put(client, tokens["runner"], attempt_id, body)
    assert first.status_code == 503
    assert first.json()["error"]["code"] == "submission_outcome_unconfirmed"
    assert power.sbatch_calls == 1

    retry = put(client, tokens["runner"], attempt_id, body)
    assert retry.status_code == 200
    assert retry.json()["state"] == "submitted"
    assert power.sbatch_calls == 1


def test_ambiguous_sbatch_is_uncertain_and_never_resubmitted(client, tokens, power):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    power.submit_exception = TimeoutError("connection lost after sbatch")

    first = put(client, tokens["runner"], attempt_id, body)
    assert first.status_code == 409
    assert first.json()["error"]["code"] == "submission_uncertain"
    assert first.json()["error"]["attempt"]["state"] == "submission_uncertain"
    for _ in range(2):
        retry = put(client, tokens["runner"], attempt_id, body)
        assert retry.status_code == 409
        assert retry.json()["error"]["code"] == "submission_uncertain"
    assert power.sbatch_calls == 1
    lookup = get(client, tokens["runner"], attempt_id)
    assert lookup.status_code == 200
    assert lookup.json()["state"] == "submission_uncertain"
    assert lookup.json()["submission"]["job_id"] is None


def test_rejected_sbatch_releases_the_reservation_and_stays_prepared(client, tokens, power, state_dir):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    power.submit_error_result = RemoteCommandResult(command="sbatch", returncode=1, stderr="sbatch: error: invalid account")
    failed = put(client, tokens["runner"], attempt_id, body)
    assert failed.status_code == 502
    assert failed.json()["error"]["code"] == "submit_failed"
    assert "invalid account" not in failed.text
    record = ledger_record(state_dir, attempt_id)
    assert record["state"] == "prepared"
    assert record["submission"]["status"] == "released"

    retried = put(client, tokens["runner"], attempt_id, body)
    assert retried.json()["state"] == "submitted"
    assert power.sbatch_calls == 2


def test_submission_that_never_reached_sbatch_keeps_its_reservation(client, tokens, power, state_dir):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    power.fail_next_submitting_state_write = True
    failed = put(client, tokens["runner"], attempt_id, body)
    assert failed.status_code == 409
    assert failed.json()["error"]["code"] == "submission_not_started"
    assert power.sbatch_calls == 0
    assert ledger_record(state_dir, attempt_id)["submission"]["status"] == "reserved"

    retried = put(client, tokens["runner"], attempt_id, body)
    assert retried.json()["state"] == "submitted"
    assert power.sbatch_calls == 1
    record = ledger_record(state_dir, attempt_id)
    assert record["submission"]["status"] == "submitted"


def test_runtime_package_change_does_not_replace_a_prepared_attempt(client, tokens, power, monkeypatch):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=False)
    assert put(client, tokens["runner"], attempt_id, body).json()["state"] == "prepared"
    snapshot = dict(power.files)

    monkeypatch.setattr(execution, "current_runtime_digest", lambda: "0" * 64)
    for submit in (False, True):
        response = put(client, tokens["runner"], attempt_id, {**body, "submit": submit})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "runtime_package_changed"
    assert power.files == snapshot
    assert power.sbatch_calls == 0


def test_rebuilt_fingerprint_drift_is_rejected(client, tokens, power, state_dir):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=False)
    assert put(client, tokens["runner"], attempt_id, body).status_code == 200
    record = ledger_record(state_dir, attempt_id)
    record["attempt_fingerprint"] = "e" * 64
    write_ledger_record(state_dir, record)
    connections = power.connections
    response = put(client, tokens["runner"], attempt_id, {**body, "submit": True})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "attempt_fingerprint_mismatch"
    assert power.connections == connections and power.sbatch_calls == 0


def test_server_restart_recovers_from_durable_records(client, tokens, power, state_dir, tmp_path, monkeypatch):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=False)
    assert put(client, tokens["runner"], attempt_id, body).status_code == 200
    # A "restarted" service: a different process would see only the on-disk ledger.
    moved = tmp_path / "restarted_state"
    shutil.copytree(state_dir, moved)
    os.chmod(moved, 0o700)
    monkeypatch.setenv(ledger_module.STATE_DIR_ENV, str(moved))
    submitted = put(client, tokens["runner"], attempt_id, {**body, "submit": True})
    assert submitted.status_code == 200
    assert submitted.json()["state"] == "submitted"
    assert power.sbatch_calls == 1
    assert len(power.run_dirs()) == 1


# --------------------------------------------------------------------- caps


def _submit_new(client, tokens, body, principal="runner"):
    attempt_id = new_id()
    return attempt_id, put(client, tokens[principal], attempt_id, body)


def test_active_job_cap_is_enforced_and_refreshed_from_slurm(client, tokens, power):
    body = attempt_body(client, tokens, submit=True)
    jobs = []
    for _ in range(2):
        _, response = _submit_new(client, tokens, body)
        assert response.status_code == 200
        jobs.append(response.json()["submission"]["job_id"])
        power.job_states[jobs[-1]] = "RUNNING"

    third_id, third = _submit_new(client, tokens, body)
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "active_job_cap_exceeded"
    assert power.sbatch_calls == 2
    # Prepare-only stays possible while the cap is full; the third is prepared.
    assert get(client, tokens["runner"], third_id).json()["state"] == "prepared"
    # Another principal has its own cap.
    _, other = _submit_new(client, tokens, body, principal="other")
    assert other.status_code == 200

    power.job_states[jobs[0]] = "COMPLETED"
    retried = put(client, tokens["runner"], third_id, body)
    assert retried.status_code == 200, retried.text
    assert retried.json()["state"] == "submitted"


def test_daily_submission_cap_is_durable_and_rolling(client, tokens, power, state_dir, tmp_path, monkeypatch):
    body = attempt_body(client, tokens, submit=True)
    submitted = []
    for _ in range(5):
        attempt_id, response = _submit_new(client, tokens, body)
        assert response.status_code == 200
        power.job_states[response.json()["submission"]["job_id"]] = "COMPLETED"
        get(client, tokens["runner"], attempt_id)
        submitted.append(attempt_id)

    _, sixth = _submit_new(client, tokens, body)
    assert sixth.status_code == 429
    assert sixth.json()["error"]["code"] == "submission_cap_exceeded"
    assert power.sbatch_calls == 5

    # A restarted service reading the same durable records keeps the cap.
    moved = tmp_path / "restarted_state"
    shutil.copytree(state_dir, moved)
    os.chmod(moved, 0o700)
    monkeypatch.setenv(ledger_module.STATE_DIR_ENV, str(moved))
    _, after_restart = _submit_new(client, tokens, body)
    assert after_restart.status_code == 429

    # The window is rolling: submissions older than 24 hours stop counting.
    for attempt_id in submitted:
        record = ledger_record(moved, attempt_id)
        record["submission"]["reserved_at"] = ledger_module.isoformat(ledger_module.utc_now() - timedelta(hours=25))
        (moved / "attempts" / f"{attempt_id}.json").write_text(json.dumps(record), encoding="utf-8")
    _, later = _submit_new(client, tokens, body)
    assert later.status_code == 200
    assert power.sbatch_calls == 6


def test_new_attempt_cap_bounds_preparations(client, tokens, power, monkeypatch):
    monkeypatch.setattr(execution, "POLICY", dataclasses.replace(execution.POLICY, max_new_attempts_per_window=2))
    body = attempt_body(client, tokens, submit=False)
    for _ in range(2):
        assert _submit_new(client, tokens, body)[1].status_code == 200
    connections = power.connections
    _, third = _submit_new(client, tokens, body)
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "attempt_cap_exceeded"
    assert power.connections == connections


# ------------------------------------------------------------------- lookup


def test_lookup_reports_bounded_scheduler_state(client, tokens, power):
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    job_id = put(client, tokens["runner"], attempt_id, body).json()["submission"]["job_id"]
    power.job_states[job_id] = "TIMEOUT"
    response = get(client, tokens["runner"], attempt_id)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "schema",
        "schema_version",
        "api_version",
        "attempt_id",
        "plan_digest",
        "state",
        "labels",
        "created_at",
        "resources",
        "submission",
        "scheduler",
    }
    assert payload["plan_digest"] == body["expected_plan_digest"]
    assert payload["scheduler"]["summary"] == "FAILURE"
    assert payload["scheduler"]["state"] == "TIMEOUT"
    assert payload["scheduler"]["terminal"] is True
    assert payload["scheduler"]["started_at"] == "2026-10-10T10:00:00"
    assert payload["scheduler"]["ended_at"] is None
    for needle in ("/bmd-db", "bmdguest", "identity_token", "fingerprint", "ssh", "run_dir", "submission_attempts"):
        assert needle not in response.text, needle
    # All remote access used the server's own connection profile.
    assert {profile.ssh_config_host for profile in power.profiles} == {"powerslurm-bmdguest"}


def test_tokens_never_appear_in_execution_responses_or_logs(client, tokens, power, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    attempt_id = new_id()
    body = attempt_body(client, tokens, submit=True)
    responses = [
        put(client, tokens["runner"], attempt_id, body),
        get(client, tokens["runner"], attempt_id),
        put(client, tokens["other"], attempt_id, body),
        put(client, tokens["preparer"], new_id(), body),
    ]
    for response in responses:
        rendered = response.text + json.dumps(dict(response.headers))
        for token in tokens.values():
            assert token not in rendered
            assert token.rsplit(".", 1)[1] not in rendered
    for token in tokens.values():
        assert token not in caplog.text


# ---------------------------------------------------------- browser routes


def test_browser_routes_and_openapi_are_unchanged(tokens):
    assert sorted(main.app.openapi()["paths"]) == [
        "/",
        "/analyze",
        "/build-calculation",
        "/build-workflow",
        "/monitor",
        "/prepare-remote",
        "/resume",
        "/submit",
    ]


def test_execution_and_planning_resolve_the_same_plan(client, tokens, power):
    body = attempt_body(client, tokens, submit=False, desired_output="electronic_dos")
    response = put(client, tokens["runner"], new_id(), body)
    assert response.status_code == 200
    assert response.json()["plan_digest"] == body["expected_plan_digest"]
