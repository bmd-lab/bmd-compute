import json
from copy import deepcopy

import pytest
from starlette.requests import Request

import backend.remote_runtime as remote_runtime
import main
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.config import DEFAULT_LOGS_DIR
from backend.monitoring import monitor_job, monitor_submitted_job
from backend.remote import RemoteJobStatus, RemoteOperationBusy
from backend.remote_runtime import default_connection_profile
from backend.submission import create_submission_identity_token


ATTEMPT_ID = "12345678-1234-5678-9234-567812345678"
OTHER_ATTEMPT_ID = "12345678-1234-5678-9234-567812345679"
JOB_ID = "123456"


def request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/monitor",
            "headers": [],
        }
    )


def workflow():
    return WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])


def submission_spec(*, attempt_id=ATTEMPT_ID):
    run_dir = "/bmd-db/guest/flows/vasp_run_static-20261007-120000"
    logs_dir = DEFAULT_LOGS_DIR
    return {
        "schema": "bmd_compute.submission",
        "schema_version": 1,
        "status": "pending",
        "label": "vasp_run_static",
        "run_name": "vasp_run_static-20261007-120000",
        "created_at": "20261007-120000",
        "flow_spec": {
            "workflow_spec": workflow().to_dict(),
            "workflow": "static",
            "potcar_functional": "PBE_64",
        },
        "paths": {
            "run_dir": run_dir,
            "remote_script": f"{run_dir}.sbatch.sh",
            "logs_dir": logs_dir,
            "flows_dir": "/bmd-db/guest/flows",
            "log_out": f"{logs_dir}/run.out",
            "log_err": f"{logs_dir}/run.err",
            "slurm_out": f"{logs_dir}/run.slurm.out",
            "slurm_err": f"{logs_dir}/run.slurm.err",
            "stage_dirs": {},
            "result_dir": run_dir,
        },
        "cluster": {
            "ssh_config_host": "powerslurm-bmdguest",
            "remote_host": "powerslurm-bmdguest",
            "username": "bmdguest",
            "port": 22,
            "partition": "leeburton-pool",
            "account": "power-leeburton-users_v2",
        },
        "resources": {
            "nodes": 1,
            "ntasks": 24,
            "mem_gb": 128,
            "walltime": "72:00:00",
        },
        "potcar": {
            "functional": "PBE_64",
            "species": [],
            "symbols": [],
            "symbol_source": "none",
            "repository": "shared",
            "target": "/bmd-db/potcars/PBE_64",
            "symlink_targets": [],
        },
        "modules": {"purge_first": True, "load": []},
        "environment": {
            "VASP_CMD": "mpirun -n $SLURM_NTASKS vasp_std",
        },
        "runner": {
            "python": "python",
            "script_name": "run_job.py",
            "working_directory": run_dir,
        },
        "submission": {
            "attempt_id": attempt_id,
            "attempt_state": f"{logs_dir}/submission_attempts/{attempt_id}.json",
            "identity_token": create_submission_identity_token(
                "20261007-120000",
                attempt_id,
            ),
        },
    }


def job_record(*, spec=None, attempt_id=ATTEMPT_ID, job_id=JOB_ID):
    spec = deepcopy(spec or submission_spec(attempt_id=attempt_id))
    return {
        "schema": "bmd_compute.job_record",
        "schema_version": 1,
        "job_id": job_id,
        "run_name": spec["run_name"],
        "run_dir": spec["paths"]["run_dir"],
        "attempt_id": attempt_id,
        "submission_spec": spec,
    }


def run_dir_mismatch_record():
    record = job_record()
    record["submission_spec"]["paths"]["run_dir"] = "/different/run"
    return record


class BoundaryRunner:
    def __init__(self, record, *, record_exists=True):
        self.record = record
        self.record_exists = record_exists
        self.connected_profile = None
        self.record_path = None
        self.record_max_bytes = None
        self.query_calls = []
        self.closed = False

    def connect(self, profile):
        self.connected_profile = profile

    def close(self):
        self.closed = True

    def is_file(self, path):
        self.record_path = path
        return self.record_exists

    def read_text(self, path, *, max_bytes=None):
        assert path == self.record_path
        self.record_max_bytes = max_bytes
        return self.record if isinstance(self.record, str) else json.dumps(self.record)

    def query_job(self, job_id):
        self.query_calls.append(job_id)
        return RemoteJobStatus(
            job_id=job_id,
            state="RUNNING",
            raw={"summary": "RUNNING", "brief": f"{job_id}|RUNNING"},
        )


class ConnectionFailureRunner(BoundaryRunner):
    def __init__(self):
        super().__init__(None)

    def connect(self, profile):
        self.connected_profile = profile
        raise OSError("temporary SSH connection failure")


def refresh_with_factory(remote_factory, state_json):
    original_factory = remote_runtime.create_remote_runner
    remote_runtime.create_remote_runner = lambda runner_factory=None: remote_factory()
    try:
        return main.refresh_monitoring(
            request(),
            structure="Si structure remains preserved",
            fmt="poscar",
            purpose="static",
            theory="pbe",
            modifiers=None,
            cpus="24",
            memory_gb="128",
            walltime="72:00:00",
            queue="leeburton-pool",
            created_at="20261007-120000",
            job_id=JOB_ID,
            submitted_at="2026-10-07 12:00:00",
            monitor_state_json=state_json,
            workflow_spec_json=json.dumps(workflow().to_dict()),
        )
    finally:
        remote_runtime.create_remote_runner = original_factory


def refresh(runner, caller_spec):
    return refresh_with_factory(
        lambda: runner,
        main.monitor_state_json(submission_spec=caller_spec),
    )


def response_html(response):
    return response.body.decode("utf-8")


def test_monitor_ignores_all_caller_connection_authority_at_remote_boundary():
    trusted_spec = submission_spec()
    caller_spec = deepcopy(trusted_spec)
    attacker_values = {
        "ssh_config_host": "attacker-alias.invalid",
        "remote_host": "attacker-host.invalid",
        "username": "attacker-user",
        "port": 2222,
        "key_file": "/tmp/attacker-selected-key",
        "keepalive_s": 9,
    }
    caller_spec["cluster"].update(attacker_values)
    caller_spec["cluster"]["account"] = "attacker-account"
    caller_spec["paths"]["run_dir"] = "/tmp/attacker-run"
    runner = BoundaryRunner(job_record(spec=trusted_spec))

    response = refresh(runner, caller_spec)

    assert response.status_code == 200
    assert runner.connected_profile == default_connection_profile()
    assert not set(attacker_values.values()) & set(runner.connected_profile.__dict__.values())
    assert runner.record_path == f"{DEFAULT_LOGS_DIR}/job_{JOB_ID}.json"
    assert runner.query_calls == [JOB_ID]
    assert runner.closed is True
    assert response.context["submission_spec"]["cluster"] == trusted_spec["cluster"]
    assert response.context["submission_spec"]["paths"]["run_dir"] == trusted_spec["paths"]["run_dir"]


def test_matching_authenticated_attempt_allows_scheduler_query():
    runner = BoundaryRunner(job_record())

    result, record = monitor_submitted_job(
        JOB_ID,
        authenticated_attempt_id=ATTEMPT_ID,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "success"
    assert record["attempt_id"] == ATTEMPT_ID
    assert runner.query_calls == [JOB_ID]


def test_attempt_mismatch_fails_before_scheduler_query():
    runner = BoundaryRunner(job_record())

    result, record = monitor_submitted_job(
        JOB_ID,
        authenticated_attempt_id=OTHER_ATTEMPT_ID,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Monitoring Authorization"
    assert record is None
    assert runner.query_calls == []
    assert result["identity_retry_allowed"] is False


def test_canonical_job_id_mismatch_fails_before_scheduler_query():
    runner = BoundaryRunner(job_record(job_id="654321"))

    result, record = monitor_submitted_job(
        JOB_ID,
        authenticated_attempt_id=ATTEMPT_ID,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Monitoring Authorization"
    assert result["identity_retry_allowed"] is False
    assert record is None
    assert runner.query_calls == []


def test_embedded_submission_attempt_mismatch_fails_before_scheduler_query():
    record = job_record()
    record["submission_spec"] = submission_spec(attempt_id=OTHER_ATTEMPT_ID)
    record["submission_spec"]["paths"]["run_dir"] = record["run_dir"]
    runner = BoundaryRunner(record)

    result, authoritative = monitor_submitted_job(
        JOB_ID,
        authenticated_attempt_id=ATTEMPT_ID,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Monitoring Authorization"
    assert result["identity_retry_allowed"] is False
    assert authoritative is None
    assert runner.query_calls == []


def test_canonical_binding_failure_does_not_create_retry_state():
    caller_spec = submission_spec()
    runner = BoundaryRunner(job_record(attempt_id=OTHER_ATTEMPT_ID))

    response = refresh(runner, caller_spec)

    assert response.status_code == 200
    assert response.context["monitor_state_json"] == ""
    assert runner.query_calls == []
    assert '<form action="/monitor"' not in response_html(response)


def test_tampered_identity_token_fails_before_remote_connection(monkeypatch):
    caller_spec = submission_spec()
    token = caller_spec["submission"]["identity_token"]
    encoded, signature = token.split(".", 1)
    replacement = "A" if signature[0] != "A" else "B"
    caller_spec["submission"]["identity_token"] = (
        f"{encoded}.{replacement}{signature[1:]}"
    )
    remote_calls = []
    monkeypatch.setattr(
        remote_runtime,
        "create_remote_runner",
        lambda runner_factory=None: remote_calls.append("connect"),
    )

    response = main.refresh_monitoring(
        request(),
        structure="Si\n1\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nDirect\n0 0 0\n",
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus="24",
        memory_gb="128",
        walltime="72:00:00",
        queue="leeburton-pool",
        created_at="20261007-120000",
        job_id=JOB_ID,
        submitted_at="2026-10-07 12:00:00",
        monitor_state_json=main.monitor_state_json(submission_spec=caller_spec),
        workflow_spec_json=json.dumps(workflow().to_dict()),
    )

    assert response.status_code == 400
    assert remote_calls == []


@pytest.mark.parametrize(
    ("record", "exists"),
    [
        ({}, False),
        ("not-json", True),
        ({"job_id": JOB_ID}, True),
        (run_dir_mismatch_record(), True),
    ],
)
def test_missing_or_untrusted_job_record_fails_before_scheduler_query(record, exists):
    runner = BoundaryRunner(record, record_exists=exists)

    result, authoritative = monitor_submitted_job(
        JOB_ID,
        authenticated_attempt_id=ATTEMPT_ID,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Monitoring Authorization"
    assert result["identity_retry_allowed"] is False
    assert authoritative is None
    assert runner.query_calls == []


def test_malformed_job_id_fails_before_connection_or_query():
    remote_calls = []

    result, record = monitor_submitted_job(
        "12345;touch /tmp/nope",
        authenticated_attempt_id=ATTEMPT_ID,
        runner_factory=lambda: remote_calls.append("connect"),
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Job ID"
    assert record is None
    assert remote_calls == []


def test_monitor_job_validates_job_id_even_with_submission_state():
    remote_calls = []

    result = monitor_job(
        "12345;touch /tmp/nope",
        submission_spec={"cluster": {"ssh_config_host": "attacker"}},
        runner_factory=lambda: remote_calls.append("connect"),
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Job ID"
    assert remote_calls == []


def test_unchanged_current_v1_monitor_state_remains_compatible():
    spec = submission_spec()
    runner = BoundaryRunner(job_record(spec=spec))

    response = refresh(runner, spec)

    assert response.status_code == 200
    assert response.context["monitoring_result"]["summary"] == "RUNNING"
    assert response.context["submission_spec"] == spec


def test_ssh_failure_preserves_minimal_refresh_state_and_retry_rebinds_authority():
    initial_spec = submission_spec()
    failed_runner = ConnectionFailureRunner()

    failed_response = refresh(failed_runner, initial_spec)

    assert failed_response.status_code == 200
    assert failed_response.context["submission_spec"] is None
    assert '<form action="/monitor"' in response_html(failed_response)
    retry_state = json.loads(failed_response.context["monitor_state_json"])
    assert retry_state["submission_spec"] == {
        "submission": {
            "attempt_id": ATTEMPT_ID,
            "identity_token": initial_spec["submission"]["identity_token"],
        }
    }

    retry_state["submission_spec"]["cluster"] = {
        "ssh_config_host": "attacker.invalid",
        "remote_host": "attacker.invalid",
        "username": "attacker",
        "port": 2222,
        "key_file": "/tmp/attacker-key",
    }
    trusted_runner = BoundaryRunner(job_record(spec=initial_spec))
    retry_response = refresh_with_factory(
        lambda: trusted_runner,
        json.dumps(retry_state),
    )

    assert retry_response.status_code == 200
    assert trusted_runner.connected_profile == default_connection_profile()
    assert trusted_runner.query_calls == [JOB_ID]
    assert retry_response.context["submission_spec"] == initial_spec

    tampered_retry_state = json.loads(failed_response.context["monitor_state_json"])
    token = tampered_retry_state["submission_spec"]["submission"]["identity_token"]
    encoded, signature = token.split(".", 1)
    replacement = "A" if signature[0] != "A" else "B"
    tampered_retry_state["submission_spec"]["submission"]["identity_token"] = (
        f"{encoded}.{replacement}{signature[1:]}"
    )
    unexpected_remote_calls = []

    def unexpected_remote_factory():
        unexpected_remote_calls.append("called")
        return BoundaryRunner(job_record())

    tampered_response = refresh_with_factory(
        unexpected_remote_factory,
        json.dumps(tampered_retry_state),
    )

    assert tampered_response.status_code == 400
    assert unexpected_remote_calls == []


def test_remote_capacity_failure_preserves_refresh_form():
    spec = submission_spec()

    def busy_factory():
        raise RemoteOperationBusy(limit=4, active=4, timeout_s=5.0)

    response = refresh_with_factory(
        busy_factory,
        main.monitor_state_json(submission_spec=spec),
    )

    assert response.status_code == 200
    assert response.context["submission_spec"] is None
    assert response.context["monitoring_result"]["stage"] == "Remote Capacity"
    assert '<form action="/monitor"' in response_html(response)
    retry_state = json.loads(response.context["monitor_state_json"])
    assert set(retry_state["submission_spec"]) == {"submission"}
    assert set(retry_state["submission_spec"]["submission"]) == {
        "attempt_id",
        "identity_token",
    }
