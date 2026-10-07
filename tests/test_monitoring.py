from backend.monitoring import (
    MonitoringStageError,
    classify_slurm_state,
    is_valid_slurm_job_id,
    monitor_job,
    parse_sacct_row,
    parse_scontrol_output,
    remote_job_status_from_slurm_outputs,
    strip_job_id,
)
from backend.remote import RemoteJobStatus
from backend.config import DEFAULT_REMOTE_HOST, DEFAULT_USERNAME, NOTEBOOK_DEFAULTS
from backend.remote_runtime import (
    connection_profile_from_submission_spec,
    default_connection_profile,
)


def test_strip_job_id_removes_slurm_suffixes():
    assert strip_job_id("12345.batch") == "12345"
    assert strip_job_id("12345.extern") == "12345"
    assert strip_job_id(" 12345 ") == "12345"


def test_slurm_job_id_validation():
    assert is_valid_slurm_job_id("12345")
    assert is_valid_slurm_job_id("12345.batch")
    assert is_valid_slurm_job_id("12345_7")
    assert not is_valid_slurm_job_id("")
    assert not is_valid_slurm_job_id("abc123")
    assert not is_valid_slurm_job_id("12345;rm")
    assert not is_valid_slurm_job_id("１２３４５")


def test_classify_slurm_state_matches_notebook_rules():
    assert classify_slurm_state("PENDING") == "PENDING"
    assert classify_slurm_state("CONFIGURING") == "PENDING"
    assert classify_slurm_state("RUNNING") == "RUNNING"
    assert classify_slurm_state("COMPLETING") == "RUNNING"
    assert classify_slurm_state("COMPLETED", "0:0") == "SUCCESS"
    assert classify_slurm_state("FAILED", "1:0") == "FAILURE"
    assert classify_slurm_state("BOOT_FAIL", "1:0") == "FAILURE"
    assert classify_slurm_state("DEADLINE", "1:0") == "FAILURE"
    assert classify_slurm_state("SPECIAL_EXIT", "1:0") == "FAILURE"
    assert classify_slurm_state("COMPLETED", "2:0") == "FAILURE"
    assert classify_slurm_state("SCHEDULER_SURPRISE") == "UNKNOWN"
    assert classify_slurm_state("") == "UNKNOWN"


def test_parse_scontrol_output_extracts_monitoring_fields():
    output = (
        "JobId=12345 JobName=TiO2-static\n"
        "   JobState=RUNNING Reason=None StdOut=/logs/job.out StdErr=/logs/job.err WorkDir=/flows/run\n"
    )

    parsed = parse_scontrol_output(output)

    assert parsed["state"] == "RUNNING"
    assert parsed["stdout"] == "/logs/job.out"
    assert parsed["stderr"] == "/logs/job.err"
    assert parsed["workdir"] == "/flows/run"
    assert parsed["jobname"] == "TiO2-static"


def test_parse_sacct_row_prefers_matching_job_id():
    output = (
        "999|FAILED|1:0|other|/logs/other.out|/flows/other\n"
        "12345|COMPLETED|0:0|TiO2-static|/logs/job.out|/logs/job.err|/flows/run\n"
    )

    parsed = parse_sacct_row("12345.batch", output)

    assert parsed["state"] == "COMPLETED"
    assert parsed["exit"] == "0:0"
    assert parsed["jobname"] == "TiO2-static"
    assert parsed["stdout"] == "/logs/job.out"
    assert parsed["stderr"] == "/logs/job.err"
    assert parsed["workdir"] == "/flows/run"


def test_remote_job_status_merges_squeue_scontrol_and_sacct():
    status = remote_job_status_from_slurm_outputs(
        "12345",
        squeue_output="",
        scontrol_output="JobId=12345 JobName=TiO2-static JobState=COMPLETED StdOut=/logs/current.out StdErr=/logs/current.err WorkDir=/flows/current",
        sacct_output="12345|COMPLETED|0:0|TiO2-static|/logs/final.out|/logs/final.err|/flows/final",
        sacct_brief_output="12345|TiO2-static|COMPLETED|00:02:10|2026|2026|leeburton-pool",
    )

    assert isinstance(status, RemoteJobStatus)
    assert status.job_id == "12345"
    assert status.state == "COMPLETED"
    assert status.exit_code == "0:0"
    assert status.stdout_path == "/logs/final.out"
    assert status.stderr_path == "/logs/final.err"
    assert status.workdir == "/flows/final"
    assert status.job_name == "TiO2-static"
    assert status.raw["summary"] == "SUCCESS"
    assert "TiO2-static" in status.raw["brief"]


def test_remote_job_status_reports_parsing_stage_with_raw_output():
    try:
        remote_job_status_from_slurm_outputs(
            "12345",
            squeue_output="RUNNING\n",
            scontrol_output=object(),
            command="/usr/bin/scontrol show job 12345",
        )
    except MonitoringStageError as exc:
        assert exc.stage == "parsing"
        assert "scontrol" in exc.command
        assert "RUNNING" in exc.stdout
        assert exc.exception_text
    else:
        raise AssertionError("MonitoringStageError was not raised")


class FakeRunner:
    def __init__(self, status, expected_job_id="12345"):
        self.status = status
        self.expected_job_id = expected_job_id
        self.connected = False
        self.closed = False

    def connect(self, profile):
        self.connected = True
        self.profile = profile

    def query_job(self, job_id):
        assert job_id == self.expected_job_id
        return self.status

    def close(self):
        self.closed = True


def test_monitor_job_uses_runner_and_closes_connection():
    status = RemoteJobStatus(
        job_id="12345",
        state="RUNNING",
        stdout_path="/logs/job.out",
        workdir="/flows/run",
        job_name="TiO2-static",
        raw={"summary": "RUNNING", "brief": "12345|TiO2-static|RUNNING"},
    )
    runner = FakeRunner(status)
    submission_spec = {
        "cluster": {
            "ssh_config_host": NOTEBOOK_DEFAULTS["ssh_config_host"],
            "remote_host": NOTEBOOK_DEFAULTS["remote_host"],
            "username": NOTEBOOK_DEFAULTS["username"],
            "port": NOTEBOOK_DEFAULTS["port"],
        }
    }

    result = monitor_job(
        "12345.batch",
        submission_spec=submission_spec,
        runner_factory=lambda: runner,
    )

    assert runner.connected is True
    assert runner.closed is True
    assert result["status"] == "success"
    assert result["job_id"] == "12345"
    assert result["slurm_state"] == "RUNNING"
    assert result["summary"] == "RUNNING"
    assert result["brief"] == "12345|TiO2-static|RUNNING"


def test_resume_monitoring_uses_canonical_profile_for_running_job():
    status = RemoteJobStatus(
        job_id="12345",
        state="RUNNING",
        stdout_path="/logs/job.out",
        workdir="/flows/run",
        job_name="TiO2-static",
        raw={"summary": "RUNNING", "brief": "12345|TiO2-static|RUNNING"},
    )
    runner = FakeRunner(status)

    result = monitor_job(
        "12345",
        runner_factory=lambda: runner,
    )

    assert runner.connected is True
    assert runner.closed is True
    assert runner.profile.host == DEFAULT_REMOTE_HOST
    assert runner.profile.username == DEFAULT_USERNAME
    assert result["status"] == "success"
    assert result["job_id"] == "12345"
    assert result["slurm_state"] == "RUNNING"
    assert result["summary"] == "RUNNING"


def test_resume_monitoring_handles_completed_job():
    status = RemoteJobStatus(
        job_id="12345",
        state="COMPLETED",
        exit_code="0:0",
        stdout_path="/logs/final.out",
        workdir="/flows/run",
        job_name="TiO2-static",
        raw={"summary": "SUCCESS", "brief": "12345|TiO2-static|COMPLETED"},
    )
    runner = FakeRunner(status)

    result = monitor_job(
        "12345.batch",
        runner_factory=lambda: runner,
    )

    assert result["status"] == "success"
    assert result["job_id"] == "12345"
    assert result["slurm_state"] == "COMPLETED"
    assert result["summary"] == "SUCCESS"
    assert result["exit_code"] == "0:0"


def test_resume_monitoring_reports_nonexistent_job():
    status = RemoteJobStatus(
        job_id="99999",
        state="UNKNOWN",
        raw={
            "summary": "UNKNOWN",
            "brief": "99999|UNKNOWN",
            "command": "/usr/bin/scontrol show job 99999\n/usr/bin/squeue -j 99999 -h -o %T",
            "scontrol": "",
            "squeue": "",
            "sacct": "",
            "sacct_brief": "",
            "scontrol_stderr": "",
            "squeue_stderr": "",
            "sacct_stderr": "",
            "sacct_brief_stderr": "",
        },
    )
    runner = FakeRunner(status, expected_job_id="99999")

    result = monitor_job(
        "99999",
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Job lookup"
    assert "No SLURM record" in result["reason"]
    assert "/usr/bin/scontrol" in result["command"]


def test_resume_monitoring_rejects_malformed_input_without_connecting():
    class UnusedRunner(FakeRunner):
        def __init__(self):
            super().__init__(None)

        def connect(self, profile):
            raise AssertionError("Malformed job IDs should not connect")

    result = monitor_job(
        "12345;rm -rf /",
        runner_factory=UnusedRunner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Job ID"
    assert "Malformed" in result["reason"]


def test_monitor_job_rejects_missing_job_id():
    result = monitor_job("", submission_spec={})

    assert result["status"] == "failed"
    assert result["stage"] == "Job ID"


class FailingQueryRunner(FakeRunner):
    def __init__(self):
        super().__init__(None)

    def query_job(self, job_id):
        raise MonitoringStageError(
            "squeue",
            "squeue command failed before returning a result.",
            command="/usr/bin/squeue -j 12345 -h -o %T",
            stdout="partial stdout",
            stderr="squeue stderr",
            exception=TimeoutError("squeue timed out"),
        )


def test_monitor_job_preserves_query_failure_stage_and_streams():
    runner = FailingQueryRunner()
    submission_spec = {
        "cluster": {
            "ssh_config_host": NOTEBOOK_DEFAULTS["ssh_config_host"],
            "remote_host": NOTEBOOK_DEFAULTS["remote_host"],
            "username": NOTEBOOK_DEFAULTS["username"],
            "port": NOTEBOOK_DEFAULTS["port"],
        }
    }

    result = monitor_job(
        "12345",
        submission_spec=submission_spec,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "squeue"
    assert "/usr/bin/squeue" in result["command"]
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "squeue stderr"
    assert result["exception_text"] == "squeue timed out"


class FailingConnectRunner(FakeRunner):
    def __init__(self):
        super().__init__(None)

    def connect(self, profile):
        raise OSError("network is unreachable")


def test_monitor_job_reports_connection_failures_only_during_connect():
    result = monitor_job(
        "12345",
        submission_spec={
            "cluster": {
                "ssh_config_host": NOTEBOOK_DEFAULTS["ssh_config_host"],
                "remote_host": NOTEBOOK_DEFAULTS["remote_host"],
                "username": NOTEBOOK_DEFAULTS["username"],
                "port": NOTEBOOK_DEFAULTS["port"],
            }
        },
        runner_factory=FailingConnectRunner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "SSH Connection"
    assert "network is unreachable" in result["exception_text"]


def test_resume_uses_canonical_connection_profile_defaults():
    profile = default_connection_profile()
    submitted_profile = connection_profile_from_submission_spec({
        "cluster": {
            "ssh_config_host": NOTEBOOK_DEFAULTS["ssh_config_host"],
            "remote_host": NOTEBOOK_DEFAULTS["remote_host"],
            "username": NOTEBOOK_DEFAULTS["username"],
            "port": NOTEBOOK_DEFAULTS["port"],
        },
    })

    assert profile == submitted_profile
    assert profile.host == DEFAULT_REMOTE_HOST
    assert profile.username == DEFAULT_USERNAME


if __name__ == "__main__":
    test_strip_job_id_removes_slurm_suffixes()
    test_slurm_job_id_validation()
    test_classify_slurm_state_matches_notebook_rules()
    test_parse_scontrol_output_extracts_monitoring_fields()
    test_parse_sacct_row_prefers_matching_job_id()
    test_remote_job_status_merges_squeue_scontrol_and_sacct()
    test_remote_job_status_reports_parsing_stage_with_raw_output()
    test_monitor_job_uses_runner_and_closes_connection()
    test_resume_monitoring_uses_canonical_profile_for_running_job()
    test_resume_monitoring_handles_completed_job()
    test_resume_monitoring_reports_nonexistent_job()
    test_resume_monitoring_rejects_malformed_input_without_connecting()
    test_monitor_job_rejects_missing_job_id()
    test_monitor_job_preserves_query_failure_stage_and_streams()
    test_monitor_job_reports_connection_failures_only_during_connect()
    test_resume_uses_canonical_connection_profile_defaults()
    print("monitoring smoke test passed")
