import json

from backend.config import (
    DEFAULT_ACCOUNT,
    DEFAULT_FLOWS_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_PARTITION,
    DEFAULT_POTCAR_DIR,
    DEFAULT_RESOURCES,
    POTCAR_LINK_MAP,
)
from backend.monitoring import MonitoringStageError
from backend.paramiko_remote import (
    DEFAULT_MONITOR_COMMAND_TIMEOUT_S,
    JobRecord,
    ParamikoRemoteRunner,
    _identity_files_from_ssh_config,
    _openssh_host_patterns_match,
    _parse_openssh_config_output,
    _paramiko_connect_details_from_ssh_options,
)
from backend.remote import (
    BatchSubmissionResult,
    RemoteCommandResult,
    RemoteConnectionProfile,
    RemotePathInfo,
)
from backend.submission import (
    create_submission_spec,
    parse_sbatch_job_id,
    remote_preparation_file_groups,
)


RUN_NAME = "TiO2-static-20260629-120000"
RUN_DIR = f"{DEFAULT_FLOWS_DIR}/{RUN_NAME}"
REMOTE_SCRIPT = f"{DEFAULT_FLOWS_DIR}/{RUN_NAME}.sbatch.sh"
LOG_OUT = f"{DEFAULT_LOGS_DIR}/{RUN_NAME}.out"
SLURM_OUT = f"{DEFAULT_LOGS_DIR}/{RUN_NAME}.slurm.out"
POTCAR_TARGET = f"{DEFAULT_POTCAR_DIR}/PBE_64"
SUBMISSION_JSON = f"{RUN_DIR}/submission.json"
BACKEND_DIR = f"{RUN_DIR}/backend"
RUN_JOB = f"{RUN_DIR}/run_job.py"


ssh_g_options = _parse_openssh_config_output(
    """
hostname powerslurm-login.tau.ac.il
user bmdguest
port 22
identityfile ~/.ssh/bmd_guest_ed25519
userknownhostsfile ~/.ssh/known_hosts ~/.ssh/known_hosts2
globalknownhostsfile /etc/ssh/ssh_known_hosts
proxycommand none
"""
)
assert ssh_g_options["hostname"] == "powerslurm-login.tau.ac.il"
assert ssh_g_options["user"] == "bmdguest"
assert ssh_g_options["port"] == "22"
assert ssh_g_options["userknownhostsfile"] == [
    "~/.ssh/known_hosts",
    "~/.ssh/known_hosts2",
]
assert ssh_g_options["globalknownhostsfile"] == ["/etc/ssh/ssh_known_hosts"]
identity_file = _identity_files_from_ssh_config(ssh_g_options)
assert identity_file.replace("\\", "/").endswith("/.ssh/bmd_guest_ed25519")

quoted_identity_options = _parse_openssh_config_output(
    'identityfile "C:/Users/lalbu/.ssh/key with spaces"\n'
)
assert _identity_files_from_ssh_config(quoted_identity_options).replace("\\", "/").endswith(
    "/.ssh/key with spaces"
)


class FakeParamikoModule:
    class ProxyCommand:
        def __init__(self, command):
            self.command = command


profile = RemoteConnectionProfile(
    host="powerslurm-bmdguest",
    username="bmdguest",
    port=22,
    key_file="C:/explicit/key",
    ssh_config_host="powerslurm-bmdguest",
)

connect_kwargs, connect_diagnostics = _paramiko_connect_details_from_ssh_options(
    profile,
    "powerslurm-bmdguest",
    {
        "hostname": "powerslurm-login.tau.ac.il",
        "user": "cluster-user",
        "port": "2222",
        "identityfile": ["C:/Users/lalbu/.ssh/bmd_guest_ed25519"],
        "proxycommand": "ssh jump-host nc powerslurm-login.tau.ac.il 2222",
    },
    {
        "source": "openssh ssh -G",
        "host_entry_found": True,
        "loaded_config_files": ["C:/Users/lalbu/.ssh/config"],
        "matching_host_patterns": ["powerslurm-bmdguest"],
    },
    FakeParamikoModule,
)
assert connect_kwargs["hostname"] == "powerslurm-login.tau.ac.il"
assert connect_kwargs["username"] == "cluster-user"
assert connect_kwargs["port"] == 2222
assert connect_kwargs["key_filename"].replace("\\", "/").endswith(
    "/.ssh/bmd_guest_ed25519"
)
assert connect_kwargs["sock"].command == "ssh jump-host nc powerslurm-login.tau.ac.il 2222"
assert connect_diagnostics["ssh_config_applied"] is True
assert connect_diagnostics["resolved_hostname"] == "powerslurm-login.tau.ac.il"
assert connect_diagnostics["resolved_username"] == "cluster-user"
assert connect_diagnostics["resolved_port"] == 2222
assert connect_diagnostics["ssh_config_lookup"]["loaded_config_files"] == [
    "C:/Users/lalbu/.ssh/config"
]
assert connect_diagnostics["ssh_config_lookup"]["host_entry_found"] is True

fallback_kwargs, fallback_diagnostics = _paramiko_connect_details_from_ssh_options(
    profile,
    "powerslurm-bmdguest",
    {
        "hostname": "powerslurm-bmdguest",
        "user": "local-dev-user",
        "port": "22",
        "identityfile": ["~/.ssh/id_ed25519"],
    },
    {
        "source": "openssh ssh -G",
        "host_entry_found": False,
        "loaded_config_files": [],
        "matching_host_patterns": [],
    },
    FakeParamikoModule,
)
assert fallback_kwargs["hostname"] == "powerslurm-bmdguest"
assert fallback_kwargs["username"] == "bmdguest"
assert fallback_kwargs["port"] == 22
assert fallback_kwargs["key_filename"] == "C:/explicit/key"
assert "sock" not in fallback_kwargs
assert fallback_diagnostics["ssh_config_applied"] is False

included_config_kwargs, included_config_diagnostics = _paramiko_connect_details_from_ssh_options(
    profile,
    "powerslurm-bmdguest",
    {
        "hostname": "powerslurm-login.tau.ac.il",
        "user": "included-user",
        "port": "22",
    },
    {
        "source": "openssh ssh -G",
        "host_entry_found": False,
        "loaded_config_files": [],
        "matching_host_patterns": [],
    },
    FakeParamikoModule,
)
assert included_config_kwargs["hostname"] == "powerslurm-login.tau.ac.il"
assert included_config_kwargs["username"] == "included-user"
assert included_config_diagnostics["ssh_config_applied"] is True

assert _openssh_host_patterns_match("powerslurm-bmdguest", ["powerslurm-*"])
assert not _openssh_host_patterns_match(
    "powerslurm-bmdguest",
    ["powerslurm-*", "!powerslurm-bmdguest"],
)


class RecordingRunner(ParamikoRemoteRunner):
    def __init__(self):
        super().__init__(client=object())
        self.commands = []
        self.batch_requests = []
        self.directories = []
        self.remote_writes = []
        self.remote_files = {}
        self.remote_reads = []
        self.symlinks = []
        self.renames = []

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
        if command.startswith("test -f"):
            return RemoteCommandResult(command=command, returncode=0)
        return RemoteCommandResult(command=command, returncode=0)

    def ensure_directory(self, remote_path):
        self.directories.append(remote_path)
        return RemotePathInfo(path=remote_path, exists=True, kind="dir")

    def is_dir(self, remote_path):
        return True

    def is_file(self, remote_path):
        return remote_path in self.remote_files

    def symlink(self, target, link_name, *, overwrite=True):
        self.symlinks.append((target, link_name, overwrite))
        return RemotePathInfo(path=link_name, exists=True, kind="dir")

    def put_text(self, remote_path, text, *, mode=0o640):
        self.remote_writes.append((remote_path, text, mode))
        self.remote_files[remote_path] = text

    def _replace_file(self, source, destination):
        # Atomic record writes land at the destination only through a rename.
        text = self.remote_files.pop(source)
        self.remote_files[destination] = text
        self.renames.append((source, destination))
        self.remote_writes.append((destination, text, None))

    def read_text(self, remote_path, *, max_bytes=None):
        self.remote_reads.append(remote_path)
        return self.remote_files[remote_path]

    def submit_batch(self, request):
        self.batch_requests.append(request)
        return BatchSubmissionResult(
            job_id="123456",
            raw_output="123456\n",
            command=(
                f"sbatch -p {request.partition} -A {request.account} "
                f"-N {request.nodes} -n {request.ntasks} "
                f"--mem={request.mem_gb}G -t {request.walltime} --parsable "
                f"{request.script_path}"
            ),
        )


class FakeSftpFile:
    def __init__(self, sftp, remote_path):
        self.sftp = sftp
        self.remote_path = remote_path
        self.parts = []

    def write(self, data):
        self.parts.append(data.decode("utf-8") if isinstance(data, bytes) else data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.sftp.files[self.remote_path] = "".join(self.parts)


class FakeSftp:
    def __init__(self):
        self.files = {}
        self.modes = {}
        self.close_count = 0

    def file(self, remote_path, mode):
        assert mode == "w"
        return FakeSftpFile(self, remote_path)

    def chmod(self, remote_path, mode):
        self.modes[remote_path] = mode

    def posix_rename(self, source, destination):
        self.files[destination] = self.files.pop(source)
        if source in self.modes:
            self.modes[destination] = self.modes.pop(source)

    def remove(self, remote_path):
        self.files.pop(remote_path, None)
        self.modes.pop(remote_path, None)

    def close(self):
        self.close_count += 1


class FakeSftpClient:
    def __init__(self):
        self.sftp = FakeSftp()
        self.open_count = 0

    def open_sftp(self):
        self.open_count += 1
        return self.sftp


class SftpPreparationRunner(ParamikoRemoteRunner):
    def __init__(self):
        self.fake_client = FakeSftpClient()
        super().__init__(client=self.fake_client)
        self.commands = []
        self.directories = []

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
        self._ssh_exec_count += 1
        self.commands.append(command)
        return RemoteCommandResult(command=command, returncode=0)

    def ensure_directory(self, remote_path):
        self.directories.append(remote_path)
        return RemotePathInfo(path=remote_path, exists=True, kind="dir")

    def is_dir(self, remote_path):
        return True

    def is_file(self, remote_path):
        return remote_path in self.fake_client.sftp.files

    def submit_batch(self, request):
        raise AssertionError("large dry-run regression should not submit a batch job")


flow_spec = {
    "workflow": "static",
    "potcar_functional": "PBE_64",
    "kpoints": None,
    "incar": {},
    "structure": {"type": "path", "path": "/remote/POSCAR"},
}

submission_spec = create_submission_spec(
    flow_spec,
    label="TiO2 static",
    timestamp="20260629-120000",
    env={},
)

runner = RecordingRunner()
runner.submit(submission_spec, dry_run=True)
record = runner.submit(submission_spec)

assert isinstance(record, JobRecord)
assert record.job_id == "123456"
assert record.status == "submitted"
assert record.run_name == RUN_NAME
assert record.run_dir == RUN_DIR
assert record.remote_script == REMOTE_SCRIPT
assert record.log_paths["stdout"] == LOG_OUT
assert record.log_paths["slurm_out"] == SLURM_OUT
assert record.remote_state_path == f"{DEFAULT_LOGS_DIR}/job_123456.json"

assert runner.commands[0] == "test -f /remote/POSCAR || test -d /remote/POSCAR"
assert runner.batch_requests
batch_request = runner.batch_requests[0]
assert batch_request.script_path == REMOTE_SCRIPT
assert batch_request.partition == DEFAULT_PARTITION
assert batch_request.account == DEFAULT_ACCOUNT
assert batch_request.nodes == DEFAULT_RESOURCES["nodes"]
assert batch_request.ntasks == DEFAULT_RESOURCES["ntasks"]
assert batch_request.mem_gb == DEFAULT_RESOURCES["mem_gb"]
assert batch_request.walltime == DEFAULT_RESOURCES["walltime"]
assert (
    f"-N {DEFAULT_RESOURCES['nodes']} "
    f"-n {DEFAULT_RESOURCES['ntasks']} "
    f"--mem={DEFAULT_RESOURCES['mem_gb']}G "
    f"-t {DEFAULT_RESOURCES['walltime']} --parsable"
) in record.raw_output

assert runner.remote_writes
writes = {path: text for path, text, _mode in runner.remote_writes}
assert SUBMISSION_JSON in writes
assert f"{BACKEND_DIR}/execution.py" in writes
assert f"{BACKEND_DIR}/parser.py" in writes
assert f"{BACKEND_DIR}/workflows.py" in writes
assert RUN_JOB in writes
assert REMOTE_SCRIPT in writes
assert f"{DEFAULT_LOGS_DIR}/job_123456.json" in writes
assert "__SPEC_JSON__" not in writes[RUN_JOB]
assert "json.load(handle)" in writes[RUN_JOB]
assert "from backend.execution import run_submission" in writes[RUN_JOB]
assert f"export BMD_SUBMISSION_SPEC={SUBMISSION_JSON}" in writes[REMOTE_SCRIPT]
assert f"Missing submission.json at {SUBMISSION_JSON}" in writes[REMOTE_SCRIPT]
assert f"Missing execution module at {BACKEND_DIR}/execution.py" in writes[REMOTE_SCRIPT]
assert f"Missing backend module parser.py at {BACKEND_DIR}/parser.py" in writes[REMOTE_SCRIPT]
assert f"Missing backend module workflows.py at {BACKEND_DIR}/workflows.py" in writes[REMOTE_SCRIPT]
assert f"Missing run_job.py at {RUN_JOB}" in writes[REMOTE_SCRIPT]
assert all("cat >" not in command for command in runner.commands)

dry_runner = RecordingRunner()
dry_record = dry_runner.submit(submission_spec, dry_run=True)

assert isinstance(dry_record, JobRecord)
assert dry_record.job_id is None
assert dry_record.status == "dry_run"
assert dry_record.run_name == RUN_NAME
assert dry_record.run_dir == RUN_DIR
assert dry_record.remote_script == REMOTE_SCRIPT
assert dry_record.remote_state_path is None
assert dry_record.raw_output.endswith("DRY RUN\n")
assert "PREP_OK=Remote directories prepared" in dry_record.raw_output
assert "PREP_OK=Working directory created" in dry_record.raw_output
assert "PREP_OK=submission.json uploaded" in dry_record.raw_output
assert "PREP_OK=Execution module uploaded" in dry_record.raw_output
assert "PREP_OK=run_job.py uploaded" in dry_record.raw_output
assert "PREP_OK=Runtime import preflight" in dry_record.raw_output
assert "PREP_OK=Submission script written" in dry_record.raw_output
assert "PREP_OK=Ready for submission" in dry_record.raw_output

assert dry_runner.commands[0] == "test -f /remote/POSCAR || test -d /remote/POSCAR"
assert not dry_runner.batch_requests
dry_writes = {path: text for path, text, _mode in dry_runner.remote_writes}
assert SUBMISSION_JSON in dry_writes
assert RUN_JOB in dry_writes
assert REMOTE_SCRIPT in dry_writes
assert "__SPEC_JSON__" not in dry_writes[RUN_JOB]
assert all("cat >" not in command for command in dry_runner.commands)
assert "Submitting with: sbatch" not in dry_record.raw_output

large_payload = "KPOINTS\n" + (
    "0.0000000000 0.0000000000 0.0000000000 0 ! Gamma\n" * 15000
)
large_flow_spec = {
    **flow_spec,
    "generated_input_regression_payload": large_payload,
}
large_submission_spec = create_submission_spec(
    large_flow_spec,
    label="TiO2 large static",
    timestamp="20260629-120000",
    env={},
)
large_runner = SftpPreparationRunner()
large_record = large_runner.submit(large_submission_spec, dry_run=True)
large_submission_json = f"{large_submission_spec['paths']['run_dir']}/submission.json"
large_files = large_runner.fake_client.sftp.files
assert large_record.status == "dry_run"
assert "PREP_OK=Runtime import preflight" in large_record.raw_output
assert "PREP_OK=Ready for submission" in large_record.raw_output
large_uploaded_spec = json.loads(large_files[large_submission_json])
assert large_uploaded_spec["flow_spec"]["generated_input_regression_payload"] == large_payload
assert large_runner.fake_client.sftp.modes[large_submission_json] == 0o640
assert all("cat >" not in command for command in large_runner.commands)
assert all(large_payload[:200] not in command for command in large_runner.commands)
assert max((len(command) for command in large_runner.commands), default=0) < 4096
assert large_runner.fake_client.open_count == 1
assert large_runner.fake_client.sftp.close_count == 1
expected_preparation_files = sum(
    len(group.get("files", []))
    for group in remote_preparation_file_groups(large_submission_spec)
)
assert large_runner.preparation_metrics["files_uploaded"] == expected_preparation_files + 1
assert large_runner.preparation_metrics["bytes_uploaded"] > len(large_payload.encode("utf-8"))
assert large_runner.preparation_metrics["sftp_session_count"] == 1
assert large_runner.preparation_metrics["ssh_exec_command_count"] == len(
    large_runner.commands
)
assert large_runner.preparation_metrics["runtime_preflight_s"] >= 0
assert large_runner.preparation_metrics["attempt_state_s"] >= 0
backend_init_path = f"{large_submission_spec['paths']['run_dir']}/backend/__init__.py"
execution_group = next(
    group
    for group in remote_preparation_file_groups(large_submission_spec)
    if group["step"] == "Execution module uploaded"
)
assert [item["path"] for item in execution_group["files"]].count(backend_init_path) == 1
expected_uploads = {
    item["path"]: item["mode"]
    for group in remote_preparation_file_groups(large_submission_spec)
    for item in group.get("files", [])
}
assert expected_uploads.keys() <= large_files.keys()
assert all(
    large_runner.fake_client.sftp.modes[path] == mode
    for path, mode in expected_uploads.items()
)
attempt_state_path = large_submission_spec["submission"]["attempt_state"]
assert json.loads(large_files[attempt_state_path])["state"] == "PREPARED"
assert large_runner.fake_client.sftp.modes[attempt_state_path] == 0o640
assert sum(
    command.count(
        f"mkdir -p -- {large_submission_spec['paths']['run_dir']}/backend &&"
    )
    for command in large_runner.commands
) == 1


class MissingUploadedFileRunner(SftpPreparationRunner):
    def run(
        self,
        command,
        *,
        check=False,
        modules=False,
        export_env=False,
        timeout_s=None,
    ):
        self._ssh_exec_count += 1
        self.commands.append(command)
        missing_path = f"{self.missing_run_dir}/backend/execution.py"
        if f"test -f {missing_path}" in command:
            return RemoteCommandResult(
                command=command,
                returncode=1,
                stderr=f"Expected file does not exist: {missing_path}\n",
            )
        return RemoteCommandResult(command=command, returncode=0)


missing_upload_runner = MissingUploadedFileRunner()
missing_upload_runner.missing_run_dir = large_submission_spec["paths"]["run_dir"]
try:
    missing_upload_runner.submit(large_submission_spec, dry_run=True)
except Exception as exc:
    assert "Execution module uploaded" in str(exc)
    assert f"{missing_upload_runner.missing_run_dir}/backend/execution.py" in str(exc)
else:
    raise AssertionError("Missing uploaded file verification should fail preparation")
assert missing_upload_runner.fake_client.open_count == 1
assert missing_upload_runner.fake_client.sftp.close_count == 1

private_potcars_dir = "/private/bmd-potcars"
private_target = f"{private_potcars_dir}/PBE_64"
private_link = f"{private_potcars_dir}/{POTCAR_LINK_MAP['PBE_64'][0]}"
private_submission_spec = create_submission_spec(
    flow_spec,
    label="TiO2 static",
    timestamp="20260629-120000",
    env={},
    potcars_dir=private_potcars_dir,
)

private_runner = RecordingRunner()
private_runner.submit(private_submission_spec, dry_run=True)
private_runner.submit(private_submission_spec)
assert (private_target, private_link, True) in private_runner.symlinks
assert any(private_link in command for command in private_runner.commands)
assert all(not path.startswith(private_potcars_dir) for path in private_runner.remote_reads)

private_dry_runner = RecordingRunner()
private_dry_record = private_dry_runner.submit(private_submission_spec, dry_run=True)
assert (private_target, private_link, True) in private_dry_runner.symlinks
assert any(private_link in command for command in private_dry_runner.commands)
assert "PREP_OK=POTCAR links prepared" in private_dry_record.raw_output

assert parse_sbatch_job_id("Submitted batch job 42") == "42"
assert parse_sbatch_job_id("SBATCH_RAW_OUT=77") == "77"


class QueryRunner(ParamikoRemoteRunner):
    def __init__(self, outputs):
        super().__init__(client=object())
        self.outputs = outputs
        self.commands = []
        self.timeout_values = []

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
        self.timeout_values.append(timeout_s)
        if "/usr/bin/scontrol" in command:
            key = "scontrol"
            stdout = self.outputs.get("scontrol", "")
        elif "/usr/bin/squeue" in command:
            key = "squeue"
            stdout = self.outputs.get("squeue", "")
        elif "JobIDRaw,State,ExitCode,JobName,StdOut,StdErr,WorkDir" in command:
            key = "sacct"
            stdout = self.outputs.get("sacct", "")
        elif "JobID,JobName%30,State,Elapsed,Start,End,Partition%20" in command:
            key = "sacct_brief"
            stdout = self.outputs.get("sacct_brief", "")
        else:
            key = ""
            stdout = ""
        return RemoteCommandResult(
            command=command,
            returncode=self.outputs.get("returncodes", {}).get(key, 0),
            stdout=stdout,
            stderr=self.outputs.get("stderr", {}).get(key, ""),
        )


running_query_runner = QueryRunner(
    {
        "scontrol": "JobId=123456 JobName=TiO2-static JobState=RUNNING StdOut=/logs/job.out WorkDir=/flows/run\n",
        "squeue": "RUNNING\n",
    }
)
running_status = running_query_runner.query_job("123456.batch")

assert running_status.job_id == "123456"
assert running_status.state == "RUNNING"
assert running_status.raw["summary"] == "RUNNING"
assert running_status.stdout_path == "/logs/job.out"
assert running_status.workdir == "/flows/run"
assert running_status.job_name == "TiO2-static"
assert len(running_query_runner.commands) == 2
assert all("/usr/bin/sacct" not in command for command in running_query_runner.commands)
assert all("timeout 2s" not in command for command in running_query_runner.commands)
assert running_query_runner.timeout_values == [
    DEFAULT_MONITOR_COMMAND_TIMEOUT_S,
    DEFAULT_MONITOR_COMMAND_TIMEOUT_S,
]

completed_query_runner = QueryRunner(
    {
        "scontrol": "",
        "squeue": "",
        "sacct": "123456|COMPLETED|0:0|TiO2-static|/logs/final.out|/logs/final.err|/flows/run\n",
        "sacct_brief": "123456|TiO2-static|COMPLETED|00:02:10|2026-07-14T10:00|2026-07-14T10:02|leeburton-pool\n",
    }
)
completed_status = completed_query_runner.query_job("123456")

assert completed_status.job_id == "123456"
assert completed_status.state == "COMPLETED"
assert completed_status.exit_code == "0:0"
assert completed_status.raw["summary"] == "SUCCESS"
assert completed_status.stdout_path == "/logs/final.out"
assert completed_status.stderr_path == "/logs/final.err"
assert "TiO2-static" in completed_status.raw["brief"]
assert len(completed_query_runner.commands) == 4
assert any("/usr/bin/sacct" in command for command in completed_query_runner.commands)
assert all("timeout 2s" not in command for command in completed_query_runner.commands)
assert completed_query_runner.timeout_values == [
    DEFAULT_MONITOR_COMMAND_TIMEOUT_S,
    DEFAULT_MONITOR_COMMAND_TIMEOUT_S,
    DEFAULT_MONITOR_COMMAND_TIMEOUT_S,
    DEFAULT_MONITOR_COMMAND_TIMEOUT_S,
]

completed_after_scontrol_miss_runner = QueryRunner(
    {
        "scontrol": "",
        "squeue": "",
        "sacct": "123456|COMPLETED|0:0|TiO2-static|/logs/final.out|/logs/final.err|/flows/run\n",
        "sacct_brief": "123456|TiO2-static|COMPLETED|00:02:10|2026-07-14T10:00|2026-07-14T10:02|leeburton-pool\n",
        "returncodes": {"scontrol": 1},
        "stderr": {"scontrol": "slurm_load_jobs error: Invalid job id specified\n"},
    }
)
completed_after_scontrol_miss = completed_after_scontrol_miss_runner.query_job("123456")

assert completed_after_scontrol_miss.state == "COMPLETED"
assert completed_after_scontrol_miss.raw["summary"] == "SUCCESS"
assert completed_after_scontrol_miss.raw["scontrol_stderr"] == "slurm_load_jobs error: Invalid job id specified"


class FailingSqueueRunner(QueryRunner):
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
        self.timeout_values.append(timeout_s)
        if "/usr/bin/squeue" in command:
            raise TimeoutError("squeue timeout")
        return RemoteCommandResult(command=command, returncode=0, stdout="")


failing_squeue_runner = FailingSqueueRunner({"scontrol": ""})
try:
    failing_squeue_runner.query_job("123456")
except MonitoringStageError as exc:
    assert exc.stage == "squeue"
    assert "/usr/bin/squeue" in exc.command
    assert "timeout 2s" not in exc.command
    assert exc.stdout == ""
    assert exc.stderr == ""
    assert exc.exception_text == "squeue timeout"
else:
    raise AssertionError("MonitoringStageError was not raised")

print("paramiko remote submit smoke test passed")
