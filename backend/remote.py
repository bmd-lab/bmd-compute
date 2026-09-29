from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


NOTEBOOK_REMOTE_INTERACTIONS = {
    "session_lifecycle": [
        "open SSH session",
        "verify active transport",
        "set keepalive",
        "close tunnels",
        "close remote wrapper",
        "close SSH client",
    ],
    "tunnels": [
        "open local-to-remote MongoDB port forward",
        "close port-forward server",
        "stop tunnel thread",
    ],
    "commands": [
        "run shell commands",
        "run commands with module loads",
        "run commands with exported VASP/jobflow environment",
        "stream long-running command output",
        "run remote Python snippets",
    ],
    "remote_files": [
        "create directories",
        "write text files",
        "write binary files",
        "read text files",
        "read binary files",
        "upload local files",
        "download remote files",
        "inspect remote path existence/type",
        "set file mode",
        "create or replace symlinks",
        "move or rename files",
        "remove temporary files",
    ],
    "submission": [
        "write sbatch script",
        "prepare POTCAR symlinks",
        "submit script with sbatch",
        "parse scheduler job id",
    ],
    "monitoring": [
        "query squeue",
        "query scontrol",
        "query sacct",
        "install remote watcher script",
        "start remote watcher process",
        "read watcher JSON",
        "stop remote watcher process",
    ],
    "results": [
        "execute remote parser",
        "save parse_summary.json",
        "locate CONTCAR/POSCAR outputs",
        "download vasprun.xml/KPOINTS",
        "copy generated plots back to remote run directory",
    ],
}


class RemoteExecutionError(RuntimeError):
    """Raised by callers when a remote command result is unsuccessful."""

    def __init__(self, result: "RemoteCommandResult"):
        self.result = result
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"Remote command failed with return code {result.returncode}"
        )
        super().__init__(message)


REMOTE_OPERATION_BUSY_MESSAGE = (
    "BMD Compute is currently handling several remote operations. "
    "Please try again in a few seconds."
)


class RemoteOperationBusy(RuntimeError):
    """Raised when the in-process remote-operation capacity is saturated."""

    def __init__(
        self,
        *,
        limit: int,
        active: int,
        timeout_s: float,
    ):
        self.limit = limit
        self.active = active
        self.timeout_s = timeout_s
        super().__init__(REMOTE_OPERATION_BUSY_MESSAGE)


class SubmissionAttemptError(RuntimeError):
    """Raised when a durable submission attempt cannot be submitted safely."""

    suggestion = (
        "Refresh the calculation state. If the job may already have been submitted, "
        "check SLURM before preparing a new submission attempt."
    )


class SubmissionAttemptMismatch(SubmissionAttemptError):
    """Raised when an attempt id is reused with different calculation metadata."""

    suggestion = "Prepare the calculation again before submitting."


class SubmissionAttemptNotPrepared(SubmissionAttemptError):
    """Raised when submission is requested before durable preparation exists."""

    suggestion = "Run Prepare Remote successfully before submitting the calculation."


class SubmissionAttemptInProgress(SubmissionAttemptError):
    """Raised when an attempt is already or ambiguously being submitted."""

    suggestion = (
        "Do not press Submit again for this attempt. Check SLURM or resume the "
        "calculation using the job ID if it appears in the queue."
    )


class SubmissionAttemptAlreadySubmitted(SubmissionAttemptError):
    """Raised when Prepare is requested for an attempt that was already submitted.

    A submitted attempt's run directory (``submission.json``, the uploaded
    runtime package, ``run_job.py`` and the sbatch script) is immutable.
    """

    suggestion = (
        "This calculation was already submitted, so its prepared files are kept "
        "unchanged. Monitor it with its job ID, or build the calculation again to "
        "start a new submission attempt."
    )

    def __init__(self, message: str, *, job_id: str | None = None):
        super().__init__(message)
        self.job_id = job_id


@dataclass(frozen=True)
class RemoteConnectionProfile:
    """Connection settings for the cluster, without embedding SSH secrets."""

    host: str
    username: str | None = None
    port: int = 22
    keepalive_s: int | None = None
    key_file: str | None = None
    ssh_config_host: str | None = None


@dataclass(frozen=True)
class RemoteCommandResult:
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> "RemoteCommandResult":
        if not self.ok:
            raise RemoteExecutionError(self)
        return self


@dataclass(frozen=True)
class RemotePathInfo:
    path: str
    exists: bool
    kind: str | None = None
    size_bytes: int | None = None
    mtime: float | None = None


@dataclass(frozen=True)
class RemoteTransferResult:
    remote_path: str
    local_path: str | None = None
    bytes_transferred: int | None = None
    mode: int | None = None


@dataclass(frozen=True)
class RemoteTunnel:
    tunnel_id: str
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int


@dataclass(frozen=True)
class RemoteProcess:
    process_id: str | None
    command: str
    pid_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None


@dataclass(frozen=True)
class BatchSubmissionRequest:
    script_path: str
    partition: str
    account: str
    nodes: int
    ntasks: int
    mem_gb: int
    walltime: str
    parsable: bool = True


@dataclass(frozen=True)
class BatchSubmissionResult:
    job_id: str
    raw_output: str = ""
    command: str = ""


@dataclass(frozen=True)
class RemoteJobStatus:
    job_id: str
    state: str
    exit_code: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    workdir: str | None = None
    job_name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobRecord:
    job_id: str | None
    run_name: str
    run_dir: str
    remote_script: str
    log_paths: dict[str, str]
    cluster: dict[str, Any]
    resources: dict[str, Any]
    submitted_at: str
    raw_output: str = ""
    status: str = "submitted"
    submission_spec: dict[str, Any] = field(default_factory=dict)
    remote_state_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RemoteRunner(ABC):
    """
    Abstract boundary for all operations that cross onto the remote cluster.

    Implementations may use SSH, Paramiko, an API, or another transport, but
    callers should depend only on this interface. This class intentionally
    performs no remote execution.
    """

    @abstractmethod
    def connect(self, profile: RemoteConnectionProfile) -> None:
        """Open the configured remote session."""
        raise NotImplementedError

    @abstractmethod
    def ensure_available(self) -> None:
        """Raise if the remote session cannot currently be used."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release remote resources, tunnels, and transport handles."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        check: bool = False,
        modules: bool = False,
        export_env: bool = False,
        timeout_s: float | None = None,
    ) -> RemoteCommandResult:
        """Run a non-interactive command on the cluster."""
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        command: str,
        *,
        timeout_s: float | None = None,
    ) -> RemoteCommandResult:
        """Run a command whose output may need to be streamed while it runs."""
        raise NotImplementedError

    @abstractmethod
    def run_python(
        self,
        source: str,
        *,
        python: str,
        env: Mapping[str, str] | None = None,
        check: bool = False,
        timeout_s: float | None = None,
    ) -> RemoteCommandResult:
        """Run Python source code with a selected remote Python interpreter."""
        raise NotImplementedError

    @abstractmethod
    def stat(self, remote_path: str) -> RemotePathInfo:
        """Inspect a remote path."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Return whether a remote path exists."""
        raise NotImplementedError

    @abstractmethod
    def is_file(self, remote_path: str) -> bool:
        """Return whether a remote path is a regular file."""
        raise NotImplementedError

    @abstractmethod
    def is_dir(self, remote_path: str) -> bool:
        """Return whether a remote path is a directory."""
        raise NotImplementedError

    @abstractmethod
    def ensure_directory(self, remote_path: str) -> RemotePathInfo:
        """Ensure a remote directory exists."""
        raise NotImplementedError

    @abstractmethod
    def chmod(self, remote_path: str, mode: int) -> RemotePathInfo:
        """Set a remote file mode."""
        raise NotImplementedError

    @abstractmethod
    def symlink(self, target: str, link_name: str, *, overwrite: bool = True) -> RemotePathInfo:
        """Create or replace a remote symbolic link."""
        raise NotImplementedError

    @abstractmethod
    def rename(self, source: str, destination: str, *, overwrite: bool = True) -> RemotePathInfo:
        """Move or rename a remote path."""
        raise NotImplementedError

    @abstractmethod
    def remove(self, remote_path: str, *, missing_ok: bool = True) -> RemotePathInfo:
        """Remove a remote path or temporary file."""
        raise NotImplementedError

    @abstractmethod
    def put_text(self, remote_path: str, text: str, *, mode: int = 0o640) -> RemoteTransferResult:
        """Write text to a remote path."""
        raise NotImplementedError

    @abstractmethod
    def read_text(self, remote_path: str, *, max_bytes: int | None = None) -> str:
        """Read text from a remote path."""
        raise NotImplementedError

    @abstractmethod
    def put_bytes(self, remote_path: str, data: bytes, *, mode: int = 0o640) -> RemoteTransferResult:
        """Write bytes to a remote path."""
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, remote_path: str, *, max_bytes: int | None = None) -> bytes:
        """Read bytes from a remote path."""
        raise NotImplementedError

    @abstractmethod
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        *,
        mode: int | None = None,
    ) -> RemoteTransferResult:
        """Upload a local file to the cluster."""
        raise NotImplementedError

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> RemoteTransferResult:
        """Download a remote file to a local path."""
        raise NotImplementedError

    @abstractmethod
    def open_tunnel(
        self,
        *,
        local_port: int,
        remote_host: str,
        remote_port: int,
        local_host: str = "127.0.0.1",
    ) -> RemoteTunnel:
        """Open a local-to-remote tunnel."""
        raise NotImplementedError

    @abstractmethod
    def close_tunnel(self, tunnel_id: str) -> None:
        """Close one tunnel by id."""
        raise NotImplementedError

    @abstractmethod
    def close_tunnels(self) -> None:
        """Close every tunnel owned by this runner."""
        raise NotImplementedError

    @abstractmethod
    def start_background(
        self,
        command: str,
        *,
        pid_path: str | None = None,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RemoteProcess:
        """Start a remote background process, such as the notebook watcher."""
        raise NotImplementedError

    @abstractmethod
    def stop_background(self, process: RemoteProcess | str, *, missing_ok: bool = True) -> None:
        """Stop a remote background process."""
        raise NotImplementedError

    @abstractmethod
    def submit_batch(self, request: BatchSubmissionRequest) -> BatchSubmissionResult:
        """Submit an already-written batch script to the remote scheduler."""
        raise NotImplementedError

    @abstractmethod
    def submit(self, submission_spec: Mapping[str, Any], dry_run: bool = False) -> JobRecord:
        """Submit a prepared SubmissionSpec and return a JobRecord."""
        raise NotImplementedError

    @abstractmethod
    def query_job(self, job_id: str) -> RemoteJobStatus:
        """Query scheduler state for a job."""
        raise NotImplementedError

    @abstractmethod
    def cancel_job(self, job_id: str) -> RemoteCommandResult:
        """Request cancellation of a scheduler job."""
        raise NotImplementedError

    @abstractmethod
    def read_job_accounting(self, job_id: str) -> RemoteJobStatus:
        """Read final scheduler accounting information for a job."""
        raise NotImplementedError


__all__ = [
    "BatchSubmissionRequest",
    "BatchSubmissionResult",
    "JobRecord",
    "NOTEBOOK_REMOTE_INTERACTIONS",
    "RemoteCommandResult",
    "RemoteConnectionProfile",
    "RemoteExecutionError",
    "RemoteJobStatus",
    "RemoteOperationBusy",
    "REMOTE_OPERATION_BUSY_MESSAGE",
    "RemotePathInfo",
    "RemoteProcess",
    "RemoteRunner",
    "RemoteTransferResult",
    "RemoteTunnel",
    "SubmissionAttemptError",
    "SubmissionAttemptInProgress",
    "SubmissionAttemptMismatch",
    "SubmissionAttemptNotPrepared",
]
