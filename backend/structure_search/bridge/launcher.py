"""Host-side launch loop for USPEX relaxation steps.

Design: docs/structure_search.md, section 4; issue #66.

USPEX runs inside its Singularity image and cannot call ``srun``. For each
relaxation step it runs ``bmd_bridge.sh``, which drops a request file into a
spool directory and waits. This module runs on the host, in the same SLURM
allocation, and turns each request into a SLURM job step:

* watch ``<spool>/requests`` for new requests
* run each one in its CalcFold with a configurable command, by default
  ``srun --exact --mpi=pmi2 -n {ranks} --mem={mem} vasp_std``, at most
  ``max_parallel`` at a time and never two in the same CalcFold at once
* write ``<spool>/done/<id>.done`` with the exit status, which releases the
  waiting bridge script
* stop a step when its bridge gives up (``<spool>/cancel/<id>``)
* on stop (SIGTERM, end of job, or the USPEX master exiting): terminate the
  running steps, record them as stopped, and write ``<spool>/stopped`` so
  waiting bridge scripts exit at once

Every file the launcher writes is written to ``<spool>/tmp`` first and then
renamed into place, so a reader never sees a half-written file, also when
the launcher is killed. VASP's own files in a CalcFold that was stopped mid-step
are whatever VASP left; USPEX treats such a step as failed when it resumes.

The file protocol is described in ``bmd_bridge.sh``; names and exit codes
are in ``protocol.py``.

Intel MPI: the step environment always has ``I_MPI_DEBUG=4`` (so the pinning
report is in each step's log) and never ``I_MPI_PIN_DOMAIN``: in the lab's
benchmarks that variable silently overrode the processor list and placed
concurrent jobs on the same cores. If the caller's environment contains it,
it is removed and the removal is recorded in the journal.

Use from Python with ``Launcher`` (just the loop) or ``run_search`` (start
the USPEX master and run the loop until it exits). For manual pilots there is
a command line: ``python -m backend.structure_search.bridge.launcher --help``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from backend.structure_search.bridge import protocol
from backend.structure_search.bridge.protocol import StepRequest


DEFAULT_COMMAND_TEMPLATE = "srun --exact --mpi=pmi2 -n {ranks} --mem={mem} vasp_std"
"""Default step command. ``{ranks}``, ``{mem}`` and ``{step}`` are filled in.

If ``mem`` is ``None`` every argument that contains ``{mem}`` is left out,
so write the memory option as one argument (``--mem={mem}``), not two.
"""

MPI_ENVIRONMENT_OVERRIDES = {"I_MPI_DEBUG": "4"}
FORBIDDEN_STEP_ENVIRONMENT = ("I_MPI_PIN_DOMAIN",)
EXIT_COMMAND_NOT_STARTED = 127

_MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*[KMGT]?$")


def build_step_argv(template: str, *, ranks: int, mem: str | None, step: int) -> list[str]:
    """Return the command for one step from ``template``.

    The template is split like a shell command line (no shell runs it).
    Placeholders: ``{ranks}``, ``{mem}``, ``{step}``. With ``mem=None``,
    arguments containing ``{mem}`` are dropped.
    """

    try:
        tokens = shlex.split(template)
    except ValueError as exc:
        raise ValueError(f"command template cannot be parsed: {exc}") from exc
    argv: list[str] = []
    for token in tokens:
        if mem is None and "{mem}" in token:
            continue
        try:
            argv.append(token.format(ranks=ranks, mem=mem, step=step))
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"command template argument {token!r} is invalid; "
                "allowed placeholders are {ranks}, {mem} and {step}"
            ) from exc
    if not argv:
        raise ValueError("command template is empty")
    if mem is None and "--mem" in argv:
        raise ValueError("write the memory option as one argument, '--mem={mem}'")
    return argv


def build_step_environment(base: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Return the environment for VASP steps and the variables that were removed.

    ``base`` is used as given (nothing is inherited from this process), then
    ``I_MPI_DEBUG=4`` is set and ``I_MPI_PIN_DOMAIN`` is removed.
    """

    environment = {str(key): str(value) for key, value in base.items()}
    removed = [name for name in FORBIDDEN_STEP_ENVIRONMENT if environment.pop(name, None) is not None]
    environment.update(MPI_ENVIRONMENT_OVERRIDES)
    return environment, removed


def slurm_job_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return the ``SLURM_*`` variables of ``source`` (normally the batch job's environment).

    ``srun`` finds the allocation it runs in through these variables
    (``SLURM_JOB_ID`` and others). The step environment is otherwise built
    explicitly, so it must include them, or ``srun`` would ask for a new
    allocation instead of starting a step in the current one. Passing all of
    them matches what a plain ``srun`` in the batch script sees, which is how
    the Phase 0 step test (reference/uspex_pilot/steptest.sbatch) runs.
    """

    return {key: value for key, value in source.items() if key.startswith("SLURM_")}


@dataclass(frozen=True)
class LauncherSettings:
    """Configuration of the launch loop.

    ``mem`` has no default on purpose: whether each ``srun`` step inside a
    ``--mem=0`` allocation needs an explicit ``--mem`` is still to be shown
    by the cluster step test (#59). Pass ``None`` to leave ``--mem`` out.

    ``environment`` is the complete environment of the VASP steps (modules
    already loaded); nothing is inherited from the launcher's own process.
    It must contain the job's ``SLURM_*`` variables; see
    ``slurm_job_environment``.
    """

    spool_dir: Path
    search_dir: Path
    max_parallel: int
    ranks: int
    mem: str | None
    environment: Mapping[str, str]
    ranks_per_step: Mapping[int, int] = field(default_factory=dict)
    command_template: str = DEFAULT_COMMAND_TEMPLATE
    poll_seconds: float = 1.0
    heartbeat_interval_seconds: float = 10.0
    kill_grace_seconds: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "spool_dir", Path(self.spool_dir))
        object.__setattr__(self, "search_dir", Path(self.search_dir))
        object.__setattr__(self, "ranks_per_step", dict(self.ranks_per_step))
        object.__setattr__(self, "environment", dict(self.environment))
        for name in ("spool_dir", "search_dir"):
            if not getattr(self, name).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        if self.ranks < 1:
            raise ValueError("ranks must be at least 1")
        for step, ranks in self.ranks_per_step.items():
            if not isinstance(step, int) or step < 1 or not isinstance(ranks, int) or ranks < 1:
                raise ValueError("ranks_per_step maps step numbers (from 1) to rank counts (from 1)")
        if self.mem is not None and not _MEMORY_PATTERN.match(self.mem):
            raise ValueError(f"mem must look like '200G' or '4000M', got {self.mem!r}")
        for seconds in ("poll_seconds", "heartbeat_interval_seconds", "kill_grace_seconds"):
            value = getattr(self, seconds)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{seconds} must be a positive number")
        build_step_argv(self.command_template, ranks=self.ranks, mem=self.mem, step=1)

    def ranks_for_step(self, step: int) -> int:
        return self.ranks_per_step.get(step, self.ranks)

    def argv_for_step(self, step: int) -> list[str]:
        return build_step_argv(
            self.command_template,
            ranks=self.ranks_for_step(step),
            mem=self.mem,
            step=step,
        )


@dataclass
class _RunningStep:
    request: StepRequest
    calcfold: Path
    argv: list[str]
    process: subprocess.Popen
    log: IO[bytes]
    started_wall: float
    stop_status: int | None = None
    stop_reason: str = ""
    kill_at: float | None = None
    killed: bool = False


class Launcher:
    """Run bridge requests as host commands. See the module docstring."""

    def __init__(self, settings: LauncherSettings):
        self.settings = settings
        self.spool = settings.spool_dir
        self._search_root = settings.search_dir.resolve()
        self._step_environment, self._removed_environment = build_step_environment(
            settings.environment
        )
        self._running: dict[str, _RunningStep] = {}
        self._claimed_ids: set[str] = set()
        self._last_heartbeat = -math.inf
        self._prepared = False
        self._stopped = False

    # -- lifecycle -----------------------------------------------------------

    def prepare(self) -> None:
        """Create the spool and clear what an earlier launcher left behind.

        Call before the USPEX master starts. Requests that an earlier launcher
        claimed but never finished are answered with status 121 (launcher
        lost): their result is unknown, and running them again could put two
        VASP runs in one CalcFold if the old step is somehow still alive.
        """

        if self._prepared:
            return
        for sub in protocol.SPOOL_SUBDIRS:
            (self.spool / sub).mkdir(parents=True, exist_ok=True)
        (self.spool / protocol.STOPPED_FILE).unlink(missing_ok=True)
        self._remove_launcher_temporaries()
        self.journal(
            "started",
            pid=os.getpid(),
            max_parallel=self.settings.max_parallel,
            command_template=self.settings.command_template,
            ranks=self.settings.ranks,
            ranks_per_step={str(k): v for k, v in sorted(self.settings.ranks_per_step.items())},
            mem=self.settings.mem,
        )
        if self._removed_environment:
            self.journal("environment_removed", names=self._removed_environment)

        for path in sorted((self.spool / protocol.CLAIMED_DIR).iterdir()):
            request_id = protocol.request_id_from_filename(path.name)
            if request_id is None:
                continue
            if not self._done_path(request_id).exists():
                self._write_done(
                    request_id,
                    protocol.EXIT_LAUNCHER_LOST,
                    reason="claimed by an earlier launcher that did not finish it",
                )
                self.journal("recovered", id=request_id, status=protocol.EXIT_LAUNCHER_LOST)
            path.unlink(missing_ok=True)

        self._beat(force=True)
        self._prepared = True

    def run(
        self,
        stop_event: threading.Event | None = None,
        stop_when: Callable[[], bool] | None = None,
    ) -> None:
        """Run the loop until ``stop_event`` is set or ``stop_when()`` is true."""

        event = stop_event or threading.Event()
        self.prepare()
        try:
            while not event.is_set():
                self.poll_once()
                if stop_when is not None and stop_when():
                    break
                event.wait(self.settings.poll_seconds)
        finally:
            self.shutdown()

    def poll_once(self) -> None:
        """One pass of the loop: heartbeat, cancellations, finished steps, new steps."""

        self._beat()
        self._handle_cancellations()
        self._reap()
        self._start_requests()

    def shutdown(self) -> None:
        """Stop all running steps, record them as stopped and mark the spool stopped.

        Waiting bridge scripts see ``<spool>/stopped`` and withdraw their
        requests. Running steps get SIGTERM, then SIGKILL after
        ``kill_grace_seconds``, and are recorded with status 123.
        """

        if self._stopped:
            return
        self._stopped = True
        if not self._prepared:
            return
        protocol.write_atomically(
            self.spool, self.spool / protocol.STOPPED_FILE, f"{int(time.time())}\n"
        )
        for running in list(self._running.values()):
            self._terminate(running, protocol.EXIT_STOPPED, "launcher stopped")
        # SIGKILL follows after kill_grace_seconds; a process that survives
        # even that (for example, stuck in a file-system call) is recorded as
        # stopped and left to SLURM, which ends every process of the job.
        give_up_at = time.monotonic() + 2 * self.settings.kill_grace_seconds + 1.0
        while self._running:
            self._reap()
            if not self._running:
                break
            if time.monotonic() >= give_up_at:
                for request_id, running in list(self._running.items()):
                    self._finish(request_id, running, returncode=None)
                    self.journal("abandoned", id=request_id, pid=running.process.pid)
                break
            time.sleep(min(0.1, self.settings.poll_seconds))
        self._remove_launcher_temporaries()
        self.journal("stopped")

    @property
    def running_ids(self) -> tuple[str, ...]:
        return tuple(self._running)

    # -- loop parts ------------------------------------------------------------

    def _beat(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_heartbeat >= self.settings.heartbeat_interval_seconds:
            protocol.write_atomically(
                self.spool, self.spool / protocol.HEARTBEAT_FILE, f"{int(time.time())}\n"
            )
            self._last_heartbeat = now

    def _handle_cancellations(self) -> None:
        for path in sorted((self.spool / protocol.CANCEL_DIR).iterdir()):
            request_id = path.name
            status = protocol.EXIT_TIMEOUT
            try:
                text = path.read_text(encoding="utf-8")
                for line in text.splitlines():
                    key, _, value = line.partition("=")
                    if key == "status" and value.isdigit() and int(value) <= 255:
                        status = int(value)
            except OSError:
                pass
            running = self._running.get(request_id)
            if running is not None:
                self._terminate(running, status, "the bridge gave up waiting")
                self.journal("cancel", id=request_id, status=status)
            path.unlink(missing_ok=True)

    def _reap(self) -> None:
        now = time.monotonic()
        for request_id, running in list(self._running.items()):
            returncode = running.process.poll()
            if returncode is None:
                if running.kill_at is not None and now >= running.kill_at and not running.killed:
                    _signal_group(running.process, signal.SIGKILL)
                    running.killed = True
                continue
            self._finish(request_id, running, returncode=returncode)

    def _finish(self, request_id: str, running: _RunningStep, *, returncode: int | None) -> None:
        """Write the done file for a step and forget it. ``returncode`` is None if it never exited."""

        if running.stop_status is not None:
            status = running.stop_status
        elif returncode is not None:
            status = _exit_status(returncode)
        else:
            status = protocol.EXIT_STOPPED
        running.log.close()
        self._write_done(
            request_id,
            status,
            request=running.request,
            argv=running.argv,
            started=running.started_wall,
            returncode=returncode,
            reason=running.stop_reason,
        )
        (self.spool / protocol.CLAIMED_DIR / f"{request_id}{protocol.REQUEST_SUFFIX}").unlink(
            missing_ok=True
        )
        del self._running[request_id]
        self.journal(
            "finished",
            id=request_id,
            step=running.request.step,
            calcfold=str(running.calcfold),
            status=status,
            returncode=returncode,
            seconds=round(time.time() - running.started_wall, 3),
            reason=running.stop_reason,
        )

    def _start_requests(self) -> None:
        requests_dir = self.spool / protocol.REQUESTS_DIR
        pending: list[tuple[int, str, Path, StepRequest]] = []
        for path in requests_dir.iterdir():
            request_id = protocol.request_id_from_filename(path.name)
            try:
                text = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue  # withdrawn by its bridge while we looked
            except OSError as exc:
                self._reject(path, request_id, f"unreadable request: {exc}")
                continue
            if request_id is None:
                self._reject(path, None, "file name is not '<id>.req'")
                continue
            try:
                request = protocol.parse_request(text, expected_id=request_id)
            except protocol.RequestError as exc:
                self._reject(path, request_id, f"malformed request: {exc}")
                continue
            if request_id in self._claimed_ids or self._done_path(request_id).exists():
                self._reject(path, None, "duplicate request id")
                continue
            calcfold = request.calcfold.resolve()
            if not calcfold.is_relative_to(self._search_root) or not calcfold.is_dir():
                self._reject(
                    path,
                    request_id,
                    f"CalcFold {request.calcfold} is not a directory inside {self._search_root}",
                )
                continue
            pending.append((request.created, request_id, path, request))

        busy = {running.calcfold for running in self._running.values()}
        for _created, request_id, path, request in sorted(pending, key=lambda item: item[:2]):
            if len(self._running) >= self.settings.max_parallel:
                break
            calcfold = request.calcfold.resolve()
            if calcfold in busy:
                continue  # never two steps in one CalcFold; try again next pass
            claimed = self.spool / protocol.CLAIMED_DIR / path.name
            try:
                os.rename(path, claimed)
            except FileNotFoundError:
                continue  # withdrawn by its bridge
            self._claimed_ids.add(request_id)
            if self._start(request, calcfold):
                busy.add(calcfold)

    def _start(self, request: StepRequest, calcfold: Path) -> bool:
        argv = self.settings.argv_for_step(request.step)
        log_path = self.spool / protocol.LOGS_DIR / f"{request.request_id}.log"
        log = open(log_path, "ab")
        started = time.time()
        try:
            process = subprocess.Popen(
                argv,
                cwd=calcfold,
                env=self._step_environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            log.close()
            self._write_done(
                request.request_id,
                EXIT_COMMAND_NOT_STARTED,
                request=request,
                argv=argv,
                started=started,
                reason=f"command could not be started: {exc}",
            )
            (self.spool / protocol.CLAIMED_DIR / f"{request.request_id}{protocol.REQUEST_SUFFIX}").unlink(
                missing_ok=True
            )
            self.journal("start_failed", id=request.request_id, error=str(exc))
            return False
        self._running[request.request_id] = _RunningStep(
            request=request,
            calcfold=calcfold,
            argv=argv,
            process=process,
            log=log,
            started_wall=started,
        )
        self.journal(
            "started_step",
            id=request.request_id,
            step=request.step,
            calcfold=str(calcfold),
            argv=argv,
            pid=process.pid,
        )
        return True

    # -- helpers ---------------------------------------------------------------

    def _terminate(self, running: _RunningStep, status: int, reason: str) -> None:
        if running.stop_status is None:
            running.stop_status = status
            running.stop_reason = reason
        if running.kill_at is None:
            _signal_group(running.process, signal.SIGTERM)
            running.kill_at = time.monotonic() + self.settings.kill_grace_seconds

    def _reject(self, path: Path, request_id: str | None, reason: str) -> None:
        target = self.spool / protocol.REJECTED_DIR / f"{path.name}.{time.time_ns()}"
        try:
            os.rename(path, target)
        except FileNotFoundError:
            return
        if request_id is not None and not self._done_path(request_id).exists():
            self._write_done(request_id, protocol.EXIT_REJECTED, reason=reason)
        self.journal("rejected", file=path.name, reason=reason)

    def _done_path(self, request_id: str) -> Path:
        return self.spool / protocol.DONE_DIR / f"{request_id}{protocol.DONE_SUFFIX}"

    def _write_done(
        self,
        request_id: str,
        status: int,
        *,
        request: StepRequest | None = None,
        argv: Sequence[str] | None = None,
        started: float | None = None,
        returncode: int | None = None,
        reason: str = "",
    ) -> None:
        lines = [f"status={status}", f"id={request_id}"]
        if request is not None:
            lines += [f"step={request.step}", f"calcfold={request.calcfold}"]
        if argv is not None:
            lines.append(f"command={shlex.join(argv)}")
        if returncode is not None:
            lines.append(f"returncode={returncode}")
        if started is not None:
            lines += [f"started={started:.3f}", f"finished={time.time():.3f}"]
        if reason:
            lines.append("reason=" + reason.replace("\n", " "))
        protocol.write_atomically(self.spool, self._done_path(request_id), "\n".join(lines) + "\n")

    def journal(self, event: str, **fields) -> None:
        """Append one event to ``<spool>/launcher.jsonl`` (one JSON object per line)."""

        record = {"time": round(time.time(), 3), "event": event, **fields}
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(
            self.spool / protocol.JOURNAL_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644
        )
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)

    def _remove_launcher_temporaries(self) -> None:
        for path in (self.spool / protocol.TMP_DIR).glob(".launcher-*"):
            path.unlink(missing_ok=True)


def _exit_status(returncode: int) -> int:
    """Map a ``Popen`` return code to a shell-style exit status (signal n -> 128 + n)."""

    if returncode < 0:
        return min(128 - returncode, 255)
    return min(returncode, 255)


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


# -- master and launcher together ----------------------------------------------


def install_stop_signal_handlers(event: threading.Event) -> dict[int, object]:
    """Set ``event`` on SIGTERM and SIGINT. Returns the previous handlers.

    Must be called from the main thread. SLURM sends SIGTERM to the job's
    processes when the job ends or is cancelled.
    """

    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: event.set())
    return previous


def run_search(
    master_argv: Sequence[str],
    master_env: Mapping[str, str],
    master_cwd: Path,
    settings: LauncherSettings,
    *,
    master_log: Path,
    stop_event: threading.Event | None = None,
) -> int:
    """Start the USPEX master and run the launch loop until the master exits.

    The spool is prepared before the master starts. If ``stop_event`` is set
    first (for example by ``install_stop_signal_handlers`` on SIGTERM), the
    master is stopped first (SIGTERM, then SIGKILL after the grace period) so
    it does not record the stopped steps as finished relaxations, then the
    launcher. Returns the master's exit status, or 123 if it was stopped.

    ``master_argv``/``master_env``/``master_cwd`` normally come from
    ``backend.structure_search.bridge.master.build_master_command``.
    """

    event = stop_event or threading.Event()
    launcher = Launcher(settings)
    launcher.prepare()
    with open(master_log, "ab") as log:
        master = subprocess.Popen(
            list(master_argv),
            cwd=master_cwd,
            env=dict(master_env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launcher.journal("master_started", pid=master.pid, argv=list(master_argv))
    stopped = False
    try:
        while master.poll() is None:
            if event.is_set():
                stopped = True
                break
            launcher.poll_once()
            event.wait(settings.poll_seconds)
    finally:
        if master.poll() is None:
            stopped = True
            _signal_group(master, signal.SIGTERM)
            try:
                master.wait(timeout=settings.kill_grace_seconds)
            except subprocess.TimeoutExpired:
                _signal_group(master, signal.SIGKILL)
                master.wait()
        launcher.journal("master_finished", returncode=master.returncode, stopped=stopped)
        launcher.shutdown()
    return protocol.EXIT_STOPPED if stopped else _exit_status(master.returncode)


# -- command line (manual pilots) ----------------------------------------------


def _parse_ranks_per_step(values: Sequence[str]) -> dict[int, int]:
    result = {}
    for value in values:
        step, separator, ranks = value.partition("=")
        if not separator or not step.isdigit() or not ranks.isdigit():
            raise argparse.ArgumentTypeError(f"expected STEP=RANKS, got {value!r}")
        result[int(step)] = int(ranks)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the launch loop alone until SIGTERM or SIGINT.

    For manual pilots on POWER (#59). The VASP steps get this process's
    environment (load the modules first); the application passes an explicit
    environment through ``LauncherSettings`` instead.
    """

    parser = argparse.ArgumentParser(
        prog="python -m backend.structure_search.bridge.launcher",
        description="Run USPEX bridge requests as host job steps.",
    )
    parser.add_argument("--spool", required=True, type=Path)
    parser.add_argument("--search-dir", required=True, type=Path)
    parser.add_argument("--max-parallel", required=True, type=int)
    parser.add_argument("--ranks", required=True, type=int)
    memory = parser.add_mutually_exclusive_group(required=True)
    memory.add_argument("--mem", help="memory per step, for example 200G")
    memory.add_argument("--no-mem", action="store_true", help="leave --mem out of the step command")
    parser.add_argument("--ranks-per-step", nargs="*", default=[], metavar="STEP=RANKS")
    parser.add_argument("--command-template", default=DEFAULT_COMMAND_TEMPLATE)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=10.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    settings = LauncherSettings(
        spool_dir=args.spool.absolute(),
        search_dir=args.search_dir.absolute(),
        max_parallel=args.max_parallel,
        ranks=args.ranks,
        mem=None if args.no_mem else args.mem,
        environment=dict(os.environ),
        ranks_per_step=_parse_ranks_per_step(args.ranks_per_step),
        command_template=args.command_template,
        poll_seconds=args.poll_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        kill_grace_seconds=args.kill_grace_seconds,
    )
    event = threading.Event()
    install_stop_signal_handlers(event)
    Launcher(settings).run(event)
    return 0


__all__ = [
    "DEFAULT_COMMAND_TEMPLATE",
    "EXIT_COMMAND_NOT_STARTED",
    "FORBIDDEN_STEP_ENVIRONMENT",
    "Launcher",
    "LauncherSettings",
    "MPI_ENVIRONMENT_OVERRIDES",
    "build_step_argv",
    "build_step_environment",
    "install_stop_signal_handlers",
    "main",
    "run_search",
    "slurm_job_environment",
]


if __name__ == "__main__":
    raise SystemExit(main())
