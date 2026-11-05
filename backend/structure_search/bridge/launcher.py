"""Host-side launch loop for USPEX relaxation steps.

Design: docs/structure_search.md, section 4; issue #66.

USPEX runs inside its Singularity image and cannot call ``srun``. For each
relaxation step it runs ``bmd_bridge.sh``, which drops a request file into a
spool directory and waits. This module runs on the host, in the same SLURM
allocation, and runs VASP for each request:

* watch ``<spool>/requests`` for new requests
* run each one in its CalcFold, at most ``max_parallel`` at a time and never
  two in the same CalcFold at once, in one of two launch modes (see
  ``LauncherSettings``): ``srun_steps`` (``srun --exact --mpi=pmi2 -n {ranks}
  --mem={mem} vasp_std``, SLURM places the cores) or ``mpirun_pinned``
  (``mpirun -n {ranks} vasp_std`` with a fixed core range per slot)
* before each run, append the exact command, the variables the launcher set
  and the slot index to ``<CalcFold>/bmd_launch.jsonl``, so core placement can
  be checked afterwards against the ``I_MPI_DEBUG=4`` report in the step log
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


SRUN_STEPS = "srun_steps"
MPIRUN_PINNED = "mpirun_pinned"
LAUNCH_MODES = (SRUN_STEPS, MPIRUN_PINNED)

SRUN_COMMAND_TEMPLATE = "srun --exact --mpi=pmi2 -n {ranks} --mem={mem} vasp_std"
MPIRUN_COMMAND_TEMPLATE = "mpirun -n {ranks} vasp_std"
DEFAULT_COMMAND_TEMPLATE = SRUN_COMMAND_TEMPLATE
DEFAULT_COMMAND_TEMPLATES = {
    SRUN_STEPS: SRUN_COMMAND_TEMPLATE,
    MPIRUN_PINNED: MPIRUN_COMMAND_TEMPLATE,
}
"""Default step command per launch mode. ``{ranks}``, ``{mem}`` and ``{step}`` are filled in.

If ``mem`` is ``None`` every argument that contains ``{mem}`` is left out,
so write the memory option as one argument (``--mem={mem}``), not two.
"""

# leeburton-pool compute nodes, from the lab's VASP notes (August 2026):
# 2 sockets x 96 cores, no hyperthreading. compute-0-332/333 have a different
# NUMA layout (docs/structure_search.md §5.5) and need their own values.
DEFAULT_SOCKETS = 2
DEFAULT_CORES_PER_SOCKET = 96
DEFAULT_MEMORY_MARGIN_FRACTION = 0.10

MPI_ENVIRONMENT_OVERRIDES = {"I_MPI_DEBUG": "4"}
FORBIDDEN_STEP_ENVIRONMENT = ("I_MPI_PIN_DOMAIN",)
EXIT_COMMAND_NOT_STARTED = 127
LAUNCH_RECORD_FILE = "bmd_launch.jsonl"

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


def default_step_memory_mb(node_memory_mb: int, max_parallel: int, margin_fraction: float) -> int:
    """Memory per ``srun`` step: node memory less a margin, shared by the parallel steps."""

    if node_memory_mb < 1 or max_parallel < 1 or not 0 <= margin_fraction < 1:
        raise ValueError("need node_memory_mb >= 1, max_parallel >= 1 and 0 <= margin_fraction < 1")
    per_step = int(node_memory_mb * (1 - margin_fraction)) // max_parallel
    if per_step < 1:
        raise ValueError("node memory is too small for this many parallel steps")
    return per_step


def pinned_slot_first_cores(
    *, slots: int, ranks: int, sockets: int, cores_per_socket: int
) -> tuple[int, ...]:
    """Return the first core of each slot's contiguous core range, balanced across sockets.

    Slots go round-robin over the sockets, packed from the start of each
    socket. For 4 slots x 48 ranks on 2 x 96 cores: slot 0 -> 0-47, slot 1 ->
    96-143, slot 2 -> 48-95, slot 3 -> 144-191. A slot larger than a socket
    must be a whole number of sockets and gets whole sockets. Raises
    ``ValueError`` if the slots do not fit on the node.

    The numbers are OS CPU numbers, assuming socket ``s`` holds CPUs
    ``s * cores_per_socket`` to ``(s + 1) * cores_per_socket - 1``. That is
    not yet checked on POWER: compare with ``lscpu`` and the ``I_MPI_DEBUG=4``
    pinning report in the bridge pilot.
    """

    if min(slots, ranks, sockets, cores_per_socket) < 1:
        raise ValueError("slots, ranks, sockets and cores_per_socket must all be at least 1")
    if ranks <= cores_per_socket:
        per_socket = -(-slots // sockets)
        if per_socket * ranks > cores_per_socket:
            raise ValueError(
                f"{slots} slots x {ranks} ranks do not fit: {per_socket} slots per socket "
                f"need {per_socket * ranks} of {cores_per_socket} cores"
            )
        return tuple(
            (slot % sockets) * cores_per_socket + (slot // sockets) * ranks for slot in range(slots)
        )
    if ranks % cores_per_socket:
        raise ValueError(
            f"{ranks} ranks is more than one socket ({cores_per_socket} cores) "
            "but not a whole number of sockets"
        )
    sockets_per_slot = ranks // cores_per_socket
    if slots * sockets_per_slot > sockets:
        raise ValueError(f"{slots} slots x {ranks} ranks need more than {sockets} sockets")
    return tuple(slot * sockets_per_slot * cores_per_socket for slot in range(slots))


def cpu_range(first: int, count: int) -> str:
    """Intel MPI processor list for ``count`` cores from ``first`` (``"0-47"`` or ``"5"``)."""

    return str(first) if count == 1 else f"{first}-{first + count - 1}"


@dataclass(frozen=True)
class LauncherSettings:
    """Configuration of the launch loop.

    ``launch_mode`` chooses how a step starts VASP. It is a single value so
    the default can change with further cluster tests (#59):

    ``"srun_steps"``
        ``srun --exact --mpi=pmi2 -n {ranks} --mem={mem} vasp_std``; SLURM
        places the cores. ``--mem`` is required: in the step test on POWER
        (#59, 2 Nov 2026) concurrent ``srun --exact`` steps without a
        per-step ``--mem`` ran one after another, and with one they ran
        together on separate cores. If ``mem`` is not given it defaults to
        the node memory less ``memory_margin_fraction``, divided by
        ``max_parallel``. Node memory is ``node_memory_mb`` or, if that is
        not given, ``SLURM_MEM_PER_NODE`` from ``environment`` (under
        ``--mem=0`` SLURM reports the node's memory there; not yet checked
        on POWER, so prefer passing ``node_memory_mb``).
        Confirmed on POWER (#59, 4 Nov 2026): ``srun --exact --mpi=pmi2 -n 4
        --mem=8G vasp_std`` inside a shared allocation started 4 VASP ranks
        on separate cores with the same energy as ``mpirun``. That run had
        ``I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so`` in its environment; pass
        it in ``environment`` (it is a cluster setting, so it is not set
        here). Placement of 48-rank steps on a full node is not yet tested.

    ``"mpirun_pinned"``
        ``mpirun -n {ranks} vasp_std``, the launch path of the lab's VASP
        benchmarks. Each of the ``max_parallel`` slots gets a fixed core
        range (``pinned_slot_first_cores``) passed as
        ``I_MPI_PIN_PROCESSOR_LIST``, with ``I_MPI_HYDRA_BOOTSTRAP=fork`` so
        Intel MPI starts its ranks directly rather than through SLURM on a
        single node. ``mpirun -n 4 vasp_std`` ran correctly inside a shared
        allocation on POWER (#59), but the pinned core lists and
        ``I_MPI_HYDRA_BOOTSTRAP=fork`` are not yet tested. Steps given fewer ranks
        by ``ranks_per_step`` use the first cores of their slot; more ranks
        than ``ranks`` are rejected.

    ``environment`` is the complete environment of the VASP steps (modules
    already loaded); nothing is inherited from the launcher's own process.
    For ``srun_steps`` it must contain the job's ``SLURM_*`` variables; see
    ``slurm_job_environment``.
    """

    spool_dir: Path
    search_dir: Path
    max_parallel: int
    ranks: int
    environment: Mapping[str, str]
    launch_mode: str = SRUN_STEPS
    mem: str | None = None
    node_memory_mb: int | None = None
    memory_margin_fraction: float = DEFAULT_MEMORY_MARGIN_FRACTION
    sockets: int = DEFAULT_SOCKETS
    cores_per_socket: int = DEFAULT_CORES_PER_SOCKET
    ranks_per_step: Mapping[int, int] = field(default_factory=dict)
    command_template: str | None = None
    poll_seconds: float = 1.0
    heartbeat_interval_seconds: float = 10.0
    kill_grace_seconds: float = 10.0
    step_mem: str | None = field(init=False)
    slot_first_cores: tuple[int, ...] | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spool_dir", Path(self.spool_dir))
        object.__setattr__(self, "search_dir", Path(self.search_dir))
        object.__setattr__(self, "ranks_per_step", dict(self.ranks_per_step))
        object.__setattr__(self, "environment", dict(self.environment))
        for name in ("spool_dir", "search_dir"):
            if not getattr(self, name).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if self.launch_mode not in LAUNCH_MODES:
            raise ValueError(f"launch_mode must be one of {LAUNCH_MODES}, got {self.launch_mode!r}")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        if self.ranks < 1:
            raise ValueError("ranks must be at least 1")
        for step, ranks in self.ranks_per_step.items():
            if not isinstance(step, int) or step < 1 or not isinstance(ranks, int) or ranks < 1:
                raise ValueError("ranks_per_step maps step numbers (from 1) to rank counts (from 1)")
        if self.mem is not None and not _MEMORY_PATTERN.match(self.mem):
            raise ValueError(f"mem must look like '200G' or '4000M' (not zero), got {self.mem!r}")
        for seconds in ("poll_seconds", "heartbeat_interval_seconds", "kill_grace_seconds"):
            value = getattr(self, seconds)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{seconds} must be a positive number")
        if self.command_template is None:
            object.__setattr__(self, "command_template", DEFAULT_COMMAND_TEMPLATES[self.launch_mode])

        step_mem = self.mem
        slot_first_cores = None
        if self.launch_mode == SRUN_STEPS:
            if "{mem}" not in self.command_template:
                raise ValueError(
                    "srun_steps needs '--mem={mem}' in the command template: without a per-step "
                    "--mem, concurrent srun steps run one after another (#59)"
                )
            if step_mem is None:
                step_mem = f"{self._default_step_memory_mb()}M"
        else:
            if any(ranks > self.ranks for ranks in self.ranks_per_step.values()):
                raise ValueError("mpirun_pinned: ranks_per_step may not exceed ranks (the slot size)")
            slot_first_cores = pinned_slot_first_cores(
                slots=self.max_parallel,
                ranks=self.ranks,
                sockets=self.sockets,
                cores_per_socket=self.cores_per_socket,
            )
        object.__setattr__(self, "step_mem", step_mem)
        object.__setattr__(self, "slot_first_cores", slot_first_cores)
        build_step_argv(self.command_template, ranks=self.ranks, mem=self.step_mem, step=1)

    def _default_step_memory_mb(self) -> int:
        node_memory_mb = self.node_memory_mb
        if node_memory_mb is None:
            reported = self.environment.get("SLURM_MEM_PER_NODE", "")
            if not reported.isdigit() or int(reported) < 1:
                raise ValueError(
                    "srun_steps needs a per-step --mem: pass mem, or node_memory_mb, or an "
                    "environment with SLURM_MEM_PER_NODE (concurrent steps without --mem run "
                    "one after another, #59)"
                )
            node_memory_mb = int(reported)
        return default_step_memory_mb(node_memory_mb, self.max_parallel, self.memory_margin_fraction)

    def ranks_for_step(self, step: int) -> int:
        return self.ranks_per_step.get(step, self.ranks)

    def argv_for_step(self, step: int) -> list[str]:
        return build_step_argv(
            self.command_template,
            ranks=self.ranks_for_step(step),
            mem=self.step_mem,
            step=step,
        )

    def environment_for_slot(self, step: int, slot: int) -> dict[str, str]:
        """Variables set for this step on top of the step environment."""

        if self.launch_mode != MPIRUN_PINNED:
            return {}
        assert self.slot_first_cores is not None
        return {
            "I_MPI_PIN_PROCESSOR_LIST": cpu_range(self.slot_first_cores[slot], self.ranks_for_step(step)),
            "I_MPI_HYDRA_BOOTSTRAP": "fork",
        }


@dataclass
class _RunningStep:
    request: StepRequest
    calcfold: Path
    argv: list[str]
    slot: int
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
            launch_mode=self.settings.launch_mode,
            mem=self.settings.step_mem,
            slot_cores=(
                None
                if self.settings.slot_first_cores is None
                else [cpu_range(first, self.settings.ranks) for first in self.settings.slot_first_cores]
            ),
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
            slot=running.slot,
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

    def _free_slot(self) -> int:
        used = {running.slot for running in self._running.values()}
        return min(slot for slot in range(self.settings.max_parallel) if slot not in used)

    def _start(self, request: StepRequest, calcfold: Path) -> bool:
        slot = self._free_slot()
        argv = self.settings.argv_for_step(request.step)
        slot_environment = self.settings.environment_for_slot(request.step, slot)
        environment = {**self._step_environment, **slot_environment}
        log_path = self.spool / protocol.LOGS_DIR / f"{request.request_id}.log"
        started = time.time()
        launch_record = {
            "time": round(started, 3),
            "id": request.request_id,
            "step": request.step,
            "slot": slot,
            "launch_mode": self.settings.launch_mode,
            "argv": argv,
            "cwd": str(calcfold),
            "environment_set": {**MPI_ENVIRONMENT_OVERRIDES, **slot_environment},
            "environment_removed": self._removed_environment,
            "output_log": str(log_path),
        }
        try:
            _append_line(calcfold / LAUNCH_RECORD_FILE, json.dumps(launch_record, sort_keys=True))
        except OSError as exc:
            self.journal("launch_record_failed", id=request.request_id, error=str(exc))
        log = open(log_path, "ab")
        try:
            process = subprocess.Popen(
                argv,
                cwd=calcfold,
                env=environment,
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
            slot=slot,
            process=process,
            log=log,
            started_wall=started,
        )
        self.journal(
            "started_step",
            id=request.request_id,
            step=request.step,
            slot=slot,
            calcfold=str(calcfold),
            argv=argv,
            environment_set=launch_record["environment_set"],
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
        _append_line(self.spool / protocol.JOURNAL_FILE, json.dumps(record, sort_keys=True))

    def _remove_launcher_temporaries(self) -> None:
        for path in (self.spool / protocol.TMP_DIR).glob(".launcher-*"):
            path.unlink(missing_ok=True)


def _append_line(path: Path, line: str) -> None:
    """Append one line with a single ``write`` in append mode, so lines never interleave."""

    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, (line + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


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
    parser.add_argument("--launch-mode", choices=LAUNCH_MODES, default=SRUN_STEPS)
    parser.add_argument("--mem", help="memory per srun step, for example 200G (default: from node memory)")
    parser.add_argument("--node-memory-mb", type=int, help="default: SLURM_MEM_PER_NODE")
    parser.add_argument(
        "--memory-margin-fraction", type=float, default=DEFAULT_MEMORY_MARGIN_FRACTION
    )
    parser.add_argument("--sockets", type=int, default=DEFAULT_SOCKETS)
    parser.add_argument("--cores-per-socket", type=int, default=DEFAULT_CORES_PER_SOCKET)
    parser.add_argument("--ranks-per-step", nargs="*", default=[], metavar="STEP=RANKS")
    parser.add_argument("--command-template", help="default: depends on --launch-mode")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=10.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    settings = LauncherSettings(
        spool_dir=args.spool.absolute(),
        search_dir=args.search_dir.absolute(),
        max_parallel=args.max_parallel,
        ranks=args.ranks,
        environment=dict(os.environ),
        launch_mode=args.launch_mode,
        mem=args.mem,
        node_memory_mb=args.node_memory_mb,
        memory_margin_fraction=args.memory_margin_fraction,
        sockets=args.sockets,
        cores_per_socket=args.cores_per_socket,
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
    "DEFAULT_COMMAND_TEMPLATES",
    "DEFAULT_CORES_PER_SOCKET",
    "DEFAULT_MEMORY_MARGIN_FRACTION",
    "DEFAULT_SOCKETS",
    "LAUNCH_MODES",
    "LAUNCH_RECORD_FILE",
    "MPIRUN_COMMAND_TEMPLATE",
    "MPIRUN_PINNED",
    "SRUN_COMMAND_TEMPLATE",
    "SRUN_STEPS",
    "cpu_range",
    "default_step_memory_mb",
    "pinned_slot_first_cores",
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
