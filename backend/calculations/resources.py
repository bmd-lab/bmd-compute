from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resource_policy import (
    AUTOMATIC_NCORE_STAGE_TYPES,
    stage_allows_automatic_ncore,
)
from backend.config import (
    DEFAULT_ACCOUNT,
    DEFAULT_PARTITION,
    DEFAULT_QOS,
    DEFAULT_RESOURCES,
    STRUCTURE_SEARCH_CORES_PER_SOCKET,
    STRUCTURE_SEARCH_EXCLUDED_NODES,
    STRUCTURE_SEARCH_NODE_SOCKETS,
    STRUCTURE_SEARCH_PARALLEL_RELAXATIONS,
    STRUCTURE_SEARCH_RANKS_PER_RELAXATION,
)


ALLOWED_CPU_COUNTS = (24, 48, 72, 96, 120, 144, 168, 192)
ALLOWED_MEMORY_GB = (32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512)
ALLOWED_QUEUES = (DEFAULT_PARTITION,)
# Automatic NCORE for eligible stages (backend.calculations.resource_policy).
# Reduced from 8 to 4 as a reliability correction: a 16-atom Si HSE06 static
# (VASP 6.4.1, 24 ranks, ENCUT 620, PRECFOCK Accurate) aborted at
# initialisation with NCORE = 8 (SET_INDPW_FULL), but initialised and ran with
# NCORE = 4 (fastest, lowest memory of the working settings) and with NCORE
# unset (about 2.3x the memory). NCORE = 4 divides every ALLOWED_CPU_COUNTS
# value. This is one data point, not a performance optimum for every system.
DEFAULT_NCORE = 4
# SLURM walltime as H:MM:SS (hours may exceed 24). The upper limit is the
# 100-day maximum of the owner100d QOS that BMD Compute submits under; SLURM
# would reject longer requests at submission.
WALLTIME_PATTERN = re.compile(r"^([0-9]{1,4}):([0-5][0-9]):([0-5][0-9])$")
MAX_WALLTIME_SECONDS = 100 * 24 * 3600


@dataclass(frozen=True)
class ExecutionResources:
    nodes: int = DEFAULT_RESOURCES["nodes"]
    cpus: int = DEFAULT_RESOURCES["ntasks"]
    memory_gb: int = DEFAULT_RESOURCES["mem_gb"]
    walltime: str = DEFAULT_RESOURCES["walltime"]
    queue: str = DEFAULT_PARTITION
    account: str = DEFAULT_ACCOUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _positive_int(self.nodes, "nodes"))
        object.__setattr__(self, "cpus", _allowed_cpu_count(self.cpus))
        object.__setattr__(self, "memory_gb", _allowed_memory_gb(self.memory_gb))
        object.__setattr__(self, "walltime", validate_walltime(self.walltime))
        object.__setattr__(self, "queue", validate_queue(self.queue))
        object.__setattr__(self, "account", _nonempty_text(self.account, "account"))

    @property
    def ntasks(self) -> int:
        return self.cpus

    @property
    def mem_gb(self) -> int:
        return self.memory_gb

    @property
    def partition(self) -> str:
        return self.queue

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "mem_gb": self.mem_gb,
            "walltime": self.walltime,
            "partition": self.partition,
            "account": self.account,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExecutionResources":
        values = dict(data or {})
        return cls(
            nodes=values.get("nodes", DEFAULT_RESOURCES["nodes"]),
            cpus=values.get("cpus", values.get("ntasks", DEFAULT_RESOURCES["ntasks"])),
            memory_gb=values.get(
                "memory_gb",
                values.get("mem_gb", DEFAULT_RESOURCES["mem_gb"]),
            ),
            walltime=values.get("walltime", DEFAULT_RESOURCES["walltime"]),
            queue=values.get("queue", values.get("partition", DEFAULT_PARTITION)),
            account=values.get("account", DEFAULT_ACCOUNT),
        )


# SLURM hostlist expression for --exclude, e.g. "compute-0-[332-333],compute-0-7".
# Restricted to characters that are safe on an #SBATCH line.
_HOSTLIST_PATTERN = re.compile(r"^[A-Za-z0-9._\-]+(\[[0-9,\-]+\])?(,[A-Za-z0-9._\-]+(\[[0-9,\-]+\])?)*$")
STRUCTURE_SEARCH_RESOURCE_PRESET = "structure_search_exclusive_node"


@dataclass(frozen=True)
class StructureSearchResources:
    """SLURM request for one USPEX structure search: one whole node.

    Used **only** for structure searches (docs/structure_search.md §4). Ordinary
    workflows keep using ``ExecutionResources`` and the automatic estimator,
    and this class is deliberately not accepted where ``ExecutionResources`` is.

    The request is ``-N 1 --exclusive --mem=0`` under the ``owner100d`` QOS:

    * ``--exclusive``: the USPEX master runs several VASP relaxations side by
      side on the node, so no other job may share it.
    * ``--mem=0``: all of the node's memory. The partition's default is 1 GB
      per job, far too little.
    * ``ntasks``: relaxations in parallel x ranks per relaxation (4 x 48 = 192
      by default), so that the job's ``srun`` steps can start that many ranks.
    * ``exclude``: nodes whose core layout differs from the configured
      sockets x cores per socket (compute-0-332/333 by default).

    The walltime must be given explicitly (H:MM:SS, at most 100 days, the
    owner100d limit). There is no automatic estimate: the estimator is
    calibrated on single PBE stages, and a search's length depends on how many
    generations it runs before it stops.
    """

    walltime: str
    ntasks: int = STRUCTURE_SEARCH_PARALLEL_RELAXATIONS * STRUCTURE_SEARCH_RANKS_PER_RELAXATION
    exclude: str = STRUCTURE_SEARCH_EXCLUDED_NODES
    queue: str = DEFAULT_PARTITION
    node_cores: int = field(
        default=STRUCTURE_SEARCH_NODE_SOCKETS * STRUCTURE_SEARCH_CORES_PER_SOCKET,
        repr=False,
    )

    nodes = 1
    exclusive = True
    memory = "0"
    account = DEFAULT_ACCOUNT
    qos = DEFAULT_QOS

    def __post_init__(self) -> None:
        if self.walltime is None or str(self.walltime).strip().lower() in {"", "auto", "automatic"}:
            raise CalculationValidationError(
                "A structure search needs an explicit walltime.",
                suggestion=(
                    "Enter the walltime as H:MM:SS, at most 2400:00:00 (100 days). "
                    "Structure searches have no automatic walltime estimate."
                ),
            )
        object.__setattr__(self, "walltime", validate_walltime(self.walltime))
        node_cores = _positive_int(self.node_cores, "node cores")
        ntasks = _positive_int(self.ntasks, "tasks")
        if ntasks > node_cores:
            raise CalculationValidationError(
                f"A structure search asks for {ntasks} tasks, but a node has {node_cores} cores.",
                suggestion="Lower the relaxations in parallel or the ranks per relaxation.",
            )
        object.__setattr__(self, "node_cores", node_cores)
        object.__setattr__(self, "ntasks", ntasks)
        exclude = str(self.exclude or "").strip()
        if exclude and not _HOSTLIST_PATTERN.fullmatch(exclude):
            raise CalculationValidationError(
                f"Excluded nodes {exclude!r} is not a SLURM host list.",
                suggestion="Use a form such as compute-0-[332-333],compute-0-7, or leave it empty.",
            )
        object.__setattr__(self, "exclude", exclude)
        object.__setattr__(self, "queue", validate_queue(self.queue))

    def to_dict(self) -> dict:
        return {
            "preset": STRUCTURE_SEARCH_RESOURCE_PRESET,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "exclusive": self.exclusive,
            "mem": self.memory,
            "walltime": self.walltime,
            "partition": self.queue,
            "account": self.account,
            "qos": self.qos,
            "exclude": self.exclude,
        }

    def sbatch_directives(self) -> list[str]:
        """The ``#SBATCH`` lines for this request, in a fixed order."""

        lines = [
            f"#SBATCH -p {self.queue}",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --qos={self.qos}",
            f"#SBATCH --time={self.walltime}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks={self.ntasks}",
            "#SBATCH --exclusive",
            f"#SBATCH --mem={self.memory}",
        ]
        if self.exclude:
            lines.append(f"#SBATCH --exclude={self.exclude}")
        return lines


def structure_search_resources(walltime) -> StructureSearchResources:
    """The exclusive-node request for a structure search with ``walltime``."""

    return StructureSearchResources(walltime=walltime)


def default_execution_resources() -> ExecutionResources:
    return ExecutionResources()


def normalize_execution_resources(resources=None) -> ExecutionResources:
    if isinstance(resources, ExecutionResources):
        return resources

    if isinstance(resources, dict):
        return ExecutionResources.from_dict(resources)

    if resources is None:
        return default_execution_resources()

    raise CalculationValidationError(
        "Execution resources could not be interpreted.",
        suggestion="Update the calculation using valid CPU, memory, walltime, and queue values.",
    )


def ncore_for_execution_resources(resources=None) -> int:
    normalize_execution_resources(resources)
    return DEFAULT_NCORE


def walltime_seconds(value) -> int:
    match = WALLTIME_PATTERN.fullmatch(str(value or "").strip())
    if match is None:
        raise ValueError("walltime must be H:MM:SS")
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def format_walltime(seconds: int) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_walltime(value) -> str:
    text = _nonempty_text(value, "walltime")
    try:
        total = walltime_seconds(text)
    except ValueError as exc:
        raise CalculationValidationError(
            "Walltime must be written as hours:minutes:seconds, for example 12:00:00.",
            suggestion="Enter the walltime as H:MM:SS (hours may exceed 24), or leave it on Automatic.",
        ) from exc
    if total <= 0 or total > MAX_WALLTIME_SECONDS:
        raise CalculationValidationError(
            "Walltime must be greater than zero and at most 2400:00:00 (100 days).",
            suggestion="100 days is the limit of the owner100d QOS. Enter a shorter walltime, or leave it on Automatic.",
        )
    return text


def validate_queue(value) -> str:
    queue = _nonempty_text(value, "queue")
    if queue not in ALLOWED_QUEUES:
        allowed = ", ".join(ALLOWED_QUEUES)
        raise CalculationValidationError(
            f"Queue must be one of: {allowed}.",
            suggestion="Choose one of the listed queues and update the calculation again.",
        )

    return queue



def _positive_int(value, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CalculationValidationError(
            f"{label.capitalize()} must be a positive integer.",
            suggestion="Enter a whole-number resource value and update the calculation again.",
        ) from exc

    if number < 1:
        raise CalculationValidationError(
            f"{label.capitalize()} must be a positive integer.",
            suggestion="Enter a whole-number resource value greater than zero.",
        )

    return number


def _allowed_cpu_count(value) -> int:
    cpus = _positive_int(value, "CPUs")
    if cpus not in ALLOWED_CPU_COUNTS:
        allowed = ", ".join(str(count) for count in ALLOWED_CPU_COUNTS)
        raise CalculationValidationError(
            f"CPUs must be one of: {allowed}.",
            suggestion="Choose one of the listed CPU counts and update the calculation again.",
        )

    return cpus


def _allowed_memory_gb(value) -> int:
    memory_gb = _positive_int(value, "memory")
    if memory_gb not in ALLOWED_MEMORY_GB:
        allowed = ", ".join(str(amount) for amount in ALLOWED_MEMORY_GB)
        raise CalculationValidationError(
            f"Memory must be one of: {allowed} GB.",
            suggestion="Choose one of the listed memory amounts and update the calculation again.",
        )

    return memory_gb


def _nonempty_text(value, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CalculationValidationError(
            f"{label.capitalize()} must not be empty.",
            suggestion="Enter the execution setting and update the calculation again.",
        )

    return text


__all__ = [
    "ALLOWED_CPU_COUNTS",
    "ALLOWED_MEMORY_GB",
    "ALLOWED_QUEUES",
    "AUTOMATIC_NCORE_STAGE_TYPES",
    "DEFAULT_NCORE",
    "MAX_WALLTIME_SECONDS",
    "WALLTIME_PATTERN",
    "ExecutionResources",
    "STRUCTURE_SEARCH_RESOURCE_PRESET",
    "StructureSearchResources",
    "default_execution_resources",
    "ncore_for_execution_resources",
    "normalize_execution_resources",
    "stage_allows_automatic_ncore",
    "structure_search_resources",
    "format_walltime",
    "validate_queue",
    "validate_walltime",
    "walltime_seconds",
]
