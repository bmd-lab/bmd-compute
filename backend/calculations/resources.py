from __future__ import annotations

from dataclasses import dataclass

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resource_policy import (
    AUTOMATIC_NCORE_STAGE_TYPES,
    stage_allows_automatic_ncore,
)
from backend.config import DEFAULT_ACCOUNT, DEFAULT_PARTITION, DEFAULT_RESOURCES


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
        object.__setattr__(self, "walltime", _nonempty_text(self.walltime, "walltime"))
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
    "ExecutionResources",
    "default_execution_resources",
    "ncore_for_execution_resources",
    "normalize_execution_resources",
    "stage_allows_automatic_ncore",
    "validate_queue",
]
