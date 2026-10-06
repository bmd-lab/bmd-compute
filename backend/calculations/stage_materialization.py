"""How each workflow stage type materializes into concrete calculations.

A workflow is an ordered sequence of stages, and a stage describes scientific
methodology (``StageSpec``). This module records, per stage type, how that
methodology becomes concrete calculations:

* ``SINGLE`` - the stage is exactly one calculation. Every existing stage type
  is ``SINGLE``, and nothing about those stages changes: they keep the implicit
  one stage -> one calculation model and never get a task-set record.
* ``DERIVED_TASK_SET`` - the stage is one methodology applied to a set of task
  instances that is derived deterministically at runtime from an upstream
  result (for phonons: the M1 displacement plan of the relaxed structure). The
  tasks are never workflow stages; the authoritative set is a separate
  ``bmd_compute.stage_task_set`` record (``backend.stage_task_set``).

This is materialization, not scheduling: nothing here says whether tasks run
sequentially, concurrently or as a SLURM array, and the canonical task order
of a task set is not an execution order.

Materialization says nothing about executability either. Stage types listed in
``NON_EXECUTABLE_STAGE_TYPES`` are representation only and are refused by the
calculation registry before any input generation, preparation or submission.
"""

from __future__ import annotations

from enum import Enum

from backend.calculations.models import StageType


class StageMaterialization(str, Enum):
    SINGLE = "single"
    DERIVED_TASK_SET = "derived_task_set"


# Identifies the producer of a derived task set: what the tasks are derived
# from and how their identities and canonical order are defined.
PHONON_DISPLACEMENT_MATERIALIZER = "bmd_compute.phonon_displacement_plan"

_STAGE_MATERIALIZATION: dict[StageType, StageMaterialization] = {
    StageType.RELAX: StageMaterialization.SINGLE,
    StageType.STATIC: StageMaterialization.SINGLE,
    StageType.DOS: StageMaterialization.SINGLE,
    StageType.BAND_STRUCTURE: StageMaterialization.SINGLE,
    StageType.PHONON_FORCES: StageMaterialization.DERIVED_TASK_SET,
}

_DERIVED_TASK_SET_MATERIALIZERS: dict[StageType, str] = {
    StageType.PHONON_FORCES: PHONON_DISPLACEMENT_MATERIALIZER,
}

# Stage types that exist only as representation. No route may build, prepare
# or submit a workflow containing them.
NON_EXECUTABLE_STAGE_TYPES = frozenset({StageType.PHONON_FORCES})


def stage_materialization(stage_type: StageType | str) -> StageMaterialization:
    normalized = StageType.from_value(stage_type)
    try:
        return _STAGE_MATERIALIZATION[normalized]
    except KeyError:
        raise ValueError(
            f"No materialization is defined for stage type {normalized.value!r}."
        ) from None


def derived_task_set_materializer(stage_type: StageType | str) -> str:
    """Return the materializer of a ``DERIVED_TASK_SET`` stage type, or raise."""

    normalized = StageType.from_value(stage_type)
    if stage_materialization(normalized) is not StageMaterialization.DERIVED_TASK_SET:
        raise ValueError(
            f"Stage type {normalized.value!r} is a single calculation, not a derived task set."
        )
    return _DERIVED_TASK_SET_MATERIALIZERS[normalized]


def stage_type_is_executable(stage_type: StageType | str) -> bool:
    return StageType.from_value(stage_type) not in NON_EXECUTABLE_STAGE_TYPES


__all__ = [
    "NON_EXECUTABLE_STAGE_TYPES",
    "PHONON_DISPLACEMENT_MATERIALIZER",
    "StageMaterialization",
    "derived_task_set_materializer",
    "stage_materialization",
    "stage_type_is_executable",
]
