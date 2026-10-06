"""Phonon force-stage task sets bound to M1 displacement plans (Phonopy M2a).

The phonon force calculations are one workflow stage (``PHONON_FORCES``) whose
tasks are the displacements of an M1 ``DisplacementPlan``. The plan stays the
scientific authority for the displacement set: displacement vectors, dataset
indices, displaced structures, supercell, primitive matrix, Phonopy policy and
version live only there. The stage task set binds to it by ``plan_sha256`` and
lists only each task's ID (``disp_NNN``) and displaced-structure hash, in
Phonopy dataset order.

Representation only: nothing here creates directories or runs calculations.

Verification chain (each link is a separate check):

    workflow + attempt --verify_stage_task_set_parent--> task set
    task set --verify_phonon_force_task_set--> DisplacementPlan
    DisplacementPlan --verify_displacement_plan (M1, needs Phonopy)--> structure

How the upstream structure relates to the plan's working structure is the
phonon working-structure boundary (M2b); it is recorded here, not checked.
"""

from __future__ import annotations

from backend.calculations.models import WorkflowSpec
from backend.phonons.plan import DisplacementPlan
from backend.phonons.records import DISPLACEMENT_PLAN_SCHEMA, DISPLACEMENT_PLAN_SCHEMA_VERSION
from backend.stage_task_set import (
    StageTaskSet,
    StageTaskSetContractError,
    build_stage_task_set_record,
)


def build_phonon_force_task_set(
    plan: DisplacementPlan,
    *,
    workflow: WorkflowSpec,
    stage_index: int,
    submission_attempt_id: str,
    upstream_stage_index: int,
    upstream_structure_sha256: str,
) -> StageTaskSet:
    """Materialize the task set of the ``PHONON_FORCES`` stage from ``plan``."""

    if not isinstance(plan, DisplacementPlan):
        raise StageTaskSetContractError("plan must be a DisplacementPlan")
    record = build_stage_task_set_record(
        workflow=workflow,
        stage_index=stage_index,
        submission_attempt_id=submission_attempt_id,
        upstream_stage_index=upstream_stage_index,
        upstream_structure_sha256=upstream_structure_sha256,
        source_schema=DISPLACEMENT_PLAN_SCHEMA,
        source_schema_version=DISPLACEMENT_PLAN_SCHEMA_VERSION,
        source_sha256=plan.plan_sha256,
        tasks=[(task.task_id, task.structure_sha256) for task in plan.tasks],
    )
    task_set = StageTaskSet.from_dict(record)
    verify_phonon_force_task_set(task_set, plan)
    return task_set


def verify_phonon_force_task_set(task_set: StageTaskSet, plan: DisplacementPlan) -> None:
    """Require ``task_set`` to be exactly the task set of ``plan``.

    The bound plan hash, the task IDs, their canonical (dataset) order and each
    task's displaced-structure hash must all match the plan.
    """

    if not isinstance(task_set, StageTaskSet):
        raise StageTaskSetContractError("task_set must be a StageTaskSet")
    if not isinstance(plan, DisplacementPlan):
        raise StageTaskSetContractError("plan must be a DisplacementPlan")
    source = task_set.source
    if source["schema"] != DISPLACEMENT_PLAN_SCHEMA:
        raise StageTaskSetContractError("task set is not derived from a displacement plan")
    if source["sha256"] != plan.plan_sha256:
        raise StageTaskSetContractError("task set is bound to a different displacement plan")
    expected = [(task.task_id, task.structure_sha256) for task in plan.tasks]
    actual = [(task["task_id"], task["input_sha256"]) for task in task_set.to_dict()["tasks"]]
    if actual != expected:
        raise StageTaskSetContractError(
            "task set tasks are not the displacement plan's tasks in dataset order"
        )


__all__ = ["build_phonon_force_task_set", "verify_phonon_force_task_set"]
