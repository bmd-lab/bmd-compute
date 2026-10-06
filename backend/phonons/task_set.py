"""Phonon force-stage task sets bound to M1 displacement plans (Phonopy M2a).

The phonon force calculations are one workflow stage (``PHONON_FORCES``) whose
tasks are the displacements of an M1 ``DisplacementPlan``. The plan stays the
scientific authority for the displacement set: displacement vectors, dataset
indices, displaced structures, supercell, primitive matrix, Phonopy policy and
version live only there. The stage task set binds to it by ``plan_sha256`` and
lists only each task's ID (``disp_NNN``) and displaced-structure hash, in
Phonopy dataset order.

Representation only: nothing here creates directories or runs calculations.

Identities (each answers one question):

    task_set.upstream_structure.sha256   what Stage 1 actually produced
                                         (M2b ``incoming_sha256``)
    working_structure.working_sha256     what BMD handed to Phonopy
                                         (= plan.working_structure.sha256)
    task_set.source.sha256               which displacement plan (plan_sha256)

The M2b working-structure record is the authenticated bridge between the first
two. ``verify_phonon_force_task_set_chain`` checks the whole chain from the
actual Stage-1 structure; no link is accepted on a methodology hash alone.
"""

from __future__ import annotations

from typing import Any

from backend.calculations.models import WorkflowSpec
from backend.phonons.plan import DisplacementPlan, verify_displacement_plan
from backend.phonons.records import DISPLACEMENT_PLAN_SCHEMA, DISPLACEMENT_PLAN_SCHEMA_VERSION
from backend.phonons.working_structure import (
    PhononWorkingStructure,
    verify_phonon_working_structure,
)
from backend.stage_task_set import (
    StageTaskSet,
    StageTaskSetContractError,
    build_stage_task_set_record,
    verify_stage_task_set_parent,
)


def build_phonon_force_task_set(
    plan: DisplacementPlan,
    *,
    working_structure: PhononWorkingStructure,
    workflow: WorkflowSpec,
    stage_index: int,
    submission_attempt_id: str,
    upstream_stage_index: int,
) -> StageTaskSet:
    """Materialize the task set of the ``PHONON_FORCES`` stage from ``plan``.

    ``working_structure`` is the M2b record of the Stage-1 output the plan was
    built from. The plan must have been built from exactly its working
    structure, and the task set records its incoming (Stage-1) identity.
    """

    if not isinstance(plan, DisplacementPlan):
        raise StageTaskSetContractError("plan must be a DisplacementPlan")
    _require_plan_from_working_structure(plan, working_structure)
    record = build_stage_task_set_record(
        workflow=workflow,
        stage_index=stage_index,
        submission_attempt_id=submission_attempt_id,
        upstream_stage_index=upstream_stage_index,
        upstream_structure_sha256=working_structure.incoming_sha256,
        source_schema=DISPLACEMENT_PLAN_SCHEMA,
        source_schema_version=DISPLACEMENT_PLAN_SCHEMA_VERSION,
        source_sha256=plan.plan_sha256,
        tasks=[(task.task_id, task.structure_sha256) for task in plan.tasks],
    )
    task_set = StageTaskSet.from_dict(record)
    verify_phonon_force_task_set(task_set, plan)
    return task_set


def verify_phonon_force_task_set_chain(
    task_set: StageTaskSet,
    *,
    workflow: WorkflowSpec,
    submission_attempt_id: str,
    stage1_structure: Any,
    working_structure: PhononWorkingStructure,
    plan: DisplacementPlan,
) -> None:
    """Verify every link from the actual Stage-1 structure to the task set.

    1. the task set belongs to its stage of this attempt's workflow;
    2. the working-structure record re-derives from ``stage1_structure``;
    3. the task set's upstream identity is that Stage-1 structure's identity;
    4. the plan rebuilds from exactly the working structure (M1, Phonopy);
    5. the task set lists exactly the plan's tasks, bound to its hash.
    """

    if not isinstance(task_set, StageTaskSet):
        raise StageTaskSetContractError("task_set must be a StageTaskSet")
    if not isinstance(plan, DisplacementPlan):
        raise StageTaskSetContractError("plan must be a DisplacementPlan")
    verify_stage_task_set_parent(task_set, workflow, submission_attempt_id=submission_attempt_id)
    verify_phonon_working_structure(working_structure, stage1_structure)
    if task_set.upstream_structure["sha256"] != working_structure.incoming_sha256:
        raise StageTaskSetContractError(
            "task set upstream identity is not the Stage-1 structure's identity"
        )
    _require_plan_from_working_structure(plan, working_structure)
    plan_data = plan.to_dict()
    verify_displacement_plan(
        plan,
        working_structure.working_structure(),
        plan.policy,
        supercell_matrix=plan_data["supercell_matrix"],
    )
    verify_phonon_force_task_set(task_set, plan)


def _require_plan_from_working_structure(
    plan: DisplacementPlan, working_structure: PhononWorkingStructure
) -> None:
    if not isinstance(working_structure, PhononWorkingStructure):
        raise StageTaskSetContractError("working_structure must be a PhononWorkingStructure")
    if plan.to_dict()["working_structure"] != working_structure.to_dict()["working"]:
        raise StageTaskSetContractError(
            "displacement plan was not built from this working structure"
        )


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


__all__ = [
    "build_phonon_force_task_set",
    "verify_phonon_force_task_set",
    "verify_phonon_force_task_set_chain",
]
