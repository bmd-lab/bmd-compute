"""Phonon force-stage task sets bound to real M1 displacement plans (Phonopy M2a)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("phonopy")

from pymatgen.core import Lattice, Structure  # noqa: E402

from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec  # noqa: E402
from backend.phonons import DisplacementPlan, build_displacement_plan  # noqa: E402
from backend.phonons.working_structure import (  # noqa: E402
    PhononWorkingStructure,
    prepare_phonon_working_structure,
)
from backend.phonons.task_set import (  # noqa: E402
    build_phonon_force_task_set,
    verify_phonon_force_task_set,
)
from backend.stage_task_set import (  # noqa: E402
    StageTaskSet,
    StageTaskSetContractError,
    stage_task_dirs,
    task_set_sha256,
    verify_stage_task_set_parent,
)


ATTEMPT = "0b6f6c55-3f43-4a3b-9b0e-7d1c2f6a9e10"
DIAG2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
WORKFLOW = WorkflowSpec(
    [
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.PHONON_FORCES, Theory.PBE),
    ]
)


def si() -> Structure:
    return Structure(
        Lattice([[0.0, 2.734, 2.734], [2.734, 0.0, 2.734], [2.734, 2.734, 0.0]]),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def nacl() -> Structure:
    return Structure(
        Lattice([[0.0, 2.845, 2.845], [2.845, 0.0, 2.845], [2.845, 2.845, 0.0]]),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def gan() -> Structure:
    return Structure(
        Lattice.hexagonal(3.19, 5.19),
        ["Ga", "Ga", "N", "N"],
        [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.377], [2 / 3, 1 / 3, 0.877]],
    )


def mos2() -> Structure:
    z = 0.621
    return Structure(
        Lattice.hexagonal(3.16, 12.3),
        ["Mo", "Mo", "S", "S", "S", "S"],
        [
            [1 / 3, 2 / 3, 0.25],
            [2 / 3, 1 / 3, 0.75],
            [1 / 3, 2 / 3, z],
            [2 / 3, 1 / 3, z - 0.5],
            [2 / 3, 1 / 3, 1 - z],
            [1 / 3, 2 / 3, 1.5 - z],
        ],
    )


# The same reference cases as the M1 tests: name, structure, supercell, tasks.
REFERENCE_CASES = [
    ("Si", si, DIAG2, 1),
    ("NaCl", nacl, DIAG2, 2),
    ("GaN", gan, [[3, 0, 0], [0, 3, 0], [0, 0, 2]], 4),
    ("MoS2", mos2, [[3, 0, 0], [0, 3, 0], [0, 0, 1]], 3),
]


# Plan hash -> the M2b working-structure record the plan was built from.
WORKING: dict[str, PhononWorkingStructure] = {}


def plan_from_stage1(structure: Structure, matrix) -> DisplacementPlan:
    working = prepare_phonon_working_structure(structure)
    plan = build_displacement_plan(working.working_structure(), supercell_matrix=matrix)
    WORKING[plan.plan_sha256] = working
    return plan


@pytest.fixture(scope="module")
def plans() -> dict[str, DisplacementPlan]:
    return {
        name: plan_from_stage1(factory(), matrix)
        for name, factory, matrix, _count in REFERENCE_CASES
    }


def task_set_for(plan: DisplacementPlan, **overrides) -> StageTaskSet:
    arguments = dict(
        working_structure=WORKING[plan.plan_sha256],
        workflow=WORKFLOW,
        stage_index=2,
        submission_attempt_id=ATTEMPT,
        upstream_stage_index=1,
    )
    arguments.update(overrides)
    return build_phonon_force_task_set(plan, **arguments)


def forged(task_set: StageTaskSet, mutate) -> StageTaskSet:
    data = task_set.to_dict()
    mutate(data)
    data["task_set_sha256"] = task_set_sha256(data)
    return StageTaskSet.from_dict(data)


@pytest.mark.parametrize("name, _factory, _matrix, count", REFERENCE_CASES, ids=[c[0] for c in REFERENCE_CASES])
def test_task_set_lists_exactly_the_plan_tasks_in_dataset_order(plans, name, _factory, _matrix, count):
    plan = plans[name]
    task_set = task_set_for(plan)

    assert task_set.task_count == plan.task_count == count
    assert task_set.task_ids == tuple(task.task_id for task in plan.tasks)
    assert task_set.task_ids == tuple(f"disp_{index:03d}" for index in range(1, count + 1))
    assert [task["input_sha256"] for task in task_set.to_dict()["tasks"]] == [
        task.structure_sha256 for task in plan.tasks
    ]
    assert [task.dataset_index for task in plan.tasks] == list(range(count))
    assert task_set.source == {
        "materializer": "bmd_compute.phonon_displacement_plan",
        "schema": "bmd_compute.phonon_displacement_plan",
        "schema_version": 1,
        "sha256": plan.plan_sha256,
    }
    verify_phonon_force_task_set(task_set, plan)
    verify_stage_task_set_parent(task_set, WORKFLOW, submission_attempt_id=ATTEMPT)


def test_task_set_does_not_duplicate_m1_scientific_content(plans):
    plan = plans["GaN"]
    text = task_set_for(plan).to_json()

    for word in ("displacement", "dataset", "supercell", "primitive", "policy", "phonopy_arguments",
                 "lattice", "species", "coords", "atom_index", "symmetry", "software"):
        assert f'"{word}' not in text
    assert len(text) < len(plan.to_json()) / 4


def test_one_stage_holds_all_displacement_tasks(plans):
    task_set = task_set_for(plans["GaN"])
    assert len(WORKFLOW.stages) == 2
    assert {task_set.parent["stage_index"]} == {2}
    assert list(stage_task_dirs(task_set, "/runs/r/stage_02").values()) == [
        f"/runs/r/stage_02/tasks/disp_00{index}" for index in range(1, 5)
    ]


def test_construction_is_deterministic(plans):
    first = task_set_for(plans["NaCl"])
    rebuilt = task_set_for(plan_from_stage1(nacl(), DIAG2))

    assert first.to_json() == rebuilt.to_json()
    assert StageTaskSet.from_json(first.to_json()).task_set_sha256 == first.task_set_sha256


# --- forged task sets ------------------------------------------------------------------


def test_task_set_cannot_claim_a_different_plan(plans):
    with pytest.raises(StageTaskSetContractError, match="different displacement plan"):
        verify_phonon_force_task_set(task_set_for(plans["NaCl"]), plans["GaN"])

    relabelled = forged(
        task_set_for(plans["NaCl"]),
        lambda data: data["source"].update(sha256=plans["Si"].plan_sha256),
    )
    with pytest.raises(StageTaskSetContractError):
        verify_phonon_force_task_set(relabelled, plans["Si"])


def test_swapped_task_inputs_are_rejected(plans):
    def swap(data):
        tasks = data["tasks"]
        tasks[0]["input_sha256"], tasks[1]["input_sha256"] = tasks[1]["input_sha256"], tasks[0]["input_sha256"]

    with pytest.raises(StageTaskSetContractError, match="dataset order"):
        verify_phonon_force_task_set(forged(task_set_for(plans["GaN"]), swap), plans["GaN"])


def test_missing_and_extra_tasks_are_rejected(plans):
    plan = plans["GaN"]
    missing = forged(task_set_for(plan), lambda data: data["tasks"].pop())
    extra = forged(
        task_set_for(plan),
        lambda data: data["tasks"].append(
            {"task_id": "disp_005", "input_sha256": plans["Si"].tasks[0].structure_sha256}
        ),
    )
    for task_set in (missing, extra):
        with pytest.raises(StageTaskSetContractError, match="dataset order"):
            verify_phonon_force_task_set(task_set, plan)


def test_task_input_from_another_plan_is_rejected(plans):
    def borrow(data):
        data["tasks"][0]["input_sha256"] = plans["Si"].tasks[0].structure_sha256

    with pytest.raises(StageTaskSetContractError):
        verify_phonon_force_task_set(forged(task_set_for(plans["NaCl"]), borrow), plans["NaCl"])


def test_plan_cannot_materialize_into_a_single_calculation_stage(plans):
    with pytest.raises(StageTaskSetContractError, match="single calculation"):
        task_set_for(plans["Si"], stage_index=1, upstream_stage_index=1)


def test_only_validated_plan_objects_are_accepted(plans):
    with pytest.raises(StageTaskSetContractError):
        build_phonon_force_task_set(
            plans["Si"].to_dict(),
            working_structure=WORKING[plans["Si"].plan_sha256],
            workflow=WORKFLOW,
            stage_index=2,
            submission_attempt_id=ATTEMPT,
            upstream_stage_index=1,
        )
    with pytest.raises(StageTaskSetContractError):
        verify_phonon_force_task_set(task_set_for(plans["Si"]).to_dict(), plans["Si"])
    with pytest.raises(StageTaskSetContractError):
        verify_phonon_force_task_set(task_set_for(plans["Si"]), json.loads(plans["Si"].to_json()))


def test_plan_and_task_set_records_cannot_be_confused(plans):
    plan = plans["NaCl"]
    task_set = task_set_for(plan)
    with pytest.raises(StageTaskSetContractError):
        StageTaskSet.from_dict(plan.to_dict())
    with pytest.raises(ValueError):
        DisplacementPlan.from_dict(task_set.to_dict())
    assert task_set.task_set_sha256 != plan.plan_sha256
