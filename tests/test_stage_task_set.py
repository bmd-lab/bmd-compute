"""Generic ``bmd_compute.stage_task_set`` v1 contract (Phonopy M2a).

These tests use synthetic hashes so they run without Phonopy; binding to real
M1 displacement plans is tested in ``test_phonon_task_set.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import workflow_stage_directories
from backend.phonons.records import canonical_json
from backend.stage_task_set import (
    STAGE_TASK_SET_SCHEMA,
    StageTaskSet,
    StageTaskSetContractError,
    build_stage_task_set_record,
    stage_spec_sha256,
    stage_task_dir,
    stage_task_dirs,
    stage_task_relative_dir,
    task_set_sha256,
    validate_stage_task_set_record,
    verify_stage_task_set_parent,
    workflow_spec_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = "0b6f6c55-3f43-4a3b-9b0e-7d1c2f6a9e10"
OTHER_ATTEMPT = "5d0f8a2e-91c4-4b8e-8f3a-2a6c0e7b1d44"
WORKFLOW = WorkflowSpec(
    [
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.PHONON_FORCES, Theory.PBE),
    ]
)
STAGE_ROOT = "/home/user/bmd_runs/run_x/stage_02"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_tasks(count: int = 4) -> list[tuple[str, str]]:
    return [(f"disp_{index:03d}", sha(f"displaced structure {index}")) for index in range(1, count + 1)]


def record(count: int = 4, workflow: WorkflowSpec = WORKFLOW, **overrides) -> dict:
    arguments = dict(
        workflow=workflow,
        stage_index=2,
        submission_attempt_id=ATTEMPT,
        upstream_stage_index=1,
        upstream_structure_sha256=sha("relaxed structure"),
        source_schema="bmd_compute.phonon_displacement_plan",
        source_schema_version=1,
        source_sha256=sha("plan"),
        tasks=synthetic_tasks(count),
    )
    arguments.update(overrides)
    return build_stage_task_set_record(**arguments)


def rehash(data: dict) -> dict:
    data["task_set_sha256"] = task_set_sha256(data)
    return data


def reject(data, match: str | None = None):
    with pytest.raises(StageTaskSetContractError, match=match):
        validate_stage_task_set_record(data)
    with pytest.raises(StageTaskSetContractError):
        StageTaskSet.from_dict(data)


# --- stage/task semantics ----------------------------------------------------------


def test_many_tasks_belong_to_one_stage():
    task_set = StageTaskSet.from_dict(record(count=12))

    assert task_set.task_count == 12
    assert len(WORKFLOW.stages) == 2
    assert task_set.parent["stage_index"] == 2
    assert task_set.parent["stage_type"] == "phonon_forces"
    assert task_set.parent["stage_sha256"] == stage_spec_sha256(WORKFLOW.stages[1])
    assert task_set.parent["workflow_sha256"] == workflow_spec_sha256(WORKFLOW)


def test_record_shape_has_no_methodology_scheduler_or_path_fields():
    data = record()

    assert set(data) == {
        "schema",
        "schema_version",
        "parent",
        "upstream_structure",
        "source",
        "task_order",
        "tasks",
        "task_set_sha256",
    }
    assert data["task_order"] == "source_canonical"
    assert all(set(task) == {"task_id", "input_sha256"} for task in data["tasks"])
    text = json.dumps(data).lower()
    for word in ("incar", "kpoint", "theory", "modifier", "resource", "slurm", "job_id",
                 "retry", "concurren", "parallel", "sequential", "execution", "/home", "path"):
        assert word not in text


@pytest.mark.parametrize(
    "key, value",
    [
        ("theory", "hse06"),
        ("modifiers", ["soc"]),
        ("incar", {"ENCUT": 800}),
        ("kpoints", [[4, 4, 4]]),
        ("resources", {"ntasks": 8}),
        ("job_id", "123"),
        ("execution_order", 0),
        ("path", "/tmp/x"),
        ("retry", 1),
    ],
)
def test_tasks_cannot_carry_methodology_or_execution_state(key, value):
    data = record()
    data["tasks"][0][key] = value
    reject(rehash(data), "tasks\\[0\\] keys")


@pytest.mark.parametrize("key", ["execution_order", "incar", "kpoints", "stage_dirs", "created_at", "host"])
def test_record_cannot_carry_extra_top_level_state(key):
    data = record()
    data[key] = "x"
    reject(rehash(data), "task set keys")


def test_canonical_order_is_preserved_and_part_of_identity():
    task_set = StageTaskSet.from_dict(record())
    assert task_set.task_ids == ("disp_001", "disp_002", "disp_003", "disp_004")

    data = record()
    data["tasks"][0], data["tasks"][1] = data["tasks"][1], data["tasks"][0]
    reject(rehash(data), "canonical order")


def test_task_directories_do_not_become_stages():
    existing = WorkflowSpec([StageSpec(StageType.RELAX), StageSpec(StageType.STATIC)])
    assert workflow_stage_directories(existing) == ("stage_01", "stage_02")
    assert set(stage_task_dirs(StageTaskSet.from_dict(record()), STAGE_ROOT).values()) == {
        f"{STAGE_ROOT}/tasks/disp_00{index}" for index in range(1, 5)
    }


# --- structural validation -----------------------------------------------------------


def _mutate(path, value):
    def apply(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value
        return data

    return apply


_DELETE = object()

MUTATIONS = {
    "wrong-schema": _mutate(("schema",), "bmd_compute.stage_task_plan"),
    "version-2": _mutate(("schema_version",), 2),
    "version-bool": _mutate(("schema_version",), True),
    "version-string": _mutate(("schema_version",), "1"),
    "missing-parent": _mutate(("parent",), _DELETE),
    "attempt-uppercase": _mutate(("parent", "submission_attempt_id"), ATTEMPT.upper()),
    "attempt-not-uuid": _mutate(("parent", "submission_attempt_id"), "attempt-1"),
    "stage-index-zero": _mutate(("parent", "stage_index"), 0),
    "stage-index-bool": _mutate(("parent", "stage_index"), True),
    "single-stage-parent": _mutate(("parent", "stage_type"), "relax"),
    "unknown-stage-type": _mutate(("parent", "stage_type"), "phonons"),
    "non-normalized-stage-type": _mutate(("parent", "stage_type"), "Phonon-Forces"),
    "bad-stage-sha": _mutate(("parent", "stage_sha256"), "ABC"),
    "upstream-not-earlier": _mutate(("upstream_structure", "stage_index"), 2),
    "upstream-bad-sha": _mutate(("upstream_structure", "sha256"), sha("x").upper()),
    "wrong-materializer": _mutate(("source", "materializer"), "bmd_compute.other"),
    "wrong-source-schema": _mutate(("source", "schema"), "bmd_compute.stage_task_set"),
    "wrong-source-version": _mutate(("source", "schema_version"), 2),
    "unordered": _mutate(("task_order",), "unordered"),
    "execution-order": _mutate(("task_order",), "execution"),
    "empty-tasks": _mutate(("tasks",), []),
    "tasks-not-list": _mutate(("tasks",), {"disp_001": sha("a")}),
    "non-canonical-float": _mutate(("parent", "stage_index"), 2.0),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_structural_validation_rejects(name):
    reject(rehash(MUTATIONS[name](record())))


def test_hash_must_match_content():
    data = record()
    data["source"]["sha256"] = sha("another plan")
    reject(data, "task_set_sha256")


def test_duplicate_task_ids_are_rejected():
    data = record()
    data["tasks"][1]["task_id"] = "disp_001"
    reject(rehash(data), "duplicated")


def test_missing_task_in_the_middle_is_rejected():
    data = record()
    del data["tasks"][1]
    reject(rehash(data), "canonical order")


def test_duplicate_scientific_inputs_are_rejected():
    data = record()
    data["tasks"][1]["input_sha256"] = data["tasks"][0]["input_sha256"]
    reject(rehash(data), "duplicates another task")


@pytest.mark.parametrize(
    "task_id",
    ["../disp_001", "disp_001/..", "/disp_001", "disp_001/x", "DISP_001", "disp_1", "disp_", "",
     "disp_001\x00", "disp 001", "tasks", ".", "..", "disp_0001x", 1],
)
def test_invalid_task_ids_are_rejected(task_id):
    data = record()
    data["tasks"][0]["task_id"] = task_id
    reject(rehash(data))


def test_builder_refuses_single_calculation_stages_and_out_of_range_indices():
    with pytest.raises(StageTaskSetContractError, match="single calculation"):
        record(stage_index=1, upstream_stage_index=1)
    with pytest.raises(StageTaskSetContractError, match="outside"):
        record(stage_index=3)
    with pytest.raises(StageTaskSetContractError):
        record(workflow=WORKFLOW.to_dict())


# --- binding to the parent -----------------------------------------------------------


def test_parent_verification_accepts_the_bound_stage():
    verify_stage_task_set_parent(StageTaskSet.from_dict(record()), WORKFLOW, submission_attempt_id=ATTEMPT)


def test_task_set_cannot_be_reused_for_another_attempt():
    with pytest.raises(StageTaskSetContractError, match="different submission attempt"):
        verify_stage_task_set_parent(
            StageTaskSet.from_dict(record()), WORKFLOW, submission_attempt_id=OTHER_ATTEMPT
        )


def test_task_set_cannot_be_reused_for_another_workflow_or_methodology():
    other = WorkflowSpec(
        [StageSpec(StageType.RELAX, Theory.PBE), StageSpec(StageType.PHONON_FORCES, Theory.PBE, {Modifier.SPIN_POLARIZED})]
    )
    with pytest.raises(StageTaskSetContractError, match="different workflow"):
        verify_stage_task_set_parent(StageTaskSet.from_dict(record()), other, submission_attempt_id=ATTEMPT)


def test_task_set_cannot_be_moved_to_another_stage():
    three = WorkflowSpec(
        [
            StageSpec(StageType.RELAX),
            StageSpec(StageType.PHONON_FORCES),
            StageSpec(StageType.PHONON_FORCES, label="second"),
        ]
    )
    data = record(workflow=three)
    data["parent"]["stage_index"] = 3
    forged = StageTaskSet.from_dict(rehash(data))
    with pytest.raises(StageTaskSetContractError, match="methodology differs"):
        verify_stage_task_set_parent(forged, three, submission_attempt_id=ATTEMPT)

    data = record(workflow=three)
    data["parent"]["stage_index"] = 1
    data["upstream_structure"]["stage_index"] = 0
    with pytest.raises(StageTaskSetContractError):
        StageTaskSet.from_dict(rehash(data))


# --- determinism and immutability -----------------------------------------------------


def test_construction_is_deterministic_and_round_trips():
    first = StageTaskSet.from_dict(record())
    second = StageTaskSet.from_dict(record())

    assert first.to_json() == second.to_json()
    assert first.task_set_sha256 == second.task_set_sha256
    assert StageTaskSet.from_json(first.to_json()) == first
    assert StageTaskSet.from_dict(first.to_dict()).to_json() == first.to_json()
    assert first.to_json() == canonical_json(first.to_dict())


def test_identity_ignores_key_insertion_order():
    data = record()

    def reverse(value):
        if isinstance(value, dict):
            return {key: reverse(value[key]) for key in reversed(list(value))}
        if isinstance(value, list):
            return [reverse(item) for item in value]
        return value

    assert StageTaskSet.from_dict(reverse(data)).to_json() == StageTaskSet.from_dict(data).to_json()


def test_non_canonical_text_is_rejected():
    text = StageTaskSet.from_dict(record()).to_json()
    with pytest.raises(StageTaskSetContractError, match="canonical form"):
        StageTaskSet(json.dumps(json.loads(text), indent=1))
    with pytest.raises(StageTaskSetContractError):
        StageTaskSet(json.loads(text))


def test_records_are_immutable():
    task_set = StageTaskSet.from_dict(record())
    exported = task_set.to_dict()
    exported["tasks"].append({"task_id": "disp_999", "input_sha256": sha("x")})
    exported["parent"]["stage_index"] = 7
    task_set.parent["stage_index"] = 9

    assert task_set.task_count == 4
    assert task_set.parent["stage_index"] == 2
    with pytest.raises(AttributeError):
        task_set._canonical = "{}"


def test_building_does_not_retain_caller_state():
    tasks = synthetic_tasks()
    data = record(tasks=tasks)
    before = copy.deepcopy(data)
    tasks.append(("disp_005", sha("late")))
    assert data == before


def test_identity_is_independent_of_hash_seed_cwd_and_locale(tmp_path):
    script = (
        "import json, hashlib\n"
        "from backend.calculations.models import StageSpec, WorkflowSpec\n"
        "from backend.stage_task_set import StageTaskSet, build_stage_task_set_record\n"
        "sha = lambda t: hashlib.sha256(t.encode()).hexdigest()\n"
        "wf = WorkflowSpec([StageSpec('relax'), StageSpec('phonon_forces')])\n"
        "r = build_stage_task_set_record(workflow=wf, stage_index=2,\n"
        f"    submission_attempt_id={ATTEMPT!r}, upstream_stage_index=1,\n"
        "    upstream_structure_sha256=sha('relaxed structure'),\n"
        "    source_schema='bmd_compute.phonon_displacement_plan', source_schema_version=1,\n"
        "    source_sha256=sha('plan'),\n"
        "    tasks=[(f'disp_{i:03d}', sha(f'displaced structure {i}')) for i in range(1, 5)])\n"
        "print(StageTaskSet.from_dict(r).to_json())\n"
    )
    outputs = set()
    for seed, locale, cwd in (("0", "C", tmp_path), ("4242", "C.UTF-8", REPO_ROOT)):
        env = dict(os.environ, PYTHONHASHSEED=seed, LC_ALL=locale, PYTHONPATH=str(REPO_ROOT))
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stderr
        outputs.add(completed.stdout)
    assert outputs == {StageTaskSet.from_dict(record()).to_json() + "\n"}


# --- task directories ----------------------------------------------------------------


def test_task_directories_are_derived_below_the_stage_tasks_root():
    assert stage_task_relative_dir("disp_001") == "tasks/disp_001"
    assert stage_task_dir(STAGE_ROOT, "disp_042") == f"{STAGE_ROOT}/tasks/disp_042"


@pytest.mark.parametrize(
    "task_id", ["../stage_01", "disp_001/../../x", "/etc", "disp_001/", "a/b", "..", ".", "", None, "Disp_001"]
)
def test_task_directory_rejects_unsafe_task_ids(task_id):
    with pytest.raises(StageTaskSetContractError):
        stage_task_dir(STAGE_ROOT, task_id)
    with pytest.raises(StageTaskSetContractError):
        stage_task_relative_dir(task_id)


@pytest.mark.parametrize(
    "stage_root",
    ["stage_02", "./stage_02", "/runs/x/../stage_02", "/runs/x/stage_02/", "/", "//runs", "/runs/x stage",
     "/runs/$(id)", "", None],
)
def test_task_directory_rejects_unsafe_stage_roots(stage_root):
    with pytest.raises(StageTaskSetContractError):
        stage_task_dir(stage_root, "disp_001")


def test_schema_name_is_distinct_from_existing_records():
    assert STAGE_TASK_SET_SCHEMA == "bmd_compute.stage_task_set"
    assert STAGE_TASK_SET_SCHEMA not in {"bmd_compute.submission", "bmd_compute.job_record"}
