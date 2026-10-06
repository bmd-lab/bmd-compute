"""Materialized task sets of derived-task-set workflow stages (Phonopy M2a).

A ``DERIVED_TASK_SET`` stage (``backend.calculations.stage_materialization``) is
one scientific stage whose concrete calculations are task instances derived at
runtime from an upstream result. ``bmd_compute.stage_task_set`` v1 is the
authoritative record of the task instances materialized for one such stage. It
is representation only: nothing in BMD Compute executes it yet.

Record (all keys required, no others)::

    schema            "bmd_compute.stage_task_set"
    schema_version    1
    parent            the stage the tasks belong to
        submission_attempt_id   attempt whose immutable recipe holds the stage
        workflow_sha256         canonical hash of the attempt's WorkflowSpec
        stage_index             1-based position of the stage in that workflow
        stage_type              a DERIVED_TASK_SET stage type
        stage_sha256            canonical hash of that StageSpec (methodology)
    upstream_structure  the earlier stage output the source was derived from
        stage_index             1-based, earlier than parent.stage_index
        sha256                  identity of that stage's actual output
                                structure, as the materializer defines it
                                (phonons: the M2b incoming Stage-1 identity,
                                ``PhononWorkingStructure.incoming_sha256``;
                                never the working-structure hash)
    source            the record that defines the tasks (the materializer's)
        materializer            the parent stage type's materializer
        schema, schema_version  the source record's schema
        sha256                  the source record's own canonical hash
    task_order        "source_canonical"
    tasks             [{"task_id", "input_sha256"}, ...]
    task_set_sha256   canonical hash of every other field

What a task is, and is not:

* a task has a deterministic ID, belongs to exactly one parent stage, and names
  its scientific input by hash (``input_sha256``); the source record holds the
  input itself;
* a task carries no theory, modifiers, INCAR, KPOINTS, resources, job IDs,
  retry state or paths. Methodology belongs to the parent ``StageSpec`` and is
  shared by every task (shown once, never per task); execution state belongs to
  future execution records;
* ``tasks`` are listed in the source's canonical association order (for
  phonons, Phonopy dataset order, which force assembly needs). That order is
  part of the identity and is never an execution order: tasks may later run in
  any order or concurrently.

Canonical form and hashing reuse the M1 rules (``backend.phonons.records``).
The record holds no timestamps, hosts, absolute paths or scheduler data, so its
hash depends only on scientific identity.

Validation has two levels, as for M1 displacement plans:

* ``validate_stage_task_set_record`` checks the record on its own: structure,
  schema, identifiers, parent/materializer consistency and hash. A record that
  passes is internally consistent, not authoritative;
* re-verification checks it against what it claims to bind to:
  ``verify_stage_task_set_parent`` against the workflow and attempt, and the
  materializer's own check against the source record (for phonons,
  ``backend.phonons.task_set.verify_phonon_force_task_set``).

Task directories are derived, never recorded: ``stage_task_dir`` maps a
validated task ID to ``<stage root>/tasks/<task_id>``. Directory names are not
scientific authority, and task directories are not stages: ``paths.stage_dirs``
keeps listing stage roots only.
"""

from __future__ import annotations

import json
import posixpath
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.calculations.models import StageSpec, StageType, WorkflowSpec
from backend.calculations.stage_materialization import (
    PHONON_DISPLACEMENT_MATERIALIZER,
    StageMaterialization,
    derived_task_set_materializer,
    stage_materialization,
)
from backend.phonons.records import (
    DISPLACEMENT_PLAN_SCHEMA,
    DISPLACEMENT_PLAN_SCHEMA_VERSION,
    PhononPlanContractError,
    canonical_json,
    canonical_sha256,
    canonical_value,
    task_id_for_index,
)


STAGE_TASK_SET_SCHEMA = "bmd_compute.stage_task_set"
STAGE_TASK_SET_SCHEMA_VERSION = 1
TASK_ORDER_SOURCE_CANONICAL = "source_canonical"
STAGE_TASKS_DIRNAME = "tasks"

# Any task ID is a single lower-case path segment; materializers narrow it.
TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STAGE_ROOT_PATTERN = re.compile(r"^/[A-Za-z0-9._/+\-]+$")

RECORD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "parent",
        "upstream_structure",
        "source",
        "task_order",
        "tasks",
        "task_set_sha256",
    }
)
PARENT_KEYS = frozenset(
    {"submission_attempt_id", "workflow_sha256", "stage_index", "stage_type", "stage_sha256"}
)
UPSTREAM_KEYS = frozenset({"stage_index", "sha256"})
SOURCE_KEYS = frozenset({"materializer", "schema", "schema_version", "sha256"})
TASK_KEYS = frozenset({"task_id", "input_sha256"})


class StageTaskSetContractError(ValueError):
    """Raised when a stage task set does not satisfy its contract."""


@dataclass(frozen=True)
class _MaterializerContract:
    source_schema: str
    source_schema_version: int
    # Deterministic task ID for a 0-based position in canonical order.
    task_id_for_position: Callable[[int], str]


_MATERIALIZER_CONTRACTS: dict[str, _MaterializerContract] = {
    PHONON_DISPLACEMENT_MATERIALIZER: _MaterializerContract(
        source_schema=DISPLACEMENT_PLAN_SCHEMA,
        source_schema_version=DISPLACEMENT_PLAN_SCHEMA_VERSION,
        task_id_for_position=task_id_for_index,
    ),
}


# --- identities ------------------------------------------------------------------------


def stage_spec_sha256(stage: StageSpec) -> str:
    """Canonical hash of one stage's methodology."""

    return _canonical_sha256(StageSpec.from_dict(stage.to_dict()).to_dict(), "stage")


def workflow_spec_sha256(workflow: WorkflowSpec) -> str:
    """Canonical hash of a workflow (ordered stages, label and recipe)."""

    return _canonical_sha256(WorkflowSpec.from_dict(workflow.to_dict()).to_dict(), "workflow")


def task_set_sha256(record: Mapping[str, Any]) -> str:
    """SHA-256 over every field of the record except ``task_set_sha256`` itself."""

    body = {key: value for key, value in record.items() if key != "task_set_sha256"}
    return _canonical_sha256(body, "task set")


# --- construction ----------------------------------------------------------------------


def build_stage_task_set_record(
    *,
    workflow: WorkflowSpec,
    stage_index: int,
    submission_attempt_id: str,
    upstream_stage_index: int,
    upstream_structure_sha256: str,
    source_schema: str,
    source_schema_version: int,
    source_sha256: str,
    tasks: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Return a validated v1 record for stage ``stage_index`` of ``workflow``.

    ``tasks`` are ``(task_id, input_sha256)`` pairs in the source's canonical
    order. The materializer is taken from the stage type, never from the caller.
    Materializers wrap this; it is not a public way to invent task sets.
    """

    if not isinstance(workflow, WorkflowSpec):
        raise StageTaskSetContractError("workflow must be a WorkflowSpec")
    stage = _workflow_stage(workflow, stage_index)
    try:
        materializer = derived_task_set_materializer(stage.stage_type)
    except ValueError as exc:
        raise StageTaskSetContractError(str(exc)) from None
    record = {
        "schema": STAGE_TASK_SET_SCHEMA,
        "schema_version": STAGE_TASK_SET_SCHEMA_VERSION,
        "parent": {
            "submission_attempt_id": submission_attempt_id,
            "workflow_sha256": workflow_spec_sha256(workflow),
            "stage_index": stage_index,
            "stage_type": stage.stage_type.value,
            "stage_sha256": stage_spec_sha256(stage),
        },
        "upstream_structure": {
            "stage_index": upstream_stage_index,
            "sha256": upstream_structure_sha256,
        },
        "source": {
            "materializer": materializer,
            "schema": source_schema,
            "schema_version": source_schema_version,
            "sha256": source_sha256,
        },
        "task_order": TASK_ORDER_SOURCE_CANONICAL,
        "tasks": [{"task_id": task_id, "input_sha256": input_sha} for task_id, input_sha in tasks],
    }
    record["task_set_sha256"] = task_set_sha256(record)
    validate_stage_task_set_record(record)
    return record


# --- structural validation -------------------------------------------------------------


def validate_stage_task_set_record(payload: Any) -> None:
    """Raise StageTaskSetContractError unless ``payload`` is a consistent v1 record.

    Malformed or inconsistent records are rejected, never repaired. Passing
    this check does not make a record authoritative; see the re-verification
    functions.
    """

    record = _mapping(payload, "task set")
    _exact_keys(record, RECORD_KEYS, "task set")
    if record["schema"] != STAGE_TASK_SET_SCHEMA:
        raise StageTaskSetContractError(f"schema must be {STAGE_TASK_SET_SCHEMA!r}")
    if _int(record["schema_version"], "schema_version") != STAGE_TASK_SET_SCHEMA_VERSION:
        raise StageTaskSetContractError(
            f"schema_version must be {STAGE_TASK_SET_SCHEMA_VERSION}, got {record['schema_version']!r}"
        )
    try:
        canonical_value(record)
    except PhononPlanContractError as exc:
        raise StageTaskSetContractError(f"task set is not canonical data: {exc}") from None

    parent = _mapping(record["parent"], "parent")
    _exact_keys(parent, PARENT_KEYS, "parent")
    _attempt_id(parent["submission_attempt_id"])
    _sha(parent["workflow_sha256"], "parent.workflow_sha256")
    _sha(parent["stage_sha256"], "parent.stage_sha256")
    parent_index = _positive_int(parent["stage_index"], "parent.stage_index")
    stage_type = _stage_type(parent["stage_type"])
    if stage_materialization(stage_type) is not StageMaterialization.DERIVED_TASK_SET:
        raise StageTaskSetContractError(
            f"parent.stage_type {stage_type.value!r} is a single calculation and has no task set"
        )

    upstream = _mapping(record["upstream_structure"], "upstream_structure")
    _exact_keys(upstream, UPSTREAM_KEYS, "upstream_structure")
    upstream_index = _positive_int(upstream["stage_index"], "upstream_structure.stage_index")
    if upstream_index >= parent_index:
        raise StageTaskSetContractError(
            "upstream_structure.stage_index must be an earlier stage than parent.stage_index"
        )
    _sha(upstream["sha256"], "upstream_structure.sha256")

    source = _mapping(record["source"], "source")
    _exact_keys(source, SOURCE_KEYS, "source")
    materializer = derived_task_set_materializer(stage_type)
    if source["materializer"] != materializer:
        raise StageTaskSetContractError(
            f"source.materializer must be {materializer!r} for a {stage_type.value!r} stage"
        )
    contract = _MATERIALIZER_CONTRACTS[materializer]
    if source["schema"] != contract.source_schema:
        raise StageTaskSetContractError(f"source.schema must be {contract.source_schema!r}")
    if _int(source["schema_version"], "source.schema_version") != contract.source_schema_version:
        raise StageTaskSetContractError(
            f"source.schema_version must be {contract.source_schema_version}"
        )
    _sha(source["sha256"], "source.sha256")

    if record["task_order"] != TASK_ORDER_SOURCE_CANONICAL:
        raise StageTaskSetContractError(f"task_order must be {TASK_ORDER_SOURCE_CANONICAL!r}")
    tasks = record["tasks"]
    if type(tasks) is not list or not tasks:
        raise StageTaskSetContractError("tasks must be a non-empty list")
    seen_ids: set[str] = set()
    seen_inputs: set[str] = set()
    for position, task in enumerate(tasks):
        label = f"tasks[{position}]"
        task = _mapping(task, label)
        _exact_keys(task, TASK_KEYS, label)
        task_id = validate_task_id(task["task_id"])
        if task_id in seen_ids:
            raise StageTaskSetContractError(f"{label}.task_id {task_id!r} is duplicated")
        expected = contract.task_id_for_position(position)
        if task_id != expected:
            raise StageTaskSetContractError(
                f"{label}.task_id must be {expected!r} in canonical order, got {task_id!r}"
            )
        input_sha = _sha(task["input_sha256"], f"{label}.input_sha256")
        if input_sha in seen_inputs:
            raise StageTaskSetContractError(f"{label} duplicates another task's scientific input")
        seen_ids.add(task_id)
        seen_inputs.add(input_sha)

    if _sha(record["task_set_sha256"], "task_set_sha256") != task_set_sha256(record):
        raise StageTaskSetContractError("task_set_sha256 does not match the canonical task-set content")


# --- immutable record ------------------------------------------------------------------


@dataclass(frozen=True)
class StageTaskSet:
    """Immutable, validated ``bmd_compute.stage_task_set`` v1 record.

    Held as canonical JSON text, so no caller can mutate it; every accessor
    returns fresh objects.
    """

    _canonical: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self._canonical) is not str:
            raise StageTaskSetContractError("StageTaskSet holds canonical JSON text")
        try:
            data = json.loads(self._canonical)
        except ValueError as exc:
            raise StageTaskSetContractError(f"task set is not valid JSON: {exc}") from None
        validate_stage_task_set_record(data)
        if canonical_json(data) != self._canonical:
            raise StageTaskSetContractError("StageTaskSet text is not in canonical form")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageTaskSet":
        if not isinstance(data, Mapping):
            raise StageTaskSetContractError("task set must be an object")
        validate_stage_task_set_record(data)
        return cls(canonical_json(dict(data)))

    @classmethod
    def from_json(cls, text: str) -> "StageTaskSet":
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise StageTaskSetContractError(f"task set is not valid JSON: {exc}") from None
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def to_json(self) -> str:
        return self._canonical

    @property
    def task_set_sha256(self) -> str:
        return self.to_dict()["task_set_sha256"]

    @property
    def parent(self) -> dict[str, Any]:
        return self.to_dict()["parent"]

    @property
    def source(self) -> dict[str, Any]:
        return self.to_dict()["source"]

    @property
    def upstream_structure(self) -> dict[str, Any]:
        return self.to_dict()["upstream_structure"]

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Task IDs in canonical association order (not an execution order)."""

        return tuple(task["task_id"] for task in self.to_dict()["tasks"])

    @property
    def task_count(self) -> int:
        return len(self.to_dict()["tasks"])


# --- re-verification against the parent ------------------------------------------------


def verify_stage_task_set_parent(
    task_set: StageTaskSet,
    workflow: WorkflowSpec,
    *,
    submission_attempt_id: str,
) -> None:
    """Require ``task_set`` to belong to stage ``parent.stage_index`` of ``workflow``.

    Checks the attempt, the workflow hash, the stage type and methodology hash
    of the parent stage, and that the upstream stage exists, so a task set
    cannot be reused for another attempt, workflow or stage.
    """

    if not isinstance(task_set, StageTaskSet):
        raise StageTaskSetContractError("task_set must be a StageTaskSet")
    if not isinstance(workflow, WorkflowSpec):
        raise StageTaskSetContractError("workflow must be a WorkflowSpec")
    parent = task_set.parent
    if parent["submission_attempt_id"] != _attempt_id(submission_attempt_id):
        raise StageTaskSetContractError("task set belongs to a different submission attempt")
    if parent["workflow_sha256"] != workflow_spec_sha256(workflow):
        raise StageTaskSetContractError("task set belongs to a different workflow")
    stage = _workflow_stage(workflow, parent["stage_index"])
    if stage.stage_type.value != parent["stage_type"]:
        raise StageTaskSetContractError(
            f"workflow stage {parent['stage_index']} is not a {parent['stage_type']!r} stage"
        )
    if stage_spec_sha256(stage) != parent["stage_sha256"]:
        raise StageTaskSetContractError(
            f"workflow stage {parent['stage_index']} methodology differs from the task set's parent"
        )
    _workflow_stage(workflow, task_set.upstream_structure["stage_index"])


# --- task directories ------------------------------------------------------------------


def validate_task_id(value: Any) -> str:
    if type(value) is not str or not TASK_ID_PATTERN.fullmatch(value):
        raise StageTaskSetContractError(f"invalid task ID {value!r}")
    return value


def stage_task_relative_dir(task_id: str) -> str:
    """``tasks/<task_id>``, relative to the parent stage's directory."""

    return f"{STAGE_TASKS_DIRNAME}/{validate_task_id(task_id)}"


def stage_task_dir(stage_root: str, task_id: str) -> str:
    """Directory of ``task_id`` below ``stage_root`` (a stage directory).

    Only the stage root comes from the caller, and it must be a normalized
    absolute path; the rest is derived from the validated task ID, so the
    result is always ``<stage_root>/tasks/<task_id>``.
    """

    if (
        type(stage_root) is not str
        or not _STAGE_ROOT_PATTERN.fullmatch(stage_root)
        or posixpath.normpath(stage_root) != stage_root
        # normpath keeps a leading "//" (implementation-defined in POSIX).
        or "//" in stage_root
        or stage_root == "/"
    ):
        raise StageTaskSetContractError(f"invalid stage directory {stage_root!r}")
    tasks_root = posixpath.join(stage_root, STAGE_TASKS_DIRNAME)
    path = posixpath.join(stage_root, stage_task_relative_dir(task_id))
    if posixpath.dirname(path) != tasks_root:
        raise StageTaskSetContractError(f"task directory escapes {tasks_root}")
    return path


def stage_task_dirs(task_set: StageTaskSet, stage_root: str) -> dict[str, str]:
    """Task ID -> task directory for every task, in canonical order."""

    return {task_id: stage_task_dir(stage_root, task_id) for task_id in task_set.task_ids}


# --- helpers ---------------------------------------------------------------------------


def _canonical_sha256(value: Any, label: str) -> str:
    try:
        return canonical_sha256(value)
    except PhononPlanContractError as exc:
        raise StageTaskSetContractError(f"{label} is not canonical data: {exc}") from None


def _workflow_stage(workflow: WorkflowSpec, stage_index: Any) -> StageSpec:
    index = _positive_int(stage_index, "stage_index")
    if index > len(workflow.stages):
        raise StageTaskSetContractError(
            f"stage_index {index} is outside a {len(workflow.stages)}-stage workflow"
        )
    return workflow.stages[index - 1]


def _stage_type(value: Any) -> StageType:
    if type(value) is not str:
        raise StageTaskSetContractError("parent.stage_type must be a string")
    try:
        stage_type = StageType(value)
    except ValueError:
        raise StageTaskSetContractError(f"unknown parent.stage_type {value!r}") from None
    return stage_type


def _attempt_id(value: Any) -> str:
    if type(value) is not str:
        raise StageTaskSetContractError("submission_attempt_id must be a string")
    try:
        canonical = str(uuid.UUID(value))
    except ValueError:
        raise StageTaskSetContractError(f"invalid submission_attempt_id {value!r}") from None
    if canonical != value:
        raise StageTaskSetContractError("submission_attempt_id must be a canonical lower-case UUID")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageTaskSetContractError(f"{label} must be an object")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected, label: str) -> None:
    keys = set(mapping)
    missing = sorted(set(expected) - keys)
    extra = sorted(str(key) for key in keys - set(expected))
    if missing or extra:
        raise StageTaskSetContractError(f"{label} keys: missing {missing}, unexpected {extra}")


def _int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise StageTaskSetContractError(f"{label} must be an integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if _int(value, label) < 1:
        raise StageTaskSetContractError(f"{label} must be at least 1")
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or not SHA256_PATTERN.fullmatch(value):
        raise StageTaskSetContractError(f"{label} must be a lower-case SHA-256 hex digest")
    return value


__all__ = [
    "STAGE_TASKS_DIRNAME",
    "STAGE_TASK_SET_SCHEMA",
    "STAGE_TASK_SET_SCHEMA_VERSION",
    "TASK_ORDER_SOURCE_CANONICAL",
    "StageTaskSet",
    "StageTaskSetContractError",
    "build_stage_task_set_record",
    "stage_spec_sha256",
    "stage_task_dir",
    "stage_task_dirs",
    "stage_task_relative_dir",
    "task_set_sha256",
    "validate_stage_task_set_record",
    "validate_task_id",
    "verify_stage_task_set_parent",
    "workflow_spec_sha256",
]
