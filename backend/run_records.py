"""Versioned contracts for the run records BMD Compute writes on POWER.

BMD Compute writes two JSON records that later readers, including BMD Agent
and BMD Compute's own Resume path, use to locate and describe a run:

``bmd_compute.submission`` v1 -- ``<run_dir>/submission.json``
    The Prepare-time submission specification: what BMD Compute prepared and
    asked the cluster to execute. It is written during Prepare, before sbatch.
    A later Prepare of the same run can currently rewrite it, so it is not yet
    immutable historical evidence.

``bmd_compute.job_record`` v1 -- ``<logs_dir>/job_<JOB_ID>.json``
    A run-resolution record written once, after sbatch returns a job ID. It
    links the scheduler job ID to the run directory and submission attempt.

Only the fields validated here are contractual. Both records may contain
other fields; those are BMD Compute internals and may change or disappear
without a schema change. Record ``status``/``state`` fields (for example
``status``, ``submission.submitted`` or ``submission.ready``) describe BMD
Compute's own preparation or submission step at write time. They are not
scheduler lifecycle authority: SLURM accounting and the VASP artifacts remain
authoritative for what actually executed.

Additive optional fields keep ``schema_version`` 1. Removing or changing the
meaning of a contractual field requires a new ``schema_version``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


SUBMISSION_RECORD_SCHEMA = "bmd_compute.submission"
SUBMISSION_RECORD_SCHEMA_VERSION = 1
JOB_RECORD_SCHEMA = "bmd_compute.job_record"
JOB_RECORD_SCHEMA_VERSION = 1

# ``paths.stage_dirs`` keys are explicit stage identifiers. ``stage_NN`` and
# ``relax_NN`` (used by the PBE double-relaxation workflow) both name the
# 1-based index of the stage in ``flow_spec.workflow_spec.stages``. JSON
# object key order is not contractual.
STAGE_DIRECTORY_ID_PATTERN = re.compile(r"^(?:stage|relax)_(\d{2,})$")

OPTIONAL_LOG_PATH_KEYS = ("log_out", "log_err", "slurm_out", "slurm_err")
OPTIONAL_ENVIRONMENT_KEYS = ("VASP_CMD", "PMG_VASP_PSP_DIR", "JOBFLOW_CONFIG_FILE")


class RunRecordContractError(ValueError):
    """Raised when a record does not satisfy its declared contract."""


def stage_index_from_directory_id(identifier: Any) -> int | None:
    """Return the 1-based stage index named by a stage-directory identifier."""

    if not isinstance(identifier, str):
        return None
    match = STAGE_DIRECTORY_ID_PATTERN.match(identifier)
    if match is None:
        return None
    index = int(match.group(1))
    return index if index >= 1 else None


def submission_record_header() -> dict[str, Any]:
    return {
        "schema": SUBMISSION_RECORD_SCHEMA,
        "schema_version": SUBMISSION_RECORD_SCHEMA_VERSION,
    }


def job_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``bmd_compute.job_record`` v1 document for a JobRecord dict.

    The contractual header and ``attempt_id`` come first. The remaining
    JobRecord fields are still written for BMD Compute's own Resume path but
    are not contractual.
    """

    submission_spec = record.get("submission_spec") or {}
    submission = submission_spec.get("submission") if isinstance(submission_spec, Mapping) else None
    attempt_id = submission.get("attempt_id") if isinstance(submission, Mapping) else None
    payload = {
        "schema": JOB_RECORD_SCHEMA,
        "schema_version": JOB_RECORD_SCHEMA_VERSION,
        "attempt_id": attempt_id,
    }
    payload.update(record)
    return payload


def validate_submission_record_v1(payload: Any) -> None:
    """Raise RunRecordContractError unless ``payload`` meets submission v1."""

    record = _mapping(payload, "submission record")
    _header(record, SUBMISSION_RECORD_SCHEMA, SUBMISSION_RECORD_SCHEMA_VERSION)
    _text(record, "run_name")
    _text(record, "created_at")

    submission = _mapping(record.get("submission"), "submission")
    _text(submission, "attempt_id", "submission.attempt_id")
    _text(submission, "attempt_state", "submission.attempt_state")

    flow_spec = _mapping(record.get("flow_spec"), "flow_spec")
    workflow_spec = _mapping(flow_spec.get("workflow_spec"), "flow_spec.workflow_spec")
    stages = workflow_spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise RunRecordContractError("flow_spec.workflow_spec.stages must be a non-empty list")
    for index, stage in enumerate(stages, start=1):
        label = f"flow_spec.workflow_spec.stages[{index}]"
        stage = _mapping(stage, label)
        _text(stage, "stage_type", f"{label}.stage_type")
        _text(stage, "theory", f"{label}.theory")
        modifiers = stage.get("modifiers")
        if not isinstance(modifiers, list) or not all(isinstance(item, str) and item for item in modifiers):
            raise RunRecordContractError(f"{label}.modifiers must be a list of strings")
        _mapping(stage.get("options"), f"{label}.options")
        if stage.get("label") is not None and not isinstance(stage.get("label"), str):
            raise RunRecordContractError(f"{label}.label must be a string or null")
    if "structure" in flow_spec and not isinstance(flow_spec["structure"], Mapping):
        raise RunRecordContractError("flow_spec.structure must be an object when present")

    paths = _mapping(record.get("paths"), "paths")
    _text(paths, "run_dir", "paths.run_dir")
    _text(paths, "result_dir", "paths.result_dir")
    validate_stage_directories(paths.get("stage_dirs"), len(stages))
    for key in OPTIONAL_LOG_PATH_KEYS:
        if key in paths:
            _text(paths, key, f"paths.{key}")

    cluster = _mapping(record.get("cluster"), "cluster")
    _text(cluster, "partition", "cluster.partition")
    _text(cluster, "account", "cluster.account")
    resources = _mapping(record.get("resources"), "resources")
    for key in ("nodes", "ntasks", "mem_gb"):
        if not _strict_int(resources.get(key)) or resources[key] < 1:
            raise RunRecordContractError(f"resources.{key} must be a positive integer")
    _text(resources, "walltime", "resources.walltime")

    if "environment" in record:
        environment = _mapping(record["environment"], "environment")
        for key in OPTIONAL_ENVIRONMENT_KEYS:
            if key in environment:
                _text(environment, key, f"environment.{key}")

    if "provenance" in record:
        provenance = _mapping(record["provenance"], "provenance")
        if not _strict_int(provenance.get("schema_version")):
            raise RunRecordContractError("provenance.schema_version must be an integer")


def validate_stage_directories(stage_dirs: Any, stage_count: int) -> dict[int, str]:
    """Map ``paths.stage_dirs`` identifiers to stage indices, strictly."""

    stage_dirs = _mapping(stage_dirs, "paths.stage_dirs")
    mapped: dict[int, str] = {}
    for identifier, path in stage_dirs.items():
        index = stage_index_from_directory_id(identifier)
        if index is None or index > stage_count:
            raise RunRecordContractError(
                f"paths.stage_dirs identifier {identifier!r} does not name a workflow stage"
            )
        if index in mapped:
            raise RunRecordContractError(f"paths.stage_dirs names stage {index} more than once")
        if not isinstance(path, str) or not path:
            raise RunRecordContractError(f"paths.stage_dirs[{identifier!r}] must be a non-empty string")
        mapped[index] = path
    if stage_count > 1 and set(mapped) != set(range(1, stage_count + 1)):
        raise RunRecordContractError("paths.stage_dirs must name every stage of a multi-stage workflow")
    return mapped


def validate_job_record_v1(payload: Any) -> None:
    """Raise RunRecordContractError unless ``payload`` meets job_record v1."""

    record = _mapping(payload, "job record")
    _header(record, JOB_RECORD_SCHEMA, JOB_RECORD_SCHEMA_VERSION)
    for key in ("job_id", "run_name", "run_dir", "attempt_id"):
        _text(record, key)
    if "submitted_at" in record and record["submitted_at"] is not None:
        _text(record, "submitted_at")


def submission_contract_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the contractual submission v1 fields (for drift checks)."""

    paths = payload["paths"]
    projection = {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "run_name": payload["run_name"],
        "created_at": payload["created_at"],
        "submission": {
            "attempt_id": payload["submission"]["attempt_id"],
            "attempt_state": payload["submission"]["attempt_state"],
        },
        "workflow_spec_stages": payload["flow_spec"]["workflow_spec"]["stages"],
        "paths": {
            "run_dir": paths["run_dir"],
            "result_dir": paths["result_dir"],
            "stage_dirs": dict(sorted(paths["stage_dirs"].items())),
            **{key: paths[key] for key in OPTIONAL_LOG_PATH_KEYS if key in paths},
        },
        "cluster": {key: payload["cluster"][key] for key in ("partition", "account")},
        "resources": {
            key: payload["resources"][key] for key in ("nodes", "ntasks", "mem_gb", "walltime")
        },
    }
    return projection


def job_record_contract_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("schema", "schema_version", "job_id", "run_name", "run_dir", "attempt_id")
    }


def _header(record: Mapping[str, Any], schema: str, version: int) -> None:
    if record.get("schema") != schema:
        raise RunRecordContractError(f"schema must be {schema!r}")
    if not _strict_int(record.get("schema_version")) or record["schema_version"] != version:
        raise RunRecordContractError(f"schema_version must be {version}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunRecordContractError(f"{label} must be an object")
    return value


def _text(record: Mapping[str, Any], key: str, label: str | None = None) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise RunRecordContractError(f"{label or key} must be a non-empty string")
    return value


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "JOB_RECORD_SCHEMA",
    "JOB_RECORD_SCHEMA_VERSION",
    "RunRecordContractError",
    "STAGE_DIRECTORY_ID_PATTERN",
    "SUBMISSION_RECORD_SCHEMA",
    "SUBMISSION_RECORD_SCHEMA_VERSION",
    "job_record_contract_projection",
    "job_record_payload",
    "stage_index_from_directory_id",
    "submission_contract_projection",
    "submission_record_header",
    "validate_job_record_v1",
    "validate_stage_directories",
    "validate_submission_record_v1",
]
