from __future__ import annotations

import platform
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from backend.calculations.custodian_policy import resolved_custodian_policy
from backend.workflows import (
    vasp_command_for_modifiers,
    vasp_executable_for_modifiers,
    vasp_job_kwargs_for_modifiers,
)
from backend.runtime_package import runtime_package_manifest_metadata


PROVENANCE_SCHEMA_VERSION = 1
SCIENTIFIC_PACKAGE_NAMES = ("pymatgen", "pymatgen-core", "atomate2", "jobflow", "custodian")


def build_submission_provenance(submission_spec: dict) -> dict:
    """
    Build a JSON-safe provenance snapshot for a prepared submission.

    The snapshot describes the BMD Compute source and preparation environment
    plus the execution policy that will be used remotely. It does not attempt
    to replace the authoritative VASP input/output files.
    """

    workflow_spec = _workflow_spec_dict(submission_spec)
    resources = dict(submission_spec.get("resources") or {})
    cluster = dict(submission_spec.get("cluster") or {})
    environment = dict(submission_spec.get("environment") or {})
    runner = dict(submission_spec.get("runner") or {})
    modules = dict(submission_spec.get("modules") or {})
    potcar = dict(submission_spec.get("potcar") or {})

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "bmd_compute": {
            "source": source_metadata(),
            "runtime_source": runtime_package_manifest_metadata(),
        },
        "python_environment": {
            "preparation": preparation_python_environment(),
            "remote_execution": {
                "status": "deferred_until_execution",
                "python": _json_safe_scalar(runner.get("python")),
                "packages": {
                    name: "captured by the remote runner at execution time"
                    for name in SCIENTIFIC_PACKAGE_NAMES
                },
            },
        },
        "vasp": {
            "global_command_template": _json_safe_scalar(environment.get("VASP_CMD")),
            "stages": stage_vasp_provenance(workflow_spec, environment),
            "version": {
                "status": "deferred_until_execution",
                "reason": "VASP version/build is only known on the compute node.",
            },
        },
        "potcar": {
            "functional": _json_safe_scalar(potcar.get("functional")),
            # Taken from the executable stage generators; ``species``/``symbols``
            # are null when stages differ, in which case ``stages`` is the record.
            "status": _json_safe_scalar(potcar.get("status")),
            "species": _json_safe_value(potcar.get("species")),
            "symbols": _json_safe_value(potcar.get("symbols")),
            "symbol_source": _json_safe_scalar(potcar.get("symbol_source")),
            "consistent_across_stages": _json_safe_scalar(
                potcar.get("consistent_across_stages")
            ),
            "stages": _json_safe_value(potcar.get("stages") or []),
            **(
                {"reason": _json_safe_scalar(potcar.get("reason"))}
                if potcar.get("reason")
                else {}
            ),
            "repository": _json_safe_scalar(potcar.get("repository")),
            "hashes": {
                "status": "not_recorded",
                "reason": (
                    "Actual execution POTCAR files are resolved from the configured "
                    "remote/shared repository; raw POTCAR contents are not stored."
                ),
            },
        },
        "execution": {
            "workflow_spec": _json_safe_value(workflow_spec),
            "automatic_treatments": _automatic_treatments_dict(submission_spec),
            "stage_order": stage_order_provenance(workflow_spec),
            "custodian": stage_custodian_provenance(workflow_spec),
            "resources": _json_safe_value(resources),
            "cluster": {
                "partition": _json_safe_scalar(cluster.get("partition")),
                "account": _json_safe_scalar(cluster.get("account")),
            },
            "modules": {
                "purge_first": bool(modules.get("purge_first", False)),
                "load": _json_safe_value(modules.get("load") or []),
                "loaded_module_information": {
                    "status": "deferred_until_execution",
                    "reason": "The active module list is emitted by the remote job.",
                },
            },
            "environment_policy": {
                "VASP_CMD": _json_safe_scalar(environment.get("VASP_CMD")),
                "JOBFLOW_CONFIG_FILE": _json_safe_scalar(environment.get("JOBFLOW_CONFIG_FILE")),
                "PMG_VASP_PSP_DIR": _json_safe_scalar(environment.get("PMG_VASP_PSP_DIR")),
                "CUSTODIAN_NO_GZIP": _json_safe_scalar(environment.get("CUSTODIAN_NO_GZIP")),
                "ATOMATE2_VASP_ZIP_FILES": _json_safe_scalar(
                    environment.get("ATOMATE2_VASP_ZIP_FILES")
                ),
            },
        },
    }


@lru_cache(maxsize=1)
def source_metadata() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    commit = _git_stdout(repo_root, "rev-parse", "HEAD")
    if not commit:
        return {
            "status": "unavailable",
            "git_commit": None,
            "dirty": None,
            "state": "unavailable",
            "reason": "Git metadata is unavailable in this deployment.",
        }

    status = _git_stdout(repo_root, "status", "--short")
    if status is None:
        return {
            "status": "available",
            "git_commit": commit,
            "dirty": None,
            "state": "unknown",
            "dirty_paths_count": None,
            "reason": "Git dirty state could not be determined.",
        }

    dirty_lines = [line for line in status.splitlines() if line.strip()]
    dirty = bool(dirty_lines)
    return {
        "status": "available",
        "git_commit": commit,
        "dirty": dirty,
        "state": "dirty" if dirty else "clean",
        "dirty_paths_count": len(dirty_lines),
    }


def preparation_python_environment() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {
            name: _package_version(name)
            for name in SCIENTIFIC_PACKAGE_NAMES
        },
    }


def stage_vasp_provenance(workflow_spec: dict, environment: dict) -> list[dict]:
    stages = workflow_spec.get("stages") if isinstance(workflow_spec, dict) else []
    if not isinstance(stages, list):
        return []

    base_command = environment.get("VASP_CMD")
    payload = []
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            continue
        modifiers = stage.get("modifiers") or []
        payload.append({
            "index": index,
            "stage_type": _json_safe_scalar(stage.get("stage_type")),
            "theory": _json_safe_scalar(stage.get("theory")),
            "modifiers": _json_safe_value(modifiers),
            "options": _json_safe_value(stage.get("options") or {}),
            "executable": vasp_executable_for_modifiers(modifiers),
            "command_template": vasp_command_for_modifiers(
                modifiers,
                base_command=base_command,
            ),
            "custodian_vasp_job_kwargs": _json_safe_value(
                vasp_job_kwargs_for_modifiers(modifiers)
            ),
        })
    return payload


def stage_order_provenance(workflow_spec: dict) -> list[dict]:
    stages = workflow_spec.get("stages") if isinstance(workflow_spec, dict) else []
    if not isinstance(stages, list):
        return []
    return [
        {
            "index": index,
            "stage_type": _json_safe_scalar(stage.get("stage_type")),
            "theory": _json_safe_scalar(stage.get("theory")),
            "modifiers": _json_safe_value(stage.get("modifiers") or []),
            "options": _json_safe_value(stage.get("options") or {}),
        }
        for index, stage in enumerate(stages, start=1)
        if isinstance(stage, dict)
    ]


def stage_custodian_provenance(workflow_spec: dict) -> dict:
    stages = workflow_spec.get("stages") if isinstance(workflow_spec, dict) else []
    if not isinstance(stages, list):
        stages = []

    return {
        "stages": [
            {
                "index": index,
                **resolved_custodian_policy(
                    stage.get("stage_type"),
                    stage.get("theory"),
                ),
            }
            for index, stage in enumerate(stages, start=1)
            if isinstance(stage, dict)
        ]
    }


def _automatic_treatments_dict(submission_spec: dict) -> dict | None:
    flow_spec = submission_spec.get("flow_spec") or {}
    treatments = flow_spec.get("automatic_treatments")
    return _json_safe_value(treatments) if isinstance(treatments, dict) else None


def _workflow_spec_dict(submission_spec: dict) -> dict:
    flow_spec = submission_spec.get("flow_spec") or {}
    workflow_spec = flow_spec.get("workflow_spec")
    return _json_safe_value(workflow_spec) if isinstance(workflow_spec, dict) else {}


def _git_stdout(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_root.as_posix()}",
                "-C",
                str(repo_root),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _package_version(package_name: str) -> str:
    try:
        return str(metadata.version(package_name))
    except metadata.PackageNotFoundError:
        return "unavailable"


def _json_safe_value(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    return _json_safe_scalar(value)


def _json_safe_scalar(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "SCIENTIFIC_PACKAGE_NAMES",
    "build_submission_provenance",
    "preparation_python_environment",
    "source_metadata",
    "stage_custodian_provenance",
    "stage_order_provenance",
    "stage_vasp_provenance",
]
