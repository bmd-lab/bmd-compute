"""Bounded JSON projection of a resolved calculation plan.

Only fields listed here leave the service. In particular the projection never
includes the submission identity token, attempt ID or fingerprint, SSH
connection profile, remote paths, environment paths, runner configuration,
generated SLURM scripts or provenance blocks.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

from backend.calculations.capabilities import SCHEMA_VERSION as CAPABILITY_SCHEMA_VERSION
from backend.generated_inputs import generated_input_stage_previews

from compute_api import API_VERSION
from compute_api.canonical import canonical_json, json_safe, sha256_hex
from compute_api.plan_digest import (
    PLAN_DIGEST_VERSION,
    canonical_structure,
    compute_plan_digest,
    policy_versions,
    potcar_symbols,
)


PLAN_RESPONSE_SCHEMA = "bmd_compute.api.plan"
PLAN_RESPONSE_SCHEMA_VERSION = 1
PLAN_REQUEST_SCHEMA_VERSION = 1
MAX_PLAN_RESPONSE_CHARS = 4 * 1024 * 1024

_CONSIDERATION_FIELDS = (
    "id",
    "method",
    "modifier",
    "status",
    "selection_state",
    "automatic_application_state",
    "trigger_elements",
    "trigger_classes",
    "applicable_stage_types",
    "policy_source",
)
_STRUCTURE_SUMMARY_FIELDS = (
    "formula",
    "reduced_formula",
    "natoms",
    "space_group_symbol",
    "space_group_number",
    "crystal_system",
    "volume",
)


class PlanProjectionError(RuntimeError):
    """The resolved plan could not be projected safely."""


def plan_stage_previews(plan) -> tuple[dict[str, Any], ...]:
    flow_spec = plan.submission_spec["flow_spec"]
    if json_safe(plan.workflow_spec.to_dict()) != json_safe(flow_spec["workflow_spec"]):
        raise PlanProjectionError("resolved workflow does not match the submission specification")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generated_input_stage_previews(
            plan.structure,
            plan.workflow_spec,
            resources=plan.execution_resources,
            potcar_functional=flow_spec["potcar_functional"],
        )


def _kpoints_summary(kpoints) -> dict[str, Any] | None:
    if kpoints is None:
        return None
    text = str(kpoints).rstrip()
    num_kpts = int(getattr(kpoints, "num_kpts", 0) or 0)
    style = getattr(kpoints, "style", None)
    summary = {
        "style": getattr(style, "name", str(style)),
        "num_kpts": num_kpts,
        "sha256": sha256_hex(text),
    }
    if num_kpts == 0:
        # Automatic meshes: the subdivisions and shift are small and useful.
        summary["kpts"] = json_safe(kpoints.kpts)
        summary["kpts_shift"] = json_safe(getattr(kpoints, "kpts_shift", None))
    return summary


def _stage_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    stage = preview["stage_spec"]
    input_set = preview["input_set"]
    incar = json_safe(dict(input_set.incar))
    poscar_text = str(input_set.poscar).rstrip()
    return {
        "index": preview["index"],
        "stage_type": stage.stage_type.value,
        "theory": stage.theory.value,
        "modifiers": sorted(modifier.value for modifier in stage.modifiers),
        "options": json_safe(stage.options),
        "vasp_executable": preview["vasp_executable"],
        "incar": {
            "settings": incar,
            "sha256": sha256_hex(str(input_set.incar).rstrip()),
        },
        "kpoints": _kpoints_summary(input_set.kpoints),
        "potcar_symbols": potcar_symbols(input_set),
        "poscar": {
            "num_sites": len(input_set.poscar.structure),
            "reduced_formula": str(input_set.poscar.structure.composition.reduced_formula),
            "sha256": sha256_hex(poscar_text),
        },
    }


def _method_considerations_projection(context: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not context:
        return None
    considerations = []
    for consideration in context.get("considerations") or []:
        item = {field: json_safe(consideration.get(field)) for field in _CONSIDERATION_FIELDS}
        application = consideration.get("automatic_application") or {}
        item["automatic_stage_indices"] = json_safe(application.get("stage_indices") or [])
        considerations.append(item)
    return {
        "policy_version": json_safe(context.get("policy_version")),
        "considerations": considerations,
    }


def _resources_projection(submission_spec: Mapping[str, Any]) -> dict[str, Any]:
    resources = submission_spec["resources"]
    cluster = submission_spec["cluster"]
    return {
        "nodes": resources["nodes"],
        "cpus": resources["ntasks"],
        "memory_gb": resources["mem_gb"],
        "walltime": resources["walltime"],
        "partition": cluster["partition"],
        "account": cluster["account"],
    }


def plan_response(plan, request) -> dict[str, Any]:
    stage_previews = plan_stage_previews(plan)
    submission_spec = plan.submission_spec
    flow_spec = submission_spec["flow_spec"]
    workflow = flow_spec["workflow_spec"]
    structure_summary = {
        field: json_safe(plan.summary.get(field)) for field in _STRUCTURE_SUMMARY_FIELDS
    }
    structure_summary["canonical_sha256"] = sha256_hex(canonical_json(canonical_structure(plan.structure)))
    runtime_parity = submission_spec.get("runtime_parity") or {}

    response = {
        "schema": PLAN_RESPONSE_SCHEMA,
        "schema_version": PLAN_RESPONSE_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "plan_digest": compute_plan_digest(
            structure=plan.structure,
            submission_spec=submission_spec,
            stage_previews=stage_previews,
        ),
        "plan_digest_version": PLAN_DIGEST_VERSION,
        "request": {
            "structure_format": request.structure_format,
            "workflow_mode": request.workflow_mode,
            "desired_output": request.desired_output,
        },
        "structure": structure_summary,
        "workflow": {
            "recipe": json_safe(workflow.get("recipe")),
            "stage_count": len(workflow["stages"]),
            "stages": [
                {
                    "index": index,
                    "stage_type": stage["stage_type"],
                    "theory": stage["theory"],
                    "modifiers": json_safe(stage["modifiers"]),
                    "options": json_safe(stage["options"]),
                    "vasp_executable": stage_previews[index - 1]["vasp_executable"],
                }
                for index, stage in enumerate(workflow["stages"], start=1)
            ],
        },
        "automatic_treatments": json_safe(flow_spec.get("automatic_treatments")),
        "method_considerations": _method_considerations_projection(plan.method_considerations),
        "resources": _resources_projection(submission_spec),
        "scientific_inputs": {
            "potcar_functional": flow_spec["potcar_functional"],
            "stages": [_stage_projection(preview) for preview in stage_previews],
        },
        "software": {
            "modules": json_safe((submission_spec.get("modules") or {}).get("load") or []),
            "runtime_parity_packages": json_safe(runtime_parity.get("packages") or {}),
        },
        "policies": policy_versions(),
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
    }
    if len(canonical_json(response)) > MAX_PLAN_RESPONSE_CHARS:
        raise PlanProjectionError("plan response exceeds the size limit")
    return response
