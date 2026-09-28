from __future__ import annotations

import hashlib
import json
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from backend.calculations.capabilities import (
    REPOSITORY_ID,
    git_provenance,
    unavailable_git_provenance,
)
from backend.calculations.models import (
    CalculationSpec,
    StageSpec,
    WorkflowSpec,
)
from backend.calculations.registry import (
    CalculationValidationError,
    calculation_spec_from_workflow_spec,
    modifier_display_name,
    stage_display_name,
    theory_display_name,
    validate_workflow_spec,
    workflow_display_name,
    workflow_stage_directories,
)
from backend.calculations.resources import (
    normalize_execution_resources,
)
from backend.calculations.vasp_stage_definitions import describe_stage
from backend.calculations.hse_band_kpoints import (
    FULL_ZONE_KPOINTS_ADJUSTMENT,
    uses_full_zone_weighted_kpoints,
)
from backend.generated_inputs import generated_input_stage_previews
from backend.parser import StructureValidationError, structure_from_spec


SCHEMA_VERSION = 1
# ``SCOPE`` and the contract ``scope`` are human-readable descriptions.
# Consumers must establish compatibility from ``schema_version``,
# ``producer.repository`` and ``reference_phase``, not from this text.
SCOPE = "BMD Compute generated pre-execution VASP input reference"
REFERENCE_PHASE = "generated_pre_execution"


class InputReferenceRequestError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "malformed_request",
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.suggestion = suggestion


def build_input_reference_payload(
    request: Mapping[str, Any],
    *,
    include_provenance: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-safe reference for BMD Compute generated VASP inputs.

    The reference is the pre-execution input that BMD Compute would generate for
    the supplied structure and workflow specification. It is not a Custodian- or
    VASP-mutated runtime record, and it never contains raw POTCAR contents.
    """

    try:
        parsed = _parse_request(request)
        stage_previews = _build_stage_previews(parsed)
    except CalculationValidationError as exc:
        return _unsupported_payload(
            request,
            message=exc.message,
            suggestion=exc.suggestion,
            include_provenance=include_provenance,
            repo_root=repo_root,
        )
    except (InputReferenceRequestError, StructureValidationError) as exc:
        suggestion = getattr(exc, "suggestion", None)
        code = getattr(exc, "code", "invalid_structure")
        return _error_payload(
            request,
            code=code,
            message=str(exc),
            suggestion=suggestion,
            include_provenance=include_provenance,
            repo_root=repo_root,
        )
    except Exception as exc:
        return _error_payload(
            request,
            code="reference_generation_failed",
            message=str(exc) or exc.__class__.__name__,
            suggestion="Check that the request uses supported BMD Compute structure and workflow fields.",
            include_provenance=include_provenance,
            repo_root=repo_root,
        )

    workflow_spec = parsed["workflow_spec"]
    resources = parsed["resources"]
    potcar_functional = parsed["potcar_functional"]

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "status": "ok",
        "reference_phase": REFERENCE_PHASE,
        "producer": _producer_payload(
            include_provenance=include_provenance,
            repo_root=repo_root,
        ),
        "contract": _contract_payload(),
        "request": {
            "structure_type": parsed["structure_type"],
            "workflow_spec": workflow_spec.to_dict(),
            "resources": _resource_reference_payload(resources),
            "potcar_functional": potcar_functional,
        },
        "workflow": _workflow_payload(workflow_spec),
        "reference": {
            "stages": [
                _stage_reference_payload(stage_preview)
                for stage_preview in stage_previews
            ],
        },
    }


def emit_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_safe_value(payload),
        indent=2,
        sort_keys=True,
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    if args:
        sys.stdout.write(
            emit_json(
                _error_payload(
                    {},
                    code="invalid_cli_arguments",
                    message="backend.calculations.input_reference reads one JSON request from stdin and accepts no arguments.",
                    suggestion="Pass the structured input-reference request on stdin.",
                )
            )
        )
        return 2

    try:
        raw_request = sys.stdin.read()
        request = _loads_request(raw_request)
        payload = build_input_reference_payload(request)
    except InputReferenceRequestError as exc:
        payload = _error_payload(
            {},
            code=exc.code,
            message=str(exc),
            suggestion=exc.suggestion,
        )
        sys.stdout.write(emit_json(payload))
        return 2

    sys.stdout.write(emit_json(payload))
    if payload.get("status") == "error":
        return 2
    return 0


def _parse_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise InputReferenceRequestError("The input-reference request must be a JSON object.")

    structure_spec = request.get("structure")
    if not isinstance(structure_spec, Mapping):
        raise InputReferenceRequestError(
            "The request must include a structure object.",
            suggestion="Use the existing BMD Compute structure specification, normally type=pasted_text with POSCAR or CIF text.",
        )

    structure_type = str(structure_spec.get("type") or "").strip()
    if structure_type in {"mp", "path"}:
        raise InputReferenceRequestError(
            f"Structure type {structure_type!r} is not supported by the read-only input-reference contract.",
            code="unsupported_structure_source",
            suggestion="Send the actual structure as pasted POSCAR/CIF text or an explicit builder structure.",
        )

    workflow_spec_payload = request.get("workflow_spec") or request.get("workflow")
    if not isinstance(workflow_spec_payload, Mapping):
        raise InputReferenceRequestError(
            "The request must include a workflow_spec object.",
            suggestion="Send the existing BMD Compute WorkflowSpec dictionary with an ordered stages list.",
        )

    structure = structure_from_spec(dict(structure_spec))
    workflow_spec = validate_workflow_spec(
        WorkflowSpec.from_dict(dict(workflow_spec_payload))
    )
    resources = normalize_execution_resources(request.get("resources"))
    potcar_functional = str(request.get("potcar_functional") or "PBE_64").strip() or "PBE_64"

    return {
        "structure": structure,
        "structure_type": structure_type,
        "workflow_spec": workflow_spec,
        "resources": resources,
        "potcar_functional": potcar_functional,
    }


def _loads_request(raw_request: str) -> Mapping[str, Any]:
    if not raw_request.strip():
        raise InputReferenceRequestError(
            "No input-reference request JSON was provided on stdin."
        )
    try:
        request = json.loads(raw_request)
    except json.JSONDecodeError as exc:
        raise InputReferenceRequestError(
            f"Invalid JSON request: {exc.msg}.",
            suggestion="Pass one valid JSON object on stdin.",
        ) from exc
    if not isinstance(request, Mapping):
        raise InputReferenceRequestError("The input-reference request must be a JSON object.")
    return request


def _build_stage_previews(parsed_request: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generated_input_stage_previews(
            parsed_request["structure"],
            parsed_request["workflow_spec"],
            resources=parsed_request["resources"],
            potcar_functional=parsed_request["potcar_functional"],
        )


def _workflow_payload(workflow_spec: WorkflowSpec) -> dict[str, Any]:
    compatible = calculation_spec_from_workflow_spec(workflow_spec)
    return {
        "label": workflow_display_name(workflow_spec),
        "recipe": workflow_spec.recipe,
        "stage_count": len(workflow_spec.stages),
        "compatible_calculation_spec": (
            compatible.to_dict()
            if isinstance(compatible, CalculationSpec)
            else None
        ),
        "stage_directories": list(workflow_stage_directories(workflow_spec)),
        "stages": [
            _workflow_stage_payload(index, stage)
            for index, stage in enumerate(workflow_spec.stages, start=1)
        ],
    }


def _workflow_stage_payload(index: int, stage: StageSpec) -> dict[str, Any]:
    return {
        "index": index,
        "stage_type": stage.stage_type.value,
        "stage_label": stage_display_name(stage),
        "theory": stage.theory.value,
        "theory_label": theory_display_name(stage.theory),
        "modifiers": sorted(modifier.value for modifier in stage.modifiers),
        "modifier_labels": [
            modifier_display_name(modifier)
            for modifier in sorted(stage.modifiers, key=lambda item: item.value)
        ],
        "options": _json_safe_value(stage.options),
    }


def _stage_reference_payload(stage_preview: Mapping[str, Any]) -> dict[str, Any]:
    input_set = stage_preview["input_set"]
    stage = stage_preview["stage_spec"]
    stage_description = describe_stage(stage.stage_type, stage.theory)
    poscar_text = _input_text(input_set.poscar)
    kpoints = input_set.kpoints
    generator = {
        "source": "backend.generated_inputs.generated_input_stage_previews",
        "selected_atomate2": stage_description["selected_atomate2"],
    }
    if uses_full_zone_weighted_kpoints(stage.stage_type, stage.theory, stage.modifiers):
        generator["bmd_kpoints_adjustment"] = FULL_ZONE_KPOINTS_ADJUSTMENT

    return {
        "index": stage_preview["index"],
        "directory": stage_preview.get("directory"),
        "stage_type": stage.stage_type.value,
        "stage_label": stage_preview["label"],
        "theory": stage.theory.value,
        "theory_label": stage_preview["theory_label"],
        "modifiers": sorted(modifier.value for modifier in stage.modifiers),
        "options": _json_safe_value(stage.options),
        "vasp_executable": stage_preview["vasp_executable"],
        "generator": generator,
        "incar": {
            "settings": _json_safe_value(dict(input_set.incar)),
            "text": _input_text(input_set.incar),
        },
        "kpoints": {
            "text": _input_text(kpoints),
            "as_dict": _json_safe_value(kpoints.as_dict() if kpoints is not None else None),
        },
        "poscar": {
            "text": poscar_text,
            "sha256": _sha256_text(poscar_text),
            "formula": str(input_set.poscar.structure.composition.formula),
            "reduced_formula": str(input_set.poscar.structure.composition.reduced_formula),
            "num_sites": len(input_set.poscar.structure),
        },
        "potcar": {
            "spec_text": _input_text(getattr(input_set, "potcar", "")),
            "symbols": _potcar_symbols(getattr(input_set, "potcar", "")),
            "contains_potcar_contents": False,
            "source": "pymatgen get_input_set(..., potcar_spec=True)",
        },
    }


def _contract_payload() -> dict[str, Any]:
    return {
        "request": "One JSON object on stdin containing structure, workflow_spec, optional resources, and optional potcar_functional.",
        "workflow_spec": "Existing BMD Compute WorkflowSpec dictionary with ordered stages.",
        "reference": "Generated through BMD Compute's existing preview/input-set path.",
        "reference_phase": REFERENCE_PHASE,
        "potcar": "Symbolic POTCAR.spec only; raw POTCAR file contents are never emitted.",
        "execution": "Does not submit jobs, call SLURM, open SSH, run Custodian, or run VASP.",
        "scope": (
            "Inputs generated by BMD Compute's executable calculation methodology, "
            "for read-only comparison; pre-execution only, not a record of what executed."
        ),
    }


def _producer_payload(
    *,
    include_provenance: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    source = (
        git_provenance(repo_root=repo_root)
        if include_provenance
        else unavailable_git_provenance()
    )
    return {
        "repository": REPOSITORY_ID,
        "source": source,
    }


def _unsupported_payload(
    request: Mapping[str, Any],
    *,
    message: str,
    suggestion: str | None,
    include_provenance: bool,
    repo_root: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "status": "unsupported",
        "reference_phase": REFERENCE_PHASE,
        "producer": _producer_payload(
            include_provenance=include_provenance,
            repo_root=repo_root,
        ),
        "contract": _contract_payload(),
        "request": _safe_request_echo(request),
        "error": {
            "code": "unsupported_combination",
            "message": message,
            "suggestion": suggestion,
        },
        "reference": None,
    }


def _error_payload(
    request: Mapping[str, Any],
    *,
    code: str,
    message: str,
    suggestion: str | None = None,
    include_provenance: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "status": "error",
        "reference_phase": REFERENCE_PHASE,
        "producer": _producer_payload(
            include_provenance=include_provenance,
            repo_root=repo_root,
        ),
        "contract": _contract_payload(),
        "request": _safe_request_echo(request),
        "error": {
            "code": code,
            "message": message,
            "suggestion": suggestion,
        },
        "reference": None,
    }


def _safe_request_echo(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        return {}
    return {
        "structure_type": (
            str((request.get("structure") or {}).get("type"))
            if isinstance(request.get("structure"), Mapping)
            else None
        ),
        "workflow_spec": _json_safe_value(request.get("workflow_spec")),
        "resources": _safe_resources_echo(request.get("resources")),
        "potcar_functional": _json_safe_value(request.get("potcar_functional")),
    }


def _resource_reference_payload(resources) -> dict[str, int]:
    return {
        "nodes": int(resources.nodes),
        "ntasks": int(resources.ntasks),
        "mem_gb": int(resources.mem_gb),
    }


def _safe_resources_echo(resources) -> dict[str, Any] | None:
    if not isinstance(resources, Mapping):
        return None
    return {
        key: _json_safe_value(resources.get(key))
        for key in ("nodes", "cpus", "ntasks", "memory_gb", "mem_gb")
        if key in resources
    }


def _potcar_symbols(spec_text: str) -> list[str]:
    return [
        line.strip()
        for line in str(spec_text or "").splitlines()
        if line.strip()
    ]


def _input_text(input_object) -> str:
    return str(input_object or "").rstrip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe_value(value: Any):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe_value(value.item())
    raise TypeError(f"Value of type {value.__class__.__name__} is not JSON serializable.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
