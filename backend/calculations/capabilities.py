from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.calculations import admission, geometry_convergence, geometry_relaxation
from backend.calculations.default_treatments import automatic_default_treatment_policy
from backend.calculations.dispersion import dispersion_modifier_policy
from backend.calculations.models import StageSpec, Theory
from backend.calculations.registry import (
    BROWSER_HIDDEN_DESIRED_OUTPUTS,
    PHONON_SUPPORTED_SCOPE_TEXT,
    _desired_output_workflows,
    supported_stage_modifier_combinations,
    validate_workflow_spec,
)
from backend.calculations.vasp_stage_definitions import (
    EXECUTABLE_METHODOLOGY_SCOPE,
    describe_stage,
    list_stage_definitions,
)


SCHEMA_VERSION = 1
# ``SCOPE`` is a human-readable description. Consumers must establish
# compatibility from ``schema_version`` and ``source.repository``, not from
# this text, which may be reworded without a schema change.
SCOPE = EXECUTABLE_METHODOLOGY_SCOPE
REPOSITORY_ID = "bmd_compute"

# Structure-search description. The structure-search modules (and the machine
# API router) import runtime configuration and are deliberately not imported by
# this producer, which must stay free of runtime machinery
# (tests/test_capabilities_contract.py). These values therefore restate the
# authoritative constants, and tests/test_capability_descriptions.py keeps each
# one equal to its source:
#   backend.structure_search.spec.SEARCH_DEFAULTS_POLICY_ID / _VERSION / _STATUS
#   backend.structure_search.admission.admission_scope_record()
#   compute_api.router.SEARCH_EXECUTION_ENV and its enabling value
# The producer never reads the execution setting: it describes the gate, not
# the state of any running service.
STRUCTURE_SEARCH_DEFAULTS_POLICY = {
    "policy_id": "bmd_compute.structure_search_defaults",
    "policy_version": 1,
    "status": "provisional",
}
STRUCTURE_SEARCH_ADMISSION_SCOPE = {
    "policy_id": "bmd_compute.structure_search_admission",
    "policy_version": 2,
    "calculation_type": 300,
    "theory": "pbe",
    "spin_polarized": False,
    "external_pressure_gpa": 0.0,
    "composition_screens": [
        "bmd_compute spin-polarisation screen (method_considerations.SPIN_TRIGGER_ELEMENT_CLASSES)",
        "bmd_compute heavy-element SOC screen (method_considerations.SOC_TRIGGER_ELEMENT_CLASSES)",
        "bmd_compute.dft_u policy (dft_u_policy.evaluate_dft_u)",
    ],
    "not_screened": ["DFT-D3 dispersion (depends on a structure's bonded connectivity)"],
    "limits": {
        "max_elements": 4,
        "max_cell_atoms": 40,
        "max_generations": 200,
        "status": "provisional",
    },
}
# Machine API structure-search operations, keyed by their route
# (compute_api.router; kept equal by tests/test_capability_descriptions.py).
STRUCTURE_SEARCH_MACHINE_API_OPERATIONS = (
    {"operation": "plan", "route": "POST /api/v1/structure-search-plans"},
    {"operation": "prepare", "route": "PUT /api/v1/structure-search-attempts/{attempt_id}"},
    {"operation": "stage", "route": "PUT /api/v1/structure-search-attempts/{attempt_id}/staging"},
    {"operation": "promote", "route": "PUT /api/v1/structure-search-attempts/{attempt_id}/promotion"},
    {"operation": "submit", "route": "PUT /api/v1/structure-search-attempts/{attempt_id}/submission"},
    {"operation": "record_remote_submission", "route": "PUT /api/v1/structure-search-attempts/{attempt_id}/submission/remote-record"},
    {"operation": "observe", "route": "POST /api/v1/structure-search-attempts/{attempt_id}/observations"},
    {"operation": "cancel", "route": "PUT /api/v1/structure-search-attempts/{attempt_id}/cancellation"},
    {"operation": "read", "route": "GET /api/v1/structure-search-attempts/{attempt_id}"},
)
STRUCTURE_SEARCH_EXECUTION_SETTING = "BMD_STRUCTURE_SEARCH_EXECUTION"
STRUCTURE_SEARCH_EXECUTION_ENABLED_VALUE = "enabled"


def build_capability_payload(
    *,
    include_provenance: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    base_stage_definitions = list(list_stage_definitions())
    capabilities = _supported_stage_theory_capabilities(base_stage_definitions)
    source = (
        git_provenance(repo_root=repo_root)
        if include_provenance
        else unavailable_git_provenance()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "source": source,
        "contract": {
            "base_stage_definitions": "Theory-neutral stage definitions from list_stage_definitions().",
            "capabilities": "Supported stage/theory descriptions from describe_stage(); unsupported combinations are not invented.",
            "stage_modifier_support": "Complete stage-local explicit modifier sets accepted by the BMD Compute registry for each supported stage/theory pair.",
            "modifier_policies": "Stage-local executable modifiers with controlled options; unsupported pairings are not invented.",
            "automatic_default_treatments": "Read-only policy describing BMD-managed Desired Output treatment resolution.",
            "release_identity": "Release tag and source revision of this checkout. Describes source code only; it is never evidence of what a running service executes.",
            "desired_outputs": "Every Desired Output identifier with its requested stage sequence, from the registry; structure-dependent resolution applies afterwards.",
            "new_calculation_admission": "Combinations refused for new calculations although stage validation accepts them, and the admissible modifier sets that remain, from the admission policy.",
            "protocols": "Workflow-level methodology that overrides stage-definition defaults for new calculations: the geometry relaxation protocol, the plane-wave cutoff rule and the adaptive geometry-convergence check.",
            "experimental_capabilities": "Implemented capabilities outside the stage methodology described in capabilities, with their status.",
            "validation": "What this record does and does not assert about scientific validation.",
        },
        "base_stage_definitions": base_stage_definitions,
        "capabilities": capabilities,
        "stage_modifier_support": _stage_modifier_support(capabilities),
        "modifier_policies": [dispersion_modifier_policy()],
        "automatic_default_treatments": automatic_default_treatment_policy(),
        "release_identity": release_identity(
            source,
            tag=_release_tag(repo_root=repo_root) if include_provenance else None,
        ),
        "desired_outputs": desired_output_catalogue(),
        "new_calculation_admission": new_calculation_admission_description(capabilities),
        "protocols": protocol_descriptions(),
        "experimental_capabilities": experimental_capability_descriptions(),
        "validation": validation_statement(),
    }


def desired_output_catalogue() -> list[dict[str, Any]]:
    """Every Desired Output identifier from the registry, read-only.

    Unlike the browser menu, this includes identifiers that are accepted but not
    offered in the browser (``browser_hidden``) and marks the Custom entry, which
    has no fixed workflow. Each managed entry carries the stage sequence it
    requests before structure-dependent resolution (automatic treatments and
    frozen plane-wave cutoffs). Free-text menu descriptions are not included.
    """

    return [
        {
            "value": value,
            "label": label,
            "workflow_mode": "custom_workflow" if workflow_spec is None else "bmd_managed_desired_output",
            "browser_hidden": value in BROWSER_HIDDEN_DESIRED_OUTPUTS,
            "workflow_spec": validate_workflow_spec(workflow_spec).to_dict() if workflow_spec is not None else None,
        }
        for value, label, _description, workflow_spec in _desired_output_workflows()
    ]


def release_identity(source: dict[str, Any], *, tag: str | None) -> dict[str, Any]:
    """Release tag and revision of this checkout; never a deployment claim."""

    commit = source.get("commit")
    dirty = source.get("dirty")
    return {
        "tag": tag,
        "commit": commit,
        "dirty": dirty,
        "released_code": bool(tag) and dirty is False,
        "tag_source": "git describe --tags --exact-match HEAD on this checkout (tag signature not verified)",
        "describes": "the source code of this checkout",
        "deployment_verified": False,
        "deployment_statement": (
            "This record is produced from a source checkout. It is not evidence of "
            "which release, if any, a running BMD Compute service executes."
        ),
    }


def new_calculation_admission_description(capabilities: list[dict[str, Any]]) -> dict[str, Any]:
    """The admission policy, as data, and its effect on stage_modifier_support."""

    admissible = []
    for capability in capabilities:
        stage_type = capability["stage_type"]
        theory = capability["theory"]
        combinations = supported_stage_modifier_combinations(stage_type, theory)
        admissible.append(
            {
                "stage_type": stage_type,
                "theory": theory,
                "admissible_modifier_combinations": [
                    sorted(modifier.value for modifier in modifiers)
                    for modifiers in combinations
                    if admission.stage_is_admissible_for_new_calculation(
                        StageSpec(stage_type, theory, modifiers)
                    )
                ],
                "source": "backend.calculations.admission.stage_is_admissible_for_new_calculation",
            }
        )
    return {
        "policy_id": admission.ADMISSION_POLICY_ID,
        "policy_version": admission.ADMISSION_POLICY_VERSION,
        "applies_to": "every new calculation built, prepared or submitted (browser and machine API), after automatic treatments are resolved",
        "not_applied_to": "reading historical records, which may contain combinations now closed",
        "relationship_to_stage_modifier_support": (
            "stage_modifier_support lists combinations stage validation accepts, including "
            "combinations kept only so historical records remain readable. A combination "
            "listed in closed_combinations is refused for new calculations; "
            "admissible_modifier_support lists what remains open."
        ),
        "closed_combinations": [
            {**entry, "diagnostic_code": admission.HSE06_SOC_NOT_SUPPORTED_CODE}
            for entry in admission.unsupported_new_calculation_combinations_payload()
        ],
        "admissible_modifier_support": admissible,
        "phonon_workflow_restrictions": {
            "diagnostic_code": admission.PHONONS_UNSUPPORTED_TREATMENT_CODE,
            "required_theory": Theory.PBE.value,
            "excluded_modifiers": sorted(modifier.value for modifier in admission.PHONON_EXCLUDED_MODIFIERS),
            "applies_to": "every stage of a workflow that contains a phonon force stage",
            "statement": PHONON_SUPPORTED_SCOPE_TEXT,
        },
    }


def protocol_descriptions() -> dict[str, Any]:
    """Workflow-level methodology applied on top of stage definitions."""

    managed_relaxing = [
        entry["value"]
        for entry in desired_output_catalogue()
        if entry["workflow_spec"] is not None
        and any(
            geometry_relaxation.PROTOCOL_OPTION_KEY in (stage.get("options") or {})
            for stage in entry["workflow_spec"]["stages"]
        )
    ]
    return {
        "geometry_relaxation": {
            **geometry_relaxation.protocol_payload(),
            "desired_outputs": managed_relaxing,
            "custom_workflows": (
                "A Custom workflow uses the protocol only when it marks two consecutive, identical "
                "Geometry Optimisation stages as pass 1 and pass 2. An unmarked Custom relaxation "
                "keeps single-stage convergence: pass-1 values, or pass-2 values when it feeds a "
                "phonon force stage."
            ),
        },
        "plane_wave_cutoff": {
            **geometry_relaxation.cutoff_rule_payload(),
            "applies_to_stage_types": sorted(
                stage_type.value for stage_type in geometry_relaxation._CUTOFF_STAGE_TYPES
            ),
            "applies_to": "every new calculation's Geometry Optimisation and phonon force stages, BMD-managed or Custom",
            "resolved": "per structure at planning, from the POTCARs the executable input generators select, and frozen into the workflow",
        },
        "geometry_convergence": {
            **geometry_convergence.policy_payload(),
            "applies_to": "the final pass of the geometry relaxation protocol and the phonon force stage it feeds",
        },
        "stage_definition_defaults": {
            "statement": (
                "For new calculations, every Geometry Optimisation stage runs with the resolved "
                "plane-wave cutoff, the protocol's common settings and pass convergence, which "
                "replace the relax stage's executable_defaults for the settings listed here. "
                "Those stage defaults (including ENCUT 580 eV and ADDGRID) are used only to "
                "rebuild workflows recorded before the protocol existed."
            ),
            "stage_type": "relax",
            "settings_replaced_for_new_calculations": sorted(
                {"ENCUT", *geometry_relaxation.COMMON_RELAX_SETTINGS, *geometry_relaxation.PASS_CONVERGENCE[1]}
            ),
            "source": "backend.calculations.geometry_relaxation.relaxation_incar_settings",
        },
    }


def experimental_capability_descriptions() -> list[dict[str, Any]]:
    """Implemented capabilities that are not standard methodology."""

    defaults_status = STRUCTURE_SEARCH_DEFAULTS_POLICY["status"]
    return [
        {
            "id": "structure_search",
            "label": "Crystal structure search (USPEX)",
            "status": "experimental",
            "part_of_stage_methodology": False,
            "methodology_status": defaults_status,
            "validation_status": "not_established" if defaults_status == "provisional" else "not_recorded",
            "defaults_policy": dict(STRUCTURE_SEARCH_DEFAULTS_POLICY),
            "admission": json.loads(json.dumps(STRUCTURE_SEARCH_ADMISSION_SCOPE)),
            "interfaces": "machine API structure-search routes; the browser page is a preview",
            "machine_api_operations": list(STRUCTURE_SEARCH_MACHINE_API_OPERATIONS),
            "execution_gate": {
                "service_setting": STRUCTURE_SEARCH_EXECUTION_SETTING,
                "enabled_value": STRUCTURE_SEARCH_EXECUTION_ENABLED_VALUE,
                "default": "disabled",
                "gates": "operations that write to POWER or submit",
                "not_gated": "planning, local preparation, observation and cancellation of an attempt already submitted",
                "reported_state": None,
                "statement": "This record does not report whether any running service has structure-search execution enabled.",
            },
        }
    ]


def validation_statement() -> dict[str, Any]:
    return {
        "status": "not_recorded",
        "statement": (
            "This record describes implemented, executable methodology. It does not "
            "assert scientific validation or lab adoption of any capability; that is "
            "a human judgement recorded separately."
        ),
    }


def git_provenance(*, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repository_root()
    commit = _git_output(("rev-parse", "HEAD"), root)
    dirty_output = _git_output(("status", "--porcelain"), root)
    if commit is None or dirty_output is None:
        return unavailable_git_provenance()
    return {
        "repository": REPOSITORY_ID,
        "commit": commit,
        "dirty": bool(dirty_output.strip()),
        "provenance_available": True,
        "unavailable_reason": None,
    }


def unavailable_git_provenance() -> dict[str, Any]:
    return {
        "repository": REPOSITORY_ID,
        "commit": None,
        "dirty": None,
        "provenance_available": False,
        "unavailable_reason": "git provenance unavailable",
    }


def emit_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    if args:
        print("backend.calculations.capabilities does not accept arguments.", file=sys.stderr)
        return 2

    try:
        sys.stdout.write(emit_json(build_capability_payload()))
    except Exception as exc:
        print(f"Failed to build BMD Compute capability payload: {exc}", file=sys.stderr)
        return 1
    return 0


def _supported_stage_theory_capabilities(
    base_stage_definitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    capabilities: list[dict[str, Any]] = []
    for stage_definition in base_stage_definitions:
        stage_type = stage_definition["stage_type"]
        for theory in Theory:
            description = describe_stage(stage_type, theory)
            if description["theory_supported_for_stage"] is True:
                capabilities.append(description)
    return capabilities


def _stage_modifier_support(capabilities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    support: list[dict[str, Any]] = []
    for capability in capabilities:
        stage_type = capability["stage_type"]
        theory = capability["theory"]
        combinations = supported_stage_modifier_combinations(stage_type, theory)
        support.append(
            {
                "stage_type": stage_type,
                "theory": theory,
                "supported_modifier_combinations": [
                    sorted(modifier.value for modifier in modifiers)
                    for modifiers in combinations
                ],
                "source": (
                    "backend.calculations.registry."
                    "supported_stage_modifier_combinations"
                ),
            }
        )
    return support


def _release_tag(*, repo_root: Path | None = None) -> str | None:
    tag = _git_output(("describe", "--tags", "--exact-match", "HEAD"), repo_root or _repository_root())
    return tag or None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_executable() -> str:
    return "git"


def _git_command(args: tuple[str, ...], repo_root: Path) -> tuple[str, ...]:
    """Build a read-only git command for producer provenance.

    ``--no-optional-locks`` stops commands such as ``git status`` from
    opportunistically refreshing and rewriting ``.git/index`` (and creating
    ``.git/index.lock``). Producers run against the live BMD Compute checkout,
    including when invoked by read-only observers such as BMD Agent, so
    provenance inspection must never write to that checkout.
    """

    return (
        _git_executable(),
        "--no-optional-locks",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "-C",
        str(repo_root),
        *args,
    )


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _git_output(args: tuple[str, ...], repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            _git_command(args, repo_root),
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
            env=_git_environment(),
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
