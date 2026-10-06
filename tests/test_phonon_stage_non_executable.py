"""The phonon force stage type exists as representation only (Phonopy M2a).

``StageType.PHONON_FORCES`` lets a workflow describe one stage whose
calculations are a derived task set. Nothing may build, prepare, submit or
preview it, and nothing public may offer it.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from starlette.requests import Request

import main
from backend.calculations.builder import build_calculation_flow
from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.method_considerations import _supported_modifier_stage_capabilities
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    NON_EXECUTABLE_STAGE_CODE,
    CalculationValidationError,
    calculation_form_options,
    supported_stage_modifier_combinations,
    validate_stage_spec,
    validate_user_workflow_spec,
    validate_workflow_spec,
    workflow_stage_directories,
)
from backend.calculations.stage_materialization import (
    NON_EXECUTABLE_STAGE_TYPES,
    StageMaterialization,
    stage_materialization,
    stage_type_is_executable,
)
from backend.generated_inputs import generated_input_stage_previews, preview_generated_inputs
from backend.parser import parse_structure, structure_from_spec
from backend.runtime_environment import PARITY_CRITICAL_PACKAGES, RECORDED_SUPPORTING_PACKAGES
from backend.submission import create_submission_spec
from backend.workflows import build_atomate2_flow_for_workflow_spec, build_atomate2_flow_from_spec


POSCAR = """Si
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
direct
0.0 0.0 0.0
0.25 0.25 0.25
"""
PHONON_WORKFLOW = WorkflowSpec(
    [
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.PHONON_FORCES, Theory.PBE),
    ]
)
EXISTING_STAGE_TYPES = ("relax", "static", "dos", "band_structure")
PUBLIC_DESIRED_OUTPUTS = (
    "energy_only",
    "relaxed_structure",
    "electronic_dos",
    "electronic_band_structure",
    "custom",
)


def request(path: str = "/build-workflow") -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def build(workflow_spec: WorkflowSpec, *, workflow: str = "custom"):
    return main.build_workflow(
        request(),
        structure=POSCAR,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow_spec.to_dict()),
        workflow=workflow,
        method=None,
    )


def remote_route(route, identity: dict, workflow_spec_json: str):
    kwargs = dict(
        structure=POSCAR,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        created_at=None,
        submission_attempt_id=identity["attempt_id"],
        submission_identity_token=identity["identity_token"],
        workflow_spec_json=workflow_spec_json,
        workflow="custom",
        method=None,
    )
    if route is main.submit_workflow:
        kwargs["remote_prepared"] = "true"
    return route(request(f"/{route.__name__}"), **kwargs)


def unreachable(*args, **kwargs):
    raise AssertionError("must not be reached for a non-executable stage type")


def assert_not_executable(callable_, *args, **kwargs):
    with pytest.raises(CalculationValidationError) as excinfo:
        callable_(*args, **kwargs)
    assert excinfo.value.diagnostic == {
        "code": NON_EXECUTABLE_STAGE_CODE,
        "policy": "fail_closed",
        "stage_type": "phonon_forces",
    }


# --- representation ------------------------------------------------------------------


def test_stage_spec_keeps_only_methodology_fields():
    assert [field.name for field in dataclasses.fields(StageSpec)] == [
        "stage_type",
        "theory",
        "modifiers",
        "label",
        "options",
    ]
    assert set(StageSpec(StageType.PHONON_FORCES).to_dict()) == {
        "stage_type",
        "theory",
        "modifiers",
        "label",
        "options",
    }


def test_phonon_force_stage_is_representable_as_one_stage():
    data = PHONON_WORKFLOW.to_dict()
    assert [stage["stage_type"] for stage in data["stages"]] == ["relax", "phonon_forces"]
    assert WorkflowSpec.from_dict(data) == PHONON_WORKFLOW
    assert StageSpec.from_dict({"stage_type": "phonon_forces"}).stage_type is StageType.PHONON_FORCES


def test_materialization_table_is_closed_and_existing_stages_are_single():
    for stage_type in StageType:
        stage_materialization(stage_type)
    assert {
        stage_type.value
        for stage_type in StageType
        if stage_materialization(stage_type) is StageMaterialization.SINGLE
    } == set(EXISTING_STAGE_TYPES)
    assert stage_materialization(StageType.PHONON_FORCES) is StageMaterialization.DERIVED_TASK_SET
    assert {member.value for member in StageMaterialization} == {"single", "derived_task_set"}
    assert NON_EXECUTABLE_STAGE_TYPES == frozenset({StageType.PHONON_FORCES})
    assert all(stage_type_is_executable(value) for value in EXISTING_STAGE_TYPES)


# --- calculation authority -----------------------------------------------------------


def test_registry_rejects_the_phonon_force_stage():
    assert_not_executable(validate_stage_spec, StageSpec(StageType.PHONON_FORCES))
    assert_not_executable(validate_workflow_spec, PHONON_WORKFLOW)
    assert_not_executable(validate_user_workflow_spec, PHONON_WORKFLOW)
    assert_not_executable(workflow_stage_directories, PHONON_WORKFLOW)
    assert_not_executable(validate_workflow_spec, WorkflowSpec([StageSpec("phonon_forces")]))


@pytest.mark.parametrize("theory", list(Theory))
@pytest.mark.parametrize("modifiers", [(), (Modifier.SPIN_POLARIZED,), (Modifier.SOC,)])
def test_no_theory_or_modifier_makes_the_stage_executable(theory, modifiers):
    assert_not_executable(validate_stage_spec, StageSpec(StageType.PHONON_FORCES, theory, modifiers))
    assert supported_stage_modifier_combinations(StageType.PHONON_FORCES, theory) == ()


def test_build_route_rejects_a_workflow_containing_the_stage():
    response = build(PHONON_WORKFLOW)

    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == NON_EXECUTABLE_STAGE_CODE
    assert not response.context.get("submission_spec")


@pytest.mark.parametrize("route", [main.prepare_remote, main.submit_workflow], ids=["prepare", "submit"])
def test_prepare_and_submit_reject_before_any_remote_operation(route, monkeypatch):
    built = build(WorkflowSpec([StageSpec(StageType.RELAX, Theory.PBE)]), workflow="relaxed_structure").context
    identity = built["submission_spec"]["submission"]
    monkeypatch.setattr(main, "prepare_remote_submission", unreachable)
    monkeypatch.setattr(main, "submit_remote_workflow", unreachable)
    monkeypatch.setattr(main, "remembered_successful_preparation", unreachable)

    response = remote_route(route, identity, json.dumps(PHONON_WORKFLOW.to_dict()))

    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == NON_EXECUTABLE_STAGE_CODE
    assert not response.context.get("submission_spec")


def test_direct_backend_entry_points_reject_before_generating_anything():
    structure = parse_structure(POSCAR)
    assert_not_executable(preview_generated_inputs, structure, PHONON_WORKFLOW)
    assert_not_executable(generated_input_stage_previews, structure, PHONON_WORKFLOW)
    assert_not_executable(build_calculation_flow, structure, PHONON_WORKFLOW)
    assert_not_executable(build_atomate2_flow_for_workflow_spec, structure, PHONON_WORKFLOW)
    with pytest.raises(CalculationValidationError):
        main.build_submission_state(structure_text=POSCAR, fmt="poscar", workflow_spec=PHONON_WORKFLOW)
    flow_spec = {
        "workflow": "custom_workflow",
        "workflow_spec": PHONON_WORKFLOW.to_dict(),
        "potcar_functional": "PBE_64",
        "kpoints": None,
        "incar": {},
        "structure": {"type": "pasted_text", "format": "poscar", "text": POSCAR},
    }
    with pytest.raises(CalculationValidationError):
        create_submission_spec(flow_spec, label="phonons", timestamp="20261006-120000", env={})


def test_power_runtime_reconstruction_rejects():
    flow_spec = {
        "workflow_spec": PHONON_WORKFLOW.to_dict(),
        "structure": {"type": "pasted_text", "format": "poscar", "text": POSCAR},
    }
    assert_not_executable(
        build_atomate2_flow_from_spec,
        structure_from_spec(flow_spec["structure"]),
        flow_spec,
        run_name="x",
    )


def test_input_reference_producer_reports_unsupported():
    payload = build_input_reference_payload(
        {
            "structure": {"type": "pasted_text", "format": "poscar", "text": POSCAR},
            "workflow_spec": PHONON_WORKFLOW.to_dict(),
        },
        include_provenance=False,
    )
    assert payload["status"] == "unsupported"
    assert payload["reference"] is None


# --- nothing public ------------------------------------------------------------------


def test_public_options_catalogue_does_not_offer_phonons():
    options = calculation_form_options()

    assert [item["value"] for item in options["stage_types"]] == list(EXISTING_STAGE_TYPES)
    assert [item["value"] for item in options["desired_outputs"]] == list(PUBLIC_DESIRED_OUTPUTS)
    assert "phonon" not in json.dumps(options).lower()


def test_method_considerations_do_not_list_the_stage():
    for modifier in Modifier:
        capabilities = _supported_modifier_stage_capabilities(modifier)
        assert all(item["stage_type"] in EXISTING_STAGE_TYPES for item in capabilities)


def test_phonopy_is_not_parity_critical():
    assert "phonopy" not in PARITY_CRITICAL_PACKAGES
    assert "phonopy" in RECORDED_SUPPORTING_PACKAGES
