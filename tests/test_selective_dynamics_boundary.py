"""User selective-dynamics constraints must never reach a managed VASP calculation.

pymatgen keeps POSCAR selective-dynamics flags as the ``selective_dynamics``
site property and writes them back into every generated POSCAR. BMD Compute
refuses such structures wherever VASP inputs are generated (flow builders and
input previews), so every route and direct backend caller gets the same,
deterministic rejection. Analyze, which generates no inputs, is unchanged.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Lattice, Structure
from starlette.requests import Request

import backend.calculations.structure_constraints as structure_constraints
import main
from backend.calculations.builder import build_calculation_flow
from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.models import CalculationSpec, Purpose, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import CalculationValidationError, desired_output_workflow_spec
from backend.calculations.structure_constraints import (
    SELECTIVE_DYNAMICS_UNSUPPORTED_CODE,
    SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE,
    reject_unsupported_structure_constraints,
)
from backend.generated_inputs import generated_input_stage_previews, preview_generated_inputs
from backend.parser import parse_structure
from backend.workflows import (
    build_atomate2_flow_for_spec,
    build_atomate2_flow_for_workflow_spec,
    build_atomate2_flow_from_spec,
)


HEADER = """Si
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
"""
PLAIN_POSCAR = HEADER + "direct\n0.0 0.0 0.0\n0.25 0.25 0.25\n"
ALL_T_POSCAR = HEADER + "Selective dynamics\ndirect\n0.0 0.0 0.0 T T T\n0.25 0.25 0.25 T T T\n"
FROZEN_POSCAR = HEADER + "Selective dynamics\ndirect\n0.0 0.0 0.0 F F F\n0.25 0.25 0.25 T T T\n"
PARTIAL_POSCAR = HEADER + "selective\ndirect\n0.0 0.0 0.0 T F T\n0.25 0.25 0.25 T T T\n"
CIF = """data_Si
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 3.8396
_cell_length_b 3.8396
_cell_length_c 3.8396
_cell_angle_alpha 60
_cell_angle_beta 60
_cell_angle_gamma 60
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0 0 0 1
Si2 Si 0.25 0.25 0.25 1
"""
STATIC = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
MANAGED_OUTPUTS = ("energy_only", "relaxed_structure", "electronic_dos", "electronic_band_structure")


def request(path: str = "/build-workflow") -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def build(structure_text: str, *, workflow: str | None = "energy_only", workflow_spec=None, fmt="poscar"):
    return main.build_workflow(
        request(),
        structure=structure_text,
        fmt=fmt,
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow_spec.to_dict()) if workflow_spec is not None else None,
        workflow=workflow,
        method=None,
    )


def remote_route(route, structure_text: str, identity: dict, workflow_spec_json: str):
    kwargs = dict(
        structure=structure_text,
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
        workflow="energy_only",
        method=None,
    )
    if route is main.submit_workflow:
        kwargs["remote_prepared"] = "true"
    return route(request(f"/{route.__name__}"), **kwargs)


def plain_structure(**site_properties) -> Structure:
    return Structure(
        Lattice([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        site_properties=site_properties or None,
    )


def assert_rejected(callable_, *args, **kwargs):
    with pytest.raises(CalculationValidationError) as excinfo:
        callable_(*args, **kwargs)
    assert excinfo.value.message == SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE
    assert excinfo.value.diagnostic["code"] == SELECTIVE_DYNAMICS_UNSUPPORTED_CODE
    return excinfo.value


def unreachable(*args, **kwargs):
    raise AssertionError("must not be reached for a selective-dynamics structure")


# --- pymatgen representation (why the rule is presence-based) ----------------------------


def test_pymatgen_creates_the_property_only_when_a_flag_is_false():
    assert "selective_dynamics" not in parse_structure(PLAIN_POSCAR).site_properties
    assert "selective_dynamics" not in parse_structure(ALL_T_POSCAR).site_properties
    assert "selective_dynamics" not in parse_structure(CIF, "cif").site_properties
    frozen = parse_structure(FROZEN_POSCAR).site_properties["selective_dynamics"]
    assert [list(row) for row in frozen] == [[False, False, False], [True, True, True]]
    assert "selective_dynamics" in parse_structure(PARTIAL_POSCAR).site_properties


# --- accepted input is unchanged -------------------------------------------------------


@pytest.mark.parametrize("text, fmt", [(PLAIN_POSCAR, "poscar"), (CIF, "cif")], ids=["poscar", "cif"])
def test_unconstrained_structures_are_accepted(text, fmt):
    response = build(text, fmt=fmt)
    assert response.status_code == 200
    assert "Selective" not in response.context["generated_inputs"]["poscar"]


def test_all_true_selective_dynamics_poscar_carries_no_constraint_and_builds_identically():
    # pymatgen discards an all-T block while parsing, so nothing BMD could
    # inherit survives; the inputs are those of the same POSCAR without it.
    constrained_free = build(ALL_T_POSCAR).context["generated_inputs"]
    plain = build(PLAIN_POSCAR).context["generated_inputs"]
    for key in ("incar", "kpoints", "poscar"):
        assert constrained_free[key] == plain[key]


@pytest.mark.parametrize("output", MANAGED_OUTPUTS)
def test_check_does_not_change_inputs_for_unconstrained_structures(output, monkeypatch):
    structure = parse_structure(PLAIN_POSCAR)
    workflow = desired_output_workflow_spec(output)
    guarded = [str(p["input_set"].incar) + str(p["input_set"].kpoints) + str(p["input_set"].poscar)
               for p in generated_input_stage_previews(structure, workflow)]
    monkeypatch.setattr(
        "backend.generated_inputs.reject_unsupported_structure_constraints", lambda structure: None
    )
    unguarded = [str(p["input_set"].incar) + str(p["input_set"].kpoints) + str(p["input_set"].poscar)
                 for p in generated_input_stage_previews(structure, workflow)]
    assert guarded == unguarded


def test_analyze_still_describes_a_constrained_structure():
    # Analyze generates no VASP inputs, so its semantics are unchanged.
    response = main.analyze(request("/analyze"), structure=FROZEN_POSCAR, fmt="poscar")
    assert response.status_code == 200
    assert response.context["summary"]


# --- every calculation route rejects ---------------------------------------------------


@pytest.mark.parametrize("output", MANAGED_OUTPUTS)
@pytest.mark.parametrize("text", [FROZEN_POSCAR, PARTIAL_POSCAR], ids=["FFF", "TFT"])
def test_build_rejects_constrained_structures_for_every_managed_output(output, text):
    response = build(text, workflow=output)
    assert response.status_code == 400
    error = response.context["calculation_error"]
    assert error["message"] == SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE
    assert error["diagnostic"] == {
        "code": SELECTIVE_DYNAMICS_UNSUPPORTED_CODE,
        "policy": "fail_closed",
        "site_property": "selective_dynamics",
        "sites": [1, 2],
    }
    assert "Selective dynamics" in error["suggestion"]
    assert not response.context.get("generated_inputs")
    assert not response.context.get("submission_spec")
    assert SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE in response.template.render(response.context)


def test_custom_workflow_cannot_bypass_the_boundary():
    custom = WorkflowSpec([StageSpec(StageType.RELAX, Theory.PBE), StageSpec(StageType.STATIC, Theory.PBE)])
    response = build(FROZEN_POSCAR, workflow="custom", workflow_spec=custom)
    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == SELECTIVE_DYNAMICS_UNSUPPORTED_CODE


@pytest.mark.parametrize("route", [main.prepare_remote, main.submit_workflow], ids=["prepare", "submit"])
def test_prepare_and_submit_reject_before_any_remote_operation(route, monkeypatch):
    built = build(PLAIN_POSCAR).context
    identity = built["submission_spec"]["submission"]
    workflow_spec_json = built["selected_workflow"]["json"]
    monkeypatch.setattr(main, "prepare_remote_submission", unreachable)
    monkeypatch.setattr(main, "submit_remote_workflow", unreachable)
    monkeypatch.setattr(main, "remembered_successful_preparation", unreachable)

    response = remote_route(route, FROZEN_POSCAR, identity, workflow_spec_json)

    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == SELECTIVE_DYNAMICS_UNSUPPORTED_CODE
    assert not response.context.get("submission_spec")


# --- direct backend callers --------------------------------------------------------------


def test_direct_backend_entry_points_reject():
    structure = parse_structure(FROZEN_POSCAR)
    assert_rejected(preview_generated_inputs, structure, STATIC)
    assert_rejected(generated_input_stage_previews, structure, STATIC)
    assert_rejected(build_calculation_flow, structure, STATIC)
    assert_rejected(build_atomate2_flow_for_workflow_spec, structure, STATIC)
    assert_rejected(build_atomate2_flow_for_spec, structure, CalculationSpec(purpose=Purpose.STATIC))
    assert_rejected(build_calculation_flow, structure, CalculationSpec(purpose=Purpose.RELAX))
    with pytest.raises(CalculationValidationError):
        main.build_submission_state(structure_text=FROZEN_POSCAR, fmt="poscar", workflow_spec=STATIC)


def test_power_runtime_reconstruction_rejects():
    flow_spec = {
        "workflow_spec": STATIC.to_dict(),
        "structure": {"type": "pasted_text", "format": "poscar", "text": FROZEN_POSCAR},
    }
    from backend.parser import structure_from_spec

    assert_rejected(build_atomate2_flow_from_spec, structure_from_spec(flow_spec["structure"]), flow_spec, run_name="x")


def test_input_reference_producer_reports_unsupported():
    payload = build_input_reference_payload(
        {
            "structure": {"type": "pasted_text", "format": "poscar", "text": FROZEN_POSCAR},
            "workflow_spec": STATIC.to_dict(),
        },
        include_provenance=False,
    )
    assert payload["status"] == "unsupported"
    assert payload["error"]["message"] == SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE
    assert payload["reference"] is None


# --- representations, malformed values, copies -------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        [[True, True, True], [True, True, True]],
        [(False, False, False), (True, True, True)],
        [np.array([False, True, True]), np.array([True, True, True])],
        [np.array([True, True, True]), np.array([True, True, True])],
        [[np.bool_(False)] * 3, [np.bool_(True)] * 3],
        [[0, 1, 1], [1, 1, 1]],
        [["F", "T", "T"], ["T", "T", "T"]],
        [None, None],
        [[], []],
        [[False], [True, True, True, True]],
        ["FFF", "TTT"],
        [{"x": False}, {"x": True}],
    ],
    ids=[
        "all-true-lists", "tuples", "numpy-arrays", "numpy-all-true", "numpy-bool-scalars",
        "ints", "strings", "none", "empty", "wrong-lengths", "strings-whole", "dicts",
    ],
)
def test_any_selective_dynamics_property_value_is_rejected_without_crashing(value):
    error = assert_rejected(reject_unsupported_structure_constraints, plain_structure(selective_dynamics=value))
    assert error.diagnostic["sites"] == [1, 2]


def test_a_single_constrained_site_is_enough():
    structure = plain_structure()
    structure[1].properties["selective_dynamics"] = [False, False, True]
    error = assert_rejected(preview_generated_inputs, structure, STATIC)
    assert error.diagnostic["sites"] == [2]


def test_property_survives_serialization_and_copies_and_is_still_rejected():
    original = parse_structure(FROZEN_POSCAR)
    variants = {
        "as_dict/from_dict": Structure.from_dict(original.as_dict()),
        "monty json": json.loads(json.dumps(original, cls=MontyEncoder), cls=MontyDecoder),
        "to json/from_str": Structure.from_str(original.to(fmt="json"), fmt="json"),
        "poscar round trip": Structure.from_str(original.to(fmt="poscar"), fmt="poscar"),
        "copy": original.copy(),
        "supercell": original * (1, 1, 2),
        "sorted": original.get_sorted_structure(),
    }
    for name, variant in variants.items():
        assert "selective_dynamics" in variant.site_properties, name
        assert_rejected(preview_generated_inputs, variant, STATIC)


def test_unusual_site_properties_container_fails_closed():
    class OpaqueSite:
        properties = None

    class OpaqueStructure:
        sites = [OpaqueSite()]

    assert_rejected(reject_unsupported_structure_constraints, OpaqueStructure())


def test_check_never_modifies_the_structure():
    structure = parse_structure(FROZEN_POSCAR)
    before = json.dumps(structure.as_dict(), cls=MontyEncoder, sort_keys=True)
    with pytest.raises(CalculationValidationError):
        preview_generated_inputs(structure, STATIC)
    assert json.dumps(structure.as_dict(), cls=MontyEncoder, sort_keys=True) == before


def test_check_runs_before_any_input_is_generated(monkeypatch):
    monkeypatch.setattr("backend.generated_inputs._input_set_for_stage", unreachable)
    monkeypatch.setattr("backend.generated_inputs.build_vasp_input_set_for_spec", unreachable)
    monkeypatch.setattr("backend.workflows.build_relax_input_set_generator", unreachable)
    monkeypatch.setattr("backend.workflows.build_static_input_set_generator", unreachable)
    structure = parse_structure(FROZEN_POSCAR)
    assert_rejected(preview_generated_inputs, structure, STATIC)
    assert_rejected(build_calculation_flow, structure, STATIC)


def test_rule_module_is_the_single_definition():
    assert structure_constraints.SELECTIVE_DYNAMICS_UNSUPPORTED_CODE == "selective_dynamics_unsupported"
