from __future__ import annotations

import json

import pytest

from backend.calculations.capabilities import build_capability_payload
from backend.calculations.dispersion import (
    DEFAULT_DISPERSION_METHOD,
    dispersion_method_options,
    dispersion_option_payload,
)
from backend.calculations.models import CalculationSpec, Modifier, Purpose, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    calculation_spec_from_workflow_spec,
    validate_calculation_spec,
    validate_workflow_spec,
)
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure
from backend.submission import create_submission_spec
from backend.workflows import build_relax_input_set_generator, build_static_input_set_generator


SI_POSCAR = """Si
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

FE2O3_POSCAR = """Fe2O3
5.04
1.0 0.0 0.0
-0.5 0.8660254 0.0
0.0 0.0 2.7281746
Fe O
2 3
direct
0.0 0.0 0.355
0.0 0.0 0.645
0.305 0.0 0.25
0.0 0.305 0.25
0.695 0.695 0.25
"""


def _si_structure():
    return parse_structure(SI_POSCAR)


def _workflow(stage_type, *, method="dftd3-bj", modifiers=None):
    return WorkflowSpec([
        StageSpec(
            stage_type,
            Theory.PBE,
            set(modifiers or {Modifier.DISPERSION}),
            options=dispersion_option_payload(method),
        )
    ])


def _preview_incar(workflow, structure=None):
    return preview_generated_inputs(
        structure or _si_structure(),
        workflow,
        resources={"ntasks": 24},
        potcar_functional="PBE_64",
    )["incar"]


def test_dispersion_method_options_are_controlled_and_default_to_d3bj():
    assert DEFAULT_DISPERSION_METHOD == "dftd3-bj"
    assert list(dispersion_method_options()) == [
        {"value": "dftd3", "label": "DFT-D3"},
        {"value": "dftd3-bj", "label": "DFT-D3(BJ)"},
    ]


def test_pbe_static_generates_upstream_vdw_ivdw_for_d3_and_d3bj():
    plain_workflow = WorkflowSpec([
        StageSpec(StageType.STATIC, Theory.PBE),
    ])
    default_workflow = WorkflowSpec([
        StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DISPERSION}),
    ])
    plain_incar = _preview_incar(plain_workflow)
    default_incar = _preview_incar(default_workflow)
    d3_incar = _preview_incar(_workflow(StageType.STATIC, method="dftd3"))
    d3bj_incar = _preview_incar(_workflow(StageType.STATIC, method="dftd3-bj"))

    assert default_workflow.to_dict()["stages"][0]["options"] == {}
    assert "IVDW" not in plain_incar
    assert "IVDW = 12" in default_incar
    assert "IVDW = 11" in d3_incar
    assert "IVDW = 12" in d3bj_incar
    assert "LHFCALC" not in d3bj_incar


def test_pbe_relax_static_and_double_relax_apply_consistent_dispersion():
    relax_static = WorkflowSpec([
        StageSpec(StageType.RELAX, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3-bj")),
        StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3-bj")),
    ])
    double_relax = WorkflowSpec([
        StageSpec(StageType.RELAX, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3")),
        StageSpec(StageType.RELAX, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3")),
    ])

    assert validate_workflow_spec(relax_static) == relax_static
    assert validate_workflow_spec(double_relax) == double_relax
    assert _preview_incar(relax_static).count("IVDW = 12") == 2
    assert _preview_incar(double_relax).count("IVDW = 11") == 2


def test_dispersion_composes_with_spin_dft_u_and_gamma_only():
    spin_gamma_incar = _preview_incar(
        _workflow(
            StageType.STATIC,
            modifiers={Modifier.DISPERSION, Modifier.SPIN_POLARIZED, Modifier.GAMMA_ONLY},
        )
    )
    assert "IVDW = 12" in spin_gamma_incar
    assert "ISPIN = 2" in spin_gamma_incar

    fe2o3_incar = _preview_incar(
        _workflow(StageType.STATIC, modifiers={Modifier.DISPERSION, Modifier.DFT_U}),
        structure=parse_structure(FE2O3_POSCAR),
    )
    assert "IVDW = 12" in fe2o3_incar
    assert "LDAU = True" in fe2o3_incar
    assert "LDAUU = 5.3 0" in fe2o3_incar


def test_unsupported_dispersion_combinations_fail_explicitly():
    unsupported_workflows = [
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.DISPERSION})]),
        WorkflowSpec([StageSpec(StageType.DOS, Theory.PBE, {Modifier.DISPERSION})]),
        WorkflowSpec([StageSpec(StageType.BAND_STRUCTURE, Theory.PBE, {Modifier.DISPERSION})]),
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, options=dispersion_option_payload("dftd3"))]),
    ]

    for workflow in unsupported_workflows:
        with pytest.raises(CalculationValidationError) as excinfo:
            validate_workflow_spec(workflow)
        assert "van der Waals correction" in excinfo.value.message

    with pytest.raises(CalculationValidationError) as excinfo:
        validate_calculation_spec(CalculationSpec(Purpose.STATIC, Theory.HSE06, {Modifier.DISPERSION}))
    assert "van der Waals correction" in excinfo.value.message

    with pytest.raises(CalculationValidationError) as excinfo:
        validate_calculation_spec(CalculationSpec(Purpose.DOS, Theory.PBE, {Modifier.DISPERSION}))
    assert "van der Waals correction" in excinfo.value.message

    with pytest.raises(CalculationValidationError) as excinfo:
        validate_calculation_spec(CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE, {Modifier.DISPERSION}))
    assert "van der Waals correction" in excinfo.value.message


def test_connected_relax_static_stages_must_use_same_dispersion_method():
    workflow = WorkflowSpec([
        StageSpec(StageType.RELAX, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3")),
        StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3-bj")),
    ])
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(workflow)
    assert "same van der Waals correction" in excinfo.value.message

    missing_on_static = WorkflowSpec([
        StageSpec(StageType.RELAX, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3")),
        StageSpec(StageType.STATIC, Theory.PBE),
    ])
    with pytest.raises(CalculationValidationError):
        validate_workflow_spec(missing_on_static)


def test_stage_options_prevent_legacy_collapse_and_round_trip_through_submission_provenance():
    workflow = _workflow(StageType.STATIC, method="dftd3")
    assert calculation_spec_from_workflow_spec(workflow) is None

    submission = create_submission_spec(
        {
            "workflow_spec": workflow.to_dict(),
            "potcar_functional": "PBE_64",
            "kpoints": None,
            "incar": {},
            "structure": {"type": "pasted_text", "format": "poscar", "text": SI_POSCAR},
        },
        structure=_si_structure(),
        label="Si dispersion",
        timestamp="20260825-120000",
        ntasks=24,
        mem_gb=128,
        walltime="24:00:00",
        env={},
    )

    workflow_dict = workflow.to_dict()
    assert submission["flow_spec"]["workflow_spec"] == workflow_dict
    assert "calculation_spec" not in submission["flow_spec"]
    assert submission["provenance"]["execution"]["workflow_spec"] == workflow_dict
    assert submission["provenance"]["execution"]["stage_order"][0]["options"] == workflow_dict["stages"][0]["options"]
    assert submission["provenance"]["vasp"]["stages"][0]["options"] == workflow_dict["stages"][0]["options"]
    json.dumps(submission["provenance"], sort_keys=True)


def test_relax_and_static_generators_use_upstream_vdw_keyword():
    structure = _si_structure()
    modifiers = {Modifier.DISPERSION}

    relax_generator = build_relax_input_set_generator(
        structure,
        modifiers=modifiers,
        resources={"ntasks": 24},
        potcar_functional="PBE_64",
        dispersion_method="dftd3-bj",
    )
    static_generator = build_static_input_set_generator(
        structure,
        modifiers=modifiers,
        resources={"ntasks": 24},
        potcar_functional="PBE_64",
        dispersion_method="dftd3-bj",
    )

    assert relax_generator.vdw == "dftd3-bj"
    assert static_generator.vdw == "dftd3-bj"


def test_dispersion_policy_is_exposed_in_capability_contract():
    payload = build_capability_payload(include_provenance=False)
    policies = payload["modifier_policies"]
    assert len(policies) == 1
    policy = policies[0]

    assert policy["modifier"] == "dispersion"
    assert policy["default_method"] == "dftd3-bj"
    assert policy["upstream_interface"]["atomate2_pymatgen_generator_keyword"] == "vdw"
    assert policy["phase_1_support"] == {
        "theories": ["pbe"],
        "stage_types": ["relax", "static"],
        "blocked_with_modifiers": [],
        "blocked_terminal_stage_types": ["dos", "band_structure"],
    }
    assert policy["methods"] == [
        {"value": "dftd3", "label": "DFT-D3", "incar_effect": {"IVDW": 11}},
        {"value": "dftd3-bj", "label": "DFT-D3(BJ)", "incar_effect": {"IVDW": 12}},
    ]
