"""Workflow stage options cannot carry free-form INCAR/KPOINTS or unknown settings.

Stage options may only hold BMD-supported treatment settings: ``dispersion``
(user-selectable) and ``dft_u`` (written by automatic Desired Output
resolution only). Rejection happens at the registry/validation boundary, so
no preview, preparation, input reference or runtime flow can be built from a
spec that carries anything else.
"""

from __future__ import annotations

import json

import pytest

import main
from backend.calculations.dispersion import dispersion_option_payload
from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    SUPPORTED_STAGE_OPTION_KEYS,
    USER_STAGE_OPTION_KEYS,
    CalculationValidationError,
    validate_user_workflow_spec,
    validate_workflow_spec,
)
from backend.generated_inputs import generated_input_stage_previews, preview_generated_inputs
from backend.workflows import build_atomate2_flow_for_workflow_spec, build_atomate2_flow_from_spec
from test_automatic_dft_u import FE2O3, NIO, build_route, evaluate_dft_u, poscar_text, request


INCAR_INJECTION = {"incar": {"EDIFF": 0.01}}
KPOINTS_INJECTION = {"kpoints": {"mode": "mesh", "value": [1, 1, 1]}}
UNKNOWN_KEY = {"nbands_override": 400}

INJECTIONS = {
    "incar": INCAR_INJECTION,
    "kpoints": KPOINTS_INJECTION,
    "unknown": UNKNOWN_KEY,
    "incar_beside_valid_dispersion": {**dispersion_option_payload("dftd3"), **INCAR_INJECTION},
}


def _custom(options, modifiers=frozenset()):
    return WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, set(modifiers), options=options)], recipe="custom")


def test_accepted_option_keys_are_exactly_dispersion_and_frozen_dft_u():
    assert SUPPORTED_STAGE_OPTION_KEYS == {"dispersion", "dft_u"}
    assert USER_STAGE_OPTION_KEYS == {"dispersion"}


# --- Registry boundary ----------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(INJECTIONS))
def test_validate_workflow_spec_rejects_injected_options(case):
    options = INJECTIONS[case]
    modifiers = {Modifier.DISPERSION} if "dispersion" in options else set()
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(_custom(options, modifiers))
    bad = next(key for key in options if key not in SUPPORTED_STAGE_OPTION_KEYS)
    assert repr(bad) in excinfo.value.message
    assert "INCAR, KPOINTS" in excinfo.value.message


def test_injection_is_rejected_on_any_stage_of_a_multistage_workflow():
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.HSE06),
            StageSpec(StageType.DOS, Theory.HSE06, options=KPOINTS_INJECTION),
        ],
        recipe="custom",
    )
    with pytest.raises(CalculationValidationError, match="'kpoints'"):
        validate_workflow_spec(workflow)


def test_client_workflows_cannot_supply_frozen_dft_u_parameters():
    parameters = evaluate_dft_u(NIO)["parameters"]
    forged = json.loads(json.dumps(parameters))
    forged["species"]["Ni"]["U"] = 20.0
    workflow = _custom({"dft_u": forged}, {Modifier.DFT_U})

    # Internally produced workflows (automatic resolution) may carry it ...
    validate_workflow_spec(workflow)
    # ... but a client-supplied Custom workflow may not.
    with pytest.raises(CalculationValidationError, match="'dft_u'"):
        validate_user_workflow_spec(workflow)


# --- No preview, input reference or runtime from a rejected spec ------------------------


@pytest.mark.parametrize("case", ["incar", "kpoints", "unknown"])
def test_preview_input_reference_and_runtime_refuse_the_spec(case):
    workflow = _custom(INJECTIONS[case])

    with pytest.raises(CalculationValidationError):
        preview_generated_inputs(NIO, workflow)
    with pytest.raises(CalculationValidationError):
        generated_input_stage_previews(NIO, workflow)
    with pytest.raises(CalculationValidationError):
        build_atomate2_flow_for_workflow_spec(NIO, workflow)

    flow_spec = {
        "workflow_spec": workflow.to_dict(),
        "potcar_functional": "PBE_64",
        "incar": {},
        "kpoints": None,
        "structure": {"type": "pasted_text", "format": "poscar", "text": poscar_text(NIO)},
    }
    with pytest.raises(CalculationValidationError):
        build_atomate2_flow_from_spec(NIO, flow_spec, run_name="injected")

    reference = build_input_reference_payload(
        {
            "structure": flow_spec["structure"],
            "workflow_spec": workflow.to_dict(),
            "potcar_functional": "PBE_64",
        },
        include_provenance=False,
    )
    assert reference["status"] == "unsupported"
    assert reference["reference"] is None


def _unreachable(*args, **kwargs):
    raise AssertionError("a rejected workflow must not reach preview or preparation")


@pytest.mark.parametrize("case", sorted(INJECTIONS))
def test_build_route_rejects_injection_before_any_preview(monkeypatch, case):
    monkeypatch.setattr(main, "preview_generated_inputs", _unreachable)
    monkeypatch.setattr(main, "build_submission_state_from_structure", _unreachable)
    options = INJECTIONS[case]
    modifiers = {Modifier.DISPERSION} if "dispersion" in options else set()

    response = build_route(NIO, workflow="custom", workflow_spec=_custom(options, modifiers))

    assert response.status_code == 400
    assert "unsupported stage option" in response.context["calculation_error"]["message"]
    assert not response.context.get("generated_inputs")
    assert not response.context.get("submission_spec")


def _prepare(workflow_json: str, *, workflow: str, identity: dict):
    return main.prepare_remote(
        request("/prepare-remote"),
        structure=poscar_text(NIO),
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
        workflow_spec_json=workflow_json,
        workflow=workflow,
        method=None,
    )


def _submit(workflow_json: str, *, workflow: str, identity: dict):
    return main.submit_workflow(
        request("/submit"),
        structure=poscar_text(NIO),
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
        remote_prepared="true",
        workflow_spec_json=workflow_json,
        workflow=workflow,
        method=None,
    )


def _identity(workflow: str = "custom", workflow_spec: WorkflowSpec | None = None) -> dict:
    submission = build_route(NIO, workflow=workflow, workflow_spec=workflow_spec).context["submission_spec"]
    return submission["submission"]


@pytest.mark.parametrize("case", ["incar", "kpoints", "unknown"])
def test_prepare_and_submit_routes_reject_injection_without_touching_the_cluster(monkeypatch, case):
    identity = _identity(workflow_spec=_custom({}))
    monkeypatch.setattr(main, "prepare_remote_submission", _unreachable)
    monkeypatch.setattr(main, "submit_remote_workflow", _unreachable)
    injected = json.dumps(_custom(INJECTIONS[case]).to_dict())

    for response in (
        _prepare(injected, workflow="custom", identity=identity),
        _submit(injected, workflow="custom", identity=identity),
    ):
        assert response.status_code == 400
        assert "unsupported stage option" in response.context["calculation_error"]["message"]


def test_prepare_route_rejects_client_supplied_frozen_dft_u(monkeypatch):
    identity = _identity(workflow_spec=_custom({}))
    monkeypatch.setattr(main, "prepare_remote_submission", _unreachable)
    parameters = evaluate_dft_u(NIO)["parameters"]
    forged = _custom({"dft_u": parameters}, {Modifier.DFT_U})

    response = _prepare(json.dumps(forged.to_dict()), workflow="custom", identity=identity)

    assert response.status_code == 400
    assert "'dft_u'" in response.context["calculation_error"]["message"]


# --- Supported workflows are unaffected ------------------------------------------------


@pytest.mark.parametrize(
    "desired_output",
    ["energy_only", "relaxed_structure", "electronic_dos", "electronic_band_structure"],
)
def test_all_desired_outputs_still_build_with_automatic_treatments(desired_output):
    response = build_route(NIO, workflow=desired_output)
    assert response.status_code == 200
    stages = response.context["submission_spec"]["flow_spec"]["workflow_spec"]["stages"]
    assert any("dft_u" in (stage.get("options") or {}) for stage in stages)
    assert response.context["generated_inputs"]["incar"]


def test_desired_output_prepare_still_accepts_its_resolved_workflow(monkeypatch):
    built = build_route(NIO, workflow="energy_only").context
    reached = {}

    def fake_prepare(submission_spec, **kwargs):
        reached["stages"] = submission_spec["flow_spec"]["workflow_spec"]["stages"]
        return main.remembered_successful_preparation(submission_spec)

    monkeypatch.setattr(main, "prepare_remote_submission", fake_prepare)
    response = _prepare(
        built["selected_workflow"]["json"],  # carries the automatic dft_u option
        workflow="energy_only",
        identity=built["submission_spec"]["submission"],
    )

    assert response.status_code == 200
    assert "dft_u" in reached["stages"][0]["options"]


@pytest.mark.parametrize(
    "workflow",
    [
        _custom({}, {Modifier.DFT_U}),
        _custom(dispersion_option_payload("dftd3"), {Modifier.DISPERSION}),
        _custom(dispersion_option_payload("dftd3-bj"), {Modifier.DISPERSION, Modifier.DFT_U, Modifier.SPIN_POLARIZED}),
        _custom({}, {Modifier.SOC}),
        WorkflowSpec(
            [
                StageSpec(StageType.RELAX, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload("dftd3")),
                StageSpec(StageType.STATIC, Theory.HSE06),
                StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
            ],
            recipe="custom",
        ),
    ],
)
def test_supported_custom_choices_still_build(workflow):
    response = build_route(FE2O3, workflow="custom", workflow_spec=workflow)
    assert response.status_code == 200
    assert response.context["generated_inputs"]["incar"]
    submitted = response.context["submission_spec"]["flow_spec"]["workflow_spec"]["stages"]
    assert [stage["options"] for stage in submitted] == [dict(stage.options) for stage in workflow.stages]
