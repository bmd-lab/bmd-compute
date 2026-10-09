"""HSE06 + SOC is closed to new calculations; history stays readable.

New calculations are admitted at one authoritative boundary,
``main.build_submission_state_from_structure``, after automatic default
treatments are resolved. The registry and the shared structural validators,
which also read historical records, still accept HSE06 + SOC.
"""

from __future__ import annotations

import json
import socket
from copy import deepcopy

import paramiko
import pytest
from starlette.requests import Request

import backend.remote_runtime as remote_runtime
import main
import run_record_fixtures
import test_automatic_soc as soc_cases
import test_monitor_trust_boundary as monitor_cases
from backend.calculations.admission import (
    ADMISSION_POLICY_ID,
    HSE06_SOC_NOT_SUPPORTED_CODE,
    HSE06_SOC_NOT_SUPPORTED_MESSAGE,
    HSE06_SOC_OMITTED_MESSAGE,
    HSE06_SOC_OMITTED_TITLE,
    require_admissible_new_calculation,
    stage_is_admissible_for_new_calculation,
)
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    calculation_form_options,
    supported_stage_modifier_combinations,
    validate_user_workflow_spec,
    validate_workflow_spec,
    workflow_spec_from_flow_spec,
)
from backend.parser import parse_structure, structure_from_spec
from backend.results import workflow_spec_from_submission_spec
from backend.run_records import validate_job_record_v1, validate_submission_record_v1
from backend.workflows import build_atomate2_flow_from_spec


SI_POSCAR = soc_cases.SI_POSCAR
BI2SE3_POSCAR = soc_cases.BI2SE3_POSCAR


def request(path: str) -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def custom_json(*stages: StageSpec) -> str:
    return json.dumps(WorkflowSpec(list(stages), recipe="custom").to_dict(), sort_keys=True)


HSE06_SOC_CUSTOM_WORKFLOWS = {
    "hse06_static_soc": custom_json(
        StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
    ),
    "pbe_relax_hse06_static_soc": custom_json(
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
    ),
    "hse06_soc_dos_chain": custom_json(
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
        StageSpec(StageType.DOS, Theory.HSE06, {Modifier.SOC}),
    ),
    "hse06_soc_band_chain": custom_json(
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
        StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06, {Modifier.SOC}),
    ),
}

# Every form of request that can name a new HSE06 + SOC calculation:
# Custom workflow JSON, and the legacy purpose/theory/modifiers fields.
REQUESTS = {
    **{
        f"custom:{name}": {"workflow": "custom", "workflow_spec_json": spec_json}
        for name, spec_json in HSE06_SOC_CUSTOM_WORKFLOWS.items()
    },
    "legacy:static/hse06/soc": {
        "purpose": "static",
        "theory": "hse06",
        "modifiers": ["soc"],
    },
    "legacy:static/hse06/soc+spin": {
        "purpose": "static",
        "theory": "hse06",
        "modifiers": ["soc", "spin_polarized"],
    },
}


def form(**values) -> dict:
    fields = {
        "structure": SI_POSCAR,
        "fmt": "poscar",
        "purpose": None,
        "theory": None,
        "modifiers": None,
        "cpus": None,
        "memory_gb": None,
        "walltime": None,
        "queue": None,
        "workflow_spec_json": None,
        "workflow": None,
        "method": None,
    }
    fields.update(values)
    return fields


def executables(submission_spec: dict) -> list[str]:
    return [stage["executable"] for stage in submission_spec["provenance"]["vasp"]["stages"]]


def valid_identity() -> dict:
    """A genuine signed attempt identity, obtained from a supported build."""

    response = main.build_workflow(request("/build-calculation"), **form(workflow="energy_only"))
    assert response.status_code == 200
    return response.context["submission_spec"]["submission"]


def build(**values):
    return main.build_workflow(request("/build-calculation"), **form(**values))


def prepare(identity: dict, **values):
    return main.prepare_remote(
        request("/prepare-remote"),
        **form(**values),
        created_at=None,
        submission_attempt_id=identity["attempt_id"],
        submission_identity_token=identity["identity_token"],
    )


def submit(identity: dict, **values):
    return main.submit_workflow(
        request("/submit"),
        **form(**values),
        created_at=None,
        submission_attempt_id=identity["attempt_id"],
        submission_identity_token=identity["identity_token"],
        remote_prepared="true",
    )


@pytest.fixture
def no_remote_boundary(monkeypatch):
    """Fail the test on any attempt to prepare, submit, open SSH or a socket."""

    calls = []

    def forbidden(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"rejected request reached the remote boundary: {name}")

        return _raise

    monkeypatch.setattr(main, "prepare_remote_submission", forbidden("prepare_remote_submission"))
    monkeypatch.setattr(main, "submit_remote_workflow", forbidden("submit_remote_workflow"))
    monkeypatch.setattr(remote_runtime, "create_remote_runner", forbidden("create_remote_runner"))
    monkeypatch.setattr(paramiko.SSHClient, "connect", forbidden("paramiko.SSHClient.connect"))
    monkeypatch.setattr(socket.socket, "connect", forbidden("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", forbidden("socket.create_connection"))
    return calls


def assert_rejected(response, calls):
    assert response.status_code == 400
    error = response.context["calculation_error"]
    assert error["message"] == HSE06_SOC_NOT_SUPPORTED_MESSAGE
    assert "Remove SOC from HSE06" in error["suggestion"]
    assert not response.context.get("generated_inputs")
    assert not response.context.get("submission_spec")
    assert not response.context.get("submission_result")
    assert not response.context.get("remote_preparation")
    assert calls == []
    rendered = response.template.render(response.context)
    assert "Calculation Validation Failed" in rendered
    assert "HSE06 + SOC is not currently supported for automatic execution." in rendered


# --- 1. Admission policy ------------------------------------------------------------


@pytest.mark.parametrize("stage_type", [StageType.STATIC, StageType.DOS, StageType.BAND_STRUCTURE])
def test_hse06_soc_is_structurally_valid_but_not_admissible(stage_type):
    stage = StageSpec(stage_type, Theory.HSE06, {Modifier.SOC})

    # Structural registry unchanged: historical records stay valid.
    assert frozenset({Modifier.SOC}) in supported_stage_modifier_combinations(stage_type, Theory.HSE06)
    assert stage_is_admissible_for_new_calculation(stage) is False


def test_admission_rejects_any_hse06_soc_stage_and_names_it():
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
            StageSpec(StageType.DOS, Theory.HSE06, {Modifier.SOC}),
        ]
    )
    validate_workflow_spec(workflow)

    with pytest.raises(CalculationValidationError) as raised:
        require_admissible_new_calculation(workflow)

    assert raised.value.message == HSE06_SOC_NOT_SUPPORTED_MESSAGE
    assert "stages 2 and 3" in raised.value.suggestion
    assert raised.value.diagnostic == {
        "code": HSE06_SOC_NOT_SUPPORTED_CODE,
        "policy_id": ADMISSION_POLICY_ID,
        "policy_version": 1,
        "stage_indices": [2, 3],
    }


@pytest.mark.parametrize(
    "workflow",
    [
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC})]),
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC, Modifier.SPIN_POLARIZED})]),
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06)]),
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SPIN_POLARIZED})]),
        WorkflowSpec(
            [
                StageSpec(StageType.RELAX, Theory.PBE),
                StageSpec(StageType.STATIC, Theory.HSE06),
                StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
            ]
        ),
        WorkflowSpec(
            [
                StageSpec(StageType.STATIC, Theory.PBE),
                StageSpec(StageType.DOS, Theory.PBE),
            ]
        ),
    ],
)
def test_admission_keeps_pbe_soc_and_hse06_without_soc(workflow):
    assert require_admissible_new_calculation(workflow) == validate_workflow_spec(workflow)


# --- 2. Every new-calculation route rejects explicit HSE06 + SOC ---------------------


@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_build_rejects_explicit_hse06_soc(name, no_remote_boundary):
    assert_rejected(build(**REQUESTS[name]), no_remote_boundary)


@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_prepare_remote_rejects_explicit_hse06_soc_without_remote_contact(name, no_remote_boundary):
    identity = valid_identity()
    assert_rejected(prepare(identity, **REQUESTS[name]), no_remote_boundary)


@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_submit_rejects_explicit_hse06_soc_without_remote_contact(name, no_remote_boundary):
    identity = valid_identity()
    assert_rejected(submit(identity, **REQUESTS[name]), no_remote_boundary)


def test_user_validator_alone_still_accepts_hse06_soc_so_routes_rely_on_admission():
    # Custom workflow JSON passes the user-facing structural validator; the
    # route rejection above therefore comes from the admission boundary.
    spec = WorkflowSpec.from_dict(json.loads(HSE06_SOC_CUSTOM_WORKFLOWS["hse06_static_soc"]))
    assert validate_user_workflow_spec(spec).stages[0].modifiers == frozenset({Modifier.SOC})


def test_removing_the_admission_check_reopens_hse06_soc(monkeypatch):
    # Regression guard: the admission call in build_submission_state_from_structure
    # is the only thing that refuses HSE06 + SOC. Without it the request is built.
    monkeypatch.setattr(main, "require_admissible_new_calculation", validate_workflow_spec)

    response = build(**REQUESTS["custom:hse06_static_soc"])

    assert response.status_code == 200
    assert executables(response.context["submission_spec"]) == ["vasp_ncl"]


def test_admission_runs_after_automatic_treatments_are_resolved(monkeypatch):
    # If automatic treatment resolution ever produced HSE06 + SOC again, the
    # boundary must still refuse it.
    original = main.resolve_workflow_for_structure

    def reintroduce_hse06_soc(structure_obj, workflow_spec, *, workflow=None):
        resolved, resolution = original(structure_obj, workflow_spec, workflow=workflow)
        stages = [
            StageSpec(stage.stage_type, stage.theory, {*stage.modifiers, Modifier.SOC},
                      label=stage.label, options=stage.options)
            if stage.theory is Theory.HSE06
            else stage
            for stage in resolved.stages
        ]
        return WorkflowSpec(stages, label=resolved.label, recipe=resolved.recipe), None

    monkeypatch.setattr(main, "resolve_workflow_for_structure", reintroduce_hse06_soc)

    response = main.build_workflow(
        request("/build-calculation"),
        **form(structure=BI2SE3_POSCAR, workflow="electronic_dos"),
    )

    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == HSE06_SOC_NOT_SUPPORTED_CODE


# --- 3. Supported methodology is unchanged -----------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        {"workflow": "custom", "workflow_spec_json": custom_json(StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC}))},
        {"purpose": "static", "theory": "pbe", "modifiers": ["soc"]},
    ],
)
def test_pbe_soc_static_remains_supported_and_runs_vasp_ncl(values):
    response = build(**values)

    assert response.status_code == 200
    generated = response.context["generated_inputs"]
    assert generated["vasp_executable"] == "vasp_ncl"
    assert executables(response.context["submission_spec"]) == ["vasp_ncl"]
    assert "LSORBIT = True" in generated["incar"]
    stage = response.context["submission_spec"]["provenance"]["vasp"]["stages"][0]
    assert stage["executable"] == "vasp_ncl"
    assert stage["custodian_vasp_job_kwargs"] == {"auto_gamma": False}


def test_pbe_soc_static_prepares_through_the_remote_boundary(monkeypatch):
    identity = valid_identity()
    prepared = []
    monkeypatch.setattr(
        main,
        "prepare_remote_submission",
        lambda submission_spec, **kwargs: prepared.append(submission_spec)
        or main.remembered_successful_preparation(submission_spec),
    )

    response = prepare(
        identity,
        workflow="custom",
        workflow_spec_json=custom_json(StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC})),
    )

    assert response.status_code == 200
    assert len(prepared) == 1
    assert prepared[0]["provenance"]["vasp"]["stages"][0]["executable"] == "vasp_ncl"


def test_heavy_element_energy_only_keeps_automatic_pbe_soc():
    response = main.build_workflow(
        request("/build-calculation"),
        **form(structure=soc_cases.BI_POSCAR, workflow="energy_only"),
    )

    assert response.status_code == 200
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == ["soc"]
    assert executables(response.context["submission_spec"]) == ["vasp_ncl"]
    assert "omitted_treatments" in response.context["submission_spec"]["flow_spec"]["automatic_treatments"]
    assert response.context["submission_spec"]["flow_spec"]["automatic_treatments"]["omitted_treatments"] == []
    rendered = response.template.render(response.context)
    assert HSE06_SOC_OMITTED_TITLE not in rendered
    assert "data-unsupported-omission" not in rendered


@pytest.mark.parametrize("desired_output", ["electronic_dos", "electronic_band_structure"])
def test_non_heavy_hse06_desired_outputs_are_unchanged_and_carry_no_warning(desired_output):
    response = main.build_workflow(
        request("/build-calculation"),
        **form(structure=SI_POSCAR, workflow=desired_output),
    )

    assert response.status_code == 200
    stages = response.context["selected_workflow"]["stages"]
    assert [(stage["stage_type"], stage["theory"], stage["modifiers"]) for stage in stages] == [
        ("relax", "pbe", []),
        ("static", "hse06", []),
        (desired_output.replace("electronic_", ""), "hse06", []),
    ]
    assert [item["executable"] for item in response.context["generated_inputs"]["vasp_executables"]] == [
        "vasp_std",
        "vasp_std",
        "vasp_std",
    ]
    assert HSE06_SOC_OMITTED_TITLE not in response.template.render(response.context)


@pytest.mark.parametrize("desired_output", ["electronic_dos", "electronic_band_structure"])
def test_heavy_element_hse06_desired_output_prepares_with_the_warning_visible(desired_output, monkeypatch):
    # Option A: the HSE06 workflow is kept, SOC omitted, and the red warning is
    # on the page that carries the Prepare and Submit forms.
    identity = valid_identity()
    prepared = []
    monkeypatch.setattr(
        main,
        "prepare_remote_submission",
        lambda submission_spec, **kwargs: prepared.append(submission_spec)
        or main.remembered_successful_preparation(submission_spec),
    )

    response = prepare(identity, structure=BI2SE3_POSCAR, workflow=desired_output)

    assert response.status_code == 200
    assert len(prepared) == 1
    workflow = prepared[0]["flow_spec"]["workflow_spec"]
    assert [stage["theory"] for stage in workflow["stages"]] == ["pbe", "hse06", "hse06"]
    assert all("soc" not in stage["modifiers"] for stage in workflow["stages"])
    omitted = prepared[0]["flow_spec"]["automatic_treatments"]["omitted_treatments"]
    assert [item["stage_indices"] for item in omitted] == [[2, 3]]
    rendered = response.template.render(response.context)
    assert HSE06_SOC_OMITTED_TITLE in rendered
    assert HSE06_SOC_OMITTED_MESSAGE in rendered
    assert rendered.index("data-unsupported-omission") < rendered.index('action="/submit"')


# --- 4. Browser prevention (backend remains authoritative) --------------------------


def test_browser_receives_the_unsupported_combination_without_disabling_soc_globally():
    options = calculation_form_options()

    assert options["unsupported_new_calculation_combinations"] == [
        {
            "theory": "hse06",
            "theory_label": "HSE06",
            "modifier": "soc",
            "modifier_label": "Spin-Orbit Coupling (SOC)",
            "message": HSE06_SOC_NOT_SUPPORTED_MESSAGE,
        }
    ]
    soc = next(item for item in options["modifiers"] if item["value"] == "soc")
    assert soc["enabled"] is True
    assert "HSE06 + SOC is not currently supported" in soc["tooltip"]


def test_template_disables_soc_only_for_hse06_stages():
    source = main.templates.env.loader.get_source(main.templates.env, "index.html")[0]

    assert "options.unsupported_new_calculation_combinations" in source
    assert "function applyUnsupportedCombinations()" in source
    assert "applyUnsupportedCombinations();\n        updateStageNumbers();" in source


# --- 5. History remains readable and executable -------------------------------------


def historical_records():
    submission_path, job_record_path = run_record_fixtures.fixture_paths("hse06_soc_after_pbe_relax")
    return (
        json.loads(submission_path.read_text(encoding="utf-8")),
        json.loads(job_record_path.read_text(encoding="utf-8")),
    )


def test_committed_hse06_soc_records_remain_valid_and_readable():
    submission, job_record = historical_records()

    validate_submission_record_v1(submission)
    validate_job_record_v1(job_record)

    workflow = workflow_spec_from_submission_spec(submission)
    assert workflow is not None
    assert [(stage.theory, stage.modifiers) for stage in workflow.stages] == [
        (Theory.PBE, frozenset()),
        (Theory.HSE06, frozenset({Modifier.SOC})),
    ]
    # The remote preflight and the runner validate flow_spec structurally.
    assert workflow_spec_from_flow_spec(submission["flow_spec"]) == workflow


def test_committed_hse06_soc_record_still_builds_its_vasp_ncl_runtime():
    submission, _ = historical_records()
    runtime_structure = structure_from_spec(submission["flow_spec"]["structure"])

    flow = build_atomate2_flow_from_spec(
        runtime_structure,
        submission["flow_spec"],
        run_name=submission["run_name"],
        resources=submission["resources"],
    )

    jobs = list(flow.jobs)
    kwargs = [soc_cases.job_maker(job).run_vasp_kwargs for job in jobs]
    assert "vasp_cmd" not in kwargs[0]
    assert kwargs[1]["vasp_cmd"].endswith("vasp_ncl")
    assert kwargs[1]["vasp_job_kwargs"] == {"auto_gamma": False}
    incar = soc_cases.job_maker(jobs[1]).input_set_generator.get_input_set(
        runtime_structure,
        potcar_spec=True,
    ).incar
    assert incar["LSORBIT"] is True
    assert incar["LHFCALC"] is True


def test_monitor_refresh_of_a_submitted_hse06_soc_run_is_not_refused():
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
        ]
    )
    trusted_spec = monitor_cases.submission_spec()
    trusted_spec["flow_spec"]["workflow_spec"] = workflow.to_dict()
    run_dir = trusted_spec["paths"]["run_dir"]
    trusted_spec["paths"]["stage_dirs"] = {
        "stage_01": f"{run_dir}/stage_01",
        "stage_02": f"{run_dir}/stage_02",
    }
    trusted_spec["paths"]["result_dir"] = f"{run_dir}/stage_02"
    runner = monitor_cases.BoundaryRunner(monitor_cases.job_record(spec=trusted_spec))

    original_factory = remote_runtime.create_remote_runner
    remote_runtime.create_remote_runner = lambda runner_factory=None: runner
    try:
        response = main.refresh_monitoring(
            monitor_cases.request(),
            structure=SI_POSCAR,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            cpus="24",
            memory_gb="128",
            walltime="72:00:00",
            queue="leeburton-pool",
            created_at="20261007-120000",
            job_id=monitor_cases.JOB_ID,
            submitted_at="2026-10-07 12:00:00",
            monitor_state_json=main.monitor_state_json(submission_spec=deepcopy(trusted_spec)),
            workflow_spec_json=json.dumps(workflow.to_dict()),
        )
    finally:
        remote_runtime.create_remote_runner = original_factory

    assert response.status_code == 200
    assert not response.context.get("calculation_error")
    assert runner.query_calls == [monitor_cases.JOB_ID]
    stages = response.context["selected_workflow"]["stages"]
    assert stages[1]["theory"] == "hse06"
    assert stages[1]["modifiers"] == ["soc"]


def test_page_context_renders_a_historical_hse06_soc_workflow():
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
        ]
    )
    context = main.page_context(selected_workflow=workflow)
    assert context["selected_workflow"]["stages"][1]["modifiers"] == ["soc"]
