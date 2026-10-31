"""Workflow identity is recorded and shown only from authoritative sources.

1. A Custom workflow's executable specification carries no client-supplied
   descriptive metadata: its ``recipe``, ``label`` and stage labels are
   normalised away, so a client cannot record a managed Desired Output
   identity (or display name) in the submission specification or provenance,
   and cannot introduce automatic treatments through such metadata.
2. Monitoring and Resume show the workflow identity, stages and treatments
   recorded in BMD Compute's job record, never client-posted workflow fields,
   never the browser default, and never an identity inferred from stages.
   When the record cannot identify its workflow, the workflow is not shown.

Planning and record handling only: the remote layer is faked and nothing here
contacts POWER.
"""

from __future__ import annotations

import json
import re
import warnings
from copy import deepcopy

import pytest
from starlette.requests import Request

import backend.remote_runtime as remote_runtime
import main
import test_automatic_soc as soc_cases
import test_input_reference_contract as reference_cases
from backend.calculations.dispersion import dispersion_option_payload
from backend.calculations.registry import desired_output_workflow_spec
from backend.monitoring import RESUMED_SUBMISSION_SPEC_KEY, monitor_job
from backend.parser import parse_structure
from compute_api.plan_digest import compute_plan_digest
from compute_api.projection import plan_stage_previews
from test_monitor_trust_boundary import JOB_ID, BoundaryRunner


SI_POSCAR = reference_cases.SI_POSCAR
NIO_POSCAR = reference_cases.NIO_POSCAR
BI2SE3_POSCAR = soc_cases.BI2SE3_POSCAR
MANAGED = (
    "relaxed_energy",
    "energy_only",
    "relaxed_structure",
    "electronic_dos",
    "electronic_band_structure",
    "phonons",
)


def _request(path):
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _render(response) -> str:
    """Rendered markup without scripts (which carry stage-card templates)."""

    html = response.template.render(response.context)
    return re.sub(r"<script\b.*?</script>", "", html, flags=re.S)


def _stage(stage_type, theory="pbe", *, modifiers=(), label=None, options=None):
    return {
        "stage_type": stage_type,
        "theory": theory,
        "modifiers": list(modifiers),
        "label": label,
        "options": options or {},
    }


def _build(structure, *, workflow, workflow_spec=None):
    response = _quiet(
        main.build_workflow,
        _request("/build-workflow"),
        structure=structure,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow_spec) if workflow_spec is not None else None,
        workflow=workflow,
        method=None,
    )
    assert response.status_code == 200, response.context.get("calculation_error")
    return response


def _workflow(spec) -> dict:
    return spec["flow_spec"]["workflow_spec"]


def _assert_no_client_metadata(spec):
    """The serialised executable specification and provenance are Custom."""

    for workflow in (_workflow(spec), spec["provenance"]["execution"]["workflow_spec"]):
        assert workflow["recipe"] is None
        assert workflow["label"] is None
        assert all(stage["label"] is None for stage in workflow["stages"])
    assert not spec["flow_spec"].get("automatic_treatments")
    serialized = json.dumps(spec, default=str)
    for identifier in MANAGED:
        assert f'"recipe": "{identifier}"' not in serialized
        assert f'"desired_output": "{identifier}"' not in serialized


# =========================== 1. Custom workflow metadata ===========================

MALICIOUS_METADATA = {
    "managed_recipe": {"recipe": "relaxed_structure", "label": None},
    "managed_recipe_and_label": {"recipe": "relaxed_structure", "label": "Relaxed structure"},
    "conflicting_recipe_and_label": {"recipe": "relaxed_energy", "label": "Energy only"},
    "new_default_recipe": {"recipe": "relaxed_energy", "label": "Relaxed Energy"},
    "legacy_recipe": {"recipe": "energy_only", "label": "Energy only"},
    "phonon_recipe": {"recipe": "phonons", "label": None},
    "stale_custom_recipe": {"recipe": "custom", "label": "My workflow"},
    "legacy_calculation_recipe": {"recipe": "double_relax", "label": None},
}
SHAPES = {
    "relax_relax": [_stage("relax", label="Geometry Optimisation"), _stage("relax", label="Relaxed structure")],
    "static": [_stage("static", label="Energy only")],
    "relax_relax_static": [_stage("relax"), _stage("relax"), _stage("static", label="Relaxed Energy")],
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("metadata", sorted(MALICIOUS_METADATA))
def test_custom_build_drops_client_recipe_and_labels(shape, metadata):
    workflow_spec = {"stages": SHAPES[shape], **MALICIOUS_METADATA[metadata]}
    response = _build(BI2SE3_POSCAR, workflow="custom", workflow_spec=workflow_spec)
    spec = response.context["submission_spec"]

    _assert_no_client_metadata(spec)
    # Science unchanged: the user's stages, and no automatic treatments on a
    # layered heavy-element structure (which managed outputs would give D3/SOC).
    assert [stage["stage_type"] for stage in _workflow(spec)["stages"]] == [
        stage["stage_type"] for stage in SHAPES[shape]
    ]
    assert all(stage["modifiers"] == [] for stage in _workflow(spec)["stages"])
    assert response.context["default_treatment_resolution"] is None
    assert response.context["selected_workflow"]["desired_output"] == "custom"
    # The page carries the canonical form forward.
    assert json.loads(response.context["selected_workflow"]["json"]) == _workflow(spec)


def test_custom_settings_selected_by_the_user_are_preserved():
    stages = [
        _stage(
            "relax",
            modifiers=["dispersion", "spin_polarized"],
            options=dispersion_option_payload("dftd3"),
            label="Relaxed structure",
        ),
        _stage(
            "static",
            modifiers=["dispersion", "soc"],
            options=dispersion_option_payload("dftd3"),
            label="Energy only",
        ),
    ]
    response = _build(
        BI2SE3_POSCAR,
        workflow="custom",
        workflow_spec={"stages": stages, "recipe": "relaxed_energy", "label": "Relaxed Energy"},
    )
    workflow = _workflow(response.context["submission_spec"])

    _assert_no_client_metadata(response.context["submission_spec"])
    assert workflow["stages"][0]["modifiers"] == ["dispersion", "spin_polarized"]
    # The client's option is kept; BMD adds only the resolved cutoff record.
    options = dict(workflow["stages"][0]["options"])
    assert options.pop("plane_wave_cutoff")["rule"]["id"] == "bmd_compute.plane_wave_cutoff"
    assert options == dispersion_option_payload("dftd3")
    assert workflow["stages"][1]["modifiers"] == ["dispersion", "soc"]


def test_canonical_custom_spec_matches_the_clean_browser_and_api_plans():
    stages = [_stage("relax"), _stage("relax")]
    spoofed = _build(
        BI2SE3_POSCAR,
        workflow="custom",
        workflow_spec={"stages": stages, "recipe": "relaxed_structure", "label": "Relaxed structure"},
    ).context
    clean = _build(BI2SE3_POSCAR, workflow="custom", workflow_spec={"stages": stages, "recipe": None, "label": None}).context
    api = _quiet(
        main.plan_calculation_request,
        structure_text=BI2SE3_POSCAR,
        fmt="poscar",
        custom_workflow={"stages": [{key: stage[key] for key in ("stage_type", "theory", "modifiers", "options")} for stage in stages]},
    )

    assert _workflow(spoofed["submission_spec"]) == _workflow(clean["submission_spec"]) == api.workflow_spec.to_dict()

    def digest(context):
        structure = parse_structure(BI2SE3_POSCAR, "poscar")
        plan = main.CalculationPlan(
            structure=structure,
            workflow_spec=main.WorkflowSpec.from_dict(_workflow(context["submission_spec"])),
            default_treatment_resolution=context["default_treatment_resolution"],
            method_considerations=context["method_considerations"],
            execution_resources=context["selected_resources"],
            summary=context["summary"],
            generated_inputs=context["generated_inputs"],
            submission_spec=context["submission_spec"],
        )
        return _quiet(compute_plan_digest, structure=structure, submission_spec=context["submission_spec"], stage_previews=plan_stage_previews(plan))

    api_digest = _quiet(compute_plan_digest, structure=api.structure, submission_spec=api.submission_spec, stage_previews=plan_stage_previews(api))
    assert digest(spoofed) == digest(clean) == api_digest


@pytest.fixture
def intercepted(monkeypatch):
    captured = {}

    def fake_prepare(submission_spec, **kwargs):
        captured["prepared"] = deepcopy(submission_spec)
        return main.remembered_successful_preparation(submission_spec)

    def fake_submit(submission_spec, **kwargs):
        captured["submitted"] = deepcopy(submission_spec)
        return {"status": "failed", "title": "Intercepted", "stage": "Submission", "reason": "test", "suggestion": ""}

    monkeypatch.setattr(main, "prepare_remote_submission", fake_prepare)
    monkeypatch.setattr(main, "submit_remote_workflow", fake_submit)
    monkeypatch.setattr(main, "load_results_for_completed_job", lambda *a, **k: None)
    return captured


def _route(route, built_context, *, workflow, workflow_spec_json, **extra):
    identity = built_context["submission_spec"]["submission"]
    resources = built_context["selected_resources"]
    return _quiet(
        route,
        _request("/"),
        structure=BI2SE3_POSCAR,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=resources["requested"]["cpus"],
        memory_gb=resources["requested"]["memory_gb"],
        walltime=resources["requested"]["walltime"],
        queue=None,
        created_at=None,
        submission_attempt_id=identity["attempt_id"],
        submission_identity_token=identity["identity_token"],
        workflow_spec_json=workflow_spec_json,
        workflow=workflow,
        method=None,
        planned_resources=resources["planned_token"],
        **extra,
    )


@pytest.mark.parametrize("metadata", ["managed_recipe_and_label", "conflicting_recipe_and_label", "legacy_recipe"])
def test_prepare_and_submit_with_stale_or_malicious_metadata_keep_the_custom_spec(intercepted, metadata):
    stages = [_stage("relax"), _stage("relax")]
    built = _build(BI2SE3_POSCAR, workflow="custom", workflow_spec={"stages": stages}).context
    # The client edits the recipe/label between Build and Prepare/Submit.
    tampered = json.dumps({"stages": stages, **MALICIOUS_METADATA[metadata]})

    prepared = _route(main.prepare_remote, built, workflow="custom", workflow_spec_json=tampered)
    assert prepared.status_code == 200, prepared.context.get("calculation_error")
    submitted = _route(main.submit_workflow, built, workflow="custom", workflow_spec_json=tampered, remote_prepared="true")
    assert submitted.status_code == 200, submitted.context.get("calculation_error")

    for spec in (intercepted["prepared"], intercepted["submitted"]):
        _assert_no_client_metadata(spec)
        assert _workflow(spec) == _workflow(built["submission_spec"])
        assert spec["flow_spec"] == built["submission_spec"]["flow_spec"]
    assert submitted.context["selected_workflow"]["desired_output"] == "custom"


def test_managed_desired_outputs_keep_their_recipe_and_treatments():
    for desired_output, structure in (("relaxed_structure", BI2SE3_POSCAR), ("relaxed_energy", NIO_POSCAR)):
        # Stage JSON posted with a managed selection (even spoofed) is ignored.
        spoofed = {"stages": [_stage("static")], "recipe": "custom", "label": "Mine"}
        context = _build(structure, workflow=desired_output, workflow_spec=spoofed).context
        workflow = _workflow(context["submission_spec"])
        treatments = context["submission_spec"]["flow_spec"]["automatic_treatments"]

        assert workflow["recipe"] == desired_output
        assert treatments["mode"] == "bmd_managed_desired_output"
        assert treatments["desired_output"] == desired_output
        assert any(stage["modifiers"] for stage in workflow["stages"])
        assert [stage["stage_type"] for stage in workflow["stages"]] == [
            stage.stage_type.value for stage in desired_output_workflow_spec(desired_output).stages
        ]


# ============================ 2. Monitor and Resume identity ============================


def _plan_spec(structure, *, desired_output=None, custom=None):
    plan = _quiet(
        main.plan_calculation_request,
        structure_text=structure,
        fmt="poscar",
        desired_output=desired_output,
        custom_workflow=custom,
    )
    return json.loads(json.dumps(plan.submission_spec, default=str))


def _record(spec):
    return {
        "schema": "bmd_compute.job_record",
        "schema_version": 1,
        "job_id": JOB_ID,
        "run_name": spec["run_name"],
        "run_dir": spec["paths"]["run_dir"],
        "attempt_id": spec["submission"]["attempt_id"],
        "submission_spec": spec,
    }


def _contaminated_custom_spec():
    """A record written before the fix: a Custom workflow with a client-supplied managed recipe."""

    spec = _plan_spec(BI2SE3_POSCAR, custom={"stages": [_stage("relax"), _stage("relax")]})
    spec["flow_spec"]["workflow_spec"]["recipe"] = "relaxed_structure"
    spec["flow_spec"]["workflow_spec"]["label"] = "Relaxed structure"
    return spec


RECORDS = {
    # name: (spec factory, expected identity, expected stages, expected modifiers)
    "historical_energy_only": (
        lambda: _plan_spec(SI_POSCAR, desired_output="energy_only"),
        "energy_only",
        ["static"],
        [[]],
    ),
    "relaxed_energy_with_treatments": (
        lambda: _plan_spec(NIO_POSCAR, desired_output="relaxed_energy"),
        "relaxed_energy",
        ["relax", "relax", "static"],
        [["dft_u", "spin_polarized"]] * 3,
    ),
    "relaxed_energy_d3_soc": (
        lambda: _plan_spec(BI2SE3_POSCAR, desired_output="relaxed_energy"),
        "relaxed_energy",
        ["relax", "relax", "static"],
        [["dispersion"], ["dispersion"], ["dispersion", "soc"]],
    ),
    "custom_resembling_relaxed_structure": (
        lambda: _plan_spec(BI2SE3_POSCAR, custom={"stages": [_stage("relax"), _stage("relax")]}),
        "custom",
        ["relax", "relax"],
        [[], []],
    ),
    "custom_resembling_energy_only": (
        lambda: _plan_spec(BI2SE3_POSCAR, custom={"stages": [_stage("static")]}),
        "custom",
        ["static"],
        [[]],
    ),
}
UNIDENTIFIABLE = {
    "contaminated_custom_record": _contaminated_custom_spec,
    "managed_recipe_without_resolution": lambda: _strip_treatments(_plan_spec(SI_POSCAR, desired_output="energy_only")),
    "resolution_disagrees_with_recipe": lambda: _set_recipe(_plan_spec(SI_POSCAR, desired_output="energy_only"), "relaxed_energy"),
    "unknown_resolution_mode": lambda: _set_mode(_plan_spec(SI_POSCAR, desired_output="energy_only"), "something_else"),
    "unknown_desired_output": lambda: _set_output(_plan_spec(SI_POSCAR, desired_output="energy_only"), "retired_output"),
}


def _strip_treatments(spec):
    spec["flow_spec"].pop("automatic_treatments")
    return spec


def _set_recipe(spec, recipe):
    spec["flow_spec"]["workflow_spec"]["recipe"] = recipe
    return spec


def _set_mode(spec, mode):
    spec["flow_spec"]["automatic_treatments"]["mode"] = mode
    return spec


def _set_output(spec, output):
    spec["flow_spec"]["automatic_treatments"]["desired_output"] = output
    spec["flow_spec"]["workflow_spec"]["recipe"] = output
    return spec


# A client posting a different identity than the record, through every
# workflow field the monitor form ever carried.
CONFLICTING_POST = {
    "purpose": "double_relax",
    "theory": "pbe",
    "modifiers": ["soc"],
    "workflow_spec_json": json.dumps(desired_output_workflow_spec("relaxed_energy").to_dict()),
    "workflow": "relaxed_energy",
    "method": None,
}


@pytest.fixture
def remote(monkeypatch):
    holder = {}

    def use(runner):
        holder["runner"] = runner
        monkeypatch.setattr(remote_runtime, "create_remote_runner", lambda runner_factory=None: runner)
        return runner

    return use


def _monitor(spec, posted=CONFLICTING_POST, state_spec=None):
    return _quiet(
        main.refresh_monitoring,
        _request("/monitor"),
        structure=SI_POSCAR,
        fmt="poscar",
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        created_at=None,
        job_id=JOB_ID,
        submitted_at="",
        monitor_state_json=main.monitor_state_json(
            summary=json.loads(json.dumps(main.summarize_structure(parse_structure(SI_POSCAR, "poscar")), default=str)),
            calculation_summary={"calculation_type": "Client claimed", "calculation_plan": ["Client stage"], "flow_name": "client", "number_of_jobs": 9},
            generated_inputs={"incar": "# Stage 1 - Client"},
            submission_spec=state_spec or spec,
        ),
        **posted,
    )


def _resume(load_results="false"):
    return _quiet(main.resume_existing_calculation, _request("/resume"), job_id=JOB_ID, load_results=load_results)


def _shown(response):
    selected = response.context["selected_workflow"]
    return (
        selected["desired_output"],
        [stage["stage_type"] for stage in selected["stages"]],
        [stage["modifiers"] for stage in selected["stages"]],
    )


def _assert_suppressed(response):
    html = _render(response)
    assert response.context["selected_workflow"]["unavailable"] is True
    assert response.context["selected_workflow"]["stages"] == []
    assert 'id="desired-output-select"' not in html
    assert "data-workflow-stage" not in html
    assert "data-submitted-stage" not in html
    assert 'value="relaxed_energy" selected' not in html
    return html


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_monitor_shows_the_recorded_identity_not_the_posted_one(remote, name):
    factory, identity, stages, modifiers = RECORDS[name]
    spec = factory()
    remote(BoundaryRunner(_record(spec)))
    response = _monitor(spec)
    html = _render(response)

    assert response.status_code == 200
    assert _shown(response) == (identity, stages, modifiers)
    assert re.search(rf'<option\s+value="{identity}"\s+selected[\s>]', html)
    assert html.count("data-submitted-stage") == len(stages)
    assert f'data-submitted-desired-output="{identity}"' in html
    # Calculation summary identity comes from the record, not the client.
    assert response.context["calculation"]["calculation_type"] != "Client claimed"
    assert len(response.context["calculation"]["calculation_plan"]) == len(stages)
    # The next refresh carries the authoritative record only.
    assert response.context["submission_spec"] == spec


# Every client-carried panel value a monitor request can post, all false.
FALSE_CARRIED_STATE = {
    "summary": {
        "formula": "Xe999",
        "reduced_formula": "Xe999",
        "natoms": 999,
        "volume": 999.0,
        "space_group_symbol": "P-client",
        "space_group_number": 999,
    },
    "calculation": {
        "calculation_type": "Client claimed",
        "workflow_type": "Client claimed",
        "calculation_plan": ["Client stage"],
        "flow_name": "client",
        "number_of_jobs": 9,
        "job_names": ["client_job"],
        "theory": "client",
        "theory_label": "Client theory",
        "ready_for_submission": True,
    },
    "generated_inputs": {
        "incar": "# Stage 1 - Client",
        "kpoints": "client kpoints",
        "poscar": "client poscar",
        "slurm_script": "#SBATCH --client",
        "exact_slurm_script": "#SBATCH --client-exact",
        "submission_summary": {"job_name": "client-job", "resources": {"walltime": "999:00:00"}},
    },
}


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_monitor_never_renders_client_carried_scientific_state(remote, name):
    spec = RECORDS[name][0]()
    remote(BoundaryRunner(_record(spec)))
    state = json.loads(main.monitor_state_json(submission_spec=spec, **{
        "summary": FALSE_CARRIED_STATE["summary"],
        "calculation_summary": FALSE_CARRIED_STATE["calculation"],
        "generated_inputs": FALSE_CARRIED_STATE["generated_inputs"],
    }))
    response = _quiet(
        main.refresh_monitoring,
        _request("/monitor"),
        # A posted structure that is not the submitted one.
        structure=NIO_POSCAR if "Si" in spec["flow_spec"]["structure"]["text"] else SI_POSCAR,
        fmt="poscar",
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        created_at=None,
        job_id=JOB_ID,
        submitted_at="",
        monitor_state_json=json.dumps(state),
        **CONFLICTING_POST,
    )
    html = _render(response)

    assert response.status_code == 200
    for false_value in ("Xe999", "999 atoms", "P-client", "Client claimed", "Client stage", "client · 9 jobs",
                        "Client theory", "# Stage 1 - Client", "client kpoints", "client poscar", "#SBATCH --client",
                        "client-job", "999:00:00"):
        assert false_value not in html, false_value
    # Structure summary and structure text come from the record.
    recorded = parse_structure(spec["flow_spec"]["structure"]["text"], spec["flow_spec"]["structure"]["format"])
    assert response.context["summary"] == main.summarize_structure(recorded)
    assert response.context["structure_text"] == spec["flow_spec"]["structure"]["text"]
    # Calculation summary: the recorded label and workflow; no job count is claimed.
    calculation = response.context["calculation"]
    assert calculation["flow_name"] == spec["label"]
    assert calculation["number_of_jobs"] is None
    assert len(calculation["calculation_plan"]) == len(spec["flow_spec"]["workflow_spec"]["stages"])
    assert f"{spec['label']} · " not in html
    # Generated inputs are not held by the record and are not shown.
    assert response.context["generated_inputs"] is None
    assert "<h2>Generated Inputs</h2>" not in html
    # The next refresh carries only authoritative state.
    carried = json.loads(response.context["monitor_state_json"])
    assert carried["summary"]["reduced_formula"] == recorded.composition.reduced_formula
    assert carried["calculation"]["flow_name"] == spec["label"]
    assert carried["generated_inputs"] is None


def test_monitor_without_a_recorded_structure_keeps_the_field_and_shows_no_carried_state(remote):
    spec = _plan_spec(SI_POSCAR, desired_output="energy_only")
    record_spec = deepcopy(spec)
    record_spec["flow_spec"].pop("structure")
    remote(BoundaryRunner(_record(record_spec)))
    state = main.monitor_state_json(
        summary=FALSE_CARRIED_STATE["summary"],
        calculation_summary=FALSE_CARRIED_STATE["calculation"],
        generated_inputs=FALSE_CARRIED_STATE["generated_inputs"],
        submission_spec=spec,
    )
    response = _quiet(
        main.refresh_monitoring, _request("/monitor"), structure=SI_POSCAR, fmt="poscar", cpus=None,
        memory_gb=None, walltime=None, queue=None, created_at=None, job_id=JOB_ID, submitted_at="",
        monitor_state_json=state, **CONFLICTING_POST,
    )
    html = _render(response)
    assert response.status_code == 200
    assert response.context["summary"] is None
    assert response.context["structure_text"] == SI_POSCAR
    assert "<h2>Structure Summary</h2>" not in html
    assert "Xe999" not in html and "# Stage 1 - Client" not in html and "client · 9 jobs" not in html


def test_monitor_treatment_display_comes_from_the_record(remote):
    spec = _plan_spec(NIO_POSCAR, desired_output="relaxed_energy")
    remote(BoundaryRunner(_record(spec)))
    html = _render(_monitor(spec))
    panel = html[html.index("data-submitted-workflow"):]
    panel = panel[:panel.index("</table>")]

    assert panel.count("DFT+U") == 3
    assert panel.count("Spin Polarised") == 3
    assert "Applied Treatments" in html


@pytest.mark.parametrize("name", sorted(UNIDENTIFIABLE))
def test_monitor_suppresses_workflows_the_record_cannot_identify(remote, name):
    spec = UNIDENTIFIABLE[name]()
    remote(BoundaryRunner(_record(spec)))
    response = _monitor(spec)

    assert response.status_code == 200
    html = _assert_suppressed(response)
    assert "data-submitted-workflow-unavailable" in html
    assert response.context["calculation"] is None
    assert response.context["generated_inputs"] is None
    # Prepare/Submit controls never carry an unidentified workflow.
    assert '<form action="/prepare-remote"' not in html


def test_monitor_without_an_authenticated_record_shows_no_workflow(remote):
    spec = _plan_spec(SI_POSCAR, desired_output="energy_only")
    record = _record(spec)
    record["attempt_id"] = "12345678-1234-5678-9234-567812345679"  # not this attempt
    remote(BoundaryRunner(record))
    response = _monitor(spec)

    assert response.status_code == 200
    _assert_suppressed(response)
    assert response.context["submission_spec"] is None


def test_monitor_error_view_does_not_show_the_posted_identity(remote):
    spec = _plan_spec(SI_POSCAR, desired_output="energy_only")
    tampered = deepcopy(spec)
    token = tampered["submission"]["identity_token"]
    tampered["submission"]["identity_token"] = token[:-1] + ("A" if token[-1] != "A" else "B")
    remote(BoundaryRunner(_record(spec)))
    response = _monitor(spec, state_spec=tampered)

    assert response.status_code == 400
    _assert_suppressed(response)


def test_monitor_ssh_failure_keeps_retry_state_but_shows_no_workflow(remote):
    class Unreachable(BoundaryRunner):
        def connect(self, profile):
            raise OSError("temporary SSH connection failure")

    spec = _plan_spec(SI_POSCAR, desired_output="energy_only")
    remote(Unreachable(_record(spec)))
    response = _monitor(spec)
    html = _assert_suppressed(response)

    assert response.status_code == 200
    assert '<form action="/monitor"' in html
    # The retry state still carries the original monitoring state forward.
    state = json.loads(response.context["monitor_state_json"])
    assert state["calculation"]["calculation_type"] == "Client claimed"


def test_monitor_ignores_unreadable_posted_workflow_fields(remote):
    spec = _plan_spec(SI_POSCAR, desired_output="energy_only")
    remote(BoundaryRunner(_record(spec)))
    response = _monitor(spec, posted={**CONFLICTING_POST, "workflow_spec_json": "{not json"})

    assert response.status_code == 200
    assert _shown(response) == ("energy_only", ["static"], [[]])


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_resume_recovers_the_recorded_workflow(remote, name):
    factory, identity, stages, modifiers = RECORDS[name]
    spec = factory()
    runner = remote(BoundaryRunner(_record(spec)))
    response = _resume()
    html = _render(response)

    assert response.status_code == 200
    assert _shown(response) == (identity, stages, modifiers)
    assert html.count("data-submitted-stage") == len(stages)
    assert f'data-submitted-desired-output="{identity}"' in html
    assert RESUMED_SUBMISSION_SPEC_KEY not in response.context["monitoring_result"]
    assert runner.query_calls == [JOB_ID]


def test_resumed_historical_energy_only_is_not_shown_as_relaxed_energy(remote):
    remote(BoundaryRunner(_record(_plan_spec(SI_POSCAR, desired_output="energy_only"))))
    html = _render(_resume())
    panel = html[html.index("data-submitted-workflow"):]

    assert "Energy only (legacy)" in panel[:panel.index("</table>")]
    assert "Relaxed Energy" not in panel[:panel.index("</table>")]


@pytest.mark.parametrize(
    "case",
    ["missing", "malformed_json", "contract_violation", "other_job", "run_dir_mismatch", *sorted(UNIDENTIFIABLE)],
)
def test_resume_without_a_recoverable_workflow_suppresses_it(remote, case):
    spec = _plan_spec(SI_POSCAR, desired_output="energy_only")
    record = _record(spec)
    exists = True
    if case == "missing":
        exists = False
    elif case == "malformed_json":
        record = "{not json"
    elif case == "contract_violation":
        del record["submission_spec"]["flow_spec"]["workflow_spec"]
    elif case == "other_job":
        record["job_id"] = "654321"
    elif case == "run_dir_mismatch":
        record["run_dir"] = "/elsewhere"
    else:
        record = _record(UNIDENTIFIABLE[case]())
    runner = remote(BoundaryRunner(record, record_exists=exists))
    response = _resume()

    assert response.status_code == 200
    html = _assert_suppressed(response)
    assert "data-submitted-workflow-unavailable" in html
    # Monitoring itself is unaffected.
    assert response.context["monitoring_result"]["status"] == "success"
    assert runner.query_calls == [JOB_ID]


def test_resume_results_loading_is_unchanged(remote, monkeypatch):
    remote(BoundaryRunner(_record(_plan_spec(SI_POSCAR, desired_output="energy_only"))))
    calls = []
    monkeypatch.setattr(main, "monitoring_indicates_success", lambda result: True)
    monkeypatch.setattr(
        main,
        "load_results_for_completed_job",
        lambda monitoring_result: calls.append(monitoring_result) or None,
    )
    _resume(load_results="true")

    assert len(calls) == 1
    assert RESUMED_SUBMISSION_SPEC_KEY not in calls[0]


def test_monitor_job_reads_the_record_only_when_resuming():
    spec = _plan_spec(SI_POSCAR, desired_output="energy_only")
    resumed_runner = BoundaryRunner(_record(spec))
    resumed = monitor_job(JOB_ID, runner_factory=lambda: resumed_runner)
    submitted_runner = BoundaryRunner(_record(spec))
    submitted = monitor_job(JOB_ID, submission_spec=spec, runner_factory=lambda: submitted_runner)

    assert resumed[RESUMED_SUBMISSION_SPEC_KEY] == spec
    assert RESUMED_SUBMISSION_SPEC_KEY not in submitted
    assert submitted_runner.record_path is None


def test_identity_is_never_inferred_from_stage_shape():
    # Same stages as the managed outputs, but recorded as Custom (no managed
    # resolution, no managed recipe): always Custom.
    for desired_output in MANAGED:
        stages = desired_output_workflow_spec(desired_output).to_dict()["stages"]
        spec = {"flow_spec": {"workflow_spec": {"stages": stages, "label": None, "recipe": None}}}
        assert main.authoritative_workflow_identity(spec)[1] == "custom"
    for broken in (None, {}, {"flow_spec": {}}, {"flow_spec": {"workflow_spec": {"stages": []}}}):
        assert main.authoritative_workflow_identity(broken) is None
