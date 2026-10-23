"""A workflow explicitly selected as Custom stays Custom.

The browser's Desired Output selector (the ``workflow`` form field) and the
machine API's ``desired_output``/``custom`` choice are the authoritative
selection. A Custom workflow keeps its Custom identity through Build, Update
Calculation, Prepare, Submit and execution even when its stage sequence matches
a managed Desired Output, and it never acquires automatic treatments because
of that resemblance. Managed Desired Outputs keep their automatic treatments.

Planning only: preparation and submission are intercepted before any remote
work, and nothing here contacts POWER.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import warnings

import pytest
from starlette.requests import Request

import main
from backend.calculations.models import StageSpec, WorkflowSpec
import test_automatic_soc as soc_cases
import test_input_reference_contract as reference_cases
from backend.parser import parse_structure
from compute_api.plan_digest import compute_plan_digest
from compute_api.projection import plan_stage_previews


SI_POSCAR = reference_cases.SI_POSCAR
BI2SE3_POSCAR = soc_cases.BI2SE3_POSCAR


def _stage(stage_type, theory="pbe"):
    return {"stage_type": stage_type, "theory": theory, "modifiers": [], "options": {}}


# Every Custom stage sequence that matches (or once matched) a managed
# Desired Output's shape.
CUSTOM_SHAPES = {
    "static": ([_stage("static")], "energy_only"),
    "relax": ([_stage("relax")], None),
    "relax_static": ([_stage("relax"), _stage("static")], None),
    "relax_relax": ([_stage("relax"), _stage("relax")], "relaxed_structure"),
    "relax_relax_static": ([_stage("relax"), _stage("relax"), _stage("static")], "relaxed_energy"),
    "dos": ([_stage("relax"), _stage("static", "hse06"), _stage("dos", "hse06")], "electronic_dos"),
    "band_structure": (
        [_stage("relax"), _stage("static", "hse06"), _stage("band_structure", "hse06")],
        "electronic_band_structure",
    ),
    "phonons": ([_stage("relax"), _stage("phonon_forces")], "phonons"),
}
# Bi2Se3 is layered (automatic D3) and heavy (automatic SOC), so any automatic
# treatment leaking into a Custom workflow is visible. Phonon workflows are
# exercised on Si.
SHAPE_STRUCTURES = {name: BI2SE3_POSCAR for name in CUSTOM_SHAPES}
SHAPE_STRUCTURES["phonons"] = SI_POSCAR

RESOURCE_CASES = {
    "automatic": {"cpus": "", "memory_gb": "", "walltime": ""},
    "manual": {"cpus": "48", "memory_gb": "", "walltime": "10:00:00"},
}


def _request(path):
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _render(response) -> str:
    return response.template.render(response.context)


def _form_fields(html: str, action: str) -> dict:
    """The fields a browser posts from a server-rendered Prepare/Submit form."""

    start = html.index(f'<form action="{action}"')
    form = html[start:html.index("</form>", start)]
    fields: dict = {"modifiers": []}
    for name, value in re.findall(r'<input type="hidden" name="(\w+)" value="([^"]*)">', form):
        value = html_lib.unescape(value)
        if name == "modifiers":
            fields["modifiers"].append(value)
        else:
            fields[name] = value
    structure = re.search(r'<textarea class="hidden-field" name="structure">(.*?)</textarea>', form, re.S)
    fields["structure"] = html_lib.unescape(structure.group(1))
    fields["modifiers"] = fields["modifiers"] or None
    return fields


def _build_form_fields(html: str) -> dict:
    """What Update Calculation posts: the selector, the stage JSON and the structure."""

    select = html[html.index('id="desired-output-select"'):]
    select = select[:select.index("</select>")]
    selected = re.search(r'<option\s+value="(\w+)"\s+selected[\s>]', select).group(1)
    workflow_json = re.search(r'id="workflow-spec-json"\s+name="workflow_spec_json"\s+value="([^"]*)"', html)
    form = html[html.index('id="calculation-review-form"'):]
    form = form[:form.index("</form>")]
    fields = {
        "workflow": selected,
        "workflow_spec_json": html_lib.unescape(workflow_json.group(1)),
    }
    for name in ("cpus", "memory_gb", "walltime", "queue"):
        fields[name] = html_lib.unescape(
            re.search(rf'<input type="hidden" name="{name}" value="([^"]*)">', form).group(1)
        )
    return fields


def _build(structure, *, workflow, workflow_spec_json=None, resources=None):
    resources = {"queue": None, **(resources or RESOURCE_CASES["automatic"])}
    response = _quiet(
        main.build_workflow,
        _request("/build-workflow"),
        structure=structure,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        workflow_spec_json=workflow_spec_json,
        workflow=workflow,
        method=None,
        **resources,
    )
    assert response.status_code == 200, response.context.get("calculation_error")
    return response


def _build_custom(name, resources=None):
    stages, _ = CUSTOM_SHAPES[name]
    # The browser posts workflow=custom and stage JSON with no recipe.
    return _build(
        SHAPE_STRUCTURES[name],
        workflow="custom",
        workflow_spec_json=json.dumps({"stages": stages, "label": None, "recipe": None}),
        resources=resources,
    )


def _post(route, fields, **extra):
    names = (
        "structure", "fmt", "purpose", "theory", "modifiers", "cpus", "memory_gb", "walltime",
        "queue", "created_at", "submission_attempt_id", "submission_identity_token",
        "workflow_spec_json", "workflow", "method", "planned_resources",
    )
    kwargs = {name: fields.get(name) for name in names}
    kwargs.update(extra)
    return _quiet(route, _request("/"), **kwargs)


@pytest.fixture
def intercepted(monkeypatch):
    """Capture what Prepare and Submit would hand to the remote layer."""

    captured = {}

    def fake_prepare(submission_spec, **kwargs):
        captured["prepared"] = submission_spec
        return main.remembered_successful_preparation(submission_spec)

    def fake_submit(submission_spec, **kwargs):
        captured["submitted"] = submission_spec
        return {"status": "failed", "title": "Intercepted", "stage": "Submission", "reason": "test", "suggestion": ""}

    monkeypatch.setattr(main, "prepare_remote_submission", fake_prepare)
    monkeypatch.setattr(main, "submit_remote_workflow", fake_submit)
    monkeypatch.setattr(main, "load_results_for_completed_job", lambda *a, **k: None)
    return captured


def _workflow_of(submission_spec) -> dict:
    return submission_spec["flow_spec"]["workflow_spec"]


def _modifiers(context) -> list:
    context = getattr(context, "context", context)
    return [stage["modifiers"] for stage in context["selected_workflow"]["stages"]]


def _browser_digest(structure_text, context) -> str:
    structure = parse_structure(structure_text, "poscar")
    plan = main.CalculationPlan(
        structure=structure,
        workflow_spec=main.WorkflowSpec.from_dict(_workflow_of(context["submission_spec"])),
        default_treatment_resolution=context["default_treatment_resolution"],
        method_considerations=context["method_considerations"],
        execution_resources=context["selected_resources"],
        summary=context["summary"],
        generated_inputs=context["generated_inputs"],
        submission_spec=context["submission_spec"],
    )
    return _quiet(
        compute_plan_digest,
        structure=structure,
        submission_spec=context["submission_spec"],
        stage_previews=plan_stage_previews(plan),
    )


# ------------------------------------------------------------------- Build


@pytest.mark.parametrize("name", sorted(CUSTOM_SHAPES))
def test_build_keeps_custom_identity_and_applies_no_treatments(name):
    response = _build_custom(name)
    html = _render(response)
    context = response.context

    assert context["selected_workflow"]["desired_output"] == "custom"
    assert context["selected_workflow"]["hidden_desired_output_label"] is None
    assert all(modifiers == [] for modifiers in _modifiers(context))
    assert context["default_treatment_resolution"] is None
    assert not (_workflow_of(context["submission_spec"]).get("recipe") or "").startswith(
        tuple(value for _, value in CUSTOM_SHAPES.values() if value)
    )
    # Every server-rendered form carries the Custom selection.
    assert re.search(r'<option\s+value="custom"\s+selected>', html)
    for action in ("/prepare-remote", "/submit"):
        assert _form_fields(html, action)["workflow"] == "custom"


# --------------------------------------- Update, Prepare and Submit (end to end)


@pytest.mark.parametrize("resources", sorted(RESOURCE_CASES))
@pytest.mark.parametrize("name", sorted(CUSTOM_SHAPES))
def test_custom_identity_survives_update_prepare_and_submit(intercepted, name, resources):
    structure = SHAPE_STRUCTURES[name]
    built = _build_custom(name)

    # Update Calculation: the browser re-posts its own form, with any
    # hardware override, through the same authoritative Build route.
    update_fields = _build_form_fields(_render(built))
    assert update_fields["workflow"] == "custom"
    posted_resources = {
        name: update_fields.pop(name) for name in ("cpus", "memory_gb", "walltime", "queue")
    }
    if resources == "manual":
        posted_resources.update(RESOURCE_CASES["manual"])
    updated = _build(structure, resources=posted_resources, **update_fields)
    assert updated.context["selected_workflow"]["desired_output"] == "custom"
    assert all(modifiers == [] for modifiers in _modifiers(updated))
    updated_spec = updated.context["submission_spec"]
    assert _workflow_of(updated_spec) == _workflow_of(built.context["submission_spec"])

    # Prepare from the rendered Prepare form.
    prepared = _post(main.prepare_remote, _form_fields(_render(updated), "/prepare-remote"))
    assert prepared.status_code == 200, prepared.context.get("calculation_error")
    assert prepared.context["selected_workflow"]["desired_output"] == "custom"
    assert intercepted["prepared"] == updated_spec

    # Submit from the rendered Submit form of the prepared page.
    submit_fields = _form_fields(_render(prepared), "/submit")
    assert submit_fields["workflow"] == "custom"
    submitted = _post(main.submit_workflow, submit_fields, remote_prepared=submit_fields.get("remote_prepared", "true"))
    assert submitted.status_code == 200, submitted.context.get("calculation_error")
    assert submitted.context["selected_workflow"]["desired_output"] == "custom"

    # Execution identity: the submitted specification is the one built and
    # approved, with no automatic treatments and the approved resources.
    executed = intercepted["submitted"]
    assert executed == updated_spec
    assert all(stage["modifiers"] == [] for stage in _workflow_of(executed)["stages"])
    assert not executed["flow_spec"].get("automatic_treatments")
    assert _browser_digest(structure, submitted.context) == _browser_digest(structure, updated.context)
    if resources == "manual":
        assert executed["resources"]["ntasks"] == 48
        assert executed["resources"]["walltime"] == "10:00:00"


# --------------------------------------------------- managed outputs unchanged


@pytest.mark.parametrize(
    "desired_output",
    ["relaxed_energy", "relaxed_structure", "electronic_dos", "electronic_band_structure", "energy_only"],
)
def test_managed_desired_outputs_keep_identity_and_treatments_through_submit(intercepted, desired_output):
    built = _build(BI2SE3_POSCAR, workflow=desired_output)
    assert built.context["selected_workflow"]["desired_output"] == desired_output
    assert any("dispersion" in modifiers for modifiers in _modifiers(built))
    assert built.context["default_treatment_resolution"] is not None

    prepared = _post(main.prepare_remote, _form_fields(_render(built), "/prepare-remote"))
    assert prepared.status_code == 200, prepared.context.get("calculation_error")
    submit_fields = _form_fields(_render(prepared), "/submit")
    assert submit_fields["workflow"] == desired_output
    submitted = _post(main.submit_workflow, submit_fields, remote_prepared="true")
    assert submitted.status_code == 200, submitted.context.get("calculation_error")
    assert submitted.context["selected_workflow"]["desired_output"] == desired_output
    built_spec = built.context["submission_spec"]
    for spec in (intercepted["prepared"], intercepted["submitted"]):
        assert _workflow_of(spec) == _workflow_of(built_spec)
        assert spec["flow_spec"]["automatic_treatments"] == built_spec["flow_spec"]["automatic_treatments"]
        assert spec["flow_spec"]["automatic_treatments"]


@pytest.mark.parametrize("resources", sorted(RESOURCE_CASES))
def test_legacy_energy_only_survives_update_prepare_and_submit(intercepted, resources):
    # An explicitly loaded legacy Energy Only page (a historical form post) is
    # still the single-point workflow at every step: never Relaxed Energy and
    # never Custom. As the browser submits, the queue is explicit on the first
    # Build as well as on Update.
    built = _build(
        SI_POSCAR, workflow="energy_only", resources={**RESOURCE_CASES["automatic"], "queue": main.DEFAULT_PARTITION}
    )
    assert built.context["selected_workflow"]["desired_output"] == "energy_only"
    assert _workflow_of(built.context["submission_spec"])["recipe"] == "energy_only"
    assert [stage["stage_type"] for stage in _workflow_of(built.context["submission_spec"])["stages"]] == ["static"]

    update_fields = _build_form_fields(_render(built))
    assert update_fields["workflow"] == "energy_only"
    posted_resources = {name: update_fields.pop(name) for name in ("cpus", "memory_gb", "walltime", "queue")}
    if resources == "manual":
        posted_resources.update(RESOURCE_CASES["manual"])
    updated = _build(SI_POSCAR, resources=posted_resources, **update_fields)
    updated_spec = updated.context["submission_spec"]
    assert updated.context["selected_workflow"]["desired_output"] == "energy_only"
    assert _workflow_of(updated_spec) == _workflow_of(built.context["submission_spec"])
    if resources == "automatic":
        assert _browser_digest(SI_POSCAR, updated.context) == _browser_digest(SI_POSCAR, built.context)
    else:
        assert _browser_digest(SI_POSCAR, updated.context) != _browser_digest(SI_POSCAR, built.context)

    prepared = _post(main.prepare_remote, _form_fields(_render(updated), "/prepare-remote"))
    assert prepared.status_code == 200, prepared.context.get("calculation_error")
    assert prepared.context["selected_workflow"]["desired_output"] == "energy_only"
    assert intercepted["prepared"] == updated_spec

    submit_fields = _form_fields(_render(prepared), "/submit")
    assert submit_fields["workflow"] == "energy_only"
    submitted = _post(main.submit_workflow, submit_fields, remote_prepared=submit_fields.get("remote_prepared", "true"))
    assert submitted.status_code == 200, submitted.context.get("calculation_error")
    assert submitted.context["selected_workflow"]["desired_output"] == "energy_only"
    assert intercepted["submitted"] == updated_spec
    assert _browser_digest(SI_POSCAR, submitted.context) == _browser_digest(SI_POSCAR, updated.context)
    if resources == "manual":
        assert intercepted["submitted"]["resources"]["ntasks"] == 48
        assert intercepted["submitted"]["resources"]["walltime"] == "10:00:00"


@pytest.mark.parametrize("route", ["prepare", "submit"])
def test_legacy_energy_only_rejects_resources_that_were_not_approved(intercepted, route):
    built = _build(SI_POSCAR, workflow="energy_only")
    if route == "prepare":
        fields = _form_fields(_render(built), "/prepare-remote")
        response = _post(main.prepare_remote, {**fields, "cpus": "48"})
    else:
        prepared = _post(main.prepare_remote, _form_fields(_render(built), "/prepare-remote"))
        fields = _form_fields(_render(prepared), "/submit")
        response = _post(main.submit_workflow, {**fields, "cpus": "48"}, remote_prepared="true")
    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == "planned_resources_changed"
    assert "submitted" not in intercepted


def test_managed_and_custom_versions_of_a_shape_differ_only_by_treatments():
    managed = _build(BI2SE3_POSCAR, workflow="relaxed_structure").context
    custom = _build_custom("relax_relax").context

    assert managed["selected_workflow"]["desired_output"] == "relaxed_structure"
    assert custom["selected_workflow"]["desired_output"] == "custom"
    assert _modifiers(managed) == [["dispersion"], ["dispersion"]]
    assert _modifiers(custom) == [[], []]


# ------------------------------------------------- machine API and digests


def _api_plan(structure, *, desired_output=None, custom=None, resources=None):
    return _quiet(
        main.plan_calculation_request,
        structure_text=structure,
        fmt="poscar",
        desired_output=desired_output,
        custom_workflow=custom,
        resources=resources,
    )


@pytest.mark.parametrize("name", sorted(CUSTOM_SHAPES))
def test_api_custom_plan_matches_the_browser_custom_build(name):
    stages, _ = CUSTOM_SHAPES[name]
    structure = SHAPE_STRUCTURES[name]
    plan = _api_plan(structure, custom={"stages": stages})
    browser = _build_custom(name).context

    assert all(not stage.modifiers for stage in plan.workflow_spec.stages)
    assert plan.default_treatment_resolution is None
    assert plan.workflow_spec.to_dict() == _workflow_of(browser["submission_spec"])
    api_digest = _quiet(
        compute_plan_digest,
        structure=plan.structure,
        submission_spec=plan.submission_spec,
        stage_previews=plan_stage_previews(plan),
    )
    assert api_digest == _browser_digest(structure, browser)


@pytest.mark.parametrize("name", sorted(value for value in CUSTOM_SHAPES if CUSTOM_SHAPES[value][1]))
def test_api_custom_and_managed_plans_of_the_same_shape_are_distinct(name):
    stages, managed = CUSTOM_SHAPES[name]
    structure = SHAPE_STRUCTURES[name]
    custom_plan = _api_plan(structure, custom={"stages": stages})
    managed_plan = _api_plan(structure, desired_output=managed)

    assert custom_plan.default_treatment_resolution is None
    assert managed_plan.default_treatment_resolution is not None
    assert managed_plan.workflow_spec.recipe == managed
    assert custom_plan.workflow_spec.recipe in (None, "custom")


# ----------------------------------------------- authoritative selection only


def test_a_custom_post_cannot_claim_a_managed_identity_through_its_recipe():
    # A Custom request whose stage JSON names a managed recipe is still Custom:
    # the selector, not the JSON, is authoritative.
    stages, _ = CUSTOM_SHAPES["relax_relax"]
    response = _build(
        BI2SE3_POSCAR,
        workflow="custom",
        workflow_spec_json=json.dumps({"stages": stages, "label": None, "recipe": "relaxed_structure"}),
    )
    html = _render(response)

    assert response.context["selected_workflow"]["desired_output"] == "custom"
    assert _modifiers(response) == [[], []]
    assert _form_fields(html, "/prepare-remote")["workflow"] == "custom"


def test_legacy_purpose_fields_without_a_selection_are_not_relabelled():
    # Legacy clients posting purpose/theory (no Desired Output) get no
    # automatic treatments, so they are shown and carried as Custom.
    response = _quiet(
        main.build_workflow,
        _request("/build-workflow"),
        structure=BI2SE3_POSCAR,
        fmt="poscar",
        purpose="double_relax",
        theory="pbe",
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=None,
        workflow=None,
        method=None,
    )
    assert response.status_code == 200, response.context.get("calculation_error")
    assert response.context["selected_workflow"]["desired_output"] == "custom"
    assert _modifiers(response) == [[], []]
    assert _form_fields(_render(response), "/prepare-remote")["workflow"] == "custom"


def test_page_without_a_selection_uses_only_the_recorded_recipe():
    stages = [StageSpec.from_dict(stage) for stage in CUSTOM_SHAPES["relax_relax"][0]]
    unnamed = main.page_context(selected_workflow=WorkflowSpec(stages, recipe=None))
    named = main.page_context(selected_workflow=WorkflowSpec(stages, recipe="relaxed_structure"))

    assert unnamed["selected_workflow"]["desired_output"] == "custom"
    assert named["selected_workflow"]["desired_output"] == "relaxed_structure"
