"""Hardware Specification is a result of Build Calculation.

Before a build the page shows only the scientific inputs. A successful build
resolves the workflow and its hardware with the R1 resource policy and shows
the Hardware Specification (CPUs, memory, walltime, QOS) with per-resource
overrides. Changing an override is approved by building again through the same
authoritative route, which updates the resource plan and the plan digest.

Planning only: nothing here contacts POWER.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest
from starlette.requests import Request

import main
import test_input_reference_contract as reference_cases
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from compute_api.plan_digest import compute_plan_digest
from compute_api.projection import plan_stage_previews


SI_POSCAR = reference_cases.SI_POSCAR


def _request(path):
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def _build(*, workflow="energy_only", workflow_spec_json=None, cpus="", memory_gb="", walltime="", queue=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main.build_workflow(
            _request("/build-workflow"),
            structure=SI_POSCAR,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
            workflow_spec_json=workflow_spec_json,
            workflow=workflow,
            method=None,
        )


def _html(response) -> str:
    return response.template.render(response.context)


def _calculation_form(html: str) -> str:
    start = html.index('<form\n                id="calculation-review-form"')
    return html[start:html.index("</form>", start)]


def _digest(context) -> str:
    structure = main.parse_structure(SI_POSCAR, "poscar")
    plan = main.CalculationPlan(
        structure=structure,
        workflow_spec=main.WorkflowSpec.from_dict(context["submission_spec"]["flow_spec"]["workflow_spec"]),
        default_treatment_resolution=context["default_treatment_resolution"],
        method_considerations=context["method_considerations"],
        execution_resources=context["selected_resources"],
        summary=context["summary"],
        generated_inputs=context["generated_inputs"],
        submission_spec=context["submission_spec"],
    )
    return compute_plan_digest(
        structure=structure,
        submission_spec=context["submission_spec"],
        stage_previews=plan_stage_previews(plan),
    )


def test_no_hardware_controls_or_recommendations_before_build():
    analyzed = main.analyze(_request("/analyze"), structure=SI_POSCAR, fmt="poscar")
    form = _calculation_form(_html(analyzed))

    assert "data-build-calculation-button" in form
    assert "Hardware Specification" not in form
    assert "data-resource-panel" not in form
    assert "data-resource-recommendation" not in form
    assert '<select name="cpus"' not in form
    assert '<select name="memory_gb"' not in form
    assert 'data-resource-field="walltime"' not in form
    assert "Automatic (24)" not in form
    # Automatic is what an unbuilt calculation asks for: the carried values are empty.
    for name in ("cpus", "memory_gb", "walltime"):
        assert f'<input type="hidden" name="{name}" value="">' in form
    # The queue is carried exactly as the built panel submits it (R1 behaviour).
    assert '<input type="hidden" name="queue" value="leeburton-pool">' in form


def _panel(form: str) -> str:
    start = form.index("data-resource-panel")
    return form[start:form.index("data-hardware-changed-notice", start)]


def _summary_values(panel: str) -> dict:
    return dict(re.findall(r'data-hardware-value="([a-z_]+)"[^>]*>([^<]*)<', panel))


def _selected(panel: str, control: str) -> str:
    start = panel.index(f'data-resource-control="{control}"')
    block = panel[start:panel.index("</select>", start)]
    return re.findall(r'<option value="([^"]*)" selected>', block)


def _submitted(panel: str) -> dict:
    return dict(re.findall(r'<input type="hidden" name="([a-z_]+)" value="([^"]*)">', panel))


def test_successful_build_shows_three_recommended_values_and_nothing_else():
    response = _build()
    html = _html(response)
    form = _calculation_form(html)
    panel = _panel(form)

    assert response.status_code == 200
    assert form.index("<h3>Scientific Specification</h3>") < form.index("<h3>Hardware Specification</h3>")
    assert form.index("<h3>Hardware Specification</h3>") < form.index("data-build-calculation-button")
    assert "data-hardware-specification" in panel and "data-built-output" in panel
    assert "Recommended resources for this calculation" in panel
    assert _summary_values(panel) == {"cpus": "24", "memory_gb": "128 GB", "walltime": "1 hour"}
    # One Custom button in the header; the controls are collapsed.
    assert panel.count("data-hardware-custom-toggle") == 1
    assert 'aria-expanded="false"' in panel
    assert "data-hardware-custom hidden" in panel
    # No estimator machinery or infrastructure details in this section.
    for absent in ("Automatic", "Manual", "Queue", "queue</", "QOS", "owner100d", "Account", "power-leeburton",
                   "Partition", "leeburton-pool<", "empirical", "fallback", "Stage 1", "safety", "policy", "Resources that will be submitted"):
        assert absent not in panel, absent
    # The scientific-change notice is shared; the hardware notice starts hidden.
    assert 'data-hardware-changed-notice role="status" hidden' in form


@pytest.mark.parametrize(
    ("workflow", "walltime"),
    [("relaxed_structure", "1 hour 45 min"), ("phonons", "4 hours 30 min"), ("electronic_band_structure", "3 days 1 hour")],
)
def test_summary_shows_the_resolved_values_for_each_workflow(workflow, walltime):
    context = _build(workflow=workflow).context
    panel = _panel(_calculation_form(_html(_build(workflow=workflow))))
    values = _summary_values(panel)
    assert values["walltime"] == walltime
    assert values["cpus"] == str(context["submission_spec"]["resources"]["ntasks"])
    assert values["memory_gb"] == f'{context["submission_spec"]["resources"]["mem_gb"]} GB'


def test_custom_controls_start_from_the_approved_values_and_submit_automatic():
    panel = _panel(_calculation_form(_html(_build())))
    # Only CPU, memory and walltime are editable.
    assert sorted(re.findall(r'data-resource-control="([a-z_]+)"', panel)) == ["cpus", "memory_gb", "walltime"]
    assert '<select name="queue"' not in panel and "<select" not in panel.replace('<select id="hardware-cpus"', "").replace(
        '<select id="hardware-memory"', ""
    )
    assert _selected(panel, "cpus") == ["24"]
    assert _selected(panel, "memory_gb") == ["128"]
    assert re.search(r'id="hardware-walltime"\s+value="01:00:00"', panel)
    # No Automatic option: the controls show values, and what is submitted stays Automatic
    # until the user changes a control.
    assert 'value=""' not in panel.split("data-hardware-custom", 1)[1].split('<input type="hidden"', 1)[0]
    assert _submitted(panel) == {"cpus": "", "memory_gb": "", "walltime": "", "queue": "leeburton-pool", "nodes": "1"}
    assert "Restore Recommended" in panel


def test_failed_build_reveals_no_hardware_recommendation_and_keeps_the_request():
    custom = WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC})], recipe="custom")
    response = _build(
        workflow="custom",
        workflow_spec_json=json.dumps(custom.to_dict()),
        walltime="10:00:00",
    )
    form = _calculation_form(_html(response))

    assert response.status_code == 400
    assert "Hardware Specification" not in form
    assert "data-resource-recommendation" not in form
    assert "Resources that will be submitted" not in form
    # The requested (manual) value is carried unchanged to the next build, not displayed.
    assert '<input type="hidden" name="walltime" value="10:00:00">' in form
    assert '<input type="hidden" name="cpus" value="">' in form


def test_overrides_after_build_update_the_resource_plan_and_digest():
    automatic = _build().context
    manual = _build(cpus="48", walltime="10:00:00").context
    back = _build(cpus="", walltime="").context

    # Overrides are applied through the authoritative build route.
    assert manual["submission_spec"]["resources"]["ntasks"] == 48
    assert manual["submission_spec"]["resources"]["walltime"] == "10:00:00"
    assert manual["selected_resources"]["sources"]["cpus"] == "manual"
    assert manual["selected_resources"]["sources"]["memory_gb"] == "automatic"
    assert manual["selected_resources"]["planned_token"] != automatic["selected_resources"]["planned_token"]
    assert _digest(manual) != _digest(automatic)
    # The manual plan records the automatic recommendation alongside the override.
    fields = manual["submission_spec"]["flow_spec"]["resource_selection"]["fields"]
    assert fields["walltime"] == {
        "source": "manual",
        "value": "10:00:00",
        "recommended": "01:00:00",
        "basis": "set manually",
    }

    # Returning to Automatic restores the estimator's recommendation and the automatic plan.
    assert back["submission_spec"]["resources"] == automatic["submission_spec"]["resources"]
    assert back["selected_resources"]["sources"]["walltime"] == "automatic"
    assert back["selected_resources"]["planned_token"] == automatic["selected_resources"]["planned_token"]
    assert _digest(back) == _digest(automatic)


@pytest.mark.parametrize(
    "posted",
    [
        {"cpus": "48", "memory_gb": "", "walltime": ""},  # override edited but not re-planned
        {"cpus": "", "memory_gb": "", "walltime": "10:00:00"},
    ],
)
def test_prepare_cannot_use_resources_that_were_not_approved(monkeypatch, posted):
    built = _build().context
    identity = built["submission_spec"]["submission"]
    monkeypatch.setattr(main, "prepare_remote_submission", lambda *a, **k: pytest.fail("prepared"))
    monkeypatch.setattr(main, "submit_remote_workflow", lambda *a, **k: pytest.fail("submitted"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.prepare_remote(
            _request("/prepare-remote"),
            structure=SI_POSCAR,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            queue=None,
            created_at=None,
            submission_attempt_id=identity["attempt_id"],
            submission_identity_token=identity["identity_token"],
            workflow_spec_json=None,
            workflow="energy_only",
            method=None,
            planned_resources=built["selected_resources"]["planned_token"],
            **posted,
        )
    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == "planned_resources_changed"


def test_prepare_and_submit_forms_carry_the_approved_plan_only():
    html = _html(_build(cpus="48"))
    for action in ("/prepare-remote", "/submit"):
        start = html.index(f'<form action="{action}"')
        form = html[start:html.index("</form>", start)]
        assert '<input type="hidden" name="cpus" value="48">' in form
        assert '<input type="hidden" name="walltime" value="">' in form
        assert 'name="planned_resources" value="cpus=48:manual;memory_gb=128:automatic;walltime=01:00:00:automatic;' in form


def test_reopening_custom_shows_the_manual_values_after_update_calculation():
    # Form values arrive as text; the controls must still show the manual values
    # and the submitted fields must keep them, so a later Update never reverts them.
    panel = _panel(_calculation_form(_html(_build(cpus="48", memory_gb="256", walltime="10:00:00"))))
    assert "Custom resources for this calculation" in panel
    assert _summary_values(panel) == {"cpus": "48", "memory_gb": "256 GB", "walltime": "10 hours"}
    assert _selected(panel, "cpus") == ["48"]
    assert _selected(panel, "memory_gb") == ["256"]
    assert re.search(r'id="hardware-walltime"\s+value="10:00:00"', panel)
    assert _submitted(panel)["cpus"] == "48"
    assert _submitted(panel)["memory_gb"] == "256"
    assert _submitted(panel)["walltime"] == "10:00:00"
    # Restore Recommended returns the controls to the estimator's values.
    assert 'data-recommended="24"' in panel and 'data-recommended="128"' in panel
    assert 'data-recommended="01:00:00"' in panel


def test_manual_values_equal_to_the_recommendation_stay_manual():
    automatic = _build(queue="leeburton-pool").context
    manual = _build(cpus="24", queue="leeburton-pool").context
    assert manual["submission_spec"]["resources"] == automatic["submission_spec"]["resources"]
    assert manual["selected_resources"]["sources"]["cpus"] == "manual"
    assert _digest(manual) != _digest(automatic)


def test_browser_update_back_to_automatic_restores_the_original_plan_digest():
    # As the browser submits: the queue is always explicit, before and after the build.
    queue = "leeburton-pool"
    first = _build(queue=queue).context
    manual = _build(cpus="48", queue=queue).context
    back = _build(cpus="", queue=queue).context
    assert _digest(manual) != _digest(first)
    assert back["selected_resources"]["planned_token"] == first["selected_resources"]["planned_token"]
    assert _digest(back) == _digest(first)


def _walltime_text(value: str) -> str:
    from jinja2 import Environment

    source = Path("templates/index.html").read_text(encoding="utf-8")
    start = source.index("{% macro walltime_text")
    end = source.index("{%- endmacro %}") + len("{%- endmacro %}")
    return Environment().from_string(source[start:end] + "{{ walltime_text(v) }}").render(v=value)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("01:00:00", "1 hour"),
        ("01:45:00", "1 hour 45 min"),
        ("10:00:00", "10 hours"),
        ("73:00:00", "3 days 1 hour"),
        ("00:05:00", "5 min"),
        ("01:30:45", "1 hour 30 min 45 sec"),
        ("00:00:30", "30 sec"),
        ("00:01:01", "1 min 1 sec"),
        ("72:00:30", "72 hours 30 sec"),
    ],
)
def test_walltime_summary_keeps_nonzero_seconds(value, text):
    assert _walltime_text(value) == text


@pytest.mark.parametrize(("walltime", "text"), [("01:30:45", "1 hour 30 min 45 sec"), ("00:00:30", "30 sec")])
def test_manual_walltime_with_seconds_is_shown_exactly(walltime, text):
    response = _build(walltime=walltime)
    panel = _panel(_calculation_form(_html(response)))
    assert response.context["submission_spec"]["resources"]["walltime"] == walltime
    assert _summary_values(panel)["walltime"] == text
    # The exact H:MM:SS value is kept in the tooltip, the Custom input and the submitted field.
    assert f'data-hardware-value="walltime" title="{walltime}">' in panel
    assert re.search(rf'id="hardware-walltime"\s+value="{walltime}"', panel)
    assert _submitted(panel)["walltime"] == walltime
