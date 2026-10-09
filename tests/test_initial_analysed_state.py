"""Before Build, the page shows only the user's selection.

Analysing a structure selects the default Desired Output (Energy only) as its
unresolved base recipe. Automatic treatments, Method Considerations, the
Calculation Summary and Generated Inputs are shown only after a successful
Build Calculation, and a failed request redisplays the selection as requested,
never as a resolved calculation.
"""

from __future__ import annotations

import json
import re

import pytest

import main
import test_automatic_soc as soc_cases
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import CalculationValidationError, desired_output_workflow_spec
from backend.structure_dimensionality import ANALYSIS_FAILED, StructureDimensionalityObservation


BI2SE3 = soc_cases.BI2SE3_POSCAR
RESOLVED_MARKERS = (
    "<h3>Method Considerations</h3>",
    'class="method-consideration-card',
    "<h2>Calculation Summary</h2>",
    "<h2>Generated Inputs</h2>",
    "<h2>Ready for Submission</h2>",
    "data-unsupported-omission=",
    'class="applied-treatments-field"',
    'class="applied-treatment-chip"',
)


def markup(response) -> str:
    """Rendered page without its scripts (which carry templates and options)."""

    html = response.template.render(response.context)
    return re.sub(r"<script\b.*?</script>", "", html, flags=re.S)


def selected_workflow_json(response) -> dict:
    return json.loads(response.context["selected_workflow"]["json"])


def analyze(poscar: str = BI2SE3):
    response = main.analyze(soc_cases.request("/analyze"), structure=poscar, fmt="poscar")
    assert response.status_code == 200
    return response


def build(poscar: str = BI2SE3, *, workflow: str, workflow_spec: WorkflowSpec | None = None):
    return soc_cases.build_route(poscar, workflow=workflow, workflow_spec=workflow_spec)


def assert_not_resolved(page: str) -> None:
    for marker in RESOLVED_MARKERS:
        assert marker not in page, marker


# --- Initial analysed state ------------------------------------------------------


@pytest.mark.parametrize("poscar", [BI2SE3, soc_cases.BI_POSCAR, soc_cases.FEPT_POSCAR, soc_cases.SI_POSCAR])
def test_analysed_structure_selects_energy_only_without_resolving_it(poscar):
    response = analyze(poscar)
    page = markup(response)

    assert response.context["selected_workflow"]["desired_output"] == "energy_only"
    assert re.search(r'<option\s+value="energy_only"\s+selected>', page)
    # The unresolved base recipe: a PBE Static Energy stage with no treatments.
    assert selected_workflow_json(response) == desired_output_workflow_spec("energy_only").to_dict()
    assert [stage["modifiers"] for stage in response.context["selected_workflow"]["stages"]] == [[]]
    assert response.context["method_considerations"] is None
    assert response.context["default_treatment_resolution"] is None
    assert response.context["generated_inputs"] is None
    assert response.context["submission_spec"] is None
    assert_not_resolved(page)
    assert '<span class="pill">Ready to build</span>' in page
    assert "Build Calculation" in page


def test_analysed_page_offers_every_desired_output_and_custom_before_build():
    page = markup(analyze())
    values = re.findall(r'<option\s+value="(\w+)"', page[page.index('id="desired-output-select"'):])
    assert {"energy_only", "relaxed_structure", "electronic_dos", "electronic_band_structure", "custom"} <= set(values)
    assert 'id="add-workflow-stage"' in page
    assert "data-build-calculation-button" in page


def test_switching_desired_outputs_before_build_shows_only_unresolved_recipes():
    # The browser re-renders stage cards from these base recipes when the
    # Desired Output changes, so none of them may carry automatic treatments.
    options = analyze().context["calculation_options"]["desired_outputs"]
    recipes = {option["value"]: option.get("workflow_spec") for option in options}
    for name in ("energy_only", "relaxed_structure", "electronic_dos", "electronic_band_structure"):
        stages = recipes[name]["stages"]
        assert stages, name
        assert all(stage["modifiers"] == [] for stage in stages), name
        assert recipes[name] == desired_output_workflow_spec(name).to_dict()


# --- Successful Build ---------------------------------------------------------------


def test_successful_build_reveals_the_resolved_calculation():
    response = build(workflow="energy_only")
    page = markup(response)

    assert response.status_code == 200
    stages = response.context["selected_workflow"]["stages"]
    assert stages[0]["modifiers"] == ["dispersion", "soc"]  # layered Bi2Se3: D3 and SOC
    assert 'class="applied-treatment-chip">Spin-Orbit Coupling (SOC)</span>' in page
    assert "<h3>Method Considerations</h3>" in page
    assert "Spin-orbit coupling (SOC) has been included automatically" in page
    assert "<h2>Calculation Summary</h2>" in page
    assert "<h2>Generated Inputs</h2>" in page
    assert response.context["generated_inputs"]["vasp_executable"] == "vasp_ncl"
    assert response.context["default_treatment_resolution"] is not None


def test_successful_dos_build_reveals_the_red_omission_warning():
    page = markup(build(workflow="electronic_dos"))

    assert page.count('data-method-consideration-presentation="unsupported"') == 1
    assert page.count("HSE06 + SOC is not currently supported by BMD Compute.") == 1
    assert "data-unsupported-omission" not in page


# --- Failed Build / Prepare / Submit -------------------------------------------------


def test_failed_custom_build_redisplays_the_request_not_a_resolved_calculation():
    custom = WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC})], recipe="custom")
    response = build(workflow="custom", workflow_spec=custom)
    page = markup(response)

    assert response.status_code == 400
    assert "Calculation Validation Failed" in page
    assert_not_resolved(page)
    # The user's own Custom selection remains editable, as entered.
    assert selected_workflow_json(response)["stages"][0]["theory"] == "hse06"
    assert selected_workflow_json(response)["stages"][0]["modifiers"] == ["soc"]


def test_failed_desired_output_build_shows_the_unresolved_selection(monkeypatch):
    monkeypatch.setattr(
        "backend.calculations.method_considerations.observe_structure_dimensionality",
        lambda structure: StructureDimensionalityObservation(
            status=ANALYSIS_FAILED, dimensionality=None, reason="test failure"
        ),
    )
    response = build(workflow="electronic_dos")
    page = markup(response)

    assert response.status_code == 400
    assert "dimensionality analysis failed" in response.context["calculation_error"]["message"]
    assert_not_resolved(page)
    assert response.context["selected_workflow"]["desired_output"] == "electronic_dos"
    assert all(stage["modifiers"] == [] for stage in response.context["selected_workflow"]["stages"])


def fail_after_resolution(*args, **kwargs):
    raise CalculationValidationError("Injected failure after automatic treatments were resolved.")


def test_build_failing_after_resolution_does_not_show_resolved_treatments(monkeypatch):
    monkeypatch.setattr(main, "build_submission_state_from_structure", fail_after_resolution)
    response = build(workflow="energy_only")
    page = markup(response)

    assert response.status_code == 400
    assert "Injected failure" in page
    assert_not_resolved(page)
    # SOC would have been applied by the resolver; the failed page does not show it.
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    assert response.context["selected_workflow"]["desired_output"] == "energy_only"


@pytest.mark.parametrize("route", ["prepare", "submit"])
def test_prepare_and_submit_failures_redisplay_the_requested_selection(route, monkeypatch):
    identity = build(soc_cases.SI_POSCAR, workflow="energy_only").context["submission_spec"]["submission"]
    monkeypatch.setattr(main, "build_submission_state_from_structure", fail_after_resolution)
    monkeypatch.setattr(main, "prepare_remote_submission", lambda *a, **k: pytest.fail("prepared"))
    monkeypatch.setattr(main, "submit_remote_workflow", lambda *a, **k: pytest.fail("submitted"))
    fields = dict(
        structure=BI2SE3, fmt="poscar", purpose=None, theory=None, modifiers=None,
        cpus=None, memory_gb=None, walltime=None, queue=None, created_at=None,
        submission_attempt_id=identity["attempt_id"],
        submission_identity_token=identity["identity_token"],
        workflow_spec_json=None, workflow="energy_only", method=None,
    )
    if route == "prepare":
        response = main.prepare_remote(soc_cases.request("/prepare-remote"), **fields)
    else:
        response = main.submit_workflow(soc_cases.request("/submit"), remote_prepared="true", **fields)

    assert response.status_code == 400
    assert_not_resolved(markup(response))
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []


# --- Custom Workflow ------------------------------------------------------------------


def test_custom_workflow_shows_user_options_without_automatic_treatments():
    custom = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)], recipe="custom")
    response = build(workflow="custom", workflow_spec=custom)
    page = markup(response)

    assert response.status_code == 200
    # The Custom stage is exactly as entered: SOC offered but not applied.
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    soc_box = re.search(r'<input\s+type="checkbox"\s+data-stage-modifier\s+value="soc"[^>]*>', page)
    assert soc_box and "checked" not in soc_box.group(0)
    assert 'class="applied-treatments-field"' not in page
    assert "automatic_treatments" not in response.context["submission_spec"]["flow_spec"]
    # Its considerations are advisory (yellow), never "applied".
    assert "Suggested to activate the Spin-Orbit Coupling (SOC) Advanced Option" in page


def test_custom_stage_modifiers_come_only_from_user_checkboxes():
    source = main.templates.env.loader.get_source(main.templates.env, "index.html")[0]
    read_stages = source[source.index("function readStages()"):]
    read_stages = read_stages[:read_stages.index("\n    }\n")]

    # Only checked, enabled checkboxes become stage modifiers; Desired Output
    # cards carry no checkboxes, so switching to Custom starts untreated.
    assert 'card.querySelectorAll("[data-stage-modifier]")' in read_stages
    assert "return input.checked && !input.disabled;" in read_stages
