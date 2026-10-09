"""Presentation order of the calculation page.

Required order: structure input and analysis; Desired Output / Custom workflow
selection; Method Considerations; Execution Resources; Calculation Summary and
Generated Inputs; Prepare / Submit / Monitor. Moving the panels must not change
any form field, identifier or request contract.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import main
import test_automatic_soc as soc_cases


TEMPLATE_SOURCE = Path("templates/index.html").read_text(encoding="utf-8")


def built_page(poscar: str, workflow: str) -> str:
    response = soc_cases.build_route(poscar, workflow=workflow)
    assert response.status_code == 200
    return response.template.render(response.context)


def analyzed_page(poscar: str) -> str:
    response = main.analyze(soc_cases.request("/analyze"), structure=poscar, fmt="poscar")
    assert response.status_code == 200
    return response.template.render(response.context)


def calculation_form(html: str) -> str:
    start = html.index('<form\n                id="calculation-review-form"')
    return html[start:html.index("</form>", start)]


def assert_in_order(text: str, markers: list[str]) -> None:
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions), list(zip(markers, positions))


class _Controls(HTMLParser):
    """Collect focusable controls (in DOM order) and form-field contracts."""

    def __init__(self):
        super().__init__()
        self.focus_order: list[str] = []
        self.names: list[str] = []
        self.ids: list[str] = []
        self.tabindexes: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if "name" in values:
            self.names.append(values["name"])
        if "tabindex" in values:
            self.tabindexes.append(values["tabindex"])
        hidden_field = values.get("type") == "hidden" or "hidden-field" in (values.get("class") or "")
        if tag in {"select", "input", "textarea", "button", "summary"} and not hidden_field:
            label = values.get("name") or values.get("id") or tag
            if "data-build-calculation-button" in values:
                label = "build"
            elif "data-stage-type" in values:
                label = "stage-type"
            elif "data-stage-theory" in values:
                label = "stage-theory"
            elif "data-stage-modifier" in values:
                label = f"modifier:{values.get('value')}"
            self.focus_order.append(label)


def controls(fragment: str) -> _Controls:
    parser = _Controls()
    parser.feed(fragment)
    return parser


# --- Page order ---------------------------------------------------------------


def test_built_page_follows_the_required_section_order():
    html = built_page(soc_cases.BI2SE3_POSCAR, "electronic_dos")

    assert_in_order(
        html,
        [
            "<h2>Structure</h2>",
            "<h2>Structure Summary</h2>",
            "<h2>Calculation Definition</h2>",
            'id="desired-output-select"',
            'id="workflow-stage-list"',
            "<h3>Method Considerations</h3>",
            "<h3>Execution Resources</h3>",
            'name="cpus"',
            "data-build-calculation-button",
            "<h2>Calculation Summary</h2>",
            "<h2>Generated Inputs</h2>",
            "<h2>Ready for Submission</h2>",
            'action="/prepare-remote"',
            'action="/submit"',
        ],
    )
    # Nothing about method considerations remains in the structure analysis.
    structure_analysis = html[
        html.index("<h2>Structure Summary</h2>"):html.index("<h2>Calculation Definition</h2>")
    ]
    assert "method-consideration" not in structure_analysis
    assert "Method Considerations" not in structure_analysis


def test_analyzed_page_shows_considerations_after_workflow_choice_and_before_resources():
    html = analyzed_page(soc_cases.BI2SE3_POSCAR)

    assert_in_order(
        html,
        [
            "<h2>Structure Summary</h2>",
            "<h2>Calculation Definition</h2>",
            "<h3>Scientific Specification</h3>",
            'id="desired-output-select"',
            "<h3>Method Considerations</h3>",
            "<h3>Execution Resources</h3>",
            "data-build-calculation-button",
        ],
    )


def test_red_and_yellow_presentation_survive_the_move():
    html = built_page(soc_cases.BI2SE3_POSCAR, "electronic_dos")
    panel = html[html.index("<h3>Method Considerations</h3>"):html.index("<h3>Execution Resources</h3>")]

    # Red HSE06 + SOC omission card, with its alert role and text.
    assert 'class="method-consideration-card unsupported"' in panel
    assert 'data-method-consideration-presentation="unsupported"' in panel
    assert 'role="alert"' in panel
    assert '<span class="step-mark unsupported" aria-label="Unsupported combination">!</span>' in panel
    assert "HSE06 + SOC is not currently supported by BMD Compute." in panel
    # Yellow advisory card (van der Waals correction applied for layered Bi2Se3).
    assert 'data-method-consideration-presentation="advisory"' in panel
    assert '<span class="step-mark advisory" aria-label="Advisory">!</span>' in panel
    # The red warning above the Prepare/Submit forms is unchanged.
    assert html.count('data-unsupported-omission="soc_omitted_from_hse06_stages"') == 2
    # Styles are unchanged.
    for rule in (".method-consideration-card.unsupported {", ".step-mark.unsupported {", ".step-mark.advisory {"):
        assert rule in TEMPLATE_SOURCE


def test_si_page_has_no_considerations_panel_but_keeps_resource_order():
    html = analyzed_page(soc_cases.SI_POSCAR)

    assert "<h3>Method Considerations</h3>" not in html
    assert_in_order(
        html,
        ["<h3>Scientific Specification</h3>", "<h3>Execution Resources</h3>", "data-build-calculation-button"],
    )


# --- Form contract and keyboard order ------------------------------------------


def test_form_fields_identifiers_and_request_contract_are_unchanged():
    html = built_page(soc_cases.BI2SE3_POSCAR, "electronic_dos")
    form = calculation_form(html)
    parsed = controls(form)

    assert 'action="/build-calculation"' in form
    assert 'method="post"' in form
    assert sorted(set(parsed.names)) == sorted(
        {"structure", "fmt", "workflow_spec_json", "workflow", "cpus", "memory_gb", "walltime", "queue", "nodes"}
    )
    for identifier in (
        "calculation-review-form",
        "workflow-spec-json",
        "calculation-stage-options",
        "desired-output-select",
        "workflow-stage-list",
        "add-workflow-stage",
    ):
        assert f'id="{identifier}"' in html
    for field in ("cpus", "memory_gb", "walltime", "queue"):
        assert parsed.names.count(field) == 1
    assert re.search(r'type="hidden"\s+name="nodes"', form)


def test_considerations_panel_adds_no_form_controls_or_focus_stops():
    html = built_page(soc_cases.BI2SE3_POSCAR, "electronic_dos")
    form = calculation_form(html)
    panel = form[form.index('class="method-considerations '):form.index("<h3>Execution Resources</h3>")]
    parsed = controls(panel)

    assert parsed.names == []
    assert parsed.focus_order == []
    assert 'aria-live="polite"' in panel


def test_keyboard_order_follows_the_visual_order():
    html = built_page(soc_cases.BI2SE3_POSCAR, "electronic_dos")
    parsed = controls(calculation_form(html))

    assert parsed.tabindexes == []
    order = parsed.focus_order
    assert order[0] == "workflow"  # Desired Output select
    assert order.index("workflow") < order.index("stage-type") < order.index("add-workflow-stage")
    assert order.index("add-workflow-stage") < order.index("cpus")
    assert order[order.index("cpus"):] == ["cpus", "memory_gb", "walltime", "queue", "build"]


def test_custom_workflow_keyboard_order_reaches_advanced_options_before_resources():
    workflow = soc_cases.WorkflowSpec(
        [soc_cases.StageSpec(soc_cases.StageType.STATIC, soc_cases.Theory.PBE)],
        recipe="custom",
    )
    response = soc_cases.build_route(soc_cases.SI_POSCAR, workflow="custom", workflow_spec=workflow)
    order = controls(calculation_form(response.template.render(response.context))).focus_order

    assert order.index("workflow") < order.index("stage-theory") < order.index("modifier:soc")
    assert order.index("modifier:soc") < order.index("add-workflow-stage") < order.index("cpus")
    assert order[-5:] == ["cpus", "memory_gb", "walltime", "queue", "build"]


# --- Considerations follow workflow changes; responsive layout ----------------


def test_considerations_are_marked_stale_when_the_selection_changes():
    script = TEMPLATE_SOURCE[TEMPLATE_SOURCE.index("var unsupportedCombinations"):]

    assert "data-method-considerations-stale-notice" in TEMPLATE_SOURCE
    assert 'form.querySelector("[data-method-considerations]")' in script
    assert "function syncConsiderationsFreshness()" in script
    # Every selection change goes through updateHiddenWorkflow, which re-checks.
    update = script[script.index("function updateHiddenWorkflow()"):]
    update = update[:update.index("\n    }\n")]
    assert update.rstrip().endswith("syncConsiderationsFreshness();")
    # The built workflow is captured once, after the initial normalisation.
    assert "updateHiddenWorkflow();\n    builtWorkflowJson = hiddenWorkflow.value;\n});" in script
    assert '.method-considerations[data-stale="true"] .method-consideration-card {' in TEMPLATE_SOURCE


def test_a_rebuilt_workflow_renders_its_own_considerations():
    # Considerations are recomputed by the server for each built workflow:
    # Bi2Se3 Energy only applies SOC (no red card); DOS omits it (red card).
    energy = built_page(soc_cases.BI2SE3_POSCAR, "energy_only")
    dos = built_page(soc_cases.BI2SE3_POSCAR, "electronic_dos")

    assert 'data-method-consideration-presentation="unsupported"' not in energy
    assert "Spin-orbit coupling (SOC) has been included automatically" in energy
    assert 'data-method-consideration-presentation="unsupported"' in dos


def test_mobile_layout_stacks_the_calculation_panels():
    media = TEMPLATE_SOURCE[TEMPLATE_SOURCE.index("@media"):]
    single_column = media[:media.index("grid-template-columns: 1fr;")]
    for selector in (".configuration-grid", ".resource-grid", ".form-grid"):
        assert selector in single_column
