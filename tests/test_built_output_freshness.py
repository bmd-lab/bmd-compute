"""All outputs of a Build share one freshness state.

Method Considerations, Calculation Summary, Generated Inputs, Execution
Summary and the submission controls (``data-built-output``) describe the build
that rendered the page. The browser hides them together while any field the
build form submits differs from that build, and shows them again only when
the whole build-defining form state matches it.

The behavioural tests run the page's own script in jsdom when Node.js and
jsdom are available (``node`` on PATH, jsdom resolvable, e.g. via
NODE_PATH); otherwise they are skipped and the static contract tests still run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import main
import test_automatic_soc as soc_cases
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.remote_preparation import remembered_successful_preparation


TEMPLATE_SOURCE = Path("templates/index.html").read_text(encoding="utf-8")
BUILT_SECTIONS = (
    "<h2>Calculation Summary</h2>",
    "<h2>Generated Inputs</h2>",
    "<h2>Execution Summary</h2>",
    "<h2>Ready for Submission</h2>",
)


def built_response(poscar=soc_cases.BI2SE3_POSCAR, *, workflow="electronic_dos", workflow_spec=None):
    response = soc_cases.build_route(poscar, workflow=workflow, workflow_spec=workflow_spec)
    assert response.status_code == 200
    return response


def render(response, **overrides) -> str:
    context = dict(response.context)
    context.update(overrides)
    return response.template.render(context)


def section_opening(html: str, heading: str) -> str:
    index = html.index(heading)
    return html[html.rindex("<section", 0, index):index]


# --- Static contract --------------------------------------------------------------


def test_every_resolved_output_is_marked_and_nothing_else_is():
    html = render(built_response())

    for heading in BUILT_SECTIONS:
        assert "data-built-output" in section_opening(html, heading), heading
    panel_open = html[html.rindex("<div", 0, html.index("<h3>Method Considerations</h3>")):]
    assert "data-built-output" in panel_open[:200]
    # The selection itself and the structure are never hidden.
    for heading in ("<h2>Structure Summary</h2>", "<h2>Calculation Definition</h2>"):
        assert "data-built-output" not in section_opening(html, heading)
    # Both pre-submission red warnings live inside hidden-together sections.
    marker = 'data-unsupported-omission="soc_omitted_from_hse06_stages"'
    for position in (html.index(marker), html.rindex(marker)):
        section = html[html.rindex("<section", 0, position):position]
        assert "data-built-output" in section[:60]
    assert html.count('data-built-output-notice role="status"') == 1


def test_monitoring_of_a_submitted_run_is_not_a_hideable_build_output():
    response = built_response(soc_cases.SI_POSCAR, workflow="energy_only")
    html = render(
        response,
        remote_preparation=remembered_successful_preparation(response.context["submission_spec"]),
        submission_result={
            "status": "success",
            "job_id": "123456",
            "queue_status": "PENDING",
            "job_record": {"run_dir": "/bmd-db/guest/flows/run", "remote_state_path": None},
        },
    )

    assert "<h2>Monitoring</h2>" in html
    assert "data-built-output" not in section_opening(html, "<h2>Monitoring</h2>")
    assert "The calculation already prepared or submitted is unchanged" in html


def test_page_without_a_build_has_no_freshness_notice():
    response = main.analyze(soc_cases.request("/analyze"), structure=soc_cases.BI2SE3_POSCAR, fmt="poscar")
    html = response.template.render(response.context)
    assert 'data-built-output-notice role="status"' not in html
    assert '<section class="stage" data-built-output>' not in html


def test_failed_build_renders_no_previous_outputs():
    custom = WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC})], recipe="custom")
    response = soc_cases.build_route(soc_cases.BI2SE3_POSCAR, workflow="custom", workflow_spec=custom)
    html = response.template.render(response.context)

    assert response.status_code == 400
    assert '<section class="stage" data-built-output>' not in html
    assert 'data-built-output-notice role="status"' not in html
    assert "<h3>Method Considerations</h3>" not in html


def test_script_compares_the_whole_form_state_and_listens_to_every_field():
    script = TEMPLATE_SOURCE[TEMPLATE_SOURCE.index("var unsupportedCombinations"):]

    assert 'document.querySelectorAll("[data-built-output]")' in script
    assert "new FormData(form).forEach" in script
    update = script[script.index("function updateHiddenWorkflow()"):]
    update = update[:update.index("\n    }\n")]
    assert update.rstrip().endswith("syncBuiltOutputs();")
    assert (
        "updateHiddenWorkflow();\n    builtFormState = formState();\n"
        '    form.addEventListener("change", syncBuiltOutputs);\n'
        '    form.addEventListener("input", syncBuiltOutputs);\n});'
    ) in script
    assert "syncConsiderationsFreshness" not in script


# --- Behaviour in a browser DOM (jsdom) ---------------------------------------------


def _jsdom_available() -> bool:
    if shutil.which("node") is None:
        return False
    probe = subprocess.run(["node", "-e", "require('jsdom')"], capture_output=True, text=True)
    return probe.returncode == 0


requires_jsdom = pytest.mark.skipif(not _jsdom_available(), reason="Node.js with jsdom is not available")

_HARNESS = r"""
const fs = require("fs");
const { JSDOM } = require("jsdom");
const html = fs.readFileSync(process.argv[2], "utf8");
const steps = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const dom = new JSDOM(html, { runScripts: "dangerously", virtualConsole: new (require("jsdom").VirtualConsole)() });
const w = dom.window, d = w.document;
d.dispatchEvent(new w.Event("DOMContentLoaded"));
const fire = (el, type) => el.dispatchEvent(new w.Event(type, { bubbles: true }));
const card = (i) => d.querySelectorAll("#workflow-stage-list [data-workflow-stage]")[i];
const snapshot = (label) => ({
  label,
  hidden: Array.from(d.querySelectorAll("[data-built-output]")).map((e) => e.hidden),
  notice: d.querySelector("[data-built-output-notice]") ? !d.querySelector("[data-built-output-notice]").hidden : null,
  submitDisabled: d.querySelector("[data-submit-calculation-button]") ? d.querySelector("[data-submit-calculation-button]").disabled : null,
  form: Array.from(new w.FormData(d.getElementById("calculation-review-form")).keys()),
});
const out = [snapshot("initial")];
for (const step of steps) {
  if (step.op === "desired") { const s = d.getElementById("desired-output-select"); s.value = step.value; fire(s, "change"); }
  if (step.op === "select") { const s = d.querySelector(`#calculation-review-form select[name="${step.name}"]`); s.value = step.value; fire(s, "change"); }
  if (step.op === "input") { const s = d.querySelector(`#calculation-review-form input[name="${step.name}"]`); s.value = step.value; fire(s, "input"); }
  if (step.op === "theory") { const s = card(step.stage).querySelector("[data-stage-theory]"); s.value = step.value; fire(s, "change"); }
  if (step.op === "modifier") { const s = card(step.stage).querySelector(`[data-stage-modifier][value="${step.value}"]`); s.checked = step.checked; fire(s, "change"); }
  if (step.op === "add") { d.getElementById("add-workflow-stage").click(); }
  if (step.op === "remove") { card(step.stage).querySelector("[data-remove-stage]").click(); }
  out.push(snapshot(step.label || step.op));
}
process.stdout.write(JSON.stringify(out));
"""


def run_page(html: str, steps: list[dict], tmp_path) -> list[dict]:
    page = tmp_path / "page.html"
    page.write_text(re.sub(r'<script src="[^"]*"></script>', "", html), encoding="utf-8")
    plan = tmp_path / "steps.json"
    plan.write_text(json.dumps(steps), encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    result = subprocess.run(
        ["node", str(harness), str(page), str(plan)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def all_shown(state):
    return state["hidden"] and not any(state["hidden"]) and state["notice"] is False


def all_hidden(state):
    return state["hidden"] and all(state["hidden"]) and state["notice"] is True


@requires_jsdom
def test_switching_desired_outputs_after_build_hides_all_outputs_until_restored(tmp_path):
    states = run_page(
        render(built_response()),
        [
            {"op": "desired", "value": "energy_only"},
            {"op": "desired", "value": "electronic_band_structure"},
            {"op": "desired", "value": "electronic_dos", "label": "restored"},
        ],
        tmp_path,
    )

    assert len(states[0]["hidden"]) == 5  # considerations + four sections
    assert all_shown(states[0])
    assert all_hidden(states[1]) and all_hidden(states[2])
    assert all_shown(states[3])


@requires_jsdom
def test_restoring_the_label_alone_is_not_enough_when_other_inputs_differ(tmp_path):
    states = run_page(
        render(built_response()),
        [
            {"op": "select", "name": "cpus", "value": "48"},
            {"op": "desired", "value": "energy_only"},
            {"op": "desired", "value": "electronic_dos", "label": "label restored, cpus still 48"},
            {"op": "select", "name": "cpus", "value": "24", "label": "fully restored"},
            {"op": "input", "name": "walltime", "value": "12:00:00"},
            {"op": "input", "name": "walltime", "value": "72:00:00", "label": "walltime restored"},
        ],
        tmp_path,
    )

    assert all_hidden(states[1])
    assert all_hidden(states[3])
    assert all_shown(states[4])
    assert all_hidden(states[5])
    assert all_shown(states[6])


@requires_jsdom
def test_editing_custom_stages_and_treatment_options_hides_all_outputs(tmp_path):
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)], recipe="custom")
    html = render(built_response(soc_cases.SI_POSCAR, workflow="custom", workflow_spec=workflow))
    states = run_page(
        html,
        [
            {"op": "modifier", "stage": 0, "value": "soc", "checked": True, "label": "SOC on"},
            {"op": "modifier", "stage": 0, "value": "soc", "checked": False, "label": "SOC off"},
            {"op": "theory", "stage": 0, "value": "hse06", "label": "HSE06"},
            {"op": "theory", "stage": 0, "value": "pbe", "label": "PBE again"},
            {"op": "add", "label": "stage added"},
            {"op": "remove", "stage": 1, "label": "stage removed"},
        ],
        tmp_path,
    )

    assert all_shown(states[0])
    assert all_hidden(states[1])
    assert all_shown(states[2])
    assert all_hidden(states[3])
    assert all_shown(states[4])
    assert all_hidden(states[5])
    assert all_shown(states[6])


@requires_jsdom
def test_form_submission_contract_is_unchanged(tmp_path):
    states = run_page(render(built_response()), [{"op": "desired", "value": "energy_only"}], tmp_path)
    for state in states:
        assert sorted(set(state["form"])) == sorted(
            {"structure", "fmt", "workflow_spec_json", "workflow", "cpus", "memory_gb", "walltime", "queue", "nodes"}
        )


@requires_jsdom
def test_prepared_run_is_hidden_with_the_build_and_returns_unchanged(tmp_path):
    response = built_response(soc_cases.SI_POSCAR, workflow="energy_only")
    html = render(
        response,
        remote_preparation=remembered_successful_preparation(response.context["submission_spec"]),
    )
    states = run_page(
        html,
        [
            {"op": "desired", "value": "relaxed_structure"},
            {"op": "desired", "value": "energy_only", "label": "restored"},
        ],
        tmp_path,
    )

    # Submit was enabled for the prepared attempt; editing hides it with the
    # rest of the build, and restoring the build shows it again unchanged.
    assert states[0]["submitDisabled"] is False and all_shown(states[0])
    assert all_hidden(states[1])
    assert states[2]["submitDisabled"] is False and all_shown(states[2])


@requires_jsdom
def test_submitted_run_keeps_its_monitoring_while_build_outputs_hide(tmp_path):
    response = built_response(soc_cases.SI_POSCAR, workflow="energy_only")
    html = render(
        response,
        remote_preparation=remembered_successful_preparation(response.context["submission_spec"]),
        submission_result={
            "status": "success",
            "job_id": "123456",
            "queue_status": "PENDING",
            "job_record": {"run_dir": "/bmd-db/guest/flows/run", "remote_state_path": None},
        },
    )
    states = run_page(html, [{"op": "desired", "value": "relaxed_structure"}], tmp_path)

    assert states[0]["submitDisabled"] is True  # already submitted
    assert all_hidden(states[1])
    assert states[1]["submitDisabled"] is True
