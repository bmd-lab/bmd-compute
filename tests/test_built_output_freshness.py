"""All outputs of a Build share one freshness state.

The Hardware Specification, Method Considerations, Calculation Summary,
Generated Inputs, Execution Summary and the submission controls
(``data-built-output``) describe the build that rendered the page. The browser
hides them together while any scientific field the build form submits differs
from that build, and shows them again only when the whole build-defining form
state matches it. A change to the hardware fields alone keeps the Hardware
Specification visible for editing and hides every other build output,
including Prepare and Submit, until Update Calculation re-plans the
calculation.

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
    # The single red omission warning is a card in the (hidden-together)
    # Method Considerations panel; no separate copies exist elsewhere.
    assert html.count('data-method-consideration-presentation="unsupported"') == 1
    assert html.index("<h3>Method Considerations</h3>") < html.index(
        'data-method-consideration-presentation="unsupported"'
    ) < html.index("<h2>Calculation Summary</h2>")
    assert "data-unsupported-omission" not in html
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
        "    builtScientificState = formState(true);\n"
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
  hardwareNotice: d.querySelector("[data-hardware-changed-notice]") ? !d.querySelector("[data-hardware-changed-notice]").hidden : null,
  hardware: d.querySelector("[data-hardware-specification]") ? !d.querySelector("[data-hardware-specification]").hidden : null,
  submitDisabled: d.querySelector("[data-submit-calculation-button]") ? d.querySelector("[data-submit-calculation-button]").disabled : null,
  form: Array.from(new w.FormData(d.getElementById("calculation-review-form")).keys()),
  values: Object.fromEntries(Array.from(new w.FormData(d.getElementById("calculation-review-form")).entries())),
  options: Array.from(d.getElementById("desired-output-select").options).map((o) => o.value),
  stageCards: d.querySelectorAll("#workflow-stage-list [data-workflow-stage]").length,
});
const out = [snapshot("initial")];
for (const step of steps) {
  if (step.op === "desired") { const s = d.getElementById("desired-output-select"); s.value = step.value; fire(s, "change"); }
  if (step.op === "select") { const s = d.querySelector(`#calculation-review-form [data-resource-control="${step.name}"]`); s.value = step.value; fire(s, "change"); }
  if (step.op === "input") { const s = d.querySelector(`#calculation-review-form [data-resource-control="${step.name}"]`); s.value = step.value; fire(s, "input"); }
  if (step.op === "custom") { d.querySelector("[data-hardware-custom-toggle]").click(); }
  if (step.op === "restore") { d.querySelector("[data-resource-restore]").click(); }
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


def hardware_only_changed(state):
    """Hardware edited: only the Hardware Specification stays visible."""

    return (
        state["hardware"] is True
        and state["hidden"].count(False) == 1
        and state["notice"] is False
        and state["hardwareNotice"] is True
    )


@requires_jsdom
def test_switching_desired_outputs_after_build_hides_all_outputs_until_restored(tmp_path):
    states = run_page(
        render(built_response()),
        [
            {"op": "desired", "value": "relaxed_energy"},
            {"op": "desired", "value": "electronic_band_structure"},
            {"op": "desired", "value": "electronic_dos", "label": "restored"},
        ],
        tmp_path,
    )

    assert len(states[0]["hidden"]) == 6  # hardware, considerations + four sections
    assert states[1]["hardware"] is False and states[1]["hardwareNotice"] is False
    assert all_shown(states[0])
    assert all_hidden(states[1]) and all_hidden(states[2])
    assert all_shown(states[3])


@requires_jsdom
def test_restoring_the_label_alone_is_not_enough_when_other_inputs_differ(tmp_path):
    states = run_page(
        render(built_response()),
        [
            {"op": "select", "name": "cpus", "value": "48", "label": "CPU override"},
            {"op": "desired", "value": "relaxed_energy"},
            {"op": "desired", "value": "electronic_dos", "label": "label restored, cpus still 48"},
            {"op": "restore", "label": "Restore Recommended"},
            {"op": "input", "name": "walltime", "value": "12:00:00"},
            {"op": "restore", "label": "Restore Recommended again"},
        ],
        tmp_path,
    )

    # A hardware override keeps the Hardware Specification editable and hides
    # Prepare/Submit (and every other build output) until it is re-planned.
    assert hardware_only_changed(states[1])
    # A scientific change hides the hardware recommendation too.
    assert all_hidden(states[2]) and states[2]["hardwareNotice"] is False
    assert hardware_only_changed(states[3])
    assert all_shown(states[4]) and states[4]["hardwareNotice"] is False
    assert hardware_only_changed(states[5])
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


FORM_FIELDS = sorted(
    {"structure", "fmt", "workflow_spec_json", "workflow", "cpus", "memory_gb", "walltime", "queue", "nodes"}
)
STANDARD_OUTPUTS = [
    option["value"] for option in main.calculation_form_options()["desired_outputs"] if option["value"] != "custom"
]


def posted_workflow(state) -> dict:
    """The workflow the browser would post: the selector and the hidden specification."""

    spec = json.loads(state["values"]["workflow_spec_json"])
    return {
        "workflow": state["values"].get("workflow"),
        "recipe": spec.get("recipe"),
        "label": spec.get("label"),
        "stage_types": [stage["stage_type"] for stage in spec.get("stages", [])],
    }


def managed_stage_types(desired_output: str) -> list[str]:
    return [stage.stage_type.value for stage in main.desired_output_workflow_spec(desired_output).stages]


def test_energy_only_is_not_a_standard_browser_output():
    assert STANDARD_OUTPUTS[0] == "relaxed_energy"
    assert "energy_only" not in STANDARD_OUTPUTS


@requires_jsdom
def test_form_submission_contract_is_unchanged(tmp_path):
    steps = [{"op": "desired", "value": value} for value in STANDARD_OUTPUTS + ["custom"]]
    states = run_page(render(built_response()), steps, tmp_path)

    for state in states:
        assert sorted(set(state["form"])) == FORM_FIELDS, state["label"]
        # A selection always posts both the Desired Output and its specification.
        assert state["values"]["workflow"]
        assert state["values"]["workflow_spec_json"]
    for step, state in zip(steps, states[1:]):
        assert state["values"]["workflow"] == step["value"]


@requires_jsdom
def test_legacy_energy_only_page_posts_the_same_form_contract(tmp_path):
    html = render(built_response(soc_cases.SI_POSCAR, workflow="energy_only"))
    states = run_page(
        html,
        [{"op": "desired", "value": "relaxed_energy"}, {"op": "desired", "value": "energy_only"}],
        tmp_path,
    )
    for state in states:
        assert sorted(set(state["form"])) == FORM_FIELDS, state["label"]


@requires_jsdom
def test_default_page_selects_relaxed_energy_and_offers_no_energy_only(tmp_path):
    response = main.analyze(soc_cases.request("/analyze"), structure=soc_cases.SI_POSCAR, fmt="poscar")
    (state,) = run_page(response.template.render(response.context), [], tmp_path)

    assert posted_workflow(state) == {
        "workflow": "relaxed_energy",
        "recipe": "relaxed_energy",
        "label": None,
        "stage_types": managed_stage_types("relaxed_energy"),
    }
    assert state["options"] == STANDARD_OUTPUTS + ["custom"]
    assert state["stageCards"] == 3


@requires_jsdom
def test_switching_among_standard_outputs_posts_each_managed_workflow(tmp_path):
    steps = [{"op": "desired", "value": value} for value in STANDARD_OUTPUTS[1:]]
    steps.append({"op": "desired", "value": "relaxed_energy", "label": "restored"})
    states = run_page(render(built_response(soc_cases.SI_POSCAR, workflow="relaxed_energy")), steps, tmp_path)

    assert all_shown(states[0])
    assert states[0]["options"] == STANDARD_OUTPUTS + ["custom"]
    for step, state in zip(steps, states[1:]):
        assert posted_workflow(state) == {
            "workflow": step["value"],
            "recipe": step["value"],
            "label": None,
            "stage_types": managed_stage_types(step["value"]),
        }, step
        assert state["stageCards"] == len(managed_stage_types(step["value"]))
    for state in states[1:-1]:
        assert all_hidden(state)
    assert all_shown(states[-1])
    assert states[-1]["values"] == states[0]["values"]


@requires_jsdom
def test_selecting_custom_keeps_the_stages_as_an_explicit_custom_workflow(tmp_path):
    states = run_page(
        render(built_response(soc_cases.SI_POSCAR, workflow="relaxed_energy")),
        [
            {"op": "desired", "value": "custom"},
            {"op": "desired", "value": "relaxed_energy", "label": "restored"},
        ],
        tmp_path,
    )

    # Custom starts from the current stages but is never a managed recipe.
    assert posted_workflow(states[1]) == {
        "workflow": "custom",
        "recipe": None,
        "label": None,
        "stage_types": managed_stage_types("relaxed_energy"),
    }
    assert all_hidden(states[1])
    assert all_shown(states[2]) and states[2]["values"] == states[0]["values"]


@requires_jsdom
def test_legacy_energy_only_page_keeps_its_identity_and_restores_its_build(tmp_path):
    response = built_response(soc_cases.SI_POSCAR, workflow="energy_only")
    states = run_page(
        render(response),
        [
            {"op": "desired", "value": "relaxed_energy"},
            {"op": "desired", "value": "relaxed_structure"},
            {"op": "desired", "value": "energy_only", "label": "restored from a standard output"},
            {"op": "desired", "value": "custom"},
            {"op": "desired", "value": "energy_only", "label": "restored from Custom"},
        ],
        tmp_path,
    )
    legacy = {"workflow": "energy_only", "recipe": "energy_only", "label": None, "stage_types": ["static"]}

    # Loaded explicitly: the legacy single point, shown as its own (legacy)
    # option after the standard menu, with the built form state.
    assert posted_workflow(states[0]) == legacy
    assert json.loads(states[0]["values"]["workflow_spec_json"]) == json.loads(
        response.context["selected_workflow"]["json"]
    )
    assert states[0]["options"] == STANDARD_OUTPUTS + ["custom", "energy_only"]
    assert all_shown(states[0])
    assert posted_workflow(states[1])["workflow"] == "relaxed_energy" and all_hidden(states[1])
    assert posted_workflow(states[2])["workflow"] == "relaxed_structure" and all_hidden(states[2])
    # Choosing the legacy option again restores its own single static stage,
    # not the stages of the previous selection, and the build becomes current.
    assert posted_workflow(states[3]) == legacy and states[3]["stageCards"] == 1
    assert all_shown(states[3]) and states[3]["values"] == states[0]["values"]
    assert posted_workflow(states[4])["workflow"] == "custom"
    assert posted_workflow(states[4])["recipe"] is None and all_hidden(states[4])
    assert posted_workflow(states[5]) == legacy
    assert all_shown(states[5]) and states[5]["values"] == states[0]["values"]


@requires_jsdom
def test_legacy_energy_only_hardware_override_requires_update_before_prepare(tmp_path):
    states = run_page(
        render(built_response(soc_cases.SI_POSCAR, workflow="energy_only")),
        [
            {"op": "custom", "label": "open Custom hardware"},
            {"op": "select", "name": "cpus", "value": "48", "label": "CPU override"},
            {"op": "restore", "label": "Restore Recommended"},
        ],
        tmp_path,
    )

    assert hardware_only_changed(states[2])
    assert posted_workflow(states[2])["workflow"] == "energy_only"
    assert states[2]["values"]["cpus"] == "48"
    assert all_shown(states[3]) and states[3]["values"] == states[0]["values"]


@requires_jsdom
def test_browser_update_of_a_legacy_page_rebuilds_energy_only_with_the_override(tmp_path):
    built = built_response(soc_cases.SI_POSCAR, workflow="energy_only")
    states = run_page(
        render(built),
        [
            {"op": "desired", "value": "relaxed_energy"},
            {"op": "desired", "value": "energy_only"},
            {"op": "custom"},
            {"op": "select", "name": "cpus", "value": "48"},
        ],
        tmp_path,
    )
    posted = states[-1]["values"]
    response = main.build_workflow(
        soc_cases.request("/build-workflow"),
        structure=posted["structure"],
        fmt=posted["fmt"],
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=posted["cpus"],
        memory_gb=posted["memory_gb"],
        walltime=posted["walltime"],
        queue=posted["queue"],
        workflow_spec_json=posted["workflow_spec_json"],
        workflow=posted["workflow"],
        method=None,
    )

    assert response.status_code == 200, response.context.get("calculation_error")
    selected = response.context["selected_workflow"]
    assert selected["desired_output"] == "energy_only"
    assert selected["hidden_desired_output_label"] == "Energy only (legacy)"
    workflow = response.context["submission_spec"]["flow_spec"]["workflow_spec"]
    assert workflow == built.context["submission_spec"]["flow_spec"]["workflow_spec"]
    assert workflow["recipe"] == "energy_only"
    assert response.context["submission_spec"]["resources"]["ntasks"] == 48
    assert response.context["selected_resources"]["planned_token"] != built.context["selected_resources"]["planned_token"]


@requires_jsdom
@pytest.mark.parametrize(
    ("workflow", "away"),
    [("energy_only", "relaxed_structure"), ("relaxed_energy", "relaxed_structure"), ("relaxed_energy", "custom")],
)
def test_prepared_run_is_hidden_with_the_build_and_returns_unchanged(tmp_path, workflow, away):
    response = built_response(soc_cases.SI_POSCAR, workflow=workflow)
    html = render(
        response,
        remote_preparation=remembered_successful_preparation(response.context["submission_spec"]),
    )
    states = run_page(
        html,
        [
            {"op": "desired", "value": away},
            {"op": "desired", "value": workflow, "label": "restored"},
        ],
        tmp_path,
    )

    # Submit was enabled for the prepared attempt; editing hides it with the
    # rest of the build, and restoring the build shows it again unchanged.
    assert states[0]["submitDisabled"] is False and all_shown(states[0])
    assert all_hidden(states[1])
    assert states[2]["submitDisabled"] is False and all_shown(states[2])
    assert states[2]["values"] == states[0]["values"]


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
