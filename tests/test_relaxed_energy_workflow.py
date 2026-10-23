"""Relaxed Energy Desired Output (E1).

Relaxed Energy is two PBE Geometry Optimisations followed by an accurate PBE
Static Energy, run in one allocation. It reuses the existing stage definitions
unchanged: each stage starts from the previous stage's final structure, the
automatic treatments apply exactly as they do to the same stages elsewhere,
and the energy is read from the final static while the structure is the one
the second relax produced.

The legacy ``energy_only`` identifier keeps its single-point meaning for the
machine API, BMD Run and historical records, and is only hidden from the
browser's Desired Output menu.
"""

from __future__ import annotations

import json
import re
import warnings

import pytest
from starlette.requests import Request

import main
import test_automatic_soc as soc_cases
import test_input_reference_contract as reference_cases
import test_results as results_cases
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    BROWSER_HIDDEN_DESIRED_OUTPUTS,
    calculation_form_options,
    desired_output_from_workflow_spec,
    desired_output_label,
    desired_output_workflow_spec,
    workflow_result_stage_directory,
    workflow_stage_directories,
)
from backend.calculations.resources import walltime_seconds
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure
from backend.results import parse_vasp_result_files, result_stage_directory_from_submission_spec
from backend.workflow_summary import calculation_plan_from_workflow_spec
from backend.workflows import build_atomate2_flow_for_workflow_spec
from compute_api import plan_digest as plan_digest_module
from compute_api.plan_digest import compute_plan_digest
from compute_api.projection import plan_stage_previews


SI_POSCAR = reference_cases.SI_POSCAR
NIO_POSCAR = reference_cases.NIO_POSCAR
BI2SE3_POSCAR = soc_cases.BI2SE3_POSCAR
FE_POSCAR = """Fe
2.87
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Fe
1
direct
0.0 0.0 0.0
"""

PRE_EXISTING_DESIRED_OUTPUTS = (
    "energy_only",
    "relaxed_structure",
    "electronic_dos",
    "electronic_band_structure",
    "phonons",
)


def _pairs(workflow: WorkflowSpec) -> list[tuple[str, str]]:
    return [(stage.stage_type.value, stage.theory.value) for stage in workflow.stages]


def _plan(structure=SI_POSCAR, desired_output="relaxed_energy", custom=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main.plan_calculation_request(
            structure_text=structure,
            fmt="poscar",
            desired_output=None if custom is not None else desired_output,
            custom_workflow=custom,
            timestamp="20260101-120000",
            submission_attempt_id="7f0c2a52-3d61-4c1e-9a55-0e1e5c3a7b11",
        )


def _stage_material(plan):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        material = plan_digest_module.stage_input_material(plan_stage_previews(plan))
    return [{key: value for key, value in stage.items() if key != "index"} for stage in material]


def _resolved_modifiers(plan):
    return [sorted(modifier.value for modifier in stage.modifiers) for stage in plan.workflow_spec.stages]


def _build(structure=SI_POSCAR, *, workflow, workflow_spec_json=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.build_workflow(
            Request({"type": "http", "method": "POST", "path": "/build-workflow", "headers": []}),
            structure=structure,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            cpus=None,
            memory_gb=None,
            walltime=None,
            queue=None,
            workflow_spec_json=workflow_spec_json,
            workflow=workflow,
            method=None,
        )
    assert response.status_code == 200, response.context.get("calculation_error")
    return response


def _select_block(html: str) -> str:
    start = html.index('id="desired-output-select"')
    return html[start:html.index("</select>", start)]


# ------------------------------------------------- 1. composition and order


def test_relaxed_energy_is_two_pbe_relaxes_then_a_pbe_static():
    workflow = desired_output_workflow_spec("relaxed_energy")

    assert _pairs(workflow) == [("relax", "pbe"), ("relax", "pbe"), ("static", "pbe")]
    assert workflow.recipe == "relaxed_energy"
    assert all(not stage.modifiers for stage in workflow.stages)
    assert desired_output_from_workflow_spec(workflow) == "relaxed_energy"
    assert calculation_plan_from_workflow_spec(workflow) == [
        "Geometry Optimisation",
        "Geometry Optimisation",
        "Static Energy",
    ]
    assert workflow_stage_directories(workflow) == ("stage_01", "stage_02", "stage_03")


def test_relaxed_energy_is_the_first_browser_option_with_its_label_and_description():
    options = calculation_form_options()["desired_outputs"]
    first = options[0]

    assert first["value"] == "relaxed_energy"
    assert first["label"] == "Relaxed Energy"
    assert first["description"] == "Optimises the crystal structure and calculates its final total energy."
    assert first["workflow_spec"] == desired_output_workflow_spec("relaxed_energy").to_dict()
    assert desired_output_label("relaxed_energy") == "Relaxed Energy"


# --------------------------------------------- 2. structure chaining (flow graph)


def test_each_stage_starts_from_the_previous_stage_final_structure():
    flow = build_atomate2_flow_for_workflow_spec(
        "initial-structure",
        desired_output_workflow_spec("relaxed_energy"),
        label="Si",
        resources={"ntasks": 24},
    )
    first_relax, second_relax, static = flow.jobs

    assert [job.name for job in flow.jobs] == ["stage_01", "stage_02", "stage_03"]
    assert flow.metadata["bmd_stage_directories"] == ("stage_01", "stage_02", "stage_03")
    assert first_relax.function_args == ("initial-structure",)

    second_input = second_relax.function_args[0]
    assert second_input.uuid == first_relax.uuid
    assert second_input.attributes == (("a", "structure"),)

    # The final static receives the structure produced by the SECOND relax,
    # not the input structure and not the first relax's structure.
    static_input = static.function_args[0]
    assert static_input.uuid == second_relax.uuid
    assert static_input.uuid != first_relax.uuid
    assert static_input.attributes == (("a", "structure"),)
    # As for every PBE static that follows a stage, it also reads that
    # stage's directory (the existing chaining, unchanged).
    prev_dir = static.function_kwargs["prev_dir"]
    assert prev_dir.uuid == second_relax.uuid
    assert prev_dir.attributes == (("a", "dir_name"),)
    # No stage copies charge densities or wavefunctions (unchanged policy).
    assert all(
        artifact == {"write_wavecar": False, "copy_from_previous": []}
        for artifact in flow.metadata["bmd_stage_artifacts"]
    )


# --------------------------------------------------------- 3. treatments


def test_plain_pbe_structure_receives_no_treatments():
    plan = _plan(SI_POSCAR)
    assert _resolved_modifiers(plan) == [[], [], []]


def test_dft_u_and_spin_apply_to_every_stage():
    plan = _plan(NIO_POSCAR)
    assert _resolved_modifiers(plan) == [["dft_u", "spin_polarized"]] * 3


def test_spin_applies_to_every_stage_for_a_magnetic_metal():
    plan = _plan(FE_POSCAR)
    assert all("spin_polarized" in modifiers for modifiers in _resolved_modifiers(plan))


def test_d3_on_every_stage_and_soc_only_on_the_final_static():
    plan = _plan(BI2SE3_POSCAR)

    assert _resolved_modifiers(plan) == [
        ["dispersion"],
        ["dispersion"],
        ["dispersion", "soc"],
    ]
    executables = [item["executable"] for item in plan.generated_inputs["vasp_executables"]]
    assert executables == ["vasp_std", "vasp_std", "vasp_ncl"]


@pytest.mark.parametrize("structure", [SI_POSCAR, NIO_POSCAR, BI2SE3_POSCAR, FE_POSCAR])
def test_stages_reuse_existing_definitions_with_no_new_incar_defaults(structure):
    # Relaxes 1-2 are exactly the Relaxed Structure stages and stage 3 is
    # exactly the Energy only static, input for input (INCAR, KPOINTS, POSCAR,
    # POTCAR symbols, executable) and treatment for treatment.
    relaxed_energy = _stage_material(_plan(structure, "relaxed_energy"))
    relaxed_structure = _stage_material(_plan(structure, "relaxed_structure"))
    energy_only = _stage_material(_plan(structure, "energy_only"))

    assert relaxed_energy[:2] == relaxed_structure
    assert relaxed_energy[2:] == energy_only
    assert _resolved_modifiers(_plan(structure, "relaxed_energy")) == (
        _resolved_modifiers(_plan(structure, "relaxed_structure"))
        + _resolved_modifiers(_plan(structure, "energy_only"))
    )


# ------------------------------------------------- 4. generated-input preview


def test_generated_inputs_preview_all_three_stages():
    preview = preview_generated_inputs(
        parse_structure(SI_POSCAR, "poscar"),
        desired_output_workflow_spec("relaxed_energy"),
        resources={"ntasks": 24},
    )
    headers = [line for line in preview["incar"].splitlines() if line.startswith("# Stage")]

    assert headers == [
        "# Stage 1 - Geometry Optimisation (PBE)",
        "# Stage 2 - Geometry Optimisation (PBE)",
        "# Stage 3 - Static Energy (PBE)",
    ]
    stage_texts = preview["incar"].split("# Stage ")[1:]
    assert "IBRION = 2" in stage_texts[0] and "ISIF = 3" in stage_texts[0]
    assert "IBRION = 2" in stage_texts[1] and "ISIF = 3" in stage_texts[1]
    assert "NSW = 0" in stage_texts[2]
    assert "ISMEAR = -5" in stage_texts[2]
    assert [item["stage"] for item in preview["vasp_executables"]] == [1, 2, 3]


def test_build_page_previews_all_three_stages():
    response = _build(workflow="relaxed_energy")
    context = response.context

    assert context["selected_workflow"]["desired_output"] == "relaxed_energy"
    assert [stage["stage_type"] for stage in context["selected_workflow"]["stages"]] == [
        "relax",
        "relax",
        "static",
    ]
    assert context["generated_inputs"]["incar"].count("# Stage ") == 3
    assert context["calculation"]["calculation_plan"] == [
        "Geometry Optimisation",
        "Geometry Optimisation",
        "Static Energy",
    ]


# ------------------------------------------ 5. final energy and structure


def test_results_are_read_from_the_final_static_stage():
    plan = _plan(SI_POSCAR)

    assert workflow_result_stage_directory(plan.workflow_spec) == "stage_03"
    assert result_stage_directory_from_submission_spec(plan.submission_spec) == "stage_03"


def test_final_energy_comes_from_the_static_and_no_ionic_convergence_is_claimed():
    # The parser reads the final stage's vasprun: its final energy is the
    # static energy and its final structure is the (unchanged, NSW = 0)
    # structure the second relax produced.
    files = {
        "contcar": {"path": "/remote/stage_03/CONTCAR", "text": "contcar"},
        "outcar": {"path": "/remote/stage_03/OUTCAR", "text": "outcar"},
        "vasprun": {"path": "/remote/stage_03/vasprun.xml", "text": "<modeling />"},
    }
    context = {
        **results_cases.monitoring_success,
        "workflow_spec": desired_output_workflow_spec("relaxed_energy").to_dict(),
    }
    with results_cases.fake_pymatgen_results_modules():
        result = parse_vasp_result_files(files, context)

    assert result["status"] == "success"
    assert result["final_energy_ev"] == -4.0
    assert result["final_formula"] == "Si"
    assert result["electronic_convergence"] is True
    # A static final stage reports no ionic convergence (its relaxes must
    # have finished successfully for the static to run at all).
    assert result["ionic_convergence"] is None
    assert result["converged_ionic"] is None


def test_remote_and_local_result_parsers_agree_for_relaxed_energy(tmp_path):
    context = {
        **results_cases.monitoring_success,
        "workflow_spec": desired_output_workflow_spec("relaxed_energy").to_dict(),
    }
    results_cases.assert_remote_parser_matches_local_parser(
        tmp_path, context, ("contcar", "outcar", "vasprun")
    )


# -------------------------------------------------------------- 6. resources


def test_one_allocation_sized_for_all_three_stages():
    relaxed_energy = _plan(BI2SE3_POSCAR, "relaxed_energy")
    relaxed_structure = _plan(BI2SE3_POSCAR, "relaxed_structure")
    energy_only = _plan(BI2SE3_POSCAR, "energy_only")

    estimate = relaxed_energy.submission_spec["flow_spec"]["resource_selection"]["walltime_estimate"]
    assert [stage["stage_type"] for stage in estimate["stages"]] == ["relax", "relax", "static"]

    # One SLURM allocation, single node, for the whole workflow.
    assert relaxed_energy.submission_spec["resources"]["nodes"] == 1
    assert len(relaxed_energy.submission_spec["flow_spec"]["workflow_spec"]["stages"]) == 3

    hours = {
        name: walltime_seconds(plan.execution_resources.walltime)
        for name, plan in {
            "relaxed_energy": relaxed_energy,
            "relaxed_structure": relaxed_structure,
            "energy_only": energy_only,
        }.items()
    }
    assert hours["relaxed_energy"] > hours["relaxed_structure"]
    assert hours["relaxed_energy"] > hours["energy_only"]


def test_per_stage_estimates_match_the_component_workflows():
    def stage_seconds(plan):
        estimate = plan.submission_spec["flow_spec"]["resource_selection"]["walltime_estimate"]
        return [stage["seconds"] for stage in estimate["stages"]]

    relaxed_energy = stage_seconds(_plan(SI_POSCAR, "relaxed_energy"))
    assert relaxed_energy[:2] == stage_seconds(_plan(SI_POSCAR, "relaxed_structure"))
    assert relaxed_energy[2:] == stage_seconds(_plan(SI_POSCAR, "energy_only"))


# ------------------------------------------------- 7. browser / API parity


def test_browser_build_and_api_plan_resolve_the_same_relaxed_energy_digest():
    response = _build(BI2SE3_POSCAR, workflow="relaxed_energy")
    ui = response.context
    api_plan = _plan(BI2SE3_POSCAR)
    structure_obj = parse_structure(BI2SE3_POSCAR, "poscar")
    ui_plan = main.CalculationPlan(
        structure=structure_obj,
        workflow_spec=main.WorkflowSpec.from_dict(ui["submission_spec"]["flow_spec"]["workflow_spec"]),
        default_treatment_resolution=ui["default_treatment_resolution"],
        method_considerations=ui["method_considerations"],
        execution_resources=ui["selected_resources"],
        summary=ui["summary"],
        generated_inputs=ui["generated_inputs"],
        submission_spec=ui["submission_spec"],
    )

    assert ui_plan.workflow_spec.to_dict() == api_plan.workflow_spec.to_dict()
    assert _stage_material(ui_plan) == _stage_material(api_plan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ui_digest = compute_plan_digest(
            structure=structure_obj,
            submission_spec=ui["submission_spec"],
            stage_previews=plan_stage_previews(ui_plan),
        )
        api_digest = compute_plan_digest(
            structure=api_plan.structure,
            submission_spec=api_plan.submission_spec,
            stage_previews=plan_stage_previews(api_plan),
        )
    assert ui_digest == api_digest


# ------------------------------------------- 8. historical energy_only unchanged


def test_energy_only_keeps_its_single_point_meaning():
    workflow = desired_output_workflow_spec("energy_only")

    assert _pairs(workflow) == [("static", "pbe")]
    assert workflow.recipe == "energy_only"
    assert desired_output_label("energy_only") == "Energy only"
    assert "energy_only" in BROWSER_HIDDEN_DESIRED_OUTPUTS
    assert "energy_only" not in {option["value"] for option in calculation_form_options()["desired_outputs"]}

    plan = _plan(SI_POSCAR, "energy_only")
    assert _pairs(plan.workflow_spec) == [("static", "pbe")]
    # A single-stage run is still read from the run directory itself.
    assert result_stage_directory_from_submission_spec(plan.submission_spec) is None
    assert plan.submission_spec["label"] == "vasp_run_static"
    assert len(_stage_material(plan)) == 1


def test_custom_single_static_is_unchanged_and_stays_custom_in_the_browser():
    custom = {"stages": [{"stage_type": "static", "theory": "pbe", "modifiers": [], "options": {}}]}
    custom_plan = _plan(SI_POSCAR, custom=custom)
    legacy_plan = _plan(SI_POSCAR, "energy_only")

    # The same science as the legacy single point.
    assert _stage_material(custom_plan) == _stage_material(legacy_plan)
    # A Custom single static is not relabelled as the hidden legacy output.
    context = main.page_context(
        selected_workflow=WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)], recipe=None)
    )
    assert context["selected_workflow"]["desired_output"] == "custom"
    assert context["selected_workflow"]["hidden_desired_output_label"] is None


def test_legacy_energy_only_form_post_builds_and_displays_energy_only():
    # BMD Run's v1 browser path and historical forms post workflow=energy_only.
    response = _build(SI_POSCAR, workflow="energy_only")
    html = response.template.render(response.context)
    select_block = _select_block(html)

    assert response.context["selected_workflow"]["desired_output"] == "energy_only"
    assert [stage["stage_type"] for stage in response.context["selected_workflow"]["stages"]] == ["static"]
    # The selector shows the legacy output as the selected option, never a
    # different Desired Output than the one that was built.
    assert re.search(r'<option value="energy_only" selected data-workflow-spec="[^"]*">\s*Energy only \(legacy\)\s*</option>', select_block)
    assert select_block.count("selected") == 1
    # Choosing the legacy option again in the browser restores its own managed
    # single-point workflow, which it carries on the option.
    assert json.loads(response.context["selected_workflow"]["hidden_desired_output_workflow_spec"]) == (
        desired_output_workflow_spec("energy_only").to_dict()
    )
    # Prepare and Submit carry the legacy identifier unchanged.
    assert html.count('<input type="hidden" name="workflow" value="energy_only">') == 2


# ----------------------------------------------- 9. BMD Run compatibility


def test_machine_api_still_accepts_every_pre_existing_identifier():
    for desired_output in PRE_EXISTING_DESIRED_OUTPUTS:
        assert desired_output_workflow_spec(desired_output) is not None, desired_output
    plan = _plan(SI_POSCAR, "energy_only")
    assert plan.submission_spec["status"]


def test_relaxed_energy_is_the_only_new_identifier():
    from backend.calculations.registry import _desired_output_workflows

    identifiers = {value for value, _, _, _ in _desired_output_workflows()}
    assert identifiers - set(PRE_EXISTING_DESIRED_OUTPUTS) == {"relaxed_energy", "custom"}


# ------------------------------------------ 10. existing workflows unchanged


def test_existing_desired_outputs_keep_their_stages():
    assert _pairs(desired_output_workflow_spec("relaxed_structure")) == [("relax", "pbe"), ("relax", "pbe")]
    assert _pairs(desired_output_workflow_spec("electronic_dos")) == [
        ("relax", "pbe"),
        ("static", "hse06"),
        ("dos", "hse06"),
    ]
    assert _pairs(desired_output_workflow_spec("electronic_band_structure")) == [
        ("relax", "pbe"),
        ("static", "hse06"),
        ("band_structure", "hse06"),
    ]
    assert _pairs(desired_output_workflow_spec("phonons")) == [("relax", "pbe"), ("phonon_forces", "pbe")]
    for name in PRE_EXISTING_DESIRED_OUTPUTS:
        assert desired_output_from_workflow_spec(desired_output_workflow_spec(name)) == name


@pytest.mark.parametrize(
    ("stages", "expected"),
    [
        (["relax", "relax", "static"], "custom"),
        (["static"], "custom"),
        # No Custom workflow is relabelled from its stage shape (see
        # test_custom_workflow_identity.py for every shape).
        (["relax", "relax"], "custom"),
    ],
)
def test_custom_workflows_are_not_relabelled_as_relaxed_energy_or_energy_only(stages, expected):
    # The browser sends a Custom workflow with no recipe. A Custom
    # relax -> relax -> static must stay Custom after Build so that Prepare and
    # Submit carry workflow=custom, not a managed Desired Output that would be
    # re-resolved with automatic treatments.
    custom = {
        "stages": [{"stage_type": stage, "theory": "pbe", "modifiers": [], "options": {}} for stage in stages],
        "label": None,
        "recipe": None,
    }
    response = _build(
        BI2SE3_POSCAR,
        workflow="custom",
        workflow_spec_json=json.dumps(custom),
    )
    selected = response.context["selected_workflow"]

    assert selected["desired_output"] == expected
    assert selected["hidden_desired_output_label"] is None
    # Custom workflows receive no automatic treatments.
    assert all(stage["modifiers"] == [] for stage in selected["stages"])
