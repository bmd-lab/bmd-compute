from __future__ import annotations

import json

from starlette.requests import Request

import main
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    calculation_form_options,
    desired_output_from_workflow_spec,
    desired_output_workflow_spec,
    workflow_result_stage_directory,
    workflow_stage_directories,
)
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure
from backend.workflow_summary import calculation_plan_from_workflow_spec
from backend.workflows import build_atomate2_flow_for_workflow_spec


SI_POSCAR = """Si
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
direct
0.0 0.0 0.0
0.25 0.25 0.25
"""

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


def _request(path: str = "/build-calculation") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
        }
    )


def _stage_pairs(workflow: WorkflowSpec) -> list[tuple[str, str]]:
    return [
        (stage.stage_type.value, stage.theory.value)
        for stage in workflow.stages
    ]


def _render_index(**context_overrides) -> str:
    context = {
        "structure_text": SI_POSCAR,
        "fmt": "poscar",
        "summary": main.summarize_structure(parse_structure(SI_POSCAR)),
    }
    context.update(context_overrides)
    return main.templates.get_template("index.html").render(
        main.page_context(**context)
    )


def _desired_output_select_block(html: str) -> str:
    start = html.index('id="desired-output-select"')
    end = html.index("</select>", start)
    return html[start:end]


def _workflow_stage_list_block(html: str) -> str:
    start = html.index('id="workflow-stage-list"')
    end = html.index('class="workflow-controls"', start)
    return html[start:end]


def test_desired_output_options_are_backend_owned_and_beginner_facing():
    options = calculation_form_options()
    desired_outputs = options["desired_outputs"]

    assert [option["label"] for option in desired_outputs] == [
        "Energy only",
        "Relaxed structure",
        "Electronic density of states",
        "Electronic band structure",
        "Phonons",
        "Custom workflow",
    ]
    assert [option["value"] for option in desired_outputs] == [
        "energy_only",
        "relaxed_structure",
        "electronic_dos",
        "electronic_band_structure",
        "phonons",
        "custom",
    ]

    workflows = {
        option["value"]: (
            WorkflowSpec.from_dict(option["workflow_spec"])
            if option["workflow_spec"] is not None
            else None
        )
        for option in desired_outputs
    }
    assert _stage_pairs(workflows["energy_only"]) == [("static", "pbe")]
    assert _stage_pairs(workflows["relaxed_structure"]) == [
        ("relax", "pbe"),
        ("relax", "pbe"),
    ]
    assert _stage_pairs(workflows["electronic_dos"]) == [
        ("relax", "pbe"),
        ("static", "hse06"),
        ("dos", "hse06"),
    ]
    assert _stage_pairs(workflows["electronic_band_structure"]) == [
        ("relax", "pbe"),
        ("static", "hse06"),
        ("band_structure", "hse06"),
    ]
    assert _stage_pairs(workflows["phonons"]) == [
        ("relax", "pbe"),
        ("phonon_forces", "pbe"),
    ]
    assert workflows["custom"] is None

    legacy_recipes = {recipe["value"] for recipe in options["recipes"]}
    assert {"relax_static", "double_relax", "dos", "band_structure"}.issubset(
        legacy_recipes
    )


def test_rendered_normal_selector_uses_desired_output_not_workflow_recipes():
    html = _render_index()
    select_block = _desired_output_select_block(html)

    assert "Desired Output" in html
    assert "Recommended Workflow" not in html
    assert 'id="workflow-recipe-select"' not in html
    assert 'id="desired-output-select"' in html
    assert "Energy only" in select_block
    assert "Relaxed structure" in select_block
    assert "Electronic density of states" in select_block
    assert "Electronic band structure" in select_block
    assert "Custom workflow" in select_block
    assert "Geometry Optimisation + Static Energy" not in select_block
    assert "Double Geometry Optimisation" not in select_block
    assert ">Density of States<" not in select_block
    assert ">Band Structure<" not in select_block
    assert "Gamma-only" not in html
    assert "Spin Polarised" in html
    assert "Spin-Orbit Coupling (SOC)" in html
    assert "DFT+U" in html
    assert "van der Waals correction" in html
    assert "DFT-D3" not in html
    assert "DFT-D3(BJ)" not in html


def test_bmd_managed_desired_output_shows_read_only_applied_treatments():
    response = main.build_workflow(
        _request(),
        structure=FE_POSCAR,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=None,
        workflow="energy_only",
        method=None,
    )
    html = response.template.render(response.context)
    stage_block = _workflow_stage_list_block(html)

    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == [
        "spin_polarized"
    ]
    assert "Applied Treatments" in stage_block
    assert "Spin Polarised" in stage_block
    assert "Advanced Options" not in stage_block
    assert "data-stage-modifier" not in stage_block


def test_custom_workflow_keeps_editable_advanced_options_controls():
    custom_workflow = WorkflowSpec(
        [
            StageSpec(
                StageType.STATIC,
                Theory.PBE,
                {Modifier.SPIN_POLARIZED, Modifier.DISPERSION},
            )
        ],
        recipe="custom",
    )
    html = _render_index(selected_workflow=custom_workflow)
    stage_block = _workflow_stage_list_block(html)

    assert "Advanced Options" in stage_block
    assert "Applied Treatments" not in stage_block
    assert "data-stage-modifier" in stage_block
    assert "Spin Polarised" in stage_block
    assert "van der Waals correction" in stage_block
    assert 'value="dispersion"' in stage_block
    dispersion_checkbox = stage_block[
        stage_block.index('value="dispersion"'):
        stage_block.index('value="dispersion"') + 120
    ]
    assert "checked" in dispersion_checkbox
    assert "data-stage-dispersion-method" not in stage_block
    assert "DFT-D3" not in stage_block


def test_default_desired_output_cards_keep_bmd_stage_theories_visible():
    relaxed_workflow = desired_output_workflow_spec("relaxed_structure")
    dos_workflow = desired_output_workflow_spec("electronic_dos")
    band_workflow = desired_output_workflow_spec("electronic_band_structure")

    relaxed_context = main.page_context(selected_workflow=relaxed_workflow)
    dos_context = main.page_context(selected_workflow=dos_workflow)
    band_context = main.page_context(selected_workflow=band_workflow)

    assert relaxed_context["selected_workflow"]["desired_output"] == "relaxed_structure"
    assert [
        (stage["stage_type"], stage["theory"])
        for stage in relaxed_context["selected_workflow"]["stages"]
    ] == [
        ("relax", "pbe"),
        ("relax", "pbe"),
    ]
    assert calculation_plan_from_workflow_spec(relaxed_workflow) == [
        "Geometry Optimisation",
        "Geometry Optimisation",
    ]
    assert workflow_stage_directories(relaxed_workflow) == ("relax_01", "relax_02")
    assert workflow_result_stage_directory(relaxed_workflow) == "relax_02"

    assert dos_context["selected_workflow"]["desired_output"] == "electronic_dos"
    assert [
        (stage["stage_type"], stage["theory"])
        for stage in dos_context["selected_workflow"]["stages"]
    ] == [
        ("relax", "pbe"),
        ("static", "hse06"),
        ("dos", "hse06"),
    ]
    assert calculation_plan_from_workflow_spec(dos_workflow) == [
        "Geometry Optimisation (PBE)",
        "Static Energy (HSE06)",
        "Density of States (HSE06)",
    ]
    assert band_context["selected_workflow"]["desired_output"] == (
        "electronic_band_structure"
    )
    assert [
        (stage["stage_type"], stage["theory"])
        for stage in band_context["selected_workflow"]["stages"]
    ] == [
        ("relax", "pbe"),
        ("static", "hse06"),
        ("band_structure", "hse06"),
    ]
    assert calculation_plan_from_workflow_spec(band_workflow) == [
        "Geometry Optimisation (PBE)",
        "Static Energy (HSE06)",
        "Band Structure (HSE06)",
    ]


def test_desired_output_previews_use_current_authoritative_workflows():
    structure = parse_structure(SI_POSCAR)
    expected_stage_markers = {
        "energy_only": ["ISPIN = 1", "NSW = 0"],
        "relaxed_structure": [
            "# Stage 1 - Geometry Optimisation (PBE)",
            "# Stage 2 - Geometry Optimisation (PBE)",
            "ISPIN = 1",
            "IBRION = 2",
            "ISIF = 3",
        ],
        "electronic_dos": [
            "# Stage 1 - Geometry Optimisation (PBE)",
            "# Stage 2 - Static Energy (HSE06)",
            "# Stage 3 - Density of States (HSE06)",
        ],
        "electronic_band_structure": [
            "# Stage 1 - Geometry Optimisation (PBE)",
            "# Stage 2 - Static Energy (HSE06)",
            "# Stage 3 - Band Structure (HSE06)",
        ],
    }

    for output, markers in expected_stage_markers.items():
        preview = preview_generated_inputs(
            structure,
            desired_output_workflow_spec(output),
            resources={"ntasks": 24},
        )
        for marker in markers:
            assert marker in preview["incar"]

    dos_preview = preview_generated_inputs(
        structure,
        desired_output_workflow_spec("electronic_dos"),
        resources={"ntasks": 24},
    )
    assert "LHFCALC = True" in dos_preview["incar"]
    assert "# Stage 3 - Density of States (HSE06)" in dos_preview["incar"]
    assert "ICHARG" not in dos_preview["incar"].split(
        "# Stage 3 - Density of States (HSE06)",
        1,
    )[1]

    band_preview = preview_generated_inputs(
        structure,
        desired_output_workflow_spec("electronic_band_structure"),
        resources={"ntasks": 24},
    )
    assert "# Stage 2 - Static Energy (HSE06)" in band_preview["incar"]
    assert "# Stage 3 - Band Structure (HSE06)" in band_preview["incar"]
    assert "Combined k-points" in band_preview["kpoints"]


def test_relaxed_structure_workflow_construction_chains_second_relax_to_first():
    workflow = desired_output_workflow_spec("relaxed_structure")
    flow = build_atomate2_flow_for_workflow_spec(
        "initial-structure",
        workflow,
        label="Si-relaxed",
        resources={"ntasks": 24},
    )

    assert flow.name == "Si-relaxed_double_relax"
    assert flow.metadata["bmd_stage_directories"] == ("relax_01", "relax_02")
    assert [job.name for job in flow.jobs] == ["relax_01", "relax_02"]
    assert flow.jobs[0].function_args == ("initial-structure",)
    second_relax_structure = flow.jobs[1].function_args[0]
    assert repr(second_relax_structure).endswith(", .structure)")


def test_build_route_can_reconstruct_desired_output_without_recipe_fallback():
    response = main.build_workflow(
        _request(),
        structure=SI_POSCAR,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=None,
        workflow="electronic_dos",
        method=None,
    )

    assert response.status_code == 200
    assert response.context["selected_workflow"]["desired_output"] == "electronic_dos"
    assert [
        (stage["stage_type"], stage["theory"])
        for stage in response.context["selected_workflow"]["stages"]
    ] == [
        ("relax", "pbe"),
        ("static", "hse06"),
        ("dos", "hse06"),
    ]
    assert response.context["calculation"]["calculation_type"] == (
        "Geometry Optimisation + Static Energy + Density of States"
    )
    assert response.context["calculation"]["calculation_plan"] == [
        "Geometry Optimisation (PBE)",
        "Static Energy (HSE06)",
        "Density of States (HSE06)",
    ]
    assert response.context["selected_calculation"]["theory_label"] == "Mixed"


def test_custom_workflow_remains_available_and_editable():
    custom_workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.HSE06),
        ],
        recipe="custom",
    )
    html = _render_index(selected_workflow=custom_workflow)
    context = main.page_context(selected_workflow=custom_workflow)

    assert desired_output_from_workflow_spec(custom_workflow) == "custom"
    assert context["selected_workflow"]["desired_output"] == "custom"
    select_block = _desired_output_select_block(html)
    custom_option = select_block.split('value="custom"', 1)[1].split("</option>", 1)[0]
    assert "selected" in custom_option
    assert "data-stage-type disabled" not in html
    assert "data-stage-theory disabled" not in html
    assert 'id="add-workflow-stage" hidden' not in html
    assert "Add Stage" in html
    assert "Remove" in html
    assert json.loads(context["selected_workflow"]["json"]) == custom_workflow.to_dict()
