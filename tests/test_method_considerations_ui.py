from __future__ import annotations

import json

from pymatgen.core import Lattice, Structure
from starlette.requests import Request

import main
from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import validate_workflow_spec
from backend.structure_dimensionality import ANALYSIS_FAILED, StructureDimensionalityObservation


BI2SE3_POSCAR = """Bi2Se3
1.0
4.143 0.0 0.0
-2.0715 3.5878 0.0
0.0 0.0 28.636
Bi Se
6 9
direct
0.0 0.0 0.1
0.0 0.0 0.2
0.333333 0.666667 0.433333
0.333333 0.666667 0.533333
0.666667 0.333333 0.766667
0.666667 0.333333 0.866667
0.0 0.0 0.3
0.0 0.0 0.4
0.333333 0.666667 0.633333
0.333333 0.666667 0.733333
0.666667 0.333333 0.966667
0.666667 0.333333 0.066667
0.0 0.0 0.5
0.333333 0.666667 0.833333
0.666667 0.333333 0.166667
"""

PT_SE_POSCAR = """PtSe
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Pt Se
1 1
direct
0.0 0.0 0.0
0.25 0.25 0.25
"""

BI_PT_SE_POSCAR = """BiPtSe
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Bi Pt Se
1 1 1
direct
0.0 0.0 0.0
0.25 0.25 0.25
0.5 0.5 0.5
"""

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

EU_POSCAR = """Eu
4.6
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Eu
1
direct
0.0 0.0 0.0
"""

IR_POSCAR = """Ir
3.84
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Ir
1
direct
0.0 0.0 0.0
"""

SNS2_POSCAR = """SnS2
1.0
3.648 0.0 0.0
-1.824 3.159 0.0
0.0 0.0 5.899
Sn S
1 2
direct
0.0 0.0 0.0
0.3333333333333333 0.6666666666666666 0.25
0.6666666666666666 0.3333333333333333 0.75
"""


def request(path: str = "/analyze") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
        }
    )


def render_response(response) -> str:
    return response.template.render(response.context)


def method_considerations_block(html: str) -> str:
    start = html.index('class="method-considerations ')
    end = html.index('<section class="stage">', start)
    return html[start:end]


def analyze_poscar(poscar: str):
    return main.analyze(request(), structure=poscar, fmt="poscar")


def build_energy_only(poscar: str):
    """Build the default Desired Output that /analyze resolves (Energy only).

    Method Considerations describe a successfully built calculation, so they
    are rendered on the built page, not on the analysed page.
    """

    response = main.build_workflow(
        request("/build-calculation"),
        structure=poscar,
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
    assert response.status_code == 200
    return response


def workflow_spec_json(stage_type: str, *, modifiers=None) -> str:
    workflow = validate_workflow_spec(
        WorkflowSpec(
            [
                StageSpec(
                    StageType.from_value(stage_type),
                    Theory.PBE,
                    modifiers or (),
                )
            ],
            recipe=stage_type,
        )
    )
    return json.dumps(workflow.to_dict(), sort_keys=True)


def test_bi_containing_structure_renders_method_considerations_only_after_build():
    analyzed = analyze_poscar(BI2SE3_POSCAR)
    analyzed_html = render_response(analyzed)

    assert analyzed.status_code == 200
    assert analyzed.context["summary"]["reduced_formula"] == "Bi2Se3"
    assert analyzed.context["summary"]["natoms"] == 15
    # Analysis alone has not built a calculation: nothing is rendered.
    assert "Method Considerations" not in analyzed_html
    assert "<h3>Method Considerations</h3>" not in analyzed_html
    assert 'class="method-consideration-card' not in analyzed_html

    built = build_energy_only(BI2SE3_POSCAR)
    html = render_response(built)
    assert built.context["method_considerations"]["policy_version"] == 6
    assert html.index("Structure Summary") < html.index("Calculation Definition")
    assert html.index('id="desired-output-select"') < html.index("<h3>Execution Resources</h3>")
    assert html.index("<h3>Execution Resources</h3>") < html.index("data-build-calculation-button")
    assert html.index("data-build-calculation-button") < html.index("<h3>Method Considerations</h3>")
    assert html.index("<h3>Method Considerations</h3>") < html.index("<h2>Calculation Summary</h2>")

def test_method_consideration_visual_state_uses_advisory_not_success_or_error():
    source = main.templates.get_template("index.html").render(main.page_context())

    assert "--bmd-advisory" in source
    assert ".step-mark.advisory" in source
    assert "var(--bmd-advisory)" in source
    assert ".pill.ready" in source
    assert "var(--bmd-success)" in source
    assert ".pill.failed" in source
    assert "var(--bmd-danger)" in source

    response = build_energy_only(SNS2_POSCAR)
    block = method_considerations_block(render_response(response))
    assert 'class="step-mark advisory"' in block
    assert "pill ready" not in block
    assert "pill failed" not in block


def test_real_style_bi2se3_renders_independent_dispersion_and_soc_considerations():
    response = build_energy_only(BI2SE3_POSCAR)
    html = render_response(response)
    considerations = response.context["method_considerations"]["considerations"]

    assert [consideration["id"] for consideration in considerations] == [
        "dispersion.two_dimensional_connectivity",
        "soc.heavy_elements",
    ]
    dispersion, soc = considerations
    assert dispersion["trigger_elements"] == []
    assert dispersion["trigger_classes"] == ["two_dimensional_bonded_connectivity"]
    assert dispersion["observed_evidence"]["triggers"][0]["dimensionality"] == 2
    assert soc["trigger_elements"] == ["Bi"]
    assert dispersion["display_name"] == "Dispersion Correction"
    assert dispersion["browser_display_name"] == "van der Waals correction applied"
    assert "This structure has two-dimensional bonded connectivity" in dispersion["reason"]
    assert "Spin-Orbit Coupling (SOC)" in html
    assert "van der Waals correction applied" in html
    assert "Likely 2-dimensional structure detected" in html
    assert "The van der Waals correction has been included automatically" in html
    assert soc["automatic_application_state"] == "applied"
    assert soc["browser_display_name"] == "Spin-Orbit Coupling (SOC) applied"
    assert "Bi detected. Spin-orbit coupling (SOC) has been included automatically in the PBE Static Energy stage, which runs with vasp_ncl." in html
    assert "Suggested to activate the Spin-Orbit Coupling" not in html
    block = method_considerations_block(html)
    assert "Two-dimensional bonded connectivity" not in block
    assert "heavy p-block" not in block
    assert "Se \u2014" not in html


def test_pt_containing_structure_renders_soc_consideration_with_5d_trigger():
    response = build_energy_only(PT_SE_POSCAR)
    html = render_response(response)
    consideration = response.context["method_considerations"]["considerations"][0]

    assert consideration["trigger_elements"] == ["Pt"]
    assert consideration["trigger_classes"] == ["5d_transition_metals"]
    assert consideration["automatic_application_state"] == "applied"
    assert "Pt detected. Spin-orbit coupling (SOC) has been included automatically in the PBE Static Energy stage, which runs with vasp_ncl." in html
    assert "Suggested to activate the Spin-Orbit Coupling" not in html
    assert "5d transition metal" not in method_considerations_block(html)
    assert "Se \u2014" not in html


def test_multi_trigger_bi_pt_se_renders_one_consideration_and_actual_triggers_only():
    response = build_energy_only(BI_PT_SE_POSCAR)
    html = render_response(response)
    payload = response.context["method_considerations"]

    assert len(payload["considerations"]) == 1
    consideration = payload["considerations"][0]
    assert consideration["trigger_elements"] == ["Bi", "Pt"]
    assert consideration["trigger_classes"] == [
        "5d_transition_metals",
        "heavy_p_block",
    ]
    assert html.count('data-method-consideration-id="soc.heavy_elements"') == 1
    assert "Bi and Pt detected. Spin-orbit coupling (SOC) has been included automatically in the PBE Static Energy stage, which runs with vasp_ncl." in html
    block = method_considerations_block(html)
    assert "heavy p-block" not in block
    assert "5d transition metal" not in block
    assert "Se \u2014" not in html


def test_si_does_not_render_empty_method_considerations_section():
    response = analyze_poscar(SI_POSCAR)
    html = render_response(response)

    assert response.context["method_considerations"] is None
    assert "Method Considerations" not in html
    assert "nonmagnetic" not in html.lower()
    assert "SOC not needed" not in html
    assert "No methodological issues found" not in html
    assert "dispersion unnecessary" not in html.lower()
    assert "no vdW interactions" not in html


def test_sns2_renders_dispersion_consideration_with_structural_trigger():
    response = build_energy_only(SNS2_POSCAR)
    html = render_response(response)
    considerations = response.context["method_considerations"]["considerations"]

    assert [consideration["id"] for consideration in considerations] == [
        "dispersion.two_dimensional_connectivity"
    ]
    consideration = considerations[0]
    assert consideration["trigger_elements"] == []
    assert consideration["trigger_classes"] == ["two_dimensional_bonded_connectivity"]
    assert consideration["observed_evidence"]["triggers"][0]["method"]["id"] == (
        "pymatgen.crystalnn_larsen_dimensionality"
    )
    assert consideration["observed_evidence"]["triggers"][0]["components"][0]["formula"] == "SnS2"
    assert consideration["display_name"] == "Dispersion Correction"
    assert consideration["browser_display_name"] == "van der Waals correction applied"
    assert "van der Waals correction applied" in html
    assert 'class="step-mark advisory"' in html
    assert "Likely 2-dimensional structure detected" in html
    assert "The van der Waals correction has been included automatically" in html
    # The analysed default Desired Output (Static Energy) has a single PBE
    # Static stage, so the summary names that stage and no relaxation.
    assert (
        "The van der Waals correction has been included automatically in the "
        "PBE Static Energy stage."
    ) in html
    block = method_considerations_block(html)
    assert "PBE Geometry Optimisation" not in block
    assert "Advisory</span>" not in block
    assert "Triggered by" not in block
    assert "Two-dimensional bonded connectivity" not in block
    assert "BMD Compute support" not in block
    assert "does not establish the magnitude of dispersion interactions" not in block
    assert "Sn \u2014" not in html
    assert "S \u2014" not in html
    assert "vdW material" not in html
    assert "dispersion is required" not in html.lower()


def test_dimensionality_failure_does_not_block_analyze_or_render_dispersion(monkeypatch):
    monkeypatch.setattr(
        "backend.calculations.method_considerations.observe_structure_dimensionality",
        lambda structure: StructureDimensionalityObservation(
            status=ANALYSIS_FAILED,
            dimensionality=None,
            reason="test failure",
        ),
    )

    response = analyze_poscar(SNS2_POSCAR)
    html = render_response(response)

    assert response.status_code == 200
    assert response.context["summary"]["reduced_formula"] == "SnS2"
    assert response.context["method_considerations"] is None
    assert "Method Considerations" not in html
    assert "van der Waals Correction" not in html


def test_fe_structure_renders_spin_polarisation_consideration_only():
    response = build_energy_only(FE_POSCAR)
    html = render_response(response)
    considerations = response.context["method_considerations"]["considerations"]

    assert [consideration["id"] for consideration in considerations] == [
        "spin.composition_screen"
    ]
    assert considerations[0]["trigger_elements"] == ["Fe"]
    assert considerations[0]["trigger_classes"] == ["3d_spin_screen"]
    assert "Spin Polarisation" in html
    assert "Spin Polarisation applied" in html
    assert "Fe detected. Spin Polarisation has been included automatically" in html
    assert "3d spin-screening element" not in method_considerations_block(html)
    assert 'data-method-consideration-id="soc.heavy_elements"' not in html
    assert "ISPIN=2 is required" not in html


def test_eu_and_ir_render_independent_spin_and_soc_cards():
    for poscar, symbol, spin_label, soc_label in (
        (EU_POSCAR, "Eu", "lanthanide spin-screening element", "lanthanide"),
        (IR_POSCAR, "Ir", "5d spin-screening element", "5d transition metal"),
    ):
        response = build_energy_only(poscar)
        html = render_response(response)
        considerations = response.context["method_considerations"]["considerations"]

        assert [consideration["id"] for consideration in considerations] == [
            "spin.composition_screen",
            "soc.heavy_elements",
        ]
        assert html.count('data-method-consideration-id="spin.composition_screen"') == 1
        assert html.count('data-method-consideration-id="soc.heavy_elements"') == 1
        assert "Spin Polarisation" in html
        assert "Spin-Orbit Coupling (SOC)" in html
        assert f"{symbol} detected. Spin Polarisation has been included automatically" in html
        assert f"{symbol} detected. Spin-orbit coupling (SOC) has been included automatically in the PBE Static Energy stage, which runs with vasp_ncl." in html
        assert "Suggested to activate the Spin-Orbit Coupling" not in html
        block = method_considerations_block(html)
        assert spin_label not in block
        assert soc_label not in block


def test_backend_reason_limitations_and_support_remain_structured_but_not_rendered():
    response = build_energy_only(BI2SE3_POSCAR)
    html = render_response(response)
    consideration = response.context["method_considerations"]["considerations"][0]
    block = method_considerations_block(html)

    assert "This structure has two-dimensional bonded connectivity" in consideration["reason"]
    for limitation in consideration["limitations"]:
        assert limitation not in block
    assert consideration["bmd_compute_support"]["supported_stage_capabilities"]
    assert "BMD Compute support" not in block
    assert "Triggered by" not in block
    for forbidden in (
        "SOC required",
        "SOC necessary",
        "SOC mandatory",
        "calculation invalid",
        "must enable SOC",
        "Enable SOC",
        "Apply recommendation",
    ):
        assert forbidden not in html


def test_analyze_reuses_the_parsed_structure_for_method_considerations(monkeypatch):
    sentinel = object()
    calls = []

    def fake_parse_structure(structure_text, fmt):
        calls.append((structure_text, fmt))
        return sentinel

    def fake_summarize_structure(structure_obj):
        assert structure_obj is sentinel
        return {
            "formula": "Bi1",
            "reduced_formula": "Bi",
            "natoms": 1,
            "volume": 1,
            "density": 1,
            "lattice": {"a": 1, "b": 1, "c": 1},
            "angles": {"alpha": 90, "beta": 90, "gamma": 90},
            "space_group_symbol": "P1",
            "space_group_number": 1,
            "crystal_system": "triclinic",
        }

    def fake_method_consideration_payload(structure_obj, *, workflow=None):
        assert structure_obj is sentinel
        return {
            "policy_version": 4,
            "scope": "test",
            "detections": [],
            "considerations": [],
        }

    monkeypatch.setattr(main, "parse_structure", fake_parse_structure)
    monkeypatch.setattr(main, "summarize_structure", fake_summarize_structure)
    monkeypatch.setattr(main, "method_consideration_payload", fake_method_consideration_payload)

    response = main.analyze(request(), structure="fake-poscar", fmt="poscar")

    assert response.status_code == 200
    assert calls == [("fake-poscar", "poscar")]
    assert response.context["summary"]["reduced_formula"] == "Bi"
    assert response.context["method_considerations"] is None


def test_method_consideration_ui_does_not_mutate_workflow_modifiers_or_inputs():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    workflow_before = workflow.to_dict()
    request_payload = {
        "structure": {
            "type": "pasted_text",
            "format": "poscar",
            "text": BI2SE3_POSCAR,
        },
        "workflow_spec": workflow.to_dict(),
        "resources": {"ntasks": 24, "mem_gb": 128},
        "potcar_functional": "PBE_64",
    }
    reference_before = build_input_reference_payload(request_payload, include_provenance=False)

    response = main.build_workflow(
        request("/build-calculation"),
        structure=BI2SE3_POSCAR,
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow.to_dict(), sort_keys=True),
        workflow=None,
        method=None,
    )

    reference_after = build_input_reference_payload(request_payload, include_provenance=False)
    assert response.status_code == 200
    consideration_ids = [
        consideration["id"]
        for consideration in response.context["method_considerations"]["considerations"]
    ]
    assert "soc.heavy_elements" in consideration_ids
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    assert "LSORBIT" not in response.context["generated_inputs"]["incar"]
    assert response.context["generated_inputs"]["vasp_executable"] == "vasp_std"
    assert workflow.to_dict() == workflow_before
    assert reference_after == reference_before


def test_spin_consideration_ui_does_not_mutate_workflow_modifiers_or_inputs():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    workflow_before = workflow.to_dict()
    request_payload = {
        "structure": {
            "type": "pasted_text",
            "format": "poscar",
            "text": FE_POSCAR,
        },
        "workflow_spec": workflow.to_dict(),
        "resources": {"ntasks": 24, "mem_gb": 128},
        "potcar_functional": "PBE_64",
    }
    reference_before = build_input_reference_payload(request_payload, include_provenance=False)

    response = main.build_workflow(
        request("/build-calculation"),
        structure=FE_POSCAR,
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow.to_dict(), sort_keys=True),
        workflow=None,
        method=None,
    )

    reference_after = build_input_reference_payload(request_payload, include_provenance=False)
    consideration = response.context["method_considerations"]["considerations"][0]
    assert response.status_code == 200
    assert consideration["id"] == "spin.composition_screen"
    assert consideration["selection_state"] == "not_selected"
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    assert "ISPIN = 1" in response.context["generated_inputs"]["incar"]
    assert "MAGMOM =" not in response.context["generated_inputs"]["incar"]
    assert workflow.to_dict() == workflow_before
    assert reference_after == reference_before


def test_dispersion_consideration_ui_does_not_mutate_workflow_modifiers_or_inputs():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    workflow_before = workflow.to_dict()
    request_payload = {
        "structure": {
            "type": "pasted_text",
            "format": "poscar",
            "text": SNS2_POSCAR,
        },
        "workflow_spec": workflow.to_dict(),
        "resources": {"ntasks": 24, "mem_gb": 128},
        "potcar_functional": "PBE_64",
    }
    reference_before = build_input_reference_payload(request_payload, include_provenance=False)

    response = main.build_workflow(
        request("/build-calculation"),
        structure=SNS2_POSCAR,
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow.to_dict(), sort_keys=True),
        workflow=None,
        method=None,
    )

    reference_after = build_input_reference_payload(request_payload, include_provenance=False)
    consideration = response.context["method_considerations"]["considerations"][0]
    assert response.status_code == 200
    assert consideration["id"] == "dispersion.two_dimensional_connectivity"
    assert consideration["selection_state"] == "not_selected"
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    assert "value=\"dispersion\" checked" not in render_response(response)
    assert "IVDW" not in response.context["generated_inputs"]["incar"]
    assert "dftd3" not in json.dumps(response.context["generated_inputs"], sort_keys=True).lower()
    assert workflow.to_dict() == workflow_before
    assert reference_after == reference_before


def test_existing_structure_summary_and_calculation_definition_remain_present():
    response = analyze_poscar(SI_POSCAR)
    html = render_response(response)

    assert "Structure Summary" in html
    assert "Calculation Definition" in html
    assert response.context["selected_calculation"]["purpose"] == "static"
    assert response.context["selected_workflow"]["stages"][0]["stage_type"] == "static"


def test_template_has_no_duplicated_method_consideration_policy_tables():
    source = main.templates.get_template("index.html").render(main.page_context())
    method_block = source[
        source.index("method-considerations"):
        source.index("calculation-layout")
    ]

    for element in ("Y", "Cd", "Hf", "Hg", "La", "Lu", "Ac", "Lr", "Tl", "Pb", "Bi", "Po"):
        assert f">{element}<" not in method_block
    assert "SOC_TRIGGER_CLASSES" not in method_block
    for class_name in (
        "4d_transition_metals",
        "heavy_p_block",
        "3d_spin_screen",
        "lanthanide_spin_screen",
        "actinide_spin_screen",
        "two_dimensional_bonded_connectivity",
        "pymatgen.crystalnn_larsen_dimensionality",
    ):
        assert class_name not in method_block
