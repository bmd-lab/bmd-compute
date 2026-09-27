from __future__ import annotations

import builtins
import json
import socket
import subprocess
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from starlette.requests import Request

import main
from backend.calculations.default_treatments import (
    AUTOMATIC_APPLICATION_ADVISORY,
    AUTOMATIC_APPLICATION_APPLIED,
    DISPERSION_CONSIDERATION_ID,
    SOC_CONSIDERATION_ID,
    SPIN_CONSIDERATION_ID,
    automatic_default_treatment_policy,
    resolve_default_treatments,
)
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import desired_output_workflow_spec
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure
from backend.structure_dimensionality import ANALYSIS_FAILED, StructureDimensionalityObservation


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

FES2_LAYERED_POSCAR = """FeS2
1.0
3.648 0.0 0.0
-1.824 3.159 0.0
0.0 0.0 5.899
Fe S
1 2
direct
0.0 0.0 0.0
0.3333333333333333 0.6666666666666666 0.25
0.6666666666666666 0.3333333333333333 0.75
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


def request(path: str = "/build-calculation") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
        }
    )


def structure_for_symbols(symbols: list[str]) -> Structure:
    return Structure(
        Lattice.cubic(max(5, len(symbols) + 3)),
        symbols,
        [
            [
                (index * 0.173) % 1,
                (index * 0.317) % 1,
                (index * 0.463) % 1,
            ]
            for index, _ in enumerate(symbols)
        ],
    )


def resolved_workflow(desired_output: str, poscar: str) -> WorkflowSpec:
    result = resolve_default_treatments(
        parse_structure(poscar),
        desired_output_workflow_spec(desired_output),
        desired_output=desired_output,
    )
    return result.resolved_workflow


def signature(workflow: WorkflowSpec) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (
            stage.stage_type.value,
            stage.theory.value,
            tuple(sorted(modifier.value for modifier in stage.modifiers)),
        )
        for stage in workflow.stages
    ]


def stage_options(workflow: WorkflowSpec) -> list[dict]:
    return [dict(stage.options or {}) for stage in workflow.stages]


def stage_sections(incar: str) -> list[str]:
    if "# Stage " not in incar:
        return [incar]
    return [
        "# Stage " + section
        for section in incar.split("# Stage ")[1:]
    ]


def build_route(
    poscar: str,
    *,
    workflow: str,
    workflow_spec: WorkflowSpec | None = None,
):
    return main.build_workflow(
        request(),
        structure=poscar,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=(
            json.dumps(workflow_spec.to_dict(), sort_keys=True)
            if workflow_spec is not None
            else None
        ),
        workflow=workflow,
        method=None,
    )


def test_si_desired_output_static_has_no_automatic_treatments_and_is_deterministic():
    structure = parse_structure(SI_POSCAR)
    base = desired_output_workflow_spec("energy_only")
    base_before = base.to_dict()
    structure_before = structure.as_dict()

    first = resolve_default_treatments(structure, base, desired_output="energy_only")
    second = resolve_default_treatments(structure, base, desired_output="energy_only")

    assert first == second
    assert first.applied_treatments == ()
    assert first.advisory_consideration_ids == ()
    assert first.resolved_workflow.to_dict() == base_before
    assert base.to_dict() == base_before
    assert structure.as_dict() == structure_before

    preview = preview_generated_inputs(
        structure,
        first.resolved_workflow,
        resources={"ntasks": 24},
    )
    assert "ISPIN = 1" in preview["incar"]
    assert "MAGMOM =" not in preview["incar"]
    assert "IVDW" not in preview["incar"]


@pytest.mark.parametrize(
    ("desired_output", "expected"),
    [
        (
            "energy_only",
            [("static", "pbe", ("spin_polarized",))],
        ),
        (
            "relaxed_structure",
            [
                ("relax", "pbe", ("spin_polarized",)),
                ("relax", "pbe", ("spin_polarized",)),
            ],
        ),
        (
            "electronic_dos",
            [
                ("relax", "pbe", ("spin_polarized",)),
                ("static", "hse06", ("spin_polarized",)),
                ("dos", "hse06", ("spin_polarized",)),
            ],
        ),
        (
            "electronic_band_structure",
            [
                ("relax", "pbe", ("spin_polarized",)),
                ("static", "hse06", ("spin_polarized",)),
                ("band_structure", "hse06", ("spin_polarized",)),
            ],
        ),
    ],
)
def test_spin_consideration_applies_to_current_desired_outputs(desired_output, expected):
    workflow = resolved_workflow(desired_output, FE_POSCAR)

    assert signature(workflow) == expected


@pytest.mark.parametrize(
    ("desired_output", "expected"),
    [
        (
            "energy_only",
            [("static", "pbe", ("dispersion",))],
        ),
        (
            "relaxed_structure",
            [
                ("relax", "pbe", ("dispersion",)),
                ("relax", "pbe", ("dispersion",)),
            ],
        ),
        (
            "electronic_dos",
            [
                ("relax", "pbe", ("dispersion",)),
                ("static", "hse06", ()),
                ("dos", "hse06", ()),
            ],
        ),
        (
            "electronic_band_structure",
            [
                ("relax", "pbe", ("dispersion",)),
                ("static", "hse06", ()),
                ("band_structure", "hse06", ()),
            ],
        ),
    ],
)
def test_dispersion_consideration_applies_only_to_bmd_pbe_stages(desired_output, expected):
    workflow = resolved_workflow(desired_output, SNS2_POSCAR)

    assert signature(workflow) == expected
    for stage, options in zip(workflow.stages, stage_options(workflow)):
        if Modifier.DISPERSION in stage.modifiers:
            assert options == {"dispersion": {"method": "dftd3-bj"}}
        else:
            assert options == {}


def test_spin_and_dispersion_compose_without_overwriting_stage_modifiers():
    workflow = resolved_workflow("electronic_dos", FES2_LAYERED_POSCAR)

    assert signature(workflow) == [
        ("relax", "pbe", ("dispersion", "spin_polarized")),
        ("static", "hse06", ("spin_polarized",)),
        ("dos", "hse06", ("spin_polarized",)),
    ]

    preview = preview_generated_inputs(
        parse_structure(FES2_LAYERED_POSCAR),
        workflow,
        resources={"ntasks": 24},
    )
    sections = stage_sections(preview["incar"])
    assert len(sections) == 3
    assert all("ISPIN = 2" in section for section in sections)
    assert "IVDW = 12" in sections[0]
    assert "IVDW" not in sections[1]
    assert "IVDW" not in sections[2]


def test_automatic_resolution_changes_the_actual_generated_vasp_inputs():
    fe_workflow = resolved_workflow("energy_only", FE_POSCAR)
    fe_preview = preview_generated_inputs(
        parse_structure(FE_POSCAR),
        fe_workflow,
        resources={"ntasks": 24},
    )
    assert "ISPIN = 2" in fe_preview["incar"]
    assert "MAGMOM = 1*5.0" in fe_preview["incar"]

    sns2_workflow = resolved_workflow("energy_only", SNS2_POSCAR)
    sns2_preview = preview_generated_inputs(
        parse_structure(SNS2_POSCAR),
        sns2_workflow,
        resources={"ntasks": 24},
    )
    assert "IVDW = 12" in sns2_preview["incar"]
    assert "ISPIN = 1" in sns2_preview["incar"]
    assert "MAGMOM =" not in sns2_preview["incar"]


def test_soc_is_applied_automatically_with_spin_for_desired_output():
    response = build_route(EU_POSCAR, workflow="energy_only")
    context = response.context
    considerations = context["method_considerations"]["considerations"]

    assert response.status_code == 200
    assert [
        consideration["id"]
        for consideration in considerations
    ] == [SPIN_CONSIDERATION_ID, SOC_CONSIDERATION_ID]
    assert considerations[0]["automatic_application_state"] == AUTOMATIC_APPLICATION_APPLIED
    assert considerations[1]["automatic_application_state"] == AUTOMATIC_APPLICATION_APPLIED
    assert context["selected_workflow"]["stages"][0]["modifiers"] == ["soc", "spin_polarized"]
    assert "LSORBIT = True" in context["generated_inputs"]["incar"]
    assert context["generated_inputs"]["vasp_executable"] == "vasp_ncl"
    rendered = response.template.render(context)
    assert "Spin Polarisation applied" in rendered
    assert "Spin-Orbit Coupling (SOC) applied" in rendered
    assert "Suggested to activate the Spin-Orbit Coupling" not in rendered


@pytest.mark.parametrize(
    ("poscar", "consideration_id", "forbidden_modifier"),
    [
        (FE_POSCAR, SPIN_CONSIDERATION_ID, "spin_polarized"),
        (SNS2_POSCAR, DISPERSION_CONSIDERATION_ID, "dispersion"),
        (EU_POSCAR, SPIN_CONSIDERATION_ID, "spin_polarized"),
    ],
)
def test_custom_workflow_is_not_automatically_mutated(poscar, consideration_id, forbidden_modifier):
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)], recipe="custom")
    before = workflow.to_dict()

    response = build_route(poscar, workflow="custom", workflow_spec=workflow)

    assert response.status_code == 200
    assert response.context["selected_workflow"]["desired_output"] == "custom"
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    assert forbidden_modifier not in json.dumps(
        response.context["selected_workflow"],
        sort_keys=True,
    )
    assert response.context["method_considerations"]["considerations"][0]["id"] == consideration_id
    assert response.context["method_considerations"]["considerations"][0][
        "automatic_application_state"
    ] == AUTOMATIC_APPLICATION_ADVISORY
    assert "Suggested to activate" in response.template.render(response.context)
    assert workflow.to_dict() == before


def test_desired_output_ignores_stale_manual_modifier_json_for_current_structure():
    stale_spin_workflow = WorkflowSpec(
        [
            StageSpec(
                StageType.STATIC,
                Theory.PBE,
                {Modifier.SPIN_POLARIZED},
            )
        ],
        recipe="energy_only",
    )

    response = build_route(
        SI_POSCAR,
        workflow="energy_only",
        workflow_spec=stale_spin_workflow,
    )

    assert response.status_code == 200
    assert response.context["selected_workflow"]["desired_output"] == "energy_only"
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    assert "ISPIN = 1" in response.context["generated_inputs"]["incar"]
    assert "MAGMOM =" not in response.context["generated_inputs"]["incar"]


def test_changing_structure_recomputes_from_base_recipe_without_stale_modifiers():
    base = desired_output_workflow_spec("energy_only")

    fe = resolve_default_treatments(parse_structure(FE_POSCAR), base)
    si = resolve_default_treatments(parse_structure(SI_POSCAR), base)
    sns2 = resolve_default_treatments(parse_structure(SNS2_POSCAR), base)

    assert signature(fe.resolved_workflow) == [("static", "pbe", ("spin_polarized",))]
    assert signature(sns2.resolved_workflow) == [("static", "pbe", ("dispersion",))]
    assert signature(si.resolved_workflow) == [("static", "pbe", ())]


def test_changing_desired_output_recomputes_stage_applicability_from_base_recipe():
    structure = parse_structure(SNS2_POSCAR)

    dos = resolve_default_treatments(
        structure,
        desired_output_workflow_spec("electronic_dos"),
        desired_output="electronic_dos",
    )
    band = resolve_default_treatments(
        structure,
        desired_output_workflow_spec("electronic_band_structure"),
        desired_output="electronic_band_structure",
    )

    assert [treatment.to_dict() for treatment in dos.applied_treatments] == [
        {
            "consideration_id": DISPERSION_CONSIDERATION_ID,
            "modifier": "dispersion",
            "display_name": "van der Waals correction",
            "application_state": "applied",
            "stage_indices": [1],
            "stage_applications": [
                {
                    "stage_index": 1,
                    "stage_type": "relax",
                    "stage_type_label": "Geometry Optimisation",
                    "theory": "pbe",
                    "theory_label": "PBE",
                },
            ],
            "source": "backend.calculations.default_treatments",
        }
    ]
    assert [treatment.stage_indices for treatment in band.applied_treatments] == [(1,)]


def test_resolution_is_pure_with_respect_to_io_network_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("automatic treatment resolution must stay local and pure")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(
        "backend.calculations.method_considerations.observe_structure_dimensionality",
        lambda structure: StructureDimensionalityObservation(
            status=ANALYSIS_FAILED,
            dimensionality=None,
            reason="test failure",
        ),
    )

    result = resolve_default_treatments(
        structure_for_symbols(["Fe"]),
        desired_output_workflow_spec("energy_only"),
    )

    assert signature(result.resolved_workflow) == [("static", "pbe", ("spin_polarized",))]


def test_automatic_default_treatment_policy_is_json_safe_and_contract_focused():
    policy = automatic_default_treatment_policy()

    assert json.loads(json.dumps(policy, sort_keys=True)) == policy
    assert policy["applies_to"]["workflow_mode"] == "bmd_managed_desired_output"
    assert policy["applies_to"]["custom_workflow"] == "preserved_without_automatic_changes"
    assert [treatment["consideration_id"] for treatment in policy["treatments"]] == [
        SPIN_CONSIDERATION_ID,
        DISPERSION_CONSIDERATION_ID,
        SOC_CONSIDERATION_ID,
    ]
    assert policy["treatments"][1]["method"] == "dftd3-bj"
    assert policy["treatments"][1]["incar_effect"] == {"IVDW": 12}
    assert policy["treatments"][2]["excluded_stage_types"] == ["relax"]
    assert policy["treatments"][2]["executable"] == "vasp_ncl"
    assert not [
        entry
        for entry in policy["advisory_only"]
        if entry.get("consideration_id") == SOC_CONSIDERATION_ID
    ]
