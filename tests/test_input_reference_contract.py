from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.calculations.input_reference import (
    REFERENCE_PHASE,
    SCHEMA_VERSION,
    SCOPE,
    build_input_reference_payload,
    emit_json,
)
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.resources import ExecutionResources
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure


REPO_ROOT = Path(__file__).resolve().parents[1]

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

NIO_POSCAR = """NiO
4.3
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Ni O
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

FE2O3_POSCAR = """Fe2O3
5.04
1.0 0.0 0.0
-0.5 0.8660254 0.0
0.0 0.0 2.7281746
Fe O
2 3
direct
0.0 0.0 0.355
0.0 0.0 0.645
0.305 0.0 0.25
0.0 0.305 0.25
0.695 0.695 0.25
"""


def request_for(
    stage_type: str,
    *,
    theory: str = "pbe",
    modifiers=None,
    options=None,
    structure: str = SI_POSCAR,
    resources=None,
) -> dict:
    return {
        "structure": {
            "type": "pasted_text",
            "format": "poscar",
            "text": structure,
        },
        "workflow_spec": {
            "stages": [
                {
                    "stage_type": stage_type,
                    "theory": theory,
                    "modifiers": list(modifiers or []),
                    "label": None,
                    "options": dict(options or {}),
                }
            ],
            "label": None,
            "recipe": None,
        },
        "resources": dict(resources or {}),
        "potcar_functional": "PBE_64",
    }


def first_stage(payload):
    return payload["reference"]["stages"][0]


def test_command_emits_json_only_for_supported_reference():
    completed = subprocess.run(
        [sys.executable, "-m", "backend.calculations.input_reference"],
        cwd=REPO_ROOT,
        input=json.dumps(request_for("static")),
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["scope"] == SCOPE
    assert payload["status"] == "ok"
    assert payload["reference_phase"] == REFERENCE_PHASE


def test_pbe_static_reference_matches_existing_preview_path():
    request = request_for("static")
    payload = build_input_reference_payload(request, include_provenance=False)
    stage = first_stage(payload)
    preview = preview_generated_inputs(
        parse_structure(SI_POSCAR),
        WorkflowSpec.from_dict(request["workflow_spec"]),
        resources=ExecutionResources.from_dict(request["resources"]),
        potcar_functional="PBE_64",
    )

    assert payload["status"] == "ok"
    assert payload["request"]["resources"] == {
        "nodes": 1,
        "ntasks": 24,
        "mem_gb": 128,
    }
    assert stage["incar"]["text"] == preview["incar"]
    assert stage["kpoints"]["text"] == preview["kpoints"]
    assert stage["poscar"]["text"] == preview["poscar"]
    assert stage["incar"]["settings"]["ENCUT"] == 620.0
    assert stage["incar"]["settings"]["NCORE"] == 8
    assert stage["incar"]["settings"]["ISPIN"] == 1
    assert "MAGMOM" not in stage["incar"]["settings"]
    assert stage["vasp_executable"] == "vasp_std"


def test_references_for_pbe_relax_spin_hse06_soc_dft_u_and_dispersion():
    relax = first_stage(build_input_reference_payload(request_for("relax"), include_provenance=False))
    assert relax["stage_type"] == "relax"
    assert relax["incar"]["settings"]["IBRION"] == 2
    assert relax["incar"]["settings"]["ISIF"] == 3
    assert relax["incar"]["settings"]["ISPIN"] == 1
    assert "MAGMOM" not in relax["incar"]["settings"]

    spin = first_stage(
        build_input_reference_payload(
            request_for("static", modifiers=["spin_polarized"]),
            include_provenance=False,
        )
    )
    assert spin["modifiers"] == ["spin_polarized"]
    assert spin["incar"]["settings"]["ISPIN"] == 2
    assert spin["incar"]["settings"]["MAGMOM"] == [0.6, 0.6]

    hse = first_stage(
        build_input_reference_payload(
            request_for("static", theory="hse06"),
            include_provenance=False,
        )
    )
    assert hse["theory"] == "hse06"
    assert hse["incar"]["settings"]["LHFCALC"] is True
    assert hse["incar"]["settings"]["HFSCREEN"] == 0.2
    assert hse["incar"]["settings"]["PRECFOCK"] == "Accurate"
    assert hse["incar"]["settings"]["ISPIN"] == 1
    assert "MAGMOM" not in hse["incar"]["settings"]

    soc = first_stage(
        build_input_reference_payload(
            request_for("static", modifiers=["soc"]),
            include_provenance=False,
        )
    )
    assert soc["vasp_executable"] == "vasp_ncl"
    assert soc["incar"]["settings"]["LSORBIT"] is True
    assert soc["incar"]["settings"]["LNONCOLLINEAR"] is True
    assert "ISPIN" not in soc["incar"]["settings"]
    # Si is outside the spin screen, so SOC starts from zero vector moments.
    assert soc["incar"]["settings"]["MAGMOM"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert "MAGMOM = " in soc["incar"]["text"]

    dft_u = first_stage(
        build_input_reference_payload(
            request_for("static", modifiers=["dft_u"], structure=NIO_POSCAR),
            include_provenance=False,
        )
    )
    assert dft_u["incar"]["settings"]["LDAU"] is True
    assert "LDAUU" in dft_u["incar"]["settings"]
    assert dft_u["incar"]["settings"]["ISPIN"] == 1
    assert "MAGMOM" not in dft_u["incar"]["settings"]

    dispersion = first_stage(
        build_input_reference_payload(
            request_for(
                "static",
                modifiers=["dispersion"],
                options={"dispersion": {"method": "dftd3"}},
            ),
            include_provenance=False,
        )
    )
    assert dispersion["modifiers"] == ["dispersion"]
    assert dispersion["options"] == {"dispersion": {"method": "dftd3"}}
    assert dispersion["incar"]["settings"]["IVDW"] == 11

    dispersion_bj = first_stage(
        build_input_reference_payload(
            request_for("relax", modifiers=["dispersion"]),
            include_provenance=False,
        )
    )
    assert dispersion_bj["incar"]["settings"]["IVDW"] == 12


def test_kpoints_are_structured_and_deterministic():
    request = request_for(
        "static",
        resources=ExecutionResources(cpus=24, memory_gb=128).to_dict(),
    )

    first = build_input_reference_payload(request, include_provenance=False)
    second = build_input_reference_payload(request, include_provenance=False)
    first_kpoints = first_stage(first)["kpoints"]

    assert first_kpoints == first_stage(second)["kpoints"]
    assert first_kpoints["as_dict"]["generation_style"] == "Gamma"
    assert first_kpoints["as_dict"]["kpoints"] == [[7, 7, 7]]
    assert json.loads(emit_json(first)) == first


def test_structure_dependent_values_are_generated_from_supplied_structure_when_spin_selected():
    payload = build_input_reference_payload(
        request_for("static", structure=FE2O3_POSCAR),
        include_provenance=False,
    )
    stage = first_stage(payload)

    assert stage["poscar"]["reduced_formula"] == "Fe2O3"
    assert stage["poscar"]["num_sites"] == 5
    assert stage["incar"]["settings"]["ISPIN"] == 1
    assert "MAGMOM" not in stage["incar"]["settings"]

    spin_payload = build_input_reference_payload(
        request_for("static", modifiers=["spin_polarized"], structure=FE2O3_POSCAR),
        include_provenance=False,
    )
    spin_stage = first_stage(spin_payload)

    assert spin_stage["poscar"]["reduced_formula"] == "Fe2O3"
    assert spin_stage["poscar"]["num_sites"] == 5
    assert spin_stage["incar"]["settings"]["ISPIN"] == 2
    assert spin_stage["incar"]["settings"]["MAGMOM"] == [5.0, 5.0, 0.6, 0.6, 0.6]


def test_unsupported_combination_returns_structured_json_without_fallback():
    payload = build_input_reference_payload(
        request_for("dos", theory="hse06"),
        include_provenance=False,
    )

    assert payload["status"] == "unsupported"
    assert payload["reference"] is None
    assert payload["error"]["code"] == "unsupported_combination"
    assert "must follow a converged Static Energy stage" in payload["error"]["message"]


def test_hse06_dos_reference_uses_supported_custom_stage_workflow():
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.PBE),
            StageSpec(StageType.DOS, Theory.HSE06),
        ]
    )
    request = request_for("static")
    request["workflow_spec"] = workflow.to_dict()

    payload = build_input_reference_payload(request, include_provenance=False)
    terminal = payload["reference"]["stages"][-1]

    assert payload["status"] == "ok"
    assert terminal["stage_type"] == "dos"
    assert terminal["theory"] == "hse06"
    assert terminal["generator"]["selected_atomate2"]["input_set_generator"].endswith(
        "HSEBSSetGenerator"
    )
    assert terminal["generator"]["selected_atomate2"]["maker"].endswith("HSEBSMaker")
    assert terminal["incar"]["settings"]["LHFCALC"] is True
    assert terminal["incar"]["settings"]["AEXX"] == 0.25
    assert terminal["incar"]["settings"]["HFSCREEN"] == 0.2
    assert terminal["incar"]["settings"]["NEDOS"] == 4001
    assert "ICHARG" not in terminal["incar"]["settings"]
    assert "Combined k-points" not in terminal["kpoints"]["text"]
    assert "Non SCF run along symmetry lines" not in terminal["kpoints"]["text"]


def test_malformed_request_returns_structured_json_from_command():
    completed = subprocess.run(
        [sys.executable, "-m", "backend.calculations.input_reference"],
        cwd=REPO_ROOT,
        input="{not-json",
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "malformed_request"
    assert "traceback" not in completed.stdout.lower()


def test_producer_provenance_shape_and_json_safety():
    payload = build_input_reference_payload(request_for("static"))

    assert payload["producer"]["repository"] == "bmd_compute"
    assert set(payload["producer"]["source"]) == {
        "repository",
        "commit",
        "dirty",
        "provenance_available",
        "unavailable_reason",
    }
    assert json.loads(emit_json(payload)) == payload
    assert "<" not in emit_json(payload)


def test_symbolic_potcar_only_no_raw_contents_or_deployment_details():
    payload = build_input_reference_payload(
        request_for("static", structure=NIO_POSCAR),
        include_provenance=False,
    )
    stage = first_stage(payload)
    text = json.dumps(payload, sort_keys=True)

    assert stage["potcar"] == {
        "spec_text": "Ni\nO",
        "symbols": ["Ni", "O"],
        "contains_potcar_contents": False,
        "source": "pymatgen get_input_set(..., potcar_spec=True)",
    }
    for forbidden in (
        "PAW_PBE",
        "End of Dataset",
        "TITEL",
        "ENMAX",
        "password",
        "private_key",
        "key_file",
        "remote_host",
        "partition",
        "account",
        "/bmd-db",
        "validated",
        "approved",
        "adopted",
    ):
        assert forbidden not in text


def test_no_submission_scheduler_or_execution_path_is_used(monkeypatch):
    import backend.submission as submission
    import backend.workflows as workflows

    def fail(*args, **kwargs):
        raise AssertionError("execution or scheduler path should not be used")

    monkeypatch.setattr(submission, "build_sbatch_script", fail)
    monkeypatch.setattr(workflows, "build_atomate2_flow_from_spec", fail)

    payload = build_input_reference_payload(request_for("static"), include_provenance=False)

    assert payload["status"] == "ok"


def test_reference_generation_does_not_import_web_remote_or_execution_machinery():
    code = f"""
import json
import sys
from backend.calculations.input_reference import build_input_reference_payload
request = {json.dumps(request_for("static"))!r}
build_input_reference_payload(json.loads(request), include_provenance=False)
blocked = [
    'fastapi',
    'paramiko',
    'custodian',
    'jobflow',
    'backend.remote_submission',
    'backend.paramiko_remote',
    'backend.execution',
]
print(json.dumps({{name: name in sys.modules for name in blocked}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    loaded = json.loads(completed.stdout)
    assert loaded == {name: False for name in loaded}


def test_multistage_reference_uses_workflow_stage_metadata():
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.PBE),
        ],
        recipe="relax_static",
    )
    payload = build_input_reference_payload(
        {
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": SI_POSCAR,
            },
            "workflow_spec": workflow.to_dict(),
            "resources": {"ntasks": 24, "mem_gb": 128},
            "potcar_functional": "PBE_64",
        },
        include_provenance=False,
    )

    assert payload["status"] == "ok"
    assert payload["workflow"]["stage_directories"] == ["stage_01", "stage_02"]
    assert [stage["directory"] for stage in payload["reference"]["stages"]] == [
        "stage_01",
        "stage_02",
    ]
    assert [stage["stage_type"] for stage in payload["reference"]["stages"]] == [
        "relax",
        "static",
    ]


def test_hse06_band_structure_reference_preserves_combined_line_kpoints():
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.HSE06),
            StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
        ],
        recipe="custom",
    )
    payload = build_input_reference_payload(
        {
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": SI_POSCAR,
            },
            "workflow_spec": workflow.to_dict(),
            "resources": {"ntasks": 24, "mem_gb": 128},
            "potcar_functional": "PBE_64",
        },
        include_provenance=False,
    )
    band_stage = payload["reference"]["stages"][1]
    kpoints = band_stage["kpoints"]["as_dict"]

    assert payload["status"] == "ok"
    assert band_stage["generator"]["selected_atomate2"]["input_set_generator"].endswith(
        "HSEBSSetGenerator"
    )
    assert band_stage["generator"]["selected_atomate2"]["maker"].endswith("HSEBSMaker")
    assert kpoints["generation_style"] == "Reciprocal"
    assert any(label == "\\Gamma" for label in kpoints["labels"])
    assert any(weight == 0 for weight in kpoints["kpts_weights"])
    assert any(weight > 0 for weight in kpoints["kpts_weights"])
    assert band_stage["incar"]["settings"]["LHFCALC"] is True
