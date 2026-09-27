from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.calculations.capabilities import (
    SCHEMA_VERSION,
    SCOPE,
    build_capability_payload,
    emit_json,
)
from backend.calculations.vasp_stage_definitions import (
    describe_stage,
    list_stage_definitions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_capability_command_emits_json_only():
    completed = subprocess.run(
        [sys.executable, "-m", "backend.calculations.capabilities"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["scope"] == SCOPE


def test_capability_payload_has_versioned_contract_and_provenance_shape():
    payload = build_capability_payload(include_provenance=False)

    assert payload["schema_version"] == 1
    assert payload["scope"] == "BMD Compute executable implementation, not a methodology authority"
    assert payload["contract"] == {
        "automatic_default_treatments": "Read-only policy describing BMD-managed Desired Output treatment resolution.",
        "base_stage_definitions": "Theory-neutral stage definitions from list_stage_definitions().",
        "capabilities": "Supported stage/theory descriptions from describe_stage(); unsupported combinations are not invented.",
        "modifier_policies": "Stage-local executable modifiers with controlled options; unsupported pairings are not invented.",
    }
    assert payload["source"] == {
        "repository": "bmd_compute",
        "commit": None,
        "dirty": None,
        "provenance_available": False,
        "unavailable_reason": "git provenance unavailable",
    }

    payload_with_provenance = build_capability_payload()
    assert set(payload_with_provenance["source"]) == {
        "repository",
        "commit",
        "dirty",
        "provenance_available",
        "unavailable_reason",
    }
    assert payload_with_provenance["source"]["repository"] == "bmd_compute"
    assert payload_with_provenance["source"]["commit"] is None or isinstance(
        payload_with_provenance["source"]["commit"],
        str,
    )
    assert payload_with_provenance["source"]["dirty"] is None or isinstance(
        payload_with_provenance["source"]["dirty"],
        bool,
    )


def test_base_stage_definitions_are_represented_from_existing_api():
    payload = build_capability_payload(include_provenance=False)

    assert payload["base_stage_definitions"] == list(list_stage_definitions())
    assert {entry["stage_type"] for entry in payload["base_stage_definitions"]} == {
        "relax",
        "static",
        "dos",
        "band_structure",
    }


def test_supported_capabilities_are_stage_theory_descriptions_only():
    payload = build_capability_payload(include_provenance=False)
    capability_keys = {
        (entry["stage_type"], entry["theory"])
        for entry in payload["capabilities"]
    }

    assert ("band_structure", "hse06") in capability_keys
    assert ("dos", "hse06") in capability_keys
    assert all(
        entry["theory_supported_for_stage"] is True
        for entry in payload["capabilities"]
    )


def test_hse_band_structure_description_survives_contract():
    payload = build_capability_payload(include_provenance=False)
    hse_band = next(
        entry
        for entry in payload["capabilities"]
        if entry["stage_type"] == "band_structure" and entry["theory"] == "hse06"
    )
    expected = describe_stage("band_structure", "hse06")

    assert hse_band == expected
    assert hse_band["selected_atomate2"]["input_set_generator"].endswith(
        "HSEBSSetGenerator"
    )
    assert hse_band["selected_atomate2"]["maker"].endswith("HSEBSMaker")
    assert hse_band["applicable_theory_amendments"]["LHFCALC"] is True
    assert hse_band["theory_stage_bmd_incar_amendments"]["encut_floor"] == 620
    assert "custodian_policy" not in hse_band["kpoints_policy"]["default_parameters"]


def test_hse_dos_description_survives_contract():
    payload = build_capability_payload(include_provenance=False)
    hse_dos = next(
        entry
        for entry in payload["capabilities"]
        if entry["stage_type"] == "dos" and entry["theory"] == "hse06"
    )
    expected = describe_stage("dos", "hse06")

    assert hse_dos == expected
    assert hse_dos["selected_atomate2"]["input_set_generator"].endswith(
        "HSEBSSetGenerator"
    )
    assert hse_dos["selected_atomate2"]["maker"].endswith("HSEBSMaker")
    assert hse_dos["selected_atomate2"]["generator_mode"] == "uniform"
    assert hse_dos["applicable_theory_amendments"]["LHFCALC"] is True
    assert hse_dos["applicable_theory_amendments"]["ISMEAR"] == -5
    assert hse_dos["theory_stage_bmd_incar_amendments"]["defaults"]["NEDOS"] == 4001
    assert hse_dos["restart_policy"]["incar_amendments"] == {}


def test_dispersion_modifier_policy_survives_contract():
    payload = build_capability_payload(include_provenance=False)
    policy = payload["modifier_policies"][0]

    assert policy["modifier"] == "dispersion"
    assert policy["default_method"] == "dftd3-bj"
    assert policy["methods"] == [
        {"value": "dftd3", "label": "DFT-D3", "incar_effect": {"IVDW": 11}},
        {"value": "dftd3-bj", "label": "DFT-D3(BJ)", "incar_effect": {"IVDW": 12}},
    ]
    assert policy["phase_1_support"]["theories"] == ["pbe"]
    assert policy["phase_1_support"]["stage_types"] == ["relax", "static"]
    assert policy["phase_1_support"]["blocked_with_modifiers"] == []


def test_automatic_default_treatment_policy_survives_contract():
    payload = build_capability_payload(include_provenance=False)
    policy = payload["automatic_default_treatments"]

    assert policy["applies_to"] == {
        "workflow_mode": "bmd_managed_desired_output",
        "custom_workflow": "preserved_without_automatic_changes",
    }
    assert [
        treatment["consideration_id"]
        for treatment in policy["treatments"]
    ] == [
        "spin.composition_screen",
        "dispersion.two_dimensional_connectivity",
        "soc.heavy_elements",
    ]
    assert policy["treatments"][1]["method"] == "dftd3-bj"
    assert policy["treatments"][1]["incar_effect"] == {"IVDW": 12}
    assert policy["treatments"][2]["modifier"] == "soc"
    assert policy["treatments"][2]["excluded_stage_types"] == ["relax"]
    assert policy["treatments"][2]["executable"] == "vasp_ncl"
    assert policy["advisory_only"] == [
        {"modifier": "dft_u"},
    ]


def test_capability_payload_is_json_safe_and_deterministic():
    first = build_capability_payload(include_provenance=False)
    second = build_capability_payload(include_provenance=False)

    assert first == second
    assert emit_json(first) == emit_json(second)
    assert json.loads(emit_json(first)) == first


def test_capability_generation_does_not_import_runtime_machinery():
    code = """
import json
import sys
from backend.calculations.capabilities import build_capability_payload
build_capability_payload(include_provenance=False)
blocked = [
    'backend.config',
    'backend.workflows',
    'backend.paramiko_remote',
    'fastapi',
    'paramiko',
    'pymatgen',
    'atomate2',
    'jobflow',
]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
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


def test_capability_payload_excludes_forbidden_boundary_information():
    payload = build_capability_payload(include_provenance=False)
    text = json.dumps(payload, sort_keys=True).lower()

    forbidden_fragments = (
        "ssh",
        "password",
        "private_key",
        "key_file",
        "remote_host",
        "raw_potcar",
        "potcar_content",
        "fastapi",
        "sbatch",
        "validated",
        "approved",
        "adopted",
        "standard",
    )
    for fragment in forbidden_fragments:
        assert fragment not in text
