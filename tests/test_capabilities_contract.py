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
from backend.calculations.models import Modifier, StageSpec
from backend.calculations.registry import (
    supported_stage_modifier_combinations,
    validate_stage_spec,
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
    assert payload["source"]["repository"] == "bmd_compute"
    assert payload["scope"] == SCOPE
    assert payload["contract"] == {
        "automatic_default_treatments": "Read-only policy describing BMD-managed Desired Output treatment resolution.",
        "base_stage_definitions": "Theory-neutral stage definitions from list_stage_definitions().",
        "capabilities": "Supported stage/theory descriptions from describe_stage(); unsupported combinations are not invented.",
        "stage_modifier_support": "Complete stage-local explicit modifier sets accepted by the BMD Compute registry for each supported stage/theory pair.",
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


def test_stage_modifier_support_is_derived_from_registry_and_validates():
    payload = build_capability_payload(include_provenance=False)
    support = {
        (entry["stage_type"], entry["theory"]): entry
        for entry in payload["stage_modifier_support"]
    }
    capability_keys = {
        (entry["stage_type"], entry["theory"])
        for entry in payload["capabilities"]
    }

    assert set(support) == capability_keys
    for (stage_type, theory), entry in support.items():
        expected = [
            sorted(modifier.value for modifier in modifiers)
            for modifiers in supported_stage_modifier_combinations(stage_type, theory)
        ]
        assert entry["supported_modifier_combinations"] == expected
        assert entry["source"].endswith("supported_stage_modifier_combinations")
        for modifiers in entry["supported_modifier_combinations"]:
            validate_stage_spec(StageSpec(stage_type, theory, set(modifiers)))


def test_stage_modifier_support_reports_representative_allowed_and_forbidden_sets():
    payload = build_capability_payload(include_provenance=False)
    support = {
        (entry["stage_type"], entry["theory"]): {
            tuple(combination)
            for combination in entry["supported_modifier_combinations"]
        }
        for entry in payload["stage_modifier_support"]
    }
    pbe_dos = support[("dos", "pbe")]
    pbe_static = support[("static", "pbe")]
    hse_static = support[("static", "hse06")]

    assert ("dft_u", "spin_polarized") in pbe_dos
    assert all(Modifier.SOC.value not in combination for combination in pbe_dos)
    assert ("soc",) in pbe_static
    assert ("soc",) in hse_static
    assert ("dispersion",) not in hse_static


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
    assert policy["failure_policy"]["dimensionality_analysis"] == {
        "status": "analysis_failed",
        "action": "reject_before_preview_preparation_or_submission",
        "diagnostic_code": "automatic_dispersion_dimensionality_analysis_failed",
    }
    assert [
        treatment["consideration_id"]
        for treatment in policy["treatments"]
    ] == [
        "spin.composition_screen",
        "dispersion.two_dimensional_connectivity",
        "soc.heavy_elements",
        "dftu.mp_oxide_fluoride",
    ]
    assert policy["treatments"][1]["method"] == "dftd3-bj"
    assert policy["treatments"][1]["incar_effect"] == {"IVDW": 12}
    assert policy["treatments"][2]["modifier"] == "soc"
    assert policy["treatments"][2]["excluded_stage_types"] == ["relax"]
    assert policy["treatments"][2]["executable"] == "vasp_ncl"
    dft_u = policy["treatments"][3]
    assert dft_u["modifier"] == "dft_u"
    assert dft_u["policy_id"] == "bmd_compute.dft_u"
    assert dft_u["policy_version"] == 1
    assert dft_u["application"] == [
        {"stage_type": "relax", "theory": "pbe"},
        {"stage_type": "static", "theory": "pbe"},
    ]
    assert dft_u["excluded_theories"] == ["hse06"]
    assert policy["advisory_only"] == []


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


def test_machine_readable_contract_identifies_compute_methodology_payload():
    payload = build_capability_payload(include_provenance=False)

    # Consumers rely on these machine-readable fields for compatibility.
    assert isinstance(payload["schema_version"], int)
    assert payload["schema_version"] == SCHEMA_VERSION == 1
    assert payload["source"]["repository"] == "bmd_compute"
    assert isinstance(payload["scope"], str) and payload["scope"]


def test_scope_text_states_compute_owns_executable_methodology():
    payload = build_capability_payload(include_provenance=False)
    scope_texts = [payload["scope"], payload["automatic_default_treatments"]["scope"]]
    scope_texts.extend(entry["scope"] for entry in payload["capabilities"])

    assert "executable calculation methodology" in payload["scope"]
    for text in scope_texts:
        assert "not a methodology authority" not in text
        assert "BMDex" not in text
