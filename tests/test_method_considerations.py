from __future__ import annotations

import builtins
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.method_considerations import (
    ALREADY_SELECTED,
    DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
    INVALID_WORKFLOW,
    NOT_SELECTED,
    DFT_U_CONSIDERATION_ID,
    POLICY_VERSION,
    SOC_HEAVY_ELEMENTS_CONSIDERATION_ID,
    SOC_RECOMMENDED_STATUS,
    SOC_TRIGGER_CLASSES,
    SOC_TRIGGER_ELEMENT_CLASSES,
    SPIN_COMPOSITION_CONSIDERATION_ID,
    SPIN_RECOMMENDED_STATUS,
    SPIN_TRIGGER_CLASSES,
    SPIN_TRIGGER_ELEMENT_CLASSES,
    UNSUPPORTED_FOR_WORKFLOW,
    WORKFLOW_NOT_PROVIDED,
    detect_structure_features,
    method_consideration_payload,
    method_considerations_for_structure,
)
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure
from backend.structure_dimensionality import ANALYSIS_FAILED, StructureDimensionalityObservation


BI2SE3_POSCAR = """Bi2Se3
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Bi Se
2 3
direct
0.0 0.0 0.0
0.25 0.25 0.25
0.5 0.5 0.5
0.75 0.75 0.75
0.125 0.625 0.375
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

SI_POSCAR_WITH_BI_COMMENT = """Bi appears only in this POSCAR comment
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


def structure_for_symbols(symbols: list[str]) -> Structure:
    coords = [
        [
            (index * 0.173) % 1,
            (index * 0.317) % 1,
            (index * 0.463) % 1,
        ]
        for index, _ in enumerate(symbols)
    ]
    return Structure(Lattice.cubic(max(5, len(symbols) + 3)), symbols, coords)


def bi2se3_structure() -> Structure:
    return structure_for_symbols(["Bi", "Bi", "Se", "Se", "Se"])


def layered_bi2se3_structure() -> Structure:
    return Structure.from_spacegroup(
        "R-3m",
        Lattice.hexagonal(4.143, 28.636),
        ["Bi", "Se", "Se"],
        [[0, 0, 0.399], [0, 0, 0.0], [0, 0, 0.211]],
    )


def sns2_structure() -> Structure:
    return parse_structure(SNS2_POSCAR)


def si_structure() -> Structure:
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def mos2_structure() -> Structure:
    return Structure.from_spacegroup(
        "P6_3/mmc",
        Lattice.hexagonal(3.16, 12.30),
        ["Mo", "S"],
        [[1 / 3, 2 / 3, 1 / 4], [1 / 3, 2 / 3, 0.621]],
    )


def nacl_structure() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def srtio3_structure() -> Structure:
    return Structure.from_spacegroup(
        "Pm-3m",
        Lattice.cubic(3.905),
        ["Sr", "Ti", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    )


def pbe_relax_workflow(*, dispersion: bool = False) -> WorkflowSpec:
    modifiers = {Modifier.DISPERSION} if dispersion else set()
    return WorkflowSpec([StageSpec(StageType.RELAX, Theory.PBE, modifiers)])


def pbe_static_workflow(*, soc: bool = False) -> WorkflowSpec:
    modifiers = {Modifier.SOC} if soc else set()
    return WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, modifiers)])


def pbe_static_dispersion_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        [StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DISPERSION})]
    )


def pbe_static_spin_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        [StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED})]
    )


def only_consideration(structure, *, workflow=None):
    considerations = method_considerations_for_structure(structure, workflow=workflow)
    assert len(considerations) == 1
    return considerations[0]


def consideration_by_id(structure, consideration_id, *, workflow=None):
    matches = [
        consideration
        for consideration in method_considerations_for_structure(structure, workflow=workflow)
        if consideration.id == consideration_id
    ]
    assert len(matches) == 1
    return matches[0]


def detection_id(symbol: str) -> str:
    return f"element.{symbol.lower()}.present"


def test_policy_v5_membership_is_explicit_and_centralized():
    assert POLICY_VERSION == 5
    assert SOC_TRIGGER_CLASSES == {
        "4d_transition_metals": ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"),
        "5d_transition_metals": ("Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"),
        "lanthanides": (
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        ),
        "actinides": (
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
        ),
        "heavy_p_block": ("Tl", "Pb", "Bi", "Po"),
    }
    assert SOC_TRIGGER_ELEMENT_CLASSES["Bi"] == ("heavy_p_block",)
    assert SOC_TRIGGER_ELEMENT_CLASSES["Pt"] == ("5d_transition_metals",)
    assert SPIN_TRIGGER_CLASSES == {
        "3d_spin_screen": ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"),
        "4d_spin_screen": ("Mo", "Tc", "Ru", "Rh"),
        "5d_spin_screen": ("Re", "Os", "Ir"),
        "lanthanide_spin_screen": (
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
        ),
        "actinide_spin_screen": ("U", "Np", "Pu", "Am", "Cm", "Bk", "Cf"),
    }
    assert SPIN_TRIGGER_ELEMENT_CLASSES["Fe"] == ("3d_spin_screen",)
    assert SPIN_TRIGGER_ELEMENT_CLASSES["Eu"] == ("lanthanide_spin_screen",)
    assert SPIN_TRIGGER_ELEMENT_CLASSES["Ir"] == ("5d_spin_screen",)


@pytest.mark.parametrize(
    ("symbol", "class_name"),
    [
        ("Mo", "4d_transition_metals"),
        ("Pt", "5d_transition_metals"),
        ("Eu", "lanthanides"),
        ("U", "actinides"),
        ("Pb", "heavy_p_block"),
        ("Bi", "heavy_p_block"),
        ("Y", "4d_transition_metals"),
        ("Cd", "4d_transition_metals"),
        ("Hf", "5d_transition_metals"),
        ("Hg", "5d_transition_metals"),
        ("La", "lanthanides"),
        ("Lu", "lanthanides"),
        ("Ac", "actinides"),
        ("Lr", "actinides"),
        ("Tl", "heavy_p_block"),
        ("Po", "heavy_p_block"),
    ],
)
def test_soc_trigger_elements_produce_element_level_detections(symbol, class_name):
    detections = detect_structure_features(structure_for_symbols([symbol]))

    assert len(detections) == 1
    detection = detections[0]
    assert detection.id == detection_id(symbol)
    assert detection.type == "element_present"
    assert detection.element == symbol
    assert detection.observed is True
    assert detection.soc_trigger_classes == (class_name,)
    assert detection.observed_evidence["element"] == symbol
    assert detection.observed_evidence["elements"] == [symbol]
    assert detection.observed_evidence["soc_trigger_classes"] == [class_name]
    assert detection.to_dict()["source"] == "pymatgen.Structure.composition.elements"


@pytest.mark.parametrize(
    ("symbol", "class_name"),
    [
        ("Ti", "3d_spin_screen"),
        ("Fe", "3d_spin_screen"),
        ("Ni", "3d_spin_screen"),
        ("Mo", "4d_spin_screen"),
        ("Ru", "4d_spin_screen"),
        ("Re", "5d_spin_screen"),
        ("Ir", "5d_spin_screen"),
        ("Ce", "lanthanide_spin_screen"),
        ("Eu", "lanthanide_spin_screen"),
        ("Yb", "lanthanide_spin_screen"),
        ("U", "actinide_spin_screen"),
        ("Pu", "actinide_spin_screen"),
        ("Cf", "actinide_spin_screen"),
    ],
)
def test_spin_trigger_elements_produce_factual_element_level_detections(symbol, class_name):
    detections = detect_structure_features(structure_for_symbols([symbol]))

    assert len(detections) == 1
    detection = detections[0]
    assert detection.id == detection_id(symbol)
    assert detection.type == "element_present"
    assert detection.element == symbol
    assert detection.observed is True
    assert detection.spin_trigger_classes == (class_name,)
    assert detection.observed_evidence["element"] == symbol
    assert detection.observed_evidence["elements"] == [symbol]
    assert detection.observed_evidence["spin_trigger_classes"] == [class_name]


@pytest.mark.parametrize("symbol", ["Si", "O", "Sc", "Cu", "Zn", "Sr", "In", "Ba"])
def test_elements_outside_both_screens_produce_no_method_consideration(symbol):
    structure = structure_for_symbols([symbol])

    assert detect_structure_features(structure) == ()
    assert method_considerations_for_structure(structure) == ()
    assert method_consideration_payload(structure)["considerations"] == []


@pytest.mark.parametrize("symbol", ["Si", "Sc", "Cu", "Zn", "Nb", "W", "Pt", "La", "Lu", "Th", "Pa"])
def test_spin_policy_exclusions_do_not_emit_spin_considerations(symbol):
    structure = structure_for_symbols([symbol])
    consideration_ids = [
        consideration.id
        for consideration in method_considerations_for_structure(structure)
    ]

    assert SPIN_COMPOSITION_CONSIDERATION_ID not in consideration_ids


def test_bi_is_detected_from_actual_structure_species_not_poscar_comments():
    structure = parse_structure(SI_POSCAR_WITH_BI_COMMENT, "poscar")

    assert structure.composition.reduced_formula == "Si"
    assert detect_structure_features(structure) == ()
    assert method_considerations_for_structure(structure) == ()


def test_heavy_element_detection_produces_single_conservative_soc_consideration():
    consideration = only_consideration(bi2se3_structure())

    assert consideration.id == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID
    assert consideration.method == "soc"
    assert consideration.modifier == "soc"
    assert consideration.status == SOC_RECOMMENDED_STATUS
    assert consideration.trigger_detection_ids == (detection_id("Bi"),)
    assert consideration.trigger_elements == ("Bi",)
    assert consideration.trigger_classes == ("heavy_p_block",)
    assert consideration.selection_state == WORKFLOW_NOT_PROVIDED
    assert consideration.applicable_stage_types == ("band_structure", "dos", "static")
    assert consideration.bmd_compute_support["supported_stage_capabilities"] == [
        {
            "stage_type": "band_structure",
            "stage_label": "Band Structure",
            "theory": "hse06",
            "theory_label": "HSE06",
        },
        {
            "stage_type": "dos",
            "stage_label": "Density of States",
            "theory": "hse06",
            "theory_label": "HSE06",
        },
        {
            "stage_type": "static",
            "stage_label": "Static Energy",
            "theory": "hse06",
            "theory_label": "HSE06",
        },
        {
            "stage_type": "static",
            "stage_label": "Static Energy",
            "theory": "pbe",
            "theory_label": "PBE",
        }
    ]
    assert consideration.bmd_compute_support["workflow"] == {
        "provided": False,
        "support_status": WORKFLOW_NOT_PROVIDED,
        "selection_state": WORKFLOW_NOT_PROVIDED,
        "selected_stage_indices": [],
        "supported_stage_indices": [],
        "unsupported_selected_stage_indices": [],
    }
    reason = consideration.reason.lower()
    assert "may be important" in reason
    for forbidden in ("required", "necessary", "mandatory", "invalid"):
        assert forbidden not in reason
    assert consideration.policy_source["policy_version"] == 5
    assert consideration.policy_source["rule_id"] == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID
    assert "composition-based screening" in consideration.limitations[0]


def test_spin_screen_detection_produces_single_conservative_spin_consideration():
    # Fe-O also triggers the independent DFT+U consideration (policy v5); this
    # test concerns the single spin consideration only.
    considerations = method_considerations_for_structure(structure_for_symbols(["Fe", "O"]))
    assert [item.id for item in considerations] == [
        SPIN_COMPOSITION_CONSIDERATION_ID,
        DFT_U_CONSIDERATION_ID,
    ]
    consideration = considerations[0]

    assert consideration.id == SPIN_COMPOSITION_CONSIDERATION_ID
    assert consideration.method == "spin_polarisation"
    assert consideration.modifier == "spin_polarized"
    assert consideration.display_name == "Spin Polarisation"
    assert consideration.status == SPIN_RECOMMENDED_STATUS
    assert consideration.trigger_detection_ids == (detection_id("Fe"),)
    assert consideration.trigger_elements == ("Fe",)
    assert consideration.trigger_classes == ("3d_spin_screen",)
    assert consideration.selection_state == WORKFLOW_NOT_PROVIDED
    supported_pairs = {
        (capability["stage_type"], capability["theory"])
        for capability in consideration.bmd_compute_support["supported_stage_capabilities"]
    }
    assert {
        ("relax", "pbe"),
        ("static", "pbe"),
        ("dos", "pbe"),
        ("band_structure", "pbe"),
        ("relax", "hse06"),
        ("static", "hse06"),
        ("dos", "hse06"),
        ("band_structure", "hse06"),
    }.issubset(supported_pairs)
    reason = consideration.reason.lower()
    assert "may be relevant" in reason
    assert "consider enabling spin polarised" in reason
    for forbidden in ("is magnetic", "required", "necessary", "mandatory"):
        assert forbidden not in reason
    assert consideration.policy_source["policy_version"] == 5
    assert consideration.policy_source["rule_id"] == SPIN_COMPOSITION_CONSIDERATION_ID
    assert "does not establish that the material is magnetic" in consideration.limitations[0]


def test_sns2_dimensionality_observation_produces_dispersion_consideration():
    consideration = only_consideration(sns2_structure())

    assert consideration.id == DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID
    assert consideration.method == "dispersion"
    assert consideration.modifier == "dispersion"
    assert consideration.display_name == "Dispersion Correction"
    assert consideration.status == "recommended_for_consideration"
    assert consideration.trigger_elements == ()
    assert consideration.trigger_classes == ("two_dimensional_bonded_connectivity",)
    assert consideration.selection_state == WORKFLOW_NOT_PROVIDED
    assert consideration.applicable_stage_types == ("relax", "static")
    assert consideration.bmd_compute_support["supported_stage_capabilities"] == [
        {
            "stage_type": "relax",
            "stage_label": "Geometry Optimisation",
            "theory": "pbe",
            "theory_label": "PBE",
        },
        {
            "stage_type": "static",
            "stage_label": "Static Energy",
            "theory": "pbe",
            "theory_label": "PBE",
        },
    ]
    trigger = consideration.observed_evidence["triggers"][0]
    assert trigger["type"] == "structure_dimensionality"
    assert trigger["label"] == "Two-dimensional bonded connectivity"
    assert trigger["dimensionality"] == 2
    assert trigger["method"]["id"] == "pymatgen.crystalnn_larsen_dimensionality"
    assert trigger["method"]["bonding"].endswith("CrystalNN.get_bonded_structure")
    assert trigger["method"]["dimensionality"].endswith("get_dimensionality_larsen")
    assert trigger["components"][0]["formula"] == "SnS2"
    assert trigger["components"][0]["dimensionality"] == 2
    assert trigger["components"][0]["orientation"] == [0, 0, 1]
    assert trigger["components"][0]["site_ids"] == [0, 1, 2]
    assert "may therefore be important" in consideration.reason
    assert "Consider enabling a dispersion correction" in consideration.reason
    for forbidden in ("definitely", "required", "D3(BJ) is scientifically correct", "vdW material"):
        assert forbidden.lower() not in consideration.reason.lower()
    assert consideration.policy_source["policy_version"] == 5
    assert consideration.policy_source["rule_id"] == (
        DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID
    )
    assert "does not establish the magnitude" in consideration.limitations[0]


@pytest.mark.parametrize(
    "factory",
    [
        sns2_structure,
        layered_bi2se3_structure,
        mos2_structure,
    ],
)
def test_two_dimensional_controls_emit_dispersion_consideration(factory):
    consideration_ids = [
        consideration.id
        for consideration in method_considerations_for_structure(factory())
    ]

    assert DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID in consideration_ids


@pytest.mark.parametrize(
    "factory",
    [
        si_structure,
        nacl_structure,
        srtio3_structure,
    ],
)
def test_three_dimensional_controls_do_not_emit_dispersion_consideration(factory):
    consideration_ids = [
        consideration.id
        for consideration in method_considerations_for_structure(factory())
    ]

    assert DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID not in consideration_ids


def test_bi2se3_emits_independent_dispersion_and_soc_considerations():
    considerations = method_considerations_for_structure(layered_bi2se3_structure())

    assert [consideration.id for consideration in considerations] == [
        DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
        SOC_HEAVY_ELEMENTS_CONSIDERATION_ID,
    ]
    dispersion, soc = considerations
    assert dispersion.trigger_elements == ()
    assert dispersion.trigger_classes == ("two_dimensional_bonded_connectivity",)
    assert dispersion.observed_evidence["triggers"][0]["dimensionality"] == 2
    assert soc.trigger_elements == ("Bi",)
    assert soc.trigger_classes == ("heavy_p_block",)
    assert dispersion.reason != soc.reason
    assert dispersion.limitations != soc.limitations


def test_failed_dimensionality_analysis_does_not_emit_dispersion_consideration(monkeypatch):
    monkeypatch.setattr(
        "backend.calculations.method_considerations.observe_structure_dimensionality",
        lambda structure: StructureDimensionalityObservation(
            status=ANALYSIS_FAILED,
            dimensionality=None,
            reason="test failure",
        ),
    )

    payload = method_consideration_payload(sns2_structure())

    assert payload["structure_observations"]["dimensionality"]["status"] == ANALYSIS_FAILED
    assert payload["considerations"] == []


def test_dispersion_consideration_workflow_statuses_are_stage_local_and_non_mutating():
    no_workflow = only_consideration(sns2_structure())
    assert no_workflow.selection_state == WORKFLOW_NOT_PROVIDED

    for workflow in (
        pbe_relax_workflow(),
        pbe_static_workflow(),
    ):
        before = workflow.to_dict()
        not_selected = consideration_by_id(
            sns2_structure(),
            DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
            workflow=workflow,
        )
        assert not_selected.selection_state == NOT_SELECTED
        assert not_selected.bmd_compute_support["workflow"]["supported_stage_indices"] == [1]
        assert not_selected.bmd_compute_support["workflow"]["selected_stage_indices"] == []
        assert workflow.to_dict() == before

    for workflow in (
        pbe_relax_workflow(dispersion=True),
        pbe_static_dispersion_workflow(),
    ):
        before = workflow.to_dict()
        already_selected = consideration_by_id(
            sns2_structure(),
            DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
            workflow=workflow,
        )
        assert already_selected.selection_state == ALREADY_SELECTED
        assert already_selected.bmd_compute_support["workflow"]["supported_stage_indices"] == [1]
        assert already_selected.bmd_compute_support["workflow"]["selected_stage_indices"] == [1]
        assert workflow.to_dict() == before

    # SOC + D3 is an allowed combination on PBE Static, so an SOC-only PBE
    # Static stage is a supported, not-yet-selected dispersion candidate.
    soc_static = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC})])
    soc_static_before = soc_static.to_dict()
    soc_candidate = consideration_by_id(
        sns2_structure(),
        DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
        workflow=soc_static,
    )
    assert soc_candidate.selection_state == NOT_SELECTED
    assert soc_candidate.bmd_compute_support["workflow"]["supported_stage_indices"] == [1]
    assert soc_candidate.bmd_compute_support["workflow"]["selected_stage_indices"] == []
    assert soc_static.to_dict() == soc_static_before

    for workflow in (
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06)]),
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC})]),
        WorkflowSpec([StageSpec(StageType.DOS, Theory.PBE)]),
    ):
        before = workflow.to_dict()
        unsupported = consideration_by_id(
            sns2_structure(),
            DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
            workflow=workflow,
        )
        assert unsupported.selection_state in {
            UNSUPPORTED_FOR_WORKFLOW,
            INVALID_WORKFLOW,
        }
        if unsupported.selection_state == UNSUPPORTED_FOR_WORKFLOW:
            assert unsupported.bmd_compute_support["workflow"]["supported_stage_indices"] == []
        assert workflow.to_dict() == before


def test_dispersion_consideration_does_not_select_modifier_or_change_generated_inputs():
    structure = sns2_structure()
    workflow = pbe_static_workflow()
    workflow_before = workflow.to_dict()
    structure_before = structure.as_dict()
    preview_before = preview_generated_inputs(
        structure,
        workflow,
        resources={"ntasks": 24},
        potcar_functional="PBE_64",
    )

    payload = method_consideration_payload(structure, workflow=workflow)

    preview_after = preview_generated_inputs(
        structure,
        workflow,
        resources={"ntasks": 24},
        potcar_functional="PBE_64",
    )
    assert payload["considerations"][0]["id"] == (
        DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID
    )
    assert payload["considerations"][0]["selection_state"] == NOT_SELECTED
    assert workflow.to_dict() == workflow_before
    assert structure.as_dict() == structure_before
    assert preview_after == preview_before
    assert "IVDW" not in preview_after["incar"]
    assert "dftd3" not in json.dumps(preview_after, sort_keys=True).lower()


def test_fe_mn_o_produces_one_spin_consideration_with_actual_spin_triggers_only():
    payload = method_consideration_payload(structure_for_symbols(["Fe", "Mn", "O"]))

    # Policy v5: an Fe/Mn oxide also receives the independent DFT+U
    # consideration; spin is still reported once, for Fe and Mn only.
    assert [consideration["id"] for consideration in payload["considerations"]] == [
        SPIN_COMPOSITION_CONSIDERATION_ID,
        DFT_U_CONSIDERATION_ID,
    ]
    dft_u = payload["considerations"][1]
    assert dft_u["trigger_elements"] == ["Fe", "Mn"]
    assert dft_u["observed_evidence"]["deciding_anion"] == "O"
    consideration = payload["considerations"][0]
    assert consideration["trigger_elements"] == ["Fe", "Mn"]
    assert consideration["trigger_detection_ids"] == [
        detection_id("Fe"),
        detection_id("Mn"),
    ]
    assert consideration["trigger_classes"] == ["3d_spin_screen"]
    assert [
        detection["element"]
        for detection in consideration["observed_evidence"]["detections"]
    ] == ["Fe", "Mn"]
    assert [
        detection["trigger_classes"]
        for detection in consideration["observed_evidence"]["detections"]
    ] == [["3d_spin_screen"], ["3d_spin_screen"]]
    assert "O" not in consideration["trigger_elements"]


def test_eu_and_ir_emit_independent_spin_and_soc_considerations():
    for symbol, spin_class, soc_class in (
        ("Eu", "lanthanide_spin_screen", "lanthanides"),
        ("Ir", "5d_spin_screen", "5d_transition_metals"),
    ):
        considerations = method_considerations_for_structure(structure_for_symbols([symbol]))

        assert [consideration.id for consideration in considerations] == [
            SPIN_COMPOSITION_CONSIDERATION_ID,
            SOC_HEAVY_ELEMENTS_CONSIDERATION_ID,
        ]
        spin, soc = considerations
        assert spin.trigger_elements == (symbol,)
        assert spin.trigger_classes == (spin_class,)
        assert spin.method == "spin_polarisation"
        assert soc.trigger_elements == (symbol,)
        assert soc.trigger_classes == (soc_class,)
        assert soc.method == "soc"
        assert spin.reason != soc.reason


def test_multi_trigger_structure_produces_one_aggregate_soc_consideration():
    structure = structure_for_symbols(["Bi", "Pt", "Se"])
    payload = method_consideration_payload(structure)

    assert [detection["id"] for detection in payload["detections"]] == [
        detection_id("Bi"),
        detection_id("Pt"),
    ]
    assert len(payload["considerations"]) == 1
    consideration = payload["considerations"][0]
    assert consideration["id"] == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID
    assert consideration["trigger_elements"] == ["Bi", "Pt"]
    assert consideration["trigger_detection_ids"] == [
        detection_id("Bi"),
        detection_id("Pt"),
    ]
    assert consideration["trigger_classes"] == [
        "5d_transition_metals",
        "heavy_p_block",
    ]
    assert consideration["observed_evidence"]["trigger_elements"] == ["Bi", "Pt"]
    assert consideration["observed_evidence"]["trigger_classes"] == [
        "5d_transition_metals",
        "heavy_p_block",
    ]
    assert [
        detection["element"]
        for detection in consideration["observed_evidence"]["detections"]
    ] == ["Bi", "Pt"]
    assert "Se" not in consideration["trigger_elements"]


def test_heavy_element_without_workflow_remains_structure_first_without_workflow_guessing():
    payload = method_consideration_payload(bi2se3_structure())
    consideration = payload["considerations"][0]

    assert payload["detections"][0]["id"] == detection_id("Bi")
    assert consideration["selection_state"] == WORKFLOW_NOT_PROVIDED
    assert consideration["bmd_compute_support"]["workflow"]["provided"] is False
    assert consideration["bmd_compute_support"]["workflow"]["supported_stage_indices"] == []


def test_spin_consideration_workflow_statuses_are_stage_local_and_non_mutating():
    no_workflow = consideration_by_id(
        structure_for_symbols(["Fe"]),
        SPIN_COMPOSITION_CONSIDERATION_ID,
    )
    assert no_workflow.selection_state == WORKFLOW_NOT_PROVIDED

    workflow = pbe_static_workflow()
    before = workflow.to_dict()
    not_selected = consideration_by_id(
        structure_for_symbols(["Fe"]),
        SPIN_COMPOSITION_CONSIDERATION_ID,
        workflow=workflow,
    )
    assert not_selected.selection_state == NOT_SELECTED
    assert not_selected.bmd_compute_support["workflow"]["supported_stage_indices"] == [1]
    assert not_selected.bmd_compute_support["workflow"]["selected_stage_indices"] == []
    assert workflow.to_dict() == before

    spin_workflow = pbe_static_spin_workflow()
    spin_before = spin_workflow.to_dict()
    already_selected = consideration_by_id(
        structure_for_symbols(["Fe"]),
        SPIN_COMPOSITION_CONSIDERATION_ID,
        workflow=spin_workflow,
    )
    assert already_selected.selection_state == ALREADY_SELECTED
    assert already_selected.bmd_compute_support["workflow"]["supported_stage_indices"] == [1]
    assert already_selected.bmd_compute_support["workflow"]["selected_stage_indices"] == [1]
    assert spin_workflow.to_dict() == spin_before

    unsupported = WorkflowSpec([StageSpec(StageType.DOS, Theory.HSE06)])
    unsupported_before = unsupported.to_dict()
    unsupported_consideration = consideration_by_id(
        structure_for_symbols(["Fe"]),
        SPIN_COMPOSITION_CONSIDERATION_ID,
        workflow=unsupported,
    )
    assert unsupported_consideration.selection_state in {
        UNSUPPORTED_FOR_WORKFLOW,
        "invalid_workflow",
    }
    assert unsupported.to_dict() == unsupported_before


def test_heavy_element_compatible_workflow_without_soc_is_not_selected():
    for workflow in (
        pbe_static_workflow(),
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.HSE06)]),
    ):
        consideration = only_consideration(
            bi2se3_structure(),
            workflow=workflow,
        )

        assert consideration.selection_state == NOT_SELECTED
        assert consideration.bmd_compute_support["workflow"]["provided"] is True
        assert consideration.bmd_compute_support["workflow"]["supported_stage_indices"] == [1]
        assert consideration.bmd_compute_support["workflow"]["selected_stage_indices"] == []


def test_heavy_element_compatible_workflow_with_soc_is_already_selected():
    workflow = pbe_static_workflow(soc=True)
    before = workflow.to_dict()

    consideration = only_consideration(
        bi2se3_structure(),
        workflow=workflow,
    )

    assert consideration.selection_state == ALREADY_SELECTED
    assert consideration.bmd_compute_support["workflow"]["supported_stage_indices"] == [1]
    assert consideration.bmd_compute_support["workflow"]["selected_stage_indices"] == [1]
    assert workflow.to_dict() == before
    assert workflow.stages[0].modifiers == frozenset({Modifier.SOC})


def test_incompatible_workflow_support_is_represented_without_workflow_changes():
    for workflow in (
        WorkflowSpec([StageSpec(StageType.RELAX, Theory.PBE)]),
        WorkflowSpec([StageSpec(StageType.RELAX, Theory.HSE06)]),
    ):
        before = workflow.to_dict()
        consideration = only_consideration(bi2se3_structure(), workflow=workflow)

        assert consideration.selection_state == UNSUPPORTED_FOR_WORKFLOW
        assert consideration.bmd_compute_support["workflow"]["support_status"] == (
            UNSUPPORTED_FOR_WORKFLOW
        )
        assert consideration.bmd_compute_support["workflow"]["supported_stage_indices"] == []
        assert workflow.to_dict() == before


def test_consideration_analysis_leaves_workflow_and_structure_unchanged():
    structure = bi2se3_structure()
    workflow = pbe_static_workflow()
    structure_before = structure.as_dict()
    workflow_before = workflow.to_dict()

    method_consideration_payload(structure, workflow=workflow)

    assert structure.as_dict() == structure_before
    assert workflow.to_dict() == workflow_before
    assert workflow.stages[0].modifiers == frozenset()


def test_consideration_analysis_does_not_change_generated_vasp_reference():
    workflow = pbe_static_workflow()
    request = {
        "structure": {
            "type": "pasted_text",
            "format": "poscar",
            "text": BI2SE3_POSCAR,
        },
        "workflow_spec": workflow.to_dict(),
        "resources": {"ntasks": 24, "mem_gb": 128},
        "potcar_functional": "PBE_64",
    }
    before = build_input_reference_payload(request, include_provenance=False)

    method_consideration_payload(parse_structure(BI2SE3_POSCAR), workflow=workflow)

    after = build_input_reference_payload(request, include_provenance=False)
    assert after["request"] == before["request"]
    assert after["workflow"] == before["workflow"]
    assert after["reference"] == before["reference"]


def test_consideration_analysis_does_not_touch_io_network_or_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("method consideration analysis must be pure")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)

    payload = method_consideration_payload(
        bi2se3_structure(),
        workflow=pbe_static_workflow(),
    )

    assert payload["considerations"][0]["selection_state"] == NOT_SELECTED


def test_consideration_generation_does_not_import_runtime_machinery():
    code = """
import json
import sys
from pymatgen.core import Lattice, Structure
from backend.calculations.method_considerations import method_consideration_payload

structure = Structure(Lattice.cubic(4.75), ["Bi"], [[0, 0, 0]])
method_consideration_payload(structure)
blocked = [
    "atomate2",
    "custodian",
    "fastapi",
    "jobflow",
    "paramiko",
    "backend.execution",
    "backend.generated_inputs",
    "backend.paramiko_remote",
    "backend.remote",
    "backend.remote_preparation",
    "backend.remote_submission",
    "backend.submission",
    "backend.workflows",
]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    loaded = json.loads(completed.stdout)
    assert loaded == {name: False for name in loaded}


def test_consideration_payload_is_json_safe_and_deterministic():
    workflow = pbe_static_workflow()
    first = method_consideration_payload(
        structure_for_symbols(["Bi", "Pt", "Se"]),
        workflow=workflow,
    )
    second = method_consideration_payload(
        structure_for_symbols(["Bi", "Pt", "Se"]),
        workflow=workflow,
    )

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert method_considerations_for_structure(
        bi2se3_structure(),
        workflow=workflow,
    ) == method_considerations_for_structure(
        bi2se3_structure(),
        workflow=workflow,
    )
