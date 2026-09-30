import json

from backend.calculations.models import StageType, Theory
from backend.calculations.vasp_stage_definitions import (
    BAND_STRUCTURE_LINE_DENSITY_DEFAULT,
    HSE_DOS_RECIPROCAL_DENSITY_DEFAULT,
    HSE_BAND_STRUCTURE_RECIPROCAL_DENSITY_DEFAULT,
    apply_hse_dos_base_incar_settings,
    apply_hse_band_structure_base_incar_settings,
    apply_relax_compatibility_incar_settings,
    apply_stage_base_incar_settings,
    apply_stage_restart_incar_settings,
    apply_static_compatibility_incar_settings,
    describe_stage,
    list_stage_definitions,
)


def test_stage_definition_introspection_is_json_safe_and_read_only():
    definitions = list_stage_definitions()
    assert {definition["stage_type"] for definition in definitions} == {
        "relax",
        "static",
        "dos",
        "band_structure",
    }
    json.dumps(definitions, sort_keys=True)

    first = definitions[0]
    first["base_bmd_incar_amendments"]["mutated"] = True
    assert "mutated" not in list_stage_definitions()[0]["base_bmd_incar_amendments"]


def test_relax_definition_preserves_current_executable_and_facade_defaults():
    relax_base = apply_stage_base_incar_settings(
        StageType.RELAX,
        {},
        explicit_user_settings={},
    )
    assert relax_base["ENCUT"] == 580
    assert relax_base["EDIFF"] == 1e-6
    assert relax_base["ADDGRID"] is True
    assert relax_base["EDIFFG"] == -0.01
    assert relax_base["ALGO"] == "Fast"
    assert relax_base["LCHARG"] is False
    assert relax_base["LWAVE"] is False
    assert relax_base["LORBIT"] is None

    relax_compatibility = apply_relax_compatibility_incar_settings({})
    assert "ENCUT" not in relax_compatibility
    assert "EDIFF" not in relax_compatibility
    assert relax_compatibility["ADDGRID"] is True
    assert relax_compatibility["EDIFFG"] == -0.01


def test_static_definition_preserves_final_prep_and_facade_defaults():
    static_final = apply_stage_base_incar_settings(StageType.STATIC, {}, intent="final")
    assert static_final["ENCUT"] == 620
    assert static_final["LWAVE"] is False
    assert static_final["LCHARG"] is True
    assert static_final["ISMEAR"] == -5
    assert static_final["SIGMA"] == 0.05
    assert static_final["NEDOS"] == 4001
    assert static_final["LORBIT"] == 11
    assert static_final["LVTOT"] is True
    assert static_final["LAECHG"] is True
    assert static_final["LELF"] is True

    static_prep = apply_stage_base_incar_settings(StageType.STATIC, {}, intent="prep")
    assert static_prep["ENCUT"] == 520
    assert static_prep["LWAVE"] is True
    assert static_prep["LCHARG"] is True
    assert static_prep["LVTOT"] is False
    assert static_prep["LAECHG"] is False
    assert static_prep["LELF"] is False
    assert static_prep["SIGMA"] is None
    assert static_prep["NEDOS"] == 3001

    static_compatibility = apply_static_compatibility_incar_settings({})
    assert static_compatibility["LWAVE"] is True
    assert static_compatibility["SIGMA"] is None
    assert static_compatibility["NEDOS"] == 3001


def test_dos_and_pbe_band_descriptions_expose_restart_and_resource_policy():
    static_final = apply_stage_base_incar_settings(StageType.STATIC, {}, intent="final")
    dos_restart = apply_stage_restart_incar_settings(StageType.DOS, static_final)
    assert dos_restart["ICHARG"] == 11

    dos_description = describe_stage(StageType.DOS)
    assert dos_description["selected_atomate2"]["input_set_generator"].endswith(
        "NonSCFSetGenerator"
    )
    assert dos_description["selected_atomate2"]["maker"].endswith("NonSCFMaker")
    assert dos_description["selected_atomate2"]["generator_mode"] == "uniform"
    assert dos_description["restart_policy"]["incar_amendments"] == {"ICHARG": 11}
    assert dos_description["resource_policy"]["automatic_ncore_eligible"] is True

    band_description = describe_stage(StageType.BAND_STRUCTURE, Theory.PBE)
    assert band_description["selected_atomate2"]["input_set_generator"].endswith(
        "NonSCFSetGenerator"
    )
    assert band_description["selected_atomate2"]["maker"].endswith("NonSCFMaker")
    assert band_description["selected_atomate2"]["generator_mode"] == "line"
    assert (
        band_description["kpoints_policy"]["default_parameters"]["line_density"]
        == BAND_STRUCTURE_LINE_DENSITY_DEFAULT
    )
    assert band_description["restart_policy"]["incar_amendments"] == {"ICHARG": 11}
    assert band_description["resource_policy"]["automatic_ncore_eligible"] is False


def test_hse_descriptions_compose_stage_base_and_theory_policy_separately():
    hse_static_description = describe_stage(StageType.STATIC, Theory.HSE06)
    hse_static_base = hse_static_description["base_bmd_incar_amendments"]
    hse_static_theory = hse_static_description["applicable_theory_amendments"]
    assert "LHFCALC" not in hse_static_base.get("final_defaults", {})
    assert hse_static_theory["LHFCALC"] is True
    assert hse_static_theory["PRECFOCK"] == "Accurate"
    assert hse_static_theory["ISMEAR"] == 0
    assert hse_static_description["selected_atomate2"]["input_set_generator"].endswith(
        "StaticSetGenerator"
    )

    hse_dos_description = describe_stage(StageType.DOS, Theory.HSE06)
    assert hse_dos_description["theory_supported_for_stage"] is True
    assert hse_dos_description["selected_atomate2"]["input_set_generator"].endswith(
        "HSEBSSetGenerator"
    )
    assert hse_dos_description["selected_atomate2"]["maker"].endswith("HSEBSMaker")
    assert hse_dos_description["selected_atomate2"]["generator_mode"] == "uniform"
    assert (
        hse_dos_description["selected_atomate2"]["extra_parameters"][
            "reciprocal_density"
        ]
        == HSE_DOS_RECIPROCAL_DENSITY_DEFAULT
    )
    assert hse_dos_description["restart_policy"]["requires_previous_stage"] is True
    assert hse_dos_description["restart_policy"]["incar_amendments"] == {}
    assert "may copy CHGCAR" in hse_dos_description["restart_policy"]["description"]
    assert "no ICHARG or ISTART" in hse_dos_description["restart_policy"]["description"]
    hse_dos_stage_amendments = hse_dos_description[
        "theory_stage_bmd_incar_amendments"
    ]
    assert hse_dos_stage_amendments["defaults"]["NEDOS"] == 4001
    assert hse_dos_stage_amendments["defaults"]["LORBIT"] == 11
    assert hse_dos_stage_amendments["encut_floor"] == 620
    assert hse_dos_description["applicable_theory_amendments"]["LHFCALC"] is True
    assert hse_dos_description["applicable_theory_amendments"]["PRECFOCK"] == "Fast"
    assert hse_dos_description["applicable_theory_amendments"]["ISMEAR"] == -5

    hse_band_description = describe_stage(StageType.BAND_STRUCTURE, Theory.HSE06)
    assert hse_band_description["selected_atomate2"]["input_set_generator"].endswith(
        "HSEBSSetGenerator"
    )
    assert hse_band_description["selected_atomate2"]["maker"].endswith("HSEBSMaker")
    assert (
        hse_band_description["selected_atomate2"]["extra_parameters"][
            "reciprocal_density"
        ]
        == HSE_BAND_STRUCTURE_RECIPROCAL_DENSITY_DEFAULT
    )
    assert "custodian_policy" in hse_band_description["selected_atomate2"][
        "extra_parameters"
    ]
    hse_band_kpoints = hse_band_description["kpoints_policy"]["default_parameters"]
    assert (
        hse_band_kpoints["reciprocal_density"]
        == HSE_BAND_STRUCTURE_RECIPROCAL_DENSITY_DEFAULT
    )
    assert "custodian_policy" not in hse_band_kpoints
    hse_band_stage_amendments = hse_band_description[
        "theory_stage_bmd_incar_amendments"
    ]
    assert hse_band_stage_amendments["defaults"]["LORBIT"] == 11
    assert hse_band_stage_amendments["encut_floor"] == 620
    assert "LHFCALC" not in hse_band_stage_amendments["defaults"]
    assert "LSORBIT" not in hse_band_stage_amendments["defaults"]
    assert "LDAU" not in hse_band_stage_amendments["defaults"]
    assert hse_band_description["applicable_theory_amendments"]["LHFCALC"] is True
    assert hse_band_description["applicable_theory_amendments"]["PRECFOCK"] == "Fast"
    assert hse_band_description["restart_policy"]["incar_amendments"] == {}
    assert "may copy CHGCAR" in hse_band_description["restart_policy"]["description"]
    assert "no ICHARG or ISTART" in hse_band_description["restart_policy"]["description"]


def test_hse_dos_base_helper_does_not_duplicate_hse_functional_settings():
    hse_dos_base = apply_hse_dos_base_incar_settings({})
    assert hse_dos_base["ENCUT"] == 620
    assert hse_dos_base["LORBIT"] == 11
    assert hse_dos_base["NEDOS"] == 4001
    assert hse_dos_base["PREC"] == "Accurate"
    assert "LHFCALC" not in hse_dos_base
    assert "ISMEAR" not in hse_dos_base


def test_hse_band_base_helper_does_not_duplicate_hse_functional_settings():
    hse_band_base = apply_hse_band_structure_base_incar_settings({})
    assert hse_band_base["ENCUT"] == 620
    assert hse_band_base["LORBIT"] == 11
    assert hse_band_base["PREC"] == "Accurate"
    assert "LHFCALC" not in hse_band_base
    assert "ISMEAR" not in hse_band_base


def test_unsupported_theory_stage_description_does_not_expose_amendments():
    r2scan_dos_description = describe_stage(StageType.DOS, Theory.R2SCAN)
    assert r2scan_dos_description["theory_supported_for_stage"] is False
    assert r2scan_dos_description["applicable_theory_amendments"] == {}
