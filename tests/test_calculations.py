from backend.calculations.builder import build_calculation_flow
from backend.calculations.models import CalculationSpec, Modifier, Purpose, Theory
from backend.calculations.registry import (
    CalculationValidationError,
    calculation_display_name,
    calculation_form_options,
    calculation_result_stage_directory,
    calculation_stage_directories,
    calculation_spec_from_flow_spec,
    calculation_spec_from_legacy,
    desired_output_from_workflow_spec,
    desired_output_workflow_spec,
    legacy_potcar_functional_from_spec,
    legacy_workflow_from_spec,
    supported_combinations,
    theory_display_name,
    validate_calculation_spec,
)


spec = CalculationSpec(
    purpose="relax",
    theory="PBE",
    modifiers={"ions-only"},
    label="compat",
)

assert spec.purpose is Purpose.RELAX
assert spec.theory is Theory.PBE
assert spec.modifiers == frozenset({Modifier.IONS_ONLY})
assert CalculationSpec(Purpose.STATIC, "HSE06").theory is Theory.HSE06
assert Modifier.from_value("DFT+U") is Modifier.DFT_U
assert Modifier.from_value("Spin Polarised") is Modifier.SPIN_POLARIZED
assert spec.to_dict() == {
    "purpose": "relax",
    "theory": "pbe",
    "modifiers": ["ions_only"],
    "label": "compat",
}

assert legacy_workflow_from_spec(CalculationSpec(Purpose.STATIC, Theory.PBE)) == "static"
assert legacy_workflow_from_spec(CalculationSpec(Purpose.STATIC, Theory.HSE06)) == "static"
assert legacy_workflow_from_spec(CalculationSpec(Purpose.RELAX, Theory.PBE)) == "relax"
assert legacy_workflow_from_spec(CalculationSpec(Purpose.RELAX, Theory.HSE06)) == "relax"
assert (
    legacy_workflow_from_spec(CalculationSpec(Purpose.RELAX_STATIC, Theory.PBE))
    == "relax_static"
)
assert (
    legacy_workflow_from_spec(CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06))
    == "relax_static"
)
assert (
    legacy_workflow_from_spec(CalculationSpec(Purpose.DOUBLE_RELAX, Theory.PBE))
    == "double_relax"
)
assert legacy_workflow_from_spec(CalculationSpec(Purpose.DOS, Theory.PBE)) == "dos"
assert (
    legacy_workflow_from_spec(CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE))
    == "band_structure"
)
assert calculation_stage_directories(
    CalculationSpec(Purpose.DOUBLE_RELAX, Theory.PBE)
) == ("relax_01", "relax_02")
assert calculation_stage_directories(
    CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06)
) == ("stage_01", "stage_02")
assert calculation_stage_directories(
    CalculationSpec(Purpose.DOS, Theory.PBE)
) == ("stage_01", "stage_02", "stage_03")
assert calculation_stage_directories(
    CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE)
) == ("stage_01", "stage_02", "stage_03")
assert (
    calculation_result_stage_directory(CalculationSpec(Purpose.DOUBLE_RELAX, Theory.PBE))
    == "relax_02"
)
assert (
    calculation_result_stage_directory(CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06))
    == "stage_02"
)
assert calculation_result_stage_directory(CalculationSpec(Purpose.DOS, Theory.PBE)) == "stage_03"
assert (
    calculation_result_stage_directory(CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE))
    == "stage_03"
)
assert calculation_stage_directories(CalculationSpec(Purpose.RELAX, Theory.PBE)) == ()
assert legacy_workflow_from_spec(spec) == "relax_ions"
assert legacy_potcar_functional_from_spec(spec) == "PBE_64"
assert (
    legacy_potcar_functional_from_spec(CalculationSpec(Purpose.STATIC, Theory.HSE06))
    == "PBE_64"
)
assert (
    legacy_potcar_functional_from_spec(CalculationSpec(Purpose.RELAX, Theory.HSE06))
    == "PBE_64"
)
assert (
    legacy_potcar_functional_from_spec(CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06))
    == "PBE_64"
)

legacy_spec = calculation_spec_from_legacy("relax_ions", "PBE_64")
assert legacy_spec.purpose is Purpose.RELAX
assert legacy_spec.theory is Theory.PBE
assert legacy_spec.modifiers == frozenset({Modifier.IONS_ONLY})

flow_spec_spec = calculation_spec_from_flow_spec(
    {
        "calculation_spec": {
            "purpose": "relax",
            "theory": "pbe",
            "modifiers": ["ions_only"],
        },
        "workflow": "static",
        "potcar_functional": "PBE_64",
    }
)
assert flow_spec_spec == legacy_spec

assert CalculationSpec(Purpose.STATIC, Theory.PBE) in supported_combinations()
assert CalculationSpec(Purpose.STATIC, Theory.HSE06) in supported_combinations()
assert CalculationSpec(Purpose.RELAX, Theory.HSE06) in supported_combinations()
assert CalculationSpec(Purpose.RELAX_STATIC, Theory.PBE) in supported_combinations()
assert CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06) in supported_combinations()
assert calculation_display_name(CalculationSpec(Purpose.RELAX, Theory.PBE)) == "Geometry Optimisation"
assert (
    calculation_display_name(CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06))
    == "Geometry Optimisation + Static Energy"
)
assert (
    calculation_display_name(CalculationSpec(Purpose.DOUBLE_RELAX, Theory.PBE))
    == "Double Geometry Optimisation"
)
assert calculation_display_name(CalculationSpec(Purpose.DOS, Theory.PBE)) == "Density of States"
assert (
    calculation_display_name(CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE))
    == "Band Structure"
)
assert calculation_display_name(CalculationSpec(Purpose.STATIC, Theory.PBE)) == "Static Energy"
assert theory_display_name(Theory.HSE06) == "HSE06"

form_options = calculation_form_options()
desired_outputs = {option["value"]: option for option in form_options["desired_outputs"]}
assert [option["label"] for option in form_options["desired_outputs"]] == [
    "Relaxed Energy",
    "Relaxed structure",
    "Electronic density of states",
    "Electronic band structure",
    "Phonons",
    "Custom workflow",
]
# Energy only is hidden from the browser menu but keeps its single-point meaning.
assert "energy_only" not in desired_outputs
assert [
    (stage.stage_type.value, stage.theory.value)
    for stage in desired_output_workflow_spec("energy_only").stages
] == [("static", "pbe")]
assert [
    stage["stage_type"]
    for stage in desired_outputs["relaxed_energy"]["workflow_spec"]["stages"]
] == ["relax", "relax", "static"]
assert [
    stage["stage_type"]
    for stage in desired_outputs["relaxed_structure"]["workflow_spec"]["stages"]
] == ["relax", "relax"]
assert [
    stage["theory"]
    for stage in desired_outputs["relaxed_structure"]["workflow_spec"]["stages"]
] == ["pbe", "pbe"]
assert [
    (stage["stage_type"], stage["theory"])
    for stage in desired_outputs["electronic_dos"]["workflow_spec"]["stages"]
] == [
    ("relax", "pbe"),
    ("relax", "pbe"),
    ("static", "hse06"),
    ("dos", "hse06"),
]
assert [
    (stage["stage_type"], stage["theory"])
    for stage in desired_outputs["electronic_band_structure"]["workflow_spec"]["stages"]
] == [
    ("relax", "pbe"),
    ("relax", "pbe"),
    ("static", "hse06"),
    ("band_structure", "hse06"),
]
assert [
    (stage["stage_type"], stage["theory"])
    for stage in desired_outputs["phonons"]["workflow_spec"]["stages"]
] == [
    ("relax", "pbe"),
    ("relax", "pbe"),
    ("phonon_forces", "pbe"),
]
assert desired_outputs["custom"]["workflow_spec"] is None
assert (
    desired_output_workflow_spec("electronic_dos").to_dict()
    == desired_outputs["electronic_dos"]["workflow_spec"]
)
assert (
    desired_output_from_workflow_spec(desired_output_workflow_spec("energy_only"))
    == "energy_only"
)
assert (
    desired_output_from_workflow_spec(desired_output_workflow_spec("electronic_band_structure"))
    == "electronic_band_structure"
)
assert [option["label"] for option in form_options["purposes"]] == [
    "Geometry Optimisation",
    "Geometry Optimisation + Static Energy",
    "Double Geometry Optimisation",
    "Static Energy",
    "Density of States",
    "Band Structure",
]
assert [option["label"] for option in form_options["theories"]] == ["PBE", "HSE06"]
assert [option["label"] for option in form_options["modifiers"]] == [
    "Spin Polarised",
    "Spin-Orbit Coupling (SOC)",
    "DFT+U",
    "van der Waals correction",
]
modifier_options = {option["value"]: option for option in form_options["modifiers"]}
assert "gamma_only" not in modifier_options
for modifier in ("spin_polarized", "dft_u", "dispersion"):
    assert modifier_options[modifier]["enabled"] is True
assert modifier_options["spin_polarized"]["tooltip"] == ""
assert modifier_options["dft_u"]["tooltip"] == "DFT+U is applied only when explicitly selected."
assert "van der Waals correction" in modifier_options["dispersion"]["tooltip"]
assert modifier_options["soc"]["enabled"] is True
assert "vasp_ncl" in modifier_options["soc"]["tooltip"]
assert "default_dispersion_method" not in form_options
assert "dispersion_methods" not in form_options

spin_static_spec = CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED})
spin_relax_spec = CalculationSpec(Purpose.RELAX, Theory.PBE, {Modifier.SPIN_POLARIZED})
spin_relax_static_spec = CalculationSpec(
    Purpose.RELAX_STATIC,
    Theory.PBE,
    {Modifier.SPIN_POLARIZED},
)
spin_double_relax_spec = CalculationSpec(
    Purpose.DOUBLE_RELAX,
    Theory.PBE,
    {Modifier.SPIN_POLARIZED},
)
spin_dos_spec = CalculationSpec(Purpose.DOS, Theory.PBE, {Modifier.SPIN_POLARIZED})
spin_band_spec = CalculationSpec(
    Purpose.BAND_STRUCTURE,
    Theory.PBE,
    {Modifier.SPIN_POLARIZED},
)
gamma_band_spec = CalculationSpec(
    Purpose.BAND_STRUCTURE,
    Theory.PBE,
    {Modifier.GAMMA_ONLY},
)
soc_static_spec = CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SOC})
pbe_soc_relax_spec = CalculationSpec(Purpose.RELAX, Theory.PBE, {Modifier.SOC})
pbe_soc_relax_static_spec = CalculationSpec(Purpose.RELAX_STATIC, Theory.PBE, {Modifier.SOC})
pbe_soc_dos_spec = CalculationSpec(Purpose.DOS, Theory.PBE, {Modifier.SOC})
pbe_soc_band_spec = CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE, {Modifier.SOC})
dft_u_relax_spec = CalculationSpec(Purpose.RELAX, Theory.PBE, {Modifier.DFT_U})
gamma_static_spec = CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.GAMMA_ONLY})
hse_static_spec = CalculationSpec(Purpose.STATIC, Theory.HSE06)
hse_spin_static_spec = CalculationSpec(
    Purpose.STATIC,
    Theory.HSE06,
    {Modifier.SPIN_POLARIZED},
)
hse_gamma_static_spec = CalculationSpec(
    Purpose.STATIC,
    Theory.HSE06,
    {Modifier.GAMMA_ONLY},
)
hse_relax_spec = CalculationSpec(Purpose.RELAX, Theory.HSE06)
hse_relax_static_spec = CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06)
hse_spin_relax_spec = CalculationSpec(
    Purpose.RELAX,
    Theory.HSE06,
    {Modifier.SPIN_POLARIZED},
)
hse_gamma_relax_spec = CalculationSpec(
    Purpose.RELAX,
    Theory.HSE06,
    {Modifier.GAMMA_ONLY},
)
hse_spin_relax_static_spec = CalculationSpec(
    Purpose.RELAX_STATIC,
    Theory.HSE06,
    {Modifier.SPIN_POLARIZED},
)
hse_gamma_relax_static_spec = CalculationSpec(
    Purpose.RELAX_STATIC,
    Theory.HSE06,
    {Modifier.GAMMA_ONLY},
)
hse_double_relax_spec = CalculationSpec(Purpose.DOUBLE_RELAX, Theory.HSE06)
hse_dos_spec = CalculationSpec(Purpose.DOS, Theory.HSE06)
hse_band_spec = CalculationSpec(Purpose.BAND_STRUCTURE, Theory.HSE06)
hse_dft_u_static_spec = CalculationSpec(Purpose.STATIC, Theory.HSE06, {Modifier.DFT_U})
hse_dft_u_relax_spec = CalculationSpec(Purpose.RELAX, Theory.HSE06, {Modifier.DFT_U})
hse_dft_u_relax_static_spec = CalculationSpec(
    Purpose.RELAX_STATIC,
    Theory.HSE06,
    {Modifier.DFT_U},
)
hse_soc_static_spec = CalculationSpec(Purpose.STATIC, Theory.HSE06, {Modifier.SOC})
hse_soc_relax_spec = CalculationSpec(Purpose.RELAX, Theory.HSE06, {Modifier.SOC})
hse_soc_relax_static_spec = CalculationSpec(
    Purpose.RELAX_STATIC,
    Theory.HSE06,
    {Modifier.SOC},
)
assert validate_calculation_spec(spin_static_spec) == spin_static_spec
assert validate_calculation_spec(spin_relax_spec) == spin_relax_spec
assert validate_calculation_spec(spin_relax_static_spec) == spin_relax_static_spec
assert validate_calculation_spec(spin_double_relax_spec) == spin_double_relax_spec
assert validate_calculation_spec(spin_dos_spec) == spin_dos_spec
assert validate_calculation_spec(spin_band_spec) == spin_band_spec
assert validate_calculation_spec(soc_static_spec) == soc_static_spec
assert validate_calculation_spec(dft_u_relax_spec) == dft_u_relax_spec
assert validate_calculation_spec(gamma_static_spec) == gamma_static_spec
assert validate_calculation_spec(hse_static_spec) == hse_static_spec
assert validate_calculation_spec(hse_relax_spec) == hse_relax_spec
assert validate_calculation_spec(hse_relax_static_spec) == hse_relax_static_spec
assert validate_calculation_spec(hse_spin_static_spec) == hse_spin_static_spec
assert validate_calculation_spec(hse_spin_relax_spec) == hse_spin_relax_spec
assert validate_calculation_spec(hse_spin_relax_static_spec) == hse_spin_relax_static_spec
assert validate_calculation_spec(hse_gamma_static_spec) == hse_gamma_static_spec
assert validate_calculation_spec(hse_gamma_relax_spec) == hse_gamma_relax_spec
assert validate_calculation_spec(hse_gamma_relax_static_spec) == hse_gamma_relax_static_spec
assert validate_calculation_spec(hse_soc_static_spec) == hse_soc_static_spec
assert legacy_workflow_from_spec(spin_static_spec) == "static"
assert legacy_workflow_from_spec(spin_relax_spec) == "relax"
assert legacy_workflow_from_spec(spin_relax_static_spec) == "relax_static"
assert legacy_workflow_from_spec(spin_double_relax_spec) == "double_relax"
assert legacy_workflow_from_spec(spin_dos_spec) == "dos"
assert legacy_workflow_from_spec(spin_band_spec) == "band_structure"
assert legacy_workflow_from_spec(soc_static_spec) == "static"
assert legacy_workflow_from_spec(dft_u_relax_spec) == "relax"
assert legacy_workflow_from_spec(gamma_static_spec) == "static"
assert legacy_workflow_from_spec(hse_soc_static_spec) == "static"
try:
    validate_calculation_spec(gamma_band_spec)
except CalculationValidationError as exc:
    assert "band_structure" in exc.message
else:
    raise AssertionError("Gamma-only should not be supported for line-mode bands.")

for unsupported_hse_spec in (hse_double_relax_spec, hse_dos_spec, hse_band_spec):
    try:
        validate_calculation_spec(unsupported_hse_spec)
    except CalculationValidationError as exc:
        assert "HSE06 is currently supported" in exc.message
        assert "Geometry Optimisation" in exc.message
        assert "Static Energy" in exc.message
    else:
        raise AssertionError(f"{unsupported_hse_spec} should fail clearly.")

for unsupported_modifier_spec in (
    pbe_soc_relax_spec,
    pbe_soc_relax_static_spec,
    pbe_soc_dos_spec,
    pbe_soc_band_spec,
    hse_dft_u_static_spec,
    hse_dft_u_relax_spec,
    hse_dft_u_relax_static_spec,
    hse_soc_relax_spec,
    hse_soc_relax_static_spec,
):
    try:
        validate_calculation_spec(unsupported_modifier_spec)
    except CalculationValidationError as exc:
        assert (
            "DFT+U" in exc.message
            or "Spin-Orbit Coupling (SOC)" in exc.message
        )
        if unsupported_modifier_spec.theory is Theory.HSE06:
            assert "hse06" in exc.message
    else:
        raise AssertionError(f"{unsupported_modifier_spec} should fail clearly.")

for unsupported_theory in (Theory.R2SCAN,):
    try:
        build_calculation_flow(
            structure="structure",
            spec=CalculationSpec(Purpose.STATIC, unsupported_theory),
        )
    except CalculationValidationError as exc:
        assert unsupported_theory.value in str(exc)
    else:
        raise AssertionError(
            f"{unsupported_theory.value} should not be implemented yet."
        )

import backend.workflows as workflows

calls = []
original_build_atomate2_flow_for_spec = workflows.build_atomate2_flow_for_spec


def fake_build_atomate2_flow_for_spec(**kwargs):
    calls.append(kwargs)
    return {"flow": kwargs}


workflows.build_atomate2_flow_for_spec = fake_build_atomate2_flow_for_spec
try:
    flow = build_calculation_flow(
        structure="structure",
        spec=spec,
        label="demo",
        incar={"ENCUT": 520},
        kpoints={"grid_density": 1000},
        potcar_functional="PBE_64",
    )
    hse_flow = build_calculation_flow(
        structure="structure",
        spec=hse_static_spec,
        label="hse-demo",
    )
    hse_relax_flow = build_calculation_flow(
        structure="structure",
        spec=hse_relax_spec,
        label="hse-relax-demo",
    )
    hse_relax_static_flow = build_calculation_flow(
        structure="structure",
        spec=hse_relax_static_spec,
        label="hse-relax-static-demo",
    )
finally:
    workflows.build_atomate2_flow_for_spec = original_build_atomate2_flow_for_spec

assert flow == {"flow": calls[0]}
assert hse_flow == {"flow": calls[1]}
assert hse_relax_flow == {"flow": calls[2]}
assert hse_relax_static_flow == {"flow": calls[3]}
assert calls == [
    {
        "structure": "structure",
        "spec": spec,
        "label": "demo",
        "incar": {"ENCUT": 520},
        "kpoints": {"grid_density": 1000},
        "resources": None,
        "potcar_functional": "PBE_64",
    },
    {
        "structure": "structure",
        "spec": hse_static_spec,
        "label": "hse-demo",
        "incar": None,
        "kpoints": None,
        "resources": None,
        "potcar_functional": "PBE_64",
    },
    {
        "structure": "structure",
        "spec": hse_relax_spec,
        "label": "hse-relax-demo",
        "incar": None,
        "kpoints": None,
        "resources": None,
        "potcar_functional": "PBE_64",
    },
    {
        "structure": "structure",
        "spec": hse_relax_static_spec,
        "label": "hse-relax-static-demo",
        "incar": None,
        "kpoints": None,
        "resources": None,
        "potcar_functional": "PBE_64",
    },
]

print("calculation architecture smoke test passed")
