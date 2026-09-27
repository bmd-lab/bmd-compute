from __future__ import annotations

from backend.calculations.models import (
    CalculationSpec,
    Modifier,
    Purpose,
    StageSpec,
    StageType,
    Theory,
    WorkflowSpec,
)
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure


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


def _preview_incar(structure_text: str, spec: CalculationSpec | WorkflowSpec) -> str:
    return preview_generated_inputs(
        parse_structure(structure_text),
        spec,
        resources={"ntasks": 24},
        potcar_functional="PBE_64",
    )["incar"]


def _sections(incar_text: str) -> list[str]:
    return incar_text.split("\n\n")


def _assert_spin_off(incar_text: str) -> None:
    assert "ISPIN = 1" in incar_text
    assert "MAGMOM =" not in incar_text


def _assert_spin_on(incar_text: str, expected_magmom: str) -> None:
    assert "ISPIN = 2" in incar_text
    assert expected_magmom in incar_text


def test_pbe_collinear_spin_checkbox_controls_relax_static_dos_and_band_previews():
    static_off = _preview_incar(SI_POSCAR, CalculationSpec(Purpose.STATIC, Theory.PBE))
    static_on = _preview_incar(
        SI_POSCAR,
        CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED}),
    )
    relax_off = _preview_incar(SI_POSCAR, CalculationSpec(Purpose.RELAX, Theory.PBE))
    relax_on = _preview_incar(
        SI_POSCAR,
        CalculationSpec(Purpose.RELAX, Theory.PBE, {Modifier.SPIN_POLARIZED}),
    )

    _assert_spin_off(static_off)
    _assert_spin_on(static_on, "MAGMOM = 2*0.6")
    _assert_spin_off(relax_off)
    _assert_spin_on(relax_on, "MAGMOM = 2*0.6")

    dos_off = WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.PBE),
            StageSpec(StageType.DOS, Theory.PBE),
        ],
        recipe="custom",
    )
    dos_on = WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED}),
            StageSpec(StageType.DOS, Theory.PBE, {Modifier.SPIN_POLARIZED}),
        ],
        recipe="custom",
    )
    for section in _sections(_preview_incar(SI_POSCAR, dos_off)):
        _assert_spin_off(section)
    for section in _sections(_preview_incar(SI_POSCAR, dos_on)):
        _assert_spin_on(section, "MAGMOM = 2*0.6")

    band_off = WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.PBE),
            StageSpec(StageType.BAND_STRUCTURE, Theory.PBE),
        ],
        recipe="custom",
    )
    band_on = WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED}),
            StageSpec(
                StageType.BAND_STRUCTURE,
                Theory.PBE,
                {Modifier.SPIN_POLARIZED},
            ),
        ],
        recipe="custom",
    )
    for section in _sections(_preview_incar(SI_POSCAR, band_off)):
        _assert_spin_off(section)
    for section in _sections(_preview_incar(SI_POSCAR, band_on)):
        _assert_spin_on(section, "MAGMOM = 2*0.6")


def test_sns2_static_preview_matches_deployed_spin_checkbox_acceptance_case():
    static_off = _preview_incar(SNS2_POSCAR, CalculationSpec(Purpose.STATIC, Theory.PBE))
    static_on = _preview_incar(
        SNS2_POSCAR,
        CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED}),
    )

    _assert_spin_off(static_off)
    _assert_spin_on(static_on, "MAGMOM = 3*0.6")


def test_hse06_collinear_spin_semantics_apply_through_supported_stage_paths():
    static_off = _preview_incar(SI_POSCAR, CalculationSpec(Purpose.STATIC, Theory.HSE06))
    static_on = _preview_incar(
        SI_POSCAR,
        CalculationSpec(Purpose.STATIC, Theory.HSE06, {Modifier.SPIN_POLARIZED}),
    )
    relax_off = _preview_incar(SI_POSCAR, CalculationSpec(Purpose.RELAX, Theory.HSE06))
    relax_on = _preview_incar(
        SI_POSCAR,
        CalculationSpec(Purpose.RELAX, Theory.HSE06, {Modifier.SPIN_POLARIZED}),
    )

    _assert_spin_off(static_off)
    _assert_spin_on(static_on, "MAGMOM = 2*0.6")
    _assert_spin_off(relax_off)
    _assert_spin_on(relax_on, "MAGMOM = 2*0.6")

    hse_band = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.HSE06),
            StageSpec(
                StageType.BAND_STRUCTURE,
                Theory.HSE06,
                {Modifier.SPIN_POLARIZED},
            ),
        ],
        recipe="custom",
    )
    relax_section, static_section, band_section = _sections(_preview_incar(SI_POSCAR, hse_band))

    _assert_spin_off(relax_section)
    _assert_spin_off(static_section)
    _assert_spin_on(band_section, "MAGMOM = 2*0.6")


def test_dft_u_does_not_implicitly_enable_collinear_spin():
    dft_u_off = _preview_incar(
        FE2O3_POSCAR,
        CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U}),
    )
    dft_u_on = _preview_incar(
        FE2O3_POSCAR,
        CalculationSpec(
            Purpose.STATIC,
            Theory.PBE,
            {Modifier.DFT_U, Modifier.SPIN_POLARIZED},
        ),
    )

    _assert_spin_off(dft_u_off)
    assert "LDAU = True" in dft_u_off
    _assert_spin_on(dft_u_on, "MAGMOM = 2*5.0 3*0.6")
    assert "LDAU = True" in dft_u_on


def _magmom_components(incar_text: str) -> list[float]:
    for line in incar_text.splitlines():
        if line.startswith("MAGMOM = "):
            components = []
            for token in line.removeprefix("MAGMOM = ").split():
                if "*" in token:
                    count, value = token.split("*", 1)
                    components.extend([float(value)] * int(count))
                else:
                    components.append(float(token))
            return components
    raise AssertionError("MAGMOM is missing")


def _without_magmom(incar_text: str) -> list[str]:
    return [line for line in incar_text.splitlines() if not line.startswith("MAGMOM = ")]


def test_soc_noncollinear_policy_is_not_changed_by_collinear_spin_semantics():
    soc = _preview_incar(SI_POSCAR, CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SOC}))
    soc_with_spin_modifier = _preview_incar(
        SI_POSCAR,
        CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SOC, Modifier.SPIN_POLARIZED}),
    )

    assert "ISPIN =" not in soc
    assert "LSORBIT = True" in soc
    assert "LNONCOLLINEAR = True" in soc
    # Si is outside the spin screen: SOC starts from zero vector moments, and
    # an explicit Spin Polarised choice restores magnetic starting moments
    # without changing any other non-collinear setting.
    assert _magmom_components(soc) == [0.0] * 6
    assert _magmom_components(soc_with_spin_modifier) == [0.0, 0.0, 0.6] * 2
    assert "ISPIN =" not in soc_with_spin_modifier
    assert _without_magmom(soc_with_spin_modifier) == _without_magmom(soc)

    hse_soc = _preview_incar(
        SI_POSCAR,
        CalculationSpec(Purpose.STATIC, Theory.HSE06, {Modifier.SOC}),
    )
    hse_soc_with_spin_modifier = _preview_incar(
        SI_POSCAR,
        CalculationSpec(Purpose.STATIC, Theory.HSE06, {Modifier.SOC, Modifier.SPIN_POLARIZED}),
    )

    assert "LHFCALC = True" in hse_soc
    assert "PRECFOCK = Accurate" in hse_soc
    assert "ISPIN =" not in hse_soc
    assert "LSORBIT = True" in hse_soc
    assert "LNONCOLLINEAR = True" in hse_soc
    assert "LELF =" not in hse_soc
    assert _magmom_components(hse_soc) == [0.0] * 6
    assert _magmom_components(hse_soc_with_spin_modifier) == [0.0, 0.0, 0.6] * 2
    assert _without_magmom(hse_soc_with_spin_modifier) == _without_magmom(hse_soc)
