"""The methodology declared in docs/methodology.md, checked against generated inputs.

These tests pin the scientifically meaningful v1 values, including those the
pinned atomate2/pymatgen stack supplies, so that a dependency change cannot
alter them silently. They test values, not prose.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath

from backend.calculations.default_treatments import (
    DFT_U_STAGE_TYPES,
    SOC_EXCLUDED_STAGE_TYPES,
    resolve_default_treatments,
)
from backend.calculations.method_considerations import SOC_TRIGGER_CLASSES, SPIN_TRIGGER_CLASSES
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import desired_output_workflow_spec
from backend.generated_inputs import generated_input_stage_previews
from backend.workflows import (
    build_band_structure_input_set_generator,
    build_dos_input_set_generator,
)
from backend.parser import parse_structure
from test_automatic_dft_u import NIO, build_route, job_maker, runtime_jobs
from test_automatic_soc import BI_POSCAR
from test_structure_dimensionality import sns2_structure


SI = Structure(
    Lattice([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]),
    ["Si", "Si"],
    [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
)
TIO2 = Structure.from_spacegroup(
    "P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]
)

COMMON = {"EDIFF": 1e-6, "PREC": "Accurate", "LREAL": False, "LASPH": True, "ADDGRID": True, "NELM": 200}
PBE_RELAX = {
    "ENCUT": 580,
    "IBRION": 2,
    "ISIF": 3,
    "EDIFFG": -0.01,
    "NSW": 99,
    "ALGO": "Fast",
    "ISMEAR": 0,
    "SIGMA": 0.2,
}
PBE_STATIC = {"ENCUT": 620, "ALGO": "Normal", "ISMEAR": -5, "SIGMA": 0.05, "NEDOS": 4001, "LORBIT": 11}
HSE06 = {"LHFCALC": True, "AEXX": 0.25, "HFSCREEN": 0.2}
HSE06_STATIC = {**HSE06, "ENCUT": 620, "ALGO": "Damped", "TIME": 0.4, "PRECFOCK": "Accurate", "ISMEAR": 0, "SIGMA": 0.05}
HSE06_DOS = {**HSE06, "ENCUT": 620, "ALGO": "Normal", "PRECFOCK": "Fast", "ISMEAR": -5, "NEDOS": 4001, "NELMIN": 5}
HSE06_BAND = {**HSE06, "ENCUT": 620, "ALGO": "Normal", "PRECFOCK": "Fast", "ISMEAR": 0, "SIGMA": 0.01, "NELMIN": 5}

DECLARED_SPIN_SCREEN = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Mo", "Tc", "Ru", "Rh", "Re", "Os", "Ir",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
}
DECLARED_SOC_SCREEN = (
    {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}
    | {"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
    | {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
    | {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
    | {"Tl", "Pb", "Bi", "Po"}
)


def stages(structure, workflow):
    return generated_input_stage_previews(structure, workflow, resources={"ntasks": 24})


def assert_incar(incar, expected):
    for key, value in expected.items():
        assert incar.get(key) == value, (key, incar.get(key), value)


def custom(*stage_specs):
    return WorkflowSpec(list(stage_specs), recipe="custom")


# --- 1. Desired Outputs ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("desired_output", "sequence"),
    [
        ("energy_only", [("static", "pbe")]),
        ("relaxed_structure", [("relax", "pbe"), ("relax", "pbe")]),
        ("electronic_dos", [("relax", "pbe"), ("static", "hse06"), ("dos", "hse06")]),
        ("electronic_band_structure", [("relax", "pbe"), ("static", "hse06"), ("band_structure", "hse06")]),
    ],
)
def test_desired_output_stage_sequences(desired_output, sequence):
    workflow = desired_output_workflow_spec(desired_output)
    assert [(stage.stage_type.value, stage.theory.value) for stage in workflow.stages] == sequence


# --- 2. Automatic treatments ----------------------------------------------------------------


def test_automatic_treatment_screens_and_placement():
    assert {element for elements in SPIN_TRIGGER_CLASSES.values() for element in elements} == DECLARED_SPIN_SCREEN
    assert {element for elements in SOC_TRIGGER_CLASSES.values() for element in elements} == DECLARED_SOC_SCREEN
    assert SOC_EXCLUDED_STAGE_TYPES == {StageType.RELAX}
    assert DFT_U_STAGE_TYPES == {StageType.RELAX, StageType.STATIC}

    resolved = resolve_default_treatments(
        NIO, desired_output_workflow_spec("electronic_dos"), desired_output="electronic_dos"
    ).resolved_workflow
    assert [sorted(m.value for m in stage.modifiers) for stage in resolved.stages] == [
        ["dft_u", "spin_polarized"],
        ["spin_polarized"],
        ["spin_polarized"],
    ]


def _resolved_previews(structure, desired_output):
    workflow = resolve_default_treatments(
        structure, desired_output_workflow_spec(desired_output), desired_output=desired_output
    ).resolved_workflow
    return stages(structure, workflow)


def test_two_dimensional_structure_gets_automatic_d3_bj_on_pbe_relax_and_static_only():
    structure = sns2_structure()  # layered SnS2: two-dimensional bonded connectivity

    (static,) = _resolved_previews(structure, "energy_only")
    assert static["input_set"].incar["IVDW"] == 12

    relax, hse_static, dos = _resolved_previews(structure, "electronic_dos")
    assert relax["input_set"].incar["IVDW"] == 12
    assert "IVDW" not in hse_static["input_set"].incar
    assert "IVDW" not in dos["input_set"].incar


def test_heavy_element_gets_automatic_soc_on_pbe_non_relaxation_stages_only():
    structure = parse_structure(BI_POSCAR)

    # HSE06 + SOC is closed to new calculations: the HSE06 stages keep HSE06
    # and omit SOC (recorded and shown as a red warning).
    relax, hse_static, band = _resolved_previews(structure, "electronic_band_structure")
    assert "LSORBIT" not in relax["input_set"].incar
    for preview in (hse_static, band):
        assert "LSORBIT" not in preview["input_set"].incar
        assert preview["input_set"].incar["LHFCALC"] is True
        assert preview["vasp_executable"] == "vasp_std"

    (static,) = _resolved_previews(structure, "energy_only")
    assert static["input_set"].incar["LSORBIT"] is True
    assert static["vasp_executable"] == "vasp_ncl"


def test_d3_bj_is_ivdw_12_and_d3_zero_damping_is_ivdw_11():
    from backend.calculations.dispersion import dispersion_option_payload

    for method, ivdw in (("dftd3-bj", 12), ("dftd3", 11)):
        (preview,) = stages(
            SI,
            custom(StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DISPERSION}, options=dispersion_option_payload(method))),
        )
        assert preview["input_set"].incar["IVDW"] == ivdw


# --- 3. Stage methodology ---------------------------------------------------------------------


def test_pbe_relaxation_and_static():
    for preview in stages(SI, desired_output_workflow_spec("relaxed_structure")):
        assert_incar(preview["input_set"].incar, {**COMMON, **PBE_RELAX, "ISPIN": 1})
    (static,) = stages(SI, desired_output_workflow_spec("energy_only"))
    assert_incar(static["input_set"].incar, {**COMMON, **PBE_STATIC, "NSW": 0, "ISPIN": 1})


def test_ions_only_relaxation_uses_isif_2():
    (preview,) = stages(SI, custom(StageSpec(StageType.RELAX, Theory.PBE, {Modifier.IONS_ONLY})))
    assert preview["input_set"].incar["ISIF"] == 2


def test_pbe_dos_and_band_structure_are_non_self_consistent():
    _, dos = stages(SI, custom(StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.DOS, Theory.PBE)))
    assert_incar(dos["input_set"].incar, {**COMMON, "ENCUT": 620, "ICHARG": 11, "ISMEAR": -5})
    _, band = stages(
        SI, custom(StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.BAND_STRUCTURE, Theory.PBE))
    )
    assert_incar(band["input_set"].incar, {**COMMON, "ENCUT": 620, "ICHARG": 11, "ISMEAR": 0, "SIGMA": 0.05, "ISYM": 0})


def test_hse06_stages():
    relax, static, dos = stages(SI, desired_output_workflow_spec("electronic_dos"))
    assert "LHFCALC" not in relax["input_set"].incar
    assert_incar(static["input_set"].incar, {**COMMON, **HSE06_STATIC})
    assert_incar(dos["input_set"].incar, {**COMMON, **HSE06_DOS})
    assert "ICHARG" not in dos["input_set"].incar
    _, _, band = stages(SI, desired_output_workflow_spec("electronic_band_structure"))
    assert_incar(band["input_set"].incar, {**COMMON, **HSE06_BAND})
    assert "ICHARG" not in band["input_set"].incar

    (hse_relax,) = stages(SI, custom(StageSpec(StageType.RELAX, Theory.HSE06)))
    assert_incar(
        hse_relax["input_set"].incar,
        {**COMMON, **PBE_RELAX, **HSE06, "ALGO": "Damped", "TIME": 0.4, "PRECFOCK": "Fast"},
    )


# --- 4. K-points ------------------------------------------------------------------------------


def _gamma_mesh(structure, density):
    return Kpoints.automatic_density_by_vol(structure, density, force_gamma=True)


@pytest.mark.parametrize("structure", [SI, NIO], ids=["Si", "NiO"])
def test_uniform_kpoint_densities(structure):
    relax, static, dos = stages(structure, desired_output_workflow_spec("electronic_dos"))
    for preview in (relax, static):
        kpoints = preview["input_set"].kpoints
        assert kpoints.style == Kpoints.supported_modes.Gamma
        assert kpoints.kpts == _gamma_mesh(structure, 64).kpts
    assert dos["input_set"].kpoints.kpts == _gamma_mesh(structure, 100).kpts

    _, pbe_dos = stages(structure, custom(StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.DOS, Theory.PBE)))
    assert pbe_dos["input_set"].kpoints.kpts == _gamma_mesh(structure, 100).kpts


def test_band_structure_paths():
    path_40 = HighSymmKpath(SI).get_kpoints(line_density=40, coords_are_cartesian=False)[0]

    _, pbe_band = stages(
        SI, custom(StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.BAND_STRUCTURE, Theory.PBE))
    )
    assert pbe_band["input_set"].kpoints.num_kpts == len(path_40)

    _, _, hse_band = stages(SI, desired_output_workflow_spec("electronic_band_structure"))
    kpoints = hse_band["input_set"].kpoints
    zero_weight = [w for w in kpoints.kpts_weights if w == 0]
    assert len(zero_weight) == len(path_40)
    generator = build_band_structure_input_set_generator(SI, theory=Theory.HSE06)
    assert generator.reciprocal_density == 64
    assert generator.line_density == 40


# --- 5. POTCARs -----------------------------------------------------------------------------------


def test_potcars_follow_the_pinned_atomate2_mapping_with_pbe_64():
    from atomate2.vasp.sets.base import _BASE_VASP_SET

    for structure, expected in ((NIO, {"Ni": "Ni", "O": "O"}), (TIO2, {"Ti": "Ti_sv", "O": "O"})):
        (preview,) = stages(structure, desired_output_workflow_spec("energy_only"))
        input_set = preview["input_set"]
        mapping = dict(zip(input_set.poscar.site_symbols, str(input_set.potcar).split()))
        assert mapping == expected
        assert mapping == {element: _BASE_VASP_SET["POTCAR"][element] for element in mapping}
    potcar = build_route(NIO, workflow="energy_only").context["submission_spec"]["potcar"]
    assert potcar["functional"] == "PBE_64"
    assert potcar["symbols"] == ["Ni", "O"]


# --- 6. Starting magnetic moments -----------------------------------------------------------------


def test_spin_polarised_stages_start_from_the_pinned_default_moments():
    (static,) = stages(
        NIO, resolve_default_treatments(NIO, desired_output_workflow_spec("energy_only"), desired_output="energy_only").resolved_workflow
    )
    incar = static["input_set"].incar
    assert incar["ISPIN"] == 2
    assert dict(zip(static["input_set"].poscar.site_symbols, incar["MAGMOM"])) == {"Ni": 5.0, "O": 0.6}


# --- 7-8. Chaining and unsuccessful stages ---------------------------------------------------------


def test_supported_analysis_stage_precursors():
    from backend.calculations.registry import CalculationValidationError, validate_workflow_spec

    supported = {
        (StageType.DOS, Theory.PBE): {Theory.PBE},
        (StageType.BAND_STRUCTURE, Theory.PBE): {Theory.PBE},
        (StageType.DOS, Theory.HSE06): {Theory.PBE, Theory.HSE06},
        (StageType.BAND_STRUCTURE, Theory.HSE06): {Theory.HSE06},
    }
    for (stage_type, theory), precursors in supported.items():
        for precursor in (Theory.PBE, Theory.HSE06):
            workflow = custom(StageSpec(StageType.STATIC, precursor), StageSpec(stage_type, theory))
            if precursor in precursors:
                validate_workflow_spec(workflow)
            else:
                with pytest.raises(CalculationValidationError):
                    validate_workflow_spec(workflow)


def test_analysis_stages_size_nbands_from_the_previous_stage():
    for theory in (Theory.PBE, Theory.HSE06):
        assert build_dos_input_set_generator(SI, theory=theory).nbands_factor == 1.2
        assert build_band_structure_input_set_generator(SI, theory=theory).nbands_factor == 1.2


@pytest.mark.parametrize(
    "desired_output", ["energy_only", "relaxed_structure", "electronic_dos", "electronic_band_structure"]
)
def test_every_stage_fails_the_workflow_when_unsuccessful(desired_output):
    _, jobs = runtime_jobs(build_route(NIO, workflow=desired_output).context["submission_spec"])
    assert all(job_maker(job).stop_children_kwargs == {"handle_unsuccessful": "error"} for job in jobs)
