"""Executable phonon workflows (Milestone A).

Construction, admission and planning preview of the ``phonons`` Desired Output
and of Custom workflows with a ``PHONON_FORCES`` stage. Execution with fake
VASP is covered in ``test_phonon_end_to_end_fake.py`` and the Jobflow
replacement-directory handling in ``test_execution_nested_directories.py``.
Nothing here contacts POWER.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from starlette.requests import Request

pytest.importorskip("phonopy")
pytest.importorskip("seekpath")

import main
import test_automatic_soc as soc_cases
from backend.calculations.admission import (
    PHONONS_UNSUPPORTED_TREATMENT_CODE,
    require_admissible_new_calculation,
)
from backend.calculations.default_treatments import (
    PHONONS_EXCLUDED_TREATMENT_CODE,
    _apply_dispersion,
)
from backend.calculations.dispersion import dispersion_option_payload
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    PHONON_UNSUPPORTED_MODIFIER_CODE,
    CalculationValidationError,
    calculation_form_options,
    desired_output_from_workflow_spec,
    desired_output_workflow_spec,
    supported_stage_modifier_combinations,
    validate_workflow_spec,
    workflow_stage_directories,
)
from backend.calculations.stage_materialization import (
    NON_EXECUTABLE_STAGE_TYPES,
    StageMaterialization,
    stage_materialization,
    stage_type_is_executable,
)
from backend.parser import parse_structure
from backend.phonons import build_displacement_plan
from backend.phonons.workflow import (
    PHONON_DISPLACEMENT_ANGSTROM,
    PHONON_JOB_NAMES,
    PHONON_SYMPREC,
    phonon_planning_preview,
)
from backend.workflows import build_atomate2_flow_for_workflow_spec


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
2.86
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Fe
2
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
NIO_POSCAR = """NiO
4.17
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Ni O
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
PBE_PHONONS = WorkflowSpec(
    [StageSpec(StageType.RELAX, Theory.PBE), StageSpec(StageType.PHONON_FORCES, Theory.PBE)]
)


def si():
    return parse_structure(SI_POSCAR, "poscar")


def quiet(callable_, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_(*args, **kwargs)


def browser_build(structure, *, workflow="phonons", workflow_spec=None):
    request = Request({"type": "http", "method": "POST", "path": "/build-workflow", "headers": []})
    return quiet(
        main.build_workflow,
        request,
        structure=structure,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow_spec.to_dict()) if workflow_spec is not None else None,
        workflow=workflow,
        method=None,
    )


def plan(structure=SI_POSCAR, **kwargs):
    return quiet(main.plan_calculation_request, structure_text=structure, fmt="poscar", **kwargs)


def with_dispersion(stage: StageSpec) -> StageSpec:
    return StageSpec(
        stage.stage_type,
        stage.theory,
        {*stage.modifiers, Modifier.DISPERSION},
        options=dispersion_option_payload("dftd3"),
    )


# --- 1. construction ------------------------------------------------------------------


def test_phonons_desired_output_is_a_pbe_relax_followed_by_phonon_forces():
    workflow = desired_output_workflow_spec("phonons")
    assert [(stage.stage_type, stage.theory, stage.modifiers) for stage in workflow.stages] == [
        (StageType.RELAX, Theory.PBE, frozenset()),
        (StageType.PHONON_FORCES, Theory.PBE, frozenset()),
    ]
    assert desired_output_from_workflow_spec(workflow) == "phonons"
    assert workflow_stage_directories(workflow) == ("stage_01", "stage_02")
    options = calculation_form_options()
    assert "phonons" in [item["value"] for item in options["desired_outputs"]]
    assert "phonon_forces" in [item["value"] for item in options["stage_types"]]


def test_phonon_force_stage_is_executable_and_still_a_derived_task_set():
    assert NON_EXECUTABLE_STAGE_TYPES == frozenset()
    assert stage_type_is_executable(StageType.PHONON_FORCES)
    assert stage_materialization(StageType.PHONON_FORCES) is StageMaterialization.DERIVED_TASK_SET


def test_si_phonon_workflow_uses_atomate2_phonon_maker_with_bmd_settings():
    from backend.config import DEFAULT_RESOURCES

    resources = DEFAULT_RESOURCES
    flow = quiet(build_atomate2_flow_for_workflow_spec, si(), PBE_PHONONS, label="si", resources=resources)

    relax, phonon = flow.jobs
    assert flow.name == "si_phonons"
    assert flow.metadata["bmd_stage_directories"] == ("stage_01", "stage_02")
    assert phonon.name == "stage_02"
    assert [job.name for job in phonon.jobs] == list(PHONON_JOB_NAMES)

    jobs = {job.name: job for job in phonon.jobs}
    displacements = jobs["generate_phonon_displacements"].function_kwargs
    assert displacements["displacement"] == PHONON_DISPLACEMENT_ANGSTROM == 0.01
    assert displacements["symprec"] == PHONON_SYMPREC == 1e-5
    assert displacements["kpath_scheme"] == "seekpath"
    assert displacements["use_symmetrized_structure"] is None
    supercell = jobs["get_supercell_size"].function_kwargs
    assert supercell["min_length"] == 10.0
    assert supercell["prefer_90_degrees"] is True
    analysis = jobs["generate_frequencies_eigenvectors"].function_kwargs
    assert analysis["born"] is None and analysis["epsilon_static"] is None
    assert analysis["total_dft_energy"] is None
    assert analysis["store_force_constants"] is True
    assert analysis["create_force_constants_file"] is True

    force_maker = jobs["run_phonon_displacements"].function_kwargs["phonon_maker"]
    assert type(force_maker).__name__ == "PhononDisplacementMaker"
    force_incar = force_maker.input_set_generator.user_incar_settings
    assert force_incar["ENCUT"] == 700
    assert force_incar["EDIFF"] == 1e-7
    assert force_incar["ISPIN"] == 1
    assert force_incar["GGA"] is None  # PBE, not atomate2's PBEsol base set
    assert "NCORE" not in force_incar
    assert force_maker.input_set_generator.auto_ispin is False
    assert force_maker.input_set_generator.vdw is None
    assert force_maker.stop_children_kwargs == relax.maker.stop_children_kwargs

    relax_incar = relax.maker.input_set_generator.user_incar_settings
    assert relax_incar["ENCUT"] == 700  # same cutoff as the force calculations
    assert relax_incar["EDIFF"] == 1e-7
    assert relax_incar["EDIFFG"] == -1e-3
    assert "NCORE" in relax_incar  # relax keeps the existing BMD resource policy


# --- 2. relax -> phonons ---------------------------------------------------------------


def test_relaxed_structure_feeds_every_phonon_job():
    flow = quiet(build_atomate2_flow_for_workflow_spec, si(), PBE_PHONONS, label="si")
    relax, phonon = flow.jobs
    jobs = {job.name: job for job in phonon.jobs}
    for name in ("get_supercell_size", "generate_phonon_displacements", "generate_frequencies_eigenvectors"):
        structure = jobs[name].function_kwargs["structure"]
        assert structure.uuid == relax.uuid, name
        assert structure.attributes == (("a", "structure"),), name
    run = jobs["run_phonon_displacements"].function_kwargs
    assert run["structure"].uuid == relax.uuid
    assert run["prev_dir"] is None  # no WAVECAR/CHGCAR from the relaxed cell
    assert run["displacements"].uuid == jobs["generate_phonon_displacements"].uuid
    assert run["supercell_matrix"].uuid == jobs["get_supercell_size"].uuid


def test_precursor_relax_settings_apply_only_to_the_relax_before_phonons():
    relaxed = desired_output_workflow_spec("relaxed_structure")
    flow = quiet(build_atomate2_flow_for_workflow_spec, si(), relaxed, label="si")
    for job in flow.jobs:
        settings = job.maker.input_set_generator.user_incar_settings
        assert settings["ENCUT"] == 580
        assert settings["EDIFF"] == 1e-6
        assert settings["EDIFFG"] == -0.01


# --- 3. deterministic displacements ------------------------------------------------------


def test_si_displacements_are_deterministic_and_match_the_bmd_plan():
    first = quiet(phonon_planning_preview, si())
    second = quiet(phonon_planning_preview, si())
    assert first["supercell_matrix"] == second["supercell_matrix"]
    assert first["displacement_count"] == second["displacement_count"] == 1
    assert first["first_displaced_supercell"] == second["first_displaced_supercell"]
    assert first["supercell_atom_count"] == 2 * round(abs(np.linalg.det(first["supercell_matrix"])))

    # Independent cross-check with the BMD displacement plan (same Phonopy,
    # same amplitude and symprec, same supercell matrix).
    bmd_plan = build_displacement_plan(si(), supercell_matrix=first["supercell_matrix"])
    assert bmd_plan.task_count == first["displacement_count"]
    assert bmd_plan.supercell_natom == first["supercell_atom_count"]
    supercell = bmd_plan.to_dict()["supercell"]
    reference = np.array(supercell["coords"], dtype=float)
    displaced = first["first_displaced_supercell"].cart_coords
    delta = displaced - reference
    task = bmd_plan.tasks[0]
    moved = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-8)
    assert moved.tolist() == [task.atom_index]
    assert np.allclose(delta[task.atom_index], task.displacement, atol=1e-10)
    assert np.isclose(np.linalg.norm(task.displacement), 0.01)


# --- 5. admission -----------------------------------------------------------------------


def test_non_magnetic_pbe_phonons_are_admitted_with_a_planning_preview():
    resolved = plan(desired_output="phonons")
    assert resolved.workflow_spec == PBE_PHONONS.__class__(PBE_PHONONS.stages, recipe="phonons")
    (phonon_plan,) = resolved.generated_inputs["phonon_plans"]
    assert phonon_plan["stage"] == 2
    # 10 A minimum image length: the 2x2x2 conventional cell (10.86 A cube).
    assert phonon_plan["supercell_matrix"] == [[-2, 2, 2], [2, -2, 2], [2, 2, -2]]
    assert phonon_plan["supercell_atom_count"] == 64
    assert phonon_plan["settings"]["supercell"]["min_length_angstrom"] == 10.0
    assert phonon_plan["displacement_count"] == 1
    assert phonon_plan["settings"]["displacement_angstrom"] == 0.01
    assert phonon_plan["settings"]["born_maker"] is None
    assert [item["name"] for item in phonon_plan["stages"]][1:] == list(PHONON_JOB_NAMES)
    incar = resolved.generated_inputs["incar"]
    assert "# Stage 2 - Phonon Force Calculations (PBE)" in incar
    assert "ENCUT = 700.0" in incar.split("# Stage 2")[1]
    assert "NCORE" not in incar.split("# Stage 2")[1]
    assert "EDIFFG = -0.001" in incar.split("# Stage 2")[0]
    assert "ENCUT = 700" in incar.split("# Stage 2")[0]


def test_d3_is_applied_consistently_to_relax_and_forces():
    workflow = WorkflowSpec([with_dispersion(stage) for stage in PBE_PHONONS.stages])
    resolved = plan(custom_workflow=workflow.to_dict())
    stage_2 = resolved.generated_inputs["incar"].split("# Stage 2")[1]
    stage_1 = resolved.generated_inputs["incar"].split("# Stage 2")[0]
    assert "IVDW = 11" in stage_1 and "IVDW = 11" in stage_2

    for stages in (
        [with_dispersion(PBE_PHONONS.stages[0]), PBE_PHONONS.stages[1]],
        [PBE_PHONONS.stages[0], with_dispersion(PBE_PHONONS.stages[1])],
    ):
        with pytest.raises(CalculationValidationError, match="same van der Waals correction"):
            validate_workflow_spec(WorkflowSpec(stages))

    # Automatic dispersion reaches both stages of the phonon workflow.
    stages, treatment = _apply_dispersion(PBE_PHONONS.stages)
    assert all(Modifier.DISPERSION in stage.modifiers for stage in stages)
    assert treatment.stage_indices == (1, 2)


# --- 6. rejections ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("structure", "triggered"),
    [
        (FE_POSCAR, "Spin Polarised"),
        (soc_cases.BI_POSCAR, "Spin-Orbit Coupling (SOC)"),
        (NIO_POSCAR, "DFT+U"),
    ],
    ids=["magnetic", "soc", "dft_u"],
)
def test_phonons_desired_output_rejects_structures_needing_excluded_treatments(structure, triggered):
    with pytest.raises(CalculationValidationError) as excinfo:
        plan(structure, desired_output="phonons")
    assert excinfo.value.diagnostic["code"] == PHONONS_EXCLUDED_TREATMENT_CODE
    assert triggered in excinfo.value.diagnostic["triggered_treatments"]


@pytest.mark.parametrize("modifier", [Modifier.SPIN_POLARIZED, Modifier.SOC, Modifier.DFT_U, Modifier.GAMMA_ONLY])
def test_phonon_force_stage_rejects_unsupported_modifiers(modifier):
    stage = StageSpec(StageType.PHONON_FORCES, Theory.PBE, {modifier})
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(WorkflowSpec([PBE_PHONONS.stages[0], stage]))
    assert excinfo.value.diagnostic["code"] == PHONON_UNSUPPORTED_MODIFIER_CODE
    assert supported_stage_modifier_combinations(StageType.PHONON_FORCES, Theory.PBE) == (
        frozenset(),
        frozenset({Modifier.DISPERSION}),
    )


@pytest.mark.parametrize("modifier", [Modifier.SPIN_POLARIZED, Modifier.DFT_U])
def test_magnetic_or_dft_u_relax_cannot_feed_phonons(modifier):
    workflow = WorkflowSpec(
        [StageSpec(StageType.RELAX, Theory.PBE, {modifier}), PBE_PHONONS.stages[1]]
    )
    with pytest.raises(CalculationValidationError) as excinfo:
        require_admissible_new_calculation(workflow)
    assert excinfo.value.diagnostic["code"] == PHONONS_UNSUPPORTED_TREATMENT_CODE
    response = browser_build(SI_POSCAR, workflow="custom", workflow_spec=workflow)
    assert response.status_code == 400
    assert not response.context.get("submission_spec")


@pytest.mark.parametrize(
    "stages",
    [
        [StageSpec(StageType.RELAX, Theory.HSE06), StageSpec(StageType.PHONON_FORCES, Theory.HSE06)],
        [StageSpec(StageType.RELAX, Theory.HSE06), StageSpec(StageType.PHONON_FORCES, Theory.PBE)],
        [StageSpec(StageType.PHONON_FORCES, Theory.PBE)],
        [StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.PHONON_FORCES, Theory.PBE)],
        [StageSpec(StageType.RELAX, Theory.PBE, {Modifier.IONS_ONLY}), StageSpec(StageType.PHONON_FORCES, Theory.PBE)],
        [*PBE_PHONONS.stages, StageSpec(StageType.STATIC, Theory.PBE)],
    ],
    ids=["hse06_phonons", "hse06_relax", "no_relax", "after_static", "ions_only_relax", "not_final"],
)
def test_unsupported_phonon_workflow_shapes_are_rejected(stages):
    with pytest.raises(CalculationValidationError):
        validate_workflow_spec(WorkflowSpec(stages))


# --- 7. browser and machine API ---------------------------------------------------------


def test_browser_and_planner_build_the_same_phonon_submission():
    browser = browser_build(SI_POSCAR)
    assert browser.status_code == 200
    ui = browser.context
    api = plan(desired_output="phonons")
    ui_flow = ui["submission_spec"]["flow_spec"]
    api_flow = api.submission_spec["flow_spec"]
    assert ui_flow["workflow_spec"] == api_flow["workflow_spec"]
    assert ui_flow.get("automatic_treatments") == api_flow.get("automatic_treatments")
    assert ui["generated_inputs"]["incar"] == api.generated_inputs["incar"]
    assert ui["generated_inputs"]["phonon_plans"] == api.generated_inputs["phonon_plans"]
    assert "Phonon Plan (stage 2)" in browser.body.decode()


# --- 8. no regression ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("desired_output", "shape"),
    [
        ("energy_only", [("static", "pbe")]),
        ("relaxed_structure", [("relax", "pbe"), ("relax", "pbe")]),
        ("electronic_dos", [("relax", "pbe"), ("static", "hse06"), ("dos", "hse06")]),
        ("electronic_band_structure", [("relax", "pbe"), ("static", "hse06"), ("band_structure", "hse06")]),
    ],
)
def test_existing_desired_outputs_are_unchanged(desired_output, shape):
    workflow = desired_output_workflow_spec(desired_output)
    assert [(stage.stage_type.value, stage.theory.value) for stage in workflow.stages] == shape
    resolved = plan(desired_output=desired_output)
    assert "phonon_plans" not in resolved.generated_inputs
    assert "phonon" not in resolved.generated_inputs["incar"].lower()
