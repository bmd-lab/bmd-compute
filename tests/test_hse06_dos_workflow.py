from __future__ import annotations

import sys
import types
from contextlib import contextmanager

from pymatgen.core import Lattice, Structure

from backend.calculations.models import (
    CalculationSpec,
    Modifier,
    Purpose,
    StageSpec,
    StageType,
    Theory,
    WorkflowSpec,
)
from backend.calculations.registry import (
    CalculationValidationError,
    calculation_form_options,
    calculation_spec_from_workflow_spec,
    desired_output_workflow_spec,
    validate_calculation_spec,
    validate_stage_spec,
    validate_workflow_spec,
    workflow_result_stage_directory,
    workflow_stage_directories,
)
from backend.calculations.vasp_stage_definitions import HSE_DOS_RECIPROCAL_DENSITY_DEFAULT
from backend.generated_inputs import preview_generated_inputs
from backend.submission import create_submission_spec
from backend.workflow_summary import calculation_plan_from_workflow_spec
from backend.workflows import build_atomate2_flow_for_workflow_spec


class FakeText:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class FakeIncar(dict):
    def __str__(self):
        return "\n".join(
            f"{key} = {value}"
            for key, value in sorted(self.items())
            if value is not None
        )


class FakeOutput:
    def __init__(self, *, structure, dir_name):
        self.structure = structure
        self.dir_name = dir_name


class FakeJob:
    def __init__(
        self,
        *,
        name,
        structure,
        input_set_generator,
        prev_dir=None,
        mode=None,
        maker_kwargs=None,
    ):
        self.name = name
        self.structure = structure
        self.input_set_generator = input_set_generator
        self.prev_dir = prev_dir
        self.mode = mode
        self.maker_kwargs = maker_kwargs or {}
        self.output = FakeOutput(
            structure=f"{name}_output_structure",
            dir_name=f"/remote/run/{name}",
        )


class FakeGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_input_set(self, structure, potcar_spec=True):
        assert potcar_spec is True
        return types.SimpleNamespace(
            incar=FakeIncar(self.kwargs["user_incar_settings"]),
            kpoints=FakeText("KPOINTS"),
            poscar=FakeText(f"POSCAR {structure}"),
        )


class FakeRelaxSetGenerator(FakeGenerator):
    def get_input_set(self, structure, potcar_spec=True):
        assert potcar_spec is True
        incar = {
            "IBRION": 2,
            "ISIF": 3,
            "NSW": 99,
        }
        incar.update(self.kwargs["user_incar_settings"])
        return types.SimpleNamespace(
            incar=FakeIncar(incar),
            kpoints=FakeText("KPOINTS"),
            poscar=FakeText(f"POSCAR {structure}"),
        )


class FakeHSEBSSetGenerator(FakeGenerator):
    def get_input_set(self, structure, potcar_spec=True):
        assert potcar_spec is True
        return types.SimpleNamespace(
            incar=FakeIncar(self.kwargs["user_incar_settings"]),
            kpoints=FakeText(
                "HSE KPOINTS "
                f"mode={self.kwargs['mode']} "
                f"reciprocal_density={self.kwargs['reciprocal_density']} "
                f"line_density={self.kwargs.get('line_density', '<absent>')}"
            ),
            poscar=FakeText(f"POSCAR {structure}"),
        )


class FakeVaspErrorHandler:
    error_msgs = {
        "auto_nbands": ["Automatically corrected NBANDS"],
        "brmix": ["BRMIX: very serious problems"],
        "tet": ["Tetrahedron method fails"],
    }

    def __init__(
        self,
        output_filename="vasp.out",
        errors_subset_to_catch=None,
        vtst_fixes=False,
    ):
        self.output_filename = output_filename
        self.errors_subset_to_catch = list(
            errors_subset_to_catch or self.error_msgs
        )
        self.vtst_fixes = vtst_fixes


class FakeOtherHandler:
    def __init__(self, name):
        self.name = name


class FrozenJobErrorHandler(FakeOtherHandler):
    pass


FrozenJobErrorHandler.__module__ = "custodian.vasp.handlers"


class FakeMaker:
    def __init__(self, *, input_set_generator, name, **kwargs):
        self.input_set_generator = input_set_generator
        self.name = name
        self.kwargs = kwargs

    def make(self, structure, prev_dir=None, mode=None):
        return FakeJob(
            name=self.name,
            structure=structure,
            input_set_generator=self.input_set_generator,
            prev_dir=prev_dir,
            mode=mode,
            maker_kwargs=self.kwargs,
        )


class FakeFlow:
    def __init__(self, jobs, name=None, metadata=None):
        self.jobs = jobs
        self.name = name
        self.metadata = metadata or {}


@contextmanager
def fake_atomate2_and_jobflow():
    modules = {
        "atomate2": types.ModuleType("atomate2"),
        "atomate2.vasp": types.ModuleType("atomate2.vasp"),
        "atomate2.vasp.jobs": types.ModuleType("atomate2.vasp.jobs"),
        "atomate2.vasp.jobs.core": types.ModuleType("atomate2.vasp.jobs.core"),
        "atomate2.vasp.run": types.ModuleType("atomate2.vasp.run"),
        "atomate2.vasp.sets": types.ModuleType("atomate2.vasp.sets"),
        "atomate2.vasp.sets.core": types.ModuleType("atomate2.vasp.sets.core"),
        "custodian": types.ModuleType("custodian"),
        "custodian.vasp": types.ModuleType("custodian.vasp"),
        "custodian.vasp.handlers": types.ModuleType("custodian.vasp.handlers"),
        "jobflow": types.ModuleType("jobflow"),
    }
    modules["atomate2"].vasp = modules["atomate2.vasp"]
    modules["atomate2.vasp"].jobs = modules["atomate2.vasp.jobs"]
    modules["atomate2.vasp"].run = modules["atomate2.vasp.run"]
    modules["atomate2.vasp"].sets = modules["atomate2.vasp.sets"]
    modules["atomate2.vasp.jobs"].core = modules["atomate2.vasp.jobs.core"]
    modules["atomate2.vasp.sets"].core = modules["atomate2.vasp.sets.core"]
    modules["custodian"].vasp = modules["custodian.vasp"]
    modules["custodian.vasp"].handlers = modules["custodian.vasp.handlers"]
    modules["atomate2.vasp.jobs.core"].RelaxMaker = FakeMaker
    modules["atomate2.vasp.jobs.core"].StaticMaker = FakeMaker
    modules["atomate2.vasp.jobs.core"].NonSCFMaker = FakeMaker
    modules["atomate2.vasp.jobs.core"].HSEBSMaker = FakeMaker
    modules["atomate2.vasp.run"].DEFAULT_HANDLERS = (
        FakeVaspErrorHandler(),
        FakeOtherHandler("mesh_symmetry"),
        FrozenJobErrorHandler("frozen_job"),
    )
    modules["atomate2.vasp.sets.core"].RelaxSetGenerator = FakeRelaxSetGenerator
    modules["atomate2.vasp.sets.core"].StaticSetGenerator = FakeGenerator
    modules["atomate2.vasp.sets.core"].NonSCFSetGenerator = FakeGenerator
    modules["atomate2.vasp.sets.core"].HSEBSSetGenerator = FakeHSEBSSetGenerator
    modules["custodian.vasp.handlers"].VaspErrorHandler = FakeVaspErrorHandler
    modules["custodian.vasp.handlers"].FrozenJobErrorHandler = FrozenJobErrorHandler
    modules["jobflow"].Flow = FakeFlow

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _si_structure() -> Structure:
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def _hse_dos_workflow(
    *,
    precursor_theory=Theory.PBE,
    modifiers=(),
) -> WorkflowSpec:
    return WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, precursor_theory),
            StageSpec(StageType.DOS, Theory.HSE06, modifiers),
        ],
        recipe="custom",
    )


def _terminal_section(generated_inputs: dict, label: str) -> str:
    return generated_inputs["incar"].split(label, 1)[1]


def _terminal_kpoints(generated_inputs: dict, label: str) -> str:
    return generated_inputs["kpoints"].split(label, 1)[1]


def test_hse06_dos_stage_and_supported_precursor_shapes_validate():
    stage = StageSpec(StageType.DOS, Theory.HSE06)
    assert validate_stage_spec(stage) == stage

    pbe_precursor = validate_workflow_spec(_hse_dos_workflow())
    hse_precursor = validate_workflow_spec(
        _hse_dos_workflow(precursor_theory=Theory.HSE06)
    )

    assert workflow_stage_directories(pbe_precursor) == (
        "stage_01",
        "stage_02",
        "stage_03",
    )
    assert workflow_result_stage_directory(pbe_precursor) == "stage_03"
    assert calculation_spec_from_workflow_spec(pbe_precursor) is None
    assert calculation_spec_from_workflow_spec(hse_precursor) is None
    assert calculation_plan_from_workflow_spec(pbe_precursor) == [
        "Geometry Optimisation (PBE)",
        "Static Energy (PBE)",
        "Density of States (HSE06)",
    ]
    assert calculation_plan_from_workflow_spec(hse_precursor) == [
        "Geometry Optimisation (PBE)",
        "Static Energy (HSE06)",
        "Density of States (HSE06)",
    ]


def test_hse06_dos_still_requires_a_static_precursor_and_legacy_recipe_stays_blocked():
    for invalid_workflow in (
        WorkflowSpec([StageSpec(StageType.DOS, Theory.HSE06)]),
        WorkflowSpec([
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.DOS, Theory.HSE06),
        ]),
    ):
        try:
            validate_workflow_spec(invalid_workflow)
        except CalculationValidationError:
            pass
        else:
            raise AssertionError(f"{invalid_workflow} should fail validation.")

    legacy_hse_dos = CalculationSpec(Purpose.DOS, Theory.HSE06)
    try:
        validate_calculation_spec(legacy_hse_dos)
    except CalculationValidationError as exc:
        assert "HSE06 is currently supported" in exc.message
    else:
        raise AssertionError("Legacy HSE06 DOS recipe should remain unsupported.")


def test_hse06_band_still_requires_hse06_static_precursor():
    invalid_band = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.PBE),
            StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
        ]
    )
    try:
        validate_workflow_spec(invalid_band)
    except CalculationValidationError as exc:
        assert "same level of theory" in exc.message
    else:
        raise AssertionError("HSE06 Band Structure should still require HSE06 Static.")


def test_hse06_dos_execution_uses_hsebs_maker_and_uniform_generator():
    workflow = _hse_dos_workflow()

    with fake_atomate2_and_jobflow():
        flow = build_atomate2_flow_for_workflow_spec(
            "initial_structure",
            workflow,
            label="Si-hse-dos",
            resources={"ntasks": 24},
        )

    assert flow.name == "Si-hse-dos_custom_workflow"
    assert flow.metadata["bmd_stage_directories"] == ("stage_01", "stage_02", "stage_03")
    assert [job.name for job in flow.jobs] == ["stage_01", "stage_02", "stage_03"]

    relax_job, static_job, dos_job = flow.jobs
    assert static_job.structure == relax_job.output.structure
    assert static_job.prev_dir == relax_job.output.dir_name
    assert dos_job.structure == static_job.output.structure
    assert dos_job.prev_dir == static_job.output.dir_name
    assert dos_job.mode == "uniform"
    assert isinstance(dos_job.input_set_generator, FakeHSEBSSetGenerator)

    dos_kwargs = dos_job.input_set_generator.kwargs
    assert dos_kwargs["mode"] == "uniform"
    assert dos_kwargs["reciprocal_density"] == HSE_DOS_RECIPROCAL_DENSITY_DEFAULT
    assert "line_density" not in dos_kwargs

    terminal_incar = dos_kwargs["user_incar_settings"]
    assert terminal_incar["LHFCALC"] is True
    assert terminal_incar["AEXX"] == 0.25
    assert terminal_incar["HFSCREEN"] == 0.2
    assert terminal_incar["ALGO"] == "Normal"
    assert terminal_incar["PRECFOCK"] == "Fast"
    assert terminal_incar["NEDOS"] == 4001
    assert terminal_incar["ISMEAR"] == -5
    assert terminal_incar["ISPIN"] == 1
    assert terminal_incar["NCORE"] == 8
    assert "MAGMOM" not in terminal_incar
    assert "ICHARG" not in terminal_incar

    run_vasp_kwargs = dos_job.maker_kwargs["run_vasp_kwargs"]
    handlers = run_vasp_kwargs["handlers"]
    vasp_handlers = [
        handler
        for handler in handlers
        if isinstance(handler, FakeVaspErrorHandler)
    ]
    assert len(vasp_handlers) == 1
    assert "auto_nbands" not in vasp_handlers[0].errors_subset_to_catch
    assert not any(isinstance(handler, FrozenJobErrorHandler) for handler in handlers)


def test_electronic_dos_desired_output_uses_existing_hse_static_and_hse_dos_paths():
    workflow = desired_output_workflow_spec("electronic_dos")

    with fake_atomate2_and_jobflow():
        flow = build_atomate2_flow_for_workflow_spec(
            "initial_structure",
            workflow,
            label="Si-default-dos",
            resources={"ntasks": 24},
        )

    assert [job.name for job in flow.jobs] == ["stage_01", "stage_02", "stage_03"]
    relax_job, static_job, dos_job = flow.jobs
    assert static_job.structure == relax_job.output.structure
    assert static_job.prev_dir == relax_job.output.dir_name
    assert dos_job.structure == static_job.output.structure
    assert dos_job.prev_dir == static_job.output.dir_name
    assert dos_job.mode == "uniform"
    assert isinstance(dos_job.input_set_generator, FakeHSEBSSetGenerator)

    static_incar = static_job.input_set_generator.kwargs["user_incar_settings"]
    assert static_incar["LHFCALC"] is True
    assert static_incar["AEXX"] == 0.25
    assert static_incar["HFSCREEN"] == 0.2
    assert static_incar["PRECFOCK"] == "Accurate"
    assert static_incar["ISMEAR"] == 0
    assert static_incar["NCORE"] == 8

    dos_kwargs = dos_job.input_set_generator.kwargs
    assert dos_kwargs["mode"] == "uniform"
    assert dos_kwargs["reciprocal_density"] == HSE_DOS_RECIPROCAL_DENSITY_DEFAULT
    terminal_incar = dos_kwargs["user_incar_settings"]
    assert terminal_incar["LHFCALC"] is True
    assert terminal_incar["AEXX"] == 0.25
    assert terminal_incar["HFSCREEN"] == 0.2
    assert terminal_incar["ALGO"] == "Normal"
    assert terminal_incar["PRECFOCK"] == "Fast"
    assert terminal_incar["NEDOS"] == 4001
    assert terminal_incar["ISMEAR"] == -5
    assert "ICHARG" not in terminal_incar


def test_hse06_dos_preview_uses_real_uniform_weighted_hse_inputs_without_line_path():
    generated_inputs = preview_generated_inputs(
        _si_structure(),
        _hse_dos_workflow(),
        resources={"ntasks": 24},
    )
    terminal_label = "# Stage 3 - Density of States (HSE06)"
    terminal_incar = _terminal_section(generated_inputs, terminal_label)
    terminal_kpoints = _terminal_kpoints(generated_inputs, terminal_label)

    assert "LHFCALC = True" in terminal_incar
    assert "HFSCREEN = 0.2" in terminal_incar
    assert "AEXX = 0.25" in terminal_incar
    assert "ALGO = Normal" in terminal_incar
    assert "PRECFOCK = Fast" in terminal_incar
    assert "NEDOS = 4001" in terminal_incar
    assert "ISMEAR = -5" in terminal_incar
    assert "ISPIN = 1" in terminal_incar
    assert "NCORE = 8" in terminal_incar
    assert "ICHARG" not in terminal_incar
    assert "MAGMOM" not in terminal_incar

    assert "pymatgen with grid density = 310 / number of atoms" in terminal_kpoints
    assert "Gamma" in terminal_kpoints
    assert "5 5 5" in terminal_kpoints
    assert "Combined k-points" not in terminal_kpoints
    assert "Non SCF run along symmetry lines" not in terminal_kpoints
    assert "\\Gamma" not in terminal_kpoints


def test_hse06_dos_spin_on_preview_keeps_existing_spin_semantics():
    generated_inputs = preview_generated_inputs(
        _si_structure(),
        _hse_dos_workflow(modifiers={Modifier.SPIN_POLARIZED}),
        resources={"ntasks": 24},
    )
    terminal_incar = _terminal_section(
        generated_inputs,
        "# Stage 3 - Density of States (HSE06)",
    )

    assert "ISPIN = 2" in terminal_incar
    assert "MAGMOM = 2*0.6" in terminal_incar
    assert "LHFCALC = True" in terminal_incar
    assert "ICHARG" not in terminal_incar


def test_existing_pbe_dos_preview_and_recommended_recipe_remain_unchanged():
    pbe_dos = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.PBE),
            StageSpec(StageType.DOS, Theory.PBE),
        ],
        recipe="dos",
    )
    generated_inputs = preview_generated_inputs(
        _si_structure(),
        pbe_dos,
        resources={"ntasks": 24},
    )
    terminal_incar = _terminal_section(
        generated_inputs,
        "# Stage 3 - Density of States (PBE)",
    )

    assert "ICHARG = 11" in terminal_incar
    assert "NEDOS = 4001" in terminal_incar
    assert "LHFCALC" not in terminal_incar

    recipes = {
        recipe["value"]: WorkflowSpec.from_dict(recipe["workflow_spec"])
        for recipe in calculation_form_options()["recipes"]
    }
    assert recipes["dos"] == pbe_dos


def test_hse06_dos_rejects_unsupported_modifiers():
    unsupported_workflows = (
        # The DOS stage would size NBANDS from a collinear static run.
        _hse_dos_workflow(modifiers={Modifier.SOC}),
        _hse_dos_workflow(modifiers={Modifier.DFT_U}),
        _hse_dos_workflow(modifiers={Modifier.DISPERSION}),
    )
    for workflow in unsupported_workflows:
        try:
            validate_workflow_spec(workflow)
        except CalculationValidationError as exc:
            assert (
                "Spin-Orbit Coupling (SOC)" in exc.message
                or "DFT+U" in exc.message
                or "van der Waals correction" in exc.message
            )
        else:
            raise AssertionError(f"{workflow} should fail validation.")


def test_hse06_dos_submission_records_custom_stage_workflow():
    workflow = _hse_dos_workflow()
    submission = create_submission_spec(
        {
            "workflow_spec": workflow.to_dict(),
            "potcar_functional": "PBE_64",
            "kpoints": None,
            "incar": {},
            "structure": {"type": "parsed"},
        },
        label="Si hse dos",
        timestamp="20260629-120000",
        env={},
    )

    assert submission["flow_spec"]["workflow"] == "custom_workflow"
    assert submission["flow_spec"]["workflow_spec"] == workflow.to_dict()
    assert submission["paths"]["result_dir"].endswith("/stage_03")
    assert submission["paths"]["stage_dirs"] == {
        "stage_01": "/bmd-db/guest/flows/Si-hse-dos-20260629-120000/stage_01",
        "stage_02": "/bmd-db/guest/flows/Si-hse-dos-20260629-120000/stage_02",
        "stage_03": "/bmd-db/guest/flows/Si-hse-dos-20260629-120000/stage_03",
    }
