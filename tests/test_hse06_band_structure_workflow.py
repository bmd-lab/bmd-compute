from __future__ import annotations

import sys
import types
from contextlib import contextmanager

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
    calculation_spec_from_workflow_spec,
    validate_calculation_spec,
    validate_stage_spec,
    validate_workflow_spec,
    workflow_result_stage_directory,
    workflow_stage_directories,
)
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
                f"line_density={self.kwargs['line_density']} "
                f"reciprocal_density={self.kwargs['reciprocal_density']}"
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


def assert_hsebs_custodian_policy_allows_auto_nbands(job):
    run_vasp_kwargs = job.maker_kwargs["run_vasp_kwargs"]
    handlers = run_vasp_kwargs["handlers"]
    vasp_handlers = [
        handler
        for handler in handlers
        if isinstance(handler, FakeVaspErrorHandler)
    ]

    assert len(vasp_handlers) == 1
    assert "auto_nbands" not in vasp_handlers[0].errors_subset_to_catch
    assert "brmix" in vasp_handlers[0].errors_subset_to_catch
    assert "tet" in vasp_handlers[0].errors_subset_to_catch
    assert any(isinstance(handler, FakeOtherHandler) for handler in handlers)
    assert not any(isinstance(handler, FrozenJobErrorHandler) for handler in handlers)


hse_band_stage = StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06)
assert validate_stage_spec(hse_band_stage) == hse_band_stage

hse_band_spec = CalculationSpec(Purpose.BAND_STRUCTURE, Theory.HSE06)
try:
    validate_calculation_spec(hse_band_spec)
except CalculationValidationError:
    pass
else:
    raise AssertionError("Legacy HSE06 Band Structure should remain unsupported.")

hse_band_workflow = WorkflowSpec(
    [
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.HSE06),
        StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
    ],
    recipe="custom",
)
# Free-form INCAR in stage options is not an accepted workflow input.
try:
    validate_workflow_spec(
        WorkflowSpec(
            [
                StageSpec(StageType.RELAX, Theory.PBE),
                StageSpec(StageType.STATIC, Theory.HSE06),
                StageSpec(
                    StageType.BAND_STRUCTURE,
                    Theory.HSE06,
                    options={"incar": {"NCORE": 4}},
                ),
            ],
            recipe="custom",
        )
    )
except CalculationValidationError as exc:
    assert "'incar'" in exc.message
else:
    raise AssertionError("Stage options must not carry INCAR settings.")
validated_workflow = validate_workflow_spec(hse_band_workflow)
assert calculation_spec_from_workflow_spec(validated_workflow) is None
assert workflow_stage_directories(validated_workflow) == (
    "stage_01",
    "stage_02",
    "stage_03",
)
assert workflow_result_stage_directory(validated_workflow) == "stage_03"
assert calculation_plan_from_workflow_spec(validated_workflow) == [
    "Geometry Optimisation (PBE)",
    "Static Energy (HSE06)",
    "Band Structure (HSE06)",
]

for invalid_workflow in (
    WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.PBE),
            StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
        ]
    ),
    WorkflowSpec([StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06)]),
    WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.HSE06),
            StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06, {Modifier.GAMMA_ONLY}),
        ]
    ),
):
    try:
        validate_workflow_spec(invalid_workflow)
    except CalculationValidationError:
        pass
    else:
        raise AssertionError(f"{invalid_workflow} should fail validation.")

with fake_atomate2_and_jobflow():
    flow = build_atomate2_flow_for_workflow_spec(
        "initial_structure",
        hse_band_workflow,
        label="Si-hse-bands",
        resources={"ntasks": 24},
    )
    override_flow = build_atomate2_flow_for_workflow_spec(
        "initial_structure",
        hse_band_workflow,
        label="Si-hse-bands-override",
        resources={"ntasks": 24},
        kpoints={"mode": "line_density", "value": 32},
    )
    # Internal builder-level INCAR input (not a workflow-spec field) still
    # keeps an explicit NCORE on the band stage.
    explicit_ncore_flow = build_atomate2_flow_for_workflow_spec(
        "initial_structure",
        hse_band_workflow,
        label="Si-hse-bands-explicit-ncore",
        resources={"ntasks": 24},
        incar={"NCORE": 4},
    )
    generated_inputs = preview_generated_inputs(
        "initial_structure",
        hse_band_workflow,
        resources={"ntasks": 24},
    )

assert flow.name == "Si-hse-bands_custom_workflow"
assert flow.metadata["bmd_stage_directories"] == ("stage_01", "stage_02", "stage_03")
assert [job.name for job in flow.jobs] == ["stage_01", "stage_02", "stage_03"]

relax_job, static_job, band_job = flow.jobs
assert static_job.structure == relax_job.output.structure
assert static_job.prev_dir == relax_job.output.dir_name
assert band_job.structure == static_job.output.structure
assert band_job.prev_dir == static_job.output.dir_name
assert band_job.mode == "line"
assert isinstance(band_job.input_set_generator, FakeHSEBSSetGenerator)
assert_hsebs_custodian_policy_allows_auto_nbands(band_job)

relax_incar = relax_job.input_set_generator.kwargs["user_incar_settings"]
static_incar = static_job.input_set_generator.kwargs["user_incar_settings"]
band_incar = band_job.input_set_generator.kwargs["user_incar_settings"]

assert "LHFCALC" not in relax_incar
assert static_incar["LHFCALC"] is True
assert band_incar["LHFCALC"] is True
assert relax_incar["ISPIN"] == 1
assert static_incar["ISPIN"] == 1
assert band_incar["ISPIN"] == 1
assert "MAGMOM" not in relax_incar
assert "MAGMOM" not in static_incar
assert "MAGMOM" not in band_incar
assert static_incar["PRECFOCK"] == "Accurate"
assert band_incar["PRECFOCK"] == "Fast"
assert static_incar["ISMEAR"] == 0
assert band_incar["ISMEAR"] == 0
assert band_incar["SIGMA"] == 0.01
assert band_incar["ALGO"] == "Normal"
assert band_incar["AEXX"] == 0.25
assert band_incar["HFSCREEN"] == 0.2
assert band_incar["GGA"] == static_incar["GGA"] == "PE"
assert band_incar["ENCUT"] == static_incar["ENCUT"] == 620
assert band_incar["ADDGRID"] == static_incar["ADDGRID"] is True
assert static_incar["NCORE"] == 8
assert "NCORE" not in band_incar
assert "ICHARG" not in band_incar

band_kwargs = band_job.input_set_generator.kwargs
assert band_kwargs["mode"] == "line"
assert band_kwargs["line_density"] == 40
assert band_kwargs["reciprocal_density"] == 64
assert "user_kpoints_settings" not in band_kwargs

override_band_kwargs = override_flow.jobs[2].input_set_generator.kwargs
assert override_band_kwargs["line_density"] == 32
assert override_band_kwargs["reciprocal_density"] == 64

explicit_ncore_static_incar = explicit_ncore_flow.jobs[1].input_set_generator.kwargs[
    "user_incar_settings"
]
explicit_ncore_band_incar = explicit_ncore_flow.jobs[2].input_set_generator.kwargs[
    "user_incar_settings"
]
assert explicit_ncore_static_incar["NCORE"] == 8
assert explicit_ncore_band_incar["NCORE"] == 4
assert_hsebs_custodian_policy_allows_auto_nbands(explicit_ncore_flow.jobs[2])

assert "# Stage 1 - Geometry Optimisation (PBE)" in generated_inputs["incar"]
assert "# Stage 2 - Static Energy (HSE06)" in generated_inputs["incar"]
assert "# Stage 3 - Band Structure (HSE06)" in generated_inputs["incar"]
static_section = generated_inputs["incar"].split("# Stage 2 - Static Energy (HSE06)", 1)[1].split(
    "# Stage 3 - Band Structure (HSE06)",
    1,
)[0]
band_section = generated_inputs["incar"].split("# Stage 3 - Band Structure (HSE06)", 1)[1]
relax_section = generated_inputs["incar"].split("# Stage 2 - Static Energy (HSE06)", 1)[0]
assert "ISPIN = 1" in relax_section
assert "MAGMOM" not in relax_section
assert "NCORE = 8" in static_section
assert "ISPIN = 1" in static_section
assert "MAGMOM" not in static_section
assert "LHFCALC = True" in band_section
assert "PRECFOCK = Fast" in band_section
assert "ALGO = Normal" in band_section
assert "ISMEAR = 0" in band_section
assert "SIGMA = 0.01" in band_section
assert "ISPIN = 1" in band_section
assert "MAGMOM" not in band_section
assert "NCORE" not in band_section
assert "ICHARG" not in band_section
assert "HSE KPOINTS mode=line line_density=40 reciprocal_density=64" in generated_inputs["kpoints"]

submission = create_submission_spec(
    {
        "workflow_spec": hse_band_workflow.to_dict(),
        "potcar_functional": "PBE_64",
        "kpoints": None,
        "incar": {},
        "structure": {"type": "parsed"},
    },
    label="Si hse bands",
    timestamp="20260629-120000",
    env={},
)
assert submission["flow_spec"]["workflow"] == "custom_workflow"
assert submission["flow_spec"]["workflow_spec"] == hse_band_workflow.to_dict()
assert submission["paths"]["result_dir"].endswith("/stage_03")
assert submission["paths"]["stage_dirs"] == {
    "stage_01": "/bmd-db/guest/flows/Si-hse-bands-20260629-120000/stage_01",
    "stage_02": "/bmd-db/guest/flows/Si-hse-bands-20260629-120000/stage_02",
    "stage_03": "/bmd-db/guest/flows/Si-hse-bands-20260629-120000/stage_03",
}

print("hse06 band structure workflow smoke test passed")
