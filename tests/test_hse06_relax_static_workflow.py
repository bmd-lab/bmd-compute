from __future__ import annotations

import sys
import types
from contextlib import contextmanager

from atomate2.vasp.run import DEFAULT_HANDLERS as ATOMATE2_DEFAULT_HANDLERS

from backend.calculations.models import CalculationSpec, Modifier, Purpose, Theory
from backend.calculations.registry import (
    CalculationValidationError,
    calculation_result_stage_directory,
    calculation_stage_directories,
    legacy_workflow_from_spec,
    validate_calculation_spec,
)
from backend.generated_inputs import preview_generated_inputs
from backend.workflow_summary import summarize_workflow
from backend.workflows import (
    build_atomate2_flow_for_spec,
    build_relax_input_set_generator,
    build_relax_static_flow,
    build_static_input_set_generator,
)


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
    def __init__(self, *, name, structure, input_set_generator, prev_dir=None):
        self.name = name
        self.structure = structure
        self.input_set_generator = input_set_generator
        self.prev_dir = prev_dir
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


class FakeStaticSetGenerator(FakeGenerator):
    pass


class FakeRelaxMaker:
    def __init__(self, *, input_set_generator, name, **kwargs):
        self.input_set_generator = input_set_generator
        self.name = name

    def make(self, structure):
        return FakeJob(
            name=self.name,
            structure=structure,
            input_set_generator=self.input_set_generator,
        )


class FakeStaticMaker(FakeRelaxMaker):
    def make(self, structure, prev_dir=None):
        return FakeJob(
            name=self.name,
            structure=structure,
            input_set_generator=self.input_set_generator,
            prev_dir=prev_dir,
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
        "jobflow": types.ModuleType("jobflow"),
    }
    modules["atomate2.vasp.jobs.core"].RelaxMaker = FakeRelaxMaker
    modules["atomate2.vasp.jobs.core"].StaticMaker = FakeStaticMaker
    modules["atomate2.vasp.run"].DEFAULT_HANDLERS = ATOMATE2_DEFAULT_HANDLERS
    modules["atomate2.vasp.sets.core"].RelaxSetGenerator = FakeRelaxSetGenerator
    modules["atomate2.vasp.sets.core"].StaticSetGenerator = FakeStaticSetGenerator
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


hse_relax_static_spec = CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06)
assert validate_calculation_spec(hse_relax_static_spec) == hse_relax_static_spec
assert legacy_workflow_from_spec(hse_relax_static_spec) == "relax_static"
assert calculation_stage_directories(hse_relax_static_spec) == ("stage_01", "stage_02")
assert calculation_result_stage_directory(hse_relax_static_spec) == "stage_02"

for unsupported_hse_spec in (
    CalculationSpec(Purpose.DOUBLE_RELAX, Theory.HSE06),
    CalculationSpec(Purpose.DOS, Theory.HSE06),
    CalculationSpec(Purpose.BAND_STRUCTURE, Theory.HSE06),
):
    try:
        validate_calculation_spec(unsupported_hse_spec)
    except CalculationValidationError:
        pass
    else:
        raise AssertionError(f"{unsupported_hse_spec} should remain unsupported.")

for unsupported_modifier_spec in (
    CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06, {Modifier.DFT_U}),
    CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06, {Modifier.SOC}),
):
    try:
        validate_calculation_spec(unsupported_modifier_spec)
    except CalculationValidationError:
        pass
    else:
        raise AssertionError(f"{unsupported_modifier_spec} should remain unsupported.")

with fake_atomate2_and_jobflow():
    pbe_relax_generator = build_relax_input_set_generator(
        "initial_structure",
        theory=Theory.PBE,
        resources={"ntasks": 24},
    )
    pbe_static_generator = build_static_input_set_generator(
        "initial_structure",
        theory=Theory.PBE,
        resources={"ntasks": 24},
    )
    flow = build_relax_static_flow(
        "initial_structure",
        label="Si-hse-relax-static",
        theory=Theory.HSE06,
        resources={"ntasks": 24},
    )

pbe_relax_incar = pbe_relax_generator.kwargs["user_incar_settings"]
pbe_static_incar = pbe_static_generator.kwargs["user_incar_settings"]
assert pbe_relax_incar["NCORE"] == 4
assert pbe_static_incar["NCORE"] == 4
assert "LHFCALC" not in pbe_relax_incar
assert "LHFCALC" not in pbe_static_incar
assert "PRECFOCK" not in pbe_relax_incar
assert "PRECFOCK" not in pbe_static_incar

assert flow.name == "Si-hse-relax-static_relax_static"
assert flow.metadata["bmd_stage_directories"] == ("stage_01", "stage_02")
assert [job.name for job in flow.jobs] == ["stage_01", "stage_02"]

relax_job, static_job = flow.jobs
assert static_job.structure == relax_job.output.structure
assert static_job.prev_dir == relax_job.output.dir_name

relax_incar = relax_job.input_set_generator.kwargs["user_incar_settings"]
static_incar = static_job.input_set_generator.kwargs["user_incar_settings"]
assert relax_incar["LHFCALC"] is True
assert static_incar["LHFCALC"] is True
assert relax_incar["HFSCREEN"] == static_incar["HFSCREEN"] == 0.2
assert relax_incar["AEXX"] == static_incar["AEXX"] == 0.25
assert relax_incar["GGA"] == static_incar["GGA"] == "PE"
assert relax_incar["PRECFOCK"] == "Fast"
assert static_incar["PRECFOCK"] == "Accurate"
assert static_incar["ISMEAR"] == 0
assert relax_incar["NCORE"] == 4
assert static_incar["NCORE"] == 4

summary = summarize_workflow(flow, hse_relax_static_spec)
assert summary["calculation_type"] == "Geometry Optimisation + Static Energy"
assert summary["number_of_jobs"] == 2
assert summary["job_names"] == ["stage_01", "stage_02"]
assert summary["calculation_plan"] == [
    "Geometry Optimisation",
    "Static Energy",
]
assert summary["theory"] == "hse06"
assert summary["theory_label"] == "HSE06"

with fake_atomate2_and_jobflow():
    flow_from_spec = build_atomate2_flow_for_spec(
        "initial_structure",
        hse_relax_static_spec,
        label="Si-hse",
        resources={"ntasks": 24},
    )
    generated_inputs = preview_generated_inputs(
        "initial_structure",
        hse_relax_static_spec,
        resources={"ntasks": 24},
    )

assert flow_from_spec.name == "Si-hse_relax_static"
assert [job.name for job in flow_from_spec.jobs] == ["stage_01", "stage_02"]
assert flow_from_spec.jobs[1].structure == flow_from_spec.jobs[0].output.structure
assert flow_from_spec.jobs[1].prev_dir == flow_from_spec.jobs[0].output.dir_name

assert "# Stage 1 - Geometry Optimisation (HSE06)" in generated_inputs["incar"]
assert "# Stage 2 - Static Energy (HSE06)" in generated_inputs["incar"]
assert "PRECFOCK = Fast" in generated_inputs["incar"]
assert "PRECFOCK = Accurate" in generated_inputs["incar"]
assert "ISMEAR = 0" in generated_inputs["incar"]
assert "ISMEAR = -5" not in generated_inputs["incar"]
assert "NCORE = 4" in generated_inputs["incar"]
assert "# Stage 1 - Geometry Optimisation (HSE06)" in generated_inputs["kpoints"]
assert "# Stage 2 - Static Energy (HSE06)" in generated_inputs["poscar"]

print("hse06 relax static workflow smoke test passed")
