from __future__ import annotations

import sys
import types
from contextlib import contextmanager

from atomate2.vasp.run import DEFAULT_HANDLERS as ATOMATE2_DEFAULT_HANDLERS

from backend.calculations.models import CalculationSpec, Purpose, Theory
from backend.workflow_summary import summarize_workflow
from backend.workflows import build_dos_flow


class FakeOutput:
    def __init__(self, *, structure, dir_name):
        self.structure = structure
        self.dir_name = dir_name


class FakeJob:
    def __init__(self, *, name, structure, input_set_generator, prev_dir=None, mode=None):
        self.name = name
        self.structure = structure
        self.input_set_generator = input_set_generator
        self.prev_dir = prev_dir
        self.mode = mode
        self.output = FakeOutput(
            structure=f"{name}_output_structure",
            dir_name=f"/remote/run/{name}",
        )


class FakeGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeRelaxSetGenerator(FakeGenerator):
    pass


class FakeStaticSetGenerator(FakeGenerator):
    pass


class FakeNonSCFSetGenerator(FakeGenerator):
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


class FakeNonSCFMaker(FakeRelaxMaker):
    def make(self, structure, prev_dir=None, mode="uniform"):
        return FakeJob(
            name=self.name,
            structure=structure,
            input_set_generator=self.input_set_generator,
            prev_dir=prev_dir,
            mode=mode,
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
    modules["atomate2.vasp.jobs.core"].NonSCFMaker = FakeNonSCFMaker
    modules["atomate2.vasp.run"].DEFAULT_HANDLERS = ATOMATE2_DEFAULT_HANDLERS
    modules["atomate2.vasp.sets.core"].RelaxSetGenerator = FakeRelaxSetGenerator
    modules["atomate2.vasp.sets.core"].StaticSetGenerator = FakeStaticSetGenerator
    modules["atomate2.vasp.sets.core"].NonSCFSetGenerator = FakeNonSCFSetGenerator
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


with fake_atomate2_and_jobflow():
    flow = build_dos_flow(
        "initial_structure",
        label="Si-dos",
        resources={"ntasks": 24},
    )

assert flow.name == "Si-dos_dos"
assert flow.metadata["bmd_stage_directories"] == ("stage_01", "stage_02", "stage_03")
assert [job.name for job in flow.jobs] == ["stage_01", "stage_02", "stage_03"]

relax_job, static_job, dos_job = flow.jobs
assert static_job.structure == relax_job.output.structure
assert static_job.prev_dir == relax_job.output.dir_name
assert dos_job.structure == static_job.output.structure
assert dos_job.prev_dir == static_job.output.dir_name
assert dos_job.mode == "uniform"

assert isinstance(relax_job.input_set_generator, FakeRelaxSetGenerator)
assert isinstance(static_job.input_set_generator, FakeStaticSetGenerator)
assert isinstance(dos_job.input_set_generator, FakeNonSCFSetGenerator)
assert dos_job.input_set_generator.kwargs["mode"] == "uniform"

static_incar = static_job.input_set_generator.kwargs["user_incar_settings"]
dos_incar = dos_job.input_set_generator.kwargs["user_incar_settings"]
for inherited_key in ("ENCUT", "ENAUG", "ADDGRID", "GGA"):
    assert dos_incar[inherited_key] == static_incar[inherited_key]
assert static_incar["ENCUT"] == 620
assert static_incar["ADDGRID"] is True
assert static_incar["ENAUG"] is None
assert static_incar["GGA"] is None
assert dos_incar["NCORE"] == static_incar["NCORE"] == 4
assert dos_incar["ICHARG"] == 11
assert dos_incar["NEDOS"] == static_incar["NEDOS"] == 4001
assert dos_incar["LORBIT"] == static_incar["LORBIT"] == 11

with fake_atomate2_and_jobflow():
    flow_with_overrides = build_dos_flow(
        "initial_structure",
        label="custom-dos",
        incar={
            "ENCUT": 700,
            "ENAUG": 1400,
            "ADDGRID": False,
            "GGA": "PE",
            "NEDOS": 6001,
        },
    )

_, override_static_job, override_dos_job = flow_with_overrides.jobs
override_static_incar = override_static_job.input_set_generator.kwargs[
    "user_incar_settings"
]
override_dos_incar = override_dos_job.input_set_generator.kwargs[
    "user_incar_settings"
]
for inherited_key in ("ENCUT", "ENAUG", "ADDGRID", "GGA"):
    assert override_dos_incar[inherited_key] == override_static_incar[inherited_key]
assert override_dos_incar["ENCUT"] == 700
assert override_dos_incar["ENAUG"] == 1400
assert override_dos_incar["ADDGRID"] is False
assert override_dos_incar["GGA"] == "PE"
assert override_dos_incar["NEDOS"] == override_static_incar["NEDOS"] == 6001
assert override_dos_incar["ICHARG"] == 11

summary = summarize_workflow(flow, CalculationSpec(Purpose.DOS, Theory.PBE))
assert summary["calculation_type"] == "Density of States"
assert summary["number_of_jobs"] == 3
assert summary["calculation_plan"] == [
    "Geometry Optimisation",
    "Static Energy",
    "Density of States",
]

print("dos stage chaining smoke test passed")
