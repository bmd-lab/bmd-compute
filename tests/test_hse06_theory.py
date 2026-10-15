from __future__ import annotations

import sys
import types
from contextlib import contextmanager

from atomate2.vasp.run import DEFAULT_HANDLERS as ATOMATE2_DEFAULT_HANDLERS

from backend.calculations.models import CalculationSpec, Modifier, Purpose, Theory
from backend.calculations.registry import CalculationValidationError, validate_calculation_spec
from backend.calculations.theory_policy import (
    CalculationStage,
    apply_theory_incar_settings,
    theory_incar_settings,
    theory_stage_from_purpose,
)
from backend.generated_inputs import preview_generated_inputs
from backend.workflow_summary import summarize_workflow
from backend.workflows import (
    build_atomate2_flow_for_spec,
    build_relax_input_set_generator,
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


class FakeStaticSetGenerator:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeStaticSetGenerator.last_kwargs = kwargs

    def get_input_set(self, structure, potcar_spec=True):
        assert potcar_spec is True
        return types.SimpleNamespace(
            incar=FakeIncar(self.kwargs["user_incar_settings"]),
            kpoints=FakeText("KPOINTS"),
            poscar=FakeText(f"POSCAR {structure}"),
        )


class FakeRelaxSetGenerator(FakeStaticSetGenerator):
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


class FakeOutput:
    structure = "hse_static_output_structure"
    dir_name = "/remote/run/hse_static"


class FakeJob:
    def __init__(self, *, name, structure, input_set_generator):
        self.name = name
        self.structure = structure
        self.input_set_generator = input_set_generator
        self.output = FakeOutput()


class FakeStaticMaker:
    def __init__(self, *, input_set_generator, name, **kwargs):
        self.input_set_generator = input_set_generator
        self.name = name

    def make(self, structure):
        return FakeJob(
            name=self.name,
            structure=structure,
            input_set_generator=self.input_set_generator,
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
    modules["atomate2.vasp.jobs.core"].StaticMaker = FakeStaticMaker
    modules["atomate2.vasp.jobs.core"].RelaxMaker = FakeStaticMaker
    modules["atomate2.vasp.run"].DEFAULT_HANDLERS = ATOMATE2_DEFAULT_HANDLERS
    modules["atomate2.vasp.sets.core"].StaticSetGenerator = FakeStaticSetGenerator
    modules["atomate2.vasp.sets.core"].RelaxSetGenerator = FakeRelaxSetGenerator
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


hse_static_policy = theory_incar_settings(Theory.HSE06, CalculationStage.STATIC)
assert hse_static_policy == {
    "LHFCALC": True,
    "AEXX": 0.25,
    "HFSCREEN": 0.2,
    "GGA": "PE",
    "ALGO": "Damped",
    "TIME": 0.4,
    "PRECFOCK": "Accurate",
    "ISMEAR": 0,
}
assert theory_incar_settings(Theory.HSE06, Purpose.STATIC) == hse_static_policy
assert theory_incar_settings(Theory.HSE06, purpose=Purpose.STATIC) == hse_static_policy
assert theory_stage_from_purpose(Purpose.STATIC) is CalculationStage.STATIC

hse_relax_policy = theory_incar_settings(Theory.HSE06, CalculationStage.RELAX)
assert hse_relax_policy["LHFCALC"] is True
assert hse_relax_policy["AEXX"] == 0.25
assert hse_relax_policy["HFSCREEN"] == 0.2
assert hse_relax_policy["PRECFOCK"] == "Fast"
assert theory_stage_from_purpose(Purpose.RELAX) is CalculationStage.RELAX

hse_dos_policy = theory_incar_settings(Theory.HSE06, CalculationStage.DOS)
assert hse_dos_policy["LHFCALC"] is True
assert hse_dos_policy["AEXX"] == 0.25
assert hse_dos_policy["HFSCREEN"] == 0.2
assert hse_dos_policy["ALGO"] == "Normal"
assert hse_dos_policy["PRECFOCK"] == "Fast"
assert hse_dos_policy["ISMEAR"] == -5
assert theory_stage_from_purpose(Purpose.DOS) is CalculationStage.DOS

hse_band_policy = theory_incar_settings(Theory.HSE06, CalculationStage.BAND_STRUCTURE)
assert hse_band_policy["LHFCALC"] is True
assert hse_band_policy["AEXX"] == 0.25
assert hse_band_policy["HFSCREEN"] == 0.2
assert hse_band_policy["ALGO"] == "Normal"
assert hse_band_policy["PRECFOCK"] == "Fast"
assert hse_band_policy["ISMEAR"] == 0
assert hse_band_policy["SIGMA"] == 0.01

assert theory_incar_settings(Theory.PBE, CalculationStage.STATIC) == {}
hse_applied = apply_theory_incar_settings(
    {"NCORE": 8, "KPAR": 2, "NPAR": 3, "ENCUT": 700},
    theory=Theory.HSE06,
    stage=CalculationStage.STATIC,
)
assert hse_applied["NCORE"] == 8
assert "KPAR" not in hse_applied
assert "NPAR" not in hse_applied
assert hse_applied["ENCUT"] == 700
assert hse_applied["LHFCALC"] is True
assert hse_applied["ISMEAR"] == 0

hse_static_spec = CalculationSpec(Purpose.STATIC, Theory.HSE06)
hse_relax_spec = CalculationSpec(Purpose.RELAX, Theory.HSE06)
hse_relax_static_spec = CalculationSpec(Purpose.RELAX_STATIC, Theory.HSE06)
hse_spin_static_spec = CalculationSpec(
    Purpose.STATIC,
    Theory.HSE06,
    {Modifier.SPIN_POLARIZED},
)
hse_spin_relax_spec = CalculationSpec(
    Purpose.RELAX,
    Theory.HSE06,
    {Modifier.SPIN_POLARIZED},
)
assert validate_calculation_spec(hse_static_spec) == hse_static_spec
assert validate_calculation_spec(hse_relax_spec) == hse_relax_spec
assert validate_calculation_spec(hse_relax_static_spec) == hse_relax_static_spec
assert validate_calculation_spec(hse_spin_static_spec) == hse_spin_static_spec
assert validate_calculation_spec(hse_spin_relax_spec) == hse_spin_relax_spec

for unsupported_spec in (
    CalculationSpec(Purpose.DOUBLE_RELAX, Theory.HSE06),
    CalculationSpec(Purpose.DOS, Theory.HSE06),
    CalculationSpec(Purpose.BAND_STRUCTURE, Theory.HSE06),
):
    try:
        validate_calculation_spec(unsupported_spec)
    except CalculationValidationError as exc:
        assert "HSE06 is currently supported" in exc.message
        assert "Geometry Optimisation" in exc.message
        assert "Static Energy" in exc.message
    else:
        raise AssertionError(f"{unsupported_spec} should fail validation.")

with fake_atomate2_and_jobflow():
    pbe_relax_generator = build_relax_input_set_generator(
        "Si2",
        theory=Theory.PBE,
        resources={"ntasks": 24},
    )
    pbe_generator = build_static_input_set_generator(
        "Si2",
        theory=Theory.PBE,
        resources={"ntasks": 24},
    )
    generator = build_static_input_set_generator(
        "Si2",
        theory=Theory.HSE06,
        resources={"ntasks": 24},
    )
    hse_relax_generator = build_relax_input_set_generator(
        "Si2",
        theory=Theory.HSE06,
        resources={"ntasks": 24},
    )
    hse_user_ncore_generator = build_static_input_set_generator(
        "Si2",
        theory=Theory.HSE06,
        resources={"ntasks": 24},
        incar={"NCORE": 2},
    )

pbe_relax_incar = pbe_relax_generator.kwargs["user_incar_settings"]
assert pbe_relax_incar["NCORE"] == 4
assert pbe_relax_incar["ENCUT"] == 580
assert pbe_relax_incar["ADDGRID"] is True
assert pbe_relax_incar["EDIFF"] == 1e-6
assert pbe_relax_incar["EDIFFG"] == -0.01
assert "LHFCALC" not in pbe_relax_incar
assert "HFSCREEN" not in pbe_relax_incar
assert "PRECFOCK" not in pbe_relax_incar

pbe_incar = pbe_generator.kwargs["user_incar_settings"]
assert pbe_incar["NCORE"] == 4
assert pbe_incar["ISMEAR"] == -5
assert "LHFCALC" not in pbe_incar
assert "HFSCREEN" not in pbe_incar
assert "PRECFOCK" not in pbe_incar

hse_incar = generator.kwargs["user_incar_settings"]
assert hse_incar["LHFCALC"] is True
assert hse_incar["AEXX"] == 0.25
assert hse_incar["HFSCREEN"] == 0.2
assert hse_incar["GGA"] == "PE"
assert hse_incar["ALGO"] == "Damped"
assert hse_incar["PRECFOCK"] == "Accurate"
assert hse_incar["ISMEAR"] == 0
assert hse_incar["SIGMA"] == 0.05
assert hse_incar["ENCUT"] == 620
assert hse_incar["NCORE"] == pbe_incar["NCORE"] == 4
assert hse_incar["ISPIN"] == 1
assert "MAGMOM" not in hse_incar

hse_user_ncore_incar = hse_user_ncore_generator.kwargs["user_incar_settings"]
assert hse_user_ncore_incar["NCORE"] == 4

hse_relax_incar = hse_relax_generator.kwargs["user_incar_settings"]
assert hse_relax_incar["LHFCALC"] is True
assert hse_relax_incar["AEXX"] == 0.25
assert hse_relax_incar["HFSCREEN"] == 0.2
assert hse_relax_incar["GGA"] == "PE"
assert hse_relax_incar["ALGO"] == "Damped"
assert hse_relax_incar["TIME"] == 0.4
assert hse_relax_incar["PRECFOCK"] == "Fast"
assert hse_relax_incar["ENCUT"] == 580
assert hse_relax_incar["ADDGRID"] is True
assert hse_relax_incar["EDIFF"] == 1e-6
assert hse_relax_incar["EDIFFG"] == -0.01
assert hse_relax_incar["LCHARG"] is False
assert hse_relax_incar["LWAVE"] is False
assert hse_relax_incar["NCORE"] == pbe_relax_incar["NCORE"] == 4
assert hse_relax_incar["ISPIN"] == 1
assert "MAGMOM" not in hse_relax_incar

with fake_atomate2_and_jobflow():
    spin_generator = build_static_input_set_generator(
        "Si2",
        theory=Theory.HSE06,
        modifiers={Modifier.SPIN_POLARIZED},
    )

assert spin_generator.kwargs["user_incar_settings"]["ISPIN"] == 2
assert spin_generator.kwargs["user_incar_settings"]["LHFCALC"] is True
assert spin_generator.kwargs["user_incar_settings"]["ISMEAR"] == 0
assert spin_generator.kwargs["user_incar_settings"]["NCORE"] == 4

with fake_atomate2_and_jobflow():
    generated_inputs = preview_generated_inputs("Si2", hse_static_spec)

assert "LHFCALC = True" in generated_inputs["incar"]
assert "HFSCREEN = 0.2" in generated_inputs["incar"]
assert "GGA = PE" in generated_inputs["incar"]
assert "PRECFOCK = Accurate" in generated_inputs["incar"]
assert "ISMEAR = 0" in generated_inputs["incar"]
assert "ISMEAR = -5" not in generated_inputs["incar"]
assert "NCORE = 4" in generated_inputs["incar"]
assert "ISPIN = 1" in generated_inputs["incar"]
assert generated_inputs["kpoints"] == "KPOINTS"
assert generated_inputs["poscar"] == "POSCAR Si2"

with fake_atomate2_and_jobflow():
    relax_generated_inputs = preview_generated_inputs("Si2", hse_relax_spec)

assert "LHFCALC = True" in relax_generated_inputs["incar"]
assert "HFSCREEN = 0.2" in relax_generated_inputs["incar"]
assert "GGA = PE" in relax_generated_inputs["incar"]
assert "PRECFOCK = Fast" in relax_generated_inputs["incar"]
assert "ENCUT = 580" in relax_generated_inputs["incar"]
assert "IBRION = 2" in relax_generated_inputs["incar"]
assert "ISIF = 3" in relax_generated_inputs["incar"]
assert "NSW = 99" in relax_generated_inputs["incar"]
assert "EDIFFG = -0.01" in relax_generated_inputs["incar"]
assert "NCORE = 4" in relax_generated_inputs["incar"]
assert "ISPIN = 1" in relax_generated_inputs["incar"]
assert relax_generated_inputs["kpoints"] == "KPOINTS"
assert relax_generated_inputs["poscar"] == "POSCAR Si2"

with fake_atomate2_and_jobflow():
    flow = build_atomate2_flow_for_spec(
        "Si2",
        hse_static_spec,
        label="Si-hse",
    )

assert flow.name == "Si-hse_hse_static"
assert [job.name for job in flow.jobs] == ["hse_static"]
assert flow.jobs[0].input_set_generator.kwargs["user_incar_settings"]["LHFCALC"] is True

with fake_atomate2_and_jobflow():
    relax_flow = build_atomate2_flow_for_spec(
        "Si2",
        hse_relax_spec,
        label="Si-hse-relax",
    )

assert relax_flow.name == "Si-hse-relax"
assert [job.name for job in relax_flow.jobs] == ["relax"]
assert (
    relax_flow.jobs[0]
    .input_set_generator.kwargs["user_incar_settings"]["PRECFOCK"]
    == "Fast"
)
assert (
    relax_flow.jobs[0]
    .input_set_generator.kwargs["user_incar_settings"]["LHFCALC"]
    is True
)

summary = summarize_workflow(flow, hse_static_spec)
assert summary["calculation_type"] == "Static Energy"
assert summary["calculation_plan"] == ["Static Energy"]
assert summary["theory"] == "hse06"
assert summary["theory_label"] == "HSE06"

relax_summary = summarize_workflow(relax_flow, hse_relax_spec)
assert relax_summary["calculation_type"] == "Geometry Optimisation"
assert relax_summary["calculation_plan"] == ["Geometry Optimisation"]
assert relax_summary["theory"] == "hse06"
assert relax_summary["theory_label"] == "HSE06"

print("hse06 theory smoke test passed")
