import os
import posixpath
import shlex

from backend.calculations.models import CalculationSpec, Modifier, Purpose, StageType, Theory
from backend.calculations.resources import (
    ALLOWED_CPU_COUNTS,
    ALLOWED_MEMORY_GB,
    ALLOWED_QUEUES,
    ExecutionResources,
    ncore_for_execution_resources,
    stage_allows_automatic_ncore,
)
from backend.calculations.registry import CalculationValidationError
from backend.parser import parse_structure
from backend.workflows import (
    apply_modifier_incar_settings,
    apply_dft_u_settings,
    apply_resource_incar_settings,
    apply_soc_magmom_settings,
    apply_stage_resource_incar_settings,
    apply_spin_settings,
    build_atomate2_flow_for_spec,
    incar_relax,
    incar_static,
    ksettings_for_modifiers,
    run_vasp_kwargs_for_modifiers,
    vasp_command_for_modifiers,
    vasp_executable_for_modifiers,
    validate_input_set_for_modifiers,
)


assert apply_spin_settings({}, spin_polarized=False) == {
    "ISPIN": 1,
}
assert apply_spin_settings({"ENCUT": 520}, spin_polarized=False) == {
    "ENCUT": 520,
    "ISPIN": 1,
}
assert apply_spin_settings({}, spin_polarized=True) == {
    "ISPIN": 2,
}
assert apply_spin_settings(
    {"LSORBIT": True},
    spin_polarized=False,
    non_collinear=True,
) == {
    "LSORBIT": True,
}
assert apply_spin_settings({"ISPIN": 1}, spin_polarized=True) == {
    "ISPIN": 2,
}
assert apply_spin_settings(
    {"MAGMOM": {"Fe": 5}},
    spin_polarized=True,
) == {
    "MAGMOM": {"Fe": 5},
    "ISPIN": 2,
}
assert apply_modifier_incar_settings(
    {},
    modifiers={Modifier.SPIN_POLARIZED},
)["ISPIN"] == 2
non_dft_u_settings = apply_modifier_incar_settings(
    {
        "LDAU": True,
        "LDAUTYPE": 2,
        "LDAUL": {"Ni": 2, "O": -1},
        "LDAUU": {"Ni": 6.2, "O": 0},
        "LDAUJ": {"Ni": 0, "O": 0},
        "LDAUPRINT": 1,
        "LMAXMIX": 4,
    },
    modifiers=(),
)
assert non_dft_u_settings["ISPIN"] == 1
assert "MAGMOM" not in non_dft_u_settings
for key in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT", "LMAXMIX"):
    assert non_dft_u_settings[key] is None

dft_u_settings = apply_modifier_incar_settings(
    {
        "LDAU": True,
        "LDAUTYPE": 2,
        "LDAUL": {"Ni": 2, "O": -1},
        "LDAUU": {"Ni": 6.2, "O": 0},
        "LDAUJ": {"Ni": 0, "O": 0},
        "LDAUPRINT": 1,
        "LMAXMIX": 4,
    },
    modifiers={Modifier.DFT_U},
)
assert dft_u_settings["LDAU"] is True
assert dft_u_settings["LDAUU"] == {"Ni": 6.2, "O": 0}
assert apply_dft_u_settings({"LDAU": True}, dft_u=False)["LDAU"] is None
assert ALLOWED_CPU_COUNTS == (24, 48, 72, 96, 120, 144, 168, 192)
assert ALLOWED_MEMORY_GB == (32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512)
assert ALLOWED_QUEUES == ("leeburton-pool",)
assert ExecutionResources().memory_gb == 128
assert ExecutionResources().queue == "leeburton-pool"
assert ncore_for_execution_resources(ExecutionResources(cpus=24)) == 8
assert ncore_for_execution_resources({"ntasks": 48}) == 8
try:
    ExecutionResources(cpus=25)
except CalculationValidationError as exc:
    assert "CPUs must be one of" in exc.message
else:
    raise AssertionError("Unsupported CPU counts should be rejected.")
try:
    ExecutionResources(memory_gb=100)
except CalculationValidationError as exc:
    assert "Memory must be one of" in exc.message
else:
    raise AssertionError("Unsupported memory amounts should be rejected.")
try:
    ExecutionResources(queue="debug; rm -rf /")
except CalculationValidationError as exc:
    assert "Queue must be one of" in exc.message
else:
    raise AssertionError("Unsupported queue values should be rejected.")
assert apply_resource_incar_settings(
    {"ENCUT": 520},
    resources=ExecutionResources(cpus=24),
) == {
    "ENCUT": 520,
    "NCORE": 8,
}
assert apply_resource_incar_settings(
    {"NCORE": 4},
    resources=ExecutionResources(cpus=48),
) == {
    "NCORE": 8,
}
assert apply_resource_incar_settings(
    {"NCORE": 4},
    resources=ExecutionResources(cpus=24),
    allow_ncore=False,
) == {}
assert stage_allows_automatic_ncore(StageType.RELAX)
assert stage_allows_automatic_ncore(StageType.STATIC)
assert stage_allows_automatic_ncore(StageType.DOS)
assert not stage_allows_automatic_ncore(StageType.BAND_STRUCTURE)
assert apply_stage_resource_incar_settings(
    {"ENCUT": 520},
    stage_type=StageType.STATIC,
    resources=ExecutionResources(cpus=24),
) == {
    "ENCUT": 520,
    "NCORE": 8,
}
assert apply_stage_resource_incar_settings(
    {"ENCUT": 520},
    stage_type=StageType.BAND_STRUCTURE,
    resources=ExecutionResources(cpus=24),
) == {
    "ENCUT": 520,
}
assert apply_stage_resource_incar_settings(
    {"NCORE": 4},
    stage_type=StageType.BAND_STRUCTURE,
    resources=ExecutionResources(cpus=24),
) == {
    "NCORE": 4,
}
assert apply_modifier_incar_settings(
    {"MAGMOM": {"Fe": 5}},
    modifiers={Modifier.SOC},
) == {
    "MAGMOM": {"Fe": 5},
    "LDAU": None,
    "LDAUTYPE": None,
    "LDAUL": None,
    "LDAUU": None,
    "LDAUJ": None,
    "LDAUPRINT": None,
    "LMAXMIX": None,
    "LSORBIT": True,
    "LNONCOLLINEAR": True,
    "ISYM": 0,
    "SAXIS": [0, 0, 1],
    "GGA_COMPAT": False,
    "LELF": None,
    "ISPIN": None,
}
assert apply_modifier_incar_settings(
    {"ISPIN": 2, "MAGMOM": {"Fe": 5}},
    modifiers={Modifier.SPIN_POLARIZED, Modifier.SOC},
) == {
    "MAGMOM": {"Fe": 5},
    "LDAU": None,
    "LDAUTYPE": None,
    "LDAUL": None,
    "LDAUU": None,
    "LDAUJ": None,
    "LDAUPRINT": None,
    "LMAXMIX": None,
    "LSORBIT": True,
    "LNONCOLLINEAR": True,
    "ISYM": 0,
    "SAXIS": [0, 0, 1],
    "GGA_COMPAT": False,
    "LELF": None,
    "ISPIN": None,
}
assert vasp_executable_for_modifiers({Modifier.SOC}) == "vasp_ncl"
assert vasp_executable_for_modifiers({Modifier.SPIN_POLARIZED}) == "vasp_std"
assert vasp_command_for_modifiers(
    {Modifier.SOC},
    base_command="mpirun -n $SLURM_NTASKS vasp_std",
) == "mpirun -n '$SLURM_NTASKS' vasp_ncl"
saved_slurm_ntasks = os.environ.get("SLURM_NTASKS")
try:
    os.environ["SLURM_NTASKS"] = "24"
    assert shlex.split(
        posixpath.expandvars(
            vasp_command_for_modifiers(
                {Modifier.SOC},
                base_command="mpirun -n $SLURM_NTASKS vasp_std",
            )
        )
    ) == ["mpirun", "-n", "24", "vasp_ncl"]
    os.environ["SLURM_NTASKS"] = "48"
    assert shlex.split(
        posixpath.expandvars(
            vasp_command_for_modifiers(
                {Modifier.SOC},
                base_command="mpirun -n $SLURM_NTASKS vasp_std",
            )
        )
    ) == ["mpirun", "-n", "48", "vasp_ncl"]
finally:
    if saved_slurm_ntasks is None:
        os.environ.pop("SLURM_NTASKS", None)
    else:
        os.environ["SLURM_NTASKS"] = saved_slurm_ntasks
assert run_vasp_kwargs_for_modifiers({Modifier.SOC})["vasp_cmd"].endswith("vasp_ncl")
assert run_vasp_kwargs_for_modifiers({Modifier.SOC})["vasp_job_kwargs"] == {"auto_gamma": False}
spin_run_vasp_kwargs = run_vasp_kwargs_for_modifiers({Modifier.SPIN_POLARIZED})
assert "vasp_cmd" not in spin_run_vasp_kwargs
assert "vasp_job_kwargs" not in spin_run_vasp_kwargs
assert spin_run_vasp_kwargs["handlers"]
assert all(
    type(handler).__name__ != "FrozenJobErrorHandler"
    for handler in spin_run_vasp_kwargs["handlers"]
)

soc_si = parse_structure(
    """Si
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
)
assert apply_soc_magmom_settings(
    {"LSORBIT": True, "LNONCOLLINEAR": True, "SAXIS": [0, 0, 1]},
    structure=soc_si,
    magnetic=True,
)["MAGMOM"] == {"Si": [0.0, 0.0, 0.6]}
assert apply_soc_magmom_settings(
    {"LSORBIT": True, "LNONCOLLINEAR": True, "SAXIS": [0, 0, 1]},
    structure=soc_si,
    magnetic=False,
)["MAGMOM"] == {"Si": [0.0, 0.0, 0.0]}
assert ksettings_for_modifiers(
    "structure",
    {"mode": "grid_density", "value": 1000},
    modifiers={Modifier.GAMMA_ONLY},
) == {"grid_density": 1.0}
assert ksettings_for_modifiers(
    "structure",
    {"mode": "grid_density", "value": 1000},
    modifiers=(),
) == {"grid_density": 1000.0}


class FakeInputSet:
    def __init__(self, incar):
        self.incar = incar


validate_input_set_for_modifiers(
    FakeInputSet({"LDAU": True, "LDAUU": [0, 5.3]}),
    spec=CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U}),
)
try:
    validate_input_set_for_modifiers(
        FakeInputSet({"LDAU": False, "LDAUU": [0, 0]}),
        spec=CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U}),
    )
except CalculationValidationError as exc:
    assert "DFT+U was requested" in exc.message
else:
    raise AssertionError("DFT+U should fail when no active U values are generated.")

non_spin_static = incar_static(apply_spin_settings({}, spin_polarized=False))
assert non_spin_static["ISPIN"] == 1
assert "MAGMOM" not in non_spin_static

spin_static = incar_static(apply_spin_settings({}, spin_polarized=True))
assert spin_static["ISPIN"] == 2
assert "MAGMOM" not in spin_static

non_spin_relax = incar_relax(apply_spin_settings({}, spin_polarized=False))
assert non_spin_relax["ISPIN"] == 1
assert "MAGMOM" not in non_spin_relax

spin_relax = incar_relax(apply_spin_settings({}, spin_polarized=True))
assert spin_relax["ISPIN"] == 2
assert "MAGMOM" not in spin_relax


import backend.workflows as workflows

calls = []
original_build_static_flow = workflows.build_static_flow
original_build_relax_flow = workflows.build_relax_flow


def fake_build_static_flow(structure, **kwargs):
    calls.append(("static", structure, kwargs))
    return {"flow": "static", "kwargs": kwargs}


def fake_build_relax_flow(structure, **kwargs):
    calls.append(("relax", structure, kwargs))
    return {"flow": "relax", "kwargs": kwargs}


workflows.build_static_flow = fake_build_static_flow
workflows.build_relax_flow = fake_build_relax_flow
try:
    build_atomate2_flow_for_spec(
        "structure",
        CalculationSpec(Purpose.STATIC, Theory.PBE),
    )
    build_atomate2_flow_for_spec(
        "structure",
        CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED}),
    )
    build_atomate2_flow_for_spec(
        "structure",
        CalculationSpec(Purpose.RELAX, Theory.PBE),
    )
    build_atomate2_flow_for_spec(
        "structure",
        CalculationSpec(Purpose.RELAX, Theory.PBE, {Modifier.SPIN_POLARIZED}),
    )
finally:
    workflows.build_static_flow = original_build_static_flow
    workflows.build_relax_flow = original_build_relax_flow

assert [call[2]["spin_polarized"] for call in calls] == [
    False,
    True,
    False,
    True,
]
assert [call[2]["modifiers"] for call in calls] == [
    frozenset(),
    frozenset({Modifier.SPIN_POLARIZED}),
    frozenset(),
    frozenset({Modifier.SPIN_POLARIZED}),
]
assert [call[2]["resources"] for call in calls] == [None, None, None, None]

print("spin modifier smoke test passed")
