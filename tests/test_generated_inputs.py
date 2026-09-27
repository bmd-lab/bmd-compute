import json

from backend.calculations.models import (
    CalculationSpec,
    Modifier,
    Purpose,
    StageSpec,
    StageType,
    Theory,
    WorkflowSpec,
)
from backend.calculations.resources import ALLOWED_MEMORY_GB, ALLOWED_QUEUES, ExecutionResources
from backend.calculations.registry import CalculationValidationError
from backend.config import DEFAULT_ACCOUNT, DEFAULT_PARTITION, DEFAULT_RESOURCES
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure, structure_from_spec
from backend.submission import build_sbatch_script, remote_preparation_file_groups
from backend.workflows import (
    build_atomate2_flow_from_spec,
    workflow_stage_artifact_policies,
)
from main import build_submission_state, build_workflow
from starlette.requests import Request


poscar = """Si
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

structure = parse_structure(poscar)


def incar_values(incar_text, key):
    prefix = f"{key} = "
    for line in incar_text.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return None


def magmom_components(incar_text):
    magmom = incar_values(incar_text, "MAGMOM")
    assert magmom is not None
    components = []
    for token in magmom.split():
        if "*" in token:
            count, value = token.split("*", 1)
            components.extend([float(value)] * int(count))
        else:
            components.append(float(token))
    return components


def magmom_component_count(incar_text):
    return len(magmom_components(incar_text))


def without_incar_key(incar_text, key):
    return [line for line in incar_text.splitlines() if not line.startswith(f"{key} = ")]


def workflow_spec_json(stage_type, *, theory="pbe", modifiers=None, recipe=None):
    return json.dumps(
        {
            "stages": [
                {
                    "stage_type": stage_type,
                    "theory": theory,
                    "modifiers": list(modifiers or []),
                    "label": None,
                    "options": {},
                }
            ],
            "label": None,
            "recipe": recipe,
        },
        sort_keys=True,
    )


def input_set_generator_from_job(job):
    if hasattr(job, "input_set_generator"):
        return job.input_set_generator

    bound_self = getattr(getattr(job, "function", None), "__self__", None)
    if hasattr(bound_self, "input_set_generator"):
        return bound_self.input_set_generator

    for value in getattr(job, "function_args", ()) or ():
        if hasattr(value, "input_set_generator"):
            return value.input_set_generator

    for value in (getattr(job, "function_kwargs", {}) or {}).values():
        if hasattr(value, "input_set_generator"):
            return value.input_set_generator

    raise AssertionError(f"Could not find input_set_generator on {job!r}")


def reconstructed_runtime_stage_incar(structure_text, workflow_spec, stage_index):
    _, _, _, submission_spec = build_submission_state(
        structure_text=structure_text,
        fmt="poscar",
        workflow_spec=workflow_spec,
        timestamp="20260814-120000",
    )
    runtime_structure = structure_from_spec(submission_spec["flow_spec"]["structure"])
    flow = build_atomate2_flow_from_spec(
        runtime_structure,
        submission_spec["flow_spec"],
        run_name=submission_spec["run_name"],
        resources=submission_spec["resources"],
    )
    generator = input_set_generator_from_job(flow.jobs[stage_index]).get_input_set(
        runtime_structure,
        potcar_spec=True,
    )
    return str(generator.incar)


static_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE),
    potcar_functional="PBE_64",
)
assert "ENCUT = 620.0" in static_preview["incar"]
assert "NCORE = 8" in static_preview["incar"]
assert "ISPIN = 1" in static_preview["incar"]
assert "MAGMOM =" not in static_preview["incar"]
assert "LELF = True" in static_preview["incar"]
assert "LWAVE = False" in static_preview["incar"]
assert "GGA_COMPAT" not in static_preview["incar"]
assert "Gamma" in static_preview["kpoints"]
assert "Si2" in static_preview["poscar"]
assert "direct" in static_preview["poscar"].lower()
assert "0.2500000000000000    0.2500000000000000    0.2500000000000000 Si" in static_preview["poscar"]

spin_static_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED}),
    potcar_functional="PBE_64",
)
assert "ISPIN = 2" in spin_static_preview["incar"]
assert "MAGMOM = 2*0.6" in spin_static_preview["incar"]
assert spin_static_preview["kpoints"] == static_preview["kpoints"]
assert spin_static_preview["poscar"] == static_preview["poscar"]

sns2_poscar = """SnS2
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
sns2_structure = parse_structure(sns2_poscar)
sns2_static_preview = preview_generated_inputs(
    sns2_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE),
    potcar_functional="PBE_64",
)
sns2_spin_static_preview = preview_generated_inputs(
    sns2_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED}),
    potcar_functional="PBE_64",
)
assert "ISPIN = 1" in sns2_static_preview["incar"]
assert "MAGMOM =" not in sns2_static_preview["incar"]
assert "ISPIN = 2" in sns2_spin_static_preview["incar"]
assert "MAGMOM = 3*0.6" in sns2_spin_static_preview["incar"]

high_cpu_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE),
    resources=ExecutionResources(cpus=48),
    potcar_functional="PBE_64",
)
assert high_cpu_preview["incar"] == static_preview["incar"]

soc_static_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SOC}),
    potcar_functional="PBE_64",
)
assert soc_static_preview["vasp_executable"] == "vasp_ncl"
assert "LSORBIT = True" in soc_static_preview["incar"]
assert "LNONCOLLINEAR = True" in soc_static_preview["incar"]
assert "ISPIN =" not in soc_static_preview["incar"]
assert "ISYM = 0" in soc_static_preview["incar"]
assert "GGA_COMPAT = False" in soc_static_preview["incar"]
assert "LELF =" not in soc_static_preview["incar"]
assert "SAXIS = 0 0 1" in soc_static_preview["incar"]
# Non-magnetic SOC (no Spin Polarised) starts from zero vector moments.
assert magmom_components(soc_static_preview["incar"]) == [0.0] * (3 * len(structure))
assert "NCORE = 8" in soc_static_preview["incar"]
assert "LWAVE = False" in soc_static_preview["incar"]
assert magmom_component_count(soc_static_preview["incar"]) == 3 * len(structure)
for dft_u_key in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT", "LMAXMIX"):
    assert f"{dft_u_key} =" not in soc_static_preview["incar"]

hse_soc_static_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.STATIC, Theory.HSE06, {Modifier.SOC}),
    potcar_functional="PBE_64",
)
assert hse_soc_static_preview["vasp_executable"] == "vasp_ncl"
assert "LHFCALC = True" in hse_soc_static_preview["incar"]
assert "AEXX = 0.25" in hse_soc_static_preview["incar"]
assert "HFSCREEN = 0.2" in hse_soc_static_preview["incar"]
assert "GGA = Pe" in hse_soc_static_preview["incar"]
assert "ALGO = Damped" in hse_soc_static_preview["incar"]
assert "PRECFOCK = Accurate" in hse_soc_static_preview["incar"]
assert "TIME = 0.4" in hse_soc_static_preview["incar"]
assert "ISMEAR = 0" in hse_soc_static_preview["incar"]
assert "LSORBIT = True" in hse_soc_static_preview["incar"]
assert "LNONCOLLINEAR = True" in hse_soc_static_preview["incar"]
assert "ISPIN =" not in hse_soc_static_preview["incar"]
assert "LELF =" not in hse_soc_static_preview["incar"]
assert "ISYM = 0" in hse_soc_static_preview["incar"]
assert "GGA_COMPAT = False" in hse_soc_static_preview["incar"]
assert "SAXIS = 0 0 1" in hse_soc_static_preview["incar"]
assert magmom_components(hse_soc_static_preview["incar"]) == [0.0] * (3 * len(structure))
assert "NCORE = 8" in hse_soc_static_preview["incar"]
assert "LWAVE = False" in hse_soc_static_preview["incar"]
assert magmom_component_count(hse_soc_static_preview["incar"]) == 3 * len(structure)

hse_spin_soc_static_preview = preview_generated_inputs(
    structure,
    CalculationSpec(
        Purpose.STATIC,
        Theory.HSE06,
        {Modifier.SPIN_POLARIZED, Modifier.SOC},
    ),
    potcar_functional="PBE_64",
)
assert hse_spin_soc_static_preview["vasp_executable"] == "vasp_ncl"
# Spin Polarised + SOC keeps the magnetic defaults along SAXIS; nothing else changes.
assert magmom_components(hse_spin_soc_static_preview["incar"]) == [0.0, 0.0, 0.6] * len(structure)
assert "ISPIN =" not in hse_spin_soc_static_preview["incar"]
assert without_incar_key(hse_spin_soc_static_preview["incar"], "MAGMOM") == without_incar_key(
    hse_soc_static_preview["incar"],
    "MAGMOM",
)

hse_soc_workflow = WorkflowSpec(
    [StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC})],
    recipe="custom",
)
hse_soc_runtime_incar = reconstructed_runtime_stage_incar(
    poscar,
    hse_soc_workflow,
    stage_index=0,
)
assert incar_values(hse_soc_runtime_incar, "MAGMOM") == incar_values(
    hse_soc_static_preview["incar"],
    "MAGMOM",
)
assert magmom_component_count(hse_soc_runtime_incar) == 3 * len(structure)
assert "LHFCALC = True" in hse_soc_runtime_incar
assert "PRECFOCK = Accurate" in hse_soc_runtime_incar
assert "LSORBIT = True" in hse_soc_runtime_incar
assert "LNONCOLLINEAR = True" in hse_soc_runtime_incar
assert "ISPIN =" not in hse_soc_runtime_incar
assert "LELF =" not in hse_soc_runtime_incar

static_to_soc_workflow = WorkflowSpec(
    [
        StageSpec(StageType.STATIC, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC}),
    ],
    recipe="custom",
)
static_to_soc_preview = preview_generated_inputs(
    structure,
    static_to_soc_workflow,
    potcar_functional="PBE_64",
)
static_precursor_section, static_soc_section = static_to_soc_preview["incar"].split("\n\n", 1)
assert "# Stage 1 - Static Energy (PBE)" in static_precursor_section
assert "# VASP executable - vasp_std" in static_precursor_section
assert "ISPIN = 1" in static_precursor_section
assert "MAGMOM =" not in static_precursor_section
assert "LWAVE = False" in static_precursor_section
assert "# Stage 2 - Static Energy (PBE)" in static_soc_section
assert "# VASP executable - vasp_ncl" in static_soc_section
assert "LWAVE = False" in static_soc_section
assert "LSORBIT = True" in static_soc_section
assert "ISPIN =" not in static_soc_section
assert magmom_components(static_soc_section) == [0.0] * (3 * len(structure))
assert workflow_stage_artifact_policies(static_to_soc_workflow) == (
    {"write_wavecar": False, "copy_from_previous": ()},
    {"write_wavecar": False, "copy_from_previous": ()},
)

static_soc_runtime_incar = reconstructed_runtime_stage_incar(
    poscar,
    static_to_soc_workflow,
    stage_index=1,
)
assert incar_values(static_soc_runtime_incar, "MAGMOM") == incar_values(
    static_soc_section,
    "MAGMOM",
)
assert magmom_component_count(static_soc_runtime_incar) == 3 * len(structure)
assert "LSORBIT = True" in static_soc_runtime_incar
assert "LNONCOLLINEAR = True" in static_soc_runtime_incar
assert "ISPIN =" not in static_soc_runtime_incar
assert "LELF =" not in static_soc_runtime_incar

gamma_static_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.GAMMA_ONLY}),
    potcar_functional="PBE_64",
)
assert "Gamma" in gamma_static_preview["kpoints"]
assert any(
    line.split() == ["1", "1", "1"]
    for line in gamma_static_preview["kpoints"].splitlines()
)
assert gamma_static_preview["poscar"] == static_preview["poscar"]

try:
    preview_generated_inputs(
        structure,
        CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U}),
        potcar_functional="PBE_64",
    )
except CalculationValidationError as exc:
    assert "DFT+U was requested" in exc.message
else:
    raise AssertionError("DFT+U should fail clearly when no U values are available.")

nio_poscar = """NiO
4.3
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Ni O
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
nio_structure = parse_structure(nio_poscar)
non_dft_u_nio_preview = preview_generated_inputs(
    nio_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE),
    potcar_functional="PBE_64",
)
for dft_u_key in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT", "LMAXMIX"):
    assert f"{dft_u_key} =" not in non_dft_u_nio_preview["incar"]
dft_u_preview = preview_generated_inputs(
    nio_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U}),
    potcar_functional="PBE_64",
)
assert "LDAU = True" in dft_u_preview["incar"]
assert "LDAUU" in dft_u_preview["incar"]

fe2o3_poscar = """Fe2O3
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
fe2o3_structure = parse_structure(fe2o3_poscar)
fe2o3_plain_preview = preview_generated_inputs(
    fe2o3_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE),
    potcar_functional="PBE_64",
)
assert "ISPIN = 1" in fe2o3_plain_preview["incar"]
assert "MAGMOM =" not in fe2o3_plain_preview["incar"]
for dft_u_key in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT", "LMAXMIX"):
    assert f"{dft_u_key} =" not in fe2o3_plain_preview["incar"]
fe2o3_dft_u_preview = preview_generated_inputs(
    fe2o3_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U}),
    potcar_functional="PBE_64",
)
assert "ISPIN = 1" in fe2o3_dft_u_preview["incar"]
assert "MAGMOM =" not in fe2o3_dft_u_preview["incar"]
assert "LDAU = True" in fe2o3_dft_u_preview["incar"]
assert "LDAUU = 5.3 0" in fe2o3_dft_u_preview["incar"]
fe2o3_spin_dft_u_preview = preview_generated_inputs(
    fe2o3_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U, Modifier.SPIN_POLARIZED}),
    potcar_functional="PBE_64",
)
assert "ISPIN = 2" in fe2o3_spin_dft_u_preview["incar"]
assert "MAGMOM = 2*5.0 3*0.6" in fe2o3_spin_dft_u_preview["incar"]
assert "LDAU = True" in fe2o3_spin_dft_u_preview["incar"]
assert "LDAUU = 5.3 0" in fe2o3_spin_dft_u_preview["incar"]
fe2o3_soc_preview = preview_generated_inputs(
    fe2o3_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SOC}),
    potcar_functional="PBE_64",
)
assert fe2o3_soc_preview["vasp_executable"] == "vasp_ncl"
assert "LSORBIT = True" in fe2o3_soc_preview["incar"]
assert "LNONCOLLINEAR = True" in fe2o3_soc_preview["incar"]
assert "ISPIN =" not in fe2o3_soc_preview["incar"]
assert "GGA_COMPAT = False" in fe2o3_soc_preview["incar"]
assert "LELF =" not in fe2o3_soc_preview["incar"]
assert "MAGMOM = 0.0 0.0 5.0 0.0 0.0 5.0 0.0 0.0 0.6 0.0 0.0 0.6 0.0 0.0 0.6" in fe2o3_soc_preview["incar"]
assert magmom_component_count(fe2o3_soc_preview["incar"]) == 3 * len(fe2o3_structure)
assert "NCORE = 8" in fe2o3_soc_preview["incar"]
for dft_u_key in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT", "LMAXMIX"):
    assert f"{dft_u_key} =" not in fe2o3_soc_preview["incar"]
fe2o3_spin_soc_preview = preview_generated_inputs(
    fe2o3_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.SPIN_POLARIZED, Modifier.SOC}),
    potcar_functional="PBE_64",
)
assert fe2o3_spin_soc_preview["incar"] == fe2o3_soc_preview["incar"]
fe2o3_dft_u_soc_preview = preview_generated_inputs(
    fe2o3_structure,
    CalculationSpec(Purpose.STATIC, Theory.PBE, {Modifier.DFT_U, Modifier.SOC}),
    potcar_functional="PBE_64",
)
assert "LSORBIT = True" in fe2o3_dft_u_soc_preview["incar"]
assert "ISPIN =" not in fe2o3_dft_u_soc_preview["incar"]
assert "GGA_COMPAT = False" in fe2o3_dft_u_soc_preview["incar"]
assert "LELF =" not in fe2o3_dft_u_soc_preview["incar"]
assert "MAGMOM = 0.0 0.0 5.0 0.0 0.0 5.0 0.0 0.0 0.6 0.0 0.0 0.6 0.0 0.0 0.6" in fe2o3_dft_u_soc_preview["incar"]
assert magmom_component_count(fe2o3_dft_u_soc_preview["incar"]) == 3 * len(fe2o3_structure)
assert "NCORE = 8" in fe2o3_dft_u_soc_preview["incar"]
assert "LDAU = True" in fe2o3_dft_u_soc_preview["incar"]
assert "LDAUU = 5.3 0" in fe2o3_dft_u_soc_preview["incar"]

fe12o18_structure = fe2o3_structure.copy()
fe12o18_structure.make_supercell([2, 3, 1])
from pymatgen.io.vasp.inputs import Poscar

fe12o18_poscar = str(Poscar(fe12o18_structure))
fe12o18_static_to_soc_preview = preview_generated_inputs(
    fe12o18_structure,
    static_to_soc_workflow,
    potcar_functional="PBE_64",
)
_, fe12o18_soc_section = fe12o18_static_to_soc_preview["incar"].split("\n\n", 1)
fe12o18_runtime_incar = reconstructed_runtime_stage_incar(
    fe12o18_poscar,
    static_to_soc_workflow,
    stage_index=1,
)
assert "MAGMOM =" in fe12o18_soc_section
assert incar_values(fe12o18_runtime_incar, "MAGMOM") == incar_values(
    fe12o18_soc_section,
    "MAGMOM",
)
assert magmom_component_count(fe12o18_soc_section) == 3 * len(fe12o18_structure) == 90
assert magmom_component_count(fe12o18_runtime_incar) == 90

relax_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.RELAX, Theory.PBE),
    potcar_functional="PBE_64",
)
assert "ENCUT = 580.0" in relax_preview["incar"]
assert "ISIF = 3" in relax_preview["incar"]
assert "ISPIN = 1" in relax_preview["incar"]
assert "MAGMOM =" not in relax_preview["incar"]

band_preview = preview_generated_inputs(
    structure,
    CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE),
    resources=ExecutionResources(cpus=24),
    potcar_functional="PBE_64",
)
band_relax_section, band_static_and_path = band_preview["incar"].split("\n\n", 1)
band_static_section, band_section = band_static_and_path.split("\n\n", 1)
assert "# Stage 1 - Geometry Optimisation" in band_relax_section
assert "# Stage 2 - Static Energy" in band_static_section
assert "# Stage 3 - Band Structure" in band_section
assert "NCORE = 8" in band_relax_section
assert "NCORE = 8" in band_static_section
assert "ISPIN = 1" in band_relax_section
assert "ISPIN = 1" in band_static_section
assert "ISPIN = 1" in band_section
assert "NCORE" not in band_section

summary, calculation, generated_inputs, submission_spec = build_submission_state(
    structure_text=poscar,
    fmt="poscar",
    calculation_spec=CalculationSpec(Purpose.STATIC, Theory.PBE),
    timestamp="20260728-120000",
)
assert summary["formula"] == "Si2"
assert calculation["calculation_type"] == "Static Energy"
assert generated_inputs["incar"] == static_preview["incar"]
assert "\r" not in generated_inputs["slurm_script"]
assert generated_inputs["slurm_script"].startswith("#!/bin/bash\n")
assert f"#SBATCH -p {DEFAULT_PARTITION}" in generated_inputs["slurm_script"]
assert f"#SBATCH --account={DEFAULT_ACCOUNT}" in generated_inputs["slurm_script"]
assert "#SBATCH -J vasp_run_static" in generated_inputs["slurm_script"]
assert f"#SBATCH --nodes={DEFAULT_RESOURCES['nodes']}" in generated_inputs["slurm_script"]
assert f"#SBATCH --ntasks={DEFAULT_RESOURCES['ntasks']}" in generated_inputs["slurm_script"]
assert f"#SBATCH --mem={DEFAULT_RESOURCES['mem_gb']}G" in generated_inputs["slurm_script"]
assert f"#SBATCH --time={DEFAULT_RESOURCES['walltime']}" in generated_inputs["slurm_script"]
assert "ulimit -s 81920" in generated_inputs["slurm_script"]
assert "module load intel/rocky8-oneAPI-2023" in generated_inputs["slurm_script"]
assert "module load vasp/rocky8-intel-6.4.1" in generated_inputs["slurm_script"]
assert f"{submission_spec['runner']['python']} -u run_job.py" not in generated_inputs["slurm_script"]
assert "\nmpirun -n $SLURM_NTASKS vasp_std\n" in generated_inputs["slurm_script"]
assert generated_inputs["exact_slurm_script"] == build_sbatch_script(submission_spec)
uploaded_script = next(
    item["text"]
    for group in remote_preparation_file_groups(submission_spec)
    if group["step"] == "Submission script written"
    for item in group["files"]
)
assert generated_inputs["exact_slurm_script"] == uploaded_script
assert submission_spec["flow_spec"]["calculation_spec"]["purpose"] == "static"
assert submission_spec["flow_spec"]["execution_resources"]["ntasks"] == 24
assert submission_spec["resources"]["ntasks"] == 24

request = Request(
    {
        "type": "http",
        "method": "POST",
        "path": "/build-calculation",
        "headers": [],
    }
)
response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=None,
    cpus=None,
    memory_gb=None,
    walltime=None,
    queue=None,
    workflow_spec_json=workflow_spec_json("static"),
    workflow=None,
    method=None,
)
assert response.status_code == 200
assert response.context["generated_inputs"]["incar"] == static_preview["incar"]
assert response.context["generated_inputs"]["kpoints"] == static_preview["kpoints"]
assert response.context["generated_inputs"]["poscar"] == static_preview["poscar"]
assert response.context["generated_inputs"]["exact_slurm_script"] == build_sbatch_script(
    response.context["submission_spec"]
)
assert "mpirun -n $SLURM_NTASKS vasp_std" in response.context["generated_inputs"][
    "slurm_script"
]
assert response.context["submission_spec"]["flow_spec"]["calculation_spec"] == {
    "purpose": "static",
    "theory": "pbe",
    "modifiers": [],
    "label": None,
}
assert response.context["selected_resources"]["cpus"] == 24

resource_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=None,
    cpus="48",
    memory_gb="256",
    walltime="12:00:00",
    queue=DEFAULT_PARTITION,
    workflow_spec_json=workflow_spec_json("static"),
    workflow=None,
    method=None,
)
assert resource_response.status_code == 200
assert resource_response.context["generated_inputs"]["incar"] == static_preview["incar"]
assert resource_response.context["selected_resources"] == {
    "nodes": 1,
    "cpus": 48,
    "allowed_cpu_counts": [24, 48, 72, 96, 120, 144, 168, 192],
    "memory_gb": 256,
    "allowed_memory_gb": list(ALLOWED_MEMORY_GB),
    "walltime": "12:00:00",
    "queue": DEFAULT_PARTITION,
    "allowed_queues": list(ALLOWED_QUEUES),
}
assert resource_response.context["submission_spec"]["resources"]["ntasks"] == 48
assert resource_response.context["submission_spec"]["resources"]["mem_gb"] == 256
assert resource_response.context["submission_spec"]["cluster"]["partition"] == DEFAULT_PARTITION
assert resource_response.context["submission_spec"]["cluster"]["account"] == DEFAULT_ACCOUNT
assert resource_response.context["generated_inputs"]["exact_slurm_script"] == build_sbatch_script(
    resource_response.context["submission_spec"]
)
assert f"#SBATCH -p {DEFAULT_PARTITION}" in resource_response.context["generated_inputs"]["slurm_script"]
assert f"#SBATCH --account={DEFAULT_ACCOUNT}" in resource_response.context["generated_inputs"]["slurm_script"]
assert "#SBATCH -J vasp_run_static" in resource_response.context["generated_inputs"]["slurm_script"]
assert "#SBATCH --nodes=1" in resource_response.context["generated_inputs"]["slurm_script"]
assert "#SBATCH --ntasks=48" in resource_response.context["generated_inputs"]["slurm_script"]
assert "#SBATCH --mem=256G" in resource_response.context["generated_inputs"]["slurm_script"]
assert "#SBATCH --time=12:00:00" in resource_response.context["generated_inputs"]["slurm_script"]

invalid_memory_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=None,
    cpus="48",
    memory_gb="100",
    walltime="12:00:00",
    queue=DEFAULT_PARTITION,
    workflow_spec_json=workflow_spec_json("static"),
    workflow=None,
    method=None,
)
assert invalid_memory_response.status_code == 400
assert "Memory must be one of" in invalid_memory_response.context["calculation_error"]["message"]

invalid_queue_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=None,
    cpus="48",
    memory_gb="256",
    walltime="12:00:00",
    queue="debug; rm -rf /",
    workflow_spec_json=workflow_spec_json("static"),
    workflow=None,
    method=None,
)
assert invalid_queue_response.status_code == 400
assert "Queue must be one of" in invalid_queue_response.context["calculation_error"]["message"]

spin_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=["spin_polarized"],
    cpus=None,
    memory_gb=None,
    walltime=None,
    queue=None,
    workflow_spec_json=workflow_spec_json("static", modifiers=["spin_polarized"]),
    workflow=None,
    method=None,
)
assert spin_response.status_code == 200
assert "ISPIN = 2" in spin_response.context["generated_inputs"]["incar"]
assert "MAGMOM = 2*0.6" in spin_response.context["generated_inputs"]["incar"]

soc_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=["soc"],
    cpus=None,
    memory_gb=None,
    walltime=None,
    queue=None,
    workflow_spec_json=workflow_spec_json("static", modifiers=["soc"]),
    workflow=None,
    method=None,
)
assert soc_response.status_code == 200
assert soc_response.context["generated_inputs"]["vasp_executable"] == "vasp_ncl"
assert "LSORBIT = True" in soc_response.context["generated_inputs"]["incar"]
assert "ISPIN =" not in soc_response.context["generated_inputs"]["incar"]
assert "GGA_COMPAT = False" in soc_response.context["generated_inputs"]["incar"]
assert "LELF =" not in soc_response.context["generated_inputs"]["incar"]
assert magmom_components(soc_response.context["generated_inputs"]["incar"]) == [0.0] * 6

gamma_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=["gamma_only"],
    cpus=None,
    memory_gb=None,
    walltime=None,
    queue=None,
    workflow_spec_json=workflow_spec_json("static", modifiers=["gamma_only"]),
    workflow=None,
    method=None,
)
assert gamma_response.status_code == 200
assert any(
    line.split() == ["1", "1", "1"]
    for line in gamma_response.context["generated_inputs"]["kpoints"].splitlines()
)

dft_u_error_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="static",
    theory="pbe",
    modifiers=["dft_u"],
    cpus=None,
    memory_gb=None,
    walltime=None,
    queue=None,
    workflow_spec_json=workflow_spec_json("static", modifiers=["dft_u"]),
    workflow=None,
    method=None,
)
assert dft_u_error_response.status_code == 400
assert "DFT+U was requested" in dft_u_error_response.context["calculation_error"]["message"]

relax_response = build_workflow(
    request,
    structure=poscar,
    fmt="poscar",
    purpose="relax",
    theory="pbe",
    modifiers=None,
    cpus=None,
    memory_gb=None,
    walltime=None,
    queue=None,
    workflow_spec_json=workflow_spec_json("relax"),
    workflow=None,
    method=None,
)
assert relax_response.status_code == 200
assert relax_response.context["calculation"]["calculation_type"] == "Geometry Optimisation"
assert "ENCUT = 580.0" in relax_response.context["generated_inputs"]["incar"]
assert "ISIF = 3" in relax_response.context["generated_inputs"]["incar"]
assert relax_response.context["submission_spec"]["flow_spec"]["calculation_spec"] == {
    "purpose": "relax",
    "theory": "pbe",
    "modifiers": [],
    "label": None,
}

template_source = open("templates/index.html", encoding="utf-8").read()
for heading in (
    "Calculation Definition",
    "Scientific Specification",
    "Execution Resources",
    "Calculation Summary",
    "Generated Inputs",
    "Execution Summary",
    "Ready for Submission",
):
    assert heading in template_source
assert "Computational Resources" not in template_source
assert "Submission Preview" not in template_source
assert '<div class="advanced-options-field">' in template_source
assert "<label>Advanced Options</label>" in template_source
assert "<summary>Advanced Options</summary>" not in template_source
assert '<summary aria-label="Toggle advanced options"></summary>' in template_source
assert '<div class="callout-title">Calculation Plan</div>' in template_source
assert '<div class="callout-title">Jobs</div>' not in template_source
assert "calculation.job_names" not in template_source
for field_name in ("cpus", "memory_gb", "walltime", "queue"):
    assert f'name="{field_name}"' in template_source
calculation_definition = template_source[
    template_source.index('<form\n                id="calculation-review-form"'):
]
resource_panel = calculation_definition[
    calculation_definition.index("<h3>Execution Resources</h3>"):
    calculation_definition.index("<h3>Scientific Specification</h3>")
]
queue_label_index = resource_panel.index("<label>Queue</label>")
queue_select_index = resource_panel.index('<select name="queue">')
queue_select_end = resource_panel.index("</select>", queue_select_index)
queue_select_block = resource_panel[queue_select_index:queue_select_end]
assert queue_label_index < queue_select_index
assert "selected_resources.allowed_queues" in queue_select_block
assert 'value="{{ queue }}"' in queue_select_block
assert "{% if selected_resources.queue == queue %}selected{% endif %}" in queue_select_block
assert 'name="account"' not in template_source
assert "<label>Account</label>" not in template_source
assert "submission_spec.cluster.account" not in template_source
assert "generated_inputs.submission_summary.resources.account" in template_source
assert "input-tab-poscar" in template_source
assert "tab-panel-poscar" in template_source
assert "generated_inputs.poscar" in template_source
assert "input-tab-slurm" in template_source
assert "tab-panel-slurm" in template_source
assert "generated_inputs.slurm_script" in template_source
assert "generated_inputs.exact_slurm_script" in template_source
assert "SLURM Script" in template_source
assert template_source.count("data-copy-input") == 6
assert template_source.count('class="input-copy-button" data-copy-input') == 5
assert "navigator.clipboard.writeText" in template_source
assert "\\u2713 Copied" in template_source
assert "Copy failed" in template_source
assert "input-tab-potcar" not in template_source
assert "tab-panel-potcar" not in template_source
assert "generated_inputs.potcar" not in template_source
assert "POTCAR Species" in template_source
assert 'id="calculation-review-form"' in template_source
assert "data-auto-rebuild" not in template_source
assert "requestSubmit" not in template_source
assert "data-current-calculation-action" in template_source
assert "Ionic Convergence" in template_source

print("generated inputs smoke test passed")


def test_generated_inputs_smoke_module_loaded():
    assert response.status_code == 200
