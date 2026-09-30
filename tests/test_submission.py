import hashlib

from backend.config import (
    DEFAULT_ACCOUNT,
    DEFAULT_FLOWS_DIR,
    DEFAULT_JOBFLOW_CONFIG_FILE,
    DEFAULT_LOGS_DIR,
    DEFAULT_PARTITION,
    DEFAULT_POTCAR_DIR,
    DEFAULT_REMOTE_PYTHON,
    DEFAULT_REMOTE_WORKSPACE_ROOT,
    DEFAULT_RESOURCES,
    DEFAULT_SHARED_POTCAR_ROOT,
    DEFAULT_VASP_CMD,
    MODULES,
    POTCAR_LINK_MAP,
)
from backend.calculations.registry import CalculationValidationError
from backend.submission import (
    build_backend_module_sources,
    build_execution_module_source,
    build_run_job_script,
    build_sbatch_script,
    build_submission_script_artifact,
    create_submission_spec,
)


flow_spec = {
    "workflow": "static",
    "potcar_functional": "PBE_64",
    "kpoints": None,
    "incar": {},
    "structure": {"type": "parsed"},
}


class FakeStructure:
    types_of_species = ["Ti", "O"]


def _rutile_tio2():
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "P4_2/mnm",
        Lattice.tetragonal(4.594, 2.959),
        ["Ti", "O"],
        [[0, 0, 0], [0.305, 0.305, 0]],
    )


spec = create_submission_spec(
    flow_spec,
    label="Si static!",
    timestamp="20260629-120000",
    env={},
)

assert spec["status"] == "pending"
assert spec["label"] == "Si-static"
assert spec["run_name"] == "Si-static-20260629-120000"
assert spec["paths"]["run_dir"] == f"{DEFAULT_FLOWS_DIR}/Si-static-20260629-120000"
assert DEFAULT_FLOWS_DIR == f"{DEFAULT_REMOTE_WORKSPACE_ROOT}/flows"
assert spec["paths"]["run_dir"].startswith("/bmd-db/guest/flows/")
assert spec["paths"]["log_out"] == f"{DEFAULT_LOGS_DIR}/Si-static-20260629-120000.out"
assert spec["paths"]["log_err"] == f"{DEFAULT_LOGS_DIR}/Si-static-20260629-120000.err"
assert spec["paths"]["slurm_out"] == f"{DEFAULT_LOGS_DIR}/Si-static-20260629-120000.slurm.out"
assert spec["paths"]["slurm_err"] == f"{DEFAULT_LOGS_DIR}/Si-static-20260629-120000.slurm.err"
assert spec["paths"]["stage_dirs"] == {}
assert spec["paths"]["result_dir"] == spec["paths"]["run_dir"]
assert spec["cluster"]["partition"] == DEFAULT_PARTITION
assert spec["cluster"]["account"] == DEFAULT_ACCOUNT
assert spec["resources"] == DEFAULT_RESOURCES
assert spec["environment"]["VASP_CMD"] == DEFAULT_VASP_CMD
assert spec["environment"]["JOBFLOW_CONFIG_FILE"] == DEFAULT_JOBFLOW_CONFIG_FILE
assert spec["environment"]["PMG_VASP_PSP_DIR"] == DEFAULT_POTCAR_DIR
assert spec["environment"]["PMG_VASP_PSP_DIR"] == DEFAULT_SHARED_POTCAR_ROOT
assert spec["paths"]["potcars_dir"] == DEFAULT_SHARED_POTCAR_ROOT
assert "/bmd-db/guest/potcars" not in spec["environment"]["PMG_VASP_PSP_DIR"]
assert DEFAULT_SHARED_POTCAR_ROOT not in spec["paths"]["directories_to_prepare"]
assert f"{DEFAULT_SHARED_POTCAR_ROOT}/PBE_64" not in spec["paths"]["directories_to_prepare"]
assert spec["environment"]["CUSTODIAN_NO_GZIP"] == "1"
assert spec["environment"]["ATOMATE2_VASP_ZIP_FILES"] == "False"
assert spec["modules"]["load"] == MODULES
assert spec["runner"]["python"] == DEFAULT_REMOTE_PYTHON
assert spec["runner"]["script_name"] == "run_job.py"
assert spec["runner"]["submission_spec_name"] == "submission.json"
assert spec["runner"]["backend_package_dir"] == "backend"
assert spec["runner"]["execution_module_name"] == "execution.py"
assert spec["potcar"]["repository"] == "shared"
assert spec["potcar"]["target"] == f"{DEFAULT_POTCAR_DIR}/PBE_64"
assert spec["potcar"]["symlink_targets"] == []
assert spec["potcar"]["species"] == []
assert spec["flow_spec"]["calculation_spec"] == {
    "purpose": "static",
    "theory": "pbe",
    "modifiers": [],
    "label": None,
}
assert spec["preflight"]["requires_remote_structure_check"] is False
assert spec["submission"]["ready"] is True
assert spec["submission"]["submitted"] is False

spin_flow_spec = {
    **flow_spec,
    "calculation_spec": {
        "purpose": "static",
        "theory": "pbe",
        "modifiers": ["spin_polarized"],
    },
}
spin_spec = create_submission_spec(
    spin_flow_spec,
    label="Si spin static!",
    timestamp="20260629-120000",
    env={},
)
assert spin_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "static",
    "theory": "pbe",
    "modifiers": ["spin_polarized"],
    "label": None,
}

double_relax_flow_spec = {
    **flow_spec,
    "workflow": "double_relax",
    "calculation_spec": {
        "purpose": "double_relax",
        "theory": "pbe",
        "modifiers": [],
    },
}
double_relax_spec = create_submission_spec(
    double_relax_flow_spec,
    label="Si double relax!",
    timestamp="20260629-120000",
    env={},
)
double_relax_run_dir = f"{DEFAULT_FLOWS_DIR}/Si-double-relax-20260629-120000"
assert double_relax_spec["paths"]["run_dir"] == double_relax_run_dir
assert double_relax_spec["paths"]["stage_dirs"] == {
    "relax_01": f"{double_relax_run_dir}/relax_01",
    "relax_02": f"{double_relax_run_dir}/relax_02",
}
assert double_relax_spec["paths"]["result_dir"] == f"{double_relax_run_dir}/relax_02"
assert f"{double_relax_run_dir}/relax_01" in double_relax_spec["paths"]["directories_to_prepare"]
assert f"{double_relax_run_dir}/relax_02" in double_relax_spec["paths"]["directories_to_prepare"]
assert double_relax_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "double_relax",
    "theory": "pbe",
    "modifiers": [],
    "label": None,
}

dos_flow_spec = {
    **flow_spec,
    "workflow": "dos",
    "calculation_spec": {
        "purpose": "dos",
        "theory": "pbe",
        "modifiers": [],
    },
}
dos_spec = create_submission_spec(
    dos_flow_spec,
    label="Si dos!",
    timestamp="20260629-120000",
    env={},
)
dos_run_dir = f"{DEFAULT_FLOWS_DIR}/Si-dos-20260629-120000"
assert dos_spec["paths"]["run_dir"] == dos_run_dir
assert dos_spec["paths"]["stage_dirs"] == {
    "stage_01": f"{dos_run_dir}/stage_01",
    "stage_02": f"{dos_run_dir}/stage_02",
    "stage_03": f"{dos_run_dir}/stage_03",
}
assert dos_spec["paths"]["result_dir"] == f"{dos_run_dir}/stage_03"
assert f"{dos_run_dir}/stage_01" in dos_spec["paths"]["directories_to_prepare"]
assert f"{dos_run_dir}/stage_02" in dos_spec["paths"]["directories_to_prepare"]
assert f"{dos_run_dir}/stage_03" in dos_spec["paths"]["directories_to_prepare"]
assert dos_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "dos",
    "theory": "pbe",
    "modifiers": [],
    "label": None,
}

band_flow_spec = {
    **flow_spec,
    "workflow": "band_structure",
    "calculation_spec": {
        "purpose": "band_structure",
        "theory": "pbe",
        "modifiers": [],
    },
}
band_spec = create_submission_spec(
    band_flow_spec,
    label="Si bands!",
    timestamp="20260629-120000",
    env={},
)
band_run_dir = f"{DEFAULT_FLOWS_DIR}/Si-bands-20260629-120000"
assert band_spec["paths"]["run_dir"] == band_run_dir
assert band_spec["paths"]["stage_dirs"] == {
    "stage_01": f"{band_run_dir}/stage_01",
    "stage_02": f"{band_run_dir}/stage_02",
    "stage_03": f"{band_run_dir}/stage_03",
}
assert band_spec["paths"]["result_dir"] == f"{band_run_dir}/stage_03"
assert f"{band_run_dir}/stage_01" in band_spec["paths"]["directories_to_prepare"]
assert f"{band_run_dir}/stage_02" in band_spec["paths"]["directories_to_prepare"]
assert f"{band_run_dir}/stage_03" in band_spec["paths"]["directories_to_prepare"]
assert band_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "band_structure",
    "theory": "pbe",
    "modifiers": [],
    "label": None,
}

hse_relax_static_flow_spec = {
    **flow_spec,
    "workflow": "relax_static",
    "calculation_spec": {
        "purpose": "relax_static",
        "theory": "hse06",
        "modifiers": [],
    },
}
hse_relax_static_spec = create_submission_spec(
    hse_relax_static_flow_spec,
    label="Si hse relax static!",
    timestamp="20260629-120000",
    env={},
)
hse_relax_static_run_dir = f"{DEFAULT_FLOWS_DIR}/Si-hse-relax-static-20260629-120000"
assert hse_relax_static_spec["paths"]["run_dir"] == hse_relax_static_run_dir
assert hse_relax_static_spec["paths"]["stage_dirs"] == {
    "stage_01": f"{hse_relax_static_run_dir}/stage_01",
    "stage_02": f"{hse_relax_static_run_dir}/stage_02",
}
assert hse_relax_static_spec["paths"]["result_dir"] == (
    f"{hse_relax_static_run_dir}/stage_02"
)
assert f"{hse_relax_static_run_dir}/stage_01" in hse_relax_static_spec[
    "paths"
]["directories_to_prepare"]
assert f"{hse_relax_static_run_dir}/stage_02" in hse_relax_static_spec[
    "paths"
]["directories_to_prepare"]
assert hse_relax_static_spec["flow_spec"]["workflow"] == "relax_static"
assert hse_relax_static_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "relax_static",
    "theory": "hse06",
    "modifiers": [],
    "label": None,
}
assert hse_relax_static_spec["flow_spec"]["potcar_functional"] == "PBE_64"
assert hse_relax_static_spec["potcar"]["functional"] == "PBE_64"

advanced_flow_spec = {
    **flow_spec,
    "calculation_spec": {
        "purpose": "static",
        "theory": "pbe",
        "modifiers": ["dft_u", "gamma_only", "spin_polarized"],
    },
}
advanced_spec = create_submission_spec(
    advanced_flow_spec,
    label="Advanced static!",
    timestamp="20260629-120000",
    env={},
)
assert advanced_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "static",
    "theory": "pbe",
    "modifiers": ["dft_u", "gamma_only", "spin_polarized"],
    "label": None,
}

hse_flow_spec = {
    **flow_spec,
    "calculation_spec": {
        "purpose": "static",
        "theory": "hse06",
        "modifiers": [],
    },
}
hse_spec = create_submission_spec(
    hse_flow_spec,
    label="Si hse!",
    timestamp="20260629-120000",
    env={},
)
assert hse_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "static",
    "theory": "hse06",
    "modifiers": [],
    "label": None,
}
assert hse_spec["flow_spec"]["potcar_functional"] == "PBE_64"
assert hse_spec["potcar"]["functional"] == "PBE_64"

hse_relax_flow_spec = {
    **flow_spec,
    "workflow": "relax",
    "calculation_spec": {
        "purpose": "relax",
        "theory": "hse06",
        "modifiers": [],
    },
}
hse_relax_spec = create_submission_spec(
    hse_relax_flow_spec,
    label="Si hse relax!",
    timestamp="20260629-120000",
    env={},
)
assert hse_relax_spec["paths"]["stage_dirs"] == {}
assert hse_relax_spec["paths"]["result_dir"] == hse_relax_spec["paths"]["run_dir"]
assert hse_relax_spec["flow_spec"]["workflow"] == "relax"
assert hse_relax_spec["flow_spec"]["calculation_spec"] == {
    "purpose": "relax",
    "theory": "hse06",
    "modifiers": [],
    "label": None,
}
assert hse_relax_spec["flow_spec"]["potcar_functional"] == "PBE_64"
assert hse_relax_spec["potcar"]["functional"] == "PBE_64"

private_potcars_dir = "/private/bmd-potcars"
private_potcars_spec = create_submission_spec(
    flow_spec,
    label="Si static!",
    timestamp="20260629-120000",
    env={},
    potcars_dir=private_potcars_dir,
)
assert private_potcars_spec["potcar"]["repository"] == "private"
assert private_potcars_spec["potcar"]["target"] == f"{private_potcars_dir}/PBE_64"
assert private_potcars_spec["potcar"]["symlink_targets"] == [
    f"{private_potcars_dir}/{link_name}"
    for link_name in POTCAR_LINK_MAP["PBE_64"]
]
assert private_potcars_dir in private_potcars_spec["paths"]["directories_to_prepare"]
assert f"{private_potcars_dir}/PBE_64" in private_potcars_spec["paths"]["directories_to_prepare"]

env_spec = create_submission_spec(
    flow_spec,
    timestamp="20260629-120000",
    env={
        "SLURM_ACCOUNT": "debug-users",
        "VASP_CMD": "srun -n $SLURM_NTASKS vasp_std",
    },
)

assert env_spec["cluster"]["partition"] == DEFAULT_PARTITION
assert env_spec["cluster"]["account"] == "debug-users"
assert env_spec["environment"]["VASP_CMD"] == "srun -n $SLURM_NTASKS vasp_std"

try:
    create_submission_spec(
        flow_spec,
        timestamp="20260629-120000",
        env={"SLURM_PARTITION": "debug; rm -rf /"},
    )
except CalculationValidationError as exc:
    assert "Queue must be one of" in exc.message
else:
    raise AssertionError("Unsafe SLURM partition values should be rejected.")


# The POTCAR record comes from the executable stage generators, never from a
# parallel table: an object that is not a real structure yields an explicit
# "unavailable" record rather than a guessed mapping.
fake_structure_spec = create_submission_spec(
    flow_spec,
    structure=FakeStructure(),
    timestamp="20260629-120000",
    env={},
)
assert fake_structure_spec["potcar"]["status"] == "unavailable"
assert fake_structure_spec["potcar"]["species"] == []
assert fake_structure_spec["potcar"]["symbols"] == []

# Rutile TiO2: pymatgen's MPRelaxSet would say Ti_pv; the atomate2 stage
# generator that executes the run uses Ti_sv, and that is what is recorded.
structure_spec = create_submission_spec(
    flow_spec,
    structure=_rutile_tio2(),
    timestamp="20260629-120000",
    env={},
)
assert structure_spec["potcar"]["status"] == "resolved"
assert structure_spec["potcar"]["symbol_source"] == "bmd_compute.executable_stage_generators"
assert structure_spec["potcar"]["species"] == [
    {"species": "Ti", "potcar_symbol": "Ti_sv"},
    {"species": "O", "potcar_symbol": "O"},
]
assert structure_spec["potcar"]["symbols"] == ["Ti_sv", "O"]
assert structure_spec["potcar"]["consistent_across_stages"] is True
assert [stage["symbols"] for stage in structure_spec["potcar"]["stages"]] == [["Ti_sv", "O"]]

run_job_script = build_run_job_script(spec)
assert "__SPEC_JSON__" not in run_job_script
assert "json.load(handle)" in run_job_script
assert "BMD_SUBMISSION_SPEC" in run_job_script
assert "Si-static-20260629-120000" not in run_job_script
assert "from backend.execution import run_submission" in run_job_script
assert "run_submission(spec)" in run_job_script
assert "RelaxMaker" not in run_job_script
assert "StaticSetGenerator" not in run_job_script

sbatch_script = build_sbatch_script(spec)
script_artifact = build_submission_script_artifact(spec)
assert script_artifact["text"] == sbatch_script
assert script_artifact["sha256"] == hashlib.sha256(sbatch_script.encode("utf-8")).hexdigest()
assert script_artifact["size_bytes"] == len(sbatch_script.encode("utf-8"))
assert f"#SBATCH -p {DEFAULT_PARTITION}" in sbatch_script
assert f"#SBATCH --account={DEFAULT_ACCOUNT}" in sbatch_script
assert f"#SBATCH --nodes={DEFAULT_RESOURCES['nodes']}" in sbatch_script
assert f"#SBATCH --ntasks={DEFAULT_RESOURCES['ntasks']}" in sbatch_script
assert f"#SBATCH --mem={DEFAULT_RESOURCES['mem_gb']}G" in sbatch_script
assert f"#SBATCH --time={DEFAULT_RESOURCES['walltime']}" in sbatch_script
assert f"#SBATCH --comment=bmd_attempt:{spec['submission']['attempt_id']}" in sbatch_script
assert f"export BMD_SUBMISSION_ATTEMPT_ID={spec['submission']['attempt_id']}" in sbatch_script
assert "export PMG_VASP_PSP_DIR=/bmd-db/potcars" in sbatch_script
assert "/bmd-db/guest/potcars" not in sbatch_script
assert "if ! type module >/dev/null 2>&1; then" in sbatch_script
assert "source /etc/bashrc" in sbatch_script
assert "source /usr/share/lmod/lmod/init/bash" not in sbatch_script
assert (
    sbatch_script.index("source /etc/bashrc")
    < sbatch_script.index("module purge")
)
assert "module purge\n" in sbatch_script
for module_name in MODULES:
    assert f"module load {module_name}\n" in sbatch_script
    assert f"module load {module_name} >/dev/null" not in sbatch_script
    assert f"module load {module_name} || true" not in sbatch_script
assert 'echo "[sbatch debug] PATH=$PATH"' in sbatch_script
assert "module list" in sbatch_script
assert "which vasp_std" in sbatch_script
assert "which mpirun" in sbatch_script
assert f"which {DEFAULT_REMOTE_PYTHON}" in sbatch_script
assert f"{DEFAULT_REMOTE_PYTHON} --version" in sbatch_script
assert f"{DEFAULT_REMOTE_PYTHON} -u run_job.py" in sbatch_script
assert f"[sbatch] Runner stdout: {spec['paths']['log_out']}" in sbatch_script
assert f"[sbatch] Runner stderr: {spec['paths']['log_err']}" in sbatch_script
assert f"[sbatch] SLURM stdout: {spec['paths']['slurm_out']}" in sbatch_script
assert f"[sbatch] SLURM stderr: {spec['paths']['slurm_err']}" in sbatch_script
assert sbatch_script.index("[sbatch] Runner stderr:") < sbatch_script.index(f"{DEFAULT_REMOTE_PYTHON} -u run_job.py")
assert "\npython --version" not in sbatch_script
assert "/bmd/lee/envs" not in sbatch_script

execution_module_source = build_execution_module_source()
assert "def run_submission(spec: dict)" in execution_module_source
assert "structure_from_spec" in execution_module_source
assert "build_atomate2_flow_from_spec" in execution_module_source
assert "resources=spec.get(\"resources\")" in execution_module_source
assert "run_locally" in execution_module_source
assert "def incar_static" not in execution_module_source
assert "def incar_relax" not in execution_module_source
assert "def ksettings" not in execution_module_source
assert "RelaxMaker" not in execution_module_source
assert "StaticSetGenerator" not in execution_module_source

backend_module_sources = build_backend_module_sources()
backend_module_names = set(backend_module_sources)
assert backend_module_names >= {
    "config.py",
    "calculations/__init__.py",
    "calculations/builder.py",
    "calculations/custodian_policy.py",
    "calculations/models.py",
    "calculations/resource_policy.py",
    "calculations/resources.py",
    "calculations/registry.py",
    "calculations/theory_policy.py",
    "calculations/vasp_stage_definitions.py",
    "execution.py",
    "parser.py",
    "runtime_package.py",
    "workflows.py",
}
assert all(name.endswith(".py") for name in backend_module_names)
assert all("__pycache__" not in name for name in backend_module_names)
assert all(not name.startswith("tests/") for name in backend_module_names)
assert "class CalculationSpec" in backend_module_sources["calculations/models.py"]
assert (
    "def hse_band_structure_run_vasp_kwargs"
    in backend_module_sources["calculations/custodian_policy.py"]
)
assert "ALLOWED_MEMORY_GB" in backend_module_sources["calculations/resources.py"]
assert "def ncore_for_execution_resources" in backend_module_sources["calculations/resources.py"]
assert "AUTOMATIC_NCORE_STAGE_TYPES" in backend_module_sources["calculations/resource_policy.py"]
assert "def describe_stage" in backend_module_sources["calculations/vasp_stage_definitions.py"]
assert "def calculation_spec_from_flow_spec" in backend_module_sources["calculations/registry.py"]
assert "def theory_incar_settings" in backend_module_sources["calculations/theory_policy.py"]
assert "def structure_from_spec" in backend_module_sources["parser.py"]
assert "def build_atomate2_flow_from_spec" in backend_module_sources["workflows.py"]
assert "RelaxMaker" in backend_module_sources["workflows.py"]
assert "StaticSetGenerator" in backend_module_sources["workflows.py"]

print("submission spec smoke test passed")
