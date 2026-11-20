"""Unit tests for the launch loop's step command and environment (issue #66).

The loop's behaviour with real processes is in
test_structure_search_bridge_end_to_end_fake.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from backend.runtime_package import runtime_package_relative_paths
from backend.structure_search.bridge.launcher import (
    DEFAULT_COMMAND_TEMPLATE,
    LauncherSettings,
    build_step_argv,
    build_step_environment,
    cpu_range,
    default_step_memory_mb,
    pinned_slot_first_cores,
    slurm_job_environment,
)


def _settings(tmp_path: Path, **overrides) -> LauncherSettings:
    values = dict(
        spool_dir=tmp_path / "spool",
        search_dir=tmp_path / "search",
        max_parallel=4,
        ranks=48,
        mem="200G",
        environment={"PATH": "/usr/bin:/bin"},
    )
    values.update(overrides)
    return LauncherSettings(**values)


# -- step command and environment ------------------------------------------------


def test_default_step_command_matches_the_design():
    assert DEFAULT_COMMAND_TEMPLATE == "srun --exact --mpi=pmi2 -n {ranks} --mem={mem} vasp_std"
    assert build_step_argv(DEFAULT_COMMAND_TEMPLATE, ranks=48, mem="200G", step=1) == [
        "srun", "--exact", "--mpi=pmi2", "-n", "48", "--mem=200G", "vasp_std",
    ]


def test_step_command_without_mem_drops_the_mem_argument():
    assert build_step_argv(DEFAULT_COMMAND_TEMPLATE, ranks=24, mem=None, step=3) == [
        "srun", "--exact", "--mpi=pmi2", "-n", "24", "vasp_std",
    ]


def test_step_command_rejects_bad_templates():
    for template in ("srun -n {cores} vasp_std", "srun -n {0} vasp_std", "", "srun 'unclosed"):
        with pytest.raises(ValueError):
            build_step_argv(template, ranks=1, mem="1G", step=1)
    with pytest.raises(ValueError, match="one argument"):
        build_step_argv("srun --mem {mem} vasp_std", ranks=1, mem=None, step=1)


def test_ranks_per_step_override_the_default(tmp_path):
    settings = _settings(tmp_path, ranks=48, ranks_per_step={3: 96})
    assert settings.argv_for_step(1)[4] == "48"
    assert settings.argv_for_step(3)[4] == "96"


def test_step_environment_sets_mpi_debug_and_never_pin_domain():
    environment, removed = build_step_environment(
        {"PATH": "/x", "I_MPI_PIN_DOMAIN": "core", "I_MPI_DEBUG": "0"}
    )
    assert environment == {"PATH": "/x", "I_MPI_DEBUG": "4"}
    assert removed == ["I_MPI_PIN_DOMAIN"]


def test_settings_validation(tmp_path):
    bad = [
        dict(max_parallel=0),
        dict(ranks=0),
        dict(mem="lots"),
        dict(mem="0G"),
        dict(ranks_per_step={0: 4}),
        dict(ranks_per_step={1: 0}),
        dict(poll_seconds=0),
        dict(kill_grace_seconds=float("nan")),
        dict(spool_dir=Path("relative")),
        dict(command_template="srun -n {cpus} vasp_std"),
        dict(launch_mode="sbatch"),
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            _settings(tmp_path, **overrides)


def test_srun_steps_requires_a_per_step_mem(tmp_path):
    # Concurrent srun --exact steps without --mem ran one after another (#59).
    with pytest.raises(ValueError, match="per-step --mem"):
        _settings(tmp_path, mem=None)
    with pytest.raises(ValueError, match="--mem=\\{mem\\}"):
        _settings(tmp_path, command_template="srun --exact -n {ranks} vasp_std")
    with pytest.raises(ValueError):
        _settings(tmp_path, mem="0")
    with pytest.raises(ValueError):
        _settings(tmp_path, mem=None, environment={"SLURM_MEM_PER_NODE": "0"})


def test_srun_steps_default_mem_is_node_memory_less_margin_over_parallel_steps(tmp_path):
    settings = _settings(tmp_path, mem=None, node_memory_mb=1_000_000, max_parallel=4)
    assert settings.step_mem == "225000M"
    assert "--mem=225000M" in settings.argv_for_step(1)
    settings = _settings(
        tmp_path, mem=None, node_memory_mb=1_000_000, max_parallel=4, memory_margin_fraction=0.2
    )
    assert settings.step_mem == "200000M"
    settings = _settings(
        tmp_path, mem=None, max_parallel=2, environment={"SLURM_MEM_PER_NODE": "100000"}
    )
    assert settings.step_mem == "45000M"
    # An explicit mem wins.
    assert _settings(tmp_path, mem="200G", node_memory_mb=10).step_mem == "200G"
    with pytest.raises(ValueError):
        default_step_memory_mb(1000, 4, 1.0)


def test_pinned_slot_layout_balances_sockets():
    assert [
        cpu_range(first, 48)
        for first in pinned_slot_first_cores(slots=4, ranks=48, sockets=2, cores_per_socket=96)
    ] == ["0-47", "96-143", "48-95", "144-191"]
    assert pinned_slot_first_cores(slots=2, ranks=96, sockets=2, cores_per_socket=96) == (0, 96)
    assert pinned_slot_first_cores(slots=8, ranks=24, sockets=2, cores_per_socket=96) == (
        0, 96, 24, 120, 48, 144, 72, 168,
    )
    assert pinned_slot_first_cores(slots=3, ranks=32, sockets=2, cores_per_socket=96) == (0, 96, 32)
    assert pinned_slot_first_cores(slots=1, ranks=192, sockets=2, cores_per_socket=96) == (0,)
    assert cpu_range(5, 1) == "5"


@pytest.mark.parametrize(
    "slots, ranks",
    [(5, 48), (3, 64), (2, 144), (2, 192), (1, 200), (0, 48)],
)
def test_pinned_slot_layout_rejects_layouts_that_do_not_fit(slots, ranks):
    with pytest.raises(ValueError):
        pinned_slot_first_cores(slots=slots, ranks=ranks, sockets=2, cores_per_socket=96)


def test_mpirun_pinned_settings(tmp_path):
    settings = _settings(tmp_path, launch_mode="mpirun_pinned", mem=None, ranks_per_step={2: 24})
    assert settings.command_template == "mpirun -n {ranks} vasp_std"
    assert settings.argv_for_step(1) == ["mpirun", "-n", "48", "vasp_std"]
    assert settings.argv_for_step(2) == ["mpirun", "-n", "24", "vasp_std"]
    assert settings.environment_for_slot(1, 1) == {
        "I_MPI_PIN_PROCESSOR_LIST": "96-143",
        "I_MPI_HYDRA_BOOTSTRAP": "fork",
    }
    assert settings.environment_for_slot(2, 3)["I_MPI_PIN_PROCESSOR_LIST"] == "144-167"
    assert _settings(tmp_path).environment_for_slot(1, 0) == {}
    with pytest.raises(ValueError):
        _settings(tmp_path, launch_mode="mpirun_pinned", ranks_per_step={2: 96})
    with pytest.raises(ValueError):
        _settings(tmp_path, launch_mode="mpirun_pinned", max_parallel=5)
    # compute-0-332/333 style override: a different topology is a setting.
    assert _settings(
        tmp_path, launch_mode="mpirun_pinned", max_parallel=8, ranks=24, sockets=8, cores_per_socket=24
    ).slot_first_cores == tuple(range(0, 192, 24))




def test_slurm_job_environment_keeps_only_slurm_variables():
    assert slurm_job_environment(
        {"SLURM_JOB_ID": "12", "SLURM_NTASKS": "192", "HOME": "/home/x", "XSLURM_": "1"}
    ) == {"SLURM_JOB_ID": "12", "SLURM_NTASKS": "192"}


def test_launcher_is_in_the_verified_runtime_package():
    assert "structure_search/bridge/launcher.py" in runtime_package_relative_paths()


def test_launcher_module_does_not_import_scientific_stack():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, backend.structure_search.bridge.launcher, backend.structure_search.bridge.master;"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in {'pymatgen','atomate2','jobflow','numpy'}))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


# --- memory SLURM allocated to the job (exclusive --mem=0 allocations) -------------------------

from backend.structure_search.bridge.launcher import (  # noqa: E402
    NodeMemoryUnavailable,
    allocated_node_memory,
    node_memory_from_job_record,
)

_RECORD = ("JobId=4242 JobName=bmds-x JobState=RUNNING NumNodes=1 NumCPUs=192 MinMemoryNode=0 "
           "AllocTRES=cpu=192,mem=500G,node=1,billing=192")


def _never_asked(job_id, environment):
    raise AssertionError("SLURM must not be asked when the job's variables give an amount")


def test_a_positive_slurm_mem_per_node_is_used_and_divided_as_before(tmp_path):
    memory = allocated_node_memory({"SLURM_MEM_PER_NODE": "512000", "SLURM_JOB_ID": "4242"}, show_job=_never_asked)
    assert (memory.mb, memory.source) == (512000, "SLURM_MEM_PER_NODE")
    settings = _settings(tmp_path, mem=None, max_parallel=4, node_memory_mb=memory.mb)
    assert settings.step_mem == "115200M"  # less 10 %, over 4 relaxations
    environment_only = _settings(tmp_path, mem=None, max_parallel=4, environment={"SLURM_MEM_PER_NODE": "512000"})
    assert environment_only.step_mem == settings.step_mem  # one policy, either way in


def test_per_cpu_memory_counts_the_cpus_on_the_node():
    memory = allocated_node_memory({"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_ON_NODE": "192"}, show_job=_never_asked)
    assert (memory.mb, memory.source) == (384000, "SLURM_MEM_PER_CPU")


@pytest.mark.parametrize("environment", [
    {"SLURM_JOB_ID": "4242"},                              # SLURM_MEM_PER_NODE missing
    {"SLURM_JOB_ID": "4242", "SLURM_MEM_PER_NODE": "0"},   # --mem=0: all memory, not an amount
])
def test_missing_or_zero_slurm_mem_per_node_asks_slurm_for_the_allocation(environment):
    asked = []

    def show_job(job_id, env):
        asked.append(job_id)
        return _RECORD

    memory = allocated_node_memory(environment, show_job=show_job)
    assert asked == ["4242"]
    assert (memory.mb, memory.source) == (512000, "scontrol AllocTRES mem")


@pytest.mark.parametrize("value", ["512G", "", "-1", "1e6", "512000 ", "abc"])
def test_malformed_slurm_mem_per_node_is_refused_without_guessing(value):
    with pytest.raises(NodeMemoryUnavailable, match="not a number of MB"):
        allocated_node_memory({"SLURM_JOB_ID": "4242", "SLURM_MEM_PER_NODE": value}, show_job=_never_asked)


@pytest.mark.parametrize("environment, record, message", [
    ({"SLURM_MEM_PER_NODE": "0"}, _RECORD, "no SLURM_JOB_ID"),
    ({"SLURM_MEM_PER_NODE": "0", "SLURM_JOB_ID": "4242"}, None, "did not answer"),
    ({"SLURM_MEM_PER_NODE": "0", "SLURM_JOB_ID": "4242"},
     "JobId=4242 JobState=RUNNING NumNodes=1 MinMemoryNode=0 AllocTRES=cpu=192,node=1", "no unambiguous"),
    ({"SLURM_MEM_PER_NODE": "0", "SLURM_JOB_ID": "4242"}, _RECORD.replace("4242", "4243"), "no unambiguous"),
    ({"SLURM_MEM_PER_NODE": "0", "SLURM_JOB_ID": "4242"}, _RECORD.replace("NumNodes=1", "NumNodes=2"),
     "no unambiguous"),
    ({"SLURM_MEM_PER_NODE": "0", "SLURM_JOB_ID": "4242"}, _RECORD + "\n" + _RECORD, "no unambiguous"),
])
def test_allocated_memory_fails_closed(environment, record, message):
    with pytest.raises(NodeMemoryUnavailable, match=message):
        allocated_node_memory(environment, show_job=lambda job_id, env: record)


@pytest.mark.parametrize("value, mb", [("512000", 512000), ("512000M", 512000), ("500G", 512000),
                                        ("1.5T", 1572864), ("2048K", 2)])
def test_slurm_memory_units(value, mb):
    record = f"JobId=7 JobState=RUNNING NumNodes=1 AllocTRES=cpu=1,mem={value},node=1"
    assert node_memory_from_job_record(record, "7").mb == mb


@pytest.mark.parametrize("record", [
    "JobId=7 JobState=RUNNING NumNodes=1 MinMemoryNode=250G AllocTRES=cpu=96,node=1",  # a request, no allocation
    "JobId=7 JobState=RUNNING NumNodes=1 MinMemoryNode=0 AllocTRES=cpu=96,node=1",     # --mem=0 request only
    "JobId=7 JobState=RUNNING NumNodes=1 MinMemoryNode=250G",                          # no AllocTRES at all
    "JobId=7 JobState=RUNNING NumNodes=1 AllocTRES=cpu=96,mem=0,node=1",               # zero allocation
    "JobId=7 JobState=RUNNING NumNodes=1 AllocTRES=cpu=96,mem=0M,node=1",
    "JobId=7 JobState=RUNNING NumNodes=1 AllocTRES=cpu=96,mem=0.5K,node=1",            # rounds to 0 MB
    "JobId=7 JobState=RUNNING NumNodes=1 AllocTRES=cpu=96,mem=?,node=1",               # unreadable
    "JobId=7 JobState=RUNNING NumNodes=1 AllocTRES=cpu=96,mem=500G,mem=400G,node=1",   # two allocations
    "JobId=7 JobState=RUNNING NumNodes=1 MinMemoryNode=400G AllocTRES=cpu=96,mem=500G,node=1",  # contradicts
    "JobId=7 JobState=RUNNING NumNodes=1 MinMemoryNode=lots AllocTRES=cpu=96,mem=500G,node=1",  # unreadable
    "JobId=7 JobState=PENDING NumNodes=1 AllocTRES=cpu=96,mem=500G,node=1",            # not running
    "JobId=7 JobState=RUNNING AllocTRES=cpu=96,mem=500G,node=1",                       # node count unknown
    "JobId=7 JobState=RUNNING NumNodes=1 NumNodes=2 AllocTRES=cpu=96,mem=500G,node=1",  # repeated field
])
def test_a_request_or_an_ambiguous_record_is_never_taken_as_an_allocation(record):
    assert node_memory_from_job_record(record, "7") is None


@pytest.mark.parametrize("min_memory_node", ["0", "500G", "512000M", None])
def test_a_zero_or_matching_request_does_not_block_the_allocation(min_memory_node):
    request = f" MinMemoryNode={min_memory_node}" if min_memory_node is not None else ""
    record = f"JobId=7 JobState=RUNNING NumNodes=1{request} AllocTRES=cpu=192,mem=500G,node=1"
    memory = node_memory_from_job_record(record, "7")
    assert (memory.mb, memory.source) == (512000, "scontrol AllocTRES mem")


def test_scontrol_is_found_on_the_job_path_and_its_absence_fails_closed(tmp_path):
    environment = {"SLURM_JOB_ID": "4242", "SLURM_MEM_PER_NODE": "0", "PATH": str(tmp_path)}
    with pytest.raises(NodeMemoryUnavailable, match="did not answer"):
        allocated_node_memory(environment)
    scontrol = tmp_path / "scontrol"
    scontrol.write_text(f"#!{sys.executable}\nimport sys\nassert sys.argv[1:] == ['-o', 'show', 'job', '4242']\n"
                        f"print({_RECORD!r})\n")
    scontrol.chmod(0o755)
    assert allocated_node_memory(environment).mb == 512000


def test_launcher_settings_still_refuse_zero_and_never_query_slurm(tmp_path):
    with pytest.raises(NodeMemoryUnavailable, match="allocated_node_memory"):
        _settings(tmp_path, mem=None, environment={"SLURM_MEM_PER_NODE": "0", "SLURM_JOB_ID": "4242",
                                                   "PATH": "/nonexistent"})
