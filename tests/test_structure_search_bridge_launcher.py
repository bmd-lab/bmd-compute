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
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            _settings(tmp_path, **overrides)
    assert _settings(tmp_path, mem=None).mem is None




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
