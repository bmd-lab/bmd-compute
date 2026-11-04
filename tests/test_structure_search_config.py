"""Configuration for USPEX structure searches (backend/config.py)."""

import os

import pytest

from backend import config


POWER_DEFAULTS = {
    "USPEX_SANDBOX_DIR": ("BMD_USPEX_SANDBOX_DIR", "/bmd/shared/uspex25_sandbox"),
    "USPEX_CACHE_DIR": ("BMD_USPEX_CACHE_DIR", "/bmd/shared/uspex_cache"),
    "USPEX_PAYLOAD_SHA256": (
        "BMD_USPEX_PAYLOAD_SHA256",
        "fc4a08e65ffc5f4fc3b45cb058de2cd54731ab2c191123aac8a3c69e87e95c8c",
    ),
    "SINGULARITY_EXECUTABLE": (
        "BMD_SINGULARITY_EXECUTABLE",
        "/powerapps/share/rocky9/mamba/mamba-1.5.8/envs/singularity-3.8.6/bin/singularity",
    ),
    "STRUCTURE_SEARCH_PMI_LIBRARY": ("BMD_SEARCH_I_MPI_PMI_LIBRARY", "/usr/lib64/libpmi2.so"),
    "STRUCTURE_SEARCH_LAUNCH_MODE": ("BMD_SEARCH_LAUNCH_MODE", "srun_steps"),
    "STRUCTURE_SEARCH_PARALLEL_RELAXATIONS": ("BMD_SEARCH_PARALLEL_RELAXATIONS", 4),
    "STRUCTURE_SEARCH_RANKS_PER_RELAXATION": ("BMD_SEARCH_RANKS_PER_RELAXATION", 48),
    "STRUCTURE_SEARCH_BRIDGE_TIMEOUT_S": ("BMD_SEARCH_BRIDGE_TIMEOUT_S", 14400),
    "STRUCTURE_SEARCH_NODE_SOCKETS": ("BMD_SEARCH_NODE_SOCKETS", 2),
    "STRUCTURE_SEARCH_CORES_PER_SOCKET": ("BMD_SEARCH_CORES_PER_SOCKET", 96),
    "STRUCTURE_SEARCH_EXCLUDED_NODES": ("BMD_SEARCH_EXCLUDED_NODES", "compute-0-[332-333]"),
}


@pytest.mark.parametrize("name", sorted(POWER_DEFAULTS))
def test_power_defaults_and_override_names(name):
    env_name, default = POWER_DEFAULTS[name]
    assert getattr(config, f"{name}_ENV") == env_name
    # Only check the default when the test machine does not override it.
    if env_name not in os.environ:
        assert getattr(config, name) == default


def test_default_layout_fills_one_normal_node():
    assert (
        config.STRUCTURE_SEARCH_PARALLEL_RELAXATIONS * config.STRUCTURE_SEARCH_RANKS_PER_RELAXATION
        <= config.STRUCTURE_SEARCH_NODE_SOCKETS * config.STRUCTURE_SEARCH_CORES_PER_SOCKET
    )


def test_text_override(monkeypatch):
    monkeypatch.setenv("BMD_TEST_TEXT", "  compute-0-1  ")
    assert config._text_env("BMD_TEST_TEXT", default="x") == "compute-0-1"
    monkeypatch.setenv("BMD_TEST_TEXT", "")
    assert config._text_env("BMD_TEST_TEXT", default="x") == "x"
    assert config._text_env("BMD_TEST_TEXT", default="x", allow_empty=True) == ""
    monkeypatch.delenv("BMD_TEST_TEXT")
    assert config._text_env("BMD_TEST_TEXT", default="x", allow_empty=True) == "x"


def test_path_override_must_be_absolute(monkeypatch):
    monkeypatch.setenv("BMD_TEST_PATH", "/bmd/other/sandbox")
    assert config._absolute_path_env("BMD_TEST_PATH", default="/d") == "/bmd/other/sandbox"
    monkeypatch.setenv("BMD_TEST_PATH", "relative/sandbox")
    with pytest.raises(RuntimeError, match="BMD_TEST_PATH must be an absolute path"):
        config._absolute_path_env("BMD_TEST_PATH", default="/d")


def test_launch_mode_override_is_checked(monkeypatch):
    monkeypatch.setenv("BMD_TEST_MODE", "mpirun_pinned")
    assert (
        config._choice_env("BMD_TEST_MODE", default="srun_steps", choices=config.STRUCTURE_SEARCH_LAUNCH_MODES)
        == "mpirun_pinned"
    )
    monkeypatch.setenv("BMD_TEST_MODE", "mpirun")
    with pytest.raises(RuntimeError, match="must be one of: srun_steps, mpirun_pinned"):
        config._choice_env("BMD_TEST_MODE", default="srun_steps", choices=config.STRUCTURE_SEARCH_LAUNCH_MODES)


def test_ordinary_workflow_defaults_are_unchanged():
    assert config.DEFAULT_RESOURCES == {"nodes": 1, "ntasks": 24, "mem_gb": 128, "walltime": "72:00:00"}
    assert config.DEFAULT_QOS == "owner100d"
    assert config.DEFAULT_VASP_LAUNCHER == "srun --mpi=pmi2 -n $SLURM_NTASKS vasp_std"
