from __future__ import annotations

import math
import os


def _bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise RuntimeError(
        f"{name} must be a boolean value: true/false, yes/no, on/off, or 1/0."
    )


def bmd_debug_enabled() -> bool:
    return _bool_env(BMD_DEBUG_ENV, default=False)


def _positive_int_env(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return parsed


def _positive_float_env(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number of seconds.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise RuntimeError(f"{name} must be a positive finite number of seconds.")
    return parsed


DEFAULT_SSH_CONFIG_HOST = "powerslurm-bmdguest"
DEFAULT_REMOTE_HOST = DEFAULT_SSH_CONFIG_HOST
DEFAULT_USERNAME = "bmdguest"
DEFAULT_SSH_KEY_FILE = None
DEFAULT_SSH_PORT = 22
DEFAULT_KEEPALIVE_S = 30

DEFAULT_OPEN_MONGO_TUNNEL = True
DEFAULT_MONGO_REMOTE_HOST = "132.66.112.243"
DEFAULT_MONGO_REMOTE_PORT = 27017
DEFAULT_MONGO_LOCAL_PORT = 27017

DEFAULT_VASP_CMD = "mpirun -n $SLURM_NTASKS vasp_std"
DEFAULT_VASP_LAUNCHER = "srun --mpi=pmi2 -n $SLURM_NTASKS vasp_std"
DEFAULT_REMOTE_WORKSPACE_ROOT = "/bmd-db/guest"
DEFAULT_JOBFLOW_CONFIG_FILE = f"{DEFAULT_REMOTE_WORKSPACE_ROOT}/jobflow_minimal.yaml"
DEFAULT_SHARED_POTCAR_ROOT = "/bmd-db/potcars"
DEFAULT_POTCAR_DIR = DEFAULT_SHARED_POTCAR_ROOT
DEFAULT_REMOTE_PYTHON = "/bmd/bmdguest/envs/atomate2_remote/bin/python"
DEFAULT_REMOTE_ENV_DIR = DEFAULT_REMOTE_PYTHON.removesuffix("/bin/python")
DEFAULT_FLOWS_DIR = f"{DEFAULT_REMOTE_WORKSPACE_ROOT}/flows"
DEFAULT_LOGS_DIR = f"{DEFAULT_REMOTE_WORKSPACE_ROOT}/logs"

DEFAULT_PARTITION = "leeburton-pool"
DEFAULT_ACCOUNT = "power-leeburton-users_v2"

DEFAULT_RESOURCES = {
    "nodes": 1,
    "ntasks": 24,
    "mem_gb": 128,
    "walltime": "72:00:00",
}

MODULES = [
    "intel/rocky8-oneAPI-2023",
    "vasp/rocky8-intel-6.4.1",
]
DEFAULT_MODULES = MODULES

BMD_DEBUG_ENV = "BMD_DEBUG"
BMD_DEBUG = bmd_debug_enabled()

REMOTE_OPERATION_LIMIT_ENV = "BMD_MAX_CONCURRENT_REMOTE_OPERATIONS"
REMOTE_OPERATION_SLOT_TIMEOUT_ENV = "BMD_REMOTE_OPERATION_SLOT_TIMEOUT_S"
MAX_CONCURRENT_REMOTE_OPERATIONS = _positive_int_env(
    REMOTE_OPERATION_LIMIT_ENV,
    default=4,
)
REMOTE_OPERATION_SLOT_TIMEOUT_S = _positive_float_env(
    REMOTE_OPERATION_SLOT_TIMEOUT_ENV,
    default=5.0,
)

DEFAULT_POTCAR_FUNCTIONAL = "PBE_64"
POTCAR_LINK_MAP = {
    "PBE_64": ["POT_GGA_PAW_PBE_64", "POT_PAW_PBE_64"],
    "PBE_54": ["POT_GGA_PAW_PBE_54"],
    "PBE_52": ["POT_GGA_PAW_PBE_52", "POT_GGA_PAW_PBE"],
    "LDA": ["POT_LDA_PAW"],
}

SUBMISSION_ENV_KEYS = (
    "VASP_CMD",
    "JOBFLOW_CONFIG_FILE",
    "PMG_VASP_PSP_DIR",
    "CUSTODIAN_NO_GZIP",
    "ATOMATE2_VASP_ZIP_FILES",
)

NOTEBOOK_DEFAULTS = {
    "ssh_config_host": DEFAULT_SSH_CONFIG_HOST,
    "remote_host": DEFAULT_REMOTE_HOST,
    "username": DEFAULT_USERNAME,
    "key_file": DEFAULT_SSH_KEY_FILE,
    "port": DEFAULT_SSH_PORT,
    "keepalive_s": DEFAULT_KEEPALIVE_S,
    "open_mongo_tunnel": DEFAULT_OPEN_MONGO_TUNNEL,
    "mongo_remote_host": DEFAULT_MONGO_REMOTE_HOST,
    "mongo_remote_port": DEFAULT_MONGO_REMOTE_PORT,
    "mongo_local_port": DEFAULT_MONGO_LOCAL_PORT,
    "remote_workspace_root": DEFAULT_REMOTE_WORKSPACE_ROOT,
    "VASP_CMD": DEFAULT_VASP_CMD,
    "JOBFLOW_CONFIG_FILE": DEFAULT_JOBFLOW_CONFIG_FILE,
    "PMG_VASP_PSP_DIR": DEFAULT_POTCAR_DIR,
    "remote_python": DEFAULT_REMOTE_PYTHON,
    "remote_env_dir": DEFAULT_REMOTE_ENV_DIR,
    "flows_dir": DEFAULT_FLOWS_DIR,
    "logs_dir": DEFAULT_LOGS_DIR,
}


__all__ = [
    "BMD_DEBUG",
    "BMD_DEBUG_ENV",
    "DEFAULT_ACCOUNT",
    "DEFAULT_FLOWS_DIR",
    "DEFAULT_JOBFLOW_CONFIG_FILE",
    "DEFAULT_KEEPALIVE_S",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_MODULES",
    "DEFAULT_MONGO_LOCAL_PORT",
    "DEFAULT_MONGO_REMOTE_HOST",
    "DEFAULT_MONGO_REMOTE_PORT",
    "DEFAULT_OPEN_MONGO_TUNNEL",
    "DEFAULT_PARTITION",
    "DEFAULT_POTCAR_DIR",
    "DEFAULT_POTCAR_FUNCTIONAL",
    "DEFAULT_REMOTE_ENV_DIR",
    "DEFAULT_REMOTE_HOST",
    "DEFAULT_REMOTE_PYTHON",
    "DEFAULT_REMOTE_WORKSPACE_ROOT",
    "DEFAULT_RESOURCES",
    "DEFAULT_SHARED_POTCAR_ROOT",
    "DEFAULT_SSH_CONFIG_HOST",
    "DEFAULT_SSH_KEY_FILE",
    "DEFAULT_SSH_PORT",
    "DEFAULT_USERNAME",
    "DEFAULT_VASP_CMD",
    "DEFAULT_VASP_LAUNCHER",
    "MAX_CONCURRENT_REMOTE_OPERATIONS",
    "MODULES",
    "NOTEBOOK_DEFAULTS",
    "POTCAR_LINK_MAP",
    "REMOTE_OPERATION_LIMIT_ENV",
    "REMOTE_OPERATION_SLOT_TIMEOUT_ENV",
    "REMOTE_OPERATION_SLOT_TIMEOUT_S",
    "SUBMISSION_ENV_KEYS",
    "bmd_debug_enabled",
]
