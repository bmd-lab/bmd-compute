from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import posixpath
import re
import secrets
import shlex
import time
import uuid
from copy import deepcopy

from backend.calculations.models import CalculationSpec, WorkflowSpec
from backend.calculations.registry import (
    calculation_spec_from_workflow_spec,
    legacy_workflow_from_spec,
    modifier_display_name,
    stage_display_name,
    theory_display_name,
    validate_calculation_spec,
    validate_workflow_spec,
    workflow_result_stage_directory,
    workflow_spec_from_flow_spec,
    workflow_stage_directories,
)
from backend.calculations.potcar_record import executable_potcar_record
from backend.calculations.resources import validate_queue
from backend.config import (
    DEFAULT_ACCOUNT,
    DEFAULT_PARTITION,
    DEFAULT_POTCAR_FUNCTIONAL,
    DEFAULT_REMOTE_PYTHON,
    DEFAULT_RESOURCES,
    DEFAULT_SHARED_POTCAR_ROOT,
    DEFAULT_VASP_LAUNCHER,
    MODULES,
    NOTEBOOK_DEFAULTS,
    POTCAR_LINK_MAP,
    SUBMISSION_ENV_KEYS,
    WORKFLOW_RESOURCE_OVERRIDES,
)
from backend.run_records import submission_record_header
from backend.provenance import build_submission_provenance
from backend.runtime_package import (
    RUNTIME_PACKAGE_DIR,
    build_runtime_package_sources,
    runtime_package_relative_paths,
)


SUBMISSION_SPEC_FILENAME = "submission.json"
SUBMISSION_ATTEMPTS_DIRNAME = "submission_attempts"
SUBMISSION_ATTEMPT_STATE_VERSION = 1
SUBMISSION_ATTEMPT_COMMENT_PREFIX = "bmd_attempt:"
REMOTE_BACKEND_PACKAGE_DIR = RUNTIME_PACKAGE_DIR
REMOTE_EXECUTION_MODULE_FILENAME = "execution.py"
REMOTE_BACKEND_INIT_FILENAME = "__init__.py"
REMOTE_BACKEND_MODULE_FILENAMES = runtime_package_relative_paths()
REMOTE_RUNTIME_PREFLIGHT_STEP = "Runtime import preflight"
REMOTE_RUNTIME_PREFLIGHT_TIMEOUT_S = 60
SUBMISSION_IDENTITY_VERSION = 1
SUBMISSION_IDENTITY_SECRET_ENV = "BMD_SUBMISSION_IDENTITY_SECRET"
RUN_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}$")
MACHINE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
RELATIVE_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,95}$")
SLURM_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_REMOTE_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9._/+\-]+$")
SAFE_MODULE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+\-]{0,127}$")
WALLTIME_PATTERN = re.compile(r"^[0-9]{1,3}:[0-9]{2}:[0-9]{2}$")
_EPHEMERAL_SUBMISSION_IDENTITY_SECRET = secrets.token_bytes(32)

def sanitize_label(label: str) -> str:
    label = (label or "").strip()
    label = re.sub(r"\s+", "-", label)
    label = re.sub(r"[^A-Za-z0-9._-]+", "", label)
    return (label or "vasp_run")[:60].rstrip("-_.") or "vasp_run"


def canonical_run_timestamp(value: str | None = None) -> str:
    timestamp = str(value or time.strftime("%Y%m%d-%H%M%S")).strip()
    if not RUN_TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValueError("Run timestamp must use the canonical YYYYMMDD-HHMMSS format.")
    try:
        time.strptime(timestamp, "%Y%m%d-%H%M%S")
    except ValueError as exc:
        raise ValueError("Run timestamp is not a valid calendar timestamp.") from exc
    return timestamp


def create_submission_identity_token(timestamp: str, attempt_id: str) -> str:
    payload = {
        "attempt_id": _normalized_submission_attempt_id(attempt_id),
        "run_timestamp": canonical_run_timestamp(timestamp),
        "version": SUBMISSION_IDENTITY_VERSION,
    }
    encoded = _urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    signature = hmac.new(
        _submission_identity_secret(),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{encoded}.{_urlsafe_b64encode(signature)}"


def verify_submission_identity_token(
    token: str | None,
    *,
    expected_attempt_id: str | None = None,
) -> dict:
    value = str(token or "").strip()
    if not value or len(value) > 1024 or value.count(".") != 1:
        raise ValueError("Submission identity is missing or malformed.")
    encoded, supplied_signature = value.split(".", 1)
    expected_signature = hmac.new(
        _submission_identity_secret(),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        decoded_signature = _urlsafe_b64decode(supplied_signature)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Submission identity signature is malformed.") from exc
    if not hmac.compare_digest(decoded_signature, expected_signature):
        raise ValueError("Submission identity signature is invalid.")
    try:
        payload = json.loads(_urlsafe_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Submission identity payload is malformed.") from exc
    if not isinstance(payload, dict) or payload.get("version") != SUBMISSION_IDENTITY_VERSION:
        raise ValueError("Submission identity version is unsupported.")
    if not isinstance(payload.get("attempt_id"), str) or not isinstance(
        payload.get("run_timestamp"), str
    ):
        raise ValueError("Submission identity payload is malformed.")
    attempt_id = _normalized_submission_attempt_id(payload["attempt_id"])
    if expected_attempt_id and attempt_id != _normalized_submission_attempt_id(expected_attempt_id):
        raise ValueError("Submission identity does not match the submission attempt.")
    return {
        "attempt_id": attempt_id,
        "run_timestamp": canonical_run_timestamp(payload["run_timestamp"]),
    }


def _submission_identity_secret() -> bytes:
    configured = os.environ.get(SUBMISSION_IDENTITY_SECRET_ENV)
    if configured:
        return configured.encode("utf-8")
    return _EPHEMERAL_SUBMISSION_IDENTITY_SECRET


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value or ""):
        raise ValueError("Invalid base64url value.")
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _coerce_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _env_bin(remote_env_dir: str) -> str:
    path = (remote_env_dir or "").rstrip("/")
    return path + "/bin" if path else "/usr/bin"


def _first_nonempty(*values):
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
        elif value is not None:
            return value
    return None


def _remote_path_equal(left: str | None, right: str | None) -> bool:
    return posixpath.normpath(str(left or "").rstrip("/")) == posixpath.normpath(
        str(right or "").rstrip("/")
    )


def _uses_shared_potcar_repository(potcars_dir: str | None) -> bool:
    return _remote_path_equal(potcars_dir, DEFAULT_SHARED_POTCAR_ROOT)


def new_submission_attempt_id() -> str:
    return str(uuid.uuid4())


def _normalized_submission_attempt_id(value: str | None = None) -> str:
    attempt_id = str(value or "").strip()
    if not attempt_id:
        return new_submission_attempt_id()
    try:
        return str(uuid.UUID(attempt_id))
    except ValueError as exc:
        raise ValueError("Submission attempt ID must be a UUID.") from exc


def submission_attempt_metadata(submission_spec: dict) -> dict:
    paths = submission_spec.get("paths", {})
    return {
        "version": SUBMISSION_ATTEMPT_STATE_VERSION,
        "label": submission_spec.get("label"),
        "run_name": submission_spec.get("run_name"),
        "created_at": submission_spec.get("created_at"),
        "flow_spec": deepcopy(submission_spec.get("flow_spec")),
        "paths": {
            "run_dir": paths.get("run_dir"),
            "remote_script": paths.get("remote_script"),
            "logs_dir": paths.get("logs_dir"),
            "flows_dir": paths.get("flows_dir"),
            "result_dir": paths.get("result_dir"),
            "stage_dirs": deepcopy(paths.get("stage_dirs", {})),
        },
        "cluster": deepcopy(submission_spec.get("cluster")),
        "resources": deepcopy(submission_spec.get("resources")),
        "environment": deepcopy(submission_spec.get("environment")),
        "modules": deepcopy(submission_spec.get("modules")),
        "runner": deepcopy(submission_spec.get("runner")),
        "potcar": deepcopy(submission_spec.get("potcar")),
    }


def submission_attempt_fingerprint(submission_spec: dict) -> str:
    payload = json.dumps(
        submission_attempt_metadata(submission_spec),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def submission_attempt_comment(submission_spec: dict) -> str:
    return f"{SUBMISSION_ATTEMPT_COMMENT_PREFIX}{submission_attempt_id(submission_spec)}"


def submission_attempt_id(submission_spec: dict) -> str:
    attempt_id = (
        (submission_spec.get("submission") or {}).get("attempt_id")
        or submission_spec.get("submission_attempt_id")
    )
    return _normalized_submission_attempt_id(attempt_id)


def initialize_submission_attempt(
    submission_spec: dict,
    *,
    attempt_id: str | None = None,
) -> dict:
    normalized_attempt_id = _normalized_submission_attempt_id(attempt_id)
    paths = submission_spec.setdefault("paths", {})
    logs_dir = str(paths.get("logs_dir") or NOTEBOOK_DEFAULTS["logs_dir"]).rstrip("/")
    attempts_dir = posixpath.join(logs_dir, SUBMISSION_ATTEMPTS_DIRNAME)
    state_path = posixpath.join(attempts_dir, f"{normalized_attempt_id}.json")
    lock_dir = posixpath.join(attempts_dir, f"{normalized_attempt_id}.lock")
    paths["submission_attempts_dir"] = attempts_dir
    paths["submission_attempt_state"] = state_path
    paths["submission_attempt_lock"] = lock_dir
    directories = paths.setdefault("directories_to_prepare", [])
    if attempts_dir not in directories:
        directories.append(attempts_dir)

    submission = submission_spec.setdefault("submission", {})
    submission["attempt_id"] = normalized_attempt_id
    submission["attempt_state"] = state_path
    submission["attempt_lock"] = lock_dir
    submission["attempt_comment"] = f"{SUBMISSION_ATTEMPT_COMMENT_PREFIX}{normalized_attempt_id}"
    submission["attempt_fingerprint"] = submission_attempt_fingerprint(submission_spec)
    return submission_spec


def default_resources_for_workflow(workflow: str | None = None) -> dict:
    resources = dict(DEFAULT_RESOURCES)
    workflow_name = (workflow or "").lower()
    resources.update(WORKFLOW_RESOURCE_OVERRIDES.get(workflow_name, {}))

    return resources


def default_resources_for_calculation_spec(spec: CalculationSpec | None = None) -> dict:
    if spec is not None:
        validate_calculation_spec(spec)

    return dict(DEFAULT_RESOURCES)


def default_resources_for_workflow_spec(spec: WorkflowSpec | None = None) -> dict:
    if spec is not None:
        validate_workflow_spec(spec)

    return dict(DEFAULT_RESOURCES)


def _shell_export(name: str, value: str | None) -> str:
    if value is None:
        return ""
    return f"export {name}={shlex.quote(str(value))}"


def _shell_log(message: str) -> str:
    return f"printf '%s\\n' {shlex.quote(str(message))}"


def _validate_submission_script_fields(submission_spec: dict) -> None:
    paths = submission_spec.get("paths") or {}
    cluster = submission_spec.get("cluster") or {}
    resources = submission_spec.get("resources") or {}
    runner = submission_spec.get("runner") or {}
    submission = submission_spec.get("submission") or {}

    _require_pattern("run name", submission_spec.get("run_name"), MACHINE_IDENTIFIER_PATTERN)
    _require_pattern("partition", cluster.get("partition"), SLURM_IDENTIFIER_PATTERN)
    _require_pattern("account", cluster.get("account"), SLURM_IDENTIFIER_PATTERN)
    _require_pattern("walltime", resources.get("walltime"), WALLTIME_PATTERN)
    if submission.get("attempt_comment"):
        _require_pattern(
            "submission comment",
            submission.get("attempt_comment"),
            re.compile(r"^bmd_attempt:[0-9a-f-]{36}$"),
        )

    for key in (
        "run_dir",
        "remote_script",
        "log_out",
        "log_err",
        "slurm_out",
        "slurm_err",
    ):
        _require_safe_remote_path(f"paths.{key}", paths.get(key))
    for key in ("python", "stdout", "stderr"):
        _require_safe_remote_path(f"runner.{key}", runner.get(key))
    for key in (
        "script_name",
        "submission_spec_name",
        "backend_package_dir",
        "backend_init_name",
        "execution_module_name",
    ):
        _require_relative_filename(f"runner.{key}", runner.get(key))
    for module_name in (submission_spec.get("modules") or {}).get("load") or []:
        _require_pattern("module name", module_name, SAFE_MODULE_PATTERN)


def _require_pattern(name: str, value, pattern: re.Pattern) -> str:
    text = str(value or "")
    if len(text) > 256 or not pattern.fullmatch(text):
        raise ValueError(f"Unsafe or invalid {name} in submission specification.")
    return text


def _require_safe_remote_path(name: str, value) -> str:
    path = str(value or "")
    if len(path) > 4096 or not SAFE_REMOTE_PATH_PATTERN.fullmatch(path):
        raise ValueError(f"Unsafe or invalid {name} in submission specification.")
    if ".." in path.split("/"):
        raise ValueError(f"Parent path components are not allowed in {name}.")
    return path


def _require_relative_filename(name: str, value) -> str:
    filename = str(value or "")
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise ValueError(f"Unsafe or invalid {name} in submission specification.")
    return _require_pattern(name, filename, RELATIVE_FILENAME_PATTERN)


def _module_lines(submission_spec: dict) -> list[str]:
    modules = submission_spec.get("modules", {})
    lines = []

    if modules.get("purge_first"):
        lines.append("module purge")

    for module_name in modules.get("load", []):
        lines.append(f"module load {shlex.quote(str(module_name))}")

    return lines


def _submission_env_exports(submission_spec: dict) -> list[str]:
    environment = submission_spec.get("environment", {})
    return [
        _shell_export(key, environment.get(key))
        for key in SUBMISSION_ENV_KEYS
        if environment.get(key) is not None
    ]


def _run_job_launcher_python() -> str:
    return r'''
import json
import os
import sys
import traceback

from backend.execution import run_submission


spec_path = os.environ.get("BMD_SUBMISSION_SPEC")
if not spec_path:
    spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.json")
print("[runner] submission spec:", spec_path)
with open(spec_path, "r", encoding="utf-8") as handle:
    spec = json.load(handle)

try:
    run_submission(spec)
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''.strip()


def build_run_job_script(submission_spec: dict | None = None) -> str:
    del submission_spec
    return _run_job_launcher_python()


def build_execution_module_source() -> str:
    return build_backend_module_sources()[REMOTE_EXECUTION_MODULE_FILENAME]


def build_backend_module_sources() -> dict[str, str]:
    return build_runtime_package_sources()


def _run_job_path(submission_spec: dict) -> str:
    paths = submission_spec["paths"]
    runner = submission_spec["runner"]
    script_name = runner["script_name"]

    if str(script_name).startswith("/"):
        return script_name

    return posixpath.join(paths["run_dir"], script_name)


def _submission_json_path(submission_spec: dict) -> str:
    paths = submission_spec["paths"]
    runner = submission_spec["runner"]
    spec_name = runner.get("submission_spec_name", SUBMISSION_SPEC_FILENAME)

    if str(spec_name).startswith("/"):
        return spec_name

    return posixpath.join(paths["run_dir"], spec_name)


def _remote_backend_dir(submission_spec: dict) -> str:
    paths = submission_spec["paths"]
    runner = submission_spec["runner"]
    backend_dir = runner.get("backend_package_dir", REMOTE_BACKEND_PACKAGE_DIR)

    if str(backend_dir).startswith("/"):
        return backend_dir

    return posixpath.join(paths["run_dir"], backend_dir)


def _execution_module_path(submission_spec: dict) -> str:
    runner = submission_spec["runner"]
    module_name = runner.get("execution_module_name", REMOTE_EXECUTION_MODULE_FILENAME)

    if str(module_name).startswith("/"):
        return module_name

    return posixpath.join(_remote_backend_dir(submission_spec), module_name)


def _backend_module_paths(submission_spec: dict) -> dict[str, str]:
    paths = {
        filename: posixpath.join(_remote_backend_dir(submission_spec), filename)
        for filename in build_backend_module_sources()
    }
    paths[REMOTE_EXECUTION_MODULE_FILENAME] = _execution_module_path(submission_spec)
    return paths


def _backend_init_path(submission_spec: dict) -> str:
    runner = submission_spec["runner"]
    init_name = runner.get("backend_init_name", REMOTE_BACKEND_INIT_FILENAME)

    if str(init_name).startswith("/"):
        return init_name

    return posixpath.join(_remote_backend_dir(submission_spec), init_name)


def build_job_body(submission_spec: dict) -> str:
    paths = submission_spec["paths"]
    runner = submission_spec["runner"]
    run_job_path = _run_job_path(submission_spec)
    submission_json_path = _submission_json_path(submission_spec)
    backend_module_paths = _backend_module_paths(submission_spec)
    backend_module_checks = "\n".join(
        (
            f"test -f {shlex.quote(path)} || "
            f"{{ {_shell_log(f'[sbatch] Missing execution module at {path}')}; exit 1; }}"
        )
        if filename == REMOTE_EXECUTION_MODULE_FILENAME
        else (
            f"test -f {shlex.quote(path)} || "
            f"{{ {_shell_log(f'[sbatch] Missing backend module {filename} at {path}')}; exit 1; }}"
        )
        for filename, path in backend_module_paths.items()
    )
    mp_api_key = submission_spec.get("preflight", {}).get("mp_api_key")
    mp_export = (
        f"export MP_API_KEY={shlex.quote(str(mp_api_key))}"
        if mp_api_key
        else 'echo "[sbatch] MP_API_KEY not provided for this run."'
    )

    attempt_id = (submission_spec.get("submission") or {}).get("attempt_id")

    return f"""
mkdir -p {shlex.quote(paths["run_dir"])}
cd {shlex.quote(paths["run_dir"])}
{mp_export}
{_shell_export("BMD_SUBMISSION_ATTEMPT_ID", attempt_id) if attempt_id else "true"}
{_shell_log(f'[sbatch] Using partition={submission_spec["cluster"]["partition"]} account={submission_spec["cluster"]["account"]}')}
echo "[sbatch] VASP_CMD=$VASP_CMD"
echo "[sbatch] SLURM_NTASKS=${{SLURM_NTASKS:-<unset>}}"
which srun 2>/dev/null || true; srun --version 2>/dev/null | head -n1 || true
which {shlex.quote(runner["python"])} || true
{shlex.quote(runner["python"])} --version
test -f {shlex.quote(run_job_path)} || {{ {_shell_log(f'[sbatch] Missing run_job.py at {run_job_path}')}; exit 1; }}
test -f {shlex.quote(submission_json_path)} || {{ {_shell_log(f'[sbatch] Missing submission.json at {submission_json_path}')}; exit 1; }}
{backend_module_checks}
export BMD_SUBMISSION_SPEC={shlex.quote(submission_json_path)}
{_shell_log(f'[sbatch] Runner stdout: {runner["stdout"]}')}
{_shell_log(f'[sbatch] Runner stderr: {runner["stderr"]}')}
{_shell_log(f'[sbatch] SLURM stdout: {paths["slurm_out"]}')}
{_shell_log(f'[sbatch] SLURM stderr: {paths["slurm_err"]}')}
{shlex.quote(runner["python"])} -u {shlex.quote(runner["script_name"])} 1>{shlex.quote(runner["stdout"])} 2>{shlex.quote(runner["stderr"])}
echo "Done. Logs:"; echo {shlex.quote(runner["stdout"])}; echo {shlex.quote(runner["stderr"])}
""".lstrip()


def build_sbatch_script(submission_spec: dict) -> str:
    _validate_submission_script_fields(submission_spec)
    paths = submission_spec["paths"]
    run_name = submission_spec["run_name"]
    cluster = submission_spec["cluster"]
    resources = submission_spec["resources"]
    module_lines = _module_lines(submission_spec)
    exports = _submission_env_exports(submission_spec)
    job_body = build_job_body(submission_spec)
    module_block = "\n".join(module_lines)
    export_block = "\n".join(exports)
    attempt_comment = (submission_spec.get("submission") or {}).get("attempt_comment")
    comment_line = f"#SBATCH --comment={attempt_comment}\n" if attempt_comment else ""

    header = (
        f"#SBATCH -p {cluster['partition']}\n"
        f"#SBATCH --account={cluster['account']}\n"
        f"#SBATCH --job-name={run_name}\n"
        f"{comment_line}"
        f"#SBATCH --time={resources['walltime']}\n"
        f"#SBATCH --nodes={int(resources['nodes'])}\n"
        f"#SBATCH --ntasks={int(resources['ntasks'])}\n"
        f"#SBATCH --mem={int(resources['mem_gb'])}G\n"
        f"#SBATCH --output={paths['slurm_out']}\n"
        f"#SBATCH --error={paths['slurm_err']}"
    )

    body = f"""#!/usr/bin/env bash
{header}
set -e -o pipefail

# -- module loads (compute node, verbose diagnostics enabled) --
if ! type module >/dev/null 2>&1; then
    source /etc/bashrc
fi
{module_block}
echo "[sbatch debug] PATH=$PATH"
module list
which vasp_std
which mpirun

# -- reasonable stack size for VASP --
ulimit -s 81920 || true

# -- propagate environment expected by the runner --
{export_block}

# -- POTCAR sanity (warn and show layout) --
echo "PMG_VASP_PSP_DIR=$PMG_VASP_PSP_DIR"
ls -ld "$PMG_VASP_PSP_DIR"/POT_* >/dev/null 2>&1 || echo "[warn] No POT_* dir found under $PMG_VASP_PSP_DIR"

# -- runner body --
{job_body.strip()}
"""
    return body.rstrip() + "\n"


def build_submission_script_artifact(submission_spec: dict) -> dict:
    """Return the exact UTF-8 SLURM script artifact used for display and upload."""

    text = build_sbatch_script(submission_spec)
    payload = text.encode("utf-8")
    return {
        "text": text,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _submission_stage_rows(submission_spec: dict) -> list[dict]:
    provenance = submission_spec.get("provenance") or {}
    stages = []
    for stage in (provenance.get("vasp") or {}).get("stages") or []:
        modifiers = list(stage.get("modifiers") or [])
        stage_name = stage_display_name(stage.get("stage_type"))
        theory_name = theory_display_name(stage.get("theory"))
        modifier_names = [modifier_display_name(modifier) for modifier in modifiers]
        display_name = f"{theory_name} {stage_name}"
        if modifier_names:
            display_name += f" + {', '.join(modifier_names)}"
        stages.append({
            "index": stage.get("index"),
            "display_name": display_name,
            "stage_type": stage.get("stage_type"),
            "theory": stage.get("theory"),
            "modifiers": modifiers,
            "executable": stage.get("executable"),
        })
    return stages


def build_standalone_slurm_example(submission_spec: dict) -> dict:
    """Build a direct POWER/VASP example from authoritative submission data.

    This is deliberately not the BMD submission artifact. Multi-stage workflows
    are represented as an illustrative stage sequence because BMD's atomate2
    runner manages the state transfer between those VASP invocations.
    """

    cluster = submission_spec.get("cluster") or {}
    resources = submission_spec.get("resources") or {}
    modules = submission_spec.get("modules") or {}
    stages = _submission_stage_rows(submission_spec)
    if not stages or any(not stage.get("executable") for stage in stages):
        raise ValueError("Standalone SLURM example requires resolved stage executables.")

    lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH -p {cluster.get('partition')}",
        f"#SBATCH --account={cluster.get('account')}",
        f"#SBATCH -J {submission_spec.get('label')}",
        f"#SBATCH --time={resources.get('walltime')}",
        f"#SBATCH --nodes={int(resources.get('nodes'))}",
        f"#SBATCH --ntasks={int(resources.get('ntasks'))}",
        f"#SBATCH --mem={int(resources.get('mem_gb'))}G",
        "",
        "ulimit -s 81920",
        "",
    ]
    if modules.get("purge_first"):
        lines.append("module purge")
    lines.extend(f"module load {module_name}" for module_name in modules.get("load") or [])
    lines.append("")

    illustrative = len(stages) > 1
    if illustrative:
        lines.extend([
            f"# Illustrative only: BMD workflow contains {len(stages)} sequential VASP stages.",
            "# This outline is not a standalone runnable multi-stage workflow.",
            "",
        ])
        for stage in stages:
            lines.extend([
                f"# Stage {stage['index']} - {stage['display_name']}",
                f"mpirun -n $SLURM_NTASKS {stage['executable']}",
                "",
            ])
        lines.append("# BMD Compute manages structure and result transfer between stages.")
    else:
        lines.append(f"mpirun -n $SLURM_NTASKS {stages[0]['executable']}")

    return {
        "text": "\n".join(lines).rstrip() + "\n",
        "kind": "illustrative_workflow" if illustrative else "standalone",
        "description": (
            "Illustrative direct-execution outline for this multi-stage workflow. "
            "BMD Compute manages the transfers between stages."
            if illustrative
            else "Equivalent standalone script for running this calculation directly on POWER."
        ),
    }


def build_submission_summary(submission_spec: dict) -> dict:
    """Build a human-readable summary from authoritative submission data."""

    provenance = submission_spec.get("provenance") or {}
    stages = _submission_stage_rows(submission_spec)

    script = ((provenance.get("execution") or {}).get("submission_script") or {})
    return {
        "job_name": submission_spec.get("run_name"),
        "resources": {
            "partition": (submission_spec.get("cluster") or {}).get("partition"),
            "account": (submission_spec.get("cluster") or {}).get("account"),
            "walltime": (submission_spec.get("resources") or {}).get("walltime"),
            "nodes": (submission_spec.get("resources") or {}).get("nodes"),
            "tasks": (submission_spec.get("resources") or {}).get("ntasks"),
            "memory_gb": (submission_spec.get("resources") or {}).get("mem_gb"),
        },
        "execution": {
            "run_directory": (submission_spec.get("paths") or {}).get("run_dir"),
            "runner_script": (submission_spec.get("runner") or {}).get("script_name"),
            "python": (submission_spec.get("runner") or {}).get("python"),
            "runtime_environment": (submission_spec.get("environment") or {}).get(
                "ATOMATE2_REMOTE_ENV"
            ),
            "stages": stages,
        },
        "environment": {
            "modules": list((submission_spec.get("modules") or {}).get("load") or []),
        },
        "artifact": {
            "status": "available" if script.get("sha256") else "not_generated",
            "path": script.get("path"),
            "sha256": script.get("sha256"),
            "size_bytes": script.get("size_bytes"),
        },
    }


def build_remote_runtime_preflight_source(submission_spec: dict) -> str:
    submission_json_path = _submission_json_path(submission_spec)
    return f"""
import json
import sys
import traceback

try:
    import backend.execution
    import backend.workflows
    from backend.calculations.registry import (
        validate_workflow_spec,
        workflow_spec_from_flow_spec,
    )

    with open({submission_json_path!r}, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    workflow_spec = workflow_spec_from_flow_spec(spec.get("flow_spec") or {{}})
    validate_workflow_spec(workflow_spec)
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("BMD_RUNTIME_PREFLIGHT_OK=backend.execution,backend.workflows")
""".strip()


def remote_preparation_file_groups(submission_spec: dict) -> list[dict]:
    """Return the remote files that must be uploaded before sbatch submission.

    File contents are intentionally kept as data for RemoteRunner file-transfer
    methods. Do not embed these payloads into shell command strings; generated
    inputs can be large for workflows such as hybrid band structures.
    """

    backend_module_sources = build_backend_module_sources()
    submission_script = build_submission_script_artifact(submission_spec)
    backend_module_paths = _backend_module_paths(submission_spec)
    execution_files = [
        {
            "path": backend_module_paths[filename],
            "text": backend_module_sources[filename],
            "mode": 0o640,
        }
        for filename in sorted(backend_module_sources)
    ]
    backend_init_path = _backend_init_path(submission_spec)
    if backend_init_path not in {item["path"] for item in execution_files}:
        execution_files.insert(
            0,
            {
                "path": backend_init_path,
                "text": "",
                "mode": 0o640,
            },
        )

    return [
        {
            "step": "submission.json uploaded",
            "files": [
                {
                    "path": _submission_json_path(submission_spec),
                    "text": json_dumps_for_remote_file(submission_spec),
                    "mode": 0o640,
                    # Readers must never observe a partially written record.
                    "atomic": True,
                },
            ],
        },
        {
            "step": "Execution module uploaded",
            "files": execution_files,
        },
        {
            "step": "run_job.py uploaded",
            "files": [
                {
                    "path": _run_job_path(submission_spec),
                    "text": build_run_job_script(submission_spec),
                    "mode": 0o640,
                },
            ],
        },
        {
            "step": "Submission script written",
            "files": [
                {
                    "path": submission_spec["paths"]["remote_script"],
                    "text": submission_script["text"],
                    "mode": 0o640,
                },
            ],
        },
    ]


def build_submission_command(submission_spec: dict, *, dry_run: bool = False) -> str:
    paths = submission_spec["paths"]
    resources = submission_spec["resources"]
    cluster = submission_spec["cluster"]
    potcar = submission_spec["potcar"]
    submission_json = json_dumps_for_remote_file(submission_spec)
    submission_json_path = _submission_json_path(submission_spec)
    backend_dir = _remote_backend_dir(submission_spec)
    backend_init_path = _backend_init_path(submission_spec)
    backend_module_paths = _backend_module_paths(submission_spec)
    backend_module_sources = build_backend_module_sources()
    run_job_script = build_run_job_script(submission_spec)
    run_job_path = _run_job_path(submission_spec)
    sbatch_script = build_submission_script_artifact(submission_spec)["text"]
    pot_links = " ".join(shlex.quote(link) for link in potcar.get("symlink_targets", []))
    link_command = (
        f"for L in {pot_links}; do ln -sfn {shlex.quote(potcar['target'])} \"$L\"; done"
        if pot_links
        else "true"
    )
    sbatch_line = (
        f"sbatch -p {shlex.quote(str(cluster['partition']))} "
        f"-A {shlex.quote(str(cluster['account']))} "
        f"-N {int(resources['nodes'])} "
        f"-n {int(resources['ntasks'])} "
        f"--mem={int(resources['mem_gb'])}G "
        f"-t {shlex.quote(str(resources['walltime']))} "
        f"--parsable {shlex.quote(paths['remote_script'])}"
    )

    directories = " ".join(
        shlex.quote(path)
        for path in paths.get("directories_to_prepare", [])
    )

    if dry_run:
        return _build_verified_dry_run_command(
            paths,
            potcar,
            submission_json_path,
            submission_json,
            backend_dir,
            backend_init_path,
            backend_module_paths,
            backend_module_sources,
            run_job_path,
            run_job_script,
            sbatch_script,
        )

    command = f"""set -e -o pipefail
mkdir -p {directories}
mkdir -p {shlex.quote(paths['run_dir'])}
mkdir -p {shlex.quote(backend_dir)}
{link_command}
cat > {shlex.quote(submission_json_path)} <<'JSON'
{submission_json}
JSON
test -f {shlex.quote(submission_json_path)}
cat > {shlex.quote(backend_init_path)} <<'PY'
PY
{_backend_module_write_commands(backend_module_paths, backend_module_sources)}
cat > {shlex.quote(run_job_path)} <<'PY'
{run_job_script}
PY
test -f {shlex.quote(run_job_path)}
cat > {shlex.quote(paths['remote_script'])} <<'SBATCH'
{sbatch_script}
SBATCH
test -f {shlex.quote(paths['remote_script'])}
echo "SBATCH_SCRIPT_PATH={paths['remote_script']}"
"""

    return command + f"""\
echo "Submitting with: {sbatch_line}"
set +e
out=$({sbatch_line} 2>&1)
rc=$?
set -e
echo "SBATCH_RAW_OUT=$out"
exit $rc
"""


def _build_verified_dry_run_command(
    paths: dict,
    potcar: dict,
    submission_json_path: str,
    submission_json: str,
    backend_dir: str,
    backend_init_path: str,
    backend_module_paths: dict[str, str],
    backend_module_sources: dict[str, str],
    run_job_path: str,
    run_job_script: str,
    sbatch_script: str,
) -> str:
    parent_dirs = list(paths.get("directories_to_prepare", []))
    run_dir = paths["run_dir"]
    execution_module_path = backend_module_paths[REMOTE_EXECUTION_MODULE_FILENAME]
    parent_dir_commands = "\n".join(
        _verified_mkdir_command(path, "Remote directories prepared")
        for path in parent_dirs
    )
    link_commands = "\n".join(
        _verified_symlink_command(potcar["target"], link, "POTCAR links prepared")
        for link in potcar.get("symlink_targets", [])
    )
    potcar_prep_block = (
        f"{link_commands}\nprep_ok \"POTCAR links prepared\""
        if link_commands
        else ""
    )

    return f"""set -e -o pipefail
prep_fail() {{
    echo "PREP_FAILED_STAGE=$1"
    echo "PREP_FAILED_REASON=$2"
    exit 42
}}
prep_ok() {{
    echo "PREP_OK=$1"
}}
verify_dir() {{
    test -d "$2" || prep_fail "$1" "Expected directory does not exist: $2"
}}
verify_file() {{
    test -f "$2" || prep_fail "$1" "Expected file does not exist: $2"
}}
verify_symlink() {{
    test -L "$3" || prep_fail "$1" "Expected POTCAR symlink does not exist: $3"
    target="$(readlink "$3" 2>/dev/null || true)"
    test "$target" = "$2" || prep_fail "$1" "POTCAR symlink points to $target instead of $2"
}}
{parent_dir_commands}
prep_ok "Remote directories prepared"
{_verified_mkdir_command(run_dir, "Working directory created")}
prep_ok "Working directory created"
{{
cat > {shlex.quote(submission_json_path)} <<'JSON'
{submission_json}
JSON
}} || prep_fail "submission.json uploaded" "Unable to write submission.json: {submission_json_path}"
verify_file "submission.json uploaded" {shlex.quote(submission_json_path)}
prep_ok "submission.json uploaded"
{_verified_mkdir_command(backend_dir, "Execution module uploaded")}
{{
cat > {shlex.quote(backend_init_path)} <<'PY'
PY
{_backend_module_write_commands(backend_module_paths, backend_module_sources, verify=False)}
}} || prep_fail "Execution module uploaded" "Unable to write execution module: {execution_module_path}"
{_backend_module_verify_commands(backend_module_paths, "Execution module uploaded")}
prep_ok "Execution module uploaded"
{{
cat > {shlex.quote(run_job_path)} <<'PY'
{run_job_script}
PY
}} || prep_fail "run_job.py uploaded" "Unable to write run_job.py: {run_job_path}"
verify_file "run_job.py uploaded" {shlex.quote(run_job_path)}
prep_ok "run_job.py uploaded"
{potcar_prep_block}
{{
cat > {shlex.quote(paths['remote_script'])} <<'SBATCH'
{sbatch_script}
SBATCH
}} || prep_fail "Submission script written" "Unable to write submission script: {paths['remote_script']}"
verify_file "Submission script written" {shlex.quote(paths['remote_script'])}
prep_ok "Submission script written"
prep_ok "Ready for submission"
echo DRY RUN
exit 0
"""


def _backend_module_write_commands(
    module_paths: dict[str, str],
    module_sources: dict[str, str],
    *,
    verify: bool = True,
) -> str:
    lines = []
    for filename in sorted(module_sources):
        path = module_paths[filename]
        source = module_sources[filename]
        parent = posixpath.dirname(path)
        if parent:
            lines.append(f"mkdir -p {shlex.quote(parent)}")
        lines.append(
            f"cat > {shlex.quote(path)} <<'PY'\n"
            f"{source}\n"
            "PY"
        )
        if verify:
            lines.append(f"test -f {shlex.quote(path)}")
    return "\n".join(lines)


def _backend_module_verify_commands(module_paths: dict[str, str], stage: str) -> str:
    return "\n".join(
        f"verify_file {shlex.quote(stage)} {shlex.quote(module_paths[filename])}"
        for filename in sorted(module_paths)
    )


def _verified_mkdir_command(path: str, stage: str) -> str:
    quoted_path = shlex.quote(path)
    quoted_stage = shlex.quote(stage)
    reason = shlex.quote(f"Unable to create directory: {path}")
    return (
        f"mkdir -p {quoted_path} || prep_fail {quoted_stage} {reason}\n"
        f"verify_dir {quoted_stage} {quoted_path}"
    )


def _verified_symlink_command(target: str, link: str, stage: str) -> str:
    quoted_target = shlex.quote(target)
    quoted_link = shlex.quote(link)
    quoted_stage = shlex.quote(stage)
    reason = shlex.quote(f"Unable to create POTCAR symlink: {link}")
    return (
        f"ln -sfn {quoted_target} {quoted_link} || prep_fail {quoted_stage} {reason}\n"
        f"verify_symlink {quoted_stage} {quoted_target} {quoted_link}"
    )


def json_dumps_for_remote_file(value: dict) -> str:
    import json

    return json.dumps(value, indent=2)


def parse_sbatch_job_id(output: str) -> str:
    output = output or ""
    patterns = (
        r"(?m)^JOBID=(\d+)\b",
        r"(?m)^SBATCH_RAW_OUT=(\d+)\b",
        r"(?m)^(\d+)\b",
        r"Submitted batch job\s+(\d+)",
        r"\b(\d+)\b",
    )

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return match.group(1)

    raise RuntimeError("Could not parse job id from sbatch output above.")


def create_submission_spec(
    flow_spec: dict,
    *,
    structure=None,
    label: str = "vasp_run",
    timestamp: str | None = None,
    nodes=None,
    ntasks=None,
    mem_gb=None,
    walltime: str | None = None,
    username: str | None = None,
    flows_dir: str | None = None,
    logs_dir: str | None = None,
    potcars_dir: str | None = None,
    remote_env_dir: str | None = None,
    remote_python: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    vasp_cmd: str | None = None,
    jobflow_config_file: str | None = None,
    env: dict | None = None,
    mp_api_key: str | None = None,
    submission_attempt_id: str | None = None,
) -> dict:
    """
    Describe a pending calculation using notebook submission defaults.

    This prepares the state that the notebook assembled immediately before
    writing a remote script and calling sbatch. It performs no submission,
    SSH, SLURM, remote execution, or Jobflow execution.
    """

    env_values = env if env is not None else os.environ
    flow_spec_copy = deepcopy(flow_spec)
    workflow_spec = workflow_spec_from_flow_spec(flow_spec_copy)
    calculation_spec = calculation_spec_from_workflow_spec(workflow_spec)
    flow_spec_copy["workflow_spec"] = workflow_spec.to_dict()
    if calculation_spec is not None:
        flow_spec_copy["calculation_spec"] = calculation_spec.to_dict()
        flow_spec_copy["workflow"] = legacy_workflow_from_spec(calculation_spec)
    else:
        flow_spec_copy.pop("calculation_spec", None)
        flow_spec_copy["workflow"] = flow_spec_copy.get("workflow") or "custom_workflow"
    resource_defaults = default_resources_for_workflow_spec(workflow_spec)

    sanitized_label = sanitize_label(label)
    run_timestamp = canonical_run_timestamp(timestamp)
    run_name = f"{sanitized_label}-{run_timestamp}"
    stage_directories = workflow_stage_directories(workflow_spec)
    result_stage_directory = workflow_result_stage_directory(workflow_spec)

    resolved_username = username or NOTEBOOK_DEFAULTS["username"]
    resolved_flows_dir = flows_dir or NOTEBOOK_DEFAULTS["flows_dir"]
    resolved_logs_dir = logs_dir or NOTEBOOK_DEFAULTS["logs_dir"]
    resolved_potcars_dir = potcars_dir or NOTEBOOK_DEFAULTS["PMG_VASP_PSP_DIR"]
    resolved_remote_env_dir = remote_env_dir or NOTEBOOK_DEFAULTS["remote_env_dir"]
    resolved_remote_python = remote_python or NOTEBOOK_DEFAULTS.get("remote_python") or DEFAULT_REMOTE_PYTHON
    if remote_env_dir is not None and remote_python is None:
        resolved_remote_python = posixpath.join(_env_bin(resolved_remote_env_dir), "python")

    resolved_nodes = _coerce_int(nodes, resource_defaults["nodes"])
    resolved_ntasks = _coerce_int(ntasks, resource_defaults["ntasks"])
    resolved_mem_gb = _coerce_int(mem_gb, resource_defaults["mem_gb"])
    resolved_walltime = walltime or resource_defaults["walltime"]

    resolved_partition = validate_queue(
        partition
        or env_values.get("SLURM_PARTITION")
        or env_values.get("SLURM_DEFAULT_PARTITION")
        or DEFAULT_PARTITION
    )
    resolved_account = account or env_values.get("SLURM_ACCOUNT") or DEFAULT_ACCOUNT

    resolved_vasp_cmd = _first_nonempty(
        env_values.get("VASP_CMD"),
        vasp_cmd,
        NOTEBOOK_DEFAULTS["VASP_CMD"],
        DEFAULT_VASP_LAUNCHER,
    )
    resolved_jobflow_config = _first_nonempty(
        env_values.get("JOBFLOW_CONFIG_FILE"),
        jobflow_config_file,
        NOTEBOOK_DEFAULTS["JOBFLOW_CONFIG_FILE"],
    )
    resolved_pmg_psp_dir = _first_nonempty(
        env_values.get("PMG_VASP_PSP_DIR"),
        resolved_potcars_dir,
    )

    potcar_functional = flow_spec_copy.get("potcar_functional", DEFAULT_POTCAR_FUNCTIONAL)
    potcar_record = executable_potcar_record(
        structure,
        workflow_spec,
        potcar_functional=potcar_functional,
    )
    potcar_target = posixpath.join(resolved_potcars_dir, potcar_functional)
    use_shared_potcars = _uses_shared_potcar_repository(resolved_potcars_dir)
    if use_shared_potcars:
        potcar_links = []
    else:
        potcar_links = [
            posixpath.join(resolved_potcars_dir, link_name)
            for link_name in POTCAR_LINK_MAP.get(potcar_functional, [])
        ]
    directories_to_prepare = [
        resolved_flows_dir,
        resolved_logs_dir,
    ]
    if potcar_links:
        directories_to_prepare.extend(
            [
                resolved_potcars_dir,
                potcar_target,
            ]
        )

    run_dir = posixpath.join(resolved_flows_dir, run_name)
    stage_dirs = {
        stage_directory: posixpath.join(run_dir, stage_directory)
        for stage_directory in stage_directories
    }
    log_out = posixpath.join(resolved_logs_dir, f"{run_name}.out")
    log_err = posixpath.join(resolved_logs_dir, f"{run_name}.err")
    slurm_out = posixpath.join(resolved_logs_dir, f"{run_name}.slurm.out")
    slurm_err = posixpath.join(resolved_logs_dir, f"{run_name}.slurm.err")
    remote_script = posixpath.join(resolved_flows_dir, f"{run_name}.sbatch.sh")
    directories_to_prepare.extend(stage_dirs.values())

    spec = {
        # Contract header for bmd_compute.submission v1; see backend.run_records.
        **submission_record_header(),
        "status": "pending",
        "label": sanitized_label,
        "run_name": run_name,
        "created_at": run_timestamp,
        "flow_spec": flow_spec_copy,
        "paths": {
            "run_dir": run_dir,
            "remote_script": remote_script,
            "logs_dir": resolved_logs_dir,
            "flows_dir": resolved_flows_dir,
            "potcars_dir": resolved_potcars_dir,
            "log_out": log_out,
            "log_err": log_err,
            "slurm_out": slurm_out,
            "slurm_err": slurm_err,
            "stage_dirs": stage_dirs,
            "result_dir": (
                stage_dirs[result_stage_directory]
                if result_stage_directory
                else run_dir
            ),
            "directories_to_prepare": directories_to_prepare,
        },
        "cluster": {
            "ssh_config_host": NOTEBOOK_DEFAULTS["ssh_config_host"],
            "remote_host": NOTEBOOK_DEFAULTS["remote_host"],
            "username": resolved_username,
            "key_file": NOTEBOOK_DEFAULTS["key_file"],
            "port": NOTEBOOK_DEFAULTS["port"],
            "partition": resolved_partition,
            "account": resolved_account,
        },
        "resources": {
            "nodes": resolved_nodes,
            "ntasks": resolved_ntasks,
            "mem_gb": resolved_mem_gb,
            "walltime": resolved_walltime,
        },
        "environment": {
            "VASP_CMD": resolved_vasp_cmd,
            "JOBFLOW_CONFIG_FILE": resolved_jobflow_config,
            "PMG_VASP_PSP_DIR": resolved_pmg_psp_dir,
            "CUSTODIAN_NO_GZIP": "1",
            "ATOMATE2_VASP_ZIP_FILES": "False",
            "ATOMATE2_REMOTE_ENV": resolved_remote_env_dir,
        },
        "modules": {
            "purge_first": True,
            "load": list(MODULES),
        },
        "runner": {
            "working_directory": run_dir,
            "python": resolved_remote_python,
            "script_name": "run_job.py",
            "submission_spec_name": SUBMISSION_SPEC_FILENAME,
            "backend_package_dir": REMOTE_BACKEND_PACKAGE_DIR,
            "backend_init_name": REMOTE_BACKEND_INIT_FILENAME,
            "execution_module_name": REMOTE_EXECUTION_MODULE_FILENAME,
            "stdout": log_out,
            "stderr": log_err,
        },
        "potcar": {
            "functional": potcar_functional,
            **potcar_record,
            "repository": "shared" if use_shared_potcars else "private",
            "target": potcar_target,
            "symlink_targets": potcar_links,
        },
        "preflight": {
            "requires_remote_structure_check": flow_spec_copy.get("structure", {}).get("type") == "path",
            "mp_api_key_provided": bool((mp_api_key or "").strip()),
        },
        "submission": {
            "method": "sbatch",
            "ready": True,
            "submitted": False,
            "reason": "Specification only; no submission has been performed.",
        },
    }
    spec["provenance"] = build_submission_provenance(spec)
    spec = initialize_submission_attempt(
        spec,
        attempt_id=submission_attempt_id,
    )
    spec["submission"]["identity_token"] = create_submission_identity_token(
        run_timestamp,
        spec["submission"]["attempt_id"],
    )
    script_artifact = build_submission_script_artifact(spec)
    spec["provenance"]["execution"]["submission_script"] = {
        "path": spec["paths"]["remote_script"],
        "builder": "backend.submission.build_sbatch_script",
        "encoding": "utf-8",
        "hash_algorithm": "sha256",
        "sha256": script_artifact["sha256"],
        "size_bytes": script_artifact["size_bytes"],
    }
    return spec


__all__ = [
    "DEFAULT_ACCOUNT",
    "DEFAULT_PARTITION",
    "DEFAULT_RESOURCES",
    "MODULES",
    "NOTEBOOK_DEFAULTS",
    "POTCAR_LINK_MAP",
    "REMOTE_BACKEND_INIT_FILENAME",
    "REMOTE_BACKEND_MODULE_FILENAMES",
    "REMOTE_BACKEND_PACKAGE_DIR",
    "REMOTE_EXECUTION_MODULE_FILENAME",
    "REMOTE_RUNTIME_PREFLIGHT_STEP",
    "REMOTE_RUNTIME_PREFLIGHT_TIMEOUT_S",
    "SUBMISSION_SPEC_FILENAME",
    "build_backend_module_sources",
    "build_execution_module_source",
    "build_remote_runtime_preflight_source",
    "build_sbatch_script",
    "build_standalone_slurm_example",
    "build_submission_summary",
    "build_submission_script_artifact",
    "canonical_run_timestamp",
    "create_submission_identity_token",
    "build_run_job_script",
    "build_submission_command",
    "create_submission_spec",
    "default_resources_for_calculation_spec",
    "default_resources_for_workflow_spec",
    "default_resources_for_workflow",
    "initialize_submission_attempt",
    "new_submission_attempt_id",
    "parse_sbatch_job_id",
    "remote_preparation_file_groups",
    "sanitize_label",
    "submission_attempt_comment",
    "submission_attempt_fingerprint",
    "submission_attempt_id",
    "verify_submission_identity_token",
    "submission_attempt_metadata",
]
