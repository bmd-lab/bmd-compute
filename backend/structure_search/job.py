"""The SLURM job of a managed structure search (USPEX step 4D).

Three parts:

* **Resource acknowledgement** (service side): the plan's resource request is
  a provisional estimate (an exclusive node for up to the planned walltime).
  :func:`build_resource_acknowledgement` freezes the user's explicit
  acknowledgement of exactly that request for one attempt and plan, with a
  SHA-256 over the whole record; :func:`verify_resource_acknowledgement`
  refuses any other.
* **Job script** (service side): :func:`build_job_script` renders the
  versioned batch script (schema ``bmd_compute.structure_search_job_script``
  v1) deterministically from the staged attempt's submission spec, its frozen
  plan, the acknowledgement, the uploaded runtime package and the requested
  modules. Same inputs, same bytes.
* **Job entry point** (POWER side): :func:`job_main` runs inside the job,
  from the runtime package uploaded with the attempt after the embedded guard
  has verified it. It re-verifies the staged inputs, runs the USPEX runtime
  check, then starts the USPEX master and the existing launch loop
  (``bridge.launcher.run_search``) with the existing bridge, launch settings
  and candidate-validity check, and records what happened in
  ``search_status.json``. It never submits anything.

The batch script never runs twice in one run directory: it first creates
``job_started`` exclusively and exits (status 84) without touching anything
if that file exists, so a duplicate or requeued job cannot overwrite a
search. It also asks SLURM not to requeue (``--no-requeue``).

Run directory after a job (``/bmd/bmdguest/searches/<run_name>/``)::

    bmd_search_job.sbatch   the job script (uploaded at promotion, step 4D-2)
    job_started             the SLURM job ID that started (created exclusively)
    search_status.json      execution record, rewritten atomically
    runtime_check.json      the job-start USPEX runtime check
    uspex_master.log        the USPEX master's output
    bridge_spool/           the launch bridge's spool
    slurm-<job id>.out      the batch script's output
    search/                 the USPEX run directory (CalcFolds, results*)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

JOB_SCRIPT_SCHEMA = "bmd_compute.structure_search_job_script"
JOB_SCRIPT_SCHEMA_VERSION = 1
JOB_STATUS_SCHEMA = "bmd_compute.structure_search_job_status"
JOB_STATUS_SCHEMA_VERSION = 1
ACKNOWLEDGEMENT_SCHEMA = "bmd_compute.structure_search_resource_acknowledgement"
ACKNOWLEDGEMENT_SCHEMA_VERSION = 1
ACKNOWLEDGEMENT_STATEMENT = (
    "I acknowledge that these SLURM resources are BMD Compute's provisional estimate for this structure "
    "search, not an approved allocation, and that the job may hold them for up to the requested walltime."
)

JOB_SCRIPT_NAME = "bmd_search_job.sbatch"
STATUS_FILE = "search_status.json"
STARTED_MARKER = "job_started"
SETUP_LOG = "job_setup.log"  # the shell's system and module set-up output, and Python start-up tracebacks
RUNTIME_CHECK_FILE = "runtime_check.json"
MASTER_LOG = "uspex_master.log"
SPOOL_DIRNAME = "bridge_spool"
SLURM_OUTPUT = "slurm-%j.out"
JOB_NAME_PREFIX = "bmds-"
REQUEST_MARKER = "BMD_SEARCH_JOB_REQUEST_B64 = "
PYTHON_END = "BMD_SEARCH_JOB_PYTHON_END"

# Exit statuses of the batch script. The USPEX master's own status is passed
# through when it ran (0 finished, 123 stopped by the launcher).
EXIT_REFUSED_INPUTS = 80
EXIT_RUNTIME_CHECK_FAILED = 81
EXIT_LAUNCH_SETUP_FAILED = 82
EXIT_SETUP_FAILED = 83
EXIT_ALREADY_STARTED = 84
EXIT_RUNTIME_PACKAGE_REFUSED = 85

# Directives BMD adds; the plan's own resource directives must not set them.
_BMD_DIRECTIVES = ("--job-name", "-J", "--comment", "--chdir", "-D", "--output", "-o", "--error", "-e",
                   "--requeue", "--no-requeue")
_SAFE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
_SAFE_MODULE = re.compile(r"^[A-Za-z0-9._+/-]+$")
_ATTEMPT = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class JobDefinitionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- resource acknowledgement ---------------------------------------------------------------


def _resources_of(plan) -> dict[str, Any]:
    resources = plan.record["resources"]
    return {
        "status": resources.get("status"),
        "approved": resources.get("approved"),
        "request": dict(resources["request"]),
        "sbatch_directives": list(resources["sbatch_directives"]),
    }


def resource_request_view(plan) -> dict[str, Any]:
    """What a user acknowledges: the statement and the plan's exact provisional resource request."""

    return {"statement": ACKNOWLEDGEMENT_STATEMENT, **_resources_of(plan)}


def resource_request_sha256(plan) -> str:
    """``sha256:`` of :func:`resource_request_view`; the client echoes it to acknowledge exactly this request."""

    return "sha256:" + _sha256_text(_canonical(resource_request_view(plan)))


def build_resource_acknowledgement(plan, *, attempt_id: str, principal: str, acknowledged_at: str) -> dict[str, Any]:
    """The frozen record of a user's explicit acknowledgement of the plan's resource request."""

    if not _ATTEMPT.match(str(attempt_id)):
        raise JobDefinitionError("invalid_attempt", "The attempt ID must be a canonical lowercase UUID.")
    if not str(principal or "").strip() or not str(acknowledged_at or "").strip():
        raise JobDefinitionError("invalid_acknowledgement", "An acknowledgement needs a principal and a time.")
    record = {
        "schema": ACKNOWLEDGEMENT_SCHEMA,
        "schema_version": ACKNOWLEDGEMENT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "plan_digest": plan.digest,
        "principal": str(principal),
        "acknowledged_at": str(acknowledged_at),
        "statement": ACKNOWLEDGEMENT_STATEMENT,
        "resources": _resources_of(plan),
    }
    record["acknowledgement_sha256"] = _sha256_text(_canonical(record))
    return record


def verify_resource_acknowledgement(acknowledgement: Any, *, plan, attempt_id: str) -> str:
    """Refuse an acknowledgement that is not exactly for this attempt, plan and request. Returns its SHA-256."""

    if not isinstance(acknowledgement, Mapping):
        raise JobDefinitionError("acknowledgement_required", "Submission needs an explicit resource acknowledgement.")
    record = dict(acknowledgement)
    claimed = record.pop("acknowledgement_sha256", None)
    if (record.get("schema"), record.get("schema_version")) != (ACKNOWLEDGEMENT_SCHEMA, ACKNOWLEDGEMENT_SCHEMA_VERSION):
        raise JobDefinitionError("acknowledgement_invalid", "The resource acknowledgement has an unknown schema.")
    if claimed != _sha256_text(_canonical(record)):
        raise JobDefinitionError("acknowledgement_invalid", "The resource acknowledgement was altered.")
    if record.get("attempt_id") != attempt_id or record.get("plan_digest") != plan.digest:
        raise JobDefinitionError("acknowledgement_mismatch",
                                 "The resource acknowledgement is for another attempt or plan.")
    if record.get("statement") != ACKNOWLEDGEMENT_STATEMENT or record.get("resources") != _resources_of(plan):
        raise JobDefinitionError("acknowledgement_mismatch",
                                 "The resource acknowledgement does not cover this plan's resource request.")
    if not str(record.get("principal") or "").strip() or not str(record.get("acknowledged_at") or "").strip():
        raise JobDefinitionError("acknowledgement_invalid", "The resource acknowledgement has no principal or time.")
    return claimed


# -- job script (service side) ------------------------------------------------------------------


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise JobDefinitionError(code, message)


def _directive_option(line: str) -> str:
    return line[len("#SBATCH "):].split("=", 1)[0].split(" ", 1)[0]


def build_job_script(
    *,
    spec: Mapping[str, Any],
    plan,
    acknowledgement: Mapping[str, Any],
    runtime_sources: Mapping[str, str],
    manifest: Mapping[str, str],
    modules: Sequence[str],
    guard_module: str = "runtime_package_guard.py",
) -> str:
    """The versioned batch script of one staged search. Deterministic: same inputs, same bytes."""

    from backend.runtime_package import runtime_package_manifest_digest
    from backend.structure_search.staging import BRIDGE_RELATIVE_PATH, staging_layout
    from backend.submission import SUBMISSION_ATTEMPT_COMMENT_PREFIX

    attempt_id = str((spec.get("submission") or {}).get("attempt_id") or "")
    _require(bool(_ATTEMPT.match(attempt_id)), "invalid_attempt", "The spec has no canonical attempt ID.")
    search = spec.get("structure_search") or {}
    _require(search.get("plan_digest") == plan.digest, "plan_mismatch", "The spec is for another plan.")
    bridge_sha256 = plan.record["execution"]["bridge"]["sha256"]
    _require(search.get("bridge_sha256") == bridge_sha256 and manifest.get(BRIDGE_RELATIVE_PATH) == bridge_sha256,
             "bridge_mismatch", "The runtime package's bridge is not the plan's.")
    _require(dict(manifest) == {name: _sha256_text(text) for name, text in sorted(runtime_sources.items())},
             "manifest_mismatch", "The manifest does not describe the runtime package sources.")
    acknowledgement_sha256 = verify_resource_acknowledgement(acknowledgement, plan=plan, attempt_id=attempt_id)

    paths = spec["paths"]
    layout = staging_layout(spec["run_name"], workspace_dir=paths["flows_dir"], logs_dir=paths["logs_dir"])
    _require(layout["bridge_path"] == search.get("bridge_path") and layout["search_dir"] == search.get("search_dir"),
             "layout_mismatch", "The spec's search paths are not its run directory's.")
    python = str((spec.get("runner") or {}).get("python") or "")
    for name, value in (("run directory", layout["run_dir"]), ("Python", python)):
        _require(bool(_SAFE_PATH.match(value)) and ".." not in value.split("/"), "unsafe_path",
                 f"The {name} path {value!r} is not a plain absolute path.")
    modules = [str(name) for name in modules]
    _require(bool(modules) and all(_SAFE_MODULE.match(name) for name in modules), "invalid_modules",
             "Every module name must be a plain module name.")

    directives = list(plan.record["resources"]["sbatch_directives"])
    for line in directives:
        _require(line.startswith("#SBATCH ") and "\n" not in line and "\r" not in line, "invalid_directive",
                 f"Not a single #SBATCH directive: {line!r}.")
        _require(_directive_option(line) not in _BMD_DIRECTIVES, "invalid_directive",
                 f"The plan's resource directives must not set {_directive_option(line)}.")
    run_dir = layout["run_dir"]
    directives += [
        f"#SBATCH --job-name={JOB_NAME_PREFIX}{attempt_id[:8]}",
        f"#SBATCH --comment={SUBMISSION_ATTEMPT_COMMENT_PREFIX}{attempt_id}",
        f"#SBATCH --chdir={run_dir}",
        f"#SBATCH --output={run_dir}/{SLURM_OUTPUT}",
        "#SBATCH --no-requeue",
    ]

    request = {
        "schema": JOB_SCRIPT_SCHEMA,
        "schema_version": JOB_SCRIPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "run_name": spec["run_name"],
        "run_dir": run_dir,
        "package_dir": layout["package_dir"],
        "search_dir": layout["search_dir"],
        "bridge_path": layout["bridge_path"],
        "plan_digest": plan.digest,
        "manifest": dict(sorted(manifest.items())),
        "modules_requested": modules,
        "acknowledgement_sha256": acknowledgement_sha256,
    }
    payload = base64.b64encode(_canonical(request).encode("utf-8")).decode("ascii")
    guard = runtime_sources[guard_module].strip()
    _require(PYTHON_END not in guard, "invalid_guard", "The guard source contains the script's end marker.")
    q = shlex.quote
    module_loads = "\n".join(
        f'module load {q(name)} >>"$SETUP_LOG" 2>&1 || fail {EXIT_SETUP_FAILED} setup_failed "module load {name} failed"'
        for name in modules
    )

    lines = [
        "#!/bin/bash",
        *directives,
        "#",
        f"# BMD Compute structure-search job ({JOB_SCRIPT_SCHEMA} v{JOB_SCRIPT_SCHEMA_VERSION}).",
        "# Generated from the attempt's frozen plan; do not edit. It never submits anything and never",
        "# runs twice in one run directory.",
        f"# attempt:                  {attempt_id}",
        f"# run:                      {spec['run_name']}",
        f"# plan digest:              {plan.digest}",
        f"# runtime package:          sha256:{runtime_package_manifest_digest(manifest)}",
        f"# resource acknowledgement: sha256:{acknowledgement_sha256}",
        f"# modules requested:        {' '.join(modules)}",
        "set -u",
        "umask 027",
        f"RUN_DIR={q(run_dir)}",
        f"PYTHON={q(python)}",
        f"ATTEMPT={q(attempt_id)}",
        'JOB_ID="${SLURM_JOB_ID:-}"',
        '[[ "$JOB_ID" =~ ^[0-9]+$ ]] || JOB_ID=unknown',
        "say() { printf '== %s %s\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"$*\"; }",
        "# fail STATUS OUTCOME MESSAGE: record a failure before Python started (fixed texts only), then stop.",
        "fail() {",
        '  local tmp="$RUN_DIR/.search_status.json.setup.$$"',
        "  printf '{\"schema\": \"%s\", \"schema_version\": %s, \"attempt_id\": \"%s\", \"slurm_job_id\": \"%s\", "
        "\"execution\": {\"outcome\": \"%s\", \"message\": \"%s\"}, \"scientific_validity\": {\"status\": "
        "\"not_assessed\"}, \"finalists\": {\"status\": \"not_started\"}}\\n' "
        f"{q(JOB_STATUS_SCHEMA)} {JOB_STATUS_SCHEMA_VERSION} \"$ATTEMPT\" \"$JOB_ID\" \"$2\" \"$3\" > \"$tmp\" "
        f"&& mv -f \"$tmp\" \"$RUN_DIR/{STATUS_FILE}\"",
        '  say "STOPPED: $3"',
        '  exit "$1"',
        "}",
        f'say "structure-search job $JOB_ID for attempt $ATTEMPT on $(hostname)"',
        "# 1. Never run twice in one run directory (a duplicate or requeued job stops here, changing nothing).",
        f'if ! ( set -o noclobber; printf "%s\\n" "$JOB_ID" > "$RUN_DIR/{STARTED_MARKER}" ) 2>/dev/null; then',
        f'  say "STOPPED: $RUN_DIR/{STARTED_MARKER} exists or cannot be created; this search is not started again"',
        f"  exit {EXIT_ALREADY_STARTED}",
        "fi",
        f'cd "$RUN_DIR" || fail {EXIT_SETUP_FAILED} setup_failed "the run directory is not accessible"',
        "# From here until Python starts, an unexpected stop of this shell (an error or an exit inside the system",
        "# or module scripts) is recorded, never over an existing status record; recording never fails the job.",
        f'SETUP_LOG="$RUN_DIR/{SETUP_LOG}"',
        "{ : >>\"$SETUP_LOG\"; } 2>/dev/null || SETUP_LOG=/dev/null  # a log that cannot be written never stops the job",
        'PHASE="clearing the submitting environment"',
        "on_shell_exit() {",
        "  local rc=$?",
        f'  [ -e "$RUN_DIR/{STATUS_FILE}" ] && return 0',
        '  local tmp="$RUN_DIR/.search_status.json.shell.$$"',
        "  printf '{\"schema\": \"%s\", \"schema_version\": %s, \"attempt_id\": \"%s\", \"slurm_job_id\": \"%s\", "
        "\"execution\": {\"outcome\": \"setup_failed\", \"code\": \"shell_stopped\", \"exit_status\": %s, "
        "\"message\": \"the job shell stopped during %s, before Python started; see %s\"}, "
        "\"scientific_validity\": {\"status\": \"not_assessed\"}, \"finalists\": {\"status\": \"not_started\"}}\\n' "
        f"{q(JOB_STATUS_SCHEMA)} {JOB_STATUS_SCHEMA_VERSION} \"$ATTEMPT\" \"$JOB_ID\" \"$rc\" \"${{PHASE:-set-up}}\" "
        f"{q(SETUP_LOG)} 2>/dev/null > \"$tmp\" "
        f'&& ln "$tmp" "$RUN_DIR/{STATUS_FILE}" 2>/dev/null  # ln never replaces an existing file',
        '  rm -f "$tmp" 2>/dev/null',
        f'  say "STOPPED: the job shell stopped during ${{PHASE:-set-up}} (exit $rc) before Python started; see {SETUP_LOG}" '
        "2>/dev/null >&9 || true",
        "  return 0  # the shell keeps its own exit status",
        "}",
        "exec 9>&1  # the SLURM output, for the line above even if the shell stops inside a redirected command",
        "trap on_shell_exit EXIT",
        "# 2. Nothing from the submitting environment may redirect the job.",
        "for name in $(env | sed -n 's/^\\(BMD_[A-Za-z0-9_]*\\)=.*/\\1/p'); do unset \"$name\"; done",
        "unset JOBFLOW_CONFIG_FILE ATOMATE2_CONFIG_FILE PYTHONPATH PYTHONHOME PYTHONSTARTUP",
        "# 3. The VASP modules BMD Compute configures (recorded above and in search_status.json). The system and",
        "# module scripts are not written for set -u (an unset variable there would end this shell silently), so",
        "# they run without it, and their output goes to the set-up log rather than nowhere.",
        "set +u",
        'PHASE="system shell initialisation (/etc/profile)"',
        'if ! type module >/dev/null 2>&1; then source /etc/profile >>"$SETUP_LOG" 2>&1 || true; fi',
        'PHASE="module set-up"',
        f'type module >/dev/null 2>&1 || fail {EXIT_SETUP_FAILED} setup_failed "the module command is not available"',
        'module purge >>"$SETUP_LOG" 2>&1 || true',
        module_loads,
        "set -u",
        'PHASE="starting Python"',
        f'[ -x "$PYTHON" ] || fail {EXIT_SETUP_FAILED} setup_failed "the runtime Python is not executable"',
        "exec 9>&-",
        "# 4. The job entry point, from the runtime package uploaded with this attempt, once verified.",
        f"exec \"$PYTHON\" - \"${{LOADEDMODULES:-}}\" <<'{PYTHON_END}'",
        "import sys",
        'if sys.path and sys.path[0] == "":',
        "    del sys.path[0]",
        "import base64, json, os",
        "def _bmd_unexpected(where):",
        "    # An unexpected Python failure: its traceback goes to the set-up log (and still to stderr, by the",
        "    # caller's re-raise); a status record is written only if none exists. Never raises.",
        "    try:",
        "        import traceback",
        "        request = globals().get('REQUEST') or {}",
        "        run_dir = request.get('run_dir') or os.getcwd()  # the shell started Python in the run directory",
        f"        with open(os.path.join(run_dir, {SETUP_LOG!r}), 'a', encoding='utf-8') as handle:",
        "            handle.write('== Python failure while ' + where + '\\n')",
        "            traceback.print_exc(file=handle)",
        f"        path = os.path.join(run_dir, {STATUS_FILE!r})",
        "        if not os.path.exists(path):",
        "            record = {'schema': " + repr(JOB_STATUS_SCHEMA) + ", 'schema_version': "
        + str(JOB_STATUS_SCHEMA_VERSION) + ", 'attempt_id': request.get('attempt_id'),"
        " 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'execution': {'outcome': 'setup_failed',"
        " 'code': 'python_startup_failed', 'message': 'Python failed while ' + where + '; see "
        + SETUP_LOG + "'}, 'scientific_validity': {'status': 'not_assessed'}, 'finalists': {'status': 'not_started'}}",
        "            temporary = path + '.python.' + str(os.getpid())",
        "            with open(temporary, 'w', encoding='utf-8') as handle:",
        "                json.dump(record, handle, indent=2, sort_keys=True)",
        "            try:",
        "                os.link(temporary, path)  # never over a record written meanwhile",
        "            finally:",
        "                os.unlink(temporary)",
        "    except Exception:",
        "        pass",
        f"{REQUEST_MARKER}{payload!r}",
        "try:",
        "    REQUEST = json.loads(base64.b64decode(BMD_SEARCH_JOB_REQUEST_B64))",
        "except Exception:",
        "    _bmd_unexpected('reading the job request')",
        "    raise",
        "# BMD Compute runtime package guard (embedded).",
        guard,
        "# End of embedded runtime package guard.",
        "def _bmd_refused(message):",
        f"    path = os.path.join(REQUEST['run_dir'], {STATUS_FILE!r})",
        "    record = {'schema': " + repr(JOB_STATUS_SCHEMA) + ", 'schema_version': " + str(JOB_STATUS_SCHEMA_VERSION)
        + ", 'attempt_id': REQUEST['attempt_id'], 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),"
        " 'execution': {'outcome': 'runtime_package_refused', 'message': message},"
        " 'scientific_validity': {'status': 'not_assessed'}, 'finalists': {'status': 'not_started'}}",
        "    with open(path + '.tmp', 'w', encoding='utf-8') as handle:",
        "        json.dump(record, handle, indent=2, sort_keys=True)",
        "    os.replace(path + '.tmp', path)",
        "sys.path.insert(0, REQUEST['run_dir'])",
        "try:",
        "    bmd_install_verified_runtime_package(REQUEST['package_dir'], REQUEST['manifest'])",
        "except Exception as exc:",
        "    _bmd_refused('The uploaded runtime package does not match its manifest; nothing was run.')",
        "    print(exc, file=sys.stderr)",
        f"    sys.exit({EXIT_RUNTIME_PACKAGE_REFUSED})",
        "try:",
        "    from backend.structure_search.job import job_main",
        "except Exception:",
        "    _bmd_unexpected('importing the job entry point')",
        "    raise",
        "try:",
        "    _bmd_status = job_main(REQUEST, loaded_modules=sys.argv[1] if len(sys.argv) > 1 else '')",
        "except Exception:",
        "    _bmd_unexpected('running the job entry point')",
        "    raise",
        "sys.exit(_bmd_status)",
        PYTHON_END,
    ]
    return "\n".join(lines) + "\n"


def job_script_sha256(script: str) -> str:
    return _sha256_text(script)


def request_from_job_script(script: str) -> dict[str, Any]:
    """The request embedded in a job script (for checks and tests)."""

    import ast

    for line in script.splitlines():
        if line.startswith(REQUEST_MARKER):
            return json.loads(base64.b64decode(ast.literal_eval(line[len(REQUEST_MARKER):])))
    raise JobDefinitionError("invalid_script", "No job request in the script.")


# -- job entry point (POWER side) ----------------------------------------------------------------


class _Status:
    """search_status.json, rewritten atomically at every phase."""

    def __init__(self, request: Mapping[str, Any], *, loaded_modules: str, environ: Mapping[str, str]):
        import platform
        import time

        self.path = Path(request["run_dir"]) / STATUS_FILE
        self.record: dict[str, Any] = {
            "schema": JOB_STATUS_SCHEMA,
            "schema_version": JOB_STATUS_SCHEMA_VERSION,
            "attempt_id": request["attempt_id"],
            "run_name": request.get("run_name"),
            "plan_digest": request["plan_digest"],
            "acknowledgement_sha256": request.get("acknowledgement_sha256"),
            "slurm_job_id": environ.get("SLURM_JOB_ID"),
            "host": platform.node(),
            "modules": {
                "requested": list(request.get("modules_requested") or []),
                "loaded": [name for name in str(loaded_modules or "").split(":") if name],
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_at": None,
            "execution": {"outcome": "verifying_inputs"},
            "scientific_validity": {"status": "not_assessed"},
            "finalists": {
                "status": "not_started",
                "note": "Finalist selection and adaptive relaxation are a later stage.",
            },
        }

    def write(self, **execution) -> None:
        self.record["execution"] = {**self.record["execution"], **execution}
        temporary = self.path.with_name(f".{STATUS_FILE}.{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.record, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def finish(self, **execution) -> None:
        import time

        self.record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.write(**execution)


def _step_validity_summary(search_dir: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for path in sorted(search_dir.rglob("bmd_step_validity.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                classification = str(json.loads(line).get("classification"))
            except ValueError:
                classification = "unreadable"
            counts[classification] = counts.get(classification, 0) + 1
    return counts


def job_main(request: Mapping[str, Any], *, loaded_modules: str = "", environ: Mapping[str, str] | None = None) -> int:
    """Run one staged structure search inside its SLURM job. Returns the batch script's exit status."""

    import sys
    import threading

    from backend.structure_search.bridge.launcher import (
        LauncherSettings,
        NodeMemoryUnavailable,
        _slurm_memory_mb,
        allocated_node_memory,
        install_stop_signal_handlers,
        run_search,
    )
    from backend.structure_search.bridge.master import build_master_command
    from backend.structure_search.bridge.protocol import EXIT_STOPPED
    from backend.structure_search.runtime import search_runtime_check, settings_from_record
    from backend.structure_search.search_plan import (
        ContentAddressedPotcarIdentity,
        SearchPlanError,
        load_plan_text,
        verify_execution_binding,
    )

    environ = dict(os.environ if environ is None else environ)
    status = _Status(request, loaded_modules=loaded_modules, environ=environ)
    stop = threading.Event()
    install_stop_signal_handlers(stop)
    run_dir = Path(request["run_dir"])
    search_dir = Path(request["search_dir"])

    def stopped() -> int:
        status.finish(outcome="stopped", message="The job was stopped before USPEX started.")
        return EXIT_STOPPED

    # 1. The staged inputs are still exactly what the frozen plan renders.
    status.write(outcome="verifying_inputs")
    try:
        plan = load_plan_text((search_dir / "search_plan.json").read_text(encoding="utf-8"))
        if plan.digest != request["plan_digest"]:
            raise SearchPlanError("The staged plan is not the approved plan.", code="plan_changed")
        verify_execution_binding(
            search_dir,
            bridge_path=request["bridge_path"],
            require_current=False,
            potcars="staged",
            potcar_identity=ContentAddressedPotcarIdentity(),
        )
    except (SearchPlanError, OSError, ValueError) as exc:
        status.finish(outcome="refused_inputs", code=getattr(exc, "code", type(exc).__name__),
                      message="The staged inputs no longer match the frozen plan; USPEX was not started.")
        return EXIT_REFUSED_INPUTS
    if stop.is_set():
        return stopped()

    # 2. The USPEX installation is the one the plan was made for.
    status.write(outcome="checking_runtime")
    runtime_record = plan.record["runtime"]
    check = search_runtime_check(runtime_record)
    with open(run_dir / RUNTIME_CHECK_FILE, "w", encoding="utf-8") as handle:
        json.dump(check, handle, indent=2, sort_keys=True)
    settings, problems = settings_from_record(runtime_record)
    if check["status"] != "passed" or settings is None or problems:
        status.finish(outcome="runtime_check_failed", problems=list(check["problems"]) + list(problems),
                      message="The USPEX runtime check failed; USPEX was not started.")
        return EXIT_RUNTIME_CHECK_FAILED
    if stop.is_set():
        return stopped()

    # 3. The existing master command, launch loop, bridge and step-validity check.
    try:
        spool = run_dir / SPOOL_DIRNAME
        master = build_master_command(
            singularity=settings.singularity,
            image=settings.sandbox_dir,
            search_dir=search_dir,
            xdg_cache_home=settings.cache_dir,
            spool_dir=spool,
            bridge_timeout_seconds=settings.bridge_timeout_seconds,
            bind=plan.record["execution"]["bridge"]["bind"],
        )
        environment = {
            key: value for key, value in environ.items()
            if not key.startswith("BMD_") and key not in ("JOBFLOW_CONFIG_FILE", "ATOMATE2_CONFIG_FILE")
        }
        environment["I_MPI_PMI_LIBRARY"] = settings.i_mpi_pmi_library
        # The allocation must be the frozen plan's: a shared node only with srun_steps, the planned
        # number of tasks, and (for an explicit --mem) exactly the planned memory.
        request = plan.record["resources"]["request"]
        if request.get("exclusive") is False and settings.launch_mode != "srun_steps":
            raise ValueError(f"a shared-node search needs srun_steps, not {settings.launch_mode}")
        planned_tasks = settings.parallel_relaxations * settings.ranks_per_relaxation
        if request.get("ntasks") != planned_tasks:
            raise ValueError(f"the plan requests {request.get('ntasks')} tasks for {planned_tasks} ranks")
        slurm_tasks = environ.get("SLURM_NTASKS")
        if slurm_tasks is not None and slurm_tasks != str(planned_tasks):
            raise ValueError(f"SLURM allocated {slurm_tasks} tasks; the plan requests {planned_tasks}")
        # The launcher's memory policy, fed with what SLURM allocated (fails closed).
        node_memory = allocated_node_memory(environment) if settings.launch_mode == "srun_steps" else None
        planned_mb = _slurm_memory_mb(str(request.get("mem", "")))
        if node_memory is not None and planned_mb and node_memory.mb != planned_mb:
            raise ValueError(f"SLURM allocated {node_memory.mb} MB of memory; the plan requests {planned_mb} MB")
        launcher_settings = LauncherSettings(
            spool_dir=spool,
            search_dir=search_dir,
            max_parallel=settings.parallel_relaxations,
            ranks=settings.ranks_per_relaxation,
            environment=environment,
            launch_mode=settings.launch_mode,
            sockets=settings.node_sockets,
            cores_per_socket=settings.cores_per_socket,
            node_memory_mb=node_memory.mb if node_memory else None,
            step_validity=True,
        )
    except NodeMemoryUnavailable as exc:
        status.finish(outcome="launch_setup_failed", code="node_memory_unavailable",
                      message=f"The memory allocated to this job could not be established: {exc} "
                              "USPEX was not started.")
        return EXIT_LAUNCH_SETUP_FAILED
    except (ValueError, KeyError) as exc:
        status.finish(outcome="launch_setup_failed", message=f"The launch settings are not usable: {exc}")
        return EXIT_LAUNCH_SETUP_FAILED

    status.write(
        outcome="running",
        launch={
            "mode": launcher_settings.launch_mode,
            "parallel_relaxations": launcher_settings.max_parallel,
            "ranks_per_relaxation": launcher_settings.ranks,
            "step_mem": launcher_settings.step_mem,
            "memory_margin_fraction": launcher_settings.memory_margin_fraction,
            "node_memory": None if node_memory is None else {
                "mb": node_memory.mb, "source": node_memory.source, "detail": node_memory.detail},
            "step_validity": True,
            "master_argv": list(master.argv),
        },
    )
    master_status = run_search(master.argv, master.env, master.cwd, launcher_settings,
                               master_log=run_dir / MASTER_LOG, stop_event=stop)
    outcome = "uspex_finished" if master_status == 0 else "stopped" if master_status == EXIT_STOPPED else "uspex_failed"
    status.record["scientific_validity"] = {
        "status": "not_assessed",
        "step_validity_verdicts": _step_validity_summary(search_dir),
        "note": "Each relaxation step was held to the candidate-validity rules by the launcher; "
                "the search's results are not assessed here.",
    }
    status.record["isolation"] = {name: name in sys.modules for name in ("jobflow", "atomate2")}
    status.finish(
        outcome=outcome,
        master_exit_status=master_status,
        results=sorted(path.name for path in search_dir.glob("results*") if path.is_dir()),
    )
    return master_status


__all__ = [
    "ACKNOWLEDGEMENT_SCHEMA",
    "ACKNOWLEDGEMENT_STATEMENT",
    "EXIT_ALREADY_STARTED",
    "EXIT_REFUSED_INPUTS",
    "EXIT_RUNTIME_CHECK_FAILED",
    "EXIT_RUNTIME_PACKAGE_REFUSED",
    "EXIT_SETUP_FAILED",
    "JOB_SCRIPT_NAME",
    "JOB_SCRIPT_SCHEMA",
    "JOB_STATUS_SCHEMA",
    "JobDefinitionError",
    "build_job_script",
    "build_resource_acknowledgement",
    "resource_request_sha256",
    "resource_request_view",
    "job_main",
    "job_script_sha256",
    "request_from_job_script",
    "verify_resource_acknowledgement",
]
