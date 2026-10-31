from __future__ import annotations

import gzip
import json
import logging
import posixpath
import threading
import tempfile
import time
import traceback
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Callable

from backend.calculations.models import CalculationSpec, WorkflowSpec
from backend.calculations.registry import (
    calculation_result_stage_directory,
    calculation_spec_from_flow_spec,
    workflow_result_stage_directory,
    workflow_spec_from_flow_spec,
)
from backend.config import DEFAULT_LOGS_DIR, DEFAULT_REMOTE_PYTHON, bmd_debug_enabled
from backend.remote import (
    REMOTE_OPERATION_BUSY_MESSAGE,
    RemoteOperationBusy,
    RemoteRunner,
)
from backend.remote_runtime import (
    connected_remote_runner,
    connection_profile_from_submission_spec,
    default_connection_profile,
)
from backend.remote_result_parser import (
    GEOMETRY_RELAXATION_VASPRUN_KEY,
    PRECEDING_STATIC_OUTCAR_KEY,
    parse_result_paths,
)
from backend.workflow_results import (
    PHONON_RESULT_FILES,
    workflow_has_phonon_results,
    workflow_result_file_keys,
)


RESULT_FILES = {
    "contcar": "CONTCAR",
    "outcar": "OUTCAR",
    "vasprun": "vasprun.xml",
    "doscar": "DOSCAR",
    "kpoints": "KPOINTS",
    **PHONON_RESULT_FILES,
    # Optional: the preceding PBE static's OUTCAR, for the Fermi reference of a
    # non-self-consistent PBE band structure.
    PRECEDING_STATIC_OUTCAR_KEY: "OUTCAR",
    # Optional: vasprun.xml of the final Geometry Optimisation stage when it is
    # not the reported stage, for the cell-convergence diagnostic.
    GEOMETRY_RELAXATION_VASPRUN_KEY: "vasprun.xml",
}
LOGGER = logging.getLogger(__name__)
RESULTS_CACHE_MAX_ENTRIES = 4
REMOTE_RESULT_PARSE_TIMEOUT_S = 900
REMOTE_RESULT_STDOUT_MAX_BYTES = 64 * 1024 * 1024
REMOTE_RESULT_JSON_START = "__BMD_RESULTS_JSON_START__"
REMOTE_RESULT_JSON_END = "__BMD_RESULTS_JSON_END__"
_RESULTS_CACHE_LOCK = threading.Lock()
_RESULTS_CACHE: OrderedDict[str, dict] = OrderedDict()
_RESULTS_CACHE_JOB_INDEX: dict[str, str] = {}


class RemoteResultExtractionError(RuntimeError):
    """Raised when the remote parser command cannot return compact results."""


def _log_results(message: str) -> None:
    LOGGER.debug(message)


def monitoring_indicates_success(monitoring_result: dict | None) -> bool:
    if not monitoring_result or monitoring_result.get("status") != "success":
        return False

    summary = str(monitoring_result.get("summary") or "").upper()
    state = str(monitoring_result.get("slurm_state") or "").upper()
    exit_code = monitoring_result.get("exit_code")

    if summary != "SUCCESS" or not state.startswith("COMPLETED"):
        return False

    return exit_code is None or str(exit_code).startswith("0:0")


def load_results_for_completed_job(
    monitoring_result: dict | None,
    *,
    submission_spec: dict | None = None,
    runner_factory: Callable[[], RemoteRunner] | None = None,
    parser: Callable[[dict, dict], dict] | None = None,
    cache: bool | None = None,
) -> dict | None:
    _log_results("ENTER results")
    _log_results("Check monitoring success")
    if not monitoring_indicates_success(monitoring_result):
        _log_results("Monitoring did not report SUCCESS")
        _log_results("RETURN results")
        return None
    _log_results("Monitoring reported SUCCESS")

    use_cache = (parser is None) if cache is None else bool(cache)
    local_location = (
        results_location_from_submission_spec(submission_spec)
        if submission_spec is not None
        else None
    )
    if use_cache:
        cached = cached_completed_result(monitoring_result, local_location)
        if cached is not None:
            _log_results("RETURN cached results")
            return cached

    _log_results("Build results connection profile")
    profile = (
        connection_profile_from_submission_spec(submission_spec)
        if submission_spec is not None
        else default_connection_profile()
    )
    _log_results("Results connection profile built")

    try:
        _log_results("Connect results RemoteRunner")
        with connected_remote_runner(
            profile=profile,
            runner_factory=runner_factory,
        ) as runner:
            _log_results("Results RemoteRunner connected")
            _log_results("Locate run directory")
            location = resolve_results_location(
                runner,
                monitoring_result,
                submission_spec,
            )
            run_dir = location["run_dir"]
            if not run_dir:
                _log_results("Run directory missing")
                _log_results("RETURN results")
                return _failure_result(
                    "Results Discovery",
                    "Completed job has no canonical BMD run directory.",
                    "Check that BMD Compute wrote the remote job state record for this SLURM job.",
                )
            _log_results("Run directory found")

            if use_cache:
                cached = cached_completed_result(monitoring_result, location)
                if cached is not None:
                    _log_results("RETURN cached results")
                    return cached

            _log_results("Build direct result paths")
            output_dir = location["output_dir"]
            paths = result_file_paths(
                output_dir,
                extra_file_keys=location["workflow_result_file_keys"],
            )
            _log_results("Direct result paths built")
            _log_results("Verify result files")
            missing = missing_result_files(runner, paths)
            _log_results("Result file verification complete")
            for key, remote_path in (location.get("optional_result_file_paths") or {}).items():
                # Optional outputs (phonons): include those present; the parser
                # reports any that are missing.
                if runner.is_file(remote_path):
                    paths[key] = remote_path
            if missing:
                _log_results("Required result file missing")
                _log_results("RETURN results")
                return _failure_result(
                    "Results Discovery",
                    f"Completed calculation is missing required output files: {', '.join(missing)}.",
                    "Check the BMD run directory and confirm VASP wrote CONTCAR, OUTCAR, and vasprun.xml.",
                    files=paths,
                )

            parse_context = _results_parse_context(monitoring_result, location)
            if parser is None:
                _log_results("Parse result files remotely")
                result = extract_remote_vasp_result(
                    runner,
                    paths,
                    parse_context,
                    python=location.get("remote_python"),
                )
                _log_results("Remote result parsing complete")
                files = {
                    key: {
                        "path": value,
                    }
                    for key, value in paths.items()
                }
            else:
                _log_results("Read result files")
                files = read_result_files(runner, paths)
                _log_results("Result file reads complete")
    except RemoteResultExtractionError as exc:
        _log_results("Remote results parsing failed")
        LOGGER.exception("Remote result parser failed.")
        _log_results("RETURN results")
        return _failure_result(
            "Results Parsing",
            _clean_message(exc),
            "Check that the remote pymatgen environment can parse the completed VASP outputs.",
            files=locals().get("paths", {}),
            exception=exc.__cause__ or exc,
        )
    except RemoteOperationBusy as exc:
        _log_results("Remote results busy")
        _log_results("RETURN results")
        return _failure_result(
            "Remote Capacity",
            REMOTE_OPERATION_BUSY_MESSAGE,
            "Please try again in a few seconds.",
        )
    except Exception as exc:
        _log_results("Results retrieval failed")
        LOGGER.exception("Results retrieval failed.")
        _log_results("RETURN results")
        return _failure_result(
            "Results Retrieval",
            _clean_message(exc),
            "Check remote file permissions and try refreshing monitoring.",
            exception=exc,
        )

    if parser is not None:
        parse_context = _results_parse_context(monitoring_result, location)
        try:
            _log_results("Parse result files")
            result = parser(files, parse_context)
            _log_results("Result file parsing complete")
        except Exception as exc:
            _log_results("Results parsing failed")
            LOGGER.exception("Local result parsing failed.")
            _log_results("RETURN results")
            return _failure_result(
                "Results Parsing",
                _clean_message(exc),
                "Check that pymatgen is installed and that the VASP output files are complete.",
                files={key: value["path"] for key, value in files.items()},
                exception=exc,
            )

    result.setdefault("status", "success")
    result.setdefault("title", "Results Summary")
    result.setdefault("run_dir", run_dir)
    result.setdefault("workdir", output_dir)
    result.setdefault("job_id", monitoring_result.get("job_id"))
    result.setdefault("files", {key: value["path"] for key, value in files.items()})
    result.setdefault("visualizations", [])
    if use_cache:
        store_completed_result(monitoring_result, location, result)
    _log_results("RETURN results")
    return result


def results_location_from_submission_spec(submission_spec: dict | None) -> dict:
    if not submission_spec:
        return _empty_results_location()

    run_dir = _run_dir_from_submission_or_state(submission_spec, {})
    if not run_dir:
        return _empty_results_location()

    workflow_spec = workflow_spec_from_submission_spec(submission_spec)
    calculation_spec = calculation_spec_from_submission_spec(submission_spec)
    output_dir = result_output_dir_from_submission_spec(run_dir, submission_spec)
    optional_paths = {}
    phonon_dirs = phonon_result_directories(run_dir, submission_spec, workflow_spec)
    if phonon_dirs is not None:
        # VASP results come from the relax that produced the phonon structure;
        # the phonon stage itself holds Phonopy outputs, not a VASP run.
        output_dir = phonon_dirs["vasp_dir"]
        optional_paths = {
            key: posixpath.join(phonon_dirs["analysis_dir"], filename)
            for key, filename in PHONON_RESULT_FILES.items()
        }
    static_outcar = preceding_static_outcar_path(run_dir, workflow_spec)
    if static_outcar is not None:
        optional_paths[PRECEDING_STATIC_OUTCAR_KEY] = static_outcar
    relaxation_vasprun = final_relaxation_vasprun_path(run_dir, workflow_spec, output_dir)
    if relaxation_vasprun is not None:
        optional_paths[GEOMETRY_RELAXATION_VASPRUN_KEY] = relaxation_vasprun
    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "optional_result_file_paths": optional_paths,
        "calculation_spec": calculation_spec,
        "workflow_spec": workflow_spec,
        "workflow_result_file_keys": workflow_result_file_keys(
            workflow_spec or calculation_spec
        ),
        "remote_python": (
            (submission_spec.get("runner") or {}).get("python")
            or DEFAULT_REMOTE_PYTHON
        ),
    }


def preceding_static_outcar_path(run_dir: str, workflow_spec) -> str | None:
    """The OUTCAR of the static before a final PBE (non-self-consistent) band structure."""

    stages = list(getattr(workflow_spec, "stages", None) or ())
    if len(stages) < 2:
        return None
    final, previous = stages[-1], stages[-2]
    if (
        getattr(final.stage_type, "value", None) != "band_structure"
        or getattr(final.theory, "value", None) != "pbe"
        or getattr(previous.stage_type, "value", None) != "static"
    ):
        return None
    from backend.calculations.registry import workflow_stage_directories

    directories = workflow_stage_directories(workflow_spec)
    return posixpath.join(run_dir.rstrip("/"), directories[-2], "OUTCAR")


def final_relaxation_vasprun_path(run_dir: str, workflow_spec, output_dir: str) -> str | None:
    """vasprun.xml of the last Geometry Optimisation stage, if not the reported stage."""

    stages = list(getattr(workflow_spec, "stages", None) or ())
    relax_indices = [
        index
        for index, stage in enumerate(stages)
        if getattr(stage.stage_type, "value", None) == "relax"
    ]
    if not relax_indices or len(stages) < 2:
        return None
    # Only calculations planned with the BMD Standard Geometry Relaxation
    # (plane-wave cutoff record) carry the diagnostic; older runs are unchanged.
    if "plane_wave_cutoff" not in dict(getattr(stages[relax_indices[-1]], "options", None) or {}):
        return None
    from backend.calculations.registry import workflow_stage_directories

    directories = workflow_stage_directories(workflow_spec)
    if not directories:
        return None
    path = posixpath.join(run_dir.rstrip("/"), directories[relax_indices[-1]], "vasprun.xml")
    if path == posixpath.join(str(output_dir).rstrip("/"), "vasprun.xml"):
        return None
    return path


def resolve_results_location(
    runner: RemoteRunner,
    monitoring_result: dict,
    submission_spec: dict | None,
) -> dict:
    state = {}
    resolved_spec = submission_spec

    if resolved_spec is None:
        job_id = str(monitoring_result.get("job_id") or "").strip()
        if not job_id:
            return _empty_results_location()

        state_path = remote_job_state_path(job_id)
        _log_results("Locate BMD remote job state")
        if not runner.is_file(state_path):
            _log_results("BMD remote job state missing")
            return _empty_results_location()
        _log_results("BMD remote job state found")

        _log_results("Read BMD remote job state")
        payload = runner.read_text(state_path)
        _log_results("BMD remote job state read")
        state = json.loads(payload)
        resolved_spec = state.get("submission_spec")

    run_dir = _run_dir_from_submission_or_state(resolved_spec, state)
    if not run_dir:
        return _empty_results_location()

    resolved_spec_with_run_dir = {
        **(resolved_spec or {}),
        "paths": {
            **((resolved_spec or {}).get("paths") or {}),
            "run_dir": run_dir,
        },
    }
    return results_location_from_submission_spec(resolved_spec_with_run_dir)


def results_cache_key(
    monitoring_result: dict | None,
    location: dict | None,
) -> str | None:
    job_id = str((monitoring_result or {}).get("job_id") or "").strip()
    run_dir = str((location or {}).get("run_dir") or "").strip()
    output_dir = str((location or {}).get("output_dir") or "").strip()
    if not job_id or not run_dir:
        return None
    return "|".join((job_id, run_dir, output_dir))


def cached_completed_result(
    monitoring_result: dict | None,
    location: dict | None,
) -> dict | None:
    key = results_cache_key(monitoring_result, location)
    job_id = str((monitoring_result or {}).get("job_id") or "").strip()
    if key is None:
        if not job_id:
            return None
        with _RESULTS_CACHE_LOCK:
            key = _RESULTS_CACHE_JOB_INDEX.get(job_id)
            if key is None:
                return None

    with _RESULTS_CACHE_LOCK:
        value = _RESULTS_CACHE.get(key)
        if value is None:
            return None
        _RESULTS_CACHE.move_to_end(key)
        return deepcopy(value)


def store_completed_result(
    monitoring_result: dict | None,
    location: dict | None,
    result: dict,
) -> None:
    key = results_cache_key(monitoring_result, location)
    if key is None or result.get("status") != "success":
        return
    job_id = str((monitoring_result or {}).get("job_id") or "").strip()

    with _RESULTS_CACHE_LOCK:
        _RESULTS_CACHE[key] = deepcopy(result)
        _RESULTS_CACHE.move_to_end(key)
        if job_id:
            _RESULTS_CACHE_JOB_INDEX[job_id] = key
        while len(_RESULTS_CACHE) > RESULTS_CACHE_MAX_ENTRIES:
            removed_key, _ = _RESULTS_CACHE.popitem(last=False)
            for indexed_job_id, indexed_key in list(_RESULTS_CACHE_JOB_INDEX.items()):
                if indexed_key == removed_key:
                    _RESULTS_CACHE_JOB_INDEX.pop(indexed_job_id, None)


def clear_results_cache() -> None:
    with _RESULTS_CACHE_LOCK:
        _RESULTS_CACHE.clear()
        _RESULTS_CACHE_JOB_INDEX.clear()


def results_cache_info() -> dict:
    with _RESULTS_CACHE_LOCK:
        return {
            "size": len(_RESULTS_CACHE),
            "max_entries": RESULTS_CACHE_MAX_ENTRIES,
            "keys": list(_RESULTS_CACHE),
            "job_index_size": len(_RESULTS_CACHE_JOB_INDEX),
        }


def resolve_results_run_dir(
    runner: RemoteRunner,
    monitoring_result: dict,
    submission_spec: dict | None,
) -> str:
    return resolve_results_location(
        runner,
        monitoring_result,
        submission_spec,
    )["run_dir"]


def _run_dir_from_submission_or_state(
    submission_spec: dict | None,
    state: dict,
) -> str:
    if submission_spec is not None:
        run_dir = submission_spec.get("paths", {}).get("run_dir")
        if run_dir:
            return str(run_dir)

    return str(state.get("run_dir") or "")


def result_stage_directory_from_submission_spec(submission_spec: dict | None) -> str | None:
    if not submission_spec:
        return None

    workflow_spec = workflow_spec_from_submission_spec(submission_spec)
    if workflow_spec is not None:
        return workflow_result_stage_directory(workflow_spec)

    calculation_spec = calculation_spec_from_submission_spec(submission_spec)
    if calculation_spec is None:
        return None

    return calculation_result_stage_directory(calculation_spec)


def calculation_spec_from_submission_spec(
    submission_spec: dict | None,
) -> CalculationSpec | None:
    if not submission_spec:
        return None

    try:
        return calculation_spec_from_flow_spec(
            submission_spec.get("flow_spec")
        )
    except Exception:
        return None


def workflow_spec_from_submission_spec(
    submission_spec: dict | None,
) -> WorkflowSpec | None:
    if not submission_spec:
        return None

    try:
        return workflow_spec_from_flow_spec(
            submission_spec.get("flow_spec")
        )
    except Exception:
        return None


def result_includes_dos_from_submission_spec(submission_spec: dict | None) -> bool:
    workflow_spec = workflow_spec_from_submission_spec(submission_spec)
    calculation_spec = calculation_spec_from_submission_spec(submission_spec)
    return "doscar" in workflow_result_file_keys(workflow_spec or calculation_spec)


def phonon_result_directories(
    run_dir: str,
    submission_spec: dict | None,
    workflow_spec: WorkflowSpec | None,
) -> dict | None:
    """Relax and Phonopy-analysis directories of a phonon workflow, or None."""

    if workflow_spec is None or not workflow_has_phonon_results(workflow_spec):
        return None
    from backend.calculations.registry import workflow_stage_directories
    from backend.execution import _safe_job_directory_name
    from backend.phonons.workflow import PHONON_JOB_NAMES

    try:
        stage_names = workflow_stage_directories(workflow_spec)
    except Exception:
        return None
    if len(stage_names) < 2:
        return None
    stage_dirs = ((submission_spec or {}).get("paths") or {}).get("stage_dirs") or {}
    root = run_dir.rstrip("/")

    def stage_dir(name: str) -> str:
        return str(stage_dirs.get(name) or posixpath.join(root, name))

    return {
        "vasp_dir": stage_dir(stage_names[-2]),
        "analysis_dir": posixpath.join(
            stage_dir(stage_names[-1]),
            _safe_job_directory_name(PHONON_JOB_NAMES[-1]),
        ),
    }


def _empty_results_location() -> dict:
    return {
        "run_dir": "",
        "output_dir": "",
        "optional_result_file_paths": {},
        "calculation_spec": None,
        "workflow_spec": None,
        "workflow_result_file_keys": (),
        "remote_python": DEFAULT_REMOTE_PYTHON,
    }


def result_output_dir_from_submission_spec(
    run_dir: str,
    submission_spec: dict | None,
) -> str:
    if submission_spec:
        result_dir = submission_spec.get("paths", {}).get("result_dir")
        if result_dir:
            return str(result_dir)

    stage_dir = result_stage_directory_from_submission_spec(submission_spec)
    if stage_dir:
        return posixpath.join(run_dir.rstrip("/"), stage_dir)

    return run_dir


def remote_job_state_path(job_id: str) -> str:
    safe_job_id = str(job_id or "").strip().split(".", 1)[0]
    return posixpath.join(DEFAULT_LOGS_DIR, f"job_{safe_job_id}.json")


def result_file_paths(
    run_dir: str,
    *,
    include_dos: bool = False,
    extra_file_keys: tuple[str, ...] = (),
) -> dict[str, str]:
    root = run_dir.rstrip("/")
    keys = ["contcar", "outcar", "vasprun"]
    if include_dos:
        keys.append("doscar")
    for key in extra_file_keys:
        if key not in keys:
            keys.append(key)
    return {
        key: posixpath.join(root, filename)
        for key, filename in RESULT_FILES.items()
        if key in keys
    }


def missing_result_files(runner: RemoteRunner, paths: dict[str, str]) -> list[str]:
    missing = []
    for key, remote_path in paths.items():
        _log_results(f"Verify {RESULT_FILES[key]}")
        if not runner.is_file(remote_path):
            missing.append(RESULT_FILES[key])
            _log_results(f"{RESULT_FILES[key]} missing")
        else:
            _log_results(f"{RESULT_FILES[key]} found")
    return missing


def read_result_files(runner: RemoteRunner, paths: dict[str, str | None]) -> dict:
    files = {}
    for key, remote_path in paths.items():
        if not remote_path:
            continue
        _log_results(f"Read {RESULT_FILES[key]}")
        data = runner.read_bytes(remote_path)
        _log_results(f"{RESULT_FILES[key]} read complete")
        _log_results(f"Decode {RESULT_FILES[key]}")
        files[key] = {
            "path": remote_path,
            "text": _decode_remote_output(remote_path, data),
        }
        _log_results(f"{RESULT_FILES[key]} decode complete")
    return files


def extract_remote_vasp_result(
    runner: RemoteRunner,
    paths: dict[str, str],
    context: dict,
    *,
    python: str | None = None,
    timeout_s: float = REMOTE_RESULT_PARSE_TIMEOUT_S,
) -> dict:
    source = remote_result_parser_source(paths, context)
    started = time.time()
    try:
        command_result = runner.run_python(
            source,
            python=python or DEFAULT_REMOTE_PYTHON,
            check=False,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        raise RemoteResultExtractionError(_clean_message(exc)) from exc

    if not command_result.ok:
        message = (
            (command_result.stderr or "").strip()
            or (command_result.stdout or "").strip()
            or f"Remote parser exited with code {command_result.returncode}."
        )
        raise RemoteResultExtractionError(message)

    stdout = command_result.stdout or ""
    stdout_bytes = len(stdout.encode("utf-8"))
    if stdout_bytes > REMOTE_RESULT_STDOUT_MAX_BYTES:
        raise RemoteResultExtractionError(
            "Remote result parser returned an unexpectedly large payload "
            f"({stdout_bytes} bytes)."
        )

    try:
        payload_json = _remote_result_json_from_stdout(stdout)
        result = json.loads(payload_json)
        if not isinstance(result, dict):
            raise RuntimeError("Remote result parser returned a non-object payload.")
    except Exception as exc:
        raise RemoteResultExtractionError(_clean_message(exc)) from exc

    diagnostics = result.setdefault("diagnostics", {})
    remote_parsing = diagnostics.setdefault("remote_parsing", {})
    remote_parsing["compact_result_bytes"] = len(payload_json.encode("utf-8"))
    remote_parsing["remote_command_elapsed_s"] = _round_float(command_result.elapsed_s)
    remote_parsing["local_processing_elapsed_s"] = _round_float(time.time() - started)
    remote_parsing["stdout_bytes"] = stdout_bytes
    remote_parsing["timeout_s"] = timeout_s
    return result


def remote_result_parser_source(paths: dict[str, str], context: dict) -> str:
    parser_module = Path(__file__).with_name("remote_result_parser.py")
    parser_source = parser_module.read_text(encoding="utf-8")
    paths_json = json.dumps(
        _json_safe_mapping(paths),
        sort_keys=True,
        allow_nan=False,
    )
    context_json = json.dumps(
        remote_result_parser_context(context),
        sort_keys=True,
        allow_nan=False,
    )
    start = json.dumps(REMOTE_RESULT_JSON_START)
    end = json.dumps(REMOTE_RESULT_JSON_END)
    return (
        parser_source
        + "\n\n"
        + "if __name__ == '__main__':\n"
        + "    import json as _json\n"
        + f"    _paths = _json.loads({paths_json!r})\n"
        + f"    _context = _json.loads({context_json!r})\n"
        + "    _cwd = os.getcwd()\n"
        + "    _stage_dir = os.path.dirname(_paths.get('vasprun') or '')\n"
        + "    try:\n"
        + "        if _stage_dir:\n"
        + "            os.chdir(_stage_dir)\n"
        + "        _result = parse_result_paths(\n"
        + "            _paths,\n"
        + "            _context,\n"
        + "            parser_name='pymatgen-remote',\n"
        + "            include_remote_metadata=True,\n"
        + "        )\n"
        + "    except Exception as _exc:\n"
        + "        _result = remote_failure_result(_paths, _exc)\n"
        + "    finally:\n"
        + "        os.chdir(_cwd)\n"
        + "    _payload = _json.dumps(_result, separators=(',', ':'), allow_nan=False)\n"
        + f"    print({start})\n"
        + "    print(_payload)\n"
        + f"    print({end})\n"
    )


def _remote_result_json_from_stdout(stdout: str) -> str:
    text = stdout or ""
    start_index = text.find(REMOTE_RESULT_JSON_START)
    if start_index < 0:
        raise RuntimeError("Remote result parser did not emit a result payload.")
    start_index += len(REMOTE_RESULT_JSON_START)
    end_index = text.find(REMOTE_RESULT_JSON_END, start_index)
    if end_index < 0:
        raise RuntimeError("Remote result parser payload was incomplete.")
    payload = text[start_index:end_index].strip()
    if not payload:
        raise RuntimeError("Remote result parser returned an empty payload.")
    return payload


def remote_result_parser_context(context: dict | None) -> dict:
    """
    Return the narrow JSON-native context consumed by the remote parser.

    The remote parser only needs scheduler completion fields for display and a
    calculation/workflow spec dict to decide whether DOS, band, or relaxation
    semantics apply. Application-domain Python objects such as RemoteJobStatus
    must not cross this process boundary.
    """

    source = context or {}
    payload = {
        "slurm_state": _json_safe_scalar(source.get("slurm_state")),
        "exit_code": _json_safe_scalar(source.get("exit_code")),
        "workflow_spec": _json_safe_optional_mapping(source.get("workflow_spec")),
        "calculation_spec": _json_safe_optional_mapping(source.get("calculation_spec")),
    }
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _json_safe_optional_mapping(value) -> dict | None:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, dict):
        return None
    return _json_safe_mapping(value)


def _json_safe_mapping(value: dict) -> dict:
    return {
        str(key): _json_safe_value(item)
        for key, item in value.items()
    }


def _json_safe_sequence(value) -> list:
    return [
        _json_safe_value(item)
        for item in value
    ]


def _json_safe_value(value):
    if isinstance(value, dict):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return _json_safe_sequence(value)
    return _json_safe_scalar(value)


def _json_safe_scalar(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Remote result parser context contains unsupported value "
        f"{value.__class__.__module__}.{value.__class__.__name__}."
    )


def parse_vasp_result_files(files: dict, monitoring_result: dict) -> dict:
    _log_results("ENTER parse_vasp_result_files")
    with tempfile.TemporaryDirectory(prefix="bmd-results-") as tmpdir:
        _log_results("Write temporary VASP files")
        tmp = Path(tmpdir)
        local_paths = {
            "contcar": tmp / "CONTCAR",
            "outcar": tmp / "OUTCAR",
            "vasprun": tmp / "vasprun.xml",
        }
        if "doscar" in files:
            local_paths["doscar"] = tmp / "DOSCAR"
        if "kpoints" in files:
            local_paths["kpoints"] = tmp / "KPOINTS"
        for key, filename in PHONON_RESULT_FILES.items():
            if key in files:
                local_paths[key] = tmp / filename
        if PRECEDING_STATIC_OUTCAR_KEY in files:
            local_paths[PRECEDING_STATIC_OUTCAR_KEY] = tmp / "preceding_static_OUTCAR"
        if GEOMETRY_RELAXATION_VASPRUN_KEY in files:
            local_paths[GEOMETRY_RELAXATION_VASPRUN_KEY] = tmp / "final_relaxation_vasprun.xml"

        for key, local_path in local_paths.items():
            local_path.write_text(files[key]["text"], encoding="utf-8")
        _log_results("Temporary VASP files written")

        path_context = {
            key: str(value)
            for key, value in local_paths.items()
        }
        result = parse_result_paths(
            path_context,
            monitoring_result,
            display_paths={
                key: value["path"]
                for key, value in files.items()
            },
            parser_name="pymatgen",
            include_remote_metadata=False,
        )
        result["files"] = {
            key: value["path"]
            for key, value in files.items()
        }
        _log_results("RETURN parse_vasp_result_files")
        return result


def _decode_remote_output(remote_path: str, data: bytes) -> str:
    raw = data or b""
    if remote_path.endswith(".gz") or raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "ignore")


def _results_parse_context(monitoring_result: dict | None, location: dict) -> dict:
    context = dict(monitoring_result or {})
    workflow_spec = location.get("workflow_spec")
    if workflow_spec is not None:
        context["workflow_spec"] = workflow_spec.to_dict()
    calculation_spec = location.get("calculation_spec")
    if calculation_spec is not None:
        context["calculation_spec"] = calculation_spec.to_dict()
    return context


def _failure_result(
    stage: str,
    reason: str,
    suggestion: str,
    *,
    files: dict | None = None,
    exception: Exception | None = None,
) -> dict:
    result = {
        "status": "failed",
        "title": "Results Summary Failed",
        "stage": stage,
        "reason": reason,
        "suggestion": suggestion,
        "files": files or {},
    }
    if exception is not None and not bmd_debug_enabled():
        result["display_reason"] = "Results could not be loaded."
    if exception is not None:
        result["exception_text"] = str(exception)
        result["traceback"] = "".join(
            traceback.format_exception(type(exception), exception, exception.__traceback__)
        )
    return result


def _clean_message(value) -> str:
    text = str(value or "").strip()
    return text or "Results processing failed."


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


__all__ = [
    "cached_completed_result",
    "clear_results_cache",
    "extract_remote_vasp_result",
    "load_results_for_completed_job",
    "missing_result_files",
    "monitoring_indicates_success",
    "parse_vasp_result_files",
    "read_result_files",
    "remote_result_parser_context",
    "remote_result_parser_source",
    "remote_job_state_path",
    "resolve_results_location",
    "resolve_results_run_dir",
    "results_cache_info",
    "results_cache_key",
    "results_location_from_submission_spec",
    "result_includes_dos_from_submission_spec",
    "result_output_dir_from_submission_spec",
    "result_stage_directory_from_submission_spec",
    "phonon_result_directories",
    "result_file_paths",
    "store_completed_result",
]
