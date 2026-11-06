"""Validity probe: how USPEX 25 treats a relaxation step the bridge reports as failed.

USPEX integration step 2 (docs/structure_search.md, "Candidate validity").
A tiny Si8 search through the production launch bridge and launcher, with two
injected failures (inject_vasp.sh): a genuine electronic non-convergence in
one step-2 relaxation and a non-zero exit status after one step-3
relaxation. It checks mechanics only; the INCARs are the si_bridge pilot's,
not BMD policy.

MODE=validity     (default) the launcher's validity check is on: the
                  production path (status 118/119/7 and withheld files).
                  Injections: nelm_step2 and exit7_step3.
MODE=status_only  the check is off: one step-3 failure is signalled by the
                  exit status alone, with OUTCAR and CONTCAR left in place.
                  Injection: exit7_step3 only.

Isolation. Everything is written under --work, a new directory directly
beneath VALIDATION_ROOT (/bmd/bmdguest/validation) whose name starts with
"uspex_validity_probe_"; the source tree must be a directory directly beneath
the same root. Both must lie under /bmd, the only path the production
Singularity bind (-B /bmd) shows USPEX. Paths are checked component by
component: no "..", ".", "//" or trailing "/", no symlinked component, and
never inside BMD Compute's production areas (PROTECTED_ROOTS). The work
directory must not exist (it is created with an exclusive mkdir); the root
is used as it is, or created with a plain mkdir if this user may write to
/bmd. No permissions are changed anywhere. The
probe does not use jobflow, atomate2, MongoDB or BMD Compute's submission
records, and checks at the end that neither jobflow nor atomate2 was
imported. The USPEX program, sandbox, cache and Singularity come from
backend.config (SearchRuntimeSettings.from_config), the values searches use,
and the job-start runtime check (backend.structure_search.runtime) must pass
before USPEX starts; its result is kept in runtime_check.json.

USPEX cache. bmdguest cannot write in the shared cache's uspex_sfx (the
USPEX extractor creates a lock file there on every start), so the probe, and
only the probe, uses the private cache PRIVATE_CACHE_DIR inside the
validation area: probe_runtime_settings() is SearchRuntimeSettings.from_config()
with cache_dir replaced, nothing else. It never reads BMD_* variables: any
BMD_* in its environment is refused, so the preflight, the cache
initialisation and the job build identical settings. The private cache is
filled once with the production mechanism (runtime.unpack_once, after checking
the program's payload SHA-256 and version banner) through --init-cache, run on
a compute node. The preflight prints the SHA-256 of the prepared runtime
record (--check-runtime); the job refuses to start unless it computes the
same value (--runtime-sha256).

Run from validity_probe.sbatch, inside a SLURM job, with the VASP modules loaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import re
import stat
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from backend.structure_search.bridge.launcher import (  # noqa: E402
    LauncherSettings,
    install_stop_signal_handlers,
    run_search,
)
from backend.structure_search.bridge.master import (  # noqa: E402
    bridge_script_path,
    build_master_command,
    uspex_command_executable,
)
from backend.structure_search.runtime import (  # noqa: E402
    CONTAINER_LOCALE_ENVIRONMENT,
    USPEX_VERSION,
    SearchRuntimeSettings,
    SearchRuntimeSetupError,
    prepared_search_runtime,
    read_program_header,
    search_runtime_check,
    unpack_once,
)

_PILOT = importlib.util.spec_from_file_location("run_pilot", HERE.parent / "si_bridge" / "run_pilot.py")
run_pilot = importlib.util.module_from_spec(_PILOT)
_PILOT.loader.exec_module(run_pilot)

STEPS = 3
PARALLEL = 2
RANKS = 4
STEP_MEM = "8G"
BRIDGE_TIMEOUT_SECONDS = 3600
EXPECTED_PYTHON = "/bmd/bmdguest/envs/atomate2_remote/bin/python"
MODES = {
    "validity": {"step_validity": True, "injections": ("nelm_step2", "exit7_step3")},
    "status_only": {"step_validity": False, "injections": ("exit7_step3",)},
}
# BMD Compute's production areas on POWER: the remote workspace (flows, logs,
# jobflow configuration), the shared POTCARs, the service environment and the
# shared USPEX installation. The probe never writes inside them, except that
# the USPEX extractor (like every search) creates and removes its lock file in
# the shared cache's uspex_sfx directory, and the runtime check tests that
# directory for writing the same way.
PROTECTED_ROOTS = ("/bmd-db", "/bmd/bmdguest", "/bmd/shared")
# The only place probe runs and their source trees may live.
# /bmd/bmdguest is writable by bmdguest (NFSv4: /bmd/validation is not, and
# ACLs are unsupported). Only this one subdirectory of the protected
# /bmd/bmdguest is usable; everything else there stays refused.
VALIDATION_ROOT = "/bmd/bmdguest/validation"
# The probe's own USPEX unpack cache (XDG_CACHE_HOME), inside the validation area.
PRIVATE_CACHE_DIR = VALIDATION_ROOT + "/uspex_cache_private"
WORK_NAME_PATTERN = re.compile(r"^uspex_validity_probe_[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SOURCE_NAME_PATTERN = re.compile(r"^bmd-compute-uspex-probe-[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def write_input(search_dir: Path, bridge: Path, num_parallel: int) -> None:
    executable = "\n".join(uspex_command_executable(bridge, STEPS))
    text = f"""PARAMETERS EVOLUTIONARY ALGORITHM
USPEX : calculationMethod
300 : calculationType

% atomType
Si
% EndAtomType

% numSpecies
8
% EndNumSpecies

0.0 : ExternalPressure
4 : populationSize
4 : initialPopSize
2 : numGenerations
2 : stopCrit

% abinitioCode
1 1 1
% ENDabinit

% KresolStart
0.16 0.10 0.07
% Kresolend

% commandExecutable
{executable}
% EndExecutable

-1 : whichCluster
{num_parallel} : numParallelCalcs
1 : coresPerJob
"""
    (search_dir / "INPUT.txt").write_text(text)


class PathRefused(SystemExit):
    """A work or source path the probe will not use."""


def _canonical(text: str, label: str) -> str:
    """``text`` unchanged if it is an absolute, already-normal path; refuse anything else."""

    text = str(text)
    if not text.startswith("/") or os.path.normpath(text) != text or text.startswith("//"):
        raise PathRefused(f"{label} {text!r} must be an absolute, normalised path (no '..', '.', '//' or trailing '/')")
    if any(part in ("..", ".") for part in text.split("/")):
        raise PathRefused(f"{label} {text!r} must not contain '..' or '.'")
    inside_validation_root = text == VALIDATION_ROOT or text.startswith(VALIDATION_ROOT + "/")
    for root in PROTECTED_ROOTS:
        if (text == root or text.startswith(root + "/")) and not inside_validation_root:
            raise PathRefused(f"{label} {text} is inside {root}, a BMD Compute production area")
    return text


def _existing_components_are_real(text: str, label: str) -> None:
    """Refuse a path any of whose existing components is a symbolic link."""

    current = "/"
    for part in text.strip("/").split("/"):
        current = os.path.join(current, part)
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise PathRefused(f"{label}: {current} is a symbolic link; the probe refuses symlinked paths")
        if not stat.S_ISDIR(mode) and current != text:
            raise PathRefused(f"{label}: {current} is not a directory")


def check_work_directory(work: str | Path, *, root: str = VALIDATION_ROOT) -> dict:
    """Validate ``work`` without creating anything. Returns what create_work_directory will do."""

    text = _canonical(str(work), "--work")
    root = _canonical(root, "validation root")
    parent, name = os.path.split(text)
    if parent != root:
        raise PathRefused(f"--work {text} must be directly beneath {root}")
    if not WORK_NAME_PATTERN.fullmatch(name):
        raise PathRefused(f"--work name {name!r} must start with 'uspex_validity_probe_' (letters, digits, '_', '.', '-')")
    _existing_components_are_real(text, "--work")
    if os.path.lexists(text):
        raise PathRefused(f"--work {text} already exists; the probe never reuses or overwrites a directory")
    if os.path.lexists(root):
        if not os.access(root, os.W_OK | os.X_OK):
            raise PathRefused(f"{root} exists but this user cannot create directories in it")
        return {"work": text, "root": root, "root_state": "exists"}
    grandparent = os.path.dirname(root)
    if not os.path.isdir(grandparent) or not os.access(grandparent, os.W_OK | os.X_OK):
        raise PathRefused(
            f"{root} does not exist and this user cannot create it in {grandparent}; "
            "create it first (the probe changes no permissions)"
        )
    return {"work": text, "root": root, "root_state": "will be created"}


def create_work_directory(work: str | Path, *, root: str = VALIDATION_ROOT) -> Path:
    """Create the fresh work directory (and the root if needed) with plain mkdir; no chmod."""

    plan = check_work_directory(work, root=root)
    if plan["root_state"] == "will be created":
        try:
            os.mkdir(plan["root"])
        except FileExistsError:
            pass  # created meanwhile by someone else; checked below like any existing root
    _existing_components_are_real(plan["root"], "validation root")
    if not stat.S_ISDIR(os.lstat(plan["root"]).st_mode):
        raise PathRefused(f"{plan['root']} is not a directory")
    os.mkdir(plan["work"])  # exclusive: fails if anything appeared at this path
    if not stat.S_ISDIR(os.lstat(plan["work"]).st_mode):
        raise PathRefused(f"{plan['work']} is not a plain directory after creation")
    return Path(plan["work"])


def check_source_directory(source: str | Path, *, root: str = VALIDATION_ROOT) -> str:
    """The transferred source tree: an existing real directory directly beneath ``root``."""

    text = _canonical(str(source), "source tree")
    parent, name = os.path.split(text)
    if parent != _canonical(root, "validation root") or not SOURCE_NAME_PATTERN.fullmatch(name):
        raise PathRefused(f"source tree {text} must be {root}/bmd-compute-uspex-probe-<tag>")
    _existing_components_are_real(text, "source tree")
    if not os.path.isdir(text) or stat.S_ISLNK(os.lstat(text).st_mode):
        raise PathRefused(f"source tree {text} is not an existing directory")
    return text


def refuse_bmd_environment(environ=None) -> None:
    """The probe's settings must not depend on inherited BMD_* variables."""

    names = sorted(name for name in (os.environ if environ is None else environ) if name.startswith("BMD_"))
    if names:
        raise SystemExit(
            "BMD_* variables are set (" + ", ".join(names) + "); the probe uses fixed settings. "
            "Run it through preflight.sh, init_private_cache.sh or validity_probe.sbatch, which clear them."
        )


def check_private_cache(cache: str = PRIVATE_CACHE_DIR, *, root: str = VALIDATION_ROOT) -> str:
    """The private cache: an existing, writable, real directory directly beneath ``root``."""

    text = _canonical(cache, "private USPEX cache")
    if os.path.dirname(text) != _canonical(root, "validation root"):
        raise PathRefused(f"private USPEX cache {text} must be directly beneath {root}")
    _existing_components_are_real(text, "private USPEX cache")
    if not os.path.isdir(text) or stat.S_ISLNK(os.lstat(text).st_mode):
        raise PathRefused(f"private USPEX cache {text} is not an existing directory")
    if not os.access(text, os.W_OK | os.X_OK):
        raise PathRefused(f"private USPEX cache {text} is not writable by this user")
    unpack_root = os.path.join(text, "uspex_sfx")
    if os.path.lexists(unpack_root):
        _existing_components_are_real(unpack_root, "private USPEX cache")
        if not os.path.isdir(unpack_root) or not os.access(unpack_root, os.W_OK | os.X_OK):
            raise PathRefused(f"{unpack_root} is not a writable directory")
    return text


def probe_runtime_settings(*, cache: str = PRIVATE_CACHE_DIR, root: str = VALIDATION_ROOT,
                           check_cache: bool = True) -> SearchRuntimeSettings:
    """Production runtime settings with only the cache replaced by the private one."""

    refuse_bmd_environment()
    if check_cache:
        check_private_cache(cache, root=root)
    return dataclasses.replace(SearchRuntimeSettings.from_config(), cache_dir=cache)


def runtime_record_sha256(record: dict) -> str:
    """SHA-256 of a prepared runtime record (canonical JSON), for preflight/job parity."""

    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def init_private_cache(settings: SearchRuntimeSettings, *, unpack=unpack_once) -> dict:
    """Fill the private cache with the production unpack, after checking the program.

    The program header must carry the configured payload SHA-256 and USPEX
    version banner; unpack_once checks the SHA-256 again, starts the real
    extractor and waits for its own .ready marker. Nothing is written here.
    """

    header = read_program_header(settings.program_path)
    if header["problems"]:
        raise SearchRuntimeSetupError(" ".join(header["problems"]))
    if header["payload_sha256"] != settings.payload_sha256:
        raise SearchRuntimeSetupError(
            f"{settings.program_path} has payload SHA-256 {header['payload_sha256']}, expected {settings.payload_sha256}"
        )
    if header["banner_version"] != USPEX_VERSION:
        raise SearchRuntimeSetupError(
            f"{settings.program_path} reports USPEX {header['banner_version']}, expected {USPEX_VERSION}"
        )
    result = unpack(settings)
    check = search_runtime_check(prepared_search_runtime(settings))
    return {"unpack": result, "runtime_check": check, "header": header}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", help="the new work directory (required to run, and for --check-paths)")
    parser.add_argument("--potcar", type=Path, help="PBE_64 Si POTCAR (copied, never modified)")
    parser.add_argument("--mode", choices=sorted(MODES), default="validity")
    parser.add_argument("--runtime-sha256", help="the runtime record SHA-256 the preflight printed (required to run)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-paths", action="store_true",
                        help="only validate the work and source paths, print the result and exit; creates nothing")
    action.add_argument("--check-runtime", action="store_true",
                        help="run the USPEX runtime check with the probe's settings, print it and the record SHA-256")
    action.add_argument("--init-cache", action="store_true",
                        help="unpack USPEX into the private cache (on a compute node, once)")
    args = parser.parse_args()
    mode = MODES[args.mode]

    if sys.executable != EXPECTED_PYTHON:
        raise SystemExit(f"run with {EXPECTED_PYTHON}, not {sys.executable}")
    refuse_bmd_environment()
    # The source tree as invoked (not resolved), then compared with the resolved
    # REPO, so a symlink anywhere in it is refused.
    source = check_source_directory(str(Path(os.path.abspath(__file__)).parents[3]))
    if source != str(REPO):
        raise PathRefused(f"source tree {source} resolves to {REPO}; symlinked source trees are refused")

    if args.init_cache:
        settings = probe_runtime_settings()
        try:
            result = init_private_cache(settings)
        except SearchRuntimeSetupError as exc:
            print(f"cache initialisation failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=1, sort_keys=True, default=str))
        return 0 if result["runtime_check"]["status"] == "passed" else 1
    if args.check_runtime:
        settings = probe_runtime_settings()
        record = prepared_search_runtime(settings)
        check = search_runtime_check(record)
        print(json.dumps({"runtime_record": record, "check": check}, indent=1, sort_keys=True, default=str))
        print(f"runtime_record_sha256={runtime_record_sha256(record)}")
        return 0 if check["status"] == "passed" else 1
    if args.work is None:
        raise SystemExit("--work is required")
    plan = check_work_directory(args.work)
    if args.check_paths:
        print(json.dumps({"source": source, **plan}, indent=1, sort_keys=True))
        return 0
    if args.potcar is None or args.runtime_sha256 is None:
        raise SystemExit("--potcar and --runtime-sha256 are required to run the probe")
    settings = probe_runtime_settings()
    record = prepared_search_runtime(settings)
    if runtime_record_sha256(record) != args.runtime_sha256:
        raise SystemExit(
            f"runtime settings differ from the preflight's: record SHA-256 {runtime_record_sha256(record)}, "
            f"preflight {args.runtime_sha256}"
        )
    potcar = args.potcar.resolve()
    if not potcar.is_file():
        raise SystemExit(f"--potcar {potcar} is not a file")
    bridge = bridge_script_path()
    if not str(bridge).startswith("/bmd/"):
        raise SystemExit(f"the bmd-compute checkout must be under /bmd (bridge is {bridge})")

    work = create_work_directory(args.work)
    search_dir = work / "search"
    spool = work / "spool"
    state = work / "probe_state"
    search_dir.mkdir()
    state.mkdir()

    # The USPEX installation searches use (with the private cache), checked
    # before anything starts.
    runtime = search_runtime_check(record)
    _write_json(work / "runtime_check.json", runtime)
    if runtime["status"] != "passed":
        print("USPEX runtime check failed:\n  " + "\n  ".join(runtime["problems"]), file=sys.stderr)
        return 2

    specific = search_dir / "Specific"
    shutil.copytree(HERE.parent / "si_bridge" / "Specific", specific)
    shutil.copyfile(potcar, specific / "POTCAR_Si")
    wrapper = run_pilot.write_bridge_wrapper(search_dir, bridge, spool, BRIDGE_TIMEOUT_SECONDS)
    write_input(search_dir, wrapper, PARALLEL)

    master = build_master_command(
        singularity=settings.singularity,
        image=settings.sandbox_dir,
        search_dir=search_dir,
        xdg_cache_home=settings.cache_dir,
        spool_dir=spool,
        bridge_timeout_seconds=BRIDGE_TIMEOUT_SECONDS,
    )
    master_env = dict(master.env)
    for key, value in CONTAINER_LOCALE_ENVIRONMENT.items():
        master_env.setdefault(f"SINGULARITYENV_{key}", value)

    environment = {key: value for key, value in os.environ.items() if not key.startswith("BMD_")}
    environment.update({
        "BMD_PROBE_STATE": str(state),
        "BMD_PROBE_INJECTIONS": ",".join(mode["injections"]),
        "I_MPI_PMI_LIBRARY": settings.i_mpi_pmi_library,
    })
    for name in ("JOBFLOW_CONFIG_FILE", "ATOMATE2_CONFIG_FILE"):
        environment.pop(name, None)
    launcher_settings = LauncherSettings(
        spool_dir=spool,
        search_dir=search_dir,
        max_parallel=PARALLEL,
        ranks=RANKS,
        environment=environment,
        launch_mode="srun_steps",
        mem=STEP_MEM,
        command_template=(
            f"{HERE / 'inject_vasp.sh'} {{step}} srun --exact --mpi=pmi2 -n {{ranks}} --mem={{mem}} vasp_std"
        ),
        step_validity=mode["step_validity"],
    )

    import pymatgen.core  # noqa: F401 - the validity check needs it; fail before USPEX starts

    manifest = {
        "probe": "uspex_validity_probe",
        "mode": args.mode,
        "injections": list(mode["injections"]),
        "step_validity": mode["step_validity"],
        "work": str(work),
        "repo": str(REPO),
        "repo_commit": _git_commit(),
        "repo_manifest_sha256": (
            _sha256(REPO / "MANIFEST.sha256") if (REPO / "MANIFEST.sha256").is_file() else None
        ),
        "python": sys.executable,
        "python_version": sys.version,
        "pymatgen_version": getattr(sys.modules.get("pymatgen.core"), "__version__", None),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "potcar": {"source": str(potcar), "sha256": _sha256(potcar)},
        "input_txt_sha256": _sha256(search_dir / "INPUT.txt"),
        "master_argv": list(master.argv),
        "launcher": {
            "max_parallel": PARALLEL,
            "ranks": RANKS,
            "mem": STEP_MEM,
            "command_template": launcher_settings.command_template,
            "launch_mode": launcher_settings.launch_mode,
        },
        "runtime_record_sha256": runtime_record_sha256(record),
        "uspex": {
            "sandbox": settings.sandbox_dir,
            "cache": settings.cache_dir,
            "payload_sha256": settings.payload_sha256,
            "observed_payload_sha256": runtime["observed"].get("payload_sha256"),
            "observed_banner_version": runtime["observed"].get("banner_version"),
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(work / "probe_manifest.json", manifest)

    stop = threading.Event()
    install_stop_signal_handlers(stop)
    status = run_search(
        master.argv, master_env, master.cwd, launcher_settings,
        master_log=work / "uspex_master.log", stop_event=stop,
    )
    manifest.update({
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_status": status,
        "jobflow_imported": "jobflow" in sys.modules,
        "atomate2_imported": "atomate2" in sys.modules,
    })
    _write_json(work / "probe_manifest.json", manifest)
    print(f"USPEX master finished with status {status}")
    if manifest["jobflow_imported"] or manifest["atomate2_imported"]:
        print("isolation check failed: jobflow or atomate2 was imported", file=sys.stderr)
        return 3
    return status


if __name__ == "__main__":
    raise SystemExit(main())
