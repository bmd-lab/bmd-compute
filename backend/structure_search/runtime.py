"""The USPEX program that structure searches run, and the checks that it is there.

A structure search runs USPEX 25.0 inside its Singularity image (an unpacked
sandbox directory) on one exclusive POWER node; see docs/structure_search.md
§4 and §5. This module owns three things:

1. **Settings** (``SearchRuntimeSettings``): which sandbox, unpack cache,
   program and launch layout a search uses. ``from_config()`` reads them from
   ``backend.config`` at preparation.
2. **Provenance** (``prepared_search_runtime()``): the record of those
   settings that preparation stores in ``submission.json``, under
   ``structure_search.runtime``. The job on POWER checks against this record,
   not against its own environment, so a job runs exactly what was prepared.
3. **Checks**:
   * ``search_runtime_check()`` runs on POWER at job start, before USPEX or
     VASP starts, through ``backend.runtime_environment`` (search submissions
     only). It reports every problem it finds; the caller stops the run if
     there are any.
   * ``unpack_once()`` is for environment setup, run once by an operator. It
     unpacks the program into the shared cache so that jobs never unpack it,
     and never race each other to do so.

How the program is identified. ``/opt/uspex/uspex`` in the image is a POSIX
shell self-extractor: a shell header followed by a gzip tar payload. The
header holds a ``PAYLOAD_SHA256=<hex>`` line, and the extractor unpacks into
``$XDG_CACHE_HOME/uspex_sfx/<that sha>/`` and writes a ``.ready`` marker
there when it is done. The payload SHA-256 therefore identifies the program,
so the check reads it from the header and compares it with the prepared
value. It does not hash the 1.4 GB sandbox or the whole extractor file.

Why the cache must be writable. On every start, before it checks whether it
is already unpacked, the extractor creates and removes a lock file in
``<cache>/uspex_sfx``. It also checks that ``<cache>`` is writable; if it is
not, it silently falls back to ``/tmp/uspex_sfx_$USER`` and unpacks 2 GB
there (docs/structure_search.md §5.4). So both directories are checked for
writing as the job's own user.

This module uses only the standard library and ``backend.config``, so it can
also be run from a checkout on POWER:
``python -m backend.structure_search.runtime {record,check,unpack}``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


USPEX_VERSION = "25.0"
# Path of the self-extracting program inside the image.
USPEX_PROGRAM_IN_IMAGE = "/opt/uspex/uspex"
UNPACK_DIRNAME = "uspex_sfx"
READY_MARKER = ".ready"
# What the extractor runs after unpacking, relative to the unpacked directory.
UNPACKED_PROGRAM = "uspex_core/uspex"
# The shell header is a few kilobytes; the payload follows it. Only this much
# of the file is read, and only up to the first NUL byte.
HEADER_READ_BYTES = 256 * 1024
# `srun --exact` needs SLURM 21.08 or later. POWER runs 26.05.5.
MIN_SLURM_VERSION_FOR_SRUN_STEPS = (21, 8)
LAUNCH_MODES = ("srun_steps", "mpirun_pinned")
# With the image's default ASCII locale USPEX 25 crashed with
# UnicodeEncodeError (docs/structure_search.md §5.6).
CONTAINER_LOCALE_ENVIRONMENT = {
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8",
}

# submission.json key that marks a structure-search submission.
STRUCTURE_SEARCH_SPEC_KEY = "structure_search"
SEARCH_RUNTIME_RECORD_KEY = "runtime"
SEARCH_RUNTIME_SCHEMA = "bmd_compute.structure_search_runtime"
SEARCH_RUNTIME_SCHEMA_VERSION = 1

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:export[ \t]+|readonly[ \t]+)?PAYLOAD_SHA256=[\"']?([0-9A-Fa-f]{64})[\"']?[ \t]*(?:#.*)?$",
    re.MULTILINE,
)
_BANNER_PATTERN = re.compile(r"USPEX v([0-9]+(?:\.[0-9]+)+)")
_SLURM_VERSION_PATTERN = re.compile(r"([0-9]+)\.([0-9]+)(?:\.([0-9]+))?")
_HOSTLIST_PATTERN = re.compile(
    r"^[A-Za-z0-9._\-]+(\[[0-9,\-]+\])?(,[A-Za-z0-9._\-]+(\[[0-9,\-]+\])?)*$"
)


class SearchRuntimeSetupError(RuntimeError):
    """Environment setup (``unpack_once``) could not complete."""


# -- settings -----------------------------------------------------------------------


@dataclass(frozen=True)
class SearchRuntimeSettings:
    """Everything a search needs to find and launch USPEX. See the module docstring."""

    sandbox_dir: str
    cache_dir: str
    payload_sha256: str
    singularity: str
    i_mpi_pmi_library: str
    launch_mode: str
    parallel_relaxations: int
    ranks_per_relaxation: int
    bridge_timeout_seconds: int
    node_sockets: int
    cores_per_socket: int
    excluded_nodes: str

    def __post_init__(self) -> None:
        problems = _settings_problems(self)
        if problems:
            raise ValueError("Invalid structure-search runtime settings: " + "; ".join(problems))

    @classmethod
    def from_config(cls) -> "SearchRuntimeSettings":
        from backend import config

        return cls(
            sandbox_dir=config.USPEX_SANDBOX_DIR,
            cache_dir=config.USPEX_CACHE_DIR,
            payload_sha256=config.USPEX_PAYLOAD_SHA256,
            singularity=config.SINGULARITY_EXECUTABLE,
            i_mpi_pmi_library=config.STRUCTURE_SEARCH_PMI_LIBRARY,
            launch_mode=config.STRUCTURE_SEARCH_LAUNCH_MODE,
            parallel_relaxations=config.STRUCTURE_SEARCH_PARALLEL_RELAXATIONS,
            ranks_per_relaxation=config.STRUCTURE_SEARCH_RANKS_PER_RELAXATION,
            bridge_timeout_seconds=config.STRUCTURE_SEARCH_BRIDGE_TIMEOUT_S,
            node_sockets=config.STRUCTURE_SEARCH_NODE_SOCKETS,
            cores_per_socket=config.STRUCTURE_SEARCH_CORES_PER_SOCKET,
            excluded_nodes=config.STRUCTURE_SEARCH_EXCLUDED_NODES,
        )

    @property
    def program_path(self) -> Path:
        """The self-extractor as seen from the host, inside the sandbox."""

        return Path(self.sandbox_dir) / USPEX_PROGRAM_IN_IMAGE.lstrip("/")

    @property
    def unpack_root(self) -> Path:
        return Path(self.cache_dir) / UNPACK_DIRNAME

    @property
    def unpacked_dir(self) -> Path:
        return self.unpack_root / self.payload_sha256

    @property
    def ready_marker(self) -> Path:
        return self.unpacked_dir / READY_MARKER


def _is_absolute_text(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("/") and "\n" not in value


def _settings_problems(settings: SearchRuntimeSettings) -> list[str]:
    problems = []
    for name in ("sandbox_dir", "cache_dir", "singularity", "i_mpi_pmi_library"):
        if not _is_absolute_text(getattr(settings, name)):
            problems.append(f"{name} must be an absolute path, got {getattr(settings, name)!r}")
    if str(settings.sandbox_dir).lower().endswith(".sif"):
        problems.append(
            f"sandbox_dir {settings.sandbox_dir!r} is a SIF file; searches use the unpacked sandbox directory"
        )
    if not isinstance(settings.payload_sha256, str) or not _SHA256_PATTERN.fullmatch(settings.payload_sha256):
        problems.append(
            f"payload_sha256 must be 64 lowercase hexadecimal characters, got {settings.payload_sha256!r}"
        )
    if settings.launch_mode not in LAUNCH_MODES:
        problems.append(f"launch_mode must be one of {', '.join(LAUNCH_MODES)}, got {settings.launch_mode!r}")
    integers_ok = True
    for name in (
        "parallel_relaxations",
        "ranks_per_relaxation",
        "bridge_timeout_seconds",
        "node_sockets",
        "cores_per_socket",
    ):
        value = getattr(settings, name)
        if type(value) is not int or value < 1:
            problems.append(f"{name} must be a positive integer, got {value!r}")
            integers_ok = False
    if integers_ok:
        ranks = settings.parallel_relaxations * settings.ranks_per_relaxation
        cores = settings.node_sockets * settings.cores_per_socket
        if ranks > cores:
            problems.append(
                f"{settings.parallel_relaxations} relaxations x {settings.ranks_per_relaxation} ranks = "
                f"{ranks} ranks do not fit on a node of {settings.node_sockets} x "
                f"{settings.cores_per_socket} = {cores} cores"
            )
    excluded = settings.excluded_nodes
    if not isinstance(excluded, str) or (excluded and not _HOSTLIST_PATTERN.fullmatch(excluded)):
        problems.append(f"excluded_nodes must be a SLURM host list or empty, got {excluded!r}")
    return problems


# -- provenance -----------------------------------------------------------------------


def prepared_search_runtime(settings: SearchRuntimeSettings | None = None) -> dict[str, Any]:
    """The search runtime record that preparation stores in ``submission.json``.

    It goes under ``submission["structure_search"]["runtime"]``. Not wired into
    preparation yet (#67, part 2).
    """

    settings = settings or SearchRuntimeSettings.from_config()
    return {
        "schema": SEARCH_RUNTIME_SCHEMA,
        "schema_version": SEARCH_RUNTIME_SCHEMA_VERSION,
        "uspex": {
            "version": USPEX_VERSION,
            "payload_sha256": settings.payload_sha256,
            "sandbox_dir": settings.sandbox_dir,
            "program_in_image": USPEX_PROGRAM_IN_IMAGE,
            "cache_dir": settings.cache_dir,
            "unpacked_dir": str(settings.unpacked_dir),
            "ready_marker": str(settings.ready_marker),
        },
        "singularity": settings.singularity,
        "launch": {
            "mode": settings.launch_mode,
            "parallel_relaxations": settings.parallel_relaxations,
            "ranks_per_relaxation": settings.ranks_per_relaxation,
            "i_mpi_pmi_library": settings.i_mpi_pmi_library,
            "bridge_timeout_seconds": settings.bridge_timeout_seconds,
            "node_sockets": settings.node_sockets,
            "cores_per_socket": settings.cores_per_socket,
            "excluded_nodes": settings.excluded_nodes,
        },
    }


def settings_from_record(record: Any) -> tuple[SearchRuntimeSettings | None, list[str]]:
    """Rebuild the settings from a prepared record, or say why it can't be done."""

    if not isinstance(record, Mapping):
        return None, ["submission.json has no structure_search.runtime record."]
    problems = []
    if record.get("schema") != SEARCH_RUNTIME_SCHEMA:
        problems.append(
            f"structure_search.runtime.schema is {record.get('schema')!r}, expected {SEARCH_RUNTIME_SCHEMA!r}."
        )
    version = record.get("schema_version")
    if type(version) is not int or version != SEARCH_RUNTIME_SCHEMA_VERSION:
        problems.append(
            f"structure_search.runtime.schema_version is {version!r}, expected {SEARCH_RUNTIME_SCHEMA_VERSION}."
        )
    uspex = record.get("uspex")
    launch = record.get("launch")
    if not isinstance(uspex, Mapping) or not isinstance(launch, Mapping):
        problems.append("structure_search.runtime is missing its uspex or launch block.")
        return None, problems
    if uspex.get("version") != USPEX_VERSION:
        problems.append(
            f"The submission was prepared for USPEX {uspex.get('version')!r}; this BMD Compute runs "
            f"USPEX {USPEX_VERSION}."
        )
    if uspex.get("program_in_image") != USPEX_PROGRAM_IN_IMAGE:
        problems.append(
            f"structure_search.runtime.uspex.program_in_image is {uspex.get('program_in_image')!r}, "
            f"expected {USPEX_PROGRAM_IN_IMAGE!r}."
        )
    if problems:
        return None, problems
    try:
        settings = SearchRuntimeSettings(
            sandbox_dir=uspex.get("sandbox_dir"),
            cache_dir=uspex.get("cache_dir"),
            payload_sha256=uspex.get("payload_sha256"),
            singularity=record.get("singularity"),
            i_mpi_pmi_library=launch.get("i_mpi_pmi_library"),
            launch_mode=launch.get("mode"),
            parallel_relaxations=launch.get("parallel_relaxations"),
            ranks_per_relaxation=launch.get("ranks_per_relaxation"),
            bridge_timeout_seconds=launch.get("bridge_timeout_seconds"),
            node_sockets=launch.get("node_sockets"),
            cores_per_socket=launch.get("cores_per_socket"),
            excluded_nodes=launch.get("excluded_nodes"),
        )
    except ValueError as exc:
        return None, [f"structure_search.runtime: {exc}"]
    for name, expected in (
        ("unpacked_dir", str(settings.unpacked_dir)),
        ("ready_marker", str(settings.ready_marker)),
    ):
        if uspex.get(name) != expected:
            problems.append(
                f"structure_search.runtime.uspex.{name} is {uspex.get(name)!r}, expected {expected!r}."
            )
    return (None, problems) if problems else (settings, [])


# -- reading the program ----------------------------------------------------------------


def read_program_header(path: str | Path) -> dict[str, Any]:
    """Read the payload SHA-256 and version banner from the extractor's shell header.

    Returns ``{"payload_sha256", "banner_version", "problems"}``. Only the first
    ``HEADER_READ_BYTES`` bytes, up to the first NUL byte, are read.
    """

    try:
        with open(path, "rb") as handle:
            head = handle.read(HEADER_READ_BYTES)
    except OSError as exc:
        return {
            "payload_sha256": None,
            "banner_version": None,
            "problems": [f"{path} could not be read: {exc.strerror or exc}."],
        }
    text = head.split(b"\0", 1)[0].decode("latin-1")
    values = sorted({match.lower() for match in _PAYLOAD_LINE_PATTERN.findall(text)})
    banners = sorted(set(_BANNER_PATTERN.findall(text)))
    problems = []
    if not values:
        problems.append(f"{path} has no PAYLOAD_SHA256= line in its header; it is not the USPEX self-extractor.")
    elif len(values) > 1:
        problems.append(f"{path} has more than one PAYLOAD_SHA256= value in its header: {', '.join(values)}.")
    if len(banners) > 1:
        problems.append(f"{path} names more than one USPEX version in its header: {', '.join(banners)}.")
    return {
        "payload_sha256": values[0] if len(values) == 1 else None,
        # Recorded when present. The payload SHA-256 is the identity; the
        # banner is only compared when the header has one.
        "banner_version": banners[0] if len(banners) == 1 else None,
        "problems": problems,
    }


def _file_facts(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def writable_directory_problem(path: Path) -> str | None:
    """Why the job's user cannot create files in ``path``, or None if it can.

    Checks the permission bits the way the extractor's shell test does, then
    creates and removes a probe file, which also catches read-only mounts and
    NFS permissions that the bits don't show.
    """

    if not path.is_dir():
        return f"{path} does not exist or is not a directory."
    if not os.access(path, os.W_OK | os.X_OK):
        return f"{path} is not writable by this user (uid {os.getuid()})."
    try:
        descriptor, probe = tempfile.mkstemp(prefix=".bmd-write-probe-", dir=path)
    except OSError as exc:
        return f"{path} is not writable by this user (uid {os.getuid()}): {exc.strerror or exc}."
    os.close(descriptor)
    try:
        os.unlink(probe)
    except OSError:
        pass
    return None


def parse_slurm_version(text: str | None) -> tuple[int, ...] | None:
    match = _SLURM_VERSION_PATTERN.search(text or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def default_slurm_version() -> str | None:
    """The output of ``srun --version`` (for example ``slurm 26.05.5``), or None."""

    try:
        completed = subprocess.run(
            ["srun", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


# -- job-start check ----------------------------------------------------------------------


def search_runtime_check(
    prepared: Any,
    *,
    slurm_version_lookup: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Check on POWER that the prepared USPEX runtime is in place.

    ``prepared`` is the ``structure_search.runtime`` record from
    ``submission.json``. Returns ``{"status", "problems", "prepared",
    "observed"}`` and never raises for a problem it can describe; the caller
    stops the run if ``problems`` is not empty. Nothing is unpacked or
    started here.
    """

    settings, problems = settings_from_record(prepared)
    observed: dict[str, Any] = {}
    if settings is not None:
        observed = _observe(settings, problems, slurm_version_lookup or default_slurm_version)
    return {
        "status": "failed" if problems else "passed",
        "problems": problems,
        "prepared": prepared if isinstance(prepared, Mapping) else None,
        "observed": observed,
    }


def _observe(
    settings: SearchRuntimeSettings,
    problems: list[str],
    slurm_version_lookup: Callable[[], str | None],
) -> dict[str, Any]:
    sandbox = Path(settings.sandbox_dir)
    program = settings.program_path
    observed: dict[str, Any] = {
        "sandbox_dir_exists": sandbox.is_dir(),
        "program": _file_facts(program),
    }
    if not sandbox.is_dir():
        problems.append(f"The USPEX sandbox {sandbox} does not exist or is not a directory.")
    elif not program.is_file():
        problems.append(f"The USPEX sandbox {sandbox} has no {USPEX_PROGRAM_IN_IMAGE}.")

    header = read_program_header(program) if program.is_file() else {}
    observed["payload_sha256"] = header.get("payload_sha256")
    observed["banner_version"] = header.get("banner_version")
    problems.extend(header.get("problems", ()))
    if header.get("payload_sha256") and header["payload_sha256"] != settings.payload_sha256:
        problems.append(
            f"{program} has payload SHA-256 {header['payload_sha256']}; the submission was prepared for "
            f"{settings.payload_sha256}. The USPEX program on POWER has changed."
        )
    if header.get("banner_version") and header["banner_version"] != USPEX_VERSION:
        problems.append(
            f"{program} is USPEX {header['banner_version']}; BMD Compute runs USPEX {USPEX_VERSION}."
        )

    observed["ready_marker"] = str(settings.ready_marker)
    observed["ready_marker_exists"] = settings.ready_marker.is_file()
    unpacked_program = settings.unpacked_dir / UNPACKED_PROGRAM
    observed["unpacked_program_exists"] = unpacked_program.is_file()
    if not observed["ready_marker_exists"]:
        problems.append(
            f"USPEX is not unpacked in the shared cache: {settings.ready_marker} is missing. "
            "Run the one-time unpack (docs/deployment.md, structure searches); jobs never unpack it."
        )
    elif not observed["unpacked_program_exists"]:
        problems.append(
            f"The unpacked USPEX in {settings.unpacked_dir} has no {UNPACKED_PROGRAM}. "
            "Remove that directory and run the one-time unpack again."
        )

    writable = {}
    for path in (Path(settings.cache_dir), settings.unpack_root):
        problem = writable_directory_problem(path)
        writable[str(path)] = problem is None
        if problem:
            problems.append(
                problem + " USPEX writes a lock file there on every start, and without a writable "
                f"{settings.cache_dir} it unpacks 2 GB into /tmp instead."
            )
    observed["writable"] = writable

    singularity = Path(settings.singularity)
    observed["singularity"] = {
        "path": settings.singularity,
        "executable": singularity.is_file() and os.access(singularity, os.X_OK),
    }
    if not observed["singularity"]["executable"]:
        problems.append(f"Singularity {settings.singularity} does not exist or is not executable.")

    slurm_output = slurm_version_lookup()
    slurm_version = parse_slurm_version(slurm_output)
    observed["slurm_version"] = slurm_output
    if settings.launch_mode == "srun_steps":
        library = Path(settings.i_mpi_pmi_library)
        observed["i_mpi_pmi_library_exists"] = library.is_file()
        if not library.is_file():
            problems.append(
                f"I_MPI_PMI_LIBRARY {settings.i_mpi_pmi_library} does not exist; "
                "srun --mpi=pmi2 steps need it."
            )
        minimum = ".".join(str(part) for part in MIN_SLURM_VERSION_FOR_SRUN_STEPS)
        if slurm_version is None:
            problems.append(
                f"The SLURM version could not be read (srun --version gave {slurm_output!r}); "
                f"the srun_steps launch mode needs SLURM {minimum} or later."
            )
        elif slurm_version[:2] < MIN_SLURM_VERSION_FOR_SRUN_STEPS:
            problems.append(
                f"SLURM {slurm_output} is too old for the srun_steps launch mode (srun --exact needs {minimum})."
            )
    return observed


# -- structure-search submissions ----------------------------------------------------------


def is_structure_search_spec(spec: Any) -> bool:
    """Whether a ``submission.json`` document is a structure-search submission."""

    return isinstance(spec, Mapping) and STRUCTURE_SEARCH_SPEC_KEY in spec


def prepared_record_from_spec(spec: Mapping[str, Any]) -> Any:
    block = spec.get(STRUCTURE_SEARCH_SPEC_KEY)
    return block.get(SEARCH_RUNTIME_RECORD_KEY) if isinstance(block, Mapping) else None


# -- one-time unpack (environment setup) -----------------------------------------------------


def unpack_command(settings: SearchRuntimeSettings, workdir: str | Path) -> tuple[list[str], dict[str, str]]:
    """The command that starts the extractor once with ``XDG_CACHE_HOME`` set.

    ``workdir`` must be inside ``settings.cache_dir``: only the cache is bound
    into the container.
    """

    cache = settings.cache_dir
    argv = [
        settings.singularity,
        "exec",
        "--cleanenv",
        "-B",
        cache,
        "--pwd",
        str(workdir),
        settings.sandbox_dir,
        USPEX_PROGRAM_IN_IMAGE,
    ]
    env = {key: value for key, value in os.environ.items() if not key.startswith("SINGULARITYENV_")}
    env["SINGULARITYENV_XDG_CACHE_HOME"] = cache
    env.update({f"SINGULARITYENV_{key}": value for key, value in CONTAINER_LOCALE_ENVIRONMENT.items()})
    return argv, env


def unpack_once(
    settings: SearchRuntimeSettings | None = None,
    *,
    timeout_seconds: float = 1800.0,
    poll_seconds: float = 2.0,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict[str, Any]:
    """Unpack USPEX into the shared cache, once. For environment setup only.

    Run it as an operator, on a compute node (through ``srun``), never from a
    job. If ``.ready`` already exists it does nothing. Otherwise it checks the
    program's payload SHA-256, starts the extractor once in an empty folder
    with ``XDG_CACHE_HOME`` set to the cache, and stops it as soon as
    ``.ready`` appears (or lets it exit: in an empty folder USPEX stops with
    an IndexError after unpacking, docs/structure_search.md §5.3).

    Raises ``SearchRuntimeSetupError`` with the reason if it cannot finish.
    It refuses to start when a partial ``uspex_sfx/<sha>*`` exists, because
    the extractor then nests a fresh copy inside the old one and fails
    (§5.2); remove it by hand after making sure no other unpack is running.
    """

    settings = settings or SearchRuntimeSettings.from_config()
    if settings.ready_marker.is_file():
        return {"status": "already_unpacked", "unpacked_dir": str(settings.unpacked_dir)}

    cache = Path(settings.cache_dir)
    problem = writable_directory_problem(cache)
    if problem:
        raise SearchRuntimeSetupError(problem)
    root = settings.unpack_root
    if not root.exists():
        root.mkdir()
        # Group-writable with the setgid bit, as the shared layout requires
        # (docs/structure_search.md §5.4).
        os.chmod(root, 0o2775)
    problem = writable_directory_problem(root)
    if problem:
        raise SearchRuntimeSetupError(problem)
    leftovers = sorted(str(path) for path in root.glob(settings.payload_sha256 + "*"))
    if leftovers:
        raise SearchRuntimeSetupError(
            "A partial unpack is in the way: "
            + ", ".join(leftovers)
            + f" (no {READY_MARKER}). Make sure no other unpack is running, remove it, and run again."
        )
    header = read_program_header(settings.program_path)
    if header["problems"]:
        raise SearchRuntimeSetupError(" ".join(header["problems"]))
    if header["payload_sha256"] != settings.payload_sha256:
        raise SearchRuntimeSetupError(
            f"{settings.program_path} has payload SHA-256 {header['payload_sha256']}, "
            f"but the configured value is {settings.payload_sha256}. Nothing was unpacked."
        )

    workdir = Path(tempfile.mkdtemp(prefix=".bmd-unpack-", dir=cache))
    log_path = workdir / "unpack.log"
    started = time.monotonic()
    try:
        argv, env = unpack_command(settings, workdir)
        with open(log_path, "wb") as log:
            try:
                process = popen(
                    argv,
                    env=env,
                    cwd=workdir,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                raise SearchRuntimeSetupError(
                    f"Singularity {settings.singularity} could not be started: {exc.strerror or exc}."
                ) from exc
            exit_status = _wait_for_ready(process, settings.ready_marker, timeout_seconds, poll_seconds)
        elapsed = time.monotonic() - started
        output_tail = log_path.read_bytes()[-4000:].decode("utf-8", "replace")
        if exit_status == "timeout":
            raise SearchRuntimeSetupError(
                f"The unpack did not finish within {timeout_seconds:g} s. Output:\n{output_tail}"
            )
        if not settings.ready_marker.is_file() or not (settings.unpacked_dir / UNPACKED_PROGRAM).is_file():
            raise SearchRuntimeSetupError(
                f"The extractor exited ({exit_status}) without a complete unpack in "
                f"{settings.unpacked_dir}. Output:\n{output_tail}"
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {
        "status": "unpacked",
        "unpacked_dir": str(settings.unpacked_dir),
        "seconds": round(elapsed, 1),
        "extractor_exit": exit_status,
    }


def _wait_for_ready(process, ready_marker: Path, timeout_seconds: float, poll_seconds: float):
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = process.poll()
        if status is not None:
            return status
        if ready_marker.is_file():
            # Unpacked; what runs now is USPEX itself, which is not wanted here.
            _stop_process_group(process)
            return "stopped_after_ready"
        if time.monotonic() >= deadline:
            _stop_process_group(process)
            return "timeout"
        time.sleep(poll_seconds)


def _stop_process_group(process, grace_seconds: float = 10.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


# -- command line ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.structure_search.runtime",
        description="USPEX runtime for structure searches (configured values from backend.config).",
    )
    parser.add_argument(
        "action",
        choices=("record", "check", "unpack"),
        help="record: print the provenance record; check: run the job-start check; "
        "unpack: unpack USPEX into the shared cache once (environment setup).",
    )
    args = parser.parse_args(argv)
    settings = SearchRuntimeSettings.from_config()
    if args.action == "record":
        print(json.dumps(prepared_search_runtime(settings), indent=2, sort_keys=True))
        return 0
    if args.action == "check":
        result = search_runtime_check(prepared_search_runtime(settings))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    try:
        result = unpack_once(settings)
    except SearchRuntimeSetupError as exc:
        print(f"Unpack failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "CONTAINER_LOCALE_ENVIRONMENT",
    "HEADER_READ_BYTES",
    "LAUNCH_MODES",
    "MIN_SLURM_VERSION_FOR_SRUN_STEPS",
    "READY_MARKER",
    "SEARCH_RUNTIME_RECORD_KEY",
    "SEARCH_RUNTIME_SCHEMA",
    "SEARCH_RUNTIME_SCHEMA_VERSION",
    "STRUCTURE_SEARCH_SPEC_KEY",
    "SearchRuntimeSettings",
    "SearchRuntimeSetupError",
    "UNPACKED_PROGRAM",
    "UNPACK_DIRNAME",
    "USPEX_PROGRAM_IN_IMAGE",
    "USPEX_VERSION",
    "default_slurm_version",
    "is_structure_search_spec",
    "main",
    "parse_slurm_version",
    "prepared_record_from_spec",
    "prepared_search_runtime",
    "read_program_header",
    "search_runtime_check",
    "settings_from_record",
    "unpack_command",
    "unpack_once",
    "writable_directory_problem",
]
