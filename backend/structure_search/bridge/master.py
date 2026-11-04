"""Command line and environment for the USPEX master, and the bridge command.

Design: docs/structure_search.md, sections 4 and 5; issue #66.

The USPEX 25 master runs inside the cluster's Singularity image::

    singularity exec --cleanenv -B /bmd --pwd <search dir> <sandbox> /opt/uspex/uspex

``build_master_command`` returns that command with its complete environment,
so nothing depends on the submitter's environment or ``.bashrc``:

* ``--cleanenv`` stops the host environment from leaking into the container.
* Variables the container needs are passed as ``SINGULARITYENV_<NAME>``,
  which Singularity sets inside the container as ``<NAME>``. Those are
  ``XDG_CACHE_HOME`` (where the self-extracting ``uspex`` file finds its
  unpacked copy; home quotas are too small, §5.2) and the ``BMD_BRIDGE_*``
  settings read by ``bmd_bridge.sh``. ``XDG_CACHE_HOME`` is therefore set for
  this command only, not for the launcher or for VASP.
* The host side gets only ``PATH`` and ``LC_ALL``, plus anything passed in
  ``host_environment`` (for example ``SINGULARITY_TMPDIR``).

UTF-8 locale. USPEX 25 prints non-ASCII text. Under ``--cleanenv`` with an
ASCII locale (``LC_ALL=C``) it crashed while setting up its launcher with
``UnicodeEncodeError: 'ascii' codec`` (POWER, job 23355060, #59). So the
container gets ``LC_ALL=C.UTF-8``, ``LANG=C.UTF-8`` and
``PYTHONIOENCODING=utf-8``, and the host side gets ``LC_ALL=C.UTF-8``; this
is the setting that fixed the pilot. ``C.UTF-8`` is built into the image's
glibc 2.35, so no locale files are needed.

Image. The supported image is an unpacked sandbox directory, on POWER
``/bmd/shared/uspex25_sandbox`` (docs/structure_search.md §5.4). Singularity
3.8.6 converts a ``.sif`` file to a temporary sandbox on every start and needs
``unsquashfs`` on ``PATH`` for that; on POWER ``unsquashfs`` is only in the
``uspex-25`` module's mamba environment, not on the explicit host ``PATH``.
``build_master_command`` therefore rejects ``.sif`` paths.

Every path is a parameter. Paths that the container must see (the search
directory, the spool, the cache and the bridge script) must lie under the
bind path, which is mounted at the same location inside the container, so
the same absolute path works on both sides.

Confirmed on POWER (#59, 4 Nov 2026): ``SINGULARITYENV_*`` variables reach
the container under ``--cleanenv`` with Singularity 3.8.6, and with
``whichCluster = -1`` USPEX runs each ``commandExecutable`` from inside its
CalcFold (``Calculation/Calcfold_<n>_<step>``, through ``sh run_local.sh``),
``numParallelCalcs`` at a time, with the full container environment. With
``whichCluster = 0`` USPEX never runs the command, so searches must use -1.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


BRIDGE_SCRIPT_NAME = "bmd_bridge.sh"
DEFAULT_BIND = "/bmd"
DEFAULT_USPEX_PATH = "/opt/uspex/uspex"
DEFAULT_HOST_PATH = "/usr/bin:/bin"
UTF8_LOCALE = "C.UTF-8"
CONTAINER_LOCALE_ENVIRONMENT = {
    "LC_ALL": UTF8_LOCALE,
    "LANG": UTF8_LOCALE,
    "PYTHONIOENCODING": "utf-8",
}
HOST_LOCALE_ENVIRONMENT = {"LC_ALL": UTF8_LOCALE}
DEFAULT_BRIDGE_POLL_SECONDS = 2
DEFAULT_BRIDGE_HEARTBEAT_SECONDS = 300


def bridge_script_path() -> Path:
    """Path of ``bmd_bridge.sh`` in this package (on POWER: the uploaded package)."""

    return Path(__file__).resolve().with_name(BRIDGE_SCRIPT_NAME)


def uspex_command_executable(bridge_script: str | Path, steps: int) -> list[str]:
    """Return the ``commandExecutable`` lines for ``INPUT.txt``, one per step.

    Each line is ``sh <bridge script> <step>``. USPEX runs it in the CalcFold.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    script = shlex.quote(str(bridge_script))
    return [f"sh {script} {step}" for step in range(1, steps + 1)]


@dataclass(frozen=True)
class MasterCommand:
    """A command ready for ``subprocess.Popen(argv, env=env, cwd=cwd)``."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path

    def shell_line(self) -> str:
        """Readable one-line form, for logs (``env -i`` makes the environment explicit)."""

        assignments = [f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items())]
        return shlex.join(["cd", str(self.cwd)]) + " && " + " ".join(
            ["env", "-i", *assignments, shlex.join(self.argv)]
        )


def _absolute(name: str, value: str | Path) -> PurePosixPath:
    path = PurePosixPath(str(value))
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path, got {value!r}")
    if ".." in path.parts:
        raise ValueError(f"{name} must not contain '..', got {value!r}")
    return path


def _under_bind(name: str, path: PurePosixPath, bind: PurePosixPath) -> None:
    if not path.is_relative_to(bind):
        raise ValueError(f"{name} {path} is not under the bind path {bind}, so the container cannot see it")


def build_master_command(
    *,
    singularity: str | Path,
    image: str | Path,
    search_dir: str | Path,
    xdg_cache_home: str | Path,
    spool_dir: str | Path,
    bridge_timeout_seconds: int,
    bind: str | Path = DEFAULT_BIND,
    uspex_path: str = DEFAULT_USPEX_PATH,
    bridge_poll_seconds: int | float = DEFAULT_BRIDGE_POLL_SECONDS,
    bridge_heartbeat_seconds: int = DEFAULT_BRIDGE_HEARTBEAT_SECONDS,
    host_path: str = DEFAULT_HOST_PATH,
    host_environment: Mapping[str, str] | None = None,
    uspex_args: Sequence[str] = (),
) -> MasterCommand:
    """Build the USPEX master command. See the module docstring.

    ``singularity`` is the full path of the Singularity executable and
    ``image`` the unpacked sandbox directory (``.sif`` files are rejected;
    see the module docstring). ``search_dir`` is
    the USPEX run directory (it holds ``INPUT.txt`` and the CalcFolds).
    ``bridge_timeout_seconds`` is how long one relaxation step may wait in
    total, queueing included; 0 means no limit. ``bridge_heartbeat_seconds``
    must be several times the launcher's ``heartbeat_interval_seconds``.
    """

    bind_path = _absolute("bind", bind)
    paths = {
        "singularity": _absolute("singularity", singularity),
        "image": _absolute("image", image),
        "search_dir": _absolute("search_dir", search_dir),
        "xdg_cache_home": _absolute("xdg_cache_home", xdg_cache_home),
        "spool_dir": _absolute("spool_dir", spool_dir),
    }
    for name in ("search_dir", "xdg_cache_home", "spool_dir"):
        _under_bind(name, paths[name], bind_path)
    if paths["image"].suffix.lower() == ".sif":
        raise ValueError(
            f"image {paths['image']} is a SIF file; use the unpacked sandbox directory "
            "(Singularity 3.8.6 needs unsquashfs to convert a SIF, which is not on the host PATH)"
        )
    if not PurePosixPath(uspex_path).is_absolute():
        raise ValueError("uspex_path must be an absolute path inside the image")
    if not isinstance(bridge_timeout_seconds, int) or bridge_timeout_seconds < 0:
        raise ValueError("bridge_timeout_seconds must be a non-negative integer")
    if not isinstance(bridge_heartbeat_seconds, int) or bridge_heartbeat_seconds < 1:
        raise ValueError("bridge_heartbeat_seconds must be a positive integer")
    if not bridge_poll_seconds > 0:
        raise ValueError("bridge_poll_seconds must be positive")

    container_environment = {
        **CONTAINER_LOCALE_ENVIRONMENT,
        "XDG_CACHE_HOME": str(paths["xdg_cache_home"]),
        "BMD_BRIDGE_SPOOL": str(paths["spool_dir"]),
        "BMD_BRIDGE_TIMEOUT_SECONDS": str(bridge_timeout_seconds),
        "BMD_BRIDGE_POLL_SECONDS": str(bridge_poll_seconds),
        "BMD_BRIDGE_HEARTBEAT_SECONDS": str(bridge_heartbeat_seconds),
    }
    env = {"PATH": host_path, **HOST_LOCALE_ENVIRONMENT}
    for key, value in (host_environment or {}).items():
        if key.startswith("SINGULARITYENV_") or key in env:
            raise ValueError(f"host_environment must not set {key}")
        env[str(key)] = str(value)
    env.update({f"SINGULARITYENV_{key}": value for key, value in container_environment.items()})

    argv = (
        str(paths["singularity"]),
        "exec",
        "--cleanenv",
        "-B",
        str(bind_path),
        "--pwd",
        str(paths["search_dir"]),
        str(paths["image"]),
        uspex_path,
        *[str(argument) for argument in uspex_args],
    )
    return MasterCommand(argv=argv, env=env, cwd=Path(str(paths["search_dir"])))


__all__ = [
    "BRIDGE_SCRIPT_NAME",
    "CONTAINER_LOCALE_ENVIRONMENT",
    "HOST_LOCALE_ENVIRONMENT",
    "UTF8_LOCALE",
    "DEFAULT_BIND",
    "DEFAULT_BRIDGE_HEARTBEAT_SECONDS",
    "DEFAULT_BRIDGE_POLL_SECONDS",
    "DEFAULT_HOST_PATH",
    "DEFAULT_USPEX_PATH",
    "MasterCommand",
    "bridge_script_path",
    "build_master_command",
    "uspex_command_executable",
]
