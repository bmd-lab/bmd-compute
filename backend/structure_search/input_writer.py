"""Write USPEX's ``INPUT.txt`` for an admitted structure search.

USPEX 25 does not report a missing or incomplete ``INPUT.txt``; in a pilot
run it crashed with an ``IndexError`` instead. bmd-compute therefore always
writes every keyword it relies on, from a spec that has passed admission,
and never leaves a value to a USPEX default.

File format (USPEX 25.0 manual): one scalar per line as ``value : name``,
and multi-value keywords as blocks opened by ``% name`` and closed by the
block's own end marker. The block markers used here are exactly:

========================  ===================
opening                   closing
========================  ===================
``% atomType``            ``% EndAtomType``
``% numSpecies``          ``% EndNumSpecies``
``% abinitioCode``        ``% ENDabinit``
``% KresolStart``         ``% Kresolend``
``% commandExecutable``   ``% EndExecutable``
========================  ===================

Two groups of settings go into the file:

- the user's choices, from :class:`~backend.structure_search.spec.StructureSearchSpec`;
- execution settings that bmd-compute fixes and the user never sets,
  from :class:`SearchExecutionSettings` and the constants below.

Pressure is written only as ``ExternalPressure``. The INCAR files must
never contain ``PSTRESS`` (manual section 4.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backend.calculations.registry import CalculationValidationError
from backend.structure_search.admission import require_admissible_search
from backend.structure_search.spec import StructureSearchSpec


INPUT_FILENAME = "INPUT.txt"

# Fixed execution settings (docs/structure_search.md section 4).
# whichCluster -1: local run without job submission. The USPEX master runs
# inside the SLURM allocation and starts each relaxation itself, as
# ``sh run_local.sh`` in the CalcFold, numParallelCalcs at a time, with the
# full container environment. USPEX's own CPU core control is not used;
# SLURM places the VASP ranks (via the launch bridge).
# Checked on POWER (probe job 23367013, 4 Nov 2026, issue #59): with
# whichCluster 0, USPEX 25 never ran commandExecutable, even when
# run_local.sh was made executable; with -1 it did.
WHICH_CLUSTER = -1
# coresPerJob 1: in mode 0 it would make USPEX's core control pin only the
# waiting bridge script. Whether mode -1 reads coresPerJob at all is
# untested; it is kept so the file is unchanged if it does.
CORES_PER_JOB = 1
# abinitioCode 1 = VASP, for every relaxation step.
ABINITIO_CODE_VASP = 1

# The bridge path is inserted into a shell command line
# (``sh <bridge path> <step>``), so only plain absolute paths are accepted.
_SAFE_ABSOLUTE_PATH = re.compile(r"^/[A-Za-z0-9._/+-]+$")


class SearchInputError(CalculationValidationError):
    """Execution settings for ``INPUT.txt`` are missing or invalid."""

    def __init__(self, message: str, *, code: str, **details):
        super().__init__(message, diagnostic={"code": code, **details})
        self.code = code


@dataclass(frozen=True)
class SearchExecutionSettings:
    """Execution settings bmd-compute chooses for a search; never user input.

    - ``bridge_path``: absolute path of the launch-bridge script on the
      cluster (defined by the launch bridge, #66). Relaxation step ``n``
      runs ``sh <bridge_path> n``.
    - ``num_parallel_calcs``: ``numParallelCalcs``, the number of
      concurrent relaxations. Always written explicitly: the manual gives
      two different defaults (1, and auto-detect available cores, which on a
      192-core node could mean 192 concurrent relaxations).
    - ``sleep_seconds``: ``sleepSeconds``, how often the USPEX master
      checks for finished relaxations. The manual shows this keyword only in
      prose (``sleepSeconds = 5``, section 2.8), and the Si8 pilot did not
      set it, so the ``value : sleepSeconds`` line form is not yet confirmed
      on a real run (Phase 0, #59).

    None of these has a default here; the execution policy (#67) chooses them.
    """

    bridge_path: str
    num_parallel_calcs: int
    sleep_seconds: int

    def __post_init__(self) -> None:
        path = str(self.bridge_path or "")
        if not _SAFE_ABSOLUTE_PATH.fullmatch(path) or "/../" in f"{path}/":
            raise SearchInputError(
                "The bridge path must be an absolute path made of letters, digits "
                "and . _ / + - only (it is used on a shell command line).",
                code="invalid_bridge_path",
                bridge_path=path,
            )
        for label in ("num_parallel_calcs", "sleep_seconds"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SearchInputError(
                    f"{label} must be a whole number of at least 1.",
                    code="invalid_execution_setting",
                    field=label,
                )


def command_executable_lines(bridge_path: str, num_steps: int) -> list[str]:
    """``commandExecutable`` entries: line ``n`` is ``sh <bridge_path> n``."""

    return [f"sh {bridge_path} {step}" for step in range(1, num_steps + 1)]


def render_input_text(spec: StructureSearchSpec, settings: SearchExecutionSettings) -> str:
    """Return the complete ``INPUT.txt`` text for an admissible spec.

    Raises the admission error if bmd-compute does not support ``spec``.
    """

    require_admissible_search(spec)
    if not isinstance(settings, SearchExecutionSettings):
        raise TypeError("settings must be a SearchExecutionSettings")

    steps = spec.num_steps
    lines: list[str] = []
    lines += [_scalar(spec.calculation_type, "calculationType"), ""]
    lines += _block("atomType", "EndAtomType", [" ".join(spec.elements)])
    lines += _block("numSpecies", "EndNumSpecies", [" ".join(str(n) for n in spec.counts)])
    lines += [
        _scalar(_number(spec.external_pressure_gpa), "ExternalPressure"),
        _scalar(spec.min_atoms, "minAt"),
        _scalar(spec.max_atoms, "maxAt"),
        "",
        _scalar(spec.resolved_population_size, "populationSize"),
        _scalar(spec.resolved_initial_population_size, "initialPopSize"),
        _scalar(spec.num_generations, "numGenerations"),
        _scalar(spec.stop_criterion, "stopCrit"),
        "",
    ]
    lines += _block("abinitioCode", "ENDabinit", [" ".join([str(ABINITIO_CODE_VASP)] * steps)])
    lines += _block("KresolStart", "Kresolend", [" ".join(_number(k) for k in spec.kresol_start)])
    lines += _block(
        "commandExecutable",
        "EndExecutable",
        command_executable_lines(settings.bridge_path, steps),
    )
    lines += [
        _scalar(WHICH_CLUSTER, "whichCluster"),
        _scalar(settings.num_parallel_calcs, "numParallelCalcs"),
        _scalar(CORES_PER_JOB, "coresPerJob"),
        _scalar(settings.sleep_seconds, "sleepSeconds"),
    ]
    return "\n".join(lines) + "\n"


def write_input_file(
    directory: str | Path,
    spec: StructureSearchSpec,
    settings: SearchExecutionSettings,
) -> Path:
    """Render ``INPUT.txt`` and write it into ``directory`` (which must exist)."""

    text = render_input_text(spec, settings)
    path = Path(directory) / INPUT_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def read_input_keywords(text: str) -> dict[str, list[str]]:
    """Parse ``INPUT.txt`` text written by this module back into keywords.

    Returns ``{name: [value lines]}``; a scalar has one entry. Intended for
    tests and pre-launch checks of files bmd-compute wrote, not as a general
    USPEX parser. Raises ``ValueError`` on a repeated keyword or an
    unterminated block.
    """

    keywords: dict[str, list[str]] = {}
    block_name: str | None = None
    block_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if block_name is not None:
            if line.startswith("%"):
                _store(keywords, block_name, block_lines)
                block_name, block_lines = None, []
            else:
                block_lines.append(line)
            continue
        if not line:
            continue
        if line.startswith("%"):
            block_name = line[1:].strip()
            continue
        value, separator, name = line.rpartition(":")
        if not separator:
            raise ValueError(f"Unrecognised INPUT.txt line: {raw!r}")
        _store(keywords, name.strip(), [value.strip()])
    if block_name is not None:
        raise ValueError(f"Unterminated INPUT.txt block: {block_name}")
    return keywords


def _store(keywords: dict[str, list[str]], name: str, values: Iterable[str]) -> None:
    if name in keywords:
        raise ValueError(f"Repeated INPUT.txt keyword: {name}")
    keywords[name] = list(values)


def _scalar(value, name: str) -> str:
    return f"{value} : {name}"


def _block(name: str, end: str, rows: list[str]) -> list[str]:
    return [f"% {name}", *rows, f"% {end}", ""]


def _number(value: float) -> str:
    """Shortest exact decimal for a float, without a trailing ``.0`` for integers."""

    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return "0" if text == "-0" else text


__all__ = [
    "ABINITIO_CODE_VASP",
    "CORES_PER_JOB",
    "INPUT_FILENAME",
    "SearchExecutionSettings",
    "SearchInputError",
    "WHICH_CLUSTER",
    "command_executable_lines",
    "read_input_keywords",
    "render_input_text",
    "write_input_file",
]
