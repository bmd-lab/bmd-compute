"""
Read the ``results*/`` folders written by a USPEX 25.0 structure search.

Design: ``docs/structure_search.md`` (issue #62, part of #58).

What this module does
---------------------
* :func:`summarise_search` reads every ``results1/``, ``results2/``, ...
  folder of one USPEX calculation directory and returns a compact,
  JSON-serialisable summary: the best energy per generation and a ranked
  list of candidates (ID, composition, energy, volume, space group,
  generation, results folder).
* :func:`load_structures` returns ``pymatgen.Structure`` objects for chosen
  individual IDs, read from one of the ``*POSCARS`` files.
* :func:`structure_symmetry` recomputes the space group of a structure with
  pymatgen, because USPEX's own space groups are often under-determined.

File format
-----------
The format handled here was observed in real USPEX 25.0 output (build of
11 November 2025) from the Phase 0 Si8 pilot (#59), kept as the test
fixture ``tests/fixtures/uspex_si8_pilot/results1/``. It differs from the
25.0 manual in file names (for example ``good_structures_POSCARS``, not
``goodStructures_POSCARS``). Only one search, with one element, has been
observed, so everything format-dependent is kept in a few named
definitions that are easy to correct:

* :data:`COLUMN_ALIASES` and :data:`IGNORED_COLUMNS` -- header names in
  ``Individuals`` and ``BESTIndividuals``. Columns are always located by
  header name, never by position. A header name in neither table stops
  parsing with :class:`USPEXFormatError` naming it, instead of guessing
  what it means.
* :data:`POSCAR_ID_PATTERN` -- the regular expression that takes the
  individual ID from the first line of each POSCAR block
  (``number=<id>, energy=...``).
* :data:`MISSING_VALUE_TOKENS` -- how a missing energy is recognised. The
  pilot had no failed individuals, so this is not yet confirmed.

Observed facts the code relies on:

* ``energy`` is in eV per cell. The pilot write-up (#59) describes it as the
  enthalpy at the run's pressure; USPEX labels it only "energy", and this
  module reports it as ``energy_ev`` without claiming more.
* ``num_atoms_all`` and ``parents`` are Python-style lists such as ``[8]``
  or ``[2, 8]``; a bracketed group is read as one field even if it contains
  spaces. The counts in ``num_atoms_all`` are taken to follow the order of
  ``% atomType`` in ``Parameters.txt``. This is inferred, not checked: the
  pilot had a single element.
* The tables are fixed-width (header and every row have the same length)
  and the last row is written **without** a final newline.
* Survivors of a generation are recalculated and appear again as new
  individuals with origin ``keptBest``, a new ID and the survivor as their
  single parent (see "Candidates" in :func:`summarise_search`).

Behaviour on running searches
-----------------------------
A search may still be writing its files, so:

* In ``Individuals`` and ``BESTIndividuals`` a last line without a newline
  is normal (see above). It is kept only if it is at least as long as the
  header line and parses; otherwise it is treated as still being written,
  ignored, and a warning is recorded.
* In ``*POSCARS`` files, which do end in a newline, a last line without one
  is ignored, and an incomplete last POSCAR block is ignored, with a
  warning.
* Other rows that cannot be read are skipped and counted in ``warnings``;
  they do not stop the parse. Only an unrecognised *header* is an error.
* A missing file or folder is reported as a warning, not an error.
* The summary reads only the small text tables. POSCAR blocks are scanned
  with plain string handling and only the requested blocks are passed to
  pymatgen, so runtime stays short even on large ``gatheredPOSCARS`` files.

Self-contained module
---------------------
This file imports only the Python standard library at import time, and
pymatgen lazily inside the functions that build structures. It must not
import from the ``backend`` package, so that the same source can later be
sent to POWER and run beside the results folders, in the same way as
``backend/remote_result_parser.py``. (That wiring is #63/#68, not here.)
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any


SUMMARY_SCHEMA = "bmd-compute.uspex-results.1"

INDIVIDUALS_FILE = "Individuals"
BEST_INDIVIDUALS_FILE = "BESTIndividuals"
GATHERED_POSCARS_FILE = "gatheredPOSCARS"
GOOD_POSCARS_FILE = "good_structures_POSCARS"
BEST_GATHERED_POSCARS_FILE = "BESTgatheredPOSCARS"
BEST_STRUCTURE_POSCAR_FILE = "best_structure_POSCAR"
PARAMETERS_FILE = "Parameters.txt"
POSCARS_FILES = (
    GATHERED_POSCARS_FILE,
    GOOD_POSCARS_FILE,
    BEST_GATHERED_POSCARS_FILE,
    BEST_STRUCTURE_POSCAR_FILE,
)

RESULTS_FOLDER_PATTERN = re.compile(r"^results(\d+)$")

# Origin USPEX gives to a recalculated survivor of the previous generation.
KEPT_BEST_ORIGIN = "keptBest"

# Symmetry tolerance (Angstrom) for structure_symmetry(). In the Si8 pilot the
# best structure is listed by USPEX as space group 1 but is diamond Si
# (Fd-3m) for symprec >= 0.05; 0.1 is a provisional default, not BMD policy.
SYMMETRY_SYMPREC = 0.1

# Safety limit so a damaged or unexpected file cannot make the parser read
# gigabytes on a login node. Not a scientific setting.
MAX_FILE_BYTES = 256 * 1024 * 1024

# --- Format definitions (see module docstring) -------------------------------

# Canonical field -> header names that map to it (compared lower-cased).
# Names are those of the USPEX 25.0 Si8 pilot output.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "generation": ("generation",),
    "id": ("number",),
    "composition": ("num_atoms_all",),
    "energy_ev": ("energy",),
    "volume_a3": ("cell_volume",),
    "space_group_uspex": ("space_group",),
    "origin": ("origin",),
    "parents": ("parents",),
}
# Header names that are recognised but not used.
IGNORED_COLUMNS = frozenset(
    {
        "e_above_hull",
        "opt_property",
        "quasi_entropy",
        "a_order",
        "s_order",
        "num_blocks",
        "expected_origin",
    }
)
REQUIRED_COLUMNS = ("generation", "id", "composition", "energy_ev")

# The individual ID is the first group of this pattern, searched in the first
# line of each POSCAR block, e.g. "number=35, energy=-43.279875, ..." or
# "Gen=4, generation=4, number=35, energy=...".
POSCAR_ID_PATTERN = re.compile(r"\bnumber=(\d+)\b")

# Energy values that mean "no energy" (for example a failed relaxation).
# Any value that is not a finite number is also treated as missing.
MISSING_VALUE_TOKENS = frozenset({"nan", "n/a", "na", "none", "-"})

# ---------------------------------------------------------------------------

_ROW_TOKEN_PATTERN = re.compile(r"\[[^\[\]]*\]|\S+")
_ATOM_TYPE_BLOCK_PATTERN = re.compile(
    r"^\s*%\s*atomType\s*$(.*?)^\s*%\s*EndAtomType\s*$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


class USPEXResultsError(ValueError):
    """The results could not be read (missing directory, unknown ID, ...)."""


class USPEXFormatError(USPEXResultsError):
    """A results file has a layout this parser does not recognise."""


@dataclass(frozen=True)
class IndividualRow:
    """One row of ``Individuals`` or ``BESTIndividuals``."""

    results_folder: str
    generation: int
    id: int
    origin: str | None
    parents: tuple[int, ...] | None
    composition: tuple[int, ...]
    energy_ev: float | None
    volume_a3: float | None
    space_group_uspex: int | str | None

    @property
    def natoms(self) -> int:
        return sum(self.composition)

    @property
    def energy_ev_per_atom(self) -> float | None:
        return None if self.energy_ev is None else self.energy_ev / self.natoms


# --- Folder discovery --------------------------------------------------------


def find_results_folders(calculation_dir: str) -> list[str]:
    """
    Return the names of the ``resultsN`` folders, ordered by N.

    USPEX writes ``results1/`` on the first run and a new ``results2/``,
    ``results3/`` ... on each restart (manual §3.2, §4.7). The order is
    numeric, so ``results10`` comes after ``results9``.
    """

    if not os.path.isdir(calculation_dir):
        raise USPEXResultsError(
            f"USPEX calculation directory not found: {calculation_dir}"
        )
    found = []
    for name in os.listdir(calculation_dir):
        match = RESULTS_FOLDER_PATTERN.match(name)
        if match and os.path.isdir(os.path.join(calculation_dir, name)):
            found.append((int(match.group(1)), name))
    return [name for _, name in sorted(found)]


# --- Reading text that may still be growing ----------------------------------


def _read_lines(path: str) -> tuple[list[str], bool] | None:
    """
    Return ``(lines, ends_with_newline)``, or None if the file does not exist.
    """

    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size > MAX_FILE_BYTES:
        raise USPEXResultsError(
            f"{path} is {size} bytes, larger than the {MAX_FILE_BYTES}-byte "
            "safety limit for USPEX result files."
        )
    with open(path, "rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    text = data.decode("utf-8", errors="replace")
    ends_with_newline = text.endswith("\n")
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if ends_with_newline:
        lines.pop()
    return lines, ends_with_newline


# --- Individuals / BESTIndividuals ------------------------------------------


def _column_map(header: str, path: str) -> dict[str, int]:
    alias_to_field = {
        alias: field for field, aliases in COLUMN_ALIASES.items() for alias in aliases
    }
    columns: dict[str, int] = {}
    unknown = []
    for index, token in enumerate(header.split()):
        name = token.lower()
        if name in IGNORED_COLUMNS:
            continue
        field = alias_to_field.get(name)
        if field is None:
            unknown.append(token)
            continue
        if field in columns:
            raise USPEXFormatError(
                f"{path}: header has more than one column for '{field}' "
                f"(header line: {header.strip()!r})."
            )
        columns[field] = index
    if unknown:
        raise USPEXFormatError(
            f"{path}: unrecognised column header(s) {unknown!r} "
            f"(header line: {header.strip()!r}). If USPEX writes these names, "
            "add them to COLUMN_ALIASES or IGNORED_COLUMNS in "
            "backend/structure_search/results.py."
        )
    missing = [field for field in REQUIRED_COLUMNS if field not in columns]
    if missing:
        raise USPEXFormatError(
            f"{path}: header is missing required column(s) {missing!r} "
            f"(header line: {header.strip()!r})."
        )
    return columns


def _parse_int_list(token: str) -> tuple[int, ...]:
    if not (token.startswith("[") and token.endswith("]")):
        raise ValueError(f"expected a bracketed list, got {token!r}")
    body = token[1:-1].replace(",", " ")
    return tuple(int(part) for part in body.split())


def _parse_composition(token: str) -> tuple[int, ...]:
    counts = _parse_int_list(token)
    if not counts or any(count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError(f"invalid composition {token!r}")
    return counts


def _parse_optional_float(token: str) -> float | None:
    if token.strip().lower() in MISSING_VALUE_TOKENS:
        return None
    value = float(token)
    return value if math.isfinite(value) else None


def _parse_space_group(token: str) -> int | str:
    try:
        return int(token)
    except ValueError:
        return token


def _optional_column(tokens, columns, field, convert):
    index = columns.get(field)
    if index is None:
        return None
    return convert(tokens[index])


def _parse_row(tokens, columns, results_folder: str) -> IndividualRow:
    return IndividualRow(
        results_folder=results_folder,
        generation=int(tokens[columns["generation"]]),
        id=int(tokens[columns["id"]]),
        origin=_optional_column(tokens, columns, "origin", str),
        parents=_optional_column(tokens, columns, "parents", _parse_int_list),
        composition=_parse_composition(tokens[columns["composition"]]),
        energy_ev=_parse_optional_float(tokens[columns["energy_ev"]]),
        volume_a3=_optional_column(tokens, columns, "volume_a3", _parse_optional_float),
        space_group_uspex=_optional_column(
            tokens, columns, "space_group_uspex", _parse_space_group
        ),
    )


def read_individuals_file(
    path: str,
    results_folder: str,
    warnings: list[str],
) -> list[IndividualRow] | None:
    """
    Parse one ``Individuals`` or ``BESTIndividuals`` file.

    Returns None if the file does not exist, and an empty list if it exists
    but holds no complete data rows yet. The first non-blank line is the
    header. A bracketed group such as ``[2, 8]`` counts as one field. Rows
    that cannot be read are skipped and counted in ``warnings``.

    USPEX writes the last row without a final newline. That row is kept only
    if it is at least as long as the header (the tables are fixed-width, so
    a row cut short while being written is shorter) and it parses.
    """

    read = _read_lines(path)
    if read is None:
        return None
    lines, ends_with_newline = read

    content = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    if not content:
        return []
    header_number, header = content[0]
    if len(content) == 1 and not ends_with_newline and header_number == len(lines):
        # Only a header that may itself still be being written.
        return []
    columns = _column_map(header, path)
    width = len(header.split())

    rows: list[IndividualRow] = []
    skipped: list[int] = []
    for number, line in content[1:]:
        unterminated = number == len(lines) and not ends_with_newline
        tokens = _ROW_TOKEN_PATTERN.findall(line)
        try:
            if len(tokens) != width:
                raise ValueError("wrong number of fields")
            if unterminated and len(line.rstrip()) < len(header.rstrip()):
                raise ValueError("unterminated last line shorter than header")
            rows.append(_parse_row(tokens, columns, results_folder))
        except (ValueError, IndexError):
            if unterminated:
                warnings.append(
                    f"{path}: last line (line {number}) looks unfinished and was "
                    "ignored (file may still be being written)."
                )
            else:
                skipped.append(number)

    if skipped:
        shown = ", ".join(str(number) for number in skipped[:5])
        more = "" if len(skipped) <= 5 else f" and {len(skipped) - 5} more"
        warnings.append(
            f"{path}: skipped {len(skipped)} unreadable row(s) at line(s) {shown}{more}."
        )
    return rows


# --- Parameters.txt ----------------------------------------------------------


def read_atom_types(calculation_dir: str, folders: list[str]) -> list[str] | None:
    """
    Return the element symbols from the ``% atomType`` block, or None.

    Looks in ``resultsN/Parameters.txt`` (a copy of ``INPUT.txt``) of each
    folder in order, then in ``INPUT.txt`` of the calculation directory.
    Atomic numbers are accepted and converted to symbols.
    """

    candidates = [os.path.join(calculation_dir, f, PARAMETERS_FILE) for f in folders]
    candidates.append(os.path.join(calculation_dir, "INPUT.txt"))
    for path in candidates:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read(1024 * 1024)
        except OSError:
            continue
        match = _ATOM_TYPE_BLOCK_PATTERN.search(text)
        if not match:
            continue
        tokens = match.group(1).split()
        if not tokens:
            continue
        if any(token.isdigit() for token in tokens):
            from pymatgen.core.periodic_table import Element

            tokens = [
                Element.from_Z(int(token)).symbol if token.isdigit() else token
                for token in tokens
            ]
        return tokens
    return None


# --- Summary -----------------------------------------------------------------


def summarise_search(calculation_dir: str, *, max_candidates: int = 50) -> dict:
    """
    Return a JSON-serialisable summary of a USPEX calculation directory.

    Parameters
    ----------
    calculation_dir
        The folder in which USPEX was run (the one that holds ``results1/``).
    max_candidates
        Maximum number of ranked candidates returned. ``n_candidates`` gives
        the total.

    Returns
    -------
    dict with these keys:

    ``schema``
        Version tag of this layout.
    ``results_folders``
        ``["results1", "results2", ...]`` in numeric order.
    ``atom_types``
        Element symbols from ``atomType``, or None if not found.
    ``n_individuals``, ``n_individuals_without_energy``
        Counts over all folders, every row included.
    ``generations``
        One entry per (results folder, generation), from the lowest to the
        highest generation number seen in that folder, including generations
        with no rows ("empty"). Every individual counts here, recalculated
        survivors included. ``best_id`` and the best energies are None when
        no individual of that generation has an energy.
    ``best_individuals``
        The rows of every ``BESTIndividuals`` file, as written by USPEX.
    ``candidates``
        Ranked by energy per atom, lowest first. See "Candidates" below.
    ``n_candidates``
        Total number of ranked candidates before ``max_candidates``.
    ``warnings``
        Human-readable notes about missing files, unfinished lines and
        skipped rows.

    Candidates
    ----------
    Individuals are identified by (results folder, ID); if the same ID
    appears twice in one folder, the later row wins. Whether IDs continue
    across a restart is not known, so folders are never merged.

    USPEX recalculates the survivors of each generation as new individuals
    with origin ``keptBest`` and the survivor as their single parent, so one
    structure can appear several times (in the Si8 pilot 12 -> 26 -> 35).
    Within one results folder, each such chain is listed once, as its member
    with the lowest energy per atom (ties: the latest ID). ``lineage_ids``
    lists every ID of the chain. A recalculation can end in a different
    minimum (pilot: 8 -> 17 rose by 0.32 eV per cell), so a hidden member
    is not necessarily the same structure; it remains in ``generations`` and
    in ``lineage_ids``. Otherwise, equivalent structures with different IDs
    are not merged; that needs a structure comparison, left to a later step.

    Each candidate has ``energy_ev`` (USPEX ``energy``, eV per cell),
    ``energy_ev_per_atom``, volumes, ``space_group_uspex`` (as reported by
    USPEX, often under-determined; see :func:`structure_symmetry`) and
    ``in_good_structures`` (listed in ``good_structures_POSCARS``). Energies
    are USPEX's search-grade values from its last relaxation step.
    """

    calculation_dir = os.path.abspath(calculation_dir)
    folders = find_results_folders(calculation_dir)
    warnings: list[str] = []
    if not folders:
        warnings.append(f"{calculation_dir}: no resultsN folders found yet.")

    atom_types = read_atom_types(calculation_dir, folders)
    if atom_types is None and folders:
        warnings.append(
            "No '% atomType' block found in Parameters.txt or INPUT.txt; "
            "compositions are given as counts only."
        )

    individuals: list[IndividualRow] = []
    best_rows: list[IndividualRow] = []
    good_ids: set[tuple[str, int]] = set()
    for folder in folders:
        folder_path = os.path.join(calculation_dir, folder)
        rows = read_individuals_file(
            os.path.join(folder_path, INDIVIDUALS_FILE), folder, warnings
        )
        if rows is None:
            warnings.append(f"{folder}/{INDIVIDUALS_FILE} not found.")
        else:
            individuals.extend(rows)
        best = read_individuals_file(
            os.path.join(folder_path, BEST_INDIVIDUALS_FILE), folder, warnings
        )
        if best is not None:
            best_rows.extend(best)
        good_path = os.path.join(folder_path, GOOD_POSCARS_FILE)
        for block in _scan_poscars_file(good_path, warnings) or ():
            if block.id is not None:
                good_ids.add((folder, block.id))

    unique: dict[tuple[str, int], IndividualRow] = {}
    for row in individuals:
        unique[(row.results_folder, row.id)] = row

    lineages = _kept_best_lineages(unique)
    ranked = []
    for members in lineages:
        scored = [row for row in members if row.energy_ev is not None]
        if not scored:
            continue
        representative = min(
            scored, key=lambda row: (row.energy_ev_per_atom, -row.id)
        )
        ranked.append((representative, sorted(row.id for row in members)))
    ranked.sort(
        key=lambda item: (
            item[0].energy_ev_per_atom,
            folders.index(item[0].results_folder),
            item[0].id,
        )
    )

    candidates = []
    for rank, (row, lineage_ids) in enumerate(ranked[: max(0, max_candidates)], 1):
        entry = _row_payload(row, atom_types)
        entry["rank"] = rank
        entry["lineage_ids"] = lineage_ids
        entry["in_good_structures"] = (row.results_folder, row.id) in good_ids
        candidates.append(entry)

    return {
        "schema": SUMMARY_SCHEMA,
        "calculation_dir": calculation_dir,
        "results_folders": folders,
        "atom_types": atom_types,
        "n_individuals": len(unique),
        "n_individuals_without_energy": sum(
            1 for row in unique.values() if row.energy_ev is None
        ),
        "generations": _generation_summary(unique.values(), folders),
        "best_individuals": [_row_payload(row, atom_types) for row in best_rows],
        "candidates": candidates,
        "n_candidates": len(ranked),
        "warnings": warnings,
    }


def _kept_best_lineages(
    unique: dict[tuple[str, int], IndividualRow],
) -> list[list[IndividualRow]]:
    """
    Group each ``keptBest`` individual with its parent, within one folder.

    Returns a list of groups; an individual that is neither a recalculated
    survivor nor recalculated itself forms a group of one.
    """

    root: dict[tuple[str, int], tuple[str, int]] = {}

    def find(key):
        while root.get(key, key) != key:
            key = root[key]
        return key

    for key, row in sorted(unique.items(), key=lambda item: (item[0][0], item[0][1])):
        root.setdefault(key, key)
        if row.origin != KEPT_BEST_ORIGIN or not row.parents or len(row.parents) != 1:
            continue
        parent_key = (row.results_folder, row.parents[0])
        if parent_key in unique:
            root[find(key)] = find(parent_key)

    groups: dict[tuple[str, int], list[IndividualRow]] = {}
    for key, row in unique.items():
        groups.setdefault(find(key), []).append(row)
    return list(groups.values())


def _generation_summary(rows, folders: list[str]) -> list[dict]:
    by_folder: dict[str, dict[int, list[IndividualRow]]] = {}
    for row in rows:
        by_folder.setdefault(row.results_folder, {}).setdefault(row.generation, []).append(row)

    summary = []
    for folder in folders:
        generations = by_folder.get(folder)
        if not generations:
            continue
        for generation in range(min(generations), max(generations) + 1):
            members = generations.get(generation, [])
            scored = [row for row in members if row.energy_ev is not None]
            best = min(scored, key=lambda row: (row.energy_ev_per_atom, row.id), default=None)
            summary.append(
                {
                    "results_folder": folder,
                    "generation": generation,
                    "n_individuals": len(members),
                    "n_with_energy": len(scored),
                    "best_id": best.id if best else None,
                    "best_energy_ev": _round(best.energy_ev if best else None),
                    "best_energy_ev_per_atom": _round(
                        best.energy_ev_per_atom if best else None
                    ),
                }
            )
    return summary


def _row_payload(row: IndividualRow, atom_types: list[str] | None) -> dict:
    natoms = row.natoms
    return {
        "id": row.id,
        "results_folder": row.results_folder,
        "generation": row.generation,
        "origin": row.origin,
        "parents": None if row.parents is None else list(row.parents),
        "composition": list(row.composition),
        "formula": _formula(row.composition, atom_types),
        "natoms": natoms,
        "energy_ev": _round(row.energy_ev),
        "energy_ev_per_atom": _round(row.energy_ev_per_atom),
        "volume_a3": _round(row.volume_a3),
        "volume_per_atom_a3": _round(
            row.volume_a3 / natoms if row.volume_a3 is not None else None
        ),
        "space_group_uspex": row.space_group_uspex,
    }


def _formula(composition: tuple[int, ...], atom_types: list[str] | None) -> str | None:
    if not atom_types or len(atom_types) != len(composition):
        return None
    return "".join(
        f"{symbol}{count}" for symbol, count in zip(atom_types, composition) if count
    )


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


# --- POSCAR blocks -----------------------------------------------------------


@dataclass(frozen=True)
class _PoscarBlock:
    line_number: int
    comment: str
    id: int | None
    species: tuple[str, ...] | None
    lines: tuple[str, ...]


def _scan_poscars_file(path: str, warnings: list[str]) -> list[_PoscarBlock] | None:
    read = _read_lines(path)
    if read is None:
        return None
    lines, ends_with_newline = read
    if not ends_with_newline and lines:
        last = lines.pop()
        if last.strip():
            warnings.append(
                f"{path}: last line has no newline and was ignored "
                "(file may still be being written)."
            )
    return _scan_poscar_blocks(lines, path, warnings)


def _scan_poscar_blocks(
    lines: list[str], path: str, warnings: list[str]
) -> list[_PoscarBlock]:
    """
    Split concatenated POSCARs into blocks without calling pymatgen.

    Each block is: comment, scale, three lattice lines, an optional species
    line, counts, an optional "Selective dynamics" line, the coordinate-mode
    line, then one line per atom. Blank lines between blocks are allowed.
    An incomplete last block (file still being written) is dropped with a
    warning. A malformed block stops the scan, because the start of the next
    block cannot be found reliably after it.
    """

    blocks = []
    index = 0
    total = len(lines)
    while index < total:
        if not lines[index].strip():
            index += 1
            continue
        start = index
        try:
            float(lines[start + 1].split()[0])
            for offset in (2, 3, 4):
                vector = [float(value) for value in lines[start + offset].split()[:3]]
                if len(vector) != 3:
                    raise ValueError("lattice line")
            cursor = start + 5
            tokens = lines[cursor].split()
            species = None
            if not all(token.isdigit() for token in tokens):
                species = tuple(tokens)
                cursor += 1
            counts = [int(token) for token in lines[cursor].split()]
            if not counts or (species is not None and len(species) != len(counts)):
                raise ValueError("counts line")
            cursor += 1
            if lines[cursor].strip()[:1] in ("s", "S"):
                cursor += 1
            if lines[cursor].strip()[:1] not in ("d", "D", "c", "C", "k", "K"):
                raise ValueError("coordinate mode line")
            end = cursor + 1 + sum(counts)
        except IndexError:
            warnings.append(
                f"{path}: POSCAR block at line {start + 1} is incomplete and was "
                "ignored (file may still be being written)."
            )
            break
        except ValueError:
            warnings.append(
                f"{path}: could not read the POSCAR block at line {start + 1}; "
                "the rest of the file was ignored."
            )
            break
        if end > total:
            warnings.append(
                f"{path}: POSCAR block at line {start + 1} is incomplete and was "
                "ignored (file may still be being written)."
            )
            break
        comment = lines[start]
        match = POSCAR_ID_PATTERN.search(comment)
        blocks.append(
            _PoscarBlock(
                line_number=start + 1,
                comment=comment.strip(),
                id=int(match.group(1)) if match else None,
                species=species,
                lines=tuple(lines[start:end]),
            )
        )
        index = end

    unnamed = [block.line_number for block in blocks if block.id is None]
    if unnamed:
        warnings.append(
            f"{path}: {len(unnamed)} POSCAR block(s) have no individual ID matching "
            f"POSCAR_ID_PATTERN {POSCAR_ID_PATTERN.pattern!r} (first at line {unnamed[0]})."
        )
    return blocks


def _block_to_structure(block: _PoscarBlock, atom_types: list[str] | None, path: str):
    from pymatgen.core.periodic_table import Element
    from pymatgen.io.vasp import Poscar

    lines = list(block.lines)
    species = block.species
    if species is None:
        # VASP 4 style block without an element line: take the elements from
        # atomType rather than letting pymatgen invent placeholder species.
        counts = lines[5].split()
        if not atom_types or len(atom_types) != len(counts):
            raise USPEXResultsError(
                f"{path}: POSCAR block for ID {block.id} (line {block.line_number}) "
                "has no element line and atomType does not match its counts."
            )
        species = tuple(atom_types)
        lines.insert(5, " ".join(species))
    for symbol in species:
        if not Element.is_valid_symbol(symbol):
            raise USPEXResultsError(
                f"{path}: POSCAR block for ID {block.id} (line {block.line_number}) "
                f"has an invalid element symbol {symbol!r}."
            )
    return Poscar.from_str("\n".join(lines) + "\n").structure


def available_structure_ids(
    calculation_dir: str,
    results_folder: str,
    *,
    poscars_file: str = GATHERED_POSCARS_FILE,
) -> set[int] | None:
    """
    Return the individual IDs that have a POSCAR block in one ``*POSCARS`` file.

    Returns None if the file does not exist. Blocks are scanned with plain
    string handling (no pymatgen), so this is cheap even for large files.
    Unfinished or unreadable blocks are left out, as in :func:`load_structures`.
    """

    if poscars_file not in POSCARS_FILES:
        raise USPEXResultsError(
            f"poscars_file must be one of {POSCARS_FILES!r}, not {poscars_file!r}."
        )
    path = os.path.join(calculation_dir, results_folder, poscars_file)
    blocks = _scan_poscars_file(path, [])
    if blocks is None:
        return None
    return {block.id for block in blocks if block.id is not None}


def load_structures(
    calculation_dir: str,
    ids,
    *,
    poscars_file: str = GATHERED_POSCARS_FILE,
    results_folder: str | None = None,
) -> dict[int, Any]:
    """
    Return ``{id: pymatgen.Structure}`` for the requested individual IDs.

    Parameters
    ----------
    calculation_dir
        The folder in which USPEX was run.
    ids
        Individual IDs, as in the ``candidates`` of :func:`summarise_search`.
    poscars_file
        One of :data:`POSCARS_FILES`: ``"gatheredPOSCARS"`` (default, every
        relaxed individual), ``"good_structures_POSCARS"``,
        ``"BESTgatheredPOSCARS"`` or ``"best_structure_POSCAR"``.
    results_folder
        Restrict the search to one folder, e.g. ``"results2"``. Needed when
        the same ID appears in more than one results folder.

    Raises :class:`USPEXResultsError` if an ID is not found, or is found in
    more than one results folder and ``results_folder`` was not given.
    Only the matching POSCAR blocks are passed to pymatgen.
    """

    if poscars_file not in POSCARS_FILES:
        raise USPEXResultsError(
            f"poscars_file must be one of {POSCARS_FILES!r}, not {poscars_file!r}."
        )
    wanted = {int(value) for value in ids}
    all_folders = find_results_folders(calculation_dir)
    folders = all_folders
    if results_folder is not None:
        if results_folder not in all_folders:
            raise USPEXResultsError(
                f"{results_folder} not found in {calculation_dir}."
            )
        folders = [results_folder]

    warnings: list[str] = []
    found: dict[int, list[tuple[str, _PoscarBlock]]] = {}
    for folder in folders:
        path = os.path.join(calculation_dir, folder, poscars_file)
        for block in _scan_poscars_file(path, warnings) or ():
            if block.id in wanted:
                # Within one file the later block wins, as in the summary.
                existing = [
                    item for item in found.get(block.id, []) if item[0] != folder
                ]
                found[block.id] = existing + [(folder, block)]

    missing = sorted(wanted - set(found))
    if missing:
        detail = f" Warnings: {' '.join(warnings)}" if warnings else ""
        raise USPEXResultsError(
            f"Individual ID(s) {missing} not found in {poscars_file} under "
            f"{calculation_dir} (folders searched: {folders}).{detail}"
        )
    ambiguous = {
        found_id: [folder for folder, _ in items]
        for found_id, items in found.items()
        if len(items) > 1
    }
    if ambiguous:
        raise USPEXResultsError(
            f"Individual ID(s) found in more than one results folder: {ambiguous}. "
            "Pass results_folder to choose one."
        )

    atom_types = read_atom_types(calculation_dir, all_folders)
    structures = {}
    for found_id, ((folder, block),) in found.items():
        path = os.path.join(calculation_dir, folder, poscars_file)
        structures[found_id] = _block_to_structure(block, atom_types, path)
    return structures


def structure_symmetry(structure, *, symprec: float = SYMMETRY_SYMPREC) -> dict:
    """
    Return ``{"space_group_number", "space_group_symbol", "symprec"}``.

    Computed with pymatgen's ``SpacegroupAnalyzer`` (spglib). USPEX's own
    ``space_group`` column is often under-determined: in the Si8 pilot the
    best structure is listed as 1 (P1) but is diamond Si (227, Fd-3m) at
    ``symprec >= 0.05``. Not part of :func:`summarise_search`, which stays
    standard-library only.
    """

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    return {
        "space_group_number": analyzer.get_space_group_number(),
        "space_group_symbol": analyzer.get_space_group_symbol(),
        "symprec": symprec,
    }


__all__ = [
    "COLUMN_ALIASES",
    "IGNORED_COLUMNS",
    "IndividualRow",
    "MISSING_VALUE_TOKENS",
    "POSCAR_ID_PATTERN",
    "POSCARS_FILES",
    "SYMMETRY_SYMPREC",
    "USPEXFormatError",
    "USPEXResultsError",
    "available_structure_ids",
    "find_results_folders",
    "load_structures",
    "read_atom_types",
    "read_individuals_file",
    "structure_symmetry",
    "summarise_search",
]
