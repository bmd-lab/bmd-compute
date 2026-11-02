"""
Read the ``results*/`` folders written by a USPEX 25.0 structure search.

Design: ``docs/structure_search.md`` (issue #62, part of #58).

What this module does
---------------------
* :func:`summarise_search` reads every ``results1/``, ``results2/``, ...
  folder of one USPEX calculation directory and returns a compact,
  JSON-serialisable summary: the best enthalpy per generation and a ranked
  list of candidates (ID, composition, enthalpy, volume, space group,
  generation, results folder).
* :func:`load_structures` returns ``pymatgen.Structure`` objects for chosen
  individual IDs, read from ``gatheredPOSCARS`` or ``goodStructures_POSCARS``.

PROVISIONAL FILE FORMAT -- READ BEFORE CHANGING
-----------------------------------------------
The USPEX 25.0 manual names the output files but does not document the
column layout of ``Individuals`` / ``BESTIndividuals``, the comment line of
the ``*POSCARS`` files, or which file holds the enthalpies that go with
``goodStructures_POSCARS``. None of these had been observed on real output
when this module was written; the Phase 0 pilot (#59) will provide real
files. Everything that depends on those unknowns is kept in three places
so it is easy to correct:

* :data:`COLUMN_ALIASES` -- header names recognised in ``Individuals`` and
  ``BESTIndividuals``. Columns are always located by header name, never by
  position. A header name that is not in this table stops parsing with
  :class:`USPEXFormatError`, naming the header, instead of guessing.
* :data:`POSCAR_ID_PATTERN` -- the regular expression that takes the
  individual ID from the comment (first) line of each POSCAR block.
* :data:`MISSING_ENTHALPY_TOKENS` -- how a missing enthalpy is recognised.

The test fixtures in ``tests/fixtures/uspex_synthetic/`` are synthetic and
follow these provisional choices. They must be replaced by, or checked
against, real Phase 0 output before the import UI (#63) is merged.

Behaviour on running searches
-----------------------------
A search may still be writing its files, so:

* A file that does not end in a newline is treated as still being written;
  its incomplete last line is ignored and a warning is recorded.
* Rows that cannot be read are skipped and counted in ``warnings``; they do
  not stop the parse. Only an unrecognised *header* is an error.
* A missing file or folder is reported as a warning, not an error.
* The summary reads only the small text tables. POSCAR blocks are scanned
  with plain string handling and only the requested blocks are passed to
  pymatgen, so runtime stays short even on large ``gatheredPOSCARS`` files.

Self-contained module
---------------------
This file imports only the Python standard library at import time, and
pymatgen lazily inside :func:`load_structures`. It must not import from the
``backend`` package, so that the same source can later be sent to POWER and
run beside the results folders, in the same way as
``backend/remote_result_parser.py``. (That wiring is #63/#68, not here.)
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any


SUMMARY_SCHEMA = "bmd-compute.uspex-results.provisional-1"

INDIVIDUALS_FILE = "Individuals"
BEST_INDIVIDUALS_FILE = "BESTIndividuals"
GATHERED_POSCARS_FILE = "gatheredPOSCARS"
GOOD_POSCARS_FILE = "goodStructures_POSCARS"
PARAMETERS_FILE = "Parameters.txt"
POSCARS_FILES = (GATHERED_POSCARS_FILE, GOOD_POSCARS_FILE)

RESULTS_FOLDER_PATTERN = re.compile(r"^results(\d+)$")

# Safety limit so a damaged or unexpected file cannot make the parser read
# gigabytes on a login node. Not a scientific setting.
MAX_FILE_BYTES = 256 * 1024 * 1024

# --- Provisional format definitions (see module docstring) -----------------

# Canonical field -> header names that map to it. Header names are compared
# after lower-casing; each header token is one whitespace-separated word.
# Units are part of the name on purpose: a bare "Enthalpy" header would not
# say whether the value is per cell or per atom, so it is rejected.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "generation": ("gen", "generation"),
    "id": ("id",),
    "origin": ("origin",),
    "composition": ("composition",),
    "enthalpy_ev": ("enthalpy(ev)",),
    "enthalpy_ev_per_atom": ("enthalpy(ev/atom)",),
    "volume_a3": ("volume(a^3)",),
    "space_group": ("symm", "spacegroup"),
}
REQUIRED_COLUMNS = ("generation", "id", "composition")
ENTHALPY_COLUMNS = ("enthalpy_ev", "enthalpy_ev_per_atom")

# The individual ID is the first group of this pattern, searched in the
# comment line of each POSCAR block, for example "EA17 ...".
POSCAR_ID_PATTERN = re.compile(r"\bEA(\d+)\b")

# Enthalpy values that mean "no enthalpy" (for example a failed relaxation).
# Any value that is not a finite number is also treated as missing.
MISSING_ENTHALPY_TOKENS = frozenset({"nan", "n/a", "na", "none", "-"})

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
    composition: tuple[int, ...]
    enthalpy_ev: float | None
    enthalpy_ev_per_atom: float | None
    volume_a3: float | None
    space_group: int | str | None

    @property
    def natoms(self) -> int:
        return sum(self.composition)


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


def _read_complete_lines(path: str, warnings: list[str]) -> list[str] | None:
    """
    Return the complete lines of a text file, or None if it does not exist.

    If the file does not end in a newline, the last line is assumed to be
    still being written and is dropped, with a warning.
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
    lines = text.split("\n")
    # text.split leaves "" after a final newline; otherwise the last element
    # is an unfinished line.
    last = lines.pop()
    if last.strip():
        warnings.append(
            f"{path}: last line has no newline and was ignored "
            "(file may still be being written)."
        )
    return [line.rstrip("\r") for line in lines]


# --- Individuals / BESTIndividuals ------------------------------------------


def _column_map(header: str, path: str) -> dict[str, int]:
    alias_to_field = {
        alias: field for field, aliases in COLUMN_ALIASES.items() for alias in aliases
    }
    columns: dict[str, int] = {}
    unknown = []
    for index, token in enumerate(header.split()):
        field = alias_to_field.get(token.lower())
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
        hint = ""
        if any(token.lower() == "enthalpy" for token in unknown):
            hint = (
                " A bare 'Enthalpy' header does not give its unit (eV per cell "
                "or eV per atom), so it is not guessed."
            )
        raise USPEXFormatError(
            f"{path}: unrecognised column header(s) {unknown!r} "
            f"(header line: {header.strip()!r}).{hint} If the real USPEX "
            "format uses these names, add them to COLUMN_ALIASES in "
            "backend/structure_search/results.py."
        )
    missing = [field for field in REQUIRED_COLUMNS if field not in columns]
    if not any(field in columns for field in ENTHALPY_COLUMNS):
        missing.append(" or ".join(ENTHALPY_COLUMNS))
    if missing:
        raise USPEXFormatError(
            f"{path}: header is missing required column(s) {missing!r} "
            f"(header line: {header.strip()!r})."
        )
    return columns


def _parse_int(token: str) -> int:
    return int(token)


def _parse_composition(token: str) -> tuple[int, ...]:
    body = token[1:-1] if token.startswith("[") and token.endswith("]") else token
    counts = tuple(int(part) for part in body.split())
    if not counts or any(count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError(f"invalid composition {token!r}")
    return counts


def _parse_optional_float(token: str) -> float | None:
    if token.strip().lower() in MISSING_ENTHALPY_TOKENS:
        return None
    value = float(token)
    return value if math.isfinite(value) else None


def _parse_space_group(token: str) -> int | str:
    try:
        return int(token)
    except ValueError:
        return token


def read_individuals_file(
    path: str,
    results_folder: str,
    warnings: list[str],
) -> list[IndividualRow] | None:
    """
    Parse one ``Individuals`` or ``BESTIndividuals`` file.

    Returns None if the file does not exist, and an empty list if it exists
    but holds no complete data rows yet. The first non-blank line is the
    header. In data rows, a bracketed group such as ``[4 4]`` counts as one
    field. Rows that cannot be read are skipped and counted in ``warnings``.
    """

    lines = _read_complete_lines(path, warnings)
    if lines is None:
        return None

    content = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    if not content:
        return []

    _, header = content[0]
    columns = _column_map(header, path)
    width = len(header.split())

    rows: list[IndividualRow] = []
    skipped: list[int] = []
    for number, line in content[1:]:
        tokens = _ROW_TOKEN_PATTERN.findall(line)
        if len(tokens) != width:
            skipped.append(number)
            continue
        try:
            composition = _parse_composition(tokens[columns["composition"]])
            total = _optional_column(tokens, columns, "enthalpy_ev", _parse_optional_float)
            per_atom = _optional_column(
                tokens, columns, "enthalpy_ev_per_atom", _parse_optional_float
            )
            if per_atom is None and total is not None:
                per_atom = total / sum(composition)
            if total is None and per_atom is not None and "enthalpy_ev" not in columns:
                total = per_atom * sum(composition)
            rows.append(
                IndividualRow(
                    results_folder=results_folder,
                    generation=_parse_int(tokens[columns["generation"]]),
                    id=_parse_int(tokens[columns["id"]]),
                    origin=_optional_column(tokens, columns, "origin", str),
                    composition=composition,
                    enthalpy_ev=total,
                    enthalpy_ev_per_atom=per_atom,
                    volume_a3=_optional_column(tokens, columns, "volume_a3", float),
                    space_group=_optional_column(
                        tokens, columns, "space_group", _parse_space_group
                    ),
                )
            )
        except ValueError:
            skipped.append(number)

    if skipped:
        shown = ", ".join(str(number) for number in skipped[:5])
        more = "" if len(skipped) <= 5 else f" and {len(skipped) - 5} more"
        warnings.append(
            f"{path}: skipped {len(skipped)} unreadable row(s) at line(s) {shown}{more}."
        )
    return rows


def _optional_column(tokens, columns, field, convert):
    index = columns.get(field)
    if index is None:
        return None
    return convert(tokens[index])


# --- Parameters.txt ----------------------------------------------------------


def read_atom_types(calculation_dir: str, folders: list[str]) -> list[str] | None:
    """
    Return the element symbols from the ``% atomType`` block, or None.

    Looks in ``resultsN/Parameters.txt`` (a copy of ``INPUT.txt``) of each
    folder in order, then in ``INPUT.txt`` of the calculation directory.
    The order of ``atomType`` is the order of the counts in the composition
    column. Atomic numbers are accepted and converted to symbols.
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
        Version tag of this layout. Provisional until real files are checked.
    ``results_folders``
        ``["results1", "results2", ...]`` in numeric order.
    ``atom_types``
        Element symbols from ``atomType``, or None if not found.
    ``n_individuals``, ``n_individuals_without_enthalpy``
        Counts over all folders.
    ``generations``
        One entry per (results folder, generation), from the lowest to the
        highest generation number seen in that folder, including generations
        with no rows ("empty"). ``best_enthalpy_ev_per_atom`` and ``best_id``
        are None when no individual of that generation has an enthalpy.
    ``best_individuals``
        The rows of every ``BESTIndividuals`` file, as written by USPEX.
    ``candidates``
        Individuals with an enthalpy, ranked by enthalpy per atom (lowest
        first). Unique by (results folder, ID): if the same ID appears twice
        in one folder, the later row wins. Equivalent structures with
        different IDs are not merged; that needs a structure comparison and
        is left to a later step.
    ``n_candidates``
        Total number of ranked candidates before ``max_candidates``.
    ``warnings``
        Human-readable notes about missing files, unfinished lines and
        skipped rows.

    Enthalpies are USPEX's own search-grade values (from its last relaxation
    step). Space groups are as reported by USPEX and are often
    under-determined; recompute them from the structure where it matters.
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
    ranked = sorted(
        (row for row in unique.values() if row.enthalpy_ev_per_atom is not None),
        key=lambda row: (
            row.enthalpy_ev_per_atom,
            folders.index(row.results_folder),
            row.id,
        ),
    )

    candidates = []
    for rank, row in enumerate(ranked[: max(0, max_candidates)], 1):
        entry = _row_payload(row, atom_types)
        entry["rank"] = rank
        entry["in_good_structures"] = (row.results_folder, row.id) in good_ids
        candidates.append(entry)

    return {
        "schema": SUMMARY_SCHEMA,
        "calculation_dir": calculation_dir,
        "results_folders": folders,
        "atom_types": atom_types,
        "n_individuals": len(unique),
        "n_individuals_without_enthalpy": sum(
            1 for row in unique.values() if row.enthalpy_ev_per_atom is None
        ),
        "generations": _generation_summary(unique.values(), folders),
        "best_individuals": [_row_payload(row, atom_types) for row in best_rows],
        "candidates": candidates,
        "n_candidates": len(ranked),
        "warnings": warnings,
    }


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
            scored = [row for row in members if row.enthalpy_ev_per_atom is not None]
            best = min(scored, key=lambda row: (row.enthalpy_ev_per_atom, row.id), default=None)
            summary.append(
                {
                    "results_folder": folder,
                    "generation": generation,
                    "n_individuals": len(members),
                    "n_with_enthalpy": len(scored),
                    "best_id": best.id if best else None,
                    "best_enthalpy_ev_per_atom": _round(
                        best.enthalpy_ev_per_atom if best else None
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
        "composition": list(row.composition),
        "formula": _formula(row.composition, atom_types),
        "natoms": natoms,
        "enthalpy_ev": _round(row.enthalpy_ev),
        "enthalpy_ev_per_atom": _round(row.enthalpy_ev_per_atom),
        "volume_a3": _round(row.volume_a3),
        "volume_per_atom_a3": _round(
            row.volume_a3 / natoms if row.volume_a3 is not None else None
        ),
        "space_group": row.space_group,
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
    lines = _read_complete_lines(path, warnings)
    if lines is None:
        return None
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
        ``"gatheredPOSCARS"`` (default, every relaxed individual) or
        ``"goodStructures_POSCARS"``.
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
    folders = find_results_folders(calculation_dir)
    if results_folder is not None:
        if results_folder not in folders:
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

    atom_types = read_atom_types(calculation_dir, find_results_folders(calculation_dir))
    structures = {}
    for found_id, ((folder, block),) in found.items():
        path = os.path.join(calculation_dir, folder, poscars_file)
        structures[found_id] = _block_to_structure(block, atom_types, path)
    return structures


__all__ = [
    "COLUMN_ALIASES",
    "IndividualRow",
    "MISSING_ENTHALPY_TOKENS",
    "POSCAR_ID_PATTERN",
    "USPEXFormatError",
    "USPEXResultsError",
    "find_results_folders",
    "load_structures",
    "read_atom_types",
    "read_individuals_file",
    "summarise_search",
]
