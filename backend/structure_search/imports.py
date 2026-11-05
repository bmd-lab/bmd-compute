"""
Import USPEX results run by hand and hand candidates to the workbench (#63).

Flow
----
1. :func:`import_uspex_archive` stores an uploaded ``.zip``/``.tar.gz`` under
   the app's upload folder, unpacks its result files safely
   (:mod:`backend.structure_search.archive`), reads them with the results
   parser (:mod:`backend.structure_search.results`) and saves the candidate
   table it shows as ``candidates.json``.
2. :func:`load_candidate_page` reads that table back for the browser: ranked
   candidates, an energy-per-atom vs generation plot for the shared Plotly
   renderer, and the parser's warnings.
3. :func:`hand_off_candidates` turns selected candidates into POSCAR text
   plus a signed *structure origin token* (:mod:`backend.structure_origin`).
   The browser then sends each one through the ordinary Build route, exactly
   as if the POSCAR had been pasted; the token adds the USPEX record to the
   new submission's provenance. Nothing here builds or submits a calculation.

USPEX energies are search-grade values from USPEX's own relaxation ladder.
They are shown to help choose candidates; the final ranking comes from the
BMD calculation the candidate is sent to.

Storage and lifetime
--------------------
Imports live in ``uploads/structure_search/<import id>/`` (``uploads/`` is the
app's existing, git-ignored upload folder). The uploaded archive itself is
deleted once unpacked; its SHA-256 is kept in ``import.json``. Each new
import first deletes imports older than :data:`IMPORT_RETENTION_SECONDS` and
keeps at most :data:`MAX_STORED_IMPORTS`. Deleting an import does not affect
calculations built from it: their origin travels in the signed token and the
submission record, not in the upload folder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import time
from pathlib import Path

from backend.structure_search import archive as archive_module
from backend.structure_search import results as uspex_results
from backend.structure_search.archive import ArchiveError


IMPORT_SCHEMA = "bmd-compute.uspex-import.1"
ORIGIN_SCHEMA = "bmd-compute.structure-origin.uspex-import.1"
ORIGIN_SOURCE = "uspex_import"

DEFAULT_UPLOADS_ROOT = Path(__file__).resolve().parents[2] / "uploads"
IMPORTS_SUBDIRECTORY = "structure_search"

# Lifetime of stored imports (see module docstring). Not scientific settings.
IMPORT_RETENTION_SECONDS = 7 * 24 * 3600
MAX_STORED_IMPORTS = 20
# Candidates listed on the page (the parser's ranking, best first).
MAX_CANDIDATES_SHOWN = 50
# Most candidates handed off in one request.
MAX_HANDOFF_CANDIDATES = 10

IMPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SELECTION_PATTERN = re.compile(r"^(results\d+):(\d+)$")
# POSCARS files used for hand-off, in order of preference. gatheredPOSCARS
# holds every relaxed individual; good_structures_POSCARS only the best ones.
HANDOFF_POSCARS_FILES = ("gatheredPOSCARS", "good_structures_POSCARS")
_EXTERNAL_PRESSURE_PATTERN = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*:\s*ExternalPressure\b",
    re.MULTILINE,
)

ENERGY_NOTE = (
    "USPEX 'energy' column of Individuals, eV per cell: the enthalpy at the "
    "search pressure from USPEX's last relaxation step (search-grade)."
)
PLOT_ELEMENT_ID = "structure-search-energy-plot"


class StructureSearchImportError(ValueError):
    """A user-facing problem with an import (message plus suggestion)."""

    def __init__(self, message: str, *, suggestion: str | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion


# --- Storage -----------------------------------------------------------------


def imports_root(uploads_root: str | os.PathLike | None = None) -> Path:
    return Path(uploads_root or DEFAULT_UPLOADS_ROOT) / IMPORTS_SUBDIRECTORY


def _import_dir(import_id: str, uploads_root=None) -> Path:
    if not isinstance(import_id, str) or not IMPORT_ID_PATTERN.fullmatch(import_id):
        raise StructureSearchImportError(
            "The structure-search import reference is not valid.",
            suggestion="Upload the USPEX results archive again.",
        )
    return imports_root(uploads_root) / import_id


def prune_imports(uploads_root=None, *, now: float | None = None) -> list[str]:
    """Delete expired imports and keep at most MAX_STORED_IMPORTS - 1."""

    root = imports_root(uploads_root)
    if not root.is_dir():
        return []
    now = time.time() if now is None else now
    entries = []
    for entry in root.iterdir():
        if not IMPORT_ID_PATTERN.fullmatch(entry.name) or entry.is_symlink() or not entry.is_dir():
            continue
        entries.append((entry.stat().st_mtime, entry))
    entries.sort(reverse=True)
    removed = []
    for index, (mtime, entry) in enumerate(entries):
        if now - mtime > IMPORT_RETENTION_SECONDS or index >= MAX_STORED_IMPORTS - 1:
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
    return removed


def _safe_display_filename(filename: str | None) -> str:
    name = os.path.basename(str(filename or "").replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._ +()-]+", "_", name).strip()
    return name[:120] or "upload"


def import_uspex_archive(
    fileobj,
    filename: str | None,
    *,
    uploads_root=None,
    now: float | None = None,
) -> str:
    """
    Store, check and unpack an uploaded USPEX results archive.

    ``fileobj`` is read in chunks (at most ``MAX_ARCHIVE_BYTES`` + 1 bytes).
    Returns the new import ID. Raises :class:`StructureSearchImportError`
    with a user-facing message if the upload is missing, too large, unsafe
    or holds no usable results; nothing is kept in that case.
    """

    if fileobj is None:
        raise StructureSearchImportError(
            "No file was uploaded.",
            suggestion=archive_module.ARCHIVE_FORMAT_SUGGESTION,
        )
    prune_imports(uploads_root, now=now)
    import_id = secrets.token_hex(16)
    directory = _import_dir(import_id, uploads_root)
    directory.mkdir(mode=0o700, parents=True)
    try:
        upload_path = directory / "upload.partial"
        digest = hashlib.sha256()
        size = 0
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(upload_path, flags, 0o600), "wb") as handle:
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > archive_module.MAX_ARCHIVE_BYTES:
                    raise StructureSearchImportError(
                        "The archive is larger than the "
                        f"{archive_module.MAX_ARCHIVE_BYTES // (1024 * 1024)} MB upload limit.",
                        suggestion="Archive only the results folders, not the whole calculation.",
                    )
                digest.update(chunk)
                handle.write(chunk)
        if size == 0:
            raise StructureSearchImportError(
                "No file was uploaded, or the uploaded file is empty.",
                suggestion=archive_module.ARCHIVE_FORMAT_SUGGESTION,
            )
        with open(upload_path, "rb") as handle:
            archive_format = archive_module.detect_archive_format(handle.read(512))
        extracted = archive_module.extract_uspex_archive(str(upload_path), str(directory / "search"))
        upload_path.unlink()

        record = {
            "schema": IMPORT_SCHEMA,
            "import_id": import_id,
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now)
            ),
            "archive": {
                "filename": _safe_display_filename(filename),
                "format": archive_format,
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            },
            "calculation_path_in_archive": extracted.calculation_path_in_archive,
            "calculation_dir": os.path.relpath(extracted.calculation_dir, directory),
            "results_folders": list(extracted.results_folders),
            "files_kept": extracted.files_kept,
            "members_skipped": extracted.members_skipped,
            "notes": list(extracted.notes),
        }
        page = _build_candidate_table(record, extracted.calculation_dir, directory)
        _write_json(directory / "import.json", record)
        _write_json(directory / "candidates.json", page)
        return import_id
    except ArchiveError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise StructureSearchImportError(exc.message, suggestion=exc.suggestion) from exc
    except uspex_results.USPEXResultsError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise StructureSearchImportError(
            "The USPEX results in the archive could not be read: "
            + _relative_message(str(exc), directory),
            suggestion=(
                "This import reads USPEX 25.0 output. If the files come from another "
                "USPEX version, please report the message above."
            ),
        ) from exc
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _write_json(path: Path, value) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def _read_import(import_id: str, uploads_root=None) -> tuple[Path, dict, dict]:
    directory = _import_dir(import_id, uploads_root)
    try:
        record = json.loads((directory / "import.json").read_text(encoding="utf-8"))
        page = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StructureSearchImportError(
            "This structure-search import is no longer available. Imports are kept "
            f"for {IMPORT_RETENTION_SECONDS // 86400} days.",
            suggestion="Upload the USPEX results archive again.",
        ) from exc
    return directory, record, page


def _relative_message(message: str, directory: Path) -> str:
    """Replace server paths in parser messages with paths inside the archive."""

    search_dir = str(directory / "search")
    return message.replace(search_dir + os.sep, "").replace(search_dir, "(archive)")


# --- Candidate table ---------------------------------------------------------


def read_external_pressure(calculation_dir: str, folders) -> float | None:
    """``ExternalPressure`` (GPa) from Parameters.txt or INPUT.txt, or None."""

    paths = [os.path.join(calculation_dir, folder, "Parameters.txt") for folder in folders]
    paths.append(os.path.join(calculation_dir, "INPUT.txt"))
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read(1024 * 1024)
        except OSError:
            continue
        match = _EXTERNAL_PRESSURE_PATTERN.search(text)
        if match:
            value = float(match.group(1))
            return value if math.isfinite(value) else None
    return None


def _handoff_poscars_files(calculation_dir: str, folders) -> dict[str, tuple[str | None, set[int]]]:
    """For each folder: the POSCARS file used for hand-off and its IDs."""

    chosen = {}
    for folder in folders:
        chosen[folder] = (None, set())
        for name in HANDOFF_POSCARS_FILES:
            ids = uspex_results.available_structure_ids(
                calculation_dir, folder, poscars_file=name
            )
            if ids is not None:
                chosen[folder] = (name, ids)
                break
    return chosen


def _build_candidate_table(record: dict, calculation_dir: str, import_directory: Path) -> dict:
    """Summarise the search and add BMD's symmetry and hand-off availability."""

    summary = uspex_results.summarise_search(
        calculation_dir, max_candidates=MAX_CANDIDATES_SHOWN
    )
    warnings = [_relative_message(text, import_directory) for text in summary["warnings"]]
    warnings.extend(record.get("notes") or [])

    folders = summary["results_folders"]
    poscars = _handoff_poscars_files(calculation_dir, folders)
    candidates = []
    by_folder: dict[str, list[int]] = {}
    for candidate in summary["candidates"]:
        poscars_file, ids = poscars.get(candidate["results_folder"], (None, set()))
        available = poscars_file is not None and candidate["id"] in ids
        row = dict(candidate)
        row["selection"] = f"{candidate['results_folder']}:{candidate['id']}"
        row["poscars_file"] = poscars_file if available else None
        row["handoff_available"] = available
        row["symmetry"] = None
        candidates.append(row)
        if available:
            by_folder.setdefault(candidate["results_folder"], []).append(candidate["id"])

    structures = {}
    for folder, ids in by_folder.items():
        loaded = uspex_results.load_structures(
            calculation_dir, ids, poscars_file=poscars[folder][0], results_folder=folder
        )
        for individual_id, structure in loaded.items():
            structures[(folder, individual_id)] = structure
    for row in candidates:
        structure = structures.get((row["results_folder"], row["id"]))
        if structure is not None:
            try:
                row["symmetry"] = uspex_results.structure_symmetry(structure)
            except Exception as exc:  # noqa: BLE001 - spglib failure is shown, not fatal
                warnings.append(
                    f"Symmetry of individual {row['id']} ({row['results_folder']}) "
                    f"could not be computed: {exc}"
                )

    unavailable = [row["selection"] for row in candidates if not row["handoff_available"]]
    if unavailable:
        warnings.append(
            f"{len(unavailable)} listed candidate(s) have no POSCAR block in "
            "gatheredPOSCARS or good_structures_POSCARS and cannot be handed off: "
            + ", ".join(unavailable[:10])
            + ("" if len(unavailable) <= 10 else ", ...")
            + "."
        )

    pressure = read_external_pressure(calculation_dir, folders)
    pressure_note = None
    if pressure is None:
        pressure_note = (
            "The search pressure (ExternalPressure) was not found in Parameters.txt "
            "or INPUT.txt. BMD calculations relax at zero pressure."
        )
    elif pressure != 0:
        pressure_note = (
            f"This search ran at {pressure:g} GPa. BMD's Relaxed Energy workflow has "
            "no pressure setting and relaxes at zero pressure, so USPEX enthalpies "
            "and BMD energies are not directly comparable."
        )

    return {
        "schema": IMPORT_SCHEMA,
        "results_folders": folders,
        "atom_types": summary["atom_types"],
        "n_individuals": summary["n_individuals"],
        "n_individuals_without_energy": summary["n_individuals_without_energy"],
        "n_candidates": summary["n_candidates"],
        "generations": summary["generations"],
        "candidates": candidates,
        "external_pressure_gpa": pressure,
        "pressure_note": pressure_note,
        "symmetry_symprec": uspex_results.SYMMETRY_SYMPREC,
        "warnings": warnings,
    }


def energy_generation_plot(page: dict) -> dict | None:
    """Visualization payload for the shared Plotly line-plot renderer."""

    traces = []
    folders = page.get("results_folders") or []
    for folder in folders:
        points = [
            entry
            for entry in page.get("generations") or []
            if entry["results_folder"] == folder and entry["best_energy_ev_per_atom"] is not None
        ]
        if not points:
            continue
        suffix = f" ({folder})" if len(folders) > 1 else ""
        traces.append(
            {
                "name": f"Best per generation{suffix}",
                "x": [entry["generation"] for entry in points],
                "y": [entry["best_energy_ev_per_atom"] for entry in points],
                "mode": "lines+markers",
                "text": [f"ID {entry['best_id']}" for entry in points],
            }
        )
    candidates = [
        row for row in page.get("candidates") or [] if row.get("energy_ev_per_atom") is not None
    ]
    if candidates:
        traces.append(
            {
                "name": "Listed candidates",
                "x": [row["generation"] for row in candidates],
                "y": [row["energy_ev_per_atom"] for row in candidates],
                "mode": "markers",
                "text": [
                    f"ID {row['id']} ({row['results_folder']}), {row.get('origin') or ''}".rstrip(", ")
                    for row in candidates
                ],
            }
        )
    if not traces:
        return None
    return {
        "id": "structure_search_energy",
        "element_id": PLOT_ELEMENT_ID,
        "kind": "line_plot",
        "title": "USPEX energy per atom by generation",
        "subtitle": "Search-grade USPEX energies; lower is better",
        "download_filename": "uspex_energy_by_generation.png",
        "plot": {
            "x": [],
            "traces": traces,
            "xaxis_title": "Generation",
            "yaxis_title": "USPEX energy (eV/atom)",
            "x_hover_unit": "",
            "y_hover_unit": " eV/atom",
            "x_hover_format": "d",
            "y_hover_format": ".4f",
            "energy_reference": "absolute",
            "source": "Individuals",
        },
    }


def load_candidate_page(import_id: str, *, uploads_root=None) -> dict:
    """Everything the browser shows for one import."""

    _directory, record, page = _read_import(import_id, uploads_root)
    visualization = energy_generation_plot(page)
    return {
        "import_id": record["import_id"],
        "archive": record["archive"],
        "created_at": record["created_at"],
        "calculation_path_in_archive": record["calculation_path_in_archive"],
        **page,
        "max_candidates_shown": MAX_CANDIDATES_SHOWN,
        "max_handoff_candidates": MAX_HANDOFF_CANDIDATES,
        "visualizations": [visualization] if visualization else [],
    }


# --- Hand-off ----------------------------------------------------------------


def origin_record(record: dict, page: dict, row: dict) -> dict:
    """The JSON-safe USPEX origin stored in the new submission's provenance."""

    return {
        "schema": ORIGIN_SCHEMA,
        "source": ORIGIN_SOURCE,
        "description": "Candidate of a USPEX structure search imported from an uploaded archive.",
        "archive_sha256": record["archive"]["sha256"],
        "archive_filename": record["archive"]["filename"],
        "archive_format": record["archive"]["format"],
        "import_id": record["import_id"],
        "imported_at": record["created_at"],
        "calculation_path_in_archive": record["calculation_path_in_archive"],
        "results_folder": row["results_folder"],
        "poscars_file": row["poscars_file"],
        "individual_id": row["id"],
        "generation": row["generation"],
        "origin": row.get("origin"),
        "parents": row.get("parents"),
        "lineage_ids": row.get("lineage_ids"),
        "formula": row.get("formula"),
        "natoms": row.get("natoms"),
        "uspex_energy_ev": row.get("energy_ev"),
        "uspex_energy_ev_per_atom": row.get("energy_ev_per_atom"),
        "uspex_energy_note": ENERGY_NOTE,
        "uspex_space_group": row.get("space_group_uspex"),
        "uspex_rank": row.get("rank"),
        "external_pressure_gpa": page.get("external_pressure_gpa"),
        "parser_schema": uspex_results.SUMMARY_SCHEMA,
    }


def hand_off_candidates(
    import_id: str,
    selections,
    *,
    uploads_root=None,
) -> list[dict]:
    """
    Prepare selected candidates for the workbench's normal Build route.

    ``selections`` are ``"<results folder>:<individual ID>"`` values from the
    candidate table. For each one this returns the candidate row, its POSCAR
    text (written by pymatgen from the USPEX POSCAR block and checked with
    ``backend.parser.parse_structure``), the origin record and a signed
    origin token. It does not build or submit anything.
    """

    from pymatgen.io.vasp import Poscar

    from backend.parser import StructureValidationError, parse_structure
    from backend.structure_origin import create_structure_origin_token

    directory, record, page = _read_import(import_id, uploads_root)
    chosen = []
    for value in selections or []:
        value = str(value).strip()
        if value and value not in chosen:
            chosen.append(value)
    if not chosen:
        raise StructureSearchImportError(
            "No candidate was selected.",
            suggestion="Tick one or more candidates in the table, then hand them off.",
        )
    if len(chosen) > MAX_HANDOFF_CANDIDATES:
        raise StructureSearchImportError(
            f"At most {MAX_HANDOFF_CANDIDATES} candidates can be handed off at once.",
            suggestion="Select fewer candidates.",
        )
    rows = {row["selection"]: row for row in page["candidates"]}
    calculation_dir = str(directory / record["calculation_dir"])

    handed_off = []
    for value in chosen:
        if not SELECTION_PATTERN.fullmatch(value) or value not in rows:
            raise StructureSearchImportError(
                f"Candidate {value!r} is not in this import's candidate table.",
                suggestion="Select candidates from the table shown for this import.",
            )
        row = rows[value]
        if not row["handoff_available"]:
            raise StructureSearchImportError(
                f"Candidate {row['id']} ({row['results_folder']}) has no POSCAR block "
                "in the uploaded files, so it cannot be handed off.",
                suggestion="Include gatheredPOSCARS in the archive.",
            )
        structure = uspex_results.load_structures(
            calculation_dir,
            [row["id"]],
            poscars_file=row["poscars_file"],
            results_folder=row["results_folder"],
        )[row["id"]]
        comment = f"USPEX individual {row['id']} ({row['results_folder']})"
        structure_text = Poscar(structure, comment=comment).get_str()
        try:
            parse_structure(structure_text, "poscar")
        except StructureValidationError as exc:
            raise StructureSearchImportError(
                f"Candidate {row['id']} ({row['results_folder']}) could not be read as "
                f"a structure: {exc.message}",
                suggestion=exc.suggestion,
            ) from exc
        origin = origin_record(record, page, row)
        handed_off.append(
            {
                "candidate": row,
                "structure_text": structure_text,
                "fmt": "poscar",
                "origin": origin,
                "origin_token": create_structure_origin_token(origin, structure_text),
            }
        )
    return handed_off


__all__ = [
    "HANDOFF_POSCARS_FILES",
    "IMPORT_RETENTION_SECONDS",
    "MAX_CANDIDATES_SHOWN",
    "MAX_HANDOFF_CANDIDATES",
    "MAX_STORED_IMPORTS",
    "ORIGIN_SCHEMA",
    "StructureSearchImportError",
    "energy_generation_plot",
    "hand_off_candidates",
    "import_uspex_archive",
    "imports_root",
    "load_candidate_page",
    "origin_record",
    "prune_imports",
    "read_external_pressure",
]
