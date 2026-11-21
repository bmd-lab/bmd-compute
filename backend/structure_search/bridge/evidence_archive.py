"""Append-only, attempt-local archive of each relaxation step's scientific evidence.

USPEX keeps only the last generation's CalcFolds; earlier ones are removed
with the per-step validity records and relaxed structures in them. The
launcher therefore commits, for every step it ran, one record outside the
CalcFolds, under ``<search>/bmd_evidence/``::

    bmd_evidence/
        steps/<request id>/   one committed record per step (never rewritten)
            record.json       the step's identity, launch, verdict and file hashes
            POSCAR            the input structure, as it was when the step started
            CONTCAR           the output structure, as VASP left it (if any)
        index.jsonl           one line per committed record: id and record SHA-256
        conflicts/<id>-<sha16>/  a different record for an id already committed
        interrupted/<name>/   a write a killed launcher left unfinished (moved
                              here on the next start; never completed or removed)
        .staging/             records being written

Order and atomicity: the launcher writes the record after VASP has exited
and the validity check has run, and *before* it writes the step's done file.
USPEX's bridge script waits for that done file, so USPEX cannot read the
step's result, start its next step or remove its CalcFold before the record
is on disk. Each record is written into ``.staging/``, every file flushed to
disk, ``record.json`` last, and the directory then renamed to
``steps/<id>`` in one step: a reader sees a whole record or none. The index
line follows; a record without its index line (the launcher died in
between) is indexed on the next start, after its file hashes are checked.

Never overwritten: a record for an id that already has one is compared by
SHA-256. The same bytes are a duplicate (nothing written; reported). Other
bytes are kept beside it under ``conflicts/`` (named by their own hash, so a
repeat is again a duplicate); the committed record is never replaced.

Original records, not reconstructions: ``validity`` is the verdict dictionary
the checker returned (classification, reasons, convergence evidence),
unchanged; ``validity_record_line_sha256`` is the SHA-256 of the exact line
appended to ``<CalcFold>/bmd_step_validity.jsonl``, so a surviving CalcFold's
record can be matched byte for byte. Structures are copied byte for byte.

Generation: USPEX passes the bridge only the step number and its working
directory, so the launcher does not know an individual's generation. Records
carry ``generation: null`` and say so; the independent analyser binds a
record to a generation through USPEX's own listing, never by guessing here.

Small and text-only: record.json, POSCAR and CONTCAR, each structure at most
:data:`MAX_STRUCTURE_BYTES` (a larger one is hashed, not copied). Never
POTCAR, WAVECAR, CHGCAR, OUTCAR or vasprun.xml.

Standard library only (the launcher runs in the job's Python, outside the
scientific stack).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARCHIVE_DIR = "bmd_evidence"
STEPS_DIR = "steps"
STAGING_DIR = ".staging"
CONFLICTS_DIR = "conflicts"
INTERRUPTED_DIR = "interrupted"
INDEX_FILE = "index.jsonl"
RECORD_FILE = "record.json"
INPUT_STRUCTURE = "POSCAR"
OUTPUT_STRUCTURE = "CONTCAR"
ARCHIVED_FILES = (RECORD_FILE, INPUT_STRUCTURE, OUTPUT_STRUCTURE)
SCHEMA = "bmd_compute.structure_search_step_evidence"
SCHEMA_VERSION = 1
MAX_STRUCTURE_BYTES = 1024 * 1024
GENERATION_NOTE = ("USPEX passes the bridge only the step and the CalcFold; the generation is bound "
                   "afterwards from USPEX's listing, not recorded here")
_CALCFOLD_NAME = re.compile(r"^Calcfold_(\d+)_(\d+)$")


class EvidenceError(Exception):
    """The archive could not be written as specified (nothing partial is ever left in ``steps/``)."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_root(search_dir: Path) -> Path:
    return Path(search_dir) / ARCHIVE_DIR


def read_structure(path: Path, *, name: str) -> tuple[dict[str, Any], bytes | None]:
    """Facts about one structure file and its bytes if it is small enough to keep."""

    facts: dict[str, Any] = {"file": name, "source": Path(path).name, "present": False}
    try:
        size = Path(path).stat().st_size
        if not Path(path).is_file():
            facts["reason"] = "not a regular file"
            return facts, None
    except FileNotFoundError:
        facts["reason"] = "missing"
        return facts, None
    except OSError as exc:
        facts["reason"] = f"unreadable: {type(exc).__name__}"
        return facts, None
    facts.update(present=True, bytes=size)
    try:
        if size > MAX_STRUCTURE_BYTES:
            facts.update(sha256=sha256_file(path), archived=False,
                         reason=f"larger than {MAX_STRUCTURE_BYTES} bytes: hashed, not copied")
            return facts, None
        data = Path(path).read_bytes()
    except OSError as exc:
        facts.update(present=False, reason=f"unreadable: {type(exc).__name__}")
        return facts, None
    facts.update(bytes=len(data), sha256=sha256_bytes(data), archived=True)
    return facts, data


def calcfold_name_fields(calcfold: Path) -> dict[str, int] | None:
    """The two numbers of a USPEX ``Calcfold_<a>_<b>`` name, as written; None for any other name."""

    match = _CALCFOLD_NAME.match(Path(calcfold).name)
    return {"first": int(match.group(1)), "second": int(match.group(2))} if match else None


def build_record(*, request_id: str, step: int, calcfold: Path, created: int | None, slot: int | None,
                 argv: list[str] | None, started: float | None, finished: float | None, returncode: int | None,
                 status: int, reason: str, validity: Mapping[str, Any] | None,
                 validity_record_line: str | None, step_log: Mapping[str, Any] | None,
                 input_structure: Mapping[str, Any], output_structure: Mapping[str, Any],
                 popen: float | None = None, popen_monotonic: float | None = None,
                 returned: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The record of one step. Pure: everything it holds is passed in.

    ``launch.popen`` and ``launch.popen_monotonic`` are the wall-clock and monotonic times just before the
    step's process was started, and ``launch.returned`` the launcher's ``srun_returned`` observation (None if
    the process never exited). These keys are additive within schema version 1: records written before them
    have none of them, and readers must accept both.
    """

    classification = None if validity is None else validity.get("classification")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "id": request_id,
        "step": step,
        "calcfold": str(calcfold),
        "calcfold_name": Path(calcfold).name,
        "calcfold_name_fields": calcfold_name_fields(calcfold),
        "generation": None,
        "generation_note": GENERATION_NOTE,
        "request_created": created,
        "launch": {"slot": slot, "argv": list(argv) if argv is not None else None,
                   "started": None if started is None else round(started, 3),
                   "finished": None if finished is None else round(finished, 3),
                   "returncode": returncode, "status": status, "reason": reason,
                   "popen": None if popen is None else round(popen, 6),
                   "popen_monotonic": None if popen_monotonic is None else round(popen_monotonic, 6),
                   "returned": None if returned is None else dict(returned)},
        "step_log": dict(step_log) if step_log is not None else None,
        "validity": None if validity is None else json.loads(json.dumps(validity, default=str)),
        "validity_record_line_sha256": (None if validity_record_line is None
                                        else sha256_bytes((validity_record_line + "\n").encode("utf-8"))),
        "input_structure": dict(input_structure),
        "output_structure": dict(output_structure),
        "final_relaxed_structure": bool(step == 5 and classification == "valid"
                                        and output_structure.get("archived")),
    }


def record_bytes(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _write_synced(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _append_synced(path: Path, line: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, (line + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_directory(root: Path, staging_name: str, files: Mapping[str, bytes], record_data: bytes) -> Path:
    staging = root / STAGING_DIR / staging_name
    staging.mkdir(parents=True)
    for name, data in files.items():
        _write_synced(staging / name, data)
    _write_synced(staging / RECORD_FILE, record_data)  # last: its presence means the files before it are whole
    _sync_directory(staging)
    return staging


def commit(root: Path, record: Mapping[str, Any], files: Mapping[str, bytes]) -> dict[str, Any]:
    """Commit one step record with its structure files; never replaces what is there.

    Returns ``{"outcome": "committed" | "duplicate" | "conflict", "record_sha256", "path"}``.
    Raises :class:`EvidenceError` (or ``OSError``) if nothing could be committed; then nothing
    under ``steps/`` was created or changed.
    """

    root = Path(root)
    request_id = str(record["id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", request_id):
        raise EvidenceError(f"invalid request id {request_id!r}")
    unknown = set(files) - {INPUT_STRUCTURE, OUTPUT_STRUCTURE}
    if unknown:
        raise EvidenceError(f"files outside the archive's allowlist: {sorted(unknown)}")
    for name, data in files.items():
        expected = (record.get("input_structure") if name == INPUT_STRUCTURE else record.get("output_structure")) or {}
        if expected.get("sha256") != sha256_bytes(data) or not expected.get("archived"):
            raise EvidenceError(f"{name} does not match its hash in the record")
    data = record_bytes(record)
    digest = sha256_bytes(data)
    for sub in (STEPS_DIR, STAGING_DIR, CONFLICTS_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)
    final = root / STEPS_DIR / request_id

    def against_existing(existing: Path) -> dict[str, Any]:
        try:
            existing_digest = sha256_file(existing / RECORD_FILE)
        except OSError:
            existing_digest = None
        if existing_digest == digest:
            return {"outcome": "duplicate", "record_sha256": digest, "path": str(existing)}
        conflict = root / CONFLICTS_DIR / f"{request_id}-{digest[:16]}"
        if (conflict / RECORD_FILE).is_file() and sha256_file(conflict / RECORD_FILE) == digest:
            return {"outcome": "duplicate", "record_sha256": digest, "path": str(conflict)}
        staging = _write_directory(root, f"{request_id}-conflict-{os.getpid()}-{time.time_ns()}", files, data)
        os.rename(staging, conflict)
        _sync_directory(conflict.parent)
        return {"outcome": "conflict", "record_sha256": digest, "path": str(conflict),
                "committed_record_sha256": existing_digest}

    if final.exists():
        return against_existing(final)
    staging = _write_directory(root, f"{request_id}-{os.getpid()}-{time.time_ns()}", files, data)
    try:
        os.rename(staging, final)  # atomic; refuses (ENOTEMPTY/EEXIST) if a record appeared meanwhile
    except OSError:
        if final.exists():
            result = against_existing(final)
            _set_aside(root, staging)  # our own complete copy, kept rather than deleted
            return result
        raise
    _sync_directory(final.parent)
    _append_synced(root / INDEX_FILE, json.dumps({"id": request_id, "record_sha256": digest,
                                                  "event": "committed", "time": round(time.time(), 3)},
                                                 sort_keys=True))
    return {"outcome": "committed", "record_sha256": digest, "path": str(final)}


def _set_aside(root: Path, path: Path) -> None:
    (root / INTERRUPTED_DIR).mkdir(parents=True, exist_ok=True)
    os.rename(path, root / INTERRUPTED_DIR / path.name)


def read_index(root: Path) -> tuple[list[dict[str, Any]], int]:
    """The index lines and the number of unreadable (torn) lines."""

    lines, torn = [], 0
    try:
        text = (Path(root) / INDEX_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], 0
    for line in text.splitlines():
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict) or "id" not in entry or "record_sha256" not in entry:
                raise ValueError
            lines.append(entry)
        except ValueError:
            torn += 1
    return lines, torn


def verify_record_directory(path: Path) -> dict[str, Any]:
    """Check a committed record against its own file hashes; never changes anything."""

    problems = []
    try:
        data = (Path(path) / RECORD_FILE).read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        return {"ok": False, "problems": [f"record.json unreadable: {type(exc).__name__}"], "record_sha256": None}
    for name, key in ((INPUT_STRUCTURE, "input_structure"), (OUTPUT_STRUCTURE, "output_structure")):
        facts = record.get(key) or {}
        present = (Path(path) / name).is_file()
        if facts.get("archived") and not present:
            problems.append(f"{name} is recorded as archived but absent")
        elif present and (not facts.get("archived") or sha256_file(Path(path) / name) != facts.get("sha256")):
            problems.append(f"{name} does not match its recorded hash")
    extra = sorted(p.name for p in Path(path).iterdir() if p.name not in ARCHIVED_FILES)
    if extra:
        problems.append(f"unexpected files {extra}")
    if Path(path).parent.name == STEPS_DIR and record.get("id") != Path(path).name:
        problems.append("the record's id is not its directory name")
    return {"ok": not problems, "problems": problems, "record_sha256": sha256_bytes(data), "record": record}


def recover(root: Path) -> dict[str, Any]:
    """On launcher start: make interrupted writes visible, index committed records the index missed.

    Unfinished staging directories are moved, unchanged, to ``interrupted/`` (never completed, never
    deleted). A committed record without an index line is indexed only if its files match their
    hashes; otherwise it is reported and left alone. Nothing is invented: no missing result is
    written.
    """

    root = Path(root)
    report: dict[str, Any] = {"interrupted": [], "indexed": [], "unverifiable": [], "torn_index_lines": 0}
    if not root.exists():
        return report
    staging = root / STAGING_DIR
    if staging.is_dir():
        for entry in sorted(staging.iterdir()):
            _set_aside(root, entry)
            report["interrupted"].append(entry.name)
    index, report["torn_index_lines"] = read_index(root)
    indexed = {(entry["id"], entry["record_sha256"]) for entry in index}
    steps = root / STEPS_DIR
    if steps.is_dir():
        for entry in sorted(steps.iterdir()):
            check = verify_record_directory(entry)
            if not check["ok"]:
                report["unverifiable"].append({"id": entry.name, "problems": check["problems"]})
                continue
            if (entry.name, check["record_sha256"]) not in indexed:
                _append_synced(root / INDEX_FILE, json.dumps(
                    {"id": entry.name, "record_sha256": check["record_sha256"], "event": "indexed_on_recovery",
                     "time": round(time.time(), 3)}, sort_keys=True))
                report["indexed"].append(entry.name)
    return report


__all__ = [
    "ARCHIVE_DIR", "ARCHIVED_FILES", "CONFLICTS_DIR", "EvidenceError", "GENERATION_NOTE", "INDEX_FILE",
    "INPUT_STRUCTURE", "INTERRUPTED_DIR", "MAX_STRUCTURE_BYTES", "OUTPUT_STRUCTURE", "RECORD_FILE", "SCHEMA",
    "SCHEMA_VERSION", "STAGING_DIR", "STEPS_DIR", "archive_root", "build_record", "calcfold_name_fields",
    "commit", "read_index", "read_structure", "record_bytes", "recover", "sha256_bytes", "sha256_file",
    "verify_record_directory",
]
