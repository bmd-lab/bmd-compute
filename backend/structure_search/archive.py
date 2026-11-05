"""
Safely unpack an uploaded archive of USPEX results (issue #63, part of #58).

A user who ran USPEX by hand uploads a ``.zip`` or ``.tar.gz`` of either the
USPEX calculation folder (the one that holds ``results1/``) or of one or more
``resultsN/`` folders. :func:`extract_uspex_archive` checks every member and
writes only the few result files that :mod:`backend.structure_search.results`
reads. Nothing in the archive is ever executed.

Safety rules
------------
The whole archive is rejected (:class:`ArchiveError`) if any member

* has an absolute path, a ``..`` component, a backslash, a Windows drive
  letter or a NUL character in its name;
* is a symbolic link, a hard link, a device file, a FIFO or any other type
  that is not a plain file or a directory;
* is encrypted (zip).

It is also rejected if it exceeds any of the limits below. The limits are
safety limits for the web server, not scientific settings; change them here
if real searches need more.

Only these members are written, everything else is skipped without being
decompressed to disk:

* :data:`KEPT_RESULTS_FILES` inside a folder named ``resultsN``;
* ``INPUT.txt`` anywhere (the parser reads element names from it if
  ``Parameters.txt`` has none).

Files are written with mode ``0600`` (never executable), with ``O_EXCL`` and
``O_NOFOLLOW``, and each destination is checked to lie inside the target
directory. Sizes are counted from the bytes actually decompressed, not from
the sizes the archive claims, so a "zip bomb" stops at the limit.

This module uses only the standard library.
"""

from __future__ import annotations

import gzip
import os
import re
import shutil
import stat
import tarfile
import zipfile
import zlib
from dataclasses import dataclass, field


# --- Limits (safety limits for the web server, not scientific settings) -----

# Largest uploaded archive accepted, in bytes (compressed size).
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
# Largest number of members (files and folders) an archive may list.
MAX_ARCHIVE_MEMBERS = 20_000
# Largest total size of the files kept on disk after extraction.
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
# Largest single kept file. Same as the parser's own limit
# (backend.structure_search.results.MAX_FILE_BYTES).
MAX_MEMBER_BYTES = 256 * 1024 * 1024
# Largest amount of data decompressed from a .tar.gz while reading it,
# skipped members included. A tar stream must be decompressed to be read, so
# this bounds the time spent on an archive full of large unneeded files.
MAX_DECOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024

# --- What is kept ------------------------------------------------------------

# Names as written by USPEX 25.0 (see backend/structure_search/results.py).
KEPT_RESULTS_FILES = frozenset(
    {
        "Individuals",
        "BESTIndividuals",
        "gatheredPOSCARS",
        "good_structures_POSCARS",
        "BESTgatheredPOSCARS",
        "best_structure_POSCAR",
        "Parameters.txt",
    }
)
KEPT_ANYWHERE_FILES = frozenset({"INPUT.txt"})
# A results folder needs Individuals and at least one of these to be usable.
REQUIRED_TABLE_FILE = "Individuals"
REQUIRED_POSCARS_FILES = ("gatheredPOSCARS", "good_structures_POSCARS")

RESULTS_FOLDER_PATTERN = re.compile(r"^results(\d+)$")

ARCHIVE_FORMAT_ZIP = "zip"
ARCHIVE_FORMAT_TAR_GZ = "tar.gz"

_COPY_CHUNK_BYTES = 1024 * 1024
_DRIVE_LETTER_PATTERN = re.compile(r"^[A-Za-z]:")

ARCHIVE_FORMAT_SUGGESTION = (
    "Upload a .zip or .tar.gz of the USPEX calculation folder, or of its "
    "results1/ (results2/, ...) folders."
)
MINIMUM_CONTENT_SUGGESTION = (
    "Each results folder needs Individuals and gatheredPOSCARS (or "
    "good_structures_POSCARS). Archive the results1/ folder itself, or the "
    "folder that contains it, not only the files inside it."
)


class ArchiveError(ValueError):
    """The upload is not a usable, safe archive of USPEX results."""

    def __init__(self, message: str, *, suggestion: str | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion


@dataclass(frozen=True)
class ExtractedSearch:
    """Where the results were unpacked and what was found."""

    # Absolute path of the folder that holds the resultsN/ folders.
    calculation_dir: str
    # The same folder inside the archive, POSIX style; "" for the archive root.
    calculation_path_in_archive: str
    # Usable results folders, in numeric order.
    results_folders: tuple[str, ...]
    # Results folders present in the archive but missing required files.
    incomplete_results_folders: tuple[str, ...]
    files_kept: int
    bytes_kept: int
    members_skipped: int
    notes: tuple[str, ...] = field(default=())


# --- Format detection --------------------------------------------------------


def detect_archive_format(head: bytes) -> str:
    """
    Return ``"zip"`` or ``"tar.gz"`` from the first bytes of a file.

    The file content decides, not its name, so a renamed file is still read
    correctly and an unsupported file gets a clear message.
    """

    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return ARCHIVE_FORMAT_ZIP
    if head.startswith(b"\x1f\x8b"):
        return ARCHIVE_FORMAT_TAR_GZ
    if not head:
        raise ArchiveError("The uploaded file is empty.", suggestion=ARCHIVE_FORMAT_SUGGESTION)
    if len(head) > 262 and head[257:262] == b"ustar":
        raise ArchiveError(
            "The uploaded file is an uncompressed .tar archive, which is not accepted.",
            suggestion=ARCHIVE_FORMAT_SUGGESTION,
        )
    raise ArchiveError(
        "The uploaded file is not a .zip or .tar.gz archive.",
        suggestion=ARCHIVE_FORMAT_SUGGESTION,
    )


# --- Member names and types --------------------------------------------------


def safe_member_parts(name: str) -> tuple[str, ...]:
    """
    Return the path components of an archive member name, or raise.

    ``.`` components and empty components (``a//b``, a trailing ``/``) are
    dropped. Absolute paths, ``..``, backslashes, drive letters and NUL
    characters are refused.
    """

    shown = name if len(name) <= 200 else name[:200] + "..."
    if "\x00" in name:
        raise ArchiveError(f"Archive member {shown!r} has a NUL character in its name.")
    if "\\" in name:
        raise ArchiveError(
            f"Archive member {shown!r} has a backslash in its name.",
            suggestion="Create the archive with a tool that writes '/' as the path separator.",
        )
    if name.startswith("/") or _DRIVE_LETTER_PATTERN.match(name):
        raise ArchiveError(f"Archive member {shown!r} has an absolute path.")
    parts = tuple(part for part in name.split("/") if part not in ("", "."))
    if ".." in parts:
        raise ArchiveError(f"Archive member {shown!r} has a '..' path component.")
    return parts


def _keep(parts: tuple[str, ...]) -> bool:
    if not parts:
        return False
    if parts[-1] in KEPT_ANYWHERE_FILES:
        return True
    return (
        len(parts) >= 2
        and parts[-1] in KEPT_RESULTS_FILES
        and RESULTS_FOLDER_PATTERN.match(parts[-2]) is not None
    )


def _zip_member_kind(info: zipfile.ZipInfo) -> str:
    """Return "file" or "dir", or raise for any other member type."""

    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if info.is_dir():
        if file_type not in (0, stat.S_IFDIR):
            raise ArchiveError(f"Archive member {info.filename!r} is not a plain folder.")
        return "dir"
    if stat.S_ISLNK(mode):
        raise ArchiveError(
            f"Archive member {info.filename!r} is a symbolic link; links are not accepted."
        )
    if file_type not in (0, stat.S_IFREG):
        raise ArchiveError(
            f"Archive member {info.filename!r} is not a plain file (device, FIFO or "
            "other special file)."
        )
    if info.flag_bits & 0x1:
        raise ArchiveError(
            f"Archive member {info.filename!r} is encrypted.",
            suggestion="Upload an archive without a password.",
        )
    return "file"


def _tar_member_kind(member: tarfile.TarInfo) -> str:
    if member.isdir():
        return "dir"
    if member.isreg():
        return "file"
    if member.issym() or member.islnk():
        raise ArchiveError(
            f"Archive member {member.name!r} is a link; links are not accepted."
        )
    if member.ischr() or member.isblk() or member.isfifo():
        raise ArchiveError(
            f"Archive member {member.name!r} is a device file or FIFO; only plain files "
            "and folders are accepted."
        )
    raise ArchiveError(
        f"Archive member {member.name!r} is not a plain file or folder."
    )


# --- Writing kept files ------------------------------------------------------


class _Writer:
    """Writes kept members under one root, enforcing the size limits."""

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.files_kept = 0
        self.bytes_kept = 0
        self.kept_parts: list[tuple[str, ...]] = []

    def write(self, parts: tuple[str, ...], source, display_name: str) -> None:
        target = os.path.join(self.root, *parts)
        parent = os.path.dirname(target)
        # Names were already checked by safe_member_parts() and no links are
        # ever created, so this is defence in depth only.
        os.makedirs(parent, mode=0o700, exist_ok=True)
        if os.path.commonpath([self.root, os.path.realpath(parent)]) != self.root:
            raise ArchiveError(f"Archive member {display_name!r} would be written outside the import folder.")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError as exc:
            raise ArchiveError(
                f"Archive member {display_name!r} appears more than once."
            ) from exc
        member_bytes = 0
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                member_bytes += len(chunk)
                self.bytes_kept += len(chunk)
                if member_bytes > MAX_MEMBER_BYTES:
                    raise ArchiveError(
                        f"Archive member {display_name!r} is larger than the "
                        f"{MAX_MEMBER_BYTES // (1024 * 1024)} MB limit for one result file."
                    )
                if self.bytes_kept > MAX_EXTRACTED_BYTES:
                    raise ArchiveError(
                        "The USPEX result files in the archive are larger than the "
                        f"{MAX_EXTRACTED_BYTES // (1024 * 1024)} MB limit in total."
                    )
                handle.write(chunk)
        self.files_kept += 1
        self.kept_parts.append(parts)


class _LimitedReader:
    """File-like wrapper that stops reading after ``limit`` bytes."""

    def __init__(self, raw, limit: int):
        self.raw = raw
        self.limit = limit
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.count += len(data)
        if self.count > self.limit:
            raise ArchiveError(
                "The archive decompresses to more than "
                f"{MAX_DECOMPRESSED_BYTES // (1024 * 1024 * 1024)} GB.",
                suggestion="Archive only the results folders, not the whole calculation.",
            )
        return data


def _count_member(count: int) -> int:
    count += 1
    if count > MAX_ARCHIVE_MEMBERS:
        raise ArchiveError(
            f"The archive has more than {MAX_ARCHIVE_MEMBERS} files and folders.",
            suggestion="Archive only the results folders, not the whole calculation.",
        )
    return count


def _extract_zip(archive_path: str, writer: _Writer) -> int:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                _count_member(MAX_ARCHIVE_MEMBERS)
            # Check every member before writing anything.
            checked = []
            for info in infos:
                parts = safe_member_parts(info.filename)
                kind = _zip_member_kind(info)
                checked.append((info, parts, kind))
            skipped = 0
            for info, parts, kind in checked:
                if kind == "file" and _keep(parts):
                    with archive.open(info) as source:
                        writer.write(parts, source, info.filename)
                elif kind == "file":
                    skipped += 1
            return skipped
    except ArchiveError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, EOFError,
            zlib.error, OSError, RuntimeError, ValueError) as exc:
        raise ArchiveError(
            "The .zip archive could not be read (it may be damaged or use an "
            "unsupported compression method).",
            suggestion=ARCHIVE_FORMAT_SUGGESTION,
        ) from exc


def _extract_tar_gz(archive_path: str, writer: _Writer) -> int:
    skipped = 0
    count = 0
    try:
        with open(archive_path, "rb") as raw, gzip.GzipFile(fileobj=raw) as unzipped:
            reader = _LimitedReader(unzipped, MAX_DECOMPRESSED_BYTES)
            # Stream mode: members are read once, in order, never seeked.
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    count = _count_member(count)
                    parts = safe_member_parts(member.name)
                    kind = _tar_member_kind(member)
                    if kind == "file" and _keep(parts):
                        source = archive.extractfile(member)
                        if source is None:  # pragma: no cover - regular files always have data
                            raise ArchiveError(f"Archive member {member.name!r} could not be read.")
                        writer.write(parts, source, member.name)
                    elif kind == "file":
                        skipped += 1
    except ArchiveError:
        raise
    except (tarfile.TarError, EOFError, zlib.error, OSError, ValueError) as exc:
        raise ArchiveError(
            "The .tar.gz archive could not be read (it may be damaged, or be a "
            "gzip file that is not a tar archive).",
            suggestion=ARCHIVE_FORMAT_SUGGESTION,
        ) from exc
    return skipped


# --- Locating the calculation folder ----------------------------------------


def _locate_search(writer: _Writer) -> tuple[tuple[str, ...], list[str], list[str]]:
    by_calculation: dict[tuple[str, ...], dict[str, set[str]]] = {}
    for parts in writer.kept_parts:
        if len(parts) >= 2 and RESULTS_FOLDER_PATTERN.match(parts[-2]):
            by_calculation.setdefault(parts[:-2], {}).setdefault(parts[-2], set()).add(parts[-1])

    if not by_calculation:
        raise ArchiveError(
            "No USPEX results folder (results1/, results2/, ...) was found in the archive.",
            suggestion=MINIMUM_CONTENT_SUGGESTION,
        )
    if len(by_calculation) > 1:
        shown = ", ".join("/".join(path) or "(archive root)" for path in sorted(by_calculation))
        raise ArchiveError(
            f"The archive holds results of more than one USPEX calculation ({shown}).",
            suggestion="Upload one calculation per archive.",
        )
    ((calculation, folders),) = by_calculation.items()
    usable, incomplete = [], []
    for folder in sorted(folders, key=lambda name: int(RESULTS_FOLDER_PATTERN.match(name).group(1))):
        names = folders[folder]
        if REQUIRED_TABLE_FILE in names and any(name in names for name in REQUIRED_POSCARS_FILES):
            usable.append(folder)
        else:
            incomplete.append(folder)
    if not usable:
        raise ArchiveError(
            "No results folder in the archive has both Individuals and "
            "gatheredPOSCARS (or good_structures_POSCARS): "
            + "; ".join(
                f"{folder} has {', '.join(sorted(folders[folder])) or 'none of them'}"
                for folder in incomplete
            )
            + ".",
            suggestion=MINIMUM_CONTENT_SUGGESTION,
        )
    return calculation, usable, incomplete


def _scan_for_loose_results(archive_path: str, archive_format: str) -> list[str]:
    """Names of result files that sit outside any resultsN folder (for the message)."""

    names = []
    try:
        if archive_format == ARCHIVE_FORMAT_ZIP:
            with zipfile.ZipFile(archive_path) as archive:
                names = [info.filename for info in archive.infolist()[:MAX_ARCHIVE_MEMBERS]]
        else:
            with open(archive_path, "rb") as raw, gzip.GzipFile(fileobj=raw) as unzipped:
                reader = _LimitedReader(unzipped, MAX_DECOMPRESSED_BYTES)
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    names = [member.name for member in archive]
    except (ArchiveError, tarfile.TarError, zipfile.BadZipFile, EOFError, zlib.error, OSError):
        return []
    loose = []
    for name in names:
        parts = tuple(part for part in name.split("/") if part not in ("", "."))
        if parts and parts[-1] in (REQUIRED_TABLE_FILE, *REQUIRED_POSCARS_FILES):
            if len(parts) < 2 or not RESULTS_FOLDER_PATTERN.match(parts[-2]):
                loose.append("/".join(parts))
    return sorted(loose)


# --- Public entry point ------------------------------------------------------


def extract_uspex_archive(archive_path: str, destination: str) -> ExtractedSearch:
    """
    Check an uploaded archive and unpack its USPEX result files.

    Parameters
    ----------
    archive_path
        The uploaded file. Its size must already be within
        :data:`MAX_ARCHIVE_BYTES` (checked again here).
    destination
        A folder that does not exist yet. It is created, and result files
        are written below it, keeping their paths inside the archive.

    Raises :class:`ArchiveError` with a message for the user if the archive
    is unsafe, too large, unreadable or does not contain usable USPEX
    results. On any error ``destination`` is removed again, so nothing from
    a rejected archive stays on disk.
    """

    size = os.path.getsize(archive_path)
    if size > MAX_ARCHIVE_BYTES:
        raise ArchiveError(
            f"The archive is larger than the {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB upload limit.",
            suggestion="Archive only the results folders, not the whole calculation.",
        )
    with open(archive_path, "rb") as handle:
        head = handle.read(512)
    archive_format = detect_archive_format(head)

    if os.path.exists(destination):
        raise ValueError(f"Extraction destination already exists: {destination}")
    os.makedirs(destination, mode=0o700)
    writer = _Writer(destination)
    try:
        if archive_format == ARCHIVE_FORMAT_ZIP:
            skipped = _extract_zip(archive_path, writer)
        else:
            skipped = _extract_tar_gz(archive_path, writer)
        calculation, usable, incomplete = _locate_search(writer)
    except ArchiveError as exc:
        # A streamed .tar.gz may have written members before the failing one.
        shutil.rmtree(destination, ignore_errors=True)
        loose = _scan_for_loose_results(archive_path, archive_format)
        if loose and "No USPEX results folder" in exc.message:
            raise ArchiveError(
                exc.message
                + " Result files were found outside a results folder: "
                + ", ".join(loose[:5])
                + ".",
                suggestion=exc.suggestion,
            ) from exc
        raise
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    notes = []
    if incomplete:
        notes.append(
            "Results folder(s) "
            + ", ".join(incomplete)
            + " lack Individuals or a POSCARS file and were not read."
        )
    return ExtractedSearch(
        calculation_dir=os.path.join(writer.root, *calculation),
        calculation_path_in_archive="/".join(calculation),
        results_folders=tuple(usable),
        incomplete_results_folders=tuple(incomplete),
        files_kept=writer.files_kept,
        bytes_kept=writer.bytes_kept,
        members_skipped=skipped,
        notes=tuple(notes),
    )


__all__ = [
    "ARCHIVE_FORMAT_TAR_GZ",
    "ARCHIVE_FORMAT_ZIP",
    "ArchiveError",
    "ExtractedSearch",
    "KEPT_RESULTS_FILES",
    "MAX_ARCHIVE_BYTES",
    "MAX_ARCHIVE_MEMBERS",
    "MAX_DECOMPRESSED_BYTES",
    "MAX_EXTRACTED_BYTES",
    "MAX_MEMBER_BYTES",
    "detect_archive_format",
    "extract_uspex_archive",
    "safe_member_parts",
]
