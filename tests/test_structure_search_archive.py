"""
Tests for backend/structure_search/archive.py (issue #63): safe unpacking of
uploaded USPEX results archives.

Archives are built in memory from the real Si8 fixture
(tests/fixtures/uspex_si8_pilot/results1/, see its README) or from small
hand-made members for the unsafe cases.
"""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from backend.structure_search import archive as archive_module
from backend.structure_search.archive import (
    ArchiveError,
    detect_archive_format,
    extract_uspex_archive,
    safe_member_parts,
)


PILOT_RESULTS = Path(__file__).parent / "fixtures" / "uspex_si8_pilot" / "results1"
MINIMAL = ("Individuals", "gatheredPOSCARS")


def _pilot_members(prefix: str = "", names=None) -> dict[str, bytes]:
    names = names or sorted(path.name for path in PILOT_RESULTS.iterdir())
    return {f"{prefix}results1/{name}": (PILOT_RESULTS / name).read_bytes() for name in names}


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _tar_gz_bytes(members: dict[str, bytes], extra=()) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for info in extra:
            archive.addfile(info)
    return buffer.getvalue()


def _extract(tmp_path: Path, data: bytes, suffix: str = ".zip"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"upload{suffix}"
    path.write_bytes(data)
    return extract_uspex_archive(str(path), str(tmp_path / "out"))


# --- Accepted layouts --------------------------------------------------------


@pytest.mark.parametrize("builder,suffix", [(_zip_bytes, ".zip"), (_tar_gz_bytes, ".tar.gz")])
def test_results_folder_at_archive_root(tmp_path, builder, suffix):
    extracted = _extract(tmp_path, builder(_pilot_members()), suffix)
    assert extracted.calculation_path_in_archive == ""
    assert extracted.results_folders == ("results1",)
    assert Path(extracted.calculation_dir) == (tmp_path / "out").resolve()
    kept = sorted(path.name for path in (tmp_path / "out" / "results1").iterdir())
    assert kept == sorted(archive_module.KEPT_RESULTS_FILES)
    # Individuals_all, Individuals_extended, README.md, ... are not written.
    assert extracted.members_skipped == len(list(PILOT_RESULTS.iterdir())) - len(kept)


def test_calculation_folder_with_other_files(tmp_path):
    members = _pilot_members("search/")
    members["search/INPUT.txt"] = b"% atomType\nSi\n% EndAtomType\n"
    members["search/Calculation/Calcfold_1_1/OUTCAR"] = b"x" * 1000
    members["search/Specific/INCAR_1"] = b"ENCUT = 250\n"
    members["search/run.sh"] = b"#!/bin/sh\nrm -rf /\n"
    extracted = _extract(tmp_path, _zip_bytes(members))
    assert extracted.calculation_path_in_archive == "search"
    assert (tmp_path / "out" / "search" / "INPUT.txt").is_file()
    assert not (tmp_path / "out" / "search" / "run.sh").exists()
    assert not (tmp_path / "out" / "search" / "Calculation").exists()


def test_several_results_folders(tmp_path):
    members = _pilot_members()
    members.update({f"results2/{name}": (PILOT_RESULTS / name).read_bytes() for name in MINIMAL})
    members["results10/BESTIndividuals"] = b""
    extracted = _extract(tmp_path, _tar_gz_bytes(members), ".tar.gz")
    assert extracted.results_folders == ("results1", "results2")
    assert extracted.incomplete_results_folders == ("results10",)
    assert "results10" in extracted.notes[0]


def test_kept_files_are_not_executable(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in _pilot_members(names=MINIMAL).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(info, data)
    _extract(tmp_path, buffer.getvalue())
    for path in (tmp_path / "out" / "results1").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_minimum_content_good_structures_only(tmp_path):
    extracted = _extract(
        tmp_path, _zip_bytes(_pilot_members(names=("Individuals", "good_structures_POSCARS")))
    )
    assert extracted.results_folders == ("results1",)


# --- Unsafe archives ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,message",
    [
        ("/etc/results1/Individuals", "absolute path"),
        ("C:/results1/Individuals", "absolute path"),
        ("../results1/Individuals", "'..'"),
        ("search/../../results1/Individuals", "'..'"),
        ("search\\results1\\Individuals", "backslash"),
    ],
)
@pytest.mark.parametrize("builder,suffix", [(_zip_bytes, ".zip"), (_tar_gz_bytes, ".tar.gz")])
def test_unsafe_names_reject_the_whole_archive(tmp_path, name, message, builder, suffix):
    members = _pilot_members(names=MINIMAL)
    members[name] = b"payload"
    with pytest.raises(ArchiveError, match=message):
        _extract(tmp_path, builder(members), suffix)
    assert not (tmp_path / "out").exists()


def test_safe_member_parts_normalises_dots_and_slashes():
    assert safe_member_parts("./search//results1/Individuals") == (
        "search",
        "results1",
        "Individuals",
    )
    assert safe_member_parts("results1/") == ("results1",)
    with pytest.raises(ArchiveError, match="NUL"):
        safe_member_parts("results1/Individuals\x00.txt")


def test_zip_symlink_is_rejected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in _pilot_members(names=MINIMAL).items():
            archive.writestr(name, data)
        link = zipfile.ZipInfo("results1/BESTIndividuals")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "/etc/passwd")
    with pytest.raises(ArchiveError, match="symbolic link"):
        _extract(tmp_path, buffer.getvalue())
    assert not (tmp_path / "out").exists()


def test_zip_device_entry_is_rejected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        device = zipfile.ZipInfo("results1/Individuals")
        device.external_attr = (stat.S_IFCHR | 0o666) << 16
        archive.writestr(device, b"")
    with pytest.raises(ArchiveError, match="not a plain file"):
        _extract(tmp_path, buffer.getvalue())


@pytest.mark.parametrize(
    "member_type,message",
    [
        (tarfile.SYMTYPE, "link"),
        (tarfile.LNKTYPE, "link"),
        (tarfile.CHRTYPE, "device file"),
        (tarfile.BLKTYPE, "device file"),
        (tarfile.FIFOTYPE, "FIFO"),
    ],
)
def test_tar_special_members_are_rejected(tmp_path, member_type, message):
    special = tarfile.TarInfo("results1/BESTIndividuals")
    special.type = member_type
    special.linkname = "/etc/passwd"
    data = _tar_gz_bytes(_pilot_members(names=MINIMAL), extra=[special])
    with pytest.raises(ArchiveError, match=message):
        _extract(tmp_path, data, ".tar.gz")


def test_link_with_a_skipped_name_is_still_rejected(tmp_path):
    link = tarfile.TarInfo("search/notes")
    link.type = tarfile.SYMTYPE
    link.linkname = "/"
    data = _tar_gz_bytes(_pilot_members(names=MINIMAL), extra=[link])
    with pytest.raises(ArchiveError, match="link"):
        _extract(tmp_path, data, ".tar.gz")


def test_duplicate_member_is_rejected(tmp_path):
    members = _pilot_members(names=MINIMAL)
    members["./results1/Individuals"] = b"generation number num_atoms_all energy\n"
    with pytest.raises(ArchiveError, match="more than once"):
        _extract(tmp_path, _tar_gz_bytes(members), ".tar.gz")


# --- Size and count limits ---------------------------------------------------


def test_oversize_archive_is_rejected(tmp_path, monkeypatch):
    data = _zip_bytes(_pilot_members(names=MINIMAL))
    monkeypatch.setattr(archive_module, "MAX_ARCHIVE_BYTES", len(data) - 1)
    with pytest.raises(ArchiveError, match="upload limit"):
        _extract(tmp_path, data)


def test_too_many_members_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "MAX_ARCHIVE_MEMBERS", 3)
    members = _pilot_members(names=MINIMAL)
    members.update({f"other/{index}": b"" for index in range(5)})
    for builder, suffix in ((_zip_bytes, ".zip"), (_tar_gz_bytes, ".tar.gz")):
        with pytest.raises(ArchiveError, match="more than 3 files"):
            _extract(tmp_path / suffix.strip("."), builder(members), suffix)


@pytest.mark.parametrize("builder,suffix", [(_zip_bytes, ".zip"), (_tar_gz_bytes, ".tar.gz")])
def test_member_bigger_than_its_limit_stops_while_decompressing(
    tmp_path, monkeypatch, builder, suffix
):
    # A highly compressible member: the archive is small, the content is not.
    members = _pilot_members(names=MINIMAL)
    members["results1/BESTgatheredPOSCARS"] = b"\0" * (2 * 1024 * 1024)
    monkeypatch.setattr(archive_module, "MAX_MEMBER_BYTES", 1024 * 1024)
    with pytest.raises(ArchiveError, match="limit for one result file"):
        _extract(tmp_path, builder(members), suffix)


def test_total_extracted_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "MAX_EXTRACTED_BYTES", 20_000)
    with pytest.raises(ArchiveError, match="limit in total"):
        _extract(tmp_path, _zip_bytes(_pilot_members()))


def test_decompressed_stream_limit_counts_skipped_members(tmp_path, monkeypatch):
    members = _pilot_members(names=MINIMAL)
    members["search/Calculation/WAVECAR"] = b"\0" * (4 * 1024 * 1024)
    monkeypatch.setattr(archive_module, "MAX_DECOMPRESSED_BYTES", 1024 * 1024)
    with pytest.raises(ArchiveError, match="decompresses to more than"):
        _extract(tmp_path, _tar_gz_bytes(members), ".tar.gz")


# --- Formats and missing content --------------------------------------------


def test_format_is_detected_from_content():
    assert detect_archive_format(_zip_bytes({"a": b""})[:8]) == "zip"
    assert detect_archive_format(_tar_gz_bytes({"a": b""})[:8]) == "tar.gz"
    with pytest.raises(ArchiveError, match="empty"):
        detect_archive_format(b"")
    with pytest.raises(ArchiveError, match="not a .zip or .tar.gz"):
        detect_archive_format(b"generation number num_atoms_all")


def test_uncompressed_tar_gets_its_own_message(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("results1/Individuals")
        archive.addfile(info, io.BytesIO(b""))
    with pytest.raises(ArchiveError, match="uncompressed .tar"):
        _extract(tmp_path, buffer.getvalue(), ".tar")


def test_gzip_that_is_not_tar(tmp_path):
    import gzip

    with pytest.raises(ArchiveError, match="could not be read"):
        _extract(tmp_path, gzip.compress(b"just text, not a tar archive" * 40), ".tar.gz")


def test_damaged_zip(tmp_path):
    data = _zip_bytes(_pilot_members(names=MINIMAL))
    with pytest.raises(ArchiveError, match="could not be read"):
        _extract(tmp_path, data[: len(data) // 2])


def test_archive_without_results_folder(tmp_path):
    with pytest.raises(ArchiveError, match="No USPEX results folder"):
        _extract(tmp_path, _zip_bytes({"notes/readme.txt": b"hello"}))


def test_loose_result_files_are_named_in_the_message(tmp_path):
    members = {name: (PILOT_RESULTS / name).read_bytes() for name in MINIMAL}
    with pytest.raises(ArchiveError, match="outside a results folder: .*Individuals"):
        _extract(tmp_path, _zip_bytes(members))


def test_results_folder_without_poscars(tmp_path):
    with pytest.raises(ArchiveError, match="results1 has Individuals"):
        _extract(tmp_path, _zip_bytes(_pilot_members(names=("Individuals",))))


def test_two_calculations_in_one_archive(tmp_path):
    members = _pilot_members("a/", names=MINIMAL)
    members.update(_pilot_members("b/", names=MINIMAL))
    with pytest.raises(ArchiveError, match="more than one USPEX calculation"):
        _extract(tmp_path, _zip_bytes(members))


def test_module_uses_only_the_standard_library():
    import ast

    tree = ast.parse(Path(archive_module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert modules <= {
        "__future__", "dataclasses", "gzip", "os", "re", "shutil", "stat", "tarfile",
        "zipfile", "zlib",
    }
