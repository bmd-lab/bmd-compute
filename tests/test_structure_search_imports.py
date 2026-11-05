"""
Tests for backend/structure_search/imports.py (issue #63).

The real Si8 pilot output (tests/fixtures/uspex_si8_pilot/results1/) is packed
into an archive in memory and imported into a temporary upload folder. Its
best candidate, individual 35, is diamond Si (USPEX lists it as P1).
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from backend.parser import parse_structure
from backend.structure_origin import verify_structure_origin
from backend.structure_search import imports
from backend.structure_search.imports import StructureSearchImportError


PILOT = Path(__file__).parent / "fixtures" / "uspex_si8_pilot"


def pilot_tar_gz(prefix: str = "si8_search/") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(PILOT / "results1", arcname=f"{prefix}results1")
    return buffer.getvalue()


def pilot_zip(files=None, parameters: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((PILOT / "results1").iterdir()):
            if files is None or path.name in files:
                data = path.read_bytes()
                if parameters is not None and path.name == "Parameters.txt":
                    data = parameters.encode()
                archive.writestr(f"results1/{path.name}", data)
    return buffer.getvalue()


@pytest.fixture
def uploads(tmp_path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def pilot_import(uploads) -> str:
    return imports.import_uspex_archive(
        io.BytesIO(pilot_tar_gz()), "si8 search.tar.gz", uploads_root=uploads
    )


# --- Import ------------------------------------------------------------------


def test_import_of_real_pilot_archive(uploads, pilot_import):
    data = pilot_tar_gz()
    page = imports.load_candidate_page(pilot_import, uploads_root=uploads)

    assert page["archive"]["format"] == "tar.gz"
    assert page["archive"]["size_bytes"] == len(data)
    import hashlib

    assert page["archive"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert page["archive"]["filename"] == "si8 search.tar.gz"
    assert page["calculation_path_in_archive"] == "si8_search"
    assert page["results_folders"] == ["results1"]
    assert page["atom_types"] == ["Si"]
    assert page["n_individuals"] == 35
    assert page["external_pressure_gpa"] == 0.0
    assert page["pressure_note"] is None
    assert page["warnings"] == []


def test_uploaded_archive_is_not_kept(uploads, pilot_import):
    directory = uploads / "structure_search" / pilot_import
    names = {path.name for path in directory.iterdir()}
    assert names == {"import.json", "candidates.json", "search"}
    for path in directory.rglob("*"):
        if path.is_file():
            assert not os.access(path, os.X_OK)


def test_best_candidate_is_diamond_si(uploads, pilot_import):
    page = imports.load_candidate_page(pilot_import, uploads_root=uploads)
    best = page["candidates"][0]
    assert best["id"] == 35
    assert best["generation"] == 4
    assert best["origin"] == "keptBest"
    assert best["formula"] == "Si8"
    assert best["energy_ev"] == pytest.approx(-43.280)
    assert best["energy_ev_per_atom"] == pytest.approx(-5.41)
    assert best["volume_per_atom_a3"] == pytest.approx(163.675 / 8)
    assert best["space_group_uspex"] == 1
    assert best["symmetry"] == {
        "space_group_number": 227,
        "space_group_symbol": "Fd-3m",
        "symprec": 0.1,
    }
    assert best["handoff_available"] is True
    assert best["poscars_file"] == "gatheredPOSCARS"
    assert best["selection"] == "results1:35"


def test_every_listed_candidate_has_symmetry(uploads, pilot_import):
    page = imports.load_candidate_page(pilot_import, uploads_root=uploads)
    assert len(page["candidates"]) == page["n_candidates"]
    assert all(row["symmetry"]["space_group_number"] >= 1 for row in page["candidates"])


def test_energy_generation_plot(uploads, pilot_import):
    page = imports.load_candidate_page(pilot_import, uploads_root=uploads)
    (plot,) = page["visualizations"]
    assert plot["kind"] == "line_plot"
    best, listed = plot["plot"]["traces"]
    assert best["x"] == [1, 2, 3, 4]
    assert best["y"][-1] == pytest.approx(-5.41)
    assert best["text"][-1] == "ID 35"
    assert len(listed["x"]) == page["n_candidates"]
    json.dumps(page, allow_nan=False)


def test_good_structures_only_archive(uploads):
    import_id = imports.import_uspex_archive(
        io.BytesIO(pilot_zip(files={"Individuals", "good_structures_POSCARS", "Parameters.txt"})),
        "good.zip",
        uploads_root=uploads,
    )
    page = imports.load_candidate_page(import_id, uploads_root=uploads)
    rows = {row["id"]: row for row in page["candidates"]}
    assert rows[35]["poscars_file"] == "good_structures_POSCARS"
    unavailable = [row for row in page["candidates"] if not row["handoff_available"]]
    assert unavailable and all(row["symmetry"] is None for row in unavailable)
    assert any("cannot be handed off" in warning for warning in page["warnings"])


def test_parser_warnings_are_shown_without_server_paths(uploads):
    individuals = (PILOT / "results1" / "Individuals").read_text()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("results1/Individuals", individuals[:-20])  # last row cut short
        archive.writestr(
            "results1/gatheredPOSCARS", (PILOT / "results1" / "gatheredPOSCARS").read_bytes()
        )
    import_id = imports.import_uspex_archive(
        io.BytesIO(buffer.getvalue()), "cut.zip", uploads_root=uploads
    )
    page = imports.load_candidate_page(import_id, uploads_root=uploads)
    assert any(
        warning.startswith("results1/Individuals: last line") for warning in page["warnings"]
    )
    assert any("atomType" in warning for warning in page["warnings"])
    assert not any(str(uploads) in warning for warning in page["warnings"])


def test_non_zero_pressure_is_flagged(uploads):
    parameters = (PILOT / "results1" / "Parameters.txt").read_text().replace(
        "0.0 : ExternalPressure", "10 : ExternalPressure"
    )
    import_id = imports.import_uspex_archive(
        io.BytesIO(pilot_zip(parameters=parameters)), "p.zip", uploads_root=uploads
    )
    page = imports.load_candidate_page(import_id, uploads_root=uploads)
    assert page["external_pressure_gpa"] == 10.0
    assert "10 GPa" in page["pressure_note"]


@pytest.mark.parametrize(
    "data,message",
    [
        (b"", "empty"),
        (b"generation number num_atoms_all energy\n", "not a .zip or .tar.gz"),
        (pilot_zip(files={"Individuals"}), "gatheredPOSCARS"),
    ],
)
def test_bad_uploads_leave_nothing_behind(uploads, data, message):
    with pytest.raises(StructureSearchImportError, match=message):
        imports.import_uspex_archive(io.BytesIO(data), "x.zip", uploads_root=uploads)
    assert list((uploads / "structure_search").iterdir()) == []


def test_missing_file(uploads):
    with pytest.raises(StructureSearchImportError, match="No file was uploaded"):
        imports.import_uspex_archive(None, None, uploads_root=uploads)


def test_unknown_column_header_is_a_clear_error(uploads):
    individuals = (PILOT / "results1" / "Individuals").read_text().replace(
        "cell_volume", "cell_volume_new"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("results1/Individuals", individuals)
        archive.writestr("results1/gatheredPOSCARS", "")
    with pytest.raises(StructureSearchImportError, match="cell_volume_new") as caught:
        imports.import_uspex_archive(io.BytesIO(buffer.getvalue()), "x.zip", uploads_root=uploads)
    assert str(uploads) not in caught.value.message
    assert "results1/Individuals" in caught.value.message


def test_oversize_upload_is_rejected_while_reading(uploads, monkeypatch):
    from backend.structure_search import archive as archive_module

    monkeypatch.setattr(archive_module, "MAX_ARCHIVE_BYTES", 1000)
    with pytest.raises(StructureSearchImportError, match="upload limit"):
        imports.import_uspex_archive(io.BytesIO(pilot_tar_gz()), "big.tar.gz", uploads_root=uploads)
    assert list((uploads / "structure_search").iterdir()) == []


def test_old_imports_are_pruned(uploads, pilot_import):
    directory = uploads / "structure_search" / pilot_import
    old = time.time() - imports.IMPORT_RETENTION_SECONDS - 60
    os.utime(directory, (old, old))
    newer = imports.import_uspex_archive(io.BytesIO(pilot_tar_gz()), "b.tar.gz", uploads_root=uploads)
    assert not directory.exists()
    assert (uploads / "structure_search" / newer).is_dir()
    with pytest.raises(StructureSearchImportError, match="no longer available"):
        imports.load_candidate_page(pilot_import, uploads_root=uploads)


def test_number_of_stored_imports_is_bounded(uploads, monkeypatch):
    monkeypatch.setattr(imports, "MAX_STORED_IMPORTS", 2)
    made = []
    for age in (30, 20, 0):
        made.append(
            imports.import_uspex_archive(io.BytesIO(pilot_tar_gz()), "a.tar.gz", uploads_root=uploads)
        )
        stamp = time.time() - age
        os.utime(uploads / "structure_search" / made[-1], (stamp, stamp))
    remaining = {path.name for path in (uploads / "structure_search").iterdir()}
    assert remaining == set(made[1:])


@pytest.mark.parametrize("import_id", ["../../etc", "ABC", "", "0" * 31])
def test_invalid_import_id(uploads, import_id):
    with pytest.raises(StructureSearchImportError, match="not valid"):
        imports.load_candidate_page(import_id, uploads_root=uploads)


# --- Hand-off ----------------------------------------------------------------


def test_hand_off_candidate_35(uploads, pilot_import):
    (handed,) = imports.hand_off_candidates(pilot_import, ["results1:35"], uploads_root=uploads)
    structure = parse_structure(handed["structure_text"], "poscar")
    assert structure.composition.reduced_formula == "Si"
    assert len(structure) == 8
    assert handed["structure_text"].startswith("USPEX individual 35 (results1)\n")

    origin = verify_structure_origin(handed["origin_token"], handed["structure_text"])
    assert origin == handed["origin"]
    page = imports.load_candidate_page(pilot_import, uploads_root=uploads)
    assert origin["archive_sha256"] == page["archive"]["sha256"]
    assert origin["results_folder"] == "results1"
    assert origin["individual_id"] == 35
    assert origin["uspex_energy_ev"] == pytest.approx(-43.28)
    assert origin["uspex_energy_ev_per_atom"] == pytest.approx(-5.41)
    assert origin["origin"] == "keptBest"
    assert origin["lineage_ids"] == [12, 26, 35]
    assert origin["generation"] == 4
    assert origin["poscars_file"] == "gatheredPOSCARS"
    assert origin["source"] == "uspex_import"


def test_hand_off_several_candidates_keeps_order_and_drops_repeats(uploads, pilot_import):
    handed = imports.hand_off_candidates(
        pilot_import, ["results1:35", "results1:20", "results1:35"], uploads_root=uploads
    )
    assert [item["origin"]["individual_id"] for item in handed] == [35, 20]


@pytest.mark.parametrize(
    "selections,message",
    [
        ([], "No candidate was selected"),
        (["results1:999"], "not in this import"),
        (["results1:12"], "not in this import"),  # hidden member of the 12 -> 26 -> 35 chain
        (["../x:35"], "not in this import"),
    ],
)
def test_hand_off_rejects_bad_selections(uploads, pilot_import, selections, message):
    with pytest.raises(StructureSearchImportError, match=message):
        imports.hand_off_candidates(pilot_import, selections, uploads_root=uploads)


def test_hand_off_limit(uploads, pilot_import, monkeypatch):
    monkeypatch.setattr(imports, "MAX_HANDOFF_CANDIDATES", 1)
    with pytest.raises(StructureSearchImportError, match="At most 1"):
        imports.hand_off_candidates(
            pilot_import, ["results1:35", "results1:20"], uploads_root=uploads
        )
