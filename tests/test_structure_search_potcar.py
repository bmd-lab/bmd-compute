"""Copying BMD's POTCAR choice into a search's Specific/ folder (placeholder files only)."""

import hashlib

import pytest

from backend.structure_search.potcar import (
    POTCAR_FUNCTIONAL_DIR,
    SpecificPotcarError,
    bmd_potcar_symbol,
    potcar_source_path,
    write_specific_potcars,
)


def _placeholder_repository(root, symbols, header=None):
    """Fake POTCAR repository; files contain only a header line, never real potentials."""

    for symbol in symbols:
        folder = root / POTCAR_FUNCTIONAL_DIR / symbol
        folder.mkdir(parents=True)
        first = header or f"  PAW_PBE {symbol} 01Jan2000"
        (folder / "POTCAR").write_text(f"{first}\nPLACEHOLDER, NOT A REAL POTCAR\n", encoding="utf-8")


def test_functional_dir_matches_pymatgen():
    from pymatgen.io.vasp.inputs import PotcarSingle

    assert POTCAR_FUNCTIONAL_DIR == PotcarSingle.functional_dir["PBE_64"]


def test_source_path_uses_bmd_symbol(tmp_path):
    assert potcar_source_path(tmp_path, "Ti") == tmp_path / "POT_PAW_PBE_64" / "Ti_sv" / "POTCAR"


def test_writes_one_potcar_per_element(tmp_path):
    psp = tmp_path / "psp"
    specific = tmp_path / "Specific"
    specific.mkdir()
    _placeholder_repository(psp, ["Ti_sv", "O"])

    records = write_specific_potcars(["Ti", "O"], psp, specific)

    assert sorted(path.name for path in specific.iterdir()) == ["POTCAR_O", "POTCAR_Ti"]
    assert (specific / "POTCAR_Ti").read_bytes() == (psp / "POT_PAW_PBE_64" / "Ti_sv" / "POTCAR").read_bytes()
    assert [(row["element"], row["potcar_symbol"], row["target"]) for row in records] == [
        ("Ti", "Ti_sv", "POTCAR_Ti"), ("O", "O", "POTCAR_O"),
    ]
    assert records[0]["sha256"] == hashlib.sha256((specific / "POTCAR_Ti").read_bytes()).hexdigest()
    assert all(row["functional"] == "PBE_64" for row in records)


def test_missing_potcar_writes_nothing(tmp_path):
    psp = tmp_path / "psp"
    specific = tmp_path / "Specific"
    specific.mkdir()
    _placeholder_repository(psp, ["O"])
    with pytest.raises(SpecificPotcarError, match="Ti_sv"):
        write_specific_potcars(["O", "Ti"], psp, specific)
    assert list(specific.iterdir()) == []


def test_wrong_potcar_file_is_refused(tmp_path):
    psp = tmp_path / "psp"
    specific = tmp_path / "Specific"
    specific.mkdir()
    _placeholder_repository(psp, ["Ti_sv"], header="  PAW_PBE Ti 08Apr2002")
    with pytest.raises(SpecificPotcarError, match="does not start with"):
        write_specific_potcars(["Ti"], psp, specific)
    assert list(specific.iterdir()) == []


def test_unmapped_element_is_refused(tmp_path):
    assert bmd_potcar_symbol("Og") is None
    with pytest.raises(SpecificPotcarError):
        potcar_source_path(tmp_path, "Og")
