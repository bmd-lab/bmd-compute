"""Copying BMD's POTCAR choice into a search's Specific/ folder (placeholder files only)."""

import hashlib

import pytest

from backend.structure_search.incar_policy import IncarPolicyError, require_encut_covers_enmax
from backend.structure_search.potcar import (
    POTCAR_FUNCTIONAL_DIR,
    SpecificPotcarError,
    bmd_potcar_symbol,
    max_enmax,
    potcar_source_path,
    read_potcar_enmax,
    write_specific_potcars,
)


def _placeholder_text(symbol, enmax=400.0, header=None):
    """Header lines laid out like a POTCAR's; no potential data, never a real POTCAR."""

    first = header or f"  PAW_PBE {symbol} 01Jan2000"
    return (
        f"{first}\n"
        "PLACEHOLDER, NOT A REAL POTCAR\n"
        "   parameters from PSCTR are:\n"
        f"   TITEL  = PAW_PBE {symbol} 01Jan2000\n"
        f"   ENMAX  =  {enmax:.3f}; ENMIN  =  {0.75 * enmax:.3f} eV\n"
        "   END of PSCTR-controll parameters\n"
    )


def _placeholder_repository(root, symbols, header=None, enmax=None):
    """Fake POTCAR repository of placeholder files."""

    for symbol in symbols:
        folder = root / POTCAR_FUNCTIONAL_DIR / symbol
        folder.mkdir(parents=True)
        value = (enmax or {}).get(symbol, 400.0)
        (folder / "POTCAR").write_text(_placeholder_text(symbol, value, header), encoding="utf-8")


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
    assert [row["enmax"] for row in records] == [400.0, 400.0]


def test_read_potcar_enmax():
    assert read_potcar_enmax(_placeholder_text("Li_sv", 499.034)) == 499.034
    # ENMAX outside the PSCTR block, or none, or two values, is refused.
    with pytest.raises(SpecificPotcarError):
        read_potcar_enmax("  PAW_PBE O 08Apr2002\n   ENMAX  =  400.000; ENMIN = 300.000 eV\n")
    with pytest.raises(SpecificPotcarError):
        read_potcar_enmax("   parameters from PSCTR are:\n   END of PSCTR-controll parameters\n")
    doubled = _placeholder_text("O").replace(
        "   END of", "   ENMAX  =  500.000; ENMIN  =  375.000 eV\n   END of"
    )
    with pytest.raises(SpecificPotcarError):
        read_potcar_enmax(doubled)


def test_potcar_without_enmax_writes_nothing(tmp_path):
    psp = tmp_path / "psp"
    specific = tmp_path / "Specific"
    specific.mkdir()
    folder = psp / POTCAR_FUNCTIONAL_DIR / "O"
    folder.mkdir(parents=True)
    (folder / "POTCAR").write_text("  PAW_PBE O 08Apr2002\nPLACEHOLDER\n", encoding="utf-8")
    with pytest.raises(SpecificPotcarError, match="ENMAX"):
        write_specific_potcars(["O"], psp, specific)
    assert list(specific.iterdir()) == []


def test_search_with_a_hard_potcar_is_refused_by_the_ladder(tmp_path):
    """Li_sv (ENMAX 499.034 eV in BMD's PBE_64 metadata) is above the early-step cutoff."""

    psp = tmp_path / "psp"
    specific = tmp_path / "Specific"
    specific.mkdir()
    assert bmd_potcar_symbol("Li") == "Li_sv"
    _placeholder_repository(psp, ["Li_sv", "O"], enmax={"Li_sv": 499.034})
    records = write_specific_potcars(["Li", "O"], psp, specific)
    assert max_enmax(records) == 499.034
    with pytest.raises(IncarPolicyError) as excinfo:
        require_encut_covers_enmax(max_enmax(records))
    assert excinfo.value.code == "encut_below_potcar_enmax"


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
