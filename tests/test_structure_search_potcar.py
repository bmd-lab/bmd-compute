"""Copying BMD's POTCAR choice into a search's Specific/ folder (placeholder files only)."""

import hashlib

import pytest
from conftest import synthetic_enmax_metadata, use_enmax_metadata

from backend.structure_search.incar_policy import (
    IncarPolicyError,
    ladder_cutoffs,
    require_encut_covers_enmax,
    require_potcars_match_cutoff_record,
    search_cutoff_record,
)
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


# The only POTCARs BMD selects whose ENMAX is above 430 eV, the fixed early-step
# cutoff the ladder used before it followed BMD's species-dependent cutoff
# rule. Read on POWER from /bmd-db/potcars/POT_PAW_PBE_64 (4 Nov 2026); they
# match backend/calculations/potcar_enmax_PBE_64.json.
HARD_BMD_POTCARS = [("Li", "Li_sv", 499.034), ("He", "He", 478.896)]


@pytest.fixture
def hard_potcar_metadata(monkeypatch, tmp_path):
    """SYNTHETIC test metadata with the published values (tests never read the real file)."""

    values = {symbol: enmax for _, symbol, enmax in HARD_BMD_POTCARS} | {"O": 400.0}
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", synthetic_enmax_metadata(values))


@pytest.mark.parametrize(("element", "symbol", "enmax"), HARD_BMD_POTCARS)
def test_search_with_a_hard_potcar_gets_cutoffs_above_its_enmax(
    tmp_path, hard_potcar_metadata, element, symbol, enmax
):
    psp = tmp_path / "psp"
    specific = tmp_path / "Specific"
    specific.mkdir()
    assert bmd_potcar_symbol(element) == symbol
    _placeholder_repository(psp, [symbol, "O"], enmax={symbol: enmax, "O": 400.0})
    text = (psp / POTCAR_FUNCTIONAL_DIR / symbol / "POTCAR").read_text(encoding="utf-8")
    assert f"   ENMAX  =  {enmax:.3f}; ENMIN  =" in text  # the standard header layout
    records = write_specific_potcars([element, "O"], psp, specific)
    assert max_enmax(records) == enmax

    cutoff_record = search_cutoff_record([element, "O"])
    assert cutoff_record["max_enmax_ev"] == enmax
    require_potcars_match_cutoff_record(records, cutoff_record)
    require_encut_covers_enmax(max_enmax(records), cutoff_record)
    assert ladder_cutoffs(cutoff_record)["early"] == 500  # was 430, below this ENMAX


def test_mismatched_potcar_is_refused_against_the_cutoff_record(tmp_path, hard_potcar_metadata):
    """A POTCAR whose ENMAX differs from the metadata the cutoffs came from."""

    psp = tmp_path / "psp"
    specific = tmp_path / "Specific"
    specific.mkdir()
    _placeholder_repository(psp, ["Li_sv", "O"], enmax={"Li_sv": 520.0, "O": 400.0})
    records = write_specific_potcars(["Li", "O"], psp, specific)
    with pytest.raises(IncarPolicyError) as excinfo:
        require_potcars_match_cutoff_record(records, search_cutoff_record(["Li", "O"]))
    assert excinfo.value.code == "potcar_enmax_differs_from_cutoff_record"
    assert excinfo.value.diagnostic["symbols"] == ["Li_sv"]
