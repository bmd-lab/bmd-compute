"""
Tests for backend/structure_search/results.py (issue #62).

The fixtures are synthetic (see tests/fixtures/uspex_synthetic/README.md) and
must be checked against real Phase 0 USPEX output before #63 is merged.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from backend.structure_search import results as uspex_results
from backend.structure_search.results import (
    USPEXFormatError,
    USPEXResultsError,
    find_results_folders,
    load_structures,
    summarise_search,
)


FIXTURE = Path(__file__).parent / "fixtures" / "uspex_synthetic" / "mgo_two_restarts"
HEADER = "Gen ID Origin Composition Enthalpy(eV) Volume(A^3) SYMM\n"


@pytest.fixture
def search(tmp_path) -> Path:
    target = tmp_path / "search"
    shutil.copytree(FIXTURE, target)
    return target


def _single_folder(search: Path) -> Path:
    shutil.rmtree(search / "results2")
    return search


# --- Folder discovery --------------------------------------------------------


def test_results_folders_are_ordered_numerically(tmp_path):
    for name in ("results10", "results2", "results1", "results_old", "notes"):
        (tmp_path / name).mkdir()
    (tmp_path / "results3").write_text("a file, not a folder")
    assert find_results_folders(str(tmp_path)) == ["results1", "results2", "results10"]


def test_missing_calculation_directory_is_an_error(tmp_path):
    with pytest.raises(USPEXResultsError, match="not found"):
        summarise_search(str(tmp_path / "absent"))


def test_directory_without_results_folders_gives_empty_summary(tmp_path):
    summary = summarise_search(str(tmp_path))
    assert summary["results_folders"] == []
    assert summary["candidates"] == []
    assert summary["generations"] == []
    assert any("no resultsN folders" in warning for warning in summary["warnings"])


# --- Summary -----------------------------------------------------------------


def test_single_results_folder(search):
    summary = summarise_search(str(_single_folder(search)))

    assert summary["results_folders"] == ["results1"]
    assert summary["atom_types"] == ["Mg", "O"]
    assert summary["n_individuals"] == 7
    assert summary["n_individuals_without_enthalpy"] == 2
    assert summary["n_candidates"] == 5
    assert summary["warnings"] == []

    ranked_ids = [candidate["id"] for candidate in summary["candidates"]]
    assert ranked_ids == [6, 1, 7, 2, 3]
    best = summary["candidates"][0]
    assert best == {
        "id": 6,
        "results_folder": "results1",
        "generation": 3,
        "origin": "Heredity",
        "composition": [4, 4],
        "formula": "Mg4O4",
        "natoms": 8,
        "enthalpy_ev": -46.736,
        "enthalpy_ev_per_atom": -5.842,
        "volume_a3": 75.151,
        "volume_per_atom_a3": pytest.approx(75.151 / 8),
        "space_group": 225,
        "rank": 1,
        "in_good_structures": True,
    }
    assert summary["candidates"][2]["in_good_structures"] is False


def test_several_results_folders(search):
    summary = summarise_search(str(search))

    assert summary["results_folders"] == ["results1", "results2"]
    assert summary["n_individuals"] == 10
    assert [c["id"] for c in summary["candidates"][:3]] == [8, 10, 6]
    assert summary["candidates"][0]["results_folder"] == "results2"
    assert [(g["results_folder"], g["generation"], g["best_id"]) for g in summary["generations"]] == [
        ("results1", 1, 1),
        ("results1", 2, None),
        ("results1", 3, 6),
        ("results2", 4, 8),
        ("results2", 5, 10),
    ]
    assert [(row["results_folder"], row["id"]) for row in summary["best_individuals"]] == [
        ("results1", 1),
        ("results1", 6),
        ("results2", 8),
        ("results2", 10),
    ]


def test_summary_is_strict_json(search):
    summary = summarise_search(str(search))
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_max_candidates_limits_list_but_not_total(search):
    summary = summarise_search(str(search), max_candidates=2)
    assert len(summary["candidates"]) == 2
    assert summary["n_candidates"] == 8


def test_empty_generation(search):
    """Generation 2 has only failed individuals; a gap generation has no rows."""

    folder = _single_folder(search) / "results1"
    lines = (folder / "Individuals").read_text().splitlines(keepends=True)
    # Renumber generation 3 to 4, leaving generation 3 with no rows at all.
    lines = [line.replace("   3    ", "   4    ", 1) for line in lines]
    (folder / "Individuals").write_text("".join(lines))

    generations = {g["generation"]: g for g in summarise_search(str(search))["generations"]}
    assert generations[2]["n_individuals"] == 2
    assert generations[2]["n_with_enthalpy"] == 0
    assert generations[2]["best_enthalpy_ev_per_atom"] is None
    assert generations[3]["n_individuals"] == 0
    assert generations[3]["best_id"] is None
    assert generations[4]["best_id"] == 6


def test_truncated_last_line_is_ignored(search):
    individuals = search / "results2" / "Individuals"
    text = individuals.read_text()
    individuals.write_text(text + "   5   11  Mutation [4 4]   -46.9")

    summary = summarise_search(str(search))

    assert 11 not in [c["id"] for c in summary["candidates"]]
    assert summary["n_individuals"] == 10
    assert any("last line has no newline" in w for w in summary["warnings"])


def test_file_with_only_partial_header_is_treated_as_empty(search):
    (search / "results2" / "Individuals").write_text("Gen ID Ori")
    summary = summarise_search(str(search))
    assert summary["results_folders"] == ["results1", "results2"]
    assert summary["n_individuals"] == 7


def test_unreadable_rows_are_skipped_with_a_warning(search):
    individuals = search / "results1" / "Individuals"
    individuals.write_text(
        individuals.read_text()
        + "   3   99  Mutation [4 4]   -46.0\n"  # too few fields
        + "   3   98  Mutation [x y]   -46.0  70.0  225\n"  # bad composition
    )
    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 10
    assert any("skipped 2 unreadable row(s)" in w for w in summary["warnings"])


def test_duplicate_id_in_one_folder_keeps_later_row(search):
    individuals = search / "results1" / "Individuals"
    individuals.write_text(
        individuals.read_text() + "   3    7  Mutation [2 2]   -23.5000    46.194  186\n"
    )
    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 10
    row = next(c for c in summary["candidates"] if c["id"] == 7)
    assert row["enthalpy_ev"] == -23.5


def test_missing_individuals_file_is_a_warning(search):
    (search / "results2" / "Individuals").unlink()
    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 7
    assert any("results2/Individuals not found" in w for w in summary["warnings"])


def test_missing_atom_types_keeps_counts(search):
    for folder in ("results1", "results2"):
        (search / folder / "Parameters.txt").unlink()
    summary = summarise_search(str(search))
    assert summary["atom_types"] is None
    assert summary["candidates"][0]["formula"] is None
    assert summary["candidates"][0]["composition"] == [4, 4]


def test_atom_types_from_atomic_numbers(search):
    for folder in ("results1", "results2"):
        (search / folder / "Parameters.txt").write_text(
            "% atomType\n12 8\n% EndAtomType\n"
        )
    assert summarise_search(str(search))["atom_types"] == ["Mg", "O"]


# --- Header handling ---------------------------------------------------------


def _replace_header(search: Path, header: str) -> None:
    individuals = search / "results1" / "Individuals"
    lines = individuals.read_text().splitlines(keepends=True)
    individuals.write_text(header + "".join(lines[1:]))


def test_columns_are_found_by_name_not_position(search):
    individuals = search / "results1" / "Individuals"
    rows = individuals.read_text().splitlines()[1:]
    reordered = ["SYMM Volume(A^3) Enthalpy(eV) Composition Origin ID Gen"]
    for row in rows:
        gen, ident, origin, comp, enthalpy, volume, symm = (
            uspex_results._ROW_TOKEN_PATTERN.findall(row)
        )
        reordered.append(" ".join([symm, volume, enthalpy, comp, origin, ident, gen]))
    individuals.write_text("\n".join(reordered) + "\n")

    best = summarise_search(str(_single_folder(search)))["candidates"][0]
    assert (best["id"], best["generation"], best["space_group"]) == (6, 3, 225)
    assert best["enthalpy_ev_per_atom"] == -5.842


def test_per_atom_enthalpy_column(search):
    individuals = search / "results1" / "Individuals"
    individuals.write_text(
        "Gen ID Composition Enthalpy(eV/atom)\n"
        "1 1 [4 4] -5.81\n"
        "1 2 [2 2] -5.90\n"
    )
    summary = summarise_search(str(_single_folder(search)))
    best = summary["candidates"][0]
    assert best["id"] == 2
    assert best["enthalpy_ev"] == pytest.approx(-23.6)
    assert best["volume_a3"] is None
    assert best["origin"] is None


def test_unrecognised_header_names_the_column(search):
    _replace_header(search, "Gen ID Origin Composition Enthalpy(eV) Volume(A^3) Fitness\n")
    with pytest.raises(USPEXFormatError, match=r"unrecognised column header\(s\) \['Fitness'\]"):
        summarise_search(str(search))


def test_bare_enthalpy_header_is_not_guessed(search):
    _replace_header(search, "Gen ID Origin Composition Enthalpy Volume(A^3) SYMM\n")
    with pytest.raises(USPEXFormatError, match="does not give its unit"):
        summarise_search(str(search))


def test_missing_required_column(search):
    _replace_header(search, "Gen Origin Composition Enthalpy(eV) Volume(A^3) SYMM\n")
    with pytest.raises(USPEXFormatError, match="missing required column"):
        summarise_search(str(search))


def test_duplicate_column(search):
    _replace_header(search, "Gen ID Origin Composition Enthalpy(eV) Volume(A^3) SYMM Generation\n")
    with pytest.raises(USPEXFormatError, match="more than one column for 'generation'"):
        summarise_search(str(search))


# --- Structures --------------------------------------------------------------


def test_load_structures_from_gathered_poscars(search):
    structures = load_structures(str(search), [8, 2])

    assert set(structures) == {2, 8}
    assert structures[8].composition.reduced_formula == "MgO"
    assert len(structures[8]) == 8
    assert structures[8].volume == pytest.approx(74.618, abs=1e-3)
    assert len(structures[2]) == 4


def test_load_structures_from_good_structures(search):
    structures = load_structures(str(search), [6], poscars_file="goodStructures_POSCARS")
    assert structures[6].lattice.a == pytest.approx(4.22)
    with pytest.raises(USPEXResultsError, match=r"\[2\] not found"):
        load_structures(str(search), [2], poscars_file="goodStructures_POSCARS")


def test_load_structures_rejects_unknown_file(search):
    with pytest.raises(USPEXResultsError, match="poscars_file must be one of"):
        load_structures(str(search), [1], poscars_file="Individuals")


def test_id_in_two_folders_needs_results_folder(search):
    shutil.copy(search / "results1" / "gatheredPOSCARS", search / "results2" / "gatheredPOSCARS")
    with pytest.raises(USPEXResultsError, match="more than one results folder"):
        load_structures(str(search), [1])
    assert 1 in load_structures(str(search), [1], results_folder="results2")
    with pytest.raises(USPEXResultsError, match="results7 not found"):
        load_structures(str(search), [1], results_folder="results7")


def test_incomplete_last_poscar_block_is_ignored(search):
    gathered = search / "results2" / "gatheredPOSCARS"
    complete = gathered.read_text()
    first_block = "".join(complete.splitlines(keepends=True)[:12])
    gathered.write_text(complete + first_block.replace("EA8 ", "EA11 "))

    assert set(load_structures(str(search), [10])) == {10}
    with pytest.raises(USPEXResultsError, match=r"\[11\] not found.*incomplete"):
        load_structures(str(search), [11])


def test_poscar_block_without_element_line_uses_atom_types(search):
    good = search / "results1" / "goodStructures_POSCARS"
    lines = good.read_text().splitlines(keepends=True)
    assert lines[5].split() == ["Mg", "O"]
    good.write_text("".join(lines[:5] + lines[6:]))

    structure = load_structures(str(search), [1], poscars_file="goodStructures_POSCARS")[1]
    assert structure.composition.reduced_formula == "MgO"

    for folder in ("results1", "results2"):
        (search / folder / "Parameters.txt").unlink()
    with pytest.raises(USPEXResultsError, match="no element line"):
        load_structures(str(search), [1], poscars_file="goodStructures_POSCARS")


def test_poscar_id_pattern_is_read_from_comment_line():
    assert uspex_results.POSCAR_ID_PATTERN.search("EA17 -5.85 74.6 Sym.group: 225").group(1) == "17"
    assert uspex_results.POSCAR_ID_PATTERN.search("REA17") is None


def test_unmatched_comment_line_is_reported(search):
    gathered = search / "results2" / "gatheredPOSCARS"
    gathered.write_text(gathered.read_text().replace("EA9 ", "structure nine "))
    summary_warnings = []
    blocks = uspex_results._scan_poscars_file(str(gathered), summary_warnings)
    assert [block.id for block in blocks] == [8, None, 10]
    assert "POSCAR_ID_PATTERN" in summary_warnings[0]


# --- Module boundary ---------------------------------------------------------


def test_module_is_self_contained():
    """
    The parser must import only the standard library at module level and
    nothing from the backend package, so it can later be sent to POWER and
    run beside the results folders, like backend/remote_result_parser.py.
    """

    source = Path(uspex_results.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert top_level <= {"__future__", "math", "os", "re", "dataclasses", "typing"}

    every_import = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            every_import.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            every_import.add((node.module or "").split(".")[0])
    assert every_import - top_level == {"pymatgen"}


def test_package_init_is_one_line():
    init = Path(uspex_results.__file__).with_name("__init__.py")
    assert init.read_text(encoding="utf-8") == (
        '"""USPEX structure search (see docs/structure_search.md)."""\n'
    )
