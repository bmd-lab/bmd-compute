"""
Tests for backend/structure_search/results.py (issue #62).

The fixture is real USPEX 25.0 output from the Phase 0 Si8 pilot (#59), see
tests/fixtures/uspex_si8_pilot/README.md. Cases that need damaged, growing or
restarted searches copy it to a temporary directory and modify the copy.
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
    structure_symmetry,
    summarise_search,
)


PILOT = Path(__file__).parent / "fixtures" / "uspex_si8_pilot"
GENERATION_BEST_ENERGIES = [-40.903, -43.277, -43.278, -43.280]


@pytest.fixture
def search(tmp_path) -> Path:
    target = tmp_path / "search"
    shutil.copytree(PILOT, target)
    return target


def _individuals_lines(search: Path) -> list[str]:
    return (search / "results1" / "Individuals").read_text().split("\n")


def _write_individuals(search: Path, lines: list[str]) -> None:
    # Written as USPEX does: no newline after the last row.
    (search / "results1" / "Individuals").write_text("\n".join(lines))


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


# --- Summary of the real pilot output ---------------------------------------


def test_pilot_summary_counts():
    summary = summarise_search(str(PILOT))

    assert summary["results_folders"] == ["results1"]
    assert summary["atom_types"] == ["Si"]
    assert summary["n_individuals"] == 35
    assert summary["n_individuals_without_energy"] == 0
    assert summary["warnings"] == []
    assert [g["generation"] for g in summary["generations"]] == [1, 2, 3, 4]
    assert [g["n_individuals"] for g in summary["generations"]] == [8, 9, 9, 9]


def test_pilot_best_candidate():
    best = summarise_search(str(PILOT))["candidates"][0]

    # Individuals gives energies to 3 decimals; good_structures gives
    # -43.2799 eV and 163.6749 A^3 for the same individual.
    assert best["id"] == 35
    assert best["rank"] == 1
    assert best["energy_ev"] == pytest.approx(-43.2799, abs=5e-4)
    assert best["energy_ev_per_atom"] == pytest.approx(-43.2799 / 8, abs=1e-4)
    assert best["volume_a3"] == pytest.approx(163.6749, abs=5e-4)
    assert best["volume_per_atom_a3"] == pytest.approx(163.6749 / 8, abs=1e-4)
    assert best["formula"] == "Si8"
    assert best["composition"] == [8]
    assert best["natoms"] == 8
    assert best["generation"] == 4
    assert best["origin"] == "keptBest"
    assert best["parents"] == [26]
    assert best["space_group_uspex"] == 1
    assert best["results_folder"] == "results1"
    assert best["in_good_structures"] is True


def test_pilot_best_energy_per_generation():
    generations = summarise_search(str(PILOT))["generations"]
    assert [g["best_energy_ev"] for g in generations] == pytest.approx(
        GENERATION_BEST_ENERGIES
    )
    # Recalculated survivors count in the per-generation data.
    assert [g["best_id"] for g in generations] == [8, 12, 26, 35]


def test_kept_best_chains_are_listed_once():
    summary = summarise_search(str(PILOT), max_candidates=100)
    by_id = {c["id"]: c for c in summary["candidates"]}

    # 35 individuals; 17, 26 and 35 are keptBest recalculations of 8, 12, 26.
    assert summary["n_candidates"] == 32
    assert by_id[35]["lineage_ids"] == [12, 26, 35]
    assert 12 not in by_id and 26 not in by_id
    # 17 (recalculation of 8) ended 0.32 eV higher, so 8 represents the chain.
    assert by_id[8]["lineage_ids"] == [8, 17]
    assert 17 not in by_id
    assert by_id[21]["lineage_ids"] == [21]


def test_ranking_matches_uspex_good_structures_order():
    ranked = [c["id"] for c in summarise_search(str(PILOT))["candidates"][:7]]
    assert ranked == [35, 21, 16, 19, 32, 20, 18]


def test_best_individuals_rows():
    rows = summarise_search(str(PILOT))["best_individuals"]
    assert [(row["generation"], row["id"]) for row in rows] == [(1, 8), (2, 12), (3, 26), (4, 35)]
    assert rows[-1]["volume_a3"] is None  # BESTIndividuals has no volume column


def test_summary_is_strict_json():
    summary = summarise_search(str(PILOT))
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_max_candidates_limits_list_but_not_total():
    summary = summarise_search(str(PILOT), max_candidates=2)
    assert len(summary["candidates"]) == 2
    assert summary["n_candidates"] == 32


# --- Growing and damaged files ----------------------------------------------


def test_real_individuals_has_no_final_newline_and_last_row_is_kept():
    assert not (PILOT / "results1" / "Individuals").read_bytes().endswith(b"\n")
    assert summarise_search(str(PILOT))["candidates"][0]["id"] == 35


def test_last_line_cut_mid_row_is_ignored(search):
    lines = _individuals_lines(search)
    lines[-1] = lines[-1][:60]
    _write_individuals(search, lines)

    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 34
    assert summary["candidates"][0]["id"] == 26
    assert any("looks unfinished" in w for w in summary["warnings"])


def test_last_line_cut_in_last_column_is_ignored(search):
    lines = _individuals_lines(search)
    assert lines[-1].endswith("4.111")
    lines[-1] = lines[-1][:-2]
    _write_individuals(search, lines)

    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 34
    assert any("looks unfinished" in w for w in summary["warnings"])


def test_file_with_only_partial_header_is_treated_as_empty(search):
    (search / "results1" / "Individuals").write_text("generation   number   num_at")
    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 0
    assert summary["generations"] == []


def test_unreadable_rows_are_skipped_with_a_warning(search):
    lines = _individuals_lines(search)
    lines.insert(5, "         1       99             [8]   -40.375")  # too few fields
    lines.insert(6, lines[1].replace("[8]", "[x]"))  # bad composition
    _write_individuals(search, lines)

    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 35
    assert any("skipped 2 unreadable row(s) at line(s) 6, 7" in w for w in summary["warnings"])


def test_empty_generation(search):
    """Generation 3 removed entirely: it is still listed, with no best."""

    lines = _individuals_lines(search)
    _write_individuals(search, [line for line in lines if not line.startswith("         3 ")])

    summary = summarise_search(str(search))
    generations = {g["generation"]: g for g in summary["generations"]}
    assert generations[3] == {
        "results_folder": "results1",
        "generation": 3,
        "n_individuals": 0,
        "n_with_energy": 0,
        "best_id": None,
        "best_energy_ev": None,
        "best_energy_ev_per_atom": None,
    }
    assert generations[4]["best_id"] == 35
    # With 26 gone, 35 no longer joins 12's chain.
    by_id = {c["id"]: c for c in summary["candidates"]}
    assert by_id[35]["lineage_ids"] == [35]
    assert by_id[12]["lineage_ids"] == [12]


def test_generation_without_energies(search):
    """
    How USPEX marks a failed individual has not been observed; this uses NaN,
    one of MISSING_VALUE_TOKENS, to check the handling.
    """

    lines = _individuals_lines(search)
    edited = []
    for line in lines:
        if line.startswith("         1 "):
            tokens = line.split()
            line = line.replace(f" {tokens[3]} ", f" {'NaN':>{len(tokens[3])}} ", 1)
        edited.append(line)
    _write_individuals(search, edited)

    summary = summarise_search(str(search))
    first = summary["generations"][0]
    assert (first["n_individuals"], first["n_with_energy"], first["best_id"]) == (8, 0, None)
    assert summary["n_individuals_without_energy"] == 8
    # 8 has no energy, so its chain is represented by its recalculation 17.
    by_id = {c["id"]: c for c in summary["candidates"]}
    assert by_id[17]["lineage_ids"] == [8, 17]


def test_duplicate_id_in_one_folder_keeps_later_row(search):
    lines = _individuals_lines(search)
    lines.append(lines[21].replace("-41.398", "-41.500"))
    _write_individuals(search, lines)

    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 35
    assert next(c for c in summary["candidates"] if c["id"] == 21)["energy_ev"] == -41.5


def test_missing_individuals_file_is_a_warning(search):
    (search / "results1" / "Individuals").unlink()
    summary = summarise_search(str(search))
    assert summary["n_individuals"] == 0
    assert any("results1/Individuals not found" in w for w in summary["warnings"])


def test_missing_atom_types_keeps_counts(search):
    (search / "results1" / "Parameters.txt").unlink()
    summary = summarise_search(str(search))
    assert summary["atom_types"] is None
    assert summary["candidates"][0]["formula"] is None
    assert summary["candidates"][0]["composition"] == [8]


def test_atom_types_from_atomic_numbers(search):
    (search / "results1" / "Parameters.txt").write_text("% atomType\n14\n% EndAtomType\n")
    assert summarise_search(str(search))["atom_types"] == ["Si"]


# --- Several results folders -------------------------------------------------


def test_several_results_folders(search):
    shutil.copytree(search / "results1", search / "results2")
    shutil.copytree(search / "results1", search / "results10")

    summary = summarise_search(str(search), max_candidates=100)
    assert summary["results_folders"] == ["results1", "results2", "results10"]
    # Unique by (folder, ID): nothing is merged across folders.
    assert summary["n_individuals"] == 105
    assert summary["n_candidates"] == 96
    assert [(c["results_folder"], c["id"]) for c in summary["candidates"][:3]] == [
        ("results1", 35),
        ("results2", 35),
        ("results10", 35),
    ]
    assert [g["results_folder"] for g in summary["generations"]] == (
        ["results1"] * 4 + ["results2"] * 4 + ["results10"] * 4
    )
    assert len(summary["best_individuals"]) == 12


def test_id_in_two_folders_needs_results_folder(search):
    shutil.copytree(search / "results1", search / "results2")
    with pytest.raises(USPEXResultsError, match="more than one results folder"):
        load_structures(str(search), [35])
    assert 35 in load_structures(str(search), [35], results_folder="results2")
    with pytest.raises(USPEXResultsError, match="results7 not found"):
        load_structures(str(search), [35], results_folder="results7")


# --- Header handling ---------------------------------------------------------


def _replace_header(search: Path, header: str) -> None:
    lines = _individuals_lines(search)
    _write_individuals(search, [header] + lines[1:])


def test_columns_are_found_by_name_not_position(search):
    lines = _individuals_lines(search)
    reordered = []
    for line in lines:
        tokens = uspex_results._ROW_TOKEN_PATTERN.findall(line)
        reordered.append(" ".join(tokens[::-1]))
    (search / "results1" / "Individuals").write_text("\n".join(reordered) + "\n")

    best = summarise_search(str(search))["candidates"][0]
    assert (best["id"], best["generation"], best["space_group_uspex"]) == (35, 4, 1)
    assert best["energy_ev"] == -43.28


def test_unrecognised_header_names_the_column(search):
    header = _individuals_lines(search)[0].replace("quasi_entropy", "fitness")
    _replace_header(search, header)
    with pytest.raises(USPEXFormatError, match=r"unrecognised column header\(s\) \['fitness'\]"):
        summarise_search(str(search))


def test_missing_required_column(search):
    header = _individuals_lines(search)[0].replace("   energy", "   energy_x")
    _replace_header(search, header)
    with pytest.raises(USPEXFormatError, match="unrecognised column header"):
        summarise_search(str(search))
    header = _individuals_lines(search)[0].replace(" energy_x", " a_order ")
    _replace_header(search, header)
    with pytest.raises(USPEXFormatError, match=r"missing required column\(s\) \['energy_ev'\]"):
        summarise_search(str(search))


def test_duplicate_column(search):
    header = _individuals_lines(search)[0].replace("s_order", "generation")
    _replace_header(search, header)
    with pytest.raises(USPEXFormatError, match="more than one column for 'generation'"):
        summarise_search(str(search))


# --- Structures --------------------------------------------------------------


def _ideal_diamond_si():
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.47), ["Si"], [[0, 0, 0]])


def test_best_structure_is_diamond_si():
    from pymatgen.analysis.structure_matcher import StructureMatcher

    structure = load_structures(str(PILOT), [35])[35]

    assert len(structure) == 8
    assert structure.composition.reduced_formula == "Si"
    assert structure.volume == pytest.approx(163.6749, abs=1e-3)
    assert StructureMatcher().fit(structure, _ideal_diamond_si())


def test_recomputed_symmetry_of_best_structure():
    structure = load_structures(str(PILOT), [35])[35]
    assert structure_symmetry(structure) == {
        "space_group_number": 227,
        "space_group_symbol": "Fd-3m",
        "symprec": 0.1,
    }
    # The tolerance matters: USPEX reports space group 1 for this structure.
    assert structure_symmetry(structure, symprec=1e-3)["space_group_number"] != 227


@pytest.mark.parametrize(
    ("poscars_file", "ids"),
    [
        ("gatheredPOSCARS", list(range(1, 36))),
        ("good_structures_POSCARS", [35, 21, 8, 17]),
        ("BESTgatheredPOSCARS", [8, 12, 26, 35]),
        ("best_structure_POSCAR", [35]),
    ],
)
def test_load_structures_from_each_poscars_file(poscars_file, ids):
    structures = load_structures(str(PILOT), ids, poscars_file=poscars_file)
    assert sorted(structures) == sorted(ids)
    assert all(len(structure) == 8 for structure in structures.values())


def test_structure_not_in_file():
    with pytest.raises(USPEXResultsError, match=r"\[12\] not found in good_structures_POSCARS"):
        load_structures(str(PILOT), [12], poscars_file="good_structures_POSCARS")


def test_load_structures_rejects_unknown_file():
    with pytest.raises(USPEXResultsError, match="poscars_file must be one of"):
        load_structures(str(PILOT), [1], poscars_file="Individuals")


def test_incomplete_last_poscar_block_is_ignored(search):
    gathered = search / "results1" / "gatheredPOSCARS"
    complete = gathered.read_text()
    first_block = "".join(complete.splitlines(keepends=True)[:12])
    gathered.write_text(complete + first_block.replace("number=1,", "number=36,"))

    assert set(load_structures(str(search), [35])) == {35}
    with pytest.raises(USPEXResultsError, match=r"\[36\] not found.*incomplete"):
        load_structures(str(search), [36])


def test_poscar_line_without_newline_is_ignored(search):
    gathered = search / "results1" / "gatheredPOSCARS"
    gathered.write_text(gathered.read_text().rstrip("\n"))
    with pytest.raises(USPEXResultsError, match=r"\[35\] not found.*no newline"):
        load_structures(str(search), [35])
    assert 34 in load_structures(str(search), [34])


def test_poscar_block_without_element_line_uses_atom_types(search):
    best = search / "results1" / "best_structure_POSCAR"
    lines = best.read_text().splitlines(keepends=True)
    assert lines[5].split() == ["Si"]
    best.write_text("".join(lines[:5] + lines[6:]))

    structure = load_structures(str(search), [35], poscars_file="best_structure_POSCAR")[35]
    assert structure.composition.reduced_formula == "Si"

    (search / "results1" / "Parameters.txt").unlink()
    with pytest.raises(USPEXResultsError, match="no element line"):
        load_structures(str(search), [35], poscars_file="best_structure_POSCAR")


def test_poscar_id_pattern_reads_real_comment_lines():
    pattern = uspex_results.POSCAR_ID_PATTERN
    assert pattern.search("number=35, energy=-43.279875, e_above_hull=0.0").group(1) == "35"
    assert pattern.search("Gen=4, generation=4, number=35, energy=-43.279875").group(1) == "35"
    assert pattern.search("EA35 -43.28") is None


def test_unmatched_comment_line_is_reported(search):
    gathered = search / "results1" / "gatheredPOSCARS"
    gathered.write_text(gathered.read_text().replace("number=9,", "individual nine,"))
    warnings: list[str] = []
    blocks = uspex_results._scan_poscars_file(str(gathered), warnings)
    assert [block.id for block in blocks][7:10] == [8, None, 10]
    assert "POSCAR_ID_PATTERN" in warnings[0]


# --- Module boundary ---------------------------------------------------------


def test_module_is_self_contained():
    """
    The parser must import only the standard library at module level and
    nothing from the backend package, so it can later be sent to POWER and
    run beside the results folders, like backend/remote_result_parser.py.
    """

    tree = ast.parse(Path(uspex_results.__file__).read_text(encoding="utf-8"))

    def imported(nodes):
        names = set()
        for node in nodes:
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        return names

    top_level = imported(tree.body)
    assert top_level <= {"__future__", "math", "os", "re", "dataclasses", "typing"}
    assert imported(ast.walk(tree)) - top_level == {"pymatgen"}


def test_package_init_is_one_line():
    init = Path(uspex_results.__file__).with_name("__init__.py")
    assert init.read_text(encoding="utf-8") == (
        '"""USPEX structure search (see docs/structure_search.md)."""\n'
    )
