"""Corrections to the scientific execution acceptance after attempt 209898bc (job 23624369, 14 Nov 2026).

A1: between intermediate ladder steps USPEX changes the structure (strained cells, moved atoms, shifts,
other lattice bases). Those changes are measured and reported, not failed and not judged; the final step
(4 -> 5) must still start from exactly its predecessor's output, now independent of the lattice basis it
is written in. Candidate identity and composition are still required between every pair of steps.
A2: a recorded atomate2 import (the documented POTCAR-table import) no longer fails the run; any other
recorded workflow machinery (jobflow, or a package the analyser does not know) still does.
A3: VASP steps are described by their own MPI ranks, apart from the job's CPU allocation.
C1: the collector counts Intel MPI startup, version and SLURM cpu-bind lines and keeps a few short sample
lines, so a pinning report that is absent can be told apart from one in an unrecognised format.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from test_uspex_execution_acceptance import (ATTEMPT, JOB, RUN_NAME, SHARED_EXPECTED, analyse, archive_check,
                                             check_detail, expected, load_analyser, make, status_of)
from uspex_execution_fakes import SHARED_48, Scenario, rewrite_poscar, run_collector

ANALYSER = load_analyser()
HEX = b"""x
1.0
3.9 0.0 0.0
-1.95 3.3775 0.0
0.0 0.0 3.2
Si
2
Direct
0.0 0.0 0.0
0.3333333 0.6666667 0.5
"""
OTHER_BASIS = ((1, 1, 0), (0, 1, 0), (0, 0, -1))
USPEX_LIKE = {"basis": OTHER_BASIS, "strain": (0.02, -0.01, 0.0), "move_first_atom_a": 0.05,
              "shift": (0.01, 0.0, 0.02)}


def transformation(before: bytes, after: bytes) -> dict:
    return ANALYSER.structure_transformation(ANALYSER.parse_poscar(before.decode()),
                                             ANALYSER.parse_poscar(after.decode()))


# --- the measurement itself ----------------------------------------------------------------------------------


def test_an_unchanged_structure_measures_zero():
    t = transformation(HEX, HEX)
    assert t["comparable"] and t["max_abs_principal_strain"] < 1e-9 and t["max_atomic_displacement_a"] < 1e-6
    assert t["rigid_translation_a"] < 1e-6 and t["volume_change_relative"] == 0


def test_the_same_cell_in_another_lattice_basis_measures_zero():
    t = transformation(HEX, rewrite_poscar(HEX, basis=OTHER_BASIS))
    assert t["comparable"] and t["max_abs_principal_strain"] < 1e-8 and t["max_atomic_displacement_a"] < 1e-6
    assert abs(round(np.linalg.det(np.array(t["lattice_correspondence"])))) == 1
    assert ANALYSER.same_structure(ANALYSER.parse_poscar(HEX.decode()),
                                   ANALYSER.parse_poscar(rewrite_poscar(HEX, basis=OTHER_BASIS).decode())) is False


def test_a_strain_is_measured_as_its_green_lagrange_strain_whatever_the_basis():
    t = transformation(HEX, rewrite_poscar(HEX, basis=OTHER_BASIS, strain=(0.02, 0.0, 0.0)))
    assert t["max_abs_principal_strain"] == pytest.approx(0.02 + 0.02 ** 2 / 2, abs=1e-6)  # HEX is hexagonal to 7 digits
    assert t["volume_change_relative"] == pytest.approx(0.02, abs=1e-7)
    assert t["max_atomic_displacement_a"] < 1e-6  # a uniform strain alone moves no atom


def test_moved_atoms_and_a_rigid_shift_are_reported_separately():
    t = transformation(HEX, rewrite_poscar(HEX, move_first_atom_a=0.05, shift=(0.1, 0.0, 0.0)))
    # Two atoms: moving one by 0.05 A is a 0.025 A move of each about their centre, plus a 0.025 A translation.
    assert t["max_atomic_displacement_a"] == pytest.approx(0.025, abs=1e-6)
    assert t["rigid_translation_a"] == pytest.approx(0.1 * 3.9 + 0.025, abs=1e-5)
    assert t["max_abs_principal_strain"] < 1e-8


def test_random_cells_bases_and_strains_match_an_independent_numpy_calculation():
    rng = random.Random(7)
    bases = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), OTHER_BASIS, ((0, 1, 0), (1, 0, 0), (1, 1, 1)),
             ((1, -1, 0), (0, 1, 0), (0, 1, 1))]
    for _ in range(12):
        lattice = np.diag([rng.uniform(2.5, 4.5) for _ in range(3)]) + np.triu(
            [[0, rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)], [0, 0, rng.uniform(-0.6, 0.6)], [0, 0, 0]])
        text = ("x\n1.0\n" + "\n".join(" ".join(f"{v:.10f}" for v in row) for row in lattice)
                + "\nSi\n2\nDirect\n0.1 0.2 0.3\n0.6 0.55 0.8\n").encode()
        strain = tuple(rng.uniform(-0.03, 0.03) for _ in range(3))
        t = transformation(text, rewrite_poscar(text, basis=rng.choice(bases), strain=strain))
        deform = np.diag([1 + e for e in strain])
        # The cell's own deformation in its Cartesian frame: C = L^-1 (L F^T)(L F^T)^T L^-T.
        lf = lattice @ deform.T
        c = np.linalg.inv(lattice) @ lf @ lf.T @ np.linalg.inv(lattice).T
        reference = 0.5 * (np.linalg.eigvalsh(c) - 1)
        assert t["comparable"]
        assert t["max_abs_principal_strain"] == pytest.approx(np.abs(reference).max(), abs=1e-7)
        assert t["max_atomic_displacement_a"] < 1e-6


def test_another_composition_or_an_unreadable_structure_is_not_comparable():
    other = transformation(HEX, rewrite_poscar(HEX, species=["Ge"]))
    assert other["comparable"] is False and other["composition_same"] is False
    assert ANALYSER.structure_transformation(None, ANALYSER.parse_poscar(HEX.decode()))["comparable"] is False


# --- A1 in the analysis ------------------------------------------------------------------------------------------


def intermediate_changes(individuals=(4, 9, 15)):
    return {(i, step): USPEX_LIKE for i in individuals for step in (2, 3, 4)}


def test_uspex_changes_between_intermediate_steps_are_measured_and_reported_not_failed(tmp_path):
    report = analyse(tmp_path, Scenario(input_changes=intermediate_changes()))
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    detail = archive_check(report)["detail"]
    assert detail["intermediate_transitions"] == 63 and detail["intermediate_transitions_changed"] == 9
    assert detail["final_step_continuity_breaks_total"] == 0 and detail["contradicted_steps_total"] == 0
    moved = report["findings"]["intermediate_transformations"]
    assert moved["changed"] == 9 and moved["not_quantified"] == 0 and moved["composition_changed"] == 0
    assert "not assessed as physically negligible and are not validated" in moved["note"]
    for pair in ("2->3", "3->4"):  # strained and moved on top of the predecessor's own output
        row = moved["by_transition"][pair]
        assert row["changed"] == 3 and row["max_abs_principal_strain"]["max"] == pytest.approx(0.0202, abs=1e-4)
        assert row["max_atomic_displacement_a"]["max"] == pytest.approx(0.025, abs=2e-3)
    changed_rows = [r for r in moved["rows"] if not r["identical_bytes"]]
    assert {r["individual"] for r in changed_rows} == {4, 9, 15}
    assert all(r["composition"] == {"species": ["Si"], "counts": [2]} for r in changed_rows)
    assert all(r["before"] and r["after"] and r["steps"][1] == r["steps"][0] + 1 for r in changed_rows)
    line = next(line for line in ANALYSER.summary_lines(report)
                if line.startswith("FINDING") and "intermediate steps" in line)
    assert "9 of 63" in line and "not judged negligible or validated" in line


def test_an_intermediate_input_whose_change_cannot_be_quantified_is_missing_evidence(tmp_path):
    # A cell tens of Å long against one of 3 Å: no correspondence within the search, so the change is not
    # quantified. That is reported and keeps the result from passing; it is not a scientific rejection.
    report = analyse(tmp_path, Scenario(continuity_break=((6, 3),)))
    assert report["result"] == "INCOMPLETE EVIDENCE" and report["missing"] == ["evidence_archive"]
    detail = archive_check(report)["detail"]
    assert detail["intermediate_transitions_not_quantified_total"] == 1
    assert detail["intermediate_transitions_not_quantified"][0][2] == "no lattice correspondence found"
    assert report["findings"]["intermediate_transformations"]["not_quantified"] == 1


@pytest.mark.parametrize("change, fragment", [
    ({"strain": (0.001, 0.0, 0.0)}, "the final step did not start from its predecessor's output"),
    ({"move_first_atom_a": 0.01}, "the final step did not start from its predecessor's output"),
    ({"shift": (0.01, 0.0, 0.0)}, "the final step did not start from its predecessor's output"),
    ({"species": ["Ge"]}, "the final step did not start from its predecessor's output"),
])
def test_the_final_step_must_still_start_from_exactly_its_predecessors_output(tmp_path, change, fragment):
    report = analyse(tmp_path, Scenario(input_changes={(7, 5): change}))
    assert report["result"] == "FAILED" and "evidence_archive" in report["failed"]
    detail = archive_check(report)["detail"]
    assert detail["final_step_continuity_breaks_total"] == 1
    assert any(fragment in p for ps in detail["contradictions"].values() for p in ps)


def test_the_final_step_input_in_another_lattice_basis_is_the_same_crystal(tmp_path):
    report = analyse(tmp_path, Scenario(input_changes={(7, 5): {"basis": OTHER_BASIS}}))
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    detail = archive_check(report)["detail"]
    assert detail["final_step_basis_rewrites_total"] == 1 and detail["final_step_continuity_breaks_total"] == 0


def test_the_old_wholesale_final_step_break_still_fails(tmp_path):
    report = analyse(tmp_path, Scenario(continuity_break=((6, 5),)))
    assert report["result"] == "FAILED" and "evidence_archive" in report["failed"]


def test_an_intermediate_change_of_composition_fails(tmp_path):
    report = analyse(tmp_path, Scenario(input_changes={(4, 3): {"species": ["Ge"]}}))
    assert report["result"] == "FAILED" and "evidence_archive" in report["failed"]
    problems = [p for ps in archive_check(report)["detail"]["contradictions"].values() for p in ps]
    assert any("different composition" in p for p in problems)
    assert report["findings"]["intermediate_transformations"]["composition_changed"] == 1


def test_the_integrity_checks_are_unchanged_alongside_measured_changes(tmp_path):
    report = analyse(tmp_path, Scenario(input_changes=intermediate_changes(), archive_tamper=((4, 3),)))
    assert report["result"] == "FAILED" and "evidence_archive" in report["failed"]
    problems = [p for ps in archive_check(report)["detail"]["contradictions"].values() for p in ps]
    assert any("reasons: archive" in p for p in problems)


def test_an_altered_intermediate_input_still_fails_its_hash_check(tmp_path):
    code, output, evidence, _ = make(tmp_path)
    assert code == 0
    record = next((evidence / "files" / "search" / "bmd_evidence" / "steps").glob("s3-0004-*"))
    (record / "POSCAR").write_text("not a structure\n")
    report = ANALYSER.analyse(evidence, expected())
    assert "evidence_archive" in report["failed"]
    problems = [p for ps in archive_check(report)["detail"]["contradictions"].values() for p in ps]
    assert any("POSCAR does not match its recorded hash" in p for p in problems)


# --- A2 ------------------------------------------------------------------------------------------------------------


def test_the_documented_atomate2_import_does_not_fail_and_is_attributed_not_proven(tmp_path):
    report = analyse(tmp_path, Scenario(isolation={"jobflow": False, "atomate2": True}))
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    detail = check_detail(report, "isolation")
    assert "potcar.py" in detail["documented_imports_recorded"]["atomate2"]
    assert detail["unexpected_workflow_machinery"] == [] and "not proven" in detail["limitation"]
    assert status_of(report, "search_status") == "verified"


@pytest.mark.parametrize("isolation, status", [
    ({"jobflow": True, "atomate2": False}, "failed"),
    ({"jobflow": True, "atomate2": True}, "failed"),
    ({"jobflow": False, "atomate2": False, "fireworks": True}, "failed"),
    ({"jobflow": False, "atomate2": "yes"}, "failed"),
    ({"atomate2": True}, "missing"),
    (None, "missing"),
])
def test_other_workflow_machinery_or_an_incomplete_record_is_never_accepted(tmp_path, isolation, status):
    report = analyse(tmp_path, Scenario(isolation=isolation))
    assert status_of(report, "isolation") == status and report["result"] != "PASSED"


# --- A3 ------------------------------------------------------------------------------------------------------------


def test_vasp_steps_are_described_by_their_own_ranks_apart_from_the_job_allocation(tmp_path):
    report = analyse(tmp_path, Scenario(**SHARED_48), SHARED_EXPECTED)
    what = {c["check"]: c["what"] for c in report["checks"]}
    assert "16 MPI ranks each" in what["vasp_steps"] and "48-CPU job allocation" in what["vasp_steps"]
    assert "16 MPI ranks each, up to 3 at once in the 48-CPU allocation" in what["pinning"]
    assert not any("48-rank" in text for text in what.values())


# --- C1 ------------------------------------------------------------------------------------------------------------


def logs_of(evidence):
    return json.loads((evidence / "step_logs.json").read_text())


def test_the_collector_counts_placement_lines_and_keeps_short_samples(tmp_path):
    code, output, evidence, _ = make(tmp_path, Scenario(**SHARED_48))
    assert code == 0
    some = next(iter(logs_of(evidence).values()))
    assert len(some["pin_rows"]) == 16 and some["mpi_startup_lines"] == 18  # version, header, 16 rows
    assert some["intel_mpi_version"].startswith("Intel(R) MPI Library, Version 2021.9")
    assert some["cpu_bind_lines"] == 0 and 0 < len(some["placement_samples"]) <= 8
    assert all(len(line) <= 200 for line in some["placement_samples"])


def test_an_unrecognised_pinning_format_is_told_apart_from_no_output(tmp_path):
    unrecognised = analyse(tmp_path / "a", Scenario(**{**SHARED_48, "pin": "unrecognised"}), SHARED_EXPECTED)
    silent = analyse(tmp_path / "b", Scenario(**{**SHARED_48, "pin": "none"}), SHARED_EXPECTED)
    version = analyse(tmp_path / "c", Scenario(**{**SHARED_48, "pin": "version_only"}), SHARED_EXPECTED)
    for report in (unrecognised, silent, version):
        assert report["missing"] == ["pinning"] and report["result"] == "INCOMPLETE EVIDENCE"
    assert (check_detail(unrecognised, "pinning")["placement_diagnostics"]["interpretation"]
            == "Intel MPI startup output not in the recognised pinning format")
    assert check_detail(unrecognised, "pinning")["placement_diagnostics"]["samples"]
    assert (check_detail(silent, "pinning")["placement_diagnostics"]["interpretation"]
            == "no Intel MPI startup output in the step logs")
    diagnostics = check_detail(version, "pinning")["placement_diagnostics"]
    assert diagnostics["steps_with_intel_mpi_version_line"] == diagnostics["steps"]
    assert diagnostics["steps_with_pinning_rows"] == 0


def test_evidence_collected_before_the_diagnostics_existed_says_so(tmp_path):
    code, output, evidence, _ = make(tmp_path, Scenario(**{**SHARED_48, "pin": "none"}))
    assert code == 0
    old = {k: {f: v for f, v in s.items() if f in ("bytes", "sha256", "ionic_steps", "final_free_energy_ev",
                                                   "final_e0_ev", "pin_rows", "error_lines")}
           for k, s in logs_of(evidence).items()}
    (evidence / "step_logs.json").write_text(json.dumps(old))
    report = ANALYSER.analyse(evidence, expected(**SHARED_EXPECTED))
    diagnostics = check_detail(report, "pinning")["placement_diagnostics"]
    assert diagnostics["collected"] is False and "cannot be told" in diagnostics["reason"]
    assert report["missing"] == ["pinning"]


def test_sample_lines_are_bounded_and_printable(tmp_path):
    code, output, evidence, run = make(tmp_path, Scenario(pin="none"))
    assert code == 0
    log = next((run / "bridge_spool" / "logs").glob("*.log"))
    log.write_bytes(b"".join(b"[0] MPI startup(): \x1b[31m" + b"X" * 500 + b"\n" for _ in range(40)) + log.read_bytes())
    sacct = (evidence / "sacct_steps.txt").read_text().splitlines()
    code, output, again = run_collector(tmp_path, attempt=ATTEMPT, run_name=RUN_NAME, job=JOB,
                                        workspace=tmp_path / "searches", state_dir=tmp_path / "submission_attempts",
                                        sacct=sacct)
    assert code == 0 and again != evidence, output
    summary = logs_of(again)[log.stem]
    assert summary["mpi_startup_lines"] == 40 and len(summary["placement_samples"]) == 8
    assert all(len(line) <= 200 and "\x1b" not in line for line in summary["placement_samples"])


# --- D1: the strict step-5 comparator (17 Nov 2026) -----------------------------------------------------------

DIAMOND = b"""d
1.0
0.0 2.735 2.735
2.735 0.0 2.735
2.735 2.735 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""
# Three species at general positions in a triclinic cell: no symmetry, so a mirror image is another structure.
CHIRAL = b"""c
1.0
4.1 0.0 0.0
0.7 3.9 0.0
0.4 0.9 3.7
Si Ge C
1 1 1
Direct
0.11 0.07 0.13
0.52 0.31 0.44
0.29 0.73 0.81
"""


def strict(before: bytes, after: bytes) -> dict:
    return ANALYSER.same_cell_in_another_basis(ANALYSER.parse_poscar(before.decode()),
                                               ANALYSER.parse_poscar(after.decode()))


@pytest.mark.parametrize("structure", [HEX, DIAMOND, CHIRAL], ids=["hex", "diamond", "chiral"])
@pytest.mark.parametrize("rewrite", [
    {},
    {"basis": OTHER_BASIS},
    {"basis": ((1, 3, 0), (0, 1, 0), (0, 0, 1))},  # coefficients beyond any search range
    {"basis": ((0, 1, 0), (1, 0, 0), (1, 1, 1))},  # a left-handed basis of the same cell
    {"wrap": False, "lattice_offsets": "per-atom"},  # atoms written in neighbouring cells
    {"reverse_atoms": True},  # atoms listed in another order
    {"basis": OTHER_BASIS, "wrap": False, "lattice_offsets": "per-atom", "reverse_atoms": True},
], ids=["unchanged", "basis", "large-coefficients", "left-handed-basis", "unwrapped", "reordered", "all"])
def test_the_same_cell_and_atoms_in_any_lattice_basis_or_periodic_image_are_accepted(structure, rewrite):
    options = dict(rewrite)
    if options.get("reverse_atoms") and structure is CHIRAL:
        pytest.skip("listing atoms of several species in reverse order would change the species order")
    if options.get("lattice_offsets") == "per-atom":
        n = sum(int(v) for v in structure.decode().splitlines()[6].split())
        options["lattice_offsets"] = [[(-1) ** k, 2 * k - 1, 1 - k] for k in range(n)]
    result = strict(structure, rewrite_poscar(structure, **options))
    assert result["same"] is True, result
    assert result["lattice_residual_a"] <= 1e-6 and result["max_atom_residual_a"] <= 1e-6
    assert abs(round(np.linalg.det(np.array(result["basis_change"])))) == 1


def test_a_value_on_the_cell_boundary_is_the_same_position():
    near_one = HEX.replace(b"0.0 0.0 0.0\n0.333", b"0.99999999 0.0 -0.00000001\n0.333")
    assert strict(HEX, near_one)["same"] is True


@pytest.mark.parametrize("structure", [HEX, DIAMOND, CHIRAL], ids=["hex", "diamond", "chiral"])
@pytest.mark.parametrize("rewrite, reason", [
    ({"shift": (0.001, 0.0, 0.0)}, "no counterpart"),  # an origin shift, however small beyond write precision
    ({"shift": (0.5, 0.5, 0.5)}, "no counterpart"),
    ({"move_first_atom_a": 5e-4}, "no counterpart"),
    ({"strain": (1e-4, 0.0, 0.0)}, "same Cartesian frame"),
    ({"rotate_z_deg": 30.0}, "same Cartesian frame"),  # identical metric, rotated crystal
    ({"mirror_x": True}, "same Cartesian frame"),  # identical metric, mirror image
    ({"basis": OTHER_BASIS, "rotate_z_deg": 90.0}, "same Cartesian frame"),
], ids=["origin-shift", "half-cell-shift", "moved-atom", "strain", "rotation", "mirror", "basis-and-rotation"])
def test_genuine_geometric_differences_are_never_accepted(structure, rewrite, reason):
    if structure is not CHIRAL and any(k in rewrite for k in ("rotate_z_deg", "mirror_x")):
        pytest.skip("a symmetric structure can be its own rotated or mirrored image: see the test below")
    result = strict(structure, rewrite_poscar(structure, **rewrite))
    assert result["same"] is False, result
    if reason == "same Cartesian frame" and (result.get("lattice_residual_a") or 1) <= 1e-6:
        # A rotation or mirror that is a symmetry of the lattice (hexagonal, cubic): the cell is found in
        # another basis in the same frame, and the atoms, now elsewhere, are what fails.
        assert "no counterpart" in result["reason"], result
    else:
        assert reason in result["reason"], result


def test_positions_not_descriptions_are_compared_under_rotations_and_mirrors():
    # Mirroring the hexagonal test cell in x puts every atom exactly where an atom already was: the same
    # structure, accepted. Rotating diamond by 90 degrees about z gives the same lattice but puts the
    # second atom elsewhere: rejected on the atoms. The chiral cell's mirror image is rejected.
    assert strict(HEX, rewrite_poscar(HEX, mirror_x=True))["same"] is True
    rotated = strict(DIAMOND, rewrite_poscar(DIAMOND, rotate_z_deg=90.0))
    assert rotated["same"] is False and rotated["lattice_residual_a"] <= 1e-6 and "no counterpart" in rotated["reason"]
    assert strict(CHIRAL, rewrite_poscar(CHIRAL, mirror_x=True))["same"] is False


def test_a_movement_within_write_precision_is_accepted_and_beyond_it_is_not():
    assert strict(HEX, rewrite_poscar(HEX, move_first_atom_a=5e-5))["same"] is True
    assert strict(HEX, rewrite_poscar(HEX, move_first_atom_a=2e-4))["same"] is False


def test_symmetry_equivalent_structures_with_atoms_elsewhere_are_not_the_same():
    # Diamond Si inverted through the origin: the same crystal up to symmetry, atoms at other positions.
    lines = DIAMOND.decode().splitlines()
    inverted = "\n".join(lines[:8] + ["0.0 0.0 0.0", "-0.25 -0.25 -0.25"]) + "\n"
    result = strict(DIAMOND, inverted.encode())
    assert result["same"] is False and "no counterpart" in result["reason"]
    # The chiral cell's mirror image has exactly the same metric (the measurement sees no strain).
    mirror = rewrite_poscar(CHIRAL, mirror_x=True)
    assert transformation(CHIRAL, mirror)["max_abs_principal_strain"] < 1e-8
    assert strict(CHIRAL, mirror)["same"] is False


def test_a_rotated_or_mirrored_crystal_is_measured_with_zero_strain_but_reports_handedness():
    rotated = transformation(CHIRAL, rewrite_poscar(CHIRAL, rotate_z_deg=30.0))
    mirrored = transformation(CHIRAL, rewrite_poscar(CHIRAL, mirror_x=True))
    assert rotated["max_abs_principal_strain"] < 1e-8 and rotated["orientation_preserving"] is True
    assert mirrored["max_abs_principal_strain"] < 1e-8 and mirrored["orientation_preserving"] is False


def test_another_composition_or_an_unreadable_structure_is_never_the_same():
    assert strict(HEX, rewrite_poscar(HEX, species=["Ge"]))["same"] is False
    assert ANALYSER.same_cell_in_another_basis(None, ANALYSER.parse_poscar(HEX.decode()))["same"] is None


@pytest.mark.parametrize("change, accepted", [
    ({"basis": OTHER_BASIS, "wrap": False, "lattice_offsets": [[1, 0, -1], [0, 2, 0]], "reverse_atoms": True}, True),
    ({"rotate_z_deg": 15.0}, False),
    ({"basis": OTHER_BASIS, "shift": (0.0, 0.0, 0.01)}, False),
])
def test_the_analysis_applies_the_strict_comparator_to_the_final_step(tmp_path, change, accepted):
    report = analyse(tmp_path, Scenario(input_changes={(7, 5): change}))
    detail = archive_check(report)["detail"]
    if accepted:
        assert report["result"] == "PASSED", report["failed"] + report["missing"]
        assert detail["final_step_basis_rewrites_total"] == 1 and len(detail["final_step_basis_rewrites"][0]) == 3
    else:
        assert report["result"] == "FAILED" and detail["final_step_continuity_breaks_total"] == 1
        problems = [p for ps in detail["contradictions"].values() for p in ps]
        assert any("the final step did not start" in p for p in problems)
    # Final transitions are never among the intermediate findings.
    assert all(r["steps"] != [4, 5] for r in report["findings"]["intermediate_transformations"]["rows"])


# --- D3: the read-only step-log inspection and its comparison with the archive ---------------------------------

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from uspex_execution_fakes import install_slurm, tree_state  # noqa: E402

KIT = Path(__file__).resolve().parents[1] / "reference" / "uspex_pilot" / "execution_validation"
INSPECTION = KIT / "power_step_log_inspection.sh"


def load_comparer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare_step_log_inspection", KIT / "compare_step_log_inspection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect(tmp_path, *arguments, queued=False):
    script = INSPECTION.read_text()
    for old, new in {"WORKSPACE=/bmd/bmdguest/searches": f"WORKSPACE={tmp_path / 'searches'}",
                     "PY=/bmd/bmdguest/envs/atomate2_remote/bin/python": f"PY={sys.executable}"}.items():
        assert old in script
        script = script.replace(old, new)
    bin_dir, state = tmp_path / "inspect_bin", tmp_path / "inspect_slurm.json"
    install_slurm(bin_dir, state, job=JOB, sacct=[], queued=queued)
    environment = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_SLURM_EXECUTION": str(state)}
    completed = subprocess.run(["bash", "-s", "--", *(arguments or (ATTEMPT, RUN_NAME, JOB))], input=script.encode(),
                               capture_output=True, env=environment)
    return completed.returncode, completed.stdout.decode(), completed.stderr.decode()


def test_the_inspection_reads_every_log_changes_nothing_and_agrees_with_the_archive(tmp_path):
    code, output, evidence, run = make(tmp_path, Scenario(**SHARED_48))
    assert code == 0
    before = tree_state(tmp_path / "searches")
    rc, out, err = inspect(tmp_path)
    assert rc == 0 and out.rstrip().endswith("== INSPECTION COMPLETE"), out + err
    assert tree_state(tmp_path / "searches") == before
    comparer = load_comparer()
    report = comparer.compare(comparer.read_inspection(out), evidence)
    assert report["result"] == "HASHES AGREE", report
    assert report["matched"] == report["collected_logs"] == len(json.loads((evidence / "step_logs.json").read_text()))
    assert report["placement"]["with_pinning_rows"] == report["placement"]["logs"]
    assert all(len(s) <= 200 for s in report["placement"]["samples"])


def test_a_log_changed_since_collection_is_reported_not_verified(tmp_path):
    code, output, evidence, run = make(tmp_path)
    log = next((run / "bridge_spool" / "logs").glob("*.log"))
    log.write_bytes(log.read_bytes() + b"appended later\n")
    rc, out, _ = inspect(tmp_path)
    comparer = load_comparer()
    report = comparer.compare(comparer.read_inspection(out), evidence)
    assert rc == 0 and report["result"] == "NOT VERIFIED" and report["mismatched"] == [log.stem]


def test_odd_names_links_and_directories_are_reported_and_make_the_inspection_incomplete(tmp_path):
    code, output, evidence, run = make(tmp_path)
    logs = run / "bridge_spool" / "logs"
    (logs / "odd name\nwith newline.log").write_bytes(b"[0] MPI startup(): Intel(R) MPI Library, Version 2021.9\n")
    (logs / "linked.log").symlink_to(next(logs.glob("s1-*.log")))
    (logs / "folder.log").mkdir()
    (logs / "notes.txt").write_text("not a step log\n")
    before = tree_state(tmp_path / "searches")
    rc, out, err = inspect(tmp_path)
    assert rc == 1 and out.rstrip().endswith("== INSPECTION INCOMPLETE"), out + err
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert {r.get("name") for r in rows if r.get("problem") == "skipped"} == {"linked.log", "folder.log"}
    assert any(r.get("log") == "odd name\nwith newline" and r["intel_mpi_version"] for r in rows)
    assert any(r.get("note") == "ignored" and r["name"] == "notes.txt" for r in rows)
    assert tree_state(tmp_path / "searches") == before
    comparer = load_comparer()
    report = comparer.compare(comparer.read_inspection(out), evidence)
    assert report["result"] == "NOT VERIFIED" and "odd name\nwith newline" in report["not_in_evidence"]


@pytest.mark.parametrize("arguments, queued, status, message", [
    (("not-an-attempt", RUN_NAME, JOB), False, 2, "REFUSED malformed attempt ID"),
    ((ATTEMPT, "uspex_search-20261120-100000-deadbeef", JOB), False, 2, "does not belong to the attempt"),
    ((ATTEMPT, RUN_NAME, "1; rm -rf /"), False, 2, "REFUSED malformed job ID"),
    ((), True, 3, "still in the queue"),
])
def test_the_inspection_refuses_clearly(tmp_path, arguments, queued, status, message):
    make(tmp_path)
    rc, out, _ = inspect(tmp_path, *arguments, queued=queued)
    assert rc == status and message in out


def test_a_missing_log_directory_is_refused(tmp_path):
    code, output, evidence, run = make(tmp_path)
    import shutil
    shutil.rmtree(run / "bridge_spool" / "logs")
    rc, out, _ = inspect(tmp_path)
    assert rc == 2 and "REFUSED no step-log directory" in out


def test_the_inspection_never_writes_submits_or_cancels():
    text = INSPECTION.read_text()
    for forbidden in ("sbatch", "scancel", "srun ", "> \"$", ">>", "mkdir", "rm ", "mv ", "cp ", "tee ", "touch "):
        assert forbidden not in text, forbidden
    assert 'open(path, "w' not in text and "os.remove" not in text and "os.mkdir" not in text
