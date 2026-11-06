"""Post-step validity check for USPEX search steps (USPEX integration step 2)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from backend.structure_search import incar_policy, step_validity
from backend.structure_search.step_validity import (
    EXECUTION_FAILED,
    INVALID_CONVERGENCE,
    INVALID_EVIDENCE,
    VALID,
    evaluate_step,
    withhold_outputs,
)

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "fake_vasp_outputs", REPO / "tests" / "fixtures" / "structure_search_bridge" / "fake_vasp_outputs.py"
)
fake = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fake)


def _calcfold(tmp_path, step, variant="valid"):
    folder = tmp_path / f"CalcFold_{step}_{variant}"
    folder.mkdir()
    (folder / "POSCAR").write_text("Si\n1.0\n3 0 0\n0 3 0\n0 0 3\nSi\n1\nDirect\n0 0 0\n")
    (folder / "INCAR").write_text(f"STEP = {step}\n")
    fake.write_outputs(folder, step, -10.0 * step, variant)
    return folder


def _evaluate(tmp_path, step, variant="valid", returncode=0, **kwargs):
    return evaluate_step(step, _calcfold(tmp_path, step, variant), returncode=returncode, **kwargs)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_converged_steps_are_valid(tmp_path, step):
    result = _evaluate(tmp_path, step)
    assert result["classification"] == VALID, result
    assert result["reasons"] == []
    assert result["evidence"]["converged_electronic"] is True
    assert result["evidence"]["free_energy_ev"] == pytest.approx(-10.0 * step)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_nelm_exhaustion_is_invalid_in_every_step(tmp_path, step):
    result = _evaluate(tmp_path, step, "nelm")
    assert result["classification"] == INVALID_CONVERGENCE
    assert "electronic loop not converged" in result["reasons"][0]


@pytest.mark.parametrize("step", [1, 2, 3])
def test_reaching_nsw_is_allowed_in_steps_1_to_3(tmp_path, step):
    result = _evaluate(tmp_path, step, "nsw")
    assert result["evidence"]["converged_ionic"] is False
    assert result["classification"] == VALID


def test_reaching_nsw_is_invalid_in_step_4(tmp_path):
    result = _evaluate(tmp_path, 4, "nsw")
    assert result["classification"] == INVALID_CONVERGENCE
    assert result["reasons"] == ["ionic relaxation not converged: 99 ionic steps reached NSW = 99"]


@pytest.mark.parametrize("step", [1, 4, 5])
def test_non_finite_energy_is_invalid(tmp_path, step):
    result = _evaluate(tmp_path, step, "nan")
    assert result["classification"] == INVALID_CONVERGENCE
    assert result["reasons"] == ["the final energy is not finite"]


@pytest.mark.parametrize(
    ("variant", "fragment"),
    [
        ("missing_vasprun", "vasprun.xml is missing"),
        ("truncated", "vasprun.xml cannot be read"),
        ("no_energy", "no final free energy"),
        ("missing_outcar", "OUTCAR is missing"),
        ("no_toten", "no 'free energy TOTEN' line"),
        ("mismatch", "differs from vasprun.xml"),
        ("missing_contcar", "CONTCAR is missing"),
        ("bad_contcar", "CONTCAR cannot be read"),
    ],
)
def test_missing_or_malformed_evidence_is_invalid(tmp_path, variant, fragment):
    for step in (1, 5):
        result = evaluate_step(step, _calcfold_in(tmp_path, step, variant), returncode=0)
        assert result["classification"] == INVALID_EVIDENCE, result
        assert fragment in result["reasons"][0]


def _calcfold_in(tmp_path, step, variant):
    base = tmp_path / f"s{step}"
    base.mkdir(exist_ok=True)
    return _calcfold(base, step, variant)


def test_empty_and_oversized_files_are_invalid_evidence(tmp_path, monkeypatch):
    folder = _calcfold(tmp_path, 4)
    (folder / "OUTCAR").write_text("")
    assert "OUTCAR is empty" in evaluate_step(4, folder, returncode=0)["reasons"][0]
    folder = _calcfold(tmp_path, 3)
    monkeypatch.setattr(step_validity, "MAX_EVIDENCE_BYTES", 100)
    result = evaluate_step(3, folder, returncode=0)
    assert result["classification"] == INVALID_EVIDENCE
    assert "larger than 100 bytes" in result["reasons"][0]


def test_contcar_atom_count_must_match(tmp_path):
    folder = _calcfold(tmp_path, 2)
    (folder / "CONTCAR").write_text("Si2\n1.0\n3 0 0\n0 3 0\n0 0 3\nSi\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    result = evaluate_step(2, folder, returncode=0)
    assert result["classification"] == INVALID_EVIDENCE
    assert "CONTCAR has 2 atoms" in result["reasons"][0]


@pytest.mark.parametrize("returncode", [1, 7, 137, None])
def test_nonzero_exit_is_execution_failure_whatever_the_files_say(tmp_path, returncode):
    result = _evaluate(tmp_path, 4, "valid", returncode=returncode)
    assert result["classification"] == EXECUTION_FAILED
    assert result["evidence"] is None


def test_stopped_step_is_execution_failure(tmp_path):
    result = _evaluate(tmp_path, 2, "valid", returncode=0, stopped=True)
    assert result["classification"] == EXECUTION_FAILED
    assert result["reasons"] == ["the launcher stopped the step"]


def test_step_without_a_policy_fails_closed(tmp_path):
    folder = _calcfold(tmp_path, 1)
    result = evaluate_step(6, folder, returncode=0)
    assert result["classification"] == INVALID_EVIDENCE
    assert result["reasons"] == ["no validity policy for step 6"]


def test_checker_errors_fail_closed(tmp_path, monkeypatch):
    folder = _calcfold(tmp_path, 4)

    def broken(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(step_validity, "read_evidence", broken)
    result = evaluate_step(4, folder, returncode=0)
    assert result["classification"] == INVALID_EVIDENCE
    assert "RuntimeError: boom" in result["reasons"][0]


def test_evaluation_is_deterministic_and_does_not_modify_files(tmp_path):
    folder = _calcfold(tmp_path, 4, "nsw")
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    first = evaluate_step(4, folder, returncode=0)
    second = evaluate_step(4, folder, returncode=0)
    assert first == second
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


def test_withhold_preserves_every_file_and_hides_what_uspex_reads(tmp_path):
    folder = _calcfold(tmp_path, 4, "nsw")
    originals = {path.name: path.read_bytes() for path in folder.iterdir()}
    archive = tmp_path / "archive"
    files = withhold_outputs(folder, archive, "req-1")
    names = {entry["name"] for entry in files}
    assert names == {"OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml", "INCAR", "POSCAR"}
    for entry in files:
        name = entry["name"]
        assert (archive / name).read_bytes() == originals[name]
        assert entry["sha256"] == hashlib.sha256(originals[name]).hexdigest()
        if name in step_validity.USPEX_READ_FILES:
            assert not (folder / name).exists()
            assert (folder / entry["withheld_as"]).read_bytes() == originals[name]
        else:
            assert (folder / name).read_bytes() == originals[name]


def test_policy_matches_the_lab_decision_and_incar_policy():
    assert incar_policy.STEP_VALIDITY is step_validity.STEP_VALIDITY
    rows = {row["step"]: row for row in step_validity.STEP_VALIDITY}
    assert [rows[s]["ionic_convergence"] for s in (1, 2, 3, 4, 5)] == [False, False, False, True, False]
    assert all(rows[s]["electronic_convergence"] and rows[s]["valid_energy"] for s in rows)
    assert len(rows) == incar_policy.NUM_STEPS


def test_bridge_documents_the_validity_exit_codes():
    from backend.structure_search.bridge import protocol

    text = (REPO / "backend" / "structure_search" / "bridge" / "bmd_bridge.sh").read_text()
    assert f"#   {protocol.EXIT_INVALID_CONVERGENCE}    VASP exited with status 0" in text
    assert f"#   {protocol.EXIT_INVALID_EVIDENCE}    VASP exited with status 0" in text
    reserved = {120, 121, 122, 123, 124, 127, 143}
    assert {protocol.EXIT_INVALID_CONVERGENCE, protocol.EXIT_INVALID_EVIDENCE}.isdisjoint(reserved)


def test_ordinary_workflows_never_use_the_search_validity_check():
    """The check is isolated to structure-search execution."""

    users = []
    for path in sorted((REPO / "backend").rglob("*.py")) + [REPO / "main.py"]:
        if "structure_search" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"step_validity", text):
            users.append(str(path.relative_to(REPO)))
    assert users == []
    for path in sorted((REPO / "compute_api").rglob("*.py")):
        assert "step_validity" not in path.read_text(encoding="utf-8")
