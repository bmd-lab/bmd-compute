"""Scientific execution acceptance tooling (reference/uspex_pilot/execution_validation/).

The POWER evidence collector runs locally (its POWER locations mapped into a
temporary tree, fake ``squeue``/``sacct`` on the PATH) against a fake finished
run directory, and the analyser reads what it collected. Scenarios: a clean
natural completion, a rejected candidate, a missing ladder step, an
inconsistent or stale listed energy, a walltime stop, an incomplete
generation, too many or overlapping parallel steps, missing pinning evidence,
a memory mismatch, a non-zero USPEX exit; renumbered, ambiguous and
structurally contradicted candidates; per-cell, per-atom, unestablished,
inconsistent and E0 listings; candidates of removed CalcFolds traced through
the launcher records, and a lost step log; absent and partial pinning
reports. Nothing contacts POWER.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from uspex_execution_fakes import SHARED_48, Scenario, build_run, run_collector, tree_state

ATTEMPT = "c0ffee00-1111-4222-8333-444455556666"
RUN_NAME = "uspex_search-20261120-100000-c0ffee00"
JOB = "23600001"
DIGEST = "sha256:" + "ab" * 32
INCARS = {f"INCAR_{k}": f"PREC = Accurate\nENCUT = {250 + 50 * k}\nNSW = 50\n" for k in range(1, 6)}
JOB_SCRIPT = "#!/bin/bash\n#SBATCH --time=04:00:00\necho job\n"
ANALYSER = Path(__file__).resolve().parents[1] / "reference" / "uspex_pilot" / "execution_validation" / \
    "analyze_execution.py"


def load_analyser():
    spec = importlib.util.spec_from_file_location("analyze_execution_under_test", ANALYSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acknowledgement() -> dict:
    record = {"schema": "bmd_compute.structure_search_resource_acknowledgement", "schema_version": 1,
              "attempt_id": ATTEMPT, "plan_digest": DIGEST, "resources": {"walltime": "04:00:00"}}
    record["acknowledgement_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    return record


# The shared-node profile's frozen plan (3 x 16 ranks, 48 CPUs, --mem=32G, no --exclusive).
SHARED_EXPECTED = {"parallel": 3, "ranks": 16, "ncpus": 48, "exclusive": False, "job_memory": "32G"}


def expected(**overrides) -> dict:
    return {**_expected(), **overrides}


def _expected() -> dict:
    return {"attempt_id": ATTEMPT, "job_id": JOB, "plan_digest": DIGEST, "walltime": "04:00:00", "parallel": 4,
            "ranks": 48, "steps": 5, "generations": 2, "cell_atoms": 2, "pressure_gpa": 0.0,
            "job_script_sha256": hashlib.sha256(JOB_SCRIPT.encode()).hexdigest(),
            "acknowledgement_sha256": acknowledgement()["acknowledgement_sha256"],
            "incar_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in INCARS.items()}}


def make(tmp_path, scenario=None, *, queued=False, sacct_edit=None):
    workspace = tmp_path / "searches"
    state_dir = tmp_path / "submission_attempts"
    state_dir.mkdir(parents=True)
    run = workspace / RUN_NAME
    built = build_run(run, identity_run_dir=f"/bmd/bmdguest/searches/{RUN_NAME}", attempt=ATTEMPT, job=JOB,
                      plan_digest=DIGEST, scenario=scenario or Scenario(), incars=INCARS, job_script=JOB_SCRIPT,
                      acknowledgement=acknowledgement())
    (state_dir / f"{ATTEMPT}.json").write_text(json.dumps({"attempt_id": ATTEMPT, "state": "SEARCH_SUBMITTED"}))
    sacct = sacct_edit(built["sacct"]) if sacct_edit else built["sacct"]
    before = tree_state(run)
    code, output, evidence = run_collector(tmp_path, attempt=ATTEMPT, run_name=RUN_NAME, job=JOB, workspace=workspace,
                                           state_dir=state_dir, sacct=sacct, queued=queued)
    assert tree_state(run) == before  # the collector never changes the run directory
    return code, output, evidence, run


def analyse(tmp_path, scenario=None, expectation=None, **options):
    code, output, evidence, _ = make(tmp_path, scenario, **options)
    assert code == 0, output
    return load_analyser().analyse(evidence, expected(**(expectation or {})))


def status_of(report, check):
    return next(c for c in report["checks"] if c["check"] == check)["status"]


# --- the collector ---------------------------------------------------------------------------------------


def test_the_collector_copies_only_allowlisted_records_and_changes_nothing(tmp_path):
    code, output, evidence, run = make(tmp_path)
    assert code == 0 and "PASS  the run directory is unchanged" in output and "== EVIDENCE COLLECTED" in output
    copied = {str(p.relative_to(evidence / "files")) for p in (evidence / "files").rglob("*") if p.is_file()}
    assert {"search_status.json", "runtime_check.json", "bmd_search_job.sbatch", "bridge_spool/launcher.jsonl",
            "search/results1/Individuals", "search/results1/gatheredPOSCARS", "search/Specific/INCAR_5",
            "uspex_master.log"} <= copied
    assert not any(part.startswith(("POTCAR", "OUTCAR", "WAVECAR", "vasprun")) for name in copied
                   for part in Path(name).parts)
    every = b"".join(p.read_bytes() for p in evidence.rglob("*") if p.is_file())
    assert b"VRHFIN" not in every and b"never copied" not in every
    logs = json.loads((evidence / "step_logs.json").read_text())
    some = next(iter(logs.values()))
    assert some["ionic_steps"] == 2 and len(some["pin_rows"]) == 48 and some["final_free_energy_ev"] < 0
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert all(len(row["sha256"]) == 64 for row in manifest["copied"]) and manifest["refused"] == []
    assert (evidence / "sacct_steps.txt").read_text().startswith(JOB + "|")


def test_the_collector_refuses_while_the_job_is_queued(tmp_path):
    code, output, evidence, _ = make(tmp_path, queued=True)
    assert code == 3 and "still in the queue" in output and evidence is None
    assert not list((tmp_path / "evidence_parent").iterdir())


def test_the_collector_bounds_a_large_master_log(tmp_path):
    workspace = tmp_path / "searches"
    run = workspace / RUN_NAME
    built = build_run(run, identity_run_dir=f"/x/{RUN_NAME}", attempt=ATTEMPT, job=JOB, plan_digest=DIGEST,
                      scenario=Scenario(), incars=INCARS, job_script=JOB_SCRIPT, acknowledgement=acknowledgement())
    (run / "uspex_master.log").write_bytes(b"** USPEX v25.0 **\n" + b"x" * (17 * 1024 * 1024) + b"\nTHE END\n")
    (tmp_path / "submission_attempts").mkdir()
    (tmp_path / "submission_attempts" / f"{ATTEMPT}.json").write_text("{}")
    code, _, evidence = run_collector(tmp_path, attempt=ATTEMPT, run_name=RUN_NAME, job=JOB, workspace=workspace,
                                      state_dir=tmp_path / "submission_attempts", sacct=built["sacct"])
    copied = (evidence / "files" / "uspex_master.log").read_bytes()
    assert code == 0 and len(copied) < 17 * 1024 * 1024 and copied.endswith(b"THE END\n")
    assert b"truncated by the evidence collector" in copied
    row = next(r for r in json.loads((evidence / "manifest.json").read_text())["copied"]
               if r["path"] == "uspex_master.log")
    assert row["truncated"] is True and row["bytes"] > 17 * 1024 * 1024


@pytest.mark.parametrize("arguments, message", [
    ([], "usage"), (["x", RUN_NAME, JOB], "malformed attempt ID"),
    ([ATTEMPT, "uspex_search-20261120-100000-deadbeef", JOB], "does not belong"),
    ([ATTEMPT, RUN_NAME, "1;id"], "malformed job ID"),
])
def test_the_collector_refuses_malformed_arguments(arguments, message):
    import subprocess

    from uspex_execution_fakes import COLLECTOR

    result = subprocess.run(["bash", str(COLLECTOR), *arguments], capture_output=True, text=True)
    assert result.returncode == 2 and message in result.stdout


def test_the_collector_never_submits_cancels_or_starts_anything():
    from uspex_execution_fakes import COLLECTOR

    text = COLLECTOR.read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for forbidden in ("sbatch ", "scancel", "srun ", "scontrol update", "rm -", "mv ", "singularity ", "vasp_std"):
        assert forbidden not in code, forbidden


# --- the analysis ----------------------------------------------------------------------------------------


def test_a_clean_natural_completion_passes_and_records_the_findings(tmp_path):
    report = analyse(tmp_path)
    assert report["result"] == "PASSED" and report["classification"] == "natural_completion", report["checks"]
    assert report["failed"] == [] and report["missing"] == []
    assert {"slurm_outcome", "allocation", "runtime_check", "memory", "uspex_startup", "bridge", "vasp_steps",
            "parallel", "pinning", "validity_decisions", "chains", "listed_energies", "generations",
            "search_status", "identities", "collection", "energy_semantics", "retention",
            "evidence_archive"} == set(report["verified"])
    findings = report["findings"]
    assert findings["candidates"] == {"started": 21, "completed_ladder": 21, "rejected": {}}
    assert findings["steps"]["total"] == 105 and findings["steps"]["by_classification"] == {"valid": 105}
    assert findings["best"]["number"] == 13 and findings["best"]["energy_per_atom_ev"] == pytest.approx(-5.42, abs=0.01)
    assert findings["best"]["symmetry"]["space_group_0.1"][0] == 227
    assert findings["diamond"] == {"required": False, "found": True, "individuals": [13], "is_lowest_energy": True,
                                   "note": "Recorded, not a condition of operational success."}


def test_diamond_is_recorded_not_required(tmp_path):
    report = analyse(tmp_path, Scenario(diamond=False))
    assert report["result"] == "PASSED"
    assert report["findings"]["diamond"]["found"] is False and report["findings"]["diamond"]["is_lowest_energy"] is False


def test_a_rejected_candidate_stops_its_chain_and_is_not_listed(tmp_path):
    report = analyse(tmp_path, Scenario(rejected={3: (3, "invalid_convergence"), 15: (5, "execution_failed")}))
    assert report["result"] == "PASSED", report["checks"]
    assert report["findings"]["candidates"]["rejected"] == {
        "3": {"step": 3, "classification": "invalid_convergence", "delivered_status": "118"},
        "15": {"step": 5, "classification": "execution_failed", "delivered_status": "7"}}
    assert report["findings"]["steps"]["by_classification"] == {"valid": 101, "invalid_convergence": 1,
                                                                "execution_failed": 1}


def listed(report):
    return next(c for c in report["checks"] if c["check"] == "listed_energies")["detail"]


@pytest.mark.parametrize("scenario, failed, reason", [
    (Scenario(missing_steps={4: 4}), {"chains", "listed_energies"}, "no accepted step-5 energy"),
    (Scenario(energy_offsets={5: 0.05}), {"listed_energies"}, "no accepted step-5 energy"),
    (Scenario(rejected={3: (3, "invalid_convergence")}, listed_rejected={3: 2}), {"listed_energies"}, "stale"),
])
def test_unusable_or_inconsistent_listed_energies_fail(tmp_path, scenario, failed, reason):
    report = analyse(tmp_path, scenario)
    assert report["result"] == "FAILED" and set(report["failed"]) == failed, report["failed"]
    assert any(reason in row["reason"] for row in listed(report)["contradicted"]), listed(report)


# --- candidate identity -----------------------------------------------------------------------------------


def test_the_numbering_convention_is_used_only_where_the_survivors_confirm_it(tmp_path):
    report = analyse(tmp_path)
    detail = listed(report)
    assert detail["numbering_confirmed"] is True and detail["established"] == 21
    assert "structure and energy" in detail["methods"]


@pytest.mark.parametrize("offset", [3, 100])
def test_renumbered_candidates_are_associated_by_structure_and_energy(tmp_path, offset):
    report = analyse(tmp_path, Scenario(number_offset=offset))
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    detail = listed(report)
    assert detail["numbering_confirmed"] is False and detail["established"] == 21
    assert detail["numbering_counterexamples"]
    assert detail["methods"] == ["structure and energy"]  # archived final structures: every generation
    assert report["findings"]["best"]["number"] == 13 + offset and report["findings"]["best"]["accepted_chain"] == 13


def test_ambiguous_identity_is_incomplete_evidence(tmp_path):
    # Two first-generation candidates with the same step-5 F and the same relaxed structure, and numbers that
    # are not CalcFold numbers: nothing decides which listed one is which chain.
    report = analyse(tmp_path, Scenario(number_offset=100, same_energy=(3, 5), same_structure=(3, 5)))
    assert report["result"] == "INCOMPLETE EVIDENCE" and report["missing"] == ["listed_energies", "retention"], report["failed"]
    ambiguous = listed(report)["ambiguous"]
    assert {row["number"] for row in ambiguous} == {103, 105}
    assert all("[3, 5]" in row["reason"] for row in ambiguous)


def test_equal_energies_are_resolved_by_numbering_only_once_the_survivors_confirm_it(tmp_path):
    report = analyse(tmp_path, Scenario(same_energy=(3, 5), same_structure=(3, 5)))
    assert report["result"] == "PASSED"
    assert "structure, energy and confirmed numbering" in listed(report)["methods"]


def test_equal_energies_are_resolved_by_the_archived_structures(tmp_path):
    report = analyse(tmp_path, Scenario(number_offset=100, same_energy=(3, 5)))
    assert report["result"] == "PASSED" and listed(report)["methods"] == ["structure and energy"]


def test_a_structure_contradicting_the_chain_fails(tmp_path):
    # USPEX recorded diamond for individual 13; chain 13's own step-5 CONTCAR (archived and surviving) differs.
    report = analyse(tmp_path, Scenario(contcar_differs=(13,)))
    assert report["result"] == "FAILED" and report["failed"] == ["listed_energies"]
    [row] = listed(report)["contradicted"]
    assert row["number"] == 13 and "structure differs" in row["reason"]


def test_a_structure_matching_another_chain_without_its_energy_fails(tmp_path):
    # CalcFolds 14 and 15 hold each other's structures: USPEX's record of 14 is chain 15's cell at chain 14's F.
    report = analyse(tmp_path, Scenario(contcar_swap=(14, 15)))
    assert report["result"] == "FAILED" and report["failed"] == ["listed_energies"]
    contradicted = {row["number"]: row["reason"] for row in listed(report)["contradicted"]}
    assert set(contradicted) == {14, 15} and "structure matches chain(s) [15]" in contradicted[14]


def test_without_any_recorded_structure_unique_energies_still_identify(tmp_path):
    # A run without the archive (an older runtime) and with no surviving CONTCAR.
    report = analyse(tmp_path, Scenario(no_contcar=True, number_offset=100, archive=False))
    assert status_of(report, "listed_energies") == "verified" and listed(report)["methods"] == ["unique energy"]
    assert report["result"] == "INCOMPLETE EVIDENCE" and report["missing"] == ["evidence_archive", "retention"]


def test_without_any_recorded_structure_equal_energies_are_ambiguous(tmp_path):
    report = analyse(tmp_path, Scenario(no_contcar=True, same_energy=(15, 17), number_offset=100, archive=False))
    assert report["result"] == "INCOMPLETE EVIDENCE"
    assert report["missing"] == ["evidence_archive", "listed_energies", "retention"]


# --- energy semantics -------------------------------------------------------------------------------------


def semantics(report):
    return next(c for c in report["checks"] if c["check"] == "energy_semantics")


def test_per_cell_energies_are_established_from_uspex_own_tables(tmp_path):
    report = analyse(tmp_path)
    check = semantics(report)
    assert check["status"] == "verified" and check["detail"]["quantity"] == "per_cell"
    assert check["detail"]["gathered_quantity"] == "per_cell"


def test_energy_per_atom_is_normalised_to_the_cell_before_comparison(tmp_path):
    report = analyse(tmp_path, Scenario(quantity="per_atom"))
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    assert semantics(report)["detail"]["quantity"] == "per_atom"
    best = report["findings"]["best"]
    assert best["listed_quantity"] == "per_atom" and best["energy_ev_per_cell"] == pytest.approx(-10.84, abs=0.002)
    assert best["energy_per_atom_ev"] == pytest.approx(-5.42, abs=0.001)


def test_an_unestablished_quantity_is_incomplete_evidence_and_nothing_is_compared(tmp_path):
    report = analyse(tmp_path, Scenario(semantics_tables=False))
    assert report["result"] == "INCOMPLETE EVIDENCE"
    assert {"energy_semantics", "listed_energies", "retention"} <= set(report["missing"])
    assert "not compared" in listed(report)


def test_an_inconsistent_quantity_fails(tmp_path):
    report = analyse(tmp_path, Scenario(table_inconsistent=True))
    assert report["result"] == "FAILED" and "energy_semantics" in report["failed"]


def test_listing_e0_instead_of_f_fails(tmp_path):
    report = analyse(tmp_path, Scenario(listed_e0=True))
    assert report["result"] == "FAILED" and "energy_semantics" in report["failed"]


def test_nonzero_pressure_leaves_the_quantity_unestablished(tmp_path):
    report = analyse(tmp_path, Scenario(pressure="5"))
    assert report["result"] != "PASSED" and semantics(report)["status"] in ("missing", "failed")


# --- evidence retention -----------------------------------------------------------------------------------


def archive_check(report):
    return next(c for c in report["checks"] if c["check"] == "evidence_archive")


def test_candidates_of_removed_calcfolds_are_traced_through_the_archive(tmp_path):
    code, output, evidence, run = make(tmp_path)
    assert not list((run / "search" / "Calculation").glob("Calcfold_1_*"))  # generation 1 removed by USPEX
    copied = {str(p.relative_to(evidence / "files")) for p in (evidence / "files" / "search" / "bmd_evidence").rglob("*")}
    assert any(name.endswith("/CONTCAR") for name in copied) and "search/bmd_evidence/index.jsonl" in copied
    report = load_analyser().analyse(evidence, expected())
    check = next(c for c in report["checks"] if c["check"] == "retention")
    assert check["status"] == "verified" and check["detail"]["traced"] == 21
    assert check["detail"]["chains_whose_calcfolds_were_removed"] == list(range(1, 11))
    inferred = check["detail"]["inferred_generation_association"]
    assert "inferred, not execution identity" in inferred["source"]
    binding = {row["number"]: row for row in inferred["rows"]}
    assert binding[1]["listed_generation"] == 1 and binding[13]["listed_generation"] == 2
    assert binding[1]["final_record"].startswith("s5-0001-")
    # execution identity never carries a generation: every archived record says null
    records = (evidence / "files" / "search" / "bmd_evidence" / "steps").glob("*/record.json")
    assert {json.loads(p.read_text())["generation"] for p in records} == {None}
    archive = archive_check(report)
    assert archive["status"] == "verified" and archive["detail"]["verified_steps"] == 105
    assert archive["detail"]["final_structures"] == 21


@pytest.mark.parametrize("scenario, missing", [
    (Scenario(archive=False), "missing_records"),
    (Scenario(archive_missing=((4, 3),)), "write_failures"),
    (Scenario(archive_interrupted=True), "interrupted"),
])
def test_absent_or_interrupted_archive_records_are_incomplete_evidence(tmp_path, scenario, missing):
    report = analyse(tmp_path, scenario)
    assert report["result"] == "INCOMPLETE EVIDENCE" and not report["failed"]
    assert "evidence_archive" in report["missing"] and archive_check(report)["detail"][missing]


@pytest.mark.parametrize("scenario, fragment", [
    (Scenario(archive_tamper=((4, 3),)), "reasons: archive"),
    (Scenario(archive_conflict=True), "a different record"),
    (Scenario(survivor_altered=(14,)), "surviving CalcFold's CONTCAR is not the archived"),
    (Scenario(continuity_break=((6, 3),)), "did not start from its predecessor's output"),
])
def test_an_archive_that_contradicts_the_run_fails(tmp_path, scenario, fragment):
    report = analyse(tmp_path, scenario)
    assert report["result"] == "FAILED" and "evidence_archive" in report["failed"], report["failed"]
    problems = [p for ps in archive_check(report)["detail"]["contradictions"].values() for p in ps]
    assert any(fragment in p for p in problems), problems


def test_a_rejected_candidate_is_archived_with_its_verdict_and_traced(tmp_path):
    report = analyse(tmp_path, Scenario(rejected={14: (4, "invalid_convergence")}))
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    assert archive_check(report)["detail"]["verified_steps"] == 104 and archive_check(report)["detail"]["final_structures"] == 20


def test_a_lost_step_log_is_incomplete_evidence(tmp_path):
    report = analyse(tmp_path, Scenario(deleted_logs=(4,)))
    assert report["result"] == "INCOMPLETE EVIDENCE"
    assert "retention" in report["missing"] and "vasp_steps" in report["missing"] and not report["failed"]


def test_a_walltime_stop_is_classified_as_such(tmp_path):
    report = analyse(tmp_path, Scenario(slurm_state="TIMEOUT", exit_code="0:15", outcome="stopped",
                                        master_status=123, stopped=True, unfinished_after=15,
                                        listed_generations=(1,)))
    assert report["classification"] == "stopped_by_walltime" and report["result"] == "FAILED"
    assert {"slurm_outcome", "generations"} <= set(report["failed"])


def test_slurm_completed_alone_is_not_success(tmp_path):
    incomplete = analyse(tmp_path / "a", Scenario(listed_generations=(1,)))
    assert incomplete["classification"] == "failed_or_incomplete" and "generations" in incomplete["failed"]
    assert status_of(incomplete, "slurm_outcome") == "verified"  # SLURM's COMPLETED is a fact, not the verdict
    uspex_failed = analyse(tmp_path / "b", Scenario(outcome="uspex_failed", master_status=1))
    assert uspex_failed["classification"] == "failed_or_incomplete" and uspex_failed["result"] == "FAILED"


@pytest.mark.parametrize("scenario, check", [
    (Scenario(parallel=6), "parallel"),
    (Scenario(pin="shared"), "pinning"),
    (Scenario(alloc_mem="250G"), "memory"),
    (Scenario(runtime_status="failed"), "runtime_check"),
    (Scenario(timelimit="7-00:00:00"), "allocation"),
])
def test_execution_mismatches_fail(tmp_path, scenario, check):
    report = analyse(tmp_path, scenario)
    assert check in report["failed"] and report["result"] == "FAILED", report["failed"]


@pytest.mark.parametrize("pin", ["none", "partial"])
def test_missing_or_incomplete_pinning_evidence_is_never_success(tmp_path, pin):
    report = analyse(tmp_path, Scenario(pin=pin))
    assert report["result"] == "INCOMPLETE EVIDENCE" and report["missing"] == ["pinning"] and report["failed"] == []
    detail = next(c for c in report["checks"] if c["check"] == "pinning")["detail"]
    assert detail["uninterpretable"]


def test_missing_pinning_evidence_is_reported_as_missing(tmp_path):
    report = analyse(tmp_path, Scenario(pin="none"))
    assert report["result"] == "INCOMPLETE EVIDENCE" and report["missing"] == ["pinning"] and report["failed"] == []


def test_a_changed_frozen_input_fails_the_identity_check(tmp_path):
    code, output, evidence, run = make(tmp_path)
    assert code == 0
    changed = dict(expected(), incar_sha256={**expected()["incar_sha256"], "INCAR_5": "0" * 64})
    report = load_analyser().analyse(evidence, changed)
    assert "identities" in report["failed"]
    assert next(c for c in report["checks"] if c["check"] == "identities")["detail"]["incars"] is False


def test_missing_records_are_missing_not_passed(tmp_path):
    code, output, evidence, _ = make(tmp_path)
    for name in ("search_status.json", "search/results1/Individuals"):
        (evidence / "files" / name).unlink()
    report = load_analyser().analyse(evidence, expected())
    assert {"listed_energies", "generations", "search_status"} <= set(report["missing"]) | set(report["failed"])
    assert report["result"] != "PASSED"


def test_the_summary_keeps_facts_missing_evidence_and_findings_apart(tmp_path):
    analyser = load_analyser()
    report = analyse(tmp_path)
    lines = analyser.summary_lines(report)
    assert lines[0] == "== PASSED (natural_completion)"
    assert any(line.startswith("VERIFIED  listed_energies") for line in lines)
    assert any(line.startswith("FINDING   diamond-like structure found: True") for line in lines)


def test_uspex_list_columns_with_spaces_are_one_column():
    analyser = load_analyser()
    rows = analyser.read_table("generation number energy energy_per_atom num_atoms_all expected_origin origin parents\n"
                               "2 14 -10.400000 -5.200000 [2] Random Heredity [12, 13]\n")
    assert rows == [{"generation": "2", "number": "14", "energy": "-10.400000", "energy_per_atom": "-5.200000",
                     "num_atoms_all": "[2]", "expected_origin": "Random", "origin": "Heredity",
                     "parents": "[12, 13]"}]
    assert analyser.read_table("generation number energy\n1 2\n") is None


def test_two_listed_candidates_never_share_one_chain():
    analyser = load_analyser()
    chain = {"free_energy_ev": -10.5, "e0_ev": -10.504, "chain_start": 1.0}
    listed_rows = [{"generation": 1, "number": 7, "energy": -10.5, "natoms": 2},
                   {"generation": 1, "number": 8, "energy": -10.5, "natoms": 2}]
    result = analyser.associate(listed_rows, {3: chain}, {3: [chain]}, {}, {}, factor=1, gathered_factor=None)
    assert [r["outcome"] for r in result["rows"]] == ["contradicted", "contradicted"]
    assert all("associated with individuals [7, 8]" in r["reason"] for r in result["rows"])


# --- archive classifications (one table) ------------------------------------------------------------------


@pytest.mark.parametrize("scenario, archive_status, result", [
    (Scenario(), "verified", "PASSED"),
    (Scenario(archive_duplicate=True), "verified", "PASSED"),  # an identical repeat changes nothing
    (Scenario(archive_interrupted=True), "missing", "INCOMPLETE EVIDENCE"),
    (Scenario(archive_missing=((4, 3),)), "missing", "INCOMPLETE EVIDENCE"),  # evidence_write_failed
    (Scenario(archive=False), "missing", "INCOMPLETE EVIDENCE"),
    (Scenario(critical_line=True), "missing", "INCOMPLETE EVIDENCE"),  # the journal itself failed
    (Scenario(archive_conflict=True), "failed", "FAILED"),
    (Scenario(archive_tamper=((4, 3),)), "failed", "FAILED"),
])
def test_archive_states_have_the_intended_classification(tmp_path, scenario, archive_status, result):
    report = analyse(tmp_path, scenario)
    assert archive_check(report)["status"] == archive_status and report["result"] == result, (
        report["failed"], report["missing"])


def test_an_archive_write_failure_is_incomplete_evidence_never_a_scientific_rejection(tmp_path):
    report = analyse(tmp_path, Scenario(archive_missing=((4, 5),)))
    assert report["failed"] == [] and set(report["missing"]) == {"evidence_archive", "retention"}
    assert report["findings"]["candidates"]["completed_ladder"] == 21 and report["findings"]["candidates"]["rejected"] == {}
    assert status_of(report, "listed_energies") == "verified" and status_of(report, "chains") == "verified"
    detail = archive_check(report)["detail"]
    assert detail["write_failures"] and detail["missing_records"]


def test_a_record_committed_before_a_crash_is_not_counted_as_accepted(tmp_path):
    # The launcher committed step 5 of candidate 14 (verdict valid) and died before the done file;
    # the next launcher answered USPEX "lost" (121). USPEX did not list 14.
    report = analyse(tmp_path, Scenario(crash_before_done=((14, 5),)))
    detail = archive_check(report)["detail"]
    [undelivered] = detail["archived_but_not_delivered"]
    assert undelivered["done_status"] == "121" and undelivered["archived_status"] == 0
    assert archive_check(report)["status"] == "missing" and "evidence_archive" not in report["failed"]
    assert report["findings"]["candidates"]["completed_ladder"] == 20
    assert report["findings"]["candidates"]["rejected"]["14"]["classification"] == "not_delivered"
    assert report["findings"]["candidates"]["rejected"]["14"]["delivered_status"] == "121"
    assert status_of(report, "listed_energies") == "verified"  # 14 is not listed, nothing else is affected
    assert report["result"] == "FAILED" and "bridge" in report["failed"]  # the launcher crash itself


def test_a_candidate_listed_from_an_undelivered_record_fails(tmp_path):
    report = analyse(tmp_path, Scenario(crash_before_done=((14, 5),), listed_rejected={14: 5}))
    contradicted = {row["number"]: row["reason"] for row in listed(report)["contradicted"]}
    assert "stale" in contradicted[14] and "listed_energies" in report["failed"]


def test_an_ambiguous_generation_association_is_incomplete_evidence(tmp_path):
    report = analyse(tmp_path, Scenario(number_offset=100, same_energy=(3, 5), same_structure=(3, 5)))
    assert report["result"] == "INCOMPLETE EVIDENCE"
    rows = next(c for c in report["checks"] if c["check"] == "retention")["detail"]["inferred_generation_association"]
    assert {row["number"] for row in rows["rows"]}.isdisjoint({103, 105})


# --- the shared-node profile (3 x 16 ranks, 48 CPUs, --mem=32G) -------------------------------------------


def check_detail(report, name):
    return next(c for c in report["checks"] if c["check"] == name)["detail"]


def test_a_shared_node_run_passes_with_expectations_from_its_plan(tmp_path):
    report = analyse(tmp_path, Scenario(**SHARED_48), SHARED_EXPECTED)
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    allocation = check_detail(report, "allocation")
    assert allocation["ncpus"] == "48" and allocation["expected_ncpus"] == 48
    assert allocation["exclusive_requested"] is False
    memory = check_detail(report, "memory")
    assert memory["planned_job_memory_mb"] == 32768 and memory["sacct_alloc_mem_mb"] == 32768
    assert memory["expected_step_mem"] == f"{int(32768 * 0.9) // 3}M" == memory["step_mem"]
    assert check_detail(report, "parallel") == {"max_concurrent": 3, "limit": 3}
    pinning = check_detail(report, "pinning")
    assert pinning["interpretable"] == pinning["steps_ran"] and not pinning["overlaps"]


@pytest.mark.parametrize("scenario, expectation, check", [
    (Scenario(**SHARED_48), {}, "allocation"),  # a 48-CPU job is not the 192-CPU plan
    (Scenario(), SHARED_EXPECTED, "allocation"),  # nor the other way round
    (Scenario(**{**SHARED_48, "node_memory_mb": 16384, "alloc_mem": "16G"}), SHARED_EXPECTED, "memory"),
    (Scenario(**{**SHARED_48, "alloc_mem": "64G"}), SHARED_EXPECTED, "memory"),  # SLURM and the launcher disagree
    (Scenario(**{**SHARED_48, "parallel": 4}), SHARED_EXPECTED, "parallel"),  # more at once than the plan
    (Scenario(**{**SHARED_48, "pin": "shared"}), SHARED_EXPECTED, "pinning"),  # concurrent steps on the same cores
])
def test_a_resource_mismatch_with_the_shared_plan_fails(tmp_path, scenario, expectation, check):
    report = analyse(tmp_path, scenario, expectation)
    assert check in report["failed"] and report["result"] == "FAILED", report["failed"]


def test_partial_pinning_on_a_shared_node_is_incomplete_evidence(tmp_path):
    report = analyse(tmp_path, Scenario(**{**SHARED_48, "pin": "partial"}), SHARED_EXPECTED)
    assert report["result"] == "INCOMPLETE EVIDENCE" and report["missing"] == ["pinning"]
