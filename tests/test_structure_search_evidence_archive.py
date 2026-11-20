"""The launcher's append-only scientific evidence archive (``bridge/evidence_archive.py``).

Unit tests of the archive (commit, duplicates, conflicts, interrupted writes,
recovery, allowlist) and launcher runs through the real bridge script with
the fake USPEX and fake srun of the bridge end-to-end tests: every step's
record, the verdict as the checker returned it, structure identity along the
ladder, rejected candidates, CalcFolds removed as early as USPEX could
remove them, and an archive failure that must not change what USPEX is told.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from backend.structure_search.bridge import evidence_archive as ea
from backend.structure_search.bridge import protocol
from backend.structure_search.bridge.launcher import Launcher, run_search

from test_structure_search_bridge_end_to_end_fake import BRIDGE, FIXTURES, Rig

POSCAR = b"in\n1.0\n3 0 0\n0 3 0\n0 0 3\nSi\n1\nDirect\n0 0 0\n"
CONTCAR = b"out\n1.0\n3.1 0 0\n0 3.1 0\n0 0 3.1\nSi\n1\nDirect\n0 0 0\n"


def _record(request_id="s5-1-2-abc", step=5, classification="valid", contcar=CONTCAR, **changes):
    input_facts = {"file": "POSCAR", "present": True, "bytes": len(POSCAR), "archived": True,
                   "sha256": hashlib.sha256(POSCAR).hexdigest()}
    output_facts = {"file": "CONTCAR", "present": True, "bytes": len(contcar), "archived": True,
                    "sha256": hashlib.sha256(contcar).hexdigest(), "withheld": False}
    record = ea.build_record(
        request_id=request_id, step=step, calcfold=Path("/s/Calculation/Calcfold_7_5"), created=1, slot=0,
        argv=["srun", "vasp_std"], started=10.0, finished=40.0, returncode=0, status=0, reason="",
        validity={"classification": classification, "reasons": [], "evidence": {"free_energy_ev": -10.5}},
        validity_record_line='{"classification": "valid"}', step_log={"file": "x.log", "bytes": 1, "sha256": "0"},
        input_structure=input_facts, output_structure=output_facts)
    record.update(changes)
    return record


def _files(contcar=CONTCAR):
    return {"POSCAR": POSCAR, "CONTCAR": contcar}


# --- the archive -------------------------------------------------------------------------------------------


def test_a_committed_record_is_whole_indexed_and_bound_to_its_structures(tmp_path):
    result = ea.commit(tmp_path, _record(), _files())
    assert result["outcome"] == "committed"
    folder = tmp_path / "steps" / "s5-1-2-abc"
    assert sorted(p.name for p in folder.iterdir()) == ["CONTCAR", "POSCAR", "record.json"]
    check = ea.verify_record_directory(folder)
    assert check["ok"] and check["record_sha256"] == result["record_sha256"]
    record = check["record"]
    assert record["final_relaxed_structure"] is True and record["generation"] is None
    assert record["calcfold_name_fields"] == {"first": 7, "second": 5}
    assert record["output_structure"]["sha256"] == hashlib.sha256(CONTCAR).hexdigest()
    index, torn = ea.read_index(tmp_path)
    assert torn == 0 and index == [{"id": "s5-1-2-abc", "record_sha256": result["record_sha256"],
                                    "event": "committed", "time": index[0]["time"]}]
    assert list((tmp_path / ".staging").iterdir()) == []


def test_only_a_valid_step_5_has_a_final_relaxed_structure(tmp_path):
    assert ea.build_record(**{**_kwargs(), "step": 4})["final_relaxed_structure"] is False
    rejected = _kwargs()
    rejected["validity"] = {"classification": "invalid_convergence", "reasons": ["x"]}
    assert ea.build_record(**rejected)["final_relaxed_structure"] is False


def _kwargs():
    return dict(request_id="s5-1-2-abc", step=5, calcfold=Path("/s/c"), created=1, slot=0, argv=[], started=1.0,
                finished=2.0, returncode=0, status=0, reason="", validity={"classification": "valid"},
                validity_record_line=None, step_log=None, input_structure={},
                output_structure={"archived": True, "sha256": "x"})


def test_an_identical_repeat_is_a_duplicate_and_writes_nothing(tmp_path):
    first = ea.commit(tmp_path, _record(), _files())
    before = sorted(str(p) for p in tmp_path.rglob("*")), (tmp_path / "index.jsonl").read_bytes()
    second = ea.commit(tmp_path, _record(), _files())
    assert second == {"outcome": "duplicate", "record_sha256": first["record_sha256"],
                      "path": str(tmp_path / "steps" / "s5-1-2-abc")}
    assert (sorted(str(p) for p in tmp_path.rglob("*")), (tmp_path / "index.jsonl").read_bytes()) == before


def test_a_different_record_for_a_committed_step_is_kept_aside_never_replacing_it(tmp_path):
    original = ea.commit(tmp_path, _record(), _files())
    other = CONTCAR.replace(b"3.1", b"3.2")
    conflict = ea.commit(tmp_path, _record(contcar=other), _files(other))
    assert conflict["outcome"] == "conflict" and conflict["committed_record_sha256"] == original["record_sha256"]
    assert (tmp_path / "steps" / "s5-1-2-abc" / "CONTCAR").read_bytes() == CONTCAR
    assert Path(conflict["path"]).parent.name == "conflicts" and (Path(conflict["path"]) / "CONTCAR").read_bytes() == other
    repeat = ea.commit(tmp_path, _record(contcar=other), _files(other))
    assert repeat["outcome"] == "duplicate" and repeat["path"] == conflict["path"]
    assert len(list((tmp_path / "conflicts").iterdir())) == 1
    assert [e["event"] for e in ea.read_index(tmp_path)[0]] == ["committed"]


def test_the_archive_refuses_unlisted_files_and_unmatched_hashes(tmp_path):
    with pytest.raises(ea.EvidenceError, match="allowlist"):
        ea.commit(tmp_path, _record(), {**_files(), "OUTCAR": b"x"})
    with pytest.raises(ea.EvidenceError, match="does not match"):
        ea.commit(tmp_path, _record(), {"POSCAR": POSCAR, "CONTCAR": b"tampered"})
    with pytest.raises(ea.EvidenceError, match="invalid request id"):
        ea.commit(tmp_path, _record(request_id="../x"), _files())
    assert not (tmp_path / "steps").exists() or list((tmp_path / "steps").iterdir()) == []


def test_an_interrupted_write_is_detectable_recoverable_and_never_completed(tmp_path, monkeypatch):
    real = ea._write_synced

    def dies_before_the_record(path, data):
        if path.name == "record.json":
            raise OSError("killed")
        real(path, data)

    monkeypatch.setattr(ea, "_write_synced", dies_before_the_record)
    with pytest.raises(OSError):
        ea.commit(tmp_path, _record(), _files())
    monkeypatch.undo()
    assert not (tmp_path / "steps" / "s5-1-2-abc").exists()
    [left] = list((tmp_path / ".staging").iterdir())
    assert not (left / "record.json").exists()
    report = ea.recover(tmp_path)
    assert report["interrupted"] == [left.name] and report["indexed"] == []
    assert (tmp_path / "interrupted" / left.name / "POSCAR").read_bytes() == POSCAR  # kept, not deleted
    assert list((tmp_path / "steps").iterdir()) == [] and ea.read_index(tmp_path) == ([], 0)  # nothing invented
    assert ea.commit(tmp_path, _record(), _files())["outcome"] == "committed"  # the step can still be recorded


def test_a_record_committed_without_its_index_line_is_indexed_on_recovery_only_if_whole(tmp_path):
    ea.commit(tmp_path, _record(), _files())
    ea.commit(tmp_path, _record(request_id="s5-1-2-def"), _files())
    (tmp_path / "index.jsonl").write_text("")  # the launcher died after the rename, before the index line
    (tmp_path / "steps" / "s5-1-2-def" / "CONTCAR").write_bytes(b"changed afterwards")
    report = ea.recover(tmp_path)
    assert report["indexed"] == ["s5-1-2-abc"]
    assert report["unverifiable"] == [{"id": "s5-1-2-def", "problems": ["CONTCAR does not match its recorded hash"]}]
    assert [(e["id"], e["event"]) for e in ea.read_index(tmp_path)[0]] == [("s5-1-2-abc", "indexed_on_recovery")]
    assert ea.recover(tmp_path)["indexed"] == []  # idempotent


def test_a_torn_index_line_is_counted_not_trusted(tmp_path):
    ea.commit(tmp_path, _record(), _files())
    with open(tmp_path / "index.jsonl", "a") as handle:
        handle.write('{"id": "s5-1-2-x", "record_sh')
    index, torn = ea.read_index(tmp_path)
    assert len(index) == 1 and torn == 1


def test_a_large_structure_is_hashed_not_copied(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "MAX_STRUCTURE_BYTES", 10)
    (tmp_path / "CONTCAR").write_bytes(CONTCAR)
    facts, data = ea.read_structure(tmp_path / "CONTCAR", name="CONTCAR")
    assert data is None and facts["archived"] is False and facts["sha256"] == hashlib.sha256(CONTCAR).hexdigest()
    facts, data = ea.read_structure(tmp_path / "absent", name="CONTCAR")
    assert data is None and facts == {"file": "CONTCAR", "source": "absent", "present": False, "reason": "missing"}


def test_the_archive_module_uses_only_the_standard_library():
    text = Path(ea.__file__).read_text()
    for name in ("pymatgen", "numpy", "jobflow", "atomate2"):
        assert f"import {name}" not in text and f"from {name}" not in text


# --- through the launcher ----------------------------------------------------------------------------------


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def _search(rig, *, steps=5, structures=2, remove=False, **settings):
    argv = [sys.executable, str(FIXTURES / "fake_uspex.py"), "--search-dir", str(rig.search), "--bridge", BRIDGE,
            "--structures", str(structures), "--steps", str(steps), "--parallel", "2"]
    if remove:
        argv.append("--remove-calcfolds")
    status = run_search(argv, rig.bridge_environment(), rig.search, rig.settings(max_parallel=2, **settings),
                        master_log=rig.root / "master.log")
    assert status == 0, (rig.root / "master.log").read_text()
    return json.loads((rig.search / "fake_uspex_results.json").read_text())


def _records(rig):
    root = rig.search / "bmd_evidence" / "steps"
    return {path.name: ea.verify_record_directory(path) for path in sorted(root.iterdir())}


def test_every_step_is_archived_with_its_original_verdict_and_structures(rig):
    _search(rig)
    records = _records(rig)
    journal = rig.journal()
    started = [e for e in journal if e["event"] == "started_step"]
    assert set(records) == {e["id"] for e in started} and len(records) == 10
    assert all(check["ok"] for check in records.values())
    evidence_events = {e["id"]: e for e in journal if e["event"] == "evidence"}
    for done in rig.done_files():
        assert done["evidence"] == records[done["id"]]["record_sha256"] == evidence_events[done["id"]]["record_sha256"]
        assert evidence_events[done["id"]]["outcome"] == "committed"
    index, torn = ea.read_index(rig.search / "bmd_evidence")
    assert torn == 0 and {(e["id"], e["record_sha256"]) for e in index} == {
        (i, c["record_sha256"]) for i, c in records.items()}
    for name in ("CalcFold1", "CalcFold2"):
        lines = (rig.search / name / "bmd_step_validity.jsonl").read_text().splitlines()
        chain = sorted((c["record"] for c in records.values() if c["record"]["calcfold_name"] == name),
                       key=lambda r: r["step"])
        assert [r["step"] for r in chain] == [1, 2, 3, 4, 5]
        for line, record in zip(lines, chain):
            # the verdict as the checker returned it, bound byte for byte to the CalcFold's own line
            assert record["validity_record_line_sha256"] == hashlib.sha256((line + "\n").encode()).hexdigest()
            assert record["validity"] == json.loads(line)
            assert record["validity"]["evidence"]["converged_electronic"] is True
            assert record["launch"]["status"] == 0 and record["generation"] is None
        # structure identity along the ladder: step k + 1 started from step k's CONTCAR (USPEX copies it)
        for before, after in zip(chain, chain[1:]):
            assert after["input_structure"]["sha256"] == before["output_structure"]["sha256"]
        assert [r["final_relaxed_structure"] for r in chain] == [False] * 4 + [True]
        final = rig.search / "bmd_evidence" / "steps" / chain[-1]["id"] / "CONTCAR"
        assert hashlib.sha256(final.read_bytes()).hexdigest() == chain[-1]["output_structure"]["sha256"]
        assert final.read_bytes() == (rig.search / name / "CONTCAR").read_bytes()
    archived = {p.name for p in (rig.search / "bmd_evidence").rglob("*") if p.is_file()}
    assert archived <= {"record.json", "POSCAR", "CONTCAR", "index.jsonl"}


def test_candidates_survive_calcfolds_removed_as_early_as_uspex_could(rig):
    results = _search(rig, remove=True)
    assert all(r["exit_codes"] == [0] * 5 for r in results.values())
    assert not (rig.search / "CalcFold1").exists() and not (rig.search / "CalcFold2").exists()
    records = _records(rig)
    assert len(records) == 10 and all(check["ok"] for check in records.values())
    finals = [c for c in records.values() if c["record"]["final_relaxed_structure"]]
    assert len(finals) == 2
    for check in finals:
        folder = rig.search / "bmd_evidence" / "steps" / check["record"]["id"]
        assert b"relaxed-step-5" in (folder / "CONTCAR").read_bytes()
        assert check["record"]["validity"]["classification"] == "valid"
        assert check["record"]["validity"]["evidence"]["free_energy_ev"] is not None


def test_a_rejected_candidate_keeps_its_reasons_evidence_and_withheld_structure(rig):
    rig.set_control({"sleep": 0.05, "variant": {"CalcFold1:4": "nsw"}})
    results = _search(rig, remove=True)
    assert results["CalcFold1"]["exit_codes"] == [0, 0, 0, protocol.EXIT_INVALID_CONVERGENCE]
    chain = sorted((c["record"] for c in _records(rig).values() if c["record"]["calcfold_name"] == "CalcFold1"),
                   key=lambda r: r["step"])
    assert [r["step"] for r in chain] == [1, 2, 3, 4]
    rejected = chain[-1]
    assert rejected["validity"]["classification"] == "invalid_convergence"
    assert "ionic relaxation not converged" in rejected["validity"]["reasons"][0]
    assert rejected["validity"]["evidence"]["converged_ionic"] is False
    assert rejected["launch"]["status"] == protocol.EXIT_INVALID_CONVERGENCE
    assert rejected["output_structure"]["withheld"] is True and rejected["output_structure"]["archived"] is True
    withheld = next(f for f in rejected["validity"]["files"] if f["name"] == "CONTCAR")
    assert withheld["sha256"] == rejected["output_structure"]["sha256"]
    assert rejected["final_relaxed_structure"] is False
    folder = rig.search / "bmd_evidence" / "steps" / rejected["id"]
    assert sorted(p.name for p in folder.iterdir()) == ["CONTCAR", "POSCAR", "record.json"]  # no OUTCAR/vasprun


def test_a_failed_archive_is_reported_and_never_changes_what_uspex_is_told(rig):
    rig.search.mkdir(exist_ok=True)
    (rig.search / "bmd_evidence").write_text("not a directory")
    results = _search(rig, steps=2, structures=1)
    assert results["CalcFold1"]["exit_codes"] == [0, 0]
    assert all(done["status"] == "0" and "evidence" not in done for done in rig.done_files())
    failures = [e for e in rig.journal() if e["event"] == "evidence_write_failed"]
    assert len(failures) == 2


def test_a_new_launcher_sets_aside_an_interrupted_write_and_says_so(rig):
    root = rig.search / "bmd_evidence"
    (root / ".staging" / "s1-1-1-dead-99-1").mkdir(parents=True)
    (root / ".staging" / "s1-1-1-dead-99-1" / "POSCAR").write_bytes(POSCAR)
    launcher = Launcher(rig.settings())
    launcher.prepare()
    launcher.shutdown()
    [event] = [e for e in rig.journal() if e["event"] == "evidence_recovered"]
    assert event["interrupted"] == ["s1-1-1-dead-99-1"]
    assert (root / "interrupted" / "s1-1-1-dead-99-1" / "POSCAR").exists()
    assert not (root / "steps").exists() or list((root / "steps").iterdir()) == []


def test_an_abandoned_step_is_recorded_without_reading_its_output(rig):
    rig.set_control({"sleep": 0.05, "hang": ["CalcFold1:1"]})
    launcher = Launcher(rig.settings(kill_grace_seconds=0.2))
    launcher.prepare()
    folder = rig.calcfold("CalcFold1")
    bridge = rig.start_bridge(folder)
    rig.wait_for(lambda: launcher.poll_once() or launcher.running_ids)
    launcher.shutdown()
    bridge.communicate(timeout=30)
    [check] = _records(rig).values()
    record = check["record"]
    assert check["ok"] and record["validity"]["classification"] == "execution_failed"
    assert record["launch"]["status"] == protocol.EXIT_STOPPED
    assert record["input_structure"]["archived"] is True


def test_the_launcher_writes_the_record_before_the_done_file(rig, monkeypatch):
    order = []
    real_commit, real_write = ea.commit, protocol.write_atomically

    def commit(*args, **kwargs):
        order.append("evidence")
        return real_commit(*args, **kwargs)

    def write_atomically(spool, target, text):
        if target.parent.name == "done":
            order.append("done")
        return real_write(spool, target, text)

    monkeypatch.setattr(ea, "commit", commit)
    monkeypatch.setattr(protocol, "write_atomically", write_atomically)
    _search(rig, steps=2, structures=1)
    assert order == ["evidence", "done"] * 2


def test_the_launcher_and_archive_are_in_the_runtime_package():
    from backend.runtime_package import runtime_package_relative_paths

    paths = runtime_package_relative_paths()
    assert "structure_search/bridge/evidence_archive.py" in paths
    assert "structure_search/bridge/launcher.py" in paths


def test_a_crash_after_the_record_and_before_the_done_file_is_answered_as_lost(rig, monkeypatch):
    # The record is committed; the launcher dies before publishing the done file. The next launcher
    # answers USPEX "lost" (121); the archived verdict (valid, status 0) was never delivered.
    launcher = Launcher(rig.settings())
    launcher.prepare()
    folder = rig.calcfold("CalcFold1")
    bridge = rig.start_bridge(folder, timeout=60)

    def crash(*args, **kwargs):
        raise SystemExit("launcher killed")

    rig.wait_for(lambda: launcher.poll_once() or launcher.running_ids)
    monkeypatch.setattr(launcher, "_write_done", crash)
    with pytest.raises(SystemExit):
        rig.wait_for(lambda: launcher.poll_once() or False, timeout=20)
    [check] = _records(rig).values()
    assert check["ok"] and check["record"]["launch"]["status"] == 0
    assert check["record"]["validity"]["classification"] == "valid"
    request_id = check["record"]["id"]
    assert not (rig.spool / "done" / f"{request_id}.done").exists()
    assert not [e for e in rig.journal() if e["event"] == "finished"]

    Launcher(rig.settings()).prepare()  # a new launcher on the same spool
    assert rig.done(request_id)["status"] == str(protocol.EXIT_LAUNCHER_LOST)
    assert "evidence" not in rig.done(request_id)
    bridge.communicate(timeout=30)
    assert bridge.returncode == protocol.EXIT_LAUNCHER_LOST
    [again] = _records(rig).values()
    assert again["record_sha256"] == check["record_sha256"]  # the record is kept, unchanged


def test_a_journal_that_cannot_be_written_is_shouted_and_never_blocks_the_answer(rig, capfd):
    (rig.search / "bmd_evidence").write_text("not a directory")  # the archive write fails
    launcher = Launcher(rig.settings())
    launcher.prepare()
    real = launcher.journal

    def journal(event, **fields):
        if event.startswith("evidence"):
            raise OSError("journal file system full")
        return real(event, **fields)

    launcher.journal = journal
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"), timeout=60)
    rig.wait_for(lambda: launcher.poll_once() or bool(list((rig.spool / "done").glob("*.done"))))
    launcher.shutdown()
    bridge.communicate(timeout=30)
    assert bridge.returncode == 0  # the validity decision (valid) reached USPEX unchanged
    [done] = rig.done_files()
    assert done["status"] == "0" and done["validity"] == "valid" and "evidence" not in done
    err = capfd.readouterr().err
    assert "BMD-LAUNCHER-CRITICAL: could not journal evidence_write_failed" in err and done["id"] in err
