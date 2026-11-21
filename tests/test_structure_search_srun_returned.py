"""The launcher's ``srun_returned`` event: when it observes a step's process exit, before the validity check.

Real launcher and bridge, fake ``srun`` (see test_structure_search_bridge_end_to_end_fake.py). The event
records the launcher's observation, on its own clock; it is not the moment SLURM released the step's CPUs.
"""

from __future__ import annotations

import json
import signal
import sys

import pytest

from backend.structure_search.bridge import evidence_archive, protocol
from backend.structure_search.bridge.launcher import Launcher, run_search

from test_structure_search_bridge_end_to_end_fake import BRIDGE, FIXTURES, Rig, _start_launcher_cli

RETURNED_FIELDS = ("observed_at", "observed_monotonic", "monotonic_seconds", "returncode", "stop_requested",
                   "killed", "launcher_instance")


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def _events(journal, request_id):
    return [(index, event) for index, event in enumerate(journal) if event.get("id") == request_id]


def _only(journal, name, request_id=None):
    found = [e for e in journal if e.get("event") == name and (request_id is None or e.get("id") == request_id)]
    assert len(found) == 1, found
    return found[0]


def _record(rig, request_id):
    path = rig.search / evidence_archive.ARCHIVE_DIR / evidence_archive.STEPS_DIR / request_id / "record.json"
    return json.loads(path.read_text())


def _check_timing(rig, journal, request_id, srun_record=None):
    """Shared assertions: ordering, clocks, and the archived copy."""

    order = [event["event"] for _, event in _events(journal, request_id)]
    assert order.index("started_step") < order.index("srun_returned")
    for later in ("validity", "evidence", "finished"):
        if later in order:
            assert order.index("srun_returned") < order.index(later), order
    started = _only(journal, "started_step", request_id)
    returned = _only(journal, "srun_returned", request_id)
    finished = _only(journal, "finished", request_id)
    popen = started["popen_time"]
    assert popen <= started["time"] + 0.0005  # just before the process was started
    assert popen <= returned["observed_at"] <= finished["time"] + 0.0005
    assert returned["monotonic_seconds"] >= 0
    assert returned["observed_monotonic"] >= started["popen_monotonic"]
    assert abs(returned["monotonic_seconds"] - (returned["observed_monotonic"] - started["popen_monotonic"])) < 1e-5
    assert abs((returned["observed_at"] - popen) - returned["monotonic_seconds"]) < 0.5
    assert returned["launcher_instance"] == started["launcher_instance"] and len(started["launcher_instance"]) == 32
    assert returned["returncode"] == finished["returncode"]
    if srun_record is not None and srun_record.get("end") is not None:
        # The fake srun's own last moment precedes the launcher's observation of its exit.
        assert srun_record["end"] <= returned["observed_at"]
    record = _record(rig, request_id)
    assert record["launch"]["popen"] == popen
    assert record["launch"]["popen_monotonic"] == started["popen_monotonic"]
    assert record["launch"]["returned"] == {k: returned[k] for k in RETURNED_FIELDS}
    return returned


def test_successful_steps_record_srun_returned_before_their_validity_check(rig):
    status = run_search(
        [sys.executable, str(FIXTURES / "fake_uspex.py"), "--search-dir", str(rig.search), "--bridge", BRIDGE,
         "--structures", "2", "--steps", "3", "--parallel", "2"],
        rig.bridge_environment(), rig.search, rig.settings(max_parallel=2), master_log=rig.root / "master.log")
    assert status == 0
    journal = rig.journal()
    ids = [e["id"] for e in journal if e.get("event") == "started_step"]
    assert len(ids) == 6
    by_pid = {r["pid"]: r for r in rig.srun_records()}
    for request_id in ids:
        pid = _only(journal, "started_step", request_id)["pid"]
        returned = _check_timing(rig, journal, request_id, by_pid.get(pid))
        assert returned == {**returned, "returncode": 0, "stop_requested": False, "killed": False}
        assert _only(journal, "validity", request_id)["classification"] == "valid"
    rig.assert_no_partial_files()


def test_a_failed_step_records_its_return_code_before_the_verdict(rig):
    rig.set_control({"sleep": 0.2, "fail": {"CalcFold2:2": 7}})
    status = run_search(
        [sys.executable, str(FIXTURES / "fake_uspex.py"), "--search-dir", str(rig.search), "--bridge", BRIDGE,
         "--structures", "2", "--steps", "2", "--parallel", "2"],
        rig.bridge_environment(), rig.search, rig.settings(max_parallel=2), master_log=rig.root / "master.log")
    assert status == 0
    journal = rig.journal()
    failed = [e for e in journal if e.get("event") == "finished" and e["returncode"] == 7]
    assert len(failed) == 1
    returned = _check_timing(rig, journal, failed[0]["id"])
    assert returned["returncode"] == 7 and not returned["stop_requested"] and not returned["killed"]
    assert _only(journal, "validity", failed[0]["id"])["classification"] == "execution_failed"


def test_a_step_stopped_by_its_bridge_timeout_is_marked_stop_requested(rig):
    rig.set_control({"hang": ["CalcFold1:1"]})
    event = rig.start_launcher_thread(rig.settings())
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"), timeout=2)
    bridge.communicate(timeout=30)
    assert bridge.returncode == protocol.EXIT_TIMEOUT
    rig.wait_for(lambda: len(rig.done_files()) == 1)
    rig.join(event)
    journal = rig.journal()
    request_id = _only(journal, "finished")["id"]
    returned = _check_timing(rig, journal, request_id)
    assert returned["stop_requested"] is True and returned["killed"] is False


def test_a_step_killed_after_the_grace_period_is_marked_killed(rig):
    rig.set_control({"ignore_term": ["CalcFold1:1"]})
    event = rig.start_launcher_thread(rig.settings(kill_grace_seconds=0.5))
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"))
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    rig.join(event)
    bridge.communicate(timeout=30)
    journal = rig.journal()
    request_id = _only(journal, "finished")["id"]
    returned = _check_timing(rig, journal, request_id)
    assert returned["returncode"] == -signal.SIGKILL
    assert returned["stop_requested"] is True and returned["killed"] is True


def test_steps_stopped_with_the_launcher_by_sigterm_are_recorded(rig):
    rig.set_control({"hang": ["CalcFold1:1"]})
    launcher = _start_launcher_cli(rig)
    rig.wait_for(lambda: (rig.spool / protocol.HEARTBEAT_FILE).exists())
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"))
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    launcher.send_signal(signal.SIGTERM)
    launcher.communicate(timeout=30)
    bridge.communicate(timeout=30)
    journal = rig.journal()
    request_id = _only(journal, "finished")["id"]
    returned = _check_timing(rig, journal, request_id)
    assert returned["stop_requested"] is True


def test_a_journal_failure_for_srun_returned_never_changes_the_step(rig, monkeypatch, capsys):
    original = Launcher.journal

    def failing(self, event, **fields):
        if event == "srun_returned":
            raise OSError("journal full")
        return original(self, event, **fields)

    monkeypatch.setattr(Launcher, "journal", failing)
    event = rig.start_launcher_thread(rig.settings())
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"))
    bridge.communicate(timeout=30)
    rig.join(event)
    assert bridge.returncode == 0
    (done,) = rig.done_files()
    assert done["status"] == "0" and done["validity"] == "valid"
    assert "could not journal srun_returned" in capsys.readouterr().err
    journal = rig.journal()
    assert not [e for e in journal if e.get("event") == "srun_returned"]
    # The archived record still carries the observation (it is in memory, not read back from the journal).
    record = _record(rig, _only(journal, "finished")["id"])
    assert record["launch"]["returned"]["returncode"] == 0


def test_the_step_command_and_environment_are_unchanged(rig):
    settings = rig.settings()
    launcher = Launcher(settings)
    launcher.prepare()
    folder = rig.calcfold("CalcFold1")
    bridge = rig.start_bridge(folder)
    rig.wait_for(lambda: list((rig.spool / protocol.REQUESTS_DIR).iterdir()))
    launcher._start_requests()
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    (srun,) = rig.srun_records()
    started = _only(rig.journal(), "started_step")
    assert started["argv"] == settings.argv_for_step(1)
    assert srun["argv"] == settings.argv_for_step(1)[1:]  # what the fake srun received after its own name
    assert set(started) == {"time", "event", "id", "step", "slot", "calcfold", "argv", "environment_set", "pid",
                            "popen_time", "popen_monotonic", "launcher_instance"}
    rig.wait_for(lambda: (launcher._reap() or True) and not launcher.running_ids)
    launcher.shutdown()
    bridge.communicate(timeout=30)


def test_a_step_abandoned_at_shutdown_has_no_srun_returned(rig, monkeypatch):
    import os

    from backend.structure_search.bridge import launcher as launcher_module

    # The step ignores SIGTERM and no signal reaches it: its process never exits while the launcher stops.
    rig.set_control({"ignore_term": ["CalcFold1:1"], "hang": ["CalcFold1:1"]})
    monkeypatch.setattr(launcher_module, "_signal_group", lambda process, signum: None)
    event = rig.start_launcher_thread(rig.settings(kill_grace_seconds=0.2))
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"))
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    pid = rig.srun_records()[0]["pid"]
    try:
        rig.join(event)
    finally:
        os.kill(pid, signal.SIGKILL)
    bridge.communicate(timeout=30)
    journal = rig.journal()
    request_id = _only(journal, "abandoned")["id"]
    assert not [e for e in journal if e.get("event") == "srun_returned"]
    assert _only(journal, "finished", request_id)["returncode"] is None
    record = _record(rig, request_id)
    assert record["launch"]["returned"] is None and record["launch"]["popen"] is not None


def test_steps_of_one_launcher_are_ordered_on_its_monotonic_clock_and_in_its_journal(rig):
    """A later step on a freed slot is started after the earlier step's srun_returned, on every measure."""

    status = run_search(
        [sys.executable, str(FIXTURES / "fake_uspex.py"), "--search-dir", str(rig.search), "--bridge", BRIDGE,
         "--structures", "1", "--steps", "3", "--parallel", "1"],
        rig.bridge_environment(), rig.search, rig.settings(max_parallel=1), master_log=rig.root / "master.log")
    assert status == 0
    journal = rig.journal()
    starts = [(i, e) for i, e in enumerate(journal) if e.get("event") == "started_step"]
    returns = {e["id"]: (i, e) for i, e in enumerate(journal) if e.get("event") == "srun_returned"}
    assert len(starts) == 3 and len({e["launcher_instance"] for _, e in starts}) == 1
    for (_, earlier), (later_index, later) in zip(starts, starts[1:]):
        returned_index, returned = returns[earlier["id"]]
        assert returned_index < later_index
        assert returned["observed_monotonic"] < later["popen_monotonic"]
        assert returned["observed_at"] < later["popen_time"]
