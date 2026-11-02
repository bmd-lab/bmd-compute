"""Simulated end-to-end runs of the USPEX launch bridge (issue #66).

The real pieces are ``bmd_bridge.sh`` (run with the system ``sh``, which is
dash on Ubuntu, as in the USPEX image) and the host-side launcher. Faked:

* USPEX: ``fixtures/structure_search_bridge/fake_uspex.py`` makes CalcFolds,
  calls the bridge for each relaxation step (several structures at once) and
  reads OSZICAR and CONTCAR afterwards.
* ``srun``: ``fixtures/structure_search_bridge/fake_srun.py``, installed as
  ``srun`` on the step PATH. It records its command line, environment and
  start/end times, sleeps, and writes OSZICAR and CONTCAR.

No SLURM, Singularity or VASP is needed. In production the bridge variables
reach the container through ``SINGULARITYENV_*`` (see ``master.py``); here
the fake USPEX simply receives them in its environment.

This verifies the protocol and process handling, not cluster placement:
whether concurrent ``srun --exact`` steps get separate cores is checked by the
Phase 0 step test on POWER (#59).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from backend.structure_search.bridge import protocol
from backend.structure_search.bridge.launcher import Launcher, LauncherSettings, run_search
from backend.structure_search.bridge.master import bridge_script_path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "structure_search_bridge"
BRIDGE = str(bridge_script_path())


class Rig:
    """Temporary search directory, spool, fake srun and helpers for one test."""

    def __init__(self, tmp_path: Path, *, control: dict | None = None):
        self.root = tmp_path
        self.search = tmp_path / "search"
        self.spool = tmp_path / "search" / ".bmd_bridge"
        self.records = tmp_path / "srun_records"
        self.bin = tmp_path / "bin"
        for path in (self.search, self.records, self.bin):
            path.mkdir(parents=True)
        self.control_path = tmp_path / "control.json"
        self.set_control(control or {"sleep": 0.3})
        srun = self.bin / "srun"
        srun.write_text(f"#!{sys.executable}\n" + (FIXTURES / "fake_srun.py").read_text())
        srun.chmod(0o755)
        self._threads: list[threading.Thread] = []

    def set_control(self, control: dict) -> None:
        self.control_path.write_text(json.dumps(control))

    def step_environment(self, **extra) -> dict[str, str]:
        return {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "FAKE_SRUN_RECORDS": str(self.records),
            "FAKE_SRUN_CONTROL": str(self.control_path),
            **extra,
        }

    def settings(self, **overrides) -> LauncherSettings:
        values = dict(
            spool_dir=self.spool,
            search_dir=self.search,
            max_parallel=2,
            ranks=48,
            mem="200G",
            environment=self.step_environment(),
            poll_seconds=0.05,
            heartbeat_interval_seconds=0.2,
            kill_grace_seconds=2.0,
        )
        values.update(overrides)
        return LauncherSettings(**values)

    def bridge_environment(self, *, timeout: int = 0, heartbeat: int = 5) -> dict[str, str]:
        return {
            "PATH": "/usr/bin:/bin",
            "BMD_BRIDGE_SPOOL": str(self.spool),
            "BMD_BRIDGE_TIMEOUT_SECONDS": str(timeout),
            "BMD_BRIDGE_POLL_SECONDS": "0.05",
            "BMD_BRIDGE_HEARTBEAT_SECONDS": str(heartbeat),
        }

    def calcfold(self, name: str, step: int = 1) -> Path:
        folder = self.search / name
        folder.mkdir(exist_ok=True)
        (folder / "POSCAR").write_text(f"{name}\n1.0\n3 0 0\n0 3 0\n0 0 3\nSi\n1\nDirect\n0 0 0\n")
        (folder / "INCAR").write_text(f"STEP = {step}\n")
        return folder

    def start_bridge(self, folder: Path, step: int = 1, **environment) -> subprocess.Popen:
        return subprocess.Popen(
            ["sh", BRIDGE, str(step)],
            cwd=folder,
            env=self.bridge_environment(**environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def start_launcher_thread(self, settings: LauncherSettings) -> threading.Event:
        launcher = Launcher(settings)
        launcher.prepare()
        event = threading.Event()
        thread = threading.Thread(target=launcher.run, args=(event,), daemon=True)
        thread.start()
        self._threads.append(thread)
        self.launcher = launcher
        return event

    def join(self, event: threading.Event) -> None:
        event.set()
        for thread in self._threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

    def srun_records(self) -> list[dict]:
        return [json.loads(path.read_text()) for path in sorted(self.records.glob("*.json"))]

    def wait_for(self, condition, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.05)
        raise AssertionError("condition not reached in time")

    def done(self, request_id: str) -> dict[str, str]:
        text = (self.spool / protocol.DONE_DIR / f"{request_id}.done").read_text()
        return dict(line.split("=", 1) for line in text.splitlines())

    def done_files(self) -> list[dict[str, str]]:
        return [self.done(path.name[: -len(".done")]) for path in sorted((self.spool / "done").glob("*.done"))]

    def journal(self) -> list[dict]:
        return [json.loads(line) for line in (self.spool / protocol.JOURNAL_FILE).read_text().splitlines()]

    def assert_no_partial_files(self) -> None:
        assert list((self.spool / protocol.TMP_DIR).iterdir()) == []
        assert list((self.spool / protocol.CLAIMED_DIR).iterdir()) == []
        for line in (self.spool / protocol.JOURNAL_FILE).read_text().splitlines():
            json.loads(line)


def _max_overlap(records: list[dict]) -> int:
    events = []
    for record in records:
        events.append((record["start"], 1))
        events.append((record["end"], -1))
    best = current = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        current += change
        best = max(best, current)
    return best


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


# -- full search -----------------------------------------------------------------


def test_fake_search_runs_every_step_through_the_bridge_within_the_concurrency_limit(rig):
    settings = rig.settings(
        max_parallel=2,
        ranks_per_step={3: 96},
        environment=rig.step_environment(I_MPI_PIN_DOMAIN="core", I_MPI_DEBUG="0"),
    )
    status = run_search(
        [
            sys.executable, str(FIXTURES / "fake_uspex.py"),
            "--search-dir", str(rig.search), "--bridge", BRIDGE,
            "--structures", "5", "--steps", "3", "--parallel", "4",
        ],
        rig.bridge_environment(),
        rig.search,
        settings,
        master_log=rig.root / "master.log",
    )
    assert status == 0, (rig.root / "master.log").read_text()

    results = json.loads((rig.search / "fake_uspex_results.json").read_text())
    assert sorted(results) == [f"CalcFold{i}" for i in range(1, 6)]
    for name, result in results.items():
        assert result["exit_codes"] == [0, 0, 0]
        # USPEX read OSZICAR and CONTCAR after every step, in order.
        assert result["energies"] == [-10.0 * step - len(name) for step in (1, 2, 3)]
        assert result["final_poscar"] == f"{name} relaxed-step-1 relaxed-step-2 relaxed-step-3"

    records = rig.srun_records()
    assert len(records) == 15
    # USPEX had 4 relaxations waiting; the launcher ran at most 2, and did use 2.
    assert _max_overlap(records) == 2
    for record in records:
        step = int(record["key"].split(":")[1])
        ranks = "96" if step == 3 else "48"
        assert record["argv"] == ["--exact", "--mpi=pmi2", "-n", ranks, "--mem=200G", "vasp_std"]
        assert record["I_MPI_DEBUG"] == "4"
        assert record["I_MPI_PIN_DOMAIN"] is None
        assert record["XDG_CACHE_HOME"] is None
    # Never two steps in one CalcFold at once.
    for name in results:
        assert _max_overlap([r for r in records if Path(r["cwd"]).name == name]) == 1

    assert all(done["status"] == "0" for done in rig.done_files())
    assert len(rig.done_files()) == 15
    assert list((rig.spool / "requests").iterdir()) == []
    assert (rig.spool / protocol.STOPPED_FILE).exists()
    rig.assert_no_partial_files()
    events = [record["event"] for record in rig.journal()]
    assert events[0] == "started" and events[-1] == "stopped"
    assert "environment_removed" in events
    assert events.count("finished") == 15


def test_failed_step_returns_the_vasp_exit_status_and_others_continue(rig):
    rig.set_control({"sleep": 0.2, "fail": {"CalcFold2:2": 7}})
    status = run_search(
        [
            sys.executable, str(FIXTURES / "fake_uspex.py"),
            "--search-dir", str(rig.search), "--bridge", BRIDGE,
            "--structures", "3", "--steps", "3", "--parallel", "3",
        ],
        rig.bridge_environment(),
        rig.search,
        rig.settings(max_parallel=3),
        master_log=rig.root / "master.log",
    )
    assert status == 0
    results = json.loads((rig.search / "fake_uspex_results.json").read_text())
    assert results["CalcFold2"]["exit_codes"] == [0, 7]
    assert results["CalcFold1"]["exit_codes"] == [0, 0, 0]
    assert results["CalcFold3"]["exit_codes"] == [0, 0, 0]
    failed = [done for done in rig.done_files() if done["status"] != "0"]
    assert len(failed) == 1 and failed[0]["status"] == "7" and failed[0]["returncode"] == "7"
    rig.assert_no_partial_files()


def test_master_exit_status_is_returned(rig):
    status = run_search(
        [sys.executable, "-c", "raise SystemExit(3)"],
        rig.bridge_environment(),
        rig.search,
        rig.settings(),
        master_log=rig.root / "master.log",
    )
    assert status == 3
    assert (rig.spool / protocol.STOPPED_FILE).exists()


# -- timeout ---------------------------------------------------------------------


def test_bridge_timeout_exits_124_and_stops_the_vasp_step(rig):
    rig.set_control({"sleep": 0.2, "hang": ["CalcFold1:1"]})
    event = rig.start_launcher_thread(rig.settings())
    folder = rig.calcfold("CalcFold1")
    bridge = rig.start_bridge(folder, timeout=2)
    _, stderr = bridge.communicate(timeout=30)
    assert bridge.returncode == protocol.EXIT_TIMEOUT, stderr
    assert "timed out" in stderr

    # The launcher saw the cancellation, stopped srun and recorded the timeout.
    rig.wait_for(lambda: len(rig.done_files()) == 1)
    assert rig.done_files()[0]["status"] == str(protocol.EXIT_TIMEOUT)
    (record,) = rig.srun_records()
    assert record["terminated"] is True
    assert not _pid_alive(record["pid"])
    assert list((rig.spool / protocol.CANCEL_DIR).iterdir()) == []

    # The CalcFold is free again: a later step there runs normally.
    rig.set_control({"sleep": 0.2})
    second = rig.start_bridge(folder, step=2, timeout=20)
    second.communicate(timeout=30)
    assert second.returncode == 0
    rig.join(event)
    rig.assert_no_partial_files()


def test_bridge_timeout_while_queued_withdraws_the_request(rig):
    rig.set_control({"sleep": 0.2, "hang": ["CalcFold1:1"]})
    event = rig.start_launcher_thread(rig.settings(max_parallel=1))
    running = rig.start_bridge(rig.calcfold("CalcFold1"))
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    queued = rig.start_bridge(rig.calcfold("CalcFold2"), timeout=1)
    queued.communicate(timeout=30)
    assert queued.returncode == protocol.EXIT_TIMEOUT
    # It never reached srun and its request was withdrawn, not left behind.
    assert len(list((rig.spool / protocol.WITHDRAWN_DIR).iterdir())) == 1
    assert list((rig.spool / protocol.REQUESTS_DIR).iterdir()) == []
    rig.join(event)
    running.communicate(timeout=30)
    assert running.returncode == protocol.EXIT_STOPPED
    assert len(rig.srun_records()) == 1


# -- launcher stopping -----------------------------------------------------------


def _start_launcher_cli(rig: Rig, *extra: str) -> subprocess.Popen:
    env = {**rig.step_environment(), "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.Popen(
        [
            sys.executable, "-m", "backend.structure_search.bridge.launcher",
            "--spool", str(rig.spool), "--search-dir", str(rig.search),
            "--max-parallel", "2", "--ranks", "48", "--mem", "200G",
            "--poll-seconds", "0.05", "--heartbeat-interval-seconds", "0.2",
            "--kill-grace-seconds", "2", *extra,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_sigterm_stops_the_launcher_mid_step_cleanly(rig):
    rig.set_control({"hang": ["CalcFold1:1", "CalcFold2:1"]})
    launcher = _start_launcher_cli(rig)
    rig.wait_for(lambda: (rig.spool / protocol.HEARTBEAT_FILE).exists())
    bridges = [rig.start_bridge(rig.calcfold(f"CalcFold{i}")) for i in (1, 2, 3)]
    rig.wait_for(lambda: len(rig.srun_records()) == 2)

    launcher.send_signal(signal.SIGTERM)
    _, launcher_err = launcher.communicate(timeout=30)
    assert launcher.returncode == 0, launcher_err

    for bridge in bridges:
        bridge.communicate(timeout=30)
        assert bridge.returncode == protocol.EXIT_STOPPED
    # Both running steps were terminated and recorded as stopped; the queued
    # third request was withdrawn by its bridge, never started.
    records = rig.srun_records()
    assert len(records) == 2 and all(record["terminated"] for record in records)
    assert not any(_pid_alive(record["pid"]) for record in records)
    statuses = sorted(done["status"] for done in rig.done_files())
    assert statuses == [str(protocol.EXIT_STOPPED)] * 2
    assert len(list((rig.spool / protocol.WITHDRAWN_DIR).iterdir())) == 1
    assert (rig.spool / protocol.STOPPED_FILE).exists()
    rig.assert_no_partial_files()


def test_step_that_ignores_sigterm_is_killed_after_the_grace_period(rig):
    rig.set_control({"ignore_term": ["CalcFold1:1"]})
    event = rig.start_launcher_thread(rig.settings(kill_grace_seconds=0.5))
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"))
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    pid = rig.srun_records()[0]["pid"]
    started = time.monotonic()
    rig.join(event)
    assert 0.4 < time.monotonic() - started < 10
    assert not _pid_alive(pid)
    bridge.communicate(timeout=30)
    assert bridge.returncode == protocol.EXIT_STOPPED
    (done,) = rig.done_files()
    assert done["status"] == str(protocol.EXIT_STOPPED)
    assert done["returncode"] == str(-signal.SIGKILL)
    rig.assert_no_partial_files()


def test_bridge_started_after_the_launcher_stopped_exits_at_once(rig):
    event = rig.start_launcher_thread(rig.settings())
    rig.join(event)
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"))
    bridge.communicate(timeout=30)
    assert bridge.returncode == protocol.EXIT_STOPPED
    assert rig.srun_records() == []


def test_run_search_stop_event_stops_master_then_steps(rig):
    rig.set_control({"hang": ["CalcFold1:1", "CalcFold2:1"]})
    event = threading.Event()
    result: dict = {}

    def target():
        result["status"] = run_search(
            [
                sys.executable, str(FIXTURES / "fake_uspex.py"),
                "--search-dir", str(rig.search), "--bridge", BRIDGE,
                "--structures", "2", "--steps", "2", "--parallel", "2",
            ],
            rig.bridge_environment(),
            rig.search,
            rig.settings(),
            master_log=rig.root / "master.log",
            stop_event=event,
        )

    thread = threading.Thread(target=target)
    thread.start()
    rig.wait_for(lambda: len(rig.srun_records()) == 2)
    event.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert result["status"] == protocol.EXIT_STOPPED
    assert all(record["terminated"] for record in rig.srun_records())
    journal = [record["event"] for record in rig.journal()]
    assert journal.index("master_finished") < journal.index("stopped")
    rig.assert_no_partial_files()


# -- lost launcher, lost and malformed requests ------------------------------------


def test_bridge_gives_up_when_the_launcher_dies_and_a_new_launcher_recovers(rig):
    rig.set_control({"hang": ["CalcFold1:1"]})
    launcher = _start_launcher_cli(rig)
    rig.wait_for(lambda: (rig.spool / protocol.HEARTBEAT_FILE).exists())
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"), heartbeat=2)
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    orphan_pid = rig.srun_records()[0]["pid"]
    try:
        launcher.kill()  # SIGKILL: no clean shutdown, no done file
        launcher.communicate(timeout=30)
        _, stderr = bridge.communicate(timeout=30)
        assert bridge.returncode == protocol.EXIT_LAUNCHER_LOST, stderr
        assert "no launcher heartbeat" in stderr
        # The request stays claimed by the dead launcher; the bridge asked for
        # it to be cancelled.
        assert len(list((rig.spool / protocol.CLAIMED_DIR).iterdir())) == 1

        # A new launcher answers the leftover claim as "launcher lost" instead
        # of running VASP a second time in that CalcFold.
        fresh = Launcher(rig.settings())
        fresh.prepare()
        (done,) = rig.done_files()
        assert done["status"] == str(protocol.EXIT_LAUNCHER_LOST)
        assert list((rig.spool / protocol.CLAIMED_DIR).iterdir()) == []
        fresh.shutdown()
    finally:
        try:
            os.killpg(orphan_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_request_whose_bridge_died_is_still_run_and_answered(rig):
    """A request left behind by a bridge that was killed (SIGKILL, no trap)."""

    event = rig.start_launcher_thread(rig.settings())
    folder = rig.calcfold("CalcFold1")
    request = "id=lost-1\nstep=1\ncalcfold={}\ncreated={}\n".format(folder, int(time.time()))
    (rig.spool / "requests" / "lost-1.req").write_text(request)
    rig.wait_for(lambda: (rig.spool / "done" / "lost-1.done").exists())
    assert rig.done("lost-1")["status"] == "0"
    rig.join(event)
    rig.assert_no_partial_files()


def test_malformed_and_out_of_tree_requests_are_rejected(rig, tmp_path):
    event = rig.start_launcher_thread(rig.settings())
    requests = rig.spool / "requests"
    (requests / "bad-1.req").write_text("id=bad-1\nstep=one\n")
    (requests / "not a request").write_text("x")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    bridge = rig.start_bridge(outside)
    _, stderr = bridge.communicate(timeout=30)
    assert bridge.returncode == protocol.EXIT_REJECTED, stderr
    rig.wait_for(lambda: (rig.spool / "done" / "bad-1.done").exists())
    assert rig.done("bad-1")["status"] == str(protocol.EXIT_REJECTED)
    assert "malformed" in rig.done("bad-1")["reason"]
    assert list(requests.iterdir()) == []
    assert len(list((rig.spool / protocol.REJECTED_DIR).iterdir())) == 3
    assert rig.srun_records() == []
    rig.join(event)


def test_missing_step_command_is_answered_with_127(rig):
    event = rig.start_launcher_thread(rig.settings(command_template="no-such-srun -n {ranks} vasp_std"))
    bridge = rig.start_bridge(rig.calcfold("CalcFold1"))
    bridge.communicate(timeout=30)
    assert bridge.returncode == 127
    assert "could not be started" in rig.done_files()[0]["reason"]
    rig.join(event)
    rig.assert_no_partial_files()


# -- duplicates ------------------------------------------------------------------


def test_duplicate_request_id_runs_vasp_once(rig):
    rig.set_control({"sleep": 1.0})
    event = rig.start_launcher_thread(rig.settings())
    folder = rig.calcfold("CalcFold1")
    text = "id=dup-1\nstep=1\ncalcfold={}\ncreated={}\n".format(folder, int(time.time()))
    (rig.spool / "requests" / "dup-1.req").write_text(text)
    rig.wait_for(lambda: len(rig.srun_records()) == 1)
    # Same ID again while the first is running, and once more after it finished.
    (rig.spool / "requests" / "dup-1.req").write_text(text)
    rig.wait_for(lambda: not (rig.spool / "requests" / "dup-1.req").exists())
    rig.wait_for(lambda: (rig.spool / "done" / "dup-1.done").exists())
    (rig.spool / "requests" / "dup-1.req").write_text(text)
    rig.wait_for(lambda: not (rig.spool / "requests" / "dup-1.req").exists())
    rig.join(event)
    assert len(rig.srun_records()) == 1
    assert rig.done("dup-1")["status"] == "0"
    rejected = [r for r in rig.journal() if r["event"] == "rejected"]
    assert [r["reason"] for r in rejected] == ["duplicate request id"] * 2


def test_two_requests_for_one_calcfold_never_overlap(rig):
    rig.set_control({"sleep": 0.5})
    event = rig.start_launcher_thread(rig.settings(max_parallel=4))
    folder = rig.calcfold("CalcFold1")
    bridges = [rig.start_bridge(folder), rig.start_bridge(folder)]
    for bridge in bridges:
        bridge.communicate(timeout=30)
        assert bridge.returncode == 0
    rig.join(event)
    records = rig.srun_records()
    assert len(records) == 2
    assert _max_overlap(records) == 1


def test_concurrent_bridges_get_distinct_request_ids(rig):
    rig.set_control({"sleep": 0.05})
    event = rig.start_launcher_thread(rig.settings(max_parallel=8))
    bridges = [rig.start_bridge(rig.calcfold(f"CalcFold{i}")) for i in range(1, 13)]
    for bridge in bridges:
        bridge.communicate(timeout=60)
        assert bridge.returncode == 0
    rig.join(event)
    ids = [done["id"] for done in rig.done_files()]
    assert len(ids) == len(set(ids)) == 12
