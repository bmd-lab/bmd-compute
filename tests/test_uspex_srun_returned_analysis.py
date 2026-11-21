"""The execution analyser's use of the launcher's ``srun_returned`` event.

``srun_returned`` is when the launcher observed a step's ``srun`` exit, on its own clock; it is not the moment
SLURM released the step's CPUs. It may settle sequential CPU reuse; it must never make a verdict more
favourable when it is missing, inconsistent or contradicted, and older evidence must be analysed as before.
"""

from __future__ import annotations

import pytest

from test_uspex_execution_acceptance import SHARED_EXPECTED, analyse, check_detail, load_analyser
from test_uspex_execution_pinning import rows, sacct, step
from uspex_execution_fakes import SHARED_48, Scenario

ANALYSER = load_analyser()


INSTANCE = "launcher-1"
OFFSET = 1000.0  # the launcher's monotonic clock: wall-clock time minus a constant, unless a test moves the wall clock


def proc(popen, observed, *, started_index, returned_index, instance=INSTANCE, wall_shift=0.0):
    """Usable timing as launch_timing returns it (monotonic = wall - OFFSET; wall_shift moves only the wall clock)."""

    return {"popen": popen + wall_shift, "observed": observed + wall_shift, "popen_monotonic": popen - OFFSET,
            "observed_monotonic": observed - OFFSET, "instance": instance, "started_index": started_index,
            "returned_index": returned_index}


def timed(name, start, end, pin_rows, process, returncode=0):
    return {**step(name, start, end, pin_rows, returncode), "process": process}


def assess(steps, slurm):
    return ANALYSER.assess_pinning(steps, slurm, ranks=16)


SAME_SECOND = sacct((101, 190), (190, 290))  # SLURM's whole seconds meet: sacct alone settles nothing


def pair(a_process, b_process, slurm=SAME_SECOND):
    steps = [timed("a", 100, 260, rows(0), a_process), timed("b", 189.5, 300, rows(0, pid=2000), b_process)]
    return assess(steps, slurm)


# --- assess_pinning with launcher timing -----------------------------------------------------------------------------

def test_cpu_reuse_settled_on_one_launchers_monotonic_clock_wall_clock_and_journal():
    plain = [step("a", 100, 260, rows(0)), step("b", 189.5, 300, rows(0, pid=2000))]
    assert assess(plain, SAME_SECOND)[0] == "missing"
    status, detail = pair(proc(99.9, 189.4, started_index=0, returned_index=1),
                          proc(189.45, 295, started_index=2, returned_index=3))
    assert status == "verified" and detail["srun_returned"]["reuse_settled_by_srun_returned"] == 1
    assert detail["unresolved_total"] == 0 and "not the moment SLURM released" in detail["srun_returned"]["basis"]


def test_a_wall_clock_adjustment_cannot_create_a_separation():
    # Monotonic order: a returned at 189.6, after b started at 189.45 (they overlap). The wall clock of b's
    # records was stepped forward by 1 s, so on the wall clock alone a would appear to return first.
    status, detail = pair(proc(99.9, 189.6, started_index=0, returned_index=3),
                          proc(189.45, 295, started_index=1, returned_index=4, wall_shift=1.0))
    assert status == "missing" and detail["srun_returned"]["reuse_settled_by_srun_returned"] == 0
    assert detail["srun_returned"]["pairs_whose_launcher_timings_disagree"] == 1


def test_a_wall_clock_adjustment_cannot_hide_a_separation_into_a_pass_either_way():
    # Monotonic says separated, the wall clock (stepped back) does not: launcher timing settles nothing.
    status, detail = pair(proc(99.9, 189.4, started_index=0, returned_index=1),
                          proc(189.45, 295, started_index=2, returned_index=3, wall_shift=-1.0))
    assert status == "missing" and detail["srun_returned"]["pairs_whose_launcher_timings_disagree"] == 1


def test_journal_order_must_agree_with_the_clocks():
    status, detail = pair(proc(99.9, 189.4, started_index=0, returned_index=3),
                          proc(189.45, 295, started_index=2, returned_index=4))
    assert status == "missing" and detail["srun_returned"]["pairs_whose_launcher_timings_disagree"] == 1


def test_steps_of_different_launcher_instances_are_never_ordered_by_launcher_timing():
    status, detail = pair(proc(99.9, 189.4, started_index=0, returned_index=1),
                          proc(189.45, 295, started_index=2, returned_index=3, instance="launcher-2"))
    assert status == "missing" and detail["srun_returned"]["reuse_settled_by_srun_returned"] == 0
    assert detail["srun_returned"]["pairs_whose_launcher_timings_disagree"] == 0


def test_genuine_overlap_is_still_a_failure():
    steps = [timed("a", 100, 260, rows(0), proc(99.9, 250, started_index=0, returned_index=2)),
             timed("b", 150, 300, rows(8, pid=2000), proc(149.9, 290, started_index=1, returned_index=3))]
    status, detail = assess(steps, sacct((101, 240), (151, 280)))
    assert status == "failed" and detail["overlaps_total"] == 1


def test_equal_return_and_start_times_settle_nothing():
    status, detail = pair(proc(99.9, 189.5, started_index=0, returned_index=1),
                          proc(189.5, 295, started_index=2, returned_index=3))
    assert status == "missing" and detail["srun_returned"]["reuse_settled_by_srun_returned"] == 0


def test_a_separation_that_slurm_contradicts_is_unresolved_not_reuse():
    steps = [timed("a", 100, 260, rows(0), proc(99.9, 150, started_index=0, returned_index=1)),
             timed("b", 151, 300, rows(0, pid=2000), proc(150.9, 295, started_index=2, returned_index=3))]
    status, detail = assess(steps, sacct((101, 200), (152, 290)))
    assert status == "missing" and "timings disagree" in detail["unresolved"][0][2]


def test_one_step_without_usable_timing_falls_back_to_sacct():
    steps = [timed("a", 100, 260, rows(0), proc(99.9, 189.4, started_index=0, returned_index=1)),
             {**step("b", 189.5, 300, rows(0, pid=2000)), "process": None, "process_unusable": "no finished event"}]
    status, detail = assess(steps, SAME_SECOND)
    assert status == "missing" and detail["srun_returned"]["unusable_reasons"] == {"no finished event": 1}


def test_steps_without_timing_evidence_are_reported_exactly_as_before():
    steps = [step("a", 100, 260, rows(0)), step("b", 200, 300, rows(0, pid=2000))]
    status, detail = assess(steps, sacct((101, 190), (201, 290)))
    assert status == "verified" and "srun_returned" not in detail


# --- launch_timing: when a step's timing is usable -------------------------------------------------------------

def journal_for(*, popen=100.0, start=100.0004, observed=130.0, mono_shift=0.0, monotonic=None, returncode=0,
                finished=131.0, finished_returncode=None, stop_requested=False, killed=False,
                order=("started_step", "srun_returned", "validity", "finished"), duplicate=False, no_event=False,
                no_popen=False, instance=INSTANCE, returned_instance=None, drop=()):
    popen_mono = popen - OFFSET
    observed_mono = observed - OFFSET + mono_shift
    started = {"event": "started_step", "id": "s", "time": start}
    if not no_popen:
        started.update(popen_time=popen, popen_monotonic=popen_mono, launcher_instance=instance)
    returned = {"event": "srun_returned", "id": "s", "observed_at": observed, "observed_monotonic": observed_mono,
                "monotonic_seconds": observed_mono - popen_mono if monotonic is None else monotonic,
                "returncode": returncode, "stop_requested": stop_requested, "killed": killed,
                "launcher_instance": instance if returned_instance is None else returned_instance}
    for key in drop:
        started.pop(key, None)
        returned.pop(key, None)
    events = {"started_step": started, "srun_returned": returned,
              "validity": {"event": "validity", "id": "s", "classification": "valid"},
              "finished": {"event": "finished", "id": "s", "time": finished,
                           "returncode": returncode if finished_returncode is None else finished_returncode}}
    journal = [events[name] for name in order if not (no_event and name == "srun_returned")]
    if duplicate:
        journal.insert(2, dict(returned))
    starts = {e["id"]: e for e in journal if e["event"] == "started_step"}
    ends = {e["id"]: e for e in journal if e["event"] == "finished"}
    return ANALYSER.launch_timing(journal, starts, ends)


def test_a_consistent_event_gives_the_steps_timing():
    usable, unusable = journal_for()
    assert unusable == {}
    timing = usable["s"]
    assert (timing["popen"], timing["observed"]) == (100.0, 130.0)
    assert (timing["popen_monotonic"], timing["observed_monotonic"]) == (100.0 - OFFSET, 130.0 - OFFSET)
    assert timing["instance"] == INSTANCE and timing["started_index"] < timing["returned_index"]


def test_failed_steps_with_a_normal_exit_are_usable():
    assert "s" in journal_for(returncode=7)[0]


@pytest.mark.parametrize("options, reason", [
    (dict(no_event=True), "0 srun_returned events"),
    (dict(duplicate=True), "2 srun_returned events"),
    (dict(drop=("popen_monotonic",)), "no numeric popen times"),
    (dict(drop=("observed_monotonic",)), "no numeric times"),
    (dict(drop=("launcher_instance",)), "one launcher instance"),
    (dict(returned_instance="launcher-2"), "one launcher instance"),
    (dict(returncode=-9), "died from a signal"),
    (dict(returncode=True), "died from a signal"),
    (dict(stop_requested=True), "stopped or killed"),
    (dict(killed=True), "stopped or killed"),
    (dict(finished_returncode=3), "return code differs"),
    (dict(observed=99.0), "negative monotonic duration"),
    (dict(monotonic=-1.0), "negative monotonic duration"),
    (dict(monotonic=29.0), "is not observed_monotonic - popen_monotonic"),
    (dict(mono_shift=-5.0, monotonic=25.0), "wall-clock and monotonic durations disagree"),
    (dict(popen=100.1, start=100.0), "popen_time is after"),
    (dict(observed=131.5), "after the step finished"),
    (dict(order=("started_step", "validity", "srun_returned", "finished")), "not between"),
    (dict(order=("srun_returned", "started_step", "validity", "finished")), "not between"),
])
def test_missing_inconsistent_or_unsafe_timing_is_never_used(options, reason):
    usable, unusable = journal_for(**options)
    assert usable == {} and reason in unusable["s"]


def test_an_event_without_its_started_step_is_unusable():
    journal = [{"event": "srun_returned", "id": "x", "observed_at": 1.0, "observed_monotonic": 1.0,
                "monotonic_seconds": 1.0, "returncode": 0, "stop_requested": False, "killed": False,
                "launcher_instance": INSTANCE}]
    usable, unusable = ANALYSER.launch_timing(journal, {}, {})
    assert usable == {} and "without started_step" in unusable["x"]


def test_a_journal_from_before_the_event_has_no_timing_at_all():
    usable, unusable = journal_for(no_event=True, no_popen=True)
    assert usable == {} and unusable == {}


# --- end to end through the collector's evidence ------------------------------------------------------------

def test_a_run_with_srun_returned_passes_and_reports_the_intervals(tmp_path):
    report = analyse(tmp_path, Scenario(**SHARED_48, pin_format="bare", srun_returned="ok"), SHARED_EXPECTED)
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    timing = check_detail(report, "pinning")["srun_returned"]
    assert timing["steps_with_process_interval"] == 105 and timing["steps_with_unusable_timing"] == 0


def test_an_archived_observation_that_is_not_the_journals_fails_the_archive(tmp_path):
    report = analyse(tmp_path, Scenario(**SHARED_48, pin_format="bare", srun_returned="archive_differs"),
                     SHARED_EXPECTED)
    assert "evidence_archive" in report["failed"]


def test_events_journaled_after_the_verdict_are_not_used(tmp_path):
    report = analyse(tmp_path, Scenario(**SHARED_48, pin_format="bare", srun_returned="after_verdict"),
                     SHARED_EXPECTED)
    timing = check_detail(report, "pinning")["srun_returned"]
    assert timing["steps_with_process_interval"] == 0 and timing["steps_with_unusable_timing"] == 105
    assert report["result"] == "PASSED"  # the earlier evidence still settles these steps; nothing is assumed


def test_older_evidence_without_the_event_keeps_its_pinning_detail(tmp_path):
    report = analyse(tmp_path, Scenario(**SHARED_48, pin_format="bare"), SHARED_EXPECTED)
    assert report["result"] == "PASSED" and "srun_returned" not in check_detail(report, "pinning")
