"""Intel MPI pinning evidence after attempt 209898bc (job 23624369).

Intel MPI 2021.10 on POWER, launched by ``srun``, prints its pinning table with bare CPU numbers
(``[0] MPI startup(): 0 1549873 compute-0-270.powerslurm 12``), not the ``{cpus}`` form the collector
expected, so every step's table was missed and pinning was ``missing``. Both formats are now read, every
rank of every step is checked (once each, distinct PIDs, one host, CPUs), and CPUs are compared only
between steps whose SLURM steps genuinely overlapped. Tables read later from the original logs come in
as a pinning supplement, used only for logs whose SHA-256 and size equal the collector's and the
archived record's.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from test_uspex_execution_acceptance import SHARED_EXPECTED, analyse, check_detail, expected, load_analyser, make
from test_uspex_execution_acceptance_corrections import inspect, load_comparer
from uspex_execution_fakes import COLLECTOR, SHARED_48, Scenario

ANALYSER = load_analyser()
HOST = "compute-0-270.powerslurm"


def rows(base, *, ranks=16, host=HOST, pid=1549873):
    return [[r, pid + r, host, str(base + r)] for r in range(ranks)]


def utc(epoch):
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def step(name, start, end, pin_rows, returncode=0):
    return {"id": name, "start": start, "end": end, "pin_rows": pin_rows, "returncode": returncode}


def sacct(*intervals):
    return [[f"9.{k}", "vasp_std", "COMPLETED", "0:0", utc(a), utc(b), "", "16", "", ""]
            for k, (a, b) in enumerate(intervals)]


def assess(steps, slurm):
    return ANALYSER.assess_pinning(steps, slurm, ranks=16)


# --- one step's table ---------------------------------------------------------------------------------------------


def test_both_row_formats_are_parsed_by_the_collector_pattern():
    import re
    text = COLLECTOR.read_text()
    pattern = re.search(r'PIN = re\.compile\((rb".*?")\)\n', text).group(1)
    pin = re.compile(eval(pattern))  # noqa: S307 - the collector's own literal
    assert pin.search(b"[0] MPI startup(): 0       1549873  compute-0-270.powerslurm  12\n").group(5) == b"12"
    assert pin.search(b"[0] MPI startup(): 3 2003 n1 {3,4}\n").group(4) == b"3,4"
    for line in (b"[0] MPI startup(): Rank    Pid      Node name                 Pin cpu\n",
                 b'[0] MPI startup(): File "" not found\n', b"[0] MPI startup(): 0 1 n1 12 extra\n",
                 b"[0] MPI startup(): 0 1 n1 cpu12\n"):
        assert pin.search(line) is None, line


def test_a_complete_table_of_sixteen_ranks_is_interpretable():
    placement, reason, shared = ANALYSER.pinning_table(rows(12), 16)
    assert reason is None and shared is None
    assert placement["host"] == HOST and placement["cpus"] == set(range(12, 28))


@pytest.mark.parametrize("table, reason", [
    ([], "no pinning table"),
    (rows(0, ranks=15), "are not 0..15"),  # incomplete
    (rows(0) + [[3, 99, HOST, "40"]], "listed more than once"),  # duplicate rank
    ([[r, 7 if r in (2, 5) else 100 + r, HOST, str(r)] for r in range(16)], "same PID"),
    ([[r, 100 + r, HOST if r < 8 else "compute-0-271.powerslurm", str(r)] for r in range(16)], "2 hosts"),
    ([[r, 100 + r, HOST, "12-x" if r == 4 else str(r)] for r in range(16)], "unreadable CPU set"),  # malformed
    ([[r, 100 + r, HOST, "" if r == 4 else str(r)] for r in range(16)], "a rank without CPUs"),
    ([[r + 1, 100 + r, HOST, str(r)] for r in range(16)], "are not 0..15"),  # ranks 1..16
])
def test_malformed_incomplete_or_duplicate_tables_are_uninterpretable(table, reason):
    placement, why, _ = ANALYSER.pinning_table(table, 16)
    assert placement is None and reason in why


def test_ranks_sharing_a_cpu_within_a_step_fail():
    table = rows(0)
    table[5][3] = table[4][3]
    status, detail = assess([step("a", 100, 200, table)], sacct((101, 199)))
    assert status == "failed" and detail["ranks_sharing_a_core"][0][1]["ranks"] == [4, 5]


# --- steps against each other ---------------------------------------------------------------------------------------


def test_concurrent_steps_on_disjoint_cpus_pass():
    steps = [step("a", 100, 200, rows(0)), step("b", 101, 201, rows(16, pid=2000)), step("c", 102, 202, rows(32, pid=3000))]
    status, detail = assess(steps, sacct((101, 190), (102, 191), (103, 192)))
    assert status == "verified" and detail["interpretable"] == 3 and detail["distinct_cpus"] == 48


def test_genuinely_overlapping_steps_on_the_same_cpus_fail():
    steps = [step("a", 100, 200, rows(0)), step("b", 150, 260, rows(8, pid=2000))]
    status, detail = assess(steps, sacct((101, 190), (151, 250)))
    assert status == "failed" and detail["overlaps"][0][:2] == ["a", "b"] and detail["overlaps"][0][2][0] == 8


def test_cpu_reuse_by_steps_whose_launcher_intervals_are_disjoint_is_never_rejected():
    steps = [step("a", 100, 200, rows(0)), step("b", 200, 300, rows(0, pid=2000))]
    status, detail = assess(steps, sacct((101, 190), (201, 290)))
    assert status == "verified" and detail["cpu_reuse_between_non_overlapping_steps"] == 1


def test_reuse_when_only_the_launcher_intervals_overlap_is_decided_by_slurm():
    # The launcher's interval also covers the validity check after VASP (up to 73 s in 209898bc): the
    # next step on the same CPUs may start before the launcher finishes the previous one.
    steps = [step("a", 100, 260, rows(0)), step("b", 200, 300, rows(0, pid=2000))]
    status, detail = assess(steps, sacct((101, 190), (201, 290)))
    assert status == "verified" and detail["cpu_reuse_between_non_overlapping_steps"] == 1


def test_slurm_intervals_meeting_within_one_second_are_unresolved_not_passed():
    steps = [step("a", 100, 260, rows(0)), step("b", 189.5, 300, rows(0, pid=2000))]
    status, detail = assess(steps, sacct((101, 190), (190, 290)))
    assert status == "missing" and detail["unresolved_total"] == 1


def test_steps_that_cannot_be_mapped_to_slurm_steps_are_unresolved_when_they_share_cpus():
    steps = [step("a", 100, 260, rows(0)), step("b", 200, 300, rows(0, pid=2000))]
    status, detail = assess(steps, [])
    assert status == "missing" and "not mapped" in detail["unresolved"][0][2]


def test_interchangeable_slurm_steps_decide_only_when_every_assignment_agrees():
    # Two steps starting within the tolerance of two SLURM steps: either assignment is possible.
    steps = [step("a", 100, 200, rows(0)), step("b", 100.5, 200, rows(0, pid=2000))]
    status, detail = assess(steps, sacct((101, 150), (101, 160)))
    assert detail["slurm_steps_with_several_candidates"] == 2
    assert status == "failed"  # they overlap whichever SLURM step each was


def test_the_same_cpu_numbers_on_different_hosts_do_not_conflict():
    steps = [step("a", 100, 200, rows(0)), step("b", 100, 200, rows(0, host="compute-0-271.powerslurm", pid=2000))]
    assert assess(steps, sacct((101, 190), (101, 191)))[0] == "verified"


def test_a_failed_step_with_a_table_is_compared_and_one_without_is_not_required():
    steps = [step("a", 100, 200, rows(0)), step("b", 120, 220, rows(4, pid=2000), returncode=1)]
    status, detail = assess(steps, sacct((101, 190), (121, 210)))
    assert status == "failed" and detail["failed_steps_compared"] == ["b"]
    steps = [step("a", 100, 200, rows(0)), step("b", 120, 220, [], returncode=1)]
    assert assess(steps, sacct((101, 190), (121, 210)))[0] == "verified"


def test_a_step_that_ran_without_a_table_is_missing_evidence():
    steps = [step("a", 100, 200, rows(0)), step("b", 200, 300, [])]
    status, detail = assess(steps, sacct((101, 190), (201, 290)))
    assert status == "missing" and detail["uninterpretable_reasons"] == {"no pinning table": 1}


# --- the whole analysis --------------------------------------------------------------------------------------------


def test_a_run_with_powers_bare_pinning_format_passes(tmp_path):
    report = analyse(tmp_path, Scenario(**SHARED_48, pin_format="bare"), SHARED_EXPECTED)
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    detail = check_detail(report, "pinning")
    assert detail["interpretable"] == detail["steps_ran"] == 105 and detail["hosts"] == ["compute-0-271.powerslurm"]
    assert detail["cpu_reuse_between_non_overlapping_steps"] > 0 and detail["overlaps"] == []
    # Steps started together have identical SLURM intervals here: interchangeable, and decided for all of them.
    assert detail["slurm_steps_mapped"] + detail["slurm_steps_with_several_candidates"] == 105
    assert detail["unresolved_total"] == 0


def test_concurrent_steps_on_the_same_cpus_in_the_bare_format_fail(tmp_path):
    report = analyse(tmp_path, Scenario(**{**SHARED_48, "pin": "shared"}, pin_format="bare"), SHARED_EXPECTED)
    assert "pinning" in report["failed"] and check_detail(report, "pinning")["overlaps_total"] > 0


def test_the_collector_records_the_table_header_and_all_rows(tmp_path):
    code, output, evidence, _ = make(tmp_path, Scenario(**SHARED_48, pin_format="bare"))
    logs = json.loads((evidence / "step_logs.json").read_text())
    some = next(iter(logs.values()))
    assert some["pin_header_lines"] == 1 and [r[0] for r in some["pin_rows"]] == list(range(16))
    assert some["pin_rows"][0][2] == "compute-0-271.powerslurm" and some["pin_rows"][0][3].isdigit()


# --- the pinning supplement from the original logs --------------------------------------------------------------


def supplement_for(tmp_path, evidence):
    rc, out, err = inspect(tmp_path)
    assert rc == 0, out + err
    comparer = load_comparer()
    inspection = comparer.read_inspection(out)
    report = comparer.compare(inspection, evidence)
    return comparer.pinning_supplement(inspection, report, source_sha256=hashlib.sha256(out.encode()).hexdigest())


def old_evidence(tmp_path, scenario):
    """Evidence as the original collector left it: the logs carry bare tables, the summaries no rows."""

    code, output, evidence, run = make(tmp_path, scenario)
    logs = json.loads((evidence / "step_logs.json").read_text())
    for summary in logs.values():
        summary["pin_rows"] = []
    (evidence / "step_logs.json").write_text(json.dumps(logs))
    return evidence


def test_tables_read_later_from_the_original_logs_complete_the_evidence(tmp_path):
    evidence = old_evidence(tmp_path, Scenario(**SHARED_48, pin_format="bare"))
    assert ANALYSER.analyse(evidence, expected(**SHARED_EXPECTED))["missing"] == ["pinning"]
    supplement = supplement_for(tmp_path, evidence)
    assert all(len(entry["pin_rows"]) == 16 and entry["pin_header_lines"] == 1 for entry in supplement["logs"].values())
    report = ANALYSER.analyse(evidence, expected(**SHARED_EXPECTED), pinning_supplement=supplement)
    assert report["result"] == "PASSED", report["failed"] + report["missing"]
    sources = check_detail(report, "pinning")["sources"]
    assert sources["supplement"] == 105 and sources["refused_total"] == 0
    assert sources["supplement_inspection_sha256"] == supplement["inspection_sha256"]


def test_a_supplement_whose_log_hash_differs_is_refused(tmp_path):
    evidence = old_evidence(tmp_path, Scenario(**SHARED_48, pin_format="bare"))
    supplement = supplement_for(tmp_path, evidence)
    name = sorted(supplement["logs"])[0]
    supplement["logs"][name]["sha256"] = "0" * 64
    report = ANALYSER.analyse(evidence, expected(**SHARED_EXPECTED), pinning_supplement=supplement)
    detail = check_detail(report, "pinning")
    assert report["missing"] == ["pinning"] and detail["sources"]["refused"] == [name]


def test_a_supplement_that_contradicts_the_collected_table_is_not_accepted(tmp_path):
    code, output, evidence, run = make(tmp_path, Scenario(**SHARED_48, pin_format="bare"))
    supplement = supplement_for(tmp_path, evidence)
    name = sorted(supplement["logs"])[0]
    supplement["logs"][name]["pin_rows"][0][3] = "99"
    report = ANALYSER.analyse(evidence, expected(**SHARED_EXPECTED), pinning_supplement=supplement)
    assert "pinning" in report["missing"] and check_detail(report, "pinning")["sources"]["disagreements"] == [name]


def test_no_supplement_is_written_from_an_inspection_that_disagrees_or_has_no_tables(tmp_path):
    evidence = old_evidence(tmp_path, Scenario(**SHARED_48, pin_format="bare"))
    rc, out, _ = inspect(tmp_path)
    comparer = load_comparer()
    inspection = comparer.read_inspection(out)
    report = comparer.compare(inspection, evidence)
    for row in inspection["logs"].values():
        del row["pin_table"]
    with pytest.raises(ValueError, match="no pinning tables"):
        comparer.pinning_supplement(inspection, report, source_sha256="x")
    report["result"] = "NOT VERIFIED"
    with pytest.raises(ValueError, match="does not agree"):
        comparer.pinning_supplement(inspection, report, source_sha256="x")


def test_the_inspection_prints_complete_tables_and_all_startup_lines_bounded(tmp_path):
    code, output, evidence, run = make(tmp_path, Scenario(**SHARED_48, pin_format="bare"))
    log = next((run / "bridge_spool" / "logs").glob("*.log"))
    log.write_bytes(b"".join(b"[0] MPI startup(): noise %d\n" % k for k in range(100)) + log.read_bytes())
    rc, out, _ = inspect(tmp_path)
    assert rc == 0
    by_name = {json.loads(line)["log"]: json.loads(line) for line in out.splitlines() if line.startswith('{"bytes"')}
    row = by_name[log.stem]
    assert len(row["startup_lines"]) == 64 and row["mpi_startup_lines"] == 100 + 18
    assert [r[0] for r in row["pin_table"]] == list(range(16)) and row["pin_rows"] == 16
    other = by_name[sorted(set(by_name) - {log.stem})[0]]
    assert len(other["startup_lines"]) == 18 and other["pin_header_lines"] == 1


# --- the verdict holds under every feasible matching of launcher steps to SLURM steps --------------------------


def mapping_of(edges, intervals):
    return ANALYSER.feasible_mapping({k: list(v) for k, v in edges.items()}, intervals)


def all_maximum_matchings(edges):
    """Every maximum one-to-one matching, by brute force (an independent reference)."""

    import itertools

    steps = list(edges)
    found = []
    for choice in itertools.product(*[[None] + list(edges[s]) for s in steps]):
        used = [c for c in choice if c is not None]
        if len(used) == len(set(used)):
            found.append(dict((s, c) for s, c in zip(steps, choice) if c is not None))
    best = max(len(m) for m in found)
    return [m for m in found if len(m) == best]


def reference_verdict(edges, intervals, first, second):
    verdicts = set()
    for matching in all_maximum_matchings(edges):
        if first not in matching or second not in matching:
            return "unresolved"
        (sa, ea), (sb, eb) = intervals[matching[first]], intervals[matching[second]]
        verdicts.add("separated" if ea < sb or eb < sa else "unresolved" if ea == sb or eb == sa else "overlapping")
    return verdicts.pop() if len(verdicts) == 1 else "unresolved"


def test_a_definite_overlap_is_found_even_when_unrelated_pairings_would_disagree():
    # a can be s0 or s1, b s1 or s2, c s0 or s2: the feasible matchings give (a, b) = (s0, s1) or (s1, s2),
    # both overlapping. The pairing (s0, s2), separated, is impossible (c would have no SLURM step).
    edges = {"a": ["s0", "s1"], "b": ["s1", "s2"], "c": ["s0", "s2"]}
    intervals = {"s0": (100, 150), "s1": (120, 200), "s2": (160, 260)}
    mapping = mapping_of(edges, intervals)
    assert ANALYSER._overlap_verdict(mapping, "a", "b") == "overlapping" == reference_verdict(edges, intervals, "a", "b")


def test_feasible_matchings_that_disagree_leave_the_verdict_unresolved():
    edges = {"a": ["s0", "s1"], "b": ["s1", "s2"], "c": ["s0", "s2"]}
    intervals = {"s0": (100, 110), "s1": (120, 200), "s2": (150, 260)}  # (s0,s1) separated, (s1,s2) overlapping
    mapping = mapping_of(edges, intervals)
    assert ANALYSER._overlap_verdict(mapping, "a", "b") == "unresolved" == reference_verdict(edges, intervals, "a", "b")


def test_a_step_unmatched_in_some_feasible_matching_is_unresolved():
    edges = {"a": ["s0"], "b": ["s0"]}  # only one of them can have run as s0
    mapping = mapping_of(edges, {"s0": (100, 200)})
    assert mapping["candidates"] == {} and ANALYSER._overlap_verdict(mapping, "a", "b") == "unresolved"


def test_disjoint_placement_must_hold_under_every_feasible_matching_end_to_end():
    # Two launcher steps on the same CPUs whose launcher intervals overlap; two SLURM steps each could be.
    shared = [step("a", 100, 300, rows(0)), step("b", 100.5, 300, rows(0, pid=2000))]
    separated = sacct((101, 150), (152, 290))  # either assignment: separated
    status, detail = assess(shared, separated)
    assert status == "verified" and detail["slurm_steps_with_several_candidates"] == 2
    touching = sacct((101, 150), (150, 290))
    assert assess(shared, touching)[0] == "missing"
    overlapping = sacct((101, 200), (150, 290))
    assert assess(shared, overlapping)[0] == "failed"


def test_verdicts_equal_a_brute_force_enumeration_of_every_feasible_matching():
    import random

    rng = random.Random(20261118)
    for _ in range(400):
        slurm = [f"s{k}" for k in range(rng.randint(2, 6))]
        intervals = {}
        for name in slurm:
            start = rng.randint(0, 40)
            intervals[name] = (start, start + rng.randint(1, 30))
        edges = {f"l{k}": rng.sample(slurm, rng.randint(1, min(3, len(slurm)))) for k in range(rng.randint(2, 5))}
        mapping = mapping_of(edges, intervals)
        names = list(edges)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                assert ANALYSER._overlap_verdict(mapping, first, second) == reference_verdict(
                    edges, intervals, first, second), (edges, intervals, first, second)
