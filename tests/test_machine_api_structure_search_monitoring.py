"""Monitoring and cancellation of submitted structure searches (USPEX step 4D-4), against fake POWER and SLURM.

The fake SLURM (in the staging tests) keeps a job table, a POWER clock (local
time, two hours from UTC), ``squeue``/``sacct`` output with optional
accounting delays, missing comments and lingering queue entries, and a
``scancel`` that can succeed, fail, lose its answer or find the job ended.
Nothing contacts POWER or a real scheduler, and no job runs.
"""

from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timedelta, timezone

import pytest

from compute_api import structure_search_monitoring as monitoring
from compute_api.structure_search_monitoring import assess_job, parse_scheduler_output
from compute_api.structure_search_submission import job_identity
from test_machine_api_execution import (  # noqa: F401 - pytest fixtures
    attempt_body,
    bearer,
    client,
    new_id,
    power,
    put as put_calculation,
    state_dir,
    tokens,
)
from test_machine_api_structure_search_attempts import get_search, search_record, write_search_record
from test_machine_api_structure_search_promotion import remote_state, run_dir_of
from test_machine_api_structure_search_staging import (  # noqa: F401 - pytest fixtures
    SEARCH_LOGS,
    cluster,
    tree,
)
from test_machine_api_structure_search_submission import error_code, go, lose_remote_record, prepared_attempt

SLURM_TIME = "%Y-%m-%dT%H:%M:%S"


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    monkeypatch.setattr(monitoring, "MIN_OBSERVATION_INTERVAL_SECONDS", 0)


@pytest.fixture
def submitted(client, tokens, state_dir, cluster):
    prepared = prepared_attempt(client, tokens, cluster)
    response = go(client, tokens, prepared)
    assert response.status_code == 200, response.text
    return (*prepared, response.json()["submission"]["job_id"])


def observe(client, token, attempt):
    return client.post(f"/api/v1/structure-search-attempts/{attempt}/observations", headers=bearer(token))


def cancel(client, token, attempt, digest, job_id, **extra):
    body = {"expected_plan_digest": digest, "job_id": job_id, "confirm": True, **extra}
    return client.put(f"/api/v1/structure-search-attempts/{attempt}/cancellation", headers=bearer(token), json=body)


def run_dir(cluster, state_dir, attempt):
    return run_dir_of(cluster, state_dir, attempt)


def start(cluster, state_dir, attempt, job_id, *, outcome="running"):
    job = cluster.jobs[job_id]
    job.update(state="RUNNING", start=job["submit"])
    directory = run_dir(cluster, state_dir, attempt)
    (directory / "job_started").write_text(f"{job_id}\n")
    write_status(cluster, state_dir, attempt, job_id, outcome=outcome)
    (directory / "bridge_spool").mkdir(exist_ok=True)
    epoch = int(cluster.clock.replace(tzinfo=timezone.utc).timestamp())
    (directory / "bridge_spool" / "heartbeat").write_text(f"{epoch - 5}\n")


def write_status(cluster, state_dir, attempt, job_id, *, outcome, master=None, slurm_job_id=None):
    document = {"schema": "bmd_compute.structure_search_job_status", "schema_version": 1, "attempt_id": attempt,
                "slurm_job_id": slurm_job_id or job_id, "finished_at": None,
                "execution": {"outcome": outcome, **({"master_exit_status": master} if master is not None else {})}}
    (run_dir(cluster, state_dir, attempt) / "search_status.json").write_text(json.dumps(document))


def _earlier(text, seconds, fmt):
    return (datetime.strptime(text, fmt) - timedelta(seconds=seconds)).strftime(fmt)


def elapse(cluster, state_dir, attempt, job_id, seconds):
    """As if ``seconds`` had passed since the submission: the attempt's and the job's times move back together.

    (POWER's clock and the service's are real time, so the submission window and the skew stay real.)
    """

    record = search_record(state_dir, attempt)
    submission = record["submission"]
    for key in ("requested_at", "submitted_at"):
        if submission.get(key):
            submission[key] = _earlier(submission[key], seconds, "%Y-%m-%dT%H:%M:%SZ")
    write_search_record(state_dir, record)
    job = cluster.jobs[job_id]
    for key in ("submit", "start", "end"):
        if job.get(key) and job[key] != "Unknown":
            job[key] = _earlier(job[key], seconds, SLURM_TIME)


def end(cluster, state_dir, attempt, job_id, state, *, ago=300):
    elapse(cluster, state_dir, attempt, job_id, ago + 60)  # submitted a minute before it ended
    job = cluster.jobs[job_id]
    job.update(state=state, end=(cluster.clock - timedelta(seconds=ago)).strftime(SLURM_TIME),
               exit_code="0:0" if state == "COMPLETED" else "1:0")


def finish(cluster, state_dir, attempt, job_id, state="COMPLETED", *, outcome="uspex_finished", master=0, ago=300):
    start(cluster, state_dir, attempt, job_id)
    write_status(cluster, state_dir, attempt, job_id, outcome=outcome, master=master)
    end(cluster, state_dir, attempt, job_id, state, ago=ago)


def evidence_files(state_dir, attempt):
    directory = state_dir / "structure_search" / "observations" / attempt
    return sorted(directory.glob("[0-9]*.json")) if directory.is_dir() else []


def last(payload):
    return payload["monitoring"]["last"]


# --- observation -----------------------------------------------------------------------------------------


def test_an_observation_reads_power_and_records_separate_answers(client, tokens, state_dir, cluster, submitted):
    attempt, job_id = submitted[0], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    remote_before = remote_state(cluster, attempt)
    files_before = tree(run_dir(cluster, state_dir, attempt))
    commands = len(cluster.commands)

    response = observe(client, tokens["runner"], attempt)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"] == "SUBMITTED" and payload["observation"]["performed"] is True
    observation = last(payload)
    assert observation["scheduler"]["state"] == "RUNNING" and observation["scheduler"]["source"] == "squeue"
    assert observation["identity"]["status"] == "verified" and observation["identity"]["problems"] == []
    assert observation["execution"]["outcome"] == "running" and observation["execution"]["job_started"] == job_id
    assert observation["execution"]["sha256"]
    assert 5 <= observation["execution"]["heartbeat_age_s"] <= 8  # POWER's clock (real time passes in between)
    assert observation["classification"] == {"classifier_version": 2, "derived_state": None,
                                             "blocked": "not_terminal"}
    submission = payload["submission"]
    assert submission["scheduler"]["refreshed"] is True and submission["scheduler"]["state"] == "RUNNING"
    assert submission["execution"]["outcome"] == "running" and submission["execution"]["read"] is True
    assert submission["scientific_validity"] == {"status": "not_assessed"}
    assert payload["observation_available"] is True and payload["cancellation_available"] is True

    # Read-only on POWER: no state or run-directory change, no sbatch, no scancel, no SLURM output read.
    assert remote_state(cluster, attempt) == remote_before
    assert tree(run_dir(cluster, state_dir, attempt)) == files_before
    issued = cluster.commands[commands:]
    assert not any("sbatch" in c or "scancel" in c or "slurm-" in c for c in issued)
    assert cluster.scancel_runs == [] and cluster.sbatch_runs == 1

    files = evidence_files(state_dir, attempt)
    assert [path.name for path in files] == ["000001.json"]
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    evidence = json.loads(files[0].read_text())
    assert evidence["previous_sha256"] is None and evidence["observed_by"] == "runner"
    assert evidence["identity"]["expected"]["comment"] == f"bmd_attempt:{attempt}"

    connections = cluster.connections  # ordinary reads stay local
    assert get_search(client, tokens["runner"], attempt).json()["monitoring"] == payload["monitoring"]
    assert cluster.connections == connections


def test_observations_are_appended_as_a_hash_chain(client, tokens, state_dir, cluster, submitted):
    attempt = submitted[0]
    for _ in range(3):
        assert observe(client, tokens["runner"], attempt).status_code == 200
    files = evidence_files(state_dir, attempt)
    assert [path.name for path in files] == ["000001.json", "000002.json", "000003.json"]
    for previous, current in zip(files, files[1:]):
        assert json.loads(current.read_text())["previous_sha256"] == hashlib.sha256(previous.read_bytes()).hexdigest()
    assert search_record(state_dir, attempt)["monitoring"]["observation_count"] == 3
    with pytest.raises(FileExistsError):  # the link step never replaces existing evidence
        import os

        os.link(files[0], files[1])


def test_requests_inside_the_interval_return_the_last_observation(client, tokens, state_dir, cluster, submitted,
                                                                  monkeypatch):
    monkeypatch.setattr(monitoring, "MIN_OBSERVATION_INTERVAL_SECONDS", 60)
    attempt = submitted[0]
    assert observe(client, tokens["runner"], attempt).json()["observation"]["performed"] is True
    connections = cluster.connections
    again = observe(client, tokens["runner"], attempt).json()
    assert again["observation"] == {"performed": False, "reason": "rate_limited",
                                    "retry_after_s": again["observation"]["retry_after_s"], "sequence": 1}
    assert cluster.connections == connections and len(evidence_files(state_dir, attempt)) == 1


def test_observation_refusals(client, tokens, state_dir, cluster, submitted):
    attempt = submitted[0]
    assert observe(client, tokens["other"], attempt).status_code == 403  # not the owner
    assert observe(client, tokens["submit_only"], attempt).status_code == 403  # no read scope
    ready = prepared_attempt(client, tokens, cluster)
    refused = observe(client, tokens["runner"], ready[0])
    assert refused.status_code == 409 and error_code(refused) == "attempt_not_submitted"
    uncertain = prepared_attempt(client, tokens, cluster)
    cluster.sbatch_mode = "lost_before"
    go(client, tokens, uncertain)
    refused = observe(client, tokens["runner"], uncertain[0])
    assert refused.status_code == 409 and error_code(refused) == "submission_uncertain"
    assert evidence_files(state_dir, ready[0]) == [] and evidence_files(state_dir, uncertain[0]) == []


# --- accounting delays and the settling period ------------------------------------------------------------


@pytest.mark.parametrize("change, scheduler_state", [
    ({"hidden": True}, "UNKNOWN"),                                   # not visible anywhere yet
    ({"state": "COMPLETED", "in_queue": False, "hidden": "sacct"}, "UNKNOWN"),  # left the queue, not accounted
    ({"state": "COMPLETED", "in_queue": False, "sacct_state": "RUNNING"}, "RUNNING"),  # accounting lags
])
def test_missing_or_delayed_accounting_is_never_terminal(client, tokens, state_dir, cluster, submitted, change,
                                                         scheduler_state):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    cluster.jobs[job_id].update(change)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED" and last(payload)["scheduler"]["state"] == scheduler_state
    assert last(payload)["classification"]["derived_state"] is None
    assert payload["monitoring"]["terminal"] is None
    cluster.jobs[job_id].update(hidden=False, sacct_state="COMPLETED")
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_COMPLETED"


def test_a_terminal_state_is_recorded_only_after_settling(client, tokens, state_dir, cluster, submitted):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id, ago=30)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED"
    assert last(payload)["classification"]["blocked"] == "terminal_settling"
    # The remaining time is SETTLE_SECONDS less the time from the job's end to POWER's clock as this observation
    # read it (both whole seconds, both recorded in the observation). Real time passes between writing the end
    # time (30 s before POWER's clock then) and the observation, so it is at most 90 s, not exactly 90.
    observed = last(payload)
    end = datetime.strptime(observed["scheduler"]["end"], SLURM_TIME)
    now = datetime.strptime(observed["power_clock"]["utc"], SLURM_TIME)
    assert observed["scheduler"]["remaining_s"] == round(monitoring.SETTLE_SECONDS - (now - end).total_seconds())
    assert 0 < observed["scheduler"]["remaining_s"] <= 90
    elapse(cluster, state_dir, attempt, job_id, 95)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SEARCH_COMPLETED"
    terminal = payload["monitoring"]["terminal"]
    assert terminal["scheduler_state"] == "COMPLETED" and terminal["execution_outcome"] == "uspex_finished"
    assert terminal["observation_sequence"] == 2 and terminal["identity"] == "verified"


def test_a_lingering_queue_entry_does_not_block_a_settled_terminal_state(client, tokens, state_dir, cluster,
                                                                         submitted):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    cluster.jobs[job_id]["in_queue"] = True  # squeue still lists the ended job (MinJobAge)
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_COMPLETED"


def test_the_clock_or_scheduler_not_answering_records_nothing_terminal(client, tokens, state_dir, cluster,
                                                                       submitted):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    cluster.clock_answers = False  # no clock: neither the submission window nor settling can be checked
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED" and last(payload)["classification"]["blocked"] == "identity_not_established"
    assert last(payload)["identity"]["times"]["status"] == "unverified"
    cluster.clock_answers = True
    cluster.scheduler_answers = False
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED" and last(payload)["scheduler"]["state"] == "UNKNOWN"
    cluster.scheduler_answers = True
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_COMPLETED"


# --- terminal-state classification ------------------------------------------------------------------------


@pytest.mark.parametrize("scheduler_state, outcome, master, expected", [
    ("COMPLETED", "uspex_finished", 0, "SEARCH_COMPLETED"),
    ("COMPLETED", "uspex_failed", 1, "SEARCH_FAILED"),         # SLURM's COMPLETED is not USPEX's success
    ("COMPLETED", None, None, "SEARCH_FAILED"),                # no execution record
    ("COMPLETED", "refused_inputs", None, "SEARCH_FAILED"),
    ("FAILED", "running", None, "SEARCH_FAILED"),
    ("OUT_OF_MEMORY", "running", None, "SEARCH_FAILED"),
    ("NODE_FAIL", "running", None, "SEARCH_FAILED"),
    ("TIMEOUT", "running", None, "SEARCH_TIMED_OUT"),
    ("CANCELLED", "running", None, "SEARCH_CANCELLED"),       # cancelled outside BMD Compute
])
def test_terminal_states_follow_slurm_and_the_execution_record(client, tokens, state_dir, cluster, submitted,
                                                              scheduler_state, outcome, master, expected):
    attempt, job_id = submitted[0], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    if outcome is None:
        (run_dir(cluster, state_dir, attempt) / "search_status.json").unlink()
    else:
        write_status(cluster, state_dir, attempt, job_id, outcome=outcome, master=master)
    end(cluster, state_dir, attempt, job_id, scheduler_state)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == expected
    assert payload["submission"]["scientific_validity"] == {"status": "not_assessed"}
    assert payload["submission"]["finalists"] == {"status": "not_started"}
    assert payload["cancellation"] is None


def test_a_recorded_terminal_state_is_preserved_and_contradictions_kept(client, tokens, state_dir, cluster,
                                                                       submitted):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_COMPLETED"
    cluster.jobs[job_id]["sacct_state"] = "FAILED"
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SEARCH_COMPLETED"
    assert payload["monitoring"]["operator_review"] is True
    contradiction = payload["monitoring"]["contradictions"][-1]
    assert contradiction["kind"] == "terminal_state"
    assert contradiction["detail"] == {"recorded": "SEARCH_COMPLETED", "observed": "SEARCH_FAILED"}
    assert payload["monitoring"]["terminal"]["observation_sequence"] == 1
    assert len(evidence_files(state_dir, attempt)) == 2


# --- identity -----------------------------------------------------------------------------------------------


def _contradict(cluster, state_dir, attempt, job_id, kind):
    job = cluster.jobs[job_id]
    if kind == "other_directory":
        job["workdir"] = "/bmd/bmdguest/searches/elsewhere"
    elif kind == "other_name":
        job["job_name"] = "bmds-someone"
    elif kind == "other_comment":
        job["comment"] = "bmd_attempt:00000000-0000-4000-8000-000000000000"
    elif kind == "reused_job_id":
        job["submit"] = (cluster.clock - timedelta(days=30)).strftime(SLURM_TIME)
    elif kind == "second_job_with_our_identity":
        cluster.extra_jobs.append({**{k: job[k] for k in ("job_name", "comment", "workdir", "submit")},
                                   "job_id": "9500", "state": "RUNNING"})
    elif kind == "job_started_names_another_job":
        (run_dir(cluster, state_dir, attempt) / "job_started").write_text("9500\n")
    elif kind == "status_written_by_another_job":
        write_status(cluster, state_dir, attempt, job_id, outcome="uspex_finished", master=0, slurm_job_id="9500")
    elif kind == "remote_record_names_another_job":
        path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
        state = json.loads(path.read_text())
        output = json.loads(state["output"])
        output["submission"]["job_id"] = "9500"
        state["output"] = json.dumps(output)
        path.write_text(json.dumps(state))
    else:
        raise AssertionError(kind)


CONTRADICTIONS = ["other_directory", "other_name", "other_comment", "reused_job_id", "second_job_with_our_identity",
                  "job_started_names_another_job", "status_written_by_another_job",
                  "remote_record_names_another_job"]


@pytest.mark.parametrize("kind", CONTRADICTIONS)
def test_identity_contradictions_never_change_the_state(client, tokens, state_dir, cluster, submitted, kind):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)  # would otherwise be SEARCH_COMPLETED
    _contradict(cluster, state_dir, attempt, job_id, kind)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED" and payload["monitoring"]["terminal"] is None
    assert last(payload)["classification"] == {"classifier_version": 2, "derived_state": None,
                                               "blocked": "contradiction"}
    assert payload["monitoring"]["operator_review"] is True
    assert payload["monitoring"]["contradictions"][0]["kind"] == "identity"


def test_an_unreported_comment_is_noted_but_name_directory_and_time_still_bind(client, tokens, state_dir, cluster,
                                                                              submitted):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    cluster.scheduler_comments = False  # accounting keeps no comment
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SEARCH_COMPLETED"
    assert last(payload)["identity"]["status"] == "partial"
    assert "comment_not_reported" in last(payload)["identity"]["notes"]
    assert payload["monitoring"]["terminal"]["identity"] == "partial"


def test_an_unreported_name_or_directory_does_not_bind(client, tokens, state_dir, cluster, submitted):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    cluster.jobs[job_id]["workdir"] = ""
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED"
    assert last(payload)["classification"]["blocked"] == "identity_not_established"


NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
CONTEXT = {"requested_at": "2026-11-20T10:00:00Z", "bound_at": "2026-11-20T10:00:03Z", "power_now": NOW,
           "skew_s": 0.0, "skew_uncertainty_s": 1.0, "power_offset": "+0200"}


def test_assess_job_units():
    identity = {"comment": "bmd_attempt:a", "job_name": "bmds-x", "workdir": "/bmd/r"}

    def assess(squeue="", sacct="", times=CONTEXT):
        records = (parse_scheduler_output(squeue, monitoring.SQUEUE_FIELDS, "squeue")
                   + parse_scheduler_output(sacct, monitoring.SACCT_FIELDS, "sacct"))
        return assess_job(records, identity=identity, job_id="9001", times=times)

    running = assess(squeue="9001|bmds-x|bmd_attempt:a|/bmd/r|RUNNING|None|2026-11-20T10:00:01\n")
    assert running["identity"]["status"] == "verified" and running["scheduler"]["state"] == "RUNNING"
    assert running["identity"]["times"]["status"] == "consistent"
    ended = assess(sacct="9001|bmds-x||/bmd/r|COMPLETED|0:0|2026-11-20T10:00:01|2026-11-20T10:01:00|"
                         "2026-11-20T11:00:00\n")
    assert ended["identity"]["status"] == "partial" and "comment_not_reported" in ended["identity"]["notes"]
    assert ended["scheduler"]["terminal_reported"] is True and ended["scheduler"]["end"] == "2026-11-20T11:00:00"
    assert assess()["identity"]["status"] == "not_reported" and assess()["scheduler"]["state"] == "UNKNOWN"
    other_comment = assess(sacct="9001|bmds-x|bmd_attempt:b|/bmd/r|COMPLETED|0:0|2026-11-20T10:00:01|"
                                 "2026-11-20T10:01:00|2026-11-20T11:00:00\n")
    assert other_comment["identity"]["status"] == "contradicted"  # a different non-empty comment always blocks
    no_directory = assess(sacct="9001|bmds-x|||COMPLETED|0:0|2026-11-20T10:00:01|2026-11-20T10:01:00|"
                                "2026-11-20T11:00:00\n")
    assert no_directory["identity"]["status"] == "unverified"
    split = assess(sacct="9001|bmds-x|bmd_attempt:a|/bmd/r|COMPLETED|0:0|2026-11-20T10:00:01||\n"
                         "9001|bmds-x|bmd_attempt:a|/bmd/r|FAILED|1:0|2026-11-20T10:00:01||\n")
    assert split["identity"]["status"] == "contradicted"
    disagree = assess(squeue="9001|bmds-x|bmd_attempt:a|/bmd/r|CANCELLED|None|2026-11-20T10:00:01\n",
                      sacct="9001|bmds-x|bmd_attempt:a|/bmd/r|COMPLETED|0:0|2026-11-20T10:00:01||\n")
    assert disagree["identity"]["status"] == "contradicted" and disagree["scheduler"]["terminal_reported"] is False
    reused = assess(squeue="9001|bmds-x|bmd_attempt:a|/bmd/r|RUNNING|None|2026-10-01T00:00:00\n")
    assert reused["identity"]["status"] == "contradicted"


@pytest.mark.parametrize("times, context, status, hint", [
    ({"submit": "2026-11-20T10:00:01"}, {}, "consistent", None),
    ({"submit": "2026-11-20T09:59:30"}, {}, "consistent", None),         # within the slack before the request
    ({"submit": "2026-10-21T10:00:01"}, {}, "contradicted", None),       # a reused job ID, 30 days earlier
    ({"submit": "2026-11-27T10:00:01"}, {"power_now": NOW + timedelta(days=8)}, "contradicted", None),  # bound +7 d
    ({"submit": "2026-11-20T10:05:00"}, {}, "unverified", None),         # outside, but within a zone offset
    ({"submit": "2026-11-20T08:00:01"}, {}, "unverified", None),         # two hours early: a zone, or not ours?
    ({"submit": "2026-11-20T12:00:01"}, {"power_now": NOW + timedelta(hours=1)}, "unverified", "local offset"),
    ({"submit": "2026-11-20T12:00:30"}, {}, "unverified", "after POWER's UTC clock"),  # printed in UTC+2
    ({"submit": "2026-11-20 10:00:01"}, {}, "unverified", "unreadable"),
    ({"submit": "2026-11-20T10:00:01Z"}, {}, "unverified", "unreadable"),
    ({"submit": "11/20-10:00:01"}, {}, "unverified", "unreadable"),
    ({"submit": "2026-13-40T10:00:01"}, {}, "unverified", "unreadable"),
    ({"submit": ""}, {}, "unverified", "not reported"),
    ({"submit": "2026-11-20T10:00:01", "start": "2026-11-20T10:00:00"}, {}, "unverified", "out of order"),
    ({"submit": "2026-11-20T10:00:01", "end": "2026-11-20T12:10:00"}, {}, "unverified", "after POWER's UTC clock"),
    ({"submit": "2026-11-20T10:00:01"}, {"power_now": None}, "unverified", "clock"),
    ({"submit": "2026-11-20T10:00:01"}, {"skew_s": 400.0}, "unverified", "skew"),
    ({"submit": "2026-11-20T10:03:20"}, {"skew_s": 200.0}, "consistent", None),  # POWER 200 s ahead
    ({"submit": "2026-11-20T10:00:01"}, {"skew_s": 200.0}, "unverified", None),  # same instant, wrong side
    ({"submit": "2026-11-20T10:00:01"}, {"requested_at": "2026-11-20T10:00:00"}, "unverified", "not recorded"),
])
def test_check_times(times, context, status, hint):
    result = monitoring.check_times(times, {**CONTEXT, **context})
    assert result["status"] == status, result
    if hint:
        assert hint in result["reason"], result


# --- binding and timestamps, end to end ------------------------------------------------------------------


def test_a_job_bound_by_verified_reconciliation_is_followed_with_a_missing_comment(client, tokens, state_dir,
                                                                                 cluster):
    prepared = prepared_attempt(client, tokens, cluster)
    cluster.sbatch_mode = "lost_after"
    assert error_code(go(client, tokens, prepared)) == "submission_uncertain"
    assert go(client, tokens, prepared).json()["submission"]["reconciled"] is True  # comment verified here
    attempt, job_id = prepared[0], "9001"
    finish(cluster, state_dir, attempt, job_id)
    cluster.scheduler_comments = False
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SEARCH_COMPLETED"
    assert last(payload)["identity"]["binding"] == "verified_reconciliation"
    assert payload["monitoring"]["terminal"]["identity"] == "partial"


def test_a_job_id_that_was_not_reliably_bound_is_never_classified_or_cancelled(client, tokens, state_dir, cluster,
                                                                              submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    record = search_record(state_dir, attempt)
    record["submission"].update(reconciled=True, reconciliation={"outcome": "unverified"})
    write_search_record(state_dir, record)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED"
    assert last(payload)["classification"]["blocked"] == "job_binding_not_established"
    refused = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert refused.status_code == 409 and error_code(refused) == "job_binding_not_established"
    assert cluster.scancel_runs == [] and search_record(state_dir, attempt)["state"] == "SUBMITTED"


@pytest.mark.parametrize("offset_hours, hint", [(2, "after POWER's UTC clock"), (-5, "local offset")])
def test_slurm_times_not_in_utc_are_not_interpreted(client, tokens, state_dir, cluster, submitted, offset_hours,
                                                   hint):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    cluster.local_offset = timedelta(hours=offset_hours)
    cluster.slurm_honours_tz = False  # as if SLURM printed POWER's local time despite TZ=UTC
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED"
    identity = last(payload)["identity"]
    assert identity["status"] == "unverified" and identity["problems"] == []  # not taken for another job either
    assert hint in identity["times"]["reason"]
    assert last(payload)["classification"]["blocked"] == "identity_not_established"
    cluster.slurm_honours_tz = True
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_COMPLETED"


@pytest.mark.parametrize("skew, expected", [(200, "SEARCH_COMPLETED"), (-200, "SEARCH_COMPLETED"),
                                            (400, "SUBMITTED")])
def test_clock_skew_is_measured_and_bounded(client, tokens, state_dir, cluster, skew, expected):
    cluster.clock_skew = timedelta(seconds=skew)  # POWER's clock differs from the service's, before submission
    prepared = prepared_attempt(client, tokens, cluster)
    assert go(client, tokens, prepared).status_code == 200
    attempt, job_id = prepared[0], "9001"
    finish(cluster, state_dir, attempt, job_id)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == expected
    assert abs(last(payload)["power_clock"]["skew_s"] - skew) < 5


@pytest.mark.parametrize("submit", ["2026/11/20 10:00:01", "11/20-10:00:01", "Unknown", "1700000000"])
def test_malformed_or_missing_submission_times_leave_the_identity_unverified(client, tokens, state_dir, cluster,
                                                                            submitted, submit):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    cluster.jobs[job_id]["submit"] = submit
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED" and last(payload)["identity"]["status"] == "unverified"
    assert last(payload)["identity"]["problems"] == []


# --- interrupted observation writes -----------------------------------------------------------------------


def test_an_interrupted_record_update_is_recovered_from_the_evidence(client, tokens, state_dir, cluster, submitted,
                                                                     monkeypatch):
    from compute_api.structure_search_execution import SearchAttemptLedger

    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    real = SearchAttemptLedger.replace
    failures = []

    def replace(self, record):
        if record.get("monitoring") and not failures:
            failures.append(record["attempt_id"])
            raise OSError("disk full")
        return real(self, record)

    monkeypatch.setattr(SearchAttemptLedger, "replace", replace)
    response = observe(client, tokens["runner"], attempt)
    assert response.status_code == 503 and error_code(response) == "observation_unavailable"
    assert len(evidence_files(state_dir, attempt)) == 1  # the evidence was kept
    assert search_record(state_dir, attempt)["state"] == "SUBMITTED"
    monkeypatch.setattr(monitoring, "MIN_OBSERVATION_INTERVAL_SECONDS", 3600)  # recovery needs no new observation
    connections = cluster.connections
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SEARCH_COMPLETED" and payload["monitoring"]["terminal"]["observation_sequence"] == 1
    assert payload["observation"]["performed"] is False and cluster.connections == connections


# --- cancellation ---------------------------------------------------------------------------------------------


def test_cancellation_requested_attempted_then_observed(client, tokens, state_dir, cluster, submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    response = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"] == "CANCEL_REQUESTED"
    cancellation = payload["cancellation"]
    assert cancellation["status"] == "command_sent" and cancellation["requested"]["by"] == "runner"
    assert cancellation["command"]["count"] == 1 and cancellation["command"]["last"]["command"] == f"scancel {job_id}"
    assert cancellation["command"]["last"]["outcome"] == "sent" and cancellation["observed"] is None
    assert cancellation["last_check"]["identity"] == "verified"
    assert cluster.scancel_runs == [job_id] and cluster.sbatch_runs == 1
    state = remote_state(cluster, attempt)
    assert state["state"] == "SEARCH_CANCEL_REQUESTED"
    remote = json.loads(state["output"])
    assert remote["cancellation"]["job_id"] == job_id and remote["submission"]["job_id"] == job_id

    payload = observe(client, tokens["runner"], attempt).json()  # the job ended just now: still settling
    assert payload["state"] == "CANCEL_REQUESTED" and payload["cancellation"]["observed"] is None
    elapse(cluster, state_dir, attempt, job_id, 150)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SEARCH_CANCELLED"
    assert payload["cancellation"]["status"] == "observed"
    assert payload["cancellation"]["observed"]["outcome"] == "cancelled"
    assert payload["cancellation_available"] is False
    assert go(client, tokens, submitted[:4]).json()["state"] == "SEARCH_CANCELLED"  # never submitted again
    assert cluster.sbatch_runs == 1


def test_cancellation_refusals_before_contacting_power(client, tokens, state_dir, cluster, submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    connections = cluster.connections
    assert cancel(client, tokens["preparer"], attempt, digest, job_id).status_code == 403  # no submit scope
    assert cancel(client, tokens["other"], attempt, digest, job_id).status_code == 403  # not the owner
    assert cancel(client, tokens["runner"], attempt, digest, job_id, confirm=False).status_code == 422
    unconfirmed = client.put(f"/api/v1/structure-search-attempts/{attempt}/cancellation",
                             headers=bearer(tokens["runner"]), json={"expected_plan_digest": digest, "job_id": job_id})
    assert unconfirmed.status_code == 422
    assert cancel(client, tokens["runner"], attempt, digest, f"{job_id}_1").status_code == 422  # never a pattern
    wrong = cancel(client, tokens["runner"], attempt, digest, "9002")
    assert wrong.status_code == 409 and error_code(wrong) == "job_id_mismatch"
    stale = cancel(client, tokens["runner"], attempt, "sha256:" + "0" * 64, job_id)
    assert stale.status_code == 409 and error_code(stale) == "plan_digest_mismatch"
    assert cluster.connections == connections and cluster.scancel_runs == []
    assert search_record(state_dir, attempt)["state"] == "SUBMITTED"

    ready = prepared_attempt(client, tokens, cluster)
    refused = cancel(client, tokens["runner"], ready[0], ready[1], "9001")
    assert refused.status_code == 409 and error_code(refused) == "attempt_not_submitted"
    uncertain = prepared_attempt(client, tokens, cluster)
    cluster.sbatch_mode = "lost_before"
    go(client, tokens, uncertain)
    record = search_record(state_dir, uncertain[0])
    refused = cancel(client, tokens["runner"], uncertain[0], uncertain[1], "9001")
    assert refused.status_code == 409 and error_code(refused) == "submission_uncertain"
    assert search_record(state_dir, uncertain[0]) == record and cluster.scancel_runs == []


def test_cancellation_waits_for_a_pending_remote_record(client, tokens, state_dir, cluster):
    prepared = prepared_attempt(client, tokens, cluster)
    assert lose_remote_record(client, tokens, cluster, prepared).status_code == 200
    refused = cancel(client, tokens["runner"], prepared[0], prepared[1], "9001")
    assert refused.status_code == 409 and error_code(refused) == "remote_record_pending"
    assert cluster.scancel_runs == [] and search_record(state_dir, prepared[0])["state"] == "SUBMITTED"


def test_a_terminal_search_is_not_cancelled(client, tokens, state_dir, cluster, submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    finish(cluster, state_dir, attempt, job_id)
    observe(client, tokens["runner"], attempt)
    refused = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert refused.status_code == 409 and error_code(refused) == "attempt_terminal"
    assert cluster.scancel_runs == []


def _refuse(cluster, state_dir, attempt, job_id, change):
    if change == "comment_not_reported":
        cluster.jobs[job_id]["comment"] = ""
    elif change == "not_visible":
        cluster.jobs[job_id]["hidden"] = True
    elif change == "remote_names_another_job":
        _contradict(cluster, state_dir, attempt, job_id, "remote_record_names_another_job")
    elif change == "slurm_times_not_utc":
        cluster.slurm_honours_tz = False
    elif change == "scheduler_unavailable":
        cluster.scheduler_answers = False
    elif change == "claim_held":
        cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.lock").mkdir()
    elif change == "remote_state_changed":
        path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
        state = json.loads(path.read_text())
        state["state"] = "SEARCH_SUBMITTING"
        path.write_text(json.dumps(state))
    else:
        _contradict(cluster, state_dir, attempt, job_id, change)


REFUSALS = [
    ("comment_not_reported", 409, "job_identity_unverified"),   # partial is never enough to cancel
    ("slurm_times_not_utc", 409, "job_identity_unverified"),
    ("other_directory", 409, "submission_conflict"),
    ("second_job_with_our_identity", 409, "submission_conflict"),
    ("reused_job_id", 409, "submission_conflict"),
    ("not_visible", 409, "job_not_visible"),
    ("remote_names_another_job", 409, "submission_conflict"),
    ("remote_state_changed", 409, "transition_not_allowed"),
    ("scheduler_unavailable", 503, "scheduler_unavailable"),
    ("claim_held", 409, "cancellation_in_progress"),
]


@pytest.mark.parametrize("change, status, code", REFUSALS)
def test_a_refusal_before_any_remote_request_returns_the_attempt_to_submitted(client, tokens, state_dir, cluster,
                                                                              submitted, change, status, code):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    _refuse(cluster, state_dir, attempt, job_id, change)
    remote_before = remote_state(cluster, attempt)
    response = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert response.status_code == status and error_code(response) == code, response.text
    assert cluster.scancel_runs == [] and remote_state(cluster, attempt) == remote_before
    payload = get_search(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SUBMITTED" and payload["cancellation_available"] is True
    cancellation = payload["cancellation"]
    assert cancellation["status"] == "refused_before_remote_request" and cancellation["command"]["count"] == 0
    assert cancellation["last_refusal"]["code"] == code
    assert cancellation["last_refusal"]["stage"] == "before_remote_request"


def test_a_connection_failure_before_power_answers_returns_the_attempt_to_submitted(client, tokens, state_dir,
                                                                                   cluster, submitted, monkeypatch):
    from compute_api import execution

    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]

    def unreachable():
        raise ConnectionError("no route to host")

    monkeypatch.setattr(execution, "RUNNER_FACTORY", unreachable)
    response = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert response.status_code == 503 and error_code(response) == "cancellation_unavailable"
    record = search_record(state_dir, attempt)
    assert record["state"] == "SUBMITTED" and record["cancellation"]["status"] == "refused_before_remote_request"


def test_a_later_request_after_a_refusal_starts_afresh(client, tokens, state_dir, cluster, submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    cluster.scheduler_answers = False
    assert cancel(client, tokens["runner"], attempt, digest, job_id).status_code == 503
    cluster.scheduler_answers = True
    payload = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    assert payload["state"] == "CANCEL_REQUESTED" and payload["cancellation"]["status"] == "command_sent"
    assert search_record(state_dir, attempt)["cancellation"]["earlier_requests"][0]["status"] == (
        "refused_before_remote_request")
    assert cluster.scancel_runs == [job_id]


def test_a_refusal_after_the_remote_request_keeps_the_request(client, tokens, state_dir, cluster, submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    cluster.scancel_mode = "error"
    assert cancel(client, tokens["runner"], attempt, digest, job_id).json()["cancellation"]["status"] == (
        "command_failed")
    cluster.jobs[job_id]["comment"] = ""  # the identity can no longer be verified
    refused = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert refused.status_code == 409 and error_code(refused) == "job_identity_unverified"
    record = search_record(state_dir, attempt)
    assert record["state"] == "CANCEL_REQUESTED" and record["cancellation"]["status"] == "command_failed"
    assert record["cancellation"]["last_refusal"]["stage"] == "after_remote_request"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_CANCEL_REQUESTED" and cluster.scancel_runs == [job_id]


def _fail_replace(monkeypatch, *failing_calls):
    from compute_api.structure_search_execution import SearchAttemptLedger

    real = SearchAttemptLedger.replace
    calls = []

    def replace(self, record):
        calls.append(record.get("state"))
        if len(calls) in failing_calls:
            raise OSError("disk full")
        return real(self, record)

    monkeypatch.setattr(SearchAttemptLedger, "replace", replace)
    return lambda: monkeypatch.setattr(SearchAttemptLedger, "replace", real)


def test_remote_request_recorded_but_scancel_not_attempted_is_pending_and_resumed_with_a_fresh_check(
        client, tokens, state_dir, cluster, submitted, monkeypatch):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    restore = _fail_replace(monkeypatch, 2)  # 1: requested; 2: "remote recorded" (fails, so nothing is sent)
    response = cancel(client, tokens["runner"], attempt, digest, job_id)
    restore()
    assert response.status_code == 503 and cluster.scancel_runs == []
    assert remote_state(cluster, attempt)["state"] == "SEARCH_CANCEL_REQUESTED"
    payload = get_search(client, tokens["runner"], attempt).json()
    assert payload["state"] == "CANCEL_REQUESTED"
    assert payload["cancellation"]["status"] == "remote_recorded_command_pending"
    assert payload["cancellation"]["last_refusal"]["stage"] == "after_remote_request"

    observe(client, tokens["runner"], attempt)  # observations never send anything
    assert cluster.scancel_runs == []
    commands = len(cluster.commands)
    resumed = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    issued = cluster.commands[commands:]
    assert any(" squeue " in c for c in issued) and any(" sacct " in c for c in issued)  # a fresh identity check
    assert issued.index(f"scancel {job_id}") > max(i for i, c in enumerate(issued) if " squeue " in c)
    assert resumed["cancellation"]["status"] == "command_sent" and cluster.scancel_runs == [job_id]


def test_a_pending_remote_request_is_never_sent_to_an_unverified_job(client, tokens, state_dir, cluster, submitted,
                                                                    monkeypatch):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    restore = _fail_replace(monkeypatch, 2, 3)  # both local notes lost: the record still says "requested"
    assert cancel(client, tokens["runner"], attempt, digest, job_id).status_code == 503
    restore()
    assert search_record(state_dir, attempt)["cancellation"]["status"] == "requested"
    cluster.jobs[job_id]["comment"] = ""
    refused = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert refused.status_code == 409 and error_code(refused) == "job_identity_unverified"
    record = search_record(state_dir, attempt)  # POWER holds the request: never back to SUBMITTED
    assert record["state"] == "CANCEL_REQUESTED"
    assert record["cancellation"]["status"] == "remote_recorded_command_pending" and cluster.scancel_runs == []


def test_repeated_cancellation_is_safe(client, tokens, state_dir, cluster, submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    cluster.scancel_mode = "error"
    first = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    assert first["cancellation"]["status"] == "command_failed" and cluster.jobs[job_id]["state"] == "RUNNING"
    recorded = json.loads(remote_state(cluster, attempt)["output"])["cancellation"]
    cluster.scancel_mode = "accept"
    second = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    assert second["cancellation"]["status"] == "command_sent" and second["cancellation"]["command"]["count"] == 2
    assert json.loads(remote_state(cluster, attempt)["output"])["cancellation"] == recorded  # recorded once
    third = cancel(client, tokens["runner"], attempt, digest, job_id).json()  # the job has left the queue
    assert third["cancellation"]["status"] == "not_sent_job_not_active"
    assert cluster.scancel_runs == [job_id, job_id] and cluster.sbatch_runs == 1


def test_a_lost_scancel_answer_is_resolved_by_observation(client, tokens, state_dir, cluster, submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    cluster.scancel_mode = "lost"
    payload = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    assert payload["cancellation"]["status"] == "command_outcome_unknown"
    again = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    assert again["cancellation"]["status"] == "not_sent_job_not_active" and cluster.scancel_runs == [job_id]
    elapse(cluster, state_dir, attempt, job_id, 150)
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_CANCELLED"


@pytest.mark.parametrize("when", ["before_request", "while_scancel_runs"])
def test_a_job_that_ends_before_cancellation_takes_effect_keeps_its_slurm_state(client, tokens, state_dir, cluster,
                                                                               submitted, when):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    finish(cluster, state_dir, attempt, job_id)  # COMPLETED with USPEX finished, settled
    if when == "before_request":
        payload = cancel(client, tokens["runner"], attempt, digest, job_id).json()
        assert payload["cancellation"]["status"] == "not_sent_job_not_active" and cluster.scancel_runs == []
    else:
        cluster.jobs[job_id]["in_queue"] = True
        cluster.jobs[job_id]["state"] = "RUNNING"  # squeue still says RUNNING; scancel finds it ended
        cluster.scancel_mode = "ended"
        payload = cancel(client, tokens["runner"], attempt, digest, job_id).json()
        assert payload["cancellation"]["status"] == "command_found_job_ended"
        cluster.jobs[job_id].update(state="COMPLETED", in_queue=False)
    assert payload["state"] == "CANCEL_REQUESTED"
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "SEARCH_COMPLETED"
    assert payload["cancellation"]["observed"]["outcome"] == "job_ended_before_cancellation_took_effect"


# --- interrupted cancellation writes --------------------------------------------------------------------------


@pytest.mark.parametrize("point", ["partial", "before_rename"])
def test_an_interrupted_remote_cancellation_write_sends_nothing_and_resumes(client, tokens, state_dir, cluster,
                                                                            submitted, point):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    cluster.fail_put = (f"{attempt}.json", point)
    response = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert response.status_code == 503 and error_code(response) == "cancellation_unavailable"
    assert cluster.scancel_runs == [] and remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTED"
    record = search_record(state_dir, attempt)
    assert record["state"] == "CANCEL_REQUESTED" and record["cancellation"]["status"] == "remote_request_uncertain"
    assert record["cancellation"]["last_refusal"]["stage"] == "remote_write"
    resumed = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    assert resumed["cancellation"]["status"] == "command_sent" and cluster.scancel_runs == [job_id]
    assert remote_state(cluster, attempt)["state"] == "SEARCH_CANCEL_REQUESTED"


def test_a_lost_local_write_after_scancel_is_recovered(client, tokens, state_dir, cluster, submitted, monkeypatch):
    from compute_api.structure_search_execution import SearchAttemptLedger

    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    real = SearchAttemptLedger.replace
    calls = []

    def replace(self, record):
        calls.append(record.get("state"))
        if len(calls) == 3:  # 1: requested (before POWER); 2: remote recorded; 3: the command's result
            raise OSError("disk full")
        return real(self, record)

    monkeypatch.setattr(SearchAttemptLedger, "replace", replace)
    response = cancel(client, tokens["runner"], attempt, digest, job_id)
    assert response.status_code == 503 and cluster.scancel_runs == [job_id]
    monkeypatch.setattr(SearchAttemptLedger, "replace", real)
    record = search_record(state_dir, attempt)
    assert record["state"] == "CANCEL_REQUESTED" and record["cancellation"]["command_count"] == 0
    assert record["cancellation"]["status"] == "remote_recorded_command_pending"
    again = cancel(client, tokens["runner"], attempt, digest, job_id).json()
    assert again["cancellation"]["status"] == "not_sent_job_not_active" and cluster.scancel_runs == [job_id]
    elapse(cluster, state_dir, attempt, job_id, 150)
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_CANCELLED"


def test_a_cancellation_power_records_but_the_local_record_lost_is_adopted(client, tokens, state_dir, cluster,
                                                                          submitted):
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    cluster.scancel_mode = "error"  # the request reaches POWER, the job keeps running
    cancel(client, tokens["runner"], attempt, digest, job_id)
    record = search_record(state_dir, attempt)
    record["state"] = "SUBMITTED"  # as if the local record had been restored from before the request
    record.pop("cancellation")
    write_search_record(state_dir, record)
    payload = observe(client, tokens["runner"], attempt).json()
    assert payload["state"] == "CANCEL_REQUESTED"
    assert payload["cancellation"]["requested"]["adopted_from_power"] is True
    assert payload["cancellation"]["requested"]["by"] == "runner" and cluster.scancel_runs == [job_id]


# --- isolation: submissions, calculations, rollback, runtime package -------------------------------------------


def test_monitoring_never_resubmits_and_reconciliation_is_unchanged(client, tokens, state_dir, cluster, submitted):
    attempt, job_id = submitted[0], submitted[4]
    finish(cluster, state_dir, attempt, job_id, state="FAILED", outcome="uspex_failed", master=1)
    assert observe(client, tokens["runner"], attempt).json()["state"] == "SEARCH_FAILED"
    again = go(client, tokens, submitted[:4])
    assert again.status_code == 200 and again.json()["state"] == "SEARCH_FAILED"
    assert cluster.sbatch_runs == 1 and list(cluster.jobs) == [job_id]


def test_calculations_and_the_v120_reader_are_untouched(client, tokens, state_dir, cluster, submitted):
    from compute_api import execution
    from test_machine_api_structure_search_attempts import _v120_ledger_module

    calculation = new_id()
    previous = execution.RUNNER_FACTORY
    execution.RUNNER_FACTORY = cluster.calculation_factory
    try:
        assert put_calculation(client, tokens["runner"], calculation,
                               attempt_body(client, tokens, submit=False)).status_code == 200
    finally:
        execution.RUNNER_FACTORY = previous
    before = (state_dir / "attempts" / f"{calculation}.json").read_bytes()
    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    observe(client, tokens["runner"], attempt)
    cancel(client, tokens["runner"], attempt, digest, job_id)
    assert (state_dir / "attempts" / f"{calculation}.json").read_bytes() == before
    assert sorted(path.name for path in (state_dir / "attempts").iterdir()) == [f"{calculation}.json"]
    v120 = _v120_ledger_module().AttemptLedger.from_environment()
    assert [record["attempt_id"] for record in v120.records()] == [calculation]
    record = json.loads((state_dir / "structure_search" / "attempts" / f"{attempt}.json").read_text())
    assert (record["schema"], record["schema_version"]) == ("bmd_compute.api_structure_search_attempt_record", 1)


def test_the_calculation_path_refuses_a_cancellation_state(client, tokens, state_dir, cluster, submitted):
    from backend.remote import SubmissionAttemptNotPrepared

    attempt, digest, job_id = submitted[0], submitted[1], submitted[4]
    start(cluster, state_dir, attempt, job_id)
    cancel(client, tokens["runner"], attempt, digest, job_id)
    spec = search_record(state_dir, attempt)["staging"]["spec"]
    with pytest.raises(SubmissionAttemptNotPrepared):
        cluster.factory()._submit_prepared_submission_attempt(spec, remote_state(cluster, attempt), "")
    assert cluster.sbatch_calls == 0


def test_the_runtime_package_is_the_reviewed_one():
    """4D-4 was service-side only (the package the 4D-2 validation staged, b7bd184, ``fca63b4b…``).

    Since then: the launcher's scientific evidence archive (``bridge/evidence_archive.py``, ``92913a28…``,
    commit 0b9925e), and configurable node exclusivity and job memory for structure searches (shared nodes):
    80 files. Attempts staged with an earlier package are refused with ``runtime_package_changed`` and need
    a new attempt ID.
    """

    from backend.runtime_package import build_runtime_package_manifest, runtime_package_manifest_digest

    manifest = build_runtime_package_manifest()
    assert len(manifest) == 80
    assert runtime_package_manifest_digest(manifest) == (
        "f31cfab5bf5c2cbc52c03594351b705cf1307922a0eeff001f577487844afc36")


def test_scancel_command_takes_only_a_numeric_job_id():
    assert monitoring._scancel_command("9001") == "scancel 9001"
    for bad in ("9001_1", "9001 --user=x", "-u bmdguest", "", "bmds-x"):
        with pytest.raises(ValueError):
            monitoring._scancel_command(bad)
    assert job_identity("a" * 8 + "-0000-4000-8000-000000000000", "/r/")["workdir"] == "/r"
