"""Exactly-once SLURM submission of structure searches (USPEX step 4D-3), against fake POWER and fake SLURM.

The fake POWER is the staging tests' (real finalize and inventory code on a
local tree); its fake SLURM answers ``sbatch --parsable`` in configurable
ways (accept, reject, lose the answer before or after creating the job,
unreadable output, exit 255) and ``squeue``/``sacct`` from its job table.
Nothing contacts POWER or a real scheduler, and no job runs.
"""

from __future__ import annotations

import json

import pytest

from compute_api.structure_search_submission import (
    definite_rejection,
    job_identity,
    parse_parsable_job_id,
    reconcile_jobs,
    scheduler_records,
)
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
from test_machine_api_structure_search_promotion import promote, remote_state, run_dir_of, staged_attempt
from test_machine_api_structure_search_staging import (  # noqa: F401 - pytest fixtures
    SEARCH_LOGS,
    WORKSPACE,
    cluster,
    tree,
)


def prepared_attempt(client, tokens, cluster, *, principal="runner"):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster, principal=principal)
    promoted = promote(client, tokens[principal], attempt, digest, resources_sha256)
    assert promoted.status_code == 200, promoted.text
    return attempt, digest, resources_sha256, promoted.json()["promotion"]["job_script"]["sha256"]


def submit(client, token, attempt, digest, resources_sha256, job_script_sha256, **extra):
    body = {"expected_plan_digest": digest, "job_script_sha256": job_script_sha256,
            "resource_acknowledgement": {"acknowledged": True, "resources_sha256": resources_sha256}, **extra}
    return client.put(f"/api/v1/structure-search-attempts/{attempt}/submission", headers=bearer(token), json=body)


@pytest.fixture
def prepared(client, tokens, state_dir, cluster):
    return prepared_attempt(client, tokens, cluster)


def go(client, tokens, prepared, token="runner"):
    return submit(client, tokens[token], *prepared)


def error_code(response):
    return response.json()["error"]["code"]


# --- success ----------------------------------------------------------------------------------


def test_submission_verifies_afresh_then_runs_sbatch_once(client, tokens, state_dir, cluster, prepared):
    attempt = prepared[0]
    before = get_search(client, tokens["runner"], attempt).json()
    assert before["state"] == "PREPARED" and before["submission_available"] is True
    checks = cluster.runtime_checks
    run_dir = run_dir_of(cluster, state_dir, attempt)
    staged = tree(run_dir)

    response = go(client, tokens, prepared)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"] == "SUBMITTED" and payload["submission_available"] is False
    submission = payload["submission"]
    assert submission["job_id"] == "9001" and submission["status"] == "submitted"
    assert submission["reconciled"] is False and submission["uncertain"] is False
    assert submission["job_name"] == f"bmds-{attempt[:8]}" and submission["attempt_comment"] == f"bmd_attempt:{attempt}"
    assert submission["scheduler"] == {"state": None, "refreshed": False}
    assert submission["execution"] == {"outcome": None, "read": False}
    assert submission["scientific_validity"] == {"status": "not_assessed"}
    assert submission["finalists"] == {"status": "not_started"}

    assert cluster.sbatch_runs == 1 and cluster.runtime_checks == checks + 1  # fresh verification first
    job = cluster.jobs["9001"]
    assert (job["comment"], job["job_name"]) == (f"bmd_attempt:{attempt}", f"bmds-{attempt[:8]}")
    assert job["workdir"] == search_record(state_dir, attempt)["staging"]["spec"]["paths"]["run_dir"]
    record = search_record(state_dir, attempt)
    planned = _plan_directives(state_dir, attempt)
    assert {key: job["directives"][key] for key in ("time", "nodes", "ntasks", "mem", "exclude")} == planned
    assert tree(run_dir) == staged  # nothing written into the run directory by submission

    state = remote_state(cluster, attempt)
    assert state["state"] == "SEARCH_SUBMITTED"
    output = json.loads(state["output"])
    assert output["submission"]["job_id"] == "9001"
    assert output["promotion"]["job_script"]["sha256"] == record["promotion"]["job_script"]["sha256"]
    sbatch_commands = [c for c in cluster.commands if " && sbatch " in c]
    assert sbatch_commands == [f"cd {state['run_dir']} && sbatch --parsable bmd_search_job.sbatch"]
    assert cluster.sbatch_calls == 0  # never through the calculation submit_batch path
    assert get_search(client, tokens["runner"], attempt).json() == payload  # reads stay local


def _plan_directives(state_dir, attempt):
    from test_machine_api_structure_search_promotion import _plan

    request = _plan(state_dir, attempt).record["resources"]["request"]
    return {"time": request["walltime"], "nodes": str(request["nodes"]), "ntasks": str(request["ntasks"]),
            "mem": request["mem"], "exclude": request["exclude"]}


def test_duplicate_requests_never_submit_twice(client, tokens, state_dir, cluster, prepared):
    first = go(client, tokens, prepared)
    connections = cluster.connections
    second = go(client, tokens, prepared)
    assert second.status_code == 200 and second.json() == first.json()
    assert cluster.sbatch_runs == 1 and cluster.connections == connections
    # Promotion and staging requests return the stored record too.
    attempt, digest, resources_sha256, _ = prepared
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 409


def test_a_concurrent_request_holding_the_claim_blocks_without_sbatch(client, tokens, state_dir, cluster, prepared):
    attempt = prepared[0]
    lock = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.lock")
    lock.mkdir()
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "submission_in_progress"
    assert cluster.sbatch_runs == 0 and remote_state(cluster, attempt)["state"] == "SEARCH_PREPARED"
    lock.rmdir()
    assert go(client, tokens, prepared).status_code == 200 and cluster.sbatch_runs == 1


# --- refusals before sbatch ------------------------------------------------------------------------


def test_refusals_before_contacting_power(client, tokens, state_dir, cluster, prepared):
    attempt, digest, resources_sha256, script = prepared
    connections = cluster.connections
    assert submit(client, tokens["preparer"], *prepared).status_code == 403  # no submit scope
    assert submit(client, tokens["other"], *prepared).status_code == 403  # not the owner
    wrong_script = submit(client, tokens["runner"], attempt, digest, resources_sha256, "0" * 64)
    assert wrong_script.status_code == 409 and error_code(wrong_script) == "job_script_mismatch"
    wrong_ack = submit(client, tokens["runner"], attempt, digest, "sha256:" + "1" * 64, script)
    assert wrong_ack.status_code == 409 and error_code(wrong_ack) == "acknowledgement_mismatch"
    assert cluster.connections == connections
    staged, staged_digest, staged_resources = staged_attempt(client, tokens, cluster)
    connections = cluster.connections
    unpromoted = submit(client, tokens["runner"], staged, staged_digest, staged_resources, script)
    assert unpromoted.status_code == 409 and error_code(unpromoted) == "attempt_not_prepared"
    bad = client.put(f"/api/v1/structure-search-attempts/{attempt}/submission", headers=bearer(tokens["runner"]),
                     json={"expected_plan_digest": digest, "job_script_sha256": script})
    assert bad.status_code == 422
    assert cluster.connections == connections and cluster.sbatch_runs == 0


def test_damaged_inputs_or_job_script_are_refused_and_the_attempt_stays_ready(client, tokens, state_dir, cluster,
                                                                              prepared):
    attempt = prepared[0]
    run_dir = run_dir_of(cluster, state_dir, attempt)
    (run_dir / "bmd_search_job.sbatch").write_text("#!/bin/bash\necho changed\n")
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "job_script_mismatch"
    assert cluster.sbatch_runs == 0 and remote_state(cluster, attempt)["state"] == "SEARCH_PREPARED"
    record = search_record(state_dir, attempt)
    assert record["state"] == "PREPARED" and record["submission"]["status"] == "not_submitted"


def test_a_failed_readiness_check_refuses_before_sbatch(client, tokens, state_dir, cluster, prepared):
    cluster.runtime_problems = ["USPEX is not unpacked in the shared cache."]
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "runtime_check_failed"
    assert cluster.sbatch_runs == 0 and search_record(state_dir, prepared[0])["state"] == "PREPARED"
    cluster.runtime_problems = []
    assert go(client, tokens, prepared).status_code == 200 and cluster.sbatch_runs == 1


def test_a_stale_plan_is_refused_before_contacting_power(client, tokens, state_dir, cluster, prepared, monkeypatch):
    from backend.structure_search import search_plan

    def stale(plan, **kwargs):
        raise search_plan.SearchPlanError("policy changed", code="plan_stale")

    monkeypatch.setattr(search_plan, "require_current_plan", stale)
    connections = cluster.connections
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "plan_changed"
    assert cluster.connections == connections and search_record(state_dir, prepared[0])["state"] == "PREPARED"


# --- definite rejection ----------------------------------------------------------------------------


@pytest.mark.parametrize("returncode, stderr, definite", [
    (1, "sbatch: error: Batch job submission failed: Invalid qos specification", True),
    (127, "bash: sbatch: command not found", True),
    (255, "", False),
    (1, "ssh: connection reset", False),
    (0, "", False),
])
def test_only_a_definite_refusal_counts_as_rejection(returncode, stderr, definite):
    assert definite_rejection(returncode, "", stderr) is definite
    assert definite_rejection(1, "9001\n", "sbatch: error: x") is False  # a job ID is never a rejection


def test_a_definite_rejection_returns_the_attempt_to_ready_and_a_retry_submits_once(client, tokens, state_dir,
                                                                                     cluster, prepared):
    attempt = prepared[0]
    cluster.sbatch_mode = "reject"
    response = go(client, tokens, prepared)
    assert response.status_code == 502 and error_code(response) == "submit_rejected"
    assert "Invalid account" in response.json()["error"]["sbatch_error"]
    assert cluster.jobs == {}
    state = remote_state(cluster, attempt)
    assert state["state"] == "SEARCH_PREPARED" and "Invalid account" in json.loads(state["output"])["last_submission_error"]
    record = search_record(state_dir, attempt)
    assert record["state"] == "PREPARED" and record["submission"]["status"] == "rejected"

    cluster.sbatch_mode = "accept"
    retried = go(client, tokens, prepared)
    assert retried.status_code == 200 and retried.json()["submission"]["job_id"] == "9001"
    assert cluster.sbatch_runs == 2 and list(cluster.jobs) == ["9001"]


# --- uncertain outcomes and reconciliation ---------------------------------------------------------


def test_lost_response_after_submission_is_reconciled_without_resubmitting(client, tokens, state_dir, cluster,
                                                                           prepared):
    attempt = prepared[0]
    cluster.sbatch_mode = "lost_after"
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "submission_uncertain"
    assert search_record(state_dir, attempt)["state"] == "SUBMISSION_UNCERTAIN"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"
    read = get_search(client, tokens["runner"], attempt).json()
    assert read["submission"]["uncertain"] is True and read["submission_available"] is False

    cluster.sbatch_mode = "accept"  # would create a second job if it were ever run again
    response = go(client, tokens, prepared)
    assert response.status_code == 200, response.text
    submission = response.json()["submission"]
    assert submission["job_id"] == "9001" and submission["reconciled"] is True
    assert submission["reconciliation"]["outcome"] == "matched"
    assert cluster.sbatch_runs == 1 and list(cluster.jobs) == ["9001"]
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTED"


@pytest.mark.parametrize("mode", ["lost_before", "exit255"])
def test_an_uncertain_submission_with_no_job_stays_uncertain_and_is_never_repeated(client, tokens, state_dir,
                                                                                   cluster, prepared, mode):
    attempt = prepared[0]
    cluster.sbatch_mode = mode
    assert error_code(go(client, tokens, prepared)) == "submission_uncertain"
    cluster.sbatch_mode = "accept"
    for _ in range(2):
        again = go(client, tokens, prepared)
        assert again.status_code == 409 and error_code(again) == "submission_uncertain"
        assert again.json()["error"]["reconciliation"]["outcome"] == "none"
    assert cluster.sbatch_runs == 1 and cluster.jobs == {}
    assert search_record(state_dir, attempt)["state"] == "SUBMISSION_UNCERTAIN"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"


def test_unreadable_sbatch_output_is_uncertain_and_reconciled(client, tokens, state_dir, cluster, prepared):
    cluster.sbatch_mode = "garbled"
    assert error_code(go(client, tokens, prepared)) == "submission_uncertain"
    response = go(client, tokens, prepared)
    assert response.status_code == 200 and response.json()["submission"]["job_id"] == "9001"
    assert cluster.sbatch_runs == 1


def test_an_unavailable_comment_is_never_taken_as_identity(client, tokens, state_dir, cluster, prepared):
    attempt = prepared[0]
    cluster.sbatch_mode = "lost_after"
    go(client, tokens, prepared)
    cluster.jobs["9001"]["state"] = "COMPLETED"  # left the queue: only sacct reports it
    cluster.scheduler_comments = False  # accounting kept no comment: name and directory alone are not enough
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "submission_unverified"
    reconciliation = response.json()["error"]["reconciliation"]
    assert reconciliation["outcome"] == "unverified" and reconciliation["matches"] == []
    assert [job["job_id"] for job in reconciliation["unverified"]] == ["9001"]
    assert search_record(state_dir, attempt)["state"] == "SUBMISSION_UNCERTAIN"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING" and cluster.sbatch_runs == 1
    cluster.scheduler_comments = True  # once SLURM reports the attempt comment, the job is verified
    adopted = go(client, tokens, prepared)
    assert adopted.status_code == 200 and adopted.json()["submission"]["job_id"] == "9001"
    assert cluster.sbatch_runs == 1


def test_delayed_accounting_leaves_the_attempt_uncertain_until_the_job_is_visible(client, tokens, state_dir,
                                                                                  cluster, prepared):
    attempt = prepared[0]
    cluster.sbatch_mode = "lost_after"
    go(client, tokens, prepared)
    cluster.jobs["9001"]["hidden"] = True  # SLURM does not report it yet
    for _ in range(2):
        response = go(client, tokens, prepared)
        assert response.status_code == 409 and error_code(response) == "submission_uncertain"
        assert response.json()["error"]["reconciliation"]["outcome"] == "none"
    assert search_record(state_dir, attempt)["submission"]["uncertain_reason"].startswith("no job")
    cluster.jobs["9001"]["state"] = "COMPLETED"
    cluster.jobs["9001"]["hidden"] = "sacct"  # gone from the queue and not yet in accounting
    assert error_code(go(client, tokens, prepared)) == "submission_uncertain"
    cluster.jobs["9001"]["hidden"] = False
    response = go(client, tokens, prepared)
    assert response.status_code == 200 and response.json()["submission"]["job_id"] == "9001"
    assert cluster.sbatch_runs == 1 and list(cluster.jobs) == ["9001"]


@pytest.mark.parametrize("other, code", [
    ({"job_name": "bmds-other", "comment": "COMMENT"}, "submission_conflict"),      # our comment, other name
    ({"job_name": "NAME", "comment": "COMMENT", "workdir": "/bmd/elsewhere"}, "submission_conflict"),
    ({"job_name": "NAME", "comment": "bmd_attempt:someone-else"}, "submission_conflict"),  # our dir and name
    ({"job_name": "NAME", "comment": "COMMENT", "job_id": "9500"}, "submission_ambiguous"),  # two of ours
    ({"job_name": "NAME", "comment": ""}, "submission_ambiguous"),  # ours, plus a candidate with no comment
    ({"job_id": "9001", "job_name": "NAME", "comment": "", "workdir": "/bmd/elsewhere"},
     "submission_conflict"),  # SLURM reports two directories for the one job
    ({"job_id": "9001", "job_name": "NAME", "comment": "bmd_attempt:someone-else"},
     "submission_conflict"),  # two comments for the one job
])
def test_conflicting_or_ambiguous_identities_are_never_adopted(client, tokens, state_dir, cluster, prepared, other,
                                                               code):
    attempt = prepared[0]
    cluster.sbatch_mode = "lost_after"
    go(client, tokens, prepared)
    identity = job_identity(attempt, remote_state(cluster, attempt)["run_dir"])
    job = {"job_id": "9500", "job_name": identity["job_name"], "comment": identity["comment"],
           "workdir": identity["workdir"], "state": "RUNNING", **other}
    job["job_name"] = identity["job_name"] if job["job_name"] == "NAME" else job["job_name"]
    job["comment"] = identity["comment"] if job["comment"] == "COMMENT" else job["comment"]
    cluster.extra_jobs.append(job)
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == code
    assert search_record(state_dir, attempt)["state"] == "SUBMISSION_UNCERTAIN"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING" and cluster.sbatch_runs == 1


def test_reconciliation_needs_the_scheduler_to_answer(client, tokens, state_dir, cluster, prepared):
    cluster.sbatch_mode = "lost_after"
    go(client, tokens, prepared)
    cluster.scheduler_answers = False
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "submission_uncertain"
    cluster.scheduler_answers = True
    assert go(client, tokens, prepared).status_code == 200 and cluster.sbatch_runs == 1


def test_a_locally_uncertain_attempt_is_never_submitted_even_if_power_shows_prepared(client, tokens, state_dir,
                                                                                     cluster, prepared):
    attempt = prepared[0]
    cluster.sbatch_mode = "lost_before"
    go(client, tokens, prepared)
    path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    state = json.loads(path.read_text())
    state["state"] = "SEARCH_PREPARED"  # as if someone reset POWER by hand
    state["output"] = json.dumps(json.loads(state["output"])["promotion"])
    path.write_text(json.dumps(state))
    cluster.sbatch_mode = "accept"
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "submission_uncertain"
    assert cluster.sbatch_runs == 1 and cluster.jobs == {}


# --- isolation between attempts -------------------------------------------------------------------


def test_reconciliation_never_adopts_another_attempts_job(client, tokens, state_dir, cluster):
    first = prepared_attempt(client, tokens, cluster)
    second = prepared_attempt(client, tokens, cluster)
    cluster.sbatch_mode = "lost_after"
    for attempt in (first, second):
        assert error_code(go(client, tokens, attempt)) == "submission_uncertain"
    assert set(cluster.jobs) == {"9001", "9002"}  # one job each, both verified for their own attempt
    cluster.jobs["9002"]["hidden"] = True  # the second attempt's job is not visible yet
    refused = go(client, tokens, second)
    assert refused.status_code == 409 and error_code(refused) == "submission_uncertain"
    assert refused.json()["error"]["reconciliation"]["outcome"] == "none"  # 9001 is not even a candidate
    adopted = go(client, tokens, first)
    assert adopted.status_code == 200 and adopted.json()["submission"]["job_id"] == "9001"
    cluster.jobs["9002"]["hidden"] = False
    assert go(client, tokens, second).json()["submission"]["job_id"] == "9002"
    assert cluster.sbatch_runs == 2


def _foreign_job(attempt, cluster, job_id="9500", **fields):
    """A job of another BMD attempt whose UUID shares this attempt's first eight characters (so its name)."""

    other = attempt[:8] + "-0000-4000-8000-000000000000"
    identity = job_identity(attempt, remote_state(cluster, attempt)["run_dir"])
    return {"job_id": job_id, "job_name": identity["job_name"], "comment": f"bmd_attempt:{other}",
            "workdir": f"{WORKSPACE}/uspex_search-other-{other[:8]}", "state": "RUNNING", **fields}


@pytest.mark.parametrize("comments", [True, False])
def test_a_job_sharing_the_name_prefix_is_never_adopted(client, tokens, state_dir, cluster, prepared, comments):
    attempt = prepared[0]
    cluster.sbatch_mode = "lost_before"  # no job of ours exists
    go(client, tokens, prepared)
    cluster.extra_jobs.append(_foreign_job(attempt, cluster))
    cluster.scheduler_comments = comments
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "submission_uncertain"
    assert response.json()["error"]["reconciliation"]["outcome"] == "none"
    assert search_record(state_dir, attempt)["state"] == "SUBMISSION_UNCERTAIN" and cluster.sbatch_runs == 1


def test_recovery_never_accepts_a_job_id_alone(client, tokens, state_dir, cluster, pending):
    attempt, digest = pending[0], pending[1]
    # The record names a job that SLURM reports for another attempt (and ours is not visible).
    cluster.jobs["9001"]["hidden"] = True
    cluster.extra_jobs.append(_foreign_job(attempt, cluster))
    record = search_record(state_dir, attempt)
    record["submission"]["job_id"] = "9500"
    write_search_record(state_dir, record)
    commands = len(cluster.commands)
    response = recover(client, tokens["runner"], attempt, digest, job_id="9500")
    assert response.status_code == 409 and error_code(response) == "job_not_visible"
    # And a job carrying no identity at all under the recorded ID is not evidence either.
    cluster.extra_jobs[:] = [{"job_id": "9500", "job_name": "", "comment": "", "workdir": "", "state": "COMPLETED"}]
    response = recover(client, tokens["runner"], attempt, digest, job_id="9500")
    assert response.status_code == 409 and error_code(response) == "job_not_visible"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"
    assert not any(" sbatch " in command for command in cluster.commands[commands:])
    assert cluster.sbatch_runs == 1


def test_the_recovery_route_never_reaches_sbatch(client, tokens, state_dir, cluster, pending, monkeypatch):
    from compute_api import structure_search_submission as module

    for function in (module._recover_remote, module.recover_remote_record):  # no route to the submission code
        assert "_submit_remote" not in function.__code__.co_names
        assert not any(isinstance(c, str) and "sbatch --" in c for c in function.__code__.co_consts)
    attempt, digest = pending[0], pending[1]
    cluster.sbatch_mode = "accept"
    monkeypatch.setattr(module, "_submit_remote", lambda **kwargs: pytest.fail("submission path used"))
    commands = len(cluster.commands)
    assert recover(client, tokens["runner"], attempt, digest).status_code == 200
    issued = cluster.commands[commands:]
    assert not any("sbatch" in command for command in issued)
    assert any(command.startswith("squeue ") for command in issued) and cluster.sbatch_runs == 1


# --- interrupted state transitions -----------------------------------------------------------------


@pytest.mark.parametrize("point", ["partial", "before_rename"])
def test_interrupted_search_submitting_write_never_runs_sbatch_and_resumes(client, tokens, state_dir, cluster,
                                                                           prepared, point):
    attempt = prepared[0]
    cluster.fail_put = (f"{attempt}.json", point)
    response = go(client, tokens, prepared)
    assert response.status_code == 503 and error_code(response) == "submission_unavailable"
    assert cluster.sbatch_runs == 0 and remote_state(cluster, attempt)["state"] == "SEARCH_PREPARED"
    assert search_record(state_dir, attempt)["state"] == "SUBMITTING"  # only POWER's evidence moves it
    resumed = go(client, tokens, prepared)
    assert resumed.status_code == 200 and resumed.json()["submission"]["job_id"] == "9001"
    assert cluster.sbatch_runs == 1


class _ProcessDied(BaseException):
    """Stands in for the service stopping (not an ordinary error any handler would catch)."""


def _failing_submitted_write(cluster, attempt, failure):
    """A runner factory whose second state write for the attempt (SEARCH_SUBMITTED) raises ``failure``."""

    original = cluster.factory
    writes = []

    def factory():
        runner = original()
        real = runner._replace_file

        def replace(source, destination):
            if destination.endswith(f"{attempt}.json"):
                writes.append(destination)
                if len(writes) == 2:  # SEARCH_SUBMITTING succeeds, SEARCH_SUBMITTED fails
                    raise failure
            return real(source, destination)

        runner._replace_file = replace
        return runner

    return factory


def lose_remote_record(client, tokens, cluster, prepared, failure=None):
    """Submit, losing the SEARCH_SUBMITTED write after sbatch reported job 9001."""

    from compute_api import execution

    original = execution.RUNNER_FACTORY
    execution.RUNNER_FACTORY = _failing_submitted_write(cluster, prepared[0], failure or TimeoutError("timed out"))
    try:
        return go(client, tokens, prepared)
    finally:
        execution.RUNNER_FACTORY = original


def recover(client, token, attempt, digest, job_id="9001", **extra):
    return client.put(f"/api/v1/structure-search-attempts/{attempt}/submission/remote-record",
                      headers=bearer(token), json={"expected_plan_digest": digest, "job_id": job_id, **extra})


@pytest.fixture
def pending(client, tokens, state_dir, cluster, prepared):
    response = lose_remote_record(client, tokens, cluster, prepared)
    assert response.status_code == 200, response.text
    assert response.json()["submission"]["remote_record_pending"] is True
    return prepared


def test_interrupted_search_submitted_write_keeps_the_job_and_never_resubmits(client, tokens, state_dir, cluster,
                                                                              pending):
    attempt = pending[0]
    record = search_record(state_dir, attempt)
    assert record["state"] == "SUBMITTED" and record["submission"]["job_id"] == "9001"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"
    read = get_search(client, tokens["runner"], attempt).json()["submission"]
    assert read["remote_record_pending"] is True and read["remote_record_recovery_available"] is True
    assert go(client, tokens, pending).json()["submission"]["job_id"] == "9001"  # stored, no second sbatch
    assert cluster.sbatch_runs == 1


def test_the_job_id_is_kept_locally_even_if_the_service_stops_after_sbatch(client, tokens, state_dir, cluster,
                                                                           prepared):
    attempt, digest = prepared[0], prepared[1]
    with pytest.raises(_ProcessDied):
        lose_remote_record(client, tokens, cluster, prepared, failure=_ProcessDied())
    record = search_record(state_dir, attempt)
    assert record["state"] == "SUBMITTED" and record["submission"]["job_id"] == "9001"
    assert record["submission"]["remote_record_pending"] is True
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"
    response = recover(client, tokens["runner"], attempt, digest)
    assert response.status_code == 200, response.text
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTED" and cluster.sbatch_runs == 1


def test_recovery_completes_the_remote_record_without_submitting(client, tokens, state_dir, cluster, pending):
    attempt, digest = pending[0], pending[1]
    before = search_record(state_dir, attempt)["submission"]
    response = recover(client, tokens["runner"], attempt, digest)
    assert response.status_code == 200, response.text
    submission = response.json()["submission"]
    assert submission["job_id"] == "9001" and submission["remote_record_pending"] is False
    assert submission["remote_record_recovery_available"] is False
    assert submission["remote_record_recovery"]["scheduler"]["outcome"] == "matched"
    assert submission["submitted_at"] == before["submitted_at"]  # the original evidence is kept
    state = remote_state(cluster, attempt)
    assert state["state"] == "SEARCH_SUBMITTED"
    remote = json.loads(state["output"])["submission"]
    assert remote["job_id"] == "9001" and remote["submitted_at"] == before["submitted_at"]
    assert remote["requested_at"] == before["remote_submission"]["requested_at"]
    assert remote["remote_record_recovery"]["recovered_by"] == "runner"
    assert cluster.sbatch_runs == 1 and list(cluster.jobs) == ["9001"]
    assert not [c for c in cluster.commands if " && sbatch " in c][1:]

    connections = cluster.connections  # idempotent: nothing pending, POWER is not contacted again
    again = recover(client, tokens["runner"], attempt, digest)
    assert again.status_code == 200 and again.json() == response.json() and cluster.connections == connections


def test_recovery_adopts_a_remote_record_an_earlier_recovery_completed(client, tokens, state_dir, cluster, pending):
    attempt, digest = pending[0], pending[1]
    assert recover(client, tokens["runner"], attempt, digest).status_code == 200
    written = remote_state(cluster, attempt)
    record = search_record(state_dir, attempt)
    record["submission"]["remote_record_pending"] = True  # as if the local write after POWER's was lost
    write_search_record(state_dir, record)
    response = recover(client, tokens["runner"], attempt, digest)
    assert response.status_code == 200 and response.json()["submission"]["remote_record_pending"] is False
    assert remote_state(cluster, attempt) == written and cluster.sbatch_runs == 1


def test_recovery_refusals_before_contacting_power(client, tokens, state_dir, cluster, pending, prepared):
    attempt, digest = pending[0], pending[1]
    connections = cluster.connections
    assert recover(client, tokens["preparer"], attempt, digest).status_code == 403
    assert recover(client, tokens["other"], attempt, digest).status_code == 403
    wrong = recover(client, tokens["runner"], attempt, digest, job_id="9002")
    assert wrong.status_code == 409 and error_code(wrong) == "job_id_mismatch"
    assert recover(client, tokens["runner"], attempt, digest, job_id="x").status_code == 422
    assert recover(client, tokens["runner"], attempt, digest, extra="1").status_code == 422
    assert cluster.connections == connections
    ready = prepared_attempt(client, tokens, cluster)
    connections = cluster.connections
    not_submitted = recover(client, tokens["runner"], ready[0], ready[1])
    assert not_submitted.status_code == 409 and error_code(not_submitted) == "attempt_not_submitted"
    assert cluster.connections == connections and remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"


def test_recovery_waits_for_delayed_accounting(client, tokens, state_dir, cluster, pending):
    attempt, digest = pending[0], pending[1]
    cluster.jobs["9001"]["hidden"] = True
    response = recover(client, tokens["runner"], attempt, digest)
    assert response.status_code == 409 and error_code(response) == "job_not_visible"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"
    assert search_record(state_dir, attempt)["submission"]["remote_record_pending"] is True
    cluster.scheduler_answers = False
    unavailable = recover(client, tokens["runner"], attempt, digest)
    assert unavailable.status_code == 503 and error_code(unavailable) == "scheduler_unavailable"
    cluster.scheduler_answers = True
    cluster.jobs["9001"]["hidden"] = False
    assert recover(client, tokens["runner"], attempt, digest).status_code == 200
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTED" and cluster.sbatch_runs == 1


def test_recovery_never_relies_on_an_unavailable_comment(client, tokens, state_dir, cluster, pending):
    attempt, digest = pending[0], pending[1]
    cluster.jobs["9001"]["state"] = "COMPLETED"
    cluster.scheduler_comments = False
    response = recover(client, tokens["runner"], attempt, digest)
    assert response.status_code == 409 and error_code(response) == "job_identity_unverified"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_SUBMITTING"
    assert search_record(state_dir, attempt)["submission"]["remote_record_pending"] is True


@pytest.mark.parametrize("change, code", [
    ("second_job_of_ours", "submission_ambiguous"),
    ("candidate_without_comment", "submission_ambiguous"),
    ("our_directory_other_comment", "submission_conflict"),
    ("only_another_job_id", "submission_conflict"),
    ("remote_names_another_job", "submission_conflict"),
    ("remote_names_another_identity", "submission_conflict"),
    ("remote_prepared", "transition_not_allowed"),
])
def test_recovery_refuses_contradictory_or_ambiguous_records(client, tokens, state_dir, cluster, pending, change,
                                                             code):
    attempt, digest = pending[0], pending[1]
    identity = job_identity(attempt, remote_state(cluster, attempt)["run_dir"])
    ours = {"job_id": "9500", "job_name": identity["job_name"], "comment": identity["comment"],
            "workdir": identity["workdir"], "state": "RUNNING"}
    path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    state = json.loads(path.read_text())
    output = json.loads(state["output"])
    if change == "second_job_of_ours":
        cluster.extra_jobs.append(ours)
    elif change == "candidate_without_comment":
        cluster.extra_jobs.append({**ours, "comment": ""})
    elif change == "our_directory_other_comment":
        cluster.extra_jobs.append({**ours, "comment": "bmd_attempt:someone-else", "job_name": "bmds-other"})
    elif change == "only_another_job_id":
        cluster.jobs["9001"]["hidden"] = True
        cluster.extra_jobs.append(ours)
    elif change == "remote_names_another_job":
        output["submission"]["job_id"] = "9555"
    elif change == "remote_names_another_identity":
        output["submission"]["comment"] = "bmd_attempt:someone-else"
    elif change == "remote_prepared":
        state["state"] = "SEARCH_PREPARED"
        output = output["promotion"]
    state["output"] = json.dumps(output)
    path.write_text(json.dumps(state))
    written = path.read_text()
    response = recover(client, tokens["runner"], attempt, digest)
    assert response.status_code == 409 and error_code(response) == code, response.text
    assert path.read_text() == written  # nothing changed on POWER
    record = search_record(state_dir, attempt)
    assert record["submission"]["job_id"] == "9001" and record["submission"]["remote_record_pending"] is True
    assert cluster.sbatch_runs == 1


def test_search_submitted_before_the_local_record_is_adopted_without_sbatch(client, tokens, state_dir, cluster,
                                                                            prepared):
    attempt = prepared[0]
    assert go(client, tokens, prepared).status_code == 200
    record = search_record(state_dir, attempt)
    record["state"] = "SUBMITTING"  # as if the service stopped after POWER recorded the job
    record["submission"].update(status="submitting", job_id=None)
    write_search_record(state_dir, record)
    response = go(client, tokens, prepared)
    assert response.status_code == 200 and response.json()["submission"]["job_id"] == "9001"
    assert response.json()["submission"]["adopted"] is True and cluster.sbatch_runs == 1


def test_a_remote_submission_of_another_promotion_is_refused(client, tokens, state_dir, cluster, prepared):
    attempt = prepared[0]
    cluster.sbatch_mode = "lost_after"
    go(client, tokens, prepared)
    path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    state = json.loads(path.read_text())
    output = json.loads(state["output"])
    output["promotion"]["job_script"]["sha256"] = "0" * 64
    state["output"] = json.dumps(output)
    path.write_text(json.dumps(state))
    response = go(client, tokens, prepared)
    assert response.status_code == 409 and error_code(response) == "promotion_mismatch"
    assert cluster.sbatch_runs == 1


# --- pure helpers ----------------------------------------------------------------------------------


@pytest.mark.parametrize("stdout, job_id", [("9001\n", "9001"), ("9001;power\n", "9001"), ("", None),
                                            ("Submitted batch job 9001\n", None), ("9001\n9002\n", None),
                                            ("abc\n", None)])
def test_parsable_job_id(stdout, job_id):
    assert parse_parsable_job_id(stdout) == job_id


def test_scheduler_records_skip_steps_and_normalise():
    records = scheduler_records("9001|bmds-x|bmd_attempt:a|/bmd/r/|RUNNING\n",
                                "9001|bmds-x|(null)|/bmd/r|RUNNING\n9001.batch|batch||/bmd/r|RUNNING\nbad line\n")
    assert [(r["source"], r["job_id"], r["comment"], r["workdir"]) for r in records] == [
        ("squeue", "9001", "bmd_attempt:a", "/bmd/r"), ("sacct", "9001", "", "/bmd/r")]
    identity = {"comment": "bmd_attempt:a", "job_name": "bmds-x", "workdir": "/bmd/r"}
    assert reconcile_jobs(records, identity)["outcome"] == "matched"  # the same job from both sources


def test_reconcile_jobs_requires_the_full_identity():
    identity = {"comment": "bmd_attempt:a", "job_name": "bmds-x", "workdir": "/bmd/r"}

    def outcome(*lines):
        return reconcile_jobs(scheduler_records("", "\n".join(lines)), identity)

    assert outcome()["outcome"] == "none"
    assert outcome("9001|bmds-x||/bmd/r|COMPLETED")["outcome"] == "unverified"  # no comment: not proof
    assert outcome("9001|bmds-x|bmd_attempt:a||COMPLETED")["outcome"] == "unverified"  # no directory
    assert outcome("9001|bmds-x||/bmd/elsewhere|COMPLETED")["outcome"] == "none"  # another attempt's name prefix
    assert outcome("9001|bmds-x|bmd_attempt:a|/bmd/r|RUNNING",
                   "9001|bmds-x|bmd_attempt:b|/bmd/r|RUNNING")["outcome"] == "conflict"
    assert outcome("9001|bmds-x|bmd_attempt:a|/bmd/r|RUNNING",
                   "9002|bmds-x||/bmd/r|COMPLETED")["outcome"] == "ambiguous"
    assert outcome("9001|bmds-y|bmd_attempt:a|/bmd/r|RUNNING")["outcome"] == "conflict"
    matched = outcome("9001|bmds-x|bmd_attempt:a|/bmd/r|RUNNING", "9100|other|bmd_attempt:c|/bmd/s|RUNNING")
    assert (matched["outcome"], matched["job_id"]) == ("matched", "9001")
    # Another attempt with the same name prefix runs elsewhere: not a candidate, with or without its comment.
    for foreign in ("9100|bmds-x|bmd_attempt:b|/bmd/s|RUNNING", "9100|bmds-x||/bmd/s|COMPLETED"):
        assert outcome(foreign)["outcome"] == "none"
        assert outcome("9001|bmds-x|bmd_attempt:a|/bmd/r|RUNNING", foreign)["job_id"] == "9001"
    # Array tasks and ranges are never merged into, or hidden behind, a plain job ID.
    split = outcome("9001|bmds-x|bmd_attempt:a|/bmd/r|RUNNING", "9001_[1-2]|bmds-x|bmd_attempt:a|/bmd/r|PENDING")
    assert split["outcome"] == "ambiguous" and split["matches"] == ["9001", "9001_[1-2]"]


# --- calculations and rollback ---------------------------------------------------------------------


def test_submission_leaves_calculations_and_the_v120_reader_untouched(client, tokens, state_dir, cluster, prepared):
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
    assert go(client, tokens, prepared).status_code == 200
    assert (state_dir / "attempts" / f"{calculation}.json").read_bytes() == before
    v120 = _v120_ledger_module().AttemptLedger.from_environment()
    assert [record["attempt_id"] for record in v120.records()] == [calculation]
    assert not cluster.at("/bmd-db/guest/logs/submission_attempts").exists()
    record = json.loads((state_dir / "structure_search" / "attempts" / f"{prepared[0]}.json").read_text())
    assert (record["schema"], record["schema_version"]) == ("bmd_compute.api_structure_search_attempt_record", 1)


def test_the_calculation_path_refuses_every_submission_state(client, tokens, state_dir, cluster, prepared):
    from backend.remote import SubmissionAttemptNotPrepared

    attempt = prepared[0]
    cluster.sbatch_mode = "lost_after"
    go(client, tokens, prepared)
    spec = search_record(state_dir, attempt)["staging"]["spec"]
    runner = cluster.factory()
    for state in ("SEARCH_SUBMITTING", "SEARCH_SUBMITTED"):
        with pytest.raises(SubmissionAttemptNotPrepared):
            runner._submit_prepared_submission_attempt(spec, {**remote_state(cluster, attempt), "state": state}, "")
    assert cluster.sbatch_calls == 0 and cluster.sbatch_runs == 1


def test_the_runtime_package_is_unchanged_by_submission_code():
    """Submission is service-side only, so attempts promoted by 4D-2 (b7bd184) stay submittable."""

    from pathlib import Path

    from backend.runtime_package import runtime_package_relative_paths

    assert not any("submission" in path and "structure_search" in path for path in runtime_package_relative_paths())
    assert Path(__file__).resolve().parents[1].joinpath("compute_api", "structure_search_submission.py").exists()
