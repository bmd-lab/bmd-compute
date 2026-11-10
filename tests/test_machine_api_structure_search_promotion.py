"""Promotion of a STAGED structure search to submission-ready (USPEX step 4D-2), against a fake POWER.

Uses the staging tests' fake POWER (a directory tree, the real finalize and
inventory code run in-process or as scripts) with a stand-in for the USPEX
runtime readiness check. Nothing contacts POWER, runs sbatch, srun, USPEX or
VASP, or submits anything.
"""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from backend.structure_search import job as search_job
from backend.structure_search import lifecycle, staging
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
from test_machine_api_structure_search_attempts import (
    attempt_dir,
    get_search,
    put_search,
    search_body,
    search_record,
    search_record_path,
    write_search_record,
)
from test_machine_api_structure_search_staging import (  # noqa: F401 - pytest fixtures
    SEARCH_LOGS,
    WORKSPACE,
    cluster,
    hidden,
    stage,
    tree,
)

SCRIPT = search_job.JOB_SCRIPT_NAME
ACK = "resource_acknowledgement.json"


def staged_attempt(client, tokens, cluster, *, principal="runner", formula="Si"):
    attempt = new_id()
    body = search_body(client, tokens, formula)
    assert put_search(client, tokens[principal], attempt, body).status_code == 200
    response = stage(client, tokens[principal], attempt, body["expected_plan_digest"])
    assert response.status_code == 200, response.text
    return attempt, body["expected_plan_digest"], response.json()["resources"]["resources_sha256"]


def promote(client, token, attempt, digest, resources_sha256, *, acknowledged=True, **extra):
    body = {"expected_plan_digest": digest,
            "resource_acknowledgement": {"acknowledged": acknowledged, "resources_sha256": resources_sha256},
            **extra}
    return client.put(f"/api/v1/structure-search-attempts/{attempt}/promotion", headers=bearer(token), json=body)


def run_dir_of(cluster, state_dir, attempt) -> Path:
    return cluster.at(f"{WORKSPACE}/{search_record(state_dir, attempt)['staging']['run_name']}")


def remote_state(cluster, attempt) -> dict:
    return json.loads(cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json").read_text())


def expected_script(state_dir, attempt) -> str:
    from backend.runtime_package import runtime_package_manifest_for_sources
    from backend.structure_search.search_plan import PLAN_FILE, load_plan_text
    from backend.submission import build_backend_module_sources

    record = search_record(state_dir, attempt)
    plan = load_plan_text((attempt_dir(state_dir, attempt) / "preparation-1" / PLAN_FILE).read_text())
    sources = build_backend_module_sources()
    return search_job.build_job_script(spec=record["staging"]["spec"], plan=plan,
                                       acknowledgement=record["promotion"]["acknowledgement"],
                                       runtime_sources=sources, manifest=runtime_package_manifest_for_sources(sources),
                                       modules=record["promotion"]["modules"])


# --- success ----------------------------------------------------------------------------------


def test_promotion_uploads_the_frozen_job_script_and_reaches_search_prepared(client, tokens, state_dir, cluster):
    from backend import config

    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    staged = get_search(client, tokens["runner"], attempt).json()
    assert staged["state"] == "STAGED" and staged["promotion_available"] is True and staged["promotion"] is None
    resources = staged["resources"]
    assert resources["status"] == "provisional_estimate" and resources["approved"] is False
    assert resources["request"]["exclusive"] is True and "#SBATCH --exclusive" in resources["sbatch_directives"]
    assert resources["statement"] == search_job.ACKNOWLEDGEMENT_STATEMENT
    run_dir = run_dir_of(cluster, state_dir, attempt)
    staged_files = tree(run_dir)
    connections = cluster.connections

    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"] == "PREPARED" and payload["submission_ready"] is True
    assert payload["submission_available"] is False and payload["promotion_available"] is False
    promotion = payload["promotion"]
    assert promotion["remote_state"] == "SEARCH_PREPARED" and promotion["remote_integrity"] == "last_verified"
    assert promotion["in_progress"] is False and promotion["verified_at"]
    assert promotion["modules"] == list(config.MODULES)
    assert promotion["runtime_check"]["status"] == "passed"
    assert promotion["acknowledgement"]["principal"] == "runner"
    assert promotion["acknowledgement"]["resources_sha256"] == resources_sha256
    assert cluster.connections == connections + 1 and cluster.runtime_checks == 1

    script = (run_dir / SCRIPT).read_text()
    assert script == expected_script(state_dir, attempt)  # rebuilt from the frozen inputs, byte for byte
    assert promotion["job_script"]["sha256"] == hashlib.sha256(script.encode()).hexdigest()
    assert stat.S_IMODE((run_dir / SCRIPT).stat().st_mode) == 0o640
    record = search_record(state_dir, attempt)
    assert json.loads((run_dir / ACK).read_text()) == record["promotion"]["acknowledgement"]
    search_job.verify_resource_acknowledgement(record["promotion"]["acknowledgement"],
                                               plan=_plan(state_dir, attempt), attempt_id=attempt)
    after = tree(run_dir)
    assert {name: after[name] for name in staged_files} == staged_files  # nothing staged was rewritten
    assert sorted(set(after) - set(staged_files)) == sorted([SCRIPT, ACK])
    assert hidden(run_dir) == []

    state = remote_state(cluster, attempt)
    assert state["state"] == "SEARCH_PREPARED" and state["attempt_id"] == attempt
    output = json.loads(state["output"])
    assert output["schema"] == "bmd_compute.structure_search_promotion"
    assert output["job_script"]["sha256"] == promotion["job_script"]["sha256"]
    assert output["acknowledgement_sha256"] == record["promotion"]["acknowledgement"]["acknowledgement_sha256"]
    assert output["plan_digest"] == digest
    assert f"{SEARCH_LOGS}/submission_attempts/{attempt}.json" in cluster.replaced  # written atomically
    assert cluster.sbatch_calls == 0

    # Reads stay local, and report the same.
    read_back = get_search(client, tokens["runner"], attempt)
    assert read_back.status_code == 200 and read_back.json() == payload
    assert cluster.connections == connections + 1


def _plan(state_dir, attempt):
    from backend.structure_search.search_plan import PLAN_FILE, load_plan_text

    return load_plan_text((attempt_dir(state_dir, attempt) / "preparation-1" / PLAN_FILE).read_text())


def test_identical_repeat_returns_the_stored_result_without_contacting_power(client, tokens, state_dir, cluster):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    first = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    connections = cluster.connections
    files = tree(run_dir_of(cluster, state_dir, attempt))
    second = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert second.status_code == 200 and second.json() == first.json()
    assert cluster.connections == connections and tree(run_dir_of(cluster, state_dir, attempt)) == files
    # Staging again returns the stored result too.
    assert stage(client, tokens["runner"], attempt, digest).json()["state"] == "PREPARED"
    assert cluster.connections == connections


# --- refusals before any remote work --------------------------------------------------------


def test_promotion_needs_submit_scope_ownership_and_a_staged_attempt(client, tokens, state_dir, cluster):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    connections = cluster.connections
    assert promote(client, tokens["preparer"], attempt, digest, resources_sha256).status_code == 403  # no submit
    assert promote(client, tokens["other"], attempt, digest, resources_sha256).status_code == 403  # not the owner
    assert promote(client, tokens["runner"], new_id(), digest, resources_sha256).status_code == 404
    wrong = promote(client, tokens["runner"], attempt, "sha256:" + "0" * 64, resources_sha256)
    assert wrong.status_code == 409 and wrong.json()["error"]["code"] == "plan_digest_mismatch"
    unstaged = new_id()
    body = search_body(client, tokens)
    put_search(client, tokens["runner"], unstaged, body)
    refused = promote(client, tokens["runner"], unstaged, body["expected_plan_digest"], resources_sha256)
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "attempt_not_staged"
    calculation = new_id()
    previous = cluster.factory
    from compute_api import execution

    execution.RUNNER_FACTORY = cluster.calculation_factory
    try:
        assert put_calculation(client, tokens["runner"], calculation,
                               attempt_body(client, tokens, submit=False)).status_code == 200
    finally:
        execution.RUNNER_FACTORY = previous
    kind = promote(client, tokens["runner"], calculation, digest, resources_sha256)
    assert kind.status_code == 409 and kind.json()["error"]["code"] == "attempt_kind_mismatch"
    assert cluster.connections == connections and search_record(state_dir, attempt)["state"] == "STAGED"


@pytest.mark.parametrize("body, problem", [
    ({"resource_acknowledgement": None}, "required"),
    ({"expected_plan_digest": None}, "required"),
    ({"resource_acknowledgement": {"acknowledged": False}}, "must_be_true"),
    ({"resource_acknowledgement": {"acknowledged": "yes"}}, "must_be_true"),
    ({"resource_acknowledgement": {"acknowledged": True, "resources_sha256": "abc"}}, "must_be_sha256_digest"),
    ({"resource_acknowledgement": {"acknowledged": True, "resources_sha256": "sha256:" + "0" * 64,
                                   "walltime": "1:00:00"}}, "unknown_field"),
    ({"submit": True}, "submission_not_available"),
])
def test_malformed_promotion_requests_are_refused(client, tokens, state_dir, cluster, body, problem):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    document = {"expected_plan_digest": digest,
                "resource_acknowledgement": {"acknowledged": True, "resources_sha256": resources_sha256}}
    for key, value in body.items():
        if value is None:
            document.pop(key)
        else:
            document[key] = value
    response = client.put(f"/api/v1/structure-search-attempts/{attempt}/promotion",
                          headers=bearer(tokens["runner"]), json=document)
    assert response.status_code == 422
    assert problem in {field["problem"] for field in response.json()["error"]["fields"]}


def test_a_changed_acknowledgement_is_refused(client, tokens, state_dir, cluster):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    connections = cluster.connections
    other = "sha256:" + "1" * 64
    response = promote(client, tokens["runner"], attempt, digest, other)
    assert response.status_code == 409 and response.json()["error"]["code"] == "resources_changed"
    assert response.json()["error"]["resources_sha256"] == resources_sha256
    assert "promotion" not in search_record(state_dir, attempt) or not search_record(state_dir, attempt)["promotion"]
    assert cluster.connections == connections

    cluster.runtime_problems = ["The USPEX cache is not ready."]  # leaves a frozen, unfinished promotion
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 409
    frozen = search_record(state_dir, attempt)["promotion"]
    changed = promote(client, tokens["runner"], attempt, digest, other)
    assert changed.status_code == 409 and changed.json()["error"]["code"] == "acknowledgement_mismatch"
    assert search_record(state_dir, attempt)["promotion"] == frozen


def test_a_stale_plan_is_refused_before_contacting_power(client, tokens, state_dir, cluster, monkeypatch):
    from backend.structure_search import search_plan

    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    connections = cluster.connections

    def stale(plan, **kwargs):
        raise search_plan.SearchPlanError("policy changed", code="plan_stale", changed=["incar_files"])

    monkeypatch.setattr(search_plan, "require_current_plan", stale)
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 409 and response.json()["error"]["code"] == "plan_changed"
    assert cluster.connections == connections and search_record(state_dir, attempt)["state"] == "STAGED"


def test_a_changed_runtime_package_is_refused_before_contacting_power(client, tokens, state_dir, cluster,
                                                                       monkeypatch):
    import backend.submission as submission

    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    connections = cluster.connections
    real_sources = submission.build_backend_module_sources
    monkeypatch.setattr(submission, "build_backend_module_sources",
                        lambda: {**real_sources(), "structure_search/extra_module.py": "X = 1\n"})
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 409 and response.json()["error"]["code"] == "runtime_package_changed"
    assert cluster.connections == connections


def test_an_altered_uploaded_package_on_power_is_refused(client, tokens, state_dir, cluster):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    run_dir = run_dir_of(cluster, state_dir, attempt)
    module = run_dir / "backend" / "structure_search" / "job.py"
    module.write_text(module.read_text() + "\n# changed on POWER\n")
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 409 and response.json()["error"]["code"] in {"staging_damaged",
                                                                               "runtime_package_mismatch"}
    assert remote_state(cluster, attempt)["state"] == "STAGED"
    assert search_record(state_dir, attempt)["state"] == "STAGED"


# --- damaged inputs and failed checks --------------------------------------------------------


@pytest.mark.parametrize("damage", ["incar", "foreign_script", "job_started"])
def test_damaged_or_foreign_files_are_refused_and_never_replaced(client, tokens, state_dir, cluster, damage):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    run_dir = run_dir_of(cluster, state_dir, attempt)
    if damage == "incar":
        incar = run_dir / "search" / "Specific" / "INCAR_3"
        incar.write_text(incar.read_text() + "EDIFF = 1E-2\n")
    elif damage == "foreign_script":
        (run_dir / SCRIPT).write_text("#!/bin/bash\necho not ours\n")
    else:
        (run_dir / "job_started").write_text("123\n")  # a job already ran here
    before = tree(run_dir)
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 409 and response.json()["error"]["code"] == "staging_damaged"
    assert tree(run_dir) == before  # nothing uploaded, nothing replaced
    assert remote_state(cluster, attempt)["state"] == "STAGED"
    assert search_record(state_dir, attempt)["state"] == "STAGED"


def test_a_failed_runtime_check_keeps_the_attempt_staged_and_a_retry_completes_it(client, tokens, state_dir,
                                                                                  cluster):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    cluster.runtime_problems = ["USPEX is not unpacked in the shared cache: .ready is missing."]
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 409 and response.json()["error"]["code"] == "runtime_check_failed"
    assert ".ready is missing" in response.json()["error"]["problems"][0]
    assert remote_state(cluster, attempt)["state"] == "STAGED"
    record = search_record(state_dir, attempt)
    assert record["state"] == "STAGED" and record["promotion"]["verified_at"] is None
    staged = get_search(client, tokens["runner"], attempt).json()
    assert staged["promotion"]["in_progress"] is True
    run_dir = run_dir_of(cluster, state_dir, attempt)
    uploaded = tree(run_dir)[SCRIPT]

    cluster.runtime_problems = []
    retried = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert retried.status_code == 200 and retried.json()["state"] == "PREPARED"
    assert tree(run_dir)[SCRIPT] == uploaded  # the frozen script, never rewritten
    assert retried.json()["promotion"]["job_script"]["sha256"] == record["promotion"]["job_script"]["sha256"]


# --- interruptions ------------------------------------------------------------------------------


@pytest.mark.parametrize("target, point", [
    (SCRIPT, "partial"), (SCRIPT, "before_rename"), (ACK, "partial"), (ACK, "before_chmod"),
])
def test_interrupted_upload_resumes_from_the_frozen_inputs(client, tokens, state_dir, cluster, target, point):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    cluster.fail_put = (target, point)
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 503
    run_dir = run_dir_of(cluster, state_dir, attempt)
    assert not (run_dir / target).exists() and len(hidden(run_dir)) == 1
    frozen = search_record(state_dir, attempt)["promotion"]
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 200 and response.json()["state"] == "PREPARED", response.text
    assert search_record(state_dir, attempt)["promotion"]["acknowledgement"] == frozen["acknowledgement"]
    assert hidden(run_dir) == []


@pytest.mark.parametrize("when", ["before", "after"])
def test_a_finalize_timeout_during_promotion_is_recovered(client, tokens, state_dir, cluster, when):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    cluster.finalize_timeout = when
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 503
    assert remote_state(cluster, attempt)["state"] == "STAGED"  # written only after verification answered
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 200 and response.json()["state"] == "PREPARED"


@pytest.mark.parametrize("point", ["partial", "before_chmod", "before_rename"])
def test_an_interrupted_state_write_leaves_the_previous_state_intact(client, tokens, state_dir, cluster, point):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    state_path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    before = state_path.read_bytes()
    cluster.fail_put = (f"{attempt}.json", point)
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 503
    assert state_path.read_bytes() == before  # the old, complete state: never a partial file
    assert json.loads(before)["state"] == "STAGED"
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 200 and response.json()["state"] == "PREPARED"
    assert remote_state(cluster, attempt)["state"] == "SEARCH_PREPARED"


def test_remote_search_prepared_is_adopted_after_verifying_matching_identities(client, tokens, state_dir, cluster):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 200
    run_dir = run_dir_of(cluster, state_dir, attempt)
    files = tree(run_dir)
    record = search_record(state_dir, attempt)
    record["state"] = "STAGED"  # as if the service stopped after POWER recorded SEARCH_PREPARED
    record["promotion"].update(verified_at=None, result=None)
    write_search_record(state_dir, record)
    checks = cluster.runtime_checks
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 200 and response.json()["state"] == "PREPARED"
    assert cluster.runtime_checks == checks + 1  # verified again before adoption
    assert tree(run_dir) == files


def test_remote_search_prepared_with_other_identities_is_refused(client, tokens, state_dir, cluster):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 200
    state_path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    state = json.loads(state_path.read_text())
    output = json.loads(state["output"])
    output["job_script"]["sha256"] = "0" * 64
    state["output"] = json.dumps(output)
    state_path.write_text(json.dumps(state))
    before = state_path.read_bytes()
    record = search_record(state_dir, attempt)
    record["state"] = "STAGED"
    record["promotion"].update(verified_at=None, result=None)
    write_search_record(state_dir, record)
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 409 and response.json()["error"]["code"] == "promotion_mismatch"
    assert state_path.read_bytes() == before and search_record(state_dir, attempt)["state"] == "STAGED"


@pytest.mark.parametrize("remote", ["SEARCH_SUBMITTING", "SEARCH_SUBMITTED", "STAGING"])
def test_promotion_follows_the_lifecycle(client, tokens, state_dir, cluster, remote):
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    state_path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    state = json.loads(state_path.read_text())
    state["state"] = remote
    state_path.write_text(json.dumps(state))
    response = promote(client, tokens["runner"], attempt, digest, resources_sha256)
    assert response.status_code == 409 and response.json()["error"]["code"] == "transition_not_allowed"
    assert json.loads(state_path.read_text())["state"] == remote


# --- no SLURM job, and nothing else changes -----------------------------------------------------


def test_promotion_never_submits_or_starts_anything(client, tokens, state_dir, cluster):
    from backend.paramiko_remote import ParamikoRemoteRunner
    from backend.remote import SubmissionAttemptNotPrepared

    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    commands = len(cluster.commands)
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 200
    for command in cluster.commands[commands:]:
        first = command.split()[0]
        assert first in {"mkdir", "rmdir", "cd"} or staging.INVENTORY_DIR_MARKER in command, command[:80]
    assert cluster.sbatch_calls == 0
    spec = search_record(state_dir, attempt)["staging"]["spec"]
    state = remote_state(cluster, attempt)
    runner = cluster.factory()
    with pytest.raises(SubmissionAttemptNotPrepared):  # the calculation path refuses SEARCH_PREPARED
        runner._submit_prepared_submission_attempt(spec, state, "")
    assert isinstance(runner, ParamikoRemoteRunner) and cluster.sbatch_calls == 0


def test_promotion_readiness_check_is_the_login_node_scope_and_runs_no_command(monkeypatch):
    from backend.structure_search import runtime

    seen = {}
    monkeypatch.setattr(runtime, "search_runtime_check",
                        lambda record, *, scope="job", slurm_version_lookup=None: seen.update(scope=scope) or
                        {"status": "passed", "problems": [], "observed": {}, "scope": scope})
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("no command at promotion"))
    assert staging.promotion_runtime_check({"schema": "x"})["scope"] == "login_node"
    assert seen == {"scope": "login_node"}


def test_job_script_check_on_power(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = "#!/bin/bash\necho ok\n"
    (run_dir / SCRIPT).write_text(script)
    request = {"run_dir": str(run_dir), "job_script": {"name": SCRIPT,
                                                        "sha256": hashlib.sha256(script.encode()).hexdigest()}}
    assert staging._verify_job_script(request, "/") == {"name": SCRIPT, "sha256": request["job_script"]["sha256"],
                                                        "syntax": "ok"}
    for text, code in (("#!/bin/bash\nif then\n", "job_script_mismatch"),):
        (run_dir / SCRIPT).write_text(text)
        with pytest.raises(staging.StagingError) as refused:
            staging._verify_job_script(request, "/")
        assert refused.value.code == code
    bad = "#!/bin/bash\nif then\n"
    (run_dir / SCRIPT).write_text(bad)
    with pytest.raises(staging.StagingError) as refused:
        staging._verify_job_script({**request, "job_script": {"name": SCRIPT, "sha256":
                                                             hashlib.sha256(bad.encode()).hexdigest()}}, "/")
    assert refused.value.code == "job_script_invalid"
    (run_dir / SCRIPT).unlink()
    with pytest.raises(staging.StagingError) as refused:
        staging._verify_job_script(request, "/")
    assert refused.value.code == "job_script_missing"


def test_promotion_leaves_calculations_and_the_v120_reader_untouched(client, tokens, state_dir, cluster):
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
    attempt, digest, resources_sha256 = staged_attempt(client, tokens, cluster)
    assert promote(client, tokens["runner"], attempt, digest, resources_sha256).status_code == 200
    assert (state_dir / "attempts" / f"{calculation}.json").read_bytes() == before
    v120 = _v120_ledger_module().AttemptLedger.from_environment()
    assert [record["attempt_id"] for record in v120.records()] == [calculation]
    assert not cluster.at("/bmd-db/guest/logs/submission_attempts").exists()  # calculation states untouched
    # The search ledger's schema is unchanged: the 4C-era reader still reads a PREPARED record.
    record = json.loads(search_record_path(state_dir, attempt).read_text())
    assert (record["schema"], record["schema_version"]) == ("bmd_compute.api_structure_search_attempt_record", 1)
