"""Structure-search attempts: managed, idempotent local preparation (USPEX step 4B).

The fixtures are the calculation-attempt tests' (a private API state root,
fake POWER that records any connection, tokens with prepare scope), so both
kinds of attempt share one ledger here exactly as in the service.
"""

from __future__ import annotations

import json
import pathlib
import os
import stat
import subprocess

import pytest

from backend.structure_search import search_plan
from backend.structure_search.search_plan import (
    PLAN_FILE,
    PREPARED_FILE,
    UNBOUND_BRIDGE_PATH,
    build_search_plan_from_formula,
    load_plan_text,
    verify_prepared_directory,
)
from compute_api import structure_search_execution as search_execution
from test_machine_api_execution import (  # noqa: F401 - pytest fixtures
    attempt_body,
    bearer,
    client,
    get as get_calculation,
    ledger_record,
    new_id,
    no_remote_activity,
    power,
    put as put_calculation,
    state_dir,
    tokens,
    write_ledger_record,
)


ROUTE = "/api/v1/structure-search-attempts"


def search_digest(client, tokens, formula="Si", search=None):
    body = {"formula": formula, **({"search": search} if search else {})}
    response = client.post("/api/v1/structure-search-plans", headers=bearer(tokens["planner"]), json=body)
    assert response.status_code == 200, response.text
    return response.json()["plan_digest"]


def search_body(client, tokens, formula="Si", search=None):
    body = {"formula": formula, "expected_plan_digest": search_digest(client, tokens, formula, search)}
    if search:
        body["search"] = search
    return body


def put_search(client, token, attempt_id, body):
    return client.put(f"{ROUTE}/{attempt_id}", headers=bearer(token), json=body)


def get_search(client, token, attempt_id):
    return client.get(f"{ROUTE}/{attempt_id}", headers=bearer(token))


def attempt_dir(state_dir, attempt_id):
    return state_dir / "structure_search" / "preparations" / attempt_id


def search_record_path(state_dir, attempt_id):
    return state_dir / "structure_search" / "attempts" / f"{attempt_id}.json"


def search_record(state_dir, attempt_id):
    return json.loads(search_record_path(state_dir, attempt_id).read_text())


def write_search_record(state_dir, record):
    search_record_path(state_dir, record["attempt_id"]).write_text(json.dumps(record))


def tree(root):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def everything(state_dir):
    return sorted(str(path.relative_to(state_dir)) for path in state_dir.rglob("*"))


# --- refusals before any write -------------------------------------------------------


def test_stale_digest_is_refused_before_any_ledger_or_file_write(client, tokens, state_dir, power):
    body = {"formula": "Si", "expected_plan_digest": "sha256:" + "0" * 64}
    response = put_search(client, tokens["preparer"], new_id(), body)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "plan_changed"
    assert response.json()["error"]["plan_digest"] == search_digest(client, tokens)
    assert everything(state_dir) == []
    assert no_remote_activity(power)


def test_policy_change_since_approval_is_refused_before_writing(client, tokens, state_dir, monkeypatch):
    from backend.structure_search import incar_policy

    body = search_body(client, tokens)
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    assert put_search(client, tokens["preparer"], new_id(), body).status_code == 409
    assert everything(state_dir) == []


@pytest.mark.parametrize(
    ("mutate", "field", "problem"),
    [
        (lambda body: body.pop("expected_plan_digest"), "expected_plan_digest", "required"),
        (lambda body: body.update(submit=True), "submit", "submission_not_available"),
        (lambda body: body.update(submit=False), "submit", "submission_not_available"),
        (lambda body: body.update(structure={"format": "poscar", "text": "x"}), "structure", "not_accepted_for_structure_search"),
        (lambda body: body.update(search={"external_pressure_gpa": 5}), "search.external_pressure_gpa", "fixed_by_bmd_policy"),
        (lambda body: body.update(formula=""), "formula", "must_be_nonempty_string"),
    ],
)
def test_malformed_requests_are_refused_before_any_write(client, tokens, state_dir, mutate, field, problem):
    body = search_body(client, tokens)
    mutate(body)
    response = put_search(client, tokens["preparer"], new_id(), body)
    assert response.status_code == 422
    assert {"field": field, "problem": problem} in response.json()["error"]["fields"]
    assert everything(state_dir) == []


def test_scope_attempt_id_and_formula_are_checked_before_any_write(client, tokens, state_dir):
    body = search_body(client, tokens)
    assert put_search(client, tokens["planner"], new_id(), body).status_code == 403  # no prepare scope
    assert put_search(client, tokens["preparer"], "NOT-A-UUID", body).status_code == 422
    unsupported = {"formula": "Fe2O3", "expected_plan_digest": "sha256:" + "0" * 64}
    response = put_search(client, tokens["preparer"], new_id(), unsupported)
    assert response.status_code == 422 and response.json()["error"]["code"] == "calculation_invalid"
    assert everything(state_dir) == []


# --- preparation ---------------------------------------------------------------------


def test_preparation_writes_the_frozen_plan_inputs_locally(client, tokens, state_dir, power):
    attempt = new_id()
    body = search_body(client, tokens, "Mg2SiO4", {"num_generations": 20})
    response = put_search(client, tokens["preparer"], attempt, body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"] == "PREPARED_LOCALLY"
    assert payload["attempt_kind"] == "structure_search"
    assert payload["plan_digest"] == body["expected_plan_digest"]
    assert payload["submission_available"] is False and payload["remote_staging_available"] is True  # step 4C
    assert payload["preparation"]["bridge_path_bound"] is False
    assert payload["preparation"]["contains_potcar_contents"] is False
    assert payload["preparation"]["integrity"] == "verified"
    assert str(state_dir) not in response.text  # no local paths in responses

    directory = attempt_dir(state_dir, attempt) / "preparation-1"
    manifest = verify_prepared_directory(directory)
    assert manifest["files"] == payload["preparation"]["files"]
    assert manifest["bindings"] == {"bridge_path": UNBOUND_BRIDGE_PATH, "bound": False}
    plan = load_plan_text((directory / PLAN_FILE).read_text())
    assert plan.digest == body["expected_plan_digest"]
    from backend.structure_search.prediction_plan import AdvancedSettings

    assert plan.canonical == build_search_plan_from_formula("Mg2SiO4", AdvancedSettings(num_generations=20)).canonical
    for name, entry in plan.record["incar_files"].items():
        assert (directory / "Specific" / name).read_text() == entry["text"]
    assert (directory / "POTCAR.spec").read_text() == "Mg\nSi\nO\n"
    assert UNBOUND_BRIDGE_PATH in (directory / "INPUT.txt").read_text()
    for path in directory.rglob("*"):
        assert not path.name.startswith("POTCAR_") and path.name != "POTCAR"
        if path.is_file():
            data = path.read_bytes()
            assert b"VRHFIN" not in data and b"PSCTR" not in data
    for private in (state_dir / "structure_search", state_dir / "structure_search" / "attempts",
                    state_dir / "structure_search" / "preparations", attempt_dir(state_dir, attempt)):
        assert stat.S_IMODE(private.stat().st_mode) == 0o700

    record = search_record(state_dir, attempt)
    assert record["schema"] == "bmd_compute.api_structure_search_attempt_record"
    assert record["attempt_kind"] == "structure_search"
    assert not (state_dir / "attempts" / f"{attempt}.json").exists()
    assert record["state"] == "PREPARED_LOCALLY" and record["principal"] == "preparer"
    assert record["preparation"]["manifest_sha256"] == payload["preparation"]["manifest_sha256"]
    assert no_remote_activity(power)


def test_repeat_preparation_is_idempotent(client, tokens, state_dir, power):
    attempt = new_id()
    body = search_body(client, tokens)
    first = put_search(client, tokens["preparer"], attempt, body)
    record_bytes = search_record_path(state_dir, attempt).read_bytes()
    files = tree(attempt_dir(state_dir, attempt))
    second = put_search(client, tokens["preparer"], attempt, {**body})
    reordered = put_search(client, tokens["preparer"], attempt, json.loads(json.dumps(body, sort_keys=True)))
    assert first.status_code == second.status_code == reordered.status_code == 200
    assert first.json() == second.json() == reordered.json()
    assert search_record_path(state_dir, attempt).read_bytes() == record_bytes
    assert tree(attempt_dir(state_dir, attempt)) == files
    assert sorted(path.name for path in attempt_dir(state_dir, attempt).iterdir()) == ["preparation-1"]
    assert no_remote_activity(power)


def test_conflicting_requests_under_one_attempt_id_are_refused(client, tokens, state_dir):
    attempt = new_id()
    assert put_search(client, tokens["preparer"], attempt, search_body(client, tokens, "Si")).status_code == 200
    before = search_record_path(state_dir, attempt).read_bytes()
    for body in (
        search_body(client, tokens, "GaAs"),
        search_body(client, tokens, "Si", {"num_generations": 20}),
    ):
        response = put_search(client, tokens["preparer"], attempt, body)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "attempt_request_mismatch"
    # The same request from another principal is refused too.
    assert put_search(client, tokens["other"], attempt, search_body(client, tokens, "Si")).status_code == 403
    assert search_record_path(state_dir, attempt).read_bytes() == before


def test_equivalent_formula_is_still_a_different_request(client, tokens):
    attempt = new_id()
    assert put_search(client, tokens["preparer"], attempt, search_body(client, tokens, "SiO2")).status_code == 200
    response = put_search(client, tokens["preparer"], attempt, search_body(client, tokens, "O2Si"))
    assert response.status_code == 409


# --- read-back -----------------------------------------------------------------------


def test_read_back_after_a_lost_response(client, tokens, state_dir, power):
    attempt = new_id()
    body = search_body(client, tokens)
    put_search(client, tokens["preparer"], attempt, body)  # the client never sees this response
    response = get_search(client, tokens["preparer"], attempt)
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "PREPARED_LOCALLY" and payload["preparation"]["integrity"] == "verified"
    assert payload["plan_digest"] == body["expected_plan_digest"]
    assert put_search(client, tokens["preparer"], attempt, body).json() == payload
    assert no_remote_activity(power)


def test_read_back_rules(client, tokens, state_dir):
    attempt = new_id()
    put_search(client, tokens["preparer"], attempt, search_body(client, tokens))
    assert get_search(client, tokens["other"], attempt).status_code == 403
    assert get_search(client, tokens["submit_only"], attempt).status_code == 403  # no read scope
    missing = get_search(client, tokens["preparer"], new_id())
    assert missing.status_code == 404
    assert get_search(client, tokens["preparer"], "nope").status_code == 422


# --- interrupted and damaged preparation --------------------------------------------


def test_interrupted_preparation_is_completed_without_touching_the_partial_directory(
    client, tokens, state_dir, monkeypatch
):
    attempt = new_id()
    body = search_body(client, tokens)
    real_prepare = search_plan.prepare_search_directory

    def crash(plan, destination, **kwargs):
        os.mkdir(destination, 0o755)
        (destination / "INPUT.txt").write_text("partial")
        raise RuntimeError("service stopped")  # no clean-up, as after a kill

    monkeypatch.setattr(search_plan, "prepare_search_directory", crash)
    failed = put_search(client, tokens["preparer"], attempt, body)
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "preparation_failed"
    assert search_record(state_dir, attempt)["state"] == "PREPARING"
    pending = get_search(client, tokens["preparer"], attempt).json()
    assert pending["state"] == "PREPARING" and pending["preparation"]["integrity"] == "incomplete"

    monkeypatch.setattr(search_plan, "prepare_search_directory", real_prepare)
    partial = attempt_dir(state_dir, attempt) / "preparation-1" / "INPUT.txt"
    partial_before = (partial.read_bytes(), partial.stat().st_mtime_ns)
    resumed = put_search(client, tokens["preparer"], attempt, body)
    assert resumed.status_code == 200
    assert resumed.json()["state"] == "PREPARED_LOCALLY"
    assert resumed.json()["preparation"]["interrupted_preparations"] == 1
    assert (partial.read_bytes(), partial.stat().st_mtime_ns) == partial_before
    record = search_record(state_dir, attempt)
    assert record["preparation"]["directory"] == "preparation-2"
    assert record["preparation"]["abandoned"] == ["preparation-1"]
    verify_prepared_directory(attempt_dir(state_dir, attempt) / "preparation-2")


def test_preparation_finished_before_the_record_update_is_adopted(client, tokens, state_dir):
    attempt = new_id()
    body = search_body(client, tokens)
    put_search(client, tokens["preparer"], attempt, body)
    files = tree(attempt_dir(state_dir, attempt))
    record = search_record(state_dir, attempt)
    record["state"] = "PREPARING"  # as if the service stopped before recording success
    record["preparation"].update(manifest_sha256=None, files=None, prepared_at=None)
    write_search_record(state_dir, record)
    response = put_search(client, tokens["preparer"], attempt, body)
    assert response.status_code == 200 and response.json()["state"] == "PREPARED_LOCALLY"
    assert tree(attempt_dir(state_dir, attempt)) == files  # adopted, not rewritten
    assert search_record(state_dir, attempt)["preparation"]["directory"] == "preparation-1"


def test_damaged_preparation_is_reported_and_never_replaced(client, tokens, state_dir):
    attempt = new_id()
    body = search_body(client, tokens)
    put_search(client, tokens["preparer"], attempt, body)
    incar = attempt_dir(state_dir, attempt) / "preparation-1" / "Specific" / "INCAR_3"
    incar.write_text(incar.read_text() + "EXTRA = 1\n")
    damaged = incar.read_bytes()
    assert get_search(client, tokens["preparer"], attempt).json()["preparation"]["integrity"] == "damaged"
    response = put_search(client, tokens["preparer"], attempt, body)
    assert response.status_code == 409 and response.json()["error"]["code"] == "preparation_damaged"
    assert incar.read_bytes() == damaged
    assert sorted(path.name for path in attempt_dir(state_dir, attempt).iterdir()) == ["preparation-1"]


# --- one ledger, two kinds of attempt -------------------------------------------------


def test_calculation_and_search_attempts_cannot_use_each_others_ids(client, tokens, state_dir, power):
    search_attempt = new_id()
    put_search(client, tokens["runner"], search_attempt, search_body(client, tokens))
    calculation = attempt_body(client, tokens, submit=False)
    response = put_calculation(client, tokens["runner"], search_attempt, calculation)
    assert response.status_code == 409 and response.json()["error"]["code"] == "attempt_kind_mismatch"
    response = get_calculation(client, tokens["runner"], search_attempt)
    assert response.status_code == 409 and response.json()["error"]["code"] == "attempt_kind_mismatch"

    calculation_attempt = new_id()
    assert put_calculation(client, tokens["runner"], calculation_attempt, calculation).status_code == 200
    response = put_search(client, tokens["runner"], calculation_attempt, search_body(client, tokens))
    assert response.status_code == 409 and response.json()["error"]["code"] == "attempt_kind_mismatch"
    response = get_search(client, tokens["runner"], calculation_attempt)
    assert response.status_code == 409


def test_calculation_records_are_unchanged_and_still_readable(client, tokens, state_dir, power):
    calculation_attempt = new_id()
    calculation = attempt_body(client, tokens, submit=False)
    first = put_calculation(client, tokens["runner"], calculation_attempt, calculation)
    assert first.status_code == 200
    legacy = ledger_record(state_dir, calculation_attempt)
    assert "attempt_kind" not in legacy  # calculation records keep their historical form

    put_search(client, tokens["runner"], new_id(), search_body(client, tokens))
    # Calculation attempts keep working with a search record in the same ledger.
    assert get_calculation(client, tokens["runner"], calculation_attempt).status_code == 200
    another = new_id()
    assert put_calculation(client, tokens["runner"], another, calculation).status_code == 200
    assert ledger_record(state_dir, calculation_attempt).keys() == legacy.keys()


def test_search_attempts_never_connect_or_run_programs(client, tokens, state_dir, power, monkeypatch):
    real_run = subprocess.run

    def only_git(command, *args, **kwargs):
        assert os.path.basename(str(list(command)[0])) == "git", command
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", only_git)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Popen")))
    import compute_api.execution as execution

    monkeypatch.setattr(execution, "prepare_remote_submission", lambda *a, **k: (_ for _ in ()).throw(AssertionError("remote")))
    monkeypatch.setattr(execution, "submit_remote_workflow", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sbatch")))
    attempt = new_id()
    body = search_body(client, tokens)
    assert put_search(client, tokens["preparer"], attempt, body).status_code == 200
    assert put_search(client, tokens["preparer"], attempt, body).status_code == 200
    assert get_search(client, tokens["preparer"], attempt).status_code == 200
    assert no_remote_activity(power)


# --- backward compatibility (rollback to v1.2.0) --------------------------------------

V120_LEDGER = pathlib.Path(__file__).parent / "fixtures" / "ledger_v1_2_0" / "ledger.py"


def _v120_ledger_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bmd_compute_v120_ledger", V120_LEDGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_is_the_released_v120_ledger():
    import hashlib

    # git show v1.2.0:compute_api/ledger.py
    assert hashlib.sha256(V120_LEDGER.read_bytes()).hexdigest() == (
        "89c621f7c76088d14e72503846b07e003bbb27292ed00eaac7d2f96615bf960c"
    )


def _calculation_records(state_dir):
    return {path.name: path.read_bytes() for path in sorted((state_dir / "attempts").glob("*.json"))}


def test_v120_ledger_still_handles_calculation_attempts_after_search_attempts(client, tokens, state_dir, power):
    calculation_attempt = new_id()
    calculation = attempt_body(client, tokens, submit=False)
    assert put_calculation(client, tokens["runner"], calculation_attempt, calculation).status_code == 200
    for formula in ("Si", "GaAs"):
        assert put_search(client, tokens["runner"], new_id(), search_body(client, tokens, formula)).status_code == 200

    v120 = _v120_ledger_module()
    ledger = v120.AttemptLedger.from_environment()
    records = ledger.records()  # v1.2.0's run-timestamp collision scan reads every record
    assert [record["attempt_id"] for record in records] == [calculation_attempt]
    assert all("run_timestamp" in record for record in records)
    assert ledger.read(calculation_attempt)["principal"] == "runner"

    # v1.2.0 creates and updates a calculation record next to the search records.
    record = dict(records[0], attempt_id=new_id())
    assert ledger.create(record) is True
    with ledger.locked():
        ledger.replace({**record, "state": "PREPARED"})
    assert ledger.read(record["attempt_id"])["state"] == "PREPARED"
    assert len(ledger.records()) == 2


def test_search_attempts_never_modify_calculation_records(client, tokens, state_dir, power, monkeypatch):
    calculation_attempt = new_id()
    assert put_calculation(client, tokens["runner"], calculation_attempt, attempt_body(client, tokens, submit=False)).status_code == 200
    before = _calculation_records(state_dir)
    attempt = new_id()
    body = search_body(client, tokens)
    put_search(client, tokens["runner"], attempt, body)
    put_search(client, tokens["runner"], attempt, body)
    get_search(client, tokens["runner"], attempt)
    put_search(client, tokens["runner"], attempt, search_body(client, tokens, "GaAs"))  # refused
    assert _calculation_records(state_dir) == before
    assert sorted(path.name for path in (state_dir / "attempts").iterdir()) == sorted(before)


def test_rollback_needs_no_migration_in_either_direction(client, tokens, state_dir, power):
    # Records written by v1.2.0's ledger code are read unchanged by this release.
    v120 = _v120_ledger_module()
    calculation_attempt = new_id()
    assert put_calculation(client, tokens["runner"], calculation_attempt, attempt_body(client, tokens, submit=False)).status_code == 200
    legacy = ledger_record(state_dir, calculation_attempt)
    assert "attempt_kind" not in legacy and legacy["schema"] == "bmd_compute.api_attempt"
    search_attempt = new_id()
    put_search(client, tokens["runner"], search_attempt, search_body(client, tokens))
    # "Rollback": v1.2.0 code works on the same state root ...
    assert v120.AttemptLedger.from_environment().read(calculation_attempt) == legacy
    # ... and "roll forward": this release still serves both attempts.
    assert get_calculation(client, tokens["runner"], calculation_attempt).status_code == 200
    assert get_search(client, tokens["runner"], search_attempt).json()["state"] == "PREPARED_LOCALLY"


def test_attempt_ids_are_unique_across_kinds_even_after_a_rollback_collision(client, tokens, state_dir, power):
    shared = new_id()
    assert put_search(client, tokens["runner"], shared, search_body(client, tokens)).status_code == 200
    # An older release, unaware of search attempts, could create a calculation with the same ID.
    v120 = _v120_ledger_module()
    calculation = new_id()
    put_calculation(client, tokens["runner"], calculation, attempt_body(client, tokens, submit=False))
    v120.AttemptLedger.from_environment().create(dict(ledger_record(state_dir, calculation), attempt_id=shared))
    for response in (
        get_calculation(client, tokens["runner"], shared),
        put_calculation(client, tokens["runner"], shared, attempt_body(client, tokens, submit=False)),
        get_search(client, tokens["runner"], shared),
        put_search(client, tokens["runner"], shared, search_body(client, tokens)),
    ):
        assert response.status_code == 409 and response.json()["error"]["code"] == "attempt_kind_mismatch"


# --- idempotency against policy changes --------------------------------------------


def test_prepared_attempt_stays_idempotent_after_a_policy_change(client, tokens, state_dir, monkeypatch):
    from backend.structure_search import incar_policy, planning

    attempt = new_id()
    body = search_body(client, tokens)
    first = put_search(client, tokens["preparer"], attempt, body)
    record_bytes = search_record_path(state_dir, attempt).read_bytes()
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    monkeypatch.setattr(planning, "build_search_plan_from_formula",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not re-plan")))
    again = put_search(client, tokens["preparer"], attempt, body)
    assert again.status_code == 200 and again.json() == first.json()
    assert search_record_path(state_dir, attempt).read_bytes() == record_bytes
    # A changed request or principal is still refused, without planning.
    assert put_search(client, tokens["preparer"], attempt, {**body, "formula": "Si2"}).status_code == 409
    assert put_search(client, tokens["other"], attempt, body).status_code == 403


def test_new_attempts_after_a_policy_change_are_refused(client, tokens, state_dir, monkeypatch):
    from backend.structure_search import incar_policy

    body = search_body(client, tokens)
    put_search(client, tokens["preparer"], new_id(), body)
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    fresh = new_id()
    assert put_search(client, tokens["preparer"], fresh, body).status_code == 409
    assert not search_record_path(state_dir, fresh).exists()
    assert not attempt_dir(state_dir, fresh).exists()


def test_interrupted_preparation_cannot_resume_against_stale_policy(client, tokens, state_dir, monkeypatch):
    from backend.structure_search import incar_policy

    attempt = new_id()
    body = search_body(client, tokens)

    def crash(plan, destination, **kwargs):
        raise RuntimeError("service stopped")

    real = search_plan.prepare_search_directory
    monkeypatch.setattr(search_plan, "prepare_search_directory", crash)
    assert put_search(client, tokens["preparer"], attempt, body).status_code == 500
    monkeypatch.setattr(search_plan, "prepare_search_directory", real)
    record_bytes = search_record_path(state_dir, attempt).read_bytes()
    assert search_record(state_dir, attempt)["state"] == "PREPARING"

    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    response = put_search(client, tokens["preparer"], attempt, body)
    assert response.status_code == 409 and response.json()["error"]["code"] == "plan_changed"
    assert search_record_path(state_dir, attempt).read_bytes() == record_bytes
    assert list(attempt_dir(state_dir, attempt).iterdir()) == []
