"""Managed structure-search planning (USPEX integration step 4A).

``POST /api/v1/structure-search-plans`` and the browser's New structure
preview plan through the same entry point
(``backend.structure_search.planning.plan_structure_search``). Planning is
read-only: nothing here prepares, submits, contacts POWER or writes the API
attempt ledger.
"""

from __future__ import annotations

import builtins
import json
import os
import pathlib
import subprocess
import uuid
import warnings

import pytest
from starlette.testclient import TestClient

import main
from backend.structure_search import planning, search_plan
from backend.structure_search.planning import StructureSearchPlanRequest, plan_structure_search
from backend.structure_search.search_plan import build_search_plan_from_formula, load_plan
from compute_api import auth as api_auth
from compute_api.ledger import STATE_DIR_ENV


LOOPBACK = ("127.0.0.1", 50000)
ROUTE = "/api/v1/structure-search-plans"


@pytest.fixture
def tokens(tmp_path, monkeypatch):
    issued = {}
    entries = []
    for principal, scopes in (
        ("planner", ["plan", "read"]),
        ("reader", ["read"]),
        ("operator", ["plan", "read", "prepare", "submit"]),
    ):
        token, token_id, verifier = api_auth.generate_token()
        issued[principal] = token
        entries.append({"principal": principal, "token_id": token_id, "verifier": verifier,
                        "scopes": scopes, "enabled": True})
    path = tmp_path / "api_tokens.json"
    path.write_text(json.dumps({"schema": "bmd_compute.api_tokens", "schema_version": 1, "principals": entries}))
    os.chmod(path, 0o600)
    monkeypatch.setenv(api_auth.TOKENS_FILE_ENV, str(path))
    monkeypatch.delenv("BMD_API_ALLOW_NON_LOOPBACK", raising=False)
    return issued


@pytest.fixture
def client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TestClient(main.app, client=LOOPBACK)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def post_search(client, token, body):
    return client.post(ROUTE, headers=bearer(token), json=body)


def _unreachable(*args, **kwargs):
    raise AssertionError("search planning must not start")


@pytest.fixture
def no_search_science(monkeypatch):
    monkeypatch.setattr(planning, "build_search_plan_from_formula", _unreachable)


def _browser_plan(client, formula, **advanced):
    form = {"formula": formula}
    names = {"min_atoms": "min_atoms", "max_atoms": "max_atoms",
             "num_generations": "num_generations", "walltime": "search_walltime"}
    form.update({names[key]: str(value) for key, value in advanced.items()})
    response = client.post("/predict", data=form)
    return response


# --- authentication and request shape ---------------------------------------------


def test_authentication_and_scope_precede_any_planning(client, tokens, no_search_science):
    assert client.post(ROUTE, json={"formula": "Si"}).status_code == 401
    response = post_search(client, tokens["reader"], {"formula": "Si"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"
    response = client.post(ROUTE + "?formula=Si", headers=bearer(tokens["planner"]), json={"formula": "Si"})
    assert response.status_code == 400
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        remote = TestClient(main.app, client=("10.0.0.5", 50000))
    assert remote.post(ROUTE, headers=bearer(tokens["planner"]), json={"formula": "Si"}).status_code == 403


@pytest.mark.parametrize(
    ("body", "field", "problem"),
    [
        ({}, "formula", "required"),
        ({"formula": ""}, "formula", "must_be_nonempty_string"),
        ({"formula": 7}, "formula", "must_be_nonempty_string"),
        ({"formula": "Si" * 40}, "formula", "too_long"),
        ({"formula": "Si\x00"}, "formula", "invalid_characters"),
        ({"formula": "Si", "colour": "red"}, "colour", "unknown_field"),
        ({"formula": "Si", "structure": {"format": "poscar", "text": "x"}}, "structure", "not_accepted_for_structure_search"),
        ({"formula": "Si", "resources": {"cpus": 4}}, "resources", "not_accepted_for_structure_search"),
        ({"formula": "Si", "external_pressure_gpa": 10}, "external_pressure_gpa", "fixed_by_bmd_policy"),
        ({"formula": "Si", "search": {"external_pressure_gpa": 10}}, "search.external_pressure_gpa", "fixed_by_bmd_policy"),
        ({"formula": "Si", "search": {"spin_polarized": True}}, "search.spin_polarized", "fixed_by_bmd_policy"),
        ({"formula": "Si", "search": {"kresol_start": [0.1] * 5}}, "search.kresol_start", "fixed_by_bmd_policy"),
        ({"formula": "Si", "search": {"theory": "hse06"}}, "search.theory", "fixed_by_bmd_policy"),
        ({"formula": "Si", "search": {"max_atoms": "12"}}, "search.max_atoms", "must_be_positive_integer"),
        ({"formula": "Si", "search": {"max_atoms": True}}, "search.max_atoms", "must_be_positive_integer"),
        ({"formula": "Si", "search": {"num_generations": 0}}, "search.num_generations", "must_be_positive_integer"),
        ({"formula": "Si", "search": {"walltime": "7 days"}}, "search.walltime", "must_be_hh_mm_ss"),
        ({"formula": "Si", "search": []}, "search", "must_be_object"),
        ({"formula": "Si", "expected_plan_digest": "abc"}, "expected_plan_digest", "must_be_sha256_digest"),
    ],
)
def test_malformed_requests_are_refused_before_planning(client, tokens, no_search_science, body, field, problem):
    response = post_search(client, tokens["planner"], body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert {"field": field, "problem": problem} in error["fields"]


def test_non_json_bodies_are_refused(client, tokens, no_search_science):
    response = client.post(ROUTE, headers={**bearer(tokens["planner"]), "content-type": "text/plain"}, content=b"Si")
    assert response.status_code == 415


# --- the plan --------------------------------------------------------------------


def test_valid_request_returns_the_frozen_search_plan(client, tokens):
    response = post_search(client, tokens["planner"], {"formula": "Mg2SiO4"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert set(payload) == {
        "api_version", "schema", "schema_version", "request_kind", "desired_output", "plan_schema",
        "plan_schema_version", "plan_digest", "plan_digest_scope", "formula", "search",
        "finalist_refinement", "resources", "preparation_available", "submission_available", "plan",
    }
    assert payload["schema"] == "bmd_compute.structure_search_plan_response"
    assert payload["request_kind"] == "structure_search"
    assert payload["desired_output"] == "new_structure"
    assert payload["plan_schema"] == search_plan.PLAN_SCHEMA
    expected = build_search_plan_from_formula("Mg2SiO4")
    assert payload["plan_digest"] == expected.digest
    frozen = load_plan(payload["plan"])
    assert frozen.digest == expected.digest and frozen.canonical == expected.canonical
    assert payload["search"]["energy_grade"] == "search-grade"
    assert payload["search"]["equivalent_to_bmd_adaptive_relaxation"] is False
    assert payload["finalist_refinement"]["status"] == "planned, not yet available"
    assert payload["resources"]["status"] == "provisional_estimate"
    assert payload["resources"]["approved"] is False
    assert payload["preparation_available"] is False and payload["submission_available"] is False
    assert payload["plan"]["potcar"]["contains_potcar_contents"] is False


def test_entered_formula_is_kept_and_equivalent_formulas_share_the_digest(client, tokens):
    first = post_search(client, tokens["planner"], {"formula": "Ca(OH)2"}).json()
    second = post_search(client, tokens["planner"], {"formula": "H2CaO2"}).json()
    assert first["plan_digest"] == second["plan_digest"]
    assert first["formula"]["entered"] == "Ca(OH)2" and first["formula"]["display"] == "Ca(OH)2"
    assert second["formula"]["display"] == "H2CaO2"
    assert first["formula"]["canonical"] == second["formula"]["canonical"] == "CaH2O2"
    assert first["plan"]["provenance"]["request"]["entered_formula"] == "Ca(OH)2"
    multiple = post_search(client, tokens["planner"], {"formula": "C2H8"}).json()
    assert multiple["formula"]["formula_unit_as_entered"] == "CH4"
    assert multiple["formula"]["reduced_from_entered"] is True


@pytest.mark.parametrize(
    ("formula", "advanced"),
    [("Si", {}), ("SiO2", {"max_atoms": 12, "num_generations": 30}), ("GaAs", {"walltime": "48:00:00"}),
     ("O2Si", {"min_atoms": 3, "max_atoms": 9})],
)
def test_browser_and_machine_planning_give_the_same_plan(client, tokens, formula, advanced):
    api = post_search(client, tokens["planner"], {"formula": formula, "search": advanced})
    assert api.status_code == 200, api.text
    browser = _browser_plan(client, formula, **advanced)
    assert browser.status_code == 200
    preview = browser.context["prediction"]["plan"]
    assert preview["plan_digest"] == api.json()["plan_digest"]
    direct = plan_structure_search(StructureSearchPlanRequest(formula=formula, **{
        key: str(value) for key, value in advanced.items()
    }))
    assert direct.preview() == preview
    api_record = {key: value for key, value in api.json()["plan"].items() if key not in ("plan_digest", "provenance")}
    assert api_record == direct.plan.record


@pytest.mark.parametrize(
    "formula",
    ["Xx2", "Si(O", "si", "Fe2O3", "CoO", "Bi2Se3", "LiNaKRbCs"],
)
def test_invalid_and_unsupported_formulas_are_refused_consistently(client, tokens, formula):
    api = post_search(client, tokens["planner"], {"formula": formula})
    assert api.status_code == 422, api.text
    error = api.json()["error"]
    assert error["code"] == "calculation_invalid"
    browser = _browser_plan(client, formula)
    assert browser.status_code == 400
    context = browser.context["prediction"]["error"]
    assert context["diagnostic"]["code"] == error["diagnostic_code"]
    assert " ".join(context["message"].split()) == error["message"]


def test_out_of_scope_overrides_are_refused_consistently(client, tokens):
    from backend.structure_search import admission

    api = post_search(client, tokens["planner"], {"formula": "Si", "search": {"max_atoms": admission.MAX_CELL_ATOMS + 1}})
    assert api.status_code == 422
    browser = _browser_plan(client, "Si", max_atoms=admission.MAX_CELL_ATOMS + 1)
    assert browser.status_code == 400
    assert api.json()["error"]["diagnostic_code"] == browser.context["prediction"]["error"]["diagnostic"]["code"]


def test_stale_expected_digest_is_refused(client, tokens):
    current = post_search(client, tokens["planner"], {"formula": "Si"}).json()["plan_digest"]
    ok = post_search(client, tokens["planner"], {"formula": "Si", "expected_plan_digest": current})
    assert ok.status_code == 200
    stale = post_search(client, tokens["planner"], {"formula": "Si", "expected_plan_digest": "sha256:" + "0" * 64})
    assert stale.status_code == 409
    error = stale.json()["error"]
    assert error["code"] == "plan_changed" and error["plan_digest"] == current


def test_policy_change_makes_an_earlier_digest_stale(client, tokens, monkeypatch):
    from backend.structure_search import incar_policy

    before = post_search(client, tokens["planner"], {"formula": "Si"}).json()["plan_digest"]
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    response = post_search(client, tokens["planner"], {"formula": "Si", "expected_plan_digest": before})
    assert response.status_code == 409


def test_unexpected_failures_return_a_generic_error(client, tokens, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret /bmd/path detail")

    monkeypatch.setattr(planning, "build_search_plan_from_formula", boom)
    response = post_search(client, tokens["planner"], {"formula": "Si"})
    assert response.status_code == 500
    assert response.json()["error"] == {"code": "plan_failed", "message": "BMD Compute could not plan this search."}


# --- distinct from structure-based planning and never executed -------------------


def test_structure_prediction_is_not_a_desired_output_of_plans_or_attempts(client, tokens, tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    monkeypatch.setenv(STATE_DIR_ENV, str(state))
    body = {"structure": {"format": "poscar", "text": "Si\n1\n"}, "workflow": {"desired_output": "new_structure"}}
    response = client.post("/api/v1/plans", headers=bearer(tokens["planner"]), json=body)
    assert response.status_code == 422
    assert {"field": "workflow.desired_output", "problem": "use_structure_search_plans"} in response.json()["error"]["fields"]

    attempt = str(uuid.uuid4())
    response = client.put(
        f"/api/v1/attempts/{attempt}",
        headers=bearer(tokens["operator"]),
        json={**body, "expected_plan_digest": "sha256:" + "0" * 64, "submit": True},
    )
    assert response.status_code == 422
    assert {"field": "workflow.desired_output", "problem": "use_structure_search_plans"} in response.json()["error"]["fields"]
    assert list(state.iterdir()) == []  # the attempt ledger was never opened
    # There is no search counterpart of attempts.
    assert client.put(f"/api/v1/structure-search-plans/{attempt}", headers=bearer(tokens["operator"]), json={}).status_code in {404, 405}
    assert client.post("/api/v1/structure-search-attempts", headers=bearer(tokens["operator"]), json={}).status_code == 404


def test_search_planning_performs_no_remote_preparation_or_file_writing(client, tokens, monkeypatch):
    import backend.paramiko_remote as paramiko_remote
    import backend.remote_runtime as remote_runtime

    for name in ("prepare_remote_submission", "submit_remote_workflow", "monitor_job", "monitor_submitted_job",
                 "load_results_for_completed_job"):
        monkeypatch.setattr(main, name, _unreachable)
    monkeypatch.setattr(remote_runtime, "connected_remote_runner", _unreachable)
    monkeypatch.setattr(paramiko_remote.ParamikoRemoteRunner, "connect", _unreachable)
    monkeypatch.setattr(search_plan, "prepare_search_directory", _unreachable)
    import compute_api.execution as execution

    monkeypatch.setattr(execution, "execute_attempt", _unreachable)
    monkeypatch.setattr(execution, "_ledger", _unreachable)

    real_run = subprocess.run
    real_open = builtins.open

    def guarded_run(args, *pargs, **kwargs):
        command = list(args) if not isinstance(args, str) else args.split()
        if not command or os.path.basename(str(command[0])) != "git":
            raise AssertionError(f"unexpected subprocess: {command[:1]}")
        assert {"commit", "push", "fetch", "checkout", "reset", "add"}.isdisjoint(map(str, command))
        return real_run(args, *pargs, **kwargs)

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write: {file}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    monkeypatch.setattr(subprocess, "Popen", _unreachable)
    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in ("replace", "rename", "remove", "unlink", "mkdir", "makedirs", "rmdir"):
        monkeypatch.setattr(os, name, _unreachable)
    for name in ("write_text", "write_bytes", "mkdir", "touch", "unlink"):
        monkeypatch.setattr(pathlib.Path, name, _unreachable)

    for body in ({"formula": "Si"}, {"formula": "Mg2SiO4", "search": {"num_generations": 20}}):
        response = post_search(client, tokens["planner"], body)
        assert response.status_code == 200, response.text
    assert client.post("/predict", data={"formula": "GaAs"}).status_code == 200


def test_identity_and_browser_openapi_surface_are_unchanged(client, tokens):
    payload = client.get("/api/v1/identity", headers=bearer(tokens["planner"])).json()
    assert "structure_search" not in json.dumps(sorted(payload))
    assert ROUTE not in main.app.openapi()["paths"]


# --- equivalent walltime spellings -----------------------------------------------


def test_api_and_browser_agree_on_equivalent_walltimes(client, tokens):
    digests = set()
    for spelling in ("48:00:00", "048:00:00"):
        api = post_search(client, tokens["planner"], {"formula": "Si", "search": {"walltime": spelling}})
        assert api.status_code == 200
        assert api.json()["search"]["walltime"] == "48:00:00"
        assert api.json()["plan"]["provenance"]["request"]["advanced_overrides_as_entered"]["walltime"] == spelling
        digests.add(api.json()["plan_digest"])
        digests.add(_browser_plan(client, "Si", walltime=spelling).context["prediction"]["plan"]["plan_digest"])
    assert len(digests) == 1


@pytest.mark.parametrize(("walltime", "status"), [("2-00:00:00", 422), ("48:60:00", 422), ("0:00:00", 422), ("2400:00:01", 422)])
def test_api_refuses_malformed_or_out_of_range_walltimes(client, tokens, walltime, status):
    response = post_search(client, tokens["planner"], {"formula": "Si", "search": {"walltime": walltime}})
    assert response.status_code == status


# --- the optional random seed and per-search node exclusion (S1, S2) ----------------------------


@pytest.mark.parametrize(
    ("search", "field", "problem"),
    [
        ({"random_seed": 0}, "search.random_seed", "must_be_nonzero_64_bit_integer"),
        ({"random_seed": 2**63}, "search.random_seed", "must_be_nonzero_64_bit_integer"),
        ({"random_seed": -(2**63) - 1}, "search.random_seed", "must_be_nonzero_64_bit_integer"),
        ({"random_seed": "20261126"}, "search.random_seed", "must_be_nonzero_64_bit_integer"),
        ({"random_seed": True}, "search.random_seed", "must_be_nonzero_64_bit_integer"),
        ({"random_seed": 1.5}, "search.random_seed", "must_be_nonzero_64_bit_integer"),
        ({"excluded_nodes": ""}, "search.excluded_nodes", "must_be_slurm_hostlist_string"),
        ({"excluded_nodes": ["compute-0-270"]}, "search.excluded_nodes", "must_be_slurm_hostlist_string"),
        ({"excluded_nodes": "compute-0-270 compute-0-271"}, "search.excluded_nodes", "must_be_slurm_hostlist_string"),
        ({"excluded_nodes": "n" * 300}, "search.excluded_nodes", "must_be_slurm_hostlist_string"),
        ({"fixRndSeed": 5}, "search.fixRndSeed", "unknown_field"),
        ({"exclude": "compute-0-270"}, "search.exclude", "unknown_field"),
    ],
)
def test_malformed_seed_or_exclusion_is_refused_before_planning(client, tokens, no_search_science, search, field,
                                                                problem):
    response = post_search(client, tokens["planner"], {"formula": "Si", "search": search})
    assert response.status_code == 422
    assert {"field": field, "problem": problem} in response.json()["error"]["fields"]


def test_an_unusable_host_list_is_refused_by_planning(client, tokens):
    response = post_search(client, tokens["planner"], {"formula": "Si", "search": {"excluded_nodes": "n[5-3]"}})
    assert response.status_code == 422, response.text


def test_seed_and_exclusion_are_planned_and_reported(client, tokens):
    body = {"formula": "Si", "search": {"min_atoms": 8, "max_atoms": 8, "num_generations": 30,
                                        "random_seed": 20261126, "excluded_nodes": "compute-0-270"}}
    response = post_search(client, tokens["planner"], body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["search"]["random_seed"]["requested"] == 20261126
    assert data["search"]["random_seed"]["effective_rng_determinism"].startswith("not verified")
    assert data["resources"]["request"]["exclude"] == "compute-0-[332-333],compute-0-270"
    plan = load_plan(data["plan"])
    assert plan.digest == data["plan_digest"]
    assert plan.record["search"]["spec"]["random_seed"] == 20261126
    # Planned again with the digest, the same request is current; without the new fields it is another plan.
    again = post_search(client, tokens["planner"], {**body, "expected_plan_digest": data["plan_digest"]})
    assert again.status_code == 200
    plain = post_search(client, tokens["planner"], {"formula": "Si", "search": {"min_atoms": 8, "max_atoms": 8,
                                                                                "num_generations": 30}}).json()
    assert plain["plan_digest"] != data["plan_digest"]
    assert "random_seed" not in plain["search"]
    assert plain["resources"]["request"]["exclude"] == "compute-0-[332-333]"


def test_null_seed_and_exclusion_mean_omitted(client, tokens):
    plain = post_search(client, tokens["planner"], {"formula": "Si"}).json()
    nulls = post_search(client, tokens["planner"],
                        {"formula": "Si", "search": {"random_seed": None, "excluded_nodes": None}}).json()
    assert nulls["plan_digest"] == plain["plan_digest"]


@pytest.mark.parametrize("seed", [-2000, -1, 2**31, 2**63 - 1, -(2**63)])
def test_negative_and_large_seeds_are_planned(client, tokens, seed):
    response = post_search(client, tokens["planner"], {"formula": "Si", "search": {"random_seed": seed}})
    assert response.status_code == 200, response.text
    assert response.json()["search"]["random_seed"]["requested"] == seed
    assert response.json()["search"]["random_seed"]["input_txt_line"] == f"{seed} : fixRndSeed"
