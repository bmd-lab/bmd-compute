"""Authenticated machine API (Phase 1A): planning only.

These tests use mocked remote boundaries throughout. Nothing here contacts
POWER, opens SSH, calls SLURM or writes calculation files.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import uuid
import warnings

import pytest
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Incar
from starlette.requests import Request
from starlette.testclient import TestClient

import main
import test_automatic_soc as soc_cases
import test_input_reference_contract as reference_cases
from backend.calculations.admission import HSE06_SOC_NOT_SUPPORTED_CODE
from backend.calculations.default_treatments import AUTOMATIC_APPLICATION_OMITTED_UNSUPPORTED
from backend.calculations.method_considerations import SOC_HEAVY_ELEMENTS_CONSIDERATION_ID
from backend.parser import parse_structure
from compute_api import auth as api_auth
from compute_api import plan_digest as plan_digest_module
from compute_api.canonical import json_safe
from compute_api.plan_digest import compute_plan_digest
from compute_api.projection import plan_stage_previews
from compute_api.tokens import main as tokens_main


SI_POSCAR = reference_cases.SI_POSCAR
NIO_POSCAR = reference_cases.NIO_POSCAR
BI2SE3_POSCAR = soc_cases.BI2SE3_POSCAR
LOOPBACK = ("127.0.0.1", 50000)
EXPECTED_BROWSER_OPENAPI_PATHS = [
    "/",
    "/analyze",
    "/build-calculation",
    "/build-workflow",
    "/monitor",
    "/prepare-remote",
    "/resume",
    "/submit",
]


# --------------------------------------------------------------------------- fixtures


def _token_document(entries):
    return {"schema": "bmd_compute.api_tokens", "schema_version": 1, "principals": entries}


def _write_tokens(path: pathlib.Path, entries) -> pathlib.Path:
    path.write_text(json.dumps(_token_document(entries)), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


@pytest.fixture
def tokens(tmp_path, monkeypatch):
    plan_token, plan_id, plan_verifier = api_auth.generate_token()
    read_token, read_id, read_verifier = api_auth.generate_token()
    disabled_token, disabled_id, disabled_verifier = api_auth.generate_token()
    path = _write_tokens(
        tmp_path / "api_tokens.json",
        [
            {"principal": "planner", "token_id": plan_id, "verifier": plan_verifier, "scopes": ["plan", "read"], "enabled": True},
            {"principal": "reader", "token_id": read_id, "verifier": read_verifier, "scopes": ["read"], "enabled": True},
            {"principal": "retired", "token_id": disabled_id, "verifier": disabled_verifier, "scopes": ["plan"], "enabled": False},
        ],
    )
    monkeypatch.setenv(api_auth.TOKENS_FILE_ENV, str(path))
    monkeypatch.delenv("BMD_API_ALLOW_NON_LOOPBACK", raising=False)
    return {"plan": plan_token, "read": read_token, "disabled": disabled_token, "path": path}


@pytest.fixture
def client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TestClient(main.app, client=LOOPBACK)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def plan_body(structure=SI_POSCAR, *, fmt="poscar", desired_output="energy_only", custom=None, resources=None):
    workflow = {"custom": custom} if custom is not None else {"desired_output": desired_output}
    body = {"structure": {"format": fmt, "text": structure}, "workflow": workflow}
    if resources is not None:
        body["resources"] = resources
    return body


def post_plan(client, token, body):
    return client.post("/api/v1/plans", headers=bearer(token), json=body)


def custom_stage(stage_type, theory, modifiers=()):
    return {"stage_type": stage_type, "theory": theory, "modifiers": list(modifiers)}


def _unreachable(*args, **kwargs):
    raise AssertionError("scientific processing must not start")


@pytest.fixture
def no_science(monkeypatch):
    for name in ("parse_structure", "workflow_spec_from_form", "resolve_workflow_for_structure", "build_submission_state_from_structure"):
        monkeypatch.setattr(main, name, _unreachable)


# --------------------------------------------------------------- 1. authentication


def test_missing_token_is_rejected_before_scientific_processing(client, tokens, no_science):
    response = client.post("/api/v1/plans", json=plan_body())
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"
    assert response.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer",
        "Bearer ",
        "Basic dXNlcjpwYXNz",
        "bearer bmdc1.0000000000000000.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Bearer not-a-token",
        "Bearer " + "x" * 400,
    ],
)
def test_malformed_or_unknown_tokens_are_rejected(client, tokens, no_science, authorization):
    response = client.post("/api/v1/plans", headers={"Authorization": authorization}, json=plan_body())
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


def test_tampered_secret_and_disabled_token_are_rejected(client, tokens, no_science):
    token = tokens["plan"]
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert post_plan(client, tampered, plan_body()).status_code == 401
    assert post_plan(client, tokens["disabled"], plan_body()).status_code == 401


def test_token_without_plan_scope_is_rejected_before_scientific_processing(client, tokens, no_science):
    response = post_plan(client, tokens["read"], plan_body())
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"


def test_authentication_precedes_body_parsing(client, tokens, no_science):
    response = client.post(
        "/api/v1/plans",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401


def test_duplicate_authorization_headers_are_rejected(client, tokens, no_science):
    response = client.post(
        "/api/v1/plans",
        headers=[("Authorization", f"Bearer {tokens['plan']}"), ("Authorization", f"Bearer {tokens['plan']}")],
        json=plan_body(),
    )
    assert response.status_code == 401


def test_query_strings_are_refused_so_tokens_never_travel_in_urls(client, tokens, no_science):
    response = client.post(f"/api/v1/plans?token={tokens['plan']}", headers=bearer(tokens["plan"]), json=plan_body())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "query_not_allowed"


def test_non_loopback_clients_are_refused_even_with_a_valid_token(tokens, no_science):
    remote = TestClient(main.app, client=("132.66.1.2", 40000))
    response = remote.post("/api/v1/plans", headers=bearer(tokens["plan"]), json=plan_body())
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "loopback_required"
    assert remote.get("/api/v1/identity", headers=bearer(tokens["plan"])).status_code == 403


def test_unconfigured_api_fails_closed(client, monkeypatch, no_science):
    monkeypatch.delenv(api_auth.TOKENS_FILE_ENV, raising=False)
    response = client.get("/api/v1/identity", headers=bearer("bmdc1.0000000000000000." + "A" * 43))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "api_not_configured"


def test_unsafe_or_malformed_token_files_fail_closed(client, tokens, tmp_path, monkeypatch, no_science):
    _, token_id, verifier = api_auth.generate_token()
    valid_entry = {"principal": "p", "token_id": token_id, "verifier": verifier, "scopes": ["plan"], "enabled": True}

    readable = _write_tokens(tmp_path / "readable.json", [valid_entry])
    os.chmod(readable, 0o640)
    unknown_scope = _write_tokens(tmp_path / "unknown_scope.json", [{**valid_entry, "scopes": ["plan", "admin"]}])
    extra_key = _write_tokens(tmp_path / "extra_key.json", [{**valid_entry, "comment": "x"}])
    duplicate = _write_tokens(tmp_path / "duplicate.json", [valid_entry, valid_entry])
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    os.chmod(malformed, 0o600)
    inside_repository = pathlib.Path(main.__file__).resolve().parent / "README.md"

    for path in (readable, unknown_scope, extra_key, duplicate, malformed, inside_repository, tmp_path / "missing.json"):
        monkeypatch.setenv(api_auth.TOKENS_FILE_ENV, str(path))
        response = client.post("/api/v1/plans", headers=bearer(tokens["plan"]), json=plan_body())
        assert response.status_code == 503, path.name
    monkeypatch.setenv(api_auth.TOKENS_FILE_ENV, "relative/tokens.json")
    assert client.get("/api/v1/identity", headers=bearer(tokens["plan"])).status_code == 503


def test_token_file_edits_take_effect_on_the_next_request(client, tokens):
    assert client.get("/api/v1/identity", headers=bearer(tokens["plan"])).status_code == 200
    document = json.loads(tokens["path"].read_text(encoding="utf-8"))
    for entry in document["principals"]:
        entry["enabled"] = False
    tokens["path"].write_text(json.dumps(document), encoding="utf-8")
    assert client.get("/api/v1/identity", headers=bearer(tokens["plan"])).status_code == 401


def test_identity_reports_principal_scopes_and_versions_only(client, tokens):
    response = client.get("/api/v1/identity", headers=bearer(tokens["read"]))
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "api_version",
        "service",
        "compute_source",
        "principal",
        "scopes",
        "capability_schema_version",
        "plan_request_schema_version",
        "plan_response_schema_version",
        "plan_digest_version",
    }
    assert payload["principal"] == "reader"
    assert payload["scopes"] == ["read"]
    assert payload["api_version"] == "v1"
    assert set(payload["compute_source"]) == {"git_commit", "dirty"}
    assert response.headers["cache-control"] == "no-store"


def test_identity_requires_authentication(client, tokens):
    assert client.get("/api/v1/identity").status_code == 401


def test_token_generator_output_authenticates(tmp_path, monkeypatch, capsys):
    assert tokens_main(["--principal", "bench-lee", "--scope", "plan"]) == 0
    output = capsys.readouterr().out
    token = next(line for line in output.splitlines() if line.startswith("bmdc1."))
    entry = json.loads(output[output.index("{"):])
    path = _write_tokens(tmp_path / "generated.json", [entry])
    principal = api_auth.authenticate([f"Bearer {token}"], environ={api_auth.TOKENS_FILE_ENV: str(path)})
    assert principal.principal == "bench-lee"
    assert principal.scopes == frozenset({"plan"})
    assert token not in json.dumps(entry)


# ---------------------------------------------------------- 2. request validation


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda body: body.update(cluster={"remote_host": "evil"}), "cluster"),
        (lambda body: body.update(paths={"run_dir": "/tmp/x"}), "paths"),
        (lambda body: body.update(ssh_config_host="evil"), "ssh_config_host"),
        (lambda body: body.update(script="#!/bin/sh"), "script"),
        (lambda body: body.update(submission_identity_token="x"), "submission_identity_token"),
        (lambda body: body["structure"].update(path="/etc/passwd"), "structure.path"),
        (lambda body: body.update(resources={"account": "other"}), "resources.account"),
        (lambda body: body.update(resources={"nodes": 4}), "resources.nodes"),
        (lambda body: body.update(resources={"partition": "other"}), "resources.partition"),
        (lambda body: body["workflow"].update(vasp_cmd="rm -rf /"), "workflow.vasp_cmd"),
    ],
)
def test_unknown_and_authority_fields_are_rejected(client, tokens, no_science, mutate, field):
    body = plan_body()
    mutate(body)
    response = post_plan(client, tokens["plan"], body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert {"field": field, "problem": "unknown_field"} in error["fields"]
    assert "evil" not in response.text and "/etc/passwd" not in response.text


@pytest.mark.parametrize(
    "custom",
    [
        {"stages": [custom_stage("static", "pbe")], "recipe": "energy_only"},
        {"stages": [{**custom_stage("static", "pbe"), "label": "x"}]},
        {"stages": [{**custom_stage("static", "pbe"), "incar": {"ENCUT": 2000}}]},
        {"stages": []},
        {"stages": [custom_stage("static", "pbe")] * 17},
        {"stages": [custom_stage("../static", "pbe")]},
    ],
)
def test_custom_workflow_shape_is_closed(client, tokens, no_science, custom):
    response = post_plan(client, tokens["plan"], plan_body(custom=custom))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_workflow_must_name_exactly_one_request_kind(client, tokens, no_science):
    body = plan_body()
    body["workflow"]["custom"] = {"stages": [custom_stage("static", "pbe")]}
    response = post_plan(client, tokens["plan"], body)
    assert response.status_code == 422
    assert response.json()["error"]["fields"] == [
        {"field": "workflow", "problem": "exactly_one_of_desired_output_or_custom"}
    ]


def test_free_form_incar_options_are_rejected_by_compute_methodology(client, tokens):
    custom = {"stages": [{**custom_stage("static", "pbe"), "options": {"incar": {"ENCUT": 2000}}}]}
    response = post_plan(client, tokens["plan"], plan_body(custom=custom))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "calculation_invalid"
    assert "2000" not in response.text


@pytest.mark.parametrize(
    "body_bytes, content_type, status, code",
    [
        (b"{not json", "application/json", 400, "invalid_json"),
        (b"\xff\xfe", "application/json", 400, "invalid_json"),
        (b"{}", "text/plain", 415, "unsupported_media_type"),
        (b"[]", "application/json", 422, "invalid_request"),
    ],
)
def test_malformed_bodies_fail_safely(client, tokens, no_science, body_bytes, content_type, status, code):
    response = client.post(
        "/api/v1/plans",
        content=body_bytes,
        headers={**bearer(tokens["plan"]), "Content-Type": content_type},
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_oversized_requests_are_refused(client, tokens, no_science):
    response = post_plan(client, tokens["plan"], plan_body(structure="x" * (3 * 1024 * 1024 + 10)))
    assert response.status_code == 413


def test_malformed_structures_fail_with_classified_errors(client, tokens):
    poscar = post_plan(client, tokens["plan"], plan_body(structure="not a poscar\n1.0\n"))
    assert poscar.status_code == 422
    assert poscar.json()["error"]["code"] == "structure_invalid"
    # backend.parser lets pymatgen's CIF errors propagate (the browser route has
    # that gap); the API classifies them without the parser's text.
    cif = post_plan(client, tokens["plan"], plan_body(structure="data_x\n_cell_length_a 1\n", fmt="cif"))
    assert cif.status_code == 422
    assert cif.json()["error"]["code"] == "structure_invalid"
    assert "no structures" not in cif.text
    bad_format = post_plan(client, tokens["plan"], plan_body(fmt="xyz"))
    assert bad_format.status_code == 422
    assert bad_format.json()["error"]["fields"] == [{"field": "structure.format", "problem": "must_be_poscar_or_cif"}]


def test_malformed_workflows_and_resources_fail_with_classified_errors(client, tokens):
    unknown_output = post_plan(client, tokens["plan"], plan_body(desired_output="not_an_output"))
    assert unknown_output.status_code == 422
    assert unknown_output.json()["error"]["diagnostic_code"] == "unknown_desired_output"

    for stage in (custom_stage("phonons", "pbe"), custom_stage("static", "b3lyp"), custom_stage("static", "pbe", ["nonsense"])):
        unknown = post_plan(client, tokens["plan"], plan_body(custom={"stages": [stage]}))
        assert unknown.status_code == 422
        assert unknown.json()["error"]["code"] == "calculation_invalid"
        assert unknown.json()["error"]["diagnostic_code"] == "unsupported_workflow_vocabulary"

    bad_cpus = post_plan(client, tokens["plan"], plan_body(resources={"cpus": 25}))
    assert bad_cpus.status_code == 422
    assert bad_cpus.json()["error"]["code"] == "calculation_invalid"

    bad_walltime = post_plan(client, tokens["plan"], plan_body(resources={"walltime": "72h"}))
    assert bad_walltime.status_code == 422
    assert bad_walltime.json()["error"]["code"] == "invalid_request"


def test_unexpected_failures_return_a_generic_error(client, tokens, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("internal detail /bmd-db/guest/secret")

    monkeypatch.setattr(main, "build_submission_state_from_structure", explode)
    response = post_plan(client, tokens["plan"], plan_body())
    assert response.status_code == 500
    assert response.json()["error"] == {"code": "plan_failed", "message": "BMD Compute could not generate this plan."}
    assert "bmd-db" not in response.text


# ------------------------------------------------------------- 3. valid requests


def test_valid_poscar_request_returns_the_bounded_plan_contract(client, tokens):
    response = post_plan(client, tokens["plan"], plan_body(resources={"cpus": 48, "memory_gb": 64, "walltime": "02:00:00"}))
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "schema",
        "schema_version",
        "api_version",
        "plan_digest",
        "plan_digest_version",
        "request",
        "structure",
        "workflow",
        "automatic_treatments",
        "method_considerations",
        "resources",
        "scientific_inputs",
        "software",
        "policies",
        "capability_schema_version",
    }
    assert payload["schema"] == "bmd_compute.api.plan"
    assert payload["plan_digest"].startswith("sha256:")
    assert payload["resources"] == {
        "nodes": 1,
        "cpus": 48,
        "memory_gb": 64,
        "walltime": "02:00:00",
        "partition": "leeburton-pool",
        "account": "power-leeburton-users_v2",
    }
    stage = payload["scientific_inputs"]["stages"][0]
    assert stage["stage_type"] == "static" and stage["theory"] == "pbe"
    assert stage["vasp_executable"] == "vasp_std"
    assert stage["potcar_symbols"] == ["Si"]
    assert stage["kpoints"]["num_kpts"] == 0
    assert payload["policies"]["new_calculation_admission"]["version"] >= 1


def test_valid_cif_request_matches_the_equivalent_poscar_plan(client, tokens):
    structure = parse_structure(SI_POSCAR, "poscar")
    cif_text = str(CifWriter(structure))
    cif = post_plan(client, tokens["plan"], plan_body(cif_text, fmt="CIF"))
    poscar = post_plan(client, tokens["plan"], plan_body())
    assert cif.status_code == 200, cif.text
    assert cif.json()["request"]["structure_format"] == "cif"
    assert cif.json()["workflow"] == poscar.json()["workflow"]
    assert cif.json()["structure"]["natoms"] == 2


FORBIDDEN_RESPONSE_TEXT = (
    "identity_token",
    "attempt_id",
    "attempt_fingerprint",
    "/bmd-db",
    "/bmd/",
    "bmdguest",
    "powerslurm",
    "ssh",
    "#SBATCH",
    "run_job",
    "PMG_VASP_PSP_DIR",
    "JOBFLOW_CONFIG_FILE",
    "remote_python",
    "mpirun",
)


@pytest.mark.parametrize(
    "body",
    [
        plan_body(),
        plan_body(BI2SE3_POSCAR, desired_output="electronic_dos"),
        plan_body(NIO_POSCAR),
    ],
)
def test_responses_never_contain_submission_ssh_path_or_script_material(client, tokens, body):
    response = post_plan(client, tokens["plan"], body)
    assert response.status_code == 200
    text = response.text
    for needle in FORBIDDEN_RESPONSE_TEXT:
        assert needle not in text, needle


# -------------------------------------------------------- 4. UI / API parity


def _browser_build(structure, *, workflow=None, workflow_spec=None, cpus=None):
    request = Request({"type": "http", "method": "POST", "path": "/build-workflow", "headers": []})
    response = main.build_workflow(
        request,
        structure=structure,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=cpus,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=json.dumps(workflow_spec) if workflow_spec is not None else None,
        workflow=workflow,
        method=None,
    )
    assert response.status_code == 200
    return response.context


def _browser_stage_incars(generated_inputs):
    text = generated_inputs["incar"]
    if "# Stage " not in text:
        return [Incar.from_str(text)]
    sections = text.split("# Stage ")[1:]
    incars = []
    for section in sections:
        lines = [line for line in section.splitlines()[1:] if not line.startswith("#")]
        incars.append(Incar.from_str("\n".join(lines)))
    return incars


def _browser_executables(generated_inputs):
    if "vasp_executables" in generated_inputs:
        return [item["executable"] for item in generated_inputs["vasp_executables"]]
    return [generated_inputs["vasp_executable"]]


PARITY_CASES = {
    "pbe_static": (SI_POSCAR, "energy_only", None),
    "pbe_soc_static": (BI2SE3_POSCAR, "energy_only", None),
    "hse06_dos_without_soc": (SI_POSCAR, "electronic_dos", None),
    "hse06_dos_soc_omitted": (BI2SE3_POSCAR, "electronic_dos", None),
    "hse06_band_structure_soc_omitted": (BI2SE3_POSCAR, "electronic_band_structure", None),
    "pbe_dft_u": (NIO_POSCAR, "energy_only", None),
    "custom_pbe_soc": (SI_POSCAR, None, {"stages": [custom_stage("static", "pbe", ["soc"])]}),
    "custom_hse06": (SI_POSCAR, None, {"stages": [custom_stage("relax", "pbe"), custom_stage("static", "hse06")]}),
}


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_ui_and_api_share_the_scientific_resolution(client, tokens, case):
    structure, desired_output, custom = PARITY_CASES[case]
    ui = _browser_build(
        structure,
        workflow=desired_output if desired_output else "custom",
        workflow_spec=custom,
    )
    api_response = post_plan(client, tokens["plan"], plan_body(structure, desired_output=desired_output, custom=custom))
    assert api_response.status_code == 200, api_response.text
    api = api_response.json()
    ui_spec = ui["submission_spec"]
    ui_flow = ui_spec["flow_spec"]

    # Resolved workflow stages, theories and modifiers (SOC applied or omitted).
    assert [
        {key: stage[key] for key in ("stage_type", "theory", "modifiers", "options")}
        for stage in api["workflow"]["stages"]
    ] == [
        {key: stage[key] for key in ("stage_type", "theory", "modifiers", "options")}
        for stage in json_safe(ui_flow["workflow_spec"]["stages"])
    ]
    # Automatic treatments, including omissions and the prepared DFT+U record.
    assert api["automatic_treatments"] == json_safe(ui_flow.get("automatic_treatments"))
    # VASP executable selection.
    assert [stage["vasp_executable"] for stage in api["workflow"]["stages"]] == _browser_executables(ui["generated_inputs"])
    # Generated scientific inputs.
    api_incars = [Incar(stage["incar"]["settings"]) for stage in api["scientific_inputs"]["stages"]]
    assert api_incars == _browser_stage_incars(ui["generated_inputs"])
    # Execution resources.
    assert api["resources"] == {
        "nodes": ui_spec["resources"]["nodes"],
        "cpus": ui_spec["resources"]["ntasks"],
        "memory_gb": ui_spec["resources"]["mem_gb"],
        "walltime": ui_spec["resources"]["walltime"],
        "partition": ui_spec["cluster"]["partition"],
        "account": ui_spec["cluster"]["account"],
    }
    # Method-consideration classification.
    ui_considerations = (ui["method_considerations"] or {}).get("considerations") or []
    api_considerations = (api["method_considerations"] or {}).get("considerations") or []
    assert [(item["id"], item["automatic_application_state"]) for item in api_considerations] == [
        (item["id"], item["automatic_application_state"]) for item in ui_considerations
    ]
    # The digest of the browser's own submission specification equals the API digest.
    structure_obj = parse_structure(structure, "poscar")
    ui_plan = main.CalculationPlan(
        structure=structure_obj,
        workflow_spec=main.WorkflowSpec.from_dict(ui_flow["workflow_spec"]),
        default_treatment_resolution=ui["default_treatment_resolution"],
        method_considerations=ui["method_considerations"],
        execution_resources=ui["selected_resources"],
        summary=ui["summary"],
        generated_inputs=ui["generated_inputs"],
        submission_spec=ui_spec,
    )
    assert compute_plan_digest(
        structure=structure_obj,
        submission_spec=ui_spec,
        stage_previews=plan_stage_previews(ui_plan),
    ) == api["plan_digest"]


def test_admission_regression_guard_applies_to_the_api_too(client, tokens, monkeypatch):
    # The API reaches admission through the same module-level function as the
    # browser, so the browser's regression guard also governs the API.
    monkeypatch.setattr(main, "require_admissible_new_calculation", main.validate_workflow_spec)
    custom = {"stages": [custom_stage("static", "hse06", ["soc"])]}
    response = post_plan(client, tokens["plan"], plan_body(custom=custom))
    assert response.status_code == 200
    assert response.json()["workflow"]["stages"][0]["vasp_executable"] == "vasp_ncl"


# ------------------------------------------- 5. HSE06+SOC admission and treatments


@pytest.mark.parametrize(
    "custom",
    [
        {"stages": [custom_stage("static", "hse06", ["soc"])]},
        {"stages": [custom_stage("relax", "pbe"), custom_stage("static", "hse06", ["soc"])]},
        {"stages": [custom_stage("relax", "pbe"), custom_stage("static", "hse06", ["soc"]), custom_stage("dos", "hse06", ["soc"])]},
    ],
)
def test_hse06_soc_is_not_admitted(client, tokens, custom):
    response = post_plan(client, tokens["plan"], plan_body(custom=custom))
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "calculation_invalid"
    assert error["diagnostic_code"] == HSE06_SOC_NOT_SUPPORTED_CODE


def test_pbe_soc_is_admitted_and_runs_vasp_ncl(client, tokens):
    response = post_plan(client, tokens["plan"], plan_body(BI2SE3_POSCAR))
    assert response.status_code == 200
    stage = response.json()["workflow"]["stages"][0]
    assert stage["theory"] == "pbe" and "soc" in stage["modifiers"]
    assert stage["vasp_executable"] == "vasp_ncl"


def test_soc_omission_on_hse06_stages_is_preserved(client, tokens):
    payload = post_plan(client, tokens["plan"], plan_body(BI2SE3_POSCAR, desired_output="electronic_dos")).json()
    hse06_stages = [stage for stage in payload["workflow"]["stages"] if stage["theory"] == "hse06"]
    assert hse06_stages and all("soc" not in stage["modifiers"] for stage in hse06_stages)
    assert all(stage["vasp_executable"] == "vasp_std" for stage in hse06_stages)
    omitted = payload["automatic_treatments"]["omitted_treatments"]
    assert omitted, "the HSE06 SOC omission must be recorded"
    states = {item["id"]: item["automatic_application_state"] for item in payload["method_considerations"]["considerations"]}
    assert states[SOC_HEAVY_ELEMENTS_CONSIDERATION_ID] == AUTOMATIC_APPLICATION_OMITTED_UNSUPPORTED
    assert {item["consideration_id"] for item in omitted} == {SOC_HEAVY_ELEMENTS_CONSIDERATION_ID}


def test_heavy_element_band_structure_keeps_hse06_and_records_the_soc_omission(client, tokens):
    ui = _browser_build(BI2SE3_POSCAR, workflow="electronic_band_structure")
    response = post_plan(client, tokens["plan"], plan_body(BI2SE3_POSCAR, desired_output="electronic_band_structure"))
    assert response.status_code == 200, response.text
    payload = response.json()

    stages = payload["workflow"]["stages"]
    hse06_stages = [stage for stage in stages if stage["theory"] == "hse06"]
    assert [stage["stage_type"] for stage in hse06_stages] == ["static", "band_structure"]
    assert all("soc" not in stage["modifiers"] for stage in hse06_stages)
    assert all(stage["vasp_executable"] == "vasp_std" for stage in hse06_stages)
    # Relaxation stays non-SOC as well; SOC is applied to no stage in this workflow.
    assert all("soc" not in stage["modifiers"] for stage in stages)

    omitted = payload["automatic_treatments"]["omitted_treatments"]
    assert {item["consideration_id"] for item in omitted} == {SOC_HEAVY_ELEMENTS_CONSIDERATION_ID}
    assert payload["automatic_treatments"] == json_safe(ui["submission_spec"]["flow_spec"]["automatic_treatments"])

    # The browser's red warning ("unsupported" presentation) is the machine
    # state omitted_unsupported_combination for the same consideration.
    ui_soc = next(
        item for item in ui["method_considerations"]["considerations"]
        if item["id"] == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID
    )
    api_soc = next(
        item for item in payload["method_considerations"]["considerations"]
        if item["id"] == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID
    )
    assert ui_soc["presentation"] == "unsupported"
    assert ui_soc["automatic_application_state"] == AUTOMATIC_APPLICATION_OMITTED_UNSUPPORTED
    assert api_soc["automatic_application_state"] == AUTOMATIC_APPLICATION_OMITTED_UNSUPPORTED


def test_automatic_dft_u_record_is_preserved(client, tokens):
    payload = post_plan(client, tokens["plan"], plan_body(NIO_POSCAR)).json()
    record = payload["automatic_treatments"]["dft_u"]
    assert record is not None
    stage = payload["scientific_inputs"]["stages"][0]
    assert stage["incar"]["settings"].get("LDAU") is True
    assert payload["workflow"]["stages"][0]["options"], "frozen DFT+U parameters travel with the stage"


# ----------------------------------------------------------------- 6. plan digest


def _plan_and_digest(structure=SI_POSCAR, *, timestamp, attempt_id, desired_output="energy_only"):
    structure_obj = parse_structure(structure, "poscar")
    workflow_spec = main.workflow_spec_from_form(workflow=desired_output)
    workflow_spec, resolution = main.resolve_workflow_for_structure(structure_obj, workflow_spec, workflow=desired_output)
    resources = main.default_execution_resources()
    summary, _, generated_inputs, spec = main.build_submission_state_from_structure(
        structure_obj=structure_obj,
        structure_text=structure,
        fmt="poscar",
        workflow_spec=workflow_spec,
        execution_resources=resources,
        timestamp=timestamp,
        submission_attempt_id=attempt_id,
        default_treatment_resolution=resolution,
    )
    plan = main.CalculationPlan(structure_obj, workflow_spec, resolution, None, resources, summary, generated_inputs, spec)
    return spec, compute_plan_digest(structure=structure_obj, submission_spec=spec, stage_previews=plan_stage_previews(plan))


def test_plan_digest_ignores_timestamps_attempts_run_names_and_paths():
    first_spec, first = _plan_and_digest(timestamp="20260101-000000", attempt_id=str(uuid.uuid4()))
    second_spec, second = _plan_and_digest(timestamp="20261231-235959", attempt_id=str(uuid.uuid4()))
    assert first_spec["run_name"] != second_spec["run_name"]
    assert first_spec["paths"]["run_dir"] != second_spec["paths"]["run_dir"]
    assert first_spec["submission"]["attempt_fingerprint"] != second_spec["submission"]["attempt_fingerprint"]
    assert first == second


def test_plan_digest_is_deterministic_across_requests_and_poscar_comments(client, tokens):
    digests = {post_plan(client, tokens["plan"], plan_body()).json()["plan_digest"] for _ in range(2)}
    renamed = SI_POSCAR.replace(SI_POSCAR.splitlines()[0], "a different comment line", 1)
    digests.add(post_plan(client, tokens["plan"], plan_body(renamed)).json()["plan_digest"])
    assert len(digests) == 1


@pytest.mark.parametrize(
    "variant",
    [
        {"resources": {"cpus": 48}},
        {"resources": {"memory_gb": 64}},
        {"resources": {"walltime": "24:00:00"}},
        {"desired_output": "relaxed_structure"},
        {"structure": SI_POSCAR.replace("0.25 0.25 0.25", "0.26 0.25 0.25")},
        {"custom": {"stages": [custom_stage("static", "pbe", ["soc"])]}},
    ],
)
def test_meaningful_changes_alter_the_plan_digest(client, tokens, variant):
    baseline = post_plan(client, tokens["plan"], plan_body()).json()["plan_digest"]
    body = plan_body(
        variant.get("structure", SI_POSCAR),
        desired_output=variant.get("desired_output", "energy_only"),
        custom=variant.get("custom"),
        resources=variant.get("resources"),
    )
    response = post_plan(client, tokens["plan"], body)
    assert response.status_code == 200, response.text
    assert response.json()["plan_digest"] != baseline


def test_policy_version_changes_alter_the_plan_digest(client, tokens, monkeypatch):
    baseline = post_plan(client, tokens["plan"], plan_body()).json()["plan_digest"]
    monkeypatch.setattr(plan_digest_module, "ADMISSION_POLICY_VERSION", 999)
    assert post_plan(client, tokens["plan"], plan_body()).json()["plan_digest"] != baseline


def _multistage_plan(structure=SI_POSCAR, desired_output="electronic_dos"):
    plan = main.plan_calculation_request(structure_text=structure, fmt="poscar", desired_output=desired_output)
    return plan, list(plan_stage_previews(plan))


def _with_stage_poscar(previews, index, poscar):
    """Return previews where only stage ``index``'s generated POSCAR differs."""

    from types import SimpleNamespace

    changed = list(previews)
    preview = dict(changed[index])
    input_set = preview["input_set"]
    preview["input_set"] = SimpleNamespace(
        incar=input_set.incar,
        kpoints=input_set.kpoints,
        potcar=input_set.potcar,
        poscar=poscar,
    )
    changed[index] = preview
    return changed


@pytest.mark.parametrize("stage_index", [0, 1, 2])
def test_changing_only_one_stage_generated_poscar_changes_the_plan_digest(stage_index):
    from pymatgen.io.vasp.inputs import Poscar

    plan, previews = _multistage_plan()
    assert len(previews) == 3
    baseline = compute_plan_digest(structure=plan.structure, submission_spec=plan.submission_spec, stage_previews=previews)

    original = previews[stage_index]["input_set"].poscar
    perturbed = original.structure.copy()
    perturbed.translate_sites([1], [0.001, 0.0, 0.0], frac_coords=True)
    mutated = _with_stage_poscar(previews, stage_index, Poscar(perturbed))
    # Only the POSCAR of this one stage differs; every other digest input is unchanged.
    assert str(mutated[stage_index]["input_set"].poscar) != str(original)
    assert compute_plan_digest(
        structure=plan.structure, submission_spec=plan.submission_spec, stage_previews=mutated
    ) != baseline

    # Identical generated POSCAR content (a fresh but equal object) keeps the digest.
    same = _with_stage_poscar(previews, stage_index, Poscar(original.structure.copy(), comment=original.comment))
    assert str(same[stage_index]["input_set"].poscar) == str(original)
    assert compute_plan_digest(
        structure=plan.structure, submission_spec=plan.submission_spec, stage_previews=same
    ) == baseline


def test_plan_digest_material_hashes_every_stage_poscar():
    plan, previews = _multistage_plan()
    material = plan_digest_module.plan_digest_material(
        structure=plan.structure, submission_spec=plan.submission_spec, stage_previews=previews
    )
    assert [stage["poscar_sha256"] for stage in material["stages"]] == [
        hashlib.sha256(str(preview["input_set"].poscar).rstrip().encode("utf-8")).hexdigest()
        for preview in previews
    ]


def test_plan_digest_does_not_replace_the_attempt_fingerprint(client, tokens):
    spec, digest = _plan_and_digest(timestamp="20260101-000000", attempt_id=str(uuid.uuid4()))
    assert spec["submission"]["attempt_fingerprint"] and not digest.endswith(spec["submission"]["attempt_fingerprint"])


# ------------------------------------------- 7. no remote or writing operations


def test_planning_performs_no_remote_ssh_slurm_or_file_writing_operation(client, tokens, monkeypatch):
    import backend.paramiko_remote as paramiko_remote
    import backend.remote_runtime as remote_runtime

    for name in ("prepare_remote_submission", "submit_remote_workflow", "monitor_job", "monitor_submitted_job", "load_results_for_completed_job"):
        monkeypatch.setattr(main, name, _unreachable)
    monkeypatch.setattr(remote_runtime, "connected_remote_runner", _unreachable)
    monkeypatch.setattr(paramiko_remote.ParamikoRemoteRunner, "connect", _unreachable)

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

    for body in (
        plan_body(),
        plan_body(BI2SE3_POSCAR, desired_output="electronic_dos"),
        plan_body(NIO_POSCAR),
    ):
        response = post_plan(client, tokens["plan"], body)
        assert response.status_code == 200, response.text


# ---------------------------------------------- 8. tokens never leak anywhere


def test_tokens_never_appear_in_responses_logs_or_errors(client, tokens, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    token = tokens["plan"]
    secret = token.rsplit(".", 1)[1]
    responses = [
        client.get("/api/v1/identity", headers=bearer(token)),
        post_plan(client, token, plan_body()),
        post_plan(client, token, plan_body(structure="garbage")),
        post_plan(client, token, {"unexpected": token}),
        post_plan(client, tokens["read"], plan_body()),
        post_plan(client, token[:-2] + "zz", plan_body()),
    ]

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "build_submission_state_from_structure", explode)
    responses.append(post_plan(client, token, plan_body()))

    for response in responses:
        rendered = response.text + json.dumps(dict(response.headers))
        for value in (token, secret, tokens["read"], tokens["read"].rsplit(".", 1)[1]):
            assert value not in rendered
    for value in (token, secret, tokens["read"]):
        assert value not in caplog.text
    assert "Machine API plan generation failed (RuntimeError)" in caplog.text


# ------------------------------------------------ 9. browser routes unchanged


def test_browser_routes_do_not_require_api_tokens(tokens):
    ui = _browser_build(SI_POSCAR, workflow="energy_only")
    assert ui["submission_spec"]["submission"]["identity_token"]


def test_browser_openapi_surface_is_unchanged_and_api_is_not_advertised():
    paths = sorted(main.app.openapi()["paths"])
    assert paths == EXPECTED_BROWSER_OPENAPI_PATHS


def test_api_routes_accept_only_their_declared_methods(client, tokens):
    assert client.get("/api/v1/plans", headers=bearer(tokens["plan"])).status_code == 405
    assert client.post("/api/v1/identity", headers=bearer(tokens["plan"])).status_code == 405
    assert client.post("/api/v1/prepare", headers=bearer(tokens["plan"])).status_code == 404
    assert client.post("/api/v1/submit", headers=bearer(tokens["plan"])).status_code == 404


@pytest.mark.parametrize("path", ["/api/v1/plans/", "/api/v1//plans", "/API/v1/plans", "/api/v1/./plans"])
def test_alternative_paths_never_bypass_authentication(tokens, no_science, path):
    client = TestClient(main.app, client=LOOPBACK, follow_redirects=False)
    response = client.post(path, json=plan_body())
    assert response.status_code in {307, 401, 404, 405}
    if response.status_code == 307:
        followed = TestClient(main.app, client=LOOPBACK).post(path, json=plan_body())
        assert followed.status_code == 401
