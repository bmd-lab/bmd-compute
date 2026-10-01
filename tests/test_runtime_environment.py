"""Runtime scientific-stack parity: preparation versions must be the POWER versions."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

import backend.runtime_environment as runtime_environment
import backend.workflows as workflows
from backend.execution import run_submission
from backend.runtime_environment import (
    PARITY_CRITICAL_PACKAGES,
    RECORDED_SUPPORTING_PACKAGES,
    RUNTIME_ENVIRONMENT_FILENAME,
    RUNTIME_ENVIRONMENT_SCHEMA,
    RuntimeParityError,
    enforce_runtime_environment,
    prepared_runtime_parity,
)
import backend.paramiko_remote as paramiko_remote
import backend.submission as submission
import main
from backend.calculations.registry import CalculationValidationError
from backend.submission import submission_attempt_fingerprint
from test_automatic_dft_u import NIO, build_route
from test_default_treatment_resolution import _managed_identity, _prepare_managed, _submit_managed


PRODUCTION = {
    "atomate2": "0.1.5",
    "pymatgen": "2026.5.4",
    "pymatgen-core": "2026.7.16",
    "custodian": "2025.12.14",
    "emmet-core": "0.87.1",
    "jobflow": "0.1.19",
    "spglib": "2.7.0",
    "monty": "2026.7.16",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "pydantic": "2.13.4",
    "pydantic-settings": "2.14.2",
    "maggma": "0.74.0",
    "ruamel.yaml": "0.19.1",
}


class Settings:
    CONFIG_FILE = "~/.atomate2.yaml"
    VASP_HANDLE_UNSUCCESSFUL = "error"
    VASP_INCAR_UPDATES: dict = {}
    VASP_INHERIT_INCAR = False
    SYMPREC = 0.1
    BANDGAP_TOL = 0.0001
    VASP_CUSTODIAN_MAX_ERRORS = 5
    VASP_ZIP_FILES = "atomate"
    VASP_GAMMA_CMD = "vasp_gam"
    VASP_NCL_CMD = "vasp_ncl"
    CUSTODIAN_SCRATCH_DIR = None


def lookup(versions):
    def version(name):
        if versions.get(name) is None:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    return version


def prepared(versions=PRODUCTION):
    return prepared_runtime_parity(version_lookup=lookup(versions))


def spec_for(tmp_path, runtime_parity):
    return {
        "run_name": "NiO-static-20260929-120000",
        "submission": {"attempt_id": "3c8f2d8e-2f7a-4c1b-9a51-2d0d1b6b0e11"},
        "paths": {"run_dir": str(tmp_path)},
        "runtime_parity": runtime_parity,
    }


def enforce(tmp_path, runtime_parity, runtime_versions=PRODUCTION, settings=None):
    return enforce_runtime_environment(
        spec_for(tmp_path, runtime_parity),
        version_lookup=lookup(runtime_versions),
        settings=settings or Settings(),
        environ={"SLURM_JOB_ID": "22400000"},
    )


def read_record(tmp_path):
    return json.loads((tmp_path / RUNTIME_ENVIRONMENT_FILENAME).read_text(encoding="utf-8"))


# --- Parity outcomes -------------------------------------------------------------


def test_identical_stack_is_allowed_and_recorded(tmp_path):
    record = enforce(tmp_path, prepared())

    written = read_record(tmp_path)
    assert written == record
    assert written["schema"] == RUNTIME_ENVIRONMENT_SCHEMA
    assert written["schema_version"] == 1
    assert written["status"] == "passed"
    assert written["problems"] == []
    assert written["prepared_packages"] == {name: PRODUCTION[name] for name in PARITY_CRITICAL_PACKAGES}
    assert written["runtime_packages"] == {name: PRODUCTION[name] for name in PARITY_CRITICAL_PACKAGES}
    assert written["supporting_packages"] == {name: PRODUCTION[name] for name in RECORDED_SUPPORTING_PACKAGES}
    assert written["slurm_job_id"] == "22400000"
    assert written["attempt_id"] == "3c8f2d8e-2f7a-4c1b-9a51-2d0d1b6b0e11"
    assert written["atomate2_settings"]["VASP_HANDLE_UNSUCCESSFUL"] == "error"
    assert [path.name for path in tmp_path.iterdir()] == [RUNTIME_ENVIRONMENT_FILENAME]


@pytest.mark.parametrize(
    ("package", "newer"),
    [
        ("atomate2", "0.1.6"),
        ("pymatgen-core", "2026.9.23"),
        ("emmet-core", "0.87.2"),
        ("jobflow", "0.3.1"),
        ("custodian", "2026.1.1"),
        ("spglib", "2.7.1"),
        ("pymatgen", "2026.5.5"),
    ],
)
def test_newer_parity_critical_package_on_power_stops_execution(tmp_path, package, newer):
    runtime_versions = {**PRODUCTION, package: newer}

    with pytest.raises(RuntimeParityError) as excinfo:
        enforce(tmp_path, prepared(), runtime_versions)

    expected = f"{package}: prepared {PRODUCTION[package]}, runtime {newer}."
    assert expected in excinfo.value.problems
    assert expected in str(excinfo.value)
    assert "VASP was not started" in str(excinfo.value)
    record = read_record(tmp_path)
    assert record["status"] == "failed"
    assert record["problems"] == [expected]
    assert record["runtime_packages"][package] == newer


def test_missing_parity_critical_package_stops_execution(tmp_path):
    runtime_versions = {**PRODUCTION, "custodian": None}
    with pytest.raises(RuntimeParityError, match="custodian: prepared 2025.12.14, not installed at runtime"):
        enforce(tmp_path, prepared(), runtime_versions)
    assert read_record(tmp_path)["runtime_packages"]["custodian"] is None


def test_supporting_package_difference_is_recorded_but_does_not_stop(tmp_path):
    runtime_versions = {**PRODUCTION, "numpy": "2.5.3", "maggma": None}
    record = enforce(tmp_path, prepared(), runtime_versions)
    assert record["status"] == "passed"
    assert record["supporting_packages"]["numpy"] == "2.5.3"
    assert record["supporting_packages"]["maggma"] is None


def _malformed_blocks():
    good = prepared()
    return {
        "absent": None,
        "not_an_object": ["atomate2==0.1.5"],
        "wrong_policy": {**good, "policy_id": "something.else"},
        "string_policy_version": {**good, "policy_version": "1"},
        "future_policy_version": {**good, "policy_version": 2},
        "packages_not_object": {**good, "packages": "atomate2==0.1.5"},
        "missing_package": {**good, "packages": {k: v for k, v in good["packages"].items() if k != "spglib"}},
        "extra_package": {**good, "packages": {**good["packages"], "numpy": "2.5.1"}},
        "empty_version": {**good, "packages": {**good["packages"], "jobflow": ""}},
        "non_string_version": {**good, "packages": {**good["packages"], "jobflow": 0.1}},
    }


@pytest.mark.parametrize("case", sorted(_malformed_blocks()))
def test_malformed_prepared_data_stops_execution(tmp_path, case):
    with pytest.raises(RuntimeParityError):
        enforce(tmp_path, _malformed_blocks()[case])
    record = read_record(tmp_path)
    assert record["status"] == "failed"
    assert record["problems"]


def test_unreadable_runtime_versions_stop_execution(tmp_path):
    def broken(name):
        raise ValueError("corrupt metadata")

    with pytest.raises(RuntimeParityError, match="not installed at runtime"):
        enforce_runtime_environment(
            spec_for(tmp_path, prepared()), version_lookup=broken, settings=Settings(), environ={}
        )


def test_preparation_refuses_when_a_parity_package_is_missing():
    with pytest.raises(RuntimeParityError, match="emmet-core"):
        prepared({**PRODUCTION, "emmet-core": None})


def test_missing_parity_package_stops_preparation_and_submission_routes(monkeypatch):
    identity, workflow_spec_json = _managed_identity()
    real = runtime_environment._default_version_lookup

    def preparation_lookup(name):
        if name == "custodian":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    def unreachable(*args, **kwargs):
        raise AssertionError("a missing parity package must stop before any remote work")

    monkeypatch.setattr(runtime_environment, "_default_version_lookup", preparation_lookup)
    monkeypatch.setattr(main, "prepare_remote_submission", unreachable)
    monkeypatch.setattr(main, "submit_remote_workflow", unreachable)
    monkeypatch.setattr(paramiko_remote, "ParamikoRemoteRunner", unreachable)

    with pytest.raises(CalculationValidationError) as excinfo:
        submission._prepared_runtime_parity_or_error()
    assert isinstance(excinfo.value.__cause__, RuntimeParityError)

    responses = (
        build_route(NIO, workflow="energy_only"),
        _prepare_managed(identity, workflow_spec_json),
        _submit_managed(identity, workflow_spec_json),
    )
    for response in responses:
        assert response.status_code == 400
        error = response.context["calculation_error"]
        assert error["message"] == (
            "The preparation environment is missing parity-critical package(s): custodian"
        )
        assert "constraints/scientific-runtime.txt" in error["suggestion"]
        assert not response.context.get("generated_inputs")
        assert not response.context.get("submission_spec")
        assert not response.context.get("remote_preparation")
        assert not response.context.get("submission_result")
        assert "custodian" in response.template.render(response.context)


# --- Mutable atomate2 settings ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value", "fragment"),
    [
        ("VASP_INCAR_UPDATES", {"ALGO": "All"}, "VASP_INCAR_UPDATES"),
        ("VASP_INHERIT_INCAR", True, "VASP_INHERIT_INCAR"),
        ("VASP_INHERIT_INCAR", ["ENCUT"], "VASP_INHERIT_INCAR"),
    ],
)
def test_input_altering_atomate2_settings_stop_execution(tmp_path, name, value, fragment):
    settings = Settings()
    setattr(settings, name, value)
    with pytest.raises(RuntimeParityError, match=fragment):
        enforce(tmp_path, prepared(), settings=settings)
    assert read_record(tmp_path)["atomate2_settings"][name] == value


# --- Real environment and the runner ---------------------------------------------------


def test_record_reflects_the_actual_installed_versions(tmp_path):
    record = enforce_runtime_environment(spec_for(tmp_path, prepared_runtime_parity()), environ={})

    assert record["status"] == "passed"
    for name in PARITY_CRITICAL_PACKAGES:
        assert record["runtime_packages"][name] == metadata.version(name)
    for name in RECORDED_SUPPORTING_PACKAGES:
        assert record["supporting_packages"][name] == metadata.version(name)
    from atomate2 import SETTINGS

    assert record["atomate2_settings"]["SYMPREC"] == SETTINGS.SYMPREC


def test_submission_records_parity_and_it_is_fingerprinted():
    spec = build_route(NIO, workflow="energy_only").context["submission_spec"]

    parity = spec["runtime_parity"]
    assert parity["policy_id"] == "bmd_compute.runtime_parity"
    assert parity["packages"] == {name: metadata.version(name) for name in PARITY_CRITICAL_PACKAGES}
    assert spec["paths"]["runtime_environment"] == f"{spec['paths']['run_dir']}/{RUNTIME_ENVIRONMENT_FILENAME}"
    remote = spec["provenance"]["python_environment"]["remote_execution"]
    assert remote["record"] == spec["paths"]["runtime_environment"]
    assert remote["parity_critical_packages"] == list(PARITY_CRITICAL_PACKAGES)

    tampered = json.loads(json.dumps(spec))
    tampered["runtime_parity"]["packages"]["jobflow"] = "0.3.1"
    assert submission_attempt_fingerprint(tampered) != submission_attempt_fingerprint(spec)


def _runner_spec(tmp_path):
    spec = json.loads(json.dumps(build_route(NIO, workflow="energy_only").context["submission_spec"]))
    spec["paths"]["run_dir"] = str(tmp_path)
    return spec


def test_runner_stops_before_building_the_workflow_on_mismatch(tmp_path, monkeypatch):
    spec = _runner_spec(tmp_path)
    spec["runtime_parity"]["packages"]["atomate2"] = "0.1.4"  # prepared elsewhere

    def must_not_run(*args, **kwargs):
        raise AssertionError("workflow must not be built")

    monkeypatch.setattr(workflows, "build_atomate2_flow_from_spec", must_not_run)
    with pytest.raises(RuntimeParityError, match=r"atomate2: prepared 0\.1\.4, runtime 0\.1\.5"):
        run_submission(spec)
    assert read_record(tmp_path)["status"] == "failed"


def test_runner_stops_when_power_is_one_version_ahead(tmp_path, monkeypatch):
    spec = _runner_spec(tmp_path)
    real = runtime_environment._default_version_lookup

    def power_lookup(name):
        return "0.87.2" if name == "emmet-core" else real(name)

    monkeypatch.setattr(runtime_environment, "_default_version_lookup", power_lookup)
    monkeypatch.setattr(workflows, "build_atomate2_flow_from_spec", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(RuntimeParityError, match="emmet-core"):
        run_submission(spec)


class _Reached(Exception):
    pass


def test_runner_proceeds_to_the_workflow_when_the_stack_matches(tmp_path, monkeypatch):
    spec = _runner_spec(tmp_path)

    def reached(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(workflows, "build_atomate2_flow_from_spec", reached)
    with pytest.raises(_Reached):
        run_submission(spec)
    assert read_record(tmp_path)["status"] == "passed"


# --- Atomic write ------------------------------------------------------------------------


def test_failed_record_write_leaves_the_previous_record_and_no_temporary(tmp_path, monkeypatch):
    enforce(tmp_path, prepared())
    before = (tmp_path / RUNTIME_ENVIRONMENT_FILENAME).read_bytes()

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runtime_environment.json, "dump", broken_dump)
    with pytest.raises(OSError):
        enforce(tmp_path, prepared())

    assert (tmp_path / RUNTIME_ENVIRONMENT_FILENAME).read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == [RUNTIME_ENVIRONMENT_FILENAME]
