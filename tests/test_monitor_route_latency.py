import json
import re

import main
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.config import DEFAULT_REMOTE_HOST, DEFAULT_USERNAME
from backend.submission import create_submission_identity_token
from starlette.requests import Request


ATTEMPT_ID = "12345678-1234-5678-9234-567812345678"


def request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/monitor",
            "headers": [],
        }
    )


def build_request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/build-calculation",
            "headers": [],
        }
    )


def resume_request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/resume",
            "headers": [],
        }
    )


def response_html(response):
    return response.body.decode("utf-8")


def structure_input_is_open(response):
    match = re.search(
        r'<details\s+id="structure-input-details"(?P<attrs>.*?)>',
        response_html(response),
        re.DOTALL,
    )
    assert match, "Structure Input disclosure should render"
    return bool(re.search(r"\bopen\b", match.group("attrs")))


def monitor_submission_spec():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    return {
        "schema": "bmd_compute.submission",
        "schema_version": 1,
        "status": "pending",
        "run_name": "vasp_run_static-20260817-120000",
        "created_at": "20260817-120000",
        "flow_spec": {
            "workflow_spec": workflow.to_dict(),
            "workflow": "static",
            "potcar_functional": "PBE_64",
        },
        "paths": {
            "run_dir": "/bmd/flows/vasp_run_static-20260817-120000",
            "remote_script": "/bmd/flows/vasp_run_static-20260817-120000.sbatch.sh",
            "logs_dir": "/bmd/logs",
            "log_out": "/bmd/logs/run.out",
            "log_err": "/bmd/logs/run.err",
            "slurm_out": "/bmd/logs/run.slurm.out",
            "slurm_err": "/bmd/logs/run.slurm.err",
            "stage_dirs": {},
            "result_dir": "/bmd/flows/vasp_run_static-20260817-120000",
        },
        "cluster": {
            "ssh_config_host": DEFAULT_REMOTE_HOST,
            "remote_host": DEFAULT_REMOTE_HOST,
            "username": DEFAULT_USERNAME,
            "port": 22,
            "partition": "leeburton-pool",
            "account": "default",
        },
        "resources": {
            "nodes": 1,
            "ntasks": 24,
            "mem_gb": 128,
            "walltime": "24:00:00",
        },
        "potcar": {
            "functional": "PBE_64",
            "species": [],
            "symbols": [],
            "symbol_source": "none",
            "repository": "shared",
            "target": "/potcars/PBE_64",
            "symlink_targets": [],
        },
        "modules": {
            "purge_first": True,
            "load": [],
        },
        "environment": {
            "VASP_CMD": "mpirun -n $SLURM_NTASKS vasp_std",
        },
        "runner": {
            "python": "python",
            "script_name": "run_job.py",
            "working_directory": "/bmd/flows/vasp_run_static-20260817-120000",
        },
        "submission": {
            "reason": "submitted",
            "attempt_id": ATTEMPT_ID,
            "attempt_state": f"/bmd/logs/submission_attempts/{ATTEMPT_ID}.json",
            "identity_token": create_submission_identity_token(
                "20260817-120000",
                ATTEMPT_ID,
            ),
        },
    }


def authoritative_job_record():
    spec = monitor_submission_spec()
    return {
        "schema": "bmd_compute.job_record",
        "schema_version": 1,
        "job_id": "123456",
        "run_name": spec["run_name"],
        "run_dir": spec["paths"]["run_dir"],
        "attempt_id": ATTEMPT_ID,
        "submission_spec": spec,
    }


def monitor_state():
    return main.monitor_state_json(
        summary={
            "formula": "Si2",
            "reduced_formula": "Si",
            "natoms": 2,
            "volume": 40.0,
            "density": 2.3,
            "lattice": {"a": 1, "b": 1, "c": 1},
            "angles": {"alpha": 90, "beta": 90, "gamma": 90},
            "space_group_symbol": "Fd-3m",
            "space_group_number": 227,
            "crystal_system": "cubic",
        },
        calculation_summary={
            "flow_name": "vasp_run_static",
            "number_of_jobs": 1,
            "calculation_type": "Static Energy",
            "calculation_plan": ["Static Energy"],
            "ready_for_submission": True,
        },
        generated_inputs={
            "incar": "LWAVE = False\n",
            "kpoints": "KPOINTS\n",
            "poscar": "POSCAR\n",
            "slurm_script": "#!/bin/bash\n",
            "vasp_executable": "vasp_std",
        },
        submission_spec=monitor_submission_spec(),
    )


def refresh_with_state(monkeypatch, *, state_summary):
    expensive_calls = []

    def expensive(name):
        def fail(*args, **kwargs):
            expensive_calls.append(name)
            raise AssertionError(f"{name} should not run during /monitor")

        return fail

    for name in (
        "build_submission_state",
        "parse_structure",
        "summarize_structure",
        "preview_generated_inputs",
        "build_calculation_flow",
    ):
        monkeypatch.setattr(main, name, expensive(name))

    monitor_calls = []

    def fake_monitor(job_id, *, authenticated_attempt_id):
        submission_spec = authoritative_job_record()["submission_spec"]
        assert authenticated_attempt_id == ATTEMPT_ID
        monitor_calls.append(
            {
                "job_id": job_id,
                "run_dir": submission_spec["paths"]["run_dir"],
                "has_structure_text": bool(
                    submission_spec.get("flow_spec", {})
                    .get("structure", {})
                    .get("text")
                ),
            }
        )
        return (
            {
                "status": "success",
                "job_id": job_id,
                "slurm_state": state_summary,
                "summary": state_summary,
                "exit_code": None,
                "brief": f"{job_id}|{state_summary}",
            },
            authoritative_job_record(),
        )

    monkeypatch.setattr(main, "monitor_submitted_job", fake_monitor)
    monkeypatch.setattr(
        main,
        "load_results_for_completed_job",
        expensive("load_results_for_completed_job"),
    )

    response = main.refresh_monitoring(
        request(),
        structure="Si POSCAR would be expensive to parse",
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus="24",
        memory_gb="128",
        walltime="24:00:00",
        queue="leeburton-pool",
        created_at="20260817-120000",
        job_id="123456",
        submitted_at="2026-08-17 12:00:00",
        monitor_state_json=monitor_state(),
        workflow_spec_json=json.dumps(
            WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)]).to_dict()
        ),
    )

    assert expensive_calls == []
    assert monitor_calls == [
        {
            "job_id": "123456",
            "run_dir": "/bmd/flows/vasp_run_static-20260817-120000",
            "has_structure_text": False,
        }
    ]
    assert response.status_code == 200
    assert response.context["summary"]["formula"] == "Si2"
    assert response.context["calculation"]["flow_name"] == "vasp_run_static"
    assert response.context["generated_inputs"]["incar"] == "LWAVE = False\n"
    assert response.context["monitoring_result"]["summary"] == state_summary
    assert response.context["collapse_structure_input"] is True
    assert structure_input_is_open(response) is False
    assert "Si POSCAR would be expensive to parse" in response_html(response)
    assert "Current SLURM State" in response_html(response)


def test_pending_monitor_uses_persisted_state_without_rebuilding(monkeypatch):
    refresh_with_state(monkeypatch, state_summary="PENDING")


def test_running_monitor_uses_persisted_state_without_rebuilding(monkeypatch):
    refresh_with_state(monkeypatch, state_summary="RUNNING")


def test_completed_monitor_is_monitoring_only_and_shows_load_results(monkeypatch):
    def fail_results(*args, **kwargs):
        raise AssertionError("Completed /monitor should not load results automatically")

    monitor_calls = []

    def fake_monitor(job_id, *, authenticated_attempt_id):
        monitor_calls.append(job_id)
        assert authenticated_attempt_id == ATTEMPT_ID
        assert authoritative_job_record()["submission_spec"]["paths"]["run_dir"] == (
            "/bmd/flows/vasp_run_static-20260817-120000"
        )
        return (
            {
                "status": "success",
                "job_id": job_id,
                "slurm_state": "COMPLETED",
                "summary": "SUCCESS",
                "exit_code": "0:0",
                "brief": f"{job_id}|COMPLETED",
            },
            authoritative_job_record(),
        )

    monkeypatch.setattr(main, "monitor_submitted_job", fake_monitor)
    monkeypatch.setattr(main, "load_results_for_completed_job", fail_results)

    response = main.refresh_monitoring(
        request(),
        structure="Si POSCAR remains preserved",
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus="24",
        memory_gb="128",
        walltime="24:00:00",
        queue="leeburton-pool",
        created_at="20260817-120000",
        job_id="123456",
        submitted_at="2026-08-17 12:00:00",
        monitor_state_json=monitor_state(),
        workflow_spec_json=json.dumps(
            WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)]).to_dict()
        ),
    )

    html = response_html(response)
    assert response.status_code == 200
    assert monitor_calls == ["123456"]
    assert response.context["monitoring_result"]["slurm_state"] == "COMPLETED"
    assert response.context["monitoring_result"]["summary"] == "SUCCESS"
    assert response.context["results_summary"] is None
    assert response.context["collapse_structure_input"] is True
    assert structure_input_is_open(response) is False
    assert "Si POSCAR remains preserved" in html
    assert "COMPLETED" in html
    assert "Load Results" in html
    assert 'name="load_results" value="true"' in html
    assert 'action="/resume"' in html


def test_home_does_not_touch_results_loading(monkeypatch):
    def slow_results(*args, **kwargs):
        raise AssertionError("GET / should not wait for results work")

    monkeypatch.setattr(main, "load_results_for_completed_job", slow_results)
    response = main.home(
        Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [],
            }
        )
    )

    assert response.status_code == 200
    assert response.context["results_summary"] is None


def test_initial_page_structure_input_is_expanded(monkeypatch):
    def slow_results(*args, **kwargs):
        raise AssertionError("GET / should not wait for results work")

    monkeypatch.setattr(main, "load_results_for_completed_job", slow_results)

    response = main.home(
        Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [],
            }
        )
    )

    assert response.status_code == 200
    assert response.context["collapse_structure_input"] is False
    assert structure_input_is_open(response) is True


def test_build_calculation_keeps_structure_input_expanded(monkeypatch):
    parsed_structure = object()

    def fake_parse_structure(structure_text, fmt):
        assert structure_text == "Si edit structure"
        assert fmt == "poscar"
        return parsed_structure

    def fake_method_considerations_context(structure_obj, *, workflow=None):
        assert structure_obj is parsed_structure
        assert workflow is not None
        return None

    def fake_build_submission_state(**kwargs):
        assert kwargs["structure_obj"] is parsed_structure
        assert kwargs["structure_text"] == "Si edit structure"
        return (
            {
                "formula": "Si2",
                "reduced_formula": "Si",
                "natoms": 2,
                "volume": 40.0,
                "density": 2.3,
                "lattice": {"a": 1, "b": 1, "c": 1},
                "angles": {"alpha": 90, "beta": 90, "gamma": 90},
                "space_group_symbol": "Fd-3m",
                "space_group_number": 227,
                "crystal_system": "cubic",
            },
            {
                "flow_name": "vasp_run_static",
                "number_of_jobs": 1,
                "calculation_type": "Static Energy",
                "calculation_plan": ["Static Energy"],
                "ready_for_submission": True,
            },
            {
                "incar": "LWAVE = False\n",
                "kpoints": "KPOINTS\n",
                "poscar": "POSCAR\n",
                "slurm_script": "#!/bin/bash\n",
                "vasp_executable": "vasp_std",
            },
            monitor_submission_spec(),
        )

    monkeypatch.setattr(main, "parse_structure", fake_parse_structure)
    monkeypatch.setattr(main, "method_considerations_context", fake_method_considerations_context)
    monkeypatch.setattr(
        main,
        "build_submission_state_from_structure",
        fake_build_submission_state,
    )

    response = main.build_workflow(
        build_request(),
        structure="Si edit structure",
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus="24",
        memory_gb="128",
        walltime="24:00:00",
        queue="leeburton-pool",
        workflow_spec_json=json.dumps(
            WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)]).to_dict()
        ),
    )

    assert response.status_code == 200
    assert response.context["collapse_structure_input"] is False
    assert structure_input_is_open(response) is True
    assert "Si edit structure" in response_html(response)


def test_resume_restores_completed_monitoring_without_synchronous_results(monkeypatch):
    def fake_monitor(job_id):
        return {
            "status": "success",
            "job_id": job_id,
            "slurm_state": "COMPLETED",
            "summary": "SUCCESS",
            "exit_code": "0:0",
            "brief": f"{job_id}|COMPLETED",
        }

    def slow_results(*args, **kwargs):
        raise AssertionError("Resume should not synchronously load completed results")

    monkeypatch.setattr(main, "monitor_job", fake_monitor)
    monkeypatch.setattr(main, "load_results_for_completed_job", slow_results)

    response = main.resume_existing_calculation(
        resume_request(),
        job_id="20791016",
        load_results="false",
    )

    assert response.status_code == 200
    assert response.context["monitoring_result"]["summary"] == "SUCCESS"
    assert response.context["results_summary"] is None
    assert response.context["resume_job_id"] == "20791016"
    assert response.context["collapse_structure_input"] is True
    assert structure_input_is_open(response) is False
    assert "Current SLURM State" in response_html(response)


def test_resume_load_results_button_explicitly_loads_completed_results(monkeypatch):
    result_calls = []

    def fake_monitor(job_id):
        return {
            "status": "success",
            "job_id": job_id,
            "slurm_state": "COMPLETED",
            "summary": "SUCCESS",
            "exit_code": "0:0",
            "brief": f"{job_id}|COMPLETED",
        }

    def fake_results(monitoring_result):
        result_calls.append(monitoring_result["job_id"])
        return {
            "status": "success",
            "final_formula": "Si",
            "visualizations": [],
            "viewer": {},
            "files": {},
            "diagnostics": {},
        }

    monkeypatch.setattr(main, "monitor_job", fake_monitor)
    monkeypatch.setattr(main, "load_results_for_completed_job", fake_results)

    response = main.resume_existing_calculation(
        resume_request(),
        job_id="20791016",
        load_results="true",
    )

    assert response.status_code == 200
    assert response.context["results_summary"]["final_formula"] == "Si"
    assert result_calls == ["20791016"]
    assert response.context["collapse_structure_input"] is True
    assert structure_input_is_open(response) is False
    assert "<h2>Results</h2>" in response_html(response)
