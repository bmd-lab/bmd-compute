import json

import pytest
from starlette.requests import Request

import main
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.config import bmd_debug_enabled
from backend.submission import create_submission_identity_token


ATTEMPT_ID = "12345678-1234-5678-9234-567812345678"


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
        }
    )


def _html(response) -> str:
    return response.body.decode("utf-8")


def _workflow_spec_json() -> str:
    return json.dumps(
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)]).to_dict(),
        sort_keys=True,
    )


def _monitor_state_json() -> str:
    return json.dumps(
        {
            "submission_spec": {
                "submission": {
                    "attempt_id": ATTEMPT_ID,
                    "identity_token": create_submission_identity_token(
                        "20260818-120000",
                        ATTEMPT_ID,
                    ),
                }
            }
        },
        sort_keys=True,
    )


def _failed_monitoring_result() -> dict:
    return {
        "status": "failed",
        "title": "Monitoring Failed",
        "stage": "Monitoring",
        "reason": "BMD Compute could not check the queue status.",
        "suggestion": "Please try again shortly.",
        "command": "python C:\\Users\\student\\bmd_compute\\backend\\monitoring.py",
        "stdout": "Traceback (most recent call last): stdout internals",
        "stderr": "File C:\\Users\\student\\bmd_compute\\backend\\paramiko_remote.py",
        "exception_text": "RuntimeError: internal monitoring detail",
        "exception_debug": {
            "type": "<class 'RuntimeError'>",
            "module": "backend.monitoring",
            "class_name": "RuntimeError",
            "repr": "RuntimeError('internal monitoring detail')",
            "traceback": (
                "Traceback (most recent call last):\n"
                "  File C:\\Users\\student\\bmd_compute\\backend\\monitoring.py\n"
                "RuntimeError: internal monitoring detail\n"
            ),
        },
    }


def _failed_results_summary() -> dict:
    return {
        "status": "failed",
        "title": "Results Summary Failed",
        "stage": "Results Retrieval",
        "reason": "Results could not be loaded.",
        "suggestion": "Check the job status and try again.",
        "files": {
            "vasprun": "/bmd-db/guest/flows/run/stage_01/vasprun.xml",
        },
        "exception_text": "ValueError: internal parser detail",
        "traceback": (
            "Traceback (most recent call last):\n"
            "  File C:\\Users\\student\\bmd_compute\\backend\\results.py\n"
            "ValueError: internal parser detail\n"
        ),
    }


def test_bmd_debug_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("BMD_DEBUG", "sometimes")

    with pytest.raises(RuntimeError):
        bmd_debug_enabled()


def test_monitoring_traceback_hidden_when_debug_disabled(monkeypatch):
    monkeypatch.setenv("BMD_DEBUG", "false")
    monkeypatch.setattr(
        main,
        "monitor_submitted_job",
        lambda *args, **kwargs: (_failed_monitoring_result(), None),
    )

    response = main.refresh_monitoring(
        _request("/monitor"),
        structure="Si structure remains available",
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus="24",
        memory_gb="128",
        walltime="24:00:00",
        queue="leeburton-pool",
        created_at="20260818-120000",
        job_id="12345",
        submitted_at="2026-08-18 12:00:00",
        monitor_state_json=_monitor_state_json(),
        workflow_spec_json=_workflow_spec_json(),
    )

    html = _html(response)
    assert "BMD Compute could not check the queue status." in html
    assert "Please try again shortly." in html
    assert "Traceback" not in html
    assert "backend\\monitoring.py" not in html
    assert "backend\\paramiko_remote.py" not in html
    assert "internal monitoring detail" not in html


def test_monitoring_traceback_visible_when_debug_enabled(monkeypatch):
    monkeypatch.setenv("BMD_DEBUG", "true")
    monkeypatch.setattr(
        main,
        "monitor_submitted_job",
        lambda *args, **kwargs: (_failed_monitoring_result(), None),
    )

    response = main.refresh_monitoring(
        _request("/monitor"),
        structure="Si structure remains available",
        fmt="poscar",
        purpose="static",
        theory="pbe",
        modifiers=None,
        cpus="24",
        memory_gb="128",
        walltime="24:00:00",
        queue="leeburton-pool",
        created_at="20260818-120000",
        job_id="12345",
        submitted_at="2026-08-18 12:00:00",
        monitor_state_json=_monitor_state_json(),
        workflow_spec_json=_workflow_spec_json(),
    )

    html = _html(response)
    assert "Traceback" in html
    assert "backend\\monitoring.py" in html
    assert "internal monitoring detail" in html


def test_results_traceback_hidden_when_debug_disabled(monkeypatch):
    monkeypatch.setenv("BMD_DEBUG", "false")
    monkeypatch.setattr(
        main,
        "monitor_job",
        lambda job_id: {
            "status": "success",
            "job_id": job_id,
            "slurm_state": "COMPLETED",
            "summary": "SUCCESS",
            "exit_code": "0:0",
            "brief": f"{job_id}|COMPLETED",
        },
    )
    monkeypatch.setattr(
        main,
        "load_results_for_completed_job",
        lambda monitoring_result: _failed_results_summary(),
    )

    response = main.resume_existing_calculation(
        _request("/resume"),
        job_id="20791016",
        load_results="true",
    )

    html = _html(response)
    assert "Results could not be loaded." in html
    assert "Check the job status and try again." in html
    assert "Traceback" not in html
    assert "backend\\results.py" not in html
    assert "internal parser detail" not in html
    assert "vasprun.xml" not in html


def test_results_traceback_visible_when_debug_enabled(monkeypatch):
    monkeypatch.setenv("BMD_DEBUG", "true")
    monkeypatch.setattr(
        main,
        "monitor_job",
        lambda job_id: {
            "status": "success",
            "job_id": job_id,
            "slurm_state": "COMPLETED",
            "summary": "SUCCESS",
            "exit_code": "0:0",
            "brief": f"{job_id}|COMPLETED",
        },
    )
    monkeypatch.setattr(
        main,
        "load_results_for_completed_job",
        lambda monitoring_result: _failed_results_summary(),
    )

    response = main.resume_existing_calculation(
        _request("/resume"),
        job_id="20791016",
        load_results="true",
    )

    html = _html(response)
    assert "Traceback" in html
    assert "backend\\results.py" in html
    assert "internal parser detail" in html
    assert "vasprun.xml" in html
