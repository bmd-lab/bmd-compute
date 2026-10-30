from __future__ import annotations

import io
import ast
import json
import re
import sys
import types
from contextlib import redirect_stdout
from contextlib import contextmanager
from copy import deepcopy

from backend.calculations.models import CalculationSpec, Purpose, Theory
from backend.config import (
    DEFAULT_FLOWS_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_REMOTE_HOST,
    DEFAULT_USERNAME,
)
from backend.results import (
    REMOTE_RESULT_JSON_END,
    REMOTE_RESULT_JSON_START,
    clear_results_cache,
    load_results_for_completed_job,
    monitoring_indicates_success,
    parse_vasp_result_files,
    remote_result_parser_context,
    remote_result_parser_source,
    remote_job_state_path,
    results_cache_info,
)
from backend.remote import RemoteCommandResult, RemoteJobStatus


monitoring_success = {
    "status": "success",
    "job_id": "123456",
    "slurm_state": "COMPLETED",
    "summary": "SUCCESS",
    "exit_code": "0:0",
    "workdir": "/home/bmdguest",
}

RUN_DIR = f"{DEFAULT_FLOWS_DIR}/TiO2-static-20260629-120000"

submission_spec = {
    "cluster": {
        "ssh_config_host": DEFAULT_REMOTE_HOST,
        "remote_host": DEFAULT_REMOTE_HOST,
        "username": DEFAULT_USERNAME,
        "port": 22,
    },
    "paths": {
        "run_dir": RUN_DIR,
    },
}


class ResultsRunner:
    def __init__(self, *, missing: str | None = None, state_payload: dict | None = None):
        self.missing = missing
        self.state_payload = state_payload or {"run_dir": RUN_DIR}
        self.connected_profile = None
        self.closed = False
        self.checked_paths = []
        self.read_paths = []
        self.state_path = None

    def connect(self, profile):
        self.connected_profile = profile

    def close(self):
        self.closed = True

    def run(self, *args, **kwargs):
        raise AssertionError("Results should not use recursive shell discovery")

    def is_file(self, remote_path):
        self.checked_paths.append(remote_path)
        if remote_path == remote_job_state_path("123456"):
            return self.missing != "state"
        if remote_path.endswith("/vasprun.xml"):
            return self.missing != "vasprun"
        if remote_path.endswith("/DOSCAR"):
            return self.missing != "doscar"
        if remote_path.endswith("/KPOINTS"):
            return self.missing != "kpoints"
        if remote_path.endswith("/CONTCAR"):
            return self.missing != "contcar"
        if remote_path.endswith("/OUTCAR"):
            return self.missing != "outcar"
        return False

    def read_text(self, remote_path, *, max_bytes=None):
        self.state_path = remote_path
        return json.dumps(self.state_payload)

    def read_bytes(self, remote_path, *, max_bytes=None):
        self.read_paths.append(remote_path)
        return f"contents for {remote_path}\n".encode()


class RemoteParserRunner(ResultsRunner):
    def __init__(
        self,
        *,
        payload: dict | None = None,
        returncode: int = 0,
        raise_exc: Exception | None = None,
        state_payload: dict | None = None,
    ):
        super().__init__(state_payload=state_payload)
        self.payload = payload or fake_parser_payload()
        self.returncode = returncode
        self.raise_exc = raise_exc
        self.python = None
        self.timeout_s = None
        self.run_python_calls = 0
        self.source = ""

    def run_python(self, source, *, python, env=None, check=False, timeout_s=None):
        del env, check
        self.run_python_calls += 1
        self.source = source
        self.python = python
        self.timeout_s = timeout_s
        assert "parse_result_paths" in source
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.returncode != 0:
            return RemoteCommandResult(
                command="remote parser",
                returncode=self.returncode,
                stdout="",
                stderr="remote parser failed",
                elapsed_s=1.25,
            )
        payload = json.dumps(self.payload, separators=(",", ":"))
        return RemoteCommandResult(
            command="remote parser",
            returncode=0,
            stdout=(
                "diagnostic before payload\n"
                f"{REMOTE_RESULT_JSON_START}\n"
                f"{payload}\n"
                f"{REMOTE_RESULT_JSON_END}\n"
            ),
            stderr="",
            elapsed_s=1.25,
        )

    def read_bytes(self, remote_path, *, max_bytes=None):
        raise AssertionError(f"Raw result file should not be downloaded: {remote_path}")


def fake_parser_payload() -> dict:
    return {
        "status": "success",
        "completion_status": "COMPLETED (ExitCode 0:0)",
        "final_energy_ev": -10.25,
        "energy_per_atom_ev": -5.125,
        "ionic_steps": 7,
        "electronic_convergence": True,
        "converged_electronic": True,
        "ionic_convergence": None,
        "converged_ionic": None,
        "final_formula": "TiO2",
        "natoms": 3,
        "files": {
            "contcar": f"{RUN_DIR}/CONTCAR",
            "outcar": f"{RUN_DIR}/OUTCAR",
            "vasprun": f"{RUN_DIR}/vasprun.xml",
        },
        "diagnostics": {
            "parser": "pymatgen-remote",
            "outcar_parsed": True,
            "outcar_error": "",
            "remote_parsing": {
                "source_bytes": 145000000,
                "source_file_bytes": {
                    "contcar": 1024,
                    "outcar": 1800000,
                    "vasprun": 143198976,
                },
                "remote_parse_elapsed_s": 4.2,
            },
        },
        "visualizations": [],
        "viewer": {"format": "cif", "source": "CONTCAR", "cif": "data_TiO2\n"},
    }


def fake_parser(files, monitoring_result):
    assert monitoring_result["job_id"] == "123456"
    assert set(files) == {"contcar", "outcar", "vasprun"}
    return {
        "status": "success",
        "completion_status": "COMPLETED (ExitCode 0:0)",
        "final_energy_ev": -10.25,
        "energy_per_atom_ev": -5.125,
        "ionic_steps": 7,
        "electronic_convergence": True,
        "final_formula": "TiO2",
        "natoms": 3,
        "diagnostics": {"parser": "fake", "outcar_parsed": True, "outcar_error": ""},
        "viewer": {"format": "cif", "source": "CONTCAR", "cif": "data_TiO2\n"},
    }


def fake_dos_parser(files, monitoring_result):
    assert monitoring_result["job_id"] == "123456"
    assert set(files) == {"contcar", "outcar", "vasprun", "doscar"}
    result = fake_parser(
        {
            "contcar": files["contcar"],
            "outcar": files["outcar"],
            "vasprun": files["vasprun"],
        },
        monitoring_result,
    )
    result["dos"] = {
        "available": True,
        "source": "vasprun.xml",
        "doscar": files["doscar"]["path"],
        "energy_points": 4001,
        "energy_min_ev": -10.0,
        "energy_max_ev": 10.0,
        "spin_channels": 1,
    }
    return result


def fake_band_parser(files, monitoring_result):
    assert monitoring_result["job_id"] == "123456"
    assert set(files) == {"contcar", "outcar", "vasprun", "kpoints", "preceding_static_outcar"}
    result = fake_parser(
        {
            "contcar": files["contcar"],
            "outcar": files["outcar"],
            "vasprun": files["vasprun"],
        },
        monitoring_result,
    )
    result["band_structure"] = {
        "available": True,
        "source": "vasprun.xml",
        "kpoints": files["kpoints"]["path"],
        "bands": 12,
        "kpoints_count": 80,
        "spin_channels": 1,
    }
    result["visualizations"] = [
        {
            "id": "band_structure",
            "kind": "line_plot",
        },
    ]
    return result


def test_monitoring_success_detection_requires_completed_success():
    assert monitoring_indicates_success(monitoring_success)
    assert not monitoring_indicates_success({**monitoring_success, "summary": "RUNNING"})
    assert not monitoring_indicates_success({**monitoring_success, "exit_code": "1:0"})
    assert not monitoring_indicates_success({**monitoring_success, "slurm_state": "FAILED"})


def test_results_load_for_submitted_completed_job_uses_submission_profile():
    runner = ResultsRunner()
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=lambda: runner,
        parser=fake_parser,
    )

    assert runner.connected_profile.host == DEFAULT_REMOTE_HOST
    assert runner.connected_profile.username == DEFAULT_USERNAME
    assert runner.closed is True
    assert runner.checked_paths == [
        f"{RUN_DIR}/CONTCAR",
        f"{RUN_DIR}/OUTCAR",
        f"{RUN_DIR}/vasprun.xml",
    ]
    assert runner.state_path is None
    assert len(runner.read_paths) == 3
    assert result["status"] == "success"
    assert result["final_formula"] == "TiO2"
    assert result["files"]["contcar"] == f"{RUN_DIR}/CONTCAR"
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == RUN_DIR


def test_results_for_double_relax_use_final_stage_directory():
    double_relax_spec = {
        **submission_spec,
        "flow_spec": {
            "calculation_spec": {
                "purpose": "double_relax",
                "theory": "pbe",
                "modifiers": [],
            },
            "workflow": "double_relax",
            "potcar_functional": "PBE_64",
        },
    }
    runner = ResultsRunner()
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=double_relax_spec,
        runner_factory=lambda: runner,
        parser=fake_parser,
    )

    final_stage_dir = f"{RUN_DIR}/relax_02"
    assert runner.checked_paths == [
        f"{final_stage_dir}/CONTCAR",
        f"{final_stage_dir}/OUTCAR",
        f"{final_stage_dir}/vasprun.xml",
    ]
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == final_stage_dir
    assert result["files"]["contcar"] == f"{final_stage_dir}/CONTCAR"


def test_results_for_relax_static_use_final_stage_directory():
    relax_static_spec = {
        **submission_spec,
        "flow_spec": {
            "calculation_spec": {
                "purpose": "relax_static",
                "theory": "hse06",
                "modifiers": [],
            },
            "workflow": "relax_static",
            "potcar_functional": "PBE_64",
        },
    }
    runner = ResultsRunner()
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=relax_static_spec,
        runner_factory=lambda: runner,
        parser=fake_parser,
    )

    final_stage_dir = f"{RUN_DIR}/stage_02"
    assert runner.checked_paths == [
        f"{final_stage_dir}/CONTCAR",
        f"{final_stage_dir}/OUTCAR",
        f"{final_stage_dir}/vasprun.xml",
    ]
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == final_stage_dir
    assert result["files"]["contcar"] == f"{final_stage_dir}/CONTCAR"


def test_results_for_dos_use_final_stage_directory_and_doscar():
    dos_spec = {
        **submission_spec,
        "flow_spec": {
            "calculation_spec": {
                "purpose": "dos",
                "theory": "pbe",
                "modifiers": [],
            },
            "workflow": "dos",
            "potcar_functional": "PBE_64",
        },
    }
    runner = ResultsRunner()
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=dos_spec,
        runner_factory=lambda: runner,
        parser=fake_dos_parser,
    )

    final_stage_dir = f"{RUN_DIR}/stage_03"
    assert runner.checked_paths == [
        f"{final_stage_dir}/CONTCAR",
        f"{final_stage_dir}/OUTCAR",
        f"{final_stage_dir}/vasprun.xml",
        f"{final_stage_dir}/DOSCAR",
    ]
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == final_stage_dir
    assert result["files"]["doscar"] == f"{final_stage_dir}/DOSCAR"
    assert result["dos"]["available"] is True
    assert result["dos"]["energy_points"] == 4001


def test_results_for_band_structure_use_final_stage_directory_and_kpoints():
    band_spec = {
        **submission_spec,
        "flow_spec": {
            "calculation_spec": {
                "purpose": "band_structure",
                "theory": "pbe",
                "modifiers": [],
            },
            "workflow": "band_structure",
            "potcar_functional": "PBE_64",
        },
    }
    runner = ResultsRunner()
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=band_spec,
        runner_factory=lambda: runner,
        parser=fake_band_parser,
    )

    final_stage_dir = f"{RUN_DIR}/stage_03"
    assert runner.checked_paths == [
        f"{final_stage_dir}/CONTCAR",
        f"{final_stage_dir}/OUTCAR",
        f"{final_stage_dir}/vasprun.xml",
        f"{final_stage_dir}/KPOINTS",
        # The preceding PBE static's OUTCAR: the band structure's Fermi reference.
        f"{RUN_DIR}/stage_02/OUTCAR",
    ]
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == final_stage_dir
    assert result["files"]["kpoints"] == f"{final_stage_dir}/KPOINTS"
    assert result["band_structure"]["available"] is True
    assert result["band_structure"]["kpoints_count"] == 80
    assert result["visualizations"][0]["id"] == "band_structure"


def test_results_load_for_resumed_completed_job_uses_default_profile():
    runner = ResultsRunner()
    result = load_results_for_completed_job(
        monitoring_success,
        runner_factory=lambda: runner,
        parser=fake_parser,
    )

    assert runner.connected_profile.host == DEFAULT_REMOTE_HOST
    assert runner.connected_profile.username == DEFAULT_USERNAME
    assert runner.closed is True
    assert runner.state_path == f"{DEFAULT_LOGS_DIR}/job_123456.json"
    assert runner.checked_paths == [
        f"{DEFAULT_LOGS_DIR}/job_123456.json",
        f"{RUN_DIR}/CONTCAR",
        f"{RUN_DIR}/OUTCAR",
        f"{RUN_DIR}/vasprun.xml",
    ]
    assert result["status"] == "success"
    assert result["job_id"] == "123456"
    assert result["run_dir"] == RUN_DIR


def test_resumed_double_relax_uses_final_stage_directory_from_job_state():
    double_relax_state = {
        "run_dir": RUN_DIR,
        "submission_spec": {
            **submission_spec,
            "flow_spec": {
                "calculation_spec": {
                    "purpose": "double_relax",
                    "theory": "pbe",
                    "modifiers": [],
                },
                "workflow": "double_relax",
                "potcar_functional": "PBE_64",
            },
        },
    }
    runner = ResultsRunner(state_payload=double_relax_state)
    result = load_results_for_completed_job(
        monitoring_success,
        runner_factory=lambda: runner,
        parser=fake_parser,
    )

    final_stage_dir = f"{RUN_DIR}/relax_02"
    assert runner.checked_paths == [
        f"{DEFAULT_LOGS_DIR}/job_123456.json",
        f"{final_stage_dir}/CONTCAR",
        f"{final_stage_dir}/OUTCAR",
        f"{final_stage_dir}/vasprun.xml",
    ]
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == final_stage_dir
    assert result["files"]["vasprun"] == f"{final_stage_dir}/vasprun.xml"


def test_resumed_dos_uses_final_stage_directory_from_job_state():
    dos_state = {
        "run_dir": RUN_DIR,
        "submission_spec": {
            **submission_spec,
            "flow_spec": {
                "calculation_spec": {
                    "purpose": "dos",
                    "theory": "pbe",
                    "modifiers": [],
                },
                "workflow": "dos",
                "potcar_functional": "PBE_64",
            },
        },
    }
    runner = ResultsRunner(state_payload=dos_state)
    result = load_results_for_completed_job(
        monitoring_success,
        runner_factory=lambda: runner,
        parser=fake_dos_parser,
    )

    final_stage_dir = f"{RUN_DIR}/stage_03"
    assert runner.checked_paths == [
        f"{DEFAULT_LOGS_DIR}/job_123456.json",
        f"{final_stage_dir}/CONTCAR",
        f"{final_stage_dir}/OUTCAR",
        f"{final_stage_dir}/vasprun.xml",
        f"{final_stage_dir}/DOSCAR",
    ]
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == final_stage_dir
    assert result["files"]["doscar"] == f"{final_stage_dir}/DOSCAR"
    assert result["dos"]["doscar"] == f"{final_stage_dir}/DOSCAR"


def test_resumed_band_structure_uses_final_stage_directory_from_job_state():
    band_state = {
        "run_dir": RUN_DIR,
        "submission_spec": {
            **submission_spec,
            "flow_spec": {
                "calculation_spec": {
                    "purpose": "band_structure",
                    "theory": "pbe",
                    "modifiers": [],
                },
                "workflow": "band_structure",
                "potcar_functional": "PBE_64",
            },
        },
    }
    runner = ResultsRunner(state_payload=band_state)
    result = load_results_for_completed_job(
        monitoring_success,
        runner_factory=lambda: runner,
        parser=fake_band_parser,
    )

    final_stage_dir = f"{RUN_DIR}/stage_03"
    assert runner.checked_paths == [
        f"{DEFAULT_LOGS_DIR}/job_123456.json",
        f"{final_stage_dir}/CONTCAR",
        f"{final_stage_dir}/OUTCAR",
        f"{final_stage_dir}/vasprun.xml",
        f"{final_stage_dir}/KPOINTS",
        # The preceding PBE static's OUTCAR: the band structure's Fermi reference.
        f"{RUN_DIR}/stage_02/OUTCAR",
    ]
    assert result["run_dir"] == RUN_DIR
    assert result["workdir"] == final_stage_dir
    assert result["files"]["kpoints"] == f"{final_stage_dir}/KPOINTS"
    assert result["band_structure"]["kpoints"] == f"{final_stage_dir}/KPOINTS"


def test_results_are_skipped_until_monitoring_reports_success():
    class UnusedRunner(ResultsRunner):
        def connect(self, profile):
            raise AssertionError("Results should not connect before successful completion")

    result = load_results_for_completed_job(
        {**monitoring_success, "summary": "RUNNING", "slurm_state": "RUNNING"},
        runner_factory=UnusedRunner,
        parser=fake_parser,
    )

    assert result is None


def test_completed_results_are_cached_without_repeated_download_or_parse():
    clear_results_cache()
    parse_calls = []

    def counting_parser(files, monitoring_result):
        parse_calls.append(monitoring_result["job_id"])
        return fake_parser(files, monitoring_result)

    first_runner = ResultsRunner()
    first = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=lambda: first_runner,
        parser=counting_parser,
        cache=True,
    )
    first["final_formula"] = "mutated"

    class UnusedRunner(ResultsRunner):
        def connect(self, profile):
            raise AssertionError("Cached submitted results should not reconnect")

    second = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=UnusedRunner,
        parser=counting_parser,
        cache=True,
    )

    assert first_runner.read_paths == [
        f"{RUN_DIR}/CONTCAR",
        f"{RUN_DIR}/OUTCAR",
        f"{RUN_DIR}/vasprun.xml",
    ]
    assert parse_calls == ["123456"]
    assert second["final_formula"] == "TiO2"
    assert second["files"]["vasprun"] == f"{RUN_DIR}/vasprun.xml"

    resumed = load_results_for_completed_job(
        monitoring_success,
        runner_factory=UnusedRunner,
        parser=counting_parser,
        cache=True,
    )

    assert parse_calls == ["123456"]
    assert resumed["final_formula"] == "TiO2"


def test_completed_results_cache_is_bounded():
    clear_results_cache()

    def generic_parser(files, monitoring_result):
        return {
            "status": "success",
            "completion_status": "COMPLETED (ExitCode 0:0)",
            "final_formula": f"Si{monitoring_result['job_id']}",
            "diagnostics": {
                "parser": "fake",
                "outcar_parsed": True,
                "outcar_error": "",
            },
            "viewer": {"format": "cif", "source": "CONTCAR", "cif": "data_Si\n"},
        }

    max_entries = results_cache_info()["max_entries"]
    for index in range(max_entries + 2):
        run_dir = f"{RUN_DIR}-{index}"
        spec = {
            **submission_spec,
            "paths": {
                **submission_spec["paths"],
                "run_dir": run_dir,
            },
        }
        result = load_results_for_completed_job(
            {
                **monitoring_success,
                "job_id": str(200000 + index),
            },
            submission_spec=spec,
            runner_factory=ResultsRunner,
            parser=generic_parser,
            cache=True,
        )
        assert result["status"] == "success"

    info = results_cache_info()
    assert info["size"] == max_entries
    assert all("200000|" not in key for key in info["keys"])


def test_remote_completed_results_parse_without_downloading_raw_vasprun():
    clear_results_cache()
    runner = RemoteParserRunner()
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "success"
    assert result["final_formula"] == "TiO2"
    assert runner.run_python_calls == 1
    assert runner.read_paths == []
    assert runner.timeout_s == 900
    assert result["diagnostics"]["parser"] == "pymatgen-remote"
    remote_metadata = result["diagnostics"]["remote_parsing"]
    assert remote_metadata["source_bytes"] == 145000000
    assert remote_metadata["compact_result_bytes"] < remote_metadata["source_bytes"]
    assert remote_metadata["remote_command_elapsed_s"] == 1.25
    assert result["files"]["vasprun"] == f"{RUN_DIR}/vasprun.xml"
    assert runner.closed is True


def test_remote_completed_results_cache_reuses_compact_payload_without_reconnecting():
    clear_results_cache()
    runner = RemoteParserRunner()
    first = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=lambda: runner,
    )
    first["final_formula"] = "mutated"

    class UnusedRunner(RemoteParserRunner):
        def connect(self, profile):
            raise AssertionError("Cached remote results should not reconnect")

    second = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=UnusedRunner,
    )

    assert runner.run_python_calls == 1
    assert second["final_formula"] == "TiO2"
    assert second["diagnostics"]["remote_parsing"]["source_bytes"] == 145000000


def json_context_from_remote_source(source: str) -> dict:
    match = re.search(r"_context = _json\.loads\((?P<literal>.+?)\)\n", source)
    assert match, "remote parser source should embed a JSON context literal"
    return json.loads(ast.literal_eval(match.group("literal")))


def assert_json_native(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            assert_json_native(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            assert_json_native(item)
        return
    raise AssertionError(f"non-JSON value leaked into remote context: {value!r}")


def test_resumed_remote_results_context_drops_remote_job_status_object():
    clear_results_cache()
    status = RemoteJobStatus(
        job_id="123456",
        state="COMPLETED",
        exit_code="0:0",
        stdout_path="/remote/slurm.out",
        workdir="/remote/workdir",
        job_name="vasp_run_static",
        raw={"summary": "SUCCESS", "brief": "123456|COMPLETED"},
    )
    workflow_submission_spec = {
        **submission_spec,
        "flow_spec": {
            "calculation_spec": CalculationSpec(Purpose.STATIC, Theory.PBE).to_dict(),
            "workflow": "static",
            "potcar_functional": "PBE_64",
        },
    }
    runner = RemoteParserRunner(
        state_payload={
            "run_dir": RUN_DIR,
            "submission_spec": workflow_submission_spec,
        }
    )
    result = load_results_for_completed_job(
        {
            **monitoring_success,
            "job_status": status,
        },
        runner_factory=lambda: runner,
    )

    assert result["status"] == "success"
    assert runner.run_python_calls == 1
    serialized_context = json_context_from_remote_source(runner.source)
    assert_json_native(serialized_context)
    assert "job_status" not in serialized_context
    assert "RemoteJobStatus" not in runner.source
    assert serialized_context["slurm_state"] == "COMPLETED"
    assert serialized_context["exit_code"] == "0:0"
    assert serialized_context["calculation_spec"] == CalculationSpec(
        Purpose.STATIC,
        Theory.PBE,
    ).to_dict()


def test_remote_result_parser_context_is_narrow_json_contract():
    status = RemoteJobStatus(
        job_id="654321",
        state="COMPLETED",
        exit_code="0:0",
        raw={"summary": "SUCCESS"},
    )
    context = remote_result_parser_context(
        {
            **monitoring_success,
            "job_status": status,
            "brief": "not used remotely",
            "workflow_spec": {
                "stages": [
                    {
                        "stage_type": "static",
                        "theory": "pbe",
                        "modifiers": [],
                        "label": None,
                        "options": {},
                    }
                ],
                "label": None,
                "recipe": None,
            },
        }
    )

    assert set(context) == {"slurm_state", "exit_code", "workflow_spec"}
    assert_json_native(context)
    payload = json.dumps(context, sort_keys=True, allow_nan=False)
    assert "RemoteJobStatus" not in payload
    assert "job_status" not in payload
    assert "not used remotely" not in payload


def test_remote_result_parser_failure_does_not_download_oversized_raw_fallback():
    clear_results_cache()
    runner = RemoteParserRunner(returncode=2)
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Results Parsing"
    assert "remote parser failed" in result["reason"]
    assert runner.read_paths == []
    assert runner.closed is True


def test_remote_result_parser_timeout_closes_connection_without_raw_fallback():
    clear_results_cache()
    runner = RemoteParserRunner(raise_exc=TimeoutError("remote parser timed out"))
    result = load_results_for_completed_job(
        monitoring_success,
        submission_spec=submission_spec,
        runner_factory=lambda: runner,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Results Parsing"
    assert "remote parser timed out" in result["reason"]
    assert runner.read_paths == []
    assert runner.closed is True


def test_results_report_missing_required_output_file():
    runner = ResultsRunner(missing="vasprun")
    result = load_results_for_completed_job(
        monitoring_success,
        runner_factory=lambda: runner,
        parser=fake_parser,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Results Discovery"
    assert "vasprun.xml" in result["reason"]
    assert runner.closed is True


def test_results_report_missing_bmd_job_state_for_resume():
    runner = ResultsRunner(missing="state")
    result = load_results_for_completed_job(
        monitoring_success,
        runner_factory=lambda: runner,
        parser=fake_parser,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "Results Discovery"
    assert "canonical BMD run directory" in result["reason"]
    assert runner.checked_paths == [f"{DEFAULT_LOGS_DIR}/job_123456.json"]
    assert runner.closed is True


class FakeComposition:
    reduced_formula = "Si"


class FakeStructure:
    composition = FakeComposition()

    def __len__(self):
        return 2

    def to(self, fmt):
        assert fmt == "cif"
        return "data_Si\n"


class FakeSpin:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return self.name


class FakeTotalDos:
    energies = [-1.0, 0.0, 1.0]
    efermi = 0.0
    densities = {
        FakeSpin("up", 1): [0.0, 1.0, 0.0],
    }


class FakeKpoint:
    def __init__(self, label=None):
        self.label = label


class FakeBandStructure:
    efermi = 0.0
    distance = [0.0, 1.0, 2.0]
    kpoints = [
        FakeKpoint("\\Gamma"),
        FakeKpoint(),
        FakeKpoint("X"),
    ]
    bands = {
        FakeSpin("up", 1): [
            [-1.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
    }

    def get_band_gap(self):
        return {
            "energy": 1.0,
            "direct": True,
        }

    def is_metal(self):
        return False


class FakeVasprun:
    calls = []
    band_calls = []

    def __init__(self, path, **kwargs):
        self.calls.append({"path": path, "kwargs": kwargs})
        self.final_structure = FakeStructure()
        self.final_energy = -4.0
        self.ionic_steps = [1]
        self.converged_electronic = True
        self.converged_ionic = False
        self.tdos = FakeTotalDos()
        self.efermi = None if kwargs.get("parse_dos") is False else 0.0

    def get_band_structure(self, **kwargs):
        self.band_calls.append(kwargs)
        if self.efermi is None:
            raise ValueError("e_fermi is None.")
        return FakeBandStructure()


class FakeOutcar:
    def __init__(self, path):
        self.path = path


@contextmanager
def fake_pymatgen_results_modules():
    modules = {
        "pymatgen": types.ModuleType("pymatgen"),
        "pymatgen.core": types.ModuleType("pymatgen.core"),
        "pymatgen.io": types.ModuleType("pymatgen.io"),
        "pymatgen.io.vasp": types.ModuleType("pymatgen.io.vasp"),
        "pymatgen.io.vasp.outputs": types.ModuleType("pymatgen.io.vasp.outputs"),
    }
    modules["pymatgen.core"].Structure = types.SimpleNamespace(
        from_file=lambda path: FakeStructure(),
    )
    modules["pymatgen.io.vasp.outputs"].Outcar = FakeOutcar
    modules["pymatgen.io.vasp.outputs"].Vasprun = FakeVasprun

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def fixture_result_files(tmp_path, keys=("contcar", "outcar", "vasprun")):
    filenames = {
        "contcar": "CONTCAR",
        "outcar": "OUTCAR",
        "vasprun": "vasprun.xml",
        "doscar": "DOSCAR",
        "kpoints": "KPOINTS",
    }
    paths = {}
    files = {}
    for key in keys:
        path = tmp_path / filenames[key]
        path.write_text(f"{key} fixture\n", encoding="utf-8")
        paths[key] = str(path)
        files[key] = {
            "path": str(path),
            "text": path.read_text(encoding="utf-8"),
        }
    return paths, files


def execute_remote_parser_source(paths, context):
    source = remote_result_parser_source(paths, context)
    stream = io.StringIO()
    with redirect_stdout(stream):
        exec(source, {"__name__": "__main__"})
    stdout = stream.getvalue()
    start = stdout.index(REMOTE_RESULT_JSON_START) + len(REMOTE_RESULT_JSON_START)
    end = stdout.index(REMOTE_RESULT_JSON_END, start)
    return json.loads(stdout[start:end].strip())


def comparable_result_payload(result):
    payload = deepcopy(result)
    diagnostics = payload.get("diagnostics") or {}
    diagnostics.pop("parser", None)
    diagnostics.pop("remote_parsing", None)
    return payload


def assert_remote_parser_matches_local_parser(tmp_path, context, keys):
    paths, files = fixture_result_files(tmp_path, keys)
    with fake_pymatgen_results_modules():
        local_result = parse_vasp_result_files(files, context)
    with fake_pymatgen_results_modules():
        remote_result = execute_remote_parser_source(paths, context)

    assert remote_result["diagnostics"]["parser"] == "pymatgen-remote"
    assert remote_result["diagnostics"]["remote_parsing"]["source_bytes"] > 0
    assert comparable_result_payload(remote_result) == comparable_result_payload(local_result)


def test_remote_result_parser_matches_local_static_parser(tmp_path):
    context = {
        **monitoring_success,
        "calculation_spec": CalculationSpec(Purpose.STATIC, Theory.PBE).to_dict(),
    }

    assert_remote_parser_matches_local_parser(
        tmp_path,
        context,
        ("contcar", "outcar", "vasprun"),
    )


def test_remote_result_parser_matches_local_dos_parser_and_plot_arrays(tmp_path):
    context = {
        **monitoring_success,
        "calculation_spec": CalculationSpec(Purpose.DOS, Theory.PBE).to_dict(),
    }

    assert_remote_parser_matches_local_parser(
        tmp_path,
        context,
        ("contcar", "outcar", "vasprun", "doscar"),
    )


def test_remote_result_parser_matches_local_band_parser_and_plot_arrays(tmp_path):
    context = {
        **monitoring_success,
        "calculation_spec": CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE).to_dict(),
    }

    assert_remote_parser_matches_local_parser(
        tmp_path,
        context,
        ("contcar", "outcar", "vasprun", "kpoints"),
    )


def test_remote_result_parser_preserves_soc_static_convergence_semantics(tmp_path):
    context = {
        **monitoring_success,
        "calculation_spec": CalculationSpec(Purpose.STATIC, Theory.PBE).to_dict(),
    }
    context["calculation_spec"]["modifiers"] = ["soc"]

    paths, _files = fixture_result_files(tmp_path, ("contcar", "outcar", "vasprun"))
    with fake_pymatgen_results_modules():
        result = execute_remote_parser_source(paths, context)

    assert result["status"] == "success"
    assert result["final_formula"] == "Si"
    assert result["electronic_convergence"] is True
    assert result["converged_electronic"] is True
    assert result["ionic_convergence"] is None
    assert result["converged_ionic"] is None


def test_parse_vasp_result_files_uses_vasprun_dos_for_dos_workflow():
    FakeVasprun.calls = []
    files = {
        "contcar": {"path": "/remote/stage_03/CONTCAR", "text": "contcar"},
        "outcar": {"path": "/remote/stage_03/OUTCAR", "text": "outcar"},
        "vasprun": {"path": "/remote/stage_03/vasprun.xml", "text": "<modeling />"},
        "doscar": {"path": "/remote/stage_03/DOSCAR", "text": "doscar"},
    }
    context = {
        **monitoring_success,
        "calculation_spec": CalculationSpec(Purpose.DOS, Theory.PBE).to_dict(),
    }

    with fake_pymatgen_results_modules():
        result = parse_vasp_result_files(files, context)

    assert FakeVasprun.calls
    assert FakeVasprun.calls[-1]["kwargs"]["parse_dos"] is True
    assert result["final_energy_ev"] == -4.0
    assert result["energy_per_atom_ev"] == -2.0
    assert result["electronic_convergence"] is True
    assert result["converged_electronic"] is True
    assert result["ionic_convergence"] is None
    assert result["converged_ionic"] is None
    assert result["viewer"]["cif"] == "data_Si\n"
    assert result["dos"]["available"] is True
    assert result["dos"]["source"] == "vasprun.xml"
    assert result["dos"]["doscar"] == "/remote/stage_03/DOSCAR"
    assert result["visualizations"][0]["id"] == "dos"
    assert result["visualizations"][0]["download_filename"] == "density_of_states.png"
    assert result["visualizations"][0]["plot"]["x"] == [-1.0, 0.0, 1.0]


def test_parse_vasp_result_files_uses_vasprun_band_structure_for_band_workflow():
    FakeVasprun.calls = []
    FakeVasprun.band_calls = []
    files = {
        "contcar": {"path": "/remote/stage_03/CONTCAR", "text": "contcar"},
        "outcar": {"path": "/remote/stage_03/OUTCAR", "text": "outcar"},
        "vasprun": {"path": "/remote/stage_03/vasprun.xml", "text": "<modeling />"},
        "kpoints": {"path": "/remote/stage_03/KPOINTS", "text": "line-mode"},
    }
    context = {
        **monitoring_success,
        "calculation_spec": CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE).to_dict(),
    }

    with fake_pymatgen_results_modules():
        result = parse_vasp_result_files(files, context)

    assert FakeVasprun.calls
    vasprun_kwargs = FakeVasprun.calls[-1]["kwargs"]
    assert "parse_dos" not in vasprun_kwargs
    assert "parse_eigenvalues" not in vasprun_kwargs
    assert vasprun_kwargs["exception_on_bad_xml"] is False
    assert vasprun_kwargs["parse_potcar_file"] is False
    assert FakeVasprun.band_calls
    assert FakeVasprun.band_calls[-1]["line_mode"] is True
    assert FakeVasprun.band_calls[-1]["kpoints_filename"].endswith("KPOINTS")
    assert "efermi" not in FakeVasprun.band_calls[-1]
    assert result["final_energy_ev"] == -4.0
    assert result["energy_per_atom_ev"] == -2.0
    assert result["ionic_convergence"] is None
    assert result["converged_ionic"] is None
    assert result["viewer"]["cif"] == "data_Si\n"
    assert result["band_structure"]["available"] is True
    assert result["band_structure"]["source"] == "vasprun.xml"
    assert result["band_structure"]["kpoints"] == "/remote/stage_03/KPOINTS"
    assert result["band_structure"]["band_gap_ev"] == 1.0
    assert result["visualizations"][0]["id"] == "band_structure"
    assert result["visualizations"][0]["download_filename"] == "band_structure.png"
    assert result["visualizations"][0]["plot"]["x"] == [0.0, 1.0, 2.0]
    assert result["visualizations"][0]["plot"]["yaxis_range"] == [-10, 10]
    assert result["visualizations"][0]["plot"]["ticktext"] == ["Γ", "X"]
    assert result["visualizations"][0]["plot"]["traces"][0]["name"] == "Bands"
    assert result["visualizations"][0]["plot"]["traces"][0]["showlegend"] is False


def test_parse_vasp_result_files_carries_ionic_convergence_for_relaxation_results():
    files = {
        "contcar": {"path": "/remote/CONTCAR", "text": "contcar"},
        "outcar": {"path": "/remote/OUTCAR", "text": "outcar"},
        "vasprun": {"path": "/remote/vasprun.xml", "text": "<modeling />"},
    }
    context = {
        **monitoring_success,
        "calculation_spec": CalculationSpec(Purpose.RELAX, Theory.PBE).to_dict(),
    }

    with fake_pymatgen_results_modules():
        result = parse_vasp_result_files(files, context)

    assert result["electronic_convergence"] is True
    assert result["converged_electronic"] is True
    assert result["ionic_convergence"] is False
    assert result["converged_ionic"] is False


if __name__ == "__main__":
    test_monitoring_success_detection_requires_completed_success()
    test_results_load_for_submitted_completed_job_uses_submission_profile()
    test_results_for_double_relax_use_final_stage_directory()
    test_results_for_relax_static_use_final_stage_directory()
    test_results_for_dos_use_final_stage_directory_and_doscar()
    test_results_for_band_structure_use_final_stage_directory_and_kpoints()
    test_results_load_for_resumed_completed_job_uses_default_profile()
    test_resumed_double_relax_uses_final_stage_directory_from_job_state()
    test_resumed_dos_uses_final_stage_directory_from_job_state()
    test_resumed_band_structure_uses_final_stage_directory_from_job_state()
    test_results_are_skipped_until_monitoring_reports_success()
    test_completed_results_are_cached_without_repeated_download_or_parse()
    test_completed_results_cache_is_bounded()
    test_results_report_missing_required_output_file()
    test_results_report_missing_bmd_job_state_for_resume()
    test_parse_vasp_result_files_uses_vasprun_dos_for_dos_workflow()
    test_parse_vasp_result_files_uses_vasprun_band_structure_for_band_workflow()
    test_parse_vasp_result_files_carries_ionic_convergence_for_relaxation_results()
    print("results smoke test passed")
