"""The validation kit's ``execution`` profile and ``execution-evidence`` command (scientific execution acceptance).

The kit drives the machine API in-process against the fake POWER and SLURM of
the staging tests; the finished job is a fake run directory
(``uspex_execution_fakes``), the collector runs through a fake ``ssh`` with its
POWER locations mapped into the fake tree, and the evidence is fetched back
with ``tar`` and analysed. Nothing contacts POWER and no job runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from compute_api import structure_search_monitoring as monitoring
from test_machine_api_execution import power, state_dir, tokens  # noqa: F401 - pytest fixtures
from test_machine_api_structure_search_attempts import search_record
from test_machine_api_structure_search_monitoring import end
from test_machine_api_structure_search_staging import SEARCH_LOGS, WORKSPACE, cluster  # noqa: F401
from test_uspex_acceptance_kit import acceptance_root, attempt_of, step
from test_uspex_staging_validation_kit import (  # noqa: F401 - pytest fixtures
    EVIDENCE_PARENT,
    clean_env,
    kit,
    staged_validation,
)
from uspex_execution_fakes import Scenario, build_run, install_slurm

EXECUTION_SEARCH = {"min_atoms": 2, "max_atoms": 2, "num_generations": 2, "walltime": "04:00:00"}


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    monkeypatch.setattr(monitoring, "MIN_OBSERVATION_INTERVAL_SECONDS", 0)


@pytest.fixture
def test_digest(kit, tmp_path, cluster, clean_env, monkeypatch):
    """The fake environment's digest of the approved search (its POTCAR library differs from POWER's)."""

    from backend.structure_search.planning import StructureSearchPlanRequest, plan_structure_search

    digest = plan_structure_search(StructureSearchPlanRequest(formula="Si", **EXECUTION_SEARCH)).digest
    monkeypatch.setattr(kit, "EXECUTION_PLAN_DIGEST", digest)
    return digest


def execution_root(kit, tmp_path, cluster, clean_env, name="execution"):
    root = tmp_path / name
    kit.setup(root, execution=True)
    step(kit, clean_env, "run", root, cluster)
    step(kit, clean_env, "promote", root, cluster)
    return root


def submit(kit, clean_env, root, cluster, *, walltime="04:00:00", attempt=None):
    return step(kit, clean_env, "submit", root, cluster, confirm_attempt=attempt or attempt_of(root),
                confirm_walltime=walltime)


# --- the profile -------------------------------------------------------------------------------------------


def test_the_pinned_digest_is_the_planners_for_the_approved_search(kit):
    """In a clean process the planner gives exactly the pinned digest (the real POTCAR identities)."""

    code = ("from backend.structure_search.planning import StructureSearchPlanRequest as R, plan_structure_search as p;"
            "import json; r = p(R(formula='Si', min_atoms=2, max_atoms=2, num_generations=2, walltime='04:00:00'));"
            "print(json.dumps([r.digest, r.plan.record['resources']['sbatch_directives'],"
            " r.plan.record['runtime']['launch'], r.plan.record['search']['spec']]))")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("BMD_")}
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True,
                            cwd=Path(kit.REPO), env=environment)
    digest, directives, launch, spec = json.loads(result.stdout.strip().splitlines()[-1])
    assert digest == kit.EXECUTION_PLAN_DIGEST == "sha256:1989c888f58012590920ebbf4b5ca31c2d5aa0dd53449cc6bcbe3951b9b4898e"
    assert tuple(directives) == kit.EXECUTION_DIRECTIVES
    assert {key: launch[key] for key in kit.EXECUTION_LAUNCH} == kit.EXECUTION_LAUNCH
    assert {key: spec[key] for key in kit.EXECUTION_SPEC} == kit.EXECUTION_SPEC


def test_setup_records_the_execution_profile(kit, tmp_path, test_digest):
    kit.setup(tmp_path / "root", execution=True)
    profile = kit.acceptance_profile(tmp_path / "root")
    assert profile == {"profile": "execution", "walltime": "04:00:00", "search": EXECUTION_SEARCH,
                       "plan_digest": test_digest}
    assert kit.search_body(tmp_path / "root") == {"formula": "Si", "search": EXECUTION_SEARCH}
    with pytest.raises(SystemExit, match="choose one profile"):
        kit.setup(tmp_path / "both", execution=True, acceptance_walltime="00:15:00")
    marker = json.loads((tmp_path / "root" / "VALIDATION_ROOT.json").read_text())
    marker["acceptance"]["search"]["num_generations"] = 3
    (tmp_path / "root" / "VALIDATION_ROOT.json").write_text(json.dumps(marker))
    with pytest.raises(SystemExit, match="not the approved one"):
        kit.acceptance_profile(tmp_path / "root")


def test_run_refuses_a_plan_that_is_not_the_pinned_one(kit, tmp_path, cluster, clean_env, test_digest,
                                                      monkeypatch):
    root = tmp_path / "execution"
    kit.setup(root, execution=True)
    monkeypatch.setattr(kit, "EXECUTION_PLAN_DIGEST", "sha256:" + "0" * 64)
    marker = json.loads((root / "VALIDATION_ROOT.json").read_text())
    marker["acceptance"]["plan_digest"] = "sha256:" + "0" * 64
    (root / "VALIDATION_ROOT.json").write_text(json.dumps(marker))
    with pytest.raises(SystemExit, match="not the approved"):
        step(kit, clean_env, "run", root, cluster)
    assert not (root / "state" / "structure_search" / "attempts").exists() or not list(
        (root / "state" / "structure_search" / "attempts").iterdir())


def test_the_execution_submission_is_the_frozen_four_hour_search(kit, tmp_path, cluster, clean_env, test_digest):
    root = execution_root(kit, tmp_path, cluster, clean_env)
    with pytest.raises(SystemExit, match="--confirm-walltime must be exactly 04:00:00"):
        submit(kit, clean_env, root, cluster, walltime=None)
    with pytest.raises(SystemExit, match="--confirm-walltime must be exactly 04:00:00"):
        submit(kit, clean_env, root, cluster, walltime="00:15:00")
    assert cluster.sbatch_runs == 0
    job = submit(kit, clean_env, root, cluster)
    assert job["profile"] == "execution" and job["walltime"] == "04:00:00" and cluster.sbatch_runs == 1
    directives = cluster.jobs["9001"]["directives"]
    assert (directives["time"], directives["ntasks"], directives["mem"]) == ("04:00:00", "192", "0")
    [check] = sorted((root / "evidence").glob("*-execution_allocation_check.json"))
    recorded = json.loads(check.read_text())
    assert recorded["plan_digest"] == test_digest
    assert recorded["execution"]["spec"] == {"min_atoms": 2, "max_atoms": 2, "num_generations": 2,
                                             "population_size": 10, "initial_population_size": 10,
                                             "stop_criterion": 2, "walltime": "04:00:00"}
    assert recorded["execution"]["launch"] == {"mode": "srun_steps", "parallel_relaxations": 4,
                                               "ranks_per_relaxation": 48}
    assert len(recorded["execution"]["incar_sha256"]) == 5
    with pytest.raises(SystemExit, match="already sent its one submission request"):
        submit(kit, clean_env, root, cluster)
    assert cluster.sbatch_runs == 1


@pytest.mark.parametrize("origin", ["operational", "default"])
def test_attempts_from_other_profiles_are_never_submitted_as_the_execution_test(kit, tmp_path, cluster, clean_env,
                                                                               test_digest, origin):
    if origin == "operational":
        root = acceptance_root(kit, tmp_path, cluster, clean_env)  # the 15-minute profile
    else:
        root = tmp_path / "default"
        staged_validation(kit, root, cluster, clean_env)
        step(kit, clean_env, "promote", root, cluster)
    marker = json.loads((root / "VALIDATION_ROOT.json").read_text())
    marker["acceptance"] = {"profile": "execution", "walltime": "04:00:00", "search": EXECUTION_SEARCH,
                            "plan_digest": test_digest}
    (root / "VALIDATION_ROOT.json").write_text(json.dumps(marker))
    with pytest.raises(SystemExit, match="not the approved execution plan"):
        submit(kit, clean_env, root, cluster)
    assert cluster.sbatch_runs == 0 and not (root / "evidence" / "submission_requested.json").exists()


def test_a_changed_job_script_or_acknowledgement_stops_before_sbatch(kit, tmp_path, cluster, clean_env,
                                                                    test_digest, monkeypatch):
    root = execution_root(kit, tmp_path, cluster, clean_env)
    real = kit.rebuilt_job_script
    monkeypatch.setattr(kit, "rebuilt_job_script",
                        lambda state, attempt: real(state, attempt).replace("#SBATCH --ntasks=192",
                                                                            "#SBATCH --ntasks=96"))
    with pytest.raises(SystemExit, match="resource directives|not the promoted one"):
        submit(kit, clean_env, root, cluster)
    monkeypatch.setattr(kit, "rebuilt_job_script", real)
    path = root / "state" / "structure_search" / "attempts" / f"{attempt_of(root)}.json"
    record = json.loads(path.read_text())
    record["promotion"]["acknowledgement"]["resources"]["sbatch_directives"][3] = "#SBATCH --time=168:00:00"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        submit(kit, clean_env, root, cluster)
    assert cluster.sbatch_runs == 0


def test_the_execution_root_keeps_the_cancellation_gate(kit, tmp_path, cluster, clean_env, test_digest):
    root = execution_root(kit, tmp_path, cluster, clean_env)
    submit(kit, clean_env, root, cluster)
    cluster.jobs["9001"]["state"] = "RUNNING"
    with pytest.raises(SystemExit, match="must be exactly"):
        step(kit, clean_env, "cancel", root, cluster, confirm_attempt=attempt_of(root), confirm_job="9002")
    assert cluster.scancel_runs == []
    summary = step(kit, clean_env, "cancel", root, cluster, confirm_attempt=attempt_of(root), confirm_job="9001")
    assert summary["status"] == "command_sent" and cluster.scancel_runs == ["9001"]


# --- execution-evidence, end to end -----------------------------------------------------------------------


FAKE_SSH = r'''#!{python}
import json, os, subprocess, sys
args = sys.argv[1:]
i = 0
while args[i].startswith("-"):
    i += 2 if args[i] == "-o" else 1
assert args[i] == "powerslurm-bmdguest", args
command = " ".join(args[i + 1:])
script = sys.stdin.buffer.read().decode()
if script:
    for old, new in json.loads(os.environ["FAKE_SSH_MAP"]).items():
        assert old in script, old
        script = script.replace(old, new)
completed = subprocess.run(["bash", "-c", command], input=script.encode(), capture_output=True)
sys.stdout.buffer.write(completed.stdout)
sys.stderr.buffer.write(completed.stderr)
sys.exit(completed.returncode)
'''


@pytest.fixture
def finished_root(kit, tmp_path, cluster, clean_env, test_digest, monkeypatch):
    """An execution root whose job ran to its natural end (a fake run), observed to SEARCH_COMPLETED."""

    def make(scenario=None):
        root = execution_root(kit, tmp_path, cluster, clean_env)
        submit(kit, clean_env, root, cluster)
        attempt = attempt_of(root)
        record = search_record(root / "state", attempt)
        run_dir_on_power = record["staging"]["spec"]["paths"]["run_dir"]
        run = cluster.at(run_dir_on_power)
        incars = {p.name: p.read_text() for p in sorted((run / "search" / "Specific").glob("INCAR_*"))}
        built = build_run(run, identity_run_dir=run_dir_on_power, attempt=attempt, job="9001",
                          plan_digest=record["plan_digest"], scenario=scenario or Scenario(), incars=incars,
                          job_script=(run / "bmd_search_job.sbatch").read_text(),
                          acknowledgement=json.loads((run / "resource_acknowledgement.json").read_text()))
        cluster.jobs["9001"]["start"] = cluster.jobs["9001"]["submit"]
        end(cluster, root / "state", attempt, "9001", (scenario or Scenario()).slurm_state)
        observed = step(kit, clean_env, "observe", root, cluster)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        ssh = bin_dir / "ssh"
        ssh.write_text(FAKE_SSH.replace("{python}", sys.executable))
        ssh.chmod(0o755)
        install_slurm(bin_dir, tmp_path / "fake_slurm_execution.json", job="9001", sacct=built["sacct"])
        cluster.at(EVIDENCE_PARENT).mkdir(parents=True, exist_ok=True)
        mapping = {"WORKSPACE=/bmd/bmdguest/searches": f"WORKSPACE={cluster.at(WORKSPACE)}",
                   "STATE_DIR=/bmd-db/guest/logs/structure_search/submission_attempts":
                       f"STATE_DIR={cluster.at(SEARCH_LOGS)}/submission_attempts",
                   "PY=/bmd/bmdguest/envs/atomate2_remote/bin/python": f"PY={sys.executable}",
                   f"EVIDENCE_PARENT={EVIDENCE_PARENT}": f"EVIDENCE_PARENT={cluster.at(EVIDENCE_PARENT)}"}
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("FAKE_SSH_MAP", json.dumps(mapping))
        monkeypatch.setenv("FAKE_SLURM_EXECUTION", str(tmp_path / "fake_slurm_execution.json"))
        monkeypatch.setattr(kit, "POWER_EVIDENCE_PARENT", str(cluster.at(EVIDENCE_PARENT)))
        return root, observed

    return make


def test_execution_evidence_end_to_end(kit, finished_root):
    root, observed = finished_root()
    assert observed["state"] == "SEARCH_COMPLETED" and observed["terminal"]["state"] == "SEARCH_COMPLETED"
    assert kit.execution_evidence(root, proc=root / "no-proc") == 0
    report = json.loads((root / "execution_analysis_01.json").read_text())
    assert report["result"] == "PASSED" and report["classification"] == "natural_completion", report["checks"]
    assert report["findings"]["diamond"]["found"] is True and report["findings"]["diamond"]["required"] is False
    evidence = root / "execution_evidence_01"
    assert (evidence / "files" / "search" / "results1" / "Individuals").is_file()
    assert not [p for p in evidence.rglob("*") if p.name.startswith(("POTCAR", "OUTCAR", "WAVECAR", "vasprun"))]
    target = kit.archive(root)  # also checks every member for tokens and POTCAR text
    with tarfile.open(target) as tar:
        names = tar.getnames()
    assert any(name.endswith("execution_analysis_01.json") for name in names)
    assert any(name.endswith("execution_evidence_01/step_logs.json") for name in names)


def test_execution_evidence_reports_a_failed_analysis(kit, finished_root):
    root, _ = finished_root(Scenario(energy_offsets={5: 0.05}))
    assert kit.execution_evidence(root, proc=root / "no-proc") == 1
    report = json.loads((root / "execution_analysis_01.json").read_text())
    assert report["result"] == "FAILED" and "listed_energies" in report["failed"]


def test_execution_evidence_needs_an_ended_execution_attempt(kit, tmp_path, cluster, clean_env, test_digest):
    root = execution_root(kit, tmp_path, cluster, clean_env)
    with pytest.raises(SystemExit, match="not one of"):
        kit.execution_evidence(root, proc=root / "no-proc")
    (tmp_path / "other").mkdir()
    operational = acceptance_root(kit, tmp_path / "other", cluster, clean_env)
    with pytest.raises(SystemExit, match="not set up for the execution acceptance"):
        kit.execution_evidence(operational, proc=operational / "no-proc")


def test_fetch_refuses_an_unexpected_evidence_directory(kit, tmp_path):
    for remote in ("/etc", "/bmd-db/guest/validation/../logs", "/bmd-db/guest/validation/uspex_staging_4c_x_y",
                   "/bmd/bmdguest/searches/uspex_execution_4d_20261120T100000Z_c0ffee00_1"):
        with pytest.raises(SystemExit, match="unexpected evidence directory"):
            kit.fetch_power_directory(remote, tmp_path / "target")
