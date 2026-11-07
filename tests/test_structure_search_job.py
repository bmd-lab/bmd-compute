"""USPEX step 4D-1: the structure-search job script, job entry point and state model.

The end-to-end tests run the generated batch script itself with ``bash``, as
SLURM would, against fakes: an ``environment-modules``-style ``module``
function, ``srun`` (the bridge tests' fake, which writes VASP outputs the real
candidate-validity check accepts), a Singularity executable that runs the
bridge tests' fake USPEX master, and a fake USPEX installation for the job's
runtime check. The real pieces are the script, the embedded runtime-package
guard, ``job_main``, the input verification, the runtime check, the master
command, the launch loop, ``bmd_bridge.sh`` and the step-validity check.

Paths: the USPEX container sees only its bind path (``/bmd`` on POWER). Here
the runtime package uploaded with the attempt is the real one except that
``bridge.master.DEFAULT_BIND`` names the test's temporary bind directory; the
guard verifies it against its own manifest, as on POWER.

Nothing contacts POWER or SLURM, and nothing is submitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from backend.structure_search import job as search_job
from backend.structure_search import lifecycle
from backend.structure_search.job import JobDefinitionError
from conftest import synthetic_enmax_metadata, use_enmax_metadata

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "structure_search_bridge"
PAYLOAD_SHA = "fc4a08e65ffc5f4fc3b45cb058de2cd54731ab2c191123aac8a3c69e87e95c8c"
MODULES = ["intel/test-oneAPI", "vasp/test-6.4.1"]
TIMESTAMP = "2026-11-08T12:00:00Z"

# A parseable, clearly labelled TEST-ONLY dataset: not from any POTCAR library.
TEST_POTCAR = """  PAW_PBE Si TEST-ONLY
 4.00000000000000
 parameters from PSCTR are:
   VRHFIN =Si: s2p2
   LEXCH  = PE
   TITEL  = PAW_PBE Si TEST-ONLY 08Nov2026
   POMASS =   28.085; ZVAL   =    4.000    mass and valenz
   ENMAX  =  245.345; ENMIN  =  184.009 eV
 END of PSCTR-controll parameters
 local part
  0.12345678E+01  0.23456789E+01  0.34567890E+01  0.45678901E+01  0.56789012E+01
  0.67890123E+01  0.78901234E+01  0.89012345E+01  0.90123456E+01  0.11111111E+01
 End of Dataset
"""

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="the job script is a bash script")


def _identity(text: str) -> dict:
    import warnings

    from backend.structure_search.search_plan import potcar_identity_from_text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return potcar_identity_from_text(text)


# --- the staged search a job runs -----------------------------------------------------------


class JobRig:
    """One staged Si search under a temporary bind path, with fake POWER tools."""

    def __init__(self, root: Path, monkeypatch):
        from backend import config
        from backend.runtime_package import runtime_package_manifest_for_sources
        from backend.structure_search.bridge import master
        from backend.structure_search.planning import StructureSearchPlanRequest, plan_structure_search
        from backend.structure_search.search_plan import prepare_search_directory
        from backend.structure_search.staging import build_search_submission_spec, staging_layout
        from backend.submission import build_backend_module_sources

        self.root = root
        self.bind = root / "bmd"
        self.bin = root / "bin"
        self.records = root / "srun_records"
        for path in (self.bind / "searches", self.bin, self.records, root / "lib"):
            path.mkdir(parents=True)
        monkeypatch.setattr(master, "DEFAULT_BIND", str(self.bind))

        # A fake USPEX installation that passes the job-start runtime check.
        sandbox = self.bind / "shared" / "sandbox"
        program = sandbox / "opt" / "uspex" / "uspex"
        program.parent.mkdir(parents=True)
        program.write_bytes(b"#!/bin/sh\n# USPEX v25.0, 11 November 2025\nPAYLOAD_SHA256=" + PAYLOAD_SHA.encode()
                            + b"\nexit 0\n\0payload")
        cache = self.bind / "cache"
        unpacked = cache / "uspex_sfx" / PAYLOAD_SHA
        (unpacked / "uspex_core").mkdir(parents=True)
        (unpacked / "uspex_core" / "uspex").write_text("fake")
        (unpacked / ".ready").write_text("")
        self.ready_marker = unpacked / ".ready"
        (root / "lib" / "libpmi2.so").write_text("")
        for name, value in {
            "USPEX_SANDBOX_DIR": str(sandbox),
            "USPEX_CACHE_DIR": str(cache),
            "USPEX_PAYLOAD_SHA256": PAYLOAD_SHA,
            "SINGULARITY_EXECUTABLE": str(self.bin / "singularity"),
            "STRUCTURE_SEARCH_PMI_LIBRARY": str(root / "lib" / "libpmi2.so"),
            "STRUCTURE_SEARCH_WORKSPACE_DIR": str(self.bind / "searches"),
            "STRUCTURE_SEARCH_LOGS_DIR": str(root / "logs"),
        }.items():
            monkeypatch.setattr(config, name, value)

        # The plan records the TEST-ONLY POTCAR's identity for Si.
        metadata = synthetic_enmax_metadata()
        identity = _identity(TEST_POTCAR)
        metadata["potcars"]["Si"] = {"TITEL": identity["TITEL"], "ENMAX": identity["ENMAX"],
                                     "sha256": identity["sha256"]}
        use_enmax_metadata(monkeypatch, root / "enmax.json", metadata)
        self.plan = plan_structure_search(StructureSearchPlanRequest(formula="Si")).plan

        # The runtime package uploaded with the attempt, staged as in step 4C.
        sources = build_backend_module_sources()
        original = sources["structure_search/bridge/master.py"]
        assert original.count('DEFAULT_BIND = "/bmd"') == 1
        sources["structure_search/bridge/master.py"] = original.replace(
            'DEFAULT_BIND = "/bmd"', f'DEFAULT_BIND = "{self.bind}"')
        self.sources = sources
        self.manifest = runtime_package_manifest_for_sources(sources)
        self.attempt = str(uuid.uuid4())
        self.spec = build_search_submission_spec(plan=self.plan, attempt_id=self.attempt,
                                                 run_timestamp="20261108-120000", created_at=TIMESTAMP,
                                                 runtime_manifest=self.manifest)
        self.spec["runner"]["python"] = sys.executable
        layout = staging_layout(self.spec["run_name"])
        self.run_dir = Path(layout["run_dir"])
        self.search_dir = Path(layout["search_dir"])
        self.run_dir.mkdir()
        for name, text in sources.items():
            path = Path(layout["package_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
        prepare_search_directory(self.plan, self.search_dir, bridge_path=layout["bridge_path"])
        potcar = self.search_dir / "Specific" / "POTCAR_Si"
        potcar.write_text(TEST_POTCAR)
        potcar.chmod(0o600)

        self.acknowledgement = search_job.build_resource_acknowledgement(
            self.plan, attempt_id=self.attempt, principal="lee", acknowledged_at=TIMESTAMP)
        self.script = self.build_script()
        self.script_path = self.run_dir / search_job.JOB_SCRIPT_NAME
        self.script_path.write_text(self.script)
        self._write_fakes(layout["bridge_path"])

    def build_script(self, **overrides) -> str:
        values = dict(spec=self.spec, plan=self.plan, acknowledgement=self.acknowledgement,
                      runtime_sources=self.sources, manifest=self.manifest, modules=MODULES)
        values.update(overrides)
        return search_job.build_job_script(**values)

    def _write_fakes(self, bridge_path: str) -> None:
        python = sys.executable
        (self.bin / "singularity").write_text(f"""#!{python}
import json, os, subprocess, sys
args = sys.argv[1:]
with open({str(self.root / 'singularity_calls.jsonl')!r}, "a") as log:
    log.write(json.dumps({{"argv": args, "env": sorted(os.environ)}}) + "\\n")
assert args[:3] == ["exec", "--cleanenv", "-B"], args
pwd = args[args.index("--pwd") + 1]
assert args[args.index("--pwd") + 3] == "/opt/uspex/uspex", args
env = {{"PATH": "/usr/bin:/bin"}}
env.update({{k[len("SINGULARITYENV_"):]: v for k, v in os.environ.items() if k.startswith("SINGULARITYENV_")}})
sys.exit(subprocess.run([{python!r}, {str(FIXTURES / 'fake_uspex.py')!r}, "--search-dir", pwd,
                         "--bridge", {bridge_path!r}, "--structures", "2", "--steps", "5", "--parallel", "2"],
                        cwd=pwd, env=env).returncode)
""")
        (self.bin / "srun").write_text(f"""#!{python}
import json, os, sys
if sys.argv[1:] == ["--version"]:
    print("slurm 23.02.7")
    sys.exit(0)
with open({str(self.root / 'srun_env.jsonl')!r}, "a") as log:
    log.write(json.dumps({{"bmd": sorted(k for k in os.environ if k.startswith("BMD_")),
                          "I_MPI_PMI_LIBRARY": os.environ.get("I_MPI_PMI_LIBRARY"),
                          "LOADEDMODULES": os.environ.get("LOADEDMODULES"),
                          "PYTHONPATH": os.environ.get("PYTHONPATH")}}) + "\\n")
os.execv({python!r}, [{python!r}, {str(self.root / 'fake_srun' / 'srun')!r}, *sys.argv[1:]])
""")
        (self.root / "fake_srun").mkdir()
        (self.root / "fake_srun" / "srun").write_text((FIXTURES / "fake_srun.py").read_text())
        (self.bin / "sbatch").write_text(f"#!/bin/sh\necho \"$@\" >> {self.root / 'sbatch_calls'}\nexit 1\n")
        for tool in ("singularity", "srun", "sbatch"):
            (self.bin / tool).chmod(0o755)
        self.set_control({"sleep": 0.05})

    def set_control(self, control: dict) -> None:
        (self.root / "control.json").write_text(json.dumps(control))

    def environment(self, **extra) -> dict:
        module = ('() { case "$1" in purge) export LOADEDMODULES=;; '
                  'load) [ "$2" = "${FAKE_MODULE_FAIL:-}" ] && return 1; '
                  'export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$2";; esac; return 0; }')
        return {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.root),
            "SLURM_JOB_ID": "4242",
            "SLURM_MEM_PER_NODE": "512000",
            "FAKE_SRUN_RECORDS": str(self.records),
            "FAKE_SRUN_CONTROL": str(self.root / "control.json"),
            "FAKE_SRUN_FIXTURES": str(FIXTURES),
            "BMD_API_STATE_DIR": "/srv/production-state",  # must never reach the job
            "BMD_SEARCH_WORKSPACE_DIR": "/elsewhere",
            "PYTHONPATH": str(self.root / "not-on-the-path"),
            "BASH_FUNC_module%%": module,
            **extra,
        }

    def fake_scontrol(self, record: str) -> None:
        """``scontrol -o show job <id>`` answering with ``record`` (what SLURM reports for the job)."""

        path = self.bin / "scontrol"
        path.write_text(f"""#!{sys.executable}
import sys
with open({str(self.root / 'scontrol_calls')!r}, "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
assert sys.argv[1:4] == ["-o", "show", "job"], sys.argv
print({record!r})
""")
        path.chmod(0o755)

    def run(self, *, timeout: float = 240, **extra) -> subprocess.CompletedProcess:
        environment = {key: value for key, value in self.environment(**extra).items() if value is not None}
        return subprocess.run(["bash", str(self.script_path)], env=environment, cwd=self.root,
                              capture_output=True, text=True, timeout=timeout)

    def status(self) -> dict:
        return json.loads((self.run_dir / search_job.STATUS_FILE).read_text())

    def singularity_calls(self) -> list[dict]:
        path = self.root / "singularity_calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def srun_env(self) -> list[dict]:
        path = self.root / "srun_env.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return JobRig(tmp_path, monkeypatch)


# --- the job script ----------------------------------------------------------------------------


def test_job_script_is_deterministic_and_carries_the_plan_directives(rig):
    assert rig.build_script() == rig.script  # same inputs, same bytes
    lines = rig.script.splitlines()
    assert lines[0] == "#!/bin/bash"
    directives = [line for line in lines if line.startswith("#SBATCH ")]
    planned = rig.plan.record["resources"]["sbatch_directives"]
    assert directives[: len(planned)] == planned  # verbatim, in order: exclusive node, mem=0, exclusions
    assert directives[len(planned):] == [
        f"#SBATCH --job-name=bmds-{rig.attempt[:8]}",
        f"#SBATCH --comment=bmd_attempt:{rig.attempt}",
        f"#SBATCH --chdir={rig.run_dir}",
        f"#SBATCH --output={rig.run_dir}/slurm-%j.out",
        "#SBATCH --no-requeue",
    ]
    assert all(line.startswith("#") for line in lines[: lines.index("set -u")])  # directives precede commands
    assert f"# plan digest:              {rig.plan.digest}" in lines
    assert f"# resource acknowledgement: sha256:{rig.acknowledgement['acknowledgement_sha256']}" in lines
    assert "# modules requested:        intel/test-oneAPI vasp/test-6.4.1" in lines
    request = search_job.request_from_job_script(rig.script)
    assert request["attempt_id"] == rig.attempt and request["plan_digest"] == rig.plan.digest
    assert request["manifest"] == rig.manifest and request["modules_requested"] == MODULES
    subprocess.run(["bash", "-n", str(rig.script_path)], check=True)
    for command in ("sbatch ", "scancel", "squeue", "sacct"):
        assert command not in rig.script  # the job never submits, cancels or queries jobs


def test_job_script_changes_with_any_bound_input(rig):
    other_modules = rig.build_script(modules=["intel/test-oneAPI", "vasp/other"])
    assert other_modules != rig.script
    other_ack = search_job.build_resource_acknowledgement(rig.plan, attempt_id=rig.attempt, principal="lee",
                                                          acknowledged_at="2026-11-08T12:00:01Z")
    assert rig.build_script(acknowledgement=other_ack) != rig.script
    assert search_job.job_script_sha256(rig.script) == hashlib.sha256(rig.script.encode()).hexdigest()


@pytest.mark.parametrize("change, code", [
    (lambda r: {"modules": []}, "invalid_modules"),
    (lambda r: {"modules": ["vasp/6.4.1; rm -rf /"]}, "invalid_modules"),
    (lambda r: {"acknowledgement": None}, "acknowledgement_required"),
    (lambda r: {"manifest": {**r.manifest, "extra.py": "0" * 64}}, "manifest_mismatch"),
    (lambda r: {"spec": {**r.spec, "runner": {**r.spec["runner"], "python": "/opt/my python/bin/python"}}},
     "unsafe_path"),
    (lambda r: {"spec": {**r.spec, "structure_search": {**r.spec["structure_search"],
                                                         "plan_digest": "sha256:" + "0" * 64}}}, "plan_mismatch"),
])
def test_job_script_refuses_inconsistent_or_unsafe_inputs(rig, change, code):
    with pytest.raises(JobDefinitionError) as refused:
        rig.build_script(**change(rig))
    assert refused.value.code == code


def _plan_with(rig, change):
    from types import SimpleNamespace

    record = rig.plan.record
    change(record)
    return SimpleNamespace(record=record, digest=rig.plan.digest)


@pytest.mark.parametrize("directive", ["#SBATCH --requeue", "#SBATCH --comment=other", "#SBATCH -o /tmp/x",
                                       "#SBATCH --chdir=/tmp", "#SBATCH --job-name=x"])
def test_plan_directives_may_not_set_what_bmd_sets(rig, directive):
    plan = _plan_with(rig, lambda record: record["resources"]["sbatch_directives"].append(directive))
    ack = search_job.build_resource_acknowledgement(plan, attempt_id=rig.attempt, principal="lee",
                                                    acknowledged_at=TIMESTAMP)
    with pytest.raises(JobDefinitionError) as refused:
        rig.build_script(plan=plan, acknowledgement=ack)
    assert refused.value.code == "invalid_directive"


# --- the resource acknowledgement ------------------------------------------------------------------


def test_acknowledgement_is_frozen_and_bound_to_attempt_plan_and_request(rig):
    ack = rig.acknowledgement
    assert ack["resources"]["status"] == "provisional_estimate" and ack["resources"]["approved"] is False
    assert ack["resources"]["request"]["exclusive"] is True
    assert ack["statement"] == search_job.ACKNOWLEDGEMENT_STATEMENT
    assert search_job.verify_resource_acknowledgement(ack, plan=rig.plan, attempt_id=rig.attempt) == \
        ack["acknowledgement_sha256"]
    for field, value in (("principal", "someone"), ("acknowledged_at", "2030-01-01T00:00:00Z"),
                         ("statement", "ok"), ("resources", {**ack["resources"], "approved": True})):
        with pytest.raises(JobDefinitionError) as refused:
            search_job.verify_resource_acknowledgement({**ack, field: value}, plan=rig.plan, attempt_id=rig.attempt)
        assert refused.value.code == "acknowledgement_invalid"
    with pytest.raises(JobDefinitionError) as refused:
        search_job.verify_resource_acknowledgement(ack, plan=rig.plan, attempt_id=str(uuid.uuid4()))
    assert refused.value.code == "acknowledgement_mismatch"
    other = _plan_with(rig, lambda record: record["resources"]["request"].update(walltime="200:00:00"))
    with pytest.raises(JobDefinitionError) as refused:
        search_job.verify_resource_acknowledgement(ack, plan=other, attempt_id=rig.attempt)
    assert refused.value.code == "acknowledgement_mismatch"


# --- running the job script with fake SLURM, modules, Singularity and USPEX -------------------------


@needs_bash
def test_job_runs_the_search_through_the_existing_launcher_bridge_and_validity_check(rig):
    staged_before = {p.name: p.read_bytes() for p in (rig.search_dir / "Specific").iterdir()}
    result = rig.run()
    assert result.returncode == 0, result.stdout + result.stderr
    status = rig.status()
    assert status["schema"] == search_job.JOB_STATUS_SCHEMA and status["attempt_id"] == rig.attempt
    assert status["slurm_job_id"] == "4242" and status["plan_digest"] == rig.plan.digest
    assert status["acknowledgement_sha256"] == rig.acknowledgement["acknowledgement_sha256"]
    assert status["modules"] == {"requested": MODULES, "loaded": MODULES}
    assert status["execution"]["outcome"] == "uspex_finished" and status["execution"]["master_exit_status"] == 0
    assert status["execution"]["launch"]["mode"] == "srun_steps"
    assert status["execution"]["launch"]["parallel_relaxations"] == 4
    assert status["execution"]["launch"]["ranks_per_relaxation"] == 48
    # SLURM_MEM_PER_NODE = 512000 MB, less 10 %, over 4 relaxations: the launcher's existing division.
    assert status["execution"]["launch"]["step_mem"] == "115200M"
    assert status["execution"]["launch"]["node_memory"]["source"] == "SLURM_MEM_PER_NODE"
    assert status["scientific_validity"]["status"] == "not_assessed"
    assert status["scientific_validity"]["step_validity_verdicts"] == {"valid": 10}  # 2 structures x 5 steps
    assert status["finalists"]["status"] == "not_started"
    assert status["finished_at"] is not None
    assert (rig.run_dir / "job_started").read_text().strip() == "4242"
    assert json.loads((rig.run_dir / "runtime_check.json").read_text())["status"] == "passed"
    assert (rig.run_dir / "uspex_master.log").exists()

    [call] = rig.singularity_calls()
    assert call["argv"][:7] == ["exec", "--cleanenv", "-B", str(rig.bind), "--pwd", str(rig.search_dir),
                                str(rig.bind / "shared" / "sandbox")]
    steps = rig.srun_env()
    assert len(steps) == 10
    assert all(step["bmd"] == [] for step in steps)  # no BMD_ setting from the submitting shell
    assert all(step["I_MPI_PMI_LIBRARY"] == str(rig.root / "lib" / "libpmi2.so") for step in steps)
    assert all(step["LOADEDMODULES"] == ":".join(MODULES) for step in steps)
    assert all(step["PYTHONPATH"] is None for step in steps)
    for calcfold in rig.search_dir.glob("CalcFold*"):
        records = [json.loads(line) for line in (calcfold / "bmd_step_validity.jsonl").read_text().splitlines()]
        assert [record["step"] for record in records] == [1, 2, 3, 4, 5]
    assert {p.name: p.read_bytes() for p in (rig.search_dir / "Specific").iterdir()} == staged_before
    assert not (rig.root / "sbatch_calls").exists()


@needs_bash
def test_a_second_or_requeued_job_never_runs_the_search_again(rig):
    assert rig.run().returncode == 0
    status = (rig.run_dir / search_job.STATUS_FILE).read_bytes()
    calls = len(rig.singularity_calls())
    again = rig.run(SLURM_JOB_ID="4243")
    assert again.returncode == search_job.EXIT_ALREADY_STARTED
    assert "is not started again" in again.stdout
    assert (rig.run_dir / search_job.STATUS_FILE).read_bytes() == status  # the first job's record is kept
    assert (rig.run_dir / "job_started").read_text().strip() == "4242"
    assert len(rig.singularity_calls()) == calls


@needs_bash
def test_changed_staged_inputs_are_refused_before_uspex_starts(rig):
    incar = rig.search_dir / "Specific" / "INCAR_4"
    incar.write_text(incar.read_text() + "EDIFFG = -1\n")
    result = rig.run()
    assert result.returncode == search_job.EXIT_REFUSED_INPUTS, result.stdout + result.stderr
    assert rig.status()["execution"]["outcome"] == "refused_inputs"
    assert rig.singularity_calls() == [] and rig.srun_env() == []


@needs_bash
def test_changed_staged_potcar_is_refused_before_uspex_starts(rig):
    potcar = rig.search_dir / "Specific" / "POTCAR_Si"
    potcar.write_text(TEST_POTCAR.replace("245.345", "250.000"))
    assert rig.run().returncode == search_job.EXIT_REFUSED_INPUTS
    assert rig.status()["execution"]["code"] == "structure_search_potcar_identity_mismatch"
    assert rig.singularity_calls() == []


@needs_bash
def test_failed_runtime_check_stops_the_job_before_uspex_starts(rig):
    rig.ready_marker.unlink()
    assert rig.run().returncode == search_job.EXIT_RUNTIME_CHECK_FAILED
    status = rig.status()
    assert status["execution"]["outcome"] == "runtime_check_failed"
    assert any(".ready is missing" in problem for problem in status["execution"]["problems"])
    assert rig.singularity_calls() == []


@needs_bash
def test_changed_runtime_package_is_refused_before_any_package_code_runs(rig):
    module = rig.run_dir / "backend" / "structure_search" / "job.py"
    module.write_text(module.read_text() + "\n# changed on POWER\n")
    result = rig.run()
    assert result.returncode == search_job.EXIT_RUNTIME_PACKAGE_REFUSED, result.stdout + result.stderr
    assert rig.status()["execution"]["outcome"] == "runtime_package_refused"
    assert rig.singularity_calls() == []


@needs_bash
def test_a_module_that_does_not_load_stops_the_job_with_a_record(rig):
    result = rig.run(FAKE_MODULE_FAIL="vasp/test-6.4.1")
    assert result.returncode == search_job.EXIT_SETUP_FAILED
    status = rig.status()
    assert status["execution"] == {"outcome": "setup_failed", "message": "module load vasp/test-6.4.1 failed"}
    assert status["attempt_id"] == rig.attempt and status["slurm_job_id"] == "4242"
    assert rig.singularity_calls() == []


@needs_bash
def test_sigterm_stops_uspex_and_the_steps_and_is_recorded(rig):
    rig.set_control({"sleep": 60})
    process = subprocess.Popen(["bash", str(rig.script_path)], env=rig.environment(), cwd=rig.root,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    deadline = time.monotonic() + 120
    while not list(rig.records.glob("*.json")):
        assert process.poll() is None, process.stdout.read()
        assert time.monotonic() < deadline
        time.sleep(0.1)
    os.killpg(process.pid, signal.SIGTERM)  # what SLURM does on scancel or at the walltime
    output, _ = process.communicate(timeout=120)
    assert process.returncode == 123, output
    status = rig.status()
    assert status["execution"]["outcome"] == "stopped" and status["execution"]["master_exit_status"] == 123
    records = [json.loads(p.read_text()) for p in rig.records.glob("*.json")]
    assert records and all(record["end"] is not None for record in records)


# --- the state model -----------------------------------------------------------------------------


def test_remote_states_never_use_calculation_state_names():
    assert not set(lifecycle.REMOTE_STATES) & set(lifecycle.CALCULATION_REMOTE_STATES)
    from backend.structure_search import staging

    assert (staging.REMOTE_STATE_STAGING, staging.REMOTE_STATE_STAGED) == (lifecycle.REMOTE_STAGING,
                                                                            lifecycle.REMOTE_STAGED)
    for state in lifecycle.CALCULATION_REMOTE_STATES:
        with pytest.raises(lifecycle.LifecycleError):
            lifecycle.require_remote_transition(lifecycle.REMOTE_STAGED, state)


@pytest.mark.parametrize("current, target", [
    (None, "STAGING"), ("STAGING", "STAGED"), ("STAGED", "SEARCH_PREPARED"),
    ("SEARCH_PREPARED", "SEARCH_PREPARED"), ("SEARCH_PREPARED", "SEARCH_SUBMITTING"),
    ("SEARCH_SUBMITTING", "SEARCH_SUBMITTED"), ("SEARCH_SUBMITTING", "SEARCH_PREPARED"),
])
def test_allowed_remote_transitions(current, target):
    assert lifecycle.require_remote_transition(current, target).target == target


@pytest.mark.parametrize("current, target", [
    ("STAGING", "SEARCH_PREPARED"),           # never promoted without completed staging
    ("STAGED", "SEARCH_SUBMITTING"),          # never submitted without fresh promotion
    ("SEARCH_SUBMITTED", "SEARCH_SUBMITTING"),  # never submitted twice
    ("SEARCH_SUBMITTED", "SEARCH_PREPARED"),
    ("SEARCH_SUBMITTING", "SEARCH_SUBMITTING"),
    ("SEARCH_SUBMITTED", "STAGED"),
    (None, "SEARCH_SUBMITTED"),
])
def test_forbidden_remote_transitions(current, target):
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.require_remote_transition(current, target)


def test_uncertain_submission_is_never_repeated_automatically():
    assert lifecycle.resubmission_allowed("SEARCH_PREPARED")
    for state in ("STAGED", "SEARCH_SUBMITTING", "SEARCH_SUBMITTED", None):
        assert not lifecycle.resubmission_allowed(state)
    with pytest.raises(lifecycle.LifecycleError, match="operator"):
        lifecycle.require_local_transition("SUBMISSION_UNCERTAIN", "PREPARED")
    assert not lifecycle.require_local_transition("SUBMISSION_UNCERTAIN", "PREPARED",
                                                  operator_confirmed=True).automatic
    assert lifecycle.require_local_transition("SUBMISSION_UNCERTAIN", "SUBMITTED").automatic
    for state in lifecycle.TERMINAL_STATES:  # terminal states are final
        for target in lifecycle.LOCAL_STATES:
            with pytest.raises(lifecycle.LifecycleError):
                lifecycle.require_local_transition(state, target, operator_confirmed=True)


def test_calculation_submission_machinery_refuses_every_search_state():
    from backend.paramiko_remote import ParamikoRemoteRunner
    from backend.remote import SubmissionAttemptError, SubmissionAttemptNotPrepared

    spec = {"submission": {"attempt_id": str(uuid.uuid4())}, "paths": {}}
    runner = ParamikoRemoteRunner(client=object())
    runner._ensure_submission_attempt_matches = lambda spec, state: None
    for state in lifecycle.REMOTE_STATES:
        with pytest.raises(SubmissionAttemptNotPrepared):
            runner._submit_prepared_submission_attempt(spec, {"state": state}, "")
    runner._require_run_directory_owned_by_attempt = lambda spec: None
    for state in lifecycle.REMOTE_STATES:
        runner._read_submission_attempt_state = lambda spec, state=state: {"state": state}
        with pytest.raises(SubmissionAttemptError):
            runner._require_preparable_submission_attempt(spec)


@pytest.mark.parametrize("scheduler, outcome, master, expected", [
    ("RUNNING", "running", None, "SUBMITTED"),
    ("PENDING", None, None, "SUBMITTED"),
    ("UNKNOWN", None, None, "SUBMITTED"),
    ("COMPLETED", "uspex_finished", 0, "SEARCH_COMPLETED"),
    ("COMPLETED", "uspex_failed", 1, "SEARCH_FAILED"),        # SLURM's COMPLETED is not success
    ("COMPLETED", "refused_inputs", None, "SEARCH_FAILED"),
    ("COMPLETED", None, None, "SEARCH_FAILED"),              # no execution record
    ("FAILED", "uspex_finished", 0, "SEARCH_FAILED"),
    ("NODE_FAIL", "running", None, "SEARCH_FAILED"),
    ("CANCELLED+", "stopped", 123, "SEARCH_CANCELLED"),
    ("TIMEOUT", "stopped", 123, "SEARCH_TIMED_OUT"),
])
def test_scheduler_execution_and_scientific_validity_are_kept_apart(scheduler, outcome, master, expected):
    attempt = str(uuid.uuid4())
    status = None if outcome is None else {
        "attempt_id": attempt, "execution": {"outcome": outcome, "master_exit_status": master}}
    result = lifecycle.classify_job(scheduler, status, attempt_id=attempt)
    assert result["state"] == expected
    assert result["scientific_validity"] == {"status": "not_assessed"}
    assert result["finalists"] == {"status": "not_started"}


def test_a_foreign_execution_record_never_completes_a_search():
    attempt = str(uuid.uuid4())
    status = {"attempt_id": str(uuid.uuid4()), "execution": {"outcome": "uspex_finished", "master_exit_status": 0}}
    result = lifecycle.classify_job("COMPLETED", status, attempt_id=attempt)
    assert result["state"] == "SEARCH_FAILED" and result["execution"]["outcome"] == "foreign_record"


# --- one POTCAR identity per content within one operation ----------------------------------------


def test_content_addressed_identity_parses_each_distinct_content_once(tmp_path):
    from backend.structure_search.search_plan import ContentAddressedPotcarIdentity

    parsed = []

    def identify_text(text):
        parsed.append(text)
        return _identity(text)

    library, staged = tmp_path / "POTCAR", tmp_path / "POTCAR_Si"
    library.write_text(TEST_POTCAR)
    staged.write_bytes(library.read_bytes())
    identify = ContentAddressedPotcarIdentity(identify_text)
    first, copy, again = identify(library), identify(staged), identify(staged)
    assert first == copy == again and len(parsed) == 1  # library, staged copy, re-check: one parse
    staged.write_text(TEST_POTCAR.replace("245.345", "250.000"))  # changed between checks
    changed = identify(staged)
    assert changed["ENMAX"] == 250.0 and changed != first and len(parsed) == 2
    assert ContentAddressedPotcarIdentity(identify_text)(library) == first and len(parsed) == 3  # nothing kept


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_content_addressed_identity_equals_pymatgen_from_file(tmp_path, newline):
    import warnings

    from backend.structure_search.search_plan import ContentAddressedPotcarIdentity, default_potcar_identity

    path = tmp_path / "POTCAR_Si"
    path.write_bytes(TEST_POTCAR.replace("\n", newline).encode())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ContentAddressedPotcarIdentity()(path) == default_potcar_identity(path)


def test_finalization_parses_each_potcar_content_once_per_operation(tmp_path, monkeypatch):
    from backend.structure_search import search_plan

    calls = []
    real = search_plan.ContentAddressedPotcarIdentity

    class Counting(real):
        def __init__(self, identify_text=None):
            super().__init__(identify_text)
            calls.append(self)

    monkeypatch.setattr(search_plan, "ContentAddressedPotcarIdentity", Counting)
    from backend.structure_search import staging

    # Finalization builds one fresh identity per operation (nothing is kept across requests).
    for _ in range(2):
        staging.finalize_staged_search({"package_dir": str(tmp_path / "missing"), "manifest": {}})
    assert len(calls) == 2 and len({id(instance) for instance in calls}) == 2


# --- memory per relaxation under an exclusive --mem=0 allocation --------------------------------

ALLOCATED = ("JobId=4242 JobName=bmds-x UserId=bmdguest(1) JobState=RUNNING NumNodes=1 NumCPUs=192 "
             "MinMemoryNode=0 AllocTRES=cpu=192,mem=500G,node=1,billing=192")


@needs_bash
def test_mem_0_allocation_is_read_from_slurm_and_divided_as_before(rig):
    rig.fake_scontrol(ALLOCATED)
    result = rig.run(SLURM_MEM_PER_NODE="0")
    assert result.returncode == 0, result.stdout + result.stderr
    launch = rig.status()["execution"]["launch"]
    assert launch["node_memory"] == {"mb": 512000, "source": "scontrol AllocTRES mem",
                                     "detail": "scontrol show job 4242: AllocTRES mem=500G, MinMemoryNode=0"}
    assert launch["step_mem"] == "115200M" and launch["memory_margin_fraction"] == 0.1
    assert (rig.root / "scontrol_calls").read_text().split("\n")[0] == "-o show job 4242"
    records = [json.loads(p.read_text()) for p in rig.records.glob("*.json")]
    assert records and all("--mem=115200M" in record["argv"] for record in records)


@needs_bash
@pytest.mark.parametrize("value, scontrol", [
    (None, None),            # not set, and SLURM cannot be asked
    ("0", None),             # "all memory" is not an amount
    ("0", "JobId=4242 JobState=RUNNING NumNodes=1 MinMemoryNode=0 AllocTRES=cpu=192,node=1"),  # no memory tracked
    ("0", "JobId=4242 JobState=RUNNING NumNodes=1 MinMemoryNode=500G AllocTRES=cpu=192,node=1"),  # request only
    ("512G", ALLOCATED),     # malformed: refused, not guessed
])
def test_job_stops_before_uspex_when_allocated_memory_is_not_established(rig, value, scontrol):
    if scontrol is not None:
        rig.fake_scontrol(scontrol)
    result = rig.run(SLURM_MEM_PER_NODE=value)
    assert result.returncode == search_job.EXIT_LAUNCH_SETUP_FAILED, result.stdout + result.stderr
    execution = rig.status()["execution"]
    assert execution["outcome"] == "launch_setup_failed" and execution["code"] == "node_memory_unavailable"
    assert rig.singularity_calls() == [] and rig.srun_env() == []

