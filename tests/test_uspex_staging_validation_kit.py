"""The staging-only POWER validation kit (reference/uspex_pilot/staging_validation/).

Runs the driver against the fake POWER of the staging tests, and the POWER
scripts through a fake ``ssh`` that behaves like the real one (the remote
command is the remaining arguments joined by spaces and run by a shell, with
the script on standard input), with the scripts' POWER locations mapped into
the fake tree. Nothing contacts POWER.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from test_machine_api_execution import power, state_dir, tokens  # noqa: F401 - pytest fixtures
from test_machine_api_structure_search_staging import (  # noqa: F401 - pytest fixtures
    LIBRARY,
    SEARCH_LOGS,
    WORKSPACE,
    cluster,
    tree,
)

KIT = Path(__file__).resolve().parents[1] / "reference" / "uspex_pilot" / "staging_validation"
EVIDENCE_PARENT = "/bmd-db/guest/validation"


def load_kit():
    spec = importlib.util.spec_from_file_location("run_staging_validation", KIT / "run_staging_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kit():
    return load_kit()


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("BMD_API_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("BMD_SUBMISSION_IDENTITY_SECRET", "production-secret-not-to-be-used")
    return monkeypatch


FAKE_SSH = r'''#!{python}
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_SSH_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
i = 0
while args[i].startswith("-"):
    i += 2 if args[i] == "-o" else 1
assert args[i] == "powerslurm-bmdguest", args
command = " ".join(args[i + 1:])
script = sys.stdin.buffer.read().decode()
for old, new in json.loads(os.environ["FAKE_SSH_MAP"]).items():
    assert old in script, old
    script = script.replace(old, new)
sys.exit(subprocess.run(["bash", "-c", command], input=script.encode()).returncode)
'''


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch, cluster):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(FAKE_SSH.replace("{python}", sys.executable))
    ssh.chmod(0o755)
    log = tmp_path / "ssh.log"
    log.write_text("")
    cluster.at(EVIDENCE_PARENT).mkdir(parents=True)
    mapping = {
        "WORKSPACE=/bmd/bmdguest/searches": f"WORKSPACE={cluster.at(WORKSPACE)}",
        "STATE_DIR=/bmd-db/guest/logs/structure_search/submission_attempts":
            f"STATE_DIR={cluster.at(SEARCH_LOGS)}/submission_attempts",
        "LIBRARY=/bmd-db/potcars/POT_PAW_PBE_64": f"LIBRARY={cluster.at(LIBRARY)}/POT_PAW_PBE_64",
        "PY=/bmd/bmdguest/envs/atomate2_remote/bin/python": f"PY={sys.executable}",
        f"EVIDENCE_PARENT={EVIDENCE_PARENT}": f"EVIDENCE_PARENT={cluster.at(EVIDENCE_PARENT)}",
    }
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    monkeypatch.setenv("FAKE_SSH_MAP", json.dumps(mapping))
    return log


def calls(log: Path):
    return [json.loads(line) for line in log.read_text().splitlines()]


def every_file(root: Path):
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}


def staged_validation(kit, root, cluster, clean_env):
    kit.setup(root)
    summary = kit.run(root, allow_dirty=True, runner_factory=cluster.factory, proc=root / "no-proc")
    for name in ("BMD_API_STATE_DIR", "BMD_API_TOKENS_FILE"):  # each step is a new process
        clean_env.delenv(name)
    return summary


# --- 1. the attempt ID and run name reach power_evidence.sh intact ------------------------------


def test_evidence_arguments_are_quoted_and_arrive_exactly(kit):
    command = kit.remote_command("a b", "c;d", "$(touch x)", "e'f")
    out = subprocess.run(["bash", "-c", command], input=b'printf "%s|" "$#" "$@"', capture_output=True)
    assert out.stdout.decode() == "4|a b|c;d|$(touch x)|e'f|"
    assert kit.remote_command() == "bash -s"


@pytest.mark.parametrize("arguments, message", [
    ((), "usage"),
    (("0f0f0f0f-1111-4222-8333-444455556666",), "usage"),
    (("0f0f0f0f-1111-4222-8333-444455556666", "uspex_search-20261107-120000-0f0f0f0f", "extra"), "usage"),
    (("0F0F0F0F-1111-4222-8333-444455556666", "uspex_search-20261107-120000-0f0f0f0f"), "malformed attempt"),
    (("../../etc", "uspex_search-20261107-120000-0f0f0f0f"), "malformed attempt"),
    (("0f0f0f0f-1111-4222-8333-444455556666", "uspex_search-20261107-120000-0f0f0f0f/.."), "malformed run name"),
    (("0f0f0f0f-1111-4222-8333-444455556666", "$(touch pwned)"), "malformed run name"),
    (("0f0f0f0f-1111-4222-8333-444455556666", "uspex_search-20261107-120000-1a1a1a1a"), "does not belong"),
])
def test_evidence_script_refuses_missing_or_malformed_arguments(tmp_path, arguments, message):
    out = subprocess.run(["bash", "-s", "--", *arguments], stdin=open(KIT / "power_evidence.sh", "rb"),
                         capture_output=True, cwd=tmp_path)
    assert out.returncode == 2 and message in out.stdout.decode()
    assert list(tmp_path.iterdir()) == []  # nothing created, nothing run


def test_driver_passes_its_own_attempt_and_run_name_and_refuses_bad_ones(
        kit, tmp_path, cluster, clean_env, fake_ssh):
    root = tmp_path / "validation"
    summary = staged_validation(kit, root, cluster, clean_env)
    kit.evidence(root, proc=root / "no-proc")
    [args] = calls(fake_ssh)
    assert args[:3] == ["-o", "BatchMode=yes", "powerslurm-bmdguest"]
    assert args[3:] == [f"bash -s -- {summary['attempt_id']} {summary['run_name']}"]
    assert f"evidence:      {cluster.at(EVIDENCE_PARENT)}/uspex_staging_4c_" in (root / "power_evidence.txt").read_text()

    with pytest.raises(SystemExit, match="is not overwritten"):
        kit.evidence(root, proc=root / "no-proc")  # the recorded output is never replaced
    (root / "power_evidence.txt").unlink()
    record_path = root / "state" / "structure_search" / "attempts" / f"{summary['attempt_id']}.json"
    record = json.loads(record_path.read_text())
    record["staging"]["run_name"] = "uspex_search-20261107-120000-1a1a1a1a; rm -rf ~"
    record_path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="does not match the attempt"):
        kit.evidence(root, proc=root / "no-proc")
    assert len(calls(fake_ssh)) == 1  # refused before any SSH


# --- 2. evidence collection never modifies the staged run -----------------------------------


def test_evidence_script_reads_the_run_without_changing_it(kit, tmp_path, cluster, clean_env, fake_ssh):
    root = tmp_path / "validation"
    summary = staged_validation(kit, root, cluster, clean_env)
    run_dir = cluster.at(summary["run_dir"])
    state_file = cluster.at(summary["remote_state_file"])
    before_run, before_state = tree(run_dir), state_file.read_bytes()
    before_all = every_file(cluster.root)

    kit.evidence(root, proc=root / "no-proc")

    assert tree(run_dir) == before_run  # bytes, modification times and modes
    assert state_file.read_bytes() == before_state
    [out_dir] = list(cluster.at(EVIDENCE_PARENT).iterdir())
    added = every_file(cluster.root) - before_all
    assert added and all(Path(cluster.root / name).is_relative_to(out_dir) for name in added)
    assert not out_dir.is_relative_to(run_dir)
    report = (root / "power_evidence.txt").read_text()
    assert "PASS  the run directory is unchanged" in report
    assert "PASS  the run directory's file hashes are unchanged" in report
    # (Path-binding and POTCAR-parsing checks cannot pass on the fake tree: its paths are
    # mapped and its POTCARs are synthetic. The read-only guarantees above do not depend on them.)
    assert "PASS  no hidden or temporary file anywhere in the run directory" in report
    # 3. No POTCAR is copied and no POTCAR text appears, only identities and hashes.
    assert not any(Path(name).name.startswith("POTCAR_") for name in added)
    for name in added:
        assert "synthetic:" not in (cluster.root / name).read_text(errors="replace"), name
    assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700


# --- 3. the archive holds no token, credential or POTCAR text ----------------------------------


def test_archive_holds_evidence_only(kit, tmp_path, cluster, clean_env, fake_ssh):
    root = tmp_path / "validation"
    kit.setup(root)
    kit.preflight(root, proc=root / "no-proc")
    kit.run(root, allow_dirty=True, runner_factory=cluster.factory, proc=root / "no-proc")
    for name in ("BMD_API_STATE_DIR", "BMD_API_TOKENS_FILE"):
        clean_env.delenv(name)
    (root / "run.log").write_text("stage: HTTP 200\n")
    kit.evidence(root, proc=root / "no-proc")
    target = kit.archive(root)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    token = (root / "client_token").read_text().strip()
    with tarfile.open(target) as tar:
        names = tar.getnames()
        texts = {name: tar.extractfile(name).read().decode() for name in names}
    assert not any(name.endswith(("client_token", "tokens.json")) for name in names)
    assert not any("/attempts/" in name for name in names)  # raw ledger records are not archived
    assert any(name.endswith("-local_record.json") for name in names)
    assert any("/preparations/" in name for name in names)
    for name, text in texts.items():
        assert token not in text and token.split(".")[-1] not in text, name
        assert "synthetic:" not in text and "Bearer" not in text, name
    record = next(text for name, text in texts.items() if name.endswith("-local_record.json"))
    assert "identity_token" not in json.loads(record)["staging"]["spec"]["submission"]  # search specs carry none
    assert kit.redact({"a": [{"identity_token": "x.y"}]}) == {"a": [{"identity_token": kit.REDACTED}]}
    with pytest.raises(SystemExit, match="is not overwritten"):
        kit.archive(root)


def test_archive_refuses_a_token_or_potcar_text_in_any_member(kit, tmp_path, cluster, clean_env):
    root = tmp_path / "validation"
    staged_validation(kit, root, cluster, clean_env)
    (root / "run.log").write_text("debug: " + (root / "client_token").read_text())
    with pytest.raises(SystemExit, match="must not be archived"):
        kit.archive(root)
    (root / "run.log").write_text("VRHFIN =Si: s2p2\n")
    with pytest.raises(SystemExit, match="must not be archived"):
        kit.archive(root)
    assert not root.with_name(root.name + ".tgz").exists()


# --- 4. an isolated ledger, never production state or credentials -------------------------------


def test_run_never_signs_with_the_shell_secret(kit, tmp_path, cluster, clean_env):
    root = tmp_path / "validation"
    staged_validation(kit, root, cluster, clean_env)
    assert "BMD_SUBMISSION_IDENTITY_SECRET" not in os.environ


@pytest.mark.parametrize("variable", ["BMD_API_STATE_DIR", "BMD_API_TOKENS_FILE", "BMD_API_ALLOW_NON_LOOPBACK"])
def test_inherited_api_settings_are_refused(kit, tmp_path, clean_env, variable):
    root = tmp_path / "validation"
    kit.setup(root)
    clean_env.setenv(variable, "/srv/bmd/production")
    with pytest.raises(SystemExit, match=variable):
        kit.run(root, allow_dirty=True, proc=root / "no-proc")


def test_only_a_root_made_by_setup_is_accepted(kit, tmp_path, clean_env):
    production = tmp_path / "production"
    for name in ("state", "evidence"):
        (production / name).mkdir(parents=True, mode=0o700)
    production.chmod(0o700)
    with pytest.raises(SystemExit, match="was not made by 'setup'"):
        kit.check_isolation(production, proc=tmp_path / "no-proc")
    with pytest.raises(FileExistsError):
        kit.setup(production)  # an existing directory is never adopted
    with pytest.raises(SystemExit, match="outside"):
        kit.setup(kit.REPO / "validation-root")


@pytest.mark.parametrize("change, message", [
    (lambda root: (root / "state" / "attempts").mkdir() or (root / "state" / "attempts" / "x.json").write_text("{}"),
     "calculation attempts"),
    (lambda root: (root / "state" / "structure_search" / "attempts").mkdir(parents=True)
     or (root / "state" / "structure_search" / "attempts" / "y.json").write_text("{}"), "did not make"),
    (lambda root: (root / "state" / "jobs.db").write_text(""), "unexpected entries"),
    (lambda root: (root / "tokens.json").write_text(json.dumps({"principals": [
        {"principal": "lee", "scopes": ["submit"]}]})), "only the validation principal"),
    (lambda root: (root / "state").chmod(0o755), "private"),
])
def test_state_or_tokens_that_are_not_the_validation_own_are_refused(kit, tmp_path, clean_env, change, message):
    root = tmp_path / "validation"
    kit.setup(root)
    change(root)
    with pytest.raises(SystemExit, match=message):
        kit.check_isolation(root, proc=tmp_path / "no-proc")


def test_a_running_service_using_the_same_paths_is_refused(kit, tmp_path, clean_env):
    root = tmp_path / "validation"
    kit.setup(root)
    proc = tmp_path / "proc"
    service = proc / "4242"
    service.mkdir(parents=True)
    (service / "cmdline").write_bytes(b"/opt/env/bin/python\0-m\0uvicorn\0main:app\0")
    (service / "environ").write_bytes(f"PATH=/bin\0BMD_API_STATE_DIR={root / 'state'}\0".encode())
    with pytest.raises(SystemExit, match="running BMD Compute service"):
        kit.check_isolation(root, proc=proc)
    (service / "environ").write_bytes(b"BMD_API_STATE_DIR=/srv/bmd/state\0BMD_API_TOKENS_FILE=/srv/bmd/tokens.json\0")
    assert kit.live_service_paths(proc) == {Path("/srv/bmd/state"), Path("/srv/bmd/tokens.json")}
    kit.check_isolation(root, proc=proc)  # a production service elsewhere is fine


# --- the full staging-only flow --------------------------------------------------------------------


def test_validation_stages_one_si_search_resumes_and_stops_at_staged(kit, tmp_path, cluster, clean_env):
    root = tmp_path / "validation"
    summary = staged_validation(kit, root, cluster, clean_env)
    assert summary["result"].startswith("STAGED") and summary["calculation_ledger_entries"] == []
    assert summary["run_dir"] == f"{WORKSPACE}/{summary['run_name']}"
    assert json.loads(cluster.at(summary["remote_state_file"]).read_text())["state"] == "STAGED"
    evidence = sorted(p.name for p in (root / "evidence").iterdir())
    assert "SHA256SUMS" in evidence and any(name.endswith("-fresh_reverification.json") for name in evidence)
    requests = [json.loads((root / "evidence" / name).read_text()).get("request", {})
                for name in evidence if name[0].isdigit()]
    assert all("/api/v1/attempts" not in request.get("url", "") for request in requests)
    assert all("headers" not in request for request in requests)
    assert cluster.sbatch_calls == 0

    again = kit.run(root, allow_dirty=True, runner_factory=cluster.factory, proc=root / "no-proc")
    assert (again["attempt_id"], again["run_name"]) == (summary["attempt_id"], summary["run_name"])
    assert sorted(p.name for p in cluster.at(WORKSPACE).iterdir()) == [summary["run_name"]]


def test_kit_never_submits_or_starts_a_program():
    for script in ("power_preflight.sh", "power_evidence.sh"):
        subprocess.run(["bash", "-n", str(KIT / script)], check=True)
    evidence = (KIT / "power_evidence.sh").read_text()
    compile(evidence.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0], "power_evidence_checker", "exec")
    for name in ("power_preflight.sh", "power_evidence.sh", "run_staging_validation.py"):
        code = "\n".join(line for line in (KIT / name).read_text().splitlines() if not line.lstrip().startswith("#"))
        for forbidden in ("sbatch ", "srun ", "singularity ", "rm -", "/opt/uspex/uspex", "vasp_std", "unpack"):
            assert forbidden not in code, (name, forbidden)
    copies = evidence.split("# 2. Copies", 1)[1].split("# 3.", 1)[0]
    assert "POTCAR_" not in copies and "Specific/INCAR_*" in copies


# --- promotion-only validation (step 4D-2) ------------------------------------------------------


def test_promotion_only_validation_stops_at_search_prepared(kit, tmp_path, cluster, clean_env, fake_ssh):
    root = tmp_path / "validation"
    summary = staged_validation(kit, root, cluster, clean_env)
    promoted = kit.promote(root, allow_dirty=True, runner_factory=cluster.factory, proc=root / "no-proc")
    for name in ("BMD_API_STATE_DIR", "BMD_API_TOKENS_FILE"):
        clean_env.delenv(name)
    assert promoted["result"].startswith("SEARCH_PREPARED") and promoted["calculation_ledger_entries"] == []
    assert promoted["attempt_id"] == summary["attempt_id"] and promoted["runtime_check"]["status"] == "passed"
    state = json.loads(cluster.at(summary["remote_state_file"]).read_text())
    assert state["state"] == "SEARCH_PREPARED"
    run_dir = cluster.at(summary["run_dir"])
    script = (run_dir / "bmd_search_job.sbatch").read_bytes()
    assert hashlib.sha256(script).hexdigest() == promoted["job_script"]["sha256"]
    assert cluster.sbatch_calls == 0

    again = kit.promote(root, allow_dirty=True, runner_factory=cluster.factory, proc=root / "no-proc")
    for name in ("BMD_API_STATE_DIR", "BMD_API_TOKENS_FILE"):
        clean_env.delenv(name)
    assert again["job_script"] == promoted["job_script"]  # the stored result

    before = tree(run_dir)
    kit.evidence(root, proc=root / "no-proc")
    report = (root / "power_evidence.txt").read_text()
    for line in ("remote state:  SEARCH_PREPARED",
                 "PASS  the job script is the one the promotion record names",
                 "PASS  the resource acknowledgement is the one the promotion record names",
                 "PASS  bash can parse the job script",
                 "PASS  no job has started in the run directory",
                 "PASS  the run directory is unchanged"):
        assert line in report, line
    assert tree(run_dir) == before
    [out_dir] = list(cluster.at(EVIDENCE_PARENT).iterdir())
    assert (out_dir / "files" / "bmd_search_job.sbatch").read_bytes() == script
    target = kit.archive(root)
    with tarfile.open(target) as tar:
        names = tar.getnames()
    assert any(name.endswith("-promotion_summary.json") for name in names)


def test_promotion_validation_needs_a_staged_attempt(kit, tmp_path, cluster, clean_env):
    root = tmp_path / "validation"
    kit.setup(root)
    with pytest.raises(SystemExit, match="run 'run' first"):
        kit.promote(root, allow_dirty=True, runner_factory=cluster.factory, proc=root / "no-proc")
