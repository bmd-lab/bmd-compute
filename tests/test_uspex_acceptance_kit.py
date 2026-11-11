"""The controlled POWER acceptance commands of the validation kit (USPEX step 4D, part D).

``submit``, ``observe`` and ``cancel`` drive the machine API in-process against
the fake POWER and SLURM of the staging tests; ``acceptance-evidence`` runs
power_acceptance_evidence.sh through the fake ``ssh`` of the kit tests, with
fake ``squeue``/``sacct``/``scontrol`` answering from the fake job table.
Nothing contacts POWER and no job runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from compute_api import structure_search_monitoring as monitoring
from test_machine_api_execution import power, state_dir, tokens  # noqa: F401 - pytest fixtures
from test_machine_api_structure_search_attempts import search_record
from test_machine_api_structure_search_monitoring import elapse
from test_machine_api_structure_search_staging import SEARCH_LOGS, WORKSPACE, cluster  # noqa: F401
from test_uspex_staging_validation_kit import (  # noqa: F401 - pytest fixtures
    EVIDENCE_PARENT,
    KIT,
    calls,
    clean_env,
    fake_ssh,
    kit,
    staged_validation,
)

SLURM_TIME = "%Y-%m-%dT%H:%M:%S"


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    monkeypatch.setattr(monitoring, "MIN_OBSERVATION_INTERVAL_SECONDS", 0)


def _fresh_process(clean_env):
    for name in ("BMD_API_STATE_DIR", "BMD_API_TOKENS_FILE"):  # each kit step is a new process
        clean_env.delenv(name, raising=False)


def step(kit, clean_env, name, root, cluster, **options):
    try:
        return getattr(kit, name)(root, allow_dirty=True, runner_factory=cluster.factory, proc=root / "no-proc",
                                  **options)
    finally:
        _fresh_process(clean_env)


WALLTIME = "00:15:00"


def acceptance_root(kit, tmp_path, cluster, clean_env, walltime="0:15:00"):
    root = tmp_path / "acceptance"
    kit.setup(root, acceptance_walltime=walltime)
    step(kit, clean_env, "run", root, cluster)
    step(kit, clean_env, "promote", root, cluster)
    return root


@pytest.fixture
def prepared_root(kit, tmp_path, cluster, clean_env):
    return acceptance_root(kit, tmp_path, cluster, clean_env)


def attempt_of(root):
    return json.loads((root / "evidence" / "attempt.json").read_text())["attempt_id"]


def submit(kit, clean_env, root, cluster, confirm=None):
    return step(kit, clean_env, "submit", root, cluster, confirm_attempt=confirm or attempt_of(root))


# --- submit ---------------------------------------------------------------------------------------------


def test_submit_needs_the_exact_attempt_id_and_sends_one_request(kit, prepared_root, cluster, clean_env):
    root, attempt = prepared_root, attempt_of(prepared_root)
    altered = attempt[:-1] + ("1" if attempt[-1] == "0" else "0")  # always differs from the attempt
    for wrong in ("", attempt.upper(), altered, "14bfbbac-8599-48a6-bcc2-cfea917fd967"):
        with pytest.raises(SystemExit, match="--confirm-attempt must be exactly"):
            submit(kit, clean_env, root, cluster, confirm=wrong or "x")
    assert cluster.sbatch_runs == 0 and not (root / "evidence" / "submission_requested.json").exists()

    job = submit(kit, clean_env, root, cluster)
    assert job["job_id"] == "9001" and job["attempt_id"] == attempt and cluster.sbatch_runs == 1
    assert json.loads((root / "evidence" / "job.json").read_text())["job_id"] == "9001"
    with pytest.raises(SystemExit, match="already sent its one submission request"):
        submit(kit, clean_env, root, cluster)
    assert cluster.sbatch_runs == 1 and list(cluster.jobs) == ["9001"]


def test_submit_refuses_an_attempt_that_is_not_promoted(kit, tmp_path, cluster, clean_env):
    root = tmp_path / "acceptance"
    kit.setup(root, acceptance_walltime=WALLTIME)
    step(kit, clean_env, "run", root, cluster)
    with pytest.raises(SystemExit, match="not a submission-ready PREPARED attempt"):
        submit(kit, clean_env, root, cluster)
    assert cluster.sbatch_runs == 0


def test_historical_validation_attempts_are_never_used(kit, tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    assert "c6edaf94-a2ab-4be7-a817-ce52f3d14852" in kit.HISTORICAL_ATTEMPTS  # the 4D acceptance of 10 Nov 2026
    for attempt in sorted(kit.HISTORICAL_ATTEMPTS):
        (evidence / "attempt.json").write_text(json.dumps({"attempt_id": attempt, "plan_digest": "sha256:x"}))
        with pytest.raises(SystemExit, match="historical validation attempt"):
            kit._own_attempt(evidence)


@pytest.mark.parametrize("mode, message", [
    ("lost_after", "UNCERTAIN SUBMISSION"),
    ("lost_before", "UNCERTAIN SUBMISSION"),
    ("garbled", "UNCERTAIN SUBMISSION"),
    ("reject", "returned 502 \\(submit_rejected\\)"),
])
def test_submit_stops_on_anything_but_a_clean_submission_and_never_repeats(kit, prepared_root, cluster, clean_env,
                                                                          mode, message):
    cluster.sbatch_mode = mode
    with pytest.raises(SystemExit, match=message):
        submit(kit, clean_env, prepared_root, cluster)
    assert cluster.sbatch_runs == 1 and not (prepared_root / "evidence" / "job.json").exists()
    cluster.sbatch_mode = "accept"
    with pytest.raises(SystemExit, match="already sent its one submission request"):
        submit(kit, clean_env, prepared_root, cluster)
    assert cluster.sbatch_runs == 1
    with pytest.raises(SystemExit, match="has not submitted"):
        step(kit, clean_env, "observe", prepared_root, cluster)


def test_submit_stops_when_the_remote_record_is_pending(kit, prepared_root, cluster, clean_env, monkeypatch):
    from test_machine_api_structure_search_submission import _failing_submitted_write

    monkeypatch.setattr(cluster, "factory", _failing_submitted_write(cluster, attempt_of(prepared_root),
                                                                      TimeoutError("timed out")))
    with pytest.raises(SystemExit, match="SEARCH_SUBMITTED record is pending"):
        submit(kit, clean_env, prepared_root, cluster)
    assert cluster.sbatch_runs == 1


# --- observe --------------------------------------------------------------------------------------------


@pytest.fixture
def submitted_root(kit, prepared_root, cluster, clean_env):
    submit(kit, clean_env, prepared_root, cluster)
    return prepared_root


def test_observe_is_read_only_and_checks_the_identity_and_evidence_chain(kit, submitted_root, cluster, clean_env):
    commands = len(cluster.commands)
    summary = step(kit, clean_env, "observe", submitted_root, cluster)
    assert summary["state"] == "SUBMITTED" and summary["identity"] == "verified" and summary["times"] == "consistent"
    assert summary["binding"] == "sbatch" and summary["scheduler"]["source"] == "squeue"
    cluster.jobs["9001"]["state"] = "RUNNING"
    assert step(kit, clean_env, "observe", submitted_root, cluster)["evidence_files"] == 2
    issued = cluster.commands[commands:]
    assert not any("sbatch" in c or "scancel" in c for c in issued) and cluster.sbatch_runs == 1
    assert cluster.scancel_runs == []


@pytest.mark.parametrize("change, message", [
    ("queued_without_comment", "squeue reports the queued job without its comment"),
    ("second_job_with_our_identity", "contradiction for operator review"),
    ("times_not_utc", "identity unverified"),
    ("not_visible", "identity not_reported"),
])
def test_observe_stops_on_an_identity_that_is_not_verified(kit, submitted_root, cluster, clean_env, change,
                                                           message):
    job = cluster.jobs["9001"]
    if change == "queued_without_comment":
        job["comment"] = ""
    elif change == "second_job_with_our_identity":
        cluster.extra_jobs.append({**job, "job_id": "9500", "state": "RUNNING"})
    elif change == "times_not_utc":
        cluster.slurm_honours_tz = False
    elif change == "not_visible":
        job["hidden"] = True
    with pytest.raises(SystemExit, match=f"STOP: {message}"):
        step(kit, clean_env, "observe", submitted_root, cluster)
    assert cluster.sbatch_runs == 1 and cluster.scancel_runs == []


def test_observe_detects_a_broken_evidence_chain(kit, submitted_root, cluster, clean_env):
    step(kit, clean_env, "observe", submitted_root, cluster)
    step(kit, clean_env, "observe", submitted_root, cluster)
    directory = submitted_root / "state" / "structure_search" / "observations" / attempt_of(submitted_root)
    first = directory / "000001.json"
    document = json.loads(first.read_text())
    document["scheduler"]["state"] = "COMPLETED"  # edited after the fact
    first.write_text(json.dumps(document))
    with pytest.raises(SystemExit, match="does not continue the chain"):
        step(kit, clean_env, "observe", submitted_root, cluster)


# --- cancel ---------------------------------------------------------------------------------------------


def cancel(kit, clean_env, root, cluster, *, attempt=None, job="9001"):
    return step(kit, clean_env, "cancel", root, cluster, confirm_attempt=attempt or attempt_of(root),
                confirm_job=job)


def test_cancel_needs_the_exact_attempt_and_job(kit, submitted_root, cluster, clean_env):
    attempt = attempt_of(submitted_root)
    altered = attempt[:-1] + ("1" if attempt[-1] == "0" else "0")  # always differs from the attempt
    for wrong_attempt, wrong_job in ((attempt, "9002"), (attempt, "9001 "), (altered, "9001"),
                                     (attempt, "bmds-" + attempt[:8])):
        with pytest.raises(SystemExit, match="must be exactly"):
            cancel(kit, clean_env, submitted_root, cluster, attempt=wrong_attempt, job=wrong_job)
    assert cluster.scancel_runs == []
    assert search_record(submitted_root / "state", attempt)["state"] == "SUBMITTED"


def test_cancel_stops_when_the_identity_cannot_be_verified(kit, submitted_root, cluster, clean_env):
    cluster.jobs["9001"].update(state="RUNNING", comment="")
    with pytest.raises(SystemExit, match="STOP: cancellation returned 409 \\(job_identity_unverified\\)"):
        cancel(kit, clean_env, submitted_root, cluster)
    assert cluster.scancel_runs == []
    assert search_record(submitted_root / "state", attempt_of(submitted_root))["state"] == "SUBMITTED"


def test_cancel_stops_when_scancel_fails(kit, submitted_root, cluster, clean_env):
    cluster.jobs["9001"]["state"] = "RUNNING"
    cluster.scancel_mode = "error"
    with pytest.raises(SystemExit, match="STOP: cancellation command_failed"):
        cancel(kit, clean_env, submitted_root, cluster)
    with pytest.raises(SystemExit, match="nothing to resume"):  # a new decision, not a blind repeat
        cancel(kit, clean_env, submitted_root, cluster)
    assert cluster.scancel_runs == ["9001"]


# --- the whole acceptance, end to end -------------------------------------------------------------------


FAKE_SLURM = r'''#!{python}
import json, os, sys
from datetime import datetime, timedelta
state = json.load(open(os.environ["FAKE_SLURM_STATE"]))
tool, args = os.path.basename(sys.argv[0]), sys.argv[1:]
if tool == "scontrol":
    print("AccountingStoreFlags    = " + state["flags"])
    print("MinJobAge               = 300 sec")
    sys.exit(0)
utc = os.environ.get("TZ") == "UTC"
def shown(value):
    try:
        moment = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return "Unknown"
    return (moment if utc else moment + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
fields = args[args.index("-o") + 1]
only = args[args.index("-j") + 1] if "-j" in args else None
for job in state["jobs"]:
    if only and job["job_id"] != only:
        continue
    if tool == "squeue":
        if job["state"] not in ("PENDING", "RUNNING"):
            continue
        codes = {"%i": job["job_id"], "%j": job["job_name"], "%k": job["comment"], "%Z": job["workdir"],
                 "%T": job["state"], "%r": "None", "%V": shown(job["submit"]),
                 "%l": job["timelimit"].lstrip("0:") if job["timelimit"].startswith("00:") else job["timelimit"]}
        print("|".join(codes[code] for code in fields.split("|")))
    else:
        comment = job["comment"] if state["accounting_comments"] else ""
        names = {"JobID": job["job_id"], "JobName": job["job_name"], "Comment": comment, "WorkDir": job["workdir"],
                 "State": job["state"], "ExitCode": job.get("exit_code", "0:0"), "Submit": shown(job["submit"]),
                 "Start": shown(job.get("start")), "End": shown(job.get("end")), "Timelimit": job["timelimit"]}
        print("|".join(names[name] for name in fields.split(",")))
'''


@pytest.fixture
def fake_slurm(fake_ssh, tmp_path, cluster, monkeypatch):
    bin_dir = tmp_path / "bin"
    for tool in ("squeue", "sacct", "scontrol"):
        path = bin_dir / tool
        path.write_text(FAKE_SLURM.replace("{python}", sys.executable))
        path.chmod(0o755)
    mapping = json.loads(os.environ["FAKE_SSH_MAP"])
    mapping.pop("LIBRARY=/bmd-db/potcars/POT_PAW_PBE_64")  # not used by the acceptance script
    monkeypatch.setenv("FAKE_SSH_MAP", json.dumps(mapping))
    monkeypatch.setenv("TZ", "Etc/GMT-2")  # POWER's local time is UTC+2 (date +%z prints +0200)
    state_file = tmp_path / "fake_slurm.json"
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_file))

    def publish(*, accounting_comments=True, flags="job_comment"):
        jobs = [{**{key: (value.strftime(SLURM_TIME) if isinstance(value, datetime) else value)
                    for key, value in job.items() if key != "directives"},
                 "timelimit": job.get("timelimit") or (job.get("directives") or {}).get("time", "UNLIMITED")}
                for job in list(cluster.jobs.values()) + list(cluster.extra_jobs)]
        state_file.write_text(json.dumps({"jobs": jobs, "accounting_comments": accounting_comments,
                                          "flags": flags}))

    return publish


def test_the_acceptance_end_to_end(kit, submitted_root, cluster, clean_env, fake_slurm):
    root, attempt = submitted_root, attempt_of(submitted_root)
    job = cluster.jobs["9001"]
    job.update(state="RUNNING", start=job["submit"])
    observed = step(kit, clean_env, "observe", root, cluster)
    assert observed["identity"] == "verified" and observed["scheduler"]["state"] == "RUNNING"

    cancelled = cancel(kit, clean_env, root, cluster)
    assert cancelled["state"] == "CANCEL_REQUESTED" and cancelled["status"] == "command_sent"
    assert cancelled["check"]["identity"] == "verified" and cluster.scancel_runs == ["9001"]
    settling = step(kit, clean_env, "observe", root, cluster)
    assert settling["state"] == "CANCEL_REQUESTED" and settling["blocked"] == "terminal_settling"
    elapse(cluster, root / "state", attempt, "9001", 150)
    final = step(kit, clean_env, "observe", root, cluster)
    assert final["state"] == "SEARCH_CANCELLED" and final["cancellation"] == "observed"
    assert final["identity"] in ("verified", "partial") and cluster.sbatch_runs == 1

    fake_slurm()
    assert kit.acceptance_evidence(root, proc=root / "no-proc") == 0
    report = (root / "power_acceptance_evidence_01.txt").read_text()
    for line in ("PASS  POWER's UTC clock is readable (epoch and ISO agree)",
                 "RESULT POWER's local offset: +0200",
                 "RESULT AccountingStoreFlags: job_comment",
                 "PASS  the remote state is SEARCH_SUBMITTED or SEARCH_CANCEL_REQUESTED",
                 "PASS  the remote submission record names this job and identity",
                 "PASS  SLURM reports the job (queue or accounting)",
                 "PASS  sacct: job name is bmds-" + attempt[:8],
                 "PASS  accounting submit time is ISO 8601 (UTC request honoured)",
                 "difference 7200.0 s, POWER's offset now +0200",
                 "RESULT comment while queued: job not queued now",
                 "RESULT comment in accounting: present",
                 "PASS  no other job carries this attempt's comment or run directory",
                 "PASS  exactly one run directory exists for this attempt",
                 "PASS  the job script on POWER is the one the remote record names",
                 "PASS  the job script asks for exactly --time=00:15:00",
                 "PASS  sacct: the time limit 00:15:00 is the acceptance walltime 00:15:00",
                 "== ACCEPTANCE EVIDENCE CHECKS PASSED (0 diagnostic(s) flagged for review)"):
        assert line in report, line
    [ssh_call] = [c for c in calls(fake_ssh_log(root)) if "bash -s --" in c[-1]]
    run_name = kit.evidence_arguments(root, states={"SEARCH_CANCELLED"})[1]
    assert ssh_call[-1] == f"bash -s -- {attempt} {run_name} 9001 {WALLTIME}"
    [out_dir] = [p for p in cluster.at(EVIDENCE_PARENT).iterdir() if p.name.startswith("uspex_acceptance_4d_")]
    assert (out_dir / "remote_state.json").exists() and not list(out_dir.rglob("POTCAR*"))

    fake_slurm(accounting_comments=False)  # a second, numbered evidence run: the comment did not persist
    assert kit.acceptance_evidence(root, proc=root / "no-proc") == 0
    assert "RESULT comment in accounting: absent" in (root / "power_acceptance_evidence_02.txt").read_text()

    target = kit.archive(root)
    with tarfile.open(target) as tar:
        names = tar.getnames()
    assert any(name.endswith("/observations/" + attempt + "/000003.json") for name in names)
    assert any(name.endswith("power_acceptance_evidence_02.txt") for name in names)
    assert not any(name.endswith(("client_token", "tokens.json")) for name in names)


def fake_ssh_log(root):
    return Path(os.environ["FAKE_SSH_LOG"])


def test_acceptance_evidence_fails_on_another_job_with_the_identity(kit, submitted_root, cluster, clean_env,
                                                                    fake_slurm):
    job = cluster.jobs["9001"]
    cluster.extra_jobs.append({**job, "job_id": "9500", "state": "RUNNING"})
    fake_slurm()
    assert kit.acceptance_evidence(submitted_root, proc=submitted_root / "no-proc") == 1
    report = (submitted_root / "power_acceptance_evidence_01.txt").read_text()
    assert "FAIL  no other job carries this attempt's comment or run directory" in report
    assert "== ACCEPTANCE EVIDENCE CHECKS FAILED" in report


def test_acceptance_evidence_needs_a_submitted_attempt(kit, prepared_root, clean_env):
    with pytest.raises(SystemExit, match="not one of"):
        kit.acceptance_evidence(prepared_root, proc=prepared_root / "no-proc")


@pytest.mark.parametrize("arguments, message", [
    ([], "usage"),
    (["not-an-id", "uspex_search-20261110-120854-aaaaaaaa", "1", "00:15:00"], "malformed attempt ID"),
    (["aaaaaaaa-0000-4000-8000-000000000000", "uspex_search-20261110-120854-bbbbbbbb", "1", "00:15:00"],
     "does not belong"),
    (["aaaaaaaa-0000-4000-8000-000000000000", "uspex_search-20261110-120854-aaaaaaaa", "9001;id", "00:15:00"],
     "malformed job ID"),
    (["aaaaaaaa-0000-4000-8000-000000000000", "uspex_search-20261110-120854-aaaaaaaa", "9001", "15 min"],
     "malformed walltime"),
])
def test_acceptance_script_refuses_malformed_arguments(arguments, message):
    result = subprocess.run(["bash", str(KIT / "power_acceptance_evidence.sh"), *arguments], capture_output=True,
                            text=True)
    assert result.returncode == 2 and message in result.stdout


def test_the_kit_itself_never_submits_cancels_or_starts_a_program():
    script = (KIT / "power_acceptance_evidence.sh").read_text()
    subprocess.run(["bash", "-n", str(KIT / "power_acceptance_evidence.sh")], check=True)
    compile(script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0], "power_acceptance_checker", "exec")
    for name in ("power_acceptance_evidence.sh", "run_staging_validation.py"):
        code = "\n".join(line for line in (KIT / name).read_text().splitlines() if not line.lstrip().startswith("#"))
        for forbidden in ("sbatch ", "scancel ", "srun ", "scontrol update", "scontrol hold", "singularity ",
                          "rm -", "/opt/uspex/uspex", "vasp_std"):
            assert forbidden not in code, (name, forbidden)
    assert "slurm-*.out' -printf '%s %f" in script  # the SLURM output's size only, never its content


# --- the short allocation ----------------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [("0:15:00", "00:15:00"), ("00:10:00", "00:10:00"),
                                             ("0:12:30", "00:12:30")])
def test_the_acceptance_walltime_is_the_planner_canonical_form(kit, tmp_path, value, expected):
    kit.setup(tmp_path / "root", acceptance_walltime=value)
    assert kit.acceptance_walltime(tmp_path / "root") == expected
    assert kit.search_body(tmp_path / "root") == {"formula": "Si", "search": {"walltime": expected}}


@pytest.mark.parametrize("value, message", [("168:00:00", "10-15 minutes"), ("00:30:00", "10-15 minutes"),
                                            ("00:05:00", "10-15 minutes"), ("0:15", "refused by the planner"),
                                            ("00:00:00", "refused by the planner")])
def test_other_acceptance_walltimes_are_refused(kit, tmp_path, value, message):
    with pytest.raises(SystemExit, match=message):
        kit.setup(tmp_path / "root", acceptance_walltime=value)
    assert not (tmp_path / "root").exists()


def test_the_short_walltime_reaches_the_digest_acknowledgement_job_script_and_remote_record(
        kit, prepared_root, cluster, clean_env):
    from test_machine_api_structure_search_promotion import remote_state

    root, attempt = prepared_root, attempt_of(prepared_root)
    digest = json.loads((root / "evidence" / "attempt.json").read_text())["plan_digest"]
    assert digest != "sha256:014ef52c3c737e146eb096779de04817357df36decd4a3e70e394e4f74f0ebbc"  # not the default
    record = search_record(root / "state", attempt)
    acknowledgement = record["promotion"]["acknowledgement"]
    assert acknowledgement["plan_digest"] == digest and acknowledgement["resources"]["request"]["walltime"] == WALLTIME
    run_dir = cluster.at(record["staging"]["spec"]["paths"]["run_dir"])
    script = (run_dir / "bmd_search_job.sbatch").read_text()
    assert [line for line in script.splitlines() if "--time" in line] == [f"#SBATCH --time={WALLTIME}"]
    job = submit(kit, clean_env, root, cluster)
    assert job["walltime"] == WALLTIME and cluster.jobs["9001"]["directives"]["time"] == WALLTIME
    assert "168:00:00" not in json.dumps(cluster.jobs["9001"]["directives"])
    submitted = json.loads(remote_state(cluster, attempt)["output"])
    assert submitted["submission"]["job_script_sha256"] == job["job_script_sha256"]
    assert submitted["promotion"]["job_script"]["sha256"] == job["job_script_sha256"]
    checks = sorted((root / "evidence").glob("*-short_allocation_check.json"))
    assert json.loads(checks[0].read_text())["job_script_time_directives"] == [f"#SBATCH --time={WALLTIME}"]


def test_a_root_without_the_acceptance_walltime_is_never_submitted(kit, tmp_path, cluster, clean_env):
    root = tmp_path / "default"
    staged_validation(kit, root, cluster, clean_env)
    step(kit, clean_env, "promote", root, cluster)
    with pytest.raises(SystemExit, match="never submits a default \\(168-hour\\) allocation"):
        submit(kit, clean_env, root, cluster)
    assert cluster.sbatch_runs == 0 and not (root / "evidence" / "submission_requested.json").exists()


@pytest.mark.parametrize("mismatch, message", [
    ("root_walltime", "the requested walltime is 00:15:00, not 00:10:00"),
    ("job_script_time", "the job script's time directives are"),
    ("two_time_lines", "the job script's time directives are"),
    ("rebuilt_script_differs", "the rebuilt job script is not the promoted one"),
    ("acknowledgement", "the resource acknowledgement"),
])
def test_any_allocation_mismatch_stops_before_sbatch(kit, prepared_root, cluster, clean_env, monkeypatch, mismatch,
                                                     message):
    root, attempt = prepared_root, attempt_of(prepared_root)
    if mismatch == "root_walltime":
        marker = json.loads((root / "VALIDATION_ROOT.json").read_text())
        marker["acceptance"].update(walltime="00:10:00", search={"walltime": "00:10:00"})
        (root / "VALIDATION_ROOT.json").write_text(json.dumps(marker))
    elif mismatch in ("job_script_time", "two_time_lines", "rebuilt_script_differs"):
        real = kit.rebuilt_job_script

        def altered(state, attempt_id):
            script = real(state, attempt_id)
            if mismatch == "rebuilt_script_differs":
                return script + "# changed\n"
            new = "#SBATCH --time=168:00:00" if mismatch == "job_script_time" else (
                f"#SBATCH --time={WALLTIME}\n#SBATCH --time-min=168:00:00")
            return script.replace(f"#SBATCH --time={WALLTIME}", new)

        monkeypatch.setattr(kit, "rebuilt_job_script", altered)
    elif mismatch == "acknowledgement":
        path = root / "state" / "structure_search" / "attempts" / f"{attempt}.json"
        record = json.loads(path.read_text())
        record["promotion"]["acknowledgement"]["plan_digest"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match=message):
        submit(kit, clean_env, root, cluster)
    assert cluster.sbatch_runs == 0 and not (root / "evidence" / "submission_requested.json").exists()


def test_acceptance_evidence_fails_on_a_time_limit_other_than_the_walltime(kit, submitted_root, cluster,
                                                                         clean_env, fake_slurm):
    cluster.jobs["9001"]["timelimit"] = "7-00:00:00"
    fake_slurm()
    assert kit.acceptance_evidence(submitted_root, proc=submitted_root / "no-proc") == 1
    report = (submitted_root / "power_acceptance_evidence_01.txt").read_text()
    assert "FAIL  squeue: the time limit 7-00:00:00 is the acceptance walltime 00:15:00" in report


def test_the_short_walltime_changes_nothing_but_the_walltime():
    """Ladder, INCARs, POTCARs, validity rules and USPEX settings are the default plan's; only time differs."""

    from backend.structure_search.planning import StructureSearchPlanRequest, plan_structure_search

    default = plan_structure_search(StructureSearchPlanRequest(formula="Si")).plan.record
    short = plan_structure_search(StructureSearchPlanRequest(formula="Si", walltime="0:15:00")).plan.record
    differences = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in set(a) | set(b):
                walk(a.get(key), b.get(key), f"{path}.{key}")
        elif a != b:
            differences.append(path)

    walk(default, short, "")
    assert sorted(differences) == [".request.advanced_overrides.walltime", ".resources.request.walltime",
                                   ".resources.sbatch_directives", ".search.spec.walltime"]
    assert [d for d in short["resources"]["sbatch_directives"] if d not in default["resources"]["sbatch_directives"]] \
        == ["#SBATCH --time=00:15:00"]


def test_a_utc_local_mismatch_is_flagged_for_review_not_failed(kit, submitted_root, cluster, clean_env, fake_slurm,
                                                               monkeypatch):
    monkeypatch.setenv("TZ", "Etc/GMT-3")  # date +%z says +0300, but the local sacct times are UTC+2
    fake_slurm()
    assert kit.acceptance_evidence(submitted_root, proc=submitted_root / "no-proc") == 0
    report = (submitted_root / "power_acceptance_evidence_01.txt").read_text()
    assert "RESULT UTC/local diagnostic: submit " in report and "difference 7200.0 s, POWER's offset now +0300" in report
    assert "REVIEW the UTC/local difference is not POWER's current offset" in report
    assert "== ACCEPTANCE EVIDENCE CHECKS PASSED (1 diagnostic(s) flagged for review)" in report
