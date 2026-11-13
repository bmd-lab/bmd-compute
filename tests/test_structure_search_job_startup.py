"""Early start-up of the structure-search job: the shell before Python, and Python before job_main.

Job 23616052 (13 Nov 2026) stopped one second after its banner with exit 1, ``job_started`` written and
nothing else: no status record, no further output. Reproduced here with the real generated job script:
the script runs ``set -u``, and the system shell initialisation (``/etc/profile``) or the ``module``
function then expands an unset variable with its output sent to /dev/null, so bash exits with status 1
and prints nothing. These tests check that system and module scripts now run without ``set -u`` with
their output kept in ``job_setup.log``, and that any other unexpected stop of the shell, or failure of
Python before ``job_main``, leaves a status record (never over an existing one) and a log.

``/etc/profile`` is replaced only inside a private mount namespace (``unshare``); the tests that need it
are skipped where that is not available.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from test_structure_search_job import JobRig, needs_bash, search_job

# The parts of a RHEL/Rocky 9 /etc/profile that matter here: a test of HISTCONTROL, which is unset in a batch
# shell, and the profile.d loop; profile.d/lmod.sh defines and exports the module function.
ROCKY_LIKE_PROFILE = r'''
HISTSIZE=1000
if [ "$HISTCONTROL" = "ignorespace" ] ; then
    export HISTCONTROL=ignoreboth
else
    export HISTCONTROL=ignoredups
fi
for i in PROFILE_D/*.sh ; do
    if [ -r "$i" ]; then
        if [ "${-#*i}" != "$-" ]; then . "$i"; else . "$i" >/dev/null; fi
    fi
done
'''
LMOD_SH = r'''
module() { case "$1" in purge) export LOADEDMODULES=;; load) export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$2";; esac; return 0; }
export -f module
'''
# Like Lmod's module function under set -u: it expands a variable the batch shell does not have.
UNSET_VARIABLE_MODULE = ('() { x="$__LMOD_REF_COUNT_MODULEPATH"; case "$1" in purge) export LOADEDMODULES=;; '
                         'load) export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$2";; esac; return 0; }')


def _can_replace_etc_profile() -> bool:
    if shutil.which("unshare") is None:
        return False
    probe = subprocess.run(["unshare", "--mount", "--propagation", "private", "true"], capture_output=True)
    return probe.returncode == 0


needs_namespace = pytest.mark.skipif(not _can_replace_etc_profile(), reason="needs a private mount namespace")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return JobRig(tmp_path, monkeypatch)


def run_with_profile(rig, profile_text, **extra):
    """The job with no module function inherited and /etc/profile replaced in a private mount namespace."""

    profile_d = rig.root / "profile.d"
    profile_d.mkdir(exist_ok=True)
    (profile_d / "lmod.sh").write_text(LMOD_SH)
    profile = rig.root / "fake_etc_profile"
    profile.write_text(profile_text.replace("PROFILE_D", str(profile_d)))
    environment = {k: v for k, v in rig.environment(**{"BASH_FUNC_module%%": None, **extra}).items()
                   if v is not None}
    return subprocess.run(["unshare", "--mount", "--propagation", "private", "bash", "-c",
                           'mount --bind "$1" /etc/profile && exec bash "$2"', "_", str(profile),
                           str(rig.script_path)], env=environment, cwd=rig.root, capture_output=True, text=True,
                          timeout=240)


def execution(rig) -> dict:
    return rig.status()["execution"]


# -- the cause of job 23616052, now handled --------------------------------------------------------------------


@needs_bash
@needs_namespace
def test_a_system_profile_written_without_set_u_no_longer_ends_the_job(rig):
    result = run_with_profile(rig, ROCKY_LIKE_PROFILE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert execution(rig)["outcome"] == "uspex_finished"
    assert (rig.run_dir / search_job.SETUP_LOG).exists()


@needs_bash
def test_a_module_function_written_without_set_u_no_longer_ends_the_job(rig):
    result = rig.run(**{"BASH_FUNC_module%%": UNSET_VARIABLE_MODULE})
    assert result.returncode == 0, result.stdout + result.stderr
    assert execution(rig)["outcome"] == "uspex_finished"
    assert execution(rig).get("code") is None


# -- any other unexpected stop of the shell is recorded ---------------------------------------------------------


@needs_bash
@needs_namespace
def test_a_shell_exit_inside_the_system_profile_is_recorded(rig):
    result = run_with_profile(rig, 'echo "profile says goodbye" >&2\nexit 3\n')
    assert result.returncode == 3
    record = execution(rig)
    assert (record["outcome"], record["code"], record["exit_status"]) == ("setup_failed", "shell_stopped", 3)
    assert "system shell initialisation (/etc/profile)" in record["message"]
    assert "STOPPED: the job shell stopped during system shell initialisation" in result.stdout
    assert "profile says goodbye" in (rig.run_dir / search_job.SETUP_LOG).read_text()
    assert (rig.run_dir / "job_started").read_text().strip() == "4242"
    assert rig.singularity_calls() == []


@needs_bash
def test_a_shell_exit_inside_the_module_function_is_recorded(rig):
    result = rig.run(**{"BASH_FUNC_module%%": '() { echo "lmod exploded" >&2; exit 7; }'})
    assert result.returncode == 7
    record = execution(rig)
    assert (record["code"], record["exit_status"]) == ("shell_stopped", 7) and "module set-up" in record["message"]
    assert "lmod exploded" in (rig.run_dir / search_job.SETUP_LOG).read_text()


@needs_bash
def test_a_recorded_setup_failure_is_never_replaced(rig):
    result = rig.run(**{"BASH_FUNC_module%%": '() { case "$1" in load) return 1;; esac; return 0; }'})
    assert result.returncode == search_job.EXIT_SETUP_FAILED
    record = execution(rig)
    assert record["outcome"] == "setup_failed" and record["message"] == "module load intel/test-oneAPI failed"
    assert "code" not in record and "shell stopped" not in result.stdout


@needs_bash
def test_an_existing_status_record_is_never_replaced_by_the_shell(rig):
    earlier = '{"execution": {"outcome": "an earlier record"}}\n'
    (rig.run_dir / search_job.STATUS_FILE).write_text(earlier)
    result = rig.run(**{"BASH_FUNC_module%%": "() { exit 7; }"})
    assert result.returncode == 7
    assert (rig.run_dir / search_job.STATUS_FILE).read_text() == earlier


@needs_bash
def test_a_second_job_is_still_stopped_by_job_started_and_records_nothing(rig):
    assert rig.run().returncode == 0
    status = (rig.run_dir / search_job.STATUS_FILE).read_bytes()
    again = rig.run(SLURM_JOB_ID="4243", **{"BASH_FUNC_module%%": "() { exit 7; }"})
    assert again.returncode == search_job.EXIT_ALREADY_STARTED
    assert (rig.run_dir / search_job.STATUS_FILE).read_bytes() == status


@needs_bash
def test_a_log_that_cannot_be_written_never_stops_the_job(rig):
    (rig.run_dir / search_job.SETUP_LOG).mkdir()  # a directory: the shell cannot append to it
    result = rig.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert execution(rig)["outcome"] == "uspex_finished"


# -- Python before job_main ---------------------------------------------------------------------------------------


@needs_bash
def test_a_python_failure_before_job_main_is_recorded_and_still_printed(rig):
    rig.script_path.write_text(rig.script.replace("from backend.structure_search.job import job_main",
                                                  "from backend.structure_search.no_such_module import job_main"))
    result = rig.run()
    assert result.returncode == 1
    record = execution(rig)
    assert (record["outcome"], record["code"]) == ("setup_failed", "python_startup_failed")
    assert "importing the job entry point" in record["message"]
    assert "ModuleNotFoundError" in result.stderr  # the traceback still reaches the SLURM output
    log = (rig.run_dir / search_job.SETUP_LOG).read_text()
    assert "Python failure while importing the job entry point" in log and "ModuleNotFoundError" in log


@needs_bash
def test_a_python_failure_never_replaces_an_existing_record(rig):
    earlier = '{"execution": {"outcome": "an earlier record"}}\n'
    (rig.run_dir / search_job.STATUS_FILE).write_text(earlier)
    rig.script_path.write_text(rig.script.replace("from backend.structure_search.job import job_main",
                                                  "from backend.structure_search.no_such_module import job_main"))
    assert rig.run().returncode == 1
    assert (rig.run_dir / search_job.STATUS_FILE).read_text() == earlier
    assert "ModuleNotFoundError" in (rig.run_dir / search_job.SETUP_LOG).read_text()


def test_the_job_script_runs_system_and_module_scripts_without_set_u_and_never_discards_their_output(rig):
    lines = rig.script.splitlines()
    start, end = lines.index("set +u"), lines.index('PHASE="starting Python"')
    section = lines[start:end]
    assert section[-1] == "set -u"
    assert any(line.startswith("if ! type module") and 'source /etc/profile >>"$SETUP_LOG"' in line for line in section)
    assert not [line for line in section if "/etc/profile" in line and ">/dev/null 2>&1 ||" in line]
    assert all('>>"$SETUP_LOG" 2>&1' in line for line in section if line.startswith(("module purge", "module load")))
    assert lines.index("trap on_shell_exit EXIT") < start
    assert lines.index("trap on_shell_exit EXIT") > next(i for i, l in enumerate(lines) if "job_started" in l
                                                         and "noclobber" in l)


# -- the two strict-mode routes of job 23616052, reproduced with set -u kept on --------------------------------


def _strict(rig) -> str:
    """The job script as job 23616052 ran it: system and module scripts under set -u."""

    assert rig.script.count("\nset +u\n") == 1
    return rig.script.replace("\nset +u\n", "\n")


@needs_bash
@needs_namespace
def test_route_one_under_set_u_the_system_profile_stops_the_shell_and_is_now_recorded(rig):
    rig.script_path.write_text(_strict(rig))
    result = run_with_profile(rig, ROCKY_LIKE_PROFILE)
    assert result.returncode == 1  # what job 23616052 exited with
    record = execution(rig)
    assert (record["code"], record["exit_status"]) == ("shell_stopped", 1)
    assert "system shell initialisation (/etc/profile)" in record["message"]
    assert "HISTCONTROL: unbound variable" in (rig.run_dir / search_job.SETUP_LOG).read_text()
    assert rig.singularity_calls() == []


@needs_bash
def test_route_two_under_set_u_the_module_function_stops_the_shell_and_is_now_recorded(rig):
    rig.script_path.write_text(_strict(rig))
    result = rig.run(**{"BASH_FUNC_module%%": UNSET_VARIABLE_MODULE})
    assert result.returncode == 1
    record = execution(rig)
    assert (record["code"], record["exit_status"]) == ("shell_stopped", 1) and "module set-up" in record["message"]
    assert "__LMOD_REF_COUNT_MODULEPATH: unbound variable" in (rig.run_dir / search_job.SETUP_LOG).read_text()


# -- strict mode, exit status and the trap's own safety ---------------------------------------------------------


def _inject(rig, line: str) -> None:
    marker = 'PHASE="starting Python"'
    assert rig.script.count(marker) == 1
    rig.script_path.write_text(rig.script.replace(marker, f"{marker}\n{line}"))


@needs_bash
def test_strict_mode_is_restored_after_the_module_set_up(rig):
    _inject(rig, 'case $- in *u*) echo "NOUNSET_IS_ON" >&9;; *) echo "NOUNSET_IS_OFF" >&9;; esac')
    result = rig.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOUNSET_IS_ON" in result.stdout and "NOUNSET_IS_OFF" not in result.stdout


@needs_bash
def test_an_unset_variable_outside_the_module_set_up_still_stops_the_shell_with_its_own_status(rig):
    _inject(rig, 'echo "$BMD_NO_SUCH_VARIABLE"')
    result = rig.run()
    assert result.returncode == 1
    record = execution(rig)
    assert (record["code"], record["exit_status"]) == ("shell_stopped", 1) and "starting Python" in record["message"]


@needs_bash
@pytest.mark.parametrize("status", [5, 42, 125])
def test_the_shell_keeps_its_own_exit_status(rig, status):
    _inject(rig, f"exit {status}")
    assert rig.run().returncode == status
    assert execution(rig)["exit_status"] == status


@needs_bash
def test_a_trap_that_cannot_record_adds_no_failure_and_keeps_the_exit_status(rig):
    (rig.bin / "ln").write_text("#!/bin/sh\nexit 1\n")  # the record cannot be linked into place
    (rig.bin / "ln").chmod(0o755)
    _inject(rig, "exit 6")
    result = rig.run()
    assert result.returncode == 6
    assert not (rig.run_dir / search_job.STATUS_FILE).exists()
    assert "STOPPED: the job shell stopped during starting Python (exit 6)" in result.stdout
    assert result.stderr == ""
    assert not list(rig.run_dir.glob(".search_status.json.shell.*"))  # its temporary file is removed


@needs_bash
def test_a_trap_with_its_output_closed_adds_no_failure(rig):
    _inject(rig, "exec 9>&-\nexit 4")
    result = rig.run()
    assert result.returncode == 4 and result.stderr == ""
    assert execution(rig)["exit_status"] == 4


@needs_bash
def test_a_python_failure_reading_the_job_request_is_recorded(rig):
    lines = rig.script.splitlines()
    [index] = [i for i, line in enumerate(lines) if line.startswith(search_job.REQUEST_MARKER)]
    lines[index] = search_job.REQUEST_MARKER + repr("bm90IGpzb24=")  # valid base64, not JSON
    rig.script_path.write_text("\n".join(lines) + "\n")
    result = rig.run()
    assert result.returncode == 1
    record = execution(rig)
    assert record["code"] == "python_startup_failed" and rig.status()["attempt_id"] is None
    assert "reading the job request" in record["message"]
    assert "JSONDecodeError" in (rig.run_dir / search_job.SETUP_LOG).read_text() and "JSONDecodeError" in result.stderr
