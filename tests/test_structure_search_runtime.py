"""USPEX runtime check, one-time unpack and provenance (backend/structure_search/runtime.py).

Everything runs against a fake POWER layout in a temporary directory: a
sandbox with a stand-in self-extractor, a shared cache, a fake Singularity
executable and a fake PMI library. No SLURM, Singularity or USPEX is needed.
"""

import json
import os
import stat
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from backend import config
from backend.structure_search import runtime
from backend.structure_search.runtime import (
    SearchRuntimeSettings,
    SearchRuntimeSetupError,
    prepared_search_runtime,
    read_program_header,
    search_runtime_check,
    settings_from_record,
    unpack_once,
)


SHA = "fc4a08e65ffc5f4fc3b45cb058de2cd54731ab2c191123aac8a3c69e87e95c8c"
OTHER_SHA = "0" * 64


def _extractor(sha=SHA, banner="USPEX v25.0, 11 November 2025", extra=""):
    header = textwrap.dedent(
        f"""\
        #!/bin/sh
        # {banner}
        PAYLOAD_SHA256={sha}
        {extra}
        exit 0
        """
    ).encode()
    return header + b"\0\x1f\x8b\x08" + os.urandom(64)


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def power(tmp_path):
    """A fake POWER layout and the settings that point at it."""

    sandbox = tmp_path / "shared" / "uspex25_sandbox"
    program = sandbox / "opt" / "uspex" / "uspex"
    program.parent.mkdir(parents=True)
    program.write_bytes(_extractor())
    cache = tmp_path / "shared" / "uspex_cache"
    unpacked = cache / "uspex_sfx" / SHA
    (unpacked / "uspex_core").mkdir(parents=True)
    (unpacked / "uspex_core" / "uspex").write_text("", encoding="utf-8")
    (unpacked / ".ready").write_text("", encoding="utf-8")
    singularity = _executable(tmp_path / "bin" / "singularity", "#!/bin/sh\nexit 0\n")
    pmi = tmp_path / "lib64" / "libpmi2.so"
    pmi.parent.mkdir()
    pmi.write_bytes(b"\x7fELF")
    settings = SearchRuntimeSettings(
        sandbox_dir=str(sandbox),
        cache_dir=str(cache),
        payload_sha256=SHA,
        singularity=str(singularity),
        i_mpi_pmi_library=str(pmi),
        launch_mode="srun_steps",
        parallel_relaxations=4,
        ranks_per_relaxation=48,
        bridge_timeout_seconds=14400,
        node_sockets=2,
        cores_per_socket=96,
        excluded_nodes="compute-0-[332-333]",
    )
    return settings


def _check(settings, slurm="slurm 26.05.5"):
    return search_runtime_check(prepared_search_runtime(settings), slurm_version_lookup=lambda: slurm)


# -- settings and provenance ---------------------------------------------------------------


def test_settings_default_to_the_configured_power_values():
    settings = SearchRuntimeSettings.from_config()
    assert settings.sandbox_dir == config.USPEX_SANDBOX_DIR
    assert settings.cache_dir == config.USPEX_CACHE_DIR
    assert settings.payload_sha256 == config.USPEX_PAYLOAD_SHA256
    assert settings.launch_mode == config.STRUCTURE_SEARCH_LAUNCH_MODE
    assert str(settings.program_path) == config.USPEX_SANDBOX_DIR + "/opt/uspex/uspex"
    assert str(settings.ready_marker) == (
        f"{config.USPEX_CACHE_DIR}/uspex_sfx/{config.USPEX_PAYLOAD_SHA256}/.ready"
    )


def test_provenance_record_lists_version_sha_paths_and_launch_layout(power):
    record = prepared_search_runtime(power)
    assert record["schema"] == "bmd_compute.structure_search_runtime"
    assert record["schema_version"] == 1
    assert record["uspex"] == {
        "version": "25.0",
        "payload_sha256": SHA,
        "sandbox_dir": power.sandbox_dir,
        "program_in_image": "/opt/uspex/uspex",
        "cache_dir": power.cache_dir,
        "unpacked_dir": f"{power.cache_dir}/uspex_sfx/{SHA}",
        "ready_marker": f"{power.cache_dir}/uspex_sfx/{SHA}/.ready",
    }
    assert record["singularity"] == power.singularity
    assert record["launch"] == {
        "mode": "srun_steps",
        "parallel_relaxations": 4,
        "ranks_per_relaxation": 48,
        "i_mpi_pmi_library": power.i_mpi_pmi_library,
        "bridge_timeout_seconds": 14400,
        "node_sockets": 2,
        "cores_per_socket": 96,
        "excluded_nodes": "compute-0-[332-333]",
    }
    assert json.loads(json.dumps(record)) == record
    assert settings_from_record(record) == (power, [])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sandbox_dir": "shared/uspex25_sandbox"}, "sandbox_dir must be an absolute path"),
        ({"sandbox_dir": "/bmd/shared/uspex_25.sif"}, "is a SIF file"),
        ({"payload_sha256": SHA.upper()}, "64 lowercase hexadecimal"),
        ({"payload_sha256": SHA[:-1]}, "64 lowercase hexadecimal"),
        ({"launch_mode": "mpirun"}, "launch_mode must be one of"),
        ({"parallel_relaxations": 5}, "240 ranks do not fit on a node of 2 x 96 = 192 cores"),
        ({"ranks_per_relaxation": 0}, "ranks_per_relaxation must be a positive integer"),
        ({"bridge_timeout_seconds": True}, "bridge_timeout_seconds must be a positive integer"),
        ({"excluded_nodes": "compute-0-1 --qos=x"}, "excluded_nodes must be a SLURM host list"),
    ],
)
def test_invalid_settings_are_rejected(power, change, message):
    with pytest.raises(ValueError, match=message):
        replace(power, **change)


def test_empty_node_exclusion_is_allowed(power):
    assert replace(power, excluded_nodes="").excluded_nodes == ""


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(schema="other"), "schema is 'other'"),
        (lambda record: record.update(schema_version=2), "schema_version is 2"),
        (lambda record: record.pop("launch"), "missing its uspex or launch block"),
        (lambda record: record["uspex"].update(version="10.5"), "prepared for USPEX '10.5'"),
        (lambda record: record["uspex"].update(program_in_image="/x"), "program_in_image"),
        (lambda record: record["uspex"].update(ready_marker="/tmp/.ready"), "ready_marker"),
        (lambda record: record["launch"].update(mode="other"), "launch_mode must be one of"),
    ],
)
def test_malformed_prepared_record_stops_the_run(power, mutate, message):
    record = prepared_search_runtime(power)
    mutate(record)
    result = search_runtime_check(record, slurm_version_lookup=lambda: "slurm 26.05.5")
    assert result["status"] == "failed"
    assert any(message in problem for problem in result["problems"]), result["problems"]
    assert result["observed"] == {}


def test_missing_prepared_record_stops_the_run():
    result = search_runtime_check(None, slurm_version_lookup=lambda: "slurm 26.05.5")
    assert result["status"] == "failed"
    assert result["problems"] == ["submission.json has no structure_search.runtime record."]


# -- the program header -------------------------------------------------------------------


def test_header_gives_payload_sha_and_banner(tmp_path):
    path = tmp_path / "uspex"
    path.write_bytes(_extractor())
    assert read_program_header(path) == {"payload_sha256": SHA, "banner_version": "25.0", "problems": []}


@pytest.mark.parametrize(
    "line", [f'PAYLOAD_SHA256="{SHA}"', f"PAYLOAD_SHA256='{SHA}'", f"  export PAYLOAD_SHA256={SHA}  # payload"]
)
def test_header_accepts_quoted_and_exported_forms(tmp_path, line):
    path = tmp_path / "uspex"
    path.write_bytes(b"#!/bin/sh\n" + line.encode() + b"\nexit 0\n\0payload")
    assert read_program_header(path)["payload_sha256"] == SHA


def test_header_reading_stops_at_the_payload(tmp_path):
    path = tmp_path / "uspex"
    path.write_bytes(b"#!/bin/sh\nexit 0\n\0" + f"\nPAYLOAD_SHA256={SHA}\n".encode())
    header = read_program_header(path)
    assert header["payload_sha256"] is None
    assert "no PAYLOAD_SHA256= line" in header["problems"][0]


def test_header_with_two_payload_values_is_ambiguous(tmp_path):
    path = tmp_path / "uspex"
    path.write_bytes(_extractor(extra=f"PAYLOAD_SHA256={OTHER_SHA}"))
    header = read_program_header(path)
    assert header["payload_sha256"] is None
    assert "more than one PAYLOAD_SHA256= value" in header["problems"][0]


# -- the job-start check ------------------------------------------------------------------


def test_check_passes_on_a_complete_layout(power):
    result = _check(power)
    assert result["status"] == "passed", result["problems"]
    assert result["problems"] == []
    observed = result["observed"]
    assert observed["payload_sha256"] == SHA
    assert observed["banner_version"] == "25.0"
    assert observed["program"]["exists"] is True
    assert observed["program"]["size_bytes"] == power.program_path.stat().st_size
    assert observed["ready_marker_exists"] is True
    assert observed["unpacked_program_exists"] is True
    assert observed["writable"] == {power.cache_dir: True, f"{power.cache_dir}/uspex_sfx": True}
    assert observed["singularity"] == {"path": power.singularity, "executable": True}
    assert observed["i_mpi_pmi_library_exists"] is True
    assert observed["slurm_version"] == "slurm 26.05.5"
    assert result["prepared"] == prepared_search_runtime(power)
    # The probe file is removed again.
    assert not list(Path(power.cache_dir).glob(".bmd-write-probe-*"))


def test_check_does_not_hash_the_sandbox_or_the_program(power, monkeypatch):
    import hashlib

    def refuse(*args, **kwargs):
        raise AssertionError("the check must not hash files")

    monkeypatch.setattr(hashlib, "sha256", refuse)
    assert _check(power)["status"] == "passed"


def test_changed_program_stops_the_run(power):
    power.program_path.write_bytes(_extractor(sha=OTHER_SHA))
    result = _check(power)
    assert result["status"] == "failed"
    assert result["observed"]["payload_sha256"] == OTHER_SHA
    assert any(f"payload SHA-256 {OTHER_SHA}" in p and "has changed" in p for p in result["problems"])


def test_other_uspex_version_stops_the_run(power):
    power.program_path.write_bytes(_extractor(banner="USPEX v26.0, 1 March 2027"))
    result = _check(power)
    assert any("is USPEX 26.0; BMD Compute runs USPEX 25.0" in p for p in result["problems"])


def test_header_without_banner_is_identified_by_its_sha(power):
    power.program_path.write_bytes(_extractor(banner="self-extracting archive"))
    result = _check(power)
    assert result["status"] == "passed", result["problems"]
    assert result["observed"]["banner_version"] is None


def test_not_a_self_extractor_stops_the_run(power):
    power.program_path.write_bytes(b"#!/bin/sh\nexec something\n")
    result = _check(power)
    assert any("no PAYLOAD_SHA256= line" in p for p in result["problems"])


def test_missing_sandbox_stops_the_run(power):
    result = _check(replace(power, sandbox_dir=power.sandbox_dir + "_missing"))
    assert any("does not exist or is not a directory" in p for p in result["problems"])


def test_sandbox_without_the_program_stops_the_run(power):
    power.program_path.unlink()
    result = _check(power)
    assert any("has no /opt/uspex/uspex" in p for p in result["problems"])


def test_not_unpacked_stops_the_run_and_never_unpacks(power):
    power.ready_marker.unlink()
    result = _check(power)
    assert result["status"] == "failed"
    assert any(".ready is missing" in p and "one-time unpack" in p for p in result["problems"])
    assert not power.ready_marker.exists()


def test_ready_marker_without_the_program_stops_the_run(power):
    (power.unpacked_dir / "uspex_core" / "uspex").unlink()
    result = _check(power)
    assert any("has no uspex_core/uspex" in p for p in result["problems"])


def test_unwritable_cache_stops_the_run(power, monkeypatch):
    real_access = os.access
    cache = power.cache_dir

    def access(path, mode, *args, **kwargs):
        if str(path) == cache and mode & os.W_OK:
            return False
        return real_access(path, mode, *args, **kwargs)

    monkeypatch.setattr(runtime.os, "access", access)
    result = _check(power)
    assert result["observed"]["writable"][cache] is False
    assert any(
        f"{cache} is not writable" in p and "unpacks 2 GB into /tmp" in p for p in result["problems"]
    )


def test_cache_where_files_cannot_be_created_stops_the_run(power, monkeypatch):
    unpack_root = str(power.unpack_root)
    real_mkstemp = runtime.tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        if str(kwargs.get("dir")) == unpack_root:
            raise PermissionError(13, "Permission denied")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(runtime.tempfile, "mkstemp", mkstemp)
    result = _check(power)
    assert result["observed"]["writable"][unpack_root] is False
    assert any(f"{unpack_root} is not writable" in p for p in result["problems"])


def test_missing_singularity_stops_the_run(power):
    Path(power.singularity).unlink()
    assert any("Singularity" in p for p in _check(power)["problems"])


@pytest.mark.parametrize(
    ("slurm", "message"),
    [
        ("slurm 20.11.9", "too old for the srun_steps launch mode"),
        (None, "SLURM version could not be read"),
    ],
)
def test_srun_steps_needs_slurm_21_08(power, slurm, message):
    result = _check(power, slurm=slurm)
    assert any(message in p for p in result["problems"])


def test_srun_steps_needs_the_pmi_library(power):
    Path(power.i_mpi_pmi_library).unlink()
    assert any("I_MPI_PMI_LIBRARY" in p for p in _check(power)["problems"])


def test_mpirun_pinned_needs_neither_srun_nor_the_pmi_library(power):
    Path(power.i_mpi_pmi_library).unlink()
    result = _check(replace(power, launch_mode="mpirun_pinned"), slurm=None)
    assert result["status"] == "passed", result["problems"]


def test_every_problem_is_reported_at_once(power):
    power.program_path.write_bytes(_extractor(sha=OTHER_SHA))
    power.ready_marker.unlink()
    Path(power.singularity).unlink()
    assert len(_check(power, slurm=None)["problems"]) == 4


@pytest.mark.parametrize(
    ("text", "version"),
    [("slurm 26.05.5", (26, 5, 5)), ("slurm 21.08", (21, 8)), ("", None), (None, None), ("srun: x", None)],
)
def test_parse_slurm_version(text, version):
    assert runtime.parse_slurm_version(text) == version


def test_structure_search_submissions_are_recognised():
    assert runtime.is_structure_search_spec({"structure_search": {"runtime": {}}})
    assert not runtime.is_structure_search_spec({"flow_spec": {}})
    assert not runtime.is_structure_search_spec(None)
    assert runtime.prepared_record_from_spec({"structure_search": {"runtime": {"a": 1}}}) == {"a": 1}
    assert runtime.prepared_record_from_spec({"structure_search": "x"}) is None


# -- one-time unpack ------------------------------------------------------------------------


FAKE_SINGULARITY = """\
#!/bin/sh
# Stand-in for `singularity exec ... <sandbox> /opt/uspex/uspex`: records how it
# was called, then behaves like the extractor according to FAKE_MODE.
printf '%s\\n' "$@" > "$FAKE_ARGS_FILE"
env | grep '^SINGULARITYENV_' | sort > "$FAKE_ENV_FILE"
pwd > "$FAKE_PWD_FILE"
d="$SINGULARITYENV_XDG_CACHE_HOME/uspex_sfx/$FAKE_SHA"
case "$FAKE_MODE" in
  fail) echo "extractor failed"; exit 2 ;;
  slow) exec sleep 60 ;;
esac
mkdir -p "$d/uspex_core" && : > "$d/uspex_core/uspex" && : > "$d/.ready"
case "$FAKE_MODE" in
  crash) echo "IndexError in calculate_formation_energies"; exit 1 ;;
  keeps_running) exec sleep 60 ;;
esac
"""


@pytest.fixture
def fresh_power(power, tmp_path, monkeypatch):
    """The fake layout before the one-time unpack, with a fake Singularity."""

    import shutil

    shutil.rmtree(power.unpack_root)
    _executable(Path(power.singularity), FAKE_SINGULARITY)
    for name in ("ARGS", "ENV", "PWD"):
        monkeypatch.setenv(f"FAKE_{name}_FILE", str(tmp_path / f"fake_{name.lower()}.txt"))
    monkeypatch.setenv("FAKE_SHA", SHA)
    monkeypatch.setenv("SINGULARITYENV_STRAY", "must not reach the container")
    return power


@pytest.mark.parametrize("mode", ["crash", "keeps_running"])
def test_unpack_once_unpacks_into_the_shared_cache(fresh_power, tmp_path, monkeypatch, mode):
    monkeypatch.setenv("FAKE_MODE", mode)
    result = unpack_once(fresh_power, poll_seconds=0.05)

    assert result["status"] == "unpacked"
    assert result["extractor_exit"] == (1 if mode == "crash" else "stopped_after_ready")
    assert fresh_power.ready_marker.is_file()
    assert stat.S_IMODE(fresh_power.unpack_root.stat().st_mode) & 0o2070 == 0o2070
    args = (tmp_path / "fake_args.txt").read_text().splitlines()
    workdir = (tmp_path / "fake_pwd.txt").read_text().strip()
    assert args == [
        "exec", "--cleanenv", "-B", fresh_power.cache_dir, "--pwd", workdir,
        fresh_power.sandbox_dir, "/opt/uspex/uspex",
    ]
    assert Path(workdir).parent == Path(fresh_power.cache_dir)
    assert not Path(workdir).exists()  # the scratch folder is removed
    assert (tmp_path / "fake_env.txt").read_text().splitlines() == [
        "SINGULARITYENV_LANG=C.UTF-8",
        "SINGULARITYENV_LC_ALL=C.UTF-8",
        "SINGULARITYENV_PYTHONIOENCODING=utf-8",
        f"SINGULARITYENV_XDG_CACHE_HOME={fresh_power.cache_dir}",
    ]
    # A job-start check now passes.
    assert _check(fresh_power)["status"] == "passed"


def test_unpack_once_does_nothing_when_already_unpacked(power):
    def popen(*args, **kwargs):
        raise AssertionError("must not start the extractor")

    assert unpack_once(power, popen=popen) == {
        "status": "already_unpacked",
        "unpacked_dir": str(power.unpacked_dir),
    }


def test_unpack_once_refuses_a_partial_unpack(fresh_power):
    partial = fresh_power.unpack_root / (SHA + ".tmp.12345")
    partial.mkdir(parents=True)
    with pytest.raises(SearchRuntimeSetupError, match="partial unpack is in the way"):
        unpack_once(fresh_power)
    (fresh_power.unpacked_dir).mkdir()
    partial.rmdir()
    with pytest.raises(SearchRuntimeSetupError, match="partial unpack is in the way"):
        unpack_once(fresh_power)


def test_unpack_once_refuses_a_different_program(fresh_power):
    fresh_power.program_path.write_bytes(_extractor(sha=OTHER_SHA))
    with pytest.raises(SearchRuntimeSetupError, match="Nothing was unpacked"):
        unpack_once(fresh_power)
    assert not fresh_power.unpacked_dir.exists()


def test_unpack_once_reports_a_failed_extractor(fresh_power, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    with pytest.raises(SearchRuntimeSetupError, match="(?s)without a complete unpack.*extractor failed"):
        unpack_once(fresh_power, poll_seconds=0.05)


def test_unpack_once_times_out_and_stops_the_extractor(fresh_power, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    with pytest.raises(SearchRuntimeSetupError, match="did not finish within 0.5 s"):
        unpack_once(fresh_power, timeout_seconds=0.5, poll_seconds=0.05)


def test_unpack_once_reports_a_missing_singularity(fresh_power):
    Path(fresh_power.singularity).unlink()
    with pytest.raises(SearchRuntimeSetupError, match="could not be started"):
        unpack_once(fresh_power)


def test_command_line_prints_the_record(capsys):
    assert runtime.main(["record"]) == 0
    assert json.loads(capsys.readouterr().out) == prepared_search_runtime()
