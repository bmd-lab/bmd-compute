"""Local checks of the POWER validity probe harness (reference/uspex_pilot/validity_probe).

The probe itself needs USPEX, Singularity, SLURM and VASP on POWER; here the
same launcher, bridge, step-validity check and injection wrapper run against a
fake USPEX master (USPEX 25 CalcFold naming) and a fake srun/VASP, and the
analysis script reads the result.
"""

from __future__ import annotations

import importlib.util
import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from backend.structure_search.bridge import protocol
from backend.structure_search.bridge.launcher import LauncherSettings, run_search
from backend.structure_search.bridge.master import bridge_script_path

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "reference" / "uspex_pilot" / "validity_probe"
FIXTURES = REPO / "tests" / "fixtures" / "structure_search_bridge"
BRIDGE = str(bridge_script_path())


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _module("run_validity_probe", PROBE / "run_validity_probe.py")
analysis = _module("analyze_probe", PROBE / "analyze_probe.py")


def _fake_probe(tmp_path: Path, injections: str, *, step_validity: bool = True, individuals: int = 4):
    work = tmp_path / "work"
    search, spool, state, bin_dir, records = (work / "search", work / "spool", work / "probe_state",
                                              tmp_path / "bin", tmp_path / "records")
    for path in (search, state, bin_dir, records):
        path.mkdir(parents=True)
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"sleep": 0.05}))
    fake = bin_dir / "srun"
    fake.write_text(f"#!{sys.executable}\n" + (FIXTURES / "fake_srun.py").read_text())
    fake.chmod(0o755)
    environment = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "FAKE_SRUN_RECORDS": str(records),
        "FAKE_SRUN_CONTROL": str(control),
        "FAKE_SRUN_FIXTURES": str(FIXTURES),
        "BMD_PROBE_STATE": str(state),
        "BMD_PROBE_INJECTIONS": injections,
    }
    settings = LauncherSettings(
        spool_dir=spool,
        search_dir=search,
        max_parallel=2,
        ranks=4,
        environment=environment,
        mem="8G",
        command_template=f"{PROBE / 'inject_vasp.sh'} {{step}} srun --exact --mpi=pmi2 -n {{ranks}} --mem={{mem}} vasp_std",
        poll_seconds=0.05,
        heartbeat_interval_seconds=0.2,
        kill_grace_seconds=2.0,
        step_validity=step_validity,
    )
    bridge_env = {
        "PATH": "/usr/bin:/bin",
        "BMD_BRIDGE_SPOOL": str(spool),
        "BMD_BRIDGE_TIMEOUT_SECONDS": "0",
        "BMD_BRIDGE_POLL_SECONDS": "0.05",
        "BMD_BRIDGE_HEARTBEAT_SECONDS": "5",
    }
    status = run_search(
        [sys.executable, str(FIXTURES / "fake_uspex_calcfolds.py"), "--search-dir", str(search),
         "--bridge", BRIDGE, "--individuals", str(individuals), "--steps", "3", "--parallel", "2"],
        bridge_env, search, settings, master_log=work / "uspex_master.log",
    )
    (work / "probe_manifest.json").write_text(json.dumps({"master_status": status, "jobflow_imported": False,
                                                          "atomate2_imported": False}))
    return work, status


# Launcher journal fields shown in a failure message: step identity and outcome only (no environment).
_EVENT_FIELDS = ("time", "event", "id", "step", "slot", "calcfold", "returncode", "status", "reason")


def _probe_evidence(work: Path) -> str:
    """The probe's own records, for an assertion message only (evaluated only when the assertion fails).

    CI run 44397423332 listed exit7_step3 twice and its log held nothing else. This shows, from the
    probe's work directory only: every VASP call in calls.tsv; the calls that were *eligible* for each
    injection (attempts: inject_vasp.sh tries to claim exit7_step3 only for a step-3 call whose VASP exited
    0, and nelm_step2 for a step-2 call) apart from those that *claimed* it (only a call whose atomic
    ``mkdir claimed_<name>`` succeeded is marked injected); injections.tsv; the claim directories with their
    creation times; and the launcher's step events, reduced to identity and outcome fields.
    """

    state, spool = work / "probe_state", work / "spool"
    parts = []
    calls = analysis.read_calls(state)
    parts.append("--- calls.tsv\n" + ((state / "calls.tsv").read_text(errors="replace")
                                       if (state / "calls.tsv").is_file() else "(absent)\n"))
    eligible = {"exit7_step3": [c for c in calls if c["step"] == 3 and c["vasp_status"] == "0"],
                "nelm_step2": [c for c in calls if c["step"] == 2]}
    for name, attempts in eligible.items():
        claimed = [c for c in calls if c["injected"] == name]
        parts.append(f"--- {name}: eligible (attempted) {[(c['calcfold'], c['started']) for c in attempts]}; "
                     f"claimed {[(c['calcfold'], c['started'], c['finished']) for c in claimed]}")
    path = state / "injections.tsv"
    parts.append("--- injections.tsv\n" + (path.read_text(errors="replace") if path.is_file() else "(absent)\n"))
    claims = sorted(p for p in state.iterdir() if p.name.startswith("claimed_")) if state.is_dir() else []
    parts.append("--- claim directories: " + ", ".join(f"{p.name} (mtime {p.stat().st_mtime:.3f})" for p in claims))
    events = []
    journal = spool / "launcher.jsonl"
    for line in (journal.read_text(errors="replace").splitlines() if journal.is_file() else []):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") in ("started_step", "finished", "rejected", "recovered", "start_failed", "abandoned"):
            events.append(json.dumps({k: event[k] for k in _EVENT_FIELDS if k in event}, sort_keys=True))
    parts.append("--- launcher step events\n" + "\n".join(events))
    return "\n".join(parts)


def test_each_injection_happens_once_and_is_reported_as_failure(tmp_path):
    work, status = _fake_probe(tmp_path, "nelm_step2,exit7_step3")
    assert status == 0
    injections = (work / "probe_state" / "injections.tsv").read_text().splitlines()
    assert sorted(line.split("\t")[3] for line in injections) == ["exit7_step3", "nelm_step2"], _probe_evidence(work)
    report = analysis.analyse(work)
    by_kind = {item["injection"]: item for item in report["injections"]}
    nelm, exit7 = by_kind["nelm_step2"], by_kind["exit7_step3"]
    assert nelm["step"] == 2 and nelm["vasp_status"] == "0"
    assert nelm["validity_records"][0]["classification"] == "invalid_convergence"
    assert nelm["bridge_status_to_uspex"] == [str(protocol.EXIT_INVALID_CONVERGENCE)]
    assert exit7["bridge_status_to_uspex"] == ["7"]
    assert exit7["step"] == 3 and exit7["vasp_status"] == "0" and exit7["reported_status"] == "7"
    assert exit7["validity_records"][0]["classification"] == "execution_failed"
    assert nelm["individual"] != exit7["individual"]
    for item in (nelm, exit7):
        observations = item["observations"]
        assert observations["chain_stopped_at_failed_step"] is True
        assert observations["retried"] is False
        assert observations["listed_by_uspex"] is False  # the fake master rejects failed candidates
        assert observations["stale_energy_reuse"] is False
    # Every other candidate ran all three steps, uninjected, and is listed.
    other = {call["individual"] for call in analysis.read_calls(work / "probe_state")} - {
        nelm["individual"], exit7["individual"]}
    assert len(other) == 2
    listed = {row["id"] for row in report["individual_tables"]["results1/Individuals"] if "id" in row}
    assert listed == other
    # The bridge reported 118 for the NELM step, as the production path does.
    done_statuses = {fields["status"] for fields in analysis.read_done(work / "spool").values()}
    assert {"0", str(protocol.EXIT_INVALID_CONVERGENCE), "7"} == done_statuses
    # The injected CalcFold's INCAR was restored after VASP; the injected copy is kept.
    folder = work / "search" / "Calculation" / nelm["calcfold"]
    assert "NELM = 60" in (folder / "INCAR").read_text()
    assert "NELM = 3" in (folder / "INCAR.bmd_probe_injected").read_text()
    assert (work / "probe_state" / f"INCAR.injected.{nelm['calcfold']}").is_file()
    assert report["isolation"] == {"jobflow_imported": False, "atomate2_imported": False}


def test_status_only_mode_leaves_the_files_in_place(tmp_path):
    work, _ = _fake_probe(tmp_path, "exit7_step3", step_validity=False)
    report = analysis.analyse(work)
    assert [item["injection"] for item in report["injections"]] == ["exit7_step3"], _probe_evidence(work)
    folder = work / "search" / "Calculation" / report["injections"][0]["calcfold"]
    assert (folder / "OUTCAR").is_file() and (folder / "CONTCAR").is_file()
    assert report["validity_archive"] == []


def test_analysis_detects_stale_energy_reuse_and_retries(tmp_path):
    work, _ = _fake_probe(tmp_path, "nelm_step2,exit7_step3")
    calls = analysis.read_calls(work / "probe_state")
    injected = next(call for call in calls if call["injected"] == "nelm_step2")
    earlier = next(call for call in calls if call["individual"] == injected["individual"] and call["step"] == 1)
    # Pretend USPEX listed the rejected candidate with its step-1 energy, and ran its step 2 again.
    individuals = work / "search" / "results1" / "Individuals"
    fake_master = _module("fake_uspex_calcfolds", FIXTURES / "fake_uspex_calcfolds.py")
    individuals.write_text(individuals.read_text() + "\n"
                           + fake_master.individuals_row(injected["individual"], earlier["outcar_toten"]))
    with open(work / "probe_state" / "calls.tsv", "a") as handle:
        fields = [injected[c] if c != "injected" else "none" for c in analysis.CALLS_COLUMNS]
        handle.write("\t".join("" if v is None else str(v) for v in fields) + "\n")
    report = analysis.analyse(work)
    item = next(i for i in report["injections"] if i["injection"] == "nelm_step2")
    assert item["observations"]["stale_energy_reuse"] is True
    assert item["observations"]["retried"] is True
    assert item["runs_of_this_step"] == 2


@pytest.mark.parametrize(
    "work",
    [
        "/bmd-db/guest/flows/uspex_validity_probe_20261106_a",
        "/bmd-db/guest/logs/uspex_validity_probe_20261106_a",
        "/bmd-db/guest/validation/uspex_validity_probe_20261106_a",
        "/bmd-db/guest/uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/envs/uspex_validity_probe_20261106_a",
        "/bmd/shared/uspex_validity_probe_20261106_a",
        "/bmd/shared/uspex_cache/uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/validation/../envs/uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/validation/../uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/validation/./uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/validation//uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/validation/uspex_validity_probe_20261106_a/",
        "/bmd/bmdguest/validation/nested/uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/validation/bmd-compute-uspex-probe-20261106c",
        "/bmd/bmdguest/validation/si8_pilot",
        "/bmd/bmdguest/validation/uspex_validity_probe_",
        "/bmd/bmdguest/validation/uspex_validity_probe_a;rm -rf x",
        "/bmd/bmdguest/validation",
        "/bmd/bmdguest/validationx/uspex_validity_probe_20261106_a",
        "/bmd/bmdguest/envs/atomate2_remote/uspex_validity_probe_20261106_a",
        "/bmd/validation/uspex_validity_probe_20261106_a",
        "/bmd/uspex_validity_probe_20261106_a",
        "/tmp/uspex_validity_probe_20261106_a",
        "uspex_validity_probe_20261106_a",
    ],
)
def test_work_paths_outside_the_validation_root_are_refused(work):
    with pytest.raises(runner.PathRefused):
        runner.check_work_directory(work)


def test_the_preferred_work_directory_is_the_only_shape_accepted():
    assert runner.VALIDATION_ROOT == "/bmd/bmdguest/validation"
    assert runner.WORK_NAME_PATTERN.fullmatch("uspex_validity_probe_20261106_a")
    assert runner.SOURCE_NAME_PATTERN.fullmatch("bmd-compute-uspex-probe-20261106c")
    # Only the validation subdirectory of the protected /bmd/bmdguest passes the
    # production-area check; its siblings and the parent do not.
    assert runner._canonical("/bmd/bmdguest/validation/uspex_validity_probe_20261106_a", "w")
    for protected in ("/bmd/bmdguest", "/bmd/bmdguest/envs", "/bmd/bmdguest/envs/atomate2_remote/bin/python",
                      "/bmd/bmdguest/validation_old", "/bmd/shared/uspex_cache"):
        with pytest.raises(runner.PathRefused, match="production area"):
            runner._canonical(protected, "w")
    # The production Singularity bind is unchanged and covers the root.
    from backend.structure_search.bridge.master import DEFAULT_BIND

    assert DEFAULT_BIND == "/bmd"
    assert runner.VALIDATION_ROOT.startswith(DEFAULT_BIND + "/")
    assert "/bmd/bmdguest" in runner.PROTECTED_ROOTS and "/bmd/shared" in runner.PROTECTED_ROOTS


@pytest.fixture
def validation_root(tmp_path):
    root = tmp_path / "bmd" / "validation"
    root.mkdir(parents=True)
    return root


def test_fresh_work_directory_is_created_once_and_never_reused(validation_root):
    work = validation_root / "uspex_validity_probe_20261106_a"
    mode_before = validation_root.stat().st_mode
    plan = runner.check_work_directory(str(work), root=str(validation_root))
    assert plan == {"work": str(work), "root": str(validation_root), "root_state": "exists"}
    assert not work.exists()  # checking creates nothing
    assert runner.create_work_directory(str(work), root=str(validation_root)) == work
    assert work.is_dir() and not work.is_symlink()
    assert validation_root.stat().st_mode == mode_before  # no permission change
    with pytest.raises(runner.PathRefused, match="already exists"):
        runner.check_work_directory(str(work), root=str(validation_root))
    with pytest.raises(runner.PathRefused, match="already exists"):
        runner.create_work_directory(str(work), root=str(validation_root))
    (validation_root / "uspex_validity_probe_20261106_b").write_text("a file, not a directory")
    with pytest.raises(runner.PathRefused, match="already exists"):
        runner.check_work_directory(str(validation_root / "uspex_validity_probe_20261106_b"), root=str(validation_root))


def test_missing_root_is_created_only_where_this_user_may_write(tmp_path, monkeypatch):
    bmd = tmp_path / "bmd"
    bmd.mkdir()
    root = bmd / "validation"
    work = root / "uspex_validity_probe_20261106_a"
    assert runner.check_work_directory(str(work), root=str(root))["root_state"] == "will be created"
    runner.create_work_directory(str(work), root=str(root))
    assert root.is_dir() and work.is_dir()

    other = tmp_path / "locked"
    other.mkdir()
    real_access = os.access
    monkeypatch.setattr(runner.os, "access", lambda path, mode: False if str(path) == str(other) else real_access(path, mode))
    with pytest.raises(runner.PathRefused, match="cannot create it"):
        runner.check_work_directory(str(other / "validation" / "uspex_validity_probe_x"), root=str(other / "validation"))
    assert not (other / "validation").exists()


def test_unwritable_existing_root_is_refused(validation_root, monkeypatch):
    real_access = os.access
    monkeypatch.setattr(runner.os, "access",
                        lambda path, mode: False if str(path) == str(validation_root) else real_access(path, mode))
    with pytest.raises(runner.PathRefused, match="cannot create directories"):
        runner.check_work_directory(str(validation_root / "uspex_validity_probe_x"), root=str(validation_root))


def test_symlinked_ancestors_and_targets_are_refused(tmp_path):
    real = tmp_path / "real_validation"
    real.mkdir()
    linked_root = tmp_path / "validation"
    linked_root.symlink_to(real)
    with pytest.raises(runner.PathRefused, match="symbolic link"):
        runner.check_work_directory(str(linked_root / "uspex_validity_probe_x"), root=str(linked_root))

    parent_real = tmp_path / "elsewhere"
    (parent_real / "validation").mkdir(parents=True)
    linked_parent = tmp_path / "bmd_link"
    linked_parent.symlink_to(parent_real)
    root = linked_parent / "validation"
    with pytest.raises(runner.PathRefused, match="symbolic link"):
        runner.check_work_directory(str(root / "uspex_validity_probe_x"), root=str(root))

    root = tmp_path / "bmd" / "validation"
    root.mkdir(parents=True)
    dangling = root / "uspex_validity_probe_dangling"
    dangling.symlink_to(tmp_path / "nowhere")
    with pytest.raises(runner.PathRefused):
        runner.check_work_directory(str(dangling), root=str(root))
    with pytest.raises(runner.PathRefused):
        runner.create_work_directory(str(dangling), root=str(root))
    assert not (tmp_path / "nowhere").exists()


def test_source_tree_must_be_a_real_directory_beneath_the_root(validation_root, tmp_path):
    source = validation_root / "bmd-compute-uspex-probe-20261106"
    source.mkdir()
    assert runner.check_source_directory(str(source), root=str(validation_root)) == str(source)
    for bad in (validation_root / "bmd-compute", validation_root / "nested" / "bmd-compute-uspex-probe-x"):
        with pytest.raises(runner.PathRefused):
            runner.check_source_directory(str(bad), root=str(validation_root))
    link = validation_root / "bmd-compute-uspex-probe-link"
    link.symlink_to(source)
    with pytest.raises(runner.PathRefused, match="symbolic link"):
        runner.check_source_directory(str(link), root=str(validation_root))
    for production in ("/bmd/bmdguest/bmd-compute-uspex-probe-x", "/bmd-db/guest/validation/bmd-compute-uspex-probe-x",
                       "/bmd/bmdguest/envs/bmd-compute-uspex-probe-x", "/bmd/validation/bmd-compute-uspex-probe-x"):
        with pytest.raises(runner.PathRefused):
            runner.check_source_directory(production)


def test_runner_input_is_the_small_si8_search(tmp_path):
    runner.write_input(tmp_path, Path("/bmd/x/bmd_bridge_pilot.sh"), runner.PARALLEL)
    text = (tmp_path / "INPUT.txt").read_text()
    for line in ("4 : populationSize", "4 : initialPopSize", "2 : numGenerations", "-1 : whichCluster",
                 "2 : numParallelCalcs", "0.0 : ExternalPressure"):
        assert line in text.splitlines()


def test_job_script_uses_the_approved_account_and_fixed_interpreter():
    text = (PROBE / "validity_probe.sbatch").read_text()
    assert "case \"$WORK\" in /bmd/bmdguest/validation/uspex_validity_probe_*)" in text
    assert "case \"$REPO\" in /bmd/bmdguest/validation/bmd-compute-uspex-probe-*)" in text
    assert "/bmd/validation/" not in text and "/bmd/validation/" not in (PROBE / "preflight.sh").read_text()
    assert "--check-paths" in (PROBE / "preflight.sh").read_text()
    for line in ("#SBATCH -p leeburton-pool", "#SBATCH --account=power-leeburton-users_v2",
                 "#SBATCH --qos=owner100d", "#SBATCH -n 8", "#SBATCH --mem=24G", "#SBATCH -t 01:30:00",
                 "PYTHON=/bmd/bmdguest/envs/atomate2_remote/bin/python"):
        assert line in text.splitlines()
    assert "/usr/bin/python3" not in text
    assert "PYTHON=${PYTHON" not in text
    assert re.search(r"unset JOBFLOW_CONFIG_FILE ATOMATE2_CONFIG_FILE", text)
    from backend import config

    assert (config.DEFAULT_ACCOUNT, config.DEFAULT_PARTITION, config.DEFAULT_QOS) == (
        "power-leeburton-users_v2", "leeburton-pool", "owner100d")
    assert config.DEFAULT_REMOTE_PYTHON == runner.EXPECTED_PYTHON
    for script in ("validity_probe.sbatch", "preflight.sh"):
        subprocess.run(["bash", "-n", str(PROBE / script)], check=True)
    subprocess.run(["sh", "-n", str(PROBE / "inject_vasp.sh")], check=True)


# -- private USPEX cache (bmdguest cannot write the shared cache's uspex_sfx) -------------


@pytest.fixture
def private_cache(validation_root):
    cache = validation_root / "uspex_cache_private"
    cache.mkdir()
    return cache


def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("BMD_"):
            monkeypatch.delenv(name)


def test_probe_settings_differ_from_production_only_in_the_private_cache(private_cache, validation_root, monkeypatch):
    from backend import config
    from backend.structure_search.runtime import SearchRuntimeSettings

    _clean_env(monkeypatch)
    assert runner.PRIVATE_CACHE_DIR == "/bmd/bmdguest/validation/uspex_cache_private"
    production = SearchRuntimeSettings.from_config()
    assert production.cache_dir == config.USPEX_CACHE_DIR == "/bmd/bmdguest/uspex_cache"  # the approved production cache
    probe = runner.probe_runtime_settings(cache=str(private_cache), root=str(validation_root))
    assert probe.cache_dir == str(private_cache)
    for field in dataclasses.fields(production):
        if field.name != "cache_dir":
            assert getattr(probe, field.name) == getattr(production, field.name), field.name
    assert str(probe.ready_marker).startswith(str(private_cache) + "/uspex_sfx/")
    # The default is the private cache, without any environment variable.
    assert runner.probe_runtime_settings(check_cache=False).cache_dir == runner.PRIVATE_CACHE_DIR


@pytest.mark.parametrize("name", ["BMD_USPEX_CACHE_DIR", "BMD_USPEX_SANDBOX_DIR", "BMD_ANYTHING"])
def test_inherited_bmd_variables_are_refused(monkeypatch, name):
    monkeypatch.setenv(name, "/somewhere")
    with pytest.raises(SystemExit, match=name):
        runner.probe_runtime_settings(check_cache=False)


def test_private_cache_must_be_a_writable_real_directory_in_the_validation_area(private_cache, validation_root, tmp_path, monkeypatch):
    assert runner.check_private_cache(str(private_cache), root=str(validation_root)) == str(private_cache)
    for bad in ("/bmd/shared/uspex_cache", "/bmd/bmdguest/uspex_cache", "/bmd/bmdguest/uspex_cache_private",
                "/bmd/bmdguest/envs/uspex_cache_private", "/bmd/bmdguest/validation/../uspex_cache_private"):
        with pytest.raises(runner.PathRefused):
            runner.check_private_cache(bad)
    with pytest.raises(runner.PathRefused, match="directly beneath"):
        runner.check_private_cache(str(private_cache / "nested"), root=str(validation_root))
    with pytest.raises(runner.PathRefused, match="not an existing directory"):
        runner.check_private_cache(str(validation_root / "missing_cache"), root=str(validation_root))
    link = validation_root / "cache_link"
    link.symlink_to(private_cache)
    with pytest.raises(runner.PathRefused, match="symbolic link"):
        runner.check_private_cache(str(link), root=str(validation_root))
    real_access = os.access
    monkeypatch.setattr(runner.os, "access", lambda p, m: False if str(p) == str(private_cache) else real_access(p, m))
    with pytest.raises(runner.PathRefused, match="not writable"):
        runner.check_private_cache(str(private_cache), root=str(validation_root))
    monkeypatch.setattr(runner.os, "access", real_access)
    (private_cache / "uspex_sfx").symlink_to(tmp_path)
    with pytest.raises(runner.PathRefused, match="symbolic link"):
        runner.check_private_cache(str(private_cache), root=str(validation_root))


def _settings_with_program(tmp_path, private_cache, header: str):
    from backend.structure_search.runtime import SearchRuntimeSettings

    sandbox = tmp_path / "sandbox"
    program = sandbox / "opt" / "uspex" / "uspex"
    program.parent.mkdir(parents=True)
    program.write_text(header)
    base = SearchRuntimeSettings.from_config()
    return dataclasses.replace(base, sandbox_dir=str(sandbox), cache_dir=str(private_cache))


def _header(sha, version="25.0"):
    return f"#!/bin/sh\n# USPEX v{version} self-extractor\nPAYLOAD_SHA256={sha}\nexit 0\n"


def test_cache_initialisation_uses_the_production_unpack_after_checking_sha_and_version(tmp_path, private_cache, monkeypatch):
    from backend import config

    settings = _settings_with_program(tmp_path, private_cache, _header(config.USPEX_PAYLOAD_SHA256))
    calls = []

    def fake_unpack(received):
        calls.append(received)
        return {"status": "unpacked"}

    result = runner.init_private_cache(settings, unpack=fake_unpack)
    assert calls == [settings] and calls[0].cache_dir == str(private_cache)
    assert result["unpack"] == {"status": "unpacked"}
    assert result["header"]["banner_version"] == "25.0"
    # The harness writes no marker of its own; without a real unpack the check fails.
    assert not settings.ready_marker.exists()
    assert result["runtime_check"]["status"] == "failed"
    assert any(".ready" in problem or "not unpacked" in problem for problem in result["runtime_check"]["problems"])


@pytest.mark.parametrize(
    ("sha", "version", "message"),
    [("0" * 64, "25.0", "payload SHA-256"), (None, "24.0", "reports USPEX 24.0")],
)
def test_cache_initialisation_refuses_a_different_program(tmp_path, private_cache, sha, version, message):
    from backend import config
    from backend.structure_search.runtime import SearchRuntimeSetupError

    settings = _settings_with_program(tmp_path, private_cache, _header(sha or config.USPEX_PAYLOAD_SHA256, version))

    def must_not_unpack(_):
        raise AssertionError("unpack must not start for a different program")

    with pytest.raises(SearchRuntimeSetupError, match=message):
        runner.init_private_cache(settings, unpack=must_not_unpack)


def test_cache_initialisation_with_the_real_unpack_waits_for_the_extractors_marker(tmp_path, private_cache):
    """unpack_once (production) with a fake extractor process that writes .ready as USPEX's does."""

    from backend import config
    from backend.structure_search import runtime

    settings = _settings_with_program(tmp_path, private_cache, _header(config.USPEX_PAYLOAD_SHA256))

    def fake_popen(argv, env, cwd, **kwargs):
        assert env["SINGULARITYENV_XDG_CACHE_HOME"] == str(private_cache)
        assert argv[argv.index("-B") + 1] == str(private_cache)
        unpacked = settings.unpacked_dir / "uspex_core"
        unpacked.mkdir(parents=True)
        (unpacked / "uspex").write_text("program")
        settings.ready_marker.write_text("")
        return subprocess.Popen(["true"])

    result = runner.init_private_cache(
        settings, unpack=lambda s: runtime.unpack_once(s, poll_seconds=0.01, popen=fake_popen)
    )
    assert result["unpack"]["status"] == "unpacked"
    assert settings.ready_marker.is_file()
    assert runner.init_private_cache(settings, unpack=runtime.unpack_once)["unpack"]["status"] == "already_unpacked"


def test_runtime_record_digest_is_stable_and_sensitive(private_cache, validation_root, monkeypatch):
    from backend.structure_search.runtime import prepared_search_runtime

    _clean_env(monkeypatch)
    settings = runner.probe_runtime_settings(cache=str(private_cache), root=str(validation_root))
    record = prepared_search_runtime(settings)
    digest = runner.runtime_record_sha256(record)
    assert digest == runner.runtime_record_sha256(json.loads(json.dumps(record)))
    shared = prepared_search_runtime(dataclasses.replace(settings, cache_dir="/bmd/shared/uspex_cache"))
    assert runner.runtime_record_sha256(shared) != digest


def test_preflight_cache_initialisation_and_job_build_the_same_settings():
    clear = "for name in $(env | sed -n 's/^\\(BMD_[A-Za-z0-9_]*\\)=.*/\\1/p'); do unset \"$name\"; done"
    scripts = {name: (PROBE / name).read_text() for name in ("preflight.sh", "init_private_cache.sh", "validity_probe.sbatch")}
    for name, text in scripts.items():
        assert clear in text, name
        assert "PYTHON=/bmd/bmdguest/envs/atomate2_remote/bin/python" in text, name
        assert "backend.structure_search.runtime check" not in text, name  # no second settings path
    assert "--check-runtime" in scripts["preflight.sh"]
    assert "--init-cache" in scripts["init_private_cache.sh"]
    assert '--runtime-sha256 "$PROBE_RUNTIME_SHA256"' in scripts["validity_probe.sbatch"]
    assert "env -u BMD_" not in scripts["preflight.sh"]
    source = (PROBE / "run_validity_probe.py").read_text()
    assert source.count("probe_runtime_settings()") >= 3  # init-cache, check-runtime and the run itself
    for script in ("init_private_cache.sh",):
        subprocess.run(["bash", "-n", str(PROBE / script)], check=True)


def test_production_bind_and_runtime_code_are_unchanged():
    from backend.structure_search.bridge.master import DEFAULT_BIND
    from backend.structure_search import runtime

    assert DEFAULT_BIND == "/bmd"
    assert runner.PRIVATE_CACHE_DIR.startswith(DEFAULT_BIND + "/")
    assert runtime.SearchRuntimeSettings.from_config().cache_dir == "/bmd/bmdguest/uspex_cache"
