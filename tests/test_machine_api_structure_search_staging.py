"""Remote staging of structure-search attempts (USPEX step 4C), against a fake POWER.

The fake POWER is a directory tree on local disk: every absolute remote path
maps below a temporary root. It accepts exactly the remote operations staging
uses (exclusive mkdir/rmdir locks, directory creation, atomic uploads, file
reads) and runs the staging script's POWER-side function in-process against
that tree, with test POTCAR datasets whose identities match the SYNTHETIC
ENMAX metadata the tests plan with. Nothing contacts POWER, submits a job or
runs USPEX or VASP.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shlex
import stat
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from backend.paramiko_remote import ParamikoRemoteRunner
from backend.remote import RemoteCommandResult, RemotePathInfo, SubmissionAttemptError
from backend.structure_search import incar_policy, staging
from backend.structure_search.search_plan import (
    PLAN_FILE,
    UNBOUND_BRIDGE_PATH,
    load_plan_text,
    render_prepared_files,
    verify_execution_binding,
)
from backend.structure_search.runtime import SearchRuntimeSettings
from compute_api import execution
from test_machine_api_execution import (  # noqa: F401 - pytest fixtures
    attempt_body,
    bearer,
    client,
    new_id,
    power,
    put as put_calculation,
    state_dir,
    tokens,
)
from test_machine_api_structure_search_attempts import (
    attempt_dir,
    put_search,
    get_search,
    search_body,
    search_record,
    search_record_path,
    write_search_record,
)


WORKSPACE = "/bmd/bmdguest/searches"
SEARCH_LOGS = "/bmd-db/guest/logs/structure_search"
LIBRARY = "/bmd-db/potcars"


def potcar_text(symbol, *, tampered=False):
    return f"SYNTHETIC {symbol}\nsynthetic:{symbol}{'!' if tampered else ''}"


def fake_identity(path):
    """Identity of a test POTCAR, matching tests/conftest.py's SYNTHETIC metadata."""

    title, data = Path(path).read_text().split("\n", 1)
    return {"TITEL": title, "ENMAX": 400.0, "sha256": hashlib.sha256(data.encode()).hexdigest()}


class FakePower:
    def __init__(self, root: Path):
        self.root = root
        self.commands: list[str] = []
        self.connections = 0
        self.sbatch_calls = 0
        self.fail_finalize = 0
        self.fail_inventory = 0
        self.finalize_timeout = None  # "before" or "after": the finalize command times out (once)
        self.timeouts: list[tuple[str, object]] = []  # (command kind, timeout_s) of every command
        self.fail_put = None  # (target suffix, "partial" | "before_chmod" | "before_rename")
        self.after_uploads = None

    def at(self, remote_path) -> Path:
        return self.root / str(remote_path).lstrip("/")

    def factory(self):
        return FakePowerRunner(self)

    def make_library(self, elements, *, tampered=()):
        from backend.structure_search.potcar import bmd_potcar_symbols

        for element, symbol in bmd_potcar_symbols(elements).items():
            path = self.at(f"{LIBRARY}/POT_PAW_PBE_64/{symbol}/POTCAR")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(potcar_text(symbol, tampered=element in tampered))

    def make_cache(self):
        marker = self.at(str(SearchRuntimeSettings.from_config().ready_marker))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")


class SimulatedCrash(Exception):
    """The connection or the service dies at this point."""


class FakePowerRunner(ParamikoRemoteRunner):
    def __init__(self, power: FakePower):
        super().__init__(client=object())
        self.power = power

    def connect(self, profile):
        self.power.connections += 1

    def close(self):
        return None

    def ensure_available(self):
        return None

    def _preparation_sftp_session(self):
        return nullcontext()

    def run(self, command, *, check=False, modules=False, export_env=False, timeout_s=None):
        self.power.commands.append(command)
        kind = ("finalize" if staging.REQUEST_MARKER in command else
                "inventory" if staging.INVENTORY_DIR_MARKER in command else command.split()[0])
        self.power.timeouts.append((kind, timeout_s))
        parts = shlex.split(command) if "\n" not in command else []
        if len(parts) == 2 and parts[0] in {"mkdir", "rmdir"}:
            try:
                (os.mkdir if parts[0] == "mkdir" else os.rmdir)(self.power.at(parts[1]))
            except OSError:
                return RemoteCommandResult(command=command, returncode=1)
            return RemoteCommandResult(command=command, returncode=0)
        if staging.INVENTORY_DIR_MARKER in command:
            if self.power.fail_inventory:
                self.power.fail_inventory -= 1
                raise SimulatedCrash("connection lost before listing the run directory")
            return self._run_inventory(command)
        if staging.REQUEST_MARKER in command:
            assert "bmd_install_verified_runtime_package(REQUEST['package_dir'], REQUEST['manifest'])" in command
            if self.power.fail_finalize:
                self.power.fail_finalize -= 1
                raise ConnectionError("connection lost during staging")
            request = staging.request_from_script(command)
            if self.power.finalize_timeout == "before":
                self.power.finalize_timeout = None
                raise TimeoutError("timed out")  # as Paramiko's stdout.read() on POWER
            result = staging.finalize_staged_search(
                request, filesystem_root=str(self.power.root), potcar_identity=fake_identity
            )
            if self.power.finalize_timeout == "after":  # POWER finished, but after the client gave up
                self.power.finalize_timeout = None
                raise TimeoutError("timed out")
            return RemoteCommandResult(command=command, returncode=0,
                                       stdout=staging.RESULT_MARKER + json.dumps(result) + "\n")
        raise AssertionError(f"unexpected remote command: {command[:80]}")

    def _run_inventory(self, command):
        """Run the generated inventory script itself, on the fake tree."""

        import contextlib
        import io

        script = command.split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0]
        remote_dir = staging.inventory_dir_from_script(script)
        script = script.replace(f"{staging.INVENTORY_DIR_MARKER}{remote_dir!r}",
                                f"{staging.INVENTORY_DIR_MARKER}{str(self.power.at(remote_dir))!r}")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(script, "<inventory>", "exec"), {"__name__": "__bmd_inventory__"})
        return RemoteCommandResult(command=command, returncode=0, stdout=out.getvalue())

    def ensure_directory(self, remote_path):
        self.power.at(remote_path).mkdir(parents=True, exist_ok=True)
        return RemotePathInfo(path=remote_path, exists=True, kind="dir")

    def is_file(self, remote_path):
        return self.power.at(remote_path).is_file()

    def is_dir(self, remote_path):
        return self.power.at(remote_path).is_dir()

    def read_text(self, remote_path, *, max_bytes=None):
        return self.power.at(remote_path).read_bytes().decode("utf-8")

    def _failing(self, remote_path, point):
        fail = self.power.fail_put
        name = posixpath.basename(remote_path)
        parsed = staging.parse_staging_temporary(name)
        target = posixpath.join(posixpath.dirname(remote_path), parsed["target"]) if parsed else remote_path
        if fail and fail[1] == point and target.endswith(fail[0]):
            self.power.fail_put = None
            return True
        return False

    def put_text(self, remote_path, text, *, mode=0o640):
        path = self.power.at(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode("utf-8")
        if self._failing(remote_path, "partial"):
            path.write_bytes(data[: len(data) // 2])
            raise SimulatedCrash("connection lost during an upload")
        path.write_bytes(data)
        if self._failing(remote_path, "before_chmod"):
            os.chmod(path, 0o644)
            raise SimulatedCrash("connection lost before the upload's mode was set")
        os.chmod(path, mode)
        if self.power.after_uploads and "/search/" in str(remote_path):
            hook, self.power.after_uploads = self.power.after_uploads, None
            hook()

    def _replace_file(self, source, destination):
        if self._failing(destination, "before_rename"):
            raise SimulatedCrash("connection lost before the upload was moved into place")
        assert staging.parse_staging_temporary(posixpath.basename(source))  # only attributable temporaries
        assert not self.power.at(destination).exists()  # never replaces a file
        os.replace(self.power.at(source), self.power.at(destination))

    def submit_batch(self, request):
        self.power.sbatch_calls += 1
        raise AssertionError("staging must never call sbatch")


@pytest.fixture
def cluster(tmp_path, monkeypatch, tokens, power):
    fake = FakePower(tmp_path / "power")
    fake.calculation_factory = power.factory
    for path in (WORKSPACE, SEARCH_LOGS):
        fake.at(path).mkdir(parents=True)
    fake.make_library(["Si", "Ga", "As", "Mg", "O", "Li"])
    fake.make_cache()
    monkeypatch.setattr(execution, "RUNNER_FACTORY", fake.factory)
    return fake


def put_calculation_attempt(client, tokens, cluster, attempt):
    """Create a calculation attempt through the calculation fake, then restore the staging fake."""
    previous = execution.RUNNER_FACTORY
    execution.RUNNER_FACTORY = cluster.calculation_factory
    try:
        return put_calculation(client, tokens["runner"], attempt, attempt_body(client, tokens, submit=False))
    finally:
        execution.RUNNER_FACTORY = previous


def prepared_attempt(client, tokens, formula="Si"):
    attempt = new_id()
    body = search_body(client, tokens, formula)
    response = put_search(client, tokens["preparer"], attempt, body)
    assert response.status_code == 200, response.text
    return attempt, body["expected_plan_digest"]


def stage(client, token, attempt, digest):
    return client.put(f"/api/v1/structure-search-attempts/{attempt}/staging", headers=bearer(token),
                      json={"expected_plan_digest": digest})


def tree(root: Path):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns, stat.S_IMODE(path.stat().st_mode))
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def staged_run(cluster, state_dir, attempt):
    run_name = search_record(state_dir, attempt)["staging"]["run_name"]
    return cluster.at(f"{WORKSPACE}/{run_name}"), run_name


# --- successful staging ------------------------------------------------------------


def test_staging_writes_a_verified_bound_search_on_power(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens, "GaAs")
    local = attempt_dir(state_dir, attempt) / "preparation-1"
    local_before = tree(local)

    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"] == "STAGED"
    assert payload["submission_available"] is False and payload["remote_staging_available"] is False
    staged = payload["staging"]
    assert staged["remote_state"] == "STAGED" and staged["bridge_path_bound"] is True
    assert staged["contains_potcar_contents"] is False
    assert [row["element"] for row in staged["potcars"]] == ["Ga", "As"]
    assert all(row["verified"] for row in staged["potcars"])
    assert "synthetic:" not in response.text  # no POTCAR content
    assert WORKSPACE not in response.text and "/bmd-db" not in response.text  # no remote paths

    run_dir, run_name = staged_run(cluster, state_dir, attempt)
    assert run_name.startswith("uspex_search-") and attempt[:8] in run_name
    bridge = f"{WORKSPACE}/{run_name}/backend/structure_search/bridge/bmd_bridge.sh"
    plan = load_plan_text((local / PLAN_FILE).read_text())
    local_manifest = json.loads((local / "prepared_inputs.json").read_text())
    expected = render_prepared_files(plan, bridge_path=bridge, generator=local_manifest["generator"])
    for name, data in expected.items():
        assert (run_dir / "search" / name).read_bytes() == data, name
    assert bridge in (run_dir / "search" / "INPUT.txt").read_text()
    assert UNBOUND_BRIDGE_PATH not in (run_dir / "search" / "INPUT.txt").read_text()
    assert json.loads((run_dir / "search" / "prepared_inputs.json").read_text())["bindings"] == {
        "bridge_path": bridge, "bound": True}
    assert hashlib.sha256((run_dir / bridge.split(run_name + "/")[1]).read_bytes()).hexdigest() == \
        plan.record["execution"]["bridge"]["sha256"]
    for element in ("Ga", "As"):
        potcar = run_dir / "search" / "Specific" / f"POTCAR_{element}"
        from backend.structure_search.potcar import bmd_potcar_symbol

        assert potcar.read_text() == potcar_text(bmd_potcar_symbol(element))
        assert stat.S_IMODE(potcar.stat().st_mode) == 0o600
    verify_execution_binding(run_dir / "search", bridge_path=bridge, filesystem_root=str(cluster.root),
                             require_current=False, potcars="staged", potcar_identity=fake_identity)

    state = json.loads(cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json").read_text())
    assert state["state"] == "STAGED" and state["attempt_id"] == attempt
    assert state["run_dir"] == f"{WORKSPACE}/{run_name}"
    submission = json.loads((run_dir / "submission.json").read_text())
    assert submission["submission"]["attempt_id"] == attempt
    assert submission["structure_search"]["plan_digest"] == digest
    assert submission["structure_search"]["submission_available"] is False
    assert submission["submission"]["attempt_fingerprint"] == state["fingerprint"]
    assert not cluster.at("/bmd-db/guest/logs/submission_attempts").exists()  # calculations untouched

    assert tree(local) == local_before  # the local preparation is never changed
    record = search_record(state_dir, attempt)
    assert record["state"] == "STAGED" and record["staging"]["result"]["remote_state"] == "STAGED"
    assert cluster.sbatch_calls == 0


def test_staging_is_idempotent_and_read_back_needs_no_remote_contact(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    first = stage(client, tokens["preparer"], attempt, digest)
    connections = cluster.connections
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    files = tree(run_dir)
    second = stage(client, tokens["preparer"], attempt, digest)
    assert second.status_code == 200 and second.json() == first.json()
    read_back = get_search(client, tokens["preparer"], attempt)
    assert read_back.status_code == 200 and read_back.json() == first.json()
    assert cluster.connections == connections
    assert tree(run_dir) == files
    # Step 4B's PUT keeps returning the stored result for a staged attempt.
    body = search_body(client, tokens)
    assert put_search(client, tokens["preparer"], attempt, body).json()["state"] == "STAGED"


# --- refusals before any remote work ---------------------------------------------


def test_staging_refusals(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    assert stage(client, tokens["planner"], attempt, digest).status_code == 403  # no prepare scope
    assert stage(client, tokens["other"], attempt, digest).status_code == 403
    response = stage(client, tokens["preparer"], attempt, "sha256:" + "0" * 64)
    assert response.status_code == 409 and response.json()["error"]["code"] == "plan_digest_mismatch"
    assert stage(client, tokens["preparer"], new_id(), digest).status_code == 404
    response = client.put(f"/api/v1/structure-search-attempts/{attempt}/staging",
                          headers=bearer(tokens["preparer"]), json={"expected_plan_digest": digest, "submit": True})
    assert response.status_code == 422
    assert {"field": "submit", "problem": "submission_not_available"} in response.json()["error"]["fields"]
    assert cluster.connections == 0
    assert search_record(state_dir, attempt)["state"] == "PREPARED_LOCALLY"


def test_stale_policy_bridge_or_package_is_refused_before_contacting_power(client, tokens, state_dir, cluster,
                                                                           monkeypatch):
    attempt, digest = prepared_attempt(client, tokens)
    with monkeypatch.context() as patch:
        patch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
        response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "plan_changed"

    import backend.submission as submission

    real_sources = submission.build_backend_module_sources

    def altered_bridge():
        sources = dict(real_sources())
        sources[staging.BRIDGE_RELATIVE_PATH] += "\n# changed\n"
        return sources

    with monkeypatch.context() as patch:
        patch.setattr(submission, "build_backend_module_sources", altered_bridge)
        response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "bridge_changed"
    assert cluster.connections == 0
    assert search_record(state_dir, attempt)["state"] == "PREPARED_LOCALLY"


def test_a_calculation_attempt_id_cannot_be_staged(client, tokens, state_dir, cluster):
    calculation = new_id()
    assert put_calculation_attempt(client, tokens, cluster, calculation).status_code == 200
    response = stage(client, tokens["runner"], calculation, "sha256:" + "0" * 64)
    assert response.status_code == 409 and response.json()["error"]["code"] == "attempt_kind_mismatch"


# --- POTCAR identity --------------------------------------------------------------


def test_wrong_power_potcar_is_refused_and_never_staged(client, tokens, state_dir, cluster):
    cluster.make_library(["Si"], tampered={"Si"})
    attempt, digest = prepared_attempt(client, tokens)
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "potcar_identity_mismatch"
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    assert not (run_dir / "search" / "Specific" / "POTCAR_Si").exists()
    assert search_record(state_dir, attempt)["state"] == "STAGING"
    state = json.loads(cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json").read_text())
    assert state["state"] == "STAGING"  # never claimed as staged

    cluster.make_library(["Si"])  # the library is repaired
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED", response.text


def test_uninitialised_cache_is_refused(client, tokens, state_dir, cluster):
    marker = cluster.at(str(SearchRuntimeSettings.from_config().ready_marker))
    marker.unlink()
    attempt, digest = prepared_attempt(client, tokens)
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "uspex_cache_not_ready"
    assert search_record(state_dir, attempt)["state"] == "STAGING"


# --- interruptions -----------------------------------------------------------------


def test_interrupted_staging_resumes_in_the_same_run_directory_without_rewriting(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_finalize = 1
    failed = stage(client, tokens["preparer"], attempt, digest)
    assert failed.status_code == 503
    assert search_record(state_dir, attempt)["state"] == "STAGING"
    run_dir, run_name = staged_run(cluster, state_dir, attempt)
    uploaded = tree(run_dir)
    assert not (run_dir / "search" / "Specific" / "POTCAR_Si").exists()
    assert not cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.lock").exists()  # claim released

    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED"
    assert staged_run(cluster, state_dir, attempt)[1] == run_name
    after = tree(run_dir)
    assert {name: after[name] for name in uploaded} == uploaded  # nothing rewritten
    assert sorted(path.name for path in cluster.at(WORKSPACE).iterdir()) == [run_name]


def test_remote_staged_before_local_record_is_adopted_read_only(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    stage(client, tokens["preparer"], attempt, digest)
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    files = tree(run_dir)
    record = search_record(state_dir, attempt)
    record["state"] = "STAGING"  # as if the service stopped before recording success
    record["staging"].update(staged_at=None, result=None)
    write_search_record(state_dir, record)
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED"
    assert tree(run_dir) == files  # re-verified, nothing written


def test_damaged_partial_staging_is_never_overwritten(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_finalize = 1
    stage(client, tokens["preparer"], attempt, digest)
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    incar = run_dir / "search" / "Specific" / "INCAR_2"
    incar.write_text(incar.read_text() + "EXTRA = 1\n")
    damaged = incar.read_bytes()
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "staging_damaged"
    assert incar.read_bytes() == damaged
    assert search_record(state_dir, attempt)["state"] == "STAGING"


def test_runtime_package_change_during_interrupted_staging_is_refused(client, tokens, state_dir, cluster,
                                                                     monkeypatch):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_finalize = 1
    stage(client, tokens["preparer"], attempt, digest)
    import backend.submission as submission

    real_sources = submission.build_backend_module_sources
    monkeypatch.setattr(submission, "build_backend_module_sources",
                        lambda: {**real_sources(), "structure_search/extra_module.py": "X = 1\n"})
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "runtime_package_changed"


def test_existing_run_directory_is_never_reused(client, tokens, state_dir, cluster, monkeypatch):
    import backend.submission as submission

    monkeypatch.setattr(submission, "canonical_run_timestamp", lambda value=None: "20261107-120000")
    attempt, digest = prepared_attempt(client, tokens)
    occupied = cluster.at(f"{WORKSPACE}/uspex_search-20261107-120000-{attempt[:8]}")
    occupied.mkdir()
    (occupied / "keep").write_text("someone else's")
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "run_directory_exists"
    assert (occupied / "keep").read_text() == "someone else's"
    assert sorted(path.name for path in occupied.iterdir()) == ["keep"]


# --- STAGED is not PREPARED -------------------------------------------------------


def test_existing_submission_machinery_refuses_a_staged_search(client, tokens, state_dir, cluster):
    from backend.remote import SubmissionAttemptNotPrepared

    attempt, digest = prepared_attempt(client, tokens)
    stage(client, tokens["preparer"], attempt, digest)
    spec = search_record(state_dir, attempt)["staging"]["spec"]
    state_path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    before = state_path.read_bytes()
    runner = cluster.factory()
    with pytest.raises(SubmissionAttemptNotPrepared):
        runner.submit(spec, dry_run=False)
    with pytest.raises(SubmissionAttemptError):
        runner._submit_prepared_submission_attempt(spec, json.loads(before), "")
    with pytest.raises((SubmissionAttemptError, KeyError)):
        runner.submit(spec, dry_run=True)  # the calculation preparation path cannot re-prepare it either
    assert state_path.read_bytes() == before
    assert cluster.sbatch_calls == 0


def test_staging_runs_no_scientific_program(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens, "Li2O")
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 200
    for command in cluster.commands:
        first = command.split()[0]
        if staging.INVENTORY_DIR_MARKER in command:
            assert "<<'PY'" in command and staging.REQUEST_MARKER not in command
        else:
            assert first in {"mkdir", "rmdir", "cd"}, command[:60]
        lowered = command.lower()
        for program in ("sbatch", "srun", "singularity", "vasp_std", "mpirun", "/opt/uspex"):
            assert program not in lowered.replace("bmd_bridge", ""), program
    assert cluster.sbatch_calls == 0


# --- rollback ------------------------------------------------------------------------


def test_staging_leaves_calculation_state_and_v120_reader_untouched(client, tokens, state_dir, cluster):
    from test_machine_api_structure_search_attempts import _v120_ledger_module

    calculation = new_id()
    assert put_calculation_attempt(client, tokens, cluster, calculation).status_code == 200
    before = (state_dir / "attempts" / f"{calculation}.json").read_bytes()
    attempt, digest = prepared_attempt(client, tokens)
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 200
    v120 = _v120_ledger_module().AttemptLedger.from_environment()
    assert [record["attempt_id"] for record in v120.records()] == [calculation]
    assert (state_dir / "attempts" / f"{calculation}.json").read_bytes() == before
    assert search_record_path(state_dir, attempt).exists()


# --- interrupted uploads (finalization) ------------------------------------------------


def hidden(run_dir: Path):
    return sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob(".*"))


def remote_state(cluster, attempt):
    return json.loads(cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json").read_text())


UPLOAD_TARGETS = {
    "submission": "submission.json",
    "package_module": "backend/structure_search/staging.py",
    "bridge": "backend/structure_search/bridge/bmd_bridge.sh",
    "search_input": "search/INPUT.txt",
    "search_manifest": "search/prepared_inputs.json",
}


@pytest.mark.parametrize("point", ["partial", "before_chmod", "before_rename"])
@pytest.mark.parametrize("target", sorted(UPLOAD_TARGETS))
def test_interrupted_upload_is_recovered_from_its_own_temporary(client, tokens, state_dir, cluster, point, target):
    relative = UPLOAD_TARGETS[target]
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_put = (relative, point)
    failed = stage(client, tokens["preparer"], attempt, digest)
    assert failed.status_code == 503, failed.text
    assert search_record(state_dir, attempt)["state"] == "STAGING"
    assert remote_state(cluster, attempt)["state"] == "STAGING"
    run_dir, run_name = staged_run(cluster, state_dir, attempt)
    assert not (run_dir / relative).exists()
    leftovers = hidden(run_dir)
    assert len(leftovers) == 1
    parsed = staging.parse_staging_temporary(Path(leftovers[0]).name)
    assert parsed["attempt"] == attempt and parsed["target"] == Path(relative).name
    completed = {name: entry for name, entry in tree(run_dir).items() if not Path(name).name.startswith(".")}

    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED", response.text
    after = tree(run_dir)
    assert {name: after[name] for name in completed} == completed  # completed files never rewritten
    assert hidden(run_dir) == []  # the temporary was completed, nothing is left behind
    assert stat.S_IMODE((run_dir / relative).stat().st_mode) == 0o640
    assert remote_state(cluster, attempt)["state"] == "STAGED"
    assert sorted(path.name for path in cluster.at(WORKSPACE).iterdir()) == [run_name]


def test_interruption_after_reserving_the_run_name_creates_the_directory_on_retry(
        client, tokens, state_dir, cluster, monkeypatch):
    from compute_api import structure_search_execution as sse

    original = sse._RemoteStaging.create_run_directory
    calls = {"n": 0}

    def crash_once(self, runner):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SimulatedCrash("connection lost after the remote state was written")
        return original(self, runner)

    monkeypatch.setattr(sse._RemoteStaging, "create_run_directory", crash_once)
    attempt, digest = prepared_attempt(client, tokens)
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 503
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    assert remote_state(cluster, attempt)["state"] == "STAGING" and not run_dir.exists()
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED", response.text


def test_interruption_before_any_upload_adopts_the_empty_reserved_directory(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_inventory = 1
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 503
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    assert run_dir.is_dir() and not any(run_dir.iterdir())
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED", response.text


def _crash_once(monkeypatch, name, *, when):
    real = getattr(os, name)
    state = {"done": False}

    def wrapper(*args, **kwargs):
        if not state["done"] and when(*args):
            state["done"] = True
            if name == "unlink":
                raise SimulatedCrash(f"service stopped before os.{name}")
            raise SimulatedCrash(f"service stopped during os.{name}")
        return real(*args, **kwargs)

    monkeypatch.setattr(os, name, wrapper)


def test_interrupted_potcar_copy_is_superseded_on_retry(client, tokens, state_dir, cluster, monkeypatch):
    attempt, digest = prepared_attempt(client, tokens, "GaAs")
    _crash_once(monkeypatch, "link", when=lambda source, destination, *rest: "POTCAR_As" in str(destination))
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 503
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    specific = run_dir / "search" / "Specific"
    assert (specific / "POTCAR_Ga").exists() and not (specific / "POTCAR_As").exists()
    leftover = [path.name for path in specific.iterdir() if path.name.startswith(".")]
    assert len(leftover) == 1 and staging.parse_staging_temporary(leftover[0])["attempt"] == attempt
    (specific / leftover[0]).write_text("half a POTCAR")  # as if the copy itself was cut short
    ga = tree(specific)["POTCAR_Ga"]

    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED", response.text
    assert hidden(run_dir) == []
    assert tree(specific)["POTCAR_Ga"] == ga  # the completed copy was not touched
    assert fake_identity(specific / "POTCAR_As")["sha256"] == fake_identity(
        cluster.at(f"{LIBRARY}/POT_PAW_PBE_64/As/POTCAR"))["sha256"]


def test_potcar_copy_interrupted_after_its_link_is_completed(client, tokens, state_dir, cluster, monkeypatch):
    attempt, digest = prepared_attempt(client, tokens)
    _crash_once(monkeypatch, "unlink", when=lambda path, *rest: "POTCAR_Si.bmdstage-" in str(path))
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 503
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    specific = run_dir / "search" / "Specific"
    leftover = [path for path in specific.iterdir() if path.name.startswith(".")]
    assert len(leftover) == 1 and os.path.samefile(leftover[0], specific / "POTCAR_Si")
    potcar = (specific / "POTCAR_Si").read_bytes()
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED", response.text
    assert hidden(run_dir) == [] and (specific / "POTCAR_Si").read_bytes() == potcar


# --- ambiguous states are refused, never guessed ------------------------------------------


OTHER_ATTEMPT = "0f0f0f0f-1111-4222-8333-444455556666"


def _plant(case, run_dir: Path, attempt: str):
    search = run_dir / "search"
    if case == "other_attempt":
        path = search / staging.staging_temporary_name("INPUT.txt", OTHER_ATTEMPT, "ab" * 32)
    elif case == "unattributed_temporary":
        path = search / f".INPUT.txt.{'c' * 32}.tmp"
    elif case == "different_content":
        path = search / staging.staging_temporary_name("INPUT.txt", attempt, "ab" * 32)
    elif case == "temporary_beside_completed_file":
        expected = hashlib.sha256((run_dir / "submission.json").read_bytes()).hexdigest()
        path = run_dir / staging.staging_temporary_name("submission.json", attempt, expected)
    elif case == "temporary_for_unknown_target":
        path = search / staging.staging_temporary_name("NOTES.txt", attempt, "ab" * 32)
    elif case == "unexpected_file":
        path = run_dir / "notes.txt"
    elif case == "unexpected_directory":
        (run_dir / "extra").mkdir()
        return
    elif case == "symbolic_link":
        (search / "Specific" / "POTCAR_Si").symlink_to(cluster_library_path(run_dir))
        return
    path.write_text("not ours")


def cluster_library_path(run_dir: Path) -> Path:
    root = run_dir
    while root.name != "power":
        root = root.parent
    return root / LIBRARY.lstrip("/") / "POT_PAW_PBE_64" / "Si" / "POTCAR"


@pytest.mark.parametrize("case, code", [
    ("other_attempt", "staging_ambiguous"),
    ("unattributed_temporary", "staging_ambiguous"),
    ("different_content", "staging_ambiguous"),
    ("temporary_beside_completed_file", "staging_ambiguous"),
    ("temporary_for_unknown_target", "staging_ambiguous"),
    ("unexpected_file", "staging_damaged"),
    ("unexpected_directory", "staging_damaged"),
    ("symbolic_link", "staging_damaged"),
])
def test_ambiguous_or_foreign_state_is_refused_and_left_untouched(client, tokens, state_dir, cluster, case, code):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_put = ("search/INPUT.txt", "partial")  # INPUT.txt missing, its own partial temporary present
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 503
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    _plant(case, run_dir, attempt)
    before = tree(run_dir)
    links = sorted(str(path) for path in run_dir.rglob("*") if path.is_symlink())
    state_before = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json").read_bytes()

    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == code, response.text
    assert tree(run_dir) == before
    assert sorted(str(path) for path in run_dir.rglob("*") if path.is_symlink()) == links
    assert cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json").read_bytes() == state_before
    assert search_record(state_dir, attempt)["state"] == "STAGING"
    assert not cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.lock").exists()


def test_potcar_temporary_that_is_not_the_completed_copy_is_refused(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_finalize = 1
    stage(client, tokens["preparer"], attempt, digest)
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    specific = run_dir / "search" / "Specific"
    (specific / "POTCAR_Si").write_text(potcar_text("Si"))  # a completed copy ...
    plan_row = load_plan_text((run_dir / "search" / PLAN_FILE).read_text()).record["potcar"]["elements"][0]
    other = specific / staging.staging_temporary_name("POTCAR_Si", attempt, plan_row["dataset_sha256"])
    other.write_text(potcar_text("Si"))  # ... and a separate temporary file beside it
    before = tree(run_dir)
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "staging_ambiguous"
    assert tree(run_dir) == before


def test_a_claim_left_by_a_dead_request_is_refused_without_touching_anything(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_finalize = 1
    stage(client, tokens["preparer"], attempt, digest)
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    lock = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.lock")
    lock.mkdir()  # the service died while holding the claim
    before = tree(run_dir)
    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 409 and response.json()["error"]["code"] == "staging_in_progress"
    assert tree(run_dir) == before and lock.is_dir()
    lock.rmdir()  # an operator, having checked no request is running, removes it
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 200


# --- STAGED is the last verified state; fresh verification is explicit -------------------


def test_read_back_reports_last_verified_state_and_never_contacts_power(client, tokens, state_dir, cluster):
    attempt, digest = prepared_attempt(client, tokens)
    staged = stage(client, tokens["preparer"], attempt, digest).json()["staging"]
    assert staged["remote_integrity"] == "last_verified"
    assert staged["last_verified_at"] == staged["staged_at"]
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    (run_dir / "search" / "INPUT.txt").write_text("changed on POWER afterwards\n")
    connections = cluster.connections
    read_back = get_search(client, tokens["preparer"], attempt)
    assert read_back.status_code == 200 and read_back.json()["state"] == "STAGED"
    assert read_back.json()["staging"]["remote_integrity"] == "last_verified"
    assert read_back.json()["submission_available"] is False
    assert stage(client, tokens["preparer"], attempt, digest).json()["state"] == "STAGED"  # stored, no contact
    assert cluster.connections == connections


def test_fresh_reverification_is_read_only_and_detects_later_changes(client, tokens, state_dir, cluster):
    from compute_api.execution import ExecutionError
    from compute_api.structure_search_execution import reverify_staged_search

    attempt, digest = prepared_attempt(client, tokens, "GaAs")
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 200
    run_dir, _ = staged_run(cluster, state_dir, attempt)
    record_path = search_record_path(state_dir, attempt)
    state_path = cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json")
    snapshot = (tree(run_dir), record_path.read_bytes(), state_path.read_bytes())

    verification = reverify_staged_search(principal="preparer", attempt_id=attempt)
    assert verification["remote_state"] == "STAGED" and verification["plan_digest"] == digest
    assert verification["contains_potcar_contents"] is False
    assert [row["element"] for row in verification["potcars"]] == ["Ga", "As"]
    assert "synthetic:" not in json.dumps(verification)
    assert (tree(run_dir), record_path.read_bytes(), state_path.read_bytes()) == snapshot  # wrote nothing

    with pytest.raises(ExecutionError) as refused:
        reverify_staged_search(principal="runner", attempt_id=attempt)
    assert refused.value.status_code == 403

    for change, code in (
        (lambda: (run_dir / "notes.txt").write_text("x"), "staging_damaged"),
        (lambda: (run_dir / "search" / "Specific" / "POTCAR_As").write_text(potcar_text("As", tampered=True)),
         "potcar_identity_mismatch"),
        (lambda: (run_dir / "backend" / "structure_search" / "bridge" / "bmd_bridge.sh").write_text("#!/bin/sh\n"),
         "runtime_package_mismatch"),
    ):
        saved = tree(run_dir)
        change()
        with pytest.raises((ExecutionError, staging.StagingError)) as caught:
            reverify_staged_search(principal="preparer", attempt_id=attempt)
        assert getattr(caught.value, "code", None) == code, caught.value
        assert search_record(state_dir, attempt)["state"] == "STAGED"
        assert state_path.read_bytes() == snapshot[2]
        # restore for the next case
        for name in set(tree(run_dir)) - set(saved):
            (run_dir / name).unlink()
        for name, (data, _, mode) in saved.items():
            (run_dir / name).write_bytes(data)
            os.chmod(run_dir / name, mode)
    assert reverify_staged_search(principal="preparer", attempt_id=attempt)["remote_state"] == "STAGED"


def test_reverification_refuses_an_attempt_that_is_not_staged(client, tokens, state_dir, cluster):
    from compute_api.execution import ExecutionError
    from compute_api.structure_search_execution import reverify_staged_search

    attempt, digest = prepared_attempt(client, tokens)
    connections = cluster.connections
    with pytest.raises(ExecutionError) as refused:
        reverify_staged_search(principal="preparer", attempt_id=attempt)
    assert refused.value.code == "attempt_not_staged" and cluster.connections == connections


def test_inventory_script_is_self_contained_and_returns_no_contents(tmp_path):
    import subprocess
    import sys

    run_dir = tmp_path / "run"
    (run_dir / "search" / "Specific").mkdir(parents=True)
    (run_dir / "search" / "Specific" / "POTCAR_Si").write_text("licensed text")
    (run_dir / "backend" / "__pycache__").mkdir(parents=True)
    (run_dir / "backend" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (run_dir / "link").symlink_to(run_dir / "search")
    output = subprocess.run([sys.executable, "-I", "-"], input=staging.build_inventory_script(str(run_dir)),
                            capture_output=True, text=True, cwd=tmp_path, check=True).stdout
    assert "licensed text" not in output
    inventory = staging.parse_inventory_output(output)["entries"]
    assert inventory["search/Specific/POTCAR_Si"]["sha256"] == hashlib.sha256(b"licensed text").hexdigest()
    assert inventory["link"] == {"kind": "symlink"}
    assert "backend/__pycache__" not in inventory and "backend" in inventory
    name = staging.staging_temporary_name("INPUT.txt", OTHER_ATTEMPT, "ab" * 32)
    assert staging.parse_staging_temporary(name) == {"target": "INPUT.txt", "attempt": OTHER_ATTEMPT,
                                                     "tag": "ab" * 8}
    assert staging.parse_staging_temporary(".INPUT.txt." + "c" * 32 + ".tmp") is None


# --- finalization timeout and diagnostics ----------------------------------------------------


def test_only_finalization_gets_the_longer_timeout(client, tokens, state_dir, cluster):
    from backend.paramiko_remote import DEFAULT_REMOTE_COMMAND_TIMEOUT_S
    from compute_api.structure_search_execution import FINALIZE_TIMEOUT_S, reverify_staged_search

    assert (DEFAULT_REMOTE_COMMAND_TIMEOUT_S, FINALIZE_TIMEOUT_S) == (60, 300)
    attempt, digest = prepared_attempt(client, tokens)
    assert stage(client, tokens["preparer"], attempt, digest).status_code == 200
    reverify_staged_search(principal="preparer", attempt_id=attempt)
    finalize = [timeout for kind, timeout in cluster.timeouts if kind == "finalize"]
    assert finalize == [300, 300]  # staging, then the fresh re-verification
    assert {timeout for kind, timeout in cluster.timeouts if kind != "finalize"} == {None}  # runner default


class _TimingOutClient:
    """A Paramiko client whose command produces no output before the channel timeout."""

    def __init__(self):
        self.timeouts = []

    def get_transport(self):
        class _Transport:
            def is_active(self):
                return True

        return _Transport()

    def exec_command(self, command, get_pty=False, timeout=None):
        self.timeouts.append(timeout)

        class _Stream:
            channel = None

            def read(self):
                raise TimeoutError("timed out")

            def close(self):
                pass

        return _Stream(), _Stream(), _Stream()


def _remote_staging_for(state_dir, attempt):
    from backend.runtime_package import runtime_package_manifest_for_sources
    from backend.submission import build_backend_module_sources
    from compute_api.structure_search_execution import _RemoteStaging

    record = search_record(state_dir, attempt)
    local = attempt_dir(state_dir, attempt) / "preparation-1"
    sources = build_backend_module_sources()
    return _RemoteStaging(spec=record["staging"]["spec"], plan=load_plan_text((local / PLAN_FILE).read_text()),
                          local_manifest=json.loads((local / "prepared_inputs.json").read_text()),
                          sources=sources, manifest=runtime_package_manifest_for_sources(sources))


def test_the_finalize_timeout_reaches_paramiko_and_the_default_is_unchanged(client, tokens, state_dir, cluster,
                                                                          caplog):
    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_finalize = 1
    stage(client, tokens["preparer"], attempt, digest)  # leaves a STAGING attempt with its spec
    fake = _TimingOutClient()
    runner = ParamikoRemoteRunner(client=fake)
    remote = _remote_staging_for(state_dir, attempt)
    with caplog.at_level("INFO", logger="compute_api.structure_search_execution"):
        with pytest.raises(TimeoutError):
            remote.finalize(runner, copy_potcars=True)
    with pytest.raises(TimeoutError):
        runner.run("true")  # an ordinary command
    assert fake.timeouts == [300, 60]
    assert "finalize (copy_potcars=True) gave no answer: TimeoutError" in caplog.text
    assert "(timeout 300 s)" in caplog.text


@pytest.mark.parametrize("when", ["before", "after"])
def test_a_finalize_timeout_is_recovered_by_retrying_the_same_attempt(client, tokens, state_dir, cluster, when):
    attempt, digest = prepared_attempt(client, tokens, "GaAs")
    cluster.finalize_timeout = when
    failed = stage(client, tokens["preparer"], attempt, digest)
    assert failed.status_code == 503 and failed.json()["error"]["code"] == "staging_unavailable"
    assert search_record(state_dir, attempt)["state"] == "STAGING"
    state = json.loads(cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.json").read_text())
    assert state["state"] == "STAGING"  # never claimed as staged
    assert not cluster.at(f"{SEARCH_LOGS}/submission_attempts/{attempt}.lock").exists()
    run_dir, run_name = staged_run(cluster, state_dir, attempt)
    potcars = sorted(p.name for p in (run_dir / "search" / "Specific").glob("POTCAR_*"))
    assert potcars == ([] if when == "before" else ["POTCAR_As", "POTCAR_Ga"])
    before = tree(run_dir)

    response = stage(client, tokens["preparer"], attempt, digest)
    assert response.status_code == 200 and response.json()["state"] == "STAGED", response.text
    after = tree(run_dir)
    assert {name: after[name] for name in before} == before  # nothing completed is rewritten
    assert sorted(p.name for p in cluster.at(WORKSPACE).iterdir()) == [run_name]
    assert hidden(run_dir) == []


def test_staging_logs_its_phases(client, tokens, state_dir, cluster, caplog):
    attempt, digest = prepared_attempt(client, tokens)
    with caplog.at_level("INFO", logger="compute_api.structure_search_execution"):
        assert stage(client, tokens["preparer"], attempt, digest).status_code == 200
    text = caplog.text
    for phase in ("inventory done", "finalize done", "final_inventory done", "uploads ("):
        assert phase in text, phase
    assert f"structure-search staging {attempt}: staged after" in text
    assert "finalize (copy_potcars=True) answered after" in text
    assert "synthetic:" not in text and "Bearer" not in text


def test_instrumented_finalize_script_returns_the_same_result_with_timings(client, tokens, state_dir, cluster,
                                                                         tmp_path):
    import subprocess

    from compute_api.structure_search_execution import (
        TIMINGS_MARKER,
        instrument_finalize_script,
        parse_finalize_timings,
    )
    from backend.submission import RUNTIME_PACKAGE_GUARD_MODULE

    attempt, digest = prepared_attempt(client, tokens)
    cluster.fail_finalize = 1
    stage(client, tokens["preparer"], attempt, digest)
    remote = _remote_staging_for(state_dir, attempt)
    request = dict(remote.request)
    for key in ("run_dir", "package_dir", "search_dir", "bridge_path", "potcar_library", "cache_ready_marker"):
        request[key] = str(cluster.at(request[key]))
    run_dir = Path(request["run_dir"])
    before = tree(run_dir)
    plain = staging.build_finalize_script(request, runtime_sources=remote.sources,
                                          guard_module=RUNTIME_PACKAGE_GUARD_MODULE)
    instrumented = instrument_finalize_script(plain)
    assert plain.replace("finalize_staged_search_main(REQUEST)", "") in instrumented.replace("\n    finalize", "\nfinalize")        or plain.splitlines()[:-1] == instrumented.splitlines()[2:len(plain.splitlines()) + 1]

    def remote_run(script):
        out = subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, cwd=run_dir)
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout

    plain_out, timed_out = remote_run(plain), remote_run(instrumented)
    assert staging.parse_finalize_output(timed_out) == staging.parse_finalize_output(plain_out)
    assert TIMINGS_MARKER not in plain_out
    timings = parse_finalize_timings(timed_out)
    assert timings["finalize_calls"] == 1 and timings["package_recheck_calls"] == 1
    assert timings["plan_load_calls"] == 1 and timings["potcar_identity_calls"] >= 1
    for key in ("guard_verify_and_install_s", "imports_s", "finalize_s", "total_s"):
        assert isinstance(timings[key], float), key
    assert tree(run_dir) == before
    assert "synthetic:" not in timed_out


def test_unexpected_finalize_script_shape_is_not_instrumented():
    from compute_api.structure_search_execution import instrument_finalize_script

    with pytest.raises(RuntimeError):
        instrument_finalize_script("print('hello')\n")
