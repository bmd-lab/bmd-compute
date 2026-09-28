"""Contract tests for bmd_compute.submission v1 and bmd_compute.job_record v1."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import run_record_fixtures
from backend.paramiko_remote import ParamikoRemoteRunner
from backend.remote import BatchSubmissionResult, RemoteCommandResult, RemotePathInfo
from backend.results import resolve_results_location
from backend.run_records import (
    JOB_RECORD_SCHEMA,
    SUBMISSION_RECORD_SCHEMA,
    RunRecordContractError,
    job_record_contract_projection,
    stage_index_from_directory_id,
    submission_contract_projection,
    validate_job_record_v1,
    validate_stage_directories,
    validate_submission_record_v1,
)
from backend.submission import create_submission_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = tuple(run_record_fixtures.CASES)


def load_fixture(case: str) -> tuple[dict, dict]:
    submission_path, job_record_path = run_record_fixtures.fixture_paths(case)
    return (
        json.loads(submission_path.read_text(encoding="utf-8")),
        json.loads(job_record_path.read_text(encoding="utf-8")),
    )


# --- canonical generated fixtures -------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_committed_fixtures_satisfy_v1_contracts(case):
    submission, job_record = load_fixture(case)

    validate_submission_record_v1(submission)
    validate_job_record_v1(job_record)
    assert submission["schema"] == SUBMISSION_RECORD_SCHEMA
    assert job_record["schema"] == JOB_RECORD_SCHEMA
    assert job_record["attempt_id"] == submission["submission"]["attempt_id"]
    assert job_record["run_dir"] == submission["paths"]["run_dir"]
    assert job_record["run_name"] == submission["run_name"]


@pytest.mark.parametrize("case", CASES)
def test_committed_fixtures_match_current_producer_on_contractual_fields(case, monkeypatch):
    monkeypatch.setenv("BMD_SUBMISSION_IDENTITY_SECRET", "fixture-only")
    committed_submission, committed_job_record = load_fixture(case)
    submission, job_record = run_record_fixtures.build_v1_records(case)

    assert submission_contract_projection(submission) == submission_contract_projection(
        committed_submission
    )
    assert job_record_contract_projection(job_record) == job_record_contract_projection(
        committed_job_record
    )


def test_fixture_stage_identifiers_cover_both_identifier_forms():
    identifiers = {
        case: set(load_fixture(case)[0]["paths"]["stage_dirs"])
        for case in CASES
    }
    assert identifiers["single_stage_pbe_static"] == set()
    assert identifiers["hse06_soc_after_pbe_relax"] == {"stage_01", "stage_02"}
    assert identifiers["pbe_double_relax"] == {"relax_01", "relax_02"}


# --- producer output ------------------------------------------------------------------


def _stage(stage_type, theory, modifiers=()):
    return {"stage_type": stage_type, "theory": theory, "modifiers": list(modifiers), "label": None, "options": {}}


WORKFLOWS = {
    "static": [_stage("static", "pbe")],
    "relax_static": [_stage("relax", "pbe"), _stage("static", "pbe")],
    "double_relax": [_stage("relax", "pbe"), _stage("relax", "pbe")],
    "dos": [_stage("relax", "pbe"), _stage("static", "pbe"), _stage("dos", "pbe")],
    "band": [_stage("relax", "pbe"), _stage("static", "pbe"), _stage("band_structure", "pbe")],
    "hse_soc": [_stage("relax", "pbe"), _stage("static", "hse06", ("soc",))],
}


def submission_for(stages, *, attempt_id="00000000-0000-4000-8000-000000000201"):
    return create_submission_spec(
        {
            "workflow_spec": {"stages": stages, "label": None, "recipe": None},
            "potcar_functional": "PBE_64",
            "structure": {"type": "pasted_text", "format": "poscar", "text": run_record_fixtures.SI_POSCAR},
        },
        label="Si contract",
        timestamp="20260928-120000",
        env={},
        submission_attempt_id=attempt_id,
    )


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_every_supported_workflow_writes_a_valid_v1_submission(name):
    stages = WORKFLOWS[name]
    spec = submission_for(stages)

    validate_submission_record_v1(spec)
    mapped = validate_stage_directories(spec["paths"]["stage_dirs"], len(stages))
    if len(stages) == 1:
        assert mapped == {}
    else:
        assert set(mapped) == set(range(1, len(stages) + 1))
        for identifier, path in spec["paths"]["stage_dirs"].items():
            assert path.endswith("/" + identifier)


@pytest.mark.parametrize("name", ["relax_static", "double_relax", "dos"])
def test_stage_mapping_does_not_depend_on_json_object_order(name):
    stages = WORKFLOWS[name]
    spec = submission_for(stages)
    reversed_dirs = dict(reversed(list(spec["paths"]["stage_dirs"].items())))

    assert validate_stage_directories(reversed_dirs, len(stages)) == validate_stage_directories(
        spec["paths"]["stage_dirs"], len(stages)
    )


def test_stage_identifier_grammar():
    assert stage_index_from_directory_id("stage_01") == 1
    assert stage_index_from_directory_id("stage_12") == 12
    assert stage_index_from_directory_id("relax_02") == 2
    for identifier in ("stage_1", "stage_00", "Stage_01", "producer-alpha", "result_dir", 1, None):
        assert stage_index_from_directory_id(identifier) is None


# --- validator adversarial cases ---------------------------------------------------------


def _mutated(document, path, value=None, *, delete=False):
    mutated = copy.deepcopy(document)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return mutated


SUBMISSION_MUTATIONS = [
    (("schema",), "bmd_compute.job_record", False),
    (("schema",), None, True),
    (("schema_version",), 2, False),
    (("schema_version",), True, False),
    (("schema_version",), "1", False),
    (("run_name",), "", False),
    (("created_at",), None, True),
    (("submission", "attempt_id"), None, True),
    (("submission", "attempt_state"), 7, False),
    (("flow_spec", "workflow_spec"), None, True),
    (("flow_spec", "workflow_spec", "stages"), [], False),
    (("flow_spec", "workflow_spec", "stages", 0, "modifiers"), "soc", False),
    (("flow_spec", "workflow_spec", "stages", 0, "options"), None, False),
    (("paths", "run_dir"), None, True),
    (("paths", "result_dir"), "", False),
    (("paths", "stage_dirs"), None, True),
    (("paths", "stage_dirs", "stage_02"), None, True),
    (("paths", "stage_dirs", "stage_03"), "/x/stage_03", False),
    (("paths", "stage_dirs", "producer-alpha"), "/x/alpha", False),
    (("cluster", "partition"), None, True),
    (("resources", "mem_gb"), "128", False),
    (("resources", "nodes"), 0, False),
    (("provenance", "schema_version"), None, True),
]


@pytest.mark.parametrize(("path", "value", "delete"), SUBMISSION_MUTATIONS)
def test_submission_v1_validator_rejects_malformed_records(path, value, delete):
    submission, _ = load_fixture("hse06_soc_after_pbe_relax")
    with pytest.raises(RunRecordContractError):
        validate_submission_record_v1(_mutated(submission, path, value, delete=delete))


def test_submission_v1_non_contractual_fields_can_change_freely():
    submission, _ = load_fixture("hse06_soc_after_pbe_relax")
    trimmed = copy.deepcopy(submission)
    for key in ("status", "label", "modules", "runner", "potcar", "preflight", "environment"):
        trimmed.pop(key, None)
    trimmed["submission"] = {
        "attempt_id": submission["submission"]["attempt_id"],
        "attempt_state": submission["submission"]["attempt_state"],
    }
    trimmed["cluster"] = {"partition": "p", "account": "a"}
    trimmed["unknown_future_additive_field"] = {"anything": True}

    validate_submission_record_v1(trimmed)


JOB_RECORD_MUTATIONS = [
    (("schema",), "bmd_compute.submission", False),
    (("schema_version",), 2, False),
    (("schema_version",), None, True),
    (("job_id",), "", False),
    (("run_name",), None, True),
    (("run_dir",), 5, False),
    (("attempt_id",), None, False),
    (("submitted_at",), 12, False),
]


@pytest.mark.parametrize(("path", "value", "delete"), JOB_RECORD_MUTATIONS)
def test_job_record_v1_validator_rejects_malformed_records(path, value, delete):
    _, job_record = load_fixture("hse06_soc_after_pbe_relax")
    with pytest.raises(RunRecordContractError):
        validate_job_record_v1(_mutated(job_record, path, value, delete=delete))


def test_job_record_v1_is_minimal_and_status_is_not_contractual():
    _, job_record = load_fixture("hse06_soc_after_pbe_relax")
    minimal = {
        key: job_record[key]
        for key in ("schema", "schema_version", "job_id", "run_name", "run_dir", "attempt_id")
    }
    validate_job_record_v1(minimal)
    validate_job_record_v1({**minimal, "submitted_at": None})


# --- atomic writes and the real submit path -------------------------------------------------


class RecordStore:
    def __init__(self):
        self.files = {}
        self.direct_writes = []
        self.renames = []
        self.fail_rename_to = None


class RecordRunner(ParamikoRemoteRunner):
    def __init__(self, store: RecordStore):
        super().__init__(client=object())
        self.store = store

    def ensure_available(self):
        return None

    def run(self, command, *, check=False, modules=False, export_env=False, timeout_s=None):
        return RemoteCommandResult(command=command, returncode=0)

    def ensure_directory(self, remote_path):
        return RemotePathInfo(path=remote_path, exists=True, kind="dir")

    def is_dir(self, remote_path):
        return True

    def is_file(self, remote_path):
        return remote_path in self.store.files

    def read_text(self, remote_path, *, max_bytes=None):
        return self.store.files[remote_path]

    def put_text(self, remote_path, text, *, mode=0o640):
        self.store.direct_writes.append(remote_path)
        self.store.files[remote_path] = text

    def _replace_file(self, source, destination):
        if destination == self.store.fail_rename_to:
            raise OSError("simulated rename failure")
        self.store.renames.append((source, destination))
        self.store.files[destination] = self.store.files.pop(source)

    def _discard_temporary_file(self, remote_path):
        self.store.files.pop(remote_path, None)

    def submit_batch(self, request):
        return BatchSubmissionResult(job_id="10000301", raw_output="10000301\n", command="sbatch --parsable")


def _submitted(store: RecordStore, spec: dict):
    RecordRunner(store).submit(spec, dry_run=True)
    return RecordRunner(store).submit(spec)


def test_submit_writes_both_records_atomically_and_as_v1():
    store = RecordStore()
    spec = submission_for(WORKFLOWS["hse_soc"])
    record = _submitted(store, spec)

    submission_path = f"{spec['paths']['run_dir']}/submission.json"
    job_record_path = record.remote_state_path
    for path in (submission_path, job_record_path):
        assert path not in store.direct_writes
        sources = [source for source, destination in store.renames if destination == path]
        assert sources, path
        for source in sources:
            directory, name = source.rsplit("/", 1)
            assert directory == path.rsplit("/", 1)[0]
            assert name.startswith(".") and name.endswith(".tmp")
            assert source not in store.files
    assert not [path for path in store.files if path.endswith(".tmp")]

    validate_submission_record_v1(json.loads(store.files[submission_path]))
    job_record = json.loads(store.files[job_record_path])
    validate_job_record_v1(job_record)
    assert job_record["job_id"] == "10000301"
    assert job_record["attempt_id"] == spec["submission"]["attempt_id"]
    assert list(job_record)[:3] == ["schema", "schema_version", "attempt_id"]


def test_other_preparation_files_are_not_routed_through_the_record_path():
    store = RecordStore()
    spec = submission_for(WORKFLOWS["static"])
    RecordRunner(store).submit(spec, dry_run=True)

    renamed = {destination for _source, destination in store.renames}
    assert renamed == {f"{spec['paths']['run_dir']}/submission.json"}
    assert spec["paths"]["remote_script"] in store.direct_writes


def test_dry_run_writes_no_job_record():
    store = RecordStore()
    spec = submission_for(WORKFLOWS["static"])
    RecordRunner(store).submit(spec, dry_run=True)

    assert not [path for path in store.files if "/job_" in path]


def test_failed_job_record_rename_leaves_no_partial_record_and_submit_still_succeeds():
    store = RecordStore()
    spec = submission_for(WORKFLOWS["static"])
    store.fail_rename_to = f"{spec['paths']['logs_dir']}/job_10000301.json"

    record = _submitted(store, spec)

    assert record.job_id == "10000301"
    assert record.remote_state_path not in store.files
    assert not [path for path in store.files if path.endswith(".tmp")]


def test_failed_submission_rename_keeps_previous_submission_json():
    store = RecordStore()
    spec = submission_for(WORKFLOWS["static"])
    submission_path = f"{spec['paths']['run_dir']}/submission.json"
    store.files[submission_path] = "previous complete document"
    store.fail_rename_to = submission_path

    with pytest.raises(Exception):
        RecordRunner(store).submit(spec, dry_run=True)

    assert store.files[submission_path] == "previous complete document"
    assert not [path for path in store.files if path.endswith(".tmp")]


class FakeSftp:
    def __init__(self, *, fail_rename=False):
        self.files = {}
        self.modes = {}
        self.operations = []
        self.fail_rename = fail_rename

    def file(self, remote_path, mode):
        sftp = self

        class Handle:
            def __enter__(self):
                return self

            def write(self, data):
                sftp.files[remote_path] = sftp.files.get(remote_path, "") + data

            def __exit__(self, *exc):
                return False

        self.operations.append(("write", remote_path))
        self.files[remote_path] = ""
        return Handle()

    def chmod(self, remote_path, mode):
        self.modes[remote_path] = mode

    def posix_rename(self, source, destination):
        self.operations.append(("posix_rename", source, destination))
        if self.fail_rename:
            raise OSError("rename refused")
        self.files[destination] = self.files.pop(source)

    def remove(self, remote_path):
        self.operations.append(("remove", remote_path))
        self.files.pop(remote_path, None)

    def close(self):
        return None


class SftpRunner(ParamikoRemoteRunner):
    def __init__(self, sftp):
        super().__init__(client=object())
        self.sftp = sftp

    def ensure_available(self):
        return None

    def ensure_directory(self, remote_path):
        return None

    def _open_sftp(self):
        return self.sftp


def test_put_text_atomic_uses_sftp_posix_rename_over_a_hidden_sibling():
    sftp = FakeSftp()
    sftp.files["/logs/job_1.json"] = "old"
    SftpRunner(sftp).put_text_atomic("/logs/job_1.json", '{"new": true}')

    write, rename = sftp.operations
    assert write[0] == "write" and write[1].startswith("/logs/.job_1.json.")
    assert rename == ("posix_rename", write[1], "/logs/job_1.json")
    assert sftp.files == {"/logs/job_1.json": '{"new": true}'}


def test_put_text_atomic_failure_removes_temporary_and_keeps_destination():
    sftp = FakeSftp(fail_rename=True)
    sftp.files["/logs/job_1.json"] = "old"

    with pytest.raises(OSError):
        SftpRunner(sftp).put_text_atomic("/logs/job_1.json", "new")

    assert sftp.operations[-1][0] == "remove"
    assert sftp.files == {"/logs/job_1.json": "old"}


# --- BMD Compute's own Resume reader --------------------------------------------------------


class ResumeRunner:
    def __init__(self, files):
        self.files = files

    def is_file(self, path):
        return path in self.files

    def read_text(self, path, *, max_bytes=None):
        return self.files[path]


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_resolves_v1_and_legacy_job_records(legacy):
    _, job_record = load_fixture("hse06_soc_after_pbe_relax")
    if legacy:
        job_record = {key: value for key, value in job_record.items() if key not in ("schema", "schema_version", "attempt_id")}
    path = f"/bmd-db/guest/logs/job_{job_record['job_id']}.json"

    location = resolve_results_location(
        ResumeRunner({path: json.dumps(job_record)}),
        {"job_id": job_record["job_id"]},
        None,
    )

    assert location["run_dir"] == job_record["run_dir"]


def test_run_record_contract_documentation_states_authority_boundaries():
    text = (REPO_ROOT / "docs" / "run_records.md").read_text(encoding="utf-8")
    assert "bmd_compute.submission" in text and "bmd_compute.job_record" in text
    assert "not scheduler lifecycle authority" in text
    assert "Prepare-time submission specification" in text
    assert "insertion order is not contractual" in text
