"""The structure-search check inside the POWER runtime-environment check.

Only submissions with a ``structure_search`` block are checked for USPEX;
every other submission's runtime record is unchanged.
"""

import json

import pytest

from backend.runtime_environment import (
    RUNTIME_ENVIRONMENT_FILENAME,
    RuntimeParityError,
    enforce_runtime_environment,
)
from backend.structure_search.runtime import prepared_search_runtime
from test_runtime_environment import PRODUCTION, Settings, lookup, prepared, spec_for
from test_structure_search_runtime import power  # noqa: F401  (fixture)


def _enforce(tmp_path, spec, slurm_version_lookup):
    return enforce_runtime_environment(
        spec,
        version_lookup=lookup(PRODUCTION),
        settings=Settings(),
        environ={"SLURM_JOB_ID": "22400000"},
        slurm_version_lookup=slurm_version_lookup,
    )


def _search_spec(run_dir, runtime_record):
    spec = spec_for(run_dir, prepared())
    spec["structure_search"] = {"runtime": runtime_record}
    return spec


def test_ordinary_submission_gets_no_uspex_check(tmp_path):
    def slurm():
        raise AssertionError("ordinary submissions must not run the search check")

    record = _enforce(tmp_path, spec_for(tmp_path, prepared()), slurm)
    assert record["status"] == "passed"
    assert "structure_search_runtime" not in record


def test_search_submission_with_uspex_in_place_passes_and_is_recorded(tmp_path, power):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = _enforce(run_dir, _search_spec(run_dir, prepared_search_runtime(power)), lambda: "slurm 26.05.5")

    assert record["status"] == "passed", record["problems"]
    search = record["structure_search_runtime"]
    assert search["status"] == "passed"
    assert search["prepared"]["uspex"]["version"] == "25.0"
    assert search["observed"]["payload_sha256"] == power.payload_sha256
    written = json.loads((run_dir / RUNTIME_ENVIRONMENT_FILENAME).read_text(encoding="utf-8"))
    assert written == record


def test_search_submission_stops_when_uspex_is_not_unpacked(tmp_path, power):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    power.ready_marker.unlink()

    with pytest.raises(RuntimeParityError) as raised:
        _enforce(run_dir, _search_spec(run_dir, prepared_search_runtime(power)), lambda: "slurm 26.05.5")

    message = str(raised.value)
    assert "runtime environment check failed" in message
    assert ".ready is missing" in message
    written = json.loads((run_dir / RUNTIME_ENVIRONMENT_FILENAME).read_text(encoding="utf-8"))
    assert written["status"] == "failed"
    assert written["structure_search_runtime"]["status"] == "failed"


def test_search_submission_without_a_prepared_runtime_record_stops(tmp_path):
    spec = spec_for(tmp_path, prepared())
    spec["structure_search"] = {}
    with pytest.raises(RuntimeParityError, match="no structure_search.runtime record"):
        _enforce(tmp_path, spec, lambda: "slurm 26.05.5")
