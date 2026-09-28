"""Canonical generated fixtures for the BMD Compute run-record contracts.

BMD Compute owns these fixtures. Regenerate them from the repository root with

    BMD_SUBMISSION_IDENTITY_SECRET=fixture-only python tests/run_record_fixtures.py

The files are produced by the real ``create_submission_spec`` and
``ParamikoRemoteRunner._job_record`` code paths plus ``job_record_payload``,
exactly as written to POWER, with two normalizations for stable diffs:
``submitted_at`` is fixed, and the identity-token secret is a fixed,
fixture-only value. Provenance fields (Git state, package versions, runtime
file hashes) are whatever the generating environment reported. Only the
contractual fields are compared by the drift test; consumers such as BMD Agent
may vendor snapshots labelled with the generating BMD Compute commit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "run_records" / "v1"
FIXTURE_SUBMITTED_AT = "2026-09-28T12:00:05+03:00"

SI_POSCAR = """Si
1.0
0.0 2.715 2.715
2.715 0.0 2.715
2.715 2.715 0.0
Si
2
direct
0.0 0.0 0.0
0.25 0.25 0.25
"""


def _stage(stage_type: str, theory: str, modifiers=()) -> dict:
    return {
        "stage_type": stage_type,
        "theory": theory,
        "modifiers": list(modifiers),
        "label": None,
        "options": {},
    }


CASES = {
    "single_stage_pbe_static": {
        "label": "si_pbe_static",
        "attempt_id": "00000000-0000-4000-8000-000000000101",
        "job_id": "10000101",
        "stages": [_stage("static", "pbe")],
    },
    "hse06_soc_after_pbe_relax": {
        "label": "si_hse06_soc",
        "attempt_id": "00000000-0000-4000-8000-000000000102",
        "job_id": "10000102",
        "stages": [_stage("relax", "pbe"), _stage("static", "hse06", ("soc",))],
    },
    "pbe_double_relax": {
        "label": "si_double_relax",
        "attempt_id": "00000000-0000-4000-8000-000000000103",
        "job_id": "10000103",
        "stages": [_stage("relax", "pbe"), _stage("relax", "pbe")],
    },
}


def build_v1_records(case: str) -> tuple[dict, dict]:
    from pymatgen.core import Structure

    from backend.paramiko_remote import ParamikoRemoteRunner
    from backend.run_records import job_record_payload
    from backend.submission import create_submission_spec

    definition = CASES[case]
    flow_spec = {
        "workflow_spec": {"stages": definition["stages"], "label": None, "recipe": None},
        "potcar_functional": "PBE_64",
        "structure": {"type": "pasted_text", "format": "poscar", "text": SI_POSCAR},
    }
    submission = create_submission_spec(
        flow_spec,
        structure=Structure.from_str(SI_POSCAR, fmt="poscar"),
        label=definition["label"],
        timestamp="20260928-120000",
        env={},
        submission_attempt_id=definition["attempt_id"],
    )
    record = ParamikoRemoteRunner._job_record(
        None,
        submission,
        definition["job_id"],
        f"{definition['job_id']}\n",
    ).to_dict()
    record["submitted_at"] = FIXTURE_SUBMITTED_AT
    return submission, job_record_payload(record)


def fixture_paths(case: str) -> tuple[Path, Path]:
    directory = FIXTURE_ROOT / case
    return directory / "submission.json", directory / "job_record.json"


def write_fixtures() -> None:
    # Build every record before writing any file so the captured Git
    # provenance describes the generating commit, not the fixture writes.
    built = {case: build_v1_records(case) for case in CASES}
    for case, (submission, job_record) in built.items():
        submission_path, job_record_path = fixture_paths(case)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
        job_record_path.write_text(json.dumps(job_record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("BMD_SUBMISSION_IDENTITY_SECRET", "fixture-only")
    write_fixtures()
