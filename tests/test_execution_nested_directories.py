"""Stage directories for nested and replacement Jobflow jobs.

The BMD local runner gives every top-level stage its own directory. atomate2's
phonon workflow is a nested Flow whose ``run_phonon_displacements`` job
replaces itself at run time with one job per displacement. These tests use a
small real Jobflow flow (pure Python jobs, no VASP) with the same shape to
check that every job, including replacement jobs, runs in its own stable
directory and that no job overwrites another job's files.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from jobflow import Flow, Response, job
from jobflow.core.store import JobStore
from maggma.stores import MemoryStore

from backend.execution import (
    _flow_stage_directories,
    _run_locally_with_stage_directories,
    _safe_job_directory_name,
)


def _mark(label: str) -> str:
    cwd = Path.cwd()
    marker = cwd / "OUTCAR"
    assert not marker.exists(), f"{label} would overwrite {marker}"
    marker.write_text(label, encoding="utf-8")
    return str(cwd)


@job(name="relax")
def fake_relax(value):
    _mark("relax")
    return value + 1


@job(name="get_supercell_size")
def fake_supercell(value):
    _mark("supercell")
    return value


@job(name="phonon static")
def fake_displacement(value, index):
    _mark(f"displacement {index}")
    return {"index": index, "force": value * 10 + index, "dir": str(Path.cwd())}


@job(name="run_phonon_displacements")
def fake_run_displacements(value, count):
    _mark("run_phonon_displacements")
    jobs = []
    for index in range(count):
        displacement = fake_displacement(value, index)
        displacement.append_name(f" {index + 1}/{count}")
        jobs.append(displacement)
    outputs = {
        "forces": [item.output["force"] for item in jobs],
        "dirs": [item.output["dir"] for item in jobs],
    }
    return Response(replace=Flow(jobs, outputs))


@job(name="generate_frequencies_eigenvectors")
def fake_collect(displacement_data):
    _mark("collect")
    return {"forces": displacement_data["forces"], "dirs": displacement_data["dirs"]}


def _phonon_like_flow(count: int = 3) -> tuple[Flow, object]:
    relax = fake_relax(1)
    supercell = fake_supercell(relax.output)
    displacements = fake_run_displacements(supercell.output, count)
    collect = fake_collect(displacements.output)
    phonon = Flow([supercell, displacements, collect], output=collect.output, name="phonon")
    flow = Flow(
        [relax, phonon],
        name="phonons",
        metadata={"bmd_stage_directories": ("relax_01", "phonon_forces_02")},
    )
    return flow, collect


def _run(flow, tmpdir):
    store = JobStore(MemoryStore())
    responses = _run_locally_with_stage_directories(
        flow,
        _flow_stage_directories(flow),
        root_dir=tmpdir,
        store=store,
        ensure_success=True,
        log=False,
    )
    return responses, store


def _relative_dirs(root: Path) -> set[str]:
    return {
        str(path.parent.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("OUTCAR")
    }


def test_job_directory_names_are_readable_and_distinct():
    assert _safe_job_directory_name("phonon static 3/8") == "phonon_static_3_of_8"
    assert _safe_job_directory_name("phonon static 10/12") == "phonon_static_10_of_12"
    assert _safe_job_directory_name("") == "job"
    assert _safe_job_directory_name("../../etc") == "etc"


def test_nested_flow_needs_stage_directories_even_when_alone():
    lone = Flow(
        [Flow([fake_supercell(1)], name="phonon")],
        name="only",
        metadata={"bmd_stage_directories": ("phonon_forces_01",)},
    )
    assert _flow_stage_directories(lone) == ("phonon_forces_01",)
    plain = Flow([fake_relax(1)], name="single")
    assert _flow_stage_directories(plain) == ()


def test_replacement_jobs_get_unique_stable_directories():
    flow, collect = _phonon_like_flow(count=3)
    with tempfile.TemporaryDirectory(prefix="bmd-nested-run-") as tmpdir:
        responses, store = _run(flow, tmpdir)
        root = Path(tmpdir).resolve()

        expected = {
            "relax_01",
            "phonon_forces_02/get_supercell_size",
            "phonon_forces_02/run_phonon_displacements",
            "phonon_forces_02/run_phonon_displacements/phonon_static_1_of_3",
            "phonon_forces_02/run_phonon_displacements/phonon_static_2_of_3",
            "phonon_forces_02/run_phonon_displacements/phonon_static_3_of_3",
            "phonon_forces_02/generate_frequencies_eigenvectors",
        }
        assert _relative_dirs(root) == expected
        assert not (root / "OUTCAR").exists()

        result = store.get_output(collect.uuid)
        assert result["forces"] == [20, 21, 22]
        assert [Path(item).relative_to(root).as_posix() for item in result["dirs"]] == [
            f"phonon_forces_02/run_phonon_displacements/phonon_static_{i}_of_3" for i in (1, 2, 3)
        ]
        # Every executed job has a recorded response with its directory.
        job_dirs = {
            response.job_dir
            for by_index in responses.values()
            for response in by_index.values()
            if getattr(response, "job_dir", None) is not None
        }
        assert root / "relax_01" in job_dirs


def test_directory_layout_is_reproducible():
    layouts = []
    for _ in range(2):
        flow, _ = _phonon_like_flow(count=4)
        with tempfile.TemporaryDirectory(prefix="bmd-nested-repeat-") as tmpdir:
            _run(flow, tmpdir)
            layouts.append(sorted(_relative_dirs(Path(tmpdir).resolve())))
    assert layouts[0] == layouts[1]


def test_flat_flows_keep_existing_layout():
    first = fake_relax(1)
    second = fake_relax(first.output)
    flow = Flow([first, second], name="flat", metadata={"bmd_stage_directories": ("relax_01", "relax_02")})
    with tempfile.TemporaryDirectory(prefix="bmd-flat-run-") as tmpdir:
        _, store = _run(flow, tmpdir)
        root = Path(tmpdir).resolve()
        assert _relative_dirs(root) == {"relax_01", "relax_02"}
        assert store.get_output(second.uuid) == 3


def test_replacement_in_flat_stage_does_not_reuse_stage_directories():
    relax = fake_relax(1)
    displacements = fake_run_displacements(relax.output, 2)
    flow = Flow([relax, displacements], name="flat-replace", metadata={"bmd_stage_directories": ("relax_01", "forces_02")})
    with tempfile.TemporaryDirectory(prefix="bmd-flat-replace-") as tmpdir:
        _run(flow, tmpdir)
        root = Path(tmpdir).resolve()
        assert _relative_dirs(root) == {
            "relax_01",
            "forces_02",
            "forces_02/phonon_static_1_of_2",
            "forces_02/phonon_static_2_of_2",
        }
