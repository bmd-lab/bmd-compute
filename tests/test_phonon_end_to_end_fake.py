"""Simulated end-to-end Si phonon run through BMD Compute's execution pathway.

The submission specification comes from the same planner the browser and the
machine API use (``main.plan_calculation_request`` with the ``phonons``
Desired Output). It is executed by the POWER runner entry point
``backend.execution.run_submission`` - runtime parity check, workflow
reconstruction from the specification, and the BMD local Jobflow runner with
per-stage directories - in a temporary directory.

Only VASP is faked: ``run_vasp`` is replaced by a no-op and the task document
is built from a simple harmonic spring model of diamond-structure Si
(nearest- and next-nearest-neighbour central springs). Everything else is
real: atomate2's PhononMaker, Phonopy displacement generation, the
replacement of ``run_phonon_displacements`` by the displacement jobs, and
Phonopy's force constants and frequencies.

This verifies *integration*, not science: the frequencies are those of the
spring model, not of PBE silicon. Scientific validation requires the live
POWER acceptance run.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("phonopy")
pytest.importorskip("seekpath")

import main
from backend import execution
from backend.phonons.workflow import PHONON_FORCE_CONSTANTS_FILENAME


SI_POSCAR = """Si2
1.0
0.0 2.715 2.715
2.715 0.0 2.715
2.715 2.715 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""
A = 5.43
R0_NN = A * np.sqrt(3) / 4
R0_NNN = A / np.sqrt(2)
K_NN = 10.0  # eV/A^2
K_NNN = 1.0  # eV/A^2


def spring_forces(structure) -> np.ndarray:
    """Central harmonic springs to the first two neighbour shells of diamond."""

    forces = np.zeros((len(structure), 3))
    for i, neighbors in enumerate(structure.get_all_neighbors(4.2)):
        for neighbor in neighbors:
            vector = neighbor.coords - structure[i].coords
            distance = np.linalg.norm(vector)
            if distance < 3.1:
                k, r0 = K_NN, R0_NN
            else:
                k, r0 = K_NNN, R0_NNN
            forces[i] += k * (distance - r0) * vector / distance
    return forces


class FakeVasp:
    def __init__(self):
        self.directories: list[Path] = []

    def run_vasp(self, *args, **kwargs):
        cwd = Path.cwd()
        assert (cwd / "INCAR").exists() and (cwd / "POSCAR").exists()
        self.directories.append(cwd)
        (cwd / "OUTCAR").write_text("fake VASP\n", encoding="utf-8")

    def task_document(self, path, **kwargs):
        from emmet.core.tasks import OutputDoc, TaskDoc
        from pymatgen.core import Structure

        structure = Structure.from_file(Path(path) / "POSCAR")
        forces = spring_forces(structure)
        return TaskDoc(
            structure=structure,
            output=OutputDoc(structure=structure, forces=forces.tolist(), energy=0.0),
            state="successful",
            dir_name=str(path),
            calcs_reversed=[],
        )


def incar(directory: Path) -> dict:
    from pymatgen.io.vasp.inputs import Incar

    for name in ("INCAR", "INCAR.gz"):
        if (directory / name).exists():
            return dict(Incar.from_file(directory / name))
    raise AssertionError(f"no INCAR in {directory}")


@pytest.fixture
def fake_vasp(monkeypatch):
    import atomate2.vasp.jobs.base as vasp_base

    fake = FakeVasp()
    original_write = vasp_base.write_vasp_input_set

    def write_without_potcar_files(*args, **kwargs):
        kwargs["potcar_spec"] = True
        return original_write(*args, **kwargs)

    monkeypatch.setattr(vasp_base, "run_vasp", fake.run_vasp)
    monkeypatch.setattr(vasp_base, "get_vasp_task_document", fake.task_document)
    monkeypatch.setattr(vasp_base, "write_vasp_input_set", write_without_potcar_files)
    return fake


def test_simulated_si_phonon_run_produces_force_constants_and_frequencies(tmp_path, fake_vasp, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = main.plan_calculation_request(
            structure_text=SI_POSCAR,
            fmt="poscar",
            desired_output="phonons",
        )
    spec = dict(plan.submission_spec)
    spec["paths"] = {**spec["paths"], "run_dir": str(tmp_path)}
    displacement_count = plan.generated_inputs["phonon_plans"][0]["displacement_count"]
    supercell_atoms = plan.generated_inputs["phonon_plans"][0]["supercell_atom_count"]

    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        execution.run_submission(spec)

    root = tmp_path.resolve()
    relax_dir = root / "stage_01"
    phonon_dir = root / "stage_02"
    displacement_root = phonon_dir / "run_phonon_displacements"
    analysis_dir = phonon_dir / "generate_frequencies_eigenvectors"

    # One fake VASP run for the relax, then one per Phonopy displacement, each
    # in its own directory; nothing ran in the run root.
    displacement_dirs = sorted(displacement_root.glob("phonon_static_*_of_*"))
    assert len(displacement_dirs) == displacement_count
    assert [path.relative_to(root).as_posix() for path in fake_vasp.directories] == [
        "stage_01",
        *[path.relative_to(root).as_posix() for path in displacement_dirs],
    ]
    assert not (root / "INCAR").exists()

    # Phonon-grade relax feeds the phonons; force calculations use atomate2's
    # phonon settings, PBE, non-spin, and no automatic NCORE.
    relax_incar = incar(relax_dir)
    assert relax_incar["ENCUT"] == 700
    assert relax_incar["EDIFF"] == 1e-7
    assert relax_incar["EDIFFG"] == -1e-3
    for directory in displacement_dirs:
        settings = incar(directory)
        assert settings["ENCUT"] == 700
        assert settings["EDIFF"] == 1e-7
        assert settings["NSW"] == 0
        assert settings["ISPIN"] == 1
        assert "NCORE" not in settings
        assert "GGA" not in settings
        assert "IVDW" not in settings

    # Phonopy force constants and frequencies.
    for name in ("phonopy.yaml", PHONON_FORCE_CONSTANTS_FILENAME, "phonon_band_structure.yaml", "phonon_dos.yaml"):
        assert (analysis_dir / name).exists(), name

    import phonopy
    from phonopy.file_IO import parse_FORCE_CONSTANTS

    force_constants = parse_FORCE_CONSTANTS(str(analysis_dir / PHONON_FORCE_CONSTANTS_FILENAME))
    assert force_constants.shape[1:] == (supercell_atoms, 3, 3)
    assert np.abs(force_constants).max() > 0

    phonon = phonopy.load(
        str(analysis_dir / "phonopy.yaml"),
        force_constants_filename=str(analysis_dir / PHONON_FORCE_CONSTANTS_FILENAME),
        produce_fc=False,
        log_level=0,
    )
    phonon.run_qpoints([[0, 0, 0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]])
    frequencies = phonon.get_qpoints_dict()["frequencies"]
    gamma = np.sort(frequencies[0])
    assert np.allclose(gamma[:3], 0.0, atol=0.05)  # acoustic sum rule (THz)
    assert np.all(gamma[3:] > 1.0)  # optical modes
    assert np.all(frequencies[1:] > 0.1)  # stable spring model away from Gamma
    assert len(gamma) == 3 * 2
    assert supercell_atoms == 64 and displacement_count == 1


def test_spring_model_is_at_equilibrium_for_the_input_structure():
    from pymatgen.core import Structure

    structure = Structure.from_str(SI_POSCAR, fmt="poscar")
    assert np.abs(spring_forces(structure)).max() < 1e-8
    structure.make_supercell([2, 2, 2])
    assert np.abs(spring_forces(structure)).max() < 1e-8
