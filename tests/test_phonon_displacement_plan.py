"""Deterministic Phonopy displacement planning (backend/phonons, M1).

These tests use the real pinned Phonopy. The plan is a scientific object, not a
workflow: its tasks are instances of one future phonon force stage.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.periodic_table import DummySpecies, Species
from pymatgen.electronic_structure.core import Magmom

from backend.calculations.models import StageSpec, WorkflowSpec
from backend.phonons import (
    DISPLACEMENT_PLAN_SCHEMA,
    PHONON_PLANNING_POLICY_ID,
    PROVISIONAL_PHONON_POLICY,
    TASK_KIND,
    DisplacementPlan,
    DisplacementTask,
    PhononPlanContractError,
    PhononPolicy,
    PhononPolicyError,
    build_displacement_plan,
    canonical_json,
    verify_displacement_plan,
)
from backend.phonons.records import (
    canonical_sha256,
    canonical_value,
    displaced_structure_sha256,
    plan_sha256,
    structure_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def si() -> Structure:
    return Structure(
        Lattice([[0.0, 2.734, 2.734], [2.734, 0.0, 2.734], [2.734, 2.734, 0.0]]),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def nacl() -> Structure:
    return Structure(
        Lattice([[0.0, 2.845, 2.845], [2.845, 0.0, 2.845], [2.845, 2.845, 0.0]]),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def gan() -> Structure:
    return Structure(
        Lattice.hexagonal(3.19, 5.19),
        ["Ga", "Ga", "N", "N"],
        [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.377], [2 / 3, 1 / 3, 0.877]],
    )


def mos2() -> Structure:
    z = 0.621
    return Structure(
        Lattice.hexagonal(3.16, 12.3),
        ["Mo", "Mo", "S", "S", "S", "S"],
        [
            [1 / 3, 2 / 3, 0.25],
            [2 / 3, 1 / 3, 0.75],
            [1 / 3, 2 / 3, z],
            [2 / 3, 1 / 3, z - 0.5],
            [2 / 3, 1 / 3, 1 - z],
            [1 / 3, 2 / 3, 1.5 - z],
        ],
    )


# name, structure factory, explicit supercell, space group, supercell atoms,
# displaced atom indices in dataset order
REFERENCE_CASES = [
    ("Si", si, DIAG2, ("Fd-3m", 227), 16, [0]),
    ("NaCl", nacl, DIAG2, ("Fm-3m", 225), 16, [0, 8]),
    ("GaN", gan, [[3, 0, 0], [0, 3, 0], [0, 0, 2]], ("P6_3mc", 186), 72, [0, 0, 36, 36]),
    ("MoS2", mos2, [[3, 0, 0], [0, 3, 0], [0, 0, 1]], ("P6_3/mmc", 194), 54, [0, 18, 18]),
]


@pytest.fixture(scope="module")
def nacl_plan() -> DisplacementPlan:
    return build_displacement_plan(nacl(), supercell_matrix=DIAG2)


def _rehash(record: dict) -> dict:
    record["plan_sha256"] = plan_sha256(record)
    return record


def _reversed_keys(value):
    if isinstance(value, dict):
        return {key: _reversed_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed_keys(item) for item in value]
    return value


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


# --- representative structures ----------------------------------------------------------


@pytest.mark.parametrize(
    "name, factory, matrix, space_group, natom, atoms",
    REFERENCE_CASES,
    ids=[case[0] for case in REFERENCE_CASES],
)
def test_reference_structures_plan_as_expected(name, factory, matrix, space_group, natom, atoms):
    plan = build_displacement_plan(factory(), supercell_matrix=matrix)

    assert plan.space_group == space_group
    assert plan.supercell_natom == natom
    assert plan.task_count == len(atoms)
    assert [task.atom_index for task in plan.tasks] == atoms
    assert [task.task_id for task in plan.tasks] == [f"disp_{i:03d}" for i in range(1, len(atoms) + 1)]
    assert [task.dataset_index for task in plan.tasks] == list(range(len(atoms)))
    for task in plan.tasks:
        assert math.isclose(float(np.linalg.norm(task.displacement)), 0.01, rel_tol=1e-9)
    record = plan.to_dict()
    assert record["schema"] == DISPLACEMENT_PLAN_SCHEMA
    assert record["software"] == {
        "phonopy": phonopy.__version__,
        "spglib": __import__("importlib").metadata.version("spglib"),
    }
    assert record["symmetry"]["symprec"] == 1e-5
    assert record["supercell_matrix"] == matrix


def test_reference_datasets_have_the_expected_displacement_directions():
    """Semantic golden values (direction in supercell fractional axes, scaled to
    unit max component): catch an upstream change in Phonopy's displacement
    choices without pinning platform-sensitive last bits. Plus/minus pairs
    appear only where symmetry does not make them equivalent (PM = auto)."""

    expected = {
        "Si": [(0, (1.0, 0.0, 0.0))],
        "NaCl": [(0, (1.0, 0.0, 0.0)), (8, (1.0, 0.0, 0.0))],
        "GaN": [(0, (1.0, 0.0, 1.0)), (0, (-1.0, 0.0, -1.0)), (36, (1.0, 0.0, 1.0)), (36, (-1.0, 0.0, -1.0))],
        "MoS2": [(0, (1.0, 0.0, 1.0)), (18, (1.0, 0.0, 1.0)), (18, (-1.0, 0.0, -1.0))],
    }
    for name, factory, matrix, *_ in REFERENCE_CASES:
        plan = build_displacement_plan(factory(), supercell_matrix=matrix)
        lattice = np.array(plan.to_dict()["supercell"]["lattice"])
        got = []
        for task in plan.tasks:
            fractional = np.linalg.solve(lattice.T, np.array(task.displacement))
            direction = np.round(fractional / np.abs(fractional).max(), 6)
            got.append((task.atom_index, tuple(float(x) + 0.0 for x in direction)))
        assert got == expected[name], name


def test_non_diagonal_supercell_matrix_is_supported_and_validated():
    conventional = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    plan = build_displacement_plan(nacl(), supercell_matrix=conventional)
    assert plan.supercell_natom == 8
    assert plan.to_dict()["supercell_matrix"] == conventional
    assert plan.space_group == ("Fm-3m", 225)


# --- determinism -------------------------------------------------------------------------


def test_repeated_builds_are_identical(nacl_plan):
    again = build_displacement_plan(nacl(), supercell_matrix=DIAG2)
    assert again.to_json() == nacl_plan.to_json()
    assert again.plan_sha256 == nacl_plan.plan_sha256
    assert again == nacl_plan


def test_round_trips_preserve_identity(nacl_plan):
    assert DisplacementPlan.from_dict(nacl_plan.to_dict()) == nacl_plan
    assert DisplacementPlan.from_json(nacl_plan.to_json()) == nacl_plan
    pretty = json.dumps(nacl_plan.to_dict(), indent=4)
    assert DisplacementPlan.from_json(pretty).plan_sha256 == nacl_plan.plan_sha256


def test_key_order_cannot_alter_identity(nacl_plan):
    shuffled = _reversed_keys(nacl_plan.to_dict())
    assert list(shuffled) != list(nacl_plan.to_dict())
    assert canonical_json(shuffled) == nacl_plan.to_json()
    assert plan_sha256(shuffled) == nacl_plan.plan_sha256
    assert DisplacementPlan.from_dict(shuffled) == nacl_plan


def test_plan_hash_covers_everything_but_itself(nacl_plan):
    record = nacl_plan.to_dict()
    body = {key: value for key, value in record.items() if key != "plan_sha256"}
    assert record["plan_sha256"] == canonical_sha256(body)


def test_caller_structure_is_not_mutated():
    structure = gan()
    structure.add_site_property("magmom", [0.0] * len(structure))
    before = copy.deepcopy(structure.as_dict())
    frac_before = structure.frac_coords.copy()
    build_displacement_plan(structure, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    assert structure.as_dict() == before
    assert np.array_equal(structure.frac_coords, frac_before)


def test_returned_dicts_are_copies_and_the_plan_is_immutable(nacl_plan):
    record = nacl_plan.to_dict()
    record["tasks"][0]["atom_index"] = 99
    record["policy"]["symprec"] = 0.1
    assert nacl_plan.tasks[0].atom_index == 0
    assert nacl_plan.policy.symprec == 1e-5
    with pytest.raises(AttributeError):
        nacl_plan.tasks[0].atom_index = 3
    with pytest.raises(AttributeError):
        nacl_plan._canonical = "{}"


def test_working_structure_is_recorded_as_given_without_wrapping():
    structure = Structure(nacl().lattice, ["Na", "Cl"], [[1.0, 0.0, -0.0], [0.5, 0.5, 1.5]])
    plan = build_displacement_plan(structure, supercell_matrix=DIAG2)
    assert plan.to_dict()["working_structure"]["coords"] == [[1.0, 0.0, 0.0], [0.5, 0.5, 1.5]]


def test_negative_zero_does_not_change_identity():
    plus = Structure(nacl().lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    minus = Structure(nacl().lattice, ["Na", "Cl"], [[-0.0, 0.0, -0.0], [0.5, 0.5, 0.5]])
    assert (
        build_displacement_plan(plus, supercell_matrix=DIAG2).plan_sha256
        == build_displacement_plan(minus, supercell_matrix=DIAG2).plan_sha256
    )


def test_tiny_float_differences_are_not_hidden():
    moved = Structure(nacl().lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5 + 1e-15]])
    assert (
        build_displacement_plan(moved, supercell_matrix=DIAG2).plan_sha256
        != build_displacement_plan(nacl(), supercell_matrix=DIAG2).plan_sha256
    )


def test_identity_is_stable_across_processes_hash_seeds_locales_and_directories(nacl_plan, tmp_path):
    script = (
        "import json, warnings; warnings.filterwarnings('ignore');"
        "from tests.test_phonon_displacement_plan import nacl, DIAG2;"
        "from backend.phonons import build_displacement_plan;"
        "print(build_displacement_plan(nacl(), supercell_matrix=DIAG2).plan_sha256)"
    )
    hashes = set()
    for seed, locale in (("0", "C"), ("12345", "C.UTF-8")):
        env = {**os.environ, "PYTHONHASHSEED": seed, "LC_ALL": locale, "PYTHONPATH": str(REPO_ROOT)}
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stderr
        hashes.add(completed.stdout.strip())
    assert hashes == {nacl_plan.plan_sha256}


def test_no_path_host_or_environment_leaks_into_the_plan(nacl_plan, tmp_path):
    forbidden = {str(tmp_path), os.getcwd(), str(REPO_ROOT), platform.node(), sys.executable, sys.prefix}
    forbidden.discard("")
    for text in _strings(nacl_plan.to_dict()):
        for item in forbidden:
            assert item not in text, (item, text)
        assert "\\" not in text


# --- stage != task ---------------------------------------------------------------------


def test_displacements_are_task_instances_not_workflow_stages(nacl_plan):
    record = nacl_plan.to_dict()
    assert record["task_kind"] == TASK_KIND == "phonon_displacement"
    for task in record["tasks"]:
        assert set(task) == {"task_id", "dataset_index", "atom_index", "displacement", "structure_sha256"}
    for key in _strings(record):
        assert key not in {"stage_type", "theory", "modifiers", "stages", "workflow_spec", "stage_dirs"}
    assert all(type(task) is DisplacementTask for task in nacl_plan.tasks)
    assert not any(isinstance(task, (StageSpec, WorkflowSpec)) for task in nacl_plan.tasks)


def test_plan_carries_no_execution_or_scheduling_semantics(nacl_plan):
    text = nacl_plan.to_json().lower()
    for word in ("slurm", "sbatch", "array", "parallel", "sequential", "walltime", "ntasks", "directory", "path"):
        assert word not in text, word


# --- explicit upstream arguments ---------------------------------------------------------


class _RecordingPhonopy(phonopy.Phonopy):
    calls: list = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append(("init", args, dict(kwargs)))
        super().__init__(*args, **kwargs)

    def generate_displacements(self, *args, **kwargs):
        type(self).calls.append(("generate_displacements", args, dict(kwargs)))
        return super().generate_displacements(*args, **kwargs)


def test_every_planning_argument_is_passed_explicitly(monkeypatch):
    _RecordingPhonopy.calls = []
    monkeypatch.setattr(phonopy, "Phonopy", _RecordingPhonopy)

    plan = build_displacement_plan(gan(), supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    (init_name, init_args, init_kwargs), (gen_name, gen_args, gen_kwargs) = _RecordingPhonopy.calls
    assert init_name == "init" and gen_name == "generate_displacements"
    assert len(init_args) == 1 and gen_args == ()
    assert init_args[0].magnetic_moments is None
    assert init_kwargs == {
        "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        "primitive_matrix": "auto",
        "symprec": 1e-5,
        "is_symmetry": True,
        "distinguish_symbol_index": False,
        "use_SNF_supercell": False,
        "calculator": "vasp",
        "lang": "C",
    }
    assert gen_kwargs == {
        "distance": 0.01,
        "is_plusminus": "auto",
        "is_diagonal": True,
        "is_trigonal": False,
        "number_of_snapshots": None,
        "random_seed": None,
    }
    assert plan.to_dict()["phonopy_arguments"] == {"construct": init_kwargs, "displacements": gen_kwargs}


def test_resolved_primitive_matrix_is_recorded_explicitly():
    conventional = Structure(
        Lattice.cubic(5.69),
        ["Na"] * 4 + ["Cl"] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
         [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )
    plan = build_displacement_plan(conventional, supercell_matrix=DIAG2)
    record = plan.to_dict()
    assert np.allclose(record["primitive_matrix"], [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    assert record["primitive_natom"] == 2
    assert record["phonopy_arguments"]["construct"]["primitive_matrix"] == "auto"


def test_phonopy_backend_fallback_is_refused(monkeypatch):
    class _RustOnly(phonopy.Phonopy):
        @property
        def lang(self):
            return "Rust"

    monkeypatch.setattr(phonopy, "Phonopy", _RustOnly)
    with pytest.raises(RuntimeError, match="backend 'Rust'"):
        build_displacement_plan(nacl(), supercell_matrix=DIAG2)


def test_unexpected_upstream_dataset_layout_is_refused(monkeypatch):
    original = phonopy.Phonopy.generate_displacements

    def generate(self, **kwargs):
        original(self, **kwargs)
        self._dataset["first_atoms"][0]["forces"] = None

    monkeypatch.setattr(phonopy.Phonopy, "generate_displacements", generate)
    with pytest.raises(RuntimeError, match="unexpected Phonopy displacement entry keys"):
        build_displacement_plan(nacl(), supercell_matrix=DIAG2)


def test_plan_reconstruction_matches_phonopys_displaced_supercells(nacl_plan):
    record = nacl_plan.to_dict()
    unit = record["working_structure"]
    ph = phonopy.Phonopy(
        phonopy.structure.atoms.PhonopyAtoms(
            symbols=unit["species"], cell=unit["lattice"], scaled_positions=unit["coords"]
        ),
        **record["phonopy_arguments"]["construct"],
    )
    ph.generate_displacements(**record["phonopy_arguments"]["displacements"])
    for task, cell in zip(nacl_plan.tasks, ph.supercells_with_displacements):
        supercell = record["supercell"]
        recomputed = displaced_structure_sha256(supercell, task.atom_index, task.displacement)
        assert recomputed == task.structure_sha256
        expected = np.array(supercell["coords"])
        expected[task.atom_index] += task.displacement
        assert np.allclose(cell.positions, expected, rtol=0, atol=1e-10)


# --- symprec sensitivity and policy identity ---------------------------------------------


def _perturbed_si() -> Structure:
    return Structure(si().lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.2501, 0.25, 0.25]])


def test_symprec_changes_the_plan_so_it_is_part_of_policy_identity():
    approved = build_displacement_plan(_perturbed_si(), supercell_matrix=DIAG2)
    loose_policy = PhononPolicy(policy_id="test.loose_symprec", symprec=1e-3)
    loose = build_displacement_plan(_perturbed_si(), loose_policy, supercell_matrix=DIAG2)

    assert approved.space_group == ("C2/m", 12) and approved.task_count == 4
    assert loose.space_group == ("Fd-3m", 227) and loose.task_count == 1
    assert approved.plan_sha256 != loose.plan_sha256


@pytest.mark.parametrize(
    "overrides",
    [
        {"symprec": 1e-3},
        {"displacement_distance_angstrom": 0.02},
        {"is_plusminus": True},
        {"is_diagonal": False},
        {"policy_version": 2},
    ],
)
def test_bmd_policy_values_cannot_change_without_a_new_registered_version(overrides):
    with pytest.raises(PhononPolicyError):
        PhononPolicy(**overrides)


def test_provisional_policy_is_marked_not_executable():
    record = PROVISIONAL_PHONON_POLICY.to_dict()
    assert record["policy_id"] == PHONON_PLANNING_POLICY_ID
    assert record["status"] == "provisional_not_executable"
    assert set(record) == {
        "policy_id", "policy_version", "status", "displacement_distance_angstrom",
        "is_plusminus", "is_diagonal", "symprec", "primitive_matrix",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symprec": 0},
        {"symprec": True},
        {"displacement_distance_angstrom": float("nan")},
        {"displacement_distance_angstrom": -0.01},
        {"is_plusminus": "yes"},
        {"is_plusminus": 1},
        {"is_diagonal": 1},
        {"primitive_matrix": "P"},
        {"policy_version": True},
    ],
)
def test_malformed_policies_are_rejected(kwargs):
    with pytest.raises(PhononPolicyError):
        PhononPolicy(policy_id="test.other", **kwargs)


# --- inputs -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix",
    [
        [2, 2, 2],
        [[2, 0, 0], [0, 2, 0]],
        [[2.0, 0, 0], [0, 2, 0], [0, 0, 2]],
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 2, 0], [0, 0, 2]],
        [[-2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "2 2 2",
    ],
)
def test_supercell_matrix_must_be_explicit_integral_and_right_handed(matrix):
    with pytest.raises(ValueError):
        build_displacement_plan(nacl(), supercell_matrix=matrix)


def test_supercell_matrix_is_required():
    with pytest.raises(TypeError):
        build_displacement_plan(nacl())


def test_numpy_integer_matrix_gives_the_same_plan(nacl_plan):
    plan = build_displacement_plan(nacl(), supercell_matrix=np.diag([2, 2, 2]).astype(np.int64))
    assert plan == nacl_plan


def test_oxidation_state_decorations_do_not_change_identity(nacl_plan):
    decorated = Structure(nacl().lattice, [Species("Na", 1), Species("Cl", -1)], nacl().frac_coords)
    assert build_displacement_plan(decorated, supercell_matrix=DIAG2) == nacl_plan


def test_unsupported_structures_fail_closed():
    with pytest.raises(TypeError):
        build_displacement_plan(Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]), supercell_matrix=DIAG2)
    with pytest.raises(TypeError):
        build_displacement_plan(nacl().as_dict(), supercell_matrix=DIAG2)
    disordered = Structure(nacl().lattice, [{"Na": 0.5, "K": 0.5}, "Cl"], nacl().frac_coords)
    with pytest.raises(ValueError, match="ordered"):
        build_displacement_plan(disordered, supercell_matrix=DIAG2)
    dummy = Structure(nacl().lattice, [DummySpecies("X"), "Cl"], nacl().frac_coords)
    with pytest.raises(ValueError, match="not a chemical element"):
        build_displacement_plan(dummy, supercell_matrix=DIAG2)
    vector_magnetic = Structure(
        nacl().lattice, ["Na", "Cl"], nacl().frac_coords,
        site_properties={"magmom": [Magmom([0.0, 0.0, 1.0]), Magmom([0.0, 0.0, 0.0])]},
    )
    with pytest.raises(ValueError, match="magnetic"):
        build_displacement_plan(vector_magnetic, supercell_matrix=DIAG2)
    magnetic = nacl()
    magnetic.add_site_property("magmom", [1.0, 0.0])
    with pytest.raises(ValueError, match="magnetic"):
        build_displacement_plan(magnetic, supercell_matrix=DIAG2)
    with pytest.raises(TypeError):
        build_displacement_plan(nacl(), PROVISIONAL_PHONON_POLICY.to_dict(), supercell_matrix=DIAG2)


# --- species-level spin (non-magnetic boundary) -------------------------------------------


def _bcc_fe(specie) -> Structure:
    return Structure(Lattice.cubic(2.87), [specie, specie], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="module")
def plain_fe_plan() -> DisplacementPlan:
    return build_displacement_plan(_bcc_fe("Fe"), supercell_matrix=DIAG2)


def test_plain_non_magnetic_fe_is_accepted(plain_fe_plan):
    assert plain_fe_plan.space_group == ("Im-3m", 229)
    assert plain_fe_plan.to_dict()["working_structure"]["species"] == ["Fe", "Fe"]


@pytest.mark.parametrize(
    "specie",
    [Species("Fe", 0), Species("Fe", 0, spin=0), Species("Fe", 2, spin=0), Species("Fe", 0, spin=0.0)],
    ids=["spin-absent", "spin-0", "Fe2+-spin-0", "spin-0.0"],
)
def test_absent_or_zero_species_spin_is_not_rejected(specie, plain_fe_plan):
    assert build_displacement_plan(_bcc_fe(specie), supercell_matrix=DIAG2) == plain_fe_plan


@pytest.mark.parametrize("spin", [5, -2, 0.5, -0.5], ids=["spin+5", "spin-2", "spin+0.5", "spin-0.5"])
def test_nonzero_species_spin_is_rejected(spin):
    with pytest.raises(ValueError, match="carries a spin"):
        build_displacement_plan(_bcc_fe(Species("Fe", 0, spin=spin)), supercell_matrix=DIAG2)


def test_one_spin_decorated_site_is_enough_to_reject():
    structure = Structure(
        Lattice.cubic(2.87), ["Fe", Species("Fe", 0, spin=5)], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    with pytest.raises(ValueError, match="carries a spin"):
        build_displacement_plan(structure, supercell_matrix=DIAG2)


def test_species_spin_survives_serialization_and_is_still_rejected():
    original = _bcc_fe(Species("Fe", 0, spin=5))
    for restored in (
        Structure.from_dict(original.as_dict()),
        Structure.from_dict(json.loads(json.dumps(original.as_dict()))),
        Structure.from_str(original.to(fmt="json"), fmt="json"),
        original.copy(),
    ):
        assert restored[0].specie.spin == 5
        with pytest.raises(ValueError, match="carries a spin"):
            build_displacement_plan(restored, supercell_matrix=DIAG2)


def test_species_spin_is_rejected_before_phonopy_is_invoked(monkeypatch):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("Phonopy must not be reached for a spin-decorated structure")

    monkeypatch.setattr(phonopy, "Phonopy", _must_not_run)
    with pytest.raises(ValueError, match="carries a spin"):
        build_displacement_plan(_bcc_fe(Species("Fe", 0, spin=5)), supercell_matrix=DIAG2)


def test_magmom_site_property_is_still_rejected_alongside_species_spin():
    magnetic = _bcc_fe("Fe")
    magnetic.add_site_property("magmom", [2.2, 2.2])
    with pytest.raises(ValueError, match="magnetic moments are present"):
        build_displacement_plan(magnetic, supercell_matrix=DIAG2)


# --- record validation -------------------------------------------------------------------


def _mutations():
    def setter(path, value):
        def apply(record):
            target = record
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        return apply

    def delete(path):
        def apply(record):
            target = record
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
        return apply

    def swap_tasks(record):
        record["tasks"].reverse()

    def duplicate_task(record):
        record["tasks"].append(copy.deepcopy(record["tasks"][0]))

    def drop_task(record):
        record["tasks"].pop()

    def drop_dataset_entry_and_task(record):
        record["dataset"]["first_atoms"].pop()
        record["tasks"].pop()

    def consistent_wrong_norm(record):
        scaled = [x * 2 for x in record["dataset"]["first_atoms"][0]["displacement"]]
        record["dataset"]["first_atoms"][0]["displacement"] = scaled
        record["tasks"][0]["displacement"] = scaled
        record["tasks"][0]["structure_sha256"] = displaced_structure_sha256(
            record["supercell"], record["tasks"][0]["atom_index"], scaled
        )

    def supercell_coordinate(record):
        record["supercell"]["coords"][3][0] += 1e-9

    def working_species(record):
        record["working_structure"]["species"] = ["K", "Cl"]

    return {
        "schema": setter(["schema"], "bmd_compute.phonon_plan"),
        "schema_version_2": setter(["schema_version"], 2),
        "schema_version_bool": setter(["schema_version"], True),
        "unexpected_key": setter(["stages"], []),
        "missing_key": delete(["software"]),
        "policy_symprec": setter(["policy", "symprec"], 1e-3),
        "policy_version": setter(["policy", "policy_version"], 2),
        "policy_status": setter(["policy", "status"], "approved"),
        "policy_extra_key": setter(["policy", "supercell_matrix"], DIAG2),
        "phonopy_argument_drift": setter(["phonopy_arguments", "construct", "use_SNF_supercell"], True),
        "phonopy_argument_missing": delete(["phonopy_arguments", "displacements", "is_trigonal"]),
        "symmetry_symprec": setter(["symmetry", "symprec"], 1e-3),
        "symmetry_number": setter(["symmetry", "number"], 0),
        "supercell_matrix_float": setter(["supercell_matrix"], [[2.0, 0, 0], [0, 2, 0], [0, 0, 2]]),
        "supercell_matrix_bool": setter(["supercell_matrix"], [[True, 0, 0], [0, 2, 0], [0, 0, 2]]),
        "supercell_matrix_shape": setter(["supercell_matrix"], [[2, 0, 0], [0, 2, 0]]),
        "supercell_matrix_other": setter(["supercell_matrix"], [[3, 0, 0], [0, 2, 0], [0, 0, 2]]),
        "supercell_matrix_singular": setter(["supercell_matrix"], [[0, 0, 0], [0, 2, 0], [0, 0, 2]]),
        "primitive_matrix_shape": setter(["primitive_matrix"], [[1.0, 0.0, 0.0]]),
        "primitive_matrix_int": setter(["primitive_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "primitive_natom": setter(["primitive_natom"], 3),
        "working_hash": setter(["working_structure", "sha256"], "0" * 64),
        "working_species": working_species,
        "working_coords_nan": setter(["working_structure", "coords", 0, 0], float("nan")),
        "supercell_hash": setter(["supercell", "sha256"], "f" * 64),
        "supercell_coordinate": supercell_coordinate,
        "supercell_coords_type": setter(["supercell", "coords_type"], "fractional"),
        "dataset_natom": setter(["dataset", "natom"], 15),
        "dataset_number_out_of_range": setter(["dataset", "first_atoms", 0, "number"], 16),
        "dataset_extra_entry_key": setter(["dataset", "first_atoms", 0, "forces"], None),
        "task_kind": setter(["task_kind"], "workflow_stage"),
        "task_id_format": setter(["tasks", 0, "task_id"], "disp_1"),
        "task_id_wrong_position": setter(["tasks", 0, "task_id"], "disp_002"),
        "task_order": swap_tasks,
        "task_duplicate": duplicate_task,
        "task_missing": drop_task,
        "task_count_and_dataset_stale": drop_dataset_entry_and_task,
        "task_dataset_index": setter(["tasks", 1, "dataset_index"], 0),
        "task_dataset_index_bool": setter(["tasks", 0, "dataset_index"], False),
        "task_atom_index": setter(["tasks", 1, "atom_index"], 0),
        "task_displacement": setter(["tasks", 0, "displacement"], [0.0, 0.0, -0.01]),
        "task_structure_hash": setter(["tasks", 0, "structure_sha256"], "a" * 64),
        "task_structure_hash_case": setter(["tasks", 0, "structure_sha256"], "A" * 64),
        "task_stage_fields": setter(["tasks", 0, "stage_type"], "static"),
        "displacement_norm": consistent_wrong_norm,
        "plan_hash": setter(["plan_sha256"], "0" * 64),
    }


MUTATIONS = _mutations()


@pytest.mark.parametrize("name", sorted(MUTATIONS))
@pytest.mark.parametrize("rehash", [False, True], ids=["stale-hash", "rehashed"])
def test_malformed_or_inconsistent_records_are_rejected(nacl_plan, name, rehash):
    if name == "plan_hash" and rehash:
        pytest.skip("rehashing would undo this mutation")
    if name == "task_count_and_dataset_stale" and rehash:
        pytest.skip("a consistently truncated dataset is caught only by a rebuild; see the forgery test")
    record = nacl_plan.to_dict()
    MUTATIONS[name](record)
    if rehash:
        try:
            _rehash(record)
        except PhononPlanContractError:
            return  # not even hashable canonical data
    with pytest.raises(PhononPlanContractError):
        DisplacementPlan.from_dict(record)


def test_validation_does_not_repair_records(nacl_plan):
    record = nacl_plan.to_dict()
    record["tasks"].reverse()
    snapshot = copy.deepcopy(record)
    with pytest.raises(PhononPlanContractError):
        DisplacementPlan.from_dict(record)
    assert record == snapshot


def test_internally_consistent_forgery_is_not_scientific_authority(nacl_plan):
    """A client can build a self-consistent record; only a rebuild is authority."""

    record = nacl_plan.to_dict()
    flipped = [-x for x in record["dataset"]["first_atoms"][1]["displacement"]]
    record["dataset"]["first_atoms"][1]["displacement"] = flipped
    record["tasks"][1]["displacement"] = flipped
    record["tasks"][1]["structure_sha256"] = displaced_structure_sha256(
        record["supercell"], record["tasks"][1]["atom_index"], flipped
    )
    forged = DisplacementPlan.from_dict(_rehash(record))

    truncated_record = nacl_plan.to_dict()
    truncated_record["dataset"]["first_atoms"].pop()
    truncated_record["tasks"].pop()
    truncated = DisplacementPlan.from_dict(_rehash(truncated_record))
    assert truncated.task_count == nacl_plan.task_count - 1

    assert verify_displacement_plan(nacl_plan, nacl(), supercell_matrix=DIAG2) == nacl_plan
    for bad in (forged, truncated):
        with pytest.raises(PhononPlanContractError, match="does not match the plan rebuilt"):
            verify_displacement_plan(bad, nacl(), supercell_matrix=DIAG2)
    with pytest.raises(PhononPlanContractError):
        verify_displacement_plan(nacl_plan, nacl(), supercell_matrix=[[3, 0, 0], [0, 2, 0], [0, 0, 2]])


def test_direct_construction_requires_canonical_text(nacl_plan):
    assert DisplacementPlan(nacl_plan.to_json()) == nacl_plan
    with pytest.raises(PhononPlanContractError, match="canonical"):
        DisplacementPlan(json.dumps(nacl_plan.to_dict(), indent=2))
    with pytest.raises(PhononPlanContractError):
        DisplacementPlan(nacl_plan.to_dict())


def test_task_ids_stay_unique_and_parseable_beyond_999():
    from backend.phonons.records import TASK_ID_PATTERN, task_id_for_index

    assert [task_id_for_index(i) for i in (0, 9, 998, 999, 12344)] == [
        "disp_001", "disp_010", "disp_999", "disp_1000", "disp_12345"
    ]
    assert all(TASK_ID_PATTERN.match(task_id_for_index(i)) for i in range(0, 2000, 37))


def test_plan_json_rejects_non_json_and_non_objects():
    with pytest.raises(PhononPlanContractError):
        DisplacementPlan.from_json("not json")
    with pytest.raises(PhononPlanContractError):
        DisplacementPlan.from_json("[]")


# --- canonicalization rules ---------------------------------------------------------------


def test_canonical_value_rejects_non_plain_types():
    for bad in (np.float64(0.1), np.int64(1), np.array([1.0]), {1: "x"}, {"x": {1, 2}}, Path("a"), b"x"):
        with pytest.raises(PhononPlanContractError):
            canonical_value(bad)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(PhononPlanContractError):
            canonical_json({"x": bad})


def test_canonical_json_rules():
    assert canonical_json({"b": [1, 2.5, -0.0, True, None, "é"], "a": (0.1,)}) == (
        '{"a":[0.1],"b":[1,2.5,0.0,true,null,"\\u00e9"]}'
    )
    assert json.loads(canonical_json({"x": 0.1 + 0.2}))["x"] == 0.1 + 0.2


def test_structure_record_hash_covers_lattice_species_coords_and_frame():
    unit = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    stretched = [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    base = structure_record(unit, ["Si"], [[0.0, 0.0, 0.0]], "fractional")
    for args in (
        (stretched, ["Si"], [[0.0, 0.0, 0.0]], "fractional"),
        (unit, ["Ge"], [[0.0, 0.0, 0.0]], "fractional"),
        (unit, ["Si"], [[0.5, 0.0, 0.0]], "fractional"),
        (unit, ["Si"], [[0.0, 0.0, 0.0]], "cartesian"),
    ):
        assert structure_record(*args)["sha256"] != base["sha256"]
    with pytest.raises(PhononPlanContractError):
        structure_record([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["Si"], [[0, 0, 0]], "fractional")
