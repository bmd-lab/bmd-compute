"""Phonon working-structure boundary (Phonopy M2b).

Stage-1 structure (B) -> ``bmd_compute.phonon_working_structure`` record ->
working structure (C) -> M1 displacement plan -> M2a task set.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import spglib
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.periodic_table import DummySpecies, Species
from pymatgen.electronic_structure.core import Magmom

from backend.phonons.records import canonical_json, structure_record
from backend.phonons.working_structure import (
    DROP_ZERO_MAGMOM,
    DROP_ZERO_VELOCITIES,
    REDUCE_SPECIES_TO_ELEMENTS,
    SYMMETRY_IDEALIZATION,
    PhononWorkingStructure,
    PhononWorkingStructureError,
    WorkingStructurePolicy,
    prepare_phonon_working_structure,
    record_sha256,
    validate_working_structure_record,
    verify_phonon_working_structure,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SYMPREC = 1e-5


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


def gan(third: float = 1 / 3, z: float = 0.377) -> Structure:
    return Structure(
        Lattice.hexagonal(3.19, 5.19),
        ["Ga", "Ga", "N", "N"],
        [[third, 2 * third, 0.0], [2 * third, third, 0.5], [third, 2 * third, z], [2 * third, third, z + 0.5]],
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


REFERENCE = [
    ("Si", si, 227, "Fd-3m"),
    ("NaCl", nacl, 225, "Fm-3m"),
    ("GaN", gan, 186, "P6_3mc"),
    ("MoS2", mos2, 194, "P6_3/mmc"),
]


def rehash(data: dict) -> dict:
    data["record_sha256"] = record_sha256(data)
    return data


def reject(data, match: str | None = None):
    with pytest.raises(PhononWorkingStructureError, match=match):
        validate_working_structure_record(data)
    with pytest.raises(PhononWorkingStructureError):
        PhononWorkingStructure.from_dict(data)


def refuse(structure, code: str):
    with pytest.raises(PhononWorkingStructureError) as excinfo:
        prepare_phonon_working_structure(structure)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def snapshot(structure: Structure):
    return (
        structure.lattice.matrix.tolist(),
        [repr(site.specie) for site in structure],
        [str(type(site.specie)) for site in structure],
        structure.frac_coords.tolist(),
        copy.deepcopy([dict(site.properties) for site in structure]),
        canonical_json(json.loads(json.dumps(structure.as_dict(), default=str))),
    )


def spglib_number(lattice, coords, species, symprec):
    numbers = [{"Si": 14, "Na": 11, "Cl": 17, "Ga": 31, "N": 7, "Mo": 42, "S": 16}[s] for s in species]
    return spglib.get_symmetry_dataset((lattice, coords, numbers), symprec=symprec).number


# --- ordinary structures ---------------------------------------------------------------


@pytest.mark.parametrize("name, factory, number, symbol", REFERENCE, ids=[r[0] for r in REFERENCE])
def test_reference_structures(name, factory, number, symbol):
    structure = factory()
    first = prepare_phonon_working_structure(structure)
    second = prepare_phonon_working_structure(factory())
    data = first.to_dict()

    assert first.to_json() == second.to_json()
    assert first.incoming_sha256 != first.working_sha256
    assert first.transformations == (SYMMETRY_IDEALIZATION,)
    assert data["symmetry"]["incoming"] == data["symmetry"]["working"]
    assert data["symmetry"]["incoming"]["number"] == number
    assert data["symmetry"]["incoming"]["international"] == symbol
    working = data["working"]
    assert working["species"] == [site.specie.symbol for site in structure]
    assert working["coords_type"] == "fractional"
    assert np.max(np.abs(np.array(working["lattice"]) - structure.lattice.matrix)) < 1e-12
    assert data["idealization"]["max_site_shift_angstrom"] <= SYMPREC
    assert data["incoming"]["lattice"] == structure.lattice.matrix.tolist()
    assert data["incoming"]["coords"] == structure.frac_coords.tolist()
    assert PhononWorkingStructure.from_json(first.to_json()) == first


def test_policy_and_software_are_recorded_explicitly():
    data = prepare_phonon_working_structure(si()).to_dict()
    assert data["policy"] == {
        "policy_id": "bmd_compute.phonon_working_structure",
        "policy_version": 1,
        "status": "provisional_not_executable",
        "method": "spglib_symmetry_idealization_in_input_setting",
        "symprec_angstrom": 1e-5,
        "angle_tolerance_degrees": -1.0,
    }
    assert set(data["software"]) == {"spglib", "numpy"}
    assert data["software"]["spglib"] == spglib.__version__


def test_record_has_no_execution_or_methodology_content():
    text = prepare_phonon_working_structure(gan()).to_json().lower()
    for word in ("incar", "kpoint", "slurm", "job", "task", "force", "timestamp", "created", "host",
                 "supercell", "displacement", "primitive", "/home", "/tmp"):
        assert word not in text


def test_working_structure_object_round_trips_to_the_same_identity():
    record = prepare_phonon_working_structure(mos2())
    working = record.working_structure()
    rebuilt = structure_record(
        working.lattice.matrix.tolist(), [site.specie.symbol for site in working],
        working.frac_coords.tolist(), "fractional",
    )
    assert rebuilt["sha256"] == record.working_sha256
    assert working.site_properties == {}


# --- standardization invariants --------------------------------------------------------


def _rotation(axis, degrees):
    axis = np.array(axis, dtype=float) / np.linalg.norm(axis)
    angle = math.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def test_cartesian_frame_is_preserved_for_a_rotated_cell():
    rotation = _rotation([1, 2, 3], 37.0)
    base = gan(third=0.333333)
    rotated = Structure(Lattice(base.lattice.matrix @ rotation.T), base.species, base.frac_coords)

    plain = np.array(prepare_phonon_working_structure(base).to_dict()["working"]["lattice"])
    turned = np.array(prepare_phonon_working_structure(rotated).to_dict()["working"]["lattice"])

    assert np.max(np.abs(turned - rotated.lattice.matrix)) < 1e-9
    assert np.max(np.abs(turned - plain @ rotation.T)) < 1e-9


def test_handedness_atom_count_order_and_origin_are_preserved():
    shifted = Structure(si().lattice, ["Si", "Si"], [[0.1, 0.2, 0.3], [0.35, 0.45, 0.55]])
    reordered = Structure(nacl().lattice, ["Cl", "Na"], [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
    for structure in (shifted, reordered, gan(z=0.3770001)):
        data = prepare_phonon_working_structure(structure).to_dict()
        working = data["working"]
        assert len(working["coords"]) == len(structure)
        assert working["species"] == [site.specie.symbol for site in structure]
        assert np.linalg.det(np.array(working["lattice"])) > 0
        assert np.max(np.abs(np.array(working["coords"]) - structure.frac_coords)) < 1e-6
    # No origin shift: averaging symmetry images of an exact structure changes
    # at most the last bits of its coordinates.
    origin = prepare_phonon_working_structure(shifted).to_dict()["working"]["coords"][0]
    assert np.max(np.abs(np.array(origin) - [0.1, 0.2, 0.3])) <= 1e-15


def test_nothing_is_wrapped_into_the_unit_cell():
    outside = Structure(si().lattice, ["Si", "Si"], [[1.0, -1.0, 2.0], [1.25, 0.25, -0.75]])
    working = prepare_phonon_working_structure(outside).to_dict()["working"]["coords"]
    assert working == [[1.0, -1.0, 2.0], [1.25, 0.25, -0.75]]


def test_full_spglib_standardization_would_change_atom_count_and_frame():
    """Why M2b idealizes in the input setting instead of standardizing."""

    structure = gan(third=0.3333)
    cell = (structure.lattice.matrix, structure.frac_coords, [31, 31, 7, 7])
    std = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False, symprec=SYMPREC)
    dataset = spglib.get_symmetry_dataset(cell, symprec=SYMPREC)
    assert len(std[2]) != len(structure)
    assert not np.allclose(dataset.std_rotation_matrix, np.eye(3))
    assert len(spglib.standardize_cell((si().lattice.matrix, si().frac_coords, [14, 14]), symprec=SYMPREC)[2]) == 8


def test_idealization_is_a_fixed_point():
    first = prepare_phonon_working_structure(gan(third=0.333333))
    again = prepare_phonon_working_structure(first.working_structure())
    a, b = first.to_dict(), again.to_dict()
    assert np.max(np.abs(np.array(a["working"]["coords"]) - np.array(b["working"]["coords"]))) < 1e-14
    assert np.max(np.abs(np.array(a["working"]["lattice"]) - np.array(b["working"]["lattice"]))) < 1e-14
    assert a["symmetry"] == b["symmetry"]


# --- rounding and small distortions ----------------------------------------------------


def test_sub_tolerance_rounding_is_idealized_without_changing_symmetry():
    """Coordinates rounded to 6 decimals: symmetry is kept at 1e-5 and made exact."""

    structure = gan(third=0.333333)
    data = prepare_phonon_working_structure(structure).to_dict()
    working = data["working"]

    assert data["symmetry"]["incoming"]["number"] == 186
    assert 0 < data["idealization"]["max_site_shift_angstrom"] <= SYMPREC
    assert spglib_number(structure.lattice.matrix, structure.frac_coords, working["species"], 1e-10) != 186
    assert spglib_number(working["lattice"], working["coords"], working["species"], 1e-10) == 186


def test_rounding_beyond_tolerance_is_not_repaired():
    """Coordinates rounded to 4 decimals lose P6_3mc at 1e-5. M2b records the
    lower symmetry and does not loosen the tolerance to recover it."""

    data = prepare_phonon_working_structure(gan(third=0.3333)).to_dict()
    assert data["symmetry"]["incoming"] == {"international": "Cmc2_1", "number": 36, "operations": 4}
    assert data["symmetry"]["working"] == data["symmetry"]["incoming"]
    assert data["idealization"]["max_site_shift_angstrom"] <= SYMPREC


def test_noise_below_tolerance_is_projected_out():
    rng = np.random.default_rng(20261006)
    noisy = si()
    coords = noisy.frac_coords + rng.normal(scale=1e-7, size=(2, 3))
    noisy = Structure(noisy.lattice, ["Si", "Si"], coords)
    working = prepare_phonon_working_structure(noisy).to_dict()["working"]
    assert spglib_number(working["lattice"], working["coords"], working["species"], 1e-10) == 227


# --- site properties, species and magnetism --------------------------------------------


def with_property(structure, name, values):
    structure = structure.copy()
    structure.add_site_property(name, values)
    return structure


def test_zero_magmom_is_recorded_then_dropped():
    structure = with_property(gan(), "magmom", [0.0, -0.0, 0, 0.0])
    data = prepare_phonon_working_structure(structure).to_dict()
    plain = prepare_phonon_working_structure(gan())

    assert data["transformations"] == [DROP_ZERO_MAGMOM, SYMMETRY_IDEALIZATION]
    assert data["incoming"]["site_properties"] == {"magmom": [0.0, 0.0, 0.0, 0.0]}
    assert data["incoming"]["sha256"] != plain.incoming_sha256
    assert data["working"] == plain.to_dict()["working"]


@pytest.mark.parametrize(
    "values",
    [[[0, 0, 0]] * 2, [Magmom([0, 0, 0])] * 2, [np.float64(0.0)] * 2, [np.zeros(3)] * 2],
    ids=["vectors", "magmom-objects", "numpy-scalars", "numpy-vectors"],
)
def test_zero_magmom_representations_are_accepted(values):
    record = prepare_phonon_working_structure(with_property(si(), "magmom", values))
    assert record.transformations == (DROP_ZERO_MAGMOM, SYMMETRY_IDEALIZATION)


@pytest.mark.parametrize(
    "values",
    [[0.001, 0.0], [0.0, -0.6], [[0, 0, 0.2], [0, 0, 0]], [Magmom([0, 0, 1])] * 2, [float("nan"), 0.0],
     [float("inf"), 0.0], [None, 0.0], ["0", 0.0], [True, 0.0]],
    ids=["tiny", "negative", "vector", "magmom-object", "nan", "inf", "none", "string", "bool"],
)
def test_nonzero_or_unreadable_magmom_is_refused(values):
    refuse(with_property(si(), "magmom", values), "magnetic")


def test_magnetic_species_spin_is_refused_and_zero_spin_is_recorded():
    refuse(Structure(si().lattice, [Species("Si", 0, spin=1), "Si"], si().frac_coords), "magnetic")
    record = prepare_phonon_working_structure(
        Structure(si().lattice, [Species("Si", 0, spin=0), Species("Si", 0, spin=0)], si().frac_coords)
    )
    data = record.to_dict()
    assert data["incoming"]["species"][0] == {"element": "Si", "oxidation_state": 0.0, "spin": 0.0}
    assert data["transformations"] == [REDUCE_SPECIES_TO_ELEMENTS, SYMMETRY_IDEALIZATION]
    assert data["working"]["species"] == ["Si", "Si"]


def test_oxidation_states_are_kept_in_the_incoming_identity_only():
    decorated = nacl()
    decorated.add_oxidation_state_by_element({"Na": 1, "Cl": -1})
    record = prepare_phonon_working_structure(decorated).to_dict()
    plain = prepare_phonon_working_structure(nacl()).to_dict()

    assert [entry["oxidation_state"] for entry in record["incoming"]["species"]] == [1.0, -1.0]
    assert record["transformations"] == [REDUCE_SPECIES_TO_ELEMENTS, SYMMETRY_IDEALIZATION]
    assert record["incoming"]["sha256"] != plain["incoming"]["sha256"]
    assert record["working"] == plain["working"]


def test_zero_velocities_are_recorded_then_dropped():
    data = prepare_phonon_working_structure(with_property(si(), "velocities", [[0.0, 0.0, 0.0]] * 2)).to_dict()
    assert data["transformations"] == [DROP_ZERO_VELOCITIES, SYMMETRY_IDEALIZATION]
    assert data["incoming"]["site_properties"] == {"velocities": [[0.0, 0.0, 0.0]] * 2}


def test_transformations_are_recorded_in_a_fixed_order():
    structure = nacl()
    structure.add_oxidation_state_by_element({"Na": 1, "Cl": -1})
    structure.add_site_property("velocities", [[0, 0, 0]] * 2)
    structure.add_site_property("magmom", [0, 0])
    assert prepare_phonon_working_structure(structure).transformations == (
        DROP_ZERO_MAGMOM,
        DROP_ZERO_VELOCITIES,
        REDUCE_SPECIES_TO_ELEMENTS,
        SYMMETRY_IDEALIZATION,
    )


@pytest.mark.parametrize(
    "name, values, code",
    [
        ("selective_dynamics", [[True, True, True]] * 2, "selective_dynamics_unsupported"),
        ("selective_dynamics", [[False, True, True], [True] * 3], "selective_dynamics_unsupported"),
        ("velocities", [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]], "unsupported_property"),
        ("velocities", [[0.0, 0.0], [0.0, 0.0]], "malformed"),
        ("velocities", [0.0, 0.0], "malformed"),
        ("predictor_corrector", [[0.0, 0.0, 0.0]] * 2, "unsupported_property"),
        ("forces", [[0.0, 0.0, 0.0]] * 2, "unsupported_property"),
        ("charge", [0.0, 0.0], "unsupported_property"),
        ("label_extra", ["a", "b"], "unsupported_property"),
    ],
)
def test_unsupported_site_properties_are_refused(name, values, code):
    refuse(with_property(si(), name, values), code)


def test_property_present_on_only_some_sites_is_refused():
    structure = si()
    structure[0].properties["magmom"] = 0.0
    refuse(structure, "malformed")


@pytest.mark.parametrize(
    "structure, code",
    [
        (Structure(si().lattice, [DummySpecies("X"), "Si"], si().frac_coords), "unsupported"),
        (Structure(si().lattice, [{"Si": 0.5, "Ge": 0.5}, "Si"], si().frac_coords), "unsupported"),
        (Structure(si().lattice, [{"Si": 0.9}, "Si"], si().frac_coords), "unsupported"),
        (Structure(Lattice(-si().lattice.matrix), ["Si", "Si"], si().frac_coords), "malformed"),
        (Structure(si().lattice, ["Si", "Si"], [[0.0, 0.0, float("nan")], [0.25, 0.25, 0.25]]), "malformed"),
        (Structure(si().lattice, ["Si", "Si"], si().frac_coords, charge=1), "unsupported"),
    ],
    ids=["dummy", "disordered", "partial", "left-handed", "nan-coordinate", "charged"],
)
def test_malformed_or_unsupported_structures_are_refused(structure, code):
    refuse(structure, code)


def test_structure_level_properties_and_non_structures_are_refused():
    structure = si()
    structure.properties["source"] = "somewhere"
    refuse(structure, "unsupported_property")
    for value in (Molecule(["H"], [[0, 0, 0]]), si().as_dict(), None):
        with pytest.raises(PhononWorkingStructureError):
            prepare_phonon_working_structure(value)


def test_site_labels_are_not_identity():
    labelled = si()
    labelled[0].label = "Si_custom"
    assert prepare_phonon_working_structure(labelled).to_json() == prepare_phonon_working_structure(si()).to_json()


# --- the caller's structure is never modified ----------------------------------------------


def test_stage1_structure_is_not_modified():
    structure = gan(third=0.333333)
    structure.add_oxidation_state_by_element({"Ga": 3, "N": -3})
    structure.add_site_property("magmom", [0.0] * 4)
    structure.add_site_property("velocities", [[0.0, 0.0, 0.0]] * 4)
    properties = [site.properties for site in structure]
    before = snapshot(structure)

    record = prepare_phonon_working_structure(structure)
    record.working_structure()
    verify_phonon_working_structure(record, structure)

    assert snapshot(structure) == before
    assert all(site.properties is original for site, original in zip(structure, properties))


# --- record validation -----------------------------------------------------------------


@pytest.fixture
def record() -> dict:
    structure = gan(third=0.333333)
    structure.add_site_property("magmom", [0.0] * 4)
    return prepare_phonon_working_structure(structure).to_dict()


def test_forged_incoming_identity_is_rejected(record):
    record["incoming"]["sha256"] = record["working"]["sha256"]
    reject(rehash(record), "incoming.sha256")


def test_forged_working_identity_is_rejected(record):
    record["working"]["sha256"] = record["incoming"]["sha256"]
    reject(rehash(record))


def test_edited_working_coordinates_are_rejected(record):
    working = record["working"]
    coords = copy.deepcopy(working["coords"])
    coords[0][2] += 1e-7
    record["working"] = structure_record(working["lattice"], working["species"], coords, "fractional")
    reject(rehash(record), "idealization magnitudes")


@pytest.mark.parametrize(
    "transformations",
    [
        [SYMMETRY_IDEALIZATION],
        [SYMMETRY_IDEALIZATION, DROP_ZERO_MAGMOM],
        [DROP_ZERO_MAGMOM, REDUCE_SPECIES_TO_ELEMENTS, SYMMETRY_IDEALIZATION],
        [DROP_ZERO_MAGMOM, "standardize_conventional", SYMMETRY_IDEALIZATION],
        [DROP_ZERO_MAGMOM],
    ],
)
def test_altered_transformation_lists_are_rejected(record, transformations):
    record["transformations"] = transformations
    reject(rehash(record), "transformations")


@pytest.mark.parametrize(
    "path, value",
    [
        (("policy", "policy_version"), 2),
        (("policy", "symprec_angstrom"), 1e-3),
        (("policy", "angle_tolerance_degrees"), 5.0),
        (("policy", "method"), "spglib_standardize_cell"),
        (("policy", "status"), "approved"),
        (("schema",), "bmd_compute.phonon_displacement_plan"),
        (("schema_version",), 2),
        (("schema_version",), True),
        (("symmetry", "working", "number"), 36),
        (("symmetry", "incoming", "number"), 0),
        (("incoming", "species", 0, "spin"), 1.0),
        (("incoming", "species", 0, "oxidation_state"), 3),
        (("incoming", "site_properties", "magmom", 0), 0.5),
        (("incoming", "site_properties", "forces"), [[0.0, 0.0, 0.0]] * 4),
        (("idealization", "max_site_shift_angstrom"), 0.0),
        (("software", "spglib"), ""),
        (("record_sha256",), "0" * 64),
    ],
)
def test_structurally_invalid_records_are_rejected(record, path, value):
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path != ("record_sha256",):
        rehash(record)
    reject(record)


@pytest.mark.parametrize("where", [(), ("policy",), ("incoming",), ("software",), ("symmetry",), ("idealization",)])
def test_unknown_keys_are_rejected(record, where):
    target = record
    for key in where:
        target = target[key]
    target["extra"] = 1
    reject(rehash(record))


def test_record_kinds_cannot_be_confused(record):
    from backend.stage_task_set import StageTaskSet, StageTaskSetContractError

    reject({**record, "schema": "bmd_compute.stage_task_set"})
    with pytest.raises(StageTaskSetContractError):
        StageTaskSet.from_dict(record)


def test_records_are_immutable():
    record = prepare_phonon_working_structure(si())
    exported = record.to_dict()
    exported["working"]["coords"][0][0] = 0.5
    exported["transformations"].append("x")
    assert record.to_dict()["working"]["coords"][0][0] == 0.0
    assert record.transformations == (SYMMETRY_IDEALIZATION,)
    with pytest.raises(AttributeError):
        record._canonical = "{}"
    with pytest.raises(PhononWorkingStructureError, match="canonical form"):
        PhononWorkingStructure(json.dumps(json.loads(record.to_json()), indent=1))


def test_policy_values_cannot_drift():
    with pytest.raises(PhononWorkingStructureError):
        WorkingStructurePolicy(symprec_angstrom=1e-4)
    with pytest.raises(PhononWorkingStructureError):
        WorkingStructurePolicy(policy_version=2)
    with pytest.raises(PhononWorkingStructureError):
        WorkingStructurePolicy(angle_tolerance_degrees=-1)


# --- re-verification against the actual Stage-1 structure ------------------------------


def test_verification_accepts_the_record_of_this_structure():
    structure = gan(third=0.333333)
    record = prepare_phonon_working_structure(structure)
    assert verify_phonon_working_structure(record, gan(third=0.333333)) == record


def test_verification_rejects_a_record_of_another_structure():
    record = prepare_phonon_working_structure(gan())
    with pytest.raises(PhononWorkingStructureError) as excinfo:
        verify_phonon_working_structure(record, gan(z=0.378))
    assert excinfo.value.code == "incoming_mismatch"


def test_verification_rejects_a_consistently_forged_working_structure():
    structure = gan(third=0.333333)
    data = prepare_phonon_working_structure(structure).to_dict()
    working = data["working"]
    coords = copy.deepcopy(working["coords"])
    coords[2][2] += 1e-7  # a free coordinate, so symmetry and magnitudes stay plausible
    data["working"] = structure_record(working["lattice"], working["species"], coords, "fractional")
    data["idealization"] = {
        "max_site_shift_angstrom": data["idealization"]["max_site_shift_angstrom"],
        "max_lattice_change_angstrom": data["idealization"]["max_lattice_change_angstrom"],
    }
    from backend.phonons.working_structure import _idealization_magnitudes

    data["idealization"] = _idealization_magnitudes(data["incoming"], data["working"])
    forged = PhononWorkingStructure.from_dict(rehash(data))
    with pytest.raises(PhononWorkingStructureError) as excinfo:
        verify_phonon_working_structure(forged, structure)
    assert excinfo.value.code == "working_mismatch"


def test_verification_refuses_other_software_versions():
    data = prepare_phonon_working_structure(si()).to_dict()
    data["software"]["spglib"] = "0.0.1"
    forged = PhononWorkingStructure.from_dict(rehash(data))
    with pytest.raises(PhononWorkingStructureError) as excinfo:
        verify_phonon_working_structure(forged, si())
    assert excinfo.value.code == "software_mismatch"


def test_identity_is_independent_of_hash_seed_cwd_and_locale(tmp_path):
    script = (
        "from pymatgen.core import Lattice, Structure\n"
        "from backend.phonons.working_structure import prepare_phonon_working_structure\n"
        "t = 0.333333\n"
        "s = Structure(Lattice.hexagonal(3.19, 5.19), ['Ga','Ga','N','N'],\n"
        "    [[t,2*t,0],[2*t,t,0.5],[t,2*t,0.377],[2*t,t,0.877]])\n"
        "print(prepare_phonon_working_structure(s).to_json())\n"
    )
    outputs = set()
    for seed, locale, cwd in (("0", "C", tmp_path), ("4242", "C.UTF-8", REPO_ROOT)):
        env = dict(os.environ, PYTHONHASHSEED=seed, LC_ALL=locale, PYTHONPATH=str(REPO_ROOT))
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stderr
        outputs.add(completed.stdout)
    assert outputs == {prepare_phonon_working_structure(gan(third=0.333333)).to_json() + "\n"}


# --- M1 and M2a integration (Phonopy) --------------------------------------------------


ATTEMPT = "0b6f6c55-3f43-4a3b-9b0e-7d1c2f6a9e10"
SUPERCELLS = {
    "Si": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    "NaCl": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    "GaN": [[3, 0, 0], [0, 3, 0], [0, 0, 2]],
    "MoS2": [[3, 0, 0], [0, 3, 0], [0, 0, 1]],
}


@pytest.fixture(scope="module")
def phonopy_available():
    pytest.importorskip("phonopy")


@pytest.mark.parametrize("name, factory, _number, _symbol", REFERENCE, ids=[r[0] for r in REFERENCE])
def test_m1_plans_from_working_structures_are_deterministic(phonopy_available, name, factory, _number, _symbol):
    from backend.phonons import build_displacement_plan

    first = prepare_phonon_working_structure(factory())
    second = prepare_phonon_working_structure(factory())
    plan = build_displacement_plan(first.working_structure(), supercell_matrix=SUPERCELLS[name])
    again = build_displacement_plan(second.working_structure(), supercell_matrix=SUPERCELLS[name])

    assert plan.to_json() == again.to_json()
    assert plan.to_dict()["working_structure"] == first.to_dict()["working"]
    assert plan.to_dict()["working_structure"]["sha256"] == first.working_sha256 != first.incoming_sha256


def test_displacement_counts_follow_the_recorded_symmetry(phonopy_available):
    from backend.phonons import build_displacement_plan

    matrix = SUPERCELLS["GaN"]
    counts = {
        label: build_displacement_plan(
            prepare_phonon_working_structure(gan(third=third)).working_structure(), supercell_matrix=matrix
        ).task_count
        for label, third in (("exact", 1 / 3), ("rounded6", 0.333333), ("rounded4", 0.3333))
    }
    assert counts == {"exact": 4, "rounded6": 4, "rounded4": 8}


def _chain(stage1):
    from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
    from backend.phonons import build_displacement_plan
    from backend.phonons.task_set import build_phonon_force_task_set

    workflow = WorkflowSpec([StageSpec(StageType.RELAX, Theory.PBE), StageSpec(StageType.PHONON_FORCES, Theory.PBE)])
    working = prepare_phonon_working_structure(stage1)
    plan = build_displacement_plan(working.working_structure(), supercell_matrix=SUPERCELLS["GaN"])
    task_set = build_phonon_force_task_set(
        plan,
        working_structure=working,
        workflow=workflow,
        stage_index=2,
        submission_attempt_id=ATTEMPT,
        upstream_stage_index=1,
    )
    return workflow, working, plan, task_set


def _stage1():
    structure = gan(third=0.333333)
    structure.add_site_property("magmom", [0.0] * 4)
    return structure


def test_task_set_upstream_is_the_stage1_identity_and_the_chain_verifies(phonopy_available):
    from backend.phonons.task_set import verify_phonon_force_task_set_chain

    workflow, working, plan, task_set = _chain(_stage1())

    assert task_set.upstream_structure == {"stage_index": 1, "sha256": working.incoming_sha256}
    assert task_set.upstream_structure["sha256"] != working.working_sha256
    assert plan.to_dict()["working_structure"]["sha256"] == working.working_sha256
    verify_phonon_force_task_set_chain(
        task_set, workflow=workflow, submission_attempt_id=ATTEMPT,
        stage1_structure=_stage1(), working_structure=working, plan=plan,
    )


def test_chain_rejects_a_different_stage1_structure_with_identical_methodology(phonopy_available):
    from backend.phonons.task_set import verify_phonon_force_task_set_chain

    workflow, working, plan, task_set = _chain(_stage1())
    other = gan(third=0.333333, z=0.378)
    other.add_site_property("magmom", [0.0] * 4)
    with pytest.raises(PhononWorkingStructureError, match="does not describe this Stage-1 structure"):
        verify_phonon_force_task_set_chain(
            task_set, workflow=workflow, submission_attempt_id=ATTEMPT,
            stage1_structure=other, working_structure=working, plan=plan,
        )


def test_chain_rejects_mixed_links(phonopy_available):
    from backend.phonons.task_set import verify_phonon_force_task_set_chain
    from backend.stage_task_set import StageTaskSet, StageTaskSetContractError, task_set_sha256

    workflow, working, plan, task_set = _chain(_stage1())
    _, other_working, other_plan, other_task_set = _chain(gan(third=0.333333, z=0.378))

    def run(**overrides):
        arguments = dict(
            workflow=workflow, submission_attempt_id=ATTEMPT, stage1_structure=_stage1(),
            working_structure=working, plan=plan,
        )
        arguments.update(overrides)
        verify_phonon_force_task_set_chain(arguments.pop("task_set", task_set), **arguments)

    with pytest.raises(StageTaskSetContractError, match="upstream identity"):
        run(task_set=other_task_set)
    with pytest.raises(StageTaskSetContractError, match="not built from this working structure"):
        run(plan=other_plan)

    data = task_set.to_dict()
    data["upstream_structure"]["sha256"] = working.working_sha256
    data["task_set_sha256"] = task_set_sha256(data)
    with pytest.raises(StageTaskSetContractError, match="upstream identity"):
        run(task_set=StageTaskSet.from_dict(data))


def test_task_set_cannot_be_built_from_a_plan_of_another_working_structure(phonopy_available):
    from backend.phonons import build_displacement_plan
    from backend.phonons.task_set import build_phonon_force_task_set
    from backend.stage_task_set import StageTaskSetContractError

    workflow, working, _plan, _task_set = _chain(_stage1())
    raw_plan = build_displacement_plan(_stage1(), supercell_matrix=SUPERCELLS["GaN"])
    assert raw_plan.to_dict()["working_structure"]["sha256"] != working.working_sha256
    with pytest.raises(StageTaskSetContractError, match="not built from this working structure"):
        build_phonon_force_task_set(
            raw_plan, working_structure=working, workflow=workflow,
            stage_index=2, submission_attempt_id=ATTEMPT, upstream_stage_index=1,
        )


def test_identity_is_the_exact_representation_not_crystal_equivalence():
    wrapped = Structure(si().lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    unwrapped = Structure(si().lattice, ["Si", "Si"], [[1.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    a, b = prepare_phonon_working_structure(wrapped), prepare_phonon_working_structure(unwrapped)
    assert a.incoming_sha256 != b.incoming_sha256
    assert a.working_sha256 != b.working_sha256


def test_web_application_does_not_load_the_boundary(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-c", "import sys, main; print('backend.phonons.working_structure' in sys.modules)"],
        cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
