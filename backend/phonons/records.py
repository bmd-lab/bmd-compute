"""Canonical serialization and validation for phonon displacement plans.

``bmd_compute.phonon_displacement_plan`` v1 is the deterministic scientific
object that later phonon execution will consume: the Phonopy displacement
dataset for one working structure and one explicit supercell, with one task
instance per displacement. It is not a workflow, a schedule or a submission
record.

Canonicalization rule (``canonical_json``):

* the value must be built only from ``dict`` (``str`` keys), ``list``/``tuple``,
  ``str``, ``int``, ``float``, ``bool`` and ``None``; anything else, including
  NumPy scalars and arrays, is rejected rather than coerced;
* object keys are sorted; no insignificant whitespace; ASCII-only output;
* floats are written with Python's shortest round-trip ``repr``, so the exact
  IEEE-754 binary64 value is preserved. Nothing is rounded. ``-0.0`` is written
  as ``0.0`` (the two compare equal and carry no scientific meaning). NaN and
  infinities are rejected;
* ``bool`` is never accepted where an ``int`` is expected, and an ``int`` is
  never accepted where a ``float`` is expected (validators enforce this).

The SHA-256 of a record is taken over the UTF-8 bytes of ``canonical_json``.
Exact float preservation means two hosts whose linear algebra differs in the
last bit produce different hashes; a future cross-host check must compare
recomputed plans structurally, not only by hash.

This module imports only the standard library so that it can be used where
Phonopy is not installed. A record that passes validation is internally
consistent; it is still not scientific authority. In particular, record
validation cannot prove the dataset is *complete* (a consistently truncated or
re-chosen displacement set is self-consistent), because that needs Phonopy's
symmetry analysis. Authority comes from rebuilding the plan from the working
structure and policy (``backend.phonons.plan.verify_displacement_plan``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from backend.phonons.policy import PhononPolicy, PhononPolicyError


DISPLACEMENT_PLAN_SCHEMA = "bmd_compute.phonon_displacement_plan"
DISPLACEMENT_PLAN_SCHEMA_VERSION = 1

# Task instances of the future common phonon force-calculation stage. They are
# not workflow stages: they carry no stage type, theory or modifiers.
TASK_KIND = "phonon_displacement"
TASK_ID_PATTERN = re.compile(r"^disp_(\d{3,})$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ELEMENT_SYMBOL_PATTERN = re.compile(r"^[A-Z][a-z]{0,2}$")

COORDS_FRACTIONAL = "fractional"
COORDS_CARTESIAN = "cartesian"

# Relative tolerance for checks between independently derived quantities
# (for example the supercell lattice against M @ L). Exact identity is
# enforced by hashes; these checks only catch inconsistent records.
_DERIVED_RTOL = 1e-9
_DERIVED_ATOL = 1e-9

PLAN_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "policy",
        "software",
        "phonopy_arguments",
        "working_structure",
        "symmetry",
        "supercell_matrix",
        "primitive_matrix",
        "primitive_natom",
        "supercell",
        "dataset",
        "task_kind",
        "tasks",
        "plan_sha256",
    }
)
STRUCTURE_KEYS = frozenset({"lattice", "species", "coords", "coords_type", "sha256"})
TASK_KEYS = frozenset(
    {"task_id", "dataset_index", "atom_index", "displacement", "structure_sha256"}
)


class PhononPlanContractError(ValueError):
    """Raised when a displacement-plan record does not satisfy its contract."""


# --- canonical form ------------------------------------------------------------------


def canonical_value(value: Any) -> Any:
    """Return ``value`` as plain JSON data in canonical form, or raise."""

    if value is None or type(value) is bool or type(value) is str:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhononPlanContractError(f"non-finite float {value!r} cannot be canonicalized")
        return 0.0 if value == 0.0 else value
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PhononPlanContractError(f"object key {key!r} is not a string")
            result[key] = canonical_value(item)
        return result
    if type(value) in (list, tuple):
        return [canonical_value(item) for item in value]
    raise PhononPlanContractError(
        f"{type(value).__module__}.{type(value).__qualname__} is not canonical plan data"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# --- structures ------------------------------------------------------------------------


def structure_record(lattice, species, coords, coords_type: str) -> dict[str, Any]:
    """Canonical structure record with its own SHA-256.

    ``lattice`` rows are lattice vectors in Angstrom. ``coords`` are fractional
    or Cartesian (Angstrom) as named by ``coords_type``. Values are stored
    exactly as given; nothing is wrapped, rounded or standardized here.
    """

    body = {
        "lattice": [[_float(x, "lattice") for x in row] for row in lattice],
        "species": [str(symbol) for symbol in species],
        "coords": [[_float(x, "coords") for x in row] for row in coords],
        "coords_type": coords_type,
    }
    body["sha256"] = canonical_sha256(body)
    return body


def displaced_structure_sha256(supercell: Mapping[str, Any], atom_index: int, displacement) -> str:
    """Identity of a displaced supercell, derived only from plan data.

    The displaced cell is the supercell with ``displacement`` (Cartesian,
    Angstrom) added to atom ``atom_index``. This is exactly how Phonopy 4.8.1
    builds ``supercells_with_displacements`` (Cartesian addition on the same
    lattice), and it needs only IEEE addition, so any reader can recompute it
    without Phonopy.
    """

    if supercell.get("coords_type") != COORDS_CARTESIAN:
        raise PhononPlanContractError("supercell coordinates must be Cartesian")
    coords = [list(row) for row in supercell["coords"]]
    coords[atom_index] = [a + b for a, b in zip(coords[atom_index], displacement)]
    return structure_record(
        supercell["lattice"], supercell["species"], coords, COORDS_CARTESIAN
    )["sha256"]


def task_id_for_index(dataset_index: int) -> str:
    """Deterministic task-instance identifier for a 0-based dataset index."""

    return f"disp_{dataset_index + 1:03d}"


# --- plan body and hash ----------------------------------------------------------------


def plan_sha256(record: Mapping[str, Any]) -> str:
    """SHA-256 over every field of the plan except ``plan_sha256`` itself."""

    body = {key: value for key, value in record.items() if key != "plan_sha256"}
    return canonical_sha256(body)


# --- validation ------------------------------------------------------------------------


def validate_displacement_plan_record(payload: Any) -> None:
    """Raise PhononPlanContractError unless ``payload`` is a consistent v1 plan.

    Malformed or inconsistent records are rejected, never repaired.
    """

    record = _mapping(payload, "plan")
    _exact_keys(record, PLAN_KEYS, "plan")
    if record["schema"] != DISPLACEMENT_PLAN_SCHEMA:
        raise PhononPlanContractError(f"schema must be {DISPLACEMENT_PLAN_SCHEMA!r}")
    if type(record["schema_version"]) is not int or record["schema_version"] != DISPLACEMENT_PLAN_SCHEMA_VERSION:
        raise PhononPlanContractError(
            f"schema_version must be {DISPLACEMENT_PLAN_SCHEMA_VERSION}, got {record['schema_version']!r}"
        )
    try:
        canonical_value(record)
    except PhononPlanContractError as exc:
        raise PhononPlanContractError(f"plan is not canonical data: {exc}") from None

    policy = _policy(record["policy"])
    _software(record["software"])
    _phonopy_arguments(record["phonopy_arguments"], policy, record["supercell_matrix"])

    working = _structure(record["working_structure"], "working_structure", COORDS_FRACTIONAL)
    natom_unit = len(working["species"])

    symmetry = _mapping(record["symmetry"], "symmetry")
    _exact_keys(symmetry, {"international", "number", "symprec"}, "symmetry")
    _text(symmetry["international"], "symmetry.international")
    number = _int(symmetry["number"], "symmetry.number")
    if not 1 <= number <= 230:
        raise PhononPlanContractError("symmetry.number must be a space-group number 1-230")
    if _float(symmetry["symprec"], "symmetry.symprec") != policy.symprec:
        raise PhononPlanContractError("symmetry.symprec does not match the policy symprec")

    supercell_matrix = _int_matrix(record["supercell_matrix"], "supercell_matrix")
    multiplicity = _det3(supercell_matrix)
    if multiplicity <= 0:
        raise PhononPlanContractError("supercell_matrix must have a positive determinant")

    primitive_matrix = _float_matrix(record["primitive_matrix"], "primitive_matrix")
    primitive_det = _det3(primitive_matrix)
    primitive_natom = _int(record["primitive_natom"], "primitive_natom")
    if primitive_natom < 1 or not math.isclose(
        natom_unit * primitive_det, primitive_natom, rel_tol=_DERIVED_RTOL, abs_tol=_DERIVED_ATOL
    ):
        raise PhononPlanContractError(
            "primitive_natom is inconsistent with the primitive matrix and working structure"
        )

    supercell = _structure(record["supercell"], "supercell", COORDS_CARTESIAN)
    natom_super = len(supercell["species"])
    if natom_super != natom_unit * multiplicity:
        raise PhononPlanContractError(
            "supercell atom count does not equal working-structure atoms times det(supercell_matrix)"
        )
    expected_lattice = _matmul(supercell_matrix, working["lattice"])
    if not _matrices_close(expected_lattice, supercell["lattice"]):
        raise PhononPlanContractError("supercell lattice is not supercell_matrix @ working lattice")
    if sorted(supercell["species"]) != sorted(working["species"] * multiplicity):
        raise PhononPlanContractError("supercell species do not replicate the working structure")

    first_atoms = _dataset(record["dataset"], natom_super, policy.displacement_distance_angstrom)

    if record["task_kind"] != TASK_KIND:
        raise PhononPlanContractError(f"task_kind must be {TASK_KIND!r}")
    tasks = record["tasks"]
    if type(tasks) is not list or len(tasks) != len(first_atoms):
        raise PhononPlanContractError("tasks must list exactly one task per dataset displacement")
    seen_structures = set()
    for index, (task, entry) in enumerate(zip(tasks, first_atoms)):
        label = f"tasks[{index}]"
        task = _mapping(task, label)
        _exact_keys(task, TASK_KEYS, label)
        if task["task_id"] != task_id_for_index(index):
            raise PhononPlanContractError(
                f"{label}.task_id must be {task_id_for_index(index)!r}, got {task['task_id']!r}"
            )
        if _int(task["dataset_index"], f"{label}.dataset_index") != index:
            raise PhononPlanContractError(f"{label}.dataset_index must be {index}")
        if _int(task["atom_index"], f"{label}.atom_index") != entry["number"]:
            raise PhononPlanContractError(f"{label}.atom_index does not match dataset.first_atoms[{index}]")
        displacement = _float_vector(task["displacement"], f"{label}.displacement")
        if displacement != entry["displacement"]:
            raise PhononPlanContractError(f"{label}.displacement does not match dataset.first_atoms[{index}]")
        structure_sha = _sha(task["structure_sha256"], f"{label}.structure_sha256")
        if structure_sha != displaced_structure_sha256(supercell, entry["number"], displacement):
            raise PhononPlanContractError(f"{label}.structure_sha256 does not identify its displaced supercell")
        if structure_sha in seen_structures:
            raise PhononPlanContractError(f"{label} duplicates another task's displaced structure")
        seen_structures.add(structure_sha)

    if _sha(record["plan_sha256"], "plan_sha256") != plan_sha256(record):
        raise PhononPlanContractError("plan_sha256 does not match the canonical plan content")


def _policy(value: Any) -> PhononPolicy:
    mapping = _mapping(value, "policy")
    try:
        policy = PhononPolicy.from_dict(mapping)
    except PhononPolicyError as exc:
        raise PhononPlanContractError(f"policy: {exc}") from None
    if canonical_json(policy.to_dict()) != canonical_json(dict(mapping)):
        raise PhononPlanContractError("policy block is not the canonical form of its policy")
    return policy


def _software(value: Any) -> None:
    software = _mapping(value, "software")
    _exact_keys(software, {"phonopy", "spglib"}, "software")
    _text(software["phonopy"], "software.phonopy")
    _text(software["spglib"], "software.spglib")


def _phonopy_arguments(value: Any, policy: PhononPolicy, supercell_matrix: Any) -> None:
    arguments = _mapping(value, "phonopy_arguments")
    try:
        expected = policy.phonopy_arguments(supercell_matrix)
    except PhononPolicyError as exc:
        raise PhononPlanContractError(f"phonopy_arguments: {exc}") from None
    if canonical_json(dict(arguments)) != canonical_json(expected):
        raise PhononPlanContractError(
            "phonopy_arguments do not match the explicit arguments implied by the policy"
        )


def _structure(value: Any, label: str, coords_type: str) -> dict[str, Any]:
    structure = _mapping(value, label)
    _exact_keys(structure, STRUCTURE_KEYS, label)
    if structure["coords_type"] != coords_type:
        raise PhononPlanContractError(f"{label}.coords_type must be {coords_type!r}")
    lattice = _float_matrix(structure["lattice"], f"{label}.lattice")
    if abs(_det3(lattice)) <= _DERIVED_ATOL:
        raise PhononPlanContractError(f"{label}.lattice is singular")
    species = structure["species"]
    if type(species) is not list or not species:
        raise PhononPlanContractError(f"{label}.species must be a non-empty list")
    for symbol in species:
        if type(symbol) is not str or not ELEMENT_SYMBOL_PATTERN.match(symbol):
            raise PhononPlanContractError(f"{label}.species contains invalid symbol {symbol!r}")
    coords = structure["coords"]
    if type(coords) is not list or len(coords) != len(species):
        raise PhononPlanContractError(f"{label}.coords must have one row per species entry")
    rows = [_float_vector(row, f"{label}.coords") for row in coords]
    recomputed = structure_record(lattice, species, rows, coords_type)["sha256"]
    if _sha(structure["sha256"], f"{label}.sha256") != recomputed:
        raise PhononPlanContractError(f"{label}.sha256 does not match its content")
    return {"lattice": lattice, "species": list(species), "coords": rows, "coords_type": coords_type}


def _dataset(value: Any, natom_super: int, distance: float) -> list[dict[str, Any]]:
    dataset = _mapping(value, "dataset")
    _exact_keys(dataset, {"natom", "first_atoms"}, "dataset")
    if _int(dataset["natom"], "dataset.natom") != natom_super:
        raise PhononPlanContractError("dataset.natom does not match the supercell atom count")
    first_atoms = dataset["first_atoms"]
    if type(first_atoms) is not list or not first_atoms:
        raise PhononPlanContractError("dataset.first_atoms must be a non-empty list")
    entries = []
    for index, entry in enumerate(first_atoms):
        label = f"dataset.first_atoms[{index}]"
        entry = _mapping(entry, label)
        _exact_keys(entry, {"number", "displacement"}, label)
        number = _int(entry["number"], f"{label}.number")
        if not 0 <= number < natom_super:
            raise PhononPlanContractError(f"{label}.number is outside the supercell")
        displacement = _float_vector(entry["displacement"], f"{label}.displacement")
        norm = math.sqrt(sum(x * x for x in displacement))
        if not math.isclose(norm, distance, rel_tol=1e-6, abs_tol=0.0):
            raise PhononPlanContractError(
                f"{label}.displacement has length {norm!r} A, policy distance is {distance!r} A"
            )
        entries.append({"number": number, "displacement": displacement})
    return entries


# --- small typed accessors ---------------------------------------------------------------


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhononPlanContractError(f"{label} must be an object")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected, label: str) -> None:
    keys = set(mapping)
    expected = set(expected)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(str(key) for key in keys - expected)
        raise PhononPlanContractError(f"{label} keys: missing {missing}, unexpected {extra}")


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise PhononPlanContractError(f"{label} must be a non-empty string")
    return value


def _int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise PhononPlanContractError(f"{label} must be an integer, got {value!r}")
    return value


def _float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PhononPlanContractError(f"{label} values must be finite floats, got {value!r}")
    return 0.0 if value == 0.0 else value


def _float_vector(value: Any, label: str) -> list[float]:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise PhononPlanContractError(f"{label} must have three components")
    return [_float(x, label) for x in value]


def _float_matrix(value: Any, label: str) -> list[list[float]]:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise PhononPlanContractError(f"{label} must be a 3x3 matrix")
    return [_float_vector(row, label) for row in value]


def _int_matrix(value: Any, label: str) -> list[list[int]]:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise PhononPlanContractError(f"{label} must be a 3x3 integer matrix")
    rows = []
    for row in value:
        if type(row) not in (list, tuple) or len(row) != 3:
            raise PhononPlanContractError(f"{label} must be a 3x3 integer matrix")
        rows.append([_int(x, label) for x in row])
    return rows


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or not SHA256_PATTERN.match(value):
        raise PhononPlanContractError(f"{label} must be a lowercase hex SHA-256")
    return value


def _det3(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _matmul(a, b) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matrices_close(a, b) -> bool:
    return all(
        math.isclose(x, y, rel_tol=_DERIVED_RTOL, abs_tol=_DERIVED_ATOL)
        for row_a, row_b in zip(a, b)
        for x, y in zip(row_a, row_b)
    )


__all__ = [
    "COORDS_CARTESIAN",
    "COORDS_FRACTIONAL",
    "DISPLACEMENT_PLAN_SCHEMA",
    "DISPLACEMENT_PLAN_SCHEMA_VERSION",
    "PhononPlanContractError",
    "TASK_KIND",
    "canonical_json",
    "canonical_sha256",
    "canonical_value",
    "displaced_structure_sha256",
    "plan_sha256",
    "structure_record",
    "task_id_for_index",
    "validate_displacement_plan_record",
]
