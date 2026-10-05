"""Deterministic Phonopy displacement planning.

``build_displacement_plan(structure, policy, supercell_matrix=...)`` turns one
working structure, the provisional planning policy and an explicit supercell
matrix into a ``DisplacementPlan``: Phonopy's displacement dataset plus one
task instance per displacement, canonically serialized and hashed.

Each task is an instance of a single future scientific stage, the PBE phonon
force-calculation stage, whose INCAR/KPOINTS methodology will be common to all
of its tasks. Tasks are deliberately not ``StageSpec`` objects, carry no stage
type, theory or modifiers, and say nothing about scheduling: the task order is
Phonopy's dataset order, which defines identity and the order forces must be
supplied in, not an execution order. Sequential, concurrent or array execution
are all compatible with the same plan.

The working structure is not standardized, wrapped or symmetrized.
Standardization of a user's structure is not approved methodology yet; a caller
that needs it must do it before planning. The plan's structure identity is the
lattice, element symbols and fractional coordinates as given. Oxidation-state
decorations and other site properties (for example ``selective_dynamics``) are
not part of it, and magnetic structures (a nonzero ``magmom`` site property or
``Species.spin``) are refused.
Under the eventual R2 workflow the working structure is the relaxed structure
produced by the prerequisite relaxation, so the plan can only be built after
that relaxation; nothing here assumes it is known earlier.

Phonopy and pymatgen are imported lazily, inside the functions that need them,
so ``import backend.phonons`` works where Phonopy is not installed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from backend.phonons.policy import (
    PHONOPY_LANG,
    PROVISIONAL_PHONON_POLICY,
    PhononPolicy,
    PhononPolicyError,
)
from backend.phonons.records import (
    COORDS_CARTESIAN,
    COORDS_FRACTIONAL,
    DISPLACEMENT_PLAN_SCHEMA,
    DISPLACEMENT_PLAN_SCHEMA_VERSION,
    PhononPlanContractError,
    TASK_KIND,
    canonical_json,
    displaced_structure_sha256,
    plan_sha256,
    structure_record,
    task_id_for_index,
    validate_displacement_plan_record,
)


# Phonopy's own displaced cells must agree with the plan's Cartesian
# reconstruction to within this (Angstrom). Phonopy stores fractional
# coordinates, so its round trip differs from exact addition only by rounding.
_PHONOPY_CROSSCHECK_ATOL = 1e-10


class PhonopyUnavailableError(ImportError):
    """Raised when displacement planning is requested without Phonopy installed."""


@dataclass(frozen=True)
class DisplacementTask:
    """One displacement task instance of the future phonon force stage."""

    task_id: str
    dataset_index: int
    atom_index: int
    displacement: tuple[float, float, float]
    structure_sha256: str


@dataclass(frozen=True)
class DisplacementPlan:
    """Immutable, validated ``bmd_compute.phonon_displacement_plan`` v1 record.

    The plan is held as its canonical JSON text, so no caller can mutate it;
    every accessor returns fresh objects.
    """

    _canonical: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self._canonical) is not str:
            raise PhononPlanContractError("DisplacementPlan holds canonical JSON text")
        data = json.loads(self._canonical)
        validate_displacement_plan_record(data)
        if canonical_json(data) != self._canonical:
            raise PhononPlanContractError("DisplacementPlan text is not in canonical form")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DisplacementPlan":
        """Validate a record and wrap it. Raises PhononPlanContractError."""

        if not isinstance(data, Mapping):
            raise PhononPlanContractError("plan must be an object")
        validate_displacement_plan_record(data)
        return cls(canonical_json(dict(data)))

    @classmethod
    def from_json(cls, text: str) -> "DisplacementPlan":
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise PhononPlanContractError(f"plan is not valid JSON: {exc}") from None
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def to_json(self) -> str:
        """Canonical JSON text; its SHA-256 body excludes ``plan_sha256``."""

        return self._canonical

    @property
    def plan_sha256(self) -> str:
        return self.to_dict()["plan_sha256"]

    @property
    def policy(self) -> PhononPolicy:
        return PhononPolicy.from_dict(self.to_dict()["policy"])

    @property
    def task_kind(self) -> str:
        return self.to_dict()["task_kind"]

    @property
    def tasks(self) -> tuple[DisplacementTask, ...]:
        return tuple(
            DisplacementTask(
                task_id=task["task_id"],
                dataset_index=task["dataset_index"],
                atom_index=task["atom_index"],
                displacement=tuple(task["displacement"]),
                structure_sha256=task["structure_sha256"],
            )
            for task in self.to_dict()["tasks"]
        )

    @property
    def task_count(self) -> int:
        return len(self.to_dict()["tasks"])

    @property
    def supercell_natom(self) -> int:
        return len(self.to_dict()["supercell"]["species"])

    @property
    def space_group(self) -> tuple[str, int]:
        symmetry = self.to_dict()["symmetry"]
        return symmetry["international"], symmetry["number"]


def build_displacement_plan(
    structure: Any,
    policy: PhononPolicy = PROVISIONAL_PHONON_POLICY,
    *,
    supercell_matrix: Any,
) -> DisplacementPlan:
    """Build the deterministic displacement plan for ``structure``.

    ``structure`` is a pymatgen ``Structure``/``IStructure`` and is not
    modified. ``supercell_matrix`` must be an explicit 3x3 integer matrix;
    no automatic supercell selection is applied.
    """

    if not isinstance(policy, PhononPolicy):
        raise TypeError("policy must be a PhononPolicy")
    try:
        arguments = policy.phonopy_arguments(supercell_matrix)
    except PhononPolicyError as exc:
        raise ValueError(str(exc)) from None
    working = _working_structure_record(structure)

    phonopy, PhonopyAtoms = _import_phonopy()
    unitcell = PhonopyAtoms(
        symbols=list(working["species"]),
        cell=[list(row) for row in working["lattice"]],
        scaled_positions=[list(row) for row in working["coords"]],
    )
    # Phonopy receives copies, so the recorded arguments are exactly those passed.
    phonon = phonopy.Phonopy(unitcell, **copy.deepcopy(arguments["construct"]))
    if phonon.lang != PHONOPY_LANG:
        raise RuntimeError(
            f"Phonopy resolved backend {phonon.lang!r} instead of the requested {PHONOPY_LANG!r}"
        )
    phonon.generate_displacements(**copy.deepcopy(arguments["displacements"]))

    supercell = structure_record(
        phonon.supercell.cell.tolist(),
        list(phonon.supercell.symbols),
        phonon.supercell.positions.tolist(),
        COORDS_CARTESIAN,
    )
    dataset = _dataset_record(phonon.dataset)
    tasks = [
        {
            "task_id": task_id_for_index(index),
            "dataset_index": index,
            "atom_index": entry["number"],
            "displacement": entry["displacement"],
            "structure_sha256": displaced_structure_sha256(
                supercell, entry["number"], entry["displacement"]
            ),
        }
        for index, entry in enumerate(dataset["first_atoms"])
    ]
    _crosscheck_phonopy_displaced_cells(phonon, supercell, dataset)

    symmetry_dataset = phonon.symmetry.dataset
    record = {
        "schema": DISPLACEMENT_PLAN_SCHEMA,
        "schema_version": DISPLACEMENT_PLAN_SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "software": _software_versions(),
        "phonopy_arguments": arguments,
        "working_structure": working,
        "symmetry": {
            "international": str(symmetry_dataset.international),
            "number": int(symmetry_dataset.number),
            "symprec": policy.symprec,
        },
        "supercell_matrix": arguments["construct"]["supercell_matrix"],
        "primitive_matrix": [[float(x) for x in row] for row in phonon.primitive_matrix.tolist()],
        "primitive_natom": len(phonon.primitive),
        "supercell": supercell,
        "dataset": dataset,
        "task_kind": TASK_KIND,
        "tasks": tasks,
    }
    record["plan_sha256"] = plan_sha256(record)
    return DisplacementPlan.from_dict(record)


def verify_displacement_plan(
    plan: DisplacementPlan | Mapping[str, Any],
    structure: Any,
    policy: PhononPolicy = PROVISIONAL_PHONON_POLICY,
    *,
    supercell_matrix: Any,
) -> DisplacementPlan:
    """Rebuild the plan and require it to equal ``plan`` exactly.

    A plan record is never scientific authority on its own: this is how a
    supplied record is accepted. Raises PhononPlanContractError on mismatch.
    """

    supplied = plan if isinstance(plan, DisplacementPlan) else DisplacementPlan.from_dict(plan)
    rebuilt = build_displacement_plan(structure, policy, supercell_matrix=supercell_matrix)
    if rebuilt.to_json() != supplied.to_json():
        raise PhononPlanContractError(
            "supplied plan does not match the plan rebuilt from the structure and policy "
            f"({supplied.plan_sha256} != {rebuilt.plan_sha256})"
        )
    return rebuilt


def _working_structure_record(structure: Any) -> dict[str, Any]:
    from pymatgen.core import IStructure
    from pymatgen.core.periodic_table import DummySpecies, Element, Species

    if not isinstance(structure, IStructure):
        raise TypeError("structure must be a pymatgen Structure")
    if not structure.is_ordered:
        raise ValueError("phonon displacement planning requires an ordered structure")
    species = []
    for site in structure:
        specie = site.specie
        if isinstance(specie, DummySpecies) or not isinstance(specie, (Element, Species)):
            raise ValueError(f"site species {specie!r} is not a chemical element")
        # Checked before the species is reduced to its element symbol, which
        # would otherwise discard the spin silently.
        if isinstance(specie, Species) and _nonzero_spin(getattr(specie, "spin", None)):
            raise ValueError(
                f"site species {specie!r} carries a spin; magnetic phonon planning is not supported"
            )
        # Oxidation-state decorations are not part of planning identity.
        species.append(specie.symbol if isinstance(specie, Element) else specie.element.symbol)
    magmoms = structure.site_properties.get("magmom")
    if magmoms is not None and any(_nonzero_moment(value) for value in magmoms):
        raise ValueError(
            "magnetic moments are present; magnetic phonon planning is not supported"
        )
    return structure_record(
        [[float(x) for x in row] for row in structure.lattice.matrix.tolist()],
        species,
        [[float(x) for x in row] for row in structure.frac_coords.tolist()],
        COORDS_FRACTIONAL,
    )


def _nonzero_spin(spin: Any) -> bool:
    """True unless ``spin`` is absent or exactly zero; unreadable values fail closed."""

    if spin is None:
        return False
    try:
        return float(spin) != 0.0
    except (TypeError, ValueError):
        return True


def _nonzero_moment(value: Any) -> bool:
    try:
        components = list(value)
    except TypeError:
        components = [value]
    return any(float(component) != 0.0 for component in components)


def _dataset_record(dataset: Any) -> dict[str, Any]:
    if not isinstance(dataset, Mapping) or set(dataset) != {"natom", "first_atoms"}:
        keys = sorted(dataset) if isinstance(dataset, Mapping) else dataset
        raise RuntimeError(f"unexpected Phonopy displacement dataset layout: {keys}")
    first_atoms = []
    for entry in dataset["first_atoms"]:
        if set(entry) != {"number", "displacement"}:
            raise RuntimeError(f"unexpected Phonopy displacement entry keys: {sorted(entry)}")
        first_atoms.append(
            {
                "number": int(entry["number"]),
                "displacement": [float(x) for x in list(entry["displacement"])],
            }
        )
    return {"natom": int(dataset["natom"]), "first_atoms": first_atoms}


def _crosscheck_phonopy_displaced_cells(phonon, supercell: Mapping[str, Any], dataset) -> None:
    import numpy as np

    cells = phonon.supercells_with_displacements
    if len(cells) != len(dataset["first_atoms"]):
        raise RuntimeError("Phonopy returned a different number of displaced supercells")
    base = np.array(supercell["coords"], dtype=float)
    lattice = np.array(supercell["lattice"], dtype=float)
    for cell, entry in zip(cells, dataset["first_atoms"]):
        expected = base.copy()
        expected[entry["number"]] += np.array(entry["displacement"], dtype=float)
        if (
            list(cell.symbols) != list(supercell["species"])
            or not np.array_equal(np.asarray(cell.cell), lattice)
            or not np.allclose(np.asarray(cell.positions), expected, rtol=0.0, atol=_PHONOPY_CROSSCHECK_ATOL)
        ):
            raise RuntimeError(
                "Phonopy's displaced supercell differs from the plan's Cartesian reconstruction"
            )


def _software_versions() -> dict[str, str]:
    from importlib import metadata

    return {"phonopy": metadata.version("phonopy"), "spglib": metadata.version("spglib")}


def _import_phonopy():
    try:
        import phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as exc:
        raise PhonopyUnavailableError(
            "Phonopy is required for phonon displacement planning but is not installed"
        ) from exc
    return phonopy, PhonopyAtoms


__all__ = [
    "DisplacementPlan",
    "DisplacementTask",
    "PhonopyUnavailableError",
    "build_displacement_plan",
    "verify_displacement_plan",
]
