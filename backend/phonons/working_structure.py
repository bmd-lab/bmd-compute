"""The phonon working-structure boundary (Phonopy M2b; not executable).

Under R2 the displacement plan is built from the actual output of the
prerequisite relaxation (Stage 1). This module is the deterministic boundary
between that output and the structure handed to M1:

    A. the original user structure     (kept in the submission, not used here)
    B. the actual Stage-1 structure    -> ``incoming`` identity
    C. the BMD phonon working structure -> ``working`` identity (what M1 plans)

``prepare_phonon_working_structure(stage1_structure)`` returns a
``bmd_compute.phonon_working_structure`` v1 record that holds both identities,
every transformation between them, the policy and the software that performed
them. B is never modified, and C is never presented as the Stage-1 output.

Site and species policy (closed; anything else is rejected, never repaired):

* ``magmom``: kept in the incoming identity; must be exactly zero on every site
  (scalar or 3-vector), then dropped (``drop_zero_magmom``). Any nonzero,
  non-finite or malformed moment is rejected: magnetic phonons are not
  supported, and a magnetic result must not pass as non-magnetic.
* ``velocities`` (VASP writes a zero block to CONTCAR): kept in the incoming
  identity; must be exactly zero, then dropped (``drop_zero_velocities``).
* ``selective_dynamics`` is rejected, as everywhere in managed BMD workflows;
  ``predictor_corrector`` and every other site property are rejected.
* species: ``Element`` or ``Species``. Oxidation states and a zero spin are
  kept in the incoming identity and reduced to element symbols in the working
  structure (``reduce_species_to_elements``); a nonzero or unreadable spin is
  rejected. Dummy species, disordered or partially occupied sites are rejected.
* structure level: a nonzero charge, non-empty ``properties``, non-periodic
  directions, non-finite values or a lattice that is singular or left-handed
  are rejected. Site labels are not scientific identity and are ignored.

Symmetry idealization (``symmetry_idealization``, always applied):

The working structure is the Stage-1 structure made exactly symmetric under the
space group spglib detects at the policy tolerance, *in the input setting*: the
same lattice basis, atom count, atom order, origin and handedness. The lattice
is spglib's idealized standard lattice mapped back through its transformation
matrix and standardization rotation (``P^T L_std R``), so the Cartesian frame
is kept and only the lattice metric is symmetrized. That is a small strain, but
on long or skewed basis vectors an individual lattice component can change by
more than ``symprec``; no component-wise bound is implied or enforced. Positions
are the average
of each atom's symmetry images (the projection onto the symmetric
configuration), so no origin shift is introduced and free coordinates (for
example along a polar axis) are unchanged. Nothing is wrapped or rounded.

Full spglib standardization is deliberately not used: it changes the atom count
(conventional or primitive cell), reorders atoms, shifts the origin and can
rotate and reflect the frame, all of which would be unapproved methodology.

The idealization tolerance is the policy's own value. It equals the approved M1
planning ``symprec`` (1e-5 Angstrom), so idealization only cleans what Phonopy
would already treat as symmetric. It never recovers symmetry that Phonopy would
not detect, and it must not change the detected space group. Recovering
symmetry lost by larger distortions (for example coordinates rounded to four
decimals) would need a looser tolerance. That is an unapproved scientific
decision and is not made here.

Identity and hashing reuse the M1 canonical rules (``records.canonical_json``).
``incoming.sha256`` answers "what exact structure did Stage 1 hand over";
``working.sha256`` is the M1 ``structure_record`` hash of the structure handed
to Phonopy, so it equals ``plan.working_structure.sha256``.

Validation has two levels. ``validate_working_structure_record`` checks the
record on its own: structure, policy, identities, the transformation list
implied by the incoming structure, symmetry blocks, the recorded idealization
magnitudes and the hash. ``verify_phonon_working_structure`` re-derives the
record from the actual Stage-1 structure, which makes it authoritative.

Module import uses only the standard library; NumPy, spglib and pymatgen are
imported when a structure is transformed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.phonons.records import (
    COORDS_FRACTIONAL,
    ELEMENT_SYMBOL_PATTERN,
    PhononPlanContractError,
    _structure,
    canonical_json,
    canonical_sha256,
    canonical_value,
    structure_record,
)


WORKING_STRUCTURE_SCHEMA = "bmd_compute.phonon_working_structure"
WORKING_STRUCTURE_SCHEMA_VERSION = 1

WORKING_STRUCTURE_POLICY_ID = "bmd_compute.phonon_working_structure"
WORKING_STRUCTURE_POLICY_VERSION = 1
WORKING_STRUCTURE_POLICY_STATUS = "provisional_not_executable"
SYMMETRY_IDEALIZATION_METHOD = "spglib_symmetry_idealization_in_input_setting"

# Transformation kinds, in the order they are applied and recorded.
DROP_ZERO_MAGMOM = "drop_zero_magmom"
DROP_ZERO_VELOCITIES = "drop_zero_velocities"
REDUCE_SPECIES_TO_ELEMENTS = "reduce_species_to_elements"
SYMMETRY_IDEALIZATION = "symmetry_idealization"

# Site properties that may appear on a Stage-1 structure and are dropped only
# when exactly zero. Everything else is rejected.
DROPPABLE_ZERO_SITE_PROPERTIES = {
    "magmom": DROP_ZERO_MAGMOM,
    "velocities": DROP_ZERO_VELOCITIES,
}

RECORD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "policy",
        "software",
        "incoming",
        "working",
        "transformations",
        "symmetry",
        "idealization",
        "record_sha256",
    }
)
POLICY_KEYS = frozenset(
    {"policy_id", "policy_version", "status", "method", "symprec_angstrom", "angle_tolerance_degrees"}
)
SOFTWARE_KEYS = frozenset({"spglib", "numpy"})
INCOMING_KEYS = frozenset({"lattice", "species", "coords", "coords_type", "site_properties", "sha256"})
SPECIES_KEYS = frozenset({"element", "oxidation_state", "spin"})
SYMMETRY_KEYS = frozenset(
    {"international", "number", "hall_number", "operations", "rotations", "equivalent_atoms"}
)
IDEALIZATION_KEYS = frozenset({"max_site_shift_angstrom", "max_lattice_change_angstrom"})

_REGISTERED_POLICY_VALUES = {
    (WORKING_STRUCTURE_POLICY_ID, 1): {
        "method": SYMMETRY_IDEALIZATION_METHOD,
        "symprec_angstrom": 1e-5,
        # spglib's own angle handling, passed explicitly (never left implicit).
        "angle_tolerance_degrees": -1.0,
    },
}


class PhononWorkingStructureError(ValueError):
    """Raised when a Stage-1 structure or a working-structure record is refused."""

    def __init__(self, message: str, *, code: str = "invalid"):
        super().__init__(message)
        self.code = code


# --- policy ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkingStructurePolicy:
    policy_id: str = WORKING_STRUCTURE_POLICY_ID
    policy_version: int = WORKING_STRUCTURE_POLICY_VERSION
    method: str = SYMMETRY_IDEALIZATION_METHOD
    symprec_angstrom: float = 1e-5
    angle_tolerance_degrees: float = -1.0

    def __post_init__(self) -> None:
        registered = _REGISTERED_POLICY_VALUES.get((self.policy_id, self.policy_version))
        if registered is None or type(self.policy_version) is not int:
            raise PhononWorkingStructureError(
                f"{self.policy_id!r} version {self.policy_version!r} is not a registered "
                "working-structure policy",
                code="unsupported_policy",
            )
        values = {
            "method": self.method,
            "symprec_angstrom": self.symprec_angstrom,
            "angle_tolerance_degrees": self.angle_tolerance_degrees,
        }
        if any(type(values[key]) is not type(registered[key]) for key in registered) or values != registered:
            raise PhononWorkingStructureError(
                f"working-structure policy values {values} differ from the registered values "
                f"{registered}; changing a value requires a new policy version",
                code="unsupported_policy",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "status": WORKING_STRUCTURE_POLICY_STATUS,
            "method": self.method,
            "symprec_angstrom": self.symprec_angstrom,
            "angle_tolerance_degrees": self.angle_tolerance_degrees,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "WorkingStructurePolicy":
        if not isinstance(data, Mapping) or set(data) != POLICY_KEYS:
            raise PhononWorkingStructureError(
                f"policy must have exactly the keys {sorted(POLICY_KEYS)}", code="unsupported_policy"
            )
        if data["status"] != WORKING_STRUCTURE_POLICY_STATUS:
            raise PhononWorkingStructureError(
                f"policy status must be {WORKING_STRUCTURE_POLICY_STATUS!r}", code="unsupported_policy"
            )
        return cls(**{key: data[key] for key in POLICY_KEYS if key != "status"})


PROVISIONAL_WORKING_STRUCTURE_POLICY = WorkingStructurePolicy()


# --- immutable record ------------------------------------------------------------------


@dataclass(frozen=True)
class PhononWorkingStructure:
    """Immutable, validated ``bmd_compute.phonon_working_structure`` v1 record."""

    _canonical: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self._canonical) is not str:
            raise PhononWorkingStructureError("PhononWorkingStructure holds canonical JSON text")
        try:
            data = json.loads(self._canonical)
        except ValueError as exc:
            raise PhononWorkingStructureError(f"record is not valid JSON: {exc}") from None
        validate_working_structure_record(data)
        if canonical_json(data) != self._canonical:
            raise PhononWorkingStructureError("PhononWorkingStructure text is not in canonical form")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhononWorkingStructure":
        if not isinstance(data, Mapping):
            raise PhononWorkingStructureError("record must be an object")
        validate_working_structure_record(data)
        return cls(canonical_json(dict(data)))

    @classmethod
    def from_json(cls, text: str) -> "PhononWorkingStructure":
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise PhononWorkingStructureError(f"record is not valid JSON: {exc}") from None
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def to_json(self) -> str:
        return self._canonical

    @property
    def record_sha256(self) -> str:
        return self.to_dict()["record_sha256"]

    @property
    def incoming_sha256(self) -> str:
        """Identity of the actual Stage-1 structure (B)."""

        return self.to_dict()["incoming"]["sha256"]

    @property
    def working_sha256(self) -> str:
        """Identity of the working structure handed to M1 (C)."""

        return self.to_dict()["working"]["sha256"]

    @property
    def transformations(self) -> tuple[str, ...]:
        return tuple(self.to_dict()["transformations"])

    def working_structure(self):
        """A new pymatgen ``Structure`` of the working structure, for M1 planning."""

        from pymatgen.core import Lattice, Structure

        working = self.to_dict()["working"]
        return Structure(
            Lattice(working["lattice"]),
            list(working["species"]),
            working["coords"],
            coords_are_cartesian=False,
        )


# --- the boundary ----------------------------------------------------------------------


def prepare_phonon_working_structure(
    stage1_structure: Any,
    policy: WorkingStructurePolicy = PROVISIONAL_WORKING_STRUCTURE_POLICY,
) -> PhononWorkingStructure:
    """Return the working-structure record for the actual Stage-1 structure.

    ``stage1_structure`` (a pymatgen ``Structure``) is only read. Raises
    PhononWorkingStructureError for anything the policy does not support.
    """

    if not isinstance(policy, WorkingStructurePolicy):
        raise TypeError("policy must be a WorkingStructurePolicy")
    incoming, transformations, elements = _incoming_record(stage1_structure)

    import numpy as np

    lattice = np.array(incoming["lattice"], dtype=float)
    coords = np.array(incoming["coords"], dtype=float)
    numbers = [_atomic_number(symbol) for symbol in elements]

    incoming_symmetry, ideal_lattice, ideal_coords = _idealize(lattice, coords, numbers, policy)
    working = structure_record(
        [[float(x) for x in row] for row in ideal_lattice.tolist()],
        elements,
        [[float(x) for x in row] for row in ideal_coords.tolist()],
        COORDS_FRACTIONAL,
    )
    working_symmetry = _symmetry_summary(
        _symmetry_dataset(np.array(working["lattice"]), np.array(working["coords"]), numbers, policy)
    )
    if working_symmetry != incoming_symmetry:
        raise PhononWorkingStructureError(
            "symmetry idealization changed the detected symmetry (space group, Hall setting, "
            f"rotations or equivalent sites; {incoming_symmetry['international']} -> "
            f"{working_symmetry['international']}); the structure is too close to the symmetry "
            "tolerance to be idealized deterministically",
            code="symmetry_unstable",
        )
    idealization = _idealization_magnitudes(incoming, working)
    if idealization["max_site_shift_angstrom"] > policy.symprec_angstrom:
        raise PhononWorkingStructureError(
            "symmetry idealization moved a site further than the tolerance", code="symmetry_unstable"
        )

    record = {
        "schema": WORKING_STRUCTURE_SCHEMA,
        "schema_version": WORKING_STRUCTURE_SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "software": _software_versions(),
        "incoming": incoming,
        "working": working,
        "transformations": transformations + [SYMMETRY_IDEALIZATION],
        "symmetry": {"incoming": incoming_symmetry, "working": working_symmetry},
        "idealization": idealization,
    }
    record["record_sha256"] = record_sha256(record)
    return PhononWorkingStructure.from_dict(record)


def verify_phonon_working_structure(
    record: PhononWorkingStructure,
    stage1_structure: Any,
) -> PhononWorkingStructure:
    """Re-derive ``record`` from the actual Stage-1 structure and require equality.

    A record is never authority on its own; this is how one is accepted. The
    software that produced it must be the software verifying it.
    """

    if not isinstance(record, PhononWorkingStructure):
        raise PhononWorkingStructureError("record must be a PhononWorkingStructure")
    data = record.to_dict()
    current = _software_versions()
    if data["software"] != current:
        raise PhononWorkingStructureError(
            f"record was produced with {data['software']}, this environment has {current}; "
            "it cannot be re-verified here",
            code="software_mismatch",
        )
    rebuilt = prepare_phonon_working_structure(
        stage1_structure, WorkingStructurePolicy.from_dict(data["policy"])
    )
    if rebuilt.to_json() != record.to_json():
        if rebuilt.incoming_sha256 != record.incoming_sha256:
            raise PhononWorkingStructureError(
                "record does not describe this Stage-1 structure", code="incoming_mismatch"
            )
        raise PhononWorkingStructureError(
            "record differs from the working structure derived from this Stage-1 structure",
            code="working_mismatch",
        )
    return rebuilt


def record_sha256(record: Mapping[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return canonical_sha256(body)


# --- incoming identity -----------------------------------------------------------------


def _incoming_record(structure: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    from pymatgen.core import IStructure
    from pymatgen.core.periodic_table import DummySpecies, Element, Species

    if not isinstance(structure, IStructure):
        raise PhononWorkingStructureError("Stage-1 structure must be a pymatgen Structure", code="malformed")
    if len(structure) == 0:
        raise PhononWorkingStructureError("Stage-1 structure has no sites", code="malformed")
    pbc = getattr(structure.lattice, "pbc", (True, True, True))
    if tuple(bool(flag) for flag in pbc) != (True, True, True):
        raise PhononWorkingStructureError("Stage-1 structure must be periodic in 3D", code="malformed")
    charge = getattr(structure, "_charge", None)
    if charge is not None and _number(charge, "charge") != 0.0:
        raise PhononWorkingStructureError("charged Stage-1 structures are not supported", code="unsupported")
    if getattr(structure, "properties", None):
        raise PhononWorkingStructureError(
            f"structure-level properties {sorted(structure.properties)} are not supported",
            code="unsupported_property",
        )

    lattice = [[_number(x, "lattice") for x in row] for row in structure.lattice.matrix.tolist()]
    if _det3(lattice) <= 0.0:
        raise PhononWorkingStructureError(
            "Stage-1 lattice must be non-singular and right-handed", code="malformed"
        )

    species: list[dict[str, Any]] = []
    elements: list[str] = []
    decorated = False
    for index, site in enumerate(structure, start=1):
        if not site.is_ordered:
            raise PhononWorkingStructureError(
                f"site {index} is disordered or partially occupied", code="unsupported"
            )
        specie = site.specie
        if isinstance(specie, DummySpecies) or type(specie) not in (Element, Species):
            raise PhononWorkingStructureError(
                f"site {index} species {specie!r} is not a chemical element", code="unsupported"
            )
        if type(specie) is Element:
            symbol, oxidation, spin = specie.symbol, None, None
        else:
            decorated = True
            symbol = specie.element.symbol
            oxidation = None if specie.oxi_state is None else _number(specie.oxi_state, "oxidation state")
            raw_spin = getattr(specie, "spin", None)
            spin = None if raw_spin is None else _number(raw_spin, f"site {index} spin", code="magnetic")
            if spin is not None and spin != 0.0:
                raise PhononWorkingStructureError(
                    f"site {index} species {specie!r} carries a spin; magnetic phonons are not supported",
                    code="magnetic",
                )
        if not ELEMENT_SYMBOL_PATTERN.fullmatch(symbol):
            raise PhononWorkingStructureError(f"site {index} element {symbol!r} is not supported", code="unsupported")
        species.append({"element": symbol, "oxidation_state": oxidation, "spin": spin})
        elements.append(symbol)

    site_properties, transformations = _site_properties(structure)
    if decorated:
        transformations.append(REDUCE_SPECIES_TO_ELEMENTS)

    coords = [[_number(x, "coordinates") for x in row] for row in structure.frac_coords.tolist()]
    body = {
        "lattice": lattice,
        "species": species,
        "coords": coords,
        "coords_type": COORDS_FRACTIONAL,
        "site_properties": site_properties,
    }
    body["sha256"] = canonical_sha256(body)
    return body, transformations, elements


def _site_properties(structure: Any) -> tuple[dict[str, Any], list[str]]:
    names = set()
    for site in structure:
        properties = getattr(site, "properties", None)
        if not isinstance(properties, Mapping):
            raise PhononWorkingStructureError("site properties are not a mapping", code="malformed")
        names.update(properties)
    if "selective_dynamics" in names:
        raise PhononWorkingStructureError(
            "Selective-dynamics constraints are not supported by managed BMD Compute workflows.",
            code="selective_dynamics_unsupported",
        )
    unsupported = sorted(str(name) for name in names if name not in DROPPABLE_ZERO_SITE_PROPERTIES)
    if unsupported:
        raise PhononWorkingStructureError(
            f"site properties {unsupported} are not supported for phonon planning",
            code="unsupported_property",
        )

    recorded: dict[str, Any] = {}
    transformations: list[str] = []
    for name in sorted(names):
        values = []
        for index, site in enumerate(structure, start=1):
            if name not in site.properties:
                raise PhononWorkingStructureError(
                    f"site property {name!r} is missing on site {index}", code="malformed"
                )
            values.append(_property_value(name, site.properties[name], index))
        recorded[name] = values
    # Applied (and recorded) in a fixed order independent of the input.
    for name, kind in DROPPABLE_ZERO_SITE_PROPERTIES.items():
        if name in recorded:
            transformations.append(kind)
    return recorded, transformations


def _property_value(name: str, value: Any, index: int) -> Any:
    code = "magnetic" if name == "magmom" else "unsupported_property"
    moment = getattr(value, "moment", None)
    if moment is not None:
        value = moment
    if name == "magmom" and _is_scalar(value):
        number = _number(value, f"site {index} magmom", code="magnetic")
        components = [number]
        result: Any = number
    else:
        # Unreadable magnetic data is treated as possibly magnetic.
        shape_code = "magnetic" if name == "magmom" else "malformed"
        if value is None or _is_scalar(value) or isinstance(value, (str, bytes, Mapping)):
            raise PhononWorkingStructureError(f"site {index} {name} must be a 3-vector", code=shape_code)
        try:
            items = list(value)
        except TypeError:
            raise PhononWorkingStructureError(f"site {index} {name} is malformed", code=shape_code) from None
        if len(items) != 3:
            raise PhononWorkingStructureError(f"site {index} {name} must have 3 components", code=shape_code)
        components = [_number(item, f"site {index} {name}", code=code) for item in items]
        result = components
    if any(component != 0.0 for component in components):
        if name == "magmom":
            raise PhononWorkingStructureError(
                f"site {index} has a nonzero magnetic moment; magnetic phonons are not supported",
                code="magnetic",
            )
        raise PhononWorkingStructureError(
            f"site {index} has nonzero {name}; only an all-zero {name} block is accepted",
            code="unsupported_property",
        )
    return result


def _is_scalar(value: Any) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    shape = getattr(value, "shape", None)
    return shape == ()


def _number(value: Any, label: str, *, code: str = "malformed") -> float:
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        raise PhononWorkingStructureError(f"{label} must be a finite number", code=code)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PhononWorkingStructureError(f"{label} must be a finite number", code=code) from None
    if not math.isfinite(number):
        raise PhononWorkingStructureError(f"{label} must be a finite number", code=code)
    return 0.0 if number == 0.0 else number


# --- symmetry idealization -------------------------------------------------------------


def _symmetry_dataset(lattice, coords, numbers, policy: WorkingStructurePolicy):
    import spglib

    dataset = spglib.get_symmetry_dataset(
        (lattice, coords, numbers),
        symprec=policy.symprec_angstrom,
        angle_tolerance=policy.angle_tolerance_degrees,
        hall_number=0,
    )
    if dataset is None:
        raise PhononWorkingStructureError(
            f"spglib could not analyse the structure: {spglib.spglib.get_error_message()}",
            code="symmetry_failed",
        )
    return dataset


def _symmetry_summary(dataset) -> dict[str, Any]:
    """Setting-level symmetry that idealization must leave unchanged.

    Besides the space group: the Hall setting, the set of distinct point
    rotations in the input basis (sorted), and the partition of sites into
    symmetry-equivalent classes (each site mapped to the lowest index in its
    class). Wyckoff letters and origin shifts are deliberately not included:
    they are representation choices that can change without any physical
    change.
    """

    equivalent = [int(value) for value in dataset.equivalent_atoms]
    lowest: dict[int, int] = {}
    for index, representative in enumerate(equivalent):
        lowest.setdefault(representative, index)
    rotations = sorted(
        {tuple(int(value) for value in rotation.flatten()) for rotation in dataset.rotations}
    )
    return {
        "international": str(dataset.international),
        "number": int(dataset.number),
        "hall_number": int(dataset.hall_number),
        "operations": int(len(dataset.rotations)),
        "rotations": [[list(rotation[0:3]), list(rotation[3:6]), list(rotation[6:9])] for rotation in rotations],
        "equivalent_atoms": [lowest[representative] for representative in equivalent],
    }


def _idealize(lattice, coords, numbers, policy: WorkingStructurePolicy):
    import numpy as np

    dataset = _symmetry_dataset(lattice, coords, numbers, policy)
    # spglib: L_std = R . A . P^-1 for column lattices A. Mapping the idealized
    # standard lattice back gives the idealized lattice in the input basis and
    # Cartesian frame: rows P^T L_std R.
    ideal_lattice = (
        np.asarray(dataset.transformation_matrix, dtype=float).T
        @ np.asarray(dataset.std_lattice, dtype=float)
        @ np.asarray(dataset.std_rotation_matrix, dtype=float)
    )
    natom = len(numbers)
    types = np.asarray(numbers)
    corrections = np.zeros((natom, 3))
    for rotation, translation in zip(dataset.rotations, dataset.translations):
        images = coords @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
        targets = set()
        for j in range(natom):
            delta = images[j] - coords
            delta -= np.round(delta)
            distance = np.linalg.norm(delta @ lattice, axis=1)
            distance[types != types[j]] = np.inf
            close = np.flatnonzero(distance <= policy.symprec_angstrom)
            if len(close) != 1:
                raise PhononWorkingStructureError(
                    "a symmetry image does not map onto exactly one atom", code="symmetry_failed"
                )
            target = int(close[0])
            targets.add(target)
            corrections[target] += delta[target]
        if len(targets) != natom:
            raise PhononWorkingStructureError(
                "a symmetry operation does not permute the atoms", code="symmetry_failed"
            )
    ideal_coords = coords + corrections / len(dataset.rotations)
    if not (np.all(np.isfinite(ideal_lattice)) and np.all(np.isfinite(ideal_coords))):
        raise PhononWorkingStructureError("symmetry idealization produced non-finite values", code="symmetry_failed")
    return _symmetry_summary(dataset), ideal_lattice, ideal_coords


def _idealization_magnitudes(incoming: Mapping[str, Any], working: Mapping[str, Any]) -> dict[str, float]:
    """Largest site shift and lattice-component change; pure Python, so re-checkable."""

    lattice = working["lattice"]
    max_shift = 0.0
    for before, after in zip(incoming["coords"], working["coords"]):
        delta = [after[k] - before[k] for k in range(3)]
        cart = [sum(delta[k] * lattice[k][axis] for k in range(3)) for axis in range(3)]
        max_shift = max(max_shift, math.sqrt(sum(component * component for component in cart)))
    max_lattice = max(
        abs(working["lattice"][row][col] - incoming["lattice"][row][col])
        for row in range(3)
        for col in range(3)
    )
    return {
        "max_site_shift_angstrom": max_shift + 0.0,
        "max_lattice_change_angstrom": max_lattice + 0.0,
    }


def _software_versions() -> dict[str, str]:
    from importlib import metadata

    return {"spglib": metadata.version("spglib"), "numpy": metadata.version("numpy")}


def _atomic_number(symbol: str) -> int:
    from pymatgen.core.periodic_table import Element

    return int(Element(symbol).Z)


# --- structural validation -------------------------------------------------------------


def validate_working_structure_record(payload: Any) -> None:
    """Raise PhononWorkingStructureError unless ``payload`` is a consistent v1 record.

    Passing this check does not make a record authoritative; see
    ``verify_phonon_working_structure``.
    """

    record = _mapping(payload, "record")
    _exact_keys(record, RECORD_KEYS, "record")
    if record["schema"] != WORKING_STRUCTURE_SCHEMA:
        raise PhononWorkingStructureError(f"schema must be {WORKING_STRUCTURE_SCHEMA!r}", code="schema")
    if type(record["schema_version"]) is not int or record["schema_version"] != WORKING_STRUCTURE_SCHEMA_VERSION:
        raise PhononWorkingStructureError(
            f"schema_version must be {WORKING_STRUCTURE_SCHEMA_VERSION}", code="schema"
        )
    try:
        canonical_value(record)
    except PhononPlanContractError as exc:
        raise PhononWorkingStructureError(f"record is not canonical data: {exc}") from None

    policy = WorkingStructurePolicy.from_dict(record["policy"])
    if canonical_json(policy.to_dict()) != canonical_json(dict(record["policy"])):
        raise PhononWorkingStructureError("policy block is not the canonical form of its policy")

    software = _mapping(record["software"], "software")
    _exact_keys(software, SOFTWARE_KEYS, "software")
    for name in SOFTWARE_KEYS:
        if type(software[name]) is not str or not software[name].strip():
            raise PhononWorkingStructureError(f"software.{name} must be a version string")

    incoming = _incoming_block(record["incoming"])
    try:
        working = _structure(record["working"], "working", COORDS_FRACTIONAL)
    except PhononPlanContractError as exc:
        raise PhononWorkingStructureError(str(exc)) from None
    # Handedness is part of the working-structure invariant, not only of the
    # incoming structure: a reflected working lattice must not validate.
    if _det3(working["lattice"]) <= 0.0:
        raise PhononWorkingStructureError("working lattice must be right-handed (positive determinant)")
    if working["species"] != [entry["element"] for entry in incoming["species"]]:
        raise PhononWorkingStructureError("working species are not the incoming elements in order")
    if len(working["coords"]) != len(incoming["coords"]):
        raise PhononWorkingStructureError("working structure has a different number of sites")

    expected = [
        kind for name, kind in DROPPABLE_ZERO_SITE_PROPERTIES.items() if name in incoming["site_properties"]
    ]
    if any(entry["oxidation_state"] is not None or entry["spin"] is not None for entry in incoming["species"]):
        expected.append(REDUCE_SPECIES_TO_ELEMENTS)
    expected.append(SYMMETRY_IDEALIZATION)
    if record["transformations"] != expected:
        raise PhononWorkingStructureError(
            f"transformations must be {expected} for this incoming structure, got {record['transformations']!r}"
        )

    symmetry = _mapping(record["symmetry"], "symmetry")
    _exact_keys(symmetry, {"incoming", "working"}, "symmetry")
    blocks = [
        _symmetry_block(symmetry[name], f"symmetry.{name}", len(incoming["species"]))
        for name in ("incoming", "working")
    ]
    if blocks[0] != blocks[1]:
        raise PhononWorkingStructureError("symmetry idealization must not change the detected space group")

    idealization = _mapping(record["idealization"], "idealization")
    _exact_keys(idealization, IDEALIZATION_KEYS, "idealization")
    if dict(idealization) != _idealization_magnitudes(incoming, working):
        raise PhononWorkingStructureError("idealization magnitudes do not match the incoming and working structures")
    if idealization["max_site_shift_angstrom"] > policy.symprec_angstrom:
        raise PhononWorkingStructureError("idealization moved a site further than the policy tolerance")

    if _sha(record["record_sha256"], "record_sha256") != record_sha256(record):
        raise PhononWorkingStructureError("record_sha256 does not match the canonical record content")


def _incoming_block(value: Any) -> dict[str, Any]:
    incoming = _mapping(value, "incoming")
    _exact_keys(incoming, INCOMING_KEYS, "incoming")
    if incoming["coords_type"] != COORDS_FRACTIONAL:
        raise PhononWorkingStructureError("incoming.coords_type must be fractional")
    lattice = _float_rows(incoming["lattice"], "incoming.lattice", 3)
    if len(lattice) != 3 or _det3(lattice) <= 0.0:
        raise PhononWorkingStructureError("incoming.lattice must be a right-handed 3x3 lattice")
    species = incoming["species"]
    if type(species) is not list or not species:
        raise PhononWorkingStructureError("incoming.species must be a non-empty list")
    for index, entry in enumerate(species):
        entry = _mapping(entry, f"incoming.species[{index}]")
        _exact_keys(entry, SPECIES_KEYS, f"incoming.species[{index}]")
        if type(entry["element"]) is not str or not ELEMENT_SYMBOL_PATTERN.fullmatch(entry["element"]):
            raise PhononWorkingStructureError(f"incoming.species[{index}].element is invalid")
        if entry["oxidation_state"] is not None and type(entry["oxidation_state"]) is not float:
            raise PhononWorkingStructureError(f"incoming.species[{index}].oxidation_state must be a float or null")
        if entry["spin"] is not None and (type(entry["spin"]) is not float or entry["spin"] != 0.0):
            raise PhononWorkingStructureError(f"incoming.species[{index}].spin must be null or 0.0")
    coords = _float_rows(incoming["coords"], "incoming.coords", 3)
    if len(coords) != len(species):
        raise PhononWorkingStructureError("incoming.coords and incoming.species differ in length")
    properties = _mapping(incoming["site_properties"], "incoming.site_properties")
    for name, values in properties.items():
        if name not in DROPPABLE_ZERO_SITE_PROPERTIES:
            raise PhononWorkingStructureError(f"incoming site property {name!r} is not supported")
        if type(values) is not list or len(values) != len(species):
            raise PhononWorkingStructureError(f"incoming site property {name!r} must list every site")
        for value in values:
            components = [value] if type(value) is float else value
            if name == "velocities" and type(value) is float:
                raise PhononWorkingStructureError("incoming velocities must be 3-vectors")
            if (
                type(components) is not list
                or len(components) not in ((1,) if type(value) is float else (3,))
                or any(type(c) is not float or c != 0.0 for c in components)
            ):
                raise PhononWorkingStructureError(f"incoming site property {name!r} must be exactly zero")
    body = {key: incoming[key] for key in INCOMING_KEYS if key != "sha256"}
    if _sha(incoming["sha256"], "incoming.sha256") != canonical_sha256(body):
        raise PhononWorkingStructureError("incoming.sha256 does not match its content")
    return dict(incoming)


def _symmetry_block(value: Any, label: str, natom: int) -> dict[str, Any]:
    block = _mapping(value, label)
    _exact_keys(block, SYMMETRY_KEYS, label)
    if type(block["international"]) is not str or not block["international"]:
        raise PhononWorkingStructureError(f"{label}.international must be a symbol")
    if type(block["number"]) is not int or not 1 <= block["number"] <= 230:
        raise PhononWorkingStructureError(f"{label}.number must be a space-group number 1-230")
    if type(block["hall_number"]) is not int or not 1 <= block["hall_number"] <= 530:
        raise PhononWorkingStructureError(f"{label}.hall_number must be a Hall number 1-530")
    if type(block["operations"]) is not int or block["operations"] < 1:
        raise PhononWorkingStructureError(f"{label}.operations must be a positive integer")

    rotations = block["rotations"]
    if type(rotations) is not list or not rotations:
        raise PhononWorkingStructureError(f"{label}.rotations must be a non-empty list")
    flattened = []
    for rotation in rotations:
        if (
            type(rotation) is not list
            or len(rotation) != 3
            or any(type(row) is not list or len(row) != 3 or any(type(x) is not int for x in row) for row in rotation)
        ):
            raise PhononWorkingStructureError(f"{label}.rotations must be 3x3 integer matrices")
        if abs(_det3(rotation)) != 1:
            raise PhononWorkingStructureError(f"{label}.rotations must be unimodular")
        flattened.append(tuple(x for row in rotation for x in row))
    if flattened != sorted(set(flattened)):
        raise PhononWorkingStructureError(f"{label}.rotations must be distinct and sorted")
    if (1, 0, 0, 0, 1, 0, 0, 0, 1) not in flattened or block["operations"] % len(flattened):
        raise PhononWorkingStructureError(f"{label}.rotations are inconsistent with the operations")

    equivalent = block["equivalent_atoms"]
    if (
        type(equivalent) is not list
        or len(equivalent) != natom
        or any(type(x) is not int or not 0 <= x <= index or equivalent[x] != x for index, x in enumerate(equivalent))
    ):
        raise PhononWorkingStructureError(
            f"{label}.equivalent_atoms must map each site to the lowest index of its class"
        )
    return dict(block)


def _float_rows(value: Any, label: str, width: int) -> list[list[float]]:
    if type(value) is not list:
        raise PhononWorkingStructureError(f"{label} must be a list")
    rows = []
    for row in value:
        if type(row) is not list or len(row) != width or any(type(x) is not float for x in row):
            raise PhononWorkingStructureError(f"{label} rows must be {width} floats")
        rows.append(row)
    return rows


def _det3(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhononWorkingStructureError(f"{label} must be an object")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected, label: str) -> None:
    keys = set(mapping)
    missing = sorted(set(expected) - keys)
    extra = sorted(str(key) for key in keys - set(expected))
    if missing or extra:
        raise PhononWorkingStructureError(f"{label} keys: missing {missing}, unexpected {extra}")


def _sha(value: Any, label: str) -> str:
    import re

    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PhononWorkingStructureError(f"{label} must be a lower-case SHA-256 hex digest")
    return value


__all__ = [
    "DROP_ZERO_MAGMOM",
    "DROP_ZERO_VELOCITIES",
    "PROVISIONAL_WORKING_STRUCTURE_POLICY",
    "PhononWorkingStructure",
    "PhononWorkingStructureError",
    "REDUCE_SPECIES_TO_ELEMENTS",
    "SYMMETRY_IDEALIZATION",
    "WORKING_STRUCTURE_SCHEMA",
    "WORKING_STRUCTURE_SCHEMA_VERSION",
    "WorkingStructurePolicy",
    "prepare_phonon_working_structure",
    "record_sha256",
    "validate_working_structure_record",
    "verify_phonon_working_structure",
]
