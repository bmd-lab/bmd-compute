"""Provisional BMD phonon displacement-planning policy.

This is NOT executable BMD methodology. It holds only the values needed to
build a deterministic Phonopy displacement plan. Force-calculation INCAR and
KPOINTS, supercell-size selection, NAC, DOS, thermal properties, scheduling and
execution are deliberately absent: none of them has been approved.

The supercell matrix is not part of the policy. Automatic supercell selection
(for example from a minimum image length) is unapproved methodology, so the
planning API takes an explicit 3x3 integer matrix from its caller and the plan
records it as given.

``primitive_matrix = "auto"`` is passed to Phonopy *explicitly* (Phonopy's own
default changed from the identity in v3 to ``"auto"`` in v4); the matrix Phonopy
resolves is recorded in the plan.

Policy identity: a policy that claims ``PHONON_PLANNING_POLICY_ID`` at a given
version must carry exactly the values registered for that version, so changing
a value (for example ``symprec``) requires an explicit new policy version.
Policies under any other id are accepted for tests and comparison only.

This module imports only the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


PHONON_PLANNING_POLICY_ID = "bmd_compute.phonon_displacement_planning"
PHONON_PLANNING_POLICY_VERSION = 1
PHONON_PLANNING_POLICY_STATUS = "provisional_not_executable"

# Phonopy settings BMD fixes for planning regardless of policy version. They do
# not change the displacement dataset for the reference structures (checked in
# tests), but they are passed explicitly so a future Phonopy default change
# cannot alter a plan silently.
PHONOPY_CALCULATOR = "vasp"
PHONOPY_LANG = "C"

_POLICY_KEYS = frozenset(
    {
        "policy_id",
        "policy_version",
        "status",
        "displacement_distance_angstrom",
        "is_plusminus",
        "is_diagonal",
        "symprec",
        "primitive_matrix",
    }
)

_REGISTERED_VALUES = {
    (PHONON_PLANNING_POLICY_ID, 1): {
        "displacement_distance_angstrom": 0.01,
        "is_plusminus": "auto",
        "is_diagonal": True,
        "symprec": 1e-5,
        "primitive_matrix": "auto",
    },
}


class PhononPolicyError(ValueError):
    """Raised for an invalid or misidentified phonon planning policy."""


@dataclass(frozen=True)
class PhononPolicy:
    """Values that determine a Phonopy displacement plan, and nothing else."""

    policy_id: str = PHONON_PLANNING_POLICY_ID
    policy_version: int = PHONON_PLANNING_POLICY_VERSION
    displacement_distance_angstrom: float = 0.01
    is_plusminus: str | bool = "auto"
    is_diagonal: bool = True
    symprec: float = 1e-5
    primitive_matrix: str = "auto"

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id.strip():
            raise PhononPolicyError("policy_id must be a non-empty string")
        if type(self.policy_version) is not int or self.policy_version < 1:
            raise PhononPolicyError("policy_version must be a positive integer")
        _positive_float(self.displacement_distance_angstrom, "displacement_distance_angstrom")
        _positive_float(self.symprec, "symprec")
        if not (self.is_plusminus == "auto" and type(self.is_plusminus) is str) and type(self.is_plusminus) is not bool:
            raise PhononPolicyError("is_plusminus must be 'auto', True or False")
        if type(self.is_diagonal) is not bool:
            raise PhononPolicyError("is_diagonal must be True or False")
        if self.primitive_matrix != "auto" or type(self.primitive_matrix) is not str:
            raise PhononPolicyError(
                "primitive_matrix must be 'auto'; the resolved matrix is recorded in the plan"
            )
        if self.policy_id == PHONON_PLANNING_POLICY_ID:
            registered = _REGISTERED_VALUES.get((self.policy_id, self.policy_version))
            if registered is None:
                raise PhononPolicyError(
                    f"{self.policy_id} version {self.policy_version} is not a registered policy"
                )
            actual = self._values()
            if actual != registered:
                raise PhononPolicyError(
                    f"{self.policy_id} v{self.policy_version} values {actual} differ from the "
                    f"registered values {registered}; changing a value requires a new policy version"
                )

    def _values(self) -> dict[str, Any]:
        return {
            "displacement_distance_angstrom": self.displacement_distance_angstrom,
            "is_plusminus": self.is_plusminus,
            "is_diagonal": self.is_diagonal,
            "symprec": self.symprec,
            "primitive_matrix": self.primitive_matrix,
        }

    @property
    def status(self) -> str:
        return PHONON_PLANNING_POLICY_STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "status": self.status,
            **self._values(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhononPolicy":
        if not isinstance(data, Mapping) or set(data) != _POLICY_KEYS:
            raise PhononPolicyError(f"policy must have exactly the keys {sorted(_POLICY_KEYS)}")
        if data["status"] != PHONON_PLANNING_POLICY_STATUS:
            raise PhononPolicyError(f"policy status must be {PHONON_PLANNING_POLICY_STATUS!r}")
        return cls(**{key: data[key] for key in _POLICY_KEYS if key != "status"})

    def phonopy_arguments(self, supercell_matrix: Any) -> dict[str, Any]:
        """Every planning argument passed to Phonopy, explicitly.

        ``construct`` goes to ``phonopy.Phonopy(...)`` and ``displacements`` to
        ``Phonopy.generate_displacements(...)``. Nothing is left to Phonopy's
        defaults.
        """

        return {
            "construct": {
                "supercell_matrix": validated_supercell_matrix(supercell_matrix),
                "primitive_matrix": self.primitive_matrix,
                "symprec": self.symprec,
                "is_symmetry": True,
                "distinguish_symbol_index": False,
                "use_SNF_supercell": False,
                "calculator": PHONOPY_CALCULATOR,
                "lang": PHONOPY_LANG,
            },
            "displacements": {
                "distance": self.displacement_distance_angstrom,
                "is_plusminus": self.is_plusminus,
                "is_diagonal": self.is_diagonal,
                "is_trigonal": False,
                "number_of_snapshots": None,
                "random_seed": None,
            },
        }


def validated_supercell_matrix(value: Any) -> list[list[int]]:
    """An explicit 3x3 integer matrix with positive determinant, or raise.

    No shorthand (three diagonal integers), no float coercion and no bools:
    the caller must state the matrix it means.
    """

    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 3:
        raise PhononPolicyError("supercell_matrix must be an explicit 3x3 integer matrix")
    rows = []
    for row in value:
        if isinstance(row, (str, bytes)) or not hasattr(row, "__len__") or len(row) != 3:
            raise PhononPolicyError("supercell_matrix must be an explicit 3x3 integer matrix")
        converted = []
        for item in row:
            if isinstance(item, bool) or not _is_integral(item):
                raise PhononPolicyError(f"supercell_matrix entries must be integers, got {item!r}")
            converted.append(int(item))
        rows.append(converted)
    det = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if det <= 0:
        raise PhononPolicyError("supercell_matrix must have a positive determinant")
    return rows


def _is_integral(value: Any) -> bool:
    import numbers

    return isinstance(value, numbers.Integral)


def _positive_float(value: Any, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise PhononPolicyError(f"{label} must be a positive finite float, got {value!r}")


PROVISIONAL_PHONON_POLICY = PhononPolicy()


__all__ = [
    "PHONON_PLANNING_POLICY_ID",
    "PHONON_PLANNING_POLICY_STATUS",
    "PHONON_PLANNING_POLICY_VERSION",
    "PHONOPY_CALCULATOR",
    "PHONOPY_LANG",
    "PROVISIONAL_PHONON_POLICY",
    "PhononPolicy",
    "PhononPolicyError",
    "validated_supercell_matrix",
]
