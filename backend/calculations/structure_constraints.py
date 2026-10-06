"""Structure constraints that BMD Compute's managed calculations refuse.

A user POSCAR may carry VASP selective-dynamics flags. pymatgen keeps them as
the ``selective_dynamics`` site property, and every input-set generator writes
them back into the generated POSCAR, so a user's frozen coordinates would
silently become part of a BMD-managed calculation. No BMD workflow defines
constrained relaxation, so such structures are rejected, not stripped.

The rule is presence-based: a structure is refused when any site carries a
``selective_dynamics`` property, whatever its value. pymatgen's POSCAR reader
only creates the property when at least one flag is ``F`` (an all-``T`` block is
discarded while parsing, see ``pymatgen.io.vasp.inputs.Poscar``), so ordinary
unconstrained POSCAR and CIF input is never affected. Not inspecting the value
means malformed or unusual values cannot slip through or crash the check.

The check is enforced where VASP inputs are generated: the workflow/flow
builders in ``backend.workflows`` and the input previews in
``backend.generated_inputs``. Every Build, Prepare, Submit, input-reference and
POWER-runtime path passes through one of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.calculations.registry import CalculationValidationError


SELECTIVE_DYNAMICS_SITE_PROPERTY = "selective_dynamics"
SELECTIVE_DYNAMICS_UNSUPPORTED_CODE = "selective_dynamics_unsupported"
SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE = (
    "Selective-dynamics constraints are not supported by managed BMD Compute workflows."
)
SELECTIVE_DYNAMICS_UNSUPPORTED_SUGGESTION = (
    "Remove the 'Selective dynamics' line and the T/F flags after each coordinate "
    "from the POSCAR, then build the calculation again."
)


def reject_unsupported_structure_constraints(structure: Any) -> None:
    """Raise CalculationValidationError if ``structure`` carries selective dynamics.

    The structure is only read, never modified.
    """

    sites = getattr(structure, "sites", None)
    if sites is None:
        return
    constrained_sites = []
    for index, site in enumerate(sites, start=1):
        properties = getattr(site, "properties", None)
        if not isinstance(properties, Mapping) or SELECTIVE_DYNAMICS_SITE_PROPERTY in properties:
            constrained_sites.append(index)
    if constrained_sites:
        raise CalculationValidationError(
            SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE,
            suggestion=SELECTIVE_DYNAMICS_UNSUPPORTED_SUGGESTION,
            diagnostic={
                "code": SELECTIVE_DYNAMICS_UNSUPPORTED_CODE,
                "policy": "fail_closed",
                "site_property": SELECTIVE_DYNAMICS_SITE_PROPERTY,
                "sites": constrained_sites,
            },
        )


__all__ = [
    "SELECTIVE_DYNAMICS_SITE_PROPERTY",
    "SELECTIVE_DYNAMICS_UNSUPPORTED_CODE",
    "SELECTIVE_DYNAMICS_UNSUPPORTED_MESSAGE",
    "SELECTIVE_DYNAMICS_UNSUPPORTED_SUGGESTION",
    "reject_unsupported_structure_constraints",
]
