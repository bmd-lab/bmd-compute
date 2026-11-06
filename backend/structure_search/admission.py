"""Which structure searches bmd-compute accepts (v1).

``spec.StructureSearchSpec`` checks that a search request is well formed.
This module decides the narrower question of whether bmd-compute *supports*
it. The scope is the lab decision of 6 Nov 2026: fixed-composition,
ambient-pressure, non-magnetic, plain-PBE searches. Version 2 accepts only:

- ``calculationType = 300`` (fixed-composition 3D bulk crystal search);
- PBE;
- non-magnetic calculations (``ISPIN = 1`` on every step);
- zero external pressure (``ExternalPressure = 0``);
- at most ``MAX_ELEMENTS`` elements, ``MAX_CELL_ATOMS`` atoms per cell and
  ``MAX_GENERATIONS`` generations (provisional v1 support limits);
- exactly one ``KresolStart`` value per step of the screening ladder, with
  steps 4 and 5 at BMD's k-point density (``incar_policy.require_spec_matches_ladder``);
- elements that have a BMD POTCAR mapping (``potcar.bmd_potcar_symbol``).

Non-magnetic is checked against the same composition screens the rest of
bmd-compute uses for automatic treatments (``docs/methodology.md``
section 2), because a search's INCARs are fixed for every candidate and
cannot switch treatments on per structure:

- an element in BMD's spin-polarisation screen
  (``SPIN_TRIGGER_ELEMENT_CLASSES``) is rejected: an ordinary BMD
  calculation of the same material would be spin-polarised, so a
  non-magnetic search would rank candidates with a different Hamiltonian
  from the one used afterwards;
- a composition for which BMD's automatic DFT+U rule
  (``dft_u_policy.evaluate_dft_u``) decides "apply" is rejected for the same
  reason;
- an element in BMD's heavy-element spin-orbit screen
  (``SOC_TRIGGER_ELEMENT_CLASSES``) is rejected: BMD evaluates the static
  energy of such a material with SOC, which a plain-PBE search and its
  plain-PBE finalist ranking cannot reproduce.

These are BMD's own composition screens, the ones its automatic treatments
use; admission adds no element list of its own.

The DFT-D3 dispersion trigger depends on a structure's bonded
connectivity, which a search does not have in advance; it is not evaluated
here (see ``docs/structure_search.md``).

Every rejection raises :class:`StructureSearchAdmissionError` with a message
that names the reason, and a ``diagnostic["code"]`` from this module.
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.calculations.registry import CalculationValidationError
from backend.structure_search.potcar import unmapped_elements
from backend.structure_search.spec import (
    FIXED_COMPOSITION_BULK,
    THEORY_PBE,
    StructureSearchSpec,
)


ADMISSION_POLICY_ID = "bmd_compute.structure_search_admission"
# Version 2 (6 Nov 2026): adds the ambient-pressure, spin-orbit, size and
# ladder checks. Version 1 was never used to run a search.
ADMISSION_POLICY_VERSION = 2

# Provisional v1 support limits (bmd-compute proposals, not lab decisions).
MAX_ELEMENTS = 4
MAX_CELL_ATOMS = 40
MAX_GENERATIONS = 200

UNSUPPORTED_CALCULATION_TYPE_CODE = "structure_search_calculation_type_not_supported"
UNSUPPORTED_THEORY_CODE = "structure_search_theory_not_supported"
SPIN_POLARIZED_NOT_SUPPORTED_CODE = "structure_search_spin_not_supported"
SPIN_SCREEN_ELEMENT_CODE = "structure_search_spin_screen_element"
DFT_U_COMPOSITION_CODE = "structure_search_dft_u_composition"
UNKNOWN_ELEMENT_CODE = "structure_search_unknown_element"
NO_POTCAR_MAPPING_CODE = "structure_search_no_potcar_mapping"
PRESSURE_NOT_SUPPORTED_CODE = "structure_search_pressure_not_supported"
SOC_SCREEN_ELEMENT_CODE = "structure_search_soc_screen_element"
TOO_MANY_ELEMENTS_CODE = "structure_search_too_many_elements"
CELL_TOO_LARGE_CODE = "structure_search_cell_too_large"
TOO_MANY_GENERATIONS_CODE = "structure_search_too_many_generations"


class StructureSearchAdmissionError(CalculationValidationError):
    """A well-formed structure search that bmd-compute v1 does not support."""

    def __init__(self, message: str, *, code: str, suggestion: str | None = None, **details):
        super().__init__(
            message,
            suggestion=suggestion,
            diagnostic={
                "code": code,
                "policy_id": ADMISSION_POLICY_ID,
                "policy_version": ADMISSION_POLICY_VERSION,
                **details,
            },
        )
        self.code = code


def require_admissible_search(spec: StructureSearchSpec) -> StructureSearchSpec:
    """Return ``spec`` unchanged if v1 supports it; otherwise raise.

    Checks run in a fixed order and the first failure is reported.
    """

    if not isinstance(spec, StructureSearchSpec):
        raise TypeError("require_admissible_search expects a StructureSearchSpec")

    if spec.calculation_type != FIXED_COMPOSITION_BULK:
        raise StructureSearchAdmissionError(
            f"USPEX calculationType {spec.calculation_type} is not supported. "
            f"bmd-compute currently runs only fixed-composition bulk searches "
            f"(calculationType {FIXED_COMPOSITION_BULK}).",
            code=UNSUPPORTED_CALCULATION_TYPE_CODE,
            calculation_type=spec.calculation_type,
        )
    if spec.theory != THEORY_PBE:
        raise StructureSearchAdmissionError(
            f"Theory '{spec.theory}' is not supported for structure searches; only PBE is.",
            code=UNSUPPORTED_THEORY_CODE,
            theory=spec.theory,
        )
    if spec.spin_polarized:
        raise StructureSearchAdmissionError(
            "Spin-polarised structure searches are not supported yet; searches run non-magnetic (ISPIN = 1).",
            code=SPIN_POLARIZED_NOT_SUPPORTED_CODE,
        )

    if spec.external_pressure_gpa != 0:
        raise StructureSearchAdmissionError(
            f"Structure searches run at ambient pressure only; this search asks for "
            f"{spec.external_pressure_gpa:g} GPa.",
            code=PRESSURE_NOT_SUPPORTED_CODE,
            external_pressure_gpa=spec.external_pressure_gpa,
        )
    if len(spec.elements) > MAX_ELEMENTS:
        raise StructureSearchAdmissionError(
            f"Structure searches support at most {MAX_ELEMENTS} elements; this one has "
            f"{len(spec.elements)}.",
            code=TOO_MANY_ELEMENTS_CODE,
            elements=list(spec.elements),
        )
    if spec.max_atoms > MAX_CELL_ATOMS:
        raise StructureSearchAdmissionError(
            f"Structure searches support cells of at most {MAX_CELL_ATOMS} atoms; this one "
            f"allows up to {spec.max_atoms}.",
            code=CELL_TOO_LARGE_CODE,
            suggestion="Lower the maximum atoms per cell.",
            max_atoms=spec.max_atoms,
        )
    if spec.num_generations > MAX_GENERATIONS:
        raise StructureSearchAdmissionError(
            f"Structure searches support at most {MAX_GENERATIONS} generations.",
            code=TOO_MANY_GENERATIONS_CODE,
            num_generations=spec.num_generations,
        )

    from backend.structure_search.incar_policy import require_spec_matches_ladder

    # One KresolStart value per ladder step; steps 4-5 at BMD's k-point density.
    require_spec_matches_ladder(spec)

    unknown = _unknown_elements(spec.elements)
    if unknown:
        raise StructureSearchAdmissionError(
            f"Unknown element(s): {', '.join(unknown)}.",
            code=UNKNOWN_ELEMENT_CODE,
            elements=unknown,
        )
    unmapped = unmapped_elements(spec.elements)
    if unmapped:
        raise StructureSearchAdmissionError(
            f"bmd-compute has no POTCAR mapping for: {', '.join(unmapped)}.",
            code=NO_POTCAR_MAPPING_CODE,
            elements=unmapped,
        )

    spin_elements = _spin_screen_elements(spec.elements)
    if spin_elements:
        raise StructureSearchAdmissionError(
            "Structure searches run non-magnetic, but BMD treats "
            f"{', '.join(spin_elements)} as spin-polarised in its other calculations. "
            "Searches containing these elements are not supported yet.",
            code=SPIN_SCREEN_ELEMENT_CODE,
            suggestion="Spin-polarised searches are planned after the non-magnetic version works.",
            elements=spin_elements,
        )

    dft_u = _dft_u_record(spec)
    if dft_u.get("decision") == "apply":
        triggering = list(dft_u.get("triggering_elements") or [])
        raise StructureSearchAdmissionError(
            "BMD applies DFT+U to this composition in its other calculations "
            f"({', '.join(triggering)} with {dft_u.get('deciding_anion')}); "
            "DFT+U structure searches are not supported yet.",
            code=DFT_U_COMPOSITION_CODE,
            elements=triggering,
            gate=dft_u.get("gate"),
        )
    soc_elements = _soc_screen_elements(spec.elements)
    if soc_elements:
        raise StructureSearchAdmissionError(
            "Structure searches are plain PBE, but BMD evaluates the energy of materials "
            f"containing {', '.join(soc_elements)} with spin-orbit coupling. "
            "Searches containing these elements are not supported yet.",
            code=SOC_SCREEN_ELEMENT_CODE,
            elements=soc_elements,
        )

    return spec


def _unknown_elements(elements) -> list[str]:
    from pymatgen.core import Element

    unknown = []
    for symbol in elements:
        try:
            Element(symbol)
        except ValueError:
            unknown.append(symbol)
    return unknown


def _spin_screen_elements(elements) -> list[str]:
    from backend.calculations.method_considerations import SPIN_TRIGGER_ELEMENT_CLASSES

    return [symbol for symbol in elements if symbol in SPIN_TRIGGER_ELEMENT_CLASSES]


def _soc_screen_elements(elements) -> list[str]:
    from backend.calculations.method_considerations import SOC_TRIGGER_ELEMENT_CLASSES

    return [symbol for symbol in elements if symbol in SOC_TRIGGER_ELEMENT_CLASSES]


def admission_scope_record() -> dict:
    """JSON-safe description of what this admission policy accepts."""

    return {
        "policy_id": ADMISSION_POLICY_ID,
        "policy_version": ADMISSION_POLICY_VERSION,
        "calculation_type": FIXED_COMPOSITION_BULK,
        "theory": THEORY_PBE,
        "spin_polarized": False,
        "external_pressure_gpa": 0.0,
        "composition_screens": [
            "bmd_compute spin-polarisation screen (method_considerations.SPIN_TRIGGER_ELEMENT_CLASSES)",
            "bmd_compute heavy-element SOC screen (method_considerations.SOC_TRIGGER_ELEMENT_CLASSES)",
            "bmd_compute.dft_u policy (dft_u_policy.evaluate_dft_u)",
        ],
        "not_screened": ["DFT-D3 dispersion (depends on a structure's bonded connectivity)"],
        "limits": {
            "max_elements": MAX_ELEMENTS,
            "max_cell_atoms": MAX_CELL_ATOMS,
            "max_generations": MAX_GENERATIONS,
            "status": "provisional",
        },
    }


def _dft_u_record(spec: StructureSearchSpec) -> dict:
    from pymatgen.core import Composition

    from backend.calculations.dft_u_policy import evaluate_dft_u

    composition = Composition(dict(zip(spec.elements, spec.counts)))
    # evaluate_dft_u reads only ``.composition`` from its argument.
    return evaluate_dft_u(SimpleNamespace(composition=composition))


__all__ = [
    "ADMISSION_POLICY_ID",
    "ADMISSION_POLICY_VERSION",
    "CELL_TOO_LARGE_CODE",
    "MAX_CELL_ATOMS",
    "MAX_ELEMENTS",
    "MAX_GENERATIONS",
    "PRESSURE_NOT_SUPPORTED_CODE",
    "SOC_SCREEN_ELEMENT_CODE",
    "TOO_MANY_ELEMENTS_CODE",
    "TOO_MANY_GENERATIONS_CODE",
    "admission_scope_record",
    "DFT_U_COMPOSITION_CODE",
    "NO_POTCAR_MAPPING_CODE",
    "SPIN_POLARIZED_NOT_SUPPORTED_CODE",
    "SPIN_SCREEN_ELEMENT_CODE",
    "StructureSearchAdmissionError",
    "UNKNOWN_ELEMENT_CODE",
    "UNSUPPORTED_CALCULATION_TYPE_CODE",
    "UNSUPPORTED_THEORY_CODE",
    "require_admissible_search",
]
