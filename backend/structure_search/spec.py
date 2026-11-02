"""Typed specification of one USPEX structure search.

A structure search starts from a composition, not from a ``pymatgen.Structure``.
:class:`StructureSearchSpec` holds everything the *user* chooses for a search;
everything bmd-compute fixes itself (cluster settings, the launch bridge, the
INCAR files) lives elsewhere (``input_writer``, ``incar_policy``).

This module only checks that a spec is *well formed*: types, ranges, lengths
and formats. Whether bmd-compute *supports* a well-formed spec (calculation
type, theory, magnetism, POTCAR availability) is decided separately by
``backend.structure_search.admission``, in the same way that
``backend.calculations.registry`` and ``backend.calculations.admission`` are
separated for ordinary workflows.

USPEX keyword names are given next to each field. Facts about USPEX come from
the USPEX 25.0 manual (section numbers in ``docs/structure_search.md``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import validate_walltime


SPEC_SCHEMA = "bmd_compute.structure_search_spec"
SPEC_SCHEMA_VERSION = 1

# USPEX calculationType 300: fixed-composition 3D bulk crystal search. It is
# the only type admitted in v1 (see admission.py); it is still a spec field so
# that a request for another type is rejected with a clear message instead of
# being silently rewritten.
FIXED_COMPOSITION_BULK = 300

THEORY_PBE = "pbe"

_ELEMENT_SYMBOL = re.compile(r"^[A-Z][a-z]{0,2}$")


class StructureSearchSpecError(CalculationValidationError):
    """A structure-search spec is malformed (wrong type, range or length)."""

    def __init__(self, message: str, *, code: str, suggestion: str | None = None, **details):
        super().__init__(
            message,
            suggestion=suggestion,
            diagnostic={"code": code, **details},
        )
        self.code = code


def default_population_size(atoms_per_cell: int) -> int:
    """USPEX's documented default ``populationSize`` for a cell of ``atoms_per_cell`` atoms.

    The manual gives the default as "2 x atoms per cell, rounded to 10". This
    function reads that as: round ``2 * atoms_per_cell`` to the nearest
    multiple of 10 (halves round up), and never return less than 10. bmd-compute
    always writes ``populationSize`` explicitly, so this value is a
    bmd-compute default and does not depend on how USPEX itself rounds.
    """

    if atoms_per_cell < 1:
        raise ValueError("atoms_per_cell must be at least 1")
    rounded = int(math.floor(2 * atoms_per_cell / 10 + 0.5)) * 10
    return max(10, rounded)


@dataclass(frozen=True)
class StructureSearchSpec:
    """User choices for one fixed-composition USPEX search.

    Fields and the ``INPUT.txt`` keywords they become:

    - ``elements`` -> ``atomType`` block; element symbols, in order.
    - ``counts`` -> ``numSpecies`` block; atoms of each element, same order.
    - ``external_pressure_gpa`` -> ``ExternalPressure`` (GPa). Pressure is
      set only here, never as ``PSTRESS`` in the INCARs.
    - ``min_atoms`` / ``max_atoms`` -> ``minAt`` / ``maxAt``, the
      atoms-per-cell range.
    - ``population_size`` -> ``populationSize``. ``None`` means the USPEX
      default rule (:func:`default_population_size` of ``max_atoms``).
    - ``initial_population_size`` -> ``initialPopSize``. ``None`` means equal
      to ``population_size`` (the USPEX default).
    - ``num_generations`` -> ``numGenerations``.
    - ``stop_criterion`` -> ``stopCrit``.
    - ``kresol_start`` -> ``KresolStart`` block; one k-point resolution per
      relaxation step, in units of 2 pi / Angstrom. Its length is the number
      of relaxation steps.
    - ``walltime`` -> the SLURM walltime of the search allocation (H:MM:SS).
      A search's length is adaptive, so there is no automatic estimate.
    - ``calculation_type`` -> ``calculationType``.
    - ``theory`` and ``spin_polarized`` are not USPEX keywords; they record
      the requested electronic-structure method so that admission can reject
      requests v1 does not support.

    Construction normalises types and raises :class:`StructureSearchSpecError`
    for malformed values. The resolved population sizes are available as
    :attr:`resolved_population_size` and :attr:`resolved_initial_population_size`.

    ``num_generations`` and ``stop_criterion`` have no default here: their
    USPEX defaults are not recorded in the design document, so the caller has
    to choose them.
    """

    elements: tuple[str, ...]
    counts: tuple[int, ...]
    min_atoms: int
    max_atoms: int
    num_generations: int
    stop_criterion: int
    kresol_start: tuple[float, ...]
    walltime: str
    external_pressure_gpa: float = 0.0
    population_size: int | None = None
    initial_population_size: int | None = None
    calculation_type: int = FIXED_COMPOSITION_BULK
    theory: str = THEORY_PBE
    spin_polarized: bool = False

    def __post_init__(self) -> None:
        elements = _element_tuple(self.elements)
        counts = _int_tuple(self.counts, "counts", minimum=1)
        if len(counts) != len(elements):
            raise StructureSearchSpecError(
                f"The composition lists {len(elements)} element(s) but "
                f"{len(counts)} atom count(s).",
                code="composition_length_mismatch",
                suggestion="Give exactly one atom count per element.",
            )
        min_atoms = _int(self.min_atoms, "min_atoms", minimum=1)
        max_atoms = _int(self.max_atoms, "max_atoms", minimum=1)
        if min_atoms > max_atoms:
            raise StructureSearchSpecError(
                f"The minimum number of atoms per cell ({min_atoms}) is larger "
                f"than the maximum ({max_atoms}).",
                code="atom_range_inverted",
            )
        formula_atoms = sum(counts)
        if not any(n % formula_atoms == 0 for n in range(min_atoms, max_atoms + 1)):
            raise StructureSearchSpecError(
                f"No cell size between {min_atoms} and {max_atoms} atoms is a whole "
                f"number of formula units ({formula_atoms} atoms each).",
                code="atom_range_excludes_composition",
                suggestion="Widen the atoms-per-cell range or check the composition.",
            )
        population = (
            None if self.population_size is None
            else _int(self.population_size, "population_size", minimum=1)
        )
        initial = (
            None if self.initial_population_size is None
            else _int(self.initial_population_size, "initial_population_size", minimum=1)
        )
        kresol = _float_tuple(self.kresol_start, "kresol_start")
        if not kresol:
            raise StructureSearchSpecError(
                "At least one relaxation step (one KresolStart value) is required.",
                code="no_relaxation_steps",
            )
        if any(value <= 0 for value in kresol):
            raise StructureSearchSpecError(
                "Every KresolStart value must be greater than zero.",
                code="kresol_not_positive",
            )
        pressure = _finite_float(self.external_pressure_gpa, "external_pressure_gpa")

        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "min_atoms", min_atoms)
        object.__setattr__(self, "max_atoms", max_atoms)
        object.__setattr__(self, "num_generations", _int(self.num_generations, "num_generations", minimum=1))
        object.__setattr__(self, "stop_criterion", _int(self.stop_criterion, "stop_criterion", minimum=1))
        object.__setattr__(self, "kresol_start", kresol)
        object.__setattr__(self, "walltime", validate_walltime(self.walltime))
        object.__setattr__(self, "external_pressure_gpa", pressure)
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "initial_population_size", initial)
        object.__setattr__(self, "calculation_type", _int(self.calculation_type, "calculation_type"))
        object.__setattr__(self, "theory", str(self.theory).strip().lower())
        if not isinstance(self.spin_polarized, bool):
            raise StructureSearchSpecError(
                "spin_polarized must be true or false.", code="invalid_type", field="spin_polarized"
            )

    @property
    def num_steps(self) -> int:
        """Number of relaxation steps (one ``KresolStart`` value each)."""

        return len(self.kresol_start)

    @property
    def resolved_population_size(self) -> int:
        if self.population_size is not None:
            return self.population_size
        return default_population_size(self.max_atoms)

    @property
    def resolved_initial_population_size(self) -> int:
        if self.initial_population_size is not None:
            return self.initial_population_size
        return self.resolved_population_size

    def to_record(self) -> dict[str, Any]:
        """JSON-safe record of the spec, with resolved population sizes."""

        return {
            "schema": SPEC_SCHEMA,
            "schema_version": SPEC_SCHEMA_VERSION,
            "calculation_type": self.calculation_type,
            "theory": self.theory,
            "spin_polarized": self.spin_polarized,
            "composition": [
                {"element": element, "count": count}
                for element, count in zip(self.elements, self.counts)
            ],
            "external_pressure_gpa": self.external_pressure_gpa,
            "min_atoms": self.min_atoms,
            "max_atoms": self.max_atoms,
            "population_size": self.resolved_population_size,
            "initial_population_size": self.resolved_initial_population_size,
            "num_generations": self.num_generations,
            "stop_criterion": self.stop_criterion,
            "kresol_start": list(self.kresol_start),
            "walltime": self.walltime,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StructureSearchSpec":
        """Build a spec from a plain mapping (for example a parsed form or JSON body).

        ``composition`` may be a list of ``{"element", "count"}`` rows or an
        ordered ``{element: count}`` mapping. Unknown keys are rejected so a
        misspelt field never silently falls back to a default.
        """

        allowed = {
            "composition", "external_pressure_gpa", "min_atoms", "max_atoms",
            "population_size", "initial_population_size", "num_generations",
            "stop_criterion", "kresol_start", "walltime", "calculation_type",
            "theory", "spin_polarized", "schema", "schema_version",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise StructureSearchSpecError(
                f"Unknown structure-search field(s): {', '.join(unknown)}.",
                code="unknown_fields",
                fields=unknown,
            )
        for required in ("composition", "min_atoms", "max_atoms", "num_generations",
                         "stop_criterion", "kresol_start", "walltime"):
            if data.get(required) is None:
                raise StructureSearchSpecError(
                    f"The structure-search field '{required}' is required.",
                    code="missing_field",
                    field=required,
                )
        elements, counts = _composition_rows(data["composition"])
        optional = {
            key: data[key]
            for key in ("external_pressure_gpa", "population_size", "initial_population_size",
                        "calculation_type", "theory", "spin_polarized")
            if key in data and data[key] is not None
        }
        return cls(
            elements=elements,
            counts=counts,
            min_atoms=data["min_atoms"],
            max_atoms=data["max_atoms"],
            num_generations=data["num_generations"],
            stop_criterion=data["stop_criterion"],
            kresol_start=tuple(data["kresol_start"]),
            walltime=data["walltime"],
            **optional,
        )


def _composition_rows(value) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    if isinstance(value, Mapping):
        return tuple(value.keys()), tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        elements, counts = [], []
        for row in value:
            if not isinstance(row, Mapping) or set(row) != {"element", "count"}:
                raise StructureSearchSpecError(
                    "Each composition row must have exactly 'element' and 'count'.",
                    code="invalid_composition",
                )
            elements.append(row["element"])
            counts.append(row["count"])
        return tuple(elements), tuple(counts)
    raise StructureSearchSpecError(
        "The composition must be a list of element/count rows or an element-to-count mapping.",
        code="invalid_composition",
    )


def _element_tuple(value) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StructureSearchSpecError(
            "elements must be a sequence of element symbols.", code="invalid_type", field="elements"
        )
    symbols = tuple(str(item).strip() for item in value)
    if not symbols:
        raise StructureSearchSpecError("The composition is empty.", code="empty_composition")
    bad = [symbol for symbol in symbols if not _ELEMENT_SYMBOL.fullmatch(symbol)]
    if bad:
        raise StructureSearchSpecError(
            f"Not an element symbol: {', '.join(repr(item) for item in bad)}.",
            code="invalid_element_symbol",
            symbols=bad,
        )
    duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
    if duplicates:
        raise StructureSearchSpecError(
            f"Element(s) listed more than once: {', '.join(duplicates)}.",
            code="duplicate_element",
            symbols=duplicates,
        )
    return symbols


def _int(value, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)) and not _is_integral_float(value):
        raise StructureSearchSpecError(
            f"{label} must be a whole number.", code="invalid_type", field=label
        )
    try:
        number = int(str(value).strip()) if isinstance(value, str) else int(value)
    except ValueError as exc:
        raise StructureSearchSpecError(
            f"{label} must be a whole number.", code="invalid_type", field=label
        ) from exc
    if minimum is not None and number < minimum:
        raise StructureSearchSpecError(
            f"{label} must be at least {minimum}.", code="out_of_range", field=label
        )
    return number


def _is_integral_float(value) -> bool:
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


def _int_tuple(value, label: str, *, minimum: int | None = None) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StructureSearchSpecError(
            f"{label} must be a sequence of whole numbers.", code="invalid_type", field=label
        )
    return tuple(_int(item, label, minimum=minimum) for item in value)


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool):
        raise StructureSearchSpecError(f"{label} must be a number.", code="invalid_type", field=label)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StructureSearchSpecError(
            f"{label} must be a number.", code="invalid_type", field=label
        ) from exc
    if not math.isfinite(number):
        raise StructureSearchSpecError(f"{label} must be finite.", code="out_of_range", field=label)
    return number


def _float_tuple(value, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StructureSearchSpecError(
            f"{label} must be a sequence of numbers.", code="invalid_type", field=label
        )
    return tuple(_finite_float(item, label) for item in value)


__all__ = [
    "FIXED_COMPOSITION_BULK",
    "SPEC_SCHEMA",
    "SPEC_SCHEMA_VERSION",
    "THEORY_PBE",
    "StructureSearchSpec",
    "StructureSearchSpecError",
    "default_population_size",
]
