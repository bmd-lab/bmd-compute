"""Typed specification of one USPEX structure search.

A structure search starts from a composition, not from a ``pymatgen.Structure``.
:class:`StructureSearchSpec` holds everything the *user* chooses for a search;
:func:`default_search_spec` fills in every choice a formula-only request
leaves open, from the provisional defaults in this module. This module is the
single source of those defaults: the browser's New structure preview
(``prediction_plan``) builds its plan from :func:`default_search_spec` and
:func:`search_defaults_record`, never from its own copies.
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


# Upper limit of the default populationSize (manual section 4.2).
POPULATION_SIZE_DEFAULT_MAX = 60


def default_population_size(atoms_per_cell: int) -> int:
    """USPEX's documented default ``populationSize`` for a cell of ``atoms_per_cell`` atoms.

    The manual (section 4.2) gives the default as "2 x N rounded to the
    closest 10 ... The upper limit is 60". This function rounds
    ``2 * atoms_per_cell`` to the nearest multiple of 10 (halves round up)
    and keeps the result between 10 and 60. The lower limit of 10 is a
    bmd-compute choice (it only matters below 3 atoms). bmd-compute always
    writes ``populationSize`` explicitly, so this value does not depend on
    how USPEX itself rounds.
    """

    if atoms_per_cell < 1:
        raise ValueError("atoms_per_cell must be at least 1")
    rounded = int(math.floor(2 * atoms_per_cell / 10 + 0.5)) * 10
    return min(POPULATION_SIZE_DEFAULT_MAX, max(10, rounded))


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

    ``num_generations`` and ``stop_criterion`` have no default here, so the
    caller must choose them. For reference, the manual's defaults (section
    4.2) are ``numGenerations`` = 100 and, for fixed composition,
    ``stopCrit`` = the total number of atoms; they are not applied
    automatically because both decide how long a search runs.
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


# ---------------------------------------------------------------------------
# Defaults for a formula-only request (provisional; BMD Lab sign-off pending).
# ---------------------------------------------------------------------------
#
# A New structure request gives only a composition. These values fill in the
# rest. They are bmd-compute proposals (see PROVISIONAL_DEFAULTS), recorded
# with every spec built from them, and are not yet approved scientific policy.

SEARCH_DEFAULTS_POLICY_ID = "bmd_compute.structure_search_defaults"
SEARCH_DEFAULTS_POLICY_VERSION = 1
SEARCH_DEFAULTS_STATUS = "provisional"

# Ambient pressure only (the v1 scope; admission refuses anything else).
DEFAULT_EXTERNAL_PRESSURE_GPA = 0.0
# Cells of 1 .. floor(DEFAULT_MAX_CELL_ATOMS / atoms per formula unit) formula
# units (at least one formula unit).
DEFAULT_MAX_CELL_ATOMS = 20
# The manual's default is 100 generations (section 4.2); 50 bounds the cost of
# a first search. stopCrit defaults to the maximum atoms per cell, the
# manual's fixed-composition rule ("the total number of atoms").
DEFAULT_NUM_GENERATIONS = 50
# One KresolStart value per step of the five-step screening ladder
# (incar_policy.NUM_STEPS; units 2 pi / A). Steps 4 and 5 meet BMD's k-point
# density bound (incar_policy.BMD_STANDARD_MAX_KRESOL = 0.04); steps 1-3 are
# coarser screening meshes (the progression of the student's Mg2SiO4 example).
DEFAULT_KRESOL_START = (0.12, 0.10, 0.08, 0.04, 0.04)
# Explicit walltime of the search allocation (7 days). There is no automatic
# estimate for a search.
DEFAULT_SEARCH_WALLTIME = "168:00:00"
# Finalists later given BMD's adaptive geometry relaxation and a plain-PBE
# static (lab decision, 6 Nov 2026). Selection and refinement are not
# implemented yet; the number is recorded so the preview and the later
# implementation share it.
PLANNED_FINALIST_COUNT = 5

PROVISIONAL_DEFAULTS = (
    "external_pressure_gpa", "min_atoms", "max_atoms", "population_size",
    "initial_population_size", "num_generations", "stop_criterion",
    "kresol_start", "walltime",
)


def default_cell_range(formula_atoms: int) -> tuple[int, int]:
    """(min_atoms, max_atoms) for a composition of ``formula_atoms`` atoms per formula unit."""

    formula_atoms = _int(formula_atoms, "formula_atoms", minimum=1)
    whole_units = max(1, DEFAULT_MAX_CELL_ATOMS // formula_atoms)
    return formula_atoms, whole_units * formula_atoms


def default_search_spec(
    composition: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    min_atoms: Any = None,
    max_atoms: Any = None,
    num_generations: Any = None,
    walltime: Any = None,
) -> "StructureSearchSpec":
    """The spec of a formula-only request, with optional overrides.

    ``composition`` is the composition of one formula unit (``numSpecies``),
    as an ordered ``{element: count}`` mapping or element/count rows. Blank
    overrides (None or an empty string) take the default. Pressure, theory,
    magnetism, k-point resolutions and population sizes are not overridable
    here: they are fixed by the v1 scope or by the defaults above.
    """

    elements, counts = _composition_rows(composition)
    counts = _int_tuple(counts, "counts", minimum=1)
    default_min, default_max = default_cell_range(sum(counts))

    def chosen(value, default):
        return default if value is None or (isinstance(value, str) and not value.strip()) else value

    min_value = chosen(min_atoms, default_min)
    max_value = chosen(max_atoms, default_max)
    max_int = _int(max_value, "max_atoms", minimum=1)
    return StructureSearchSpec(
        elements=elements,
        counts=counts,
        min_atoms=min_value,
        max_atoms=max_int,
        num_generations=chosen(num_generations, DEFAULT_NUM_GENERATIONS),
        stop_criterion=max_int,
        kresol_start=DEFAULT_KRESOL_START,
        walltime=chosen(walltime, DEFAULT_SEARCH_WALLTIME),
        external_pressure_gpa=DEFAULT_EXTERNAL_PRESSURE_GPA,
    )


def search_defaults_record() -> dict[str, Any]:
    """JSON-safe record of the formula-only defaults, for previews and provenance."""

    return {
        "policy_id": SEARCH_DEFAULTS_POLICY_ID,
        "policy_version": SEARCH_DEFAULTS_POLICY_VERSION,
        "status": SEARCH_DEFAULTS_STATUS,
        "provisional_fields": list(PROVISIONAL_DEFAULTS),
        "external_pressure_gpa": DEFAULT_EXTERNAL_PRESSURE_GPA,
        "max_cell_atoms": DEFAULT_MAX_CELL_ATOMS,
        "num_generations": DEFAULT_NUM_GENERATIONS,
        "stop_criterion": "max_atoms",
        "population_size": f"2 x max_atoms rounded to 10, between 10 and {POPULATION_SIZE_DEFAULT_MAX}",
        "kresol_start": list(DEFAULT_KRESOL_START),
        "walltime": DEFAULT_SEARCH_WALLTIME,
        "planned_finalist_count": PLANNED_FINALIST_COUNT,
    }


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
    "DEFAULT_EXTERNAL_PRESSURE_GPA",
    "DEFAULT_KRESOL_START",
    "DEFAULT_MAX_CELL_ATOMS",
    "DEFAULT_NUM_GENERATIONS",
    "DEFAULT_SEARCH_WALLTIME",
    "PLANNED_FINALIST_COUNT",
    "PROVISIONAL_DEFAULTS",
    "SEARCH_DEFAULTS_POLICY_ID",
    "SEARCH_DEFAULTS_POLICY_VERSION",
    "SEARCH_DEFAULTS_STATUS",
    "default_cell_range",
    "default_search_spec",
    "search_defaults_record",
    "FIXED_COMPOSITION_BULK",
    "POPULATION_SIZE_DEFAULT_MAX",
    "SPEC_SCHEMA",
    "SPEC_SCHEMA_VERSION",
    "THEORY_PBE",
    "StructureSearchSpec",
    "StructureSearchSpecError",
    "default_population_size",
]
