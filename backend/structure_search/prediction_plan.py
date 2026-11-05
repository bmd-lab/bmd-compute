"""Browser preview of a "New structure" request: formula in, search plan out.

The front page's **New structure** option asks only for a chemical formula.
This module turns that formula into the plan BMD Compute would run, and is
used only to *show* that plan; it submits nothing (search submission is
#68/#70 and is not wired yet).

What a New structure request returns: one ground-state structure, relaxed to
the same accuracy as the **Relaxed structure** Desired Output. The plan is

1. a USPEX evolutionary search (fixed composition, PBE, non-magnetic), whose
   relaxations use the loose-to-tight INCAR ladder of
   ``docs/structure_search.md`` §6, and then
2. the BMD Standard Geometry Relaxation of the best distinct candidates,
   exactly the stages of the Relaxed structure Desired Output. The candidate
   with the lowest final PBE energy per atom from those relaxations is the
   result. USPEX's own (search-grade) energies only choose the candidates.

Every default below that is marked *proposed* still needs BMD Lab sign-off
(see ``PROPOSED_DEFAULTS_NOTE``). They live here, in one place, so that the
later search spec (``backend.structure_search.spec`` on the
``structure-search-spec`` branch) can take its values from them.

Standard library and pymatgen's periodic table only; nothing here touches
POWER.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import (
    structure_search_resources,
    validate_walltime,
    walltime_seconds,
)

PLAN_SCHEMA = "bmd_compute.structure_prediction_preview"
PLAN_SCHEMA_VERSION = 1

NEW_STRUCTURE_OUTPUT = "new_structure"
GROUND_STATE_REFINEMENT_OUTPUT = "relaxed_structure"

# --- proposed defaults (BMD Lab sign-off pending) -------------------------------------------

PROPOSED_DEFAULTS_NOTE = (
    "Proposed defaults, pending BMD Lab sign-off: pressure, cell-size range, "
    "generations, stopping rule, walltime and the number of candidates refined."
)
DEFAULT_PRESSURE_GPA = 0.0
# Cells of 1 .. floor(DEFAULT_MAX_CELL_ATOMS / atoms per formula unit) formula units.
DEFAULT_MAX_CELL_ATOMS = 20
DEFAULT_NUM_GENERATIONS = 50
DEFAULT_SEARCH_WALLTIME = "168:00:00"  # 7 days
# Distinct best USPEX candidates given the BMD Standard Geometry Relaxation.
DEFAULT_REFINED_CANDIDATES = 5
# USPEX 25.0 manual §4.2: populationSize defaults to 2 x N rounded to the
# closest 10, at most 60. bmd-compute writes it explicitly.
POPULATION_SIZE_MAX = 60

# --- hard limits (what this entry point accepts at all) -------------------------------------

MAX_ELEMENTS = 4
MAX_FORMULA_ATOMS = 20
MAX_CELL_ATOMS = 40
MAX_FORMULA_LENGTH = 64
MAX_PRESSURE_GPA = 500.0
MAX_GENERATIONS = 200

_TOKEN = re.compile(r"([A-Z][a-z]?)|(\()|(\))|(\d+)")


class FormulaError(CalculationValidationError):
    """The formula or an advanced setting cannot be planned."""

    def __init__(self, message: str, *, code: str, suggestion: str | None = None):
        super().__init__(message, suggestion=suggestion, diagnostic={"code": code})
        self.code = code


def _is_element(symbol: str) -> bool:
    from pymatgen.core.periodic_table import Element

    if symbol in {"D", "T"}:  # isotopes pymatgen accepts as symbols
        return False
    return Element.is_valid_symbol(symbol)


def parse_formula(text: str) -> dict[str, int]:
    """Element -> atoms per formula unit, in order of first appearance.

    Accepts element symbols with optional whole-number counts and one or more
    levels of parentheses (``Mg2SiO4``, ``Ca(OH)2``). Rejects anything else:
    fractional or zero counts, charges, hydrates, lower-case symbols,
    unknown symbols and spaces inside the formula.
    """

    formula = (text or "").strip()
    if not formula:
        raise FormulaError("Enter a chemical formula.", code="empty_formula",
                           suggestion="For example Si, GaAs or Mg2SiO4.")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise FormulaError("The formula is too long.", code="formula_too_long")
    position = 0
    stack: list[dict[str, int]] = [{}]
    last: dict[str, int] | None = None  # what a following count multiplies
    last_symbol: str | None = None
    while position < len(formula):
        match = _TOKEN.match(formula, position)
        if not match:
            raise FormulaError(
                f"{formula!r} is not a formula BMD Compute can read (problem at "
                f"character {position + 1}, {formula[position]!r}).",
                code="unreadable_formula",
                suggestion="Use element symbols with whole-number counts, for example Mg2SiO4 or Ca(OH)2.",
            )
        symbol, opening, closing, number = match.groups()
        position = match.end()
        if symbol:
            if not _is_element(symbol):
                raise FormulaError(f"{symbol!r} is not an element symbol.", code="unknown_element",
                                   suggestion="Check the capitalisation, for example Co (cobalt) not CO.")
            stack[-1][symbol] = stack[-1].get(symbol, 0) + 1
            last, last_symbol = None, symbol
        elif opening:
            stack.append({})
            last, last_symbol = None, None
        elif closing:
            if len(stack) == 1:
                raise FormulaError("The formula has an unmatched ')'.", code="unbalanced_parentheses")
            group = stack.pop()
            if not group:
                raise FormulaError("The formula has an empty '()'.", code="unreadable_formula")
            for element, count in group.items():
                stack[-1][element] = stack[-1].get(element, 0) + count
            last, last_symbol = group, None
        else:
            value = int(number)
            if number.startswith("0") or value < 1:
                raise FormulaError("Counts must be whole numbers of at least 1.", code="invalid_count")
            if last_symbol is not None:
                stack[-1][last_symbol] += value - 1
                last_symbol = None
            elif last is not None:
                for element, count in last.items():
                    stack[-1][element] += count * (value - 1)
                last = None
            else:
                raise FormulaError("A count must follow an element or ')'.", code="unreadable_formula")
    if len(stack) != 1:
        raise FormulaError("The formula has an unmatched '('.", code="unbalanced_parentheses")
    composition = stack[0]
    if len(composition) > MAX_ELEMENTS:
        raise FormulaError(
            f"Structure prediction supports at most {MAX_ELEMENTS} elements; "
            f"{formula} has {len(composition)}.",
            code="too_many_elements",
        )
    return composition


def reduce_composition(composition: Mapping[str, int]) -> dict[str, int]:
    divisor = 0
    for count in composition.values():
        divisor = math.gcd(divisor, count)
    return {element: count // divisor for element, count in composition.items()}


def formula_text(composition: Mapping[str, int]) -> str:
    return "".join(f"{e}{n if n != 1 else ''}" for e, n in composition.items())


def walltime_display(walltime: str) -> str:
    """``168:00:00`` -> ``7 days``; ``36:30:00`` -> ``36 h 30 min``."""

    total = walltime_seconds(walltime)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours >= 48 and hours % 24 == 0 and not minutes and not seconds:
        return f"{hours // 24} days"
    words = [f"{hours} h"] if hours else []
    if minutes:
        words.append(f"{minutes} min")
    if seconds:
        words.append(f"{seconds} s")
    return " ".join(words) or "0 min"


def default_population_size(max_atoms: int) -> int:
    rounded = int(math.floor(2 * max_atoms / 10 + 0.5)) * 10
    return min(POPULATION_SIZE_MAX, max(10, rounded))


def _optional_int(value: Any, name: str, *, minimum: int, maximum: int) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        raise FormulaError(f"{name} must be a whole number.", code="invalid_number") from None
    if not minimum <= number <= maximum:
        raise FormulaError(f"{name} must be between {minimum} and {maximum}.", code="out_of_range")
    return number


def _optional_pressure(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        raise FormulaError("Pressure must be a number in GPa.", code="invalid_number") from None
    if not math.isfinite(number) or not 0 <= number <= MAX_PRESSURE_GPA:
        raise FormulaError(f"Pressure must be between 0 and {MAX_PRESSURE_GPA:g} GPa.", code="out_of_range")
    return number


@dataclass(frozen=True)
class AdvancedSettings:
    """Optional overrides from the collapsed Advanced panel (blank = default)."""

    pressure_gpa: Any = None
    min_atoms: Any = None
    max_atoms: Any = None
    num_generations: Any = None
    walltime: Any = None

    def is_empty(self) -> bool:
        return all(str(v or "").strip() == "" for v in self.__dict__.values())


def build_prediction_plan(formula: str, advanced: AdvancedSettings | None = None) -> dict[str, Any]:
    """The plan a New structure request for ``formula`` would run (preview only)."""

    advanced = advanced or AdvancedSettings()
    entered = parse_formula(formula)
    composition = reduce_composition(entered)
    formula_atoms = sum(composition.values())
    if formula_atoms > MAX_FORMULA_ATOMS:
        raise FormulaError(
            f"{formula_text(composition)} has {formula_atoms} atoms per formula unit; "
            f"structure prediction supports at most {MAX_FORMULA_ATOMS}.",
            code="formula_too_large",
        )

    default_max = max(formula_atoms, (DEFAULT_MAX_CELL_ATOMS // formula_atoms) * formula_atoms)
    min_atoms = _optional_int(advanced.min_atoms, "Minimum atoms per cell", minimum=1, maximum=MAX_CELL_ATOMS)
    max_atoms = _optional_int(advanced.max_atoms, "Maximum atoms per cell", minimum=1, maximum=MAX_CELL_ATOMS)
    min_atoms = formula_atoms if min_atoms is None else min_atoms
    max_atoms = default_max if max_atoms is None else max_atoms
    if min_atoms > max_atoms:
        raise FormulaError("The minimum atoms per cell is larger than the maximum.", code="atom_range_inverted")
    cell_sizes = [n for n in range(min_atoms, max_atoms + 1) if n % formula_atoms == 0]
    if not cell_sizes:
        raise FormulaError(
            f"No cell between {min_atoms} and {max_atoms} atoms holds a whole number of "
            f"{formula_text(composition)} formula units ({formula_atoms} atoms each).",
            code="atom_range_excludes_composition",
            suggestion="Widen the atoms-per-cell range.",
        )

    pressure = _optional_pressure(advanced.pressure_gpa)
    pressure = DEFAULT_PRESSURE_GPA if pressure is None else pressure
    generations = _optional_int(advanced.num_generations, "Generations", minimum=1, maximum=MAX_GENERATIONS)
    generations = DEFAULT_NUM_GENERATIONS if generations is None else generations
    walltime_text = str(advanced.walltime or "").strip() or DEFAULT_SEARCH_WALLTIME
    walltime = validate_walltime(walltime_text)
    resources = structure_search_resources(walltime)

    population = default_population_size(max_atoms)
    return {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "desired_output": NEW_STRUCTURE_OUTPUT,
        "entered_formula": formula.strip(),
        "formula": formula_text(composition),
        "composition": [{"element": e, "count": n} for e, n in composition.items()],
        "formula_atoms": formula_atoms,
        "reduced_from_entered": composition != entered,
        "defaults_note": PROPOSED_DEFAULTS_NOTE,
        "advanced_overrides": not advanced.is_empty(),
        "search": {
            "method": "USPEX 25.0 evolutionary search, fixed composition (calculationType 300)",
            "theory": "PBE, non-magnetic",
            "pressure_gpa": pressure,
            "min_atoms": min_atoms,
            "max_atoms": max_atoms,
            "cell_sizes": cell_sizes,
            "formula_units": [n // formula_atoms for n in cell_sizes],
            "population_size": population,
            "initial_population_size": population,
            "num_generations": generations,
            "stop_criterion": max_atoms,
            "relaxation_ladder": (
                "Loose-to-tight INCAR ladder; the last step uses BMD standards "
                "(docs/structure_search.md §6, sign-off pending, #65)."
            ),
            "walltime": walltime,
            "walltime_display": walltime_display(walltime),
            "resources": resources.to_dict(),
        },
        "refinement": {
            "desired_output": GROUND_STATE_REFINEMENT_OUTPUT,
            "candidates": DEFAULT_REFINED_CANDIDATES,
            "method": "BMD Standard Geometry Relaxation (two PBE passes), as for Relaxed structure",
            "selection": "lowest final PBE energy per atom",
            "resources": "automatic, as for any Relaxed structure calculation",
        },
        "submission_available": False,
    }


__all__ = [
    "AdvancedSettings",
    "DEFAULT_MAX_CELL_ATOMS",
    "DEFAULT_NUM_GENERATIONS",
    "DEFAULT_PRESSURE_GPA",
    "DEFAULT_REFINED_CANDIDATES",
    "DEFAULT_SEARCH_WALLTIME",
    "FormulaError",
    "GROUND_STATE_REFINEMENT_OUTPUT",
    "NEW_STRUCTURE_OUTPUT",
    "PROPOSED_DEFAULTS_NOTE",
    "build_prediction_plan",
    "default_population_size",
    "formula_text",
    "parse_formula",
    "reduce_composition",
]
