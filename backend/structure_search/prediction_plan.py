"""Browser preview of a "New structure" request: formula in, search plan out.

The front page's **New structure** option asks only for a chemical formula.
This module parses the formula and shows the plan BMD Compute would run. It
submits nothing (search submission is not implemented yet).

Every default, limit and policy shown here comes from the authoritative
search specification, never from this module:

* defaults: :func:`backend.structure_search.spec.default_search_spec` and
  :func:`~backend.structure_search.spec.search_defaults_record`;
* scope and limits: :func:`backend.structure_search.admission.require_admissible_search`;
* the screening ladder: ``backend.structure_search.incar_policy``;
* search resources: ``backend.calculations.resources.structure_search_resources``.

This module only owns how a formula is *read* (:func:`parse_formula`).

What the plan shows (lab decision, 6 Nov 2026):

1. a USPEX search whose candidates are screened by a five-step relaxation
   ladder and ranked by a *search-grade* single-point energy, and
2. (planned, not yet available) the five lowest-energy structurally distinct
   candidates refined with BMD's adaptive geometry relaxation and ranked by
   plain-PBE static energies. Search-grade energies are never reported as
   BMD final energies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import structure_search_resources, walltime_seconds

PLAN_SCHEMA = "bmd_compute.structure_prediction_preview"
PLAN_SCHEMA_VERSION = 2

NEW_STRUCTURE_OUTPUT = "new_structure"

# How long a typed formula may be (input handling only, not a scientific limit).
MAX_FORMULA_LENGTH = 64

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
    return stack[0]


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


def reduce_composition(composition: Mapping[str, int]) -> dict[str, int]:
    divisor = 0
    for count in composition.values():
        divisor = math.gcd(divisor, count)
    return {element: count // divisor for element, count in composition.items()}


def formula_text(composition: Mapping[str, int]) -> str:
    return "".join(f"{e}{n if n != 1 else ''}" for e, n in composition.items())


@dataclass(frozen=True)
class AdvancedSettings:
    """Optional overrides from the collapsed Advanced panel (blank = default).

    Pressure is not offered: searches run at ambient pressure only.
    """

    min_atoms: Any = None
    max_atoms: Any = None
    num_generations: Any = None
    walltime: Any = None

    def is_empty(self) -> bool:
        return all(str(v or "").strip() == "" for v in self.__dict__.values())


def preview_defaults() -> dict[str, Any]:
    """The defaults the Advanced panel shows as placeholders (from the spec)."""

    from backend.structure_search.spec import search_defaults_record

    record = search_defaults_record()
    return {**record, "walltime_display": walltime_display(record["walltime"])}


def build_prediction_plan(formula: str, advanced: AdvancedSettings | None = None) -> dict[str, Any]:
    """The plan a New structure request for ``formula`` would run (preview only).

    Raises :class:`FormulaError` for an unreadable formula and the spec's or
    admission's validation error for a request outside the search scope.
    """

    from backend.structure_search import admission, incar_policy, spec as spec_module

    advanced = advanced or AdvancedSettings()
    entered = parse_formula(formula)
    composition = reduce_composition(entered)
    if len(composition) > admission.MAX_ELEMENTS:
        # Checked before building a spec so the message names the formula.
        raise FormulaError(
            f"Structure prediction supports at most {admission.MAX_ELEMENTS} elements; "
            f"{formula_text(composition)} has {len(composition)}.",
            code=admission.TOO_MANY_ELEMENTS_CODE,
        )
    spec = spec_module.default_search_spec(
        composition,
        min_atoms=advanced.min_atoms,
        max_atoms=advanced.max_atoms,
        num_generations=advanced.num_generations,
        walltime=advanced.walltime,
    )
    admission.require_admissible_search(spec)

    cutoffs = incar_policy.ladder_cutoffs(incar_policy.search_cutoff_record(spec.elements))
    step_cutoff = ("early", "early", "relaxation", "relaxation", "ranking")
    ladder = [
        {
            "step": step,
            "role": incar_policy.STEP_ROLES[step - 1],
            "source": incar_policy.STEP_SOURCES[step - 1],
            "encut_ev": cutoffs[step_cutoff[step - 1]],
            "kresol": spec.kresol_start[step - 1],
            "validity": dict(incar_policy.STEP_VALIDITY[step - 1]),
        }
        for step in range(1, incar_policy.NUM_STEPS + 1)
    ]
    formula_atoms = sum(spec.counts)
    cell_sizes = [n for n in range(spec.min_atoms, spec.max_atoms + 1) if n % formula_atoms == 0]
    record = spec.to_record()
    resources = structure_search_resources(spec.walltime)
    return {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "desired_output": NEW_STRUCTURE_OUTPUT,
        "entered_formula": formula.strip(),
        "formula": formula_text(composition),
        "formula_atoms": formula_atoms,
        "reduced_from_entered": composition != entered,
        "advanced_overrides": not advanced.is_empty(),
        "spec": record,
        "defaults": spec_module.search_defaults_record(),
        "scope": admission.admission_scope_record(),
        "search": {
            "method": "USPEX 25.0 evolutionary search, fixed composition (calculationType 300)",
            "theory": "plain PBE, non-magnetic",
            "pressure_gpa": spec.external_pressure_gpa,
            "min_atoms": spec.min_atoms,
            "max_atoms": spec.max_atoms,
            "cell_sizes": cell_sizes,
            "formula_units": [n // formula_atoms for n in cell_sizes],
            "population_size": spec.resolved_population_size,
            "initial_population_size": spec.resolved_initial_population_size,
            "num_generations": spec.num_generations,
            "stop_criterion": spec.stop_criterion,
            "energy_grade": incar_policy.SEARCH_ENERGY_GRADE,
            "equivalent_to_bmd_adaptive_relaxation": incar_policy.EQUIVALENT_TO_BMD_ADAPTIVE_RELAXATION,
            "ladder": ladder,
            "cutoffs_ev": cutoffs,
            "validity_enforced": incar_policy.STEP_VALIDITY_ENFORCED,
            "walltime": spec.walltime,
            "walltime_display": walltime_display(spec.walltime),
            "resources": resources.to_dict(),
        },
        "refinement": {
            "status": "planned, not yet available",
            "candidates": spec_module.PLANNED_FINALIST_COUNT,
            "selection": "lowest search-grade energy per atom, structurally distinct",
            "method": "BMD adaptive geometry relaxation, then a plain-PBE static energy",
            "protocols": ["bmd_compute.geometry_relaxation/1", "bmd_compute.geometry_convergence/1"],
            "ranking": "final plain-PBE static energy per atom",
        },
        "submission_available": False,
    }


__all__ = [
    "AdvancedSettings",
    "FormulaError",
    "MAX_FORMULA_LENGTH",
    "NEW_STRUCTURE_OUTPUT",
    "build_prediction_plan",
    "formula_text",
    "parse_formula",
    "preview_defaults",
    "reduce_composition",
    "walltime_display",
]
