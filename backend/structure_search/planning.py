"""Managed planning of New structure (formula-based USPEX) requests.

USPEX integration step 4A. This is the one entry point through which BMD
Compute plans a structure search, for the browser's New structure preview
(``/predict``) and the machine API (``POST /api/v1/structure-search-plans``)
alike. Both build a :class:`StructureSearchPlanRequest`, call
:func:`plan_structure_search` and present the same
:class:`StructureSearchPlanningResult`: the browser through
:meth:`~StructureSearchPlanningResult.preview`, the API through
:meth:`~StructureSearchPlanningResult.response`.

Planning itself is ``search_plan.build_search_plan_from_formula``: the frozen,
digest-addressed :class:`~backend.structure_search.search_plan.StructureSearchPlan`
built from ``spec``, ``admission``, ``incar_policy``, ``potcar``,
``step_validity``, ``runtime`` and the search resource estimator. This
module adds no scientific value of its own.

Structure prediction is not a Desired Output of the structure-based
workflow: it starts from a formula, not a structure, and has its own plan
schema and digest (``bmd_compute.structure_search_plan``), distinct from the
calculation plan digest (``compute_api.plan_digest``). Planning is read-only:
it never prepares files, submits anything, contacts POWER or writes the
API attempt ledger. Search preparation, submission, monitoring, finalist
selection and the adaptive finalist relaxation are not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.structure_search.prediction_plan import AdvancedSettings, preview_from_plan
from backend.structure_search.search_plan import (
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    STALE_PLAN_CODE,
    SearchPlanError,
    StructureSearchPlan,
    build_search_plan_from_formula,
)


RESPONSE_SCHEMA = "bmd_compute.structure_search_plan_response"
RESPONSE_SCHEMA_VERSION = 1
REQUEST_KIND = "structure_search"
OVERRIDE_FIELDS = ("min_atoms", "max_atoms", "num_generations", "walltime")
# Optional fields that, unlike OVERRIDE_FIELDS, have no default value: blank
# means "not requested", and a plan without them is exactly the plan from
# before they existed (same record, same digest).
OPTIONAL_REQUEST_FIELDS = ("random_seed", "excluded_nodes")


@dataclass(frozen=True)
class StructureSearchPlanRequest:
    """A New structure request: a formula, optional overrides, an optional expected digest.

    The overrides are the browser's Advanced settings (atoms-per-cell range,
    generations, walltime); ``None`` or blank takes the search default.
    ``expected_plan_digest``, when given, must equal the digest the request
    plans to now; otherwise the request is refused as stale.
    """

    formula: str
    min_atoms: Any = None
    max_atoms: Any = None
    num_generations: Any = None
    walltime: Any = None
    expected_plan_digest: str | None = None
    random_seed: Any = None
    excluded_nodes: Any = None

    def advanced(self) -> AdvancedSettings:
        return AdvancedSettings(
            min_atoms=self.min_atoms,
            max_atoms=self.max_atoms,
            num_generations=self.num_generations,
            walltime=self.walltime,
            random_seed=self.random_seed,
            excluded_nodes=self.excluded_nodes,
        )


@dataclass(frozen=True)
class StructureSearchPlanningResult:
    """The frozen plan of one request, and its two presentations."""

    plan: StructureSearchPlan
    advanced_overrides: bool

    @property
    def digest(self) -> str:
        return self.plan.digest

    def preview(self) -> dict[str, Any]:
        """The browser's New structure preview of this plan."""

        return preview_from_plan(self.plan, advanced_overrides=self.advanced_overrides)

    def response(self) -> dict[str, Any]:
        """The machine-facing planning response for this plan (read-only)."""

        preview = self.preview()
        record = self.plan.record
        search = record["search"]
        return {
            "schema": RESPONSE_SCHEMA,
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_kind": REQUEST_KIND,
            "desired_output": record["desired_output"],
            "plan_schema": PLAN_SCHEMA,
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "plan_digest": self.plan.digest,
            "plan_digest_scope": (
                "the scientific search plan record; excludes provenance (formula as entered, "
                "generating commit). Distinct from the calculation plan digest."
            ),
            "formula": {
                "entered": preview["entered_formula"],
                "display": preview["display_formula"],
                "formula_unit_as_entered": preview["reduced_display_formula"],
                "canonical": preview["formula"],
                "reduced_from_entered": preview["reduced_from_entered"],
                "reordered_from_entered": preview["reordered_from_entered"],
                "formula_atoms": preview["formula_atoms"],
            },
            "search": {
                "method": search["method"],
                "theory": search["theory"],
                "energy_grade": search["energy_grade"],
                "equivalent_to_bmd_adaptive_relaxation": search["equivalent_to_bmd_adaptive_relaxation"],
                "cell_sizes": search["cell_sizes"],
                "formula_units": search["formula_units"],
                "population_size": search["spec"]["population_size"],
                "num_generations": search["spec"]["num_generations"],
                "stop_criterion": search["spec"]["stop_criterion"],
                "walltime": search["spec"]["walltime"],
                **(
                    {"random_seed": search["random_seed"]}
                    if "random_seed" in search else {}
                ),
                "defaults_status": search["defaults"]["status"],
                "advanced_overrides": self.advanced_overrides,
            },
            "finalist_refinement": record["refinement"],
            "resources": {
                "status": record["resources"]["status"],
                "approved": record["resources"]["approved"],
                "note": record["resources"]["note"],
                "request": record["resources"]["request"],
            },
            "preparation_available": False,
            "submission_available": False,
            "plan": self.plan.to_document(),
        }


def plan_structure_search(request: StructureSearchPlanRequest, *, runtime_settings=None) -> StructureSearchPlanningResult:
    """Plan a New structure request (read-only).

    Raises the formula, spec or admission validation error for a request
    outside the search scope, and :class:`SearchPlanError` with code
    ``STALE_PLAN_CODE`` when ``expected_plan_digest`` differs from the plan
    the request gives now.
    """

    if not isinstance(request, StructureSearchPlanRequest):
        raise TypeError("plan_structure_search expects a StructureSearchPlanRequest")
    advanced = request.advanced()
    plan = build_search_plan_from_formula(request.formula, advanced, runtime_settings=runtime_settings)
    if request.expected_plan_digest is not None and request.expected_plan_digest != plan.digest:
        raise SearchPlanError(
            "This request no longer gives the expected search plan: BMD Compute's search "
            "policy, defaults or settings have changed, or the request differs.",
            code=STALE_PLAN_CODE,
            suggestion="Plan the search again and review the new plan.",
            expected_plan_digest=request.expected_plan_digest,
            plan_digest=plan.digest,
        )
    return StructureSearchPlanningResult(plan=plan, advanced_overrides=not advanced.is_empty())


__all__ = [
    "OPTIONAL_REQUEST_FIELDS",
    "OVERRIDE_FIELDS",
    "REQUEST_KIND",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_VERSION",
    "StructureSearchPlanRequest",
    "StructureSearchPlanningResult",
    "plan_structure_search",
]
