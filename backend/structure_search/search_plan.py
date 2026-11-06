"""Frozen, digest-addressed structure-search plans and local input preparation.

USPEX integration step 3. A :class:`StructureSearchPlan` is the complete,
immutable statement of one fixed-composition USPEX search: what is searched,
with which screening and ranking methodology, POTCARs, k-point resolutions,
candidate-validity rules, runtime and launch settings, and a *provisional*
resource estimate. Its canonical digest covers all of it, in the same spirit
as the calculation plan digest (``compute_api.plan_digest``).

Nothing here is a new source of methodology. Every value is taken from the
module that owns it:

* request and defaults: :mod:`backend.structure_search.spec`
  (``default_search_spec``, ``search_defaults_record``) and the formula
  reader in :mod:`backend.structure_search.prediction_plan`;
* scope: :func:`backend.structure_search.admission.require_admissible_search`;
* the five-step screening and ranking ladder, cutoffs and k-point bounds:
  :mod:`backend.structure_search.incar_policy`;
* POTCAR choice and identity: :mod:`backend.structure_search.potcar` and the
  versioned ENMAX metadata behind ``incar_policy.search_cutoff_record``;
* candidate validity: :mod:`backend.structure_search.step_validity`;
* USPEX program and launch layout: :mod:`backend.structure_search.runtime`;
* ``INPUT.txt``: :mod:`backend.structure_search.input_writer`;
* the exclusive-node request: ``backend.calculations.resources.structure_search_resources``.

The only value defined here is the provisional ``sleepSeconds`` of
``INPUT.txt`` (:data:`SEARCH_SLEEP_SECONDS`), which the input writer leaves
to the execution policy and which no earlier module set.

Local preparation (:func:`prepare_search_directory`) writes a new directory::

    INPUT.txt                 USPEX input (input_writer)
    Specific/INCAR_1 .. 5     the frozen ladder INCARs (incar_policy)
    POTCAR.spec               POTCAR symbols, one per element, in INPUT.txt order
    search_plan.json          the frozen plan, its digest and its provenance
    prepared_inputs.json      plan digest, preparation provenance (bridge path,
                              generating commit) and file SHA-256

It never reads, copies or writes a POTCAR file: ``Specific/POTCAR_<El>`` are
copied from the POWER POTCAR library at execution, and their expected
identities (TITEL, ENMAX, dataset SHA-256) are in the plan. Preparation does
not contact POWER, start USPEX or VASP, or submit anything; submission,
monitoring, finalist selection and the adaptive finalist relaxation are not
implemented yet.

Equivalent requests, one plan. A formula is reduced to its smallest whole
formula unit (as the preview always did) and its elements are put in a
canonical order (:data:`ELEMENT_ORDER_RULE`: increasing Pauling
electronegativity, ties and elements without one by symbol, the order of
pymatgen's ``Composition.formula``). So ``SiO2`` and ``O2Si`` give the same
search, the same plan record and the same digest. Counts, the Advanced
overrides (cell-size range, generations, walltime) and the default cell range
do not depend on the order, so reordering never changes which cells are
searched. What was actually entered is kept in the plan's ``provenance``.

What the digest covers. Every field of the scientific plan record (see
:func:`plan_digest`): the canonical search composition and Advanced
overrides, the defaults, admission policy, the ladder (INCAR settings and
exact INCAR text), cutoff rule, ENMAX metadata, POTCAR identities, k-point
resolutions, validity policy, USPEX program and launch layout, ``INPUT.txt``
execution settings, the launch-bridge implementation (SHA-256) and the
resource estimate. Not covered: the plan's ``provenance`` (the formula or
spec as entered, and the generating Git commit), where the prepared directory
is written, and the absolute path of the launch bridge on POWER (a
preparation binding, recorded in ``prepared_inputs.json`` and checked at
execution by :func:`verify_execution_binding`).

Where searches will run. The USPEX container sees only ``/bmd``
(``bridge.master.DEFAULT_BIND``), so production searches need a dedicated,
writable search workspace under ``/bmd``, separate from the validation area
(``/bmd/bmdguest/validation``) and from ordinary calculations in
``/bmd-db/guest``. It does not exist yet and nothing here creates it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from backend.calculations.registry import CalculationValidationError


PLAN_SCHEMA = "bmd_compute.structure_search_plan"
PLAN_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA = "bmd_compute.structure_search_plan_provenance"
PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_FIELD = "provenance"
PREPARED_SCHEMA = "bmd_compute.structure_search_prepared_inputs"
PREPARED_SCHEMA_VERSION = 1
DIGEST_PREFIX = "sha256:"
DIGEST_FIELD = "plan_digest"

NEW_STRUCTURE_OUTPUT = "new_structure"

# Execution policy for INPUT.txt (#67). input_writer leaves sleepSeconds to
# the execution policy; no search has set it yet. 30 s is the value of the
# input-writer golden files. PROVISIONAL: the "value : sleepSeconds" line
# form has not been confirmed on a real USPEX 25 run (the Si8 pilot and the
# validity probe did not set it).
SEARCH_SLEEP_SECONDS = 30
SEARCH_SLEEP_SECONDS_STATUS = (
    "provisional: compatibility with real USPEX 25 (the 'value : sleepSeconds' line) "
    "not yet independently confirmed"
)

ELEMENT_ORDER_RULE = (
    "increasing Pauling electronegativity (pymatgen Element.X); ties, and elements "
    "without a value, by element symbol, after those with one"
)

PLAN_FILE = "search_plan.json"
PREPARED_FILE = "prepared_inputs.json"
POTCAR_SPEC_FILE = "POTCAR.spec"
SPECIFIC_DIR = "Specific"

PLAN_DIGEST_MISMATCH_CODE = "structure_search_plan_digest_mismatch"
PLAN_SCHEMA_CODE = "structure_search_plan_schema_not_supported"
STALE_PLAN_CODE = "structure_search_plan_stale"
INVALID_PLAN_CODE = "structure_search_plan_invalid"
UNSAFE_PATH_CODE = "structure_search_unsafe_path"
DESTINATION_EXISTS_CODE = "structure_search_destination_exists"
UNEXPECTED_FILE_CODE = "structure_search_unexpected_file"
PREPARED_MISMATCH_CODE = "structure_search_prepared_input_mismatch"
INVALID_BINDING_CODE = "structure_search_invalid_binding"
BINDING_MISMATCH_CODE = "structure_search_binding_mismatch"
BRIDGE_MISMATCH_CODE = "structure_search_bridge_mismatch"

_POTCAR_SYMBOL = re.compile(r"^[A-Z][a-z]?(_[A-Za-z0-9]+)*$")


class SearchPlanError(CalculationValidationError):
    """A search plan or a prepared search directory cannot be used."""

    def __init__(self, message: str, *, code: str, suggestion: str | None = None, **details):
        super().__init__(message, suggestion=suggestion, diagnostic={"code": code, **details})
        self.code = code


# -- canonical form and digest --------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite float in a search plan")
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    raise TypeError(f"not JSON-safe in a search plan: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def plan_digest(record: Mapping[str, Any]) -> str:
    """``sha256:<hex>`` of the canonical JSON of a plan record (without its digest)."""

    if DIGEST_FIELD in record:
        raise ValueError("plan_digest() takes the record without its digest")
    return DIGEST_PREFIX + hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructureSearchPlan:
    """An immutable search plan: its canonical record, digest and provenance.

    The record is held as canonical JSON text, so it cannot be changed after
    construction; :attr:`record` returns a fresh copy. ``provenance_json``
    (the request as entered and the generating Git commit) is kept beside the
    record and is not covered by the digest.
    """

    canonical: str
    digest: str
    provenance_json: str = "{}"

    def __post_init__(self) -> None:
        expected = DIGEST_PREFIX + hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()
        if self.digest != expected:
            raise SearchPlanError(
                "The search plan's digest does not match its content.",
                code=PLAN_DIGEST_MISMATCH_CODE,
                expected=expected,
                found=self.digest,
            )

    @classmethod
    def from_record(cls, record: Mapping[str, Any], provenance: Mapping[str, Any] | None = None) -> "StructureSearchPlan":
        canonical = canonical_json(record)
        return cls(
            canonical=canonical,
            digest=plan_digest(json.loads(canonical)),
            provenance_json=canonical_json(provenance or {}),
        )

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self.canonical)

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self.provenance_json)

    def to_document(self) -> dict[str, Any]:
        """The plan as stored: its record, ``plan_digest`` and ``provenance``."""

        return {**self.record, DIGEST_FIELD: self.digest, PROVENANCE_FIELD: self.provenance}

    def to_json(self) -> str:
        """``search_plan.json`` text (sorted keys, one-space indent, final newline)."""

        return json.dumps(self.to_document(), indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def load_plan(document: Mapping[str, Any]) -> StructureSearchPlan:
    """Rebuild a plan from its stored document; refuse a wrong schema or digest.

    Any change to the record after it was frozen, including an edited policy
    record, gives a digest mismatch.
    """

    if not isinstance(document, Mapping):
        raise SearchPlanError("A search plan must be a JSON object.", code=INVALID_PLAN_CODE)
    if document.get("schema") != PLAN_SCHEMA or document.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise SearchPlanError(
            f"Not a {PLAN_SCHEMA} v{PLAN_SCHEMA_VERSION} plan "
            f"({document.get('schema')!r} v{document.get('schema_version')!r}).",
            code=PLAN_SCHEMA_CODE,
        )
    claimed = document.get(DIGEST_FIELD)
    provenance = document.get(PROVENANCE_FIELD) or {}
    if not isinstance(provenance, Mapping):
        raise SearchPlanError("The search plan's provenance must be an object.", code=INVALID_PLAN_CODE)
    record = {key: value for key, value in document.items() if key not in (DIGEST_FIELD, PROVENANCE_FIELD)}
    try:
        actual = plan_digest(record)
    except TypeError as exc:
        raise SearchPlanError(f"The search plan is not valid JSON data: {exc}", code=INVALID_PLAN_CODE) from exc
    if claimed != actual:
        raise SearchPlanError(
            "The search plan's digest does not match its content; it has been changed "
            "since it was frozen.",
            code=PLAN_DIGEST_MISMATCH_CODE,
            expected=actual,
            found=claimed,
        )
    plan = StructureSearchPlan.from_record(record, provenance)
    _require_provenance_matches(plan)
    return plan


def load_plan_text(text: str) -> StructureSearchPlan:
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise SearchPlanError(f"The search plan is not valid JSON: {exc}", code=INVALID_PLAN_CODE) from exc
    return load_plan(document)


# -- building a plan ------------------------------------------------------------------


def _runtime_settings(runtime_settings=None):
    from backend.structure_search.runtime import SearchRuntimeSettings

    if runtime_settings is None:
        return SearchRuntimeSettings.from_config()
    if not isinstance(runtime_settings, SearchRuntimeSettings):
        raise TypeError("runtime_settings must be a SearchRuntimeSettings")
    return runtime_settings


def _bridge_script_sha256() -> str:
    from backend.structure_search.bridge.master import bridge_script_path

    return hashlib.sha256(bridge_script_path().read_bytes()).hexdigest()


def _composition_rows(elements, counts) -> list[dict[str, Any]]:
    return [{"element": element, "count": int(count)} for element, count in zip(elements, counts)]


def _element_order_key(symbol: str):
    import warnings

    from pymatgen.core import Element

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # noble gases have no electronegativity
        value = Element(symbol).X
    known = isinstance(value, (int, float)) and math.isfinite(value)
    return (0 if known else 1, float(value) if known else 0.0, symbol)


def canonical_composition(composition: Mapping[str, int]) -> dict[str, int]:
    """``composition`` with its elements in :data:`ELEMENT_ORDER_RULE` order (counts unchanged)."""

    return {element: int(composition[element]) for element in sorted(composition, key=_element_order_key)}


def canonical_spec(spec):
    """``spec`` with elements (and their counts) in canonical order; nothing else changes."""

    from dataclasses import replace

    order = sorted(range(len(spec.elements)), key=lambda index: _element_order_key(spec.elements[index]))
    return replace(
        spec,
        elements=tuple(spec.elements[index] for index in order),
        counts=tuple(spec.counts[index] for index in order),
    )


def _generator_provenance(generator: Mapping[str, Any] | None) -> dict[str, Any]:
    if generator is not None:
        return dict(generator)
    from backend.calculations.capabilities import git_provenance

    return git_provenance()


def build_search_plan_from_formula(
    formula: str,
    advanced=None,
    *,
    runtime_settings=None,
    generator: Mapping[str, Any] | None = None,
) -> StructureSearchPlan:
    """The plan of a New structure request: a formula and optional Advanced overrides.

    The formula is read and reduced as the browser preview reads it
    (``prediction_plan.parse_formula``); defaults come from
    ``spec.default_search_spec``. Raises the formula, spec or admission error
    for a request outside the search scope.
    """

    from backend.structure_search import admission
    from backend.structure_search.prediction_plan import (
        AdvancedSettings,
        FormulaError,
        formula_text,
        parse_formula,
        reduce_composition,
    )
    from backend.structure_search.spec import default_search_spec

    advanced = advanced or AdvancedSettings()
    if not isinstance(advanced, AdvancedSettings):
        raise TypeError("advanced must be an AdvancedSettings")
    entered = parse_formula(formula)
    reduced = reduce_composition(entered)
    composition = canonical_composition(reduced)
    if len(composition) > admission.MAX_ELEMENTS:
        # Checked before building a spec so the message names the formula.
        raise FormulaError(
            f"Structure prediction supports at most {admission.MAX_ELEMENTS} elements; "
            f"{formula_text(composition)} has {len(composition)}.",
            code=admission.TOO_MANY_ELEMENTS_CODE,
        )
    overrides = {
        name: (None if getattr(advanced, name) is None or str(getattr(advanced, name)).strip() == ""
               else str(getattr(advanced, name)).strip())
        for name in ("min_atoms", "max_atoms", "num_generations", "walltime")
    }
    spec = default_search_spec(composition, **overrides)
    request = {"kind": "formula", "advanced_overrides": overrides}
    provenance_request = {
        "kind": "formula",
        "entered_formula": str(formula).strip(),
        "entered_composition": _composition_rows(entered.keys(), entered.values()),
        "advanced_overrides_as_entered": {
            name: getattr(advanced, name) for name in ("min_atoms", "max_atoms", "num_generations", "walltime")
        },
        "reduced_from_entered": dict(reduced) != dict(entered),
        "reordered_from_entered": list(composition) != list(reduced),
    }
    return _build(spec, request=request, provenance_request=provenance_request,
                  runtime_settings=runtime_settings, generator=generator)


def build_search_plan(spec, *, runtime_settings=None, generator: Mapping[str, Any] | None = None) -> StructureSearchPlan:
    """The plan of an explicit :class:`~backend.structure_search.spec.StructureSearchSpec`.

    The spec's elements are put in canonical order (counts follow them); the
    spec as given is kept in the plan's provenance.
    """

    from backend.structure_search.spec import StructureSearchSpec

    if not isinstance(spec, StructureSearchSpec):
        raise TypeError("build_search_plan expects a StructureSearchSpec")
    ordered = canonical_spec(spec)
    provenance_request = {
        "kind": "spec",
        "spec_as_given": spec.to_record(),
        "reduced_from_entered": False,
        "reordered_from_entered": ordered.elements != spec.elements,
    }
    return _build(ordered, request={"kind": "spec"}, provenance_request=provenance_request,
                  runtime_settings=runtime_settings, generator=generator)


def _build(spec, *, request, provenance_request, runtime_settings, generator) -> StructureSearchPlan:
    from backend.calculations.resources import structure_search_resources
    from backend.structure_search import admission, incar_policy, input_writer, potcar
    from backend.structure_search import spec as spec_module
    from backend.structure_search import step_validity
    from backend.structure_search.prediction_plan import formula_text, reduce_composition
    from backend.structure_search.runtime import prepared_search_runtime

    settings = _runtime_settings(runtime_settings)
    admission.require_admissible_search(spec)

    ranks = settings.ranks_per_relaxation
    cutoff_record = incar_policy.search_cutoff_record(spec.elements)
    ladder_record = incar_policy.incar_policy_record(cutoff_record, ranks)
    incar_files = incar_policy.incar_texts(cutoff_record, ranks)
    potcar_symbols = potcar.bmd_potcar_symbols(spec.elements)
    identities = {str(row["symbol"]): row for row in cutoff_record["potcars"]}
    potcar_spec_text = "".join(f"{potcar_symbols[element]}\n" for element in spec.elements)

    reduced = canonical_composition(reduce_composition(dict(zip(spec.elements, spec.counts))))
    formula_atoms = sum(spec.counts)
    cell_sizes = [n for n in range(spec.min_atoms, spec.max_atoms + 1) if n % formula_atoms == 0]
    defaults = spec_module.search_defaults_record()
    from_defaults = request["kind"] == "formula"
    kresol_source = (
        f"{defaults['policy_id']}/{defaults['policy_version']} ({defaults['status']})"
        if from_defaults else "explicit search specification"
    )
    resources = structure_search_resources(spec.walltime)

    record = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "desired_output": NEW_STRUCTURE_OUTPUT,
        "request": request,
        "composition": {
            "search": _composition_rows(spec.elements, spec.counts),
            "search_formula": formula_text(dict(zip(spec.elements, spec.counts))),
            "reduced": _composition_rows(reduced.keys(), reduced.values()),
            "reduced_formula": formula_text(reduced),
            "formula_atoms": formula_atoms,
            "element_order": ELEMENT_ORDER_RULE,
        },
        "search": {
            "spec": spec.to_record(),
            "defaults": defaults,
            "defaults_applied": from_defaults,
            "method": "USPEX 25.0 evolutionary search, fixed composition (calculationType 300)",
            "theory": "plain PBE, non-magnetic",
            "cell_sizes": cell_sizes,
            "formula_units": [n // formula_atoms for n in cell_sizes],
            "energy_grade": incar_policy.SEARCH_ENERGY_GRADE,
            "equivalent_to_bmd_adaptive_relaxation": incar_policy.EQUIVALENT_TO_BMD_ADAPTIVE_RELAXATION,
        },
        "admission": {
            "decision": "admitted",
            **admission.admission_scope_record(),
        },
        "screening_and_ranking": ladder_record,
        "incar_files": {
            name: {"path": f"{SPECIFIC_DIR}/{name}", "text": text, "sha256": _sha256_text(text)}
            for name, text in incar_files.items()
        },
        "kpoints": {
            "keyword": "KresolStart",
            "units": "2 pi / Angstrom",
            "mesh": "generated by USPEX for each candidate cell from KresolStart",
            "steps": [
                {
                    "step": index,
                    "kresol": value,
                    "source": kresol_source,
                    "bmd_policy_bound": (
                        {
                            "max_kresol": incar_policy.BMD_STANDARD_MAX_KRESOL,
                            "reciprocal_density": incar_policy.BMD_RECIPROCAL_DENSITY,
                        }
                        if index in incar_policy.BMD_STANDARD_STEPS else None
                    ),
                }
                for index, value in enumerate(spec.kresol_start, start=1)
            ],
        },
        "potcar": {
            "functional": potcar.POTCAR_FUNCTIONAL,
            "library_folder": potcar.POTCAR_FUNCTIONAL_DIR,
            "symbol_source": potcar.POTCAR_SYMBOL_SOURCE,
            "contains_potcar_contents": False,
            "spec_file": {"path": POTCAR_SPEC_FILE, "text": potcar_spec_text,
                          "sha256": _sha256_text(potcar_spec_text)},
            "elements": [
                {
                    "element": element,
                    "symbol": potcar_symbols[element],
                    "TITEL": identities[potcar_symbols[element]]["TITEL"],
                    "ENMAX": identities[potcar_symbols[element]]["ENMAX"],
                    "dataset_sha256": identities[potcar_symbols[element]]["sha256"],
                    "staged_at_execution_as": f"{SPECIFIC_DIR}/{potcar.specific_potcar_name(element)}",
                }
                for element in spec.elements
            ],
            "enmax_metadata": dict(cutoff_record.get("metadata") or {}),
        },
        "candidate_validity": {
            **step_validity.policy_record(),
            "enforced": incar_policy.STEP_VALIDITY_ENFORCED,
            "enforced_by": incar_policy.STEP_VALIDITY_ENFORCED_BY,
            "verified_on_power": incar_policy.STEP_VALIDITY_VERIFIED_ON_POWER,
        },
        "runtime": prepared_search_runtime(settings),
        "execution": {
            "input_txt": {
                "which_cluster": input_writer.WHICH_CLUSTER,
                "cores_per_job": input_writer.CORES_PER_JOB,
                "abinitio_code": input_writer.ABINITIO_CODE_VASP,
                "num_parallel_calcs": settings.parallel_relaxations,
                "sleep_seconds": SEARCH_SLEEP_SECONDS,
                "sleep_seconds_status": SEARCH_SLEEP_SECONDS_STATUS,
            },
            "bridge": {
                "script": _bridge_script_name(),
                "sha256": _bridge_script_sha256(),
                "path": "bound at preparation: absolute path on POWER, under the container bind path",
                "bind": _bind_path(),
            },
        },
        "resources": {
            "status": "provisional_estimate",
            "approved": False,
            "note": "An estimate for review, not an approved allocation; submission is not implemented.",
            "request": resources.to_dict(),
            "sbatch_directives": resources.sbatch_directives(),
        },
        "refinement": {
            "status": "planned, not yet available",
            "candidates": spec_module.PLANNED_FINALIST_COUNT,
            "selection": "lowest search-grade energy per atom, structurally distinct",
            "method": "BMD adaptive geometry relaxation, then a plain-PBE static energy",
            "protocols": ["bmd_compute.geometry_relaxation/1", "bmd_compute.geometry_convergence/1"],
        },
        "not_implemented": [
            "SLURM submission", "search monitoring", "finalist selection", "adaptive finalist relaxation",
        ],
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "covered_by_plan_digest": False,
        "request": provenance_request,
        "generator": _generator_provenance(generator),
    }
    return StructureSearchPlan.from_record(record, provenance)


def _require_provenance_matches(plan: StructureSearchPlan) -> None:
    """The recorded request must describe the planned search (it is outside the digest)."""

    from backend.structure_search.prediction_plan import FormulaError, parse_formula, reduce_composition

    provenance, record = plan.provenance, plan.record
    request = provenance.get("request")
    if not provenance:
        return  # a plan without provenance (for example built from a bare record)
    search = [(row["element"], row["count"]) for row in record.get("composition", {}).get("search", [])]
    try:
        if request.get("kind") == "formula" and record["request"]["kind"] == "formula":
            entered = canonical_composition(reduce_composition(parse_formula(request["entered_formula"])))
            consistent = list(entered.items()) == search
        elif request.get("kind") == "spec" and record["request"]["kind"] == "spec":
            given = request["spec_as_given"]
            ordered = canonical_composition({row["element"]: row["count"] for row in given["composition"]})
            consistent = list(ordered.items()) == search and {
                key: value for key, value in given.items() if key != "composition"
            } == {key: value for key, value in record["search"]["spec"].items() if key != "composition"}
        else:
            consistent = False
    except (KeyError, TypeError, AttributeError, FormulaError):
        consistent = False
    if not consistent:
        raise SearchPlanError(
            "The search plan's provenance does not describe the planned search.",
            code=INVALID_PLAN_CODE,
        )


def _bridge_script_name() -> str:
    from backend.structure_search.bridge.master import BRIDGE_SCRIPT_NAME

    return BRIDGE_SCRIPT_NAME


def _bind_path() -> str:
    from backend.structure_search.bridge.master import DEFAULT_BIND

    return DEFAULT_BIND


def rebuild_plan(plan: StructureSearchPlan, *, runtime_settings=None) -> StructureSearchPlan:
    """The plan the same request gives with today's code, policies and settings."""

    from backend.structure_search.prediction_plan import AdvancedSettings
    from backend.structure_search.spec import StructureSearchSpec

    from backend.structure_search.prediction_plan import formula_text

    record = plan.record
    request = record.get("request") or {}
    # Rebuilt from the scientific record only; provenance cannot change the result.
    unrecorded = {"repository": None, "commit": None, "dirty": None,
                  "provenance_available": False, "unavailable_reason": "rebuild for comparison"}
    try:
        if request.get("kind") == "formula":
            composition = {row["element"]: row["count"] for row in record["composition"]["search"]}
            return build_search_plan_from_formula(
                formula_text(composition),
                AdvancedSettings(**(request.get("advanced_overrides") or {})),
                runtime_settings=runtime_settings,
                generator=unrecorded,
            )
        if request.get("kind") == "spec":
            return build_search_plan(StructureSearchSpec.from_mapping(record["search"]["spec"]),
                                     runtime_settings=runtime_settings, generator=unrecorded)
    except (KeyError, TypeError, AttributeError) as exc:
        raise SearchPlanError(f"The search plan is incomplete: {exc!r}.", code=INVALID_PLAN_CODE) from exc
    raise SearchPlanError("The search plan has no recognised request.", code=INVALID_PLAN_CODE)


def require_current_plan(plan: StructureSearchPlan, *, runtime_settings=None) -> None:
    """Refuse a plan that today's policies, defaults or settings would plan differently."""

    try:
        current = rebuild_plan(plan, runtime_settings=runtime_settings)
    except CalculationValidationError as exc:
        raise SearchPlanError(
            f"The search plan's request is no longer accepted: {exc}",
            code=STALE_PLAN_CODE,
            suggestion="Plan the search again.",
        ) from exc
    if current.digest != plan.digest:
        old, new = plan.record, current.record
        changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        raise SearchPlanError(
            "The search plan is stale: BMD Compute would now plan this request differently "
            f"({', '.join(changed)} changed).",
            code=STALE_PLAN_CODE,
            suggestion="Plan the search again and review the new plan.",
            plan_digest=plan.digest,
            current_digest=current.digest,
            changed=changed,
        )


# -- rendering the prepared files ---------------------------------------------------


def _require_bridge_path(bridge_path: str) -> str:
    from backend.structure_search.bridge.master import BRIDGE_SCRIPT_NAME, DEFAULT_BIND

    path = str(bridge_path or "")
    pure = PurePosixPath(path)
    if (
        not pure.is_absolute()
        or ".." in pure.parts
        or pure.name != BRIDGE_SCRIPT_NAME
        or not pure.is_relative_to(DEFAULT_BIND)
        or str(pure) != path
    ):
        raise SearchPlanError(
            f"The launch-bridge path must be an absolute, normalised path to {BRIDGE_SCRIPT_NAME} "
            f"under {DEFAULT_BIND} (the only path the USPEX container sees), not {path!r}.",
            code=INVALID_BINDING_CODE,
        )
    return path


def _incar_lines_match(text: str, settings: Mapping[str, Any]) -> bool:
    from backend.structure_search.incar_policy import render_incar

    return sorted(text.splitlines()) == sorted(render_incar(dict(settings)).splitlines())


def _require_consistent_record(record: Mapping[str, Any]) -> None:
    """The plan's exact file texts agree with its structured settings."""

    steps = record["screening_and_ranking"]["steps"]
    files = record["incar_files"]
    if sorted(files) != [f"INCAR_{row['step']}" for row in steps]:
        raise SearchPlanError("The plan's INCAR files do not match its ladder.", code=INVALID_PLAN_CODE)
    for row in steps:
        entry = files[f"INCAR_{row['step']}"]
        if entry["sha256"] != _sha256_text(entry["text"]) or not _incar_lines_match(entry["text"], row["incar"]):
            raise SearchPlanError(
                f"The plan's INCAR_{row['step']} text does not match its step {row['step']} settings.",
                code=INVALID_PLAN_CODE,
            )
    spec_file = record["potcar"]["spec_file"]
    symbols = [row["symbol"] for row in record["potcar"]["elements"]]
    if spec_file["text"] != "".join(f"{symbol}\n" for symbol in symbols) or spec_file["sha256"] != _sha256_text(
        spec_file["text"]
    ):
        raise SearchPlanError("The plan's POTCAR.spec does not match its POTCAR record.", code=INVALID_PLAN_CODE)
    if not all(_POTCAR_SYMBOL.fullmatch(symbol) for symbol in symbols):
        raise SearchPlanError("The plan's POTCAR.spec holds something other than symbols.", code=INVALID_PLAN_CODE)
    if record["potcar"].get("contains_potcar_contents") is not False:
        raise SearchPlanError("A search plan must not contain POTCAR contents.", code=INVALID_PLAN_CODE)


def render_prepared_files(
    plan: StructureSearchPlan,
    *,
    bridge_path: str,
    generator: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Every file preparation writes, from the frozen plan and the bridge-path binding.

    ``generator`` is the Git provenance of the code that prepares the files
    (default: this checkout), recorded in ``prepared_inputs.json``.

    The INCARs and ``POTCAR.spec`` are the plan's exact texts; ``INPUT.txt``
    is rendered by the input writer from the plan's spec and execution
    settings. Keys are POSIX paths relative to the prepared directory.
    """

    from backend.structure_search.input_writer import (
        INPUT_FILENAME,
        SearchExecutionSettings,
        render_input_text,
    )
    from backend.structure_search.spec import StructureSearchSpec

    record = plan.record
    try:
        _require_consistent_record(record)
    except (KeyError, TypeError, AttributeError) as exc:
        raise SearchPlanError(f"The search plan is incomplete: {exc!r}.", code=INVALID_PLAN_CODE) from exc
    binding = _require_bridge_path(bridge_path)
    try:
        execution = record["execution"]["input_txt"]
        spec = StructureSearchSpec.from_mapping(record["search"]["spec"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise SearchPlanError(f"The search plan is incomplete: {exc!r}.", code=INVALID_PLAN_CODE) from exc
    input_text = render_input_text(
        spec,
        SearchExecutionSettings(
            bridge_path=binding,
            num_parallel_calcs=execution["num_parallel_calcs"],
            sleep_seconds=execution["sleep_seconds"],
        ),
    )
    files: dict[str, bytes] = {INPUT_FILENAME: input_text.encode("utf-8")}
    for entry in record["incar_files"].values():
        files[entry["path"]] = entry["text"].encode("utf-8")
    files[POTCAR_SPEC_FILE] = record["potcar"]["spec_file"]["text"].encode("utf-8")
    files[PLAN_FILE] = plan.to_json().encode("utf-8")
    manifest = {
        "schema": PREPARED_SCHEMA,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "plan_digest": plan.digest,
        "plan_file": PLAN_FILE,
        "bindings": {"bridge_path": binding},
        "generator": _generator_provenance(generator),
        "bindings_and_generator_covered_by_plan_digest": False,
        "files": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
        "potcar_files_staged_at_execution": [
            {key: row[key] for key in ("staged_at_execution_as", "symbol", "TITEL", "dataset_sha256")}
            for row in record["potcar"]["elements"]
        ],
        "contains_potcar_contents": False,
        "submitted": False,
    }
    files[PREPARED_FILE] = (json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    return files


def prepared_file_names(plan: StructureSearchPlan) -> list[str]:
    return sorted(render_prepared_files(
        plan, bridge_path=f"{_bind_path()}/x/{_bridge_script_name()}", generator={}
    ))


# -- safe local preparation -----------------------------------------------------------


def _require_safe_new_path(destination) -> Path:
    raw = os.fspath(destination)
    if not raw or "\x00" in raw:
        raise SearchPlanError("The destination path is empty or invalid.", code=UNSAFE_PATH_CODE)
    if ".." in Path(raw).parts:
        raise SearchPlanError(f"The destination {raw!r} contains '..'.", code=UNSAFE_PATH_CODE)
    path = Path(os.path.abspath(raw))
    if os.path.lexists(path):
        raise SearchPlanError(
            f"{path} already exists; a search is prepared only into a new directory.",
            code=DESTINATION_EXISTS_CODE,
            suggestion="Choose a path that does not exist yet.",
        )
    parent = path.parent
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise SearchPlanError(f"{current} does not exist.", code=UNSAFE_PATH_CODE) from exc
        if stat.S_ISLNK(info.st_mode):
            raise SearchPlanError(f"{current} is a symbolic link; refusing to prepare under it.",
                                  code=UNSAFE_PATH_CODE)
        if not stat.S_ISDIR(info.st_mode):
            raise SearchPlanError(f"{current} is not a directory.", code=UNSAFE_PATH_CODE)
    return path


_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def prepare_search_directory(
    plan: StructureSearchPlan,
    destination,
    *,
    bridge_path: str,
    runtime_settings=None,
    generator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a frozen plan's inputs into the new directory ``destination``.

    Refuses a plan whose digest is wrong or that is stale (today's code,
    policies or settings would plan the same request differently), an
    existing destination, a destination under a symbolic link, and a
    missing parent directory. Files are created exclusively and never
    overwritten. After writing, the directory is verified with
    :func:`verify_prepared_directory`; if anything fails, the files this call
    created are removed again. Returns the ``prepared_inputs.json`` record.
    """

    if not isinstance(plan, StructureSearchPlan):
        raise TypeError("prepare_search_directory expects a StructureSearchPlan")
    plan = load_plan(plan.to_document())
    require_current_plan(plan, runtime_settings=runtime_settings)
    files = render_prepared_files(plan, bridge_path=bridge_path, generator=generator)
    path = _require_safe_new_path(destination)

    os.mkdir(path, 0o755)
    created_files: list[str] = []
    created_dirs: list[str] = []
    dir_fd = os.open(path, _DIR_FLAGS)
    try:
        for relative in sorted(files):
            parts = relative.split("/")
            if len(parts) == 2 and parts[0] not in created_dirs:
                os.mkdir(parts[0], 0o755, dir_fd=dir_fd)
                created_dirs.append(parts[0])
            fd = os.open(relative, _WRITE_FLAGS, 0o644, dir_fd=dir_fd)
            created_files.append(relative)
            with os.fdopen(fd, "wb") as handle:
                handle.write(files[relative])
        manifest = verify_prepared_directory(path, runtime_settings=runtime_settings, require_current=False)
    except BaseException:
        for relative in reversed(created_files):
            try:
                os.unlink(relative, dir_fd=dir_fd)
            except OSError:
                pass
        for name in reversed(created_dirs):
            try:
                os.rmdir(name, dir_fd=dir_fd)
            except OSError:
                pass
        os.close(dir_fd)
        dir_fd = -1
        try:
            os.rmdir(path)
        except OSError:
            pass
        raise
    finally:
        if dir_fd >= 0:
            os.close(dir_fd)
    return manifest


def _directory_entries(root: Path) -> dict[str, bytes]:
    """Every regular file under ``root``; refuse symlinks and other file types."""

    info = os.lstat(root)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SearchPlanError(f"{root} is not a real directory.", code=UNSAFE_PATH_CODE)
    found: dict[str, bytes] = {}

    def walk(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                relative = f"{prefix}{entry.name}"
                if entry.is_symlink():
                    raise SearchPlanError(f"{relative} is a symbolic link.", code=UNEXPECTED_FILE_CODE)
                if entry.is_dir(follow_symlinks=False):
                    walk(Path(entry.path), f"{relative}/")
                elif entry.is_file(follow_symlinks=False):
                    fd = os.open(entry.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "rb") as handle:
                        found[relative] = handle.read()
                else:
                    raise SearchPlanError(f"{relative} is not a regular file.", code=UNEXPECTED_FILE_CODE)

    walk(root, "")
    return found


def verify_prepared_directory(path, *, runtime_settings=None, require_current: bool = True) -> dict[str, Any]:
    """Check that a prepared directory holds exactly its frozen plan's inputs.

    The directory must contain exactly the files preparation writes (no
    others, no symbolic links, no empty extra folders), ``search_plan.json``
    must carry a valid digest, ``prepared_inputs.json`` must name that
    digest, and every file must equal what the plan and its recorded
    bridge-path binding render. With ``require_current`` the plan must also
    still be current. Returns the ``prepared_inputs.json`` record.
    """

    root = Path(os.path.abspath(os.fspath(path)))
    found = _directory_entries(root)
    if PLAN_FILE not in found or PREPARED_FILE not in found:
        raise SearchPlanError(
            f"{root} is not a prepared search directory (no {PLAN_FILE} or {PREPARED_FILE}).",
            code=UNEXPECTED_FILE_CODE,
        )
    plan = load_plan_text(found[PLAN_FILE].decode("utf-8"))
    try:
        manifest = json.loads(found[PREPARED_FILE].decode("utf-8"))
    except ValueError as exc:
        raise SearchPlanError(f"{PREPARED_FILE} is not valid JSON.", code=PREPARED_MISMATCH_CODE) from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != PREPARED_SCHEMA:
        raise SearchPlanError(f"{PREPARED_FILE} has the wrong schema.", code=PREPARED_MISMATCH_CODE)
    if manifest.get("plan_digest") != plan.digest:
        raise SearchPlanError(
            f"{PREPARED_FILE} names plan {manifest.get('plan_digest')}, but {PLAN_FILE} is {plan.digest}.",
            code=PREPARED_MISMATCH_CODE,
        )
    if require_current:
        require_current_plan(plan, runtime_settings=runtime_settings)
    bindings = manifest.get("bindings") or {}
    generator = manifest.get("generator")
    if not isinstance(generator, Mapping):
        raise SearchPlanError(f"{PREPARED_FILE} has no generator record.", code=PREPARED_MISMATCH_CODE)
    expected = render_prepared_files(plan, bridge_path=bindings.get("bridge_path"), generator=generator)
    directories = {name.rsplit("/", 1)[0] for name in expected if "/" in name}
    extra = sorted(set(found) - set(expected))
    missing = sorted(set(expected) - set(found))
    if extra or missing:
        raise SearchPlanError(
            f"{root} does not hold exactly the prepared files"
            + (f"; unexpected: {', '.join(extra)}" if extra else "")
            + (f"; missing: {', '.join(missing)}" if missing else "") + ".",
            code=UNEXPECTED_FILE_CODE,
            unexpected=extra,
            missing=missing,
        )
    present_dirs = {
        str(Path(dirpath).relative_to(root)).replace(os.sep, "/")
        for dirpath, _, _ in os.walk(root)
        if Path(dirpath) != root
    }
    if present_dirs != directories:
        raise SearchPlanError(
            f"{root} has unexpected folders: {', '.join(sorted(present_dirs - directories))}.",
            code=UNEXPECTED_FILE_CODE,
        )
    differing = sorted(name for name in expected if expected[name] != found[name])
    if differing:
        raise SearchPlanError(
            f"Prepared file(s) differ from the frozen plan: {', '.join(differing)}.",
            code=PREPARED_MISMATCH_CODE,
            files=differing,
        )
    return json.loads(expected[PREPARED_FILE].decode("utf-8"))


def verify_execution_binding(
    prepared_dir,
    *,
    bridge_path: str,
    runtime_settings=None,
    filesystem_root: str = "/",
) -> dict[str, Any]:
    """Check, where a search is about to run, that it runs what was prepared.

    For the job on POWER, before USPEX starts (not yet wired in; submission is
    not implemented). It requires that the prepared directory verifies and is
    current (:func:`verify_prepared_directory`), that ``bridge_path`` is the
    path recorded at preparation, and that the file there is a regular file
    (not a symbolic link) whose SHA-256 is the bridge implementation hash in
    the plan. ``filesystem_root`` is where the cluster's file system is seen
    (``/`` on POWER; tests use a temporary directory). Returns the
    ``prepared_inputs.json`` record.
    """

    manifest = verify_prepared_directory(prepared_dir, runtime_settings=runtime_settings, require_current=True)
    binding = _require_bridge_path(bridge_path)
    recorded = (manifest.get("bindings") or {}).get("bridge_path")
    if binding != recorded:
        raise SearchPlanError(
            f"The search was prepared for the launch bridge at {recorded}, but it would run {binding}.",
            code=BINDING_MISMATCH_CODE,
            recorded=recorded,
            actual=binding,
        )
    plan = load_plan_text((Path(os.path.abspath(os.fspath(prepared_dir))) / PLAN_FILE).read_text(encoding="utf-8"))
    expected = plan.record["execution"]["bridge"]["sha256"]
    location = Path(filesystem_root) / binding.lstrip("/")
    try:
        fd = os.open(location, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SearchPlanError(
            f"The launch bridge {binding} cannot be opened ({exc.strerror}); a symbolic link is not accepted.",
            code=BRIDGE_MISMATCH_CODE,
        ) from exc
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise SearchPlanError(f"The launch bridge {binding} is not a regular file.", code=BRIDGE_MISMATCH_CODE)
        found = hashlib.sha256(handle.read()).hexdigest()
    if found != expected:
        raise SearchPlanError(
            f"The launch bridge {binding} is not the implementation the plan was made for "
            f"(SHA-256 {found}, plan {expected}).",
            code=BRIDGE_MISMATCH_CODE,
            expected=expected,
            found=found,
        )
    return manifest


__all__ = [
    "DESTINATION_EXISTS_CODE",
    "INVALID_BINDING_CODE",
    "INVALID_PLAN_CODE",
    "NEW_STRUCTURE_OUTPUT",
    "PLAN_DIGEST_MISMATCH_CODE",
    "PLAN_FILE",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_CODE",
    "PLAN_SCHEMA_VERSION",
    "POTCAR_SPEC_FILE",
    "PREPARED_FILE",
    "PREPARED_MISMATCH_CODE",
    "PREPARED_SCHEMA",
    "PREPARED_SCHEMA_VERSION",
    "SEARCH_SLEEP_SECONDS",
    "SEARCH_SLEEP_SECONDS_STATUS",
    "SPECIFIC_DIR",
    "STALE_PLAN_CODE",
    "SearchPlanError",
    "StructureSearchPlan",
    "UNEXPECTED_FILE_CODE",
    "UNSAFE_PATH_CODE",
    "build_search_plan",
    "build_search_plan_from_formula",
    "canonical_composition",
    "canonical_spec",
    "verify_execution_binding",
    "BINDING_MISMATCH_CODE",
    "BRIDGE_MISMATCH_CODE",
    "ELEMENT_ORDER_RULE",
    "PROVENANCE_SCHEMA",
    "canonical_json",
    "load_plan",
    "load_plan_text",
    "plan_digest",
    "prepare_search_directory",
    "prepared_file_names",
    "rebuild_plan",
    "render_prepared_files",
    "require_current_plan",
    "verify_prepared_directory",
]
