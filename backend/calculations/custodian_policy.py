from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from importlib import metadata


BMD_CUSTODIAN_POLICY_ID = "bmd_compute.vasp"
BMD_CUSTODIAN_POLICY_VERSION = 1
BMD_CUSTODIAN_POLICY_RATIONALE = (
    "Output-file inactivity alone is not sufficient evidence for BMD to "
    "terminate a VASP calculation."
)
FROZEN_JOB_HANDLER = "custodian.vasp.handlers.FrozenJobErrorHandler"
HSE_BAND_STRUCTURE_CUSTODIAN_ERROR_EXCLUSIONS = ("auto_nbands",)

# BMD Compute v1 unsuccessful-stage policy. atomate2 decides, from its task
# document, whether a VASP stage succeeded (electronically and, for
# relaxations, ionically converged). BMD owns what happens next: an
# unsuccessful stage raises, so the flow fails and no dependent stage runs.
# This is passed explicitly to every maker as ``stop_children_kwargs`` rather
# than inherited from atomate2's mutable ``VASP_HANDLE_UNSUCCESSFUL`` setting
# (environment variable or ~/.atomate2.yaml). It does not change any
# convergence criterion.
UNSUCCESSFUL_STAGE_HANDLING = "error"
UNSUCCESSFUL_STAGE_POLICY_SOURCE = (
    "backend.calculations.custodian_policy.unsuccessful_stage_stop_children_kwargs"
)


def unsuccessful_stage_stop_children_kwargs() -> dict:
    """``stop_children_kwargs`` for every BMD-constructed atomate2 VASP maker."""

    return {"handle_unsuccessful": UNSUCCESSFUL_STAGE_HANDLING}


def bmd_custodian_handlers(*, excluded_vasp_errors=()) -> tuple:
    """Return independent atomate2 default handlers with BMD exclusions applied."""

    from atomate2.vasp.run import DEFAULT_HANDLERS
    from custodian.vasp.handlers import VaspErrorHandler

    excluded = set(excluded_vasp_errors or ())
    handlers = []
    for handler in DEFAULT_HANDLERS:
        if _qualified_name(handler) == FROZEN_JOB_HANDLER:
            continue
        if isinstance(handler, VaspErrorHandler) and excluded:
            handlers.append(
                _vasp_error_handler_without_errors(
                    handler,
                    VaspErrorHandler,
                    excluded,
                )
            )
        else:
            handlers.append(deepcopy(handler))

    return tuple(handlers)


def custodian_handlers_excluding_errors(excluded_errors) -> tuple:
    """Compatibility facade for callers that customize VaspErrorHandler."""

    return bmd_custodian_handlers(excluded_vasp_errors=excluded_errors)


def hse_band_structure_run_vasp_kwargs() -> dict:
    return {
        "handlers": bmd_custodian_handlers(
            excluded_vasp_errors=vasp_error_exclusions_for_stage(
                "band_structure",
                "hse06",
            )
        )
    }


def vasp_error_exclusions_for_stage(stage_type, theory) -> tuple[str, ...]:
    """Return reviewed stage-local exclusions from VaspErrorHandler."""

    if _enum_value(theory) == "hse06" and _enum_value(stage_type) in {
        "dos",
        "band_structure",
    }:
        return HSE_BAND_STRUCTURE_CUSTODIAN_ERROR_EXCLUSIONS
    return ()


def resolved_custodian_policy(stage_type, theory) -> dict:
    """Describe the JSON-safe policy used for one executable VASP stage."""

    exclusions = vasp_error_exclusions_for_stage(stage_type, theory)
    handlers = bmd_custodian_handlers(excluded_vasp_errors=exclusions)

    return {
        "policy_id": BMD_CUSTODIAN_POLICY_ID,
        "policy_version": BMD_CUSTODIAN_POLICY_VERSION,
        "stage": {
            "stage_type": _enum_value(stage_type),
            "theory": _enum_value(theory),
        },
        "handlers": [_policy_object_description(handler) for handler in handlers],
        "explicit_handler_exclusions": [FROZEN_JOB_HANDLER],
        "vasp_error_exclusions": list(exclusions),
        "validators": {
            "source": "atomate2.vasp.run._DEFAULT_VALIDATORS",
            "explicit_override": None,
            "resolved": _default_validator_descriptions(),
        },
        "unsuccessful_stage": {
            "stop_children_kwargs": unsuccessful_stage_stop_children_kwargs(),
            "source": UNSUCCESSFUL_STAGE_POLICY_SOURCE,
        },
        "walltime_authority": "slurm",
        "walltime_handler": None,
        "custodian_version": _package_version("custodian"),
        "implementation_source": (
            "backend.calculations.custodian_policy.bmd_custodian_handlers"
        ),
        "rationale": BMD_CUSTODIAN_POLICY_RATIONALE,
    }


def _default_validator_descriptions() -> list[dict]:
    from atomate2.vasp.run import _DEFAULT_VALIDATORS

    return [_policy_object_description(validator) for validator in _DEFAULT_VALIDATORS]


def _policy_object_description(value) -> dict:
    serialized = value.as_dict() if hasattr(value, "as_dict") else {}
    module = serialized.pop("@module", type(value).__module__)
    class_name = serialized.pop("@class", type(value).__name__)
    serialized.pop("@version", None)
    return {
        "class": f"{module}.{class_name}",
        "configuration": _json_safe_value(serialized),
    }


def _qualified_name(value) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _enum_value(value) -> str:
    return str(getattr(value, "value", value or "")).strip().lower()


def _package_version(package_name: str) -> str:
    try:
        return str(metadata.version(package_name))
    except metadata.PackageNotFoundError:
        return "unavailable"


def _json_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _vasp_error_handler_without_errors(handler, handler_cls, excluded_errors: set[str]):
    available_errors = list(getattr(handler_cls, "error_msgs", {}))
    current_subset = list(getattr(handler, "errors_subset_to_catch", available_errors))
    errors_subset = [
        error
        for error in current_subset
        if error not in excluded_errors
    ]
    output_filename = getattr(handler, "output_filename", "vasp.out")
    vtst_fixes = getattr(handler, "vtst_fixes", False)

    try:
        return handler_cls(
            output_filename=output_filename,
            errors_subset_to_catch=errors_subset,
            vtst_fixes=vtst_fixes,
        )
    except TypeError:
        return handler_cls(
            output_filename=output_filename,
            errors_subset_to_catch=errors_subset,
        )


__all__ = [
    "BMD_CUSTODIAN_POLICY_ID",
    "BMD_CUSTODIAN_POLICY_RATIONALE",
    "BMD_CUSTODIAN_POLICY_VERSION",
    "FROZEN_JOB_HANDLER",
    "HSE_BAND_STRUCTURE_CUSTODIAN_ERROR_EXCLUSIONS",
    "bmd_custodian_handlers",
    "custodian_handlers_excluding_errors",
    "hse_band_structure_run_vasp_kwargs",
    "resolved_custodian_policy",
    "vasp_error_exclusions_for_stage",
]
