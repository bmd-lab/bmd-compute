"""
New-calculation admission policy.

The registry and the shared workflow validators describe what a BMD Compute
workflow *is*: they decide whether a WorkflowSpec is structurally valid and
are also used to read historical submission records, monitor states, results
and prepared runs. They therefore keep accepting every combination BMD Compute
has ever executed, including HSE06 stages with SOC.

This module decides something narrower: which structurally valid workflows
BMD Compute accepts as *new* calculations. It is applied once, at the
authoritative build boundary (``main.build_submission_state_from_structure``),
after automatic default treatments have been resolved, so it covers Desired
Output, Custom and legacy purpose/theory/modifier requests on the build,
prepare and submit routes alike. It is never applied when reading existing
records.

HSE06 + SOC is excluded from new calculations as a product-support
limitation: the combination can be exceptionally computationally expensive
and may need system-specific convergence and resource settings. This is not a
statement that HSE06 + SOC is scientifically invalid.
"""

from __future__ import annotations

from typing import Any

from backend.calculations.models import Modifier, StageSpec, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    modifier_display_name,
    theory_display_name,
    validate_workflow_spec,
)


ADMISSION_POLICY_ID = "bmd_compute.new_calculation_admission"
ADMISSION_POLICY_VERSION = 1
HSE06_SOC_NOT_SUPPORTED_CODE = "hse06_soc_not_supported_for_new_calculations"

# (theory, modifier) pairs that are structurally valid, and remain readable in
# historical records, but are not accepted for new calculations.
UNSUPPORTED_NEW_CALCULATION_COMBINATIONS: tuple[tuple[Theory, Modifier], ...] = (
    (Theory.HSE06, Modifier.SOC),
)

HSE06_SOC_NOT_SUPPORTED_MESSAGE = (
    "HSE06 + SOC is not currently supported for automatic execution. "
    "This combination can be exceptionally computationally expensive and may "
    "require system-specific convergence and resource settings. HSE06 and "
    "PBE+SOC remain available separately."
)
HSE06_SOC_OMITTED_TITLE = "HSE06 + SOC is not currently supported by BMD Compute."
HSE06_SOC_OMITTED_MESSAGE = (
    "Spin–orbit coupling has been omitted from the HSE06 stages of this "
    "workflow. For materials containing heavy elements, this may significantly "
    "affect the predicted electronic structure, including band ordering and "
    "band gaps."
)


def stage_is_admissible_for_new_calculation(stage: StageSpec) -> bool:
    """Return whether a stage's theory/modifier combination is open to new calculations."""

    theory = Theory.from_value(stage.theory)
    modifiers = {Modifier.from_value(item) for item in stage.modifiers}
    return not any(
        theory is blocked_theory and blocked_modifier in modifiers
        for blocked_theory, blocked_modifier in UNSUPPORTED_NEW_CALCULATION_COMBINATIONS
    )


def modifier_is_admissible_for_new_theory(
    theory: Theory | str,
    modifier: Modifier | str,
) -> bool:
    normalized_theory = Theory.from_value(theory)
    normalized_modifier = Modifier.from_value(modifier)
    return (normalized_theory, normalized_modifier) not in UNSUPPORTED_NEW_CALCULATION_COMBINATIONS


def inadmissible_stage_indices(workflow: WorkflowSpec) -> tuple[int, ...]:
    """1-based indices of stages that new calculations may not contain."""

    return tuple(
        index
        for index, stage in enumerate(workflow.stages, start=1)
        if not stage_is_admissible_for_new_calculation(stage)
    )


def require_admissible_new_calculation(workflow: WorkflowSpec) -> WorkflowSpec:
    """
    Reject a new calculation that contains an unsupported combination.

    The workflow is first validated structurally with the shared validator, so
    this check only ever narrows what that validator accepts.
    """

    normalized = validate_workflow_spec(workflow)
    blocked = inadmissible_stage_indices(normalized)
    if not blocked:
        return normalized

    stage_text = _stage_list_text(blocked)
    raise CalculationValidationError(
        HSE06_SOC_NOT_SUPPORTED_MESSAGE,
        suggestion=(
            f"Remove SOC from HSE06 {stage_text}, or use PBE+SOC Static Energy "
            "for a spin–orbit-coupled calculation."
        ),
        diagnostic={
            "code": HSE06_SOC_NOT_SUPPORTED_CODE,
            "policy_id": ADMISSION_POLICY_ID,
            "policy_version": ADMISSION_POLICY_VERSION,
            "stage_indices": list(blocked),
        },
    )


def unsupported_new_calculation_combinations_payload() -> list[dict[str, Any]]:
    """Browser-facing description of the combinations new calculations exclude."""

    return [
        {
            "theory": theory.value,
            "theory_label": theory_display_name(theory),
            "modifier": modifier.value,
            "modifier_label": modifier_display_name(modifier),
            "message": HSE06_SOC_NOT_SUPPORTED_MESSAGE,
        }
        for theory, modifier in UNSUPPORTED_NEW_CALCULATION_COMBINATIONS
    ]


def _stage_list_text(indices: tuple[int, ...]) -> str:
    numbers = [str(index) for index in indices]
    if len(numbers) == 1:
        return f"stage {numbers[0]}"
    return "stages " + ", ".join(numbers[:-1]) + f" and {numbers[-1]}"


__all__ = [
    "ADMISSION_POLICY_ID",
    "ADMISSION_POLICY_VERSION",
    "HSE06_SOC_NOT_SUPPORTED_CODE",
    "HSE06_SOC_NOT_SUPPORTED_MESSAGE",
    "HSE06_SOC_OMITTED_MESSAGE",
    "HSE06_SOC_OMITTED_TITLE",
    "UNSUPPORTED_NEW_CALCULATION_COMBINATIONS",
    "inadmissible_stage_indices",
    "modifier_is_admissible_for_new_theory",
    "require_admissible_new_calculation",
    "stage_is_admissible_for_new_calculation",
    "unsupported_new_calculation_combinations_payload",
]
