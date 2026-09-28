from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from backend.calculations.dispersion import (
    DEFAULT_DISPERSION_METHOD,
    dispersion_option_payload,
)
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    modifier_display_name,
    stage_display_name,
    theory_display_name,
    validate_stage_spec,
    validate_workflow_spec,
)


"""
Automatic treatment resolution for BMD-managed Desired Output workflows.

This module turns an already parsed structure and a backend-owned Desired Output
base recipe into the executable WorkflowSpec BMD Compute will run. It reuses the
existing Method Consideration evidence layer; Custom workflow remains
user-managed and should not be passed through this resolver.
"""


SPIN_CONSIDERATION_ID = "spin.composition_screen"
DISPERSION_CONSIDERATION_ID = "dispersion.two_dimensional_connectivity"
SOC_CONSIDERATION_ID = "soc.heavy_elements"
AUTOMATIC_APPLICATION_APPLIED = "applied"
AUTOMATIC_APPLICATION_ADVISORY = "advisory"
AUTOMATIC_APPLICATION_NOT_APPLICABLE = "not_applicable"
SOC_EXCLUDED_STAGE_TYPES = frozenset({StageType.RELAX})
SOC_NOT_APPLICABLE_REASON = (
    "BMD Compute keeps geometry optimisation stages non-SOC, and this Desired "
    "Output contains no stage that receives SOC."
)
IMPLEMENTATION_SOURCE = "backend.calculations.default_treatments"


@dataclass(frozen=True)
class AppliedDefaultTreatment:
    consideration_id: str
    modifier: Modifier | str
    display_name: str
    stage_indices: tuple[int, ...]
    stage_applications: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modifier", Modifier.from_value(self.modifier))
        object.__setattr__(
            self,
            "stage_indices",
            tuple(int(index) for index in self.stage_indices),
        )
        object.__setattr__(
            self,
            "stage_applications",
            tuple(_json_safe_mapping(application) for application in self.stage_applications),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "consideration_id": self.consideration_id,
            "modifier": self.modifier.value,
            "display_name": self.display_name,
            "application_state": AUTOMATIC_APPLICATION_APPLIED,
            "stage_indices": list(self.stage_indices),
            "stage_applications": [
                dict(application)
                for application in self.stage_applications
            ],
            "source": IMPLEMENTATION_SOURCE,
        }


@dataclass(frozen=True)
class ResolvedDefaultWorkflow:
    base_workflow: WorkflowSpec
    resolved_workflow: WorkflowSpec
    applied_treatments: tuple[AppliedDefaultTreatment, ...] = field(default_factory=tuple)
    advisory_consideration_ids: tuple[str, ...] = field(default_factory=tuple)
    desired_output: str | None = None
    mode: str = "bmd_managed_desired_output"
    not_applicable_considerations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_workflow", validate_workflow_spec(self.base_workflow))
        object.__setattr__(self, "resolved_workflow", validate_workflow_spec(self.resolved_workflow))
        object.__setattr__(
            self,
            "applied_treatments",
            tuple(self.applied_treatments),
        )
        object.__setattr__(
            self,
            "advisory_consideration_ids",
            tuple(str(item) for item in self.advisory_consideration_ids),
        )
        object.__setattr__(
            self,
            "not_applicable_considerations",
            tuple(_json_safe_mapping(item) for item in self.not_applicable_considerations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "desired_output": self.desired_output,
            "base_workflow": self.base_workflow.to_dict(),
            "resolved_workflow": self.resolved_workflow.to_dict(),
            "applied_treatments": [
                treatment.to_dict()
                for treatment in self.applied_treatments
            ],
            "advisory_consideration_ids": list(self.advisory_consideration_ids),
            "not_applicable_considerations": [
                dict(item)
                for item in self.not_applicable_considerations
            ],
            "source": IMPLEMENTATION_SOURCE,
        }


def resolve_default_treatments(
    structure,
    base_workflow: WorkflowSpec,
    *,
    desired_output: str | None = None,
) -> ResolvedDefaultWorkflow:
    """
    Resolve BMD automatic treatments for a Desired Output base workflow.

    The function is deterministic for the supplied parsed structure and base
    workflow. It does not mutate the structure or the supplied WorkflowSpec.
    """

    normalized_base = validate_workflow_spec(base_workflow)
    payload = _method_consideration_payload(structure, workflow=normalized_base)
    consideration_ids = tuple(
        str(consideration.get("id"))
        for consideration in payload.get("considerations", ())
    )

    stages = tuple(normalized_base.stages)
    applied: list[AppliedDefaultTreatment] = []

    if SPIN_CONSIDERATION_ID in consideration_ids:
        stages, treatment = _apply_spin_polarisation(stages)
        applied.append(treatment)

    if DISPERSION_CONSIDERATION_ID in consideration_ids:
        stages, treatment = _apply_dispersion(stages)
        if treatment is not None:
            applied.append(treatment)

    not_applicable: list[dict[str, Any]] = []
    if SOC_CONSIDERATION_ID in consideration_ids:
        stages, treatment = _apply_soc(stages)
        if treatment is not None:
            applied.append(treatment)
        else:
            not_applicable.append(
                {
                    "consideration_id": SOC_CONSIDERATION_ID,
                    "modifier": Modifier.SOC.value,
                    "reason": SOC_NOT_APPLICABLE_REASON,
                }
            )

    resolved = validate_workflow_spec(
        WorkflowSpec(
            stages=stages,
            label=normalized_base.label,
            recipe=normalized_base.recipe,
        )
    )
    applied_ids = {treatment.consideration_id for treatment in applied}
    not_applicable_ids = {item["consideration_id"] for item in not_applicable}
    advisory_ids = tuple(
        consideration_id
        for consideration_id in consideration_ids
        if consideration_id not in applied_ids
        and consideration_id not in not_applicable_ids
    )
    return ResolvedDefaultWorkflow(
        base_workflow=normalized_base,
        resolved_workflow=resolved,
        applied_treatments=tuple(applied),
        advisory_consideration_ids=advisory_ids,
        desired_output=desired_output,
        not_applicable_considerations=tuple(not_applicable),
    )


def automatic_default_treatment_policy() -> dict[str, Any]:
    return {
        "scope": (
            "BMD Compute executable methodology for BMD-managed Desired Output "
            "workflows: the automatic treatments BMD Compute applies"
        ),
        "source": IMPLEMENTATION_SOURCE,
        "applies_to": {
            "workflow_mode": "bmd_managed_desired_output",
            "custom_workflow": "preserved_without_automatic_changes",
        },
        "treatments": [
            {
                "consideration_id": SPIN_CONSIDERATION_ID,
                "modifier": Modifier.SPIN_POLARIZED.value,
                "display_name": modifier_display_name(Modifier.SPIN_POLARIZED),
                "trigger_source": "backend.calculations.method_considerations",
                "application": "all stages in the selected BMD-managed Desired Output workflow",
                "support_guard": "backend.calculations.registry.validate_stage_spec",
            },
            {
                "consideration_id": DISPERSION_CONSIDERATION_ID,
                "modifier": Modifier.DISPERSION.value,
                "display_name": modifier_display_name(Modifier.DISPERSION),
                "trigger_source": "backend.calculations.method_considerations",
                "method": DEFAULT_DISPERSION_METHOD,
                "incar_effect": {"IVDW": 12},
                "application": [
                    {"stage_type": StageType.RELAX.value, "theory": Theory.PBE.value},
                    {"stage_type": StageType.STATIC.value, "theory": Theory.PBE.value},
                ],
                "support_guard": "backend.calculations.registry.validate_stage_spec",
            },
            {
                "consideration_id": SOC_CONSIDERATION_ID,
                "modifier": Modifier.SOC.value,
                "display_name": modifier_display_name(Modifier.SOC),
                "trigger_source": "backend.calculations.method_considerations",
                "application": "every non-relaxation stage in the selected BMD-managed Desired Output workflow",
                "excluded_stage_types": sorted(
                    stage_type.value for stage_type in SOC_EXCLUDED_STAGE_TYPES
                ),
                "executable": "vasp_ncl",
                "initial_magnetic_moments": (
                    "zero vector MAGMOM unless the structure contains an element in "
                    "the spin method-consideration screen or the stage is Spin Polarised"
                ),
                "support_guard": "backend.calculations.registry.validate_stage_spec",
            },
        ],
        "advisory_only": [
            {
                "modifier": Modifier.DFT_U.value,
            },
        ],
    }


def _apply_spin_polarisation(
    stages: Iterable[StageSpec],
) -> tuple[tuple[StageSpec, ...], AppliedDefaultTreatment]:
    resolved_stages: list[StageSpec] = []
    stage_applications: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        resolved = _stage_with_modifier(stage, Modifier.SPIN_POLARIZED)
        _validate_automatic_stage(
            resolved,
            modifier=Modifier.SPIN_POLARIZED,
            consideration_id=SPIN_CONSIDERATION_ID,
        )
        resolved_stages.append(resolved)
        stage_applications.append(_stage_application(index, resolved))

    return tuple(resolved_stages), AppliedDefaultTreatment(
        consideration_id=SPIN_CONSIDERATION_ID,
        modifier=Modifier.SPIN_POLARIZED,
        display_name=modifier_display_name(Modifier.SPIN_POLARIZED),
        stage_indices=tuple(application["stage_index"] for application in stage_applications),
        stage_applications=tuple(stage_applications),
    )


def _apply_dispersion(
    stages: Iterable[StageSpec],
) -> tuple[tuple[StageSpec, ...], AppliedDefaultTreatment | None]:
    resolved_stages: list[StageSpec] = []
    stage_applications: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        if _stage_receives_default_dispersion(stage):
            resolved = _stage_with_default_dispersion(stage)
            _validate_automatic_stage(
                resolved,
                modifier=Modifier.DISPERSION,
                consideration_id=DISPERSION_CONSIDERATION_ID,
            )
            stage_applications.append(_stage_application(index, resolved))
        else:
            resolved = stage
        resolved_stages.append(resolved)

    treatment = None
    if stage_applications:
        treatment = AppliedDefaultTreatment(
            consideration_id=DISPERSION_CONSIDERATION_ID,
            modifier=Modifier.DISPERSION,
            display_name=modifier_display_name(Modifier.DISPERSION),
            stage_indices=tuple(
                application["stage_index"]
                for application in stage_applications
            ),
            stage_applications=tuple(stage_applications),
        )
    return tuple(resolved_stages), treatment


def _apply_soc(
    stages: Iterable[StageSpec],
) -> tuple[tuple[StageSpec, ...], AppliedDefaultTreatment | None]:
    resolved_stages: list[StageSpec] = []
    stage_applications: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        if stage.stage_type in SOC_EXCLUDED_STAGE_TYPES:
            resolved_stages.append(stage)
            continue
        resolved = _stage_with_modifier(stage, Modifier.SOC)
        _validate_automatic_stage(
            resolved,
            modifier=Modifier.SOC,
            consideration_id=SOC_CONSIDERATION_ID,
        )
        resolved_stages.append(resolved)
        stage_applications.append(_stage_application(index, resolved))

    treatment = None
    if stage_applications:
        treatment = AppliedDefaultTreatment(
            consideration_id=SOC_CONSIDERATION_ID,
            modifier=Modifier.SOC,
            display_name=modifier_display_name(Modifier.SOC),
            stage_indices=tuple(
                application["stage_index"]
                for application in stage_applications
            ),
            stage_applications=tuple(stage_applications),
        )
    return tuple(resolved_stages), treatment


def _stage_receives_default_dispersion(stage: StageSpec) -> bool:
    return (
        stage.theory is Theory.PBE
        and stage.stage_type in {StageType.RELAX, StageType.STATIC}
    )


def _stage_with_modifier(stage: StageSpec, modifier: Modifier) -> StageSpec:
    return StageSpec(
        stage_type=stage.stage_type,
        theory=stage.theory,
        modifiers=frozenset({*stage.modifiers, modifier}),
        label=stage.label,
        options=stage.options,
    )


def _stage_with_default_dispersion(stage: StageSpec) -> StageSpec:
    options = dict(stage.options or {})
    options.update(dispersion_option_payload(DEFAULT_DISPERSION_METHOD))
    return StageSpec(
        stage_type=stage.stage_type,
        theory=stage.theory,
        modifiers=frozenset({*stage.modifiers, Modifier.DISPERSION}),
        label=stage.label,
        options=options,
    )


def _validate_automatic_stage(
    stage: StageSpec,
    *,
    modifier: Modifier,
    consideration_id: str,
) -> None:
    try:
        validate_stage_spec(stage)
    except CalculationValidationError as exc:
        raise CalculationValidationError(
            (
                f"BMD automatic {modifier_display_name(modifier)} treatment could "
                f"not be applied to {stage_display_name(stage)} "
                f"({theory_display_name(stage.theory)})."
            ),
            suggestion=(
                "Choose Custom workflow to configure this treatment manually, "
                "or remove the triggering structure feature."
            ),
        ) from exc


def _stage_application(index: int, stage: StageSpec) -> dict[str, Any]:
    return {
        "stage_index": index,
        "stage_type": stage.stage_type.value,
        "stage_type_label": stage_display_name(stage),
        "theory": stage.theory.value,
        "theory_label": theory_display_name(stage.theory),
    }


def _method_consideration_payload(structure, *, workflow: WorkflowSpec) -> dict[str, Any]:
    from backend.calculations.method_considerations import method_consideration_payload

    return method_consideration_payload(structure, workflow=workflow)


def _json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe_value(value)
        for key, value in sorted(dict(mapping or {}).items(), key=lambda item: str(item[0]))
    }


def _json_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    raise TypeError(f"Value is not JSON-native: {type(value).__name__}")


__all__ = [
    "AUTOMATIC_APPLICATION_ADVISORY",
    "AUTOMATIC_APPLICATION_APPLIED",
    "AUTOMATIC_APPLICATION_NOT_APPLICABLE",
    "AppliedDefaultTreatment",
    "DISPERSION_CONSIDERATION_ID",
    "ResolvedDefaultWorkflow",
    "SOC_CONSIDERATION_ID",
    "SOC_EXCLUDED_STAGE_TYPES",
    "SOC_NOT_APPLICABLE_REASON",
    "SPIN_CONSIDERATION_ID",
    "automatic_default_treatment_policy",
    "resolve_default_treatments",
]
