from __future__ import annotations

from backend.calculations.models import CalculationSpec, WorkflowSpec
from backend.calculations.registry import (
    calculation_spec_from_legacy,
    legacy_potcar_functional_from_spec,
    validate_calculation_spec,
)


def build_calculation_flow(
    structure,
    spec: CalculationSpec | WorkflowSpec,
    *,
    label: str = "vasp_run",
    incar: dict | None = None,
    kpoints: dict | None = None,
    resources=None,
    potcar_functional: str | None = None,
    dft_u_expectations: dict | None = None,
):
    """
    Build a Jobflow Flow for a calculation intent.

    A ``WorkflowSpec`` is built by ``build_atomate2_flow_for_workflow_spec``
    (the path used for preview-parity and remote execution); a legacy
    ``CalculationSpec`` is validated and built by the single-calculation
    builders in ``backend.workflows``.
    """

    if isinstance(spec, WorkflowSpec):
        from backend.workflows import build_atomate2_flow_for_workflow_spec

        return build_atomate2_flow_for_workflow_spec(
            structure=structure,
            workflow_spec=spec,
            label=label,
            incar=incar,
            kpoints=kpoints,
            resources=resources,
            potcar_functional=potcar_functional or "PBE_64",
            dft_u_expectations=dft_u_expectations,
        )

    normalized = CalculationSpec(
        purpose=spec.purpose,
        theory=spec.theory,
        modifiers=spec.modifiers,
        label=spec.label,
    )

    normalized = validate_calculation_spec(normalized)
    legacy_potcar_functional = potcar_functional or legacy_potcar_functional_from_spec(
        normalized
    )

    from backend.workflows import build_atomate2_flow_for_spec

    return build_atomate2_flow_for_spec(
        structure=structure,
        spec=normalized,
        label=label,
        incar=incar,
        kpoints=kpoints,
        resources=resources,
        potcar_functional=legacy_potcar_functional,
    )


def build_calculation_flow_from_legacy(
    structure,
    workflow: str | None,
    *,
    label: str = "vasp_run",
    incar: dict | None = None,
    kpoints: dict | None = None,
    resources=None,
    potcar_functional: str | None = None,
):
    spec = calculation_spec_from_legacy(workflow, potcar_functional)
    return build_calculation_flow(
        structure,
        spec,
        label=label,
        incar=incar,
        kpoints=kpoints,
        resources=resources,
        potcar_functional=potcar_functional,
    )


__all__ = [
    "build_calculation_flow",
    "build_calculation_flow_from_legacy",
]
