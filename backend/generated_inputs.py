from __future__ import annotations

from backend.calculations.models import (
    CalculationSpec,
    Modifier,
    Purpose,
    StageSpec,
    StageType,
    WorkflowSpec,
)
from backend.calculations.registry import (
    calculation_stage_directories,
    calculation_spec_from_workflow_spec,
    stage_display_name,
    theory_display_name,
    validate_calculation_spec,
    validate_workflow_spec,
    workflow_spec_from_calculation_spec,
    workflow_stage_directories,
)
from backend.calculations.structure_constraints import reject_unsupported_structure_constraints
from backend.workflow_summary import calculation_plan_from_spec
from backend.workflows import (
    build_band_structure_input_set_generator,
    build_dos_input_set_generator,
    build_relax_input_set_generator,
    build_static_input_set_generator,
    build_vasp_input_set_for_spec,
    frozen_dft_u_for_stage,
    validate_input_set_for_modifiers,
    vasp_executable_for_modifiers,
    apply_stage_artifact_incar_settings,
    dispersion_method_for_stage,
    workflow_stage_artifact_policies,
)


def preview_generated_inputs(
    structure,
    spec: CalculationSpec | WorkflowSpec,
    *,
    incar: dict | None = None,
    kpoints: dict | None = None,
    resources=None,
    potcar_functional: str = "PBE_64",
) -> dict:
    """
    Return template-friendly VASP input previews for a calculation.

    The preview uses the same input-set generator path as workflow construction,
    with POTCAR generation switched to symbol-only mode so POTCAR files are not
    read locally and no HPC resources are contacted.
    """

    reject_unsupported_structure_constraints(structure)
    if isinstance(spec, WorkflowSpec):
        workflow_spec = validate_workflow_spec(spec)
    else:
        calculation_spec = validate_calculation_spec(spec)
        workflow_spec = workflow_spec_from_calculation_spec(calculation_spec)

    if len(workflow_spec.stages) > 1:
        return _preview_workflow_inputs(
            structure,
            workflow_spec,
            incar=incar,
            kpoints=kpoints,
            resources=resources,
            potcar_functional=potcar_functional,
        )

    calculation_spec = calculation_spec_from_workflow_spec(workflow_spec)
    if calculation_spec is None:
        input_set = _input_set_for_stage(
            structure,
            workflow_spec.stages[0],
            incar=incar,
            kpoints=kpoints,
            resources=resources,
            potcar_functional=potcar_functional,
        )
        validate_input_set_for_modifiers(
            input_set,
            spec=_calculation_spec_for_stage(workflow_spec.stages[0]),
        )
        return {
            "incar": _input_text(input_set.incar),
            "kpoints": _input_text(input_set.kpoints),
            "poscar": _input_text(input_set.poscar),
            "vasp_executable": vasp_executable_for_modifiers(
                workflow_spec.stages[0].modifiers
            ),
        }

    input_set = build_vasp_input_set_for_spec(
        structure,
        calculation_spec,
        incar=incar,
        kpoints=kpoints,
        resources=resources,
        potcar_functional=potcar_functional,
    )

    return {
        "incar": _input_text(input_set.incar),
        "kpoints": _input_text(input_set.kpoints),
        "poscar": _input_text(input_set.poscar),
        "vasp_executable": vasp_executable_for_modifiers(calculation_spec.modifiers),
    }


def generated_input_stage_previews(
    structure,
    spec: CalculationSpec | WorkflowSpec,
    *,
    incar: dict | None = None,
    kpoints: dict | None = None,
    resources=None,
    potcar_functional: str = "PBE_64",
) -> tuple[dict, ...]:
    reject_unsupported_structure_constraints(structure)
    if isinstance(spec, WorkflowSpec):
        workflow_spec = validate_workflow_spec(spec)
    else:
        calculation_spec = validate_calculation_spec(spec)
        workflow_spec = workflow_spec_from_calculation_spec(calculation_spec)

    stage_directories = workflow_stage_directories(workflow_spec)
    stage_artifacts = workflow_stage_artifact_policies(workflow_spec)
    stage_previews = []

    for index, stage in enumerate(workflow_spec.stages):
        input_set = _input_set_for_stage(
            structure,
            stage,
            incar=incar,
            kpoints=kpoints,
            resources=resources,
            potcar_functional=potcar_functional,
            artifact_policy=stage_artifacts[index],
        )
        validate_input_set_for_modifiers(
            input_set,
            spec=_calculation_spec_for_stage(stage),
        )
        stage_previews.append(
            {
                "index": index + 1,
                "directory": stage_directories[index] if stage_directories else None,
                "label": stage_display_name(stage),
                "theory_label": theory_display_name(stage.theory),
                "vasp_executable": vasp_executable_for_modifiers(stage.modifiers),
                "artifact_policy": stage_artifacts[index],
                "stage_spec": stage,
                "input_set": input_set,
            }
        )

    return tuple(stage_previews)


def _preview_workflow_inputs(
    structure,
    workflow_spec: WorkflowSpec,
    *,
    incar: dict | None,
    kpoints: dict | None,
    resources,
    potcar_functional: str,
) -> dict:
    stage_previews = generated_input_stage_previews(
        structure,
        workflow_spec,
        incar=incar,
        kpoints=kpoints,
        resources=resources,
        potcar_functional=potcar_functional,
    )

    return {
        "incar": _combined_stage_input_text(stage_previews, "incar"),
        "kpoints": _combined_stage_input_text(stage_previews, "kpoints"),
        "poscar": _combined_stage_input_text(stage_previews, "poscar"),
        "vasp_executables": [
            {
                "stage": stage["index"],
                "label": stage["label"],
                "executable": stage["vasp_executable"],
            }
            for stage in stage_previews
        ],
    }


def _input_set_for_stage(
    structure,
    stage: StageSpec,
    *,
    incar: dict | None,
    kpoints,
    resources,
    potcar_functional: str,
    artifact_policy: dict | None = None,
):
    user_incar = dict(incar or {})
    user_incar.update(dict(stage.options or {}).get("incar") or {})
    user_incar = apply_stage_artifact_incar_settings(
        user_incar,
        artifact_policy=artifact_policy,
    )
    stage_kpoints = dict(stage.options or {}).get("kpoints", kpoints)
    spin_polarized = Modifier.SPIN_POLARIZED in stage.modifiers

    if stage.stage_type is StageType.RELAX:
        generator = build_relax_input_set_generator(
            structure,
            isif=2 if Modifier.IONS_ONLY in stage.modifiers else None,
            theory=stage.theory,
            spin_polarized=spin_polarized,
            modifiers=stage.modifiers,
            resources=resources,
            incar=user_incar,
            kpoints=stage_kpoints,
            potcar_functional=potcar_functional,
            dispersion_method=dispersion_method_for_stage(stage),
            frozen_dft_u=frozen_dft_u_for_stage(stage),
        )
    elif stage.stage_type is StageType.STATIC:
        generator = build_static_input_set_generator(
            structure,
            intent="final",
            theory=stage.theory,
            spin_polarized=spin_polarized,
            modifiers=stage.modifiers,
            resources=resources,
            incar=user_incar,
            kpoints=stage_kpoints,
            potcar_functional=potcar_functional,
            dispersion_method=dispersion_method_for_stage(stage),
            frozen_dft_u=frozen_dft_u_for_stage(stage),
        )
    elif stage.stage_type is StageType.DOS:
        generator = build_dos_input_set_generator(
            structure,
            theory=stage.theory,
            spin_polarized=spin_polarized,
            modifiers=stage.modifiers,
            resources=resources,
            incar=user_incar,
            kpoints=stage_kpoints,
            potcar_functional=potcar_functional,
        )
    elif stage.stage_type is StageType.BAND_STRUCTURE:
        generator = build_band_structure_input_set_generator(
            structure,
            theory=stage.theory,
            spin_polarized=spin_polarized,
            modifiers=stage.modifiers,
            resources=resources,
            incar=user_incar,
            kpoints=stage_kpoints,
            potcar_functional=potcar_functional,
        )
    else:
        raise ValueError(f"Unsupported stage type: {stage.stage_type.value}")

    return generator.get_input_set(structure, potcar_spec=True)


def _calculation_spec_for_stage(stage: StageSpec) -> CalculationSpec:
    purposes = {
        StageType.RELAX: Purpose.RELAX,
        StageType.STATIC: Purpose.STATIC,
        StageType.DOS: Purpose.DOS,
        StageType.BAND_STRUCTURE: Purpose.BAND_STRUCTURE,
    }
    return CalculationSpec(
        purposes[stage.stage_type],
        stage.theory,
        stage.modifiers,
        stage.label,
    )


def _preview_relax_static_inputs(
    structure,
    spec: CalculationSpec,
    *,
    incar: dict | None,
    kpoints: dict | None,
    resources,
    potcar_functional: str,
) -> dict:
    calculation_modifiers = spec.modifiers
    spin_polarized = Modifier.SPIN_POLARIZED in calculation_modifiers
    relax_generator = build_relax_input_set_generator(
        structure,
        theory=spec.theory,
        spin_polarized=spin_polarized,
        modifiers=calculation_modifiers,
        resources=resources,
        incar=incar,
        kpoints=kpoints,
        potcar_functional=potcar_functional,
    )
    static_generator = build_static_input_set_generator(
        structure,
        intent="final",
        theory=spec.theory,
        spin_polarized=spin_polarized,
        modifiers=calculation_modifiers,
        resources=resources,
        incar=incar,
        kpoints=kpoints,
        potcar_functional=potcar_functional,
    )

    stage_directories = calculation_stage_directories(spec)
    stage_labels = calculation_plan_from_spec(spec)
    theory_label = theory_display_name(spec.theory)
    stage_previews = [
        {
            "index": 1,
            "directory": stage_directories[0],
            "label": stage_labels[0],
            "theory_label": theory_label,
            "vasp_executable": vasp_executable_for_modifiers(spec.modifiers),
            "input_set": relax_generator.get_input_set(structure, potcar_spec=True),
        },
        {
            "index": 2,
            "directory": stage_directories[1],
            "label": stage_labels[1],
            "theory_label": theory_label,
            "vasp_executable": vasp_executable_for_modifiers(spec.modifiers),
            "input_set": static_generator.get_input_set(structure, potcar_spec=True),
        },
    ]
    validate_input_set_for_modifiers(stage_previews[-1]["input_set"], spec=spec)

    return {
        "incar": _combined_stage_input_text(stage_previews, "incar", theory_label),
        "kpoints": _combined_stage_input_text(stage_previews, "kpoints", theory_label),
        "poscar": _combined_stage_input_text(stage_previews, "poscar", theory_label),
        "vasp_executables": [
            {
                "stage": stage["index"],
                "label": stage["label"],
                "executable": stage["vasp_executable"],
            }
            for stage in stage_previews
        ],
    }


def _combined_stage_input_text(
    stage_previews: list[dict],
    key: str,
    theory_label: str | None = None,
) -> str:
    sections = []
    for stage in stage_previews:
        label = stage["label"]
        stage_theory_label = stage.get("theory_label") or theory_label
        sections.append(
            "\n".join(
                [
                    f"# Stage {stage['index']} - {label} ({stage_theory_label})",
                    f"# VASP executable - {stage['vasp_executable']}",
                    _input_text(getattr(stage["input_set"], key)),
                ]
            )
        )

    return "\n\n".join(sections)


def _input_text(input_object) -> str:
    return str(input_object).rstrip()


__all__ = [
    "generated_input_stage_previews",
    "preview_generated_inputs",
]
