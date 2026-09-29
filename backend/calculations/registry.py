from __future__ import annotations

from itertools import combinations

from backend.calculations.models import (
    CalculationSpec,
    Modifier,
    Purpose,
    StageSpec,
    StageType,
    Theory,
    WorkflowSpec,
)
from backend.calculations.dispersion import (
    DISPERSION_OPTION_KEY,
    dispersion_method_from_options,
)
from backend.calculations.dft_u_policy import FROZEN_OPTION_KEY as DFT_U_OPTION_KEY
from backend.calculations.vasp_stage_definitions import (
    stage_inherits_previous_band_count,
    stage_reads_previous_charge_density,
)
from backend.calculations.theory_policy import (
    CalculationStage,
    theory_default_potcar_functional,
    theory_supported_purposes,
    theory_supported_stages,
)


class CalculationValidationError(ValueError):
    def __init__(self, message: str, *, suggestion: str | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion


_ACTIVE_UI_MODIFIERS = (
    Modifier.SPIN_POLARIZED,
    Modifier.DFT_U,
    Modifier.DISPERSION,
    Modifier.GAMMA_ONLY,
)

_PBE_RELAX_STATIC_MODIFIERS = _ACTIVE_UI_MODIFIERS

_PBE_ANALYSIS_WORKFLOW_MODIFIERS = (
    Modifier.SPIN_POLARIZED,
    Modifier.DFT_U,
    Modifier.GAMMA_ONLY,
)

_PBE_STATIC_MODIFIERS = (
    Modifier.SPIN_POLARIZED,
    Modifier.SOC,
    Modifier.DFT_U,
    Modifier.DISPERSION,
    Modifier.GAMMA_ONLY,
)

_HSE06_SINGLE_STAGE_MODIFIERS = (
    Modifier.SPIN_POLARIZED,
    Modifier.GAMMA_ONLY,
)

_HSE06_STATIC_MODIFIERS = (
    Modifier.SPIN_POLARIZED,
    Modifier.SOC,
    Modifier.GAMMA_ONLY,
)

_HSE06_DOS_MODIFIERS = _HSE06_STATIC_MODIFIERS

_HSE06_BAND_STRUCTURE_MODIFIERS = (
    Modifier.SPIN_POLARIZED,
    Modifier.SOC,
)


def _modifier_subsets(
    modifiers: tuple[Modifier, ...],
) -> tuple[frozenset[Modifier], ...]:
    return tuple(
        frozenset(subset)
        for size in range(len(modifiers) + 1)
        for subset in combinations(modifiers, size)
    )


def _build_supported_compatibility_workflows() -> dict[
    tuple[Purpose, Theory, frozenset[Modifier]],
    str,
]:
    supported: dict[tuple[Purpose, Theory, frozenset[Modifier]], str] = {}

    for modifiers in _modifier_subsets(_PBE_STATIC_MODIFIERS):
        supported[(Purpose.STATIC, Theory.PBE, modifiers)] = "static"

    for modifiers in _modifier_subsets(_PBE_RELAX_STATIC_MODIFIERS):
        supported[(Purpose.RELAX, Theory.PBE, modifiers)] = "relax"
        supported[(Purpose.RELAX_STATIC, Theory.PBE, modifiers)] = "relax_static"
        supported[(Purpose.DOUBLE_RELAX, Theory.PBE, modifiers)] = "double_relax"
        supported[
            (
                Purpose.RELAX,
                Theory.PBE,
                frozenset({*modifiers, Modifier.IONS_ONLY}),
            )
        ] = "relax_ions"

    for modifiers in _modifier_subsets(_PBE_ANALYSIS_WORKFLOW_MODIFIERS):
        supported[(Purpose.DOS, Theory.PBE, modifiers)] = "dos"
        if Modifier.GAMMA_ONLY not in modifiers:
            supported[(Purpose.BAND_STRUCTURE, Theory.PBE, modifiers)] = "band_structure"

    for modifiers in _modifier_subsets(_HSE06_SINGLE_STAGE_MODIFIERS):
        supported[(Purpose.RELAX, Theory.HSE06, modifiers)] = "relax"
        supported[(Purpose.RELAX_STATIC, Theory.HSE06, modifiers)] = "relax_static"

    for modifiers in _modifier_subsets(_HSE06_STATIC_MODIFIERS):
        supported[(Purpose.STATIC, Theory.HSE06, modifiers)] = "static"

    return supported


_SUPPORTED_COMPATIBILITY_WORKFLOWS = _build_supported_compatibility_workflows()

_LEGACY_WORKFLOW_SPECS = {
    "static": CalculationSpec(Purpose.STATIC, Theory.PBE),
    "relax": CalculationSpec(Purpose.RELAX, Theory.PBE),
    "relax_static": CalculationSpec(Purpose.RELAX_STATIC, Theory.PBE),
    "double_relax": CalculationSpec(Purpose.DOUBLE_RELAX, Theory.PBE),
    "dos": CalculationSpec(Purpose.DOS, Theory.PBE),
    "band_structure": CalculationSpec(Purpose.BAND_STRUCTURE, Theory.PBE),
    "relax_ions": CalculationSpec(Purpose.RELAX, Theory.PBE, {Modifier.IONS_ONLY}),
}

_LEGACY_POTCAR_THEORIES = {
    "PBE": Theory.PBE,
    "PBE_52": Theory.PBE,
    "PBE_54": Theory.PBE,
    "PBE_64": Theory.PBE,
}

_PURPOSE_DISPLAY_NAMES = {
    Purpose.RELAX: "Geometry Optimisation",
    Purpose.RELAX_STATIC: "Geometry Optimisation + Static Energy",
    Purpose.DOUBLE_RELAX: "Double Geometry Optimisation",
    Purpose.STATIC: "Static Energy",
    Purpose.DOS: "Density of States",
    Purpose.BAND_STRUCTURE: "Band Structure",
    Purpose.DIELECTRIC: "Dielectric Properties",
}

_STAGE_DISPLAY_NAMES = {
    StageType.RELAX: "Geometry Optimisation",
    StageType.STATIC: "Static Energy",
    StageType.DOS: "Density of States",
    StageType.BAND_STRUCTURE: "Band Structure",
}

_STAGE_DESCRIPTIONS = {
    StageType.RELAX: "Optimise the atomic structure.",
    StageType.STATIC: "Calculate a converged single-point total energy.",
    StageType.DOS: "Calculate the electronic density of states after a static calculation.",
    StageType.BAND_STRUCTURE: "Calculate the electronic band structure after a static calculation.",
}

_STAGE_PURPOSES = {
    StageType.RELAX: Purpose.RELAX,
    StageType.STATIC: Purpose.STATIC,
    StageType.DOS: Purpose.DOS,
    StageType.BAND_STRUCTURE: Purpose.BAND_STRUCTURE,
}

_PURPOSE_STAGE_TYPES = {
    Purpose.RELAX: (StageType.RELAX,),
    Purpose.STATIC: (StageType.STATIC,),
    Purpose.RELAX_STATIC: (StageType.RELAX, StageType.STATIC),
    Purpose.DOUBLE_RELAX: (StageType.RELAX, StageType.RELAX),
    Purpose.DOS: (StageType.RELAX, StageType.STATIC, StageType.DOS),
    Purpose.BAND_STRUCTURE: (
        StageType.RELAX,
        StageType.STATIC,
        StageType.BAND_STRUCTURE,
    ),
}

_TERMINAL_ANALYSIS_STAGES = {
    StageType.DOS,
    StageType.BAND_STRUCTURE,
}

_PURPOSE_DESCRIPTIONS = {
    Purpose.RELAX: "Optimise the atomic structure before analysis or follow-up calculations.",
    Purpose.RELAX_STATIC: "Optimise the structure and then calculate a final static energy.",
    Purpose.DOUBLE_RELAX: "Run two consecutive geometry optimisations, using the first final structure as the second starting structure.",
    Purpose.STATIC: "Calculate a single-point total energy for the supplied structure.",
    Purpose.DOS: "Calculate the electronic density of states.",
    Purpose.BAND_STRUCTURE: "Calculate the electronic band structure.",
    Purpose.DIELECTRIC: "Calculate dielectric response properties.",
}

_PURPOSE_STAGE_DIRECTORIES = {
    Purpose.RELAX_STATIC: ("stage_01", "stage_02"),
    Purpose.DOUBLE_RELAX: ("relax_01", "relax_02"),
    Purpose.DOS: ("stage_01", "stage_02", "stage_03"),
    Purpose.BAND_STRUCTURE: ("stage_01", "stage_02", "stage_03"),
}

_THEORY_DISPLAY_NAMES = {
    Theory.PBE: "PBE",
    Theory.R2SCAN: "r2SCAN",
    Theory.HSE06: "HSE06",
}

_MODIFIER_DISPLAY_NAMES = {
    Modifier.SPIN_POLARIZED: "Spin Polarised",
    Modifier.SOC: "Spin-Orbit Coupling (SOC)",
    Modifier.DFT_U: "DFT+U",
    Modifier.DISPERSION: "van der Waals correction",
    Modifier.GAMMA_ONLY: "Gamma-only",
    Modifier.IONS_ONLY: "Ions only",
}

_UNIMPLEMENTED_TOOLTIP = "Coming soon"
_MODIFIER_TOOLTIPS = {
    Modifier.SOC: (
        "SOC is available for PBE and HSE06 Static Energy stages and HSE06 "
        "Density of States and Band Structure stages, and runs with vasp_ncl."
    ),
    Modifier.DFT_U: "DFT+U is applied only when explicitly selected.",
    Modifier.DISPERSION: "van der Waals correction for PBE Geometry Optimisation and Static Energy stages.",
}

_UI_HIDDEN_MODIFIERS = {
    Modifier.GAMMA_ONLY,
    Modifier.IONS_ONLY,
}

_UI_MODIFIER_ORDER = (
    Modifier.SPIN_POLARIZED,
    Modifier.SOC,
    Modifier.DFT_U,
    Modifier.DISPERSION,
    Modifier.GAMMA_ONLY,
)


def _recommended_workflow_recipes() -> tuple[dict, ...]:
    recipes = (
        (
            "relax",
            "Geometry Optimisation",
            "Optimise the supplied structure.",
            WorkflowSpec([StageSpec(StageType.RELAX)], recipe="relax"),
        ),
        (
            "static",
            "Static Energy",
            "Calculate a single-point total energy.",
            WorkflowSpec([StageSpec(StageType.STATIC)], recipe="static"),
        ),
        (
            "relax_static",
            "Geometry Optimisation + Static Energy",
            "Optimise the structure, then calculate a final static energy.",
            WorkflowSpec(
                [
                    StageSpec(StageType.RELAX),
                    StageSpec(StageType.STATIC),
                ],
                recipe="relax_static",
            ),
        ),
        (
            "double_relax",
            "Double Geometry Optimisation",
            "Run two consecutive geometry optimisations.",
            WorkflowSpec(
                [
                    StageSpec(StageType.RELAX),
                    StageSpec(StageType.RELAX),
                ],
                recipe="double_relax",
            ),
        ),
        (
            "dos",
            "Density of States",
            "Optimise, run a static calculation, then calculate the density of states.",
            WorkflowSpec(
                [
                    StageSpec(StageType.RELAX),
                    StageSpec(StageType.STATIC),
                    StageSpec(StageType.DOS),
                ],
                recipe="dos",
            ),
        ),
        (
            "band_structure",
            "Band Structure",
            "Optimise, run a static calculation, then calculate the band structure.",
            WorkflowSpec(
                [
                    StageSpec(StageType.RELAX),
                    StageSpec(StageType.STATIC),
                    StageSpec(StageType.BAND_STRUCTURE),
                ],
                recipe="band_structure",
            ),
        ),
    )
    return tuple(
        {
            "value": value,
            "label": label,
            "description": description,
            "workflow_spec": validate_workflow_spec(workflow_spec).to_dict(),
        }
        for value, label, description, workflow_spec in recipes
    )


def _desired_output_workflows() -> tuple[tuple[str, str, str, WorkflowSpec | None], ...]:
    return (
        (
            "energy_only",
            "Energy only",
            "Calculate a PBE static total energy.",
            WorkflowSpec(
                [StageSpec(StageType.STATIC, Theory.PBE)],
                recipe="energy_only",
            ),
        ),
        (
            "relaxed_structure",
            "Relaxed structure",
            "Run two successive PBE geometry optimisations for the final relaxed structure.",
            WorkflowSpec(
                [
                    StageSpec(StageType.RELAX, Theory.PBE),
                    StageSpec(StageType.RELAX, Theory.PBE),
                ],
                recipe="relaxed_structure",
            ),
        ),
        (
            "electronic_dos",
            "Electronic density of states",
            "Use BMD Compute's default staged workflow for an HSE06 density of states.",
            WorkflowSpec(
                [
                    StageSpec(StageType.RELAX, Theory.PBE),
                    StageSpec(StageType.STATIC, Theory.HSE06),
                    StageSpec(StageType.DOS, Theory.HSE06),
                ],
                recipe="electronic_dos",
            ),
        ),
        (
            "electronic_band_structure",
            "Electronic band structure",
            "Use BMD Compute's default staged workflow for an HSE06 band structure.",
            WorkflowSpec(
                [
                    StageSpec(StageType.RELAX, Theory.PBE),
                    StageSpec(StageType.STATIC, Theory.HSE06),
                    StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
                ],
                recipe="electronic_band_structure",
            ),
        ),
        (
            "custom",
            "Custom workflow",
            "Configure stages and theories explicitly.",
            None,
        ),
    )


def _desired_output_options() -> tuple[dict, ...]:
    options = []
    for value, label, description, workflow_spec in _desired_output_workflows():
        options.append(
            {
                "value": value,
                "label": label,
                "description": description,
                "workflow_spec": (
                    validate_workflow_spec(workflow_spec).to_dict()
                    if workflow_spec is not None
                    else None
                ),
            }
        )
    return tuple(options)


def desired_output_workflow_spec(value: str | None) -> WorkflowSpec | None:
    output = (value or "").strip()
    for candidate, _, _, workflow_spec in _desired_output_workflows():
        if candidate == output and workflow_spec is not None:
            return validate_workflow_spec(workflow_spec)
    return None


def desired_output_from_workflow_spec(workflow: WorkflowSpec) -> str:
    normalized = validate_workflow_spec(workflow)
    shape = tuple((stage.stage_type, stage.theory) for stage in normalized.stages)
    for value, _, _, workflow_spec in _desired_output_workflows():
        if workflow_spec is None:
            continue
        candidate = validate_workflow_spec(workflow_spec)
        candidate_shape = tuple(
            (stage.stage_type, stage.theory)
            for stage in candidate.stages
        )
        if shape == candidate_shape:
            return value
    return "custom"


def validate_calculation_spec(spec: CalculationSpec) -> CalculationSpec:
    normalized = CalculationSpec(
        purpose=spec.purpose,
        theory=spec.theory,
        modifiers=spec.modifiers,
        label=spec.label,
    )
    key = _compatibility_key(normalized)

    if key not in _SUPPORTED_COMPATIBILITY_WORKFLOWS:
        raise _unsupported_combination_error(normalized, key)

    return normalized


def validate_stage_spec(stage: StageSpec) -> StageSpec:
    normalized = StageSpec(
        stage_type=stage.stage_type,
        theory=stage.theory,
        modifiers=stage.modifiers,
        label=stage.label,
        options=stage.options,
    )
    _validate_stage_support(normalized)

    return normalized


def validate_workflow_spec(workflow: WorkflowSpec) -> WorkflowSpec:
    normalized = WorkflowSpec(
        stages=[
            validate_stage_spec(stage)
            for stage in WorkflowSpec(
                stages=workflow.stages,
                label=workflow.label,
                recipe=workflow.recipe,
            ).stages
        ],
        label=workflow.label,
        recipe=workflow.recipe,
    )

    if not normalized.stages:
        raise CalculationValidationError(
            "Add at least one calculation stage.",
            suggestion="Choose a recommended workflow, or add a stage in Custom Workflow.",
        )

    _validate_dispersion_workflow_consistency(normalized)
    _validate_previous_stage_soc_consistency(normalized)

    for index, stage in enumerate(normalized.stages):
        stage_number = index + 1
        if stage.stage_type in _TERMINAL_ANALYSIS_STAGES and index != len(normalized.stages) - 1:
            raise CalculationValidationError(
                f"{stage_display_name(stage)} must be the final workflow stage.",
                suggestion="Move this stage to the end of the workflow.",
            )

        if stage.stage_type not in _TERMINAL_ANALYSIS_STAGES:
            continue

        if index == 0 or normalized.stages[index - 1].stage_type is not StageType.STATIC:
            raise CalculationValidationError(
                f"{stage_display_name(stage)} must follow a converged Static Energy stage.",
                suggestion=(
                    f"Add Static Energy immediately before stage {stage_number}, "
                    "or choose the recommended workflow."
                ),
            )

        previous_stage = normalized.stages[index - 1]
        if not _analysis_stage_accepts_precursor(stage, previous_stage):
            raise CalculationValidationError(
                f"{stage_display_name(stage)} must use the same level of theory as the preceding Static Energy stage.",
                suggestion=(
                    "Use matching levels of theory for the static and analysis stages, "
                    "or choose a validated recommended workflow."
                ),
            )

    return normalized


def _validate_previous_stage_soc_consistency(workflow: WorkflowSpec) -> None:
    # Two consumers of previous-stage electronic data must agree with their
    # producer on SOC:
    # - a stage that restarts from the previous fixed charge density
    #   (ICHARG=11) reads that CHGCAR as-is, and a collinear and a
    #   non-collinear (SOC) CHGCAR are different objects;
    # - a stage whose atomate2 generator sizes NBANDS from the previous
    #   vasprun.xml (NonSCF and HSE06 DOS/Band Structure) would get roughly
    #   half the bands a non-collinear run needs, or twice what a collinear run
    #   needs, when SOC differs.
    # Stages that only inherit the structure are unaffected.
    for index, (previous_stage, current_stage) in enumerate(
        zip(workflow.stages, workflow.stages[1:]),
        start=2,
    ):
        reads_charge_density = stage_reads_previous_charge_density(
            current_stage.stage_type,
            current_stage.theory,
        )
        inherits_band_count = stage_inherits_previous_band_count(
            current_stage.stage_type,
            current_stage.theory,
        )
        if not reads_charge_density and not inherits_band_count:
            continue
        if reads_charge_density and (
            (Modifier.DFT_U in previous_stage.modifiers)
            != (Modifier.DFT_U in current_stage.modifiers)
        ):
            producer_u = "uses" if Modifier.DFT_U in previous_stage.modifiers else "does not use"
            consumer_u = "uses" if Modifier.DFT_U in current_stage.modifiers else "does not use"
            raise CalculationValidationError(
                (
                    f"{stage_display_name(current_stage)} (stage {index}) restarts from "
                    f"the fixed charge density of stage {index - 1}, which {producer_u} "
                    f"DFT+U, but stage {index} {consumer_u} DFT+U."
                ),
                suggestion=(
                    "Use the same DFT+U setting on both stages, or add a Static Energy "
                    "stage with the matching DFT+U setting immediately before this stage."
                ),
            )
        previous_soc = Modifier.SOC in previous_stage.modifiers
        current_soc = Modifier.SOC in current_stage.modifiers
        if previous_soc == current_soc:
            continue
        producer = "uses" if previous_soc else "does not use"
        consumer = "uses" if current_soc else "does not use"
        dependency = (
            f"restarts from the fixed charge density of stage {index - 1}"
            if reads_charge_density
            else f"takes its band count (NBANDS) from stage {index - 1}"
        )
        raise CalculationValidationError(
            (
                f"{stage_display_name(current_stage)} (stage {index}) {dependency}, "
                f"which {producer} Spin-Orbit Coupling (SOC), but stage {index} "
                f"{consumer} SOC."
            ),
            suggestion=(
                "Use the same SOC setting on both stages, or add a Static Energy "
                "stage with the matching SOC setting immediately before this stage."
            ),
        )


def _analysis_stage_accepts_precursor(
    stage: StageSpec,
    previous_stage: StageSpec,
) -> bool:
    if previous_stage.theory is stage.theory:
        return True

    if (
        stage.stage_type is StageType.DOS
        and stage.theory is Theory.HSE06
        and previous_stage.theory in {Theory.PBE, Theory.HSE06}
    ):
        return True

    return False


def calculation_spec_from_legacy(
    workflow: str | None,
    potcar_functional: str | None = None,
) -> CalculationSpec:
    workflow_name = (workflow or "static").strip().lower()
    try:
        spec = _LEGACY_WORKFLOW_SPECS[workflow_name]
    except KeyError as exc:
        raise CalculationValidationError(f"Unsupported legacy workflow: {workflow!r}") from exc

    theory = _theory_from_legacy_potcar(potcar_functional)
    return validate_calculation_spec(
        CalculationSpec(
            purpose=spec.purpose,
            theory=theory,
            modifiers=spec.modifiers,
            label=spec.label,
        )
    )


def calculation_spec_from_flow_spec(flow_spec: dict | None) -> CalculationSpec:
    values = dict(flow_spec or {})
    if values.get("workflow_spec") or values.get("stages"):
        calculation_spec = calculation_spec_from_workflow_spec(
            workflow_spec_from_flow_spec(values)
        )
        if calculation_spec is None:
            raise CalculationValidationError(
                "This stage workflow cannot be represented as a legacy calculation.",
                suggestion="Use the stage workflow specification for this calculation.",
            )
        return calculation_spec

    if values.get("calculation_spec"):
        return validate_calculation_spec(
            CalculationSpec.from_dict(values.get("calculation_spec"))
        )

    return calculation_spec_from_legacy(
        values.get("workflow"),
        values.get("potcar_functional"),
    )


def workflow_spec_from_calculation_spec(spec: CalculationSpec) -> WorkflowSpec:
    normalized = validate_calculation_spec(spec)
    try:
        stage_types = _PURPOSE_STAGE_TYPES[normalized.purpose]
    except KeyError as exc:
        raise CalculationValidationError(
            f"{calculation_display_name(normalized)} is not available as a stage workflow yet.",
            suggestion="Choose one of the supported scientific workflows.",
        ) from exc

    return validate_workflow_spec(
        WorkflowSpec(
            [
                StageSpec(
                    stage_type=stage_type,
                    theory=normalized.theory,
                    modifiers=normalized.modifiers,
                )
                for stage_type in stage_types
            ],
            label=normalized.label,
            recipe=legacy_workflow_from_spec(normalized),
        )
    )


def workflow_spec_from_legacy(
    workflow: str | None,
    potcar_functional: str | None = None,
) -> WorkflowSpec:
    return workflow_spec_from_calculation_spec(
        calculation_spec_from_legacy(workflow, potcar_functional)
    )


def workflow_spec_from_flow_spec(flow_spec: dict | None) -> WorkflowSpec:
    values = dict(flow_spec or {})
    if values.get("workflow_spec"):
        return validate_workflow_spec(
            WorkflowSpec.from_dict(values.get("workflow_spec"))
        )

    if values.get("stages"):
        return validate_workflow_spec(
            WorkflowSpec(
                stages=values.get("stages"),
                label=values.get("label"),
                recipe=values.get("workflow"),
            )
        )

    return workflow_spec_from_calculation_spec(
        calculation_spec_from_flow_spec(values)
    )


def calculation_spec_from_workflow_spec(workflow: WorkflowSpec) -> CalculationSpec | None:
    normalized = validate_workflow_spec(workflow)
    stages = normalized.stages
    stage_types = tuple(stage.stage_type for stage in stages)
    if any(stage.options for stage in stages):
        return None

    def same_stage_policy() -> tuple[Theory, frozenset[Modifier]] | None:
        if not stages:
            return None
        theory = stages[0].theory
        modifiers = stages[0].modifiers
        if all(stage.theory is theory and stage.modifiers == modifiers for stage in stages):
            return theory, modifiers
        return None

    shared_policy = same_stage_policy()
    if stage_types == (StageType.RELAX,):
        stage = stages[0]
        return validate_calculation_spec(
            CalculationSpec(Purpose.RELAX, stage.theory, stage.modifiers, normalized.label)
        )

    if stage_types == (StageType.STATIC,):
        stage = stages[0]
        return validate_calculation_spec(
            CalculationSpec(Purpose.STATIC, stage.theory, stage.modifiers, normalized.label)
        )

    if stage_types == (StageType.RELAX, StageType.STATIC) and shared_policy:
        theory, modifiers = shared_policy
        return validate_calculation_spec(
            CalculationSpec(Purpose.RELAX_STATIC, theory, modifiers, normalized.label)
        )

    if stage_types == (StageType.RELAX, StageType.RELAX) and shared_policy:
        theory, modifiers = shared_policy
        try:
            return validate_calculation_spec(
                CalculationSpec(Purpose.DOUBLE_RELAX, theory, modifiers, normalized.label)
            )
        except CalculationValidationError:
            return None

    if stage_types == (StageType.RELAX, StageType.STATIC, StageType.DOS) and shared_policy:
        theory, modifiers = shared_policy
        try:
            return validate_calculation_spec(
                CalculationSpec(Purpose.DOS, theory, modifiers, normalized.label)
            )
        except CalculationValidationError:
            return None

    if (
        stage_types
        == (StageType.RELAX, StageType.STATIC, StageType.BAND_STRUCTURE)
        and shared_policy
    ):
        theory, modifiers = shared_policy
        try:
            return validate_calculation_spec(
                CalculationSpec(
                    Purpose.BAND_STRUCTURE,
                    theory,
                    modifiers,
                    normalized.label,
                )
            )
        except CalculationValidationError:
            return None

    return None


def legacy_workflow_from_spec(spec: CalculationSpec) -> str:
    normalized = validate_calculation_spec(spec)
    return _SUPPORTED_COMPATIBILITY_WORKFLOWS[_compatibility_key(normalized)]


def legacy_potcar_functional_from_spec(spec: CalculationSpec) -> str:
    normalized = validate_calculation_spec(spec)
    try:
        return theory_default_potcar_functional(normalized.theory)
    except ValueError as exc:
        raise CalculationValidationError(
            f"No legacy POTCAR functional for theory: {normalized.theory.value}"
        ) from exc


def calculation_stage_directories(spec: CalculationSpec) -> tuple[str, ...]:
    return workflow_stage_directories(workflow_spec_from_calculation_spec(spec))


def calculation_result_stage_directory(spec: CalculationSpec) -> str | None:
    stage_directories = calculation_stage_directories(spec)
    if not stage_directories:
        return None

    return stage_directories[-1]


def workflow_stage_directories(workflow: WorkflowSpec) -> tuple[str, ...]:
    normalized = validate_workflow_spec(workflow)
    if len(normalized.stages) <= 1:
        return ()

    if (
        len(normalized.stages) == 2
        and all(stage.stage_type is StageType.RELAX for stage in normalized.stages)
        and all(stage.theory is Theory.PBE for stage in normalized.stages)
    ):
        return ("relax_01", "relax_02")

    return tuple(
        f"stage_{index:02d}"
        for index in range(1, len(normalized.stages) + 1)
    )


# Stage options carry only BMD-supported treatment settings, never raw VASP
# input. ``dispersion`` is the user-selectable van der Waals method;
# ``dft_u`` holds the frozen +U parameters that automatic Desired Output
# resolution writes. Anything else - including ``incar`` and ``kpoints`` - is
# rejected, so a workflow spec cannot inject INCAR or KPOINTS values.
SUPPORTED_STAGE_OPTION_KEYS = frozenset({DISPERSION_OPTION_KEY, DFT_U_OPTION_KEY})
# Keys a client may supply in a Custom workflow. Frozen +U parameters are only
# produced by BMD Compute's automatic treatment resolution.
USER_STAGE_OPTION_KEYS = frozenset({DISPERSION_OPTION_KEY})


def _validate_stage_option_keys(stage: StageSpec, allowed=SUPPORTED_STAGE_OPTION_KEYS) -> None:
    unsupported = sorted(str(key) for key in dict(stage.options or {}) if key not in allowed)
    if not unsupported:
        return
    names = ", ".join(repr(key) for key in unsupported)
    raise CalculationValidationError(
        f"{stage_display_name(stage)} has unsupported stage option(s) {names}. "
        "BMD Compute does not accept INCAR, KPOINTS or other free-form settings "
        "in a workflow; stages may only use the supported Advanced Options.",
        suggestion="Remove the unsupported options and choose supported Advanced Options instead.",
    )


def validate_user_workflow_spec(workflow: WorkflowSpec) -> WorkflowSpec:
    """Validate a workflow supplied by a client (a Custom workflow).

    In addition to :func:`validate_workflow_spec`, stage options are limited to
    those a user may choose; BMD-generated options such as frozen +U parameters
    cannot be supplied from outside.
    """

    for stage in WorkflowSpec(stages=workflow.stages, label=workflow.label, recipe=workflow.recipe).stages:
        _validate_stage_option_keys(stage, USER_STAGE_OPTION_KEYS)
    return validate_workflow_spec(workflow)


def _validate_stage_support(stage: StageSpec) -> None:
    _validate_stage_option_keys(stage)
    calculation_stage = CalculationStage(stage.stage_type.value)
    theory_label = theory_display_name(stage.theory)
    stage_label = stage_display_name(stage)

    if calculation_stage not in theory_supported_stages(stage.theory):
        supported_stages = theory_supported_stages(stage.theory)
        if supported_stages:
            supported_text = ", ".join(
                stage_display_name(StageType.from_value(item.value))
                for item in sorted(supported_stages, key=lambda candidate: candidate.value)
            )
            suggestion = (
                f"Choose {supported_text} with {theory_label}, "
                f"or choose PBE for {stage_label}."
            )
        else:
            suggestion = "Choose PBE for this stage."

        raise CalculationValidationError(
            f"{theory_label} is not available for {stage_label} stages yet.",
            suggestion=suggestion,
        )

    unsupported_modifiers = stage.modifiers.difference(
        _supported_modifiers_for_stage(stage.stage_type, stage.theory)
    )
    if unsupported_modifiers:
        modifier_text = ", ".join(
            modifier_display_name(modifier)
            for modifier in sorted(unsupported_modifiers, key=lambda item: item.value)
        )
        raise CalculationValidationError(
            f"{stage_label} with {theory_label} is not available with {modifier_text}.",
            suggestion="Adjust the advanced options, or choose PBE for this stage.",
        )

    _validate_dispersion_stage_support(stage)
    _validate_frozen_dft_u_option(stage)


def _validate_frozen_dft_u_option(stage: StageSpec) -> None:
    options = dict(stage.options or {})
    if DFT_U_OPTION_KEY not in options:
        return
    frozen = options[DFT_U_OPTION_KEY]
    if Modifier.DFT_U not in stage.modifiers:
        raise CalculationValidationError(
            "Frozen DFT+U parameters require the DFT+U advanced option on the same stage.",
            suggestion="Rebuild the calculation, then try again.",
        )
    if stage.theory is not Theory.PBE or stage.stage_type not in {StageType.RELAX, StageType.STATIC}:
        raise CalculationValidationError(
            "Frozen DFT+U parameters are only used on PBE Geometry Optimisation and Static Energy stages.",
            suggestion="Rebuild the calculation, then try again.",
        )
    species = frozen.get("species") if isinstance(frozen, dict) else None
    if (
        not isinstance(species, dict)
        or not species
        or not isinstance(frozen.get("LDAUTYPE"), int)
        or any(
            not isinstance(values, dict)
            or any(not isinstance(values.get(key), (int, float)) for key in ("L", "U", "J"))
            for values in species.values()
        )
    ):
        raise CalculationValidationError(
            "The frozen DFT+U parameters for this stage are malformed.",
            suggestion="Rebuild the calculation, then try again.",
        )


def _stage_dispersion_method(stage: StageSpec) -> str | None:
    if Modifier.DISPERSION not in stage.modifiers:
        return None
    return dispersion_method_from_options(stage.options)


def _validate_dispersion_stage_support(stage: StageSpec) -> None:
    has_dispersion_option = DISPERSION_OPTION_KEY in dict(stage.options or {})
    has_dispersion_modifier = Modifier.DISPERSION in stage.modifiers
    if has_dispersion_option and not has_dispersion_modifier:
        raise CalculationValidationError(
            "van der Waals correction options require the van der Waals correction advanced option.",
            suggestion="Enable van der Waals correction or remove the stage-local dispersion option.",
        )
    if not has_dispersion_modifier:
        return

    if stage.theory is not Theory.PBE:
        raise CalculationValidationError(
            "van der Waals correction is currently available for PBE Geometry Optimisation and Static Energy stages only.",
            suggestion="Use PBE for this van der Waals-corrected stage, or remove van der Waals correction.",
        )
    if stage.stage_type not in {StageType.RELAX, StageType.STATIC}:
        raise CalculationValidationError(
            "van der Waals correction is applied only to PBE Geometry Optimisation and Static Energy stages in Phase 1.",
            suggestion="Apply dispersion to the PBE precursor relax/static stages, not directly to DOS or Band Structure.",
        )
    try:
        _stage_dispersion_method(stage)
    except ValueError as exc:
        raise CalculationValidationError(
            str(exc),
            suggestion="Choose DFT-D3 or DFT-D3(BJ).",
        ) from exc


def _validate_dispersion_workflow_consistency(workflow: WorkflowSpec) -> None:
    relax_static_types = {StageType.RELAX, StageType.STATIC}
    for previous_stage, current_stage in zip(workflow.stages, workflow.stages[1:]):
        if (
            previous_stage.stage_type not in relax_static_types
            or current_stage.stage_type not in relax_static_types
            or previous_stage.theory is not Theory.PBE
            or current_stage.theory is not Theory.PBE
        ):
            continue
        previous_method = _stage_dispersion_method(previous_stage)
        current_method = _stage_dispersion_method(current_stage)
        if previous_method == current_method:
            continue
        if previous_method or current_method:
            raise CalculationValidationError(
                "Use the same van der Waals correction across connected PBE relax/static stages.",
                suggestion=(
                    "Enable the same DFT-D3 or DFT-D3(BJ) option on each connected "
                    "Geometry Optimisation and Static Energy stage, or remove dispersion."
                ),
            )


def _supported_modifiers_for_stage(
    stage_type: StageType,
    theory: Theory,
) -> frozenset[Modifier]:
    if theory is Theory.PBE:
        supported = set(_ACTIVE_UI_MODIFIERS)
        if stage_type is StageType.STATIC:
            supported.add(Modifier.SOC)
        if stage_type is StageType.RELAX:
            supported.add(Modifier.IONS_ONLY)
        if stage_type in _TERMINAL_ANALYSIS_STAGES:
            supported.discard(Modifier.DISPERSION)
        if stage_type is StageType.BAND_STRUCTURE:
            supported.discard(Modifier.GAMMA_ONLY)
        return frozenset(supported)

    if theory is Theory.HSE06:
        if stage_type is StageType.BAND_STRUCTURE:
            return frozenset(_HSE06_BAND_STRUCTURE_MODIFIERS)
        if stage_type is StageType.STATIC:
            return frozenset(_HSE06_STATIC_MODIFIERS)
        if stage_type is StageType.DOS:
            return frozenset(_HSE06_DOS_MODIFIERS)
        return frozenset(_HSE06_SINGLE_STAGE_MODIFIERS)

    return frozenset()


def workflow_result_stage_directory(workflow: WorkflowSpec) -> str | None:
    stage_directories = workflow_stage_directories(workflow)
    if not stage_directories:
        return None
    return stage_directories[-1]


def supported_combinations() -> tuple[CalculationSpec, ...]:
    return tuple(
        CalculationSpec(purpose=purpose, theory=theory, modifiers=modifiers)
        for purpose, theory, modifiers in _SUPPORTED_COMPATIBILITY_WORKFLOWS
    )


def calculation_display_name(spec: CalculationSpec) -> str:
    normalized = CalculationSpec(
        purpose=spec.purpose,
        theory=spec.theory,
        modifiers=spec.modifiers,
        label=spec.label,
    )
    return _PURPOSE_DISPLAY_NAMES.get(
        normalized.purpose,
        normalized.purpose.value.replace("_", " ").title(),
    )


def stage_display_name(stage: StageSpec | StageType | str) -> str:
    stage_type = stage.stage_type if isinstance(stage, StageSpec) else StageType.from_value(stage)
    return _STAGE_DISPLAY_NAMES.get(
        stage_type,
        stage_type.value.replace("_", " ").title(),
    )


def workflow_display_name(workflow: WorkflowSpec) -> str:
    normalized = validate_workflow_spec(workflow)
    if normalized.label:
        return normalized.label

    compatible_spec = calculation_spec_from_workflow_spec(normalized)
    if compatible_spec is not None:
        return calculation_display_name(compatible_spec)

    stage_names = [stage_display_name(stage) for stage in normalized.stages]
    if len(stage_names) == 1:
        return stage_names[0]

    return " + ".join(stage_names)


def theory_display_name(theory: Theory | str) -> str:
    normalized = Theory.from_value(theory)
    return _THEORY_DISPLAY_NAMES.get(
        normalized,
        normalized.value.replace("_", " ").title(),
    )


def modifier_display_name(modifier: Modifier | str) -> str:
    normalized = Modifier.from_value(modifier)
    return _MODIFIER_DISPLAY_NAMES.get(
        normalized,
        normalized.value.replace("_", " ").title(),
    )


def calculation_form_options() -> dict:
    supported = supported_combinations()
    visible_supported = [
        spec for spec in supported
        if not spec.modifiers.intersection(_UI_HIDDEN_MODIFIERS)
    ]
    supported_purposes = {spec.purpose for spec in visible_supported}
    supported_theories = {spec.theory for spec in visible_supported}
    supported_modifiers = {
        modifier
        for spec in visible_supported
        for modifier in spec.modifiers
        if modifier not in _UI_HIDDEN_MODIFIERS
    }

    return {
        "purposes": [
            {
                "value": purpose.value,
                "label": _PURPOSE_DISPLAY_NAMES.get(
                    purpose,
                    purpose.value.replace("_", " ").title(),
                ),
                "description": _PURPOSE_DESCRIPTIONS.get(purpose, ""),
                "enabled": purpose in supported_purposes,
            }
            for purpose in Purpose
            if purpose in supported_purposes
        ],
        "theories": [
            {
                "value": theory.value,
                "label": theory_display_name(theory),
                "enabled": theory in supported_theories,
            }
            for theory in Theory
            if theory in supported_theories
        ],
        "modifiers": [
            {
                "value": modifier.value,
                "label": modifier_display_name(modifier),
                "enabled": modifier in supported_modifiers,
                "tooltip": (
                    _MODIFIER_TOOLTIPS.get(modifier, "")
                    if modifier in supported_modifiers
                    else _MODIFIER_TOOLTIPS.get(modifier, _UNIMPLEMENTED_TOOLTIP)
                ),
            }
            for modifier in _UI_MODIFIER_ORDER
            if modifier not in _UI_HIDDEN_MODIFIERS
        ],
        "stage_types": [
            {
                "value": stage_type.value,
                "label": stage_display_name(stage_type),
                "description": _STAGE_DESCRIPTIONS.get(stage_type, ""),
            }
            for stage_type in StageType
        ],
        "desired_outputs": list(_desired_output_options()),
        "recipes": list(_recommended_workflow_recipes()),
    }


def _theory_from_legacy_potcar(potcar_functional: str | None) -> Theory:
    if not potcar_functional:
        return Theory.PBE

    key = str(potcar_functional).strip().upper()
    try:
        return _LEGACY_POTCAR_THEORIES[key]
    except KeyError as exc:
        raise CalculationValidationError(
            f"Unsupported legacy POTCAR functional: {potcar_functional!r}"
        ) from exc


def _compatibility_key(spec: CalculationSpec) -> tuple[Purpose, Theory, frozenset[Modifier]]:
    return (spec.purpose, spec.theory, frozenset(spec.modifiers))


def _unsupported_combination_error(
    spec: CalculationSpec,
    key: tuple[Purpose, Theory, frozenset[Modifier]],
) -> CalculationValidationError:
    purpose, theory, modifiers = key
    theory_label = theory_display_name(theory)
    purpose_label = calculation_display_name(spec)
    supported_purposes = theory_supported_purposes(theory)
    combination = _format_combination(*key)

    if not supported_purposes:
        return CalculationValidationError(
            f"{theory_label} is not available in BMD Compute yet. Combination: {combination}.",
            suggestion="Choose PBE for this calculation.",
        )

    if purpose not in supported_purposes:
        supported_text = ", ".join(
            _PURPOSE_DISPLAY_NAMES.get(item, item.value.replace("_", " ").title())
            for item in sorted(supported_purposes, key=lambda item: item.value)
        )
        return CalculationValidationError(
            f"{theory_label} is currently supported for {supported_text} only. "
            f"{purpose_label} with {theory_label} is not available yet. "
            f"Combination: {combination}.",
            suggestion=(
                f"Choose {supported_text} with {theory_label}, "
                f"or choose PBE for {purpose_label}."
            ),
        )

    supported_modifiers = {
        candidate_modifier
        for (
            candidate_purpose,
            candidate_theory,
            candidate_modifiers,
        ) in _SUPPORTED_COMPATIBILITY_WORKFLOWS
        if candidate_purpose is purpose and candidate_theory is theory
        for candidate_modifier in candidate_modifiers
    }
    unsupported_modifiers = sorted(
        modifiers.difference(supported_modifiers),
        key=lambda modifier: modifier.value,
    )
    modifier_text = (
        ", ".join(modifier_display_name(modifier) for modifier in unsupported_modifiers)
        if unsupported_modifiers
        else "the selected advanced options"
    )
    return CalculationValidationError(
        f"{purpose_label} with {theory_label} is not available with {modifier_text}. "
        f"Combination: {combination}.",
        suggestion="Adjust the advanced options, or choose PBE for this calculation.",
    )


def _format_combination(
    purpose: Purpose,
    theory: Theory,
    modifiers: frozenset[Modifier],
) -> str:
    modifier_text = ",".join(sorted(modifier.value for modifier in modifiers)) or "none"
    return f"purpose={purpose.value}, theory={theory.value}, modifiers={modifier_text}"


__all__ = [
    "CalculationValidationError",
    "SUPPORTED_STAGE_OPTION_KEYS",
    "USER_STAGE_OPTION_KEYS",
    "validate_user_workflow_spec",
    "calculation_display_name",
    "calculation_form_options",
    "calculation_result_stage_directory",
    "calculation_stage_directories",
    "calculation_spec_from_flow_spec",
    "calculation_spec_from_legacy",
    "calculation_spec_from_workflow_spec",
    "desired_output_from_workflow_spec",
    "desired_output_workflow_spec",
    "legacy_potcar_functional_from_spec",
    "legacy_workflow_from_spec",
    "modifier_display_name",
    "stage_display_name",
    "supported_combinations",
    "theory_display_name",
    "validate_calculation_spec",
    "validate_stage_spec",
    "validate_workflow_spec",
    "workflow_display_name",
    "workflow_result_stage_directory",
    "workflow_spec_from_calculation_spec",
    "workflow_spec_from_flow_spec",
    "workflow_spec_from_legacy",
    "workflow_stage_directories",
]
