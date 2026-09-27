import json
import time
from copy import deepcopy

from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

from backend.config import bmd_debug_enabled
from backend.calculations.builder import build_calculation_flow
from backend.calculations.models import (
    CalculationSpec,
    Purpose,
    StageType,
    Theory,
    WorkflowSpec,
)
from backend.calculations.default_treatments import (
    AUTOMATIC_APPLICATION_ADVISORY,
    AUTOMATIC_APPLICATION_APPLIED,
    AUTOMATIC_APPLICATION_NOT_APPLICABLE,
    ResolvedDefaultWorkflow,
    resolve_default_treatments,
)
from backend.calculations.method_considerations import (
    DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
    SOC_HEAVY_ELEMENTS_CONSIDERATION_ID,
    SPIN_COMPOSITION_CONSIDERATION_ID,
    method_consideration_payload,
)
from backend.calculations.resources import (
    ALLOWED_CPU_COUNTS,
    ALLOWED_MEMORY_GB,
    ALLOWED_QUEUES,
    ExecutionResources,
    default_execution_resources,
)
from backend.calculations.registry import (
    CalculationValidationError,
    calculation_spec_from_workflow_spec,
    calculation_display_name,
    calculation_form_options,
    calculation_spec_from_legacy,
    desired_output_from_workflow_spec,
    desired_output_workflow_spec,
    legacy_potcar_functional_from_spec,
    legacy_workflow_from_spec,
    modifier_display_name,
    stage_display_name,
    theory_display_name,
    validate_workflow_spec,
    workflow_display_name,
    workflow_spec_from_calculation_spec,
)
from backend.generated_inputs import preview_generated_inputs
from backend.monitoring import monitor_job
from backend.parser import StructureValidationError, parse_structure
from backend.remote_preparation import prepare_remote_submission, remembered_successful_preparation
from backend.remote_submission import remembered_successful_submission, submit_remote_workflow
from backend.results import load_results_for_completed_job, monitoring_indicates_success
from backend.submission import (
    build_standalone_slurm_example,
    build_submission_script_artifact,
    build_submission_summary,
    create_submission_spec,
    verify_submission_identity_token,
)
from backend.summary import summarize_structure
from backend.workflow_summary import summarize_workflow

app = FastAPI()

templates = Jinja2Templates(directory="templates")


def page_context(
    *,
    structure_text: str = "",
    fmt: str = "poscar",
    summary=None,
    selected_spec: CalculationSpec | None = None,
    selected_workflow: WorkflowSpec | None = None,
    selected_resources: ExecutionResources | None = None,
    calculation_summary=None,
    generated_inputs=None,
    submission_spec=None,
    remote_preparation=None,
    submission_result=None,
    monitoring_result=None,
    results_summary=None,
    method_considerations=None,
    default_treatment_resolution: ResolvedDefaultWorkflow | None = None,
    resume_job_id: str = "",
    structure_error=None,
    calculation_error=None,
    collapse_structure_input: bool = False,
):
    if selected_workflow is None:
        selected_workflow = (
            workflow_spec_from_calculation_spec(selected_spec)
            if selected_spec is not None
            else default_workflow_spec()
        )
    selected_workflow = validate_workflow_spec(selected_workflow)
    selected_spec = (
        selected_spec
        or calculation_spec_from_workflow_spec(selected_workflow)
        or default_calculation_spec()
    )
    selected_resources = selected_resources or default_execution_resources()
    return {
        "structure_text": structure_text,
        "fmt": fmt,
        "summary": summary,
        "selected_calculation": selected_calculation_context(
            selected_spec,
            selected_workflow=selected_workflow,
        ),
        "selected_workflow": selected_workflow_context(selected_workflow),
        "selected_resources": selected_resources_context(selected_resources),
        "calculation_options": calculation_form_options(),
        "calculation": calculation_summary,
        "workflow": calculation_summary,
        "generated_inputs": generated_inputs,
        "submission_spec": submission_spec,
        "monitor_state_json": monitor_state_json(
            summary=summary,
            calculation_summary=calculation_summary,
            generated_inputs=generated_inputs,
            submission_spec=submission_spec,
        ),
        "remote_preparation": remote_preparation,
        "submission_result": submission_result,
        "monitoring_result": monitoring_result,
        "results_summary": results_summary,
        "method_considerations": method_considerations,
        "default_treatment_resolution": default_treatment_resolution_context(
            default_treatment_resolution
        ),
        "resume_job_id": resume_job_id,
        "structure_error": structure_error,
        "calculation_error": calculation_error,
        "collapse_structure_input": collapse_structure_input,
        "bmd_debug": bmd_debug_enabled(),
    }


def monitor_state_json(
    *,
    summary=None,
    calculation_summary=None,
    generated_inputs=None,
    submission_spec=None,
) -> str:
    if not submission_spec:
        return ""

    payload = {
        "summary": summary,
        "calculation": calculation_summary,
        "generated_inputs": generated_inputs,
        "submission_spec": _compact_submission_spec_for_monitor(submission_spec),
    }
    return json.dumps(payload, sort_keys=True)


def _compact_submission_spec_for_monitor(submission_spec: dict) -> dict:
    compact = deepcopy(submission_spec)
    flow_spec = compact.get("flow_spec")
    if isinstance(flow_spec, dict):
        # Monitoring/results need the workflow and remote paths, not the full
        # pasted structure text. The original structure remains in the visible
        # form state for editing/rebuilding.
        flow_spec.pop("structure", None)
    return compact


def monitor_state_from_json(value: str | None) -> dict:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise CalculationValidationError(
            "The saved monitoring state could not be read.",
            suggestion="Rebuild the calculation, then refresh monitoring again.",
        ) from exc
    return payload if isinstance(payload, dict) else {}


def lightweight_submission_spec_from_monitor_form(
    *,
    structure_text: str,
    fmt: str,
    workflow_spec: WorkflowSpec,
    execution_resources: ExecutionResources,
    calculation_summary=None,
) -> dict:
    calculation_spec = calculation_spec_from_workflow_spec(workflow_spec)
    potcar_functional = (
        legacy_potcar_functional_from_spec(calculation_spec)
        if calculation_spec is not None
        else "PBE_64"
    )
    legacy_workflow = (
        legacy_workflow_from_spec(calculation_spec)
        if calculation_spec is not None
        else "custom_workflow"
    )
    flow_spec = {
        "workflow_spec": workflow_spec.to_dict(),
        "workflow": legacy_workflow,
        "potcar_functional": potcar_functional,
        "kpoints": None,
        "incar": {},
        "execution_resources": execution_resources.to_dict(),
        "structure": {
            "type": "pasted_text",
            "format": fmt,
            "text": structure_text,
        },
    }
    if calculation_spec is not None:
        flow_spec["calculation_spec"] = calculation_spec.to_dict()

    return create_submission_spec(
        flow_spec,
        structure=None,
        label=(calculation_summary or {}).get("flow_name", "vasp_run"),
        nodes=execution_resources.nodes,
        ntasks=execution_resources.cpus,
        mem_gb=execution_resources.memory_gb,
        walltime=execution_resources.walltime,
        partition=execution_resources.queue,
        account=execution_resources.account,
    )


def authoritative_submission_identity(
    identity_token: str | None,
    submission_attempt_id: str | None,
) -> dict:
    try:
        return verify_submission_identity_token(
            identity_token,
            expected_attempt_id=submission_attempt_id,
        )
    except ValueError as exc:
        raise CalculationValidationError(
            "The submission identity is missing, expired, or invalid.",
            suggestion="Rebuild the calculation before preparing or submitting it.",
        ) from exc


def structure_error_context(exc: StructureValidationError) -> dict:
    return {
        "message": exc.message,
        "suggestion": exc.suggestion,
    }


def structure_error_response(
    request: Request,
    *,
    structure_text: str,
    fmt: str,
    exc: StructureValidationError,
    selected_spec: CalculationSpec | None = None,
    selected_workflow: WorkflowSpec | None = None,
    selected_resources: ExecutionResources | None = None,
):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure_text,
            fmt=fmt,
            selected_spec=selected_spec,
            selected_workflow=selected_workflow,
            selected_resources=selected_resources,
            structure_error=structure_error_context(exc),
        ),
        status_code=400,
    )


def calculation_error_context(exc: CalculationValidationError) -> dict:
    return {
        "message": exc.message,
        "suggestion": exc.suggestion,
    }


def calculation_error_response(
    request: Request,
    *,
    structure_text: str,
    fmt: str,
    exc: CalculationValidationError,
    selected_spec: CalculationSpec | None = None,
    selected_workflow: WorkflowSpec | None = None,
    selected_resources: ExecutionResources | None = None,
):
    summary = None
    try:
        summary = summarize_structure(parse_structure(structure_text, fmt))
    except StructureValidationError:
        pass

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure_text,
            fmt=fmt,
            summary=summary,
            selected_spec=selected_spec,
            selected_workflow=selected_workflow,
            selected_resources=selected_resources,
            calculation_error=calculation_error_context(exc),
        ),
        status_code=400,
    )


def default_calculation_spec() -> CalculationSpec:
    return CalculationSpec(Purpose.STATIC, Theory.PBE)


def default_workflow_spec() -> WorkflowSpec:
    return desired_output_workflow_spec("energy_only") or workflow_spec_from_calculation_spec(
        default_calculation_spec()
    )


def selected_resources_context(resources: ExecutionResources) -> dict:
    return {
        "nodes": resources.nodes,
        "cpus": resources.cpus,
        "allowed_cpu_counts": list(ALLOWED_CPU_COUNTS),
        "memory_gb": resources.memory_gb,
        "allowed_memory_gb": list(ALLOWED_MEMORY_GB),
        "walltime": resources.walltime,
        "queue": resources.queue,
        "allowed_queues": list(ALLOWED_QUEUES),
    }


def selected_calculation_context(
    spec: CalculationSpec,
    *,
    selected_workflow: WorkflowSpec | None = None,
) -> dict:
    workflow_spec = (
        validate_workflow_spec(selected_workflow)
        if selected_workflow is not None
        else workflow_spec_from_calculation_spec(spec)
    )
    compatible_spec = calculation_spec_from_workflow_spec(workflow_spec)
    visible_spec = compatible_spec or spec
    modifiers = sorted(
        {
            modifier
            for stage in workflow_spec.stages
            for modifier in stage.modifiers
        }
        or set(visible_spec.modifiers),
        key=lambda modifier: modifier.value,
    )
    theories = {stage.theory for stage in workflow_spec.stages}
    if len(theories) == 1:
        theory = next(iter(theories))
        theory_value = theory.value
        theory_label = theory_display_name(theory)
    else:
        theory_value = "mixed"
        theory_label = "Mixed"
    return {
        "purpose": visible_spec.purpose.value if compatible_spec is not None else "custom",
        "theory": theory_value,
        "modifiers": [modifier.value for modifier in modifiers],
        "modifier_labels": [modifier_display_name(modifier) for modifier in modifiers],
        "purpose_label": (
            calculation_display_name(compatible_spec)
            if compatible_spec is not None
            else workflow_display_name(workflow_spec)
        ),
        "theory_label": theory_label,
    }


def selected_workflow_context(workflow_spec: WorkflowSpec) -> dict:
    workflow = validate_workflow_spec(workflow_spec)
    desired_output = (
        workflow.recipe
        if workflow.recipe in {
            "energy_only",
            "relaxed_structure",
            "electronic_dos",
            "electronic_band_structure",
            "custom",
        }
        else desired_output_from_workflow_spec(workflow)
    )
    return {
        "json": json.dumps(workflow.to_dict(), sort_keys=True),
        "label": workflow_display_name(workflow),
        "recipe": workflow.recipe,
        "desired_output": desired_output,
        "stages": [
            {
                "index": index + 1,
                "stage_type": stage.stage_type.value,
                "stage_type_label": stage_display_name(stage),
                "theory": stage.theory.value,
                "theory_label": theory_display_name(stage.theory),
                "modifiers": sorted(
                    modifier.value for modifier in stage.modifiers
                ),
                "options": dict(stage.options or {}),
                "modifier_labels": [
                    modifier_display_name(modifier)
                    for modifier in sorted(
                        stage.modifiers,
                        key=lambda modifier: modifier.value,
                    )
                ],
            }
            for index, stage in enumerate(workflow.stages)
        ],
    }


def calculation_spec_from_form(
    *,
    purpose: str | None = None,
    theory: str | None = None,
    modifiers: list[str] | None = None,
    workflow: str | None = None,
    method: str | None = None,
) -> CalculationSpec:
    purpose = optional_form_value(purpose)
    theory = optional_form_value(theory)
    workflow = optional_form_value(workflow)
    method = optional_form_value(method)
    if purpose or theory or modifiers:
        return CalculationSpec(
            purpose=purpose or Purpose.STATIC,
            theory=theory or Theory.PBE,
            modifiers=modifiers or (),
        )

    return calculation_spec_from_legacy(workflow, method)


def workflow_spec_from_form(
    *,
    workflow_spec_json: str | None = None,
    purpose: str | None = None,
    theory: str | None = None,
    modifiers: list[str] | None = None,
    workflow: str | None = None,
    method: str | None = None,
) -> WorkflowSpec:
    workflow_spec_json = optional_form_value(workflow_spec_json)
    purpose = optional_form_value(purpose)
    theory = optional_form_value(theory)
    workflow = optional_form_value(workflow)
    method = optional_form_value(method)
    desired_workflow = desired_output_workflow_spec(workflow)
    if desired_workflow is not None:
        return desired_workflow

    if workflow_spec_json:
        try:
            data = json.loads(workflow_spec_json)
        except json.JSONDecodeError as exc:
            raise CalculationValidationError(
                "The workflow stage configuration could not be read.",
                suggestion="Rebuild the calculation, then try again.",
            ) from exc
        return validate_workflow_spec(WorkflowSpec.from_dict(data))

    return workflow_spec_from_calculation_spec(
        calculation_spec_from_form(
            purpose=purpose,
            theory=theory,
            modifiers=modifiers,
            workflow=workflow,
            method=method,
        )
    )


def optional_form_value(value):
    if value is None:
        return None
    value_type = type(value)
    if value_type.__module__ == "fastapi.params" and value_type.__name__ == "Form":
        return None
    return value


def resolve_workflow_for_structure(
    structure_obj,
    workflow_spec: WorkflowSpec,
    *,
    workflow: str | None = None,
) -> tuple[WorkflowSpec, ResolvedDefaultWorkflow | None]:
    workflow = optional_form_value(workflow)
    if desired_output_workflow_spec(workflow) is None:
        return validate_workflow_spec(workflow_spec), None

    resolution = resolve_default_treatments(
        structure_obj,
        desired_output_workflow_spec(workflow),
        desired_output=workflow,
    )
    return resolution.resolved_workflow, resolution


def execution_resources_from_form(
    *,
    cpus: str | None = None,
    memory_gb: str | None = None,
    walltime: str | None = None,
    queue: str | None = None,
) -> ExecutionResources:
    defaults = default_execution_resources()
    return ExecutionResources(
        cpus=cpus or defaults.cpus,
        memory_gb=memory_gb or defaults.memory_gb,
        walltime=walltime or defaults.walltime,
        queue=queue or defaults.queue,
        account=defaults.account,
    )


def default_treatment_resolution_context(
    resolution: ResolvedDefaultWorkflow | None,
) -> dict | None:
    return resolution.to_dict() if resolution is not None else None


def method_considerations_for_workflow_state(
    structure_obj,
    *,
    workflow: WorkflowSpec | None,
    default_treatment_resolution: ResolvedDefaultWorkflow | None,
) -> dict | None:
    if default_treatment_resolution is None:
        return method_considerations_context(structure_obj, workflow=workflow)
    return method_considerations_context(
        structure_obj,
        workflow=workflow,
        default_treatment_resolution=default_treatment_resolution,
    )


def method_considerations_context(
    structure_obj,
    *,
    workflow: WorkflowSpec | None = None,
    default_treatment_resolution: ResolvedDefaultWorkflow | None = None,
) -> dict | None:
    payload = (
        method_consideration_payload(structure_obj, workflow=workflow)
        if workflow is not None
        else method_consideration_payload(structure_obj)
    )
    if not payload.get("considerations"):
        return None

    rendered = deepcopy(payload)
    applied_by_id = {
        treatment.consideration_id: treatment.to_dict()
        for treatment in (
            default_treatment_resolution.applied_treatments
            if default_treatment_resolution is not None
            else ()
        )
    }
    not_applicable_by_id = {
        str(item.get("consideration_id")): dict(item)
        for item in (
            default_treatment_resolution.not_applicable_considerations
            if default_treatment_resolution is not None
            else ()
        )
    }
    for consideration in rendered.get("considerations", []):
        automatic_application = applied_by_id.get(consideration.get("id"))
        if automatic_application is not None:
            application_state = AUTOMATIC_APPLICATION_APPLIED
        elif consideration.get("id") in not_applicable_by_id:
            application_state = AUTOMATIC_APPLICATION_NOT_APPLICABLE
        else:
            application_state = AUTOMATIC_APPLICATION_ADVISORY
        consideration["automatic_application_state"] = application_state
        consideration["automatic_application"] = automatic_application
        consideration["automatic_not_applicable"] = not_applicable_by_id.get(
            consideration.get("id")
        )
        support = consideration.get("bmd_compute_support") or {}
        consideration["display_name"] = (
            consideration.get("display_name")
            or support.get("modifier_label")
            or str(consideration.get("method") or "").upper()
        )
        evidence = consideration.get("observed_evidence") or {}
        trigger_items = []
        for trigger in evidence.get("triggers") or []:
            trigger_item = deepcopy(trigger)
            trigger_item["display_label"] = (
                trigger.get("label")
                or trigger.get("element")
                or trigger.get("id")
                or "Structural observation"
            )
            trigger_item["trigger_class_labels"] = [
                _method_consideration_class_label(class_name)
                for class_name in trigger.get("trigger_classes") or []
            ]
            trigger_items.append(trigger_item)
        for detection in evidence.get("detections") or []:
            trigger_classes = (
                detection.get("trigger_classes")
                or detection.get("soc_trigger_classes")
                or detection.get("spin_trigger_classes")
                or []
            )
            detection["trigger_class_labels"] = [
                _method_consideration_class_label(class_name)
                for class_name in trigger_classes
            ]
            trigger_items.append(
                {
                    "id": detection.get("id"),
                    "type": detection.get("type"),
                    "display_label": detection.get("element") or detection.get("id"),
                    "trigger_class_labels": detection["trigger_class_labels"],
                }
            )
        consideration["trigger_items"] = trigger_items
        consideration["browser_display_name"] = _method_consideration_browser_name(
            consideration
        )
        consideration["browser_summary"] = _method_consideration_browser_summary(
            consideration
        )
    return rendered


def _method_consideration_browser_name(consideration: dict) -> str:
    if consideration.get("automatic_application_state") == AUTOMATIC_APPLICATION_APPLIED:
        if consideration.get("id") == DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID:
            return "van der Waals correction applied"
        if consideration.get("id") == SPIN_COMPOSITION_CONSIDERATION_ID:
            return "Spin Polarisation applied"
        if consideration.get("id") == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID:
            return "Spin-Orbit Coupling (SOC) applied"
    if consideration.get("automatic_application_state") == AUTOMATIC_APPLICATION_NOT_APPLICABLE:
        if consideration.get("id") == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID:
            return "Spin-Orbit Coupling (SOC) not used in geometry optimisation"

    if consideration.get("id") == DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID:
        return "van der Waals Correction"
    return str(consideration.get("display_name") or "Method Consideration")


def _method_consideration_browser_summary(consideration: dict) -> str:
    consideration_id = consideration.get("id")
    is_applied = (
        consideration.get("automatic_application_state")
        == AUTOMATIC_APPLICATION_APPLIED
    )
    if consideration_id == DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID:
        if is_applied:
            return (
                "Likely 2-dimensional structure detected. The van der Waals "
                "correction has been included automatically in the applicable "
                "PBE stages. Choose Custom workflow to configure this manually."
            )
        support = _method_consideration_support_phrase(consideration)
        suffix = f" for {support}" if support else ""
        return (
            "Likely 2-dimensional structure detected. Suggested to activate the "
            f"van der Waals correction Advanced Option{suffix}."
        )

    elements = _human_join(consideration.get("trigger_elements") or ())
    subject = f"{elements} detected" if elements else "Relevant structure feature detected"

    if consideration_id == SPIN_COMPOSITION_CONSIDERATION_ID:
        if is_applied:
            return (
                f"{subject}. Spin Polarisation has been included automatically "
                "in the BMD workflow. Choose Custom workflow to configure this "
                "manually."
            )
        return (
            f"{subject}. Suggested to activate the Spin Polarised Advanced Option."
        )

    if consideration_id == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID:
        if is_applied:
            stage_phrase, stage_count = _automatic_application_stage_phrase(
                consideration
            )
            if stage_phrase:
                verb = "runs" if stage_count == 1 else "run"
                stage_text = f" in the {stage_phrase}, which {verb} with vasp_ncl"
            else:
                stage_text = ""
            return (
                f"{subject}. Spin-orbit coupling (SOC) has been included "
                f"automatically{stage_text}. Choose Custom workflow to configure "
                "this manually."
            )
        if (
            consideration.get("automatic_application_state")
            == AUTOMATIC_APPLICATION_NOT_APPLICABLE
        ):
            return (
                f"{subject}. BMD Compute keeps geometry optimisations non-SOC, "
                "so SOC is not part of this workflow. Choose Custom workflow to "
                "configure this manually."
            )
        support = _method_consideration_support_phrase(consideration)
        suffix = f" for {support}" if support else ""
        return (
            f"{subject}. Suggested to activate the Spin-Orbit Coupling (SOC) "
            f"Advanced Option{suffix}."
        )

    reason = str(consideration.get("reason") or "").strip()
    return reason or f"{subject}."


def _automatic_application_stage_phrase(consideration: dict) -> tuple[str, int]:
    application = consideration.get("automatic_application") or {}
    labels = [
        f"{item.get('theory_label')} {item.get('stage_type_label')}"
        for item in application.get("stage_applications") or ()
        if item.get("theory_label") and item.get("stage_type_label")
    ]
    if not labels:
        return "", 0
    noun = "stage" if len(labels) == 1 else "stages"
    return f"{_human_join(labels)} {noun}", len(labels)


def _method_consideration_support_phrase(consideration: dict) -> str:
    support = consideration.get("bmd_compute_support") or {}
    capabilities = support.get("supported_stage_capabilities") or ()
    if not capabilities:
        return ""

    stage_order = {
        "relax": 0,
        "static": 1,
        "dos": 2,
        "band_structure": 3,
    }
    unique_capabilities = {
        (
            capability.get("theory"),
            capability.get("stage_type"),
            f"{capability.get('theory_label')} {capability.get('stage_label')}",
        )
        for capability in capabilities
    }
    labels = [
        label
        for _theory, _stage_type, label in sorted(
            unique_capabilities,
            key=lambda item: (str(item[0]), stage_order.get(str(item[1]), 99), item[2]),
        )
        if "None" not in label
    ]
    if not labels:
        return ""
    return f"{_human_join(labels, conjunction='or')} stages"


def _human_join(values, *, conjunction: str = "and") -> str:
    items = [str(value) for value in values if str(value)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f" {conjunction} ".join(items)
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def _method_consideration_class_label(class_name: str) -> str:
    spin_labels = {
        "3d_spin_screen": "3d spin-screening element",
        "4d_spin_screen": "4d spin-screening element",
        "5d_spin_screen": "5d spin-screening element",
        "lanthanide_spin_screen": "lanthanide spin-screening element",
        "actinide_spin_screen": "actinide spin-screening element",
    }
    if class_name in spin_labels:
        return spin_labels[class_name]

    label = str(class_name).replace("_", " ").replace("p block", "p-block")
    for plural, singular in (
        ("metals", "metal"),
        ("lanthanides", "lanthanide"),
        ("actinides", "actinide"),
    ):
        if label.endswith(plural):
            return f"{label[:-len(plural)]}{singular}"
    return label


def build_submission_state(
    *,
    structure_text: str,
    fmt: str,
    calculation_spec: CalculationSpec | None = None,
    workflow_spec: WorkflowSpec | None = None,
    execution_resources: ExecutionResources | None = None,
    timestamp: str | None = None,
    submission_attempt_id: str | None = None,
):
    execution_resources = execution_resources or default_execution_resources()
    if workflow_spec is None:
        if calculation_spec is None:
            calculation_spec = default_calculation_spec()
        workflow_spec = workflow_spec_from_calculation_spec(calculation_spec)
    workflow_spec = validate_workflow_spec(workflow_spec)
    structure_obj = parse_structure(structure_text, fmt)
    return build_submission_state_from_structure(
        structure_obj=structure_obj,
        structure_text=structure_text,
        fmt=fmt,
        calculation_spec=calculation_spec,
        workflow_spec=workflow_spec,
        execution_resources=execution_resources,
        timestamp=timestamp,
        submission_attempt_id=submission_attempt_id,
    )


def build_submission_state_from_structure(
    *,
    structure_obj,
    structure_text: str,
    fmt: str,
    calculation_spec: CalculationSpec | None = None,
    workflow_spec: WorkflowSpec,
    execution_resources: ExecutionResources,
    timestamp: str | None = None,
    submission_attempt_id: str | None = None,
    default_treatment_resolution: ResolvedDefaultWorkflow | None = None,
):
    if (
        default_treatment_resolution is not None
        and default_treatment_resolution.resolved_workflow.to_dict()
        != validate_workflow_spec(workflow_spec).to_dict()
    ):
        raise CalculationValidationError(
            "The automatic treatment record does not match the workflow being prepared.",
            suggestion="Rebuild the calculation, then try again.",
        )
    calculation_spec = calculation_spec_from_workflow_spec(workflow_spec)
    summary = summarize_structure(structure_obj)
    potcar_functional = (
        legacy_potcar_functional_from_spec(calculation_spec)
        if calculation_spec is not None
        else "PBE_64"
    )
    legacy_workflow = (
        legacy_workflow_from_spec(calculation_spec)
        if calculation_spec is not None
        else "custom_workflow"
    )
    generated_inputs = preview_generated_inputs(
        structure_obj,
        workflow_spec,
        resources=execution_resources,
        potcar_functional=potcar_functional,
    )
    flow = build_calculation_flow(
        structure=structure_obj,
        spec=workflow_spec,
        resources=execution_resources,
        potcar_functional=potcar_functional,
    )
    calculation_summary = summarize_workflow(flow, workflow_spec)
    flow_spec = {
        "workflow_spec": workflow_spec.to_dict(),
        "workflow": legacy_workflow,
        "potcar_functional": potcar_functional,
        "kpoints": None,
        "incar": {},
        "execution_resources": execution_resources.to_dict(),
        "structure": {
            "type": "pasted_text",
            "format": fmt,
            "text": structure_text,
        },
    }
    if calculation_spec is not None:
        flow_spec["calculation_spec"] = calculation_spec.to_dict()
    if default_treatment_resolution is not None:
        flow_spec["automatic_treatments"] = default_treatment_resolution.to_dict()
    submission_spec = create_submission_spec(
        flow_spec,
        structure=structure_obj,
        label=calculation_summary["flow_name"],
        timestamp=timestamp,
        nodes=execution_resources.nodes,
        ntasks=execution_resources.cpus,
        mem_gb=execution_resources.memory_gb,
        walltime=execution_resources.walltime,
        partition=execution_resources.queue,
        account=execution_resources.account,
        submission_attempt_id=submission_attempt_id,
    )
    slurm_example = build_standalone_slurm_example(submission_spec)
    generated_inputs["submission_summary"] = build_submission_summary(submission_spec)
    generated_inputs["slurm_script"] = slurm_example["text"]
    generated_inputs["slurm_script_kind"] = slurm_example["kind"]
    generated_inputs["slurm_script_description"] = slurm_example["description"]
    generated_inputs["exact_slurm_script"] = build_submission_script_artifact(
        submission_spec
    )["text"]
    return summary, calculation_summary, generated_inputs, submission_spec


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(),
    )


@app.post("/analyze")
def analyze(
    request: Request,
    structure: str = Form(...),
    fmt: str = Form(...)
):
    try:
        structure_obj = parse_structure(structure, fmt)
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            exc=exc,
        )

    summary = summarize_structure(structure_obj)
    workflow_spec, default_treatment_resolution = resolve_workflow_for_structure(
        structure_obj,
        default_workflow_spec(),
        workflow="energy_only",
    )
    method_considerations = method_considerations_for_workflow_state(
        structure_obj,
        workflow=workflow_spec,
        default_treatment_resolution=default_treatment_resolution,
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure,
            fmt=fmt,
            summary=summary,
            selected_workflow=workflow_spec,
            method_considerations=method_considerations,
            default_treatment_resolution=default_treatment_resolution,
        ),
    )


@app.post("/resume")
def resume_existing_calculation(
    request: Request,
    job_id: str = Form(...),
    load_results: str = Form("false"),
):
    monitoring_result = monitor_job(job_id)
    should_load_results = load_results.lower() == "true"
    results_summary = (
        load_results_for_completed_job(monitoring_result)
        if should_load_results and monitoring_indicates_success(monitoring_result)
        else None
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            monitoring_result=monitoring_result,
            results_summary=results_summary,
            resume_job_id=job_id,
            collapse_structure_input=True,
        ),
    )


@app.post("/build-calculation")
@app.post("/build-workflow")
def build_workflow(
    request: Request,
    structure: str = Form(...),
    fmt: str = Form(...),
    purpose: str | None = Form(None),
    theory: str | None = Form(None),
    modifiers: list[str] | None = Form(None),
    cpus: str | None = Form(None),
    memory_gb: str | None = Form(None),
    walltime: str | None = Form(None),
    queue: str | None = Form(None),
    workflow_spec_json: str | None = Form(None),
    workflow: str | None = Form(None),
    method: str | None = Form(None),
):
    calculation_spec = default_calculation_spec()
    workflow_spec = default_workflow_spec()
    execution_resources = default_execution_resources()
    method_considerations = None
    try:
        workflow_spec = workflow_spec_from_form(
            workflow_spec_json=workflow_spec_json,
            purpose=purpose,
            theory=theory,
            modifiers=modifiers,
            workflow=workflow,
            method=method,
        )
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        execution_resources = execution_resources_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        structure_obj = parse_structure(structure, fmt)
        workflow_spec, default_treatment_resolution = resolve_workflow_for_structure(
            structure_obj,
            workflow_spec,
            workflow=workflow,
        )
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        method_considerations = method_considerations_for_workflow_state(
            structure_obj,
            workflow=workflow_spec,
            default_treatment_resolution=default_treatment_resolution,
        )
        summary, calculation_summary, generated_inputs, submission_spec = build_submission_state_from_structure(
            structure_obj=structure_obj,
            structure_text=structure,
            fmt=fmt,
            workflow_spec=workflow_spec,
            execution_resources=execution_resources,
            default_treatment_resolution=default_treatment_resolution,
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure,
            fmt=fmt,
            summary=summary,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            calculation_summary=calculation_summary,
            generated_inputs=generated_inputs,
            submission_spec=submission_spec,
            method_considerations=method_considerations,
            default_treatment_resolution=default_treatment_resolution,
        ),
    )


@app.post("/prepare-remote")
def prepare_remote(
    request: Request,
    structure: str = Form(...),
    fmt: str = Form(...),
    purpose: str | None = Form(None),
    theory: str | None = Form(None),
    modifiers: list[str] | None = Form(None),
    cpus: str | None = Form(None),
    memory_gb: str | None = Form(None),
    walltime: str | None = Form(None),
    queue: str | None = Form(None),
    created_at: str | None = Form(None),
    submission_attempt_id: str | None = Form(None),
    submission_identity_token: str | None = Form(None),
    workflow_spec_json: str | None = Form(None),
    workflow: str | None = Form(None),
    method: str | None = Form(None),
):
    preparation_started = time.perf_counter()
    # Accepted for legacy clients only; the signed server identity is authoritative.
    del created_at
    calculation_spec = default_calculation_spec()
    workflow_spec = default_workflow_spec()
    execution_resources = default_execution_resources()
    method_considerations = None
    try:
        identity = authoritative_submission_identity(
            submission_identity_token,
            submission_attempt_id,
        )
        workflow_spec = workflow_spec_from_form(
            workflow_spec_json=workflow_spec_json,
            purpose=purpose,
            theory=theory,
            modifiers=modifiers,
            workflow=workflow,
            method=method,
        )
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        execution_resources = execution_resources_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        structure_obj = parse_structure(structure, fmt)
        workflow_spec, default_treatment_resolution = resolve_workflow_for_structure(
            structure_obj,
            workflow_spec,
            workflow=workflow,
        )
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        method_considerations = method_considerations_for_workflow_state(
            structure_obj,
            workflow=workflow_spec,
            default_treatment_resolution=default_treatment_resolution,
        )
        summary, calculation_summary, generated_inputs, submission_spec = build_submission_state_from_structure(
            structure_obj=structure_obj,
            structure_text=structure,
            fmt=fmt,
            workflow_spec=workflow_spec,
            execution_resources=execution_resources,
            timestamp=identity["run_timestamp"],
            submission_attempt_id=identity["attempt_id"],
            default_treatment_resolution=default_treatment_resolution,
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    remote_preparation = prepare_remote_submission(
        submission_spec,
        initial_diagnostics={
            "local_reconstruction_s": time.perf_counter() - preparation_started,
        },
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure,
            fmt=fmt,
            summary=summary,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            calculation_summary=calculation_summary,
            generated_inputs=generated_inputs,
            submission_spec=submission_spec,
            remote_preparation=remote_preparation,
            method_considerations=method_considerations,
            default_treatment_resolution=default_treatment_resolution,
        ),
    )


@app.post("/submit")
def submit_workflow(
    request: Request,
    structure: str = Form(...),
    fmt: str = Form(...),
    purpose: str | None = Form(None),
    theory: str | None = Form(None),
    modifiers: list[str] | None = Form(None),
    cpus: str | None = Form(None),
    memory_gb: str | None = Form(None),
    walltime: str | None = Form(None),
    queue: str | None = Form(None),
    created_at: str | None = Form(None),
    submission_attempt_id: str | None = Form(None),
    submission_identity_token: str | None = Form(None),
    remote_prepared: str = Form("false"),
    workflow_spec_json: str | None = Form(None),
    workflow: str | None = Form(None),
    method: str | None = Form(None),
):
    # Accepted for legacy clients only; the signed server identity is authoritative.
    del created_at
    calculation_spec = default_calculation_spec()
    workflow_spec = default_workflow_spec()
    execution_resources = default_execution_resources()
    method_considerations = None
    try:
        identity = authoritative_submission_identity(
            submission_identity_token,
            submission_attempt_id,
        )
        workflow_spec = workflow_spec_from_form(
            workflow_spec_json=workflow_spec_json,
            purpose=purpose,
            theory=theory,
            modifiers=modifiers,
            workflow=workflow,
            method=method,
        )
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        execution_resources = execution_resources_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        structure_obj = parse_structure(structure, fmt)
        workflow_spec, default_treatment_resolution = resolve_workflow_for_structure(
            structure_obj,
            workflow_spec,
            workflow=workflow,
        )
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        method_considerations = method_considerations_for_workflow_state(
            structure_obj,
            workflow=workflow_spec,
            default_treatment_resolution=default_treatment_resolution,
        )
        summary, calculation_summary, generated_inputs, submission_spec = build_submission_state_from_structure(
            structure_obj=structure_obj,
            structure_text=structure,
            fmt=fmt,
            workflow_spec=workflow_spec,
            execution_resources=execution_resources,
            timestamp=identity["run_timestamp"],
            submission_attempt_id=identity["attempt_id"],
            default_treatment_resolution=default_treatment_resolution,
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    prepared = remote_prepared.lower() == "true"
    submission_result = submit_remote_workflow(
        submission_spec,
        remote_prepared=prepared,
    )
    remote_preparation = (
        remembered_successful_preparation(submission_spec)
        if prepared
        else None
    )
    monitoring_result = None
    if submission_result.get("status") == "success" and submission_result.get("job_id"):
        monitoring_result = monitor_job(
            submission_result["job_id"],
            submission_spec=submission_spec,
        )
    results_summary = load_results_for_completed_job(
        monitoring_result,
        submission_spec=submission_spec,
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure,
            fmt=fmt,
            summary=summary,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            calculation_summary=calculation_summary,
            generated_inputs=generated_inputs,
            submission_spec=submission_spec,
            remote_preparation=remote_preparation,
            submission_result=submission_result,
            monitoring_result=monitoring_result,
            results_summary=results_summary,
            method_considerations=method_considerations,
            default_treatment_resolution=default_treatment_resolution,
        ),
    )


@app.post("/monitor")
def refresh_monitoring(
    request: Request,
    structure: str = Form(...),
    fmt: str = Form(...),
    purpose: str | None = Form(None),
    theory: str | None = Form(None),
    modifiers: list[str] | None = Form(None),
    cpus: str | None = Form(None),
    memory_gb: str | None = Form(None),
    walltime: str | None = Form(None),
    queue: str | None = Form(None),
    created_at: str | None = Form(None),
    job_id: str = Form(...),
    submitted_at: str = Form(""),
    monitor_state_json: str | None = Form(None),
    workflow_spec_json: str | None = Form(None),
    workflow: str | None = Form(None),
    method: str | None = Form(None),
):
    # Accepted for legacy clients only; monitoring never trusts a client run timestamp.
    del created_at
    calculation_spec = default_calculation_spec()
    workflow_spec = default_workflow_spec()
    execution_resources = default_execution_resources()
    monitor_state = {}
    summary = None
    calculation_summary = None
    generated_inputs = None
    submission_spec = None
    try:
        monitor_state = monitor_state_from_json(monitor_state_json)
        workflow_spec = workflow_spec_from_form(
            workflow_spec_json=workflow_spec_json,
            purpose=purpose,
            theory=theory,
            modifiers=modifiers,
            workflow=workflow,
            method=method,
        )
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        execution_resources = execution_resources_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        summary = monitor_state.get("summary")
        calculation_summary = monitor_state.get("calculation")
        generated_inputs = monitor_state.get("generated_inputs")
        submission_spec = monitor_state.get("submission_spec")
        if not submission_spec:
            submission_spec = lightweight_submission_spec_from_monitor_form(
                structure_text=structure,
                fmt=fmt,
                workflow_spec=workflow_spec,
                execution_resources=execution_resources,
                calculation_summary=calculation_summary,
            )
        if not isinstance(submission_spec, dict):
            raise CalculationValidationError(
                "The saved submission state could not be read.",
                suggestion="Rebuild the calculation, then refresh monitoring again.",
            )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            structure_text=structure,
            fmt=fmt,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    remote_preparation = remembered_successful_preparation(submission_spec)
    submission_result = remembered_successful_submission(
        submission_spec,
        job_id,
        submitted_at=submitted_at,
    )
    monitoring_result = monitor_job(job_id, submission_spec=submission_spec)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure,
            fmt=fmt,
            summary=summary,
            selected_spec=calculation_spec,
            selected_workflow=workflow_spec,
            selected_resources=execution_resources,
            calculation_summary=calculation_summary,
            generated_inputs=generated_inputs,
            submission_spec=submission_spec,
            remote_preparation=remote_preparation,
            submission_result=submission_result,
            monitoring_result=monitoring_result,
            results_summary=None,
            collapse_structure_input=True,
        ),
    )
