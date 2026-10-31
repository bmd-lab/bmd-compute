import dataclasses
import json
import time
from copy import deepcopy
from dataclasses import dataclass

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
from backend.calculations.admission import require_admissible_new_calculation
from backend.calculations.geometry_relaxation import freeze_plane_wave_cutoffs
from backend.config import DEFAULT_POTCAR_FUNCTIONAL
from backend.calculations.default_treatments import (
    AUTOMATIC_APPLICATION_ADVISORY,
    AUTOMATIC_APPLICATION_APPLIED,
    AUTOMATIC_APPLICATION_NOT_APPLICABLE,
    AUTOMATIC_APPLICATION_OMITTED_UNSUPPORTED,
    ResolvedDefaultWorkflow,
    resolve_default_treatments,
)
from backend.calculations.dft_u_policy import (
    GATE_UNAVAILABLE as DFT_U_GATE_UNAVAILABLE,
    prepared_dft_u_record,
)
from backend.calculations.method_considerations import (
    DFT_U_CONSIDERATION_ID,
    DFT_U_SUPPRESSED_STATUS,
    DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID,
    SOC_HEAVY_ELEMENTS_CONSIDERATION_ID,
    SPIN_COMPOSITION_CONSIDERATION_ID,
    method_consideration_payload,
)
from backend.calculations.automatic_resources import (
    AUTOMATIC_CPUS,
    AUTOMATIC_MEMORY_GB,
    SOURCE_AUTOMATIC,
    ResourceRequest,
    planned_resources_token,
    resolve_execution_resources,
)
from backend.calculations.resources import (
    ALLOWED_CPU_COUNTS,
    ALLOWED_MEMORY_GB,
    ALLOWED_QUEUES,
    ExecutionResources,
    default_execution_resources,
)
from backend.config import DEFAULT_PARTITION
from backend.calculations.registry import (
    BROWSER_HIDDEN_DESIRED_OUTPUTS,
    CalculationValidationError,
    calculation_spec_from_workflow_spec,
    calculation_display_name,
    calculation_form_options,
    canonical_custom_workflow_spec,
    calculation_spec_from_legacy,
    desired_output_label,
    desired_output_workflow_spec,
    legacy_potcar_functional_from_spec,
    legacy_workflow_from_spec,
    modifier_display_name,
    stage_display_name,
    theory_display_name,
    validate_user_workflow_spec,
    validate_workflow_spec,
    workflow_display_name,
    workflow_spec_from_calculation_spec,
)
from backend.generated_inputs import generated_input_stage_previews, preview_generated_inputs
from backend.monitoring import RESUMED_SUBMISSION_SPEC_KEY, monitor_job, monitor_submitted_job
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
from backend.workflow_summary import (
    calculation_plan_from_workflow_spec,
    summarize_workflow,
    workflow_label_from_spec,
)
from compute_api.router import build_machine_api_router

app = FastAPI()

templates = Jinja2Templates(directory="templates")


def page_context(
    *,
    structure_text: str = "",
    fmt: str = "poscar",
    summary=None,
    selected_spec: CalculationSpec | None = None,
    selected_workflow: WorkflowSpec | None = None,
    selected_resources: "SelectedResources | ExecutionResources | None" = None,
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
    monitor_state_override: str | None = None,
    selected_desired_output: str | None = None,
    workflow_identity_available: bool = True,
    show_submitted_workflow: bool = False,
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
    selected_resources = (
        selected_resources if selected_resources is not None else SelectedResources()
    )
    return {
        "structure_text": structure_text,
        "fmt": fmt,
        "summary": summary,
        "selected_calculation": {
            **selected_calculation_context(
                selected_spec,
                selected_workflow=selected_workflow,
            ),
            "unavailable": not workflow_identity_available,
        },
        "selected_workflow": selected_workflow_context(
            selected_workflow,
            selected_desired_output=selected_desired_output,
            available=workflow_identity_available,
        ),
        "show_submitted_workflow": show_submitted_workflow,
        "selected_resources": selected_resources_context(selected_resources),
        "calculation_options": calculation_form_options(),
        "calculation": calculation_summary,
        "workflow": calculation_summary,
        "generated_inputs": generated_inputs,
        "submission_spec": submission_spec,
        "monitor_state_json": (
            monitor_state_override
            if monitor_state_override is not None
            else monitor_state_json(
                summary=summary,
                calculation_summary=calculation_summary,
                generated_inputs=generated_inputs,
                submission_spec=submission_spec,
            )
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


def _authenticated_monitor_retry_state_json(
    *,
    identity: dict,
    identity_token: str,
    summary=None,
    calculation_summary=None,
    generated_inputs=None,
) -> str:
    return monitor_state_json(
        summary=summary,
        calculation_summary=calculation_summary,
        generated_inputs=generated_inputs,
        submission_spec={
            "submission": {
                "attempt_id": identity["attempt_id"],
                "identity_token": identity_token,
            }
        },
    )


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
    selected_resources: "SelectedResources | ExecutionResources | None" = None,
    selected_desired_output: str | None = None,
    workflow_identity_available: bool = True,
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
            selected_desired_output=selected_desired_output,
            workflow_identity_available=workflow_identity_available,
            structure_error=structure_error_context(exc),
        ),
        status_code=400,
    )


def calculation_error_context(exc: CalculationValidationError) -> dict:
    context = {
        "message": exc.message,
        "suggestion": exc.suggestion,
    }
    if exc.diagnostic is not None:
        context["diagnostic"] = deepcopy(exc.diagnostic)
    return context


def calculation_error_response(
    request: Request,
    *,
    structure_text: str,
    fmt: str,
    exc: CalculationValidationError,
    selected_spec: CalculationSpec | None = None,
    selected_workflow: WorkflowSpec | None = None,
    selected_resources: "SelectedResources | ExecutionResources | None" = None,
    selected_desired_output: str | None = None,
    workflow_identity_available: bool = True,
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
            selected_desired_output=selected_desired_output,
            workflow_identity_available=workflow_identity_available,
            calculation_error=calculation_error_context(exc),
        ),
        status_code=400,
    )


def default_calculation_spec() -> CalculationSpec:
    return CalculationSpec(Purpose.STATIC, Theory.PBE)


def default_workflow_spec() -> WorkflowSpec:
    # The browser's default Desired Output. Energy only is no longer offered in
    # the browser menu, so its replacement in the same position is the default.
    return desired_output_workflow_spec("relaxed_energy") or workflow_spec_from_calculation_spec(
        default_calculation_spec()
    )


@dataclass(frozen=True)
class SelectedResources:
    """Resources as requested (``None`` = Automatic) and, once built, as resolved."""

    request: ResourceRequest = ResourceRequest()
    resources: ExecutionResources | None = None
    record: dict | None = None
    qos: str | None = None


def selected_resources_from_submission(
    submission_spec: dict,
    request: ResourceRequest,
) -> SelectedResources:
    resources = submission_spec["resources"]
    cluster = submission_spec["cluster"]
    resolved = ExecutionResources(
        nodes=resources["nodes"],
        cpus=resources["ntasks"],
        memory_gb=resources["mem_gb"],
        walltime=resources["walltime"],
        queue=cluster["partition"],
        account=cluster["account"],
    )
    record = (submission_spec.get("flow_spec") or {}).get("resource_selection")
    return SelectedResources(
        request=request,
        resources=resolved,
        record=record,
        qos=cluster.get("qos"),
    )


def selected_resources_context(selection) -> dict:
    if isinstance(selection, ExecutionResources):
        selection = SelectedResources(
            request=ResourceRequest.from_execution_resources(selection),
            resources=selection,
        )
    request = selection.request
    resources = selection.resources
    record = selection.record or {}
    estimate = record.get("walltime_estimate") or {}
    fields = record.get("fields") or {}

    def source(name: str) -> str:
        return SOURCE_AUTOMATIC if getattr(request, name) is None else "manual"

    cpus = resources.cpus if resources is not None else (request.cpus or AUTOMATIC_CPUS)
    memory_gb = (
        resources.memory_gb if resources is not None else (request.memory_gb or AUTOMATIC_MEMORY_GB)
    )
    walltime = resources.walltime if resources is not None else request.walltime
    return {
        "nodes": resources.nodes if resources is not None else 1,
        "cpus": cpus,
        "allowed_cpu_counts": list(ALLOWED_CPU_COUNTS),
        "memory_gb": memory_gb,
        "allowed_memory_gb": list(ALLOWED_MEMORY_GB),
        "walltime": walltime,
        "queue": resources.queue if resources is not None else (request.queue or DEFAULT_PARTITION),
        "allowed_queues": list(ALLOWED_QUEUES),
        "automatic_cpus": AUTOMATIC_CPUS,
        # The resolved automatic memory bracket (the recommendation the
        # Custom hardware controls restore), not a fixed default.
        "automatic_memory_gb": (fields.get("memory_gb") or {}).get("recommended") or AUTOMATIC_MEMORY_GB,
        "requested": {
            "cpus": "" if request.cpus is None else request.cpus,
            "memory_gb": "" if request.memory_gb is None else request.memory_gb,
            "walltime": "" if request.walltime is None else request.walltime,
            "queue": "" if request.queue is None else request.queue,
        },
        "sources": {
            "cpus": source("cpus"),
            "memory_gb": source("memory_gb"),
            "walltime": source("walltime"),
            "queue": source("queue"),
        },
        "resolved": resources is not None,
        "qos": selection.qos,
        "recommended_walltime": (fields.get("walltime") or {}).get("recommended"),
        "walltime_basis": estimate.get("basis"),
        "walltime_explanation": estimate.get("explanation"),
        "walltime_stages": [
            {
                "index": stage.get("index"),
                "stage_type": stage.get("stage_type"),
                "theory": stage.get("theory"),
                "class": stage.get("class"),
                "hours": round(float(stage.get("seconds") or 0) / 3600, 2),
                "reason": stage.get("reason"),
            }
            for stage in estimate.get("stages") or []
        ],
        "policy": record.get("policy"),
        "planned_token": (
            planned_resources_token(resources, selection.record)
            if resources is not None and selection.record
            else ""
        ),
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


def authoritative_workflow_identity(submission_spec: dict | None) -> tuple[WorkflowSpec, str] | None:
    """The submitted workflow and its identity, read from a submission record.

    Identity is taken from what BMD Compute itself recorded, never from the
    stage sequence: a managed Desired Output is identified by its recorded
    automatic-treatment resolution (``mode`` ``bmd_managed_desired_output`` and
    its ``desired_output``), which every managed plan records and must agree
    with the workflow's recipe. A workflow with no automatic-treatment
    resolution and no managed recipe is a Custom workflow. Anything else (no
    readable workflow, an unknown resolution, a managed recipe without its
    resolution, or a disagreement) cannot be identified and returns ``None``.
    """

    if not isinstance(submission_spec, dict):
        return None
    flow_spec = submission_spec.get("flow_spec")
    if not isinstance(flow_spec, dict) or not isinstance(flow_spec.get("workflow_spec"), dict):
        return None
    try:
        workflow = validate_workflow_spec(WorkflowSpec.from_dict(flow_spec["workflow_spec"]))
    except (CalculationValidationError, KeyError, TypeError, ValueError):
        return None
    if not workflow.stages:
        return None
    treatments = flow_spec.get("automatic_treatments")
    if treatments:
        if not isinstance(treatments, dict) or treatments.get("mode") != "bmd_managed_desired_output":
            return None
        desired_output = treatments.get("desired_output")
        if desired_output_workflow_spec(desired_output) is None or workflow.recipe != desired_output:
            return None
        return workflow, desired_output
    if desired_output_workflow_spec(workflow.recipe) is not None:
        return None
    return workflow, "custom"


def authoritative_structure_summary(submission_spec: dict | None) -> tuple[str | None, dict | None]:
    """The submitted structure text and its summary, read from a submission record.

    Returns ``(None, None)`` when the record holds no readable structure.
    """

    if not isinstance(submission_spec, dict):
        return None, None
    structure = (submission_spec.get("flow_spec") or {}).get("structure")
    if not isinstance(structure, dict) or not isinstance(structure.get("text"), str):
        return None, None
    try:
        parsed = parse_structure(structure["text"], str(structure.get("format") or "poscar"))
        return structure["text"], summarize_structure(parsed)
    except Exception:  # noqa: BLE001 - an unreadable recorded structure is simply not shown
        return None, None


def authoritative_calculation_summary(submission_spec: dict) -> dict:
    """The parts of the calculation summary a submission record holds.

    The flow name is the recorded run label (or, for a record without one,
    its run name). The number of jobs is not
    recorded, so it is not shown; workflow fields are added by
    ``calculation_summary_for_workflow`` from the recorded workflow.
    """

    return {
        "flow_name": str(submission_spec.get("label") or submission_spec.get("run_name") or ""),
        "number_of_jobs": None,
        "job_names": [],
        "ready_for_submission": True,
    }


def calculation_summary_for_workflow(calculation_summary, workflow: WorkflowSpec):
    """A carried calculation summary with its workflow fields taken from ``workflow``."""

    if not isinstance(calculation_summary, dict):
        return None
    workflow = validate_workflow_spec(workflow)
    compatible_spec = calculation_spec_from_workflow_spec(workflow)
    theories = {stage.theory for stage in workflow.stages}
    calculation_type = (
        workflow_label_from_spec(compatible_spec)
        if compatible_spec is not None
        else workflow_display_name(workflow)
    )
    return {
        **calculation_summary,
        "calculation_type": calculation_type,
        "workflow_type": calculation_type,
        "calculation_plan": calculation_plan_from_workflow_spec(workflow),
        "theory": next(iter(theories)).value if len(theories) == 1 else "mixed",
        "theory_label": theory_display_name(next(iter(theories))) if len(theories) == 1 else "Mixed",
    }


def selected_desired_output_from_form(workflow: str | None) -> str:
    """The authoritative Desired Output selection of a browser request.

    The ``workflow`` form field (the Desired Output selector, carried unchanged
    by the Prepare, Submit and Monitor forms) decides whether a request is a
    managed Desired Output, exactly as ``workflow_spec_from_form`` and
    ``resolve_workflow_for_structure`` decide it: a managed identifier selects
    that Desired Output; any other value (``custom`` or a legacy recipe name)
    is a user-managed Custom workflow, as is a request with no selection
    (legacy purpose/theory fields or stage JSON alone), since none of those
    receive automatic treatments.
    """

    workflow = optional_form_value(workflow)
    return workflow if desired_output_workflow_spec(workflow) is not None else "custom"


def selected_workflow_context(
    workflow_spec: WorkflowSpec,
    *,
    selected_desired_output: str | None = None,
    available: bool = True,
) -> dict:
    if not available:
        # The workflow could not be recovered from an authoritative record: its
        # identity and stages are suppressed rather than shown as the browser
        # default or as client-posted values.
        return {
            "unavailable": True,
            "json": "",
            "label": "",
            "recipe": None,
            "desired_output": None,
            "desired_output_label": None,
            "hidden_desired_output_label": None,
            "hidden_desired_output_workflow_spec": None,
            "stages": [],
        }
    workflow = validate_workflow_spec(workflow_spec)
    if selected_desired_output is not None:
        # The request's own selection is authoritative.
        desired_output = selected_desired_output
    elif desired_output_workflow_spec(workflow.recipe) is not None:
        # No selection was posted (the default page, a legacy request): only a
        # managed workflow recorded under its own identifier is shown as that
        # Desired Output.
        desired_output = workflow.recipe
    else:
        # A Desired Output is never inferred from a stage sequence: a workflow
        # that merely resembles a managed recipe was not resolved with its
        # automatic treatments, so it is a Custom workflow.
        desired_output = "custom"
    return {
        "json": json.dumps(workflow.to_dict(), sort_keys=True),
        "label": workflow_display_name(workflow),
        "recipe": workflow.recipe,
        "unavailable": False,
        "desired_output": desired_output,
        "desired_output_label": (
            "Custom workflow" if desired_output == "custom" else desired_output_label(desired_output)
        ),
        # A hidden legacy Desired Output (for example ``energy_only`` reached
        # through a legacy form post or a historical record) is not in the
        # menu; the selector shows it as its own selected option so the page
        # never displays a different Desired Output than the one built.
        "hidden_desired_output_label": (
            f"{desired_output_label(desired_output)} (legacy)"
            if desired_output in BROWSER_HIDDEN_DESIRED_OUTPUTS
            else None
        ),
        # The managed workflow of that hidden Desired Output, so that choosing
        # it again in the selector restores its own stages (the menu's
        # ``desired_outputs`` deliberately omit it).
        "hidden_desired_output_workflow_spec": (
            json.dumps(desired_output_workflow_spec(desired_output).to_dict(), sort_keys=True)
            if desired_output in BROWSER_HIDDEN_DESIRED_OUTPUTS
            and desired_output_workflow_spec(desired_output) is not None
            else None
        ),
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
        # Client-supplied (Custom) workflow: only user-selectable stage
        # options are accepted; free-form INCAR/KPOINTS and BMD-generated
        # options are rejected here, before any preview or preparation.
        return canonical_custom_workflow_spec(WorkflowSpec.from_dict(data))

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
        return with_plane_wave_cutoffs(structure_obj, validate_workflow_spec(workflow_spec)), None

    resolution = resolve_default_treatments(
        structure_obj,
        desired_output_workflow_spec(workflow),
        desired_output=workflow,
    )
    resolved = with_plane_wave_cutoffs(structure_obj, resolution.resolved_workflow)
    return resolved, dataclasses.replace(resolution, resolved_workflow=resolved)


def with_plane_wave_cutoffs(structure_obj, workflow_spec: WorkflowSpec) -> WorkflowSpec:
    """Freeze the species-dependent plane-wave cutoff of every relax/phonon stage.

    Every route that builds, prepares or submits a calculation (browser and
    machine API) resolves its workflow here, so the cutoff is decided once,
    from the POTCARs Compute's executable generators select, and carried by
    the workflow itself into previews, the plan digest and the POWER run
    (``backend.calculations.geometry_relaxation``).
    """

    return validate_workflow_spec(
        freeze_plane_wave_cutoffs(
            structure_obj,
            workflow_spec,
            potcar_functional=DEFAULT_POTCAR_FUNCTIONAL,
        )
    )


def resource_request_from_form(
    *,
    cpus=None,
    memory_gb=None,
    walltime=None,
    queue=None,
) -> ResourceRequest:
    """Requested resources: an omitted, empty or "auto" value means Automatic."""

    return ResourceRequest.from_values(
        cpus=optional_form_value(cpus),
        memory_gb=optional_form_value(memory_gb),
        walltime=optional_form_value(walltime),
        queue=optional_form_value(queue),
    )


def execution_resources_from_form(
    *,
    cpus: str | None = None,
    memory_gb: str | None = None,
    walltime: str | None = None,
    queue: str | None = None,
) -> ExecutionResources:
    """Explicit resources with defaults for missing fields (no estimation)."""

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
    workflow_theories = (
        {stage.theory for stage in validate_workflow_spec(workflow).stages}
        if workflow is not None
        else set()
    )
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
    omitted_by_id = {
        str(item.get("consideration_id")): dict(item)
        for item in (
            default_treatment_resolution.omitted_treatments
            if default_treatment_resolution is not None
            else ()
        )
    }
    for consideration in rendered.get("considerations", []):
        automatic_application = applied_by_id.get(consideration.get("id"))
        unsupported_omission = omitted_by_id.get(consideration.get("id"))
        # An omission takes precedence for presentation: even when the
        # treatment reached other stages, the red warning must be shown.
        if unsupported_omission is not None:
            application_state = AUTOMATIC_APPLICATION_OMITTED_UNSUPPORTED
        elif automatic_application is not None:
            application_state = AUTOMATIC_APPLICATION_APPLIED
        elif consideration.get("id") in not_applicable_by_id:
            application_state = AUTOMATIC_APPLICATION_NOT_APPLICABLE
        else:
            application_state = AUTOMATIC_APPLICATION_ADVISORY
        consideration["automatic_application_state"] = application_state
        consideration["automatic_application"] = automatic_application
        consideration["unsupported_omission"] = unsupported_omission
        consideration["presentation"] = (
            "unsupported" if unsupported_omission is not None else "advisory"
        )
        consideration["automatic_not_applicable"] = not_applicable_by_id.get(
            consideration.get("id")
        )
        consideration["workflow_has_hse06_stages"] = Theory.HSE06 in workflow_theories
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
    omission = consideration.get("unsupported_omission")
    if omission is not None:
        return str(omission.get("title") or "Unsupported combination")
    if consideration.get("automatic_application_state") == AUTOMATIC_APPLICATION_APPLIED:
        if consideration.get("id") == DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID:
            return "van der Waals correction applied"
        if consideration.get("id") == SPIN_COMPOSITION_CONSIDERATION_ID:
            return "Spin Polarisation applied"
        if consideration.get("id") == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID:
            return "Spin-Orbit Coupling (SOC) applied"
        if consideration.get("id") == DFT_U_CONSIDERATION_ID:
            return "DFT+U applied"
    if consideration.get("automatic_application_state") == AUTOMATIC_APPLICATION_NOT_APPLICABLE:
        if consideration.get("id") == SOC_HEAVY_ELEMENTS_CONSIDERATION_ID:
            return "Spin-Orbit Coupling (SOC) not used in geometry optimisation"
        if consideration.get("id") == DFT_U_CONSIDERATION_ID:
            return "DFT+U not applied automatically"

    if consideration.get("id") == DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID:
        return "van der Waals Correction"
    return str(consideration.get("display_name") or "Method Consideration")


def _method_consideration_browser_summary(consideration: dict) -> str:
    consideration_id = consideration.get("id")
    omission = consideration.get("unsupported_omission")
    if omission is not None:
        return str(omission.get("message") or "")
    is_applied = (
        consideration.get("automatic_application_state")
        == AUTOMATIC_APPLICATION_APPLIED
    )
    if consideration_id == DISPERSION_TWO_DIMENSIONAL_CONNECTIVITY_CONSIDERATION_ID:
        if is_applied:
            stage_phrase, _ = _automatic_application_stage_phrase(consideration)
            stage_text = stage_phrase or "applicable PBE stages"
            return (
                "Likely 2-dimensional structure detected. The van der Waals "
                f"correction has been included automatically in the {stage_text}. "
                "Choose Custom workflow to configure this manually."
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

    if consideration_id == DFT_U_CONSIDERATION_ID:
        return _dft_u_browser_summary(consideration, is_applied=is_applied)

    reason = str(consideration.get("reason") or "").strip()
    return reason or f"{subject}."


def _dft_u_browser_summary(consideration: dict, *, is_applied: bool) -> str:
    evidence = consideration.get("observed_evidence") or {}
    evaluation = evidence.get("dft_u_evaluation") or {}
    anion = evaluation.get("deciding_anion")
    compound = {"O": "oxide", "F": "fluoride"}.get(anion, "compound")
    elements = _human_join(consideration.get("trigger_elements") or ())
    subject = f"{elements} in an {compound} detected" if elements else "Relevant composition detected"

    if consideration.get("status") == DFT_U_SUPPRESSED_STATUS:
        return (
            f"{subject}. Every charge-balanced oxidation state pymatgen finds puts "
            f"{elements} at d0, so BMD Compute does not apply DFT+U automatically. "
            "Choose Custom workflow to add it manually."
        )

    species = ((evaluation.get("parameters") or {}).get("species")) or {}
    values = _human_join(
        f"{symbol} U = {parameters.get('U'):g} eV, J = {parameters.get('J'):g} eV"
        for symbol, parameters in sorted(species.items())
        if float(parameters.get("U") or 0) > 0
    )
    value_text = f" ({values})" if values else ""
    if is_applied:
        stage_phrase, _ = _automatic_application_stage_phrase(consideration)
        stage_text = f" in the {stage_phrase}" if stage_phrase else ""
        hse_text = (
            " DFT+U applies to the PBE geometry optimisation only; the HSE06 "
            "electronic-structure stages do not use +U."
            if consideration.get("workflow_has_hse06_stages")
            else ""
        )
        gate_text = (
            " pymatgen found no charge-balanced oxidation state, so the Materials "
            "Project rule was applied without the d0 check."
            if evaluation.get("gate") == DFT_U_GATE_UNAVAILABLE
            else ""
        )
        return (
            f"{subject}. DFT+U has been included automatically{stage_text}{value_text}, "
            "following the Materials Project/pymatgen GGA+U oxide/fluoride "
            "parameterisation. These are standard empirical values, not values "
            f"determined for this material.{hse_text}{gate_text} Choose Custom "
            "workflow to configure this manually."
        )
    return (
        f"{subject}. Suggested to activate the DFT+U Advanced Option"
        f"{value_text}, which uses the Materials Project/pymatgen oxide/fluoride "
        "parameterisation."
    )


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
    # Repeated stages of one kind (the two passes of the BMD Standard
    # Geometry Relaxation) are named once: "PBE Geometry Optimisation stages".
    distinct = list(dict.fromkeys(labels))
    return f"{_human_join(distinct)} {noun}", len(labels)


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
    execution_resources: ExecutionResources | ResourceRequest | None = None,
    timestamp: str | None = None,
    submission_attempt_id: str | None = None,
):
    if execution_resources is None:
        execution_resources = default_execution_resources()
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
    execution_resources: ExecutionResources | ResourceRequest,
    timestamp: str | None = None,
    submission_attempt_id: str | None = None,
    default_treatment_resolution: ResolvedDefaultWorkflow | None = None,
    planned_resources: str | None = None,
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
    # New-calculation admission boundary. Every route that builds, prepares or
    # submits a new calculation reaches this point after automatic treatments
    # are resolved, whatever form the request took (Desired Output, Custom
    # workflow JSON, legacy purpose/theory/modifiers). Structural validators,
    # which also read historical records, are deliberately left unchanged.
    workflow_spec = require_admissible_new_calculation(workflow_spec)
    # Every prepared plan carries the species-dependent plane-wave cutoff,
    # whichever route built it (deterministic; a resolved workflow is unchanged).
    workflow_spec = with_plane_wave_cutoffs(structure_obj, workflow_spec)
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
    resource_selection = resolve_resources_for_workflow(
        structure_obj,
        workflow_spec,
        execution_resources,
        potcar_functional=potcar_functional,
    )
    execution_resources = resource_selection.resources
    if planned_resources:
        current = planned_resources_token(execution_resources, resource_selection.record)
        if current != planned_resources:
            raise CalculationValidationError(
                "The resources for this calculation changed after it was built.",
                suggestion=(
                    "Build the calculation again to review the current resource "
                    "recommendation before preparing or submitting it."
                ),
                diagnostic={"code": "planned_resources_changed"},
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
        "resource_selection": resource_selection.record,
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
        if default_treatment_resolution.dft_u is not None:
            flow_spec["automatic_treatments"]["dft_u"] = prepared_dft_u_record(
                default_treatment_resolution.dft_u,
                structure_obj,
                workflow_spec,
                resources=execution_resources,
                potcar_functional=potcar_functional,
            )
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


def resolve_resources_for_workflow(
    structure_obj,
    workflow_spec: WorkflowSpec,
    requested: ExecutionResources | ResourceRequest,
    *,
    potcar_functional: str,
):
    """Authoritative resource resolution for the browser and the machine API.

    Explicit values are kept; Automatic values come from
    ``backend.calculations.automatic_resources`` using the stage inputs that
    Compute generates for this structure and workflow.
    """

    if isinstance(requested, ExecutionResources):
        requested = ResourceRequest.from_execution_resources(requested)
    preview_resources = ExecutionResources(
        cpus=requested.cpus if requested.cpus is not None else AUTOMATIC_CPUS,
        memory_gb=requested.memory_gb if requested.memory_gb is not None else AUTOMATIC_MEMORY_GB,
        queue=requested.queue if requested.queue is not None else DEFAULT_PARTITION,
    )
    stage_previews = generated_input_stage_previews(
        structure_obj,
        workflow_spec,
        resources=preview_resources,
        potcar_functional=potcar_functional,
    )
    return resolve_execution_resources(
        requested,
        stage_previews=stage_previews,
        potcar_functional=potcar_functional,
    )


@dataclass(frozen=True)
class CalculationPlan:
    """Everything the Build route resolves for one request, before any remote work."""

    structure: object
    workflow_spec: WorkflowSpec
    default_treatment_resolution: ResolvedDefaultWorkflow | None
    method_considerations: dict | None
    execution_resources: ExecutionResources
    summary: dict
    generated_inputs: dict
    submission_spec: dict


def plan_calculation_request(
    *,
    structure_text: str,
    fmt: str,
    desired_output: str | None = None,
    custom_workflow: dict | None = None,
    resources: dict | None = None,
    timestamp: str | None = None,
    submission_attempt_id: str | None = None,
) -> CalculationPlan:
    """Resolve a calculation exactly as the Build route does, with no remote work.

    This is the machine API's entry into BMD Compute's scientific resolution. It
    calls the same module-level functions, in the same order, as
    ``build_workflow``: request workflow, execution resources, structure
    parsing, workflow resolution with automatic treatments, method
    considerations, and ``build_submission_state_from_structure`` (which applies
    the new-calculation admission policy and builds the submission
    specification). It adds no methodology of its own.

    ``timestamp`` and ``submission_attempt_id`` are passed to
    ``build_submission_state_from_structure`` exactly as the browser's Prepare
    and Submit routes pass the values recovered from their signed submission
    identity, so that a retried machine attempt rebuilds the identical
    submission specification.
    """

    if (desired_output is None) == (custom_workflow is None):
        raise CalculationValidationError(
            "Request exactly one of a Desired Output or a Custom workflow.",
            suggestion="Send either a Desired Output identifier or a Custom workflow specification.",
        )
    if desired_output is not None and desired_output_workflow_spec(desired_output) is None:
        raise CalculationValidationError(
            "Unknown Desired Output.",
            suggestion="Use one of the Desired Output identifiers offered by BMD Compute.",
            diagnostic={"code": "unknown_desired_output"},
        )
    # The browser sends workflow="custom" together with the Custom workflow JSON.
    workflow = desired_output if desired_output is not None else "custom"
    try:
        workflow_spec = workflow_spec_from_form(
            workflow_spec_json=(
                json.dumps(custom_workflow, sort_keys=True)
                if custom_workflow is not None
                else None
            ),
            workflow=workflow,
        )
    except CalculationValidationError:
        raise
    except ValueError as exc:
        # Unknown stage type, theory or modifier names are raised as plain
        # ValueError by the workflow models; classify them for machine callers.
        raise CalculationValidationError(
            "The Custom workflow names an unsupported stage type, theory or modifier.",
            suggestion="Use the stage types, theories and modifiers offered by BMD Compute.",
            diagnostic={"code": "unsupported_workflow_vocabulary"},
        ) from exc
    resources = dict(resources or {})
    # Omitted resource values mean Automatic; explicit values are kept exactly.
    resource_request = resource_request_from_form(
        cpus=resources.get("cpus"),
        memory_gb=resources.get("memory_gb"),
        walltime=resources.get("walltime"),
        queue=resources.get("queue"),
    )
    try:
        structure_obj = parse_structure(structure_text, fmt)
    except StructureValidationError:
        raise
    except Exception as exc:
        # parse_structure classifies POSCAR failures itself; CIF parser errors
        # propagate from pymatgen. Machine callers get the same classification
        # for both, without the parser's internal text.
        raise StructureValidationError(
            "The structure could not be read as a valid CIF file.",
            suggestion="Check that the CIF file contains one complete crystal structure.",
        ) from exc
    workflow_spec, default_treatment_resolution = resolve_workflow_for_structure(
        structure_obj,
        workflow_spec,
        workflow=workflow,
    )
    method_considerations = method_considerations_for_workflow_state(
        structure_obj,
        workflow=workflow_spec,
        default_treatment_resolution=default_treatment_resolution,
    )
    summary, _calculation_summary, generated_inputs, submission_spec = build_submission_state_from_structure(
        structure_obj=structure_obj,
        structure_text=structure_text,
        fmt=fmt,
        workflow_spec=workflow_spec,
        execution_resources=resource_request,
        timestamp=timestamp,
        submission_attempt_id=submission_attempt_id,
        default_treatment_resolution=default_treatment_resolution,
    )
    execution_resources = selected_resources_from_submission(
        submission_spec, resource_request
    ).resources
    return CalculationPlan(
        structure=structure_obj,
        workflow_spec=workflow_spec,
        default_treatment_resolution=default_treatment_resolution,
        method_considerations=method_considerations,
        execution_resources=execution_resources,
        summary=summary,
        generated_inputs=generated_inputs,
        submission_spec=submission_spec,
    )


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
    # Before Build, show only the user's selection: the default Desired Output
    # (Energy only) as its unresolved base recipe. Automatic treatments, Method
    # Considerations, generated inputs and provenance are resolved and shown
    # only by a successful Build Calculation.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            structure_text=structure,
            fmt=fmt,
            summary=summary,
            selected_workflow=default_workflow_spec(),
        ),
    )


@app.post("/resume")
def resume_existing_calculation(
    request: Request,
    job_id: str = Form(...),
    load_results: str = Form("false"),
):
    monitoring_result = monitor_job(job_id)
    # The submitted workflow comes only from the job record monitoring read;
    # without one, the workflow is not shown (never the browser default).
    recorded_spec = (
        monitoring_result.pop(RESUMED_SUBMISSION_SPEC_KEY, None)
        if isinstance(monitoring_result, dict)
        else None
    )
    recovered = (
        authoritative_workflow_identity(recorded_spec)
        if isinstance(recorded_spec, dict)
        else None
    )
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
            selected_workflow=recovered[0] if recovered is not None else None,
            selected_desired_output=recovered[1] if recovered is not None else None,
            workflow_identity_available=recovered is not None,
            show_submitted_workflow=True,
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
    # The authoritative Desired Output selection, once the request's workflow
    # has been read (None until then: the default page identity is shown).
    selection = None
    # The selection as requested, before automatic treatments: what a failed
    # request redisplays, so it never looks like a resolved calculation.
    requested_workflow_spec = workflow_spec
    resource_request = ResourceRequest()
    execution_resources = SelectedResources(request=resource_request)
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
        selection = selected_desired_output_from_form(workflow)
        requested_workflow_spec = workflow_spec
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        resource_request = resource_request_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        execution_resources = SelectedResources(request=resource_request)
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
            execution_resources=resource_request,
            default_treatment_resolution=default_treatment_resolution,
        )
        execution_resources = selected_resources_from_submission(
            submission_spec, resource_request
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            selected_desired_output=selection,
            structure_text=structure,
            fmt=fmt,
            selected_spec=None,
            selected_workflow=requested_workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            selected_desired_output=selection,
            structure_text=structure,
            fmt=fmt,
            selected_spec=None,
            selected_workflow=requested_workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            selected_desired_output=selection,
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
    planned_resources: str | None = Form(None),
):
    preparation_started = time.perf_counter()
    # Accepted for legacy clients only; the signed server identity is authoritative.
    del created_at
    calculation_spec = default_calculation_spec()
    workflow_spec = default_workflow_spec()
    # The authoritative Desired Output selection, once the request's workflow
    # has been read (None until then: the default page identity is shown).
    selection = None
    # The selection as requested, before automatic treatments: what a failed
    # request redisplays, so it never looks like a resolved calculation.
    requested_workflow_spec = workflow_spec
    resource_request = ResourceRequest()
    execution_resources = SelectedResources(request=resource_request)
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
        selection = selected_desired_output_from_form(workflow)
        requested_workflow_spec = workflow_spec
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        resource_request = resource_request_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        execution_resources = SelectedResources(request=resource_request)
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
            execution_resources=resource_request,
            timestamp=identity["run_timestamp"],
            submission_attempt_id=identity["attempt_id"],
            default_treatment_resolution=default_treatment_resolution,
            planned_resources=optional_form_value(planned_resources),
        )
        execution_resources = selected_resources_from_submission(
            submission_spec, resource_request
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            selected_desired_output=selection,
            structure_text=structure,
            fmt=fmt,
            selected_spec=None,
            selected_workflow=requested_workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            selected_desired_output=selection,
            structure_text=structure,
            fmt=fmt,
            selected_spec=None,
            selected_workflow=requested_workflow_spec,
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
            selected_desired_output=selection,
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
    planned_resources: str | None = Form(None),
):
    # Accepted for legacy clients only; the signed server identity is authoritative.
    del created_at
    calculation_spec = default_calculation_spec()
    workflow_spec = default_workflow_spec()
    # The authoritative Desired Output selection, once the request's workflow
    # has been read (None until then: the default page identity is shown).
    selection = None
    # The selection as requested, before automatic treatments: what a failed
    # request redisplays, so it never looks like a resolved calculation.
    requested_workflow_spec = workflow_spec
    resource_request = ResourceRequest()
    execution_resources = SelectedResources(request=resource_request)
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
        selection = selected_desired_output_from_form(workflow)
        requested_workflow_spec = workflow_spec
        calculation_spec = (
            calculation_spec_from_workflow_spec(workflow_spec)
            or default_calculation_spec()
        )
        resource_request = resource_request_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        execution_resources = SelectedResources(request=resource_request)
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
            execution_resources=resource_request,
            timestamp=identity["run_timestamp"],
            submission_attempt_id=identity["attempt_id"],
            default_treatment_resolution=default_treatment_resolution,
            planned_resources=optional_form_value(planned_resources),
        )
        execution_resources = selected_resources_from_submission(
            submission_spec, resource_request
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            selected_desired_output=selection,
            structure_text=structure,
            fmt=fmt,
            selected_spec=None,
            selected_workflow=requested_workflow_spec,
            selected_resources=execution_resources,
            exc=exc,
        )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            selected_desired_output=selection,
            structure_text=structure,
            fmt=fmt,
            selected_spec=None,
            selected_workflow=requested_workflow_spec,
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
            selected_desired_output=selection,
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
    # Workflow identity, stages and treatments shown while monitoring come only
    # from the authenticated job record. The posted workflow fields (purpose,
    # theory, modifiers, workflow_spec_json, workflow, method) are accepted for
    # form compatibility and never read: until the record is read, and when it
    # cannot identify its workflow, the workflow is not shown at all.
    del purpose, theory, modifiers, workflow_spec_json, workflow, method
    resource_request = ResourceRequest()
    execution_resources = SelectedResources(request=resource_request)
    monitor_state = {}
    summary = None
    calculation_summary = None
    generated_inputs = None
    submission_spec = None
    try:
        monitor_state = monitor_state_from_json(monitor_state_json)
        resource_request = resource_request_from_form(
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=queue,
        )
        execution_resources = SelectedResources(request=resource_request)
        summary = monitor_state.get("summary")
        calculation_summary = monitor_state.get("calculation")
        generated_inputs = monitor_state.get("generated_inputs")
        submission_spec = monitor_state.get("submission_spec")
        if not isinstance(submission_spec, dict):
            raise CalculationValidationError(
                "The saved submission state could not be read.",
                suggestion="Rebuild the calculation, then refresh monitoring again.",
            )
        submission = submission_spec.get("submission")
        if not isinstance(submission, dict):
            raise CalculationValidationError(
                "The saved submission identity could not be read.",
                suggestion="Rebuild the calculation, then refresh monitoring again.",
            )
        identity = authoritative_submission_identity(
            submission.get("identity_token"),
            submission.get("attempt_id"),
        )
    except CalculationValidationError as exc:
        return calculation_error_response(
            request,
            workflow_identity_available=False,
            structure_text=structure,
            fmt=fmt,
            selected_resources=execution_resources,
            exc=exc,
        )
    except StructureValidationError as exc:
        return structure_error_response(
            request,
            workflow_identity_available=False,
            structure_text=structure,
            fmt=fmt,
            selected_resources=execution_resources,
            exc=exc,
        )
    monitoring_result, authoritative_job_record = monitor_submitted_job(
        job_id,
        authenticated_attempt_id=identity["attempt_id"],
    )
    authoritative_submission_spec = (
        authoritative_job_record.get("submission_spec")
        if authoritative_job_record is not None
        else None
    )
    submission_spec = (
        authoritative_submission_spec
        if isinstance(authoritative_submission_spec, dict)
        else None
    )
    remote_preparation = (
        remembered_successful_preparation(submission_spec)
        if submission_spec is not None
        else None
    )
    submission_result = (
        remembered_successful_submission(
            submission_spec,
            job_id,
            submitted_at=submitted_at,
        )
        if submission_spec is not None
        else None
    )
    if submission_spec is not None:
        try:
            execution_resources = selected_resources_from_submission(
                submission_spec, resource_request
            )
        except (CalculationValidationError, KeyError, TypeError, ValueError):
            pass
    recovered = (
        authoritative_workflow_identity(submission_spec)
        if submission_spec is not None
        else None
    )
    # Every panel shown while monitoring comes from the authenticated job record
    # or is not shown, and the client-carried values are never rendered: the
    # structure field and Structure Summary come from the recorded structure
    # (parsing and summarising it takes milliseconds; nothing is rebuilt), the
    # calculation summary from the recorded workflow and label, and Generated
    # Inputs are not shown (the record does not hold the generated input files).
    recorded_structure_text, display_summary = authoritative_structure_summary(submission_spec)
    display_calculation_summary = (
        calculation_summary_for_workflow(
            authoritative_calculation_summary(submission_spec),
            recovered[0],
        )
        if recovered is not None
        else None
    )
    display_generated_inputs = None
    monitor_state_override = None
    if (
        submission_spec is None
        and monitoring_result.get("identity_retry_allowed") is True
    ):
        monitor_state_override = _authenticated_monitor_retry_state_json(
            identity=identity,
            identity_token=submission["identity_token"],
            summary=summary,
            calculation_summary=calculation_summary,
            generated_inputs=generated_inputs,
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context(
            selected_desired_output=recovered[1] if recovered is not None else None,
            workflow_identity_available=recovered is not None,
            show_submitted_workflow=True,
            structure_text=recorded_structure_text if recorded_structure_text is not None else structure,
            fmt=fmt,
            summary=display_summary,
            selected_workflow=recovered[0] if recovered is not None else None,
            selected_resources=execution_resources,
            calculation_summary=display_calculation_summary,
            generated_inputs=display_generated_inputs,
            submission_spec=submission_spec,
            remote_preparation=remote_preparation,
            submission_result=submission_result,
            monitoring_result=monitoring_result,
            results_summary=None,
            collapse_structure_input=True,
            monitor_state_override=monitor_state_override,
        ),
    )


# Authenticated machine API (planning only). It is a separate router with its
# own authentication; the browser routes above are unchanged and do not use it.
app.include_router(build_machine_api_router(planner=plan_calculation_request))
