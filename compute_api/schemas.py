"""Strict validation of machine-API plan requests.

Validation is explicit and closed: every object accepts only the listed keys,
and errors name the offending field and a fixed problem code without echoing
the submitted value. This module checks request *shape* only. Whether a
Desired Output, stage, theory, modifier, option or resource value is
scientifically or operationally admissible is decided by BMD Compute's
existing methodology and resource code, never here.

Request (``schema_version`` 1)::

    {
      "structure": {"format": "poscar" | "cif", "text": "<structure file text>"},
      "workflow": {"desired_output": "<Desired Output id>"}
               | {"custom": {"stages": [{"stage_type": ..., "theory": ...,
                                         "modifiers": [...], "options": {...}}]}},
      "resources": {"cpus": int, "memory_gb": int, "walltime": "HH:MM:SS",
                    "queue": str}            # optional; every key optional
    }
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MAX_REQUEST_BYTES = 3 * 1024 * 1024
MAX_STRUCTURE_CHARS = 2 * 1024 * 1024
MAX_CUSTOM_STAGES = 16
MAX_MODIFIERS = 8
MAX_IDENTIFIER_CHARS = 64
STRUCTURE_FORMATS = ("poscar", "cif")

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_WALLTIME = re.compile(r"^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$")
_QUEUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

TOP_LEVEL_KEYS = frozenset({"structure", "workflow", "resources"})
STRUCTURE_KEYS = frozenset({"format", "text"})
WORKFLOW_KEYS = frozenset({"desired_output", "custom"})
CUSTOM_KEYS = frozenset({"stages"})
STAGE_KEYS = frozenset({"stage_type", "theory", "modifiers", "options"})
RESOURCE_KEYS = frozenset({"cpus", "memory_gb", "walltime", "queue"})


class PlanRequestError(Exception):
    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("invalid plan request")
        self.errors = errors


@dataclass(frozen=True)
class PlanRequest:
    structure_format: str
    structure_text: str
    desired_output: str | None
    custom_workflow: dict[str, Any] | None
    resources: dict[str, Any]

    @property
    def workflow_mode(self) -> str:
        return "desired_output" if self.desired_output is not None else "custom"


def _error(errors: list, field: str, problem: str) -> None:
    errors.append({"field": field, "problem": problem})


def _closed_object(errors: list, value: Any, field: str, allowed: frozenset[str]) -> dict | None:
    if not isinstance(value, dict):
        _error(errors, field, "must_be_object")
        return None
    for key in sorted(set(value) - allowed, key=str):
        _error(errors, f"{field}.{key}" if field else str(key), "unknown_field")
    return value


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_plan_request(document: Any) -> PlanRequest:
    errors: list[dict[str, str]] = []
    top = _closed_object(errors, document, "", TOP_LEVEL_KEYS)
    if top is None:
        raise PlanRequestError(errors)

    structure_format = structure_text = None
    structure = _closed_object(errors, top.get("structure"), "structure", STRUCTURE_KEYS) if "structure" in top else None
    if "structure" not in top:
        _error(errors, "structure", "required")
    elif structure is not None:
        fmt = structure.get("format")
        if not isinstance(fmt, str) or fmt.strip().lower() not in STRUCTURE_FORMATS:
            _error(errors, "structure.format", "must_be_poscar_or_cif")
        else:
            structure_format = fmt.strip().lower()
        text = structure.get("text")
        if not isinstance(text, str) or not text.strip():
            _error(errors, "structure.text", "must_be_nonempty_string")
        elif len(text) > MAX_STRUCTURE_CHARS:
            _error(errors, "structure.text", "too_large")
        elif "\x00" in text:
            _error(errors, "structure.text", "invalid_characters")
        else:
            structure_text = text

    desired_output = custom_workflow = None
    if "workflow" not in top:
        _error(errors, "workflow", "required")
    else:
        workflow = _closed_object(errors, top.get("workflow"), "workflow", WORKFLOW_KEYS)
        if workflow is not None:
            present = [key for key in ("desired_output", "custom") if key in workflow]
            if len(present) != 1:
                _error(errors, "workflow", "exactly_one_of_desired_output_or_custom")
            elif present[0] == "desired_output":
                value = workflow["desired_output"]
                if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                    _error(errors, "workflow.desired_output", "must_be_identifier")
                else:
                    desired_output = value
            else:
                custom_workflow = _parse_custom_workflow(errors, workflow["custom"])

    resources: dict[str, Any] = {}
    if "resources" in top and top["resources"] is not None:
        resource_object = _closed_object(errors, top["resources"], "resources", RESOURCE_KEYS)
        if resource_object is not None:
            for key in ("cpus", "memory_gb"):
                if key in resource_object:
                    if not _strict_int(resource_object[key]) or resource_object[key] <= 0:
                        _error(errors, f"resources.{key}", "must_be_positive_integer")
                    else:
                        resources[key] = resource_object[key]
            if "walltime" in resource_object:
                value = resource_object["walltime"]
                if not isinstance(value, str) or not _WALLTIME.fullmatch(value):
                    _error(errors, "resources.walltime", "must_be_hh_mm_ss")
                else:
                    resources["walltime"] = value
            if "queue" in resource_object:
                value = resource_object["queue"]
                if not isinstance(value, str) or not _QUEUE.fullmatch(value):
                    _error(errors, "resources.queue", "must_be_identifier")
                else:
                    resources["queue"] = value

    if errors:
        raise PlanRequestError(errors)
    return PlanRequest(
        structure_format=structure_format,
        structure_text=structure_text,
        desired_output=desired_output,
        custom_workflow=custom_workflow,
        resources=resources,
    )


def _parse_custom_workflow(errors: list, value: Any) -> dict | None:
    custom = _closed_object(errors, value, "workflow.custom", CUSTOM_KEYS)
    if custom is None:
        return None
    stages = custom.get("stages")
    if not isinstance(stages, list) or not stages:
        _error(errors, "workflow.custom.stages", "must_be_nonempty_list")
        return None
    if len(stages) > MAX_CUSTOM_STAGES:
        _error(errors, "workflow.custom.stages", "too_many_stages")
        return None
    parsed_stages = []
    for index, stage in enumerate(stages):
        field = f"workflow.custom.stages[{index}]"
        stage_object = _closed_object(errors, stage, field, STAGE_KEYS)
        if stage_object is None:
            continue
        for key in ("stage_type", "theory"):
            item = stage_object.get(key)
            if not isinstance(item, str) or not _IDENTIFIER.fullmatch(item):
                _error(errors, f"{field}.{key}", "must_be_identifier")
        modifiers = stage_object.get("modifiers", [])
        if (
            not isinstance(modifiers, list)
            or len(modifiers) > MAX_MODIFIERS
            or not all(isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in modifiers)
        ):
            _error(errors, f"{field}.modifiers", "must_be_identifier_list")
        options = stage_object.get("options", {})
        if not isinstance(options, dict):
            _error(errors, f"{field}.options", "must_be_object")
        parsed_stages.append(
            {
                "stage_type": stage_object.get("stage_type"),
                "theory": stage_object.get("theory"),
                "modifiers": list(modifiers) if isinstance(modifiers, list) else [],
                "options": dict(options) if isinstance(options, dict) else {},
            }
        )
    return {"stages": parsed_stages}
