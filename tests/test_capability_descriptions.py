"""The capability payload's description sections match what Compute actually does.

Each section added to the read-only capability producer (release identity,
Desired Outputs, new-calculation admission, workflow protocols, experimental
capabilities, validation statement) is checked here against the authoritative
planner (``main.plan_calculation_request``), the admission policy and the
registry, so a description cannot drift from behaviour unnoticed.

Planning uses the SYNTHETIC POTCAR ENMAX metadata installed by ``conftest``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import main
import test_input_reference_contract as reference_cases
from backend.calculations import admission, geometry_relaxation as gr
from backend.calculations.capabilities import (
    STRUCTURE_SEARCH_EXECUTION_ENABLED_VALUE,
    STRUCTURE_SEARCH_EXECUTION_SETTING,
    build_capability_payload,
    release_identity,
)
from backend.calculations.models import StageType
from backend.calculations.registry import (
    BROWSER_HIDDEN_DESIRED_OUTPUTS,
    CalculationValidationError,
    _desired_output_options,
    desired_output_workflow_spec,
    supported_stage_modifier_combinations,
)
from backend.calculations.vasp_stage_definitions import describe_stage
from backend.structure_search import admission as search_admission
from backend.structure_search import spec as search_spec
from compute_api.projection import plan_stage_previews

REPO_ROOT = Path(__file__).resolve().parents[1]
SI_POSCAR = reference_cases.SI_POSCAR
PAYLOAD = build_capability_payload(include_provenance=False)


def _plan(desired_output=None, custom=None, structure=SI_POSCAR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main.plan_calculation_request(
            structure_text=structure, fmt="poscar", desired_output=desired_output, custom_workflow=custom
        )


def _previews(plan):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return plan_stage_previews(plan)


def _stage(stage_type, theory="pbe", modifiers=()):
    return {"stage_type": stage_type, "theory": theory, "modifiers": list(modifiers), "options": {}}


def _shape(stages):
    return [(stage["stage_type"], stage["theory"], (stage.get("options") or {}).get("geometry_relaxation", {}).get("pass"))
            for stage in stages]


def _planned_shape(plan):
    return [(stage.stage_type.value, stage.theory.value, gr.stage_protocol_pass(stage)) for stage in plan.workflow_spec.stages]


# ------------------------------------------------------------ release identity


def test_release_identity_never_claims_a_deployment():
    identity = PAYLOAD["release_identity"]
    assert identity["deployment_verified"] is False
    assert identity["tag"] is None and identity["released_code"] is False
    assert "not evidence" in identity["deployment_statement"]


@pytest.mark.parametrize(
    ("tag", "dirty", "released"),
    [("v1.3.1", False, True), ("v1.3.1", True, False), (None, False, False), ("v1.3.1", None, False)],
)
def test_released_code_requires_an_exact_tag_and_a_clean_checkout(tag, dirty, released):
    identity = release_identity({"commit": "abc", "dirty": dirty}, tag=tag)
    assert identity["released_code"] is released
    assert identity["deployment_verified"] is False


def test_release_tag_matches_git_when_git_is_available():
    try:
        expected = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("git unavailable")
    identity = build_capability_payload()["release_identity"]
    assert identity["tag"] == (expected.stdout.strip() or None if expected.returncode == 0 else None)


# ------------------------------------------------------------ Desired Outputs


def test_desired_outputs_are_exactly_the_registry_identifiers():
    described = {entry["value"]: entry for entry in PAYLOAD["desired_outputs"]}
    for value, entry in described.items():
        if entry["workflow_mode"] == "custom_workflow":
            assert desired_output_workflow_spec(value) is None
        else:
            assert desired_output_workflow_spec(value) is not None, value
        assert entry["browser_hidden"] is (value in BROWSER_HIDDEN_DESIRED_OUTPUTS)
    menu = {option["value"] for option in _desired_output_options()}
    assert menu == {value for value, entry in described.items() if not entry["browser_hidden"]}


@pytest.mark.parametrize(
    "value",
    [entry["value"] for entry in PAYLOAD["desired_outputs"] if entry["workflow_mode"] == "bmd_managed_desired_output"],
)
def test_each_described_desired_output_plans_the_described_stage_sequence(value):
    entry = next(item for item in PAYLOAD["desired_outputs"] if item["value"] == value)
    # Silicon triggers no automatic treatment, so the planned shape equals the request.
    assert _planned_shape(_plan(desired_output=value)) == _shape(entry["workflow_spec"]["stages"])


# ------------------------------------------------------------ admission


def test_closed_combinations_are_exactly_the_admission_policy():
    closed = PAYLOAD["new_calculation_admission"]["closed_combinations"]
    assert [(item["theory"], item["modifier"]) for item in closed] == [
        (theory.value, modifier.value) for theory, modifier in admission.UNSUPPORTED_NEW_CALCULATION_COMBINATIONS
    ]
    assert PAYLOAD["new_calculation_admission"]["policy_id"] == admission.ADMISSION_POLICY_ID
    assert PAYLOAD["new_calculation_admission"]["policy_version"] == admission.ADMISSION_POLICY_VERSION


def test_stage_modifier_support_still_lists_registry_acceptance_unchanged():
    hse_static = next(
        item for item in PAYLOAD["stage_modifier_support"] if (item["stage_type"], item["theory"]) == ("static", "hse06")
    )
    assert ["soc"] in hse_static["supported_modifier_combinations"]


def test_admissible_support_is_registry_support_minus_closed_combinations():
    closed = {(item["theory"], item["modifier"]) for item in PAYLOAD["new_calculation_admission"]["closed_combinations"]}
    for entry in PAYLOAD["new_calculation_admission"]["admissible_modifier_support"]:
        registry = [
            sorted(modifier.value for modifier in combination)
            for combination in supported_stage_modifier_combinations(entry["stage_type"], entry["theory"])
        ]
        expected = [combo for combo in registry if not any((entry["theory"], m) in closed for m in combo)]
        assert entry["admissible_modifier_combinations"] == expected, (entry["stage_type"], entry["theory"])


@pytest.mark.parametrize("theory", ["pbe", "hse06"])
def test_planner_refuses_exactly_the_described_closed_static_combinations(theory):
    admission_section = PAYLOAD["new_calculation_admission"]
    closed_code = admission_section["closed_combinations"][0]["diagnostic_code"]
    admissible = next(
        item["admissible_modifier_combinations"]
        for item in admission_section["admissible_modifier_support"]
        if (item["stage_type"], item["theory"]) == ("static", theory)
    )
    registry = [
        sorted(m.value for m in combination) for combination in supported_stage_modifier_combinations("static", theory)
    ]
    for modifiers in registry:
        try:
            _plan(custom={"stages": [_stage("static", theory, modifiers)]})
            refused_code = None
        except CalculationValidationError as exc:
            refused_code = (exc.diagnostic or {}).get("code")
        if modifiers in admissible:
            assert refused_code != closed_code, (theory, modifiers)
        else:
            assert refused_code == closed_code, (theory, modifiers)


def test_planner_applies_the_described_phonon_restrictions():
    restrictions = PAYLOAD["new_calculation_admission"]["phonon_workflow_restrictions"]
    assert restrictions["excluded_modifiers"] == sorted(m.value for m in admission.PHONON_EXCLUDED_MODIFIERS)
    with pytest.raises(CalculationValidationError) as raised:
        _plan(custom={"stages": [_stage("relax", modifiers=["spin_polarized"]), _stage("phonon_forces")]})
    assert raised.value.diagnostic["code"] == restrictions["diagnostic_code"]


# ------------------------------------------------------------ protocols


def test_protocol_desired_outputs_are_those_whose_plans_use_the_two_pass_protocol():
    described = set(PAYLOAD["protocols"]["geometry_relaxation"]["desired_outputs"])
    for entry in PAYLOAD["desired_outputs"]:
        if entry["workflow_mode"] != "bmd_managed_desired_output":
            continue
        passes = [p for (_, _, p) in _planned_shape(_plan(desired_output=entry["value"])) if p is not None]
        assert (passes == [1, 2]) is (entry["value"] in described), entry["value"]


@pytest.mark.parametrize("value", PAYLOAD["protocols"]["geometry_relaxation"]["desired_outputs"])
def test_managed_relaxations_run_the_described_protocol_not_the_stage_defaults(value):
    protocol = PAYLOAD["protocols"]["geometry_relaxation"]
    stage_defaults = describe_stage("relax", "pbe")["base_bmd_incar_amendments"]["executable_defaults"]
    plan = _plan(desired_output=value)
    previews = _previews(plan)
    relaxes = [i for i, stage in enumerate(plan.workflow_spec.stages) if stage.stage_type is StageType.RELAX]
    for pass_number, index in zip((1, 2), relaxes):
        incar = previews[index]["input_set"].incar
        record = plan.submission_spec["flow_spec"]["workflow_spec"]["stages"][index]["options"]["plane_wave_cutoff"]
        assert incar["ENCUT"] == record["ENCUT"] == gr.resolved_encut([record["max_enmax_ev"]])
        assert record["rule"] == {"id": PAYLOAD["protocols"]["plane_wave_cutoff"]["id"],
                                  "version": PAYLOAD["protocols"]["plane_wave_cutoff"]["version"]}
        for key, expected in protocol["passes"][str(pass_number)].items():
            assert incar[key] == expected, key
        for key, expected in protocol["common_settings"].items():
            if expected is None:
                assert key not in incar, key
            else:
                assert incar[key] == expected, key
    # The stage-definition default the description says is replaced really is.
    assert stage_defaults["ADDGRID"] is True
    assert "ADDGRID" not in previews[relaxes[0]]["input_set"].incar


def test_custom_relaxations_also_use_the_resolved_cutoff_as_described():
    cutoff = PAYLOAD["protocols"]["plane_wave_cutoff"]
    assert "Custom" in cutoff["applies_to"] and cutoff["applies_to_stage_types"] == ["phonon_forces", "relax"]
    plan = _plan(custom={"stages": [_stage("relax")]})
    incar = _previews(plan)[0]["input_set"].incar
    record = plan.submission_spec["flow_spec"]["workflow_spec"]["stages"][0]["options"]["plane_wave_cutoff"]
    assert incar["ENCUT"] == record["ENCUT"] == gr.resolved_encut([entry["ENMAX"] for entry in record["potcars"]])
    assert (incar["EDIFF"], incar["EDIFFG"]) == tuple(PAYLOAD["protocols"]["geometry_relaxation"]["passes"]["1"].values())
    assert "ADDGRID" not in incar


def test_replaced_settings_are_the_ones_relaxation_incar_settings_sets():
    replaced = PAYLOAD["protocols"]["stage_definition_defaults"]["settings_replaced_for_new_calculations"]
    plan = _plan(desired_output="relaxed_structure")
    settings = gr.relaxation_incar_settings(plan.workflow_spec, 0)
    assert replaced == sorted(settings)


# ------------------------------------------------------------ experimental capabilities


def test_structure_search_is_described_as_experimental_and_not_established():
    (search,) = PAYLOAD["experimental_capabilities"]
    assert search["id"] == "structure_search"
    assert search["status"] == "experimental" and search["part_of_stage_methodology"] is False
    assert search["methodology_status"] == search_spec.SEARCH_DEFAULTS_STATUS == "provisional"
    assert search["validation_status"] == "not_established"
    assert search["execution_gate"]["reported_state"] is None


def test_restated_structure_search_constants_equal_their_sources():
    (search,) = PAYLOAD["experimental_capabilities"]
    assert search["defaults_policy"] == {
        "policy_id": search_spec.SEARCH_DEFAULTS_POLICY_ID,
        "policy_version": search_spec.SEARCH_DEFAULTS_POLICY_VERSION,
        "status": search_spec.SEARCH_DEFAULTS_STATUS,
    }
    assert search["admission"] == search_admission.admission_scope_record()


def test_execution_gate_is_the_routers_gate_and_defaults_to_disabled(monkeypatch):
    from compute_api import router

    assert STRUCTURE_SEARCH_EXECUTION_SETTING == router.SEARCH_EXECUTION_ENV
    monkeypatch.delenv(router.SEARCH_EXECUTION_ENV, raising=False)
    with pytest.raises(router._ApiError):
        router._require_search_execution()
    monkeypatch.setenv(router.SEARCH_EXECUTION_ENV, STRUCTURE_SEARCH_EXECUTION_ENABLED_VALUE)
    router._require_search_execution()


def test_described_structure_search_operations_are_the_routers_routes():
    from compute_api import router

    api = router.build_machine_api_router(planner=lambda **_: None)
    routes = {
        f"{method} {route.path}"
        for route in api.routes
        if "structure-search" in route.path
        for method in route.methods
    }
    (search,) = PAYLOAD["experimental_capabilities"]
    assert {entry["route"] for entry in search["machine_api_operations"]} == routes


# ------------------------------------------------------------ validation and compatibility


def test_the_record_never_asserts_validation():
    assert PAYLOAD["validation"]["status"] == "not_recorded"
    assert "does not assert" in PAYLOAD["validation"]["statement"]


def test_producer_still_needs_only_the_standard_library():
    completed = subprocess.run(
        [sys.executable, "-E", "-S", "-B", "-m", "backend.calculations.capabilities"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert set(payload) >= {"release_identity", "desired_outputs", "new_calculation_admission", "protocols",
                            "experimental_capabilities", "validation"}


def test_existing_bmd_check_consumer_still_accepts_the_payload():
    compute = pytest.importorskip("bmd_agent.resources.compute")
    capabilities = compute.parse_capability_payload(json.dumps(build_capability_payload()))
    assert len(capabilities.capabilities) == len(PAYLOAD["capabilities"])
