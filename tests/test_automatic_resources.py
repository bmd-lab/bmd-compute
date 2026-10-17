"""Automatic execution resource selection (R1).

Planning only: nothing here contacts POWER, opens SSH, calls SLURM or writes
calculation files.
"""

from __future__ import annotations

import re
import warnings

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar
from starlette.requests import Request

import main
import test_automatic_soc as soc_cases
import test_input_reference_contract as reference_cases
from backend.calculations import automatic_resources as auto
from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import ExecutionResources, walltime_seconds
from backend.config import DEFAULT_ACCOUNT, DEFAULT_PARTITION, DEFAULT_QOS
from backend.remote import BatchSubmissionRequest
from backend.submission import (
    build_sbatch_script,
    build_submission_command,
    build_standalone_slurm_example,
    create_submission_spec,
)
from compute_api.plan_digest import PLAN_DIGEST_VERSION, compute_plan_digest, policy_versions
from compute_api.projection import plan_stage_previews


SI_POSCAR = reference_cases.SI_POSCAR
NIO_POSCAR = reference_cases.NIO_POSCAR
BI2SE3_POSCAR = soc_cases.BI2SE3_POSCAR
_SI_CONVENTIONAL = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
SI8_POSCAR = Poscar(_SI_CONVENTIONAL).get_str()
SI64_POSCAR = Poscar(_SI_CONVENTIONAL * (2, 2, 2)).get_str()


def _plan(structure=SI_POSCAR, desired_output="energy_only", custom=None, resources=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main.plan_calculation_request(
            structure_text=structure,
            fmt="poscar",
            desired_output=None if custom is not None else desired_output,
            custom_workflow=custom,
            resources=resources,
        )


def _selection(plan):
    return plan.submission_spec["flow_spec"]["resource_selection"]


def _estimate(plan):
    return _selection(plan)["walltime_estimate"]


def _hours(plan) -> float:
    return walltime_seconds(plan.execution_resources.walltime) / 3600


def _stage(stage_type, theory, modifiers=()):
    return {"stage_type": stage_type, "theory": theory, "modifiers": list(modifiers), "options": {}}


def _request(path):
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def _build(structure=SI_POSCAR, workflow="energy_only", cpus="", memory_gb="", walltime=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.build_workflow(
            _request("/build-workflow"),
            structure=structure,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            cpus=cpus,
            memory_gb=memory_gb,
            walltime=walltime,
            queue=None,
            workflow_spec_json=None,
            workflow=workflow,
            method=None,
        )
    assert response.status_code == 200, response.context.get("calculation_error")
    return response.context


@pytest.fixture(scope="module")
def plans():
    return {
        "si2_static": _plan(),
        "si64_static": _plan(SI64_POSCAR),
        "si2_relax": _plan(desired_output="relaxed_structure"),
        "nio_static": _plan(NIO_POSCAR),
        "si2_hse_dos": _plan(desired_output="electronic_dos"),
        "si8_hse_band": _plan(SI8_POSCAR, desired_output="electronic_band_structure"),
        "si2_phonons": _plan(desired_output="phonons"),
    }


# ------------------------------------------------------------ descriptors


def test_descriptors_reproduce_recorded_outcar_quantities(plans):
    # Telemetry 2026-10-16: Si2 7x7x7 Gamma mesh -> 20 irreducible k-points,
    # 32^3 FFT at ENCUT 620; the Si64 phonon supercell -> 6 k-points,
    # 168 bands, 96^3 FFT at ENCUT 700.
    si2 = _estimate(plans["si2_static"])["stages"][0]["descriptors"]
    assert si2["kpoints"] == 20
    assert si2["fft_grid"] == [32, 32, 32]
    assert si2["electrons"] == 8
    phonon = _estimate(plans["si2_phonons"])["stages"][1]["descriptors"]
    assert (phonon["kpoints"], phonon["bands"], phonon["fft_grid"]) == (6, 168, [96, 96, 96])
    assert phonon["calculations"] == 1


def test_default_band_rule_matches_vasp_for_recorded_systems():
    # Cu20Mn4 spin-polarised: NELECT 272, NMAG 32 -> 179 -> 180 with 3 band groups.
    assert auto.round_to_band_groups(
        auto.default_bands(272, 24, spin_polarized=True, soc=False, magnetisation=32.0), cpus=24, ncore=8
    ) == 180
    # Fe12O18 with SOC: NELECT 204, NMAG 70.8 -> 2 x 157 -> 315.
    assert auto.round_to_band_groups(
        auto.default_bands(204, 30, spin_polarized=True, soc=True, magnetisation=70.8), cpus=24, ncore=8
    ) == 315


def test_fft_grid_matches_recorded_grids():
    assert auto.fft_grid((3.8673,) * 3, 620.0) == (32, 32, 32)
    assert auto.fft_grid((5.43,) * 3, 620.0) == (48, 48, 48)


def test_an_unsizable_stage_gets_the_unknown_fallback():
    class Broken:
        incar = {}
        poscar = None

    stage = main.WorkflowSpec.from_dict({"stages": [_stage("static", "pbe")]}).stages[0]
    record = auto.estimate_stage({"index": 1, "stage_spec": stage, "input_set": Broken()}, cpus=24, potcar_functional="PBE_64")
    assert record["class"] == auto.STAGE_UNKNOWN_FALLBACK
    assert record["seconds"] == 72 * 3600
    estimate = auto.estimate_walltime([{"index": 1, "stage_spec": stage, "input_set": Broken()}])
    assert estimate["basis"] == auto.BASIS_FALLBACK
    assert estimate["walltime"] == "73:00:00"


# ------------------------------------------------------------ recommendations


def test_cpus_and_memory_are_the_conservative_defaults(plans):
    for plan in plans.values():
        assert plan.execution_resources.cpus == 24
        assert plan.execution_resources.memory_gb == 128
        assert _selection(plan)["fields"]["cpus"]["source"] == "automatic"
        assert _selection(plan)["fields"]["memory_gb"]["source"] == "automatic"


def test_recommendations_vary_with_computational_workload(plans):
    assert _hours(plans["si2_static"]) < _hours(plans["si64_static"])
    assert _hours(plans["si2_static"]) < _hours(plans["si2_relax"]) < _hours(plans["si2_phonons"])
    # Same atom count, different cost: the descriptor, not the atom count, decides.
    si2 = _estimate(plans["si2_static"])["stages"][0]
    nio = _estimate(plans["nio_static"])["stages"][0]
    assert si2["descriptors"]["atoms"] == nio["descriptors"]["atoms"]
    assert si2["descriptors"]["descriptor"] != nio["descriptors"]["descriptor"]
    assert si2["seconds"] != nio["seconds"]


def test_small_pbe_calculations_receive_shorter_walltimes_than_hybrid_workflows(plans):
    for pbe in ("si2_static", "si64_static", "si2_relax", "nio_static", "si2_phonons"):
        for hybrid in ("si2_hse_dos", "si8_hse_band"):
            assert _hours(plans[pbe]) < _hours(plans[hybrid]), (pbe, hybrid)
    assert _hours(plans["si2_hse_dos"]) < _hours(plans["si8_hse_band"])


def test_calibrated_pbe_plans_are_empirical_estimates(plans):
    for name in ("si2_static", "si64_static", "si2_relax", "nio_static", "si2_phonons"):
        estimate = _estimate(plans[name])
        assert estimate["basis"] == auto.BASIS_EMPIRICAL, name
        assert {stage["class"] for stage in estimate["stages"]} == {auto.STAGE_CALIBRATED}
    assert plans["si2_static"].execution_resources.walltime == "01:00:00"


def test_magnetic_dft_u_plan_is_sized_as_spin_polarised(plans):
    descriptors = _estimate(plans["nio_static"])["stages"][0]["descriptors"]
    assert descriptors["dft_u"] is True
    assert descriptors["spin_polarized"] is True


def test_hybrid_stages_use_identified_fallbacks_not_the_pbe_model(plans):
    stages = _estimate(plans["si2_hse_dos"])["stages"]
    assert [stage["class"] for stage in stages] == [
        auto.STAGE_CALIBRATED,
        auto.STAGE_HYBRID_FALLBACK,
        auto.STAGE_HYBRID_FALLBACK,
    ]
    assert stages[1]["seconds"] == 36 * 3600
    assert "HSE06" in stages[1]["reason"]
    assert _estimate(plans["si2_hse_dos"])["basis"] == auto.BASIS_FALLBACK
    assert "Conservative fallback" in _estimate(plans["si2_hse_dos"])["explanation"]


def test_hybrid_with_soc_uses_the_longest_fallback():
    seconds, reason = auto.hybrid_fallback_seconds({"soc": True, "descriptor": 1e8})
    assert seconds == 21 * 24 * 3600
    assert "SOC" in reason


def test_out_of_domain_pbe_uses_a_generous_fallback():
    # Bi2Se3 with automatic SOC: cost descriptor above the calibrated range.
    plan = _plan(BI2SE3_POSCAR)
    stage = _estimate(plan)["stages"][0]
    assert stage["class"] == auto.STAGE_EXTRAPOLATED
    assert "calibrated" in stage["reason"]
    assert _hours(plan) >= 72
    # PBE DOS is not a calibrated stage type.
    custom = _plan(custom={"stages": [_stage("static", "pbe"), _stage("dos", "pbe")]})
    assert [stage["class"] for stage in _estimate(custom)["stages"]] == [
        auto.STAGE_CALIBRATED,
        auto.STAGE_EXTRAPOLATED,
    ]
    assert _estimate(custom)["minimum_floor_applied"] is True
    assert custom.execution_resources.walltime == "72:00:00"


def test_walltime_rounding_floor_and_qos_cap():
    assert auto.round_walltime_seconds(10) == (3600, False)
    assert auto.round_walltime_seconds(3601) == (3600 + 900, False)
    assert auto.round_walltime_seconds(30 * 3600 + 1) == (31 * 3600, False)
    assert auto.round_walltime_seconds(200 * 24 * 3600) == (100 * 24 * 3600, True)


def test_long_walltimes_are_requested_without_a_72_hour_ceiling():
    plan = _plan(SI64_POSCAR, desired_output="relaxed_structure")
    assert _hours(plan) > 72
    assert plan.submission_spec["cluster"]["qos"] == DEFAULT_QOS
    assert f"#SBATCH --time={plan.execution_resources.walltime}" in build_sbatch_script(plan.submission_spec)


# ------------------------------------------------------------ manual overrides


def test_explicit_values_are_kept_exactly_and_recorded_as_manual():
    plan = _plan(resources={"cpus": 48, "memory_gb": 256, "walltime": "05:30:00"})
    assert (plan.execution_resources.cpus, plan.execution_resources.memory_gb, plan.execution_resources.walltime) == (
        48,
        256,
        "05:30:00",
    )
    fields = _selection(plan)["fields"]
    assert {name: fields[name]["source"] for name in ("cpus", "memory_gb", "walltime")} == {
        "cpus": "manual",
        "memory_gb": "manual",
        "walltime": "manual",
    }
    # The automatic recommendation is still recorded for comparison.
    assert fields["walltime"]["recommended"] == "01:00:00"


def test_each_resource_can_be_overridden_individually():
    plan = _plan(resources={"walltime": "200:00:00"})
    assert plan.execution_resources.walltime == "200:00:00"
    assert plan.execution_resources.cpus == 24
    sources = {name: field["source"] for name, field in _selection(plan)["fields"].items()}
    assert sources == {"cpus": "automatic", "memory_gb": "automatic", "walltime": "manual", "queue": "automatic"}


def test_existing_explicit_requests_behave_unchanged():
    # Callers that pass ExecutionResources (every field explicit) keep them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _summary, _calc, _inputs, spec = main.build_submission_state(
            structure_text=SI_POSCAR,
            fmt="poscar",
            workflow_spec=main.desired_output_workflow_spec("energy_only"),
            execution_resources=ExecutionResources(cpus=72, memory_gb=192, walltime="12:00:00"),
        )
    assert spec["resources"] == {"nodes": 1, "ntasks": 72, "mem_gb": 192, "walltime": "12:00:00"}


@pytest.mark.parametrize("walltime", ["72h", "1-00:00:00", "10:60:00", "2400:00:01", "00:00:00"])
def test_invalid_manual_walltimes_are_rejected_with_guidance(walltime):
    with pytest.raises(CalculationValidationError):
        _plan(resources={"walltime": walltime})


def test_browser_automatic_to_manual_and_back():
    automatic = _build()
    assert automatic["selected_resources"]["sources"]["walltime"] == "automatic"
    assert automatic["selected_resources"]["requested"]["walltime"] == ""
    assert automatic["submission_spec"]["resources"]["walltime"] == "01:00:00"

    manual = _build(walltime="10:00:00", cpus="48")
    assert manual["selected_resources"]["sources"]["walltime"] == "manual"
    assert manual["selected_resources"]["sources"]["memory_gb"] == "automatic"
    assert manual["submission_spec"]["resources"]["walltime"] == "10:00:00"
    assert manual["submission_spec"]["resources"]["ntasks"] == 48
    assert manual["selected_resources"]["recommended_walltime"] == "01:00:00"

    back = _build(walltime="", cpus="auto")
    assert back["selected_resources"]["sources"]["walltime"] == "automatic"
    assert back["submission_spec"]["resources"]["walltime"] == "01:00:00"
    assert back["submission_spec"]["resources"]["ntasks"] == 24


def test_browser_rendering_shows_automatic_choices_and_submitted_resources():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.build_workflow(
            _request("/build-workflow"),
            structure=SI_POSCAR,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            cpus="",
            memory_gb="",
            walltime="",
            queue=None,
            workflow_spec_json=None,
            workflow="electronic_dos",
            method=None,
        )
    html = response.body.decode()
    # The Hardware Specification shows the three resolved values; the
    # estimator basis stays in the plan record, not on the page.
    assert re.search(r'data-hardware-value="cpus">24<', html)
    assert re.search(r'data-hardware-value="memory_gb">128 GB<', html)
    assert re.search(r'data-hardware-value="walltime" title="73:00:00">3 days 1 hour<', html)
    assert response.context["submission_spec"]["flow_spec"]["resource_selection"]["walltime_estimate"]["basis"] == "conservative_fallback"
    assert response.context["submission_spec"]["cluster"]["qos"] == "owner100d"
    # Prepare and Submit carry the requested (Automatic) values and the planned resources.
    assert html.count('name="planned_resources" value="cpus=24:automatic;memory_gb=128:automatic;walltime=73:00:00:automatic;') == 2


# ------------------------------------------------------------ consistency


def test_browser_and_machine_api_resolve_the_same_resources():
    for workflow in ("energy_only", "phonons", "electronic_band_structure"):
        ui = _build(workflow=workflow)
        api = _plan(desired_output=workflow)
        assert ui["submission_spec"]["resources"] == api.submission_spec["resources"]
        assert ui["submission_spec"]["cluster"]["qos"] == api.submission_spec["cluster"]["qos"]
        assert ui["submission_spec"]["flow_spec"]["resource_selection"] == _selection(api)


def test_plan_digest_is_deterministic_and_covers_the_resource_policy():
    first = _plan(desired_output="relaxed_structure")
    second = _plan(desired_output="relaxed_structure")

    def digest(plan):
        return compute_plan_digest(
            structure=plan.structure,
            submission_spec=plan.submission_spec,
            stage_previews=plan_stage_previews(plan),
        )

    assert digest(first) == digest(second)
    assert PLAN_DIGEST_VERSION == 2
    assert policy_versions()["automatic_resources"] == {"id": auto.POLICY_ID, "version": auto.POLICY_VERSION}
    # Choosing the recommended value manually is a different plan (manual source).
    manual = _plan(desired_output="relaxed_structure", resources={"walltime": first.execution_resources.walltime})
    assert manual.execution_resources == first.execution_resources
    assert digest(manual) != digest(first)


def test_prepare_refuses_resources_that_changed_since_planning(monkeypatch):
    built = _build()
    identity = built["submission_spec"]["submission"]
    monkeypatch.setattr(main, "prepare_remote_submission", lambda *a, **k: pytest.fail("prepared"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.prepare_remote(
            _request("/prepare-remote"),
            structure=SI_POSCAR,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            cpus="",
            memory_gb="",
            walltime="",
            queue=None,
            created_at=None,
            submission_attempt_id=identity["attempt_id"],
            submission_identity_token=identity["identity_token"],
            workflow_spec_json=None,
            workflow="energy_only",
            method=None,
            planned_resources=built["selected_resources"]["planned_token"].replace("01:00:00", "02:00:00"),
        )
    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == "planned_resources_changed"


def test_prepare_uses_the_resources_approved_at_planning(monkeypatch):
    built = _build(workflow="phonons")
    identity = built["submission_spec"]["submission"]
    captured = {}

    def fake_prepare(spec, **kwargs):
        captured["spec"] = spec
        return main.remembered_successful_preparation(spec)

    monkeypatch.setattr(main, "prepare_remote_submission", fake_prepare)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.prepare_remote(
            _request("/prepare-remote"),
            structure=SI_POSCAR,
            fmt="poscar",
            purpose=None,
            theory=None,
            modifiers=None,
            cpus="",
            memory_gb="",
            walltime="",
            queue=None,
            created_at=None,
            submission_attempt_id=identity["attempt_id"],
            submission_identity_token=identity["identity_token"],
            workflow_spec_json=None,
            workflow="phonons",
            method=None,
            planned_resources=built["selected_resources"]["planned_token"],
        )
    assert response.status_code == 200
    assert captured["spec"]["resources"] == built["submission_spec"]["resources"]
    assert captured["spec"]["submission"]["attempt_fingerprint"] == built["submission_spec"]["submission"]["attempt_fingerprint"]


# ------------------------------------------------------------ SLURM


def test_generated_slurm_submission_includes_the_qos(plans):
    spec = plans["si2_phonons"].submission_spec
    assert spec["cluster"]["qos"] == "owner100d"
    assert spec["cluster"]["account"] == DEFAULT_ACCOUNT
    assert spec["cluster"]["partition"] == DEFAULT_PARTITION
    script = build_sbatch_script(spec)
    assert "#SBATCH --qos=owner100d\n" in script
    assert f"#SBATCH --time={spec['resources']['walltime']}\n" in script
    command = build_submission_command(spec)
    assert f"sbatch -p {DEFAULT_PARTITION} -A {DEFAULT_ACCOUNT} --qos=owner100d -N 1 -n 24 --mem=128G -t {spec['resources']['walltime']} --parsable" in command
    assert "#SBATCH --qos=owner100d" in build_standalone_slurm_example(spec)["text"]


def test_batch_submission_request_carries_the_qos(plans):
    from backend.paramiko_remote import _batch_request_from_submission_spec

    request = _batch_request_from_submission_spec(plans["si2_static"].submission_spec)
    assert request.qos == "owner100d"
    assert BatchSubmissionRequest(
        script_path="/x", partition="p", account="a", nodes=1, ntasks=24, mem_gb=128, walltime="01:00:00"
    ).qos is None


def test_qos_is_only_used_with_the_authorised_account():
    flow_spec = {
        "workflow_spec": main.desired_output_workflow_spec("energy_only").to_dict(),
        "potcar_functional": "PBE_64",
    }
    structure = Structure.from_str(SI_POSCAR, fmt="poscar")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = create_submission_spec(flow_spec, structure=structure, account="another-account", timestamp="20261016-120000")
        default = create_submission_spec(flow_spec, structure=structure, timestamp="20261016-120000")
    assert other["cluster"]["qos"] is None
    assert "--qos" not in build_sbatch_script(other)
    assert default["cluster"]["qos"] == DEFAULT_QOS


def test_no_submission_restrictions_are_introduced():
    # Any walltime up to the QOS limit, and every allowed CPU and memory value,
    # remain valid explicit requests.
    plan = _plan(resources={"cpus": 192, "memory_gb": 512, "walltime": "2400:00:00"})
    assert plan.execution_resources.walltime == "2400:00:00"
    assert "#SBATCH --time=2400:00:00" in build_sbatch_script(plan.submission_spec)
