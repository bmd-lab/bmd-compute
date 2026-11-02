"""BMD Adaptive Geometry Relaxation v1 (``bmd_compute.geometry_convergence/1``).

Dynamic-execution tests use small fake VASP jobs (pure Python, no VASP) run by
BMD's own stage-directory runner, with controlled evidence. Plans use the
SYNTHETIC POTCAR ENMAX metadata of ``conftest`` (E0 = 550 eV, so E1 = 700 eV
and Ep = 850 eV); no value here is a production ENMAX or a measurement.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from jobflow import Flow, Maker, job
from jobflow.core.store import JobStore
from maggma.stores import MemoryStore
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Kpoints

import main
import test_geometry_relaxation_protocol as protocol_cases
from backend.calculations import geometry_convergence as gc
from backend.calculations import geometry_convergence_jobs as gcj
from backend.calculations import geometry_relaxation as gr
from backend.calculations.custodian_policy import LARGE_SIGMA_HANDLER
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    canonical_custom_workflow_spec,
    validate_user_workflow_spec,
    validate_workflow_spec,
)
from backend.execution import _flow_stage_directories, _run_locally_with_stage_directories
from backend.parser import parse_structure
from backend.results import results_location_from_submission_spec
from backend.workflows import build_atomate2_flow_for_workflow_spec
from compute_api.plan_digest import compute_plan_digest
from test_machine_api import client, tokens  # noqa: F401 - pytest fixtures

SI_POSCAR = protocol_cases.SI_POSCAR
NIO_POSCAR = protocol_cases.NIO_POSCAR
_plan = protocol_cases._plan
_previews = protocol_cases._previews
_stage = protocol_cases._stage


# ------------------------------------------------------------------ policy


@pytest.mark.parametrize(
    ("initial", "refinement", "probe"),
    [(300, 450, 700), (350, 500, 700), (400, 550, 700), (550, 700, 850), (650, 800, 950)],
)
def test_cutoffs_follow_from_the_initial_cutoff(initial, refinement, probe):
    record = gc.convergence_record(initial)
    assert (record["initial_encut"], record["refinement_encut"], record["probe_encut"]) == (initial, refinement, probe)
    assert record["max_refinements"] == 1
    assert record["probe_encut"] - record["refinement_encut"] >= gc.REFINEMENT_INCREMENT_EV
    assert record["acceptance"] == {
        "max_force_ev_a": 0.001,
        "max_abs_pressure_kbar": 1.0,
        "max_abs_stress_component_kbar": 2.0,
        "max_magnetisation_change_mub": 0.1,
    }


def _evidence(**values):
    base = {
        "converged_electronic": True,
        "converged_ionic": True,
        "max_force_ev_a": 0.0,
        "pressure_kbar": 0.2,
        "max_stress_kbar": 0.3,
        "spin_polarized": False,
        "magnetisation": None,
    }
    base.update(values)
    return base


def test_si_al_like_evidence_is_accepted_without_refinement():
    evaluation = gc.evaluate_probe(relaxation=_evidence(), probe=_evidence(pressure_kbar=-0.4, max_stress_kbar=0.4))
    assert evaluation["category"] == "accepted" and evaluation["reasons"] == []
    decision = gc.decide(gc.convergence_record(350), refinements_done=0, evaluation=evaluation)
    assert (decision["action"], decision["outcome"]) == ("accept", gc.OUTCOME_ACCEPTED_WITHOUT_REFINEMENT)
    recorded = {check["name"]: check for check in evaluation["checks"]}
    assert recorded["probe_pressure_kbar"] == {"name": "probe_pressure_kbar", "value": -0.4, "tolerance": 1.0, "passed": True}


@pytest.mark.parametrize(
    ("probe", "reason"),
    [
        (_evidence(max_force_ev_a=0.0012), "maximum residual force"),
        (_evidence(pressure_kbar=1.2), "hydrostatic pressure"),
        (_evidence(pressure_kbar=-1.2), "hydrostatic pressure"),
        (_evidence(max_stress_kbar=2.5), "stress component"),
    ],
)
def test_each_tolerance_rejects_and_the_single_refinement_is_requested(probe, reason):
    evaluation = gc.evaluate_probe(relaxation=_evidence(), probe=probe)
    assert evaluation["category"] == "rejected" and reason in evaluation["reasons"][0]
    record = gc.convergence_record(350)
    assert gc.decide(record, refinements_done=0, evaluation=evaluation)["action"] == "refine"
    final = gc.decide(record, refinements_done=1, evaluation=evaluation)
    assert (final["action"], final["status"], final["outcome"]) == ("stop", gc.STATUS_UNRESOLVED, gc.OUTCOME_UNRESOLVED)


def test_values_exactly_at_the_tolerances_are_accepted():
    probe = _evidence(max_force_ev_a=0.001, pressure_kbar=1.0, max_stress_kbar=2.0)
    assert gc.evaluate_probe(relaxation=_evidence(), probe=probe)["category"] == "accepted"


@pytest.mark.parametrize(
    ("relaxation", "probe"),
    [
        (_evidence(converged_ionic=False), _evidence()),
        (_evidence(converged_electronic=False), _evidence()),
        (_evidence(), _evidence(converged_electronic=False)),
    ],
)
def test_unconverged_vasp_evidence_is_an_execution_failure_never_a_refinement(relaxation, probe):
    evaluation = gc.evaluate_probe(relaxation=relaxation, probe=probe)
    assert evaluation["category"] == "vasp_not_converged"
    decision = gc.decide(gc.convergence_record(350), refinements_done=0, evaluation=evaluation)
    assert (decision["action"], decision["status"]) == ("stop", gc.STATUS_EXECUTION_FAILURE)


@pytest.mark.parametrize(
    "probe",
    [
        None,
        {"converged_electronic": True},
        _evidence(pressure_kbar=None),
        _evidence(max_stress_kbar=float("nan")),
        _evidence(max_force_ev_a="0.0"),
        _evidence(converged_electronic="yes"),
    ],
)
def test_missing_or_malformed_evidence_is_invalid_evidence_never_a_refinement(probe):
    evaluation = gc.evaluate_probe(relaxation=_evidence(), probe=probe)
    assert evaluation["category"] == "invalid_evidence"
    decision = gc.decide(gc.convergence_record(350), refinements_done=0, evaluation=evaluation)
    assert (decision["action"], decision["status"], decision["outcome"]) == ("stop", gc.STATUS_INVALID_EVIDENCE, gc.OUTCOME_INVALID_EVIDENCE)


def _magnetic(total, sites=()):
    return {"spin_polarized": True, "magnetisation": {"total": total, "sites": list(sites)}}


def test_a_magnetic_state_change_is_unresolved_and_never_refined():
    relaxation = _evidence(**_magnetic(2.0, [1.7, 0.3]))
    for probe in (_evidence(**_magnetic(1.6, [1.4, 0.2])), _evidence(**_magnetic(2.0, [1.8, 0.2])), _evidence(**_magnetic(0.0, [0.0, 0.0]))):
        evaluation = gc.evaluate_probe(relaxation=relaxation, probe=probe)
        assert evaluation["category"] == "magnetic_state_changed"
        decision = gc.decide(gc.convergence_record(550), refinements_done=0, evaluation=evaluation)
        assert (decision["action"], decision["outcome"]) == ("stop", gc.OUTCOME_UNRESOLVED_MAGNETIC)
    small = gc.evaluate_probe(relaxation=relaxation, probe=_evidence(**_magnetic(2.05, [1.75, 0.3])))
    assert small["category"] == "accepted"
    moment = {check["name"]: check for check in small["checks"]}["probe_magnetisation"]
    assert moment["applicable"] and moment["magnetic_state_verified"] and moment["total_change_mub"] == pytest.approx(0.05)
    assert moment["max_site_change_mub"] == pytest.approx(0.05)


def test_a_refined_relaxation_that_changes_magnetic_state_is_not_accepted():
    initial = _evidence(**_magnetic(2.0, [2.0]))
    refined = _evidence(**_magnetic(1.0, [1.0]))
    evaluation = gc.evaluate_probe(relaxation=refined, probe=_evidence(**_magnetic(1.0, [1.0])), reference_relaxation=initial)
    assert evaluation["category"] == "magnetic_state_changed"
    assert "refinement_magnetisation" in evaluation["reasons"][0]


@pytest.mark.parametrize(
    ("relaxation_sites", "probe_sites"),
    [((), ()), ((1.7, 0.3), ()), ((), (1.7, 0.3)), ((1.7, 0.3), (1.7,)), ((1.7, float("nan")), (1.7, 0.3))],
)
def test_total_magnetisation_alone_is_never_accepted_as_magnetic_evidence(relaxation_sites, probe_sites):
    # Codex P1: a matching total (antiferromagnets have ~0) does not show that
    # the configuration is preserved; without valid site moments the evidence
    # is invalid, never accepted.
    evaluation = gc.evaluate_probe(
        relaxation=_evidence(**_magnetic(2.0, relaxation_sites)), probe=_evidence(**_magnetic(2.0, probe_sites))
    )
    assert evaluation["category"] == "invalid_evidence"
    check = {c["name"]: c for c in evaluation["checks"]}["probe_magnetisation"]
    assert check["magnetic_state_verified"] is False and check["passed"] is False
    assert gc.decide(gc.convergence_record(550), refinements_done=0, evaluation=evaluation)["status"] == gc.STATUS_INVALID_EVIDENCE
    missing = gc.evaluate_probe(relaxation=_evidence(**_magnetic(2.0, [2.0])), probe=_evidence(spin_polarized=True, magnetisation=None))
    assert missing["category"] == "invalid_evidence"


def test_an_antiferromagnetic_sign_flip_is_a_magnetic_state_change():
    afm = _evidence(**_magnetic(0.0, [1.7, -1.7]))
    flipped = _evidence(**_magnetic(0.0, [-1.7, 1.7]))
    assert gc.evaluate_probe(relaxation=afm, probe=flipped)["category"] == "magnetic_state_changed"


def test_the_spin_polarised_final_pass_and_its_adaptive_stages_write_site_moments():
    # LORBIT is output only: VASP then writes the OUTCAR "magnetization (x)"
    # table pymatgen's Outcar.magnetization reads. Non-magnetic plans unchanged.
    nio = _plan(NIO_POSCAR, desired_output="relaxed_energy")
    first, final, static = (preview["input_set"].incar for preview in _previews(nio))
    assert final["LORBIT"] == 11 and "LORBIT" not in first
    extra = gc.worst_case_stage_previews(_previews(nio))
    assert all(p["input_set"].incar["LORBIT"] == 11 for p in extra)
    si = _previews(_plan(desired_output="relaxed_structure"))
    assert "LORBIT" not in si[1]["input_set"].incar


def test_tampered_or_misplaced_records_are_refused():
    plan = _plan(desired_output="relaxed_structure")
    stages = [stage.to_dict() for stage in plan.workflow_spec.stages]
    assert stages[1]["options"][gc.OPTION_KEY] == gc.convergence_record(550)
    tampered = json.loads(json.dumps(stages))
    tampered[1]["options"][gc.OPTION_KEY]["probe_encut"] = 600
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(WorkflowSpec.from_dict({"stages": tampered}))
    assert excinfo.value.diagnostic["code"] == "geometry_convergence_record_invalid"
    misplaced = json.loads(json.dumps(stages))
    misplaced[0]["options"][gc.OPTION_KEY] = gc.convergence_record(550)
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(WorkflowSpec.from_dict({"stages": misplaced}))
    assert excinfo.value.diagnostic["code"] == "geometry_convergence_placement_invalid"


# ---------------------------------------------- planning, digest, resources


def _ctrl_stage_inputs(plan):
    from compute_api import plan_digest

    return plan_digest.stage_input_material(_previews(plan))


def test_every_managed_relaxation_freezes_the_policy_on_its_final_pass():
    for output in protocol_cases.MANAGED_RELAXING_OUTPUTS:
        plan = _plan(desired_output=output)
        stages = plan.workflow_spec.stages
        assert gc.stage_convergence_record(stages[0]) is None
        assert gc.stage_convergence_record(stages[1]) == gc.convergence_record(550), output
        for stage in stages[2:]:
            expected = gc.convergence_record(550) if stage.stage_type is StageType.PHONON_FORCES else None
            assert gc.stage_convergence_record(stage) == expected, output


def test_plan_digest_covers_the_policy_and_its_possible_stages():
    plan = _plan(desired_output="relaxed_structure")
    material = _ctrl_stage_inputs(plan)
    convergence = material[1]["geometry_relaxation"]["geometry_convergence"]
    assert convergence["record"] == gc.convergence_record(550)
    assert convergence["possible_stages"]["refinement"]["incar_overrides"] == {"ENCUT": 700}
    assert convergence["possible_stages"]["final_probe"]["incar_overrides"] == {"ENCUT": 850, "NSW": 0, "IBRION": -1}
    digest = compute_plan_digest(structure=plan.structure, submission_spec=plan.submission_spec, stage_previews=_previews(plan))
    # Without the frozen record (a plan recorded before the adaptive controller)
    # the plan is a different plan.
    spec = json.loads(json.dumps(plan.submission_spec))
    spec["flow_spec"]["workflow_spec"]["stages"][1]["options"].pop(gc.OPTION_KEY)
    stripped = WorkflowSpec.from_dict(spec["flow_spec"]["workflow_spec"])
    from backend.generated_inputs import generated_input_stage_previews

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        previews = generated_input_stage_previews(plan.structure, stripped, resources=plan.execution_resources)
    assert digest != compute_plan_digest(structure=plan.structure, submission_spec=spec, stage_previews=previews)


def test_worst_case_walltime_and_memory_cover_every_possible_stage():
    plan = _plan(desired_output="phonons")
    selection = plan.submission_spec["flow_spec"]["resource_selection"]
    estimate = selection["walltime_estimate"]
    roles = [stage.get("geometry_convergence_role") for stage in estimate["stages"]]
    assert roles == [None, None, None, gc.ROLE_INITIAL_PROBE, gc.ROLE_REFINEMENT, gc.ROLE_FINAL_PROBE]
    by_role = {stage.get("geometry_convergence_role"): stage for stage in estimate["stages"]}
    assert by_role[gc.ROLE_REFINEMENT]["descriptors"]["encut_ev"] == 700
    assert by_role[gc.ROLE_FINAL_PROBE]["descriptors"]["encut_ev"] == 850
    assert by_role[gc.ROLE_INITIAL_PROBE]["stage_type"] == "static"
    # The request is the sum over the whole worst-case path.
    from backend.calculations.automatic_resources import estimate_memory, estimate_walltime

    with_path = estimate_walltime(_worst_case_previews(plan))
    without_path = estimate_walltime(_previews(plan))
    assert with_path["walltime_seconds"] > without_path["walltime_seconds"]
    assert plan.execution_resources.walltime == with_path["walltime"]
    assert plan.execution_resources.memory_gb == estimate_memory(with_path["stages"])["memory_gb"]
    assert "may finish earlier" in estimate["explanation"]


def _worst_case_previews(plan):
    previews = _previews(plan)
    return (*previews, *gc.worst_case_stage_previews(previews))


def test_worst_case_stage_inputs_are_the_final_pass_with_only_the_cutoff_changed():
    plan = _plan(desired_output="relaxed_structure")
    previews = _previews(plan)
    final = dict(previews[1]["input_set"].incar)
    extra = gc.worst_case_stage_previews(previews)
    probe, refinement, final_probe = (dict(p["input_set"].incar) for p in extra)
    assert {k for k in set(probe) | set(final) if probe.get(k) != final.get(k)} == {"ENCUT", "NSW", "IBRION"}
    assert {k for k in set(refinement) | set(final) if refinement.get(k) != final.get(k)} == {"ENCUT"}
    assert probe == final_probe
    assert (probe["ISMEAR"], probe["SIGMA"], probe["ENCUT"]) == (0, 0.2, 850)
    assert all(str(p["input_set"].kpoints) == str(previews[1]["input_set"].kpoints) for p in extra)


def test_downstream_cutoffs_are_plan_fixed():
    energy = _previews(_plan(desired_output="relaxed_energy"))
    assert energy[2]["input_set"].incar["ENCUT"] == 700  # max(620, refinement cutoff)
    phonons = _previews(_plan(desired_output="phonons"))
    assert phonons[2]["input_set"].incar["ENCUT"] == 850  # the probe cutoff
    dos = _previews(_plan(desired_output="electronic_dos"))
    assert (dos[2]["input_set"].incar["ENCUT"], dos[3]["input_set"].incar["ENCUT"]) == (700, 620)  # HSE06 DOS: own SCF
    custom = {
        "stages": [
            _stage("relax", options=gr.protocol_option(1)),
            _stage("relax", options=gr.protocol_option(2)),
            _stage("static"),
            _stage("dos"),
        ]
    }
    pbe = _previews(_plan(custom=custom))
    assert pbe[3]["input_set"].incar["ENCUT"] == pbe[2]["input_set"].incar["ENCUT"] == 700  # reads the static's CHGCAR


def test_browser_and_api_plan_the_same_policy_and_resources():
    from starlette.requests import Request

    request = Request({"type": "http", "method": "POST", "path": "/build-workflow", "headers": []})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.build_workflow(
            request, structure=SI_POSCAR, fmt="poscar", purpose=None, theory=None, modifiers=None,
            cpus=None, memory_gb=None, walltime=None, queue=None, workflow_spec_json=None,
            workflow="relaxed_energy", method=None,
        )
    assert response.status_code == 200
    api = _plan(desired_output="relaxed_energy")
    browser_spec = response.context["submission_spec"]
    assert browser_spec["flow_spec"]["workflow_spec"] == api.submission_spec["flow_spec"]["workflow_spec"]
    assert browser_spec["resources"] == api.submission_spec["resources"]


def test_custom_workflows_get_the_policy_and_cannot_supply_or_echo_it():
    plan = _plan(custom={"stages": [_stage("relax", options=gr.protocol_option(1)), _stage("relax", options=gr.protocol_option(2))]})
    assert gc.stage_convergence_record(plan.workflow_spec.stages[1]) == gc.convergence_record(550)
    stages = [stage.to_dict() for stage in plan.workflow_spec.stages]
    with pytest.raises(CalculationValidationError):
        validate_user_workflow_spec(WorkflowSpec.from_dict({"stages": stages}))
    canonical = canonical_custom_workflow_spec(WorkflowSpec.from_dict({"stages": stages}))
    assert all(gc.OPTION_KEY not in stage.options for stage in canonical.stages)
    # An unmarked Custom relaxation has no adaptive controller.
    single = _plan(custom={"stages": [_stage("relax")]})
    assert gc.stage_convergence_record(single.workflow_spec.stages[0]) is None


def test_recorded_workflows_without_the_policy_build_exactly_as_recorded():
    plan = _plan(desired_output="phonons")
    stages = [stage.to_dict() for stage in plan.workflow_spec.stages]
    for stage in stages:
        stage["options"].pop(gc.OPTION_KEY, None)
    recorded = validate_workflow_spec(WorkflowSpec.from_dict({"stages": stages, "recipe": "phonons"}))
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, recorded, resources={"ntasks": 24})
    assert [job.name for job in flow.jobs] == ["stage_01", "stage_02", "stage_03"]
    phonon_jobs = {job.name: job for job in flow.jobs[2].jobs}
    force = phonon_jobs["run_phonon_displacements"].function_kwargs["phonon_maker"].input_set_generator
    assert force.user_incar_settings["ENCUT"] == 550
    # A read-only tool keeps the recorded form.
    kept = gr.freeze_plane_wave_cutoffs(plan.structure, recorded, potcar_functional="PBE_64", only_missing=True)
    assert all(gc.stage_convergence_record(stage) is None for stage in kept.stages)


def test_the_controller_follows_the_final_pass_and_feeds_the_downstream_stages():
    plan = _plan(desired_output="phonons")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    assert [job.name for job in flow.jobs] == ["stage_01", "stage_02", gc.STAGE_DIRECTORY, "stage_03"]
    assert _flow_stage_directories(flow) == ("stage_01", "stage_02", gc.STAGE_DIRECTORY, "stage_03")
    controller = flow.jobs[2]
    assert controller.function_args[1].uuid == flow.jobs[1].uuid  # the final pass's directory
    assert controller.function_args[3] == gc.convergence_record(550)
    assert controller.function_kwargs["downstream_cutoffs"] == [{"stage_type": "phonon_forces", "theory": "pbe", "encut": 850}]
    referenced = {ref.uuid for phonon_job in flow.jobs[3].jobs for ref in phonon_job.input_references}
    assert controller.uuid in referenced and flow.jobs[1].uuid not in referenced


def test_probe_and_refinement_makers_keep_the_final_pass_settings_and_custodian_policy():
    plan = _plan(NIO_POSCAR, desired_output="relaxed_structure")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    final = flow.jobs[1].maker
    record = gc.convergence_record(gr.stage_cutoff_record(plan.workflow_spec.stages[1])["ENCUT"])
    probe = gcj.probe_maker(final, record, name="probe_initial")
    refinement = gcj.refinement_maker(final, record, name="refinement")
    base = final.input_set_generator.user_incar_settings
    assert {k: v for k, v in probe.input_set_generator.user_incar_settings.items() if base.get(k) != v} == {
        "ENCUT": record["probe_encut"], "NSW": 0, "IBRION": -1,
    }
    assert {k: v for k, v in refinement.input_set_generator.user_incar_settings.items() if base.get(k) != v} == {
        "ENCUT": record["refinement_encut"],
    }
    for maker in (probe, refinement):
        handlers = [f"{type(h).__module__}.{type(h).__name__}" for h in maker.run_vasp_kwargs["handlers"]]
        assert LARGE_SIGMA_HANDLER not in handlers
        assert maker.input_set_generator.user_incar_settings["LDAUU"] == base["LDAUU"]
    structure = parse_structure(NIO_POSCAR)
    incar = probe.input_set_generator.get_input_set(structure, potcar_spec=True).incar
    assert (incar["ISMEAR"], incar["SIGMA"], incar["ENCUT"], incar["NSW"]) == (0, 0.2, record["probe_encut"], 0)


def test_results_report_the_accepted_geometry_and_the_record():
    plan = _plan(desired_output="phonons")
    spec = dict(plan.submission_spec)
    spec["paths"] = {**spec["paths"], "run_dir": "/runs/x"}
    location = results_location_from_submission_spec(spec)
    assert location["output_dir"] == "/runs/x/geometry_convergence/accepted"
    assert location["optional_result_file_paths"]["geometry_convergence_record"] == "/runs/x/geometry_convergence/geometry_convergence.json"
    energy = dict(_plan(desired_output="relaxed_energy").submission_spec)
    energy["paths"] = {**energy["paths"], "run_dir": "/runs/y"}
    location = results_location_from_submission_spec(energy)
    assert location["output_dir"].endswith("/stage_03")
    assert location["optional_result_file_paths"]["geometry_relaxation_vasprun"] == "/runs/y/geometry_convergence/accepted/vasprun.xml"


def test_remote_parser_summarises_the_record(tmp_path):
    from backend.remote_result_parser import geometry_convergence_payload

    path = tmp_path / "geometry_convergence.json"
    path.write_text(json.dumps({"schema": gc.RECORD_SCHEMA, "schema_version": 1, "status": "in_progress", "pending_stage": {"operation": "refinement"}}))
    summary = geometry_convergence_payload(str(path))
    assert (summary["status"], summary["category"], summary["outcome"], summary["pending_operation"]) == (
        "interrupted", "interrupted_or_abandoned", "interrupted", "refinement"
    )
    assert any("stopped during refinement without an observed failure" in reason for reason in summary["reasons"])
    path.write_text("{")
    assert geometry_convergence_payload(str(path))["status"] == "unavailable"
    assert geometry_convergence_payload(None) is None


# ---------------------------------------------------- dynamic execution


@dataclass
class FakeVaspMaker(Maker):
    """Writes controlled evidence instead of running VASP."""

    name: str = "fake"
    scenario: dict = field(default_factory=dict)
    kpoints: Kpoints | None = None

    @job
    def make(self, structure):
        evidence = dict(self.scenario[self.name])
        if evidence.pop("raise", None) == "interrupt":
            raise KeyboardInterrupt
        lattice = evidence.pop("lattice_a", None)
        ignore_frozen = evidence.pop("ignore_frozen_kpoints", False)
        out = structure.copy()
        if lattice is not None:
            out = Structure(Lattice.cubic(lattice), out.species, out.frac_coords)
        cwd = Path.cwd()
        assert not (cwd / "evidence.json").exists(), "a stage reused a directory"
        (cwd / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        for name in ("CONTCAR", "OUTCAR", "vasprun.xml"):
            (cwd / name).write_text(f"{self.name} {name}", encoding="utf-8")
        # As the input set does: a given Kpoints is used verbatim, otherwise
        # an automatic mesh is generated from the (changed) cell.
        mesh = self.kpoints if self.kpoints is not None and not ignore_frozen else _automatic_mesh(out)
        (cwd / "KPOINTS").write_text(str(mesh), encoding="utf-8")
        return {"structure": out, "dir_name": f"fakehost:{cwd}"}


def _automatic_mesh(structure):
    n = max(1, round(24.0 / structure.lattice.a))
    return Kpoints.gamma_automatic((n, n, n))


@job(name="downstream")
def fake_downstream(structure, dir_name):
    (Path.cwd() / "DOWNSTREAM").write_text("ran", encoding="utf-8")
    return {"a": structure.lattice.a, "dir_name": dir_name}


def _read_fake_evidence(dir_name):
    path = gcj.local_directory(dir_name) / "evidence.json"
    if not path.exists():
        raise gcj.EvidenceError("missing", f"no evidence in {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def fake_vasp(monkeypatch):
    def fake_maker(maker, record, name, kpoints=None):
        return FakeVaspMaker(name=name, scenario=maker.scenario, kpoints=kpoints)

    monkeypatch.setattr(gcj, "probe_maker", fake_maker)
    monkeypatch.setattr(gcj, "refinement_maker", fake_maker)
    monkeypatch.setattr(gcj, "read_stage_evidence", _read_fake_evidence)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def _run_controller(scenario, record=None, tmp=None):
    record = record or gc.convergence_record(350)
    structure = Structure(Lattice.cubic(5.40), ["Si"], [[0, 0, 0]])
    relax = FakeVaspMaker(name="relax_02", scenario=scenario).make(structure)
    controller = gcj.controller_job(relax, record, theory="pbe")
    downstream = fake_downstream(controller.output.structure, controller.output.dir_name)
    flow = Flow(
        [relax, controller, downstream],
        metadata={"bmd_stage_directories": ("relax_02", gc.STAGE_DIRECTORY, "static_03")},
    )
    tmp = tmp or tempfile.mkdtemp(prefix="bmd-geometry-convergence-")
    responses = _run_locally_with_stage_directories(
        flow, _flow_stage_directories(flow), root_dir=tmp, store=JobStore(MemoryStore()), ensure_success=False, log=False
    )
    root = Path(tmp).resolve()
    record_path = root / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME
    run_record = json.loads(record_path.read_text()) if record_path.exists() else None
    received = responses.get(downstream.uuid, {}).get(1)
    return root, run_record, (received.output if received is not None else None)


GOOD = {"converged_electronic": True, "converged_ionic": True, "max_force_ev_a": 0.0, "pressure_kbar": 0.3, "max_stress_kbar": 0.4, "spin_polarized": False}
BAD = {**GOOD, "pressure_kbar": 4.0, "max_stress_kbar": 6.0}


def test_immediate_acceptance_releases_the_initial_geometry(fake_vasp):
    root, record, received = _run_controller({"relax_02": {**GOOD, "lattice_a": 5.43}, "probe_initial": GOOD})
    assert (record["status"], record["outcome"], record["refinements"]) == ("accepted", gc.OUTCOME_ACCEPTED_WITHOUT_REFINEMENT, 0)
    assert record["accepted_encut"] == 350 and len(record["attempts"]) == 1
    assert received["a"] == pytest.approx(5.43)
    assert received["dir_name"].endswith("relax_02")
    accepted = root / gc.STAGE_DIRECTORY / gc.ACCEPTED_DIRECTORY
    assert (accepted / "CONTCAR").read_text() == "relax_02 CONTCAR"
    assert (root / "static_03" / "DOWNSTREAM").exists()
    assert record["resource_checks"][0]["source"] == "not_slurm"


def test_refinement_then_acceptance_releases_the_refined_geometry(fake_vasp):
    root, record, received = _run_controller(
        {
            "relax_02": {**GOOD, "lattice_a": 5.43},
            "probe_initial": BAD,
            "refinement": {**GOOD, "lattice_a": 5.47},
            "probe_final": GOOD,
        }
    )
    assert (record["status"], record["outcome"], record["refinements"]) == ("accepted", gc.OUTCOME_ACCEPTED_AFTER_REFINEMENT, 1)
    assert [a["decision"] for a in record["attempts"]] == ["refine", "accept"]
    assert [a["relaxation_encut"] for a in record["attempts"]] == [350, 500]
    assert record["accepted_encut"] == 500
    assert received["a"] == pytest.approx(5.47)  # the refined geometry, not the initial one
    assert "refinement" in received["dir_name"]
    assert (root / gc.STAGE_DIRECTORY / gc.ACCEPTED_DIRECTORY / "CONTCAR").read_text() == "refinement CONTCAR"
    # Every VASP directory is kept.
    assert {p.parent.name for p in root.rglob("evidence.json")} == {"relax_02", "probe_initial", "refinement", "probe_final"}


def test_bi2se3_like_evidence_is_unresolved_and_downstream_never_starts(fake_vasp):
    root, record, received = _run_controller(
        {"relax_02": GOOD, "probe_initial": BAD, "refinement": GOOD, "probe_final": BAD}
    )
    assert (record["status"], record["outcome"]) == ("unresolved", gc.OUTCOME_UNRESOLVED)
    assert "the permitted refinement is exhausted" in record["reasons"]
    assert received is None and not (root / "static_03" / "DOWNSTREAM").exists()
    assert not (root / gc.STAGE_DIRECTORY / gc.ACCEPTED_DIRECTORY).exists()
    assert len(list(root.rglob("evidence.json"))) == 4


def test_the_bmd_runner_fails_the_run_when_the_geometry_is_unresolved(fake_vasp):
    structure = Structure(Lattice.cubic(5.40), ["Si"], [[0, 0, 0]])
    scenario = {"relax_02": GOOD, "probe_initial": BAD, "refinement": GOOD, "probe_final": BAD}
    relax = FakeVaspMaker(name="relax_02", scenario=scenario).make(structure)
    controller = gcj.controller_job(relax, gc.convergence_record(350), theory="pbe")
    downstream = fake_downstream(controller.output.structure, controller.output.dir_name)
    flow = Flow([relax, controller, downstream], metadata={"bmd_stage_directories": ("relax_02", gc.STAGE_DIRECTORY, "static_03")})
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(RuntimeError, match="did not finish"):
        _run_locally_with_stage_directories(
            flow, _flow_stage_directories(flow), root_dir=tmp, store=JobStore(MemoryStore()), ensure_success=True, log=False
        )


def test_a_magnetic_state_change_stops_without_refinement(fake_vasp):
    magnetic = {**GOOD, "spin_polarized": True, "magnetisation": {"total": 2.0, "sites": [2.0]}}
    root, record, received = _run_controller(
        {"relax_02": magnetic, "probe_initial": {**magnetic, "magnetisation": {"total": 0.0, "sites": [0.0]}}}
    )
    assert (record["status"], record["outcome"]) == ("unresolved", gc.OUTCOME_UNRESOLVED_MAGNETIC)
    assert received is None and not list(root.rglob("refinement"))


def test_missing_probe_evidence_is_invalid_evidence(fake_vasp, monkeypatch):
    original = _read_fake_evidence

    def missing_probe(dir_name):
        if "probe" in str(dir_name):
            raise gcj.EvidenceError("missing", "no vasprun.xml")
        return original(dir_name)

    monkeypatch.setattr(gcj, "read_stage_evidence", missing_probe)
    root, record, received = _run_controller({"relax_02": GOOD, "probe_initial": GOOD})
    assert (record["status"], record["outcome"]) == ("invalid_evidence", gc.OUTCOME_INVALID_EVIDENCE)
    assert "missing evidence" in record["reasons"][0] and received is None


def test_a_failed_initial_probe_is_recorded_as_failed_by_the_runner(fake_vasp):
    root, record, received = _run_controller({"relax_02": GOOD})  # the probe raises (no scenario entry)
    assert (record["status"], record["outcome"]) == (gc.STATUS_EXECUTION_FAILURE, gc.OUTCOME_EXECUTION_FAILURE)
    assert record["pending_stage"] is None
    assert _operations(record) == [("probe_initial", "failed")]
    assert record["operations"][0]["failure"].startswith("probe_initial failed: KeyError")
    assert record["reasons"] == [record["operations"][0]["failure"]]
    assert received is None


def test_resource_budget_exhaustion_stops_before_the_refinement(fake_vasp, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    remaining = iter([(10_000.0, "squeue"), (100.0, "squeue")])
    monkeypatch.setattr(gcj, "remaining_walltime_seconds", lambda: next(remaining))
    monkeypatch.setattr(gcj, "estimated_stage_seconds", lambda maker, structure, stage_type, theory: 600.0)
    root, record, received = _run_controller({"relax_02": GOOD, "probe_initial": BAD})
    assert (record["status"], record["outcome"]) == ("resource_budget_exhausted", gc.OUTCOME_RESOURCE_BUDGET)
    checks = record["resource_checks"]
    assert [c["passed"] for c in checks] == [True, False] and checks[1]["required_s"] == 1200.0
    assert received is None and not list(root.rglob("refinement"))


def test_unknown_remaining_walltime_under_slurm_stops_safely(fake_vasp, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(gcj, "remaining_walltime_seconds", lambda: (None, "unknown"))
    root, record, received = _run_controller({"relax_02": GOOD})
    assert record["status"] == gc.STATUS_RESOURCE_BUDGET and "could not be determined" in record["reasons"][0]
    assert not list(root.rglob("probe_initial"))


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("1-02:03:04", 93784.0), ("02:03:04", 7384.0), ("3:04", 184.0), ("UNLIMITED", float("inf")), ("INVALID", None)],
)
def test_squeue_time_left_is_parsed(text, seconds):
    assert gcj._parse_squeue_time(text) == seconds


def test_slurm_end_time_is_preferred(monkeypatch):
    import time

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 3600))
    seconds, source = gcj.remaining_walltime_seconds()
    assert source == "SLURM_JOB_END_TIME" and 3500 < seconds <= 3600


def test_real_evidence_reader_reports_missing_and_malformed_outputs(tmp_path):
    with pytest.raises(gcj.EvidenceError) as missing:
        gcj.read_stage_evidence(str(tmp_path))
    assert missing.value.kind == "missing"
    (tmp_path / "vasprun.xml").write_text("<modeling><broken", encoding="utf-8")
    with pytest.raises(gcj.EvidenceError) as malformed:
        gcj.read_stage_evidence(f"host:{tmp_path}")
    assert malformed.value.kind == "malformed"


def test_a_plan_built_before_the_adaptive_controller_is_refused_as_stale(monkeypatch):
    # The approved worst-case resources are bound to the plan: a page built
    # with the pre-adaptive request (relaxation passes only) cannot be
    # prepared under the adaptive plan.
    import test_automatic_resources as resource_cases

    built = resource_cases._build(workflow="relaxed_structure")
    token = built["selected_resources"]["planned_token"]
    assert "walltime=03:00:00" in token
    stale = token.replace("walltime=03:00:00", "walltime=01:30:00")
    identity = built["submission_spec"]["submission"]
    monkeypatch.setattr(main, "prepare_remote_submission", lambda *a, **k: pytest.fail("prepared"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.prepare_remote(
            resource_cases._request("/prepare-remote"),
            structure=SI_POSCAR, fmt="poscar", purpose=None, theory=None, modifiers=None,
            cpus="", memory_gb="", walltime="", queue=None, created_at=None,
            submission_attempt_id=identity["attempt_id"],
            submission_identity_token=identity["identity_token"],
            workflow_spec_json=None, workflow="relaxed_structure", method=None,
            planned_resources=stale,
        )
    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == "planned_resources_changed"


def test_the_runtime_budget_uses_the_planning_estimate_of_the_same_stage():
    plan = _plan(desired_output="relaxed_structure")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    record = gc.convergence_record(550)
    final = flow.jobs[1].maker
    planned = {
        stage["geometry_convergence_role"]: stage["seconds"]
        for stage in plan.submission_spec["flow_spec"]["resource_selection"]["walltime_estimate"]["stages"]
        if stage.get("geometry_convergence_role")
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = gcj.estimated_stage_seconds(gcj.probe_maker(final, record, name="p"), plan.structure, stage_type="static", theory="pbe")
        refinement = gcj.estimated_stage_seconds(
            gcj.refinement_maker(final, record, name="r"), plan.structure, stage_type="relax", theory="pbe"
        )
    assert probe == pytest.approx(planned[gc.ROLE_INITIAL_PROBE])
    assert refinement == pytest.approx(planned[gc.ROLE_REFINEMENT])


def test_controller_stages_fail_the_workflow_when_unsuccessful():
    plan = _plan(desired_output="relaxed_structure")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    record = gc.convergence_record(550)
    for maker in (gcj.probe_maker(flow.jobs[1].maker, record, name="p"), gcj.refinement_maker(flow.jobs[1].maker, record, name="r")):
        assert maker.stop_children_kwargs == {"handle_unsuccessful": "error"}


# -------------------------------------- P1: frozen reciprocal-space sampling


def test_the_input_set_uses_a_frozen_mesh_verbatim_where_it_would_otherwise_regenerate_it():
    plan = _plan(desired_output="relaxed_structure")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    final = flow.jobs[1].maker
    record = gc.convergence_record(550)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = final.input_set_generator.get_input_set(plan.structure, potcar_spec=True).kpoints
        changed = None
        for scale in (0.80, 0.85, 0.90, 1.10, 1.20, 1.30, 1.50):
            candidate = plan.structure.copy()
            candidate.scale_lattice(plan.structure.volume * scale**3)
            regenerated = final.input_set_generator.get_input_set(candidate, potcar_spec=True).kpoints
            if regenerated.kpts != frozen.kpts:
                changed = candidate
                break
        assert changed is not None, "no tested cell change regenerates a different automatic mesh"
        unfrozen = gcj.probe_maker(final, record, name="p").input_set_generator.get_input_set(changed, potcar_spec=True)
        probe = gcj.probe_maker(final, record, name="p", kpoints=frozen).input_set_generator.get_input_set(changed, potcar_spec=True)
        refinement = gcj.refinement_maker(final, record, name="r", kpoints=frozen).input_set_generator.get_input_set(
            changed, potcar_spec=True
        )
    assert unfrozen.kpoints.kpts != frozen.kpts  # the defect Codex found
    assert str(probe.kpoints) == str(refinement.kpoints) == str(frozen)


def test_probes_and_refinement_sample_the_preceding_relaxations_mesh_when_the_cell_changes(fake_vasp):
    # The refined cell (a = 7.0) would generate a 3x3x3 automatic mesh; the
    # rejected relaxation (a = 5.43) used 4x4x4, which every later stage keeps.
    root, record, received = _run_controller(
        {
            "relax_02": {**GOOD, "lattice_a": 5.43},
            "probe_initial": BAD,
            "refinement": {**GOOD, "lattice_a": 7.0},
            "probe_final": GOOD,
        }
    )
    assert record["outcome"] == gc.OUTCOME_ACCEPTED_AFTER_REFINEMENT
    convergence = root / gc.STAGE_DIRECTORY
    meshes = {
        path.parent.name: [[int(v) for v in row] for row in Kpoints.from_file(str(path)).kpts]
        for path in root.rglob("KPOINTS")
        if gc.ACCEPTED_DIRECTORY not in path.parts
    }
    assert meshes == {"relax_02": [[4, 4, 4]], "probe_initial": [[4, 4, 4]], "refinement": [[4, 4, 4]], "probe_final": [[4, 4, 4]]}
    assert [list(row) for row in _automatic_mesh(Structure(Lattice.cubic(7.0), ["Si"], [[0, 0, 0]])).kpts] == [[3, 3, 3]]
    frozen = record["frozen_kpoints"]
    assert frozen["kpts"] == [[4.0, 4.0, 4.0]] and frozen["source"].endswith("relax_02")
    for attempt in record["attempts"]:
        assert attempt["relaxation_kpoints"]["sha256"] == attempt["probe_kpoints"]["sha256"] == frozen["sha256"]
        assert attempt["checks"][0] == {"name": "kpoints_identical", "passed": True}
    assert (convergence / gc.RECORD_FILENAME).exists()


def test_a_probe_on_a_different_mesh_is_invalid_evidence(fake_vasp):
    root, record, received = _run_controller(
        {"relax_02": {**GOOD, "lattice_a": 5.43}, "probe_initial": {**GOOD, "lattice_a": 7.0, "ignore_frozen_kpoints": True}}
    )
    assert (record["status"], record["outcome"]) == (gc.STATUS_INVALID_EVIDENCE, gc.OUTCOME_INVALID_EVIDENCE)
    assert "frozen k-point mesh" in record["reasons"][0] and received is None


def test_a_refinement_on_a_different_mesh_stops_before_the_final_probe(fake_vasp):
    root, record, received = _run_controller(
        {
            "relax_02": {**GOOD, "lattice_a": 5.43},
            "probe_initial": BAD,
            "refinement": {**GOOD, "lattice_a": 7.0, "ignore_frozen_kpoints": True},
            "probe_final": GOOD,
        }
    )
    assert record["status"] == gc.STATUS_INVALID_EVIDENCE and "refinement did not use the frozen" in record["reasons"][0]
    assert not list(root.rglob("probe_final/evidence.json")) and received is None


# ------------------------------------------ P1: magnetic evidence in the runner


def test_missing_site_moments_are_never_accepted_in_the_runner(fake_vasp):
    total_only = {**GOOD, "spin_polarized": True, "magnetisation": {"total": 2.0, "sites": []}}
    root, record, received = _run_controller({"relax_02": total_only, "probe_initial": total_only})
    assert (record["status"], record["outcome"]) == (gc.STATUS_INVALID_EVIDENCE, gc.OUTCOME_INVALID_EVIDENCE)
    assert "site-resolved" in record["reasons"][0]
    check = {c["name"]: c for c in record["attempts"][0]["checks"]}["probe_magnetisation"]
    assert check["magnetic_state_verified"] is False
    assert received is None and not (root / gc.STAGE_DIRECTORY / gc.ACCEPTED_DIRECTORY).exists()


def test_preserved_site_moments_are_recorded_as_verified(fake_vasp):
    magnetic = {**GOOD, "spin_polarized": True, "magnetisation": {"total": 2.0, "sites": [1.7, 0.3]}}
    root, record, received = _run_controller({"relax_02": magnetic, "probe_initial": magnetic})
    check = {c["name"]: c for c in record["attempts"][0]["checks"]}["probe_magnetisation"]
    assert record["status"] == "accepted" and check["magnetic_state_verified"] is True


# ----------------------------------------- P1: pending-operation provenance


def _operations(record):
    return [(entry["operation"], entry["state"]) for entry in record["operations"]]


def test_the_record_names_each_dynamic_operation_before_it_runs(fake_vasp):
    root, record, received = _run_controller(
        {"relax_02": GOOD, "probe_initial": BAD, "refinement": GOOD, "probe_final": GOOD}
    )
    assert _operations(record) == [
        ("probe_initial", "completed"),
        ("decision_initial", "completed"),
        ("refinement", "completed"),
        ("checkpoint_refinement", "completed"),
        ("probe_final", "completed"),
        ("decision_final", "completed"),
    ]
    assert record["pending_stage"] is None


def test_a_failed_refinement_is_recorded_as_failed_by_the_runner(fake_vasp):
    root, record, received = _run_controller({"relax_02": GOOD, "probe_initial": BAD})  # refinement raises
    assert record["status"] == gc.STATUS_EXECUTION_FAILURE and record["pending_stage"] is None
    assert _operations(record) == [("probe_initial", "completed"), ("decision_initial", "completed"), ("refinement", "failed")]
    assert "refinement failed: KeyError" in record["reasons"][0]
    assert record["attempts"][0]["decision"] == "refine"  # earlier evidence kept
    assert (root / gc.STAGE_DIRECTORY / "probe_initial" / "evidence.json").exists()  # completed evidence kept
    assert received is None


def test_a_failed_final_probe_is_recorded_as_failed_by_the_runner(fake_vasp):
    root, record, received = _run_controller({"relax_02": GOOD, "probe_initial": BAD, "refinement": GOOD})
    assert record["status"] == gc.STATUS_EXECUTION_FAILURE and record["pending_stage"] is None
    assert _operations(record)[-3:] == [
        ("refinement", "completed"), ("checkpoint_refinement", "completed"), ("probe_final", "failed"),
    ]  # not the refinement
    assert "probe_final failed: KeyError" in record["reasons"][0]
    assert len(list(root.rglob("evidence.json"))) == 3 and received is None


def test_a_failing_decision_is_recorded_as_an_execution_failure(fake_vasp, monkeypatch):
    def broken(**kwargs):
        raise ValueError("synthetic evaluation failure")

    monkeypatch.setattr(gc, "evaluate_probe", broken)
    root, record, received = _run_controller({"relax_02": GOOD, "probe_initial": GOOD})
    assert (record["status"], record["outcome"]) == (gc.STATUS_EXECUTION_FAILURE, gc.OUTCOME_EXECUTION_FAILURE)
    assert "decision_initial failed: ValueError" in record["reasons"][0]
    assert _operations(record)[-1] == ("decision_initial", "failed") and received is None


def test_an_interrupted_decision_leaves_the_decision_pending(fake_vasp, monkeypatch, tmp_path):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(gcj, "_decide", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run_controller({"relax_02": GOOD, "probe_initial": GOOD}, tmp=str(tmp_path))
    record = json.loads((tmp_path / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME).read_text())
    assert record["status"] == gc.STATUS_IN_PROGRESS
    assert record["pending_stage"]["operation"] == gc.OPERATION_INITIAL_DECISION


# --------------------------------------- P1: one adaptive relaxation per workflow


TWO_PAIRS = {
    "stages": [
        _stage("relax", options=gr.protocol_option(1)),
        _stage("relax", options=gr.protocol_option(2)),
        _stage("relax", options=gr.protocol_option(1)),
        _stage("relax", options=gr.protocol_option(2)),
    ]
}


def test_custom_validation_refuses_two_canonical_pairs():
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_user_workflow_spec(WorkflowSpec.from_dict(TWO_PAIRS))
    assert excinfo.value.diagnostic["code"] == "geometry_relaxation_multiple_pairs"
    assert "one BMD Geometry Relaxation" in str(excinfo.value)
    # Ordinary explicit relaxations beside one pair remain valid.
    one_pair = {"stages": [*TWO_PAIRS["stages"][:2], _stage("relax"), _stage("static")]}
    validate_user_workflow_spec(WorkflowSpec.from_dict(one_pair))


def test_planning_refuses_two_canonical_pairs_on_every_route():
    with pytest.raises(CalculationValidationError) as excinfo:
        _plan(custom=TWO_PAIRS)
    assert excinfo.value.diagnostic["code"] == "geometry_relaxation_multiple_pairs"
    from starlette.requests import Request

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.build_workflow(
            Request({"type": "http", "method": "POST", "path": "/build-workflow", "headers": []}),
            structure=SI_POSCAR, fmt="poscar", purpose=None, theory=None, modifiers=None,
            cpus=None, memory_gb=None, walltime=None, queue=None,
            workflow_spec_json=json.dumps({**TWO_PAIRS, "label": None, "recipe": None}),
            workflow="custom", method=None,
        )
    assert response.status_code == 400
    assert response.context["calculation_error"]["diagnostic"]["code"] == "geometry_relaxation_multiple_pairs"


def test_machine_api_refuses_two_canonical_pairs(client, tokens):
    from test_machine_api import plan_body, post_plan

    response = post_plan(client, tokens["plan"], plan_body(custom=TWO_PAIRS))
    assert response.status_code in (400, 422), response.text
    assert "geometry_relaxation_multiple_pairs" in response.text


def test_workflow_construction_refuses_two_controllers_defensively(monkeypatch):
    structure = parse_structure(SI_POSCAR)
    workflow = WorkflowSpec.from_dict(TWO_PAIRS)
    monkeypatch.setattr(gr, "validate_workflow", lambda workflow: None)  # bypass validation
    frozen = gr.freeze_plane_wave_cutoffs(structure, workflow, potcar_functional="PBE_64")
    assert sum(gc.stage_convergence_record(stage) is not None for stage in frozen.stages) == 2
    with pytest.raises(CalculationValidationError) as excinfo:
        build_atomate2_flow_for_workflow_spec(structure, frozen, resources={"ntasks": 24})
    assert excinfo.value.diagnostic["code"] == "geometry_relaxation_multiple_pairs"


# --------------------------- P1: failed-job diagnostics and accepted-stage label


def _adaptive_job_record(run_dir=None):
    import test_monitor_trust_boundary as boundary

    spec = boundary.submission_spec()
    spec["flow_spec"]["workflow_spec"] = _plan(desired_output="relaxed_structure").workflow_spec.to_dict()
    spec["flow_spec"]["workflow"] = "custom_workflow"
    if run_dir is not None:
        spec["paths"]["run_dir"] = str(run_dir)
    run_dir = spec["paths"]["run_dir"]
    spec["paths"]["stage_dirs"] = {name: f"{run_dir}/{name}" for name in ("relax_01", "relax_02")}
    return boundary, spec, boundary.job_record(spec=spec)


def _write_record(run_dir, record):
    directory = Path(run_dir) / gc.STAGE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / gc.RECORD_FILENAME
    path.write_text(record if isinstance(record, str) else json.dumps(record), encoding="utf-8")
    return path


class _DiagnosticRunner:
    """The monitoring boundary: job record over the general reader; the
    convergence record only through the contained-read helper (real checks,
    on a local run directory)."""

    def __init__(self, job_record, state="FAILED"):
        self.job_record, self.state = job_record, state
        self.general_reads = []
        self.contained_reads = []
        self.query_calls = []

    def connect(self, profile):
        pass

    def close(self):
        pass

    def is_file(self, path):
        self.general_reads.append(path)
        return "geometry_convergence" not in path

    def read_text(self, path, *, max_bytes=None):
        self.general_reads.append(path)
        assert "geometry_convergence" not in path, "the record must not be read through the general reader"
        return json.dumps(self.job_record)

    def read_contained_file(self, root, parts, *, max_bytes, python=None):
        from backend.secure_remote_read import read_contained_regular_file

        self.contained_reads.append((root, tuple(parts), max_bytes))
        return read_contained_regular_file(root, parts, max_bytes)

    def query_job(self, job_id):
        from backend.remote import RemoteJobStatus

        self.query_calls.append(job_id)
        return RemoteJobStatus(job_id=job_id, state=self.state, exit_code="1:0" if self.state == "FAILED" else "0:0",
                               raw={"brief": f"{job_id}|{self.state}"})


def _monitor(job_record, *, attempt=None, state="FAILED"):
    from backend.monitoring import monitor_submitted_job

    import test_monitor_trust_boundary as boundary

    runner = _DiagnosticRunner(job_record, state=state)
    result, record = monitor_submitted_job(
        boundary.JOB_ID, authenticated_attempt_id=attempt or boundary.ATTEMPT_ID, runner_factory=lambda: runner
    )
    return result, record, runner


def _unresolved_record():
    return {
        "schema": gc.RECORD_SCHEMA,
        "schema_version": gc.RECORD_SCHEMA_VERSION,
        "status": gc.STATUS_UNRESOLVED,
        "outcome": gc.OUTCOME_UNRESOLVED,
        "reasons": ["probe |hydrostatic pressure| (kbar) 4 exceeds 1", "the permitted refinement is exhausted"],
        "policy": {**gc.convergence_record(350), "injected": "<script>"},
        "refinements": 1,
        "attempts": [
            {"role": "initial", "relaxation_stage": "final pass (stage 2 of the canonical relaxation)", "relaxation_encut": 350,
             "probe_encut": 700, "relaxation_dir": "/secret/run/relax_02", "probe": {"max_force_ev_a": 0.0, "pressure_kbar": 4.0, "max_stress_kbar": 6.0},
             "category": "rejected", "decision": "refine", "checks": [], "reasons": ["probe |hydrostatic pressure| (kbar) 4 exceeds 1"]},
            {"role": "refinement", "relaxation_stage": "refinement relaxation", "relaxation_encut": 500, "probe_encut": 700,
             "probe": {"max_force_ev_a": 0.0, "pressure_kbar": 3.0, "max_stress_kbar": 5.0},
             "category": "rejected", "decision": "stop", "checks": [], "reasons": ["probe |hydrostatic pressure| (kbar) 3 exceeds 1"]},
        ],
    }


def test_an_authorized_failed_adaptive_job_shows_its_bounded_convergence_record(tmp_path):
    boundary, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", _unresolved_record())
    result, record, runner = _monitor(job_record)
    assert result["summary"] == "FAILURE"  # still a failed calculation
    convergence = result["geometry_convergence"]
    assert (convergence["status"], convergence["category"]) == ("unresolved", "scientific_non_acceptance")
    assert [a["relaxation_encut"] for a in convergence["attempts"]] == [350, 500]
    assert convergence["attempts"][1]["probe_pressure_kbar"] == 3.0
    shown = json.dumps(convergence)
    assert "/secret/run" not in shown and "injected" not in shown  # no paths, no unapproved fields
    # Exactly one fixed file under the authenticated run directory, through the contained reader.
    assert runner.contained_reads == [(str(tmp_path / "run"), (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024 * 1024)]
    assert not [p for p in runner.general_reads if "geometry_convergence" in p]
    # The page shows it, still as a failure.
    response = boundary.refresh_with_factory(lambda: _DiagnosticRunner(job_record), main.monitor_state_json(submission_spec=spec))
    html = boundary.response_html(response)
    assert 'data-geometry-convergence="scientific_non_acceptance"' in html
    assert "Not accepted (scientific criteria)" in html and "the permitted refinement is exhausted" in html
    assert 'class="pill failed">FAILURE<' in html


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (gc.STATUS_INVALID_EVIDENCE, "missing_or_invalid_evidence"),
        (gc.STATUS_EXECUTION_FAILURE, "execution_failure"),
        (gc.STATUS_RESOURCE_BUDGET, "resource_exhaustion"),
        (gc.STATUS_IN_PROGRESS, "interrupted_or_abandoned"),
    ],
)
def test_diagnostic_categories_are_distinguished(tmp_path, status, category):
    boundary, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", {**_unresolved_record(), "status": status, "outcome": None if status == gc.STATUS_IN_PROGRESS else status,
                                     "pending_stage": {"operation": "probe_final"}})
    result, _, _ = _monitor(job_record)
    assert result["geometry_convergence"]["category"] == category


def test_an_unauthorized_request_never_reads_the_convergence_record(tmp_path):
    import test_monitor_trust_boundary as boundary

    boundary_, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", _unresolved_record())
    result, record, runner = _monitor(job_record, attempt=boundary.OTHER_ATTEMPT_ID)
    assert result["status"] == "failed" and record is None and "geometry_convergence" not in result
    assert runner.contained_reads == [] and runner.query_calls == []


def test_no_diagnostic_for_successful_or_non_adaptive_jobs(tmp_path):
    import test_monitor_trust_boundary as boundary

    _, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", _unresolved_record())
    result, _, runner = _monitor(job_record, state="COMPLETED")
    assert result["summary"] == "SUCCESS" and "geometry_convergence" not in result and runner.contained_reads == []
    result, _, runner = _monitor(boundary.job_record())
    assert result["summary"] == "FAILURE" and "geometry_convergence" not in result and runner.contained_reads == []


def test_a_missing_record_is_reported_without_failing_monitoring(tmp_path):
    _, spec, job_record = _adaptive_job_record(tmp_path / "run")
    (tmp_path / "run").mkdir()
    result, _, _ = _monitor(job_record)
    assert result["status"] == "success" and result["geometry_convergence"]["status"] == "unavailable"


def _replace_with_symlink(run_dir, target):
    path = Path(run_dir) / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    path.symlink_to(target)


@pytest.mark.parametrize("where", ["outside", "inside"])
def test_a_symlinked_record_gives_no_diagnostic(tmp_path, where):
    run = tmp_path / "run"
    _, spec, job_record = _adaptive_job_record(run)
    target = (tmp_path / "elsewhere.json") if where == "outside" else (run / "copy.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_unresolved_record()))
    _replace_with_symlink(run, target)
    result, _, runner = _monitor(job_record)
    assert result["summary"] == "FAILURE" and "geometry_convergence" not in result
    assert len(runner.contained_reads) == 1


def test_a_symlinked_record_directory_gives_no_diagnostic(tmp_path):
    run = tmp_path / "run"
    _, spec, job_record = _adaptive_job_record(run)
    _write_record(tmp_path / "other", _unresolved_record())
    run.mkdir()
    (run / gc.STAGE_DIRECTORY).symlink_to(tmp_path / "other" / gc.STAGE_DIRECTORY)
    result, _, _ = _monitor(job_record)
    assert "geometry_convergence" not in result


@pytest.mark.parametrize(
    "content",
    [
        "x" * (1024 * 1024 + 1),
        json.dumps({"schema": gc.RECORD_SCHEMA, "schema_version": 1, "status": "unresolved"}) + " " * (1024 * 1024),
    ],
    ids=["oversized", "valid-json-then-padding"],
)
def test_an_oversized_record_gives_no_diagnostic(tmp_path, content):
    _, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", content)
    result, _, _ = _monitor(job_record)
    assert "geometry_convergence" not in result


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_a_non_regular_record_gives_no_diagnostic(tmp_path, kind):
    import os

    run = tmp_path / "run"
    _, spec, job_record = _adaptive_job_record(run)
    path = run / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME
    path.parent.mkdir(parents=True)
    os.mkfifo(path) if kind == "fifo" else path.mkdir()
    result, _, _ = _monitor(job_record)
    assert "geometry_convergence" not in result


@pytest.mark.parametrize(
    "content",
    ["{", json.dumps(["not", "a", "record"]), json.dumps({"schema": "other", "status": "unresolved"}),
     json.dumps({"schema": gc.RECORD_SCHEMA, "schema_version": 1, "status": "made_up"}), b"\xff\xfe".decode("latin-1")],
)
def test_a_malformed_record_gives_no_diagnostic(tmp_path, content):
    _, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", content)
    result, _, _ = _monitor(job_record)
    assert "geometry_convergence" not in result


def test_contained_read_returns_the_checked_file_even_if_the_path_is_replaced(tmp_path):
    from backend.secure_remote_read import read_contained_regular_file

    path = _write_record(tmp_path / "run", "original")

    def replace(checked_path):
        assert checked_path == str(path)
        path.unlink()
        path.write_text("substituted")  # a new file at the same path after the checks

    data = read_contained_regular_file(str(tmp_path / "run"), (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024, _after_open=replace)
    assert data == b"original"
    os_path = tmp_path / "run" / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME
    _replace_with_symlink(tmp_path / "run", tmp_path / "target")
    (tmp_path / "target").write_text("x")
    from backend.secure_remote_read import ContainedReadError

    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(str(tmp_path / "run"), (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024)
    assert excinfo.value.code == "not_regular" and os_path.is_symlink()


def test_contained_read_rejects_a_grown_file_rather_than_truncating(tmp_path):
    from backend.secure_remote_read import ContainedReadError, read_contained_regular_file

    path = _write_record(tmp_path / "run", "ok")

    def grow(checked_path):
        with open(checked_path, "a") as handle:  # same inode, appended after the size check
            handle.write("x" * 2048)

    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(str(tmp_path / "run"), (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024, _after_open=grow)
    assert excinfo.value.code == "too_large" and path.exists()


@pytest.mark.parametrize(
    ("root", "parts", "code"),
    [
        ("relative/run", ("geometry_convergence", "geometry_convergence.json"), "invalid_root"),
        ("/tmp", ("..", "etc"), "invalid_path"),
        ("/tmp", ("a/b",), "invalid_path"),
        ("/tmp", (), "invalid_path"),
    ],
)
def test_contained_read_refuses_unsafe_paths(root, parts, code):
    from backend.secure_remote_read import ContainedReadError, read_contained_regular_file

    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(root, parts, 1024)
    assert excinfo.value.code == code


def test_a_symlinked_run_directory_is_refused(tmp_path):
    from backend.secure_remote_read import ContainedReadError, read_contained_regular_file

    _write_record(tmp_path / "real", "{}")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(str(tmp_path / "link"), (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024)
    assert excinfo.value.code == "root_unavailable"


def test_the_paramiko_helper_runs_the_same_checks_remotely_in_one_process(tmp_path, monkeypatch):
    # The remote command is the module's own source, executed here with this
    # Python instead of over SSH; the general readers are never used.
    import subprocess
    import sys

    from backend.paramiko_remote import ParamikoRemoteRunner
    from backend.remote import RemoteCommandResult
    from backend.secure_remote_read import ContainedReadError

    commands = []

    def run_python(self, source, *, python, env=None, check=False, timeout_s=None):
        commands.append(python)
        done = subprocess.run([sys.executable, "-I", "-"], input=source, capture_output=True, text=True, timeout=60)
        return RemoteCommandResult(command="python -", returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)

    monkeypatch.setattr(ParamikoRemoteRunner, "run_python", run_python)
    monkeypatch.setattr(ParamikoRemoteRunner, "read_bytes", lambda *a, **k: pytest.fail("general reader used"))
    runner = ParamikoRemoteRunner.__new__(ParamikoRemoteRunner)
    _write_record(tmp_path / "run", json.dumps(_unresolved_record()))
    parts = (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME)
    assert json.loads(runner.read_contained_file(str(tmp_path / "run"), parts, max_bytes=1024 * 1024))["status"] == "unresolved"
    _replace_with_symlink(tmp_path / "run", tmp_path / "outside.json")
    (tmp_path / "outside.json").write_text("{}")
    with pytest.raises(ContainedReadError) as excinfo:
        runner.read_contained_file(str(tmp_path / "run"), parts, max_bytes=1024 * 1024)
    assert excinfo.value.code == "not_regular"
    from backend.config import DEFAULT_REMOTE_PYTHON

    assert commands == [DEFAULT_REMOTE_PYTHON, DEFAULT_REMOTE_PYTHON]


def test_runners_without_the_contained_reader_give_no_diagnostic(tmp_path):
    from backend.monitoring import geometry_convergence_diagnostic
    from backend.remote import RemoteRunner

    _, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", _unresolved_record())

    class Plain:
        read_contained_file = RemoteRunner.read_contained_file

    assert geometry_convergence_diagnostic(Plain(), job_record, {"summary": "FAILURE"}) is None


# ------------------------------- runner-observed failures and interruptions


def test_an_interrupted_vasp_operation_stays_started_and_is_shown_as_interrupted(fake_vasp, tmp_path):
    from backend.remote_result_parser import geometry_convergence_summary

    with pytest.raises(KeyboardInterrupt):
        _run_controller({"relax_02": GOOD, "probe_initial": {"raise": "interrupt"}}, tmp=str(tmp_path))
    record = json.loads((tmp_path / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME).read_text())
    assert record["status"] == gc.STATUS_IN_PROGRESS and _operations(record) == [("probe_initial", "started")]
    summary = geometry_convergence_summary(record)
    assert (summary["status"], summary["category"], summary["pending_operation"]) == (
        "interrupted", "interrupted_or_abandoned", "probe_initial"
    )


def test_the_original_failure_still_propagates_when_the_runner_raises_immediately(fake_vasp, tmp_path):
    structure = Structure(Lattice.cubic(5.40), ["Si"], [[0, 0, 0]])
    relax = FakeVaspMaker(name="relax_02", scenario={"relax_02": GOOD}).make(structure)
    controller = gcj.controller_job(relax, gc.convergence_record(350), theory="pbe")
    flow = Flow([relax, controller], metadata={"bmd_stage_directories": ("relax_02", gc.STAGE_DIRECTORY)})
    with pytest.raises(KeyError):
        _run_locally_with_stage_directories(
            flow, _flow_stage_directories(flow), root_dir=str(tmp_path), store=JobStore(MemoryStore()),
            ensure_success=True, log=False, raise_immediately=True,
        )
    record = json.loads((tmp_path / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME).read_text())
    assert _operations(record) == [("probe_initial", "failed")]


def test_a_controller_stop_is_not_rewritten_as_an_execution_failure(tmp_path):
    gcj.write_record(tmp_path, {**_unresolved_record(), "operations": [{"operation": "decision_final", "state": "completed"}]})
    gcj.record_observed_failure({"record_dir": str(tmp_path), "operation": "decision_final"}, gcj.GeometryConvergenceStopped("x"))
    assert gcj.read_record(tmp_path)["status"] == gc.STATUS_UNRESOLVED


def test_a_failure_recorder_problem_never_masks_the_original_failure(tmp_path, caplog):
    from backend.execution import _observe_job_failure

    job = fake_downstream({}, "x")
    job.metadata[gcj.OPERATION_METADATA_KEY] = {"record_dir": str(tmp_path / "missing"), "operation": "probe_initial"}
    _observe_job_failure(job, RuntimeError("vasp"))  # no record there: logged, not raised
    assert "Could not record the failure" in caplog.text


def test_non_adaptive_jobs_are_unaffected_by_the_failure_observer(tmp_path):
    from backend.execution import _observe_job_failure

    job = fake_downstream({}, "x")
    _observe_job_failure(job, RuntimeError("vasp"))
    assert not list(tmp_path.iterdir())


def _accepted_record(stage):
    record = _unresolved_record()
    record.update(status="accepted", outcome=gc.OUTCOME_ACCEPTED_AFTER_REFINEMENT if stage == "refinement" else gc.OUTCOME_ACCEPTED_WITHOUT_REFINEMENT,
                  accepted_stage=stage, accepted_encut=500 if stage == "refinement" else 350, reasons=[])
    return record


def test_the_accepted_refinement_is_labelled_as_the_refinement_not_stage_2():
    from backend import remote_result_parser as parser

    refined = parser.geometry_convergence_summary(_accepted_record("refinement"))
    assert refined["accepted_stage"] == "refinement" and "refinement" in refined["accepted_stage_label"]
    assert "stage 2" not in refined["accepted_stage_label"]
    initial = parser.geometry_convergence_summary(_accepted_record("initial"))
    assert initial["accepted_stage_label"].startswith("stage 2")
    # The stress diagnostic of the result describes the accepted relaxation.
    plan = _plan(desired_output="relaxed_structure")
    vasprun = protocol_cases._relax_vasprun([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    relaxation = parser._geometry_relaxation_payload(
        plan.submission_spec["flow_spec"]["workflow_spec"], vasprun=vasprun, local_paths={}, vasprun_class=None
    )
    assert relaxation["stage_index"] == 2
    parser.apply_accepted_relaxation(relaxation, refined)
    assert (relaxation["stage_index"], relaxation["accepted_stage"], relaxation["stage_label"]) == (
        None, "refinement", refined["accepted_stage_label"]
    )
    kept = parser._geometry_relaxation_payload(
        plan.submission_spec["flow_spec"]["workflow_spec"], vasprun=vasprun, local_paths={}, vasprun_class=None
    )
    parser.apply_accepted_relaxation(kept, initial)
    assert kept["stage_index"] == 2 and kept["stage_label"].startswith("stage 2")
    # The results page uses the label, not "stage 2", for the refinement.
    html = _render_results({"geometry_relaxation": relaxation, "geometry_convergence": refined})
    assert "the accepted relaxation: refinement relaxation" in html
    assert "final Geometry Optimisation stage (stage None)" not in html
    assert 'data-geometry-convergence="accepted"' in html


def _render_results(extra):
    summary = {
        "status": "success", "title": "Results Summary", "final_energy_ev": -10.0, "energy_per_atom_ev": -5.0,
        "ionic_steps": 3, "electronic_convergence": True, "ionic_convergence": True, "final_formula": "Si", "natoms": 2,
        "files": {}, "diagnostics": {}, "visualizations": [], "viewer": {"format": "cif", "source": "CONTCAR", "cif": ""},
        **extra,
    }
    page = main.templates.get_template("index.html")
    from starlette.requests import Request

    request = Request({"type": "http", "method": "POST", "path": "/resume", "headers": []})
    return page.render(request=request, **main.page_context(results_summary=summary, show_submitted_workflow=True))


def test_historical_results_keep_their_stage_numbering():
    from backend import remote_result_parser as parser

    historical = {"stages": [{"stage_type": "relax", "options": {"plane_wave_cutoff": {"ENCUT": 550}}}]}
    vasprun = protocol_cases._relax_vasprun([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    payload = parser._geometry_relaxation_payload(historical, vasprun=vasprun, local_paths={}, vasprun_class=None)
    assert payload["stage_index"] == 1 and "stage_label" not in payload


# --------------------------------------------- P1: checkpoint provenance


_REFINED = {"relax_02": GOOD, "probe_initial": BAD, "refinement": GOOD, "probe_final": GOOD}


def _is_refinement_dir(dir_name):
    return gcj.local_directory(dir_name).name == "refinement"


def test_a_successful_checkpoint_completes_the_refinement_then_itself(fake_vasp, monkeypatch):
    seen = []
    original = gcj.read_kpoints

    def spy(dir_name):
        # The checkpoint's own validation runs with it durably started.
        if _is_refinement_dir(dir_name) and not seen:
            record_dir = gcj.local_directory(dir_name).parent.parent
            seen.append(_operations(json.loads((record_dir / gc.RECORD_FILENAME).read_text())))
        return original(dir_name)

    monkeypatch.setattr(gcj, "read_kpoints", spy)
    root, record, received = _run_controller(_REFINED)
    assert seen[0][-2:] == [("refinement", "executed"), ("checkpoint_refinement", "started")]
    assert _operations(record)[2:5] == [
        ("refinement", "completed"), ("checkpoint_refinement", "completed"), ("probe_final", "completed"),
    ]
    assert record["outcome"] == gc.OUTCOME_ACCEPTED_AFTER_REFINEMENT


def _checkpoint_kpoints(monkeypatch, exception):
    original = gcj.read_kpoints
    calls = []

    def patched(dir_name):
        if _is_refinement_dir(dir_name):
            calls.append(dir_name)
            if len(calls) == 1:  # the checkpoint's read (the final decision reads it again)
                raise exception
        return original(dir_name)

    monkeypatch.setattr(gcj, "read_kpoints", patched)


def test_an_observed_checkpoint_exception_marks_the_checkpoint_failed_not_the_refinement(fake_vasp, monkeypatch):
    _checkpoint_kpoints(monkeypatch, OSError("synthetic checkpoint failure"))
    root, record, received = _run_controller(_REFINED)
    assert record["status"] == gc.STATUS_EXECUTION_FAILURE
    assert _operations(record) == [
        ("probe_initial", "completed"), ("decision_initial", "completed"),
        ("refinement", "executed"), ("checkpoint_refinement", "failed"),
    ]
    assert record["reasons"] == ["checkpoint_refinement failed: OSError: synthetic checkpoint failure"]
    assert len(list(root.rglob("refinement/evidence.json"))) == 1  # the refinement's VASP output is kept
    assert not list(root.rglob("probe_final/evidence.json")) and received is None


def test_an_interrupted_checkpoint_leaves_the_checkpoint_started(fake_vasp, monkeypatch, tmp_path):
    _checkpoint_kpoints(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _run_controller(_REFINED, tmp=str(tmp_path))
    record = json.loads((tmp_path / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME).read_text())
    assert record["status"] == gc.STATUS_IN_PROGRESS
    assert _operations(record)[-2:] == [("refinement", "executed"), ("checkpoint_refinement", "started")]
    assert record["pending_stage"] == {"operation": "checkpoint_refinement"}


# ------------------------------------- P1: absolute-path traversal from "/"


def _record_under(base):
    run = base / "a" / "b" / "run"
    _write_record(run, json.dumps(_unresolved_record()))
    return run


@pytest.mark.parametrize("level", ["top", "intermediate", "run_itself"])
def test_a_symlink_in_any_ancestor_of_the_run_directory_is_refused(tmp_path, level):
    from backend.secure_remote_read import ContainedReadError, read_contained_regular_file

    run = _record_under(tmp_path)
    parts = (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME)
    assert read_contained_regular_file(str(run), parts, 1024 * 1024)  # the real path is readable
    if level == "top":
        (tmp_path / "link").symlink_to(tmp_path / "a")
        root = tmp_path / "link" / "b" / "run"
    elif level == "intermediate":
        (tmp_path / "a" / "blink").symlink_to(tmp_path / "a" / "b")
        root = tmp_path / "a" / "blink" / "run"
    else:
        (tmp_path / "a" / "b" / "runlink").symlink_to(run)
        root = tmp_path / "a" / "b" / "runlink"
    assert (root / gc.STAGE_DIRECTORY / gc.RECORD_FILENAME).read_text()  # an ordinary open follows it
    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(str(root), parts, 1024 * 1024)
    assert excinfo.value.code == "root_unavailable"


def test_a_job_whose_run_directory_has_a_symlinked_ancestor_gets_no_diagnostic(tmp_path):
    run = _record_under(tmp_path)
    (tmp_path / "a" / "blink").symlink_to(tmp_path / "a" / "b")
    _, spec, job_record = _adaptive_job_record(tmp_path / "a" / "blink" / "run")
    result, _, runner = _monitor(job_record)
    assert result["summary"] == "FAILURE" and "geometry_convergence" not in result
    assert len(runner.contained_reads) == 1
    _, spec, job_record = _adaptive_job_record(run)
    assert _monitor(job_record)[0]["geometry_convergence"]["status"] == "unresolved"


@pytest.mark.parametrize("root", ["/tmp//run", "/tmp/run/", "/tmp/./run", "/tmp/../tmp/run", "/tmp/r un", "/tmp/r\x00n"])
def test_unsafe_root_components_are_refused(root):
    from backend.secure_remote_read import ContainedReadError, read_contained_regular_file

    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(root, (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024)
    assert excinfo.value.code == "invalid_root"


@pytest.mark.parametrize("flag", ["O_NOFOLLOW", "O_DIRECTORY"])
def test_the_read_fails_closed_without_the_linux_flags(tmp_path, monkeypatch, flag):
    import os

    from backend.secure_remote_read import ContainedReadError, read_contained_regular_file

    run = _record_under(tmp_path)
    monkeypatch.delattr(os, flag)
    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(str(run), (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024 * 1024)
    assert excinfo.value.code == "unsupported_platform"


def test_the_read_fails_closed_without_descriptor_relative_open(tmp_path, monkeypatch):
    import os

    from backend.secure_remote_read import ContainedReadError, read_contained_regular_file

    run = _record_under(tmp_path)
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(ContainedReadError) as excinfo:
        read_contained_regular_file(str(run), (gc.STAGE_DIRECTORY, gc.RECORD_FILENAME), 1024 * 1024)
    assert excinfo.value.code == "unsupported_platform"


# ------------------------------------------ P1: strict record schema version


@pytest.mark.parametrize(
    "version",
    ["missing", None, 2, 0, "1", 1.0, True, [1]],
    ids=["missing", "null", "unknown-2", "unknown-0", "string", "float", "bool", "list"],
)
def test_a_record_without_the_supported_schema_version_gives_no_diagnostic(tmp_path, version):
    from backend.remote_result_parser import geometry_convergence_summary

    record = _unresolved_record()
    if version == "missing":
        record.pop("schema_version")
    else:
        record["schema_version"] = version
    assert geometry_convergence_summary(record)["status"] == "unavailable"
    _, spec, job_record = _adaptive_job_record(tmp_path / "run")
    _write_record(tmp_path / "run", record)
    result, _, _ = _monitor(job_record)
    assert result["summary"] == "FAILURE" and "geometry_convergence" not in result


def test_the_supported_schema_version_is_accepted_and_written_by_the_controller(fake_vasp):
    from backend.remote_result_parser import GEOMETRY_CONVERGENCE_RECORD_SCHEMA_VERSION, geometry_convergence_summary

    assert GEOMETRY_CONVERGENCE_RECORD_SCHEMA_VERSION == gc.RECORD_SCHEMA_VERSION == 1
    assert geometry_convergence_summary(_unresolved_record())["status"] == "unresolved"
    root, record, received = _run_controller({"relax_02": GOOD, "probe_initial": GOOD})
    assert (record["schema"], record["schema_version"]) == (gc.RECORD_SCHEMA, 1)
    assert geometry_convergence_summary(record)["status"] == "accepted"
