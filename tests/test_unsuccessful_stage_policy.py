"""BMD owns the rule that an unsuccessful (unconverged) VASP stage fails the workflow.

atomate2 decides from its task document whether a stage succeeded; what happens
next is BMD policy, passed explicitly as ``stop_children_kwargs`` to every maker
instead of inherited from atomate2's mutable ``VASP_HANDLE_UNSUCCESSFUL``.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

import atomate2.vasp.jobs.base as atomate2_vasp_base
import atomate2.vasp.run as atomate2_vasp_run
from atomate2.settings import Atomate2Settings

from backend.calculations.custodian_policy import (
    UNSUCCESSFUL_STAGE_HANDLING,
    resolved_custodian_policy,
    unsuccessful_stage_stop_children_kwargs,
)
from backend.calculations.models import (
    CalculationSpec,
    Modifier,
    Purpose,
    StageSpec,
    StageType,
    Theory,
    WorkflowSpec,
)
from backend.calculations.registry import desired_output_workflow_spec
from backend.execution import _flow_stage_directories, _run_locally_with_stage_directories
from backend.workflows import (
    build_atomate2_flow_for_spec,
    build_atomate2_flow_for_workflow_spec,
)
from test_automatic_dft_u import NIO, build_route, job_maker, resolve, runtime_jobs


POLICY = {"handle_unsuccessful": "error"}

SI = Structure(
    Lattice([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]),
    ["Si", "Si"],
    [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
)


def _all_makers(flow):
    return [job_maker(job) for job in flow.jobs]


# --- Every BMD-constructed maker carries the policy --------------------------------


def test_policy_is_error_and_returns_fresh_dicts():
    assert UNSUCCESSFUL_STAGE_HANDLING == "error"
    first = unsuccessful_stage_stop_children_kwargs()
    first["handle_unsuccessful"] = False
    assert unsuccessful_stage_stop_children_kwargs() == POLICY


@pytest.mark.parametrize(
    "desired_output",
    ["energy_only", "relaxed_structure", "electronic_dos", "electronic_band_structure"],
)
def test_every_desired_output_stage_uses_the_policy_at_runtime(desired_output):
    response = build_route(NIO, workflow=desired_output)
    _, jobs = runtime_jobs(response.context["submission_spec"])
    assert jobs
    for job in jobs:
        assert job_maker(job).stop_children_kwargs == POLICY


@pytest.mark.parametrize(
    "workflow",
    [
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.DOS, Theory.PBE)], recipe="custom"),
        WorkflowSpec(
            [StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.BAND_STRUCTURE, Theory.PBE)],
            recipe="custom",
        ),
        WorkflowSpec(
            [
                StageSpec(StageType.RELAX, Theory.HSE06),
                StageSpec(StageType.STATIC, Theory.HSE06),
                StageSpec(StageType.DOS, Theory.HSE06),
            ],
            recipe="custom",
        ),
        WorkflowSpec(
            [
                StageSpec(StageType.RELAX, Theory.PBE, {Modifier.IONS_ONLY}),
                StageSpec(StageType.STATIC, Theory.HSE06),
                StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
            ],
            recipe="custom",
        ),
    ],
)
def test_every_custom_workflow_stage_uses_the_policy(workflow):
    flow = build_atomate2_flow_for_workflow_spec(SI, workflow)
    assert all(maker.stop_children_kwargs == POLICY for maker in _all_makers(flow))


@pytest.mark.parametrize(
    "purpose",
    [Purpose.RELAX, Purpose.STATIC, Purpose.RELAX_STATIC, Purpose.DOUBLE_RELAX, Purpose.DOS, Purpose.BAND_STRUCTURE],
)
def test_single_calculation_builders_use_the_policy(purpose):
    flow = build_atomate2_flow_for_spec(SI, CalculationSpec(purpose, Theory.PBE))
    makers = _all_makers(flow)
    assert makers
    assert all(maker.stop_children_kwargs == POLICY for maker in makers)


def test_policy_is_recorded_in_stage_custodian_provenance():
    policy = resolved_custodian_policy("static", "pbe")
    assert policy["unsuccessful_stage"]["stop_children_kwargs"] == POLICY
    spec = build_route(NIO, workflow="relaxed_structure").context["submission_spec"]
    for stage in spec["provenance"]["execution"]["custodian"]["stages"]:
        assert stage["unsuccessful_stage"]["stop_children_kwargs"] == POLICY


# --- Behaviour under a hostile external atomate2 setting ---------------------------


def test_external_setting_can_change_atomate2s_own_default(monkeypatch):
    monkeypatch.setenv("ATOMATE2_VASP_HANDLE_UNSUCCESSFUL", "false")
    assert Atomate2Settings().VASP_HANDLE_UNSUCCESSFUL is False


class StubTaskDoc(dict):
    """Minimal task-document stand-in: atomate2 accepts dict outputs."""

    def __init__(self, state, structure, dir_name):
        super().__init__(
            state=state,
            structure=structure,
            dir_name=dir_name,
            task_label=None,
            custodian=[],
            calcs_reversed=[],
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _stub_vasp(monkeypatch, *, states):
    """Replace VASP I/O inside atomate2's maker; record which stages ran."""

    ran = []

    def fake_write(structure, input_set_generator, **kwargs):
        ran.append(structure.composition.reduced_formula)

    def fake_task_doc(path, **kwargs):
        index = len(ran) - 1
        return StubTaskDoc(states[index], SI, str(path))

    monkeypatch.setattr(atomate2_vasp_base, "write_vasp_input_set", fake_write)
    monkeypatch.setattr(atomate2_vasp_base, "copy_vasp_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(atomate2_vasp_base, "run_vasp", lambda **kwargs: None)
    monkeypatch.setattr(atomate2_vasp_base, "get_vasp_task_document", fake_task_doc)
    monkeypatch.setattr(atomate2_vasp_base, "gzip_output_folder", lambda **kwargs: None)
    return ran


def _external_default(monkeypatch, value):
    # What ATOMATE2_VASP_HANDLE_UNSUCCESSFUL / ~/.atomate2.yaml would set at import.
    monkeypatch.setattr(atomate2_vasp_run.should_stop_children, "__defaults__", (value,))


def _two_stage_flows():
    relax_relax = build_atomate2_flow_for_workflow_spec(SI, desired_output_workflow_spec("relaxed_structure"))
    static_dos = build_atomate2_flow_for_workflow_spec(
        SI,
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.DOS, Theory.PBE)], recipe="custom"),
    )
    return {"second_relaxation": relax_relax, "pbe_dos": static_dos}


@pytest.mark.parametrize("chain", ["second_relaxation", "pbe_dos"])
@pytest.mark.parametrize("external", [False, True, "error"])
def test_unsuccessful_upstream_stage_always_fails_the_workflow(monkeypatch, tmp_path, chain, external):
    _external_default(monkeypatch, external)
    ran = _stub_vasp(monkeypatch, states=["failed", "successful"])
    flow = _two_stage_flows()[chain]

    with pytest.raises(RuntimeError, match="did not finish"):
        _run_locally_with_stage_directories(
            flow, _flow_stage_directories(flow), ensure_success=True, root_dir=tmp_path
        )
    assert len(ran) == 1  # the dependent stage never started


@pytest.mark.parametrize("chain", ["second_relaxation", "pbe_dos"])
def test_without_bmd_policy_the_external_setting_would_let_it_proceed(monkeypatch, tmp_path, chain):
    # Control: this is the behaviour BMD must not inherit.
    _external_default(monkeypatch, False)
    ran = _stub_vasp(monkeypatch, states=["failed", "successful"])
    flow = _two_stage_flows()[chain]
    for maker in _all_makers(flow):
        maker.stop_children_kwargs = {}

    _run_locally_with_stage_directories(
        flow, _flow_stage_directories(flow), ensure_success=True, root_dir=tmp_path
    )
    assert len(ran) == 2


@pytest.mark.parametrize("chain", ["second_relaxation", "pbe_dos"])
def test_successful_stages_are_unaffected(monkeypatch, tmp_path, chain):
    _external_default(monkeypatch, False)
    ran = _stub_vasp(monkeypatch, states=["successful", "successful"])
    flow = _two_stage_flows()[chain]

    _run_locally_with_stage_directories(
        flow, _flow_stage_directories(flow), ensure_success=True, root_dir=tmp_path
    )
    assert len(ran) == 2


def test_unsuccessful_final_stage_fails_even_a_single_stage_run(monkeypatch, tmp_path):
    _external_default(monkeypatch, True)  # would otherwise mark it completed
    _stub_vasp(monkeypatch, states=["failed"])
    maker = job_maker(
        build_atomate2_flow_for_workflow_spec(SI, desired_output_workflow_spec("energy_only")).jobs[0]
    )
    with pytest.raises(RuntimeError, match="not successful"):
        atomate2_vasp_run.should_stop_children(StubTaskDoc("failed", SI, "."), **maker.stop_children_kwargs)
