"""Explicit, evidence-independent smearing for relaxations, PBE band
structures and phonon displacements.

* Every relaxation fixes ISMEAR = 0, SIGMA = 0.2 itself, so atomate2's
  band-gap-based auto_ismear cannot change it from previous-stage evidence.
* Non-self-consistent PBE band structures fix ISMEAR = 0 (SIGMA = 0.05) rather
  than relying on pymatgen's too-few-k-points fallback.
* Phonon displacement calculations run without Custodian's LargeSigmaHandler,
  so every displacement keeps SIGMA = 0.2; all other handlers are unchanged.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from conftest import planned
from atomate2.vasp.run import DEFAULT_HANDLERS

import test_input_reference_contract as reference_cases
from backend.calculations.custodian_policy import (
    FROZEN_JOB_HANDLER,
    LARGE_SIGMA_HANDLER,
    bmd_custodian_handlers,
    handler_exclusions_for_stage,
    resolved_custodian_policy,
)
from backend.calculations.models import Modifier, StageSpec, Theory
from backend.calculations.registry import desired_output_workflow_spec
from backend.parser import parse_structure
from backend.phonons.workflow import phonon_force_settings_payload
from backend.workflows import (
    build_atomate2_flow_for_workflow_spec,
    build_band_structure_input_set_generator,
    build_phonon_forces_flow,
    build_relax_input_set_generator,
)

SI = parse_structure(reference_cases.SI_POSCAR, "poscar")
NIO = parse_structure(reference_cases.NIO_POSCAR, "poscar")
PREVIOUS_STAGE_GAPS = (None, 0.0, 1.5)  # unknown, metallic, gapped


def _qualified(handler) -> str:
    return f"{type(handler).__module__}.{type(handler).__name__}"


def _incar_for_previous_gap(generator, structure, gap):
    # atomate2 sets ``bandgap`` from the previous stage's vasprun.xml at run time.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        incar = dataclasses.replace(generator, bandgap=gap).get_input_set(structure, potcar_spec=True)["INCAR"]
    return incar, [str(item.message) for item in caught]


# ----------------------------------------------------------------- relaxations


@pytest.mark.parametrize(
    ("structure", "kwargs"),
    [
        (SI, {}),
        (SI, {"isif": 2}),
        (SI, {"theory": Theory.HSE06}),
        (NIO, {"spin_polarized": True, "modifiers": {Modifier.DFT_U}}),
    ],
    ids=["pbe", "ions-only", "hse06", "spin-dft-u"],
)
def test_relaxation_smearing_ignores_previous_stage_evidence(structure, kwargs):
    generator = build_relax_input_set_generator(structure, **kwargs)

    assert (generator.user_incar_settings["ISMEAR"], generator.user_incar_settings["SIGMA"]) == (0, 0.2)
    for gap in PREVIOUS_STAGE_GAPS:
        incar, _ = _incar_for_previous_gap(generator, structure, gap)
        assert (incar["ISMEAR"], incar["SIGMA"]) == (0, 0.2), gap


def test_every_relaxation_in_the_managed_workflows_is_fixed():
    for output in ("relaxed_energy", "relaxed_structure", "electronic_dos", "electronic_band_structure", "phonons"):
        flow = build_atomate2_flow_for_workflow_spec(
            SI, planned(SI, desired_output_workflow_spec(output)), resources={"ntasks": 24}
        )
        relaxes = [job for job in flow.jobs if type(getattr(job, "maker", None)).__name__ == "RelaxMaker"]
        assert relaxes, output
        for job in relaxes:
            settings = job.maker.input_set_generator.user_incar_settings
            assert (settings["ISMEAR"], settings["SIGMA"]) == (0, 0.2), output


# ------------------------------------------------------ PBE band structure


def test_pbe_band_structure_smearing_is_explicit_and_needs_no_fallback():
    generator = build_band_structure_input_set_generator(SI, theory=Theory.PBE)

    assert (generator.user_incar_settings["ISMEAR"], generator.user_incar_settings["SIGMA"]) == (0, 0.05)
    for gap in PREVIOUS_STAGE_GAPS:
        incar, messages = _incar_for_previous_gap(generator, SI, gap)
        assert (incar["ISMEAR"], incar["SIGMA"], incar["ICHARG"]) == (0, 0.05, 11), gap
        # pymatgen's tetrahedron fallback no longer decides ISMEAR.
        assert not any("Too few KPOINTS" in message for message in messages)


def test_hse06_band_structure_keeps_its_c2_generator():
    generator = build_band_structure_input_set_generator(SI, theory=Theory.HSE06, evidence_stage_index=2)
    assert type(generator).__name__ == "HseBandSigmaHSEBSSetGenerator"
    assert (generator.user_incar_settings["ISMEAR"], generator.user_incar_settings["SIGMA"]) == (0, 0.01)


# ------------------------------------------------------- phonon displacements


def _displacement_maker(flow):
    phonon = flow.jobs[-1]
    jobs = {job.name: job for job in phonon.jobs}
    return jobs["run_phonon_displacements"].function_kwargs["phonon_maker"]


@pytest.mark.parametrize("entry", ["workflow", "direct"])
def test_phonon_displacements_share_one_sigma_without_large_sigma_handler(entry):
    stage = StageSpec.from_dict({"stage_type": "phonon_forces", "theory": "pbe", "modifiers": []})
    if entry == "workflow":
        flow = build_atomate2_flow_for_workflow_spec(
            SI, planned(SI, desired_output_workflow_spec("phonons")), resources={"ntasks": 24}
        )
        maker = _displacement_maker(flow)
    else:
        maker = _displacement_maker(type("F", (), {"jobs": [build_phonon_forces_flow(SI, stage=stage, name="stage_02")]}))
    names = [_qualified(handler) for handler in maker.run_vasp_kwargs["handlers"]]

    # One maker generates every displacement calculation, with one SIGMA.
    generator = maker.input_set_generator
    for gap in PREVIOUS_STAGE_GAPS:
        incar, _ = _incar_for_previous_gap(generator, SI, gap)
        assert (incar["ISMEAR"], incar["SIGMA"]) == (0, 0.2)
    assert LARGE_SIGMA_HANDLER not in names
    # Every other BMD handler is kept, in atomate2's order.
    assert names == [
        _qualified(handler)
        for handler in DEFAULT_HANDLERS
        if _qualified(handler) not in {FROZEN_JOB_HANDLER, LARGE_SIGMA_HANDLER}
    ]


def test_only_phonon_and_relaxation_stages_drop_large_sigma_handler():
    # Phonon displacements, and (BMD Standard Geometry Relaxation v1) every
    # Geometry Optimisation, keep the planned SIGMA; other stages are unchanged.
    default_names = [_qualified(handler) for handler in bmd_custodian_handlers()]
    assert LARGE_SIGMA_HANDLER in default_names
    for theory in ("pbe", "hse06"):
        relax = resolved_custodian_policy("relax", theory)
        assert handler_exclusions_for_stage("relax") == (LARGE_SIGMA_HANDLER,)
        assert LARGE_SIGMA_HANDLER not in [item["class"] for item in relax["handlers"]]
        assert relax["explicit_handler_exclusions"] == [FROZEN_JOB_HANDLER, LARGE_SIGMA_HANDLER]
    for stage_type, theory in [
        ("static", "pbe"),
        ("static", "hse06"),
        ("dos", "pbe"),
        ("dos", "hse06"),
        ("band_structure", "pbe"),
        ("band_structure", "hse06"),
    ]:
        assert handler_exclusions_for_stage(stage_type) == ()
        policy = resolved_custodian_policy(stage_type, theory)
        assert LARGE_SIGMA_HANDLER in [item["class"] for item in policy["handlers"]], stage_type
        assert policy["explicit_handler_exclusions"] == [FROZEN_JOB_HANDLER]

    phonon = resolved_custodian_policy("phonon_forces", "pbe")
    assert phonon["explicit_handler_exclusions"] == [FROZEN_JOB_HANDLER, LARGE_SIGMA_HANDLER]
    assert [item["class"] for item in phonon["handlers"]] == [name for name in default_names if name != LARGE_SIGMA_HANDLER]


def test_phonon_plan_identity_records_the_handler_choice():
    assert phonon_force_settings_payload()["force_custodian_handler_exclusions"] == [LARGE_SIGMA_HANDLER]
