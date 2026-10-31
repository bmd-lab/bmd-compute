"""C2-informed initial SIGMA of self-consistent HSE06 band-structure stages.

BMD Check's C2 assessment of the preceding HSE06 static chooses the initial
SIGMA of the band-structure stage (metallic 0.20 eV, otherwise 0.01 eV),
ISMEAR = 0 throughout. Verdict mapping and provenance use stubbed C2 results;
missing and invalid evidence use the real installed C2. C2's own verdicts on
recorded VASP evidence are tested in BMD Check.
"""

from __future__ import annotations

import json
import warnings

import pytest

from conftest import planned
from monty.json import MontyDecoder

import main
import test_input_reference_contract as reference_cases
from backend import runtime_environment
from backend.calculations import hse_band_sigma
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import CalculationValidationError, desired_output_workflow_spec
from backend.parser import parse_structure
from backend.runtime_environment import (
    RuntimeParityError,
    parity_critical_packages_for_workflow,
    prepared_runtime_parity,
    runtime_parity_problems,
)
from backend.workflows import build_atomate2_flow_for_workflow_spec, build_band_structure_input_set_generator
from compute_api.plan_digest import compute_plan_digest, stage_input_material
from compute_api.projection import plan_stage_previews

SI = parse_structure(reference_cases.SI_POSCAR, "poscar")
CUSTOM_HSE_BAND = {
    "stages": [
        {"stage_type": "relax", "theory": "pbe", "modifiers": []},
        {"stage_type": "static", "theory": "hse06", "modifiers": []},
        {"stage_type": "band_structure", "theory": "hse06", "modifiers": []},
    ]
}


def _plan(desired_output=None, custom=None, resources=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main.plan_calculation_request(
            structure_text=reference_cases.SI_POSCAR,
            fmt="poscar",
            desired_output=desired_output,
            custom_workflow=custom,
            resources=resources,
        )


def _digest(plan):
    return compute_plan_digest(
        structure=plan.structure, submission_spec=plan.submission_spec, stage_previews=plan_stage_previews(plan)
    )


def _generator(evidence_stage_index=2):
    return build_band_structure_input_set_generator(SI, theory=Theory.HSE06, evidence_stage_index=evidence_stage_index)


@pytest.fixture
def c2(monkeypatch):
    """Stub C2's verdict; record what Compute asked it to assess."""

    from bmd_agent.resources import electronic_character as ec

    calls = {}

    def set_verdict(classification, codes=("CODE",)):
        def assess(directory, **kwargs):
            calls.update(kwargs, directory=str(directory))
            return classification

        monkeypatch.setattr(ec, "assess_stage_directory", assess)
        monkeypatch.setattr(
            ec,
            "serialize_electronic_character_assessment",
            lambda verdict: {
                "schema_id": "bmd_check.electronic_character_assessment",
                "method_version": "1.0.0-c2",
                "classification": verdict,
                "findings": [{"reason_code": code} for code in codes],
                "evidence_references": [{"evidence_id": "stage-2-vasprun-electronic-structure"}],
            },
        )
        return calls

    return set_verdict


@pytest.fixture
def runtime_inputs(monkeypatch, tmp_path):
    """Generate band-structure inputs as at run time, from a copied static."""

    from pymatgen.io.vasp.sets import VaspInputSet

    # The stand-in files are not a parseable VASP run; pymatgen's own reading
    # of the previous calculation (structure, band gap) is not under test.
    monkeypatch.setattr(VaspInputSet, "_set_previous", lambda self, prev_dir=None: None)
    (tmp_path / "vasprun.xml").write_text("<modeling></modeling>\n", encoding="utf-8")

    def generate(generator=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (generator or _generator()).get_input_set(SI, prev_dir=str(tmp_path), potcar_spec=True)

    return generate


def _record(input_set):
    return json.loads(input_set[hse_band_sigma.PROVENANCE_FILENAME])


# ----------------------------------------------------------------- mapping


@pytest.mark.parametrize(
    ("classification", "sigma", "basis"),
    [
        ("metallic", 0.2, "metallic"),
        ("nonmetallic", 0.01, "nonmetallic"),
        ("uncertain", 0.01, "fallback: uncertain (ELECTRONIC_CHARACTER_UNDETERMINED, SINGLE_KPOINT_SAMPLING)"),
    ],
)
def test_c2_verdict_selects_the_initial_sigma_only(c2, runtime_inputs, classification, sigma, basis):
    calls = c2(classification, codes=("ELECTRONIC_CHARACTER_UNDETERMINED", "SINGLE_KPOINT_SAMPLING"))
    preview = _generator().get_input_set(SI, potcar_spec=True)
    executed = runtime_inputs()
    record = _record(executed)

    assert executed["INCAR"]["SIGMA"] == sigma
    assert executed["INCAR"]["ISMEAR"] == 0
    # Every other INCAR setting, the k-points and the structure are unchanged.
    assert {k: v for k, v in executed["INCAR"].items() if k != "SIGMA"} == {
        k: v for k, v in preview["INCAR"].items() if k != "SIGMA"
    }
    assert str(executed["KPOINTS"]) == str(preview["KPOINTS"])
    assert str(executed["POSCAR"]) == str(preview["POSCAR"])
    # C2 assessed the copied HSE06 static as the preceding stage.
    assert calls["stage_index"] == 2 and calls["stage_type"] == "static" and calls["theory"] == "hse06"
    # Provenance agrees with the executed INCAR.
    assert record["initial_SIGMA_ev"] == executed["INCAR"]["SIGMA"]
    assert record["ISMEAR"] == executed["INCAR"]["ISMEAR"]
    assert record["classification"] == classification
    assert record["basis"] == (basis if classification == "uncertain" else classification)
    assert record["method_version"] == "1.0.0-c2"
    assert record["policy"] == {"id": "bmd_compute.hse_band_structure_sigma", "version": 1}
    assert record["evidence_stage_index"] == 2
    assert len(record["evidence_vasprun_sha256"]) == 64
    assert record["assessment"]["evidence_references"]
    assert "custodian.json" in record["custodian"]


def test_previews_show_the_default_and_write_no_decision():
    preview = _generator().get_input_set(SI, potcar_spec=True)
    assert preview["INCAR"]["SIGMA"] == 0.01 and preview["INCAR"]["ISMEAR"] == 0
    assert hse_band_sigma.PROVENANCE_FILENAME not in preview


@pytest.mark.parametrize("contents", [None, "", "<modeling><truncated", "\x00\x01 not xml"])
def test_missing_or_invalid_evidence_falls_back_with_the_reason(tmp_path, contents):
    # The real installed C2 (bmd-agent 0.2.0) assesses the evidence.
    if contents is not None:
        (tmp_path / "vasprun.xml").write_text(contents, encoding="utf-8")
    decision = hse_band_sigma.select_initial_sigma(tmp_path, evidence_stage_index=2)

    assert decision["initial_SIGMA_ev"] == 0.01
    assert decision["classification"] == "uncertain"
    assert decision["basis"].startswith("fallback: uncertain (")
    assert decision["method_version"] == "1.0.0-c2"


@pytest.mark.parametrize("failure", ["raises", "unimportable", "no_stage_index"])
def test_assessment_failures_fall_back_without_stopping(monkeypatch, tmp_path, failure):
    from bmd_agent.resources import electronic_character as ec

    stage_index = 2
    if failure == "raises":
        monkeypatch.setattr(ec, "assess_stage_directory", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    elif failure == "unimportable":
        monkeypatch.setitem(__import__("sys").modules, "bmd_agent.resources.electronic_character", None)
    else:
        stage_index = None
    decision = hse_band_sigma.select_initial_sigma(tmp_path, evidence_stage_index=stage_index)

    assert decision["initial_SIGMA_ev"] == 0.01
    assert decision["classification"] is None
    assert decision["basis"].startswith("fallback: assessment unavailable (")


def test_generated_incar_must_match_the_decision(c2, runtime_inputs, monkeypatch):
    c2("metallic")
    original = hse_band_sigma.select_initial_sigma

    def inconsistent(*args, **kwargs):
        return {**original(*args, **kwargs), "ISMEAR": 1}

    monkeypatch.setattr(hse_band_sigma, "ISMEAR", 1)
    monkeypatch.setattr(hse_band_sigma, "select_initial_sigma", inconsistent)
    with pytest.raises(RuntimeError, match="does not match"):
        runtime_inputs()


def test_generators_round_trip_through_serialization():
    from backend.calculations.hse_band_kpoints import full_zone_hse_band_structure_generator_class

    full_zone = hse_band_sigma.hse_band_sigma_generator_class(full_zone_hse_band_structure_generator_class())
    for generator in (_generator(), full_zone(bmd_sigma_evidence_stage_index=3)):
        restored = MontyDecoder().process_decoded(generator.as_dict())
        assert type(restored) is type(generator)
        assert restored.bmd_sigma_evidence_stage_index == generator.bmd_sigma_evidence_stage_index


# ------------------------------------------------------------- stage graph


@pytest.mark.parametrize("workflow", ["managed", "custom"])
def test_band_structure_assesses_the_completed_static_it_follows(workflow):
    spec = (
        planned(SI, desired_output_workflow_spec("electronic_band_structure"))
        if workflow == "managed"
        else WorkflowSpec([StageSpec.from_dict(stage) for stage in CUSTOM_HSE_BAND["stages"]], recipe=None)
    )
    flow = build_atomate2_flow_for_workflow_spec(SI, spec, resources={"ntasks": 24})
    *_, static, band = flow.jobs
    maker = band.function.__self__
    generator = maker.input_set_generator

    assert type(generator).__name__ == "HseBandSigmaHSEBSSetGenerator"
    # The static is stage 3 after the two-pass relaxation; stage 2 in Custom.
    assert generator.bmd_sigma_evidence_stage_index == (3 if workflow == "managed" else 2)
    # The band structure runs after, and copies, the completed static stage.
    prev_dir = band.function_kwargs["prev_dir"]
    assert prev_dir.uuid == static.uuid and prev_dir.attributes == (("a", "dir_name"),)
    assert maker.copy_vasp_kwargs == {"additional_vasp_files": ("CHGCAR", "OSZICAR")}
    assert generator.user_incar_settings["ISMEAR"] == 0 and generator.user_incar_settings["SIGMA"] == 0.01


def test_hse06_dos_and_other_stages_are_unchanged():
    flow = build_atomate2_flow_for_workflow_spec(
        SI, planned(SI, desired_output_workflow_spec("electronic_dos")), resources={"ntasks": 24}
    )
    for job in flow.jobs:
        maker = job.function.__self__
        assert not type(maker.input_set_generator).__name__.startswith("HseBandSigma")
        assert "additional_vasp_files" not in (getattr(maker, "copy_vasp_kwargs", None) or {})


# ------------------------------------------------------- runtime parity


@pytest.mark.parametrize(
    ("workflow", "affected"),
    [
        (desired_output_workflow_spec("electronic_band_structure").to_dict(), True),
        (CUSTOM_HSE_BAND, True),
        (desired_output_workflow_spec("electronic_dos").to_dict(), False),
        (desired_output_workflow_spec("relaxed_energy").to_dict(), False),
        (desired_output_workflow_spec("phonons").to_dict(), False),
        ({"stages": [{"stage_type": "static", "theory": "pbe"}, {"stage_type": "band_structure", "theory": "pbe"}]}, False),
    ],
)
def test_bmd_check_is_parity_critical_only_for_hse06_band_structures(workflow, affected):
    assert ("bmd-agent" in parity_critical_packages_for_workflow(workflow)) is affected


def _versions(**overrides):
    base = {name: "1.0" for name in runtime_environment.PARITY_CRITICAL_PACKAGES}
    return lambda name: {**base, "bmd-agent": "0.2.0", **overrides}.get(name)


def test_preparation_requires_exactly_bmd_agent_0_2_0():
    packages = parity_critical_packages_for_workflow(CUSTOM_HSE_BAND)
    assert prepared_runtime_parity(version_lookup=_versions(), packages=packages)["packages"]["bmd-agent"] == "0.2.0"
    with pytest.raises(RuntimeParityError, match="bmd-agent 0.1.0 is installed; this BMD Compute requires 0.2.0"):
        prepared_runtime_parity(version_lookup=_versions(**{"bmd-agent": "0.1.0"}), packages=packages)
    with pytest.raises(RuntimeParityError, match="missing parity-critical package"):
        prepared_runtime_parity(version_lookup=_versions(**{"bmd-agent": None}), packages=packages)


def test_power_execution_refuses_a_missing_or_different_bmd_check():
    packages = parity_critical_packages_for_workflow(CUSTOM_HSE_BAND)
    prepared = prepared_runtime_parity(version_lookup=_versions(), packages=packages)
    runtime = {name: _versions()(name) for name in packages}

    assert runtime_parity_problems(prepared, runtime, packages) == []
    assert runtime_parity_problems(prepared, {**runtime, "bmd-agent": None}, packages)
    assert runtime_parity_problems(prepared, {**runtime, "bmd-agent": "0.2.1"}, packages)
    stale = {**prepared, "packages": {**prepared["packages"], "bmd-agent": "0.1.0"}}
    assert any("requires 0.2.0" in problem for problem in runtime_parity_problems(stale, runtime, packages))


@pytest.mark.parametrize("installed", [None, "0.1.0"])
def test_band_structure_planning_is_refused_without_a_compatible_bmd_check(monkeypatch, installed):
    real = runtime_environment._default_version_lookup

    def lookup(name):
        if name == "bmd-agent":
            if installed is None:
                raise LookupError(name)
            return installed
        return real(name)

    monkeypatch.setattr(runtime_environment, "_default_version_lookup", lookup)
    with pytest.raises(CalculationValidationError):
        _plan("electronic_band_structure")
    with pytest.raises(CalculationValidationError):
        _plan(custom=CUSTOM_HSE_BAND)
    # Unaffected workflows do not depend on BMD Check.
    assert "bmd-agent" not in _plan("electronic_dos").submission_spec["runtime_parity"]["packages"]


# ------------------------------------------------------------ plan identity


def test_plan_describes_the_rule_and_is_stable():
    plan = _plan("electronic_band_structure")
    material = stage_input_material(plan_stage_previews(plan))

    assert all("conditional_settings" not in item for item in material[:3])
    assert material[3]["conditional_settings"] == hse_band_sigma.conditional_settings_payload()
    assert material[3]["incar"]["SIGMA"] == 0.01
    assert _digest(plan) == _digest(_plan("electronic_band_structure"))
    assert plan.submission_spec["runtime_parity"]["packages"]["bmd-agent"] == "0.2.0"
    # The preview states the runtime-dependent setting beside the default.
    incar_text = plan.generated_inputs["incar"]
    notes = [line for line in incar_text.splitlines() if "BMD Check C2" in line]
    assert notes == [hse_band_sigma.preview_note(3)]
    assert "0.2 eV" in notes[0] and "Stage 3 HSE06 static" in notes[0]


def test_the_rule_version_is_part_of_plan_identity(monkeypatch):
    before = _digest(_plan("electronic_band_structure"))
    monkeypatch.setattr(hse_band_sigma, "POLICY_VERSION", 2)
    assert _digest(_plan("electronic_band_structure")) != before


@pytest.mark.parametrize("desired_output", ["relaxed_energy", "relaxed_structure", "electronic_dos", "phonons"])
def test_other_workflows_carry_no_rule_note_or_bmd_check(desired_output):
    plan = _plan(desired_output)
    assert all("conditional_settings" not in item for item in stage_input_material(plan_stage_previews(plan)))
    assert "BMD Check C2" not in plan.generated_inputs["incar"]
    assert "bmd-agent" not in plan.submission_spec["runtime_parity"]["packages"]


def test_real_atomate2_maker_path_copies_oszicar_and_writes_consistent_inputs(c2, monkeypatch, tmp_path):
    """The HSEBSMaker steps that run before VASP: copy the static, write inputs."""

    import gzip

    from atomate2.vasp.files import copy_vasp_outputs, write_vasp_input_set
    from pymatgen.io.vasp import Incar
    from pymatgen.io.vasp.sets import VaspInputSet

    calls = c2("metallic")
    # Stand-in files are not a parseable VASP run (see ``runtime_inputs``).
    monkeypatch.setattr(VaspInputSet, "_set_previous", lambda self, prev_dir=None: None)
    flow = build_atomate2_flow_for_workflow_spec(
        SI, planned(SI, desired_output_workflow_spec("electronic_band_structure")), resources={"ntasks": 24}
    )
    maker = flow.jobs[3].function.__self__
    static_dir = tmp_path / "stage_03"
    static_dir.mkdir()
    for name, text in {
        "INCAR": "ISMEAR = 0\n",
        "OUTCAR": "stand-in\n",
        "CONTCAR": str(SI.to(fmt="poscar")),
        "vasprun.xml": "<modeling></modeling>\n",
        "OSZICAR": "stand-in\n",
        "CHGCAR": "stand-in\n",
        "POTCAR.spec": "Si\n",
    }.items():
        (static_dir / f"{name}.gz").write_bytes(gzip.compress(text.encode()))
    band_dir = tmp_path / "stage_04"
    band_dir.mkdir()
    monkeypatch.chdir(band_dir)

    copy_vasp_outputs(static_dir, **maker.copy_vasp_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_vasp_input_set(SI, maker.input_set_generator, from_prev=True, potcar_spec=True)

    # OSZICAR (for C1 convergence) arrives with CHGCAR, decompressed.
    assert (band_dir / "OSZICAR").read_text() == "stand-in\n"
    assert (band_dir / "CHGCAR").is_file()
    assert calls["directory"] == "." and calls["stage_index"] == 3
    written = Incar.from_file(band_dir / "INCAR")
    record = json.loads((band_dir / hse_band_sigma.PROVENANCE_FILENAME).read_text())
    assert (written["ISMEAR"], written["SIGMA"]) == (0, 0.2)
    assert (record["ISMEAR"], record["initial_SIGMA_ev"], record["classification"]) == (0, 0.2, "metallic")
