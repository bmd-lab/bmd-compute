"""BMD Standard Geometry Relaxation Protocol v1 (``bmd_compute.geometry_relaxation/1``).

Every test plans against SYNTHETIC POTCAR ENMAX metadata (see ``conftest``);
no value here is a production ENMAX.
"""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

import main
import test_automatic_soc as soc_cases
import test_input_reference_contract as reference_cases
from backend.calculations import geometry_relaxation as gr
from backend.calculations.custodian_policy import LARGE_SIGMA_HANDLER
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    canonical_custom_workflow_spec,
    desired_output_workflow_spec,
    validate_user_workflow_spec,
    validate_workflow_spec,
)
from backend.parser import parse_structure
from backend.remote_result_parser import (
    GEOMETRY_RELAXATION_VASPRUN_KEY,
    _geometry_relaxation_payload,
    geometry_relaxation_diagnostic,
)
from backend.results import results_location_from_submission_spec
from backend.workflows import build_atomate2_flow_for_workflow_spec
from compute_api.plan_digest import compute_plan_digest
from compute_api.projection import plan_stage_previews
from conftest import planned, synthetic_enmax_metadata, synthetic_potcar_loader, use_enmax_metadata

SI_POSCAR = reference_cases.SI_POSCAR
NIO_POSCAR = reference_cases.NIO_POSCAR
BI2SE3_POSCAR = soc_cases.BI2SE3_POSCAR
AL_POSCAR = Poscar(
    Structure(Lattice.cubic(4.04), ["Al"] * 4, [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
).get_str()
LI2O_POSCAR = Poscar(
    Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.61), ["Li", "O"], [[0.25, 0.25, 0.25], [0, 0, 0]]
    ).get_primitive_structure()
).get_str()
CUMN_POSCAR = Poscar(Structure(Lattice.cubic(2.95), ["Cu", "Mn"], [[0, 0, 0], [0.5, 0.5, 0.5]])).get_str()
MANAGED_RELAXING_OUTPUTS = (
    "relaxed_structure",
    "relaxed_energy",
    "electronic_dos",
    "electronic_band_structure",
    "phonons",
)


def _plan(structure=SI_POSCAR, desired_output=None, custom=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main.plan_calculation_request(
            structure_text=structure,
            fmt="poscar",
            desired_output=desired_output,
            custom_workflow=custom,
        )


def _previews(plan):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return plan_stage_previews(plan)


def _stage(stage_type, theory="pbe", modifiers=(), options=None):
    return {"stage_type": stage_type, "theory": theory, "modifiers": list(modifiers), "options": options or {}}


def _cutoff(plan, index):
    return plan.submission_spec["flow_spec"]["workflow_spec"]["stages"][index]["options"]["plane_wave_cutoff"]


# --------------------------------------------------------------- cutoff rule


@pytest.mark.parametrize(
    ("enmax", "encut"),
    [
        ([245.345], 350),  # 318.9 -> 350
        ([400.0], 550),  # 520 -> 550
        ([499.034], 650),  # 648.7 -> 650
        ([500.0], 650),  # exactly 650: no extra increment
        ([250.0], 350),  # 325 -> 350
        ([230.283], 300),  # 299.4 -> 300
        ([1000.0 / 1.3], 1000),  # 1000.0000...: decimal arithmetic, no spurious bump
        ([200.0, 400.0, 245.0], 550),  # the highest ENMAX decides
    ],
)
def test_cutoff_is_1p3_times_the_highest_enmax_rounded_up_to_50_ev(enmax, encut):
    assert gr.resolved_encut(enmax) == encut


@pytest.mark.parametrize("bad", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_cutoff_arithmetic_refuses_unusable_enmax(bad):
    with pytest.raises(gr.PlaneWaveCutoffError):
        gr.resolved_encut(bad)


# Representative POTCAR selections of Compute's executable generators, with
# distinct SYNTHETIC ENMAX values so the deciding dataset is visible.
_SELECTION_ENMAX = {"Si": 245.0, "Al": 240.0, "Ni": 270.0, "O": 400.0, "Li_sv": 499.0, "Bi_d": 243.0, "Se": 212.0,
                    "Cu": 295.0, "Mn_pv": 270.0}


@pytest.mark.parametrize(
    ("poscar", "symbols", "encut"),
    [
        (SI_POSCAR, ["Si"], 350),
        (AL_POSCAR, ["Al"], 350),
        (NIO_POSCAR, ["Ni", "O"], 550),
        (LI2O_POSCAR, ["Li_sv", "O"], 650),
        (BI2SE3_POSCAR, ["Bi_d", "Se"], 350),
        (CUMN_POSCAR, ["Cu", "Mn_pv"], 400),
    ],
    ids=["Si", "Al", "NiO", "Li2O", "Bi2Se3", "CuMn"],
)
def test_cutoff_uses_the_potcars_the_executable_generators_select(monkeypatch, tmp_path, poscar, symbols, encut):
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", synthetic_enmax_metadata(_SELECTION_ENMAX))
    plan = _plan(poscar, desired_output="relaxed_structure")
    previews = _previews(plan)
    for index in (0, 1):
        record = _cutoff(plan, index)
        assert [item["symbol"] for item in record["potcars"]] == symbols
        assert str(previews[index]["input_set"].potcar).split() == symbols
        assert [item["ENMAX"] for item in record["potcars"]] == [_SELECTION_ENMAX[symbol] for symbol in symbols]
        assert record["ENCUT"] == encut
        assert record["rule"] == {"id": gr.CUTOFF_RULE_ID, "version": gr.CUTOFF_RULE_VERSION}
        assert (record["safety_factor"], record["increment_ev"]) == (1.3, 50)
        assert record["metadata"]["synthetic"] is True
        assert previews[index]["input_set"].incar["ENCUT"] == encut
    # The POTCAR identity record shows the same ENMAX values.
    potcar_stages = plan.submission_spec["potcar"]["stages"]
    assert potcar_stages[0]["plane_wave_cutoff"]["enmax_ev"] == {s: _SELECTION_ENMAX[s] for s in symbols}


# ------------------------------------------------------- metadata safeguards


def test_missing_metadata_fails_planning_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(gr, "metadata_path", lambda functional: tmp_path / "absent.json")
    with pytest.raises(CalculationValidationError) as excinfo:
        _plan(desired_output="relaxed_structure")
    assert excinfo.value.diagnostic["code"] == "potcar_enmax_metadata_unavailable"
    # Workflows without a relaxation or phonon stage do not need it.
    assert _plan(desired_output="energy_only").workflow_spec.stages[0].stage_type is StageType.STATIC


def test_a_symbol_missing_from_the_metadata_fails_clearly(monkeypatch, tmp_path):
    metadata = synthetic_enmax_metadata()
    del metadata["potcars"]["O"]
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", metadata)
    with pytest.raises(CalculationValidationError) as excinfo:
        _plan(NIO_POSCAR, desired_output="relaxed_structure")
    assert excinfo.value.diagnostic["code"] == "potcar_enmax_missing"
    assert excinfo.value.diagnostic["missing_symbols"] == ["O"]


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda m: m.update(schema="other"),
        lambda m: m.update(functional="PBE_54"),
        lambda m: m["potcars"]["Si"].update(ENMAX=-1.0),
        lambda m: m["potcars"]["Si"].update(ENMAX=True),
        lambda m: m["potcars"]["Si"].pop("sha256"),
        lambda m: m["potcars"]["Si"].update(TITEL=""),
    ],
)
def test_inconsistent_metadata_fails_clearly(monkeypatch, tmp_path, corrupt):
    metadata = synthetic_enmax_metadata()
    corrupt(metadata)
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", metadata)
    with pytest.raises(CalculationValidationError) as excinfo:
        _plan(desired_output="relaxed_structure")
    assert excinfo.value.diagnostic["code"] == "potcar_enmax_metadata_invalid"


def test_a_tampered_cutoff_record_is_refused():
    workflow = planned(parse_structure(SI_POSCAR), desired_output_workflow_spec("relaxed_structure"))
    stages = [stage.to_dict() for stage in workflow.stages]
    stages[1]["options"]["plane_wave_cutoff"]["ENCUT"] = 400
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(WorkflowSpec.from_dict({"stages": stages}))
    assert excinfo.value.diagnostic["code"] == "plane_wave_cutoff_record_invalid"


def test_a_protocol_stage_is_never_built_without_its_cutoff():
    with pytest.raises(CalculationValidationError) as excinfo:
        build_atomate2_flow_for_workflow_spec(parse_structure(SI_POSCAR), desired_output_workflow_spec("relaxed_structure"))
    assert excinfo.value.diagnostic["code"] == "plane_wave_cutoff_missing"


def test_metadata_builder_reads_both_library_subdirectories_and_refuses_divergent_copies(tmp_path, monkeypatch):
    import pymatgen.io.vasp.inputs as vasp_inputs

    from backend.calculations import potcar_enmax_metadata as builder

    def write(subdirectory, symbol, enmax, titel=None):
        directory = tmp_path / subdirectory / symbol
        directory.mkdir(parents=True)
        (directory / "POTCAR").write_text(json.dumps({"symbol": symbol, "ENMAX": enmax, "TITEL": titel or f"T {symbol}"}))

    def fake_from_file(path):
        values = json.loads(open(path).read())
        return SimpleNamespace(symbol=values["symbol"], data=open(path).read(),
                               keywords={"TITEL": values["TITEL"], "ENMAX": values["ENMAX"]})

    monkeypatch.setattr(vasp_inputs.PotcarSingle, "from_file", staticmethod(fake_from_file))
    write("POT_PAW_PBE_64", "Si", 245.345)
    write("PBE_64", "Si", 245.345)
    write("PBE_64", "O", 400.0)
    metadata = builder.build_enmax_metadata(tmp_path, symbols=["Si", "O"])
    assert metadata["synthetic"] is False
    assert metadata["potcars"]["Si"]["ENMAX"] == 245.345

    def library_paths(symbol):
        # Relative paths use the host's separator (POSIX on POWER, where the
        # production metadata is generated); compare them separator-independently.
        return sorted(path.replace("\\", "/") for path in metadata["potcars"][symbol]["library_paths"])

    assert library_paths("Si") == ["PBE_64/Si/POTCAR", "POT_PAW_PBE_64/Si/POTCAR"]
    assert library_paths("O") == ["PBE_64/O/POTCAR"]
    assert builder.validate_metadata(metadata, functional="PBE_64") == []

    write("POT_PAW_PBE_64", "O", 401.0)
    with pytest.raises(builder.PotcarMetadataError, match="copies differ"):
        builder.build_enmax_metadata(tmp_path, symbols=["Si", "O"])
    with pytest.raises(builder.PotcarMetadataError, match="no POTCAR"):
        builder.build_enmax_metadata(tmp_path, symbols=["Si", "Li_sv"])


def test_the_production_metadata_file_is_not_synthetic_when_present():
    from backend.calculations.potcar_enmax_metadata import (
        generator_selectable_symbols,
        metadata_path,
        validate_metadata,
    )

    path = metadata_path("PBE_64")
    if not path.exists():
        pytest.skip("production POTCAR ENMAX metadata has not been generated from POWER yet")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["synthetic"] is False
    assert validate_metadata(metadata, functional="PBE_64") == []
    # Exactly the datasets Compute's executable generators can select, each one
    # present where atomate2/pymatgen read it at run time, and header values only.
    assert set(metadata["potcars"]) == set(generator_selectable_symbols())
    assert metadata["source"]["library_root"] == "/bmd-db/potcars"
    for symbol, entry in metadata["potcars"].items():
        assert set(entry) == {"TITEL", "ENMAX", "sha256", "library_paths"}, symbol
        assert f"POT_PAW_PBE_64/{symbol}/POTCAR" in entry["library_paths"], symbol
        assert entry["TITEL"].startswith("PAW_PBE ") and symbol in entry["TITEL"].split(), symbol


@pytest.mark.parametrize(
    ("poscar", "symbols", "encut"),
    [
        (SI_POSCAR, ["Si"], 350),
        (AL_POSCAR, ["Al"], 350),
        (NIO_POSCAR, ["Ni", "O"], 550),
        (LI2O_POSCAR, ["Li_sv", "O"], 650),
        (BI2SE3_POSCAR, ["Bi_d", "Se"], 350),
        (CUMN_POSCAR, ["Cu", "Mn_pv"], 400),
    ],
    ids=["Si", "Al", "NiO", "Li2O", "Bi2Se3", "CuMn"],
)
def test_production_metadata_resolves_the_representative_cutoffs(monkeypatch, poscar, symbols, encut):
    from backend.calculations import potcar_enmax_metadata

    path = potcar_enmax_metadata.metadata_path("PBE_64")
    if not path.exists():
        pytest.skip("production POTCAR ENMAX metadata has not been generated from POWER yet")
    monkeypatch.setattr(gr, "metadata_path", potcar_enmax_metadata.metadata_path)
    plan = _plan(poscar, desired_output="relaxed_energy")
    first, final, static = (preview["input_set"].incar for preview in _previews(plan))
    record = _cutoff(plan, 1)
    assert record["metadata"]["synthetic"] is False
    assert record["metadata"]["digest"] == potcar_enmax_metadata.metadata_digest(json.loads(path.read_text()))
    assert [item["symbol"] for item in record["potcars"]] == symbols
    assert _cutoff(plan, 0) == record
    assert first["ENCUT"] == final["ENCUT"] == record["ENCUT"] == encut
    assert static["ENCUT"] == max(620, encut)


# ------------------------------------------------------ two-pass protocol


@pytest.mark.parametrize("desired_output", MANAGED_RELAXING_OUTPUTS)
def test_every_managed_relaxation_is_the_two_pass_protocol(desired_output):
    plan = _plan(desired_output=desired_output)
    stages = plan.workflow_spec.stages
    relaxes = [index for index, stage in enumerate(stages) if stage.stage_type is StageType.RELAX]
    assert relaxes == [0, 1]
    assert [gr.stage_protocol_pass(stages[index]) for index in relaxes] == [1, 2]
    previews = _previews(plan)
    first, final = (previews[index]["input_set"].incar for index in relaxes)
    assert first["ENCUT"] == final["ENCUT"] == _cutoff(plan, 0)["ENCUT"] == _cutoff(plan, 1)["ENCUT"]
    for incar in (first, final):
        for key, value in {"ISMEAR": 0, "SIGMA": 0.2, "IBRION": 2, "ISIF": 3, "PREC": "Accurate",
                           "LREAL": False, "LASPH": True}.items():
            assert incar[key] == value, key
        assert "ADDGRID" not in incar
    assert (first["EDIFF"], first["EDIFFG"]) == (1e-6, -0.01)
    assert (final["EDIFF"], final["EDIFFG"]) == (1e-7, -0.001)


def test_second_pass_starts_from_the_first_pass_structure_with_a_new_basis():
    plan = _plan(desired_output="relaxed_structure")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    first, second = flow.jobs
    reference = second.function_args[0]
    assert reference.uuid == first.uuid and reference.attributes == (("a", "structure"),)
    # No previous directory: no WAVECAR or CHGCAR, so VASP builds a new basis.
    assert "prev_dir" not in second.function_kwargs or second.function_kwargs["prev_dir"] is None
    assert not second.maker.copy_vasp_kwargs


def test_no_large_sigma_handler_in_any_relaxation():
    plan = _plan(desired_output="relaxed_energy")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    names = [[f"{type(h).__module__}.{type(h).__name__}" for h in job.maker.run_vasp_kwargs["handlers"]] for job in flow.jobs]
    assert LARGE_SIGMA_HANDLER not in names[0] and LARGE_SIGMA_HANDLER not in names[1]
    assert LARGE_SIGMA_HANDLER in names[2]  # the static stage is unchanged


def test_modifiers_are_carried_by_both_passes():
    nio = _plan(NIO_POSCAR, desired_output="relaxed_energy")
    first, final, static = (preview["input_set"].incar for preview in _previews(nio))
    for incar in (first, final, static):
        assert incar["ISPIN"] == 2 and incar["LDAU"] is True
    assert first["LDAUU"] == final["LDAUU"] == static["LDAUU"]
    bi2se3 = _plan(BI2SE3_POSCAR, desired_output="relaxed_structure")
    assert all(preview["input_set"].incar["IVDW"] == 12 for preview in _previews(bi2se3))


def test_the_final_pass_inherits_the_relaxed_magnetic_moments():
    # atomate2 hands the relaxed structure on; its magnetic moments (from the
    # first pass's OUTCAR) are site properties the generator uses for MAGMOM.
    plan = _plan(NIO_POSCAR, desired_output="relaxed_structure")
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    generator = flow.jobs[1].maker.input_set_generator
    relaxed = plan.structure.copy()
    moments = [1.7 if site.specie.symbol == "Ni" else 0.1 for site in relaxed]
    relaxed.add_site_property("magmom", moments)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        incar = generator.get_input_set(relaxed, potcar_spec=True).incar
    assert incar["MAGMOM"] == moments
    assert incar["ENCUT"] == _cutoff(plan, 1)["ENCUT"]


# ------------------------------------------------------------ Custom workflows


def test_custom_relaxations_keep_their_explicit_stages():
    single = _plan(custom={"stages": [_stage("relax")]})
    assert len(single.workflow_spec.stages) == 1
    assert gr.stage_protocol_pass(single.workflow_spec.stages[0]) is None
    incar = _previews(single)[0]["input_set"].incar
    assert (incar["EDIFF"], incar["EDIFFG"]) == (1e-6, -0.01)
    assert incar["ENCUT"] == _cutoff(single, 0)["ENCUT"]
    # An existing two-relaxation Custom workflow is not reinterpreted.
    double = _plan(custom={"stages": [_stage("relax"), _stage("relax")]})
    assert [(i["EDIFF"], i["EDIFFG"]) for i in (p["input_set"].incar for p in _previews(double))] == [
        (1e-6, -0.01),
        (1e-6, -0.01),
    ]


def test_custom_workflows_can_request_the_canonical_relaxation():
    stages = [_stage("relax", options=gr.protocol_option(1)), _stage("relax", options=gr.protocol_option(2)),
              _stage("static")]
    plan = _plan(custom={"stages": stages})
    first, final, static = (p["input_set"].incar for p in _previews(plan))
    assert (first["EDIFFG"], final["EDIFFG"]) == (-0.01, -0.001)
    assert static["ENCUT"] == 620  # the static stage keeps its own methodology
    assert "plane_wave_cutoff" not in plan.workflow_spec.stages[2].options
    from backend.calculations.registry import calculation_form_options

    recipes = {item["value"]: item for item in calculation_form_options()["recipes"]}
    assert [stage["options"]["geometry_relaxation"]["pass"] for stage in recipes["geometry_relaxation"]["workflow_spec"]["stages"]] == [1, 2]


@pytest.mark.parametrize(
    "stages",
    [
        [_stage("relax", options=gr.protocol_option(1))],
        [_stage("relax", options=gr.protocol_option(2))],
        [_stage("relax", options=gr.protocol_option(1)), _stage("static")],
        [_stage("relax", options=gr.protocol_option(1)), _stage("relax", "pbe", ["spin_polarized"], gr.protocol_option(2))],
        [_stage("static", options=gr.protocol_option(1))],
        [_stage("relax", options={"geometry_relaxation": {"protocol": gr.PROTOCOL_ID, "version": 2, "pass": 1}})],
    ],
    ids=["pass1_alone", "pass2_alone", "pass1_then_static", "mismatched_modifiers", "static_marked", "unknown_version"],
)
def test_malformed_custom_protocol_requests_are_refused(stages):
    with pytest.raises(CalculationValidationError):
        validate_user_workflow_spec(WorkflowSpec.from_dict({"stages": stages}))


def test_a_client_cannot_supply_a_cutoff_and_an_echoed_record_is_replaced():
    plan = _plan(desired_output="relaxed_structure")
    echoed = WorkflowSpec.from_dict(plan.submission_spec["flow_spec"]["workflow_spec"])
    stages = [stage.to_dict() for stage in echoed.stages]
    stages[0]["options"]["plane_wave_cutoff"]["ENCUT"] = 300  # tampered in transit
    with pytest.raises(CalculationValidationError):
        validate_user_workflow_spec(WorkflowSpec.from_dict({"stages": stages}))
    canonical = canonical_custom_workflow_spec(WorkflowSpec.from_dict({"stages": stages}))
    assert all("plane_wave_cutoff" not in stage.options for stage in canonical.stages)
    assert [gr.stage_protocol_pass(stage) for stage in canonical.stages] == [1, 2]


# ------------------------------------------------------------- phonons


def test_custom_nio_phonons_use_plain_pbe_forces_at_the_relaxation_cutoff():
    # D1 regression: the relaxation was plain PBE but pymatgen added U(Ni) to the forces.
    plan = _plan(NIO_POSCAR, custom={"stages": [_stage("relax"), _stage("phonon_forces")]})
    relax, forces = (preview["input_set"].incar for preview in _previews(plan))
    for key in ("LDAU", "LDAUU", "LDAUL", "LDAUJ", "LDAUTYPE", "LMAXMIX"):
        assert key not in relax and key not in forces, key
    assert forces["ENCUT"] == relax["ENCUT"] == _cutoff(plan, 0)["ENCUT"] == _cutoff(plan, 1)["ENCUT"]
    assert (relax["ISMEAR"], relax["SIGMA"]) == (forces["ISMEAR"], forces["SIGMA"]) == (0, 0.2)  # D2
    assert "ADDGRID" not in relax and "ADDGRID" not in forces  # D3
    assert (relax["EDIFF"], relax["EDIFFG"]) == (1e-7, -0.001)


def test_phonon_forces_share_the_final_relaxation_cutoff_and_smearing_policy():
    plan = _plan(desired_output="phonons")
    previews = _previews(plan)
    final, forces = previews[1]["input_set"].incar, previews[2]["input_set"].incar
    assert forces["ENCUT"] == final["ENCUT"]
    flow = build_atomate2_flow_for_workflow_spec(plan.structure, plan.workflow_spec, resources={"ntasks": 24})
    relax_handlers = [f"{type(h).__module__}.{type(h).__name__}" for h in flow.jobs[1].maker.run_vasp_kwargs["handlers"]]
    phonon_jobs = {job.name: job for job in flow.jobs[2].jobs}
    force_maker = phonon_jobs["run_phonon_displacements"].function_kwargs["phonon_maker"]
    force_handlers = [f"{type(h).__module__}.{type(h).__name__}" for h in force_maker.run_vasp_kwargs["handlers"]]
    assert LARGE_SIGMA_HANDLER not in relax_handlers and LARGE_SIGMA_HANDLER not in force_handlers
    assert force_maker.input_set_generator.user_incar_settings["ENCUT"] == final["ENCUT"]


def test_a_phonon_cutoff_differing_from_its_relaxation_is_refused():
    plan = _plan(desired_output="phonons")
    stages = [stage.to_dict() for stage in plan.workflow_spec.stages]
    stages[2]["options"]["plane_wave_cutoff"] = json.loads(json.dumps(stages[0]["options"]["plane_wave_cutoff"]))
    stages[2]["options"]["plane_wave_cutoff"]["potcars"][0]["sha256"] = "0" * 64
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(WorkflowSpec.from_dict({"stages": stages, "recipe": "phonons"}))
    assert excinfo.value.diagnostic["code"] == "phonon_cutoff_mismatch"


# ------------------------------------------------- downstream stages unchanged


def test_downstream_stage_inputs_do_not_depend_on_the_relaxation_protocol():
    # The same static, DOS and band-structure inputs as a workflow whose
    # relaxation carries no protocol (the historical form).
    for output in ("relaxed_energy", "electronic_dos", "electronic_band_structure"):
        plan = _plan(desired_output=output)
        protocol = _previews(plan)[2:]
        historical = WorkflowSpec(
            [StageSpec(StageType.RELAX, Theory.PBE), *desired_output_workflow_spec(output).stages[2:]],
        )
        from backend.generated_inputs import generated_input_stage_previews

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy = generated_input_stage_previews(plan.structure, historical, resources=plan.execution_resources)[1:]
        for new, old in zip(protocol, legacy):
            assert dict(new["input_set"].incar) == dict(old["input_set"].incar), output
            assert str(new["input_set"].kpoints) == str(old["input_set"].kpoints), output


# --------------------------------------------------- plan identity and parity


def test_plan_digest_follows_the_cutoff_and_the_protocol(monkeypatch, tmp_path):
    plan = _plan(desired_output="relaxed_structure")
    digest = compute_plan_digest(structure=plan.structure, submission_spec=plan.submission_spec, stage_previews=_previews(plan))
    again = _plan(desired_output="relaxed_structure")
    assert digest == compute_plan_digest(structure=again.structure, submission_spec=again.submission_spec, stage_previews=_previews(again))
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", synthetic_enmax_metadata({"Si": 430.0}))
    other = _plan(desired_output="relaxed_structure")
    assert _cutoff(other, 0)["ENCUT"] == 600
    assert digest != compute_plan_digest(structure=other.structure, submission_spec=other.submission_spec, stage_previews=_previews(other))
    monkeypatch.undo()
    from compute_api import plan_digest

    material = plan_digest.stage_input_material(_previews(plan))
    assert material[0]["geometry_relaxation"]["protocol"] == {"id": gr.PROTOCOL_ID, "version": gr.PROTOCOL_VERSION}
    assert material[1]["geometry_relaxation"]["pass"] == 2
    assert material[1]["geometry_relaxation"]["custodian_excluded_handlers"] == [LARGE_SIGMA_HANDLER]


def test_browser_and_api_freeze_the_same_cutoffs():
    from starlette.requests import Request

    request = Request({"type": "http", "method": "POST", "path": "/build-workflow", "headers": []})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = main.build_workflow(
            request, structure=SI_POSCAR, fmt="poscar", purpose=None, theory=None, modifiers=None,
            cpus=None, memory_gb=None, walltime=None, queue=None, workflow_spec_json=None,
            workflow="phonons", method=None,
        )
    assert response.status_code == 200
    api = _plan(desired_output="phonons")
    assert response.context["submission_spec"]["flow_spec"]["workflow_spec"] == api.submission_spec["flow_spec"]["workflow_spec"]
    assert response.context["generated_inputs"]["incar"] == api.generated_inputs["incar"]


# ------------------------------------------------------ historical records


def test_historical_workflows_without_cutoff_records_build_as_before():
    structure = parse_structure(SI_POSCAR)
    historical = WorkflowSpec([StageSpec(StageType.RELAX, Theory.PBE), StageSpec(StageType.PHONON_FORCES, Theory.PBE)])
    flow = build_atomate2_flow_for_workflow_spec(structure, historical, resources={"ntasks": 24})
    relax = flow.jobs[0].maker.input_set_generator.user_incar_settings
    assert (relax["ENCUT"], relax["EDIFF"], relax["EDIFFG"], relax["ADDGRID"]) == (700, 1e-7, -1e-3, True)
    phonon_jobs = {job.name: job for job in flow.jobs[1].jobs}
    force = phonon_jobs["run_phonon_displacements"].function_kwargs["phonon_maker"].input_set_generator
    assert force.user_incar_settings["ENCUT"] == 700
    assert gr.runtime_potcar_check({"workflow_spec": historical.to_dict()})["status"] == "not_applicable"


# ------------------------------------------------ run-time POTCAR check


def test_runtime_check_compares_the_recorded_enmax_with_the_installed_potcars(tmp_path):
    from backend.runtime_environment import RuntimeParityError, enforce_runtime_environment
    from test_runtime_environment import Settings, lookup, prepared, spec_for, PRODUCTION

    plan = _plan(desired_output="relaxed_structure")
    flow_spec = plan.submission_spec["flow_spec"]
    assert gr.runtime_potcar_check(flow_spec, potcar_loader=synthetic_potcar_loader)["status"] == "passed"

    def changed(symbol, functional):
        potcar = synthetic_potcar_loader(symbol, functional)
        potcar.keywords["ENMAX"] = 300.0
        return potcar

    result = gr.runtime_potcar_check(flow_spec, potcar_loader=changed)
    assert result["status"] == "failed" and "ENMAX" in result["problems"][0]

    spec = spec_for(tmp_path, prepared())
    spec["flow_spec"] = flow_spec
    with pytest.raises(RuntimeParityError, match="POTCAR Si differs"):
        enforce_runtime_environment(spec, version_lookup=lookup(PRODUCTION), settings=Settings(), environ={},
                                    potcar_loader=changed)
    record = json.loads((tmp_path / "runtime_environment.json").read_text())
    assert record["plane_wave_cutoff_potcars"]["status"] == "failed"


# --------------------------------------------------------- stress diagnostic


def _relax_vasprun(stress, initial_a=5.40, final_a=5.43, steps=3, converged=True):
    def silicon(a):
        return Structure(Lattice.cubic(a), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

    return SimpleNamespace(
        ionic_steps=[{"stress": stress}] * steps,
        converged_ionic=converged,
        converged_electronic=True,
        initial_structure=silicon(initial_a),
        final_structure=silicon(final_a),
    )


def test_stress_diagnostic_reports_pressure_largest_component_and_cell_change():
    stress = [[-0.6, 0.1, 0.0], [0.1, -0.9, 0.0], [0.0, 0.0, -1.2]]
    payload = geometry_relaxation_diagnostic(_relax_vasprun(stress), stage_index=2)
    residual = payload["residual_stress"]
    assert residual["stress_tensor_kbar"] == stress
    assert residual["hydrostatic_pressure_kbar"] == pytest.approx(-0.9)
    assert residual["max_abs_component_kbar"] == pytest.approx(1.2)
    assert residual["assessment"] == "within_reference"
    assert "never a failure" in residual["assessment_basis"]
    assert payload["vasp_reported"]["converged_ionic"] is True
    assert "EDIFFG force criterion" in payload["vasp_reported"]["criterion"]
    assert payload["ionic_steps"] == 3 and payload["stage_index"] == 2
    change = payload["cell_change"]["change_percent"]
    assert change["a"] == pytest.approx(100 * (5.43 - 5.40) / 5.40, abs=1e-4)
    assert change["volume"] == pytest.approx(100 * ((5.43 / 5.40) ** 3 - 1), abs=1e-3)
    assert payload["cell_change"]["volume_change_review"] is True  # 1.7 % > 0.5 %


@pytest.mark.parametrize(
    ("stress", "level"),
    [
        ([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], "within_reference"),
        ([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]], "review"),
        ([[9.0, 0, 0], [0, 0, 0], [0, 0, 0]], "large"),
    ],
)
def test_stress_assessment_levels_are_diagnostic_only(stress, level):
    payload = geometry_relaxation_diagnostic(_relax_vasprun(stress, converged=True))
    assert payload["residual_stress"]["assessment"] == level
    assert payload["status"] == "available"


def test_diagnostic_is_recorded_only_for_protocol_relaxations(tmp_path):
    plan = _plan(desired_output="relaxed_energy")
    spec = json.loads(json.dumps(plan.submission_spec))
    spec["paths"]["run_dir"] = "/bmd-db/guest/flows/run"
    spec["paths"].pop("result_dir", None)
    location = results_location_from_submission_spec(spec)
    assert location["optional_result_file_paths"][GEOMETRY_RELAXATION_VASPRUN_KEY] == (
        "/bmd-db/guest/flows/run/stage_02/vasprun.xml"
    )
    workflow = spec["flow_spec"]["workflow_spec"]
    vasprun = _relax_vasprun([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    payload = _geometry_relaxation_payload(
        workflow, vasprun=None, local_paths={GEOMETRY_RELAXATION_VASPRUN_KEY: "x"},
        vasprun_class=lambda *args, **kwargs: vasprun,
    )
    assert payload["status"] == "available" and payload["stage_index"] == 2
    # Historical workflows (no cutoff record) get no diagnostic.
    historical = {"stages": [{"stage_type": "relax", "options": {}}, {"stage_type": "static", "options": {}}]}
    assert _geometry_relaxation_payload(historical, vasprun=vasprun, local_paths={}, vasprun_class=None) is None
    single = _plan(desired_output="relaxed_structure")
    spec = json.loads(json.dumps(single.submission_spec))
    spec["paths"]["run_dir"] = "/bmd-db/guest/flows/run"
    spec["paths"].pop("result_dir", None)
    assert GEOMETRY_RELAXATION_VASPRUN_KEY not in results_location_from_submission_spec(spec)["optional_result_file_paths"]


# ------------------------------------------- static cutoff after a relaxation


def _encuts(plan):
    return [(p["stage_spec"].stage_type.value, p["input_set"].incar.get("ENCUT")) for p in _previews(plan)]


def test_static_never_uses_a_lower_cutoff_than_its_relaxation(monkeypatch, tmp_path):
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", synthetic_enmax_metadata({"Li_sv": 499.0, "O": 400.0}))
    # Relaxation cutoff 650 eV > the static floor 620 eV: the static follows it.
    assert _encuts(_plan(LI2O_POSCAR, desired_output="relaxed_energy")) == [
        ("relax", 650), ("relax", 650), ("static", 650)
    ]
    dos = _encuts(_plan(LI2O_POSCAR, desired_output="electronic_dos"))
    assert dos == [("relax", 650), ("relax", 650), ("static", 650), ("dos", 620)]  # HSE06 DOS: own SCF, unchanged
    # A PBE DOS / Band Structure reads the static's charge density: it keeps the static's basis.
    for terminal in ("dos", "band_structure"):
        custom = {"stages": [_stage("relax"), _stage("static"), _stage(terminal)]}
        assert _encuts(_plan(LI2O_POSCAR, custom=custom)) == [("relax", 650), ("static", 650), (terminal, 650)]
    # A static with no preceding relaxation keeps its own cutoff.
    assert _encuts(_plan(LI2O_POSCAR, custom={"stages": [_stage("static")]})) == [("static", 620)]


def test_static_keeps_its_own_cutoff_when_it_is_higher():
    # SYNTHETIC metadata: relaxation 550 eV < static floor 620 eV.
    assert _encuts(_plan(desired_output="relaxed_energy")) == [("relax", 550), ("relax", 550), ("static", 620)]


def test_static_cutoff_rule_does_not_touch_historical_workflows():
    structure = parse_structure(SI_POSCAR)
    historical = WorkflowSpec([StageSpec(StageType.RELAX, Theory.PBE), StageSpec(StageType.STATIC, Theory.PBE)])
    assert gr.downstream_cutoff_floor(historical, 1) is None
    from backend.generated_inputs import generated_input_stage_previews

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        previews = generated_input_stage_previews(structure, historical, resources={"ntasks": 24})
    assert [p["input_set"].incar["ENCUT"] for p in previews] == [580, 620]


# ---------------------------------------------- browser Custom workflow builder


def _jsdom_available() -> bool:
    import shutil
    import subprocess

    if shutil.which("node") is None:
        return False
    return subprocess.run(["node", "-e", "require('jsdom')"], capture_output=True).returncode == 0


_BUILDER_HARNESS = r"""
const fs = require("fs");
const { JSDOM, VirtualConsole } = require("jsdom");
const dom = new JSDOM(fs.readFileSync(process.argv[2], "utf8"), { runScripts: "dangerously", virtualConsole: new VirtualConsole() });
const w = dom.window, d = w.document;
d.dispatchEvent(new w.Event("DOMContentLoaded"));
const fire = (el, type) => el.dispatchEvent(new w.Event(type, { bubbles: true }));
const cards = () => Array.from(d.querySelectorAll("#workflow-stage-list [data-workflow-stage]"));
const snapshot = (label) => ({
  label,
  workflow: JSON.parse(d.getElementById("workflow-spec-json").value),
  passes: cards().map((c) => c.getAttribute("data-protocol-pass")),
  typeDisabled: cards().map((c) => c.querySelector("[data-stage-type]").disabled),
  addRelaxationHidden: d.getElementById("add-geometry-relaxation").hidden,
});
const out = [snapshot("initial")];
for (const step of JSON.parse(fs.readFileSync(process.argv[3], "utf8"))) {
  if (step.op === "desired") { const s = d.getElementById("desired-output-select"); s.value = step.value; fire(s, "change"); }
  if (step.op === "addRelaxation") { d.getElementById("add-geometry-relaxation").click(); }
  if (step.op === "add") { d.getElementById("add-workflow-stage").click(); }
  if (step.op === "modifier") { const s = cards()[step.stage].querySelector(`[data-stage-modifier][value="${step.value}"]`); s.checked = step.checked; fire(s, "change"); }
  if (step.op === "remove") { cards()[step.stage].querySelector("[data-remove-stage]").click(); }
  out.push(snapshot(step.op));
}
process.stdout.write(JSON.stringify(out));
"""


def _run_builder(html, steps, tmp_path):
    import re
    import subprocess

    page = tmp_path / "page.html"
    page.write_text(re.sub(r'<script src="[^"]*"></script>', "", html), encoding="utf-8")
    (tmp_path / "steps.json").write_text(json.dumps(steps), encoding="utf-8")
    (tmp_path / "harness.js").write_text(_BUILDER_HARNESS, encoding="utf-8")
    result = subprocess.run(
        ["node", str(tmp_path / "harness.js"), str(page), str(tmp_path / "steps.json")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _custom_page(stages):
    response = soc_cases.build_route(
        SI_POSCAR, workflow="custom", workflow_spec=WorkflowSpec.from_dict({"stages": stages})
    )
    assert response.status_code == 200, response.context.get("calculation_error")
    return response


def _stage_signature(workflow):
    return [
        (stage["stage_type"], stage["theory"], tuple(stage["modifiers"]),
         (stage["options"].get("geometry_relaxation") or {}).get("pass"))
        for stage in workflow["stages"]
    ]


@pytest.mark.skipif(not _jsdom_available(), reason="Node.js with jsdom is not available")
def test_browser_custom_builder_adds_and_keeps_the_canonical_relaxation(tmp_path):
    response = _custom_page([_stage("static")])
    states = _run_builder(
        response.template.render(response.context),
        [
            {"op": "addRelaxation"},
            {"op": "modifier", "stage": 1, "value": "spin_polarized", "checked": True},
            {"op": "add"},
            {"op": "remove", "stage": 2},
        ],
        tmp_path,
    )
    initial, added, mirrored, single, removed = states
    assert initial["addRelaxationHidden"] is False
    # The two passes carry exactly the option the machine API accepts.
    assert _stage_signature(added["workflow"]) == [("static", "pbe", (), None), ("relax", "pbe", (), 1), ("relax", "pbe", (), 2)]
    assert added["workflow"]["stages"][1]["options"] == gr.protocol_option(1)
    assert added["typeDisabled"] == [False, True, True]
    # An Advanced Option on one pass is applied to both.
    assert [stage[2] for stage in _stage_signature(mirrored["workflow"])][1:] == [("spin_polarized",), ("spin_polarized",)]
    # Add Stage still adds a single, unmarked stage.
    assert single["passes"] == [None, "1", "2", None]
    # Removing either pass removes the whole relaxation.
    assert removed["passes"] == [None, None]
    # What the browser posts is a valid Custom workflow.
    validate_user_workflow_spec(WorkflowSpec.from_dict(mirrored["workflow"]))


@pytest.mark.skipif(not _jsdom_available(), reason="Node.js with jsdom is not available")
def test_browser_custom_builder_preserves_passes_after_build_and_single_relaxations(tmp_path):
    stages = [_stage("relax", options=gr.protocol_option(1)), _stage("relax", options=gr.protocol_option(2)), _stage("relax")]
    response = _custom_page(stages)
    (initial,) = _run_builder(response.template.render(response.context), [], tmp_path)
    # Server-rendered cards keep the passes; the frozen cutoff is never posted back.
    assert initial["passes"] == ["1", "2", None]
    assert _stage_signature(initial["workflow"]) == [("relax", "pbe", (), 1), ("relax", "pbe", (), 2), ("relax", "pbe", (), None)]
    assert all(set(stage["options"]) <= {"geometry_relaxation"} for stage in initial["workflow"]["stages"])
    # Planned exactly as the machine API plans the same Custom workflow.
    api = _plan(custom={"stages": stages})
    assert response.context["submission_spec"]["flow_spec"]["workflow_spec"] == api.submission_spec["flow_spec"]["workflow_spec"]


@pytest.mark.skipif(not _jsdom_available(), reason="Node.js with jsdom is not available")
def test_switching_a_managed_output_to_custom_keeps_the_canonical_relaxation(tmp_path):
    response = soc_cases.build_route(SI_POSCAR, workflow="relaxed_energy")
    states = _run_builder(response.template.render(response.context), [{"op": "desired", "value": "custom"}], tmp_path)
    assert states[0]["addRelaxationHidden"] is True
    assert _stage_signature(states[1]["workflow"]) == [
        ("relax", "pbe", (), 1), ("relax", "pbe", (), 2), ("static", "pbe", (), None)
    ]
    assert states[1]["addRelaxationHidden"] is False
