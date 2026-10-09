"""Automatic DFT+U policy v1 (bmd_compute.dft_u).

Frozen policy: the pinned pymatgen/Materials Project GGA+U oxide/fluoride rule
and parameters, a compound-level d0 gate over every charge-balanced pymatgen
oxidation-state guess, PBE Relax/Static placement only, and preparation-time
freezing that runtime verifies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pymatgen.core import Composition, Lattice, Structure
from starlette.requests import Request

import main
from backend.calculations import dft_u_policy
from backend.calculations.default_treatments import (
    AUTOMATIC_APPLICATION_ADVISORY,
    AUTOMATIC_APPLICATION_APPLIED,
    AUTOMATIC_APPLICATION_NOT_APPLICABLE,
    DFT_U_CONSIDERATION_ID,
    DISPERSION_CONSIDERATION_ID,
    SOC_CONSIDERATION_ID,
    SPIN_CONSIDERATION_ID,
    resolve_default_treatments,
)
from backend.calculations.dft_u_policy import (
    DECISION_APPLY,
    DECISION_NOT_TRIGGERED,
    DECISION_SUPPRESS,
    GATE_ALL_D0,
    GATE_D_ELECTRONS_PRESENT,
    GATE_NOT_EVALUATED,
    GATE_UNAVAILABLE,
    DftUFreezeError,
    evaluate_dft_u,
    frozen_parameters,
    mp_hubbard_table,
)
from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    desired_output_workflow_spec,
    validate_workflow_spec,
)
from backend.structure_dimensionality import OBSERVED, StructureDimensionalityObservation
from backend.generated_inputs import generated_input_stage_previews, preview_generated_inputs
from backend.parser import parse_structure, structure_from_spec
from backend.workflows import (
    build_atomate2_flow_from_spec,
    build_static_input_set_generator,
    frozen_dft_u_for_stage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

MP_U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2}


# --- Structures ----------------------------------------------------------------


def composition_structure(formula: str) -> Structure:
    """A placeholder cell carrying exactly the formula; the gate reads composition only."""

    composition = Composition(formula)
    species: list[str] = []
    for element, amount in composition.items():
        species.extend([element.symbol] * int(round(amount)))
    side = math.ceil(len(species) ** (1 / 3))
    coords = [
        [(i % side) / side, ((i // side) % side) / side, (i // side // side) / side]
        for i in range(len(species))
    ]
    return Structure(Lattice.cubic(2.5 * side), species, coords)


def rocksalt(a: float, cation: str, anion: str) -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()


def poscar_text(structure: Structure) -> str:
    return structure.to(fmt="poscar")


NIO = rocksalt(4.17, "Ni", "O")
FE2O3 = Structure.from_spacegroup(
    "R-3c", Lattice.hexagonal(5.035, 13.75), ["Fe", "O"], [[0, 0, 0.3553], [0.3059, 0, 0.25]]
).get_primitive_structure()
WO2 = Structure.from_spacegroup(
    "P4_2/mnm", Lattice.tetragonal(4.86, 2.77), ["W", "O"], [[0, 0, 0], [0.3, 0.3, 0]]
)
WO3 = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.8), ["W", "O"], [[0, 0, 0], [0.5, 0, 0]])
FEWO4 = composition_structure("FeWO4")


# --- Route / runtime helpers ---------------------------------------------------


def request(path: str = "/build-workflow") -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def build_route(structure: Structure, *, workflow: str, workflow_spec: WorkflowSpec | None = None):
    return main.build_workflow(
        request(),
        structure=poscar_text(structure),
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=(
            json.dumps(workflow_spec.to_dict(), sort_keys=True) if workflow_spec is not None else None
        ),
        workflow=workflow,
        method=None,
    )


def resolve(desired_output: str, structure: Structure):
    return resolve_default_treatments(
        structure,
        desired_output_workflow_spec(desired_output),
        desired_output=desired_output,
    )


def signature(workflow: WorkflowSpec):
    return [
        (stage.stage_type.value, stage.theory.value, tuple(sorted(m.value for m in stage.modifiers)))
        for stage in workflow.stages
    ]


def considerations_by_id(context: dict) -> dict:
    return {item["id"]: item for item in context["method_considerations"]["considerations"]}


def stage_sections(text: str) -> list[str]:
    if "# Stage " not in text:
        return [text]
    return ["# Stage " + section for section in text.split("# Stage ")[1:]]


def incar_value(incar_text: str, key: str) -> str | None:
    for line in incar_text.splitlines():
        if line.startswith(f"{key} = "):
            return line.removeprefix(f"{key} = ")
    return None


def numbers(value: str | None) -> list[float]:
    assert value is not None
    result: list[float] = []
    for token in value.split():
        if "*" in token:
            count, number = token.split("*", 1)
            result.extend([float(number)] * int(count))
        else:
            result.append(float(token))
    return result


def runtime_jobs(submission_spec: dict):
    runtime_structure = structure_from_spec(submission_spec["flow_spec"]["structure"])
    flow = build_atomate2_flow_from_spec(
        runtime_structure,
        submission_spec["flow_spec"],
        run_name=submission_spec["run_name"],
        resources=submission_spec["resources"],
    )
    return runtime_structure, list(flow.jobs)


def job_maker(job):
    maker = getattr(getattr(job, "function", None), "__self__", None)
    assert maker is not None
    return maker


def runtime_input_set(submission_spec: dict, stage_index: int):
    runtime_structure, jobs = runtime_jobs(submission_spec)
    return job_maker(jobs[stage_index]).input_set_generator.get_input_set(
        runtime_structure, potcar_spec=True
    )


# --- Trigger and d0 gate: the agreed representative set -------------------------


APPLY = {
    "NiO": ["Ni"], "MnO": ["Mn"], "CoO": ["Co"], "FeO": ["Fe"], "Fe2O3": ["Fe"],
    "Cr2O3": ["Cr"], "NiF2": ["Ni"], "LiCoO2": ["Co"], "FeOCl": ["Fe"],
    "LiFePO4": ["Fe"], "Fe3O4": ["Fe"], "Co3O4": ["Co"], "LiMn2O4": ["Mn"],
    "VO2": ["V"], "V2O3": ["V"], "MoO2": ["Mo"], "WO2": ["W"], "FeWO4": ["Fe", "W"],
    "Li10Ni8CoMnO20": ["Co", "Mn", "Ni"],
}
SUPPRESS = {"WO3": ["W"], "MoO3": ["Mo"], "V2O5": ["V"], "CrO3": ["Cr"], "KMnO4": ["Mn"]}
NOT_TRIGGERED = ["TiO2", "ZnO", "Cu2O", "FeS2", "MoS2", "FeN"]
NO_GUESS_FALLBACK = {"BaFeO3": ["Fe"], "K2FeO4": ["Fe"]}


@pytest.mark.parametrize("formula", sorted(APPLY))
def test_representative_d_electron_oxides_and_fluorides_apply_mp_u(formula):
    record = evaluate_dft_u(composition_structure(formula))

    assert record["mp_rule_triggered"] is True
    assert record["triggering_elements"] == APPLY[formula]
    assert record["deciding_anion"] == ("F" if formula == "NiF2" else "O")
    assert record["oxidation_state_guesses"], formula
    assert record["gate"] == GATE_D_ELECTRONS_PRESENT
    assert record["decision"] == DECISION_APPLY
    species = record["parameters"]["species"]
    for symbol in APPLY[formula]:
        assert species[symbol] == {"L": 2, "U": MP_U[symbol], "J": 0}
    for symbol, values in species.items():
        if symbol not in APPLY[formula]:
            assert values["U"] == 0


@pytest.mark.parametrize("formula", sorted(SUPPRESS))
def test_d0_oxides_are_suppressed_at_compound_level(formula):
    record = evaluate_dft_u(composition_structure(formula))

    assert record["mp_rule_triggered"] is True
    assert record["triggering_elements"] == SUPPRESS[formula]
    assert record["gate"] == GATE_ALL_D0
    assert record["decision"] == DECISION_SUPPRESS
    assert "parameters" not in record
    assert record["d_counts"]
    assert all(value == 0 for counts in record["d_counts"] for value in counts.values())


@pytest.mark.parametrize("formula", NOT_TRIGGERED)
def test_non_mp_u_compositions_are_not_triggered(formula):
    record = evaluate_dft_u(composition_structure(formula))

    assert record["mp_rule_triggered"] is False
    assert record["gate"] == GATE_NOT_EVALUATED
    assert record["decision"] == DECISION_NOT_TRIGGERED
    assert record["oxidation_state_guesses"] == []
    assert "parameters" not in record


def test_sulfide_and_nitride_are_decided_by_their_own_most_electronegative_anion():
    assert evaluate_dft_u(composition_structure("FeS2"))["most_electronegative_element"] == "S"
    assert evaluate_dft_u(composition_structure("MoS2"))["most_electronegative_element"] == "S"
    assert evaluate_dft_u(composition_structure("FeN"))["most_electronegative_element"] == "N"
    # FeOCl: O is more electronegative than Cl, so the oxide rule decides.
    assert evaluate_dft_u(composition_structure("FeOCl"))["most_electronegative_element"] == "O"


@pytest.mark.parametrize("formula", sorted(NO_GUESS_FALLBACK))
def test_no_oxidation_state_guess_falls_back_to_the_mp_rule_with_provenance(formula):
    record = evaluate_dft_u(composition_structure(formula))

    assert record["oxidation_state_guesses"] == []
    assert record["gate"] == GATE_UNAVAILABLE
    assert record["decision"] == DECISION_APPLY
    assert "no charge-balanced" in record["gate_reason"]
    assert record["parameters"]["species"]["Fe"]["U"] == MP_U["Fe"]


def test_oxidation_state_guess_errors_fall_back_to_the_mp_rule(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise ValueError("synthetic refusal")

    monkeypatch.setattr(Composition, "oxi_state_guesses", refuse)
    record = evaluate_dft_u(composition_structure("NiO"))

    assert record["gate"] == GATE_UNAVAILABLE
    assert record["decision"] == DECISION_APPLY
    assert record["oxidation_state_error"] == "ValueError: synthetic refusal"
    assert record["parameters"]["species"]["Ni"]["U"] == MP_U["Ni"]


@pytest.mark.parametrize("formula", ["Fe3O4", "Co3O4", "LiMn2O4", "Li10Ni8CoMnO20"])
def test_mixed_valence_records_every_guess_and_d_count(formula):
    record = evaluate_dft_u(composition_structure(formula))

    guesses = record["oxidation_state_guesses"]
    assert guesses
    assert len(record["d_counts"]) == len(guesses)
    for guess, counts in zip(guesses, record["d_counts"]):
        assert set(counts) == set(record["triggering_elements"])
        assert all(symbol in guess for symbol in counts)
    # Mixed valence shows up as non-integral average oxidation states.
    assert any(
        not float(guess[symbol]).is_integer()
        for guess in guesses
        for symbol in record["triggering_elements"]
    )
    assert record["decision"] == DECISION_APPLY


def test_nmc_records_all_sixteen_guesses():
    record = evaluate_dft_u(composition_structure("Li10Ni8CoMnO20"))
    assert len(record["oxidation_state_guesses"]) == 16


def test_gate_never_removes_u_from_one_species():
    # FeWO4: W is d0 in the only W6+ guesses but Fe keeps d electrons; the
    # compound-level gate keeps U on every triggering species.
    record = evaluate_dft_u(FEWO4)
    assert record["decision"] == DECISION_APPLY
    assert record["parameters"]["species"]["Fe"]["U"] == MP_U["Fe"]
    assert record["parameters"]["species"]["W"]["U"] == MP_U["W"]

    preview = preview_generated_inputs(FEWO4, resolve("energy_only", FEWO4).resolved_workflow)
    symbols = preview["poscar"].splitlines()[5].split()
    u_values = dict(zip(symbols, numbers(incar_value(preview["incar"], "LDAUU"))))
    assert u_values == {"Fe": MP_U["Fe"], "W": MP_U["W"], "O": 0.0}


def test_gate_only_suppresses_and_never_introduces_u():
    for formula in NOT_TRIGGERED:
        record = evaluate_dft_u(composition_structure(formula))
        assert record["decision"] == DECISION_NOT_TRIGGERED
        resolution = resolve("energy_only", composition_structure(formula))
        assert all(Modifier.DFT_U not in stage.modifiers for stage in resolution.resolved_workflow.stages)
        assert resolution.dft_u is None or resolution.dft_u["decision"] != DECISION_APPLY


# --- Parameters come from the pinned pymatgen table ----------------------------


def test_parameters_are_the_pinned_mprelaxset_values():
    table = mp_hubbard_table()
    assert table["LDAUTYPE"] == 2
    for anion in ("O", "F"):
        assert {el: u for el, u in table["LDAUU"][anion].items() if u} == MP_U
        assert set(table["LDAUJ"][anion].values()) == {0}
        assert set(table["LDAUL"][anion].values()) == {2}

    frozen = frozen_parameters(["Li", "Ni", "O"], "O")
    assert frozen["parameter_source"] == dft_u_policy.PARAMETER_SOURCE
    assert frozen["LDAUTYPE"] == 2
    assert frozen["species"]["Ni"] == {"L": 2, "U": 6.2, "J": 0}
    assert frozen["species"]["Li"]["U"] == 0
    assert frozen["species"]["O"]["U"] == 0


def test_atomate2_generator_table_matches_mprelaxset():
    from atomate2.vasp.sets.base import _BASE_VASP_SET

    table = mp_hubbard_table()
    for key in ("LDAUU", "LDAUL", "LDAUJ", "LDAUTYPE"):
        assert _BASE_VASP_SET["INCAR"][key] == table[key]


# --- Desired Output stage placement --------------------------------------------


def test_static_energy_places_u_on_the_pbe_static():
    resolution = resolve("energy_only", NIO)
    assert signature(resolution.resolved_workflow) == [
        ("static", "pbe", ("dft_u", "spin_polarized")),
    ]
    stage = resolution.resolved_workflow.stages[0]
    assert stage.options["dft_u"]["species"]["Ni"]["U"] == 6.2
    treatment = next(t for t in resolution.applied_treatments if t.consideration_id == DFT_U_CONSIDERATION_ID)
    assert treatment.stage_indices == (1,)


def test_relaxed_structure_places_u_on_both_pbe_relaxations():
    resolution = resolve("relaxed_structure", FE2O3)
    assert signature(resolution.resolved_workflow) == [
        ("relax", "pbe", ("dft_u", "spin_polarized")),
        ("relax", "pbe", ("dft_u", "spin_polarized")),
    ]
    assert all("dft_u" in stage.options for stage in resolution.resolved_workflow.stages)


@pytest.mark.parametrize(
    ("desired_output", "terminal"),
    [("electronic_dos", "dos"), ("electronic_band_structure", "band_structure")],
)
def test_dos_and_band_place_u_on_the_pbe_relaxation_only(desired_output, terminal):
    resolution = resolve(desired_output, NIO)
    assert signature(resolution.resolved_workflow) == [
        ("relax", "pbe", ("dft_u", "spin_polarized")),
        ("static", "hse06", ("spin_polarized",)),
        (terminal, "hse06", ("spin_polarized",)),
    ]
    relax, static, analysis = resolution.resolved_workflow.stages
    assert "dft_u" in relax.options
    assert "dft_u" not in (static.options or {})
    assert "dft_u" not in (analysis.options or {})
    treatment = next(t for t in resolution.applied_treatments if t.consideration_id == DFT_U_CONSIDERATION_ID)
    assert treatment.stage_indices == (1,)

    response = build_route(NIO, workflow=desired_output)
    assert response.status_code == 200
    sections = stage_sections(response.context["generated_inputs"]["incar"])
    assert incar_value(sections[0], "LDAU") == "True"
    for section in sections[1:]:
        assert "LHFCALC = True" in section
        assert "LDAU" not in section


def test_d0_suppression_leaves_every_desired_output_without_u():
    for desired_output in ("energy_only", "relaxed_structure", "electronic_dos", "electronic_band_structure"):
        resolution = resolve(desired_output, WO3)
        assert all(Modifier.DFT_U not in stage.modifiers for stage in resolution.resolved_workflow.stages)
        suppressed = [
            item for item in resolution.not_applicable_considerations
            if item["consideration_id"] == DFT_U_CONSIDERATION_ID
        ]
        assert len(suppressed) == 1
        assert suppressed[0]["gate"] == GATE_ALL_D0
        assert resolution.dft_u["decision"] == DECISION_SUPPRESS


# --- Spin, SOC and D3 interactions ----------------------------------------------


def test_u_does_not_enable_spin():
    for desired_output in ("energy_only", "relaxed_structure", "electronic_dos"):
        resolution = resolve(desired_output, WO2)
        assert any(Modifier.DFT_U in stage.modifiers for stage in resolution.resolved_workflow.stages)
        assert all(
            Modifier.SPIN_POLARIZED not in stage.modifiers for stage in resolution.resolved_workflow.stages
        )
        assert SPIN_CONSIDERATION_ID not in {t.consideration_id for t in resolution.applied_treatments}

    preview = preview_generated_inputs(WO2, resolve("energy_only", WO2).resolved_workflow)
    assert "ISPIN = 2" not in preview["incar"]


def test_spin_is_decided_independently_and_composes_with_u():
    resolution = resolve("energy_only", NIO)
    applied = [t.consideration_id for t in resolution.applied_treatments]
    assert applied == [SPIN_CONSIDERATION_ID, DFT_U_CONSIDERATION_ID]


def test_soc_and_u_coexist_on_the_pbe_static():
    resolution = resolve("energy_only", WO2)
    assert signature(resolution.resolved_workflow) == [("static", "pbe", ("dft_u", "soc"))]
    preview = preview_generated_inputs(WO2, resolution.resolved_workflow)
    assert preview["vasp_executable"] == "vasp_ncl"
    assert "LSORBIT = True" in preview["incar"]
    assert incar_value(preview["incar"], "LDAU") == "True"


def test_soc_u_and_d3_coexist_on_one_pbe_static(monkeypatch):
    from backend.calculations import method_considerations

    monkeypatch.setattr(
        method_considerations,
        "observe_structure_dimensionality",
        lambda structure: StructureDimensionalityObservation(status=OBSERVED, dimensionality=2),
    )
    resolution = resolve("energy_only", WO2)
    assert signature(resolution.resolved_workflow) == [
        ("static", "pbe", ("dft_u", "dispersion", "soc")),
    ]
    assert [t.consideration_id for t in resolution.applied_treatments] == [
        DISPERSION_CONSIDERATION_ID,
        SOC_CONSIDERATION_ID,
        DFT_U_CONSIDERATION_ID,
    ]
    preview = preview_generated_inputs(WO2, resolution.resolved_workflow)
    assert "IVDW = 12" in preview["incar"]
    assert "LSORBIT = True" in preview["incar"]
    assert incar_value(preview["incar"], "LDAU") == "True"


def test_hse_stages_never_receive_u_when_the_soc_policy_also_triggers():
    # W triggers the heavy-element SOC policy; HSE06 + SOC is closed to new
    # calculations, so SOC is omitted from the HSE06 stages (and recorded).
    # Neither +U nor SOC reaches them.
    resolution = resolve("electronic_band_structure", WO2)
    assert signature(resolution.resolved_workflow) == [
        ("relax", "pbe", ("dft_u",)),
        ("static", "hse06", ()),
        ("band_structure", "hse06", ()),
    ]
    assert [item["consideration_id"] for item in resolution.omitted_treatments] == [
        "soc.heavy_elements",
    ]
    assert resolution.omitted_treatments[0]["stage_indices"] == [2, 3]


# --- Custom workflows -------------------------------------------------------------


def test_custom_workflow_is_never_altered_and_gets_an_advisory():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)], recipe="custom")
    response = build_route(NIO, workflow="custom", workflow_spec=workflow)

    assert response.status_code == 200
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    assert "LDAU" not in response.context["generated_inputs"]["incar"]
    assert "automatic_treatments" not in response.context["submission_spec"]["flow_spec"]
    dftu = considerations_by_id(response.context)[DFT_U_CONSIDERATION_ID]
    assert dftu["automatic_application_state"] == AUTOMATIC_APPLICATION_ADVISORY
    rendered = response.template.render(response.context)
    assert "Suggested to activate the DFT+U Advanced Option" in rendered


def test_custom_manual_dft_u_stays_the_users_choice():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DFT_U})], recipe="custom")
    response = build_route(WO3, workflow="custom", workflow_spec=workflow)
    assert response.status_code == 200
    # The d0 gate never removes a user's explicit DFT+U choice.
    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == ["dft_u"]
    assert incar_value(response.context["generated_inputs"]["incar"], "LDAU") == "True"


# --- ICHARG=11 chaining ----------------------------------------------------------


@pytest.mark.parametrize("terminal", [StageType.DOS, StageType.BAND_STRUCTURE])
@pytest.mark.parametrize(("static_u", "terminal_u"), [(True, False), (False, True)])
def test_fixed_charge_density_stage_must_match_the_source_u_state(terminal, static_u, terminal_u):
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DFT_U} if static_u else set()),
            StageSpec(terminal, Theory.PBE, {Modifier.DFT_U} if terminal_u else set()),
        ],
        recipe="custom",
    )
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(workflow)
    assert "fixed charge density" in excinfo.value.message
    assert "DFT+U" in excinfo.value.message


@pytest.mark.parametrize(
    "workflow",
    [
        WorkflowSpec(
            [StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DFT_U}), StageSpec(StageType.DOS, Theory.PBE, {Modifier.DFT_U})]
        ),
        WorkflowSpec(
            [
                StageSpec(StageType.STATIC, Theory.PBE, {Modifier.DFT_U}),
                StageSpec(StageType.BAND_STRUCTURE, Theory.PBE, {Modifier.DFT_U}),
            ]
        ),
        # Relax +U -> plain HSE06 Static shares only the structure.
        WorkflowSpec(
            [
                StageSpec(StageType.RELAX, Theory.PBE, {Modifier.DFT_U}),
                StageSpec(StageType.STATIC, Theory.HSE06),
                StageSpec(StageType.DOS, Theory.HSE06),
            ]
        ),
    ],
)
def test_matching_u_chains_remain_allowed(workflow):
    validate_workflow_spec(workflow)


def test_frozen_option_requires_a_pbe_relax_or_static_with_the_dft_u_modifier():
    parameters = evaluate_dft_u(NIO)["parameters"]
    for stage in (
        StageSpec(StageType.STATIC, Theory.PBE, options={"dft_u": parameters}),
        StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.DFT_U}, options={"dft_u": parameters}),
    ):
        with pytest.raises(CalculationValidationError):
            validate_workflow_spec(WorkflowSpec([stage], recipe="custom"))


# --- Freezing and runtime verification -------------------------------------------


def _frozen_nio_generator(expected=None):
    stage = resolve("energy_only", NIO).resolved_workflow.stages[0]
    return build_static_input_set_generator(
        NIO,
        modifiers=stage.modifiers,
        frozen_dft_u=frozen_dft_u_for_stage(stage, expected_generated=expected),
    )


def test_frozen_parameters_survive_upstream_table_drift(monkeypatch):
    from atomate2.vasp.sets.base import _BASE_VASP_SET

    drifted = json.loads(json.dumps(_BASE_VASP_SET["INCAR"]["LDAUU"]))
    drifted["O"]["Ni"] = 9.9
    monkeypatch.setitem(_BASE_VASP_SET["INCAR"], "LDAUU", drifted)

    # The unfrozen manual path follows the drifted upstream table ...
    unfrozen = build_static_input_set_generator(NIO, modifiers={Modifier.DFT_U}).get_input_set(
        NIO, potcar_spec=True
    )
    symbols = unfrozen.poscar.site_symbols
    assert dict(zip(symbols, unfrozen.incar["LDAUU"]))["Ni"] == 9.9

    # ... while the frozen automatic path keeps the prepared value.
    frozen = _frozen_nio_generator().get_input_set(NIO, potcar_spec=True)
    assert dict(zip(frozen.poscar.site_symbols, frozen.incar["LDAUU"]))["Ni"] == 6.2


def test_runtime_refuses_generated_u_that_disagrees_with_the_frozen_values():
    decorated = NIO.copy()
    decorated.add_site_property("ldauu", [4.0] * len(decorated))
    with pytest.raises(DftUFreezeError):
        _frozen_nio_generator().get_input_set(decorated, potcar_spec=True)


@pytest.mark.parametrize(
    ("key", "tampered"),
    [
        ("LDAUU", [1.0, 0.0]),
        ("LMAXMIX", 6),
        ("potcar_symbols", ["Ni", "O_h"]),
        ("poscar_symbols", ["O", "Ni"]),
    ],
)
def test_runtime_refuses_generated_inputs_that_disagree_with_preparation(key, tampered):
    good = _frozen_nio_generator().get_input_set(NIO, potcar_spec=True)
    expected = dft_u_policy.generated_dft_u_settings(good)
    _frozen_nio_generator(expected).get_input_set(NIO, potcar_spec=True)  # matches

    expected[key] = tampered
    with pytest.raises(DftUFreezeError, match="preparation recorded"):
        _frozen_nio_generator(expected).get_input_set(NIO, potcar_spec=True)


def test_prepared_record_carries_full_provenance():
    response = build_route(NIO, workflow="electronic_dos")
    submission_spec = response.context["submission_spec"]
    flow_spec = submission_spec["flow_spec"]
    record = flow_spec["automatic_treatments"]["dft_u"]

    assert record["policy_id"] == "bmd_compute.dft_u"
    assert record["policy_version"] == 1
    assert record["rule_id"] == dft_u_policy.RULE_ID
    assert record["deciding_anion"] == "O"
    assert record["triggering_elements"] == ["Ni"]
    assert record["oxidation_state_guesses"] == [{"Ni": 2.0, "O": -2.0}]
    assert record["d_counts"] == [{"Ni": 8.0}]
    assert record["gate"] == GATE_D_ELECTRONS_PRESENT
    assert record["gate_reason"]
    assert record["parameters"]["parameter_source"] == dft_u_policy.PARAMETER_SOURCE

    source = record["parameter_source_record"]
    assert len(source["mprelaxset_yaml_sha256"]) == 64
    assert source["atomate2_table_matches_mprelaxset"] is True
    assert set(source["packages"]) == {"atomate2", "pymatgen", "pymatgen-core"}
    assert all(source["packages"].values())

    assert [stage["stage_index"] for stage in record["generated_stages"]] == [1]
    stage = record["generated_stages"][0]
    assert stage["stage_type"] == "relax" and stage["theory"] == "pbe"
    assert stage["LDAU"] is True
    assert stage["LDAUTYPE"] == 2
    assert dict(zip(stage["poscar_symbols"], stage["LDAUU"])) == {"Ni": 6.2, "O": 0.0}
    assert dict(zip(stage["poscar_symbols"], stage["LDAUL"])) == {"Ni": 2, "O": 0}
    assert set(stage["LDAUJ"]) == {0}
    assert stage["LMAXMIX"] == 4
    assert len(stage["potcar_symbols"]) == len(stage["poscar_symbols"])

    assert submission_spec["provenance"]["execution"]["automatic_treatments"] == flow_spec["automatic_treatments"]


def test_suppressed_record_is_preserved_without_generated_stages():
    response = build_route(WO3, workflow="energy_only")
    record = response.context["submission_spec"]["flow_spec"]["automatic_treatments"]["dft_u"]
    assert record["decision"] == DECISION_SUPPRESS
    assert record["gate"] == GATE_ALL_D0
    assert record["generated_stages"] == []
    assert record["d_counts"] == [{"W": 0.0}]


# --- Preview / runtime / input-reference parity -----------------------------------


@pytest.mark.parametrize("structure", [NIO, FE2O3], ids=["NiO", "Fe2O3"])
@pytest.mark.parametrize("desired_output", ["energy_only", "relaxed_structure", "electronic_band_structure"])
def test_preview_runtime_and_reference_share_the_frozen_u(structure, desired_output):
    response = build_route(structure, workflow=desired_output)
    assert response.status_code == 200
    submission_spec = response.context["submission_spec"]
    record = submission_spec["flow_spec"]["automatic_treatments"]["dft_u"]
    sections = stage_sections(response.context["generated_inputs"]["incar"])

    reference = build_input_reference_payload(
        {
            "structure": {"type": "pasted_text", "format": "poscar", "text": poscar_text(structure)},
            "workflow_spec": submission_spec["flow_spec"]["workflow_spec"],
            "resources": submission_spec["resources"],
            "potcar_functional": "PBE_64",
        },
        include_provenance=False,
    )
    reference_stages = reference["reference"]["stages"]

    runtime_structure, jobs = runtime_jobs(submission_spec)
    for generated in record["generated_stages"]:
        index = generated["stage_index"] - 1
        preview_u = numbers(incar_value(sections[index], "LDAUU"))
        assert preview_u == generated["LDAUU"]
        reference_settings = reference_stages[index]["incar"]["settings"]
        assert reference_settings["LDAUU"] == generated["LDAUU"]
        assert reference_settings["LDAUL"] == generated["LDAUL"]
        assert reference_settings["LMAXMIX"] == generated["LMAXMIX"]
        assert reference_stages[index]["options"]["dft_u"] == record["parameters"]
        # The first stage's structure is known before the run; later stages
        # regenerate from the relaxed structure, which keeps species order.
        runtime = job_maker(jobs[index]).input_set_generator.get_input_set(
            runtime_structure, potcar_spec=True
        )
        assert list(runtime.incar["LDAUU"]) == generated["LDAUU"]
        assert runtime.incar["LMAXMIX"] == generated["LMAXMIX"]


def test_runtime_flow_fails_when_prepared_expectations_are_tampered():
    response = build_route(NIO, workflow="energy_only")
    submission_spec = json.loads(json.dumps(response.context["submission_spec"]))
    submission_spec["flow_spec"]["automatic_treatments"]["dft_u"]["generated_stages"][0]["LDAUU"] = [3.0, 0.0]
    runtime_structure, jobs = runtime_jobs(submission_spec)
    with pytest.raises(DftUFreezeError):
        job_maker(jobs[0]).input_set_generator.get_input_set(runtime_structure, potcar_spec=True)


# --- Method Considerations text -----------------------------------------------------


def test_applied_text_names_stages_values_and_source():
    response = build_route(NIO, workflow="relaxed_structure")
    dftu = considerations_by_id(response.context)[DFT_U_CONSIDERATION_ID]
    assert dftu["automatic_application_state"] == AUTOMATIC_APPLICATION_APPLIED
    rendered = response.template.render(response.context)
    assert "DFT+U applied" in rendered
    assert "DFT+U has been included automatically in the" in rendered
    assert "Ni U = 6.2 eV, J = 0 eV" in rendered
    assert "Materials Project/pymatgen GGA+U oxide/fluoride" in rendered
    assert "HSE06 electronic-structure stages do not use +U" not in rendered
    summary = main._dft_u_browser_summary(dftu, is_applied=True)
    assert "Geometry Optimisation stages" in summary
    assert "not values determined for this material" in summary
    for claim in ("required", "validated", "necessary"):
        assert claim not in summary.lower()


def test_dos_text_states_u_is_on_the_pbe_relaxation_only():
    response = build_route(NIO, workflow="electronic_dos")
    rendered = response.template.render(response.context)
    assert "DFT+U applies to the PBE geometry optimisation only" in rendered
    assert "the HSE06 electronic-structure stages do not use +U" in rendered


def test_suppressed_text_explains_the_d0_gate():
    response = build_route(WO3, workflow="energy_only")
    dftu = considerations_by_id(response.context)[DFT_U_CONSIDERATION_ID]
    assert dftu["automatic_application_state"] == AUTOMATIC_APPLICATION_NOT_APPLICABLE
    rendered = response.template.render(response.context)
    assert "DFT+U not applied automatically" in rendered
    assert "puts W at d0" in rendered


# --- Dependency and provenance capture -----------------------------------------------


def test_general_provenance_records_pymatgen_core():
    from backend import provenance

    assert "pymatgen-core" in provenance.SCIENTIFIC_PACKAGE_NAMES
    response = build_route(NIO, workflow="energy_only")
    packages = json.dumps(response.context["submission_spec"]["provenance"])
    assert "pymatgen-core" in packages


def test_runtime_info_reports_pymatgen_core():
    source = (REPO_ROOT / "backend" / "execution.py").read_text(encoding="utf-8")
    assert '"pymatgen-core"' in source


def test_environment_pins_the_production_stack():
    text = (REPO_ROOT / "constraints" / "scientific-runtime.txt").read_text(encoding="utf-8")
    assert "atomate2==0.1.5" in text
    assert "pymatgen==2026.5.4" in text
    assert "pymatgen-core==2026.7.16" in text


def test_running_stack_matches_the_pins():
    from importlib import metadata

    assert metadata.version("atomate2") == "0.1.5"
    assert metadata.version("pymatgen") == "2026.5.4"
    assert metadata.version("pymatgen-core") == "2026.7.16"
