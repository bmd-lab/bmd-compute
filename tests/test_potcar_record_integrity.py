"""POTCAR identity shown and recorded by Compute comes from the executing generators.

Production acceptance case (job 22392213, NiO Static Energy): the old record
said ``Ni_pv`` (pymatgen MPRelaxSet) while the executed POTCAR was PAW_PBE Ni.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.vasp import sets as pymatgen_sets
from pymatgen.io.vasp.sets import MPRelaxSet

from backend.calculations import potcar_record
from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.potcar_record import (
    POTCAR_SYMBOL_SOURCE,
    aggregate_stage_potcars,
    executable_potcar_record,
)
from backend.calculations.registry import desired_output_workflow_spec
from backend.generated_inputs import generated_input_stage_previews
from backend.submission import create_submission_spec
from test_automatic_dft_u import (
    NIO,
    build_route,
    job_maker,
    poscar_text,
    runtime_jobs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def composition_structure(formula: str) -> Structure:
    composition = Composition(formula)
    species: list[str] = []
    for element, amount in composition.items():
        species.extend([element.symbol] * int(round(amount)))
    return Structure(
        Lattice.cubic(3.0 * len(species)),
        species,
        [[index / len(species), 0.0, 0.0] for index in range(len(species))],
    )


def mprelaxset_symbols(structure: Structure) -> dict[str, str]:
    mp = MPRelaxSet(structure, user_potcar_functional="PBE_64")
    return dict(zip(mp.poscar.site_symbols, mp.potcar_symbols))


def generator_symbols(structure: Structure, desired_output: str = "energy_only") -> list[dict[str, str]]:
    previews = generated_input_stage_previews(
        structure,
        desired_output_workflow_spec(desired_output),
        potcar_functional="PBE_64",
    )
    return [
        dict(
            zip(
                preview["input_set"].poscar.site_symbols,
                str(preview["input_set"].potcar).split(),
            )
        )
        for preview in previews
    ]


def submission_potcar(structure: Structure, desired_output: str = "energy_only") -> dict:
    flow_spec = {
        "workflow_spec": desired_output_workflow_spec(desired_output).to_dict(),
        "potcar_functional": "PBE_64",
        "kpoints": None,
        "incar": {},
        "structure": {"type": "pasted_text", "format": "poscar", "text": poscar_text(structure)},
    }
    spec = create_submission_spec(flow_spec, structure=structure, timestamp="20260929-120000", env={})
    return spec["potcar"]


# --- The production acceptance case: NiO -----------------------------------------


def test_nio_records_the_executed_ni_potcar_not_mprelaxset_ni_pv():
    assert mprelaxset_symbols(NIO)["Ni"] == "Ni_pv"  # the old, wrong source

    response = build_route(NIO, workflow="energy_only")
    submission_spec = response.context["submission_spec"]
    potcar = submission_spec["potcar"]

    assert potcar["status"] == "resolved"
    assert potcar["symbol_source"] == POTCAR_SYMBOL_SOURCE
    assert potcar["symbols"] == ["Ni", "O"]
    assert potcar["species"] == [
        {"species": "Ni", "potcar_symbol": "Ni"},
        {"species": "O", "potcar_symbol": "O"},
    ]
    assert potcar["consistent_across_stages"] is True

    provenance = submission_spec["provenance"]["potcar"]
    for key in ("status", "species", "symbols", "symbol_source", "consistent_across_stages", "stages"):
        assert provenance[key] == potcar[key]

    # Execution: the runtime stage generator writes exactly these POTCARs.
    runtime_structure, jobs = runtime_jobs(submission_spec)
    runtime_set = job_maker(jobs[0]).input_set_generator.get_input_set(
        runtime_structure, potcar_spec=True
    )
    assert str(runtime_set.potcar).split() == potcar["symbols"]
    assert list(runtime_set.poscar.site_symbols) == ["Ni", "O"]

    # The frozen DFT+U record already agreed with execution.
    generated = submission_spec["flow_spec"]["automatic_treatments"]["dft_u"]["generated_stages"]
    assert generated[0]["potcar_symbols"] == potcar["symbols"]

    rendered = response.template.render(response.context)
    assert "Ni &rarr; Ni<br>" in rendered
    assert "Ni_pv" not in rendered
    assert "Ni_pv" not in json.dumps(submission_spec)


def test_input_reference_agrees_with_the_submission_record():
    response = build_route(NIO, workflow="electronic_band_structure")
    submission_spec = response.context["submission_spec"]
    reference = build_input_reference_payload(
        {
            "structure": {"type": "pasted_text", "format": "poscar", "text": poscar_text(NIO)},
            "workflow_spec": submission_spec["flow_spec"]["workflow_spec"],
            "resources": submission_spec["resources"],
            "potcar_functional": "PBE_64",
        },
        include_provenance=False,
    )
    reference_symbols = [stage["potcar"]["symbols"] for stage in reference["reference"]["stages"]]
    assert reference_symbols == [stage["symbols"] for stage in submission_spec["potcar"]["stages"]]


# --- Elements where MPRelaxSet and the atomate2 generators differ ------------------


@pytest.mark.parametrize(
    ("formula", "element", "executed", "mprelaxset"),
    [
        ("NiO", "Ni", "Ni", "Ni_pv"),
        ("TiO2", "Ti", "Ti_sv", "Ti_pv"),
        ("FeO", "Fe", "Fe", "Fe_pv"),
        ("WO3", "W", "W_sv", "W_pv"),
        ("MoO3", "Mo", "Mo_sv", "Mo_pv"),
        ("V2O5", "V", "V_sv", "V_pv"),
        ("Cu2O", "Cu", "Cu", "Cu_pv"),
        ("MgO", "Mg", "Mg", "Mg_pv"),
        ("Nb2O5", "Nb", "Nb_sv", "Nb_pv"),
        ("Bi2O3", "Bi", "Bi_d", "Bi"),
    ],
)
def test_record_follows_the_generator_where_mprelaxset_differs(formula, element, executed, mprelaxset):
    structure = composition_structure(formula)
    assert mprelaxset_symbols(structure)[element] == mprelaxset
    assert generator_symbols(structure)[0][element] == executed

    potcar = submission_potcar(structure)
    mapping = {row["species"]: row["potcar_symbol"] for row in potcar["species"]}
    assert mapping[element] == executed


# --- Ordering ---------------------------------------------------------------------


def test_species_and_symbols_follow_the_executed_poscar_order():
    # Input lists O first; the generator's POSCAR (what executes) decides the order.
    structure = Structure(
        Lattice.cubic(5.0),
        ["O", "O", "O", "O", "Li", "Fe", "P"],
        [[0.1 * index, 0.0, 0.0] for index in range(7)],
    )
    previews = generated_input_stage_previews(
        structure, desired_output_workflow_spec("energy_only"), potcar_functional="PBE_64"
    )
    executed_species = list(previews[0]["input_set"].poscar.site_symbols)
    executed_symbols = str(previews[0]["input_set"].potcar).split()

    potcar = submission_potcar(structure)
    assert [row["species"] for row in potcar["species"]] == executed_species
    assert potcar["symbols"] == executed_symbols
    assert [row["potcar_symbol"] for row in potcar["species"]] == executed_symbols


# --- Multi-stage representation -----------------------------------------------------


def test_multi_stage_record_lists_every_stage():
    potcar = submission_potcar(NIO, "electronic_band_structure")

    assert [(stage["stage_index"], stage["stage_type"], stage["theory"]) for stage in potcar["stages"]] == [
        (1, "relax", "pbe"),
        (2, "static", "hse06"),
        (3, "band_structure", "hse06"),
    ]
    assert all(stage["symbols"] == ["Ni", "O"] for stage in potcar["stages"])
    assert potcar["consistent_across_stages"] is True
    assert potcar["symbols"] == ["Ni", "O"]


def test_differing_stage_potcars_are_never_flattened():
    stages = [
        {"stage_index": 1, "stage_type": "relax", "theory": "pbe",
         "species": [{"species": "Ni", "potcar_symbol": "Ni"}], "symbols": ["Ni"]},
        {"stage_index": 2, "stage_type": "static", "theory": "hse06",
         "species": [{"species": "Ni", "potcar_symbol": "Ni_pv"}], "symbols": ["Ni_pv"]},
    ]
    record = aggregate_stage_potcars(stages)

    assert record["consistent_across_stages"] is False
    assert record["species"] is None
    assert record["symbols"] is None
    assert [stage["symbols"] for stage in record["stages"]] == [["Ni"], ["Ni_pv"]]


def test_differing_stage_potcars_reach_provenance_and_ui_per_stage(monkeypatch):
    original = potcar_record.stage_potcar_entry

    def diverging(preview):
        entry = original(preview)
        if entry["stage_index"] == 2:
            entry["species"] = [dict(row, potcar_symbol=row["potcar_symbol"] + "_x") for row in entry["species"]]
            entry["symbols"] = [symbol + "_x" for symbol in entry["symbols"]]
        return entry

    monkeypatch.setattr(potcar_record, "stage_potcar_entry", diverging)
    response = build_route(NIO, workflow="relaxed_structure")
    submission_spec = response.context["submission_spec"]

    for record in (submission_spec["potcar"], submission_spec["provenance"]["potcar"]):
        assert record["consistent_across_stages"] is False
        assert record["species"] is None
        assert record["symbols"] is None
        assert [stage["symbols"] for stage in record["stages"]] == [["Ni", "O"], ["Ni_x", "O_x"]]

    rendered = response.template.render(response.context)
    assert "Stage 1: Ni &rarr; Ni, O &rarr; O" in rendered
    assert "Stage 2: Ni &rarr; Ni_x, O &rarr; O_x" in rendered


# --- Adversarial: MPRelaxSet must never be the source again -------------------------


def test_record_does_not_use_mprelaxset(monkeypatch):
    # Poison both the MPRelaxSet POTCAR table and its instantiation. Neither may
    # influence the record; the class constant stays readable for the DFT+U
    # policy, which only reads MPRelaxSet's +U table.
    monkeypatch.setitem(MPRelaxSet.CONFIG["POTCAR"], "Ni", "Ni_MPRELAXSET_SENTINEL")

    class PoisonedMPRelaxSet(MPRelaxSet):
        def __post_init__(self):
            raise AssertionError("MPRelaxSet must not be used for the POTCAR record")

    monkeypatch.setattr(pymatgen_sets, "MPRelaxSet", PoisonedMPRelaxSet)

    response = build_route(NIO, workflow="energy_only")
    submission_spec = response.context["submission_spec"]
    assert submission_spec["potcar"]["symbols"] == ["Ni", "O"]
    assert "SENTINEL" not in json.dumps(submission_spec)
    assert "SENTINEL" not in response.template.render(response.context)


def test_no_parallel_potcar_table_or_mprelaxset_in_record_code():
    for relative in ("backend/submission.py", "backend/provenance.py", "backend/calculations/potcar_record.py"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "MPRelaxSet" not in source
        assert "MP_RECOMMENDED_POTCAR_SYMBOLS" not in source
        assert "Ni_pv" not in source


def test_unavailable_generators_give_an_explicit_record_not_a_guess():
    record = executable_potcar_record(object(), desired_output_workflow_spec("energy_only"), potcar_functional="PBE_64")
    assert record["status"] == "unavailable"
    assert record["species"] == [] and record["symbols"] == [] and record["stages"] == []
    assert record["reason"]
