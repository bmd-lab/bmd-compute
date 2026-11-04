"""Structure-search INCAR ladder (proposal under BMD review, #65)."""

import math
from pathlib import Path

import pytest
from conftest import synthetic_enmax_metadata, use_enmax_metadata

from backend.calculations.geometry_relaxation import CUTOFF_RULE_ID, resolve_cutoff_record
from backend.calculations.resources import DEFAULT_NCORE
from backend.calculations.vasp_stage_definitions import ENCUT_STATIC_FINAL_DEFAULT
from backend.structure_search import incar_policy
from backend.structure_search.incar_policy import (
    BMD_EQUIVALENT_KRESOL,
    BMD_STANDARD_MAX_KRESOL,
    NUM_STEPS,
    IncarPolicyError,
    incar_ladder,
    incar_policy_record,
    incar_texts,
    ladder_cutoffs,
    render_incar,
    require_spec_matches_ladder,
    search_cutoff_record,
    write_incar_files,
)
from backend.structure_search.spec import StructureSearchSpec


FIXTURES = Path(__file__).parent / "fixtures" / "structure_search"

# Tests never read the production ENMAX metadata (tests/conftest.py). These
# SYNTHETIC test metadata carry the published PBE_64 values for the symbols
# the golden searches use (Si 245.345, Li_sv 499.034, O 400.0 eV).
SEARCH_ENMAX = {"Si": 245.345, "Li_sv": 499.034, "O": 400.0, "He": 478.896}

# Golden searches: (fixture folder, elements, early / relaxation / ranking ENCUT).
GOLDEN_SEARCHES = [
    ("incar_Si", ["Si"], (250, 350, 620)),
    ("incar_Li2O", ["Li", "O"], (500, 650, 650)),
]


@pytest.fixture(autouse=True)
def search_enmax_metadata(monkeypatch, tmp_path):
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", synthetic_enmax_metadata(SEARCH_ENMAX))


@pytest.fixture
def si_record():
    return search_cutoff_record(["Si"])


def _settings(record, step):
    return incar_ladder(record)[step - 1].settings


@pytest.mark.parametrize(("folder", "elements", "cutoffs"), GOLDEN_SEARCHES)
@pytest.mark.parametrize("step", range(1, NUM_STEPS + 1))
def test_incar_matches_golden_file(folder, elements, cutoffs, step):
    text = incar_texts(search_cutoff_record(elements))[f"INCAR_{step}"]
    assert text == (FIXTURES / folder / f"INCAR_{step}").read_text(encoding="utf-8")


@pytest.mark.parametrize(("folder", "elements", "cutoffs"), GOLDEN_SEARCHES)
def test_cutoffs_follow_bmd_rules(folder, elements, cutoffs):
    record = search_cutoff_record(elements)
    early, relaxation, ranking = cutoffs
    assert ladder_cutoffs(record) == {"early": early, "relaxation": relaxation, "ranking": ranking}
    encuts = [_settings(record, n)["ENCUT"] for n in range(1, 6)]
    assert encuts == [early, early, relaxation, relaxation, ranking]
    # Steps 3-4 use exactly the cutoff a BMD Geometry Optimisation resolves.
    assert relaxation == record["ENCUT"]
    assert record["rule"]["id"] == CUTOFF_RULE_ID == "bmd_compute.plane_wave_cutoff"
    # Step 5 is a BMD Static Energy after a relaxation: max(620, relaxation).
    assert ranking == max(ENCUT_STATIC_FINAL_DEFAULT, relaxation)
    # No step is below any POTCAR's ENMAX.
    assert early >= record["max_enmax_ev"]


def test_search_record_uses_bmd_potcar_symbols_and_metadata():
    record = search_cutoff_record(["Li", "O"])
    assert record == resolve_cutoff_record(["Li_sv", "O"], functional="PBE_64")
    assert record["max_enmax_ev"] == 499.034


def test_si8_cutoffs_are_cheaper_than_the_old_fixed_values(si_record):
    # Previously 430 / 580 / 620 eV for every search.
    assert ladder_cutoffs(si_record) == {"early": 250, "relaxation": 350, "ranking": 620}


def test_early_cutoff_rounds_up_to_50_ev():
    assert incar_policy.early_step_encut(245.345) == 250
    assert incar_policy.early_step_encut(400.0) == 400
    assert incar_policy.early_step_encut(400.001) == 450
    assert incar_policy.early_step_encut(499.034) == 500


def test_addgrid_follows_bmd_stages(si_record):
    # Relaxation protocol: no ADDGRID (steps 3-4). Static Energy keeps it (step 5).
    assert [("ADDGRID" in _settings(si_record, n)) for n in range(1, 6)] == [True, True, False, False, True]
    assert _settings(si_record, 5)["ADDGRID"] is True


def test_ladder_shape_follows_the_manual(si_record):
    assert NUM_STEPS == 5
    assert [_settings(si_record, n)["ISIF"] for n in range(1, 6)] == [4, 4, 3, 3, 2]
    assert _settings(si_record, 5)["NSW"] == 0
    assert _settings(si_record, 5)["IBRION"] == -1
    assert (_settings(si_record, 1)["IBRION"], _settings(si_record, 1)["POTIM"]) == (2, 0.02)
    assert all(
        (_settings(si_record, n)["IBRION"], _settings(si_record, n)["POTIM"]) == (1, 0.3)
        for n in (2, 3, 4)
    )


def test_convergence_tightens_monotonically_to_bmd_standards(si_record):
    ediff = [_settings(si_record, n)["EDIFF"] for n in range(1, 6)]
    assert ediff == sorted(ediff, reverse=True)
    assert ediff[0] == 1e-2
    assert ediff[3] == ediff[4] == 1e-6  # BMD's EDIFF on every stage
    assert [_settings(si_record, n)["EDIFFG"] for n in (1, 2, 3)] == [1e-1, 1e-2, 1e-3]
    assert _settings(si_record, 4)["EDIFFG"] == -0.01  # BMD pass-1 force criterion


def test_smearing(si_record):
    for step in (1, 2, 3):
        assert _settings(si_record, step)["ISMEAR"] == 1  # Methfessel-Paxton, manual section 3.1
    assert [_settings(si_record, n)["SIGMA"] for n in (1, 2, 3)] == [0.10, 0.10, 0.05]
    final = dict(incar_policy.FINAL_SMEARING_NEEDS_LAB_DECISION)
    for step in (4, 5):
        assert {key: _settings(si_record, step)[key] for key in ("ISMEAR", "SIGMA")} == final


def test_nothing_depends_on_the_structure_and_no_pressure(si_record):
    for step in incar_ladder(si_record):
        settings = step.settings
        assert settings["ISPIN"] == 1
        assert settings["NCORE"] == DEFAULT_NCORE == 4
        assert settings["KPAR"] == incar_policy.FIXED_KPAR_NEEDS_LAB_DECISION
        assert settings["LWAVE"] is False and settings["LCHARG"] is False
        assert not incar_policy.FORBIDDEN_KEYS & set(settings)
    for text in incar_texts(si_record).values():
        for key in ("PSTRESS", "MAGMOM", "NBANDS"):
            assert key not in text


def test_render_refuses_forbidden_keys():
    with pytest.raises(ValueError, match="PSTRESS"):
        render_incar({"ENCUT": 580, "PSTRESS": 100})
    with pytest.raises(ValueError, match="NPAR"):
        render_incar({"NCORE": 4, "NPAR": 12})


@pytest.mark.parametrize(
    "record",
    [None, {}, {"rule": {"id": "other", "version": 1}, "ENCUT": 350, "max_enmax_ev": 245.345}],
)
def test_ladder_requires_a_bmd_cutoff_record(record):
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_ladder(record)
    assert excinfo.value.code == incar_policy.INVALID_CUTOFF_RECORD_CODE


@pytest.mark.parametrize(
    ("kpar", "ncore", "ranks"),
    [(1, 4, 48), (4, 4, 48), (2, 4, 48), (1, 4, 24), (1, 4, 96), (4, 4, 96)],
)
def test_parallel_tags_valid(kpar, ncore, ranks):
    incar_policy.validate_parallel_tags(kpar, ncore, ranks)


@pytest.mark.parametrize(
    ("kpar", "ncore", "ranks", "match"),
    [
        (5, 4, 48, "KPAR = 5 does not divide"),
        (8, 4, 48, "NCORE = 4 does not divide the 6"),
        (1, 4, 6, "NCORE = 4 does not divide the 6"),
        (0, 4, 48, "KPAR must be"),
        (1, 4, True, "ranks must be"),
    ],
)
def test_parallel_tags_invalid(kpar, ncore, ranks, match):
    with pytest.raises(ValueError, match=match):
        incar_policy.validate_parallel_tags(kpar, ncore, ranks)


def test_ladder_checks_parallel_tags_against_the_rank_count(si_record):
    assert incar_policy.DEFAULT_RANKS_PER_RELAXATION == 48
    assert incar_texts(si_record, ranks=24) == incar_texts(si_record)  # ranks never appear
    with pytest.raises(ValueError, match="NCORE"):
        incar_ladder(si_record, ranks=6)
    assert incar_policy_record(si_record, ranks=96)["ranks_per_relaxation"] == 96


def test_values_needing_lab_decision_are_listed(si_record):
    assert set(incar_policy.VALUES_NEEDING_LAB_DECISION) == {
        "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION",
        "EARLY_STEP_PREC_NEEDS_LAB_DECISION",
        "STEP_4_CONVERGENCE_NEEDS_LAB_DECISION",
        "FINAL_SMEARING_NEEDS_LAB_DECISION",
        "FIXED_KPAR_NEEDS_LAB_DECISION",
        "NUM_STEPS",
    }
    assert incar_policy.EARLY_STEP_ENCUT_NEEDS_LAB_DECISION == 1.0
    assert all(_settings(si_record, n)["PREC"] == "Accurate" for n in range(1, 6))
    assert {key: _settings(si_record, 4)[key] for key in ("EDIFF", "EDIFFG")} == dict(
        incar_policy.STEP_4_CONVERGENCE_NEEDS_LAB_DECISION
    )


def test_policy_record_carries_the_resolved_cutoff_record(si_record):
    record = incar_policy_record(si_record)
    assert record["plane_wave_cutoff"] == si_record
    assert record["cutoffs"] == {"early": 250, "relaxation": 350, "ranking": 620}
    assert record["values_needing_lab_decision"] == incar_policy.VALUES_NEEDING_LAB_DECISION
    assert [row["step"] for row in record["steps"]] == [1, 2, 3, 4, 5]


def test_bmd_kresol_equivalent():
    # Reciprocal density 64 -> spacing 0.25 1/A -> 0.0398 in units of 2 pi / A.
    assert BMD_EQUIVALENT_KRESOL == pytest.approx(0.25 / (2 * math.pi))
    assert BMD_EQUIVALENT_KRESOL <= BMD_STANDARD_MAX_KRESOL < BMD_EQUIVALENT_KRESOL * 1.06


def test_bmd_kresol_matches_pymatgen_meshes():
    """The pymatgen meshes BMD uses have spacings close to the KresolStart limit."""

    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp.inputs import Kpoints

    for a in (3.0, 5.0, 8.0, 12.0):
        structure = Structure(Lattice.cubic(a), ["Si"], [[0, 0, 0]])
        divisions = Kpoints.automatic_density_by_vol(structure, 64).kpts[0]
        spacing_2pi = (1 / a) / divisions[0]
        assert abs(spacing_2pi - BMD_STANDARD_MAX_KRESOL) < 0.003


def _spec(kresol):
    return StructureSearchSpec(
        elements=("Si", "O"), counts=(1, 2), min_atoms=3, max_atoms=12, num_generations=10,
        stop_criterion=5, kresol_start=kresol, walltime="24:00:00",
    )


def test_spec_must_match_the_ladder():
    require_spec_matches_ladder(_spec((0.12, 0.10, 0.08, 0.04, 0.04)))
    with pytest.raises(IncarPolicyError) as excinfo:
        require_spec_matches_ladder(_spec((0.12, 0.10, 0.08, 0.04)))
    assert excinfo.value.code == "step_count_mismatch"
    with pytest.raises(IncarPolicyError) as excinfo:
        require_spec_matches_ladder(_spec((0.12, 0.10, 0.08, 0.06, 0.04)))
    assert excinfo.value.code == "kresol_coarser_than_bmd_policy"
    assert excinfo.value.diagnostic["steps"] == [4]


@pytest.mark.parametrize(("folder", "elements", "cutoffs"), GOLDEN_SEARCHES)
def test_encut_covers_the_search_potcars(folder, elements, cutoffs):
    record = search_cutoff_record(elements)
    incar_policy.require_encut_covers_enmax(record["max_enmax_ev"], record)


def test_enmax_check_is_a_safety_net(si_record, monkeypatch):
    # An ENMAX above a step's cutoff (for example from a mismatched POTCAR)
    # is refused, never adjusted.
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_policy.require_encut_covers_enmax(300.0, si_record)
    error = excinfo.value
    assert error.code == incar_policy.ENCUT_BELOW_ENMAX_CODE == "encut_below_potcar_enmax"
    assert error.diagnostic["steps"] == [1, 2]
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_policy.require_encut_covers_enmax(700.0, si_record)
    assert excinfo.value.diagnostic["steps"] == [1, 2, 3, 4, 5]
    # A multiplier below 1 would put steps 1-2 below ENMAX.
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION", 0.8)
    with pytest.raises(IncarPolicyError):
        incar_policy.require_encut_covers_enmax(si_record["max_enmax_ev"], si_record)


@pytest.mark.parametrize("enmax", [0, -1, float("nan"), float("inf")])
def test_enmax_must_be_positive_and_finite(si_record, enmax):
    with pytest.raises(ValueError):
        incar_policy.require_encut_covers_enmax(enmax, si_record)


def test_copied_potcars_must_match_the_cutoff_record():
    record = search_cutoff_record(["Li", "O"])
    matching = [
        {"element": "Li", "potcar_symbol": "Li_sv", "enmax": 499.034},
        {"element": "O", "potcar_symbol": "O", "enmax": 400.0},
    ]
    incar_policy.require_potcars_match_cutoff_record(matching, record)
    for wrong in (
        [matching[0], {**matching[1], "enmax": 434.0}],          # different ENMAX
        [matching[0]],                                           # POTCAR missing
        [{**matching[0], "potcar_symbol": "Li"}, matching[1]],   # different dataset
    ):
        with pytest.raises(IncarPolicyError) as excinfo:
            incar_policy.require_potcars_match_cutoff_record(wrong, record)
        assert excinfo.value.code == incar_policy.POTCAR_ENMAX_MISMATCH_CODE


def test_write_incar_files(tmp_path, si_record):
    paths = write_incar_files(tmp_path, si_record)
    assert [path.name for path in paths] == [f"INCAR_{n}" for n in range(1, 6)]
    for path in paths:
        expected = (FIXTURES / "incar_Si" / path.name).read_text(encoding="utf-8")
        assert path.read_text(encoding="utf-8") == expected
