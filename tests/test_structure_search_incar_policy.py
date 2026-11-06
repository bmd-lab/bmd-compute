"""Structure-search INCAR ladder (#65; BMD lab decision of 5 Nov 2026)."""

import math
from pathlib import Path

import pytest
from conftest import synthetic_enmax_metadata, use_enmax_metadata

from backend.calculations.geometry_relaxation import (
    COMMON_RELAX_SETTINGS,
    CUTOFF_RULE_ID,
    PASS_CONVERGENCE,
    PROTOCOL_ID,
    resolve_cutoff_record,
)
from backend.calculations.resources import DEFAULT_NCORE
from backend.calculations.vasp_stage_definitions import ENCUT_STATIC_FINAL_DEFAULT
from backend.structure_search import incar_policy
from backend.structure_search.incar_policy import (
    BMD_EQUIVALENT_KRESOL,
    BMD_STANDARD_MAX_KRESOL,
    NUM_STEPS,
    IncarPolicyError,
    bmd_stage_settings,
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

KPAR = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4}  # lab decision, 5 Nov 2026


def _wide(step):
    return {"ISPIN": 1, "NCORE": 4, "KPAR": KPAR[step], "LWAVE": False, "LCHARG": False}


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
    assert relaxation == record["ENCUT"]
    assert record["rule"]["id"] == CUTOFF_RULE_ID == "bmd_compute.plane_wave_cutoff"
    assert ranking == max(ENCUT_STATIC_FINAL_DEFAULT, relaxation)
    assert early >= record["max_enmax_ev"]


def test_search_record_uses_bmd_potcar_symbols_and_metadata():
    record = search_cutoff_record(["Li", "O"])
    assert record == resolve_cutoff_record(["Li_sv", "O"], functional="PBE_64")
    assert record["max_enmax_ev"] == 499.034


def test_early_cutoff_rounds_up_to_50_ev():
    assert incar_policy.EARLY_STEP_ENCUT_MULTIPLIER == 1.0
    assert incar_policy.early_step_encut(245.345) == 250
    assert incar_policy.early_step_encut(400.0) == 400
    assert incar_policy.early_step_encut(400.001) == 450
    assert incar_policy.early_step_encut(499.034) == 500


# --- Steps 1-3: the USPEX manual's example INCARs (lab decision) -------------

MANUAL = [
    {"PREC": "Low", "EDIFF": 1e-2, "EDIFFG": 1e-1, "NSW": 65, "IBRION": 2, "POTIM": 0.02,
     "ISIF": 4, "ISMEAR": 1, "SIGMA": 0.10},
    {"PREC": "Normal", "EDIFF": 1e-3, "EDIFFG": 1e-2, "NSW": 55, "IBRION": 1, "POTIM": 0.30,
     "ISIF": 4, "ISMEAR": 1, "SIGMA": 0.08},
    {"PREC": "Normal", "EDIFF": 1e-3, "EDIFFG": 1e-2, "NSW": 65, "IBRION": 2, "POTIM": 0.02,
     "ISIF": 3, "ISMEAR": 1, "SIGMA": 0.07},
]


@pytest.mark.parametrize("step", [1, 2, 3])
def test_manual_steps_set_only_the_manual_tags_and_search_wide_tags(si_record, step):
    settings = dict(_settings(si_record, step))
    encut = settings.pop("ENCUT")
    assert encut == (250 if step < 3 else 350)
    assert settings == {**MANUAL[step - 1], **_wide(step)}
    for key in ("ALGO", "LREAL", "LASPH", "NELM", "ADDGRID"):
        assert key not in settings


# --- Step 4: screening relaxation with BMD pass-1 INCAR settings -------------

def test_step_4_uses_bmd_pass_1_incar_settings(si_record):
    settings = _settings(si_record, 4)
    pass_1, _ = bmd_stage_settings(si_record)
    # Every tag BMD's generator writes for pass 1, unchanged; the search adds only KPAR.
    assert "KPAR" not in pass_1
    assert {key: settings[key] for key in pass_1} == pass_1
    assert set(settings) == set(pass_1) | {"KPAR"}
    assert settings["KPAR"] == 4
    # ... which are the protocol's documented pass-1 settings.
    protocol = {key: value for key, value in COMMON_RELAX_SETTINGS.items() if value is not None}
    protocol.update(PASS_CONVERGENCE[1])
    assert {key: settings[key] for key in protocol} == protocol
    assert "ADDGRID" not in settings  # the protocol removes it
    assert settings["ENCUT"] == si_record["ENCUT"] == 350
    assert (settings["ALGO"], settings["NSW"], settings["NELM"]) == ("Fast", 99, 200)
    assert (settings["ISMEAR"], settings["SIGMA"]) == (0, 0.2)
    source = incar_policy_record(si_record)["step_4_incar_source"]
    assert source["id"] == PROTOCOL_ID and source["version"] == 1 and source["pass"] == 1
    assert source["search_wide_additions"] == {"KPAR": 4}
    assert "not the BMD protocol" in source["use"]


def test_ladder_is_labelled_search_grade_screening_not_bmd_relaxation(si_record):
    record = incar_policy_record(si_record)
    assert record["energy_grade"] == "search-grade"
    assert record["equivalent_to_bmd_adaptive_relaxation"] is False
    assert "step_4_protocol" not in record
    roles = [row["role"] for row in record["steps"]]
    assert roles == ["screening relaxation"] * 4 + ["search ranking single point"]
    for row in record["steps"]:
        # No step is labelled as the BMD protocol itself.
        assert not row["source"].startswith("BMD Geometry Optimisation pass 1 (")
    assert "screening only" in record["steps"][3]["source"]
    assert "search ranking only" in record["steps"][4]["source"]


def test_step_validity_follows_the_lab_decision(si_record):
    validity = {row["step"]: row for row in incar_policy.STEP_VALIDITY}
    assert set(validity) == {1, 2, 3, 4, 5}
    for step in (1, 2, 3):
        assert validity[step]["electronic_convergence"] is True
        assert validity[step]["ionic_convergence"] is False  # reaching NSW is allowed
    assert validity[4]["electronic_convergence"] and validity[4]["ionic_convergence"]
    assert validity[5]["electronic_convergence"] and validity[5]["valid_energy"]
    record = incar_policy_record(si_record)["step_validity"]
    assert record["criteria"] == [dict(row) for row in incar_policy.STEP_VALIDITY]
    # Recorded policy only: the launch bridge does not enforce it yet.
    assert record["enforced"] is False


# --- Step 5: ranking single point --------------------------------------------

def test_step_5_is_a_bmd_static_with_ranking_smearing(si_record):
    settings = _settings(si_record, 5)
    _, static = bmd_stage_settings(si_record)
    assert (settings["ISIF"], settings["IBRION"], settings["NSW"]) == (2, -1, 0)
    assert (settings["ISMEAR"], settings["SIGMA"]) == (0, 0.05)
    assert static["ISMEAR"] == -5  # BMD's static uses tetrahedron; the search does not
    assert settings["ENCUT"] == 620
    assert settings["EDIFF"] == 1e-6
    assert settings["ADDGRID"] is True and static["ADDGRID"] is True
    for key in ("ALGO", "PREC", "LREAL", "LASPH", "NELM"):
        assert settings[key] == static[key]
    for key in incar_policy.OUTPUT_ONLY_STATIC_KEYS:
        assert key not in settings
    assert {"NEDOS", "LORBIT", "LVTOT", "LAECHG", "LELF"} <= set(static)


def test_ranking_cutoff_ignores_the_adaptive_refinement_cutoff():
    # A managed BMD static after the adaptive protocol uses max(620, E0 + 150);
    # a search has no refinement, so step 5 uses max(620, E0).
    record = search_cutoff_record(["Li", "O"])
    _, static = bmd_stage_settings(record)
    assert static["ENCUT"] == 800
    assert _settings(record, 5)["ENCUT"] == 650


# --- All steps ----------------------------------------------------------------

def test_search_wide_tags_and_nothing_structure_dependent(si_record):
    for step in incar_ladder(si_record):
        settings = step.settings
        wide = _wide(step.number)
        assert {key: settings[key] for key in wide} == wide
        assert settings["NCORE"] == DEFAULT_NCORE == 4
        assert not incar_policy.FORBIDDEN_KEYS & set(settings)
    for text in incar_texts(si_record).values():
        for key in ("PSTRESS", "MAGMOM", "NBANDS"):
            assert key not in text


def test_ladder_shape():
    assert NUM_STEPS == 5
    assert len(incar_policy.STEP_SOURCES) == 5


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


def test_record_from_other_metadata_is_refused(si_record, monkeypatch, tmp_path):
    # The generator resolves pass 1 from the current metadata; a record made
    # from different ENMAX values no longer matches it.
    use_enmax_metadata(monkeypatch, tmp_path / "other.json", synthetic_enmax_metadata({"Si": 300.0}))
    stale = dict(si_record, metadata=dict(si_record["metadata"], digest="sha256:other"))
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_ladder(stale)
    assert excinfo.value.code == incar_policy.BMD_STAGE_MISMATCH_CODE


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


def test_kpar_by_step_follows_the_lab_decision(si_record):
    assert dict(incar_policy.KPAR_BY_STEP) == KPAR
    assert [_settings(si_record, n)["KPAR"] for n in range(1, 6)] == [1, 1, 1, 4, 4]
    assert incar_policy_record(si_record)["kpar_by_step"] == KPAR


@pytest.mark.parametrize("ranks", [16, 32, 48, 96, 192])
def test_rank_counts_valid_for_every_step(si_record, ranks):
    # KPAR 4 x NCORE 4: ranks must be a multiple of 16. At 48: 12 ranks per group.
    incar_ladder(si_record, ranks=ranks)


@pytest.mark.parametrize(
    ("ranks", "match"),
    [
        (24, "NCORE = 4 does not divide the 6"),   # valid for KPAR 1, not for KPAR 4
        (6, "NCORE = 4 does not divide the 6"),    # invalid already for KPAR 1
        (8, "NCORE = 4 does not divide the 2"),
        (50, "KPAR = 4 does not divide|NCORE"),
    ],
)
def test_rank_counts_invalid_for_some_step_are_refused(si_record, ranks, match):
    with pytest.raises(ValueError, match=match):
        incar_ladder(si_record, ranks=ranks)


def test_ladder_checks_parallel_tags_against_the_rank_count(si_record):
    assert incar_policy.DEFAULT_RANKS_PER_RELAXATION == 48
    assert incar_texts(si_record, ranks=96) == incar_texts(si_record)  # ranks never appear
    assert incar_policy_record(si_record, ranks=96)["ranks_per_relaxation"] == 96


def test_no_lab_decisions_are_open(si_record):
    assert incar_policy.VALUES_NEEDING_LAB_DECISION == {}
    for removed in ("EARLY_STEP_PREC_NEEDS_LAB_DECISION", "STEP_4_CONVERGENCE_NEEDS_LAB_DECISION",
                    "FINAL_SMEARING_NEEDS_LAB_DECISION", "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION",
                    "FIXED_KPAR_NEEDS_LAB_DECISION"):
        assert not hasattr(incar_policy, removed)


def test_policy_record(si_record):
    record = incar_policy_record(si_record)
    assert record["plane_wave_cutoff"] == si_record
    assert record["cutoffs"] == {"early": 250, "relaxation": 350, "ranking": 620}
    decisions = {decision["topic"]: decision for decision in record["lab_decisions"]}
    assert set(decisions) == {
        "INCAR ladder", "KPAR", "Role of the ladder", "Validity of a search step", "KPAR and NCORE",
    }
    assert decisions["INCAR ladder"]["date"] == decisions["KPAR"]["date"] == "2026-11-05"
    for topic in ("Role of the ladder", "Validity of a search step", "KPAR and NCORE"):
        assert decisions[topic]["date"] == "2026-11-06"
        assert decisions[topic]["decided_by"] == "Lee Burton"
    assert "must not be described as equivalent" in decisions["Role of the ladder"]["rule"]
    assert "final relaxation step must be consistent" in decisions["INCAR ladder"]["rule"]
    assert decisions["KPAR"]["rule"] == "KPAR = 1 on steps 1-3, KPAR = 4 on steps 4-5."
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
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_policy.require_encut_covers_enmax(300.0, si_record)
    error = excinfo.value
    assert error.code == incar_policy.ENCUT_BELOW_ENMAX_CODE == "encut_below_potcar_enmax"
    assert error.diagnostic["steps"] == [1, 2]
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_policy.require_encut_covers_enmax(700.0, si_record)
    assert excinfo.value.diagnostic["steps"] == [1, 2, 3, 4, 5]
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 0.8)
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
