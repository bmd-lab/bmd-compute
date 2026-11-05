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

SEARCH_WIDE = {"ISPIN": 1, "NCORE": 4, "KPAR": 1, "LWAVE": False, "LCHARG": False}


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
    assert settings == {**MANUAL[step - 1], **SEARCH_WIDE}
    for key in ("ALGO", "LREAL", "LASPH", "NELM", "ADDGRID"):
        assert key not in settings


# --- Step 4: BMD Geometry Optimisation pass 1, unchanged ----------------------

def test_step_4_is_bmd_geometry_optimisation_pass_1(si_record):
    settings = _settings(si_record, 4)
    pass_1, _ = bmd_stage_settings(si_record)
    # Every tag BMD's generator writes for pass 1, plus only the search-wide tags.
    assert {key: settings[key] for key in pass_1} == {**pass_1, **{k: v for k, v in SEARCH_WIDE.items() if k in pass_1}}
    assert set(settings) == set(pass_1) | set(SEARCH_WIDE)
    # ... which are the protocol's documented pass-1 settings.
    protocol = {key: value for key, value in COMMON_RELAX_SETTINGS.items() if value is not None}
    protocol.update(PASS_CONVERGENCE[1])
    assert {key: settings[key] for key in protocol} == protocol
    assert "ADDGRID" not in settings  # the protocol removes it
    assert settings["ENCUT"] == si_record["ENCUT"] == 350
    assert (settings["ALGO"], settings["NSW"], settings["NELM"]) == ("Fast", 99, 200)
    assert (settings["ISMEAR"], settings["SIGMA"]) == (0, 0.2)
    assert incar_policy_record(si_record)["step_4_protocol"] == {
        "id": PROTOCOL_ID, "version": 1, "pass": 1,
    }


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
        assert {key: settings[key] for key in SEARCH_WIDE} == SEARCH_WIDE
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


def test_ladder_checks_parallel_tags_against_the_rank_count(si_record):
    assert incar_policy.DEFAULT_RANKS_PER_RELAXATION == 48
    assert incar_texts(si_record, ranks=24) == incar_texts(si_record)  # ranks never appear
    with pytest.raises(ValueError, match="NCORE"):
        incar_ladder(si_record, ranks=6)
    assert incar_policy_record(si_record, ranks=96)["ranks_per_relaxation"] == 96


def test_only_kpar_still_needs_a_lab_decision(si_record):
    assert incar_policy.VALUES_NEEDING_LAB_DECISION == {"FIXED_KPAR_NEEDS_LAB_DECISION": 1}
    for removed in ("EARLY_STEP_PREC_NEEDS_LAB_DECISION", "STEP_4_CONVERGENCE_NEEDS_LAB_DECISION",
                    "FINAL_SMEARING_NEEDS_LAB_DECISION", "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION"):
        assert not hasattr(incar_policy, removed)


def test_policy_record(si_record):
    record = incar_policy_record(si_record)
    assert record["plane_wave_cutoff"] == si_record
    assert record["cutoffs"] == {"early": 250, "relaxation": 350, "ranking": 620}
    assert record["lab_decision"]["date"] == "2026-11-05"
    assert "final relaxation step must be consistent" in record["lab_decision"]["rule"]
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
