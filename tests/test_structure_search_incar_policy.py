"""Structure-search INCAR ladder (proposal under BMD review, #65)."""

import math
from pathlib import Path

import pytest

from backend.calculations.resources import DEFAULT_NCORE
from backend.calculations.vasp_stage_definitions import (
    ENCUT_RELAX_DEFAULT,
    ENCUT_STATIC_FINAL_DEFAULT,
)
from backend.structure_search import incar_policy
from backend.structure_search.incar_policy import (
    BMD_EQUIVALENT_KRESOL,
    BMD_STANDARD_MAX_KRESOL,
    NUM_STEPS,
    IncarPolicyError,
    incar_ladder,
    incar_policy_record,
    incar_texts,
    render_incar,
    require_spec_matches_ladder,
    write_incar_files,
)
from backend.structure_search.spec import StructureSearchSpec


FIXTURES = Path(__file__).parent / "fixtures" / "structure_search"


def _settings(step):
    return incar_ladder()[step - 1].settings


@pytest.mark.parametrize("step", range(1, NUM_STEPS + 1))
def test_incar_matches_golden_file(step):
    assert incar_texts()[f"INCAR_{step}"] == (FIXTURES / f"INCAR_{step}").read_text(encoding="utf-8")


def test_ladder_shape_follows_the_manual():
    assert NUM_STEPS == 5
    assert [_settings(n)["ISIF"] for n in range(1, 6)] == [4, 4, 3, 3, 2]
    assert _settings(5)["NSW"] == 0
    assert _settings(5)["IBRION"] == -1
    assert (_settings(1)["IBRION"], _settings(1)["POTIM"]) == (2, 0.02)
    assert all((_settings(n)["IBRION"], _settings(n)["POTIM"]) == (1, 0.3) for n in (2, 3, 4))


def test_convergence_tightens_monotonically_to_bmd_standards():
    ediff = [_settings(n)["EDIFF"] for n in range(1, 6)]
    assert ediff == sorted(ediff, reverse=True)
    assert ediff[0] == 1e-2
    assert ediff[3] == ediff[4] == 1e-6  # BMD's EDIFF on every stage
    assert [_settings(n)["EDIFFG"] for n in (1, 2, 3)] == [1e-1, 1e-2, 1e-3]
    assert _settings(4)["EDIFFG"] == -0.01  # BMD relaxation force criterion


def test_cutoffs():
    # Volume-changing steps use the BMD relaxation cutoff (no Pulay stress
    # error); only the fixed-volume steps use the lower early cutoff.
    for step in incar_ladder():
        if step.settings["ISIF"] == 3:
            assert step.settings["ENCUT"] == ENCUT_RELAX_DEFAULT == 580
        if step.settings["ISIF"] == 4:
            assert step.settings["ENCUT"] == incar_policy.EARLY_STEP_ENCUT_NEEDS_LAB_DECISION
    assert incar_policy.EARLY_STEP_ENCUT_NEEDS_LAB_DECISION < ENCUT_RELAX_DEFAULT
    # The ranking energy uses the BMD Static Energy cutoff.
    assert _settings(5)["ENCUT"] == ENCUT_STATIC_FINAL_DEFAULT == 620


def test_smearing():
    for step in (1, 2, 3):
        assert _settings(step)["ISMEAR"] == 1  # Methfessel-Paxton, manual section 3.1
    assert [_settings(n)["SIGMA"] for n in (1, 2, 3)] == [0.10, 0.10, 0.05]
    final = dict(incar_policy.FINAL_SMEARING_NEEDS_LAB_DECISION)
    for step in (4, 5):
        assert {key: _settings(step)[key] for key in ("ISMEAR", "SIGMA")} == final


def test_nothing_depends_on_the_structure_and_no_pressure():
    for step in incar_ladder():
        settings = step.settings
        assert settings["ISPIN"] == 1
        assert settings["NCORE"] == DEFAULT_NCORE == 4
        assert settings["KPAR"] == incar_policy.FIXED_KPAR_NEEDS_LAB_DECISION
        assert settings["LWAVE"] is False and settings["LCHARG"] is False
        assert not incar_policy.FORBIDDEN_KEYS & set(settings)
    for text in incar_texts().values():
        for key in ("PSTRESS", "MAGMOM", "NBANDS"):
            assert key not in text


def test_render_refuses_forbidden_keys():
    with pytest.raises(ValueError, match="PSTRESS"):
        render_incar({"ENCUT": 580, "PSTRESS": 100})
    with pytest.raises(ValueError, match="NPAR"):
        render_incar({"NCORE": 4, "NPAR": 12})


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


def test_ladder_checks_parallel_tags_against_the_rank_count():
    assert incar_policy.DEFAULT_RANKS_PER_RELAXATION == 48
    assert incar_texts(ranks=24) == incar_texts()  # ranks never appear in the INCARs
    with pytest.raises(ValueError, match="NCORE"):
        incar_ladder(ranks=6)
    assert incar_policy_record(ranks=96)["ranks_per_relaxation"] == 96


def test_values_needing_lab_decision_are_listed():
    assert set(incar_policy.VALUES_NEEDING_LAB_DECISION) == {
        "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION",
        "ENCUT_BELOW_ENMAX_ACTION_NEEDS_LAB_DECISION",
        "VOLUME_STEP_ENCUT_NEEDS_LAB_DECISION",
        "EARLY_STEP_PREC_NEEDS_LAB_DECISION",
        "STEP_4_CONVERGENCE_NEEDS_LAB_DECISION",
        "FINAL_SMEARING_NEEDS_LAB_DECISION",
        "FIXED_KPAR_NEEDS_LAB_DECISION",
        "NUM_STEPS",
    }
    # Listing a value as undecided does not change it.
    assert incar_policy.ENCUT_BELOW_ENMAX_ACTION_NEEDS_LAB_DECISION == "refuse"
    assert incar_policy.VOLUME_STEP_ENCUT_NEEDS_LAB_DECISION == ENCUT_RELAX_DEFAULT == 580
    assert all(_settings(n)["PREC"] == "Accurate" for n in range(1, 6))
    assert {key: _settings(4)[key] for key in ("EDIFF", "EDIFFG")} == dict(
        incar_policy.STEP_4_CONVERGENCE_NEEDS_LAB_DECISION
    )
    record = incar_policy_record()
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


@pytest.mark.parametrize("enmax", [250.0, 400.0, 429.999, 430.0])
def test_encut_covering_enmax_is_accepted(enmax):
    incar_policy.require_encut_covers_enmax(enmax)


def test_early_steps_below_enmax_are_refused():
    # Li_sv: ENMAX 499.034 eV in BMD's PBE_64 metadata.
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_policy.require_encut_covers_enmax(499.034)
    error = excinfo.value
    assert error.code == incar_policy.ENCUT_BELOW_ENMAX_CODE == "encut_below_potcar_enmax"
    assert error.diagnostic["steps"] == [1, 2]
    assert error.diagnostic["largest_enmax_ev"] == 499.034
    assert "499.034" in error.message


def test_every_step_below_a_very_hard_potcar_is_listed():
    with pytest.raises(IncarPolicyError) as excinfo:
        incar_policy.require_encut_covers_enmax(700.0)
    assert excinfo.value.diagnostic["steps"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("enmax", [0, -1, float("nan"), float("inf")])
def test_enmax_must_be_positive_and_finite(enmax):
    with pytest.raises(ValueError):
        incar_policy.require_encut_covers_enmax(enmax)


def test_write_incar_files(tmp_path):
    paths = write_incar_files(tmp_path)
    assert [path.name for path in paths] == [f"INCAR_{n}" for n in range(1, 6)]
    for path in paths:
        assert path.read_text(encoding="utf-8") == (FIXTURES / path.name).read_text(encoding="utf-8")
