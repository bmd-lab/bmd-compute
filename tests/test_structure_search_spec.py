"""Structure-search spec: well-formedness checks and defaults."""

import pytest

from backend.calculations.registry import CalculationValidationError
from backend.structure_search.spec import (
    SPEC_SCHEMA,
    StructureSearchSpec,
    StructureSearchSpecError,
    default_population_size,
)


def _spec(**overrides):
    values = dict(
        elements=("Mg", "Si", "O"),
        counts=(2, 1, 4),
        min_atoms=7,
        max_atoms=14,
        num_generations=40,
        stop_criterion=15,
        kresol_start=(0.12, 0.10, 0.08, 0.04, 0.04),
        walltime="120:00:00",
    )
    values.update(overrides)
    return StructureSearchSpec(**values)


@pytest.mark.parametrize(
    ("atoms", "expected"),
    [(1, 10), (2, 10), (4, 10), (7, 10), (8, 20), (12, 20), (14, 30), (20, 40), (22, 40),
     (23, 50), (27, 50), (28, 60), (30, 60), (40, 60), (200, 60)],
)
def test_default_population_size_is_twice_atoms_rounded_to_ten(atoms, expected):
    assert default_population_size(atoms) == expected


def test_population_defaults_follow_manual_rule():
    spec = _spec()
    assert spec.population_size is None
    assert spec.resolved_population_size == default_population_size(14) == 30
    assert spec.resolved_initial_population_size == 30


def test_initial_population_defaults_to_population_size():
    spec = _spec(population_size=16)
    assert spec.resolved_initial_population_size == 16
    assert _spec(population_size=16, initial_population_size=40).resolved_initial_population_size == 40


def test_values_are_normalised():
    spec = _spec(elements=["Mg", "Si", "O"], counts=["2", 1, 4.0], external_pressure_gpa="10",
                 kresol_start=[0.1, "0.05"], theory=" PBE ")
    assert spec.elements == ("Mg", "Si", "O")
    assert spec.counts == (2, 1, 4)
    assert spec.external_pressure_gpa == 10.0
    assert spec.kresol_start == (0.1, 0.05)
    assert spec.num_steps == 2
    assert spec.theory == "pbe"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"counts": (2, 1)}, "composition_length_mismatch"),
        ({"elements": (), "counts": ()}, "empty_composition"),
        ({"elements": ("Mg", "si", "O")}, "invalid_element_symbol"),
        ({"elements": ("Mg", "Mg", "O")}, "duplicate_element"),
        ({"counts": (2, 0, 4)}, "out_of_range"),
        ({"counts": (2, 1.5, 4)}, "invalid_type"),
        ({"min_atoms": 15}, "atom_range_inverted"),
        ({"min_atoms": 8, "max_atoms": 13}, "atom_range_excludes_composition"),
        ({"kresol_start": ()}, "no_relaxation_steps"),
        ({"kresol_start": (0.1, 0.0)}, "kresol_not_positive"),
        ({"external_pressure_gpa": float("nan")}, "out_of_range"),
        ({"num_generations": 0}, "out_of_range"),
        ({"stop_criterion": True}, "invalid_type"),
        ({"population_size": 0}, "out_of_range"),
        ({"spin_polarized": "no"}, "invalid_type"),
    ],
)
def test_malformed_specs_are_rejected(overrides, code):
    with pytest.raises(StructureSearchSpecError) as excinfo:
        _spec(**overrides)
    assert excinfo.value.code == code
    assert excinfo.value.diagnostic["code"] == code


@pytest.mark.parametrize("walltime", ["", "5 days", "00:00:00", "2401:00:00"])
def test_walltime_is_required_and_validated(walltime):
    with pytest.raises(CalculationValidationError):
        _spec(walltime=walltime)


def test_from_mapping_round_trip_and_composition_forms():
    data = {
        "composition": [{"element": "Mg", "count": 2}, {"element": "Si", "count": 1},
                        {"element": "O", "count": 4}],
        "min_atoms": 7,
        "max_atoms": 14,
        "num_generations": 40,
        "stop_criterion": 15,
        "kresol_start": [0.12, 0.10, 0.08, 0.04, 0.04],
        "walltime": "120:00:00",
        "external_pressure_gpa": 10,
    }
    spec = StructureSearchSpec.from_mapping(data)
    assert spec == _spec(external_pressure_gpa=10.0)
    assert StructureSearchSpec.from_mapping({**data, "composition": {"Mg": 2, "Si": 1, "O": 4}}) == spec
    record = spec.to_record()
    assert record["schema"] == SPEC_SCHEMA
    assert record["population_size"] == 30
    assert StructureSearchSpec.from_mapping(record).to_record() == record


def test_from_mapping_rejects_unknown_and_missing_fields():
    base = _spec().to_record()
    with pytest.raises(StructureSearchSpecError) as excinfo:
        StructureSearchSpec.from_mapping({**base, "pstress": 10})
    assert excinfo.value.code == "unknown_fields"
    missing = dict(base)
    del missing["walltime"]
    with pytest.raises(StructureSearchSpecError) as excinfo:
        StructureSearchSpec.from_mapping(missing)
    assert excinfo.value.code == "missing_field"


# --- formula-only defaults (single source for the preview) --------------------

from backend.structure_search import spec as spec_module  # noqa: E402


def test_default_search_spec_from_a_composition():
    spec = spec_module.default_search_spec({"Mg": 2, "Si": 1, "O": 4})
    assert (spec.elements, spec.counts) == (("Mg", "Si", "O"), (2, 1, 4))
    assert (spec.min_atoms, spec.max_atoms) == (7, 14)
    assert spec.stop_criterion == spec.max_atoms
    assert spec.num_generations == spec_module.DEFAULT_NUM_GENERATIONS
    assert spec.kresol_start == spec_module.DEFAULT_KRESOL_START
    assert spec.walltime == spec_module.DEFAULT_SEARCH_WALLTIME
    assert spec.external_pressure_gpa == 0.0
    assert spec.resolved_population_size == default_population_size(14)


@pytest.mark.parametrize(
    ("formula_atoms", "expected"),
    [(1, (1, 20)), (2, (2, 20)), (5, (5, 20)), (7, (7, 14)), (17, (17, 17)), (25, (25, 25))],
)
def test_default_cell_range(formula_atoms, expected):
    assert spec_module.default_cell_range(formula_atoms) == expected


def test_default_search_spec_overrides_and_blanks():
    spec = spec_module.default_search_spec(
        {"Si": 1}, min_atoms="8", max_atoms="16", num_generations="", walltime="72:00:00"
    )
    assert (spec.min_atoms, spec.max_atoms, spec.stop_criterion) == (8, 16, 16)
    assert spec.num_generations == spec_module.DEFAULT_NUM_GENERATIONS
    assert spec.walltime == "72:00:00"


def test_default_kpoints_match_the_ladder():
    from backend.structure_search.incar_policy import (
        BMD_STANDARD_MAX_KRESOL,
        BMD_STANDARD_STEPS,
        NUM_STEPS,
    )

    assert len(spec_module.DEFAULT_KRESOL_START) == NUM_STEPS
    for step in BMD_STANDARD_STEPS:
        assert spec_module.DEFAULT_KRESOL_START[step - 1] <= BMD_STANDARD_MAX_KRESOL


def test_defaults_are_recorded_as_provisional():
    record = spec_module.search_defaults_record()
    assert record["status"] == "provisional"
    assert record["planned_finalist_count"] == spec_module.PLANNED_FINALIST_COUNT == 5
    assert set(record["provisional_fields"]) == set(spec_module.PROVISIONAL_DEFAULTS)
