"""Structure-search admission (v1): fixed composition, PBE, non-magnetic, mapped POTCARs."""

import pytest

from backend.structure_search import admission
from backend.structure_search.admission import (
    ADMISSION_POLICY_ID,
    StructureSearchAdmissionError,
    require_admissible_search,
)
from backend.structure_search.potcar import bmd_potcar_symbol, bmd_potcar_symbols
from backend.structure_search.spec import StructureSearchSpec


def _spec(elements=("Si", "O"), counts=(1, 2), **overrides):
    values = dict(
        elements=elements,
        counts=counts,
        min_atoms=sum(counts),
        max_atoms=4 * sum(counts),
        num_generations=10,
        stop_criterion=5,
        kresol_start=(0.12, 0.10, 0.08, 0.04, 0.04),
        walltime="24:00:00",
    )
    values.update(overrides)
    return StructureSearchSpec(**values)


@pytest.mark.parametrize(
    ("elements", "counts"),
    [(("Si", "O"), (1, 2)), (("Mg", "Si", "O"), (2, 1, 4)), (("C",), (8,)), (("Ga", "As"), (1, 1))],
)
def test_supported_searches_are_admitted(elements, counts):
    spec = _spec(elements, counts)
    assert require_admissible_search(spec) is spec


def _code(spec):
    with pytest.raises(StructureSearchAdmissionError) as excinfo:
        require_admissible_search(spec)
    error = excinfo.value
    assert error.diagnostic["policy_id"] == ADMISSION_POLICY_ID
    assert error.diagnostic["code"] == error.code
    assert error.message
    return error


def test_only_fixed_composition_bulk_searches():
    error = _code(_spec(calculation_type=301))
    assert error.code == admission.UNSUPPORTED_CALCULATION_TYPE_CODE
    assert "301" in error.message


def test_only_pbe():
    assert _code(_spec(theory="hse06")).code == admission.UNSUPPORTED_THEORY_CODE


def test_spin_polarised_request_rejected():
    assert _code(_spec(spin_polarized=True)).code == admission.SPIN_POLARIZED_NOT_SUPPORTED_CODE


@pytest.mark.parametrize(("elements", "counts"), [(("Fe", "O"), (2, 3)), (("Ti", "O"), (1, 2)), (("Ce", "O"), (1, 2))])
def test_elements_bmd_treats_as_spin_polarised_are_rejected(elements, counts):
    error = _code(_spec(elements, counts))
    assert error.code == admission.SPIN_SCREEN_ELEMENT_CODE
    assert error.diagnostic["elements"] == [elements[0]]


def test_compositions_bmd_applies_dft_u_to_are_rejected():
    # W is not in the spin screen; WO2 (W4+, d2) triggers BMD's DFT+U rule.
    # (WO3, d0, is suppressed by that rule but now rejected by the SOC screen.)
    error = _code(_spec(("W", "O"), (1, 2)))
    assert error.code == admission.DFT_U_COMPOSITION_CODE
    assert error.diagnostic["elements"] == ["W"]


def test_unknown_element_rejected():
    error = _code(_spec(("Si", "Xx"), (1, 2)))
    assert error.code == admission.UNKNOWN_ELEMENT_CODE
    assert error.diagnostic["elements"] == ["Xx"]


def test_element_without_bmd_potcar_rejected():
    assert bmd_potcar_symbol("Og") is None
    error = _code(_spec(("Si", "Og"), (1, 1)))
    assert error.code == admission.NO_POTCAR_MAPPING_CODE


def test_potcar_symbols_follow_bmd_methodology_examples():
    # docs/methodology.md section 5.
    assert bmd_potcar_symbols(["Ni", "Fe", "Ti", "W", "Mo", "V", "Bi", "O"]) == {
        "Ni": "Ni", "Fe": "Fe", "Ti": "Ti_sv", "W": "W_sv",
        "Mo": "Mo_sv", "V": "V_sv", "Bi": "Bi_d", "O": "O",
    }
    with pytest.raises(KeyError):
        bmd_potcar_symbols(["Og"])


def test_potcar_symbols_match_the_executable_relax_generator():
    """The search uses exactly the POTCARs a BMD relaxation of the same elements writes."""

    from pymatgen.core import Lattice, Structure

    from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
    from backend.calculations.potcar_record import executable_potcar_record

    structure = Structure(
        Lattice.cubic(6.0),
        ["Mg", "Si", "O", "Ti"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    workflow = WorkflowSpec(stages=(StageSpec(stage_type=StageType.RELAX, theory=Theory.PBE),))
    record = executable_potcar_record(structure, workflow, potcar_functional="PBE_64")
    assert record["status"] == "resolved", record.get("reason")
    expected = {row["species"]: row["potcar_symbol"] for row in record["species"]}
    assert bmd_potcar_symbols(expected) == expected


# --- v2 scope: ambient pressure, plain PBE, size limits, ladder ---------------

@pytest.mark.parametrize("pressure", [10, 0.5, -1])
def test_nonzero_pressure_rejected(pressure):
    error = _code(_spec(external_pressure_gpa=pressure))
    assert error.code == admission.PRESSURE_NOT_SUPPORTED_CODE


def test_zero_pressure_admitted():
    spec = _spec(external_pressure_gpa=0)
    assert require_admissible_search(spec) is spec


@pytest.mark.parametrize(
    ("elements", "counts", "flagged"),
    [
        (("W", "O"), (1, 3), ["W"]),          # 5d transition metal
        (("Zr", "O"), (1, 2), ["Zr"]),        # 4d transition metal
        (("Bi", "Se"), (2, 3), ["Bi"]),       # heavy p-block
        (("La", "N"), (1, 1), ["La"]),        # lanthanide (not in the spin screen)
        (("Pb", "Te"), (1, 1), ["Pb"]),
    ],
)
def test_elements_bmd_treats_with_soc_are_rejected(elements, counts, flagged):
    error = _code(_spec(elements, counts))
    assert error.code == admission.SOC_SCREEN_ELEMENT_CODE
    assert error.diagnostic["elements"] == flagged


def test_soc_screen_is_bmds_own_policy():
    from backend.calculations.method_considerations import SOC_TRIGGER_ELEMENT_CLASSES

    # Every element BMD treats with SOC is refused, and nothing else is.
    for element in ("Sn", "Sb", "I", "Cs", "Ba", "Sr", "Te", "Se", "Ga", "In"):
        assert element not in SOC_TRIGGER_ELEMENT_CLASSES
        assert admission._soc_screen_elements([element]) == []
    for element in SOC_TRIGGER_ELEMENT_CLASSES:
        assert admission._soc_screen_elements([element]) == [element]


def test_too_many_elements_rejected():
    error = _code(_spec(("Li", "Na", "K", "Rb", "Cs"), (1, 1, 1, 1, 1)))
    assert error.code == admission.TOO_MANY_ELEMENTS_CODE


def test_cell_size_limit():
    assert require_admissible_search(_spec(("Si",), (1,), min_atoms=1, max_atoms=admission.MAX_CELL_ATOMS))
    error = _code(_spec(("Si",), (1,), min_atoms=1, max_atoms=admission.MAX_CELL_ATOMS + 1))
    assert error.code == admission.CELL_TOO_LARGE_CODE


def test_generation_limit():
    error = _code(_spec(num_generations=admission.MAX_GENERATIONS + 1))
    assert error.code == admission.TOO_MANY_GENERATIONS_CODE


@pytest.mark.parametrize(
    ("kresol", "code"),
    [
        ((0.1, 0.06), "step_count_mismatch"),
        ((0.12, 0.10, 0.08, 0.06, 0.04), "kresol_coarser_than_bmd_policy"),
        ((0.12, 0.10, 0.08, 0.04, 0.05), "kresol_coarser_than_bmd_policy"),
    ],
)
def test_kpoints_must_match_the_ladder(kresol, code):
    from backend.structure_search.incar_policy import IncarPolicyError

    with pytest.raises(IncarPolicyError) as excinfo:
        require_admissible_search(_spec(kresol_start=kresol))
    assert excinfo.value.code == code


def test_scope_record():
    record = admission.admission_scope_record()
    assert record["policy_version"] == admission.ADMISSION_POLICY_VERSION == 2
    assert record["external_pressure_gpa"] == 0.0
    assert record["spin_polarized"] is False
    assert any("SOC" in screen for screen in record["composition_screens"])
    assert record["limits"]["status"] == "provisional"
