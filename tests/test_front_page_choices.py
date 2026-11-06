"""Front page ("What would you like to calculate?") and the New structure preview."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

import main
from backend.calculations.registry import CalculationValidationError
from backend.structure_search.prediction_plan import (
    AdvancedSettings,
    FormulaError,
    build_prediction_plan,
    parse_formula,
)
import test_input_reference_contract as reference_cases


@pytest.fixture
def client(monkeypatch):
    def no_remote(*args, **kwargs):
        raise AssertionError("the front page and the New structure preview never touch POWER")

    for name in ("monitor_job", "prepare_remote_submission", "submit_remote_workflow",
                 "load_results_for_completed_job"):
        monkeypatch.setattr(main, name, no_remote)
    return TestClient(main.app)


def _choices(html):
    return re.findall(r'data-front-page-choice="([a-z_]+)"', html)


def test_home_asks_what_to_calculate_with_the_five_outputs(client):
    html = client.get("/").text
    assert "What would you like to calculate?" in html
    assert _choices(html) == [
        "new_structure",
        "relaxed_structure",
        "phonons",
        "electronic_band_structure",
        "electronic_dos",
    ]
    for label in ("New structure", "Relaxed structure", "Phonon band structure",
                  "Electronic band structure", "Electronic DOS"):
        assert label in html
    # The structure input is not shown until a structure-based output is chosen.
    assert 'id="structure-input-details"' not in html
    # Resuming a job stays one step away.
    assert 'action="/resume"' in html


@pytest.mark.parametrize(
    "output", ["relaxed_structure", "phonons", "electronic_band_structure", "electronic_dos"]
)
def test_structure_choice_opens_structure_input_with_that_output_selected(client, output):
    response = client.get(f"/calculate?output={output}")
    html = response.text
    assert response.status_code == 200
    assert response.context["selected_workflow"]["desired_output"] == output
    assert re.search(r'<details\s+id="structure-input-details"[^>]*\bopen\b', html, re.DOTALL)
    assert f'<input type="hidden" name="output" value="{output}">' in html
    assert f'data-chosen-output="{output}"' in html


@pytest.mark.parametrize("output", ["", "custom", "energy_only", "new_structure", "<script>"])
def test_unknown_choice_returns_to_the_question(client, output):
    html = client.get("/calculate", params={"output": output}).text
    assert "What would you like to calculate?" in html
    assert 'id="structure-input-details"' not in html


def test_analyze_keeps_the_chosen_output(client):
    response = client.post(
        "/analyze",
        data={"structure": reference_cases.SI_POSCAR, "fmt": "poscar", "output": "phonons"},
    )
    assert response.status_code == 200
    assert response.context["selected_workflow"]["desired_output"] == "phonons"
    assert response.context["summary"] is not None


def test_analyze_without_a_choice_keeps_the_previous_default(client):
    response = client.post("/analyze", data={"structure": reference_cases.SI_POSCAR, "fmt": "poscar"})
    assert response.context["selected_workflow"]["desired_output"] == "relaxed_energy"


def test_analyze_ignores_a_choice_that_is_not_on_the_front_page(client):
    response = client.post(
        "/analyze",
        data={"structure": reference_cases.SI_POSCAR, "fmt": "poscar", "output": "custom"},
    )
    assert response.context["selected_workflow"]["desired_output"] == "relaxed_energy"


def test_analyze_failure_keeps_the_chosen_output(client):
    response = client.post(
        "/analyze", data={"structure": "not a poscar", "fmt": "poscar", "output": "electronic_dos"}
    )
    assert response.status_code == 400
    assert response.context["selected_workflow"]["desired_output"] == "electronic_dos"
    assert '<input type="hidden" name="output" value="electronic_dos">' in response.text


def test_new_structure_asks_only_for_a_formula(client):
    html = client.get("/predict").text
    assert 'name="formula"' in html
    # Everything else is optional and collapsed.
    advanced = re.search(r"<details[^>]*data-prediction-advanced[^>]*>", html).group(0)
    assert "open" not in advanced.replace("data-prediction-advanced", "")
    assert 'name="structure"' not in html


def test_formula_preview_shows_search_and_planned_refinement_but_does_not_submit(client):
    response = client.post("/predict", data={"formula": "Mg2SiO4"})
    html = response.text
    assert response.status_code == 200
    plan = response.context["prediction"]["plan"]
    assert plan["formula"] == "Mg2SiO4"
    assert plan["search"]["cell_sizes"] == [7, 14]
    assert plan["search"]["energy_grade"] == "search-grade"
    assert plan["search"]["equivalent_to_bmd_adaptive_relaxation"] is False
    assert plan["refinement"]["status"] == "planned, not yet available"
    assert plan["submission_available"] is False
    assert "data-prediction-plan" in html
    assert re.search(r"<button[^>]*disabled[^>]*data-prediction-submit", html)
    assert "await BMD Lab sign-off" in html


def test_preview_text_separates_search_grade_ranking_from_bmd_refinement(client):
    html = client.post("/predict", data={"formula": "Si"}).text
    search = re.search(r'data-prediction-step="search">(.*?)</li>', html, re.S).group(1)
    refinement = re.search(r'data-prediction-step="refinement">(.*?)</li>', html, re.S).group(1)
    result = re.search(r'data-prediction-step="result">(.*?)</li>', html, re.S).group(1)
    assert "search-grade" in search
    assert "not BMD&#39;s geometry relaxation" in search or "not BMD's geometry relaxation" in search
    assert "planned, not yet available" in refinement
    assert "adaptive geometry relaxation" in refinement
    assert "cannot guarantee the global minimum" in result
    # The old wording that equated the ladder with BMD standards is gone.
    assert "whose last step uses BMD standards" not in html
    assert "same accuracy as Relaxed structure" not in html


def test_bad_formula_is_explained_and_redisplayed(client):
    response = client.post("/predict", data={"formula": "CO<b>"})
    assert response.status_code == 400
    assert "Formula Not Accepted" in response.text
    assert "CO&lt;b&gt;" in response.text
    assert response.context["prediction"]["plan"] is None


@pytest.mark.parametrize(
    ("formula", "code"),
    [
        ("Fe2O3", "structure_search_spin_screen_element"),
        ("Bi2Se3", "structure_search_soc_screen_element"),
        ("WO3", "structure_search_soc_screen_element"),
        ("LiNaKRbCs", "structure_search_too_many_elements"),
    ],
)
def test_out_of_scope_compositions_are_refused_by_the_spec(client, formula, code):
    response = client.post("/predict", data={"formula": formula})
    assert response.status_code == 400
    assert response.context["prediction"]["error"]["diagnostic"]["code"] == code
    assert response.context["prediction"]["plan"] is None


def test_preview_offers_no_pressure_and_ignores_a_posted_one(client):
    assert 'name="pressure_gpa"' not in client.get("/predict").text
    response = client.post("/predict", data={"formula": "Si", "pressure_gpa": "10"})
    assert response.context["prediction"]["plan"]["search"]["pressure_gpa"] == 0.0


def test_advanced_values_are_used_and_the_panel_stays_open(client):
    response = client.post(
        "/predict",
        data={"formula": "Si", "min_atoms": "8", "max_atoms": "16",
              "num_generations": "30", "search_walltime": "72:00:00"},
    )
    plan = response.context["prediction"]["plan"]
    assert plan["search"]["cell_sizes"] == list(range(8, 17))
    assert plan["search"]["num_generations"] == 30
    assert plan["search"]["stop_criterion"] == 16
    assert plan["search"]["walltime"] == "72:00:00"
    assert plan["search"]["walltime_display"] == "3 days"
    assert re.search(r"<details[^>]*data-prediction-advanced\s+open", response.text)


def test_advanced_placeholders_come_from_the_spec(client):
    from backend.structure_search import spec

    html = client.get("/predict").text
    assert f'placeholder="up to {spec.DEFAULT_MAX_CELL_ATOMS}"' in html
    assert f'placeholder="{spec.DEFAULT_NUM_GENERATIONS}"' in html
    assert f'placeholder="{spec.DEFAULT_SEARCH_WALLTIME}"' in html


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Si", {"Si": 1}),
        ("GaAs", {"Ga": 1, "As": 1}),
        ("Mg2SiO4", {"Mg": 2, "Si": 1, "O": 4}),
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("Al2(SO4)3", {"Al": 2, "S": 3, "O": 12}),
        ("SiSi", {"Si": 2}),
    ],
)
def test_parse_formula(text, expected):
    assert parse_formula(text) == expected


@pytest.mark.parametrize(
    "text, code",
    [
        ("", "empty_formula"),
        ("co", "unreadable_formula"),
        ("Xx", "unknown_element"),
        ("D2O", "unknown_element"),
        ("Fe0.5O", "invalid_count"),
        ("Si02", "invalid_count"),
        ("Si 2", "unreadable_formula"),
        ("(Si", "unbalanced_parentheses"),
        ("Si)", "unbalanced_parentheses"),
        ("()", "unreadable_formula"),
        ("2Si", "unreadable_formula"),
        ("Si" * 40, "formula_too_long"),
    ],
)
def test_parse_formula_rejects(text, code):
    with pytest.raises(FormulaError) as info:
        parse_formula(text)
    assert info.value.code == code


def test_formula_is_reduced_and_default_cells_hold_whole_formula_units():
    plan = build_prediction_plan("Si8")
    assert plan["formula"] == "Si"
    assert plan["reduced_from_entered"] is True
    assert plan["search"]["cell_sizes"] == list(range(1, 21))
    plan = build_prediction_plan("GaAs")
    assert plan["search"]["cell_sizes"] == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert plan["search"]["population_size"] == 40
    assert plan["search"]["stop_criterion"] == 20


def test_cell_limit_comes_from_admission():
    from backend.structure_search import admission

    with pytest.raises(CalculationValidationError) as info:
        build_prediction_plan("Si", AdvancedSettings(max_atoms=str(admission.MAX_CELL_ATOMS + 1)))
    assert info.value.diagnostic["code"] in {"out_of_range", admission.CELL_TOO_LARGE_CODE}


def test_atom_range_must_contain_a_whole_formula_unit():
    with pytest.raises(CalculationValidationError) as info:
        build_prediction_plan("Mg2SiO4", AdvancedSettings(min_atoms="8", max_atoms="13"))
    assert info.value.diagnostic["code"] == "atom_range_excludes_composition"


def test_search_resources_are_the_exclusive_node_preset():
    from backend.structure_search.spec import DEFAULT_SEARCH_WALLTIME

    resources = build_prediction_plan("Si")["search"]["resources"]
    assert resources["preset"] == "structure_search_exclusive_node"
    assert resources["exclusive"] is True
    assert resources["walltime"] == DEFAULT_SEARCH_WALLTIME


# --- the preview is the authoritative spec, not a copy ------------------------

@pytest.mark.parametrize("formula", ["Si", "GaAs", "Mg2SiO4", "Ca(OH)2", "Al2(SO4)3"])
def test_preview_plan_is_built_from_the_spec_defaults(formula):
    from backend.structure_search import admission, incar_policy
    from backend.structure_search.prediction_plan import reduce_composition
    from backend.structure_search.spec import default_search_spec, search_defaults_record

    plan = build_prediction_plan(formula)
    spec = default_search_spec(reduce_composition(parse_formula(formula)))
    admission.require_admissible_search(spec)
    assert plan["spec"] == spec.to_record()
    assert plan["defaults"] == search_defaults_record()
    assert plan["scope"] == admission.admission_scope_record()
    assert [row["kresol"] for row in plan["search"]["ladder"]] == list(spec.kresol_start)
    assert [row["role"] for row in plan["search"]["ladder"]] == list(incar_policy.STEP_ROLES)
    cutoffs = incar_policy.ladder_cutoffs(incar_policy.search_cutoff_record(spec.elements))
    assert [row["encut_ev"] for row in plan["search"]["ladder"]] == [
        cutoffs["early"], cutoffs["early"], cutoffs["relaxation"], cutoffs["relaxation"], cutoffs["ranking"],
    ]
    assert plan["refinement"]["candidates"] == search_defaults_record()["planned_finalist_count"]


def test_preview_module_defines_no_scientific_defaults_of_its_own():
    from backend.structure_search import prediction_plan

    for name in dir(prediction_plan):
        assert not name.startswith("DEFAULT_"), name
    for name in ("MAX_ELEMENTS", "MAX_CELL_ATOMS", "MAX_GENERATIONS", "MAX_FORMULA_ATOMS",
                 "MAX_PRESSURE_GPA", "POPULATION_SIZE_MAX"):
        assert not hasattr(prediction_plan, name), name
    assert not hasattr(prediction_plan, "default_population_size")


def test_every_structure_based_desired_output_and_custom_remain_available(client):
    """The front page leaves the existing Desired Output menu unchanged."""

    import test_input_reference_contract as reference_cases

    response = client.post(
        "/analyze", data={"structure": reference_cases.SI_POSCAR, "fmt": "poscar", "output": "phonons"}
    )
    offered = {option["value"] for option in response.context["calculation_options"]["desired_outputs"]}
    assert {"relaxed_energy", "relaxed_structure", "electronic_dos", "electronic_band_structure",
            "phonons", "custom"} <= offered
