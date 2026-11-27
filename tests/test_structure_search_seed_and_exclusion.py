"""S1 and S2 for the blind Si8 benchmark: the USPEX random seed and per-search node exclusion.

S1: an optional ``random_seed`` in the search spec, written to ``INPUT.txt`` as
``<n> : fixRndSeed`` and recorded as *requested* and *written*, distinct from
whether USPEX reproduces a search (not verified).

S2: an optional per-search ``excluded_nodes`` (SLURM host list), added to the
deployment's ``STRUCTURE_SEARCH_EXCLUDED_NODES`` and never replacing it, carried
by the resource request, the ``#SBATCH`` directives, the plan digest and the
resource acknowledgement.

Both are optional: a request that omits them plans exactly as in v1.3.1 (same
record, same digest, same files). Nothing here contacts POWER or SLURM.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import (
    MAX_ADDITIONAL_EXCLUDED_NODES,
    StructureSearchResources,
    expand_hostlist,
    merge_excluded_nodes,
    structure_search_resources,
)
from backend.structure_search import job as search_job
from backend.structure_search.input_writer import (
    SearchExecutionSettings,
    read_input_keywords,
    render_input_text,
    written_random_seed,
)
from backend.structure_search.planning import StructureSearchPlanRequest, plan_structure_search
from backend.structure_search.prediction_plan import AdvancedSettings
from backend.structure_search.search_plan import (
    INVALID_PLAN_CODE,
    PLAN_DIGEST_MISMATCH_CODE,
    PREPARED_FILE,
    SearchPlanError,
    StructureSearchPlan,
    build_search_plan,
    build_search_plan_from_formula,
    load_plan,
    plan_digest,
    prepare_search_directory,
    render_prepared_files,
    require_current_plan,
    verify_prepared_directory,
)
from backend.structure_search.spec import (
    RANDOM_SEED_MAX,
    StructureSearchSpec,
    StructureSearchSpecError,
    default_search_spec,
)
from conftest import synthetic_enmax_metadata, use_enmax_metadata

FIXTURES = Path(__file__).parent / "fixtures" / "structure_search"
GOLDEN_SI8 = (FIXTURES / "INPUT_Si8_0GPa.txt").read_text()
BRIDGE = "/bmd/placeholder/bmd_bridge.sh"
SEARCH_ENMAX = {"Si": 245.345, "Li_sv": 499.034, "O": 400.0}
SEED = 20261126
SHARED_NODE = {"STRUCTURE_SEARCH_PARALLEL_RELAXATIONS": 3, "STRUCTURE_SEARCH_RANKS_PER_RELAXATION": 16,
               "STRUCTURE_SEARCH_EXCLUSIVE_NODE": False, "STRUCTURE_SEARCH_JOB_MEMORY": "32G"}
# Sections of a plan that S1/S2 must never change (the scientific methodology).
METHODOLOGY_SECTIONS = ("screening_and_ranking", "incar_files", "kpoints", "potcar", "candidate_validity",
                        "admission", "composition", "runtime", "execution", "refinement")


@pytest.fixture(autouse=True)
def search_enmax_metadata(monkeypatch, tmp_path):
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", synthetic_enmax_metadata(SEARCH_ENMAX))


def _configured(monkeypatch, values):
    from backend import config

    for name, value in values.items():
        monkeypatch.setattr(config, name, value)


def _si8_spec(**changes):
    # The input-writer golden search (tests/test_structure_search_input_writer.py).
    spec = StructureSearchSpec(
        elements=("Si",), counts=(8,), min_atoms=8, max_atoms=8,
        num_generations=20, stop_criterion=8, kresol_start=(0.1, 0.08, 0.06, 0.04, 0.04),
        walltime="48:00:00", population_size=12, initial_population_size=24,
    )
    return replace(spec, **changes) if changes else spec


def _settings():
    return SearchExecutionSettings(bridge_path=BRIDGE, num_parallel_calcs=4, sleep_seconds=30)


def _benchmark_request(**changes):
    # The benchmark's planning request (design and STATUS.md C1): Si, 8 atoms, 30 generations; the
    # population (20/20) and stopCrit (8) are the manual defaults the planner applies itself.
    values = {"formula": "Si", "min_atoms": 8, "max_atoms": 8, "num_generations": 30, "walltime": "168:00:00",
              "random_seed": SEED, "excluded_nodes": "compute-0-270"}
    values.update(changes)
    return StructureSearchPlanRequest(**values)


# --- S1: the spec field ------------------------------------------------------------------


def test_the_seed_is_optional_and_absent_from_the_v1_spec_record():
    spec = _si8_spec()
    assert spec.random_seed is None
    assert "random_seed" not in spec.to_record()
    assert StructureSearchSpec.from_mapping(spec.to_record()) == spec


@pytest.mark.parametrize("value, expected", [(1, 1), (SEED, SEED), (RANDOM_SEED_MAX, RANDOM_SEED_MAX),
                                             ("20261126", SEED), (" 42 ", 42)])
def test_valid_seeds_are_accepted_and_recorded(value, expected):
    spec = _si8_spec(random_seed=value)
    assert spec.random_seed == expected
    assert spec.to_record()["random_seed"] == expected
    assert StructureSearchSpec.from_mapping(spec.to_record()) == spec


@pytest.mark.parametrize("value, code", [
    (0, "out_of_range"), (-2000, "out_of_range"), (RANDOM_SEED_MAX + 1, "out_of_range"),
    (True, "invalid_type"), (False, "invalid_type"), (1.0, "invalid_type"), (1.5, "invalid_type"),
    ("-5", "invalid_type"), ("1e3", "invalid_type"), ("", "invalid_type"), ("seven", "invalid_type"),
    ([1], "invalid_type"),
])
def test_invalid_seeds_are_refused(value, code):
    with pytest.raises(StructureSearchSpecError) as info:
        _si8_spec(random_seed=value)
    assert info.value.code == code
    assert info.value.diagnostic["field"] == "random_seed"


def test_default_spec_takes_an_optional_seed_and_blank_means_none():
    assert default_search_spec({"Si": 1}).random_seed is None
    assert default_search_spec({"Si": 1}, random_seed="").random_seed is None
    assert default_search_spec({"Si": 1}, random_seed=SEED).random_seed == SEED


# --- S1: INPUT.txt -----------------------------------------------------------------------


def test_without_a_seed_input_txt_is_the_v1_golden_file():
    text = render_input_text(_si8_spec(), _settings())
    assert text == GOLDEN_SI8
    assert "fixRndSeed" not in text
    assert written_random_seed(text) is None


def test_a_seed_adds_exactly_one_fixrndseed_line_after_stopcrit():
    text = render_input_text(_si8_spec(random_seed=SEED), _settings())
    golden = GOLDEN_SI8.splitlines()
    lines = text.splitlines()
    position = golden.index("8 : stopCrit") + 1
    assert lines == golden[:position] + [f"{SEED} : fixRndSeed"] + golden[position:]
    assert read_input_keywords(text)["fixRndSeed"] == [str(SEED)]
    assert written_random_seed(text) == SEED


# --- S1: the plan, its digest and the prepared files -------------------------------------


def test_a_seed_changes_only_the_spec_and_the_seed_record():
    plain = build_search_plan(_si8_spec())
    seeded = build_search_plan(_si8_spec(random_seed=SEED))
    other = build_search_plan(_si8_spec(random_seed=SEED + 1))
    assert len({plain.digest, seeded.digest, other.digest}) == 3
    for section in (*METHODOLOGY_SECTIONS, "resources", "request"):
        assert seeded.record[section] == plain.record[section], section
    search_plain = plain.record["search"]
    search_seeded = seeded.record["search"]
    assert set(search_seeded) - set(search_plain) == {"random_seed"}
    assert {k: v for k, v in search_seeded["spec"].items() if k != "random_seed"} == search_plain["spec"]
    record = search_seeded["random_seed"]
    assert record["requested"] == SEED and record["uspex_keyword"] == "fixRndSeed"
    assert record["input_txt_line"] == f"{SEED} : fixRndSeed"
    assert record["effective_rng_determinism"].startswith("not verified")


def test_seeded_preparation_records_requested_and_written_seed(tmp_path):
    plan = build_search_plan(_si8_spec(random_seed=SEED))
    manifest = prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    assert manifest["random_seed"] == {
        "requested": SEED,
        "written_in_input_txt": SEED,
        "effective_rng_determinism": plan.record["search"]["random_seed"]["effective_rng_determinism"],
    }
    text = (tmp_path / "search" / "INPUT.txt").read_text()
    assert written_random_seed(text) == SEED
    assert verify_prepared_directory(tmp_path / "search") == manifest
    # The plan stored in the prepared directory round-trips with its seed.
    stored = load_plan(json.loads((tmp_path / "search" / "search_plan.json").read_text()))
    assert stored.digest == plan.digest


def test_unseeded_preparation_has_no_seed_record(tmp_path):
    manifest = prepare_search_directory(build_search_plan(_si8_spec()), tmp_path / "search", bridge_path=BRIDGE)
    assert "random_seed" not in manifest
    assert "fixRndSeed" not in (tmp_path / "search" / "INPUT.txt").read_text()


def test_an_edited_seed_is_a_digest_mismatch():
    document = build_search_plan(_si8_spec(random_seed=SEED)).to_document()
    document["search"]["spec"]["random_seed"] = SEED + 1
    with pytest.raises(SearchPlanError) as info:
        load_plan(document)
    assert info.value.code == PLAN_DIGEST_MISMATCH_CODE


def test_a_seed_record_that_contradicts_the_spec_is_refused_even_with_a_recomputed_digest():
    record = build_search_plan(_si8_spec(random_seed=SEED)).record
    record["search"]["random_seed"]["requested"] = SEED + 1
    forged = StructureSearchPlan(canonical=json.dumps(record, sort_keys=True, separators=(",", ":")),
                                 digest=plan_digest(record))
    with pytest.raises(SearchPlanError) as info:
        render_prepared_files(forged, bridge_path=BRIDGE)
    assert info.value.code == INVALID_PLAN_CODE
    record = build_search_plan(_si8_spec()).record
    record["search"]["spec"]["random_seed"] = SEED  # a seed with no seed record
    forged = StructureSearchPlan.from_record(record)
    with pytest.raises(SearchPlanError):
        render_prepared_files(forged, bridge_path=BRIDGE)


def test_seeded_plans_are_current_and_rebuild_identically():
    for plan in (build_search_plan(_si8_spec(random_seed=SEED)),
                 build_search_plan_from_formula("Si", AdvancedSettings(random_seed=SEED))):
        require_current_plan(plan)


def test_equivalent_seed_spellings_give_one_plan_and_keep_the_entered_text():
    as_int = build_search_plan_from_formula("Si", AdvancedSettings(random_seed=SEED))
    as_text = build_search_plan_from_formula("Si", AdvancedSettings(random_seed=f" {SEED} "))
    assert as_int.digest == as_text.digest
    assert as_text.provenance["request"]["advanced_overrides_as_entered"]["random_seed"] == f" {SEED} "
    assert as_text.record["request"]["advanced_overrides"]["random_seed"] == SEED


# --- omitted fields: v1.3.1 behaviour -----------------------------------------------------


@pytest.mark.parametrize("formula, advanced", [
    ("Si", {}), ("SiO2", {"max_atoms": 12, "num_generations": 30}), ("GaAs", {"walltime": "48:00:00"}),
])
def test_blank_optional_fields_plan_exactly_as_when_omitted(formula, advanced):
    omitted = build_search_plan_from_formula(formula, AdvancedSettings(**advanced))
    for blank in (None, "", "   "):
        given = build_search_plan_from_formula(
            formula, AdvancedSettings(**advanced, random_seed=blank, excluded_nodes=blank))
        assert given.digest == omitted.digest
        assert given.canonical == omitted.canonical
    overrides = omitted.record["request"]["advanced_overrides"]
    assert set(overrides) == {"min_atoms", "max_atoms", "num_generations", "walltime"}  # the v1 keys only
    assert "random_seed" not in omitted.record["search"]
    resources = omitted.record["resources"]["request"]
    assert set(resources) == {"preset", "nodes", "ntasks", "exclusive", "mem", "walltime", "partition",
                              "account", "qos", "exclude"}


# --- S2: host lists and the merge -----------------------------------------------------------


@pytest.mark.parametrize("text, names", [
    ("compute-0-270", ["compute-0-270"]),
    ("compute-0-[332-333]", ["compute-0-332", "compute-0-333"]),
    ("compute-0-[332-333],compute-0-7", ["compute-0-332", "compute-0-333", "compute-0-7"]),
    ("n[01-03,7]", ["n01", "n02", "n03", "n7"]),
    ("", []),
])
def test_expand_hostlist(text, names):
    assert expand_hostlist(text) == names


@pytest.mark.parametrize("text", ["compute 0", "a;b", "[1-2]", "n[5-3]", "n[1-x]", "n[1--2]", "n,", "n]1["])
def test_expand_hostlist_refuses_what_is_not_a_host_list(text):
    with pytest.raises(ValueError):
        expand_hostlist(text)


@pytest.mark.parametrize("default, additional, effective, canonical", [
    ("compute-0-[332-333]", "compute-0-270", "compute-0-[332-333],compute-0-270", "compute-0-270"),
    ("compute-0-[332-333]", " compute-0-270 ", "compute-0-[332-333],compute-0-270", "compute-0-270"),
    ("compute-0-[332-333]", "compute-0-[271,270]", "compute-0-[332-333],compute-0-270,compute-0-271",
     "compute-0-270,compute-0-271"),
    ("compute-0-[332-333]", "compute-0-1000,compute-0-99", "compute-0-[332-333],compute-0-99,compute-0-1000",
     "compute-0-99,compute-0-1000"),
    # already excluded by the deployment: recorded, nothing removed or repeated
    ("compute-0-[332-333]", "compute-0-332", "compute-0-[332-333]", "compute-0-332"),
    ("compute-0-[332-333]", "compute-0-[270,333]", "compute-0-[332-333],compute-0-270", "compute-0-270,compute-0-333"),
    ("", "compute-0-270", "compute-0-270", "compute-0-270"),
    ("compute-0-[332-333]", None, "compute-0-[332-333]", ""),
    ("compute-0-[332-333]", "", "compute-0-[332-333]", ""),
])
def test_merge_keeps_the_default_and_adds_only_new_nodes(default, additional, effective, canonical):
    assert merge_excluded_nodes(default, additional) == (effective, canonical)


def test_every_default_node_stays_excluded_whatever_is_added():
    default = "compute-0-[332-333]"
    for additional in ("compute-0-270", "compute-0-332", "compute-0-[1-64]", "x"):
        effective, _ = merge_excluded_nodes(default, additional)
        assert effective.startswith(default)
        assert set(expand_hostlist(default)) <= set(expand_hostlist(effective))


@pytest.mark.parametrize("additional", [
    "compute 0-270", "compute-0-270;reboot", "--nodelist=x", f"n[1-{MAX_ADDITIONAL_EXCLUDED_NODES + 1}]",
    ["compute-0-270"], 270,
])
def test_unusable_additional_exclusions_are_refused(additional):
    with pytest.raises(CalculationValidationError):
        merge_excluded_nodes("compute-0-[332-333]", additional)


def test_resources_without_additions_are_unchanged():
    plain = structure_search_resources("48:00:00", ntasks=48)
    assert plain.to_dict() == structure_search_resources("48:00:00", ntasks=48, additional_exclude="").to_dict()
    assert "exclude_additional" not in plain.to_dict()
    assert plain.sbatch_directives()[-1] == "#SBATCH --exclude=compute-0-[332-333]"


def test_resources_with_additions_record_both_parts():
    resources = structure_search_resources("48:00:00", ntasks=48, additional_exclude="compute-0-270")
    record = resources.to_dict()
    assert record["exclude"] == "compute-0-[332-333],compute-0-270"
    assert record["exclude_deployment_default"] == "compute-0-[332-333]"
    assert record["exclude_additional"] == "compute-0-270"
    excludes = [line for line in resources.sbatch_directives() if "--exclude" in line]
    assert excludes == ["#SBATCH --exclude=compute-0-[332-333],compute-0-270"]


def test_inconsistent_resource_parts_are_refused():
    with pytest.raises(CalculationValidationError):
        StructureSearchResources(walltime="1:00:00", ntasks=48, exclude="compute-0-270",
                                 exclude_deployment_default="compute-0-[332-333]", exclude_additional="compute-0-270")
    with pytest.raises(CalculationValidationError):
        StructureSearchResources(walltime="1:00:00", ntasks=48, exclude="compute-0-[332-333]",
                                 exclude_deployment_default="compute-0-[332-333]")


def test_the_deployment_default_is_read_not_changed(monkeypatch):
    from backend import config

    before = config.STRUCTURE_SEARCH_EXCLUDED_NODES
    build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-270"))
    assert config.STRUCTURE_SEARCH_EXCLUDED_NODES == before == "compute-0-[332-333]"
    monkeypatch.setattr(config, "STRUCTURE_SEARCH_EXCLUDED_NODES", "compute-0-[1-2]")
    plan = build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-270"))
    assert plan.record["resources"]["request"]["exclude"] == "compute-0-[1-2],compute-0-270"


# --- S2: the plan, the acknowledgement and the job ------------------------------------------


def test_an_exclusion_changes_only_the_request_and_the_resources():
    plain = build_search_plan_from_formula("Si")
    excluded = build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-270"))
    assert excluded.digest != plain.digest
    for section in (*METHODOLOGY_SECTIONS, "search"):
        assert excluded.record[section] == plain.record[section], section
    assert excluded.record["request"]["advanced_overrides"]["excluded_nodes"] == "compute-0-270"
    assert search_job.resource_request_sha256(excluded) != search_job.resource_request_sha256(plain)
    directives_plain = plain.record["resources"]["sbatch_directives"]
    directives = excluded.record["resources"]["sbatch_directives"]
    assert directives[:-1] == directives_plain[:-1]
    assert directives[-1] == "#SBATCH --exclude=compute-0-[332-333],compute-0-270"


def test_equivalent_exclusions_give_one_plan():
    first = build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-[271,270]"))
    second = build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-270,compute-0-271"))
    assert first.digest == second.digest


def test_explicit_spec_plans_carry_the_exclusion_and_rebuild():
    plan = build_search_plan(_si8_spec(random_seed=SEED), excluded_nodes="compute-0-270")
    assert plan.record["request"] == {"kind": "spec", "excluded_nodes": "compute-0-270"}
    assert plan.provenance["request"]["excluded_nodes_as_given"] == "compute-0-270"
    require_current_plan(plan)
    assert build_search_plan(_si8_spec(random_seed=SEED)).record["request"] == {"kind": "spec"}


def test_a_changed_deployment_default_makes_an_excluding_plan_stale(monkeypatch):
    from backend import config

    plan = build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-270"))
    monkeypatch.setattr(config, "STRUCTURE_SEARCH_EXCLUDED_NODES", "compute-0-[332-334]")
    with pytest.raises(SearchPlanError):
        require_current_plan(plan)


def test_the_acknowledgement_binds_the_effective_exclusion():
    plan = build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-270"))
    acknowledgement = search_job.build_resource_acknowledgement(
        plan, attempt_id="00000000-0000-4000-8000-000000000000", principal="lee",
        acknowledged_at="2026-11-27T12:00:00Z")
    assert acknowledgement["resources"]["request"]["exclude"] == "compute-0-[332-333],compute-0-270"
    assert "#SBATCH --exclude=compute-0-[332-333],compute-0-270" in acknowledgement["resources"]["sbatch_directives"]


# --- the benchmark's own request, end to end in planning ---------------------------------------


def test_the_benchmark_plan_keeps_the_methodology_and_the_manual_defaults(monkeypatch):
    _configured(monkeypatch, SHARED_NODE)
    plain = plan_structure_search(_benchmark_request(random_seed=None, excluded_nodes=None)).plan
    plan = plan_structure_search(_benchmark_request()).plan
    for section in METHODOLOGY_SECTIONS:
        assert plan.record[section] == plain.record[section], section
    spec = plan.record["search"]["spec"]
    assert (spec["population_size"], spec["initial_population_size"], spec["stop_criterion"],
            spec["num_generations"], spec["random_seed"]) == (20, 20, 8, 30, SEED)
    request = plan.record["resources"]["request"]
    assert (request["ntasks"], request["exclusive"], request["mem"]) == (48, False, "32G")
    assert request["exclude"] == "compute-0-[332-333],compute-0-270"
    files = render_prepared_files(plan, bridge_path=BRIDGE)
    plain_files = render_prepared_files(plain, bridge_path=BRIDGE)
    # Only INPUT.txt (one added line), the plan and the manifest differ; the INCARs and POTCAR.spec are identical.
    differing = sorted(name for name in files if files[name] != plain_files[name])
    assert differing == ["INPUT.txt", PREPARED_FILE, "search_plan.json"]
    added = [line for line in files["INPUT.txt"].decode().splitlines()
             if line not in plain_files["INPUT.txt"].decode().splitlines()]
    assert added == [f"{SEED} : fixRndSeed"]
    responses = [plan_structure_search(_benchmark_request(random_seed=seed)).plan.digest
                 for seed in (20261126, 20261127, 20261128)]
    assert len(set(responses)) == 3  # the three replicates are three plans differing only in the seed


# --- the analyser's seed check ---------------------------------------------------------------


def _analyser():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "reference" / "uspex_pilot" / "execution_validation" / \
        "analyze_execution.py"
    spec = importlib.util.spec_from_file_location("analyze_execution_seed_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("input_text, parameters, status", [
    (f"{SEED} : fixRndSeed\n", f"{SEED} : fixRndSeed\n", "verified"),
    (f"{SEED} : fixRndSeed\n", f"{SEED + 1} : fixRndSeed\n", "failed"),
    (f"{SEED + 1} : fixRndSeed\n", f"{SEED + 1} : fixRndSeed\n", "failed"),
    (f"{SEED} : fixRndSeed\n{SEED} : fixRndSeed\n", f"{SEED} : fixRndSeed\n", "failed"),
    (f"{SEED} : fixRndSeed\n", None, "missing"),
    (f"{SEED} : fixRndSeed\n", "fixRndSeed = 20261126\n", "missing"),  # USPEX's own form not yet confirmed
])
def test_the_analyser_compares_requested_written_and_read_back_seeds(input_text, parameters, status):
    analyser = _analyser()
    manifest = {"requested": SEED, "written_in_input_txt": SEED}
    found, detail = analyser._random_seed_check(manifest, input_text, parameters)
    assert found == status
    assert detail["effective_rng_determinism"] == "not assessed by this analysis"


def test_the_analyser_uses_the_frozen_plans_seed():
    analyser = _analyser()
    line = f"{SEED} : fixRndSeed\n"
    manifest = {"requested": SEED, "written_in_input_txt": SEED}
    assert analyser._random_seed_check(manifest, line, line, planned=SEED)[0] == "verified"
    assert analyser._random_seed_check(manifest, line, line, planned=SEED + 1)[0] == "failed"
    assert analyser._random_seed_check({}, line, line, planned=SEED)[0] == "failed"  # manifest lost its record


def test_an_exclusion_inside_the_default_is_recorded_but_excludes_nothing_new():
    plain = build_search_plan_from_formula("Si")
    plan = build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-332"))
    request = plan.record["resources"]["request"]
    assert request["exclude"] == plain.record["resources"]["request"]["exclude"] == "compute-0-[332-333]"
    assert request["exclude_additional"] == "compute-0-332"
    assert plan.record["resources"]["sbatch_directives"] == plain.record["resources"]["sbatch_directives"]
    assert plan.digest != plain.digest  # the request is part of the plan


def test_an_unexpandable_deployment_default_is_named_in_the_error(monkeypatch):
    from backend import config

    monkeypatch.setattr(config, "STRUCTURE_SEARCH_EXCLUDED_NODES", "n[5-3]")
    with pytest.raises(CalculationValidationError, match="deployment's excluded nodes"):
        build_search_plan_from_formula("Si", AdvancedSettings(excluded_nodes="compute-0-270"))
