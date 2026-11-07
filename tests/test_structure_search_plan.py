"""Frozen structure-search plans and local input preparation (USPEX integration step 3)."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

from backend.calculations.registry import CalculationValidationError
from backend.structure_search import incar_policy, search_plan, step_validity
from backend.structure_search.prediction_plan import AdvancedSettings, build_prediction_plan
from backend.structure_search.runtime import SearchRuntimeSettings
from backend.structure_search.search_plan import (
    DESTINATION_EXISTS_CODE,
    INVALID_BINDING_CODE,
    INVALID_PLAN_CODE,
    PLAN_DIGEST_MISMATCH_CODE,
    PLAN_FILE,
    PLAN_SCHEMA,
    PLAN_SCHEMA_CODE,
    PREPARED_FILE,
    PREPARED_MISMATCH_CODE,
    STALE_PLAN_CODE,
    UNEXPECTED_FILE_CODE,
    UNSAFE_PATH_CODE,
    SearchPlanError,
    StructureSearchPlan,
    build_search_plan,
    build_search_plan_from_formula,
    load_plan,
    load_plan_text,
    plan_digest,
    prepare_search_directory,
    render_prepared_files,
    verify_prepared_directory,
)
from backend.structure_search.spec import StructureSearchSpec
from conftest import synthetic_enmax_metadata, use_enmax_metadata


FIXTURES = Path(__file__).parent / "fixtures" / "structure_search"
# The input-writer golden files use this placeholder bridge path.
BRIDGE = "/bmd/placeholder/bmd_bridge.sh"
# SYNTHETIC test metadata with the published PBE_64 ENMAX of the golden searches
# (as tests/test_structure_search_incar_policy.py).
SEARCH_ENMAX = {"Si": 245.345, "Li_sv": 499.034, "O": 400.0}

EXPECTED_FILES = [
    "INPUT.txt", "POTCAR.spec", PLAN_FILE, PREPARED_FILE,
    *(f"Specific/INCAR_{n}" for n in range(1, 6)),
]


@pytest.fixture(autouse=True)
def search_enmax_metadata(monkeypatch, tmp_path):
    use_enmax_metadata(monkeypatch, tmp_path / "enmax.json", synthetic_enmax_metadata(SEARCH_ENMAX))


def _si8_spec():
    # The input-writer golden search (tests/test_structure_search_input_writer.py).
    return StructureSearchSpec(
        elements=("Si",), counts=(8,), min_atoms=8, max_atoms=8,
        num_generations=20, stop_criterion=8, kresol_start=(0.1, 0.08, 0.06, 0.04, 0.04),
        walltime="48:00:00", population_size=12, initial_population_size=24,
    )


def _mg2sio4_spec():
    return StructureSearchSpec(
        elements=("Mg", "Si", "O"), counts=(2, 1, 4), min_atoms=7, max_atoms=14,
        num_generations=40, stop_criterion=15,
        kresol_start=(0.12, 0.10, 0.08, 0.04, 0.04), walltime="120:00:00",
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)).replace(os.sep, "/"): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _code(info) -> str:
    return info.value.diagnostic["code"]


# --- the plan --------------------------------------------------------------------


def test_plan_has_every_required_section():
    plan = build_search_plan_from_formula("Mg2SiO4")
    record = plan.record
    assert record["schema"] == PLAN_SCHEMA and record["schema_version"] == 1
    assert plan.digest.startswith("sha256:") and plan.digest == plan_digest(record)
    composition = record["composition"]
    assert composition["search_formula"] == "Mg2SiO4"
    assert composition["reduced_formula"] == "Mg2SiO4"
    assert "entered_formula" not in json.dumps(record)  # request provenance is outside the digest
    assert record["search"]["spec"]["calculation_type"] == 300
    assert record["admission"]["decision"] == "admitted"
    assert record["admission"]["external_pressure_gpa"] == 0.0
    assert record["admission"]["spin_polarized"] is False and record["admission"]["theory"] == "pbe"
    ladder = record["screening_and_ranking"]
    assert [row["step"] for row in ladder["steps"]] == [1, 2, 3, 4, 5]
    assert ladder["policy_id"] == incar_policy.INCAR_POLICY_ID
    assert sorted(record["incar_files"]) == [f"INCAR_{n}" for n in range(1, 6)]
    assert [row["symbol"] for row in record["potcar"]["elements"]] == ["Mg", "Si", "O"]
    for row in record["potcar"]["elements"]:
        assert {"TITEL", "ENMAX", "dataset_sha256", "staged_at_execution_as"} <= set(row)
    assert record["potcar"]["contains_potcar_contents"] is False
    assert record["potcar"]["functional"] == "PBE_64"
    assert [row["kresol"] for row in record["kpoints"]["steps"]] == [0.12, 0.10, 0.08, 0.04, 0.04]
    assert record["kpoints"]["steps"][3]["bmd_policy_bound"]["max_kresol"] == incar_policy.BMD_STANDARD_MAX_KRESOL
    assert record["kpoints"]["steps"][0]["bmd_policy_bound"] is None
    assert "structure_search_defaults" in record["kpoints"]["steps"][0]["source"]
    assert record["candidate_validity"]["policy_id"] == step_validity.STEP_VALIDITY_POLICY_ID
    assert record["candidate_validity"]["enforced"] is True
    assert record["runtime"]["uspex"]["version"] == "25.0"
    assert record["runtime"]["launch"]["parallel_relaxations"] == record["execution"]["input_txt"]["num_parallel_calcs"]
    assert record["execution"]["bridge"]["script"] == "bmd_bridge.sh"
    assert re.fullmatch(r"[0-9a-f]{64}", record["execution"]["bridge"]["sha256"])
    assert record["resources"]["status"] == "provisional_estimate"
    assert record["resources"]["approved"] is False
    assert record["resources"]["request"]["exclusive"] is True
    assert "SLURM submission" in record["not_implemented"]


def test_formula_is_recorded_as_entered_in_provenance_and_reduced_in_the_plan():
    plan = build_search_plan_from_formula("Si8")
    record, provenance = plan.record, plan.provenance
    assert record["composition"]["search"] == [{"element": "Si", "count": 1}]
    assert record["request"] == {
        "kind": "formula",
        "advanced_overrides": {"min_atoms": None, "max_atoms": None, "num_generations": None, "walltime": None},
    }
    assert provenance["schema"] == search_plan.PROVENANCE_SCHEMA
    assert provenance["covered_by_plan_digest"] is False
    assert provenance["request"]["entered_formula"] == "Si8"
    assert provenance["request"]["entered_composition"] == [{"element": "Si", "count": 8}]
    assert provenance["request"]["reduced_from_entered"] is True
    assert provenance["request"]["reordered_from_entered"] is False


# --- equivalent compositions: one canonical plan ---------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [("SiO2", "O2Si"), ("Mg2SiO4", "O4SiMg2"), ("Mg2SiO4", "SiMg2O4"), ("Ca(OH)2", "H2CaO2"),
     ("Li2O", "OLi2"), ("GaAs", "AsGa")],
)
def test_reordered_formulas_give_the_same_plan_and_digest(first, second, tmp_path):
    a = build_search_plan_from_formula(first)
    b = build_search_plan_from_formula(second)
    assert a.digest == b.digest and a.canonical == b.canonical
    assert a.provenance["request"]["entered_formula"] == first
    assert b.provenance["request"]["entered_formula"] == second
    prepare_search_directory(a, tmp_path / "a", bridge_path=BRIDGE)
    prepare_search_directory(b, tmp_path / "b", bridge_path=BRIDGE)
    tree_a, tree_b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    for name in EXPECTED_FILES:
        if name not in (PLAN_FILE, PREPARED_FILE):
            assert tree_a[name] == tree_b[name], name
    # Only the recorded provenance differs, and so the hash of search_plan.json.
    plan_a, plan_b = json.loads(tree_a[PLAN_FILE]), json.loads(tree_b[PLAN_FILE])
    assert plan_a.pop("provenance") != plan_b.pop("provenance")
    assert plan_a == plan_b
    manifest_a, manifest_b = json.loads(tree_a[PREPARED_FILE]), json.loads(tree_b[PREPARED_FILE])
    manifest_a["files"].pop(PLAN_FILE), manifest_b["files"].pop(PLAN_FILE)
    assert manifest_a == manifest_b


def test_canonical_order_is_electronegativity_then_symbol():
    record = build_search_plan_from_formula("O2Si").record
    assert [row["element"] for row in record["composition"]["search"]] == ["Si", "O"]
    assert [row["symbol"] for row in record["potcar"]["elements"]] == ["Si", "O"]
    assert record["composition"]["element_order"] == search_plan.ELEMENT_ORDER_RULE
    assert list(search_plan.canonical_composition({"O": 4, "Si": 1, "Mg": 2})) == ["Mg", "Si", "O"]
    assert list(search_plan.canonical_composition({"He": 1, "Ne": 1, "Li": 1})) == ["Li", "He", "Ne"]


def test_reordering_keeps_cell_sizes_and_formula_units(tmp_path):
    overrides = AdvancedSettings(min_atoms="3", max_atoms="12", num_generations="20")
    a = build_search_plan_from_formula("SiO2", overrides).record
    b = build_search_plan_from_formula("O2Si", overrides).record
    for record in (a, b):
        assert record["search"]["spec"]["min_atoms"] == 3 and record["search"]["spec"]["max_atoms"] == 12
        assert record["search"]["cell_sizes"] == [3, 6, 9, 12]
        assert record["search"]["formula_units"] == [1, 2, 3, 4]
    assert a == b
    default = build_search_plan_from_formula("O2Si").record["search"]
    assert default["cell_sizes"] == build_search_plan_from_formula("SiO2").record["search"]["cell_sizes"]


def test_explicit_specs_are_put_in_canonical_order_and_kept_as_given():
    given = StructureSearchSpec(
        elements=("O", "Si", "Mg"), counts=(4, 1, 2), min_atoms=7, max_atoms=14,
        num_generations=40, stop_criterion=15,
        kresol_start=(0.12, 0.10, 0.08, 0.04, 0.04), walltime="120:00:00",
    )
    plan = build_search_plan(given)
    assert plan.digest == build_search_plan(_mg2sio4_spec()).digest
    assert plan.provenance["request"]["spec_as_given"]["composition"][0] == {"element": "O", "count": 4}
    assert plan.provenance["request"]["reordered_from_entered"] is True


def test_provenance_cannot_contradict_the_plan():
    plan = build_search_plan_from_formula("SiO2")
    document = plan.to_document()
    document["provenance"]["request"]["entered_formula"] = "GaAs"
    with pytest.raises(SearchPlanError) as info:
        load_plan(document)
    assert _code(info) == INVALID_PLAN_CODE
    document["provenance"]["request"]["entered_formula"] = "Si2O4"  # same reduced composition
    assert load_plan(document).digest == plan.digest


def test_multiples_of_one_formula_unit_give_the_same_search():
    # Unchanged behaviour: a formula is reduced to one formula unit, and the
    # default cell range is built from it; the multiple entered is provenance.
    assert build_search_plan_from_formula("Si8").digest == build_search_plan_from_formula("Si").digest
    assert build_search_plan_from_formula("Si2O4").digest == build_search_plan_from_formula("SiO2").digest


def test_generating_commit_is_recorded_outside_the_digest():
    one = build_search_plan_from_formula("Si", generator={"commit": "a" * 40, "dirty": False})
    two = build_search_plan_from_formula("Si", generator={"commit": "b" * 40, "dirty": True})
    assert one.digest == two.digest
    assert one.provenance["generator"]["commit"] == "a" * 40
    assert two.provenance["generator"] == {"commit": "b" * 40, "dirty": True}
    real = build_search_plan_from_formula("Si").provenance["generator"]
    assert set(real) >= {"repository", "commit", "dirty", "provenance_available"}


def test_plan_is_immutable_and_record_is_a_copy():
    plan = build_search_plan_from_formula("Si")
    record = plan.record
    record["search"]["spec"]["num_generations"] = 1
    assert plan.record["search"]["spec"]["num_generations"] != 1
    with pytest.raises(Exception):
        plan.digest = "sha256:" + "0" * 64  # frozen dataclass
    with pytest.raises(SearchPlanError):
        StructureSearchPlan(canonical=plan.canonical, digest="sha256:" + "0" * 64)


def test_same_request_gives_identical_plans_digests_and_prepared_inputs(tmp_path):
    first = build_search_plan_from_formula("Li2O", AdvancedSettings(num_generations="30"))
    second = build_search_plan_from_formula("Li2O", AdvancedSettings(num_generations="30"))
    assert first.canonical == second.canonical and first.digest == second.digest
    assert first.to_json() == second.to_json()
    prepare_search_directory(first, tmp_path / "a", bridge_path=BRIDGE)
    prepare_search_directory(second, tmp_path / "b", bridge_path=BRIDGE)
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    assert sorted(_tree(tmp_path / "a")) == sorted(EXPECTED_FILES)


@pytest.mark.parametrize(
    ("formula", "advanced"),
    [
        ("Si", AdvancedSettings(num_generations="51")),
        ("Si", AdvancedSettings(max_atoms="16")),
        ("Si", AdvancedSettings(walltime="100:00:00")),
        ("SiC", None),
        ("Li2O", None),
    ],
)
def test_search_settings_change_the_digest(formula, advanced):
    assert build_search_plan_from_formula(formula, advanced).digest != build_search_plan_from_formula("Si").digest


def _with(monkeypatch, target, name, value):
    monkeypatch.setattr(target, name, value)
    return build_search_plan_from_formula("Si").digest


def test_numerical_policy_changes_change_the_digest(monkeypatch, tmp_path):
    base = build_search_plan_from_formula("Si").digest
    changed = {}
    with monkeypatch.context() as patch:
        changed["early cutoff rule"] = _with(patch, incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    with monkeypatch.context() as patch:
        changed["KPAR"] = _with(patch, incar_policy, "KPAR_BY_STEP", {1: 1, 2: 1, 3: 2, 4: 4, 5: 4})
    with monkeypatch.context() as patch:
        steps = [dict(row) for row in incar_policy.MANUAL_STEPS]
        steps[0]["EDIFF"] = 2e-2
        changed["step-1 EDIFF"] = _with(patch, incar_policy, "MANUAL_STEPS", tuple(steps))
    with monkeypatch.context() as patch:
        rows = [dict(row) for row in step_validity.STEP_VALIDITY]
        rows[2]["ionic_convergence"] = True
        changed["validity"] = _with(patch, step_validity, "STEP_VALIDITY", tuple(rows))
    with monkeypatch.context() as patch:
        changed["sleepSeconds"] = _with(patch, search_plan, "SEARCH_SLEEP_SECONDS", 60)
    with monkeypatch.context() as patch:
        use_enmax_metadata(patch, tmp_path / "other.json", synthetic_enmax_metadata({**SEARCH_ENMAX, "Si": 300.0}))
        changed["ENMAX metadata"] = build_search_plan_from_formula("Si").digest
    defaults = SearchRuntimeSettings.from_config()
    other = SearchRuntimeSettings(**{**defaults.__dict__, "parallel_relaxations": 2})
    changed["runtime launch layout"] = build_search_plan_from_formula("Si", runtime_settings=other).digest
    assert all(digest != base for digest in changed.values()), changed
    assert len(set(changed.values())) == len(changed)
    assert build_search_plan_from_formula("Si").digest == base


def test_out_of_scope_requests_are_refused_by_admission():
    with pytest.raises(CalculationValidationError):
        build_search_plan_from_formula("Fe2O3")  # spin-polarisation screen
    with pytest.raises(CalculationValidationError):
        build_search_plan(StructureSearchSpec(
            elements=("Si",), counts=(1,), min_atoms=1, max_atoms=8, num_generations=10,
            stop_criterion=8, kresol_start=(0.1, 0.08, 0.06, 0.04, 0.04), walltime="10:00:00",
            external_pressure_gpa=10.0,
        ))


# --- golden files ----------------------------------------------------------------


@pytest.mark.parametrize(("formula", "folder"), [("Si", "incar_Si"), ("Li2O", "incar_Li2O")])
def test_frozen_incars_are_the_golden_ladder(formula, folder, tmp_path):
    plan = build_search_plan_from_formula(formula)
    for step in range(1, 6):
        golden = (FIXTURES / folder / f"INCAR_{step}").read_text(encoding="utf-8")
        assert plan.record["incar_files"][f"INCAR_{step}"]["text"] == golden
    prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    for step in range(1, 6):
        assert (tmp_path / "search" / "Specific" / f"INCAR_{step}").read_bytes() == (
            FIXTURES / folder / f"INCAR_{step}"
        ).read_bytes()


@pytest.mark.parametrize(("spec", "golden"), [(_si8_spec, "INPUT_Si8_0GPa.txt"), (_mg2sio4_spec, "INPUT_Mg2SiO4_0GPa.txt")])
def test_prepared_input_txt_is_the_golden_file(spec, golden, tmp_path):
    plan = build_search_plan(spec())
    prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    assert (tmp_path / "search" / "INPUT.txt").read_bytes() == (FIXTURES / golden).read_bytes()


def test_prepared_files_match_the_plan_and_manifest(tmp_path):
    plan = build_search_plan(_si8_spec())
    manifest = prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    tree = _tree(tmp_path / "search")
    assert load_plan_text(tree[PLAN_FILE].decode()).digest == plan.digest
    assert manifest == json.loads(tree[PREPARED_FILE])
    assert manifest["plan_digest"] == plan.digest
    assert manifest["bindings"] == {"bridge_path": BRIDGE}
    assert manifest["bindings_and_generator_covered_by_plan_digest"] is False
    assert {"commit", "dirty"} <= set(manifest["generator"])
    assert manifest["submitted"] is False and manifest["contains_potcar_contents"] is False
    import hashlib
    for name, digest in manifest["files"].items():
        assert hashlib.sha256(tree[name]).hexdigest() == digest
    assert set(manifest["files"]) == set(EXPECTED_FILES) - {PREPARED_FILE}
    assert tree["POTCAR.spec"] == b"Si\n"
    assert [row["staged_at_execution_as"] for row in manifest["potcar_files_staged_at_execution"]] == [
        "Specific/POTCAR_Si"
    ]
    assert verify_prepared_directory(tmp_path / "search") == manifest


def test_no_potcar_contents_anywhere(tmp_path):
    plan = build_search_plan_from_formula("Mg2SiO4")
    prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    tree = _tree(tmp_path / "search")
    assert not any(Path(name).name.startswith("POTCAR_") or Path(name).name == "POTCAR" for name in tree)
    assert tree["POTCAR.spec"].decode().splitlines() == ["Mg", "Si", "O"]
    for data in tree.values():
        for marker in (b"VRHFIN", b"PSCTR", b"POMASS", b"END of PSCTR", b"Description"):
            assert marker not in data


# --- refusing plans --------------------------------------------------------------


def test_load_plan_round_trip_and_rejects_tampering():
    plan = build_search_plan_from_formula("Si")
    assert load_plan(plan.to_document()) == plan
    assert load_plan_text(plan.to_json()) == plan

    wrong = {**plan.to_document(), "plan_digest": "sha256:" + "0" * 64}
    with pytest.raises(SearchPlanError) as info:
        load_plan(wrong)
    assert _code(info) == PLAN_DIGEST_MISMATCH_CODE

    edited = plan.to_document()
    edited["screening_and_ranking"]["steps"][0]["incar"]["ENCUT"] = 999
    with pytest.raises(SearchPlanError) as info:
        load_plan(edited)
    assert _code(info) == PLAN_DIGEST_MISMATCH_CODE

    other_version = {**plan.to_document(), "schema_version": 99}
    with pytest.raises(SearchPlanError) as info:
        load_plan(other_version)
    assert _code(info) == PLAN_SCHEMA_CODE

    with pytest.raises(SearchPlanError):
        load_plan_text("{not json")


def _redigested(document):
    record = {key: value for key, value in document.items() if key not in ("plan_digest", "provenance")}
    return StructureSearchPlan.from_record(record, document.get("provenance"))


def test_modified_policy_record_with_a_recomputed_digest_is_refused(tmp_path):
    plan = build_search_plan_from_formula("Si")
    document = plan.to_document()
    document["screening_and_ranking"]["steps"][3]["incar"]["EDIFFG"] = -0.1
    forged = _redigested(document)
    with pytest.raises(SearchPlanError) as info:
        prepare_search_directory(forged, tmp_path / "search", bridge_path=BRIDGE)
    assert _code(info) == STALE_PLAN_CODE
    assert "screening_and_ranking" in info.value.diagnostic["changed"]
    assert not (tmp_path / "search").exists()
    # The record's INCAR text no longer matches its settings either.
    with pytest.raises(SearchPlanError) as info:
        render_prepared_files(forged, bridge_path=BRIDGE)
    assert _code(info) == INVALID_PLAN_CODE


def test_stale_plan_is_refused(monkeypatch, tmp_path):
    plan = build_search_plan_from_formula("Si")
    prepare_search_directory(plan, tmp_path / "before", bridge_path=BRIDGE)
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    with pytest.raises(SearchPlanError) as info:
        prepare_search_directory(plan, tmp_path / "after", bridge_path=BRIDGE)
    assert _code(info) == STALE_PLAN_CODE
    assert not (tmp_path / "after").exists()
    with pytest.raises(SearchPlanError) as info:
        verify_prepared_directory(tmp_path / "before")
    assert _code(info) == STALE_PLAN_CODE
    # Still consistent with its own frozen plan.
    verify_prepared_directory(tmp_path / "before", require_current=False)


def test_plan_for_other_runtime_settings_is_stale_under_the_configured_ones(tmp_path):
    defaults = SearchRuntimeSettings.from_config()
    other = SearchRuntimeSettings(**{**defaults.__dict__, "parallel_relaxations": 2})
    plan = build_search_plan_from_formula("Si", runtime_settings=other)
    with pytest.raises(SearchPlanError) as info:
        prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    assert _code(info) == STALE_PLAN_CODE
    prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE, runtime_settings=other)
    assert "2 : numParallelCalcs" in (tmp_path / "search" / "INPUT.txt").read_text()


# --- refusing destinations -------------------------------------------------------


def test_existing_destinations_are_refused_and_untouched(tmp_path):
    plan = build_search_plan_from_formula("Si")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SearchPlanError) as info:
        prepare_search_directory(plan, empty, bridge_path=BRIDGE)
    assert _code(info) == DESTINATION_EXISTS_CODE
    assert list(empty.iterdir()) == []

    afile = tmp_path / "file"
    afile.write_text("keep")
    with pytest.raises(SearchPlanError) as info:
        prepare_search_directory(plan, afile, bridge_path=BRIDGE)
    assert _code(info) == DESTINATION_EXISTS_CODE
    assert afile.read_text() == "keep"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "nowhere")
    with pytest.raises(SearchPlanError) as info:
        prepare_search_directory(plan, dangling, bridge_path=BRIDGE)
    assert _code(info) == DESTINATION_EXISTS_CODE
    assert not (tmp_path / "nowhere").exists()

    prepare_search_directory(plan, tmp_path / "once", bridge_path=BRIDGE)
    before = _tree(tmp_path / "once")
    with pytest.raises(SearchPlanError) as info:
        prepare_search_directory(plan, tmp_path / "once", bridge_path=BRIDGE)
    assert _code(info) == DESTINATION_EXISTS_CODE
    assert _tree(tmp_path / "once") == before


def test_unsafe_paths_are_refused(tmp_path):
    plan = build_search_plan_from_formula("Si")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    cases = [
        link / "search",                       # under a symbolic link
        tmp_path / "missing" / "search",       # parent does not exist
        str(real) + "/../real/search",         # '..'
        "",                                    # empty
    ]
    afile = tmp_path / "afile"
    afile.write_text("x")
    cases.append(afile / "search")             # parent is a file
    for destination in cases:
        with pytest.raises(SearchPlanError) as info:
            prepare_search_directory(plan, destination, bridge_path=BRIDGE)
        assert _code(info) == UNSAFE_PATH_CODE, destination
    assert list(real.iterdir()) == []
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    "bridge",
    ["bmd_bridge.sh", "/tmp/bmd_bridge.sh", "/bmd/x/other.sh", "/bmd/x/../bmd_bridge.sh",
     "/bmd/x//bmd_bridge.sh", "/bmd/x y/bmd_bridge.sh", None],
)
def test_invalid_bridge_binding_is_refused_before_anything_is_written(bridge, tmp_path):
    plan = build_search_plan_from_formula("Si")
    with pytest.raises(CalculationValidationError) as info:
        prepare_search_directory(plan, tmp_path / "search", bridge_path=bridge)
    assert info.value.diagnostic["code"] in {INVALID_BINDING_CODE, "invalid_bridge_path"}
    assert not (tmp_path / "search").exists()


def test_failed_preparation_removes_only_what_it_created(monkeypatch, tmp_path):
    plan = build_search_plan_from_formula("Si")
    (tmp_path / "neighbour").write_text("keep")

    def broken(*args, **kwargs):
        raise SearchPlanError("simulated", code="simulated")

    monkeypatch.setattr(search_plan, "verify_prepared_directory", broken)
    with pytest.raises(SearchPlanError):
        prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    assert not (tmp_path / "search").exists()
    assert (tmp_path / "neighbour").read_text() == "keep"


# --- verifying a prepared directory ---------------------------------------------


@pytest.fixture
def prepared(tmp_path):
    plan = build_search_plan_from_formula("Si")
    path = tmp_path / "search"
    prepare_search_directory(plan, path, bridge_path=BRIDGE)
    return path


def _refused(path, code):
    with pytest.raises(SearchPlanError) as info:
        verify_prepared_directory(path)
    assert _code(info) == code, info.value


def test_verify_refuses_unexpected_files_folders_and_symlinks(prepared, tmp_path):
    (prepared / "extra.txt").write_text("x")
    _refused(prepared, UNEXPECTED_FILE_CODE)
    os.unlink(prepared / "extra.txt")
    verify_prepared_directory(prepared)

    (prepared / "Specific" / "POTCAR_Si").write_text("not a real POTCAR")
    _refused(prepared, UNEXPECTED_FILE_CODE)
    os.unlink(prepared / "Specific" / "POTCAR_Si")

    (prepared / "empty_folder").mkdir()
    _refused(prepared, UNEXPECTED_FILE_CODE)
    os.rmdir(prepared / "empty_folder")

    (prepared / "link").symlink_to(prepared / "INPUT.txt")
    _refused(prepared, UNEXPECTED_FILE_CODE)
    os.unlink(prepared / "link")

    os.unlink(prepared / "Specific" / "INCAR_5")
    _refused(prepared, UNEXPECTED_FILE_CODE)

    whole = tmp_path / "whole_link"
    whole.symlink_to(prepared)
    _refused(whole, UNSAFE_PATH_CODE)


@pytest.mark.parametrize("name", ["INPUT.txt", "Specific/INCAR_2", "POTCAR.spec"])
def test_verify_refuses_modified_inputs(prepared, name):
    path = prepared / name
    path.write_bytes(path.read_bytes() + b"EXTRA = 1\n")
    _refused(prepared, PREPARED_MISMATCH_CODE)


def test_verify_refuses_edited_plan_or_manifest(prepared):
    plan_path = prepared / PLAN_FILE
    original = plan_path.read_text()
    plan_path.write_text(original.replace('"num_generations": 50', '"num_generations": 49', 1))
    _refused(prepared, PLAN_DIGEST_MISMATCH_CODE)
    plan_path.write_text(original)

    manifest_path = prepared / PREPARED_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["plan_digest"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    _refused(prepared, PREPARED_MISMATCH_CODE)

    manifest["plan_digest"] = load_plan_text(original).digest
    manifest["bindings"]["bridge_path"] = "/bmd/elsewhere/bmd_bridge.sh"
    manifest_path.write_text(json.dumps(manifest))
    _refused(prepared, PREPARED_MISMATCH_CODE)


# --- no remote work ----------------------------------------------------------------


def test_planning_and_preparation_never_run_programs_or_open_connections(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("structure-search planning and preparation must not run programs or connect")

    real_run = subprocess.run
    programs = []

    def only_read_only_git(command, *args, **kwargs):
        # The only program allowed: read-only git for the generating commit
        # (backend.calculations.capabilities.git_provenance).
        programs.append(list(command))
        if not (command[0] == "git" and command[1] == "--no-optional-locks"
                and command[-2:] in (["rev-parse", "HEAD"], ["status", "--porcelain"])
                or command[0] == "git" and tuple(command[-2:]) in {("rev-parse", "HEAD"), ("status", "--porcelain")}):
            forbidden()
        return real_run(command, *args, **kwargs)

    for name in ("Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(subprocess, "run", only_read_only_git)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    plan = build_search_plan_from_formula("Mg2SiO4")
    prepare_search_directory(plan, tmp_path / "search", bridge_path=BRIDGE)
    verify_prepared_directory(tmp_path / "search")
    build_prediction_plan("Si")
    assert programs and all(command[0] == "git" for command in programs)


# --- the New structure preview ---------------------------------------------------


@pytest.mark.parametrize("formula", ["Si", "GaAs", "Mg2SiO4"])
def test_preview_is_a_view_of_the_frozen_plan(formula):
    preview = build_prediction_plan(formula)
    plan = build_search_plan_from_formula(formula)
    assert preview["plan_digest"] == plan.digest
    assert preview["search_plan_schema"] == f"{PLAN_SCHEMA}/1"
    assert preview["spec"] == plan.record["search"]["spec"]
    assert [row["encut_ev"] for row in preview["search"]["ladder"]] == [
        row["incar"]["ENCUT"] for row in plan.record["screening_and_ranking"]["steps"]
    ]
    assert preview["search"]["resources"] == plan.record["resources"]["request"]
    assert preview["search"]["resources_status"] == "provisional_estimate"
    assert preview["search"]["resources_approved"] is False
    assert preview["submission_available"] is False


def test_preview_page_shows_the_plan_digest(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    def no_remote(*args, **kwargs):
        raise AssertionError("the New structure preview never touches POWER")

    for name in ("monitor_job", "prepare_remote_submission", "submit_remote_workflow",
                 "load_results_for_completed_job"):
        monkeypatch.setattr(main, name, no_remote)
    response = TestClient(main.app).post("/predict", data={"formula": "Si"})
    assert response.status_code == 200
    digest = build_search_plan_from_formula("Si").digest
    assert response.context["prediction"]["plan"]["plan_digest"] == digest
    shown = re.search(r"data-prediction-plan-digest>(.*?)</div>", response.text, re.S).group(1)
    assert digest in shown
    assert "Provisional estimate, not an approved allocation" in response.text


# --- execution-time binding check ------------------------------------------------


def _cluster(tmp_path, content: bytes | None = None) -> Path:
    """A stand-in for POWER's file system with the launch bridge at BRIDGE."""

    from backend.structure_search.bridge.master import bridge_script_path

    root = tmp_path / "cluster"
    target = root / BRIDGE.lstrip("/")
    target.parent.mkdir(parents=True)
    target.write_bytes(bridge_script_path().read_bytes() if content is None else content)
    return root


def test_execution_binding_accepts_the_prepared_bridge(prepared, tmp_path):
    root = _cluster(tmp_path)
    manifest = search_plan.verify_execution_binding(prepared, bridge_path=BRIDGE, filesystem_root=str(root))
    assert manifest["bindings"]["bridge_path"] == BRIDGE


def test_execution_binding_refuses_another_bridge_path(prepared, tmp_path):
    root = _cluster(tmp_path)
    with pytest.raises(SearchPlanError) as info:
        search_plan.verify_execution_binding(
            prepared, bridge_path="/bmd/elsewhere/bmd_bridge.sh", filesystem_root=str(root)
        )
    assert _code(info) == search_plan.BINDING_MISMATCH_CODE


def test_execution_binding_refuses_a_changed_or_missing_or_linked_bridge(prepared, tmp_path):
    root = _cluster(tmp_path, b"#!/bin/sh\nexit 0\n")
    with pytest.raises(SearchPlanError) as info:
        search_plan.verify_execution_binding(prepared, bridge_path=BRIDGE, filesystem_root=str(root))
    assert _code(info) == search_plan.BRIDGE_MISMATCH_CODE

    target = root / BRIDGE.lstrip("/")
    os.unlink(target)
    with pytest.raises(SearchPlanError) as info:
        search_plan.verify_execution_binding(prepared, bridge_path=BRIDGE, filesystem_root=str(root))
    assert _code(info) == search_plan.BRIDGE_MISMATCH_CODE

    from backend.structure_search.bridge.master import bridge_script_path

    target.symlink_to(bridge_script_path())
    with pytest.raises(SearchPlanError) as info:
        search_plan.verify_execution_binding(prepared, bridge_path=BRIDGE, filesystem_root=str(root))
    assert _code(info) == search_plan.BRIDGE_MISMATCH_CODE


def test_execution_binding_refuses_a_stale_plan(prepared, tmp_path, monkeypatch):
    root = _cluster(tmp_path)
    monkeypatch.setattr(incar_policy, "EARLY_STEP_ENCUT_MULTIPLIER", 1.2)
    with pytest.raises(SearchPlanError) as info:
        search_plan.verify_execution_binding(prepared, bridge_path=BRIDGE, filesystem_root=str(root))
    assert _code(info) == STALE_PLAN_CODE


def test_sleep_seconds_is_recorded_as_provisional():
    execution = build_search_plan_from_formula("Si").record["execution"]["input_txt"]
    assert execution["sleep_seconds"] == 30
    assert "not yet independently confirmed" in execution["sleep_seconds_status"]


def test_preview_keeps_the_entered_formula_and_the_canonical_one_for_the_plan(monkeypatch):
    preview = build_prediction_plan("O2Si")
    assert preview["display_formula"] == "O2Si" and preview["entered_formula"] == "O2Si"
    assert preview["formula"] == "SiO2"
    assert preview["reordered_from_entered"] is True and preview["reduced_from_entered"] is False
    assert preview["plan_digest"] == build_prediction_plan("SiO2")["plan_digest"]
    assert build_prediction_plan("C2H8")["reduced_display_formula"] == "CH4"


def _preview_page(monkeypatch, formula):
    from fastapi.testclient import TestClient

    import main

    def no_remote(*args, **kwargs):
        raise AssertionError("the New structure preview never touches POWER")

    for name in ("monitor_job", "prepare_remote_submission", "submit_remote_workflow",
                 "load_results_for_completed_job"):
        monkeypatch.setattr(main, name, no_remote)
    response = TestClient(main.app).post("/predict", data={"formula": formula})
    assert response.status_code == 200
    return response.text


def _main_formula(html):
    return re.search(r"data-prediction-formula>(.*?)</span>", html, re.S).group(1).strip()


@pytest.mark.parametrize(("formula", "canonical"), [("CH4", "H4C"), ("Ca(OH)2", "CaH2O2"), ("NH3", "H3N")])
def test_familiar_formulas_stay_the_main_formula_in_the_preview(monkeypatch, formula, canonical):
    html = _preview_page(monkeypatch, formula)
    assert _main_formula(html) == formula
    assert "data-prediction-entered" not in html  # not reduced: no "you entered" notice
    row = re.search(r"data-prediction-canonical-formula>(.*?)</tr>", html, re.S).group(1)
    assert canonical in row  # the normalised formula appears only in the details
    assert canonical not in html.split("data-prediction-canonical-formula")[0].split("data-prediction-plan>")[1]


def test_reduced_formula_notice_uses_the_entered_element_order(monkeypatch):
    html = _preview_page(monkeypatch, "C2H8")
    assert _main_formula(html) == "C2H8"
    notice = re.search(r"data-prediction-entered>(.*?)</p>", html, re.S).group(1)
    assert "C2H8" in notice and "CH4" in notice and "H4C" not in notice


# --- equivalent walltime spellings -----------------------------------------------


@pytest.mark.parametrize("spelling", ["48:00:00", "048:00:00", "0048:00:00", " 48:00:00 "])
def test_equivalent_walltimes_give_one_plan_and_keep_the_entered_text(spelling):
    from backend.structure_search.spec import canonical_walltime

    reference = build_search_plan_from_formula("Si", AdvancedSettings(walltime="48:00:00"))
    plan = build_search_plan_from_formula("Si", AdvancedSettings(walltime=spelling))
    assert plan.digest == reference.digest and plan.canonical == reference.canonical
    assert plan.record["search"]["spec"]["walltime"] == "48:00:00"
    assert plan.record["request"]["advanced_overrides"]["walltime"] == "48:00:00"
    assert plan.provenance["request"]["advanced_overrides_as_entered"]["walltime"] == spelling
    assert canonical_walltime(spelling) == "48:00:00"


def test_equivalent_integer_overrides_give_one_plan():
    a = build_search_plan_from_formula("Si", AdvancedSettings(max_atoms="08", num_generations="030"))
    b = build_search_plan_from_formula("Si", AdvancedSettings(max_atoms=8, num_generations=30))
    assert a.digest == b.digest
    assert a.record["request"]["advanced_overrides"]["max_atoms"] == 8


def test_different_durations_still_give_different_plans():
    assert (build_search_plan_from_formula("Si", AdvancedSettings(walltime="48:00:00")).digest
            != build_search_plan_from_formula("Si", AdvancedSettings(walltime="48:00:01")).digest)


@pytest.mark.parametrize(
    "walltime",
    ["2-00:00:00", "48:00", "48:60:00", "48:00:60", "0:00:00", "2400:00:01", "10000:00:00", "two days", "-1:00:00"],
)
def test_malformed_or_out_of_range_walltimes_are_refused(walltime):
    with pytest.raises(CalculationValidationError):
        build_search_plan_from_formula("Si", AdvancedSettings(walltime=walltime))


@pytest.mark.parametrize("walltime", ["5:00:00", "05:00:00", "048:00:00", "2400:00:00", "168:00:00"])
def test_resource_estimate_keeps_the_requested_duration(walltime):
    from backend.calculations.resources import structure_search_resources, walltime_seconds

    record = build_search_plan_from_formula("Si", AdvancedSettings(walltime=walltime)).record["resources"]
    entered = structure_search_resources(walltime).to_dict()
    planned = record["request"]
    assert walltime_seconds(planned["walltime"]) == walltime_seconds(walltime.strip())
    assert {k: v for k, v in planned.items() if k != "walltime"} == {k: v for k, v in entered.items() if k != "walltime"}
    assert f"#SBATCH --time={planned['walltime']}" in record["sbatch_directives"]
    assert record["status"] == "provisional_estimate" and record["approved"] is False
