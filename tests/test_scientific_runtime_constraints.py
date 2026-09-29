"""The declared production stack and the runtime-parity contract cannot drift apart."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import yaml
from packaging.requirements import Requirement

from backend.provenance import SCIENTIFIC_PACKAGE_NAMES
from backend.runtime_environment import PARITY_CRITICAL_PACKAGES, RECORDED_SUPPORTING_PACKAGES


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = REPO_ROOT / "constraints" / "scientific-runtime.txt"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9.]*)$")


def _tiers() -> dict[str, dict[str, str]]:
    tiers: dict[str, dict[str, str]] = {}
    current = None
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("# Tier 1"):
            current = "critical"
        elif line.startswith("# Tier 2"):
            current = "supporting"
        if not line or line.startswith("#"):
            continue
        match = PIN.match(line)
        assert match, f"not an exact pin: {raw!r}"
        assert current, f"pin outside a tier: {raw!r}"
        name, version = match.groups()
        assert all(name not in tier for tier in tiers.values()), f"duplicate pin: {name}"
        tiers.setdefault(current, {})[name] = version
    return tiers


def test_every_entry_is_an_exact_pin_in_a_declared_tier():
    tiers = _tiers()
    assert set(tiers) == {"critical", "supporting"}


def test_tier_one_is_exactly_the_runtime_parity_set():
    assert list(_tiers()["critical"]) == list(PARITY_CRITICAL_PACKAGES)


def test_tier_two_is_exactly_the_recorded_supporting_set():
    assert list(_tiers()["supporting"]) == list(RECORDED_SUPPORTING_PACKAGES)


def test_intentional_pins_are_preserved():
    critical = _tiers()["critical"]
    assert critical["atomate2"] == "0.1.5"
    assert critical["pymatgen"] == "2026.5.4"
    assert critical["pymatgen-core"] == "2026.7.16"


def test_monty_pin_satisfies_the_pinned_pymatgen_core():
    tiers = _tiers()
    assert metadata.version("pymatgen-core") == tiers["critical"]["pymatgen-core"]
    monty_requirements = [
        Requirement(text)
        for text in metadata.requires("pymatgen-core") or []
        if Requirement(text).name == "monty" and "extra" not in text
    ]
    assert monty_requirements
    for requirement in monty_requirements:
        assert requirement.specifier.contains(tiers["supporting"]["monty"])


def test_environment_yml_takes_the_scientific_stack_only_from_the_constraints():
    environment = yaml.safe_load((REPO_ROOT / "environment.yml").read_text(encoding="utf-8"))
    pinned = {name.lower() for tier in _tiers().values() for name in tier}
    conda_names = [item.split("=")[0].strip().lower() for item in environment["dependencies"] if isinstance(item, str)]
    pip_items = [item for item in environment["dependencies"] if isinstance(item, dict)][0]["pip"]

    assert pip_items == ["-r constraints/scientific-runtime.txt"]
    assert not pinned.intersection(conda_names)


def test_provenance_records_every_constrained_package():
    names = [name for tier in _tiers().values() for name in tier]
    assert sorted(SCIENTIFIC_PACKAGE_NAMES) == sorted(names)


def test_running_stack_matches_the_parity_critical_pins():
    for name, version in _tiers()["critical"].items():
        assert metadata.version(name) == version, name
