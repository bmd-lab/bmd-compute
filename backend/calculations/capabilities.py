from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.calculations.default_treatments import automatic_default_treatment_policy
from backend.calculations.dispersion import dispersion_modifier_policy
from backend.calculations.models import Theory
from backend.calculations.vasp_stage_definitions import (
    EXECUTABLE_METHODOLOGY_SCOPE,
    describe_stage,
    list_stage_definitions,
)


SCHEMA_VERSION = 1
# ``SCOPE`` is a human-readable description. Consumers must establish
# compatibility from ``schema_version`` and ``source.repository``, not from
# this text, which may be reworded without a schema change.
SCOPE = EXECUTABLE_METHODOLOGY_SCOPE
REPOSITORY_ID = "bmd_compute"


def build_capability_payload(
    *,
    include_provenance: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    base_stage_definitions = list(list_stage_definitions())
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "source": (
            git_provenance(repo_root=repo_root)
            if include_provenance
            else unavailable_git_provenance()
        ),
        "contract": {
            "base_stage_definitions": "Theory-neutral stage definitions from list_stage_definitions().",
            "capabilities": "Supported stage/theory descriptions from describe_stage(); unsupported combinations are not invented.",
            "modifier_policies": "Stage-local executable modifiers with controlled options; unsupported pairings are not invented.",
            "automatic_default_treatments": "Read-only policy describing BMD-managed Desired Output treatment resolution.",
        },
        "base_stage_definitions": base_stage_definitions,
        "capabilities": _supported_stage_theory_capabilities(base_stage_definitions),
        "modifier_policies": [dispersion_modifier_policy()],
        "automatic_default_treatments": automatic_default_treatment_policy(),
    }


def git_provenance(*, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repository_root()
    commit = _git_output(("rev-parse", "HEAD"), root)
    dirty_output = _git_output(("status", "--porcelain"), root)
    if commit is None or dirty_output is None:
        return unavailable_git_provenance()
    return {
        "repository": REPOSITORY_ID,
        "commit": commit,
        "dirty": bool(dirty_output.strip()),
        "provenance_available": True,
        "unavailable_reason": None,
    }


def unavailable_git_provenance() -> dict[str, Any]:
    return {
        "repository": REPOSITORY_ID,
        "commit": None,
        "dirty": None,
        "provenance_available": False,
        "unavailable_reason": "git provenance unavailable",
    }


def emit_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    if args:
        print("backend.calculations.capabilities does not accept arguments.", file=sys.stderr)
        return 2

    try:
        sys.stdout.write(emit_json(build_capability_payload()))
    except Exception as exc:
        print(f"Failed to build BMD Compute capability payload: {exc}", file=sys.stderr)
        return 1
    return 0


def _supported_stage_theory_capabilities(
    base_stage_definitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    capabilities: list[dict[str, Any]] = []
    for stage_definition in base_stage_definitions:
        stage_type = stage_definition["stage_type"]
        for theory in Theory:
            description = describe_stage(stage_type, theory)
            if description["theory_supported_for_stage"] is True:
                capabilities.append(description)
    return capabilities


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_executable() -> str:
    return "git"


def _git_command(args: tuple[str, ...], repo_root: Path) -> tuple[str, ...]:
    """Build a read-only git command for producer provenance.

    ``--no-optional-locks`` stops commands such as ``git status`` from
    opportunistically refreshing and rewriting ``.git/index`` (and creating
    ``.git/index.lock``). Producers run against the live BMD Compute checkout,
    including when invoked by read-only observers such as BMD Agent, so
    provenance inspection must never write to that checkout.
    """

    return (
        _git_executable(),
        "--no-optional-locks",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "-C",
        str(repo_root),
        *args,
    )


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _git_output(args: tuple[str, ...], repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            _git_command(args, repo_root),
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
            env=_git_environment(),
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
