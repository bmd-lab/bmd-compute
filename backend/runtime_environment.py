"""Scientific runtime-stack parity between preparation and POWER execution.

Preparation records the exact versions of the parity-critical packages from the
preparation environment into ``submission.json`` (``runtime_parity``). Before
any workflow is built or VASP is started, the POWER runner reads the same
packages from its own environment, writes ``runtime_environment.json`` into the
run directory, and stops if anything required does not match.

Package tiers (the versions themselves live in
``constraints/scientific-runtime.txt``):

* ``PARITY_CRITICAL_PACKAGES`` must be identical at preparation and execution.
  They generate inputs and symmetry/k-points (pymatgen, pymatgen-core, spglib),
  construct and run the workflow (atomate2, jobflow), correct VASP errors
  (custodian) or decide whether a stage converged (emmet-core).
* ``PHONON_PARITY_CRITICAL_PACKAGES`` (phonopy, seekpath) are parity-critical
  *only* for submissions whose workflow contains a phonon force stage; the
  required set is derived from the submission's own ``workflow_spec`` at
  preparation and again on POWER. For every other submission they are recorded
  as supporting packages.
* ``RECORDED_SUPPORTING_PACKAGES`` are pinned in the constraints for
  reproducible installs and recorded at run time, but a difference does not
  stop a run: they are numerical, serialization, configuration or job-store
  substrate that does not choose BMD's inputs or success decisions.

Atomate2 settings that could silently alter generated inputs are also checked:
a non-empty ``VASP_INCAR_UPDATES`` or a truthy ``VASP_INHERIT_INCAR`` stops the
run. This module imports nothing heavy at import time.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


RUNTIME_PARITY_POLICY_ID = "bmd_compute.runtime_parity"
RUNTIME_PARITY_POLICY_VERSION = 1
RUNTIME_ENVIRONMENT_SCHEMA = "bmd_compute.runtime_environment"
RUNTIME_ENVIRONMENT_SCHEMA_VERSION = 1
RUNTIME_ENVIRONMENT_FILENAME = "runtime_environment.json"

PARITY_CRITICAL_PACKAGES = (
    "atomate2",
    "pymatgen",
    "pymatgen-core",
    "custodian",
    "emmet-core",
    "jobflow",
    "spglib",
)
# Parity-critical only for workflows that contain a phonon force stage: they
# choose the phonon supercell, displacements, force constants and band path.
# Other workflows record their versions (or absence) without depending on them,
# so existing PBE/HSE06/DFT+U submissions keep working where they are not yet
# installed.
PHONON_PARITY_CRITICAL_PACKAGES = (
    "phonopy",
    "seekpath",
)
PHONON_STAGE_TYPE = "phonon_forces"
RECORDED_SUPPORTING_PACKAGES = (
    "monty",
    "numpy",
    "scipy",
    "pydantic",
    "pydantic-settings",
    "maggma",
    "ruamel.yaml",
)
RECORDED_ATOMATE2_SETTINGS = (
    "CONFIG_FILE",
    "VASP_HANDLE_UNSUCCESSFUL",
    "VASP_INCAR_UPDATES",
    "VASP_INHERIT_INCAR",
    "SYMPREC",
    "BANDGAP_TOL",
    "VASP_CUSTODIAN_MAX_ERRORS",
    "VASP_ZIP_FILES",
    "VASP_GAMMA_CMD",
    "VASP_NCL_CMD",
    "CUSTODIAN_SCRATCH_DIR",
)

VersionLookup = Callable[[str], str]


class RuntimeParityError(RuntimeError):
    """Raised when execution must not start under the current runtime stack."""

    def __init__(self, problems: list[str], *, record_path: str | None = None):
        self.problems = list(problems)
        self.record_path = record_path
        lines = ["BMD Compute runtime environment check failed; VASP was not started."]
        lines.extend(f"  - {problem}" for problem in self.problems)
        if record_path:
            lines.append(f"  Runtime record: {record_path}")
        super().__init__("\n".join(lines))


def _default_version_lookup(name: str) -> str:
    from importlib import metadata

    return metadata.version(name)


def installed_versions(names, *, version_lookup: VersionLookup | None = None) -> dict[str, str | None]:
    lookup = version_lookup or _default_version_lookup
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            value = lookup(name)
        except Exception:
            value = None
        versions[name] = str(value) if value else None
    return versions


def workflow_requires_phonon_runtime(workflow_spec: Any) -> bool:
    stages = workflow_spec.get("stages") if isinstance(workflow_spec, Mapping) else None
    return any(
        isinstance(stage, Mapping) and stage.get("stage_type") == PHONON_STAGE_TYPE
        for stage in stages or ()
    )


def parity_critical_packages_for_workflow(workflow_spec: Any) -> tuple[str, ...]:
    """Parity-critical package names for a submission with ``workflow_spec``."""

    if workflow_requires_phonon_runtime(workflow_spec):
        return PARITY_CRITICAL_PACKAGES + PHONON_PARITY_CRITICAL_PACKAGES
    return PARITY_CRITICAL_PACKAGES


def parity_critical_packages_for_spec(spec: Any) -> tuple[str, ...]:
    flow_spec = spec.get("flow_spec") if isinstance(spec, Mapping) else None
    workflow_spec = flow_spec.get("workflow_spec") if isinstance(flow_spec, Mapping) else None
    return parity_critical_packages_for_workflow(workflow_spec)


def recorded_supporting_packages_for(critical: tuple[str, ...]) -> tuple[str, ...]:
    return RECORDED_SUPPORTING_PACKAGES + tuple(
        name for name in PHONON_PARITY_CRITICAL_PACKAGES if name not in critical
    )


def prepared_runtime_parity(
    *,
    version_lookup: VersionLookup | None = None,
    packages: tuple[str, ...] = PARITY_CRITICAL_PACKAGES,
) -> dict[str, Any]:
    """The ``runtime_parity`` block recorded in ``submission.json`` at preparation."""

    versions = installed_versions(packages, version_lookup=version_lookup)
    missing = [name for name, version in versions.items() if not version]
    if missing:
        raise RuntimeParityError(
            [
                "The preparation environment is missing parity-critical package(s): "
                + ", ".join(missing)
            ]
        )
    return {
        "policy_id": RUNTIME_PARITY_POLICY_ID,
        "policy_version": RUNTIME_PARITY_POLICY_VERSION,
        "packages": versions,
    }


def prepared_parity_problems(
    prepared: Any,
    required: tuple[str, ...] = PARITY_CRITICAL_PACKAGES,
) -> list[str]:
    """Structural problems in a prepared ``runtime_parity`` block."""

    if not isinstance(prepared, Mapping):
        return ["submission.json has no valid runtime_parity block."]
    problems = []
    if prepared.get("policy_id") != RUNTIME_PARITY_POLICY_ID:
        problems.append(
            f"runtime_parity.policy_id is {prepared.get('policy_id')!r}, "
            f"expected {RUNTIME_PARITY_POLICY_ID!r}."
        )
    version = prepared.get("policy_version")
    if type(version) is not int or version != RUNTIME_PARITY_POLICY_VERSION:
        problems.append(
            f"runtime_parity.policy_version is {version!r}, "
            f"expected {RUNTIME_PARITY_POLICY_VERSION}."
        )
    packages = prepared.get("packages")
    if not isinstance(packages, Mapping):
        problems.append("runtime_parity.packages is not an object.")
        return problems
    if set(packages) != set(required):
        problems.append(
            "runtime_parity.packages lists "
            f"{sorted(packages)}, expected {sorted(required)}."
        )
    for name in required:
        value = packages.get(name)
        if name in packages and (not isinstance(value, str) or not value.strip()):
            problems.append(f"runtime_parity.packages[{name!r}] is not a version string: {value!r}.")
    return problems


def runtime_parity_problems(
    prepared: Any,
    actual: Mapping[str, str | None],
    required: tuple[str, ...] = PARITY_CRITICAL_PACKAGES,
) -> list[str]:
    """Every reason the runtime stack does not satisfy the prepared contract."""

    problems = prepared_parity_problems(prepared, required)
    if problems:
        return problems
    expected = prepared["packages"]
    for name in required:
        runtime_version = actual.get(name)
        if not runtime_version:
            problems.append(f"{name}: prepared {expected[name]}, not installed at runtime.")
        elif runtime_version != expected[name]:
            problems.append(f"{name}: prepared {expected[name]}, runtime {runtime_version}.")
    return problems


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def atomate2_settings_snapshot(settings: Any = None) -> dict[str, Any]:
    if settings is None:
        from atomate2 import SETTINGS as settings  # noqa: N811
    return {name: _json_safe(getattr(settings, name, None)) for name in RECORDED_ATOMATE2_SETTINGS}


def atomate2_settings_problems(snapshot: Mapping[str, Any]) -> list[str]:
    problems = []
    if snapshot.get("VASP_INCAR_UPDATES"):
        problems.append(
            "atomate2 VASP_INCAR_UPDATES is set "
            f"({snapshot['VASP_INCAR_UPDATES']!r}); it would change BMD's INCARs at write time."
        )
    if snapshot.get("VASP_INHERIT_INCAR"):
        problems.append(
            "atomate2 VASP_INHERIT_INCAR is enabled "
            f"({snapshot['VASP_INHERIT_INCAR']!r}); it would copy previous-stage INCAR settings."
        )
    return problems


def write_record_atomically(path: str | Path, record: Mapping[str, Any]) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def runtime_environment_record_path(spec: Mapping[str, Any]) -> Path:
    paths = spec.get("paths") if isinstance(spec, Mapping) else None
    run_dir = (paths or {}).get("run_dir") if isinstance(paths, Mapping) else None
    return Path(run_dir or os.getcwd()) / RUNTIME_ENVIRONMENT_FILENAME


def enforce_runtime_environment(
    spec: Mapping[str, Any],
    *,
    version_lookup: VersionLookup | None = None,
    settings: Any = None,
    record_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Record the runtime stack and stop unless it satisfies the prepared contract.

    Always writes the record (pass or fail) before returning or raising.
    """

    environ = os.environ if environ is None else environ
    prepared = spec.get("runtime_parity") if isinstance(spec, Mapping) else None
    required = parity_critical_packages_for_spec(spec)
    critical = installed_versions(required, version_lookup=version_lookup)
    supporting = installed_versions(
        recorded_supporting_packages_for(required),
        version_lookup=version_lookup,
    )
    problems = runtime_parity_problems(prepared, critical, required)
    try:
        settings_snapshot = atomate2_settings_snapshot(settings)
    except Exception as exc:
        settings_snapshot = {}
        problems.append(f"atomate2 settings could not be read: {type(exc).__name__}: {exc}")
    problems.extend(atomate2_settings_problems(settings_snapshot))

    submission = spec.get("submission") if isinstance(spec, Mapping) else None
    record = {
        "schema": RUNTIME_ENVIRONMENT_SCHEMA,
        "schema_version": RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed" if not problems else "failed",
        "problems": problems,
        "run_name": _json_safe(spec.get("run_name") if isinstance(spec, Mapping) else None),
        "attempt_id": _json_safe((submission or {}).get("attempt_id") if isinstance(submission, Mapping) else None),
        "slurm_job_id": environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "parity_policy": {
            "policy_id": RUNTIME_PARITY_POLICY_ID,
            "policy_version": RUNTIME_PARITY_POLICY_VERSION,
            "critical_packages": list(required),
        },
        "prepared_packages": _json_safe(prepared.get("packages")) if isinstance(prepared, Mapping) else None,
        "runtime_packages": critical,
        "supporting_packages": supporting,
        "atomate2_settings": settings_snapshot,
    }
    path = Path(record_path) if record_path is not None else runtime_environment_record_path(spec)
    write_record_atomically(path, record)
    if problems:
        raise RuntimeParityError(problems, record_path=str(path))
    return record


__all__ = [
    "PARITY_CRITICAL_PACKAGES",
    "PHONON_PARITY_CRITICAL_PACKAGES",
    "parity_critical_packages_for_spec",
    "parity_critical_packages_for_workflow",
    "recorded_supporting_packages_for",
    "workflow_requires_phonon_runtime",
    "RECORDED_ATOMATE2_SETTINGS",
    "RECORDED_SUPPORTING_PACKAGES",
    "RUNTIME_ENVIRONMENT_FILENAME",
    "RUNTIME_ENVIRONMENT_SCHEMA",
    "RUNTIME_ENVIRONMENT_SCHEMA_VERSION",
    "RUNTIME_PARITY_POLICY_ID",
    "RUNTIME_PARITY_POLICY_VERSION",
    "RuntimeParityError",
    "atomate2_settings_problems",
    "atomate2_settings_snapshot",
    "enforce_runtime_environment",
    "installed_versions",
    "prepared_parity_problems",
    "prepared_runtime_parity",
    "runtime_environment_record_path",
    "runtime_parity_problems",
    "write_record_atomically",
]
