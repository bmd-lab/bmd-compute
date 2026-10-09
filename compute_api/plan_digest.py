"""Stable digest of a resolved BMD Compute calculation plan.

The plan digest identifies *what would be calculated, and with which
resources*, independently of when or where it would be prepared. It is computed
from the server-built submission specification and the per-stage generated
inputs, after automatic treatments and admission. It complements, and does not
replace, the submission-attempt fingerprint
(``backend.submission.submission_attempt_fingerprint``), which binds one
immutable prepared attempt, including its timestamped paths.

Covered (any change alters the digest):

* the structure, as parsed: lattice matrix, site species, fractional
  coordinates and site properties (not the file text, so a changed POSCAR
  comment line or whitespace that parses to the same structure does not alter
  it);
* the resolved workflow: ordered stages, stage types, theories, modifiers and
  stage options, including frozen DFT+U parameters;
* the automatic-treatment record (applied, omitted, not applicable, advisory,
  and the prepared DFT+U record), or its absence for Custom workflows;
* for every stage: the VASP executable, the generated INCAR settings, the
  generated KPOINTS file text, the SHA-256 of the generated POSCAR file text and
  the POTCAR symbols. These are the inputs Compute's stage input-set generators
  produce for each stage from the submitted structure (the same generator path
  as the browser preview); at run time stage 2 onwards is regenerated from the
  previous stage's output, as described in ``docs/methodology.md``;
* the POTCAR functional and the per-stage POTCAR symbol record;
* execution resources: nodes, MPI tasks, memory, walltime, partition, account;
* the loaded software modules and the VASP command template;
* the parity-critical scientific package versions (``runtime_parity``);
* the versions of the policies that decide the plan: new-calculation
  admission, automatic DFT+U, method considerations, Custodian handling and
  runtime parity, and this digest's own version.

Not covered (changing these does not alter the digest):

* run timestamp, run name, label and creation time;
* submission attempt ID, attempt fingerprint and identity token;
* remote filesystem paths, POTCAR repository location, runner and log paths;
* SSH connection profile and remote Python environment location;
* generated SLURM script text (its scientific and resource content is covered
  above through its inputs);
* BMD Compute Git commit, dirty state and runtime-package manifest. A code
  change that alters generated inputs, treatments or policy versions changes
  the digest through those fields; a code change that alters none of them does
  not.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.calculations.admission import ADMISSION_POLICY_ID, ADMISSION_POLICY_VERSION
from backend.calculations.custodian_policy import (
    BMD_CUSTODIAN_POLICY_ID,
    BMD_CUSTODIAN_POLICY_VERSION,
)
from backend.calculations.dft_u_policy import POLICY_ID as DFT_U_POLICY_ID
from backend.calculations.dft_u_policy import POLICY_VERSION as DFT_U_POLICY_VERSION
from backend.calculations.method_considerations import (
    POLICY_VERSION as METHOD_CONSIDERATIONS_POLICY_VERSION,
)
from backend.runtime_environment import RUNTIME_PARITY_POLICY_ID, RUNTIME_PARITY_POLICY_VERSION

from compute_api.canonical import canonical_json, json_safe, sha256_hex


PLAN_DIGEST_SCHEMA = "bmd_compute.plan_digest"
PLAN_DIGEST_VERSION = 1

_POTCAR_RECORD_EXCLUDED_KEYS = frozenset({"repository", "target", "symlink_targets"})
_ENVIRONMENT_COVERED_KEYS = ("VASP_CMD",)


def policy_versions() -> dict[str, dict[str, Any]]:
    return {
        "new_calculation_admission": {"id": ADMISSION_POLICY_ID, "version": ADMISSION_POLICY_VERSION},
        "automatic_dft_u": {"id": DFT_U_POLICY_ID, "version": DFT_U_POLICY_VERSION},
        "method_considerations": {"version": METHOD_CONSIDERATIONS_POLICY_VERSION},
        "custodian": {"id": BMD_CUSTODIAN_POLICY_ID, "version": BMD_CUSTODIAN_POLICY_VERSION},
        "runtime_parity": {"id": RUNTIME_PARITY_POLICY_ID, "version": RUNTIME_PARITY_POLICY_VERSION},
        "plan_digest": {"id": PLAN_DIGEST_SCHEMA, "version": PLAN_DIGEST_VERSION},
    }


def canonical_structure(structure) -> dict[str, Any]:
    return {
        "lattice": json_safe(structure.lattice.matrix),
        "species": [site.species_string for site in structure],
        "frac_coords": json_safe(structure.frac_coords),
        "site_properties": json_safe(dict(sorted(structure.site_properties.items()))),
    }


def potcar_symbols(input_set) -> list[str]:
    """POTCAR symbols from the symbol-only (``potcar_spec``) generated input."""

    return [
        line.strip()
        for line in str(getattr(input_set, "potcar", "") or "").splitlines()
        if line.strip()
    ]


def stage_input_material(stage_previews: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    material = []
    for preview in stage_previews:
        input_set = preview["input_set"]
        kpoints = input_set.kpoints
        material.append(
            {
                "index": preview["index"],
                "vasp_executable": preview["vasp_executable"],
                "incar": json_safe(dict(input_set.incar)),
                "kpoints_text": str(kpoints).rstrip() if kpoints is not None else None,
                "poscar_sha256": sha256_hex(str(input_set.poscar).rstrip()),
                "potcar_symbols": potcar_symbols(input_set),
            }
        )
    return material


def plan_digest_material(
    *,
    structure,
    submission_spec: Mapping[str, Any],
    stage_previews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    flow_spec = submission_spec["flow_spec"]
    resources = submission_spec["resources"]
    cluster = submission_spec["cluster"]
    environment = submission_spec.get("environment") or {}
    potcar = {
        key: value
        for key, value in (submission_spec.get("potcar") or {}).items()
        if key not in _POTCAR_RECORD_EXCLUDED_KEYS
    }
    return {
        "digest": {"schema": PLAN_DIGEST_SCHEMA, "version": PLAN_DIGEST_VERSION},
        "structure": canonical_structure(structure),
        "workflow": json_safe(flow_spec["workflow_spec"]),
        "automatic_treatments": json_safe(flow_spec.get("automatic_treatments")),
        "stages": stage_input_material(stage_previews),
        "potcar": json_safe(potcar),
        "resources": {
            "nodes": resources["nodes"],
            "ntasks": resources["ntasks"],
            "mem_gb": resources["mem_gb"],
            "walltime": resources["walltime"],
            "partition": cluster["partition"],
            "account": cluster["account"],
        },
        "software": {
            "modules": json_safe(submission_spec.get("modules")),
            "environment": {key: environment.get(key) for key in _ENVIRONMENT_COVERED_KEYS},
        },
        "runtime_parity_packages": json_safe((submission_spec.get("runtime_parity") or {}).get("packages")),
        "policies": policy_versions(),
    }


def compute_plan_digest(
    *,
    structure,
    submission_spec: Mapping[str, Any],
    stage_previews: Sequence[Mapping[str, Any]],
) -> str:
    material = plan_digest_material(
        structure=structure,
        submission_spec=submission_spec,
        stage_previews=stage_previews,
    )
    return "sha256:" + sha256_hex(canonical_json(material))
