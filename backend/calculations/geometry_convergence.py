"""BMD Adaptive Geometry Relaxation v1 (``bmd_compute.geometry_convergence/1``).

A bounded, deterministic plane-wave cutoff check of the geometry produced by
the BMD Standard Geometry Relaxation
(``backend.calculations.geometry_relaxation``). It runs after the final pass of
the canonical two-pass relaxation, inside the same SLURM allocation, and
decides whether that geometry is released to the downstream stages:

1. two-pass relaxation at the species-dependent cutoff E0 (unchanged);
2. single-point probe of the final geometry at the probe cutoff Ep;
3. the probe is evaluated against the acceptance criteria below;
4. accepted: the geometry and its relaxation are released downstream;
5. rejected: one refinement relaxation at E1 (pass-2 settings, starting from
   the rejected geometry), then one final probe at Ep;
6. a final probe that fails stops the workflow with an explicit unresolved
   outcome. No more than one refinement is ever run.

Cutoffs (fixed for the run and frozen into the plan)::

    E1 = E0 + 150 eV                   (refinement relaxation)
    Ep = max(700 eV, E0 + 300 eV)      (both probes)

700 eV is the reference cutoff of the BMD validation benchmark (Si, Al, NiO,
Bi2Se3 relaxed at their species cutoffs and at 700 eV) and the historical BMD
relaxation cutoff; for every cutoff up to 400 eV the probe is that reference.
The probe always lies at least 150 eV above the highest relaxation cutoff, so
the final probe is still a higher-cutoff test of the refined geometry.

Acceptance (all required; values, tolerances and reasons are recorded):

* the relaxation converged electronically and ionically, and the probe
  converged electronically (VASP's own criteria);
* probe maximum residual force <= 0.001 eV/A;
* probe |hydrostatic pressure| <= 1 kbar;
* probe maximum |stress component| <= 2 kbar;
* for spin-polarised calculations, the probe's total and per-site
  magnetisation within 0.1 mu_B of the relaxation's (and a refined
  relaxation's within 0.1 mu_B of the initial relaxation's). The site
  moments come from OUTCAR, written because the spin-polarised final pass and
  every adaptive stage set LORBIT = 11 (output only); without them the
  evidence is invalid - the total moment alone is never accepted as proof. A
  change of magnetic state is never accepted and is not refined: it is
  unresolved.

Every probe and the refinement use the actual KPOINTS file of the relaxation
they follow (frozen, never regenerated from the changed cell), and the record
keeps that mesh's identity. One adaptive canonical relaxation per workflow.

These are engineering acceptance criteria: the geometry is a stationary point,
to these tolerances, of the probe-cutoff Hamiltonian. They are not a proof of
complete basis-set convergence. The k-point mesh, smearing, magnetic
treatment, DFT+U, dispersion and POTCARs are those of the relaxation.

Downstream stages use plan-fixed cutoffs (no runtime propagation in v1):
Static Energy stages use at least E1 (the highest relaxation cutoff the
controller may accept, on top of their own 620 eV floor; PBE DOS/Band
Structure stages restarting from that static keep its cutoff), and phonon
displacement forces use Ep, at which the accepted geometry has been verified.

The policy here is pure (no file system, no VASP); the run-time jobs are in
``backend.calculations.geometry_convergence_jobs``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from backend.calculations.models import StageType

RULE_ID = "bmd_compute.geometry_convergence"
RULE_VERSION = 1
OPTION_KEY = "geometry_convergence"

PROBE_REFERENCE_ENCUT_EV = 700
REFINEMENT_INCREMENT_EV = 150
PROBE_MARGIN_EV = 2 * REFINEMENT_INCREMENT_EV
MAX_REFINEMENTS = 1
ACCEPTANCE = {
    "max_force_ev_a": 0.001,
    "max_abs_pressure_kbar": 1.0,
    "max_abs_stress_component_kbar": 2.0,
    "max_magnetisation_change_mub": 0.1,
}

STAGE_DIRECTORY = "geometry_convergence"
RECORD_FILENAME = "geometry_convergence.json"
ACCEPTED_DIRECTORY = "accepted"
ACCEPTED_FILES = ("CONTCAR", "OUTCAR", "vasprun.xml", "INCAR", "KPOINTS", "POSCAR")
RECORD_SCHEMA = "bmd_compute.geometry_convergence_record"
RECORD_SCHEMA_VERSION = 1

# Stage roles of the worst-case path (resource estimate and record).
ROLE_INITIAL_PROBE = "initial_probe"
ROLE_REFINEMENT = "refinement"
ROLE_FINAL_PROBE = "final_probe"
WORST_CASE_ROLES = (ROLE_INITIAL_PROBE, ROLE_REFINEMENT, ROLE_FINAL_PROBE)

# Status and outcome vocabulary.
STATUS_IN_PROGRESS = "in_progress"
STATUS_ACCEPTED = "accepted"
STATUS_UNRESOLVED = "unresolved"
STATUS_EXECUTION_FAILURE = "execution_failure"
STATUS_RESOURCE_BUDGET = "resource_budget_exhausted"
STATUS_INVALID_EVIDENCE = "invalid_evidence"
OUTCOME_ACCEPTED_WITHOUT_REFINEMENT = "accepted_without_refinement"
OUTCOME_ACCEPTED_AFTER_REFINEMENT = "accepted_after_refinement"
OUTCOME_UNRESOLVED = "unresolved_after_refinement"
OUTCOME_UNRESOLVED_MAGNETIC = "unresolved_magnetic_state_change"
OUTCOME_EXECUTION_FAILURE = "execution_failure"
OUTCOME_RESOURCE_BUDGET = "resource_budget_exhausted"
OUTCOME_INVALID_EVIDENCE = "invalid_evidence"
# Dynamic operations, in order (the record names the one pending).
OPERATION_INITIAL_PROBE = "probe_initial"
OPERATION_INITIAL_DECISION = "decision_initial"
OPERATION_REFINEMENT = "refinement"
OPERATION_REFINEMENT_CHECKPOINT = "checkpoint_refinement"
OPERATION_FINAL_PROBE = "probe_final"
OPERATION_FINAL_DECISION = "decision_final"
# Output-only INCAR tag of the spin-polarised final pass and of every adaptive
# stage: VASP then writes the site-resolved moments (OUTCAR "magnetization
# (x)") the magnetic-state check needs. It does not change the Hamiltonian.
MAGNETIC_EVIDENCE_INCAR = {"LORBIT": 11}
# One adaptive canonical relaxation per workflow in v1.
MAX_ADAPTIVE_RELAXATIONS = 1


# --------------------------------------------------------------------------
# The frozen policy record
# --------------------------------------------------------------------------


def refinement_encut(initial_encut: int) -> int:
    return int(initial_encut) + REFINEMENT_INCREMENT_EV


def probe_encut(initial_encut: int) -> int:
    return max(PROBE_REFERENCE_ENCUT_EV, int(initial_encut) + PROBE_MARGIN_EV)


def convergence_record(initial_encut: int) -> dict[str, Any]:
    """The frozen policy of one final relaxation pass with cutoff ``initial_encut``."""

    initial = int(initial_encut)
    return {
        "rule": {"id": RULE_ID, "version": RULE_VERSION},
        "initial_encut": initial,
        "refinement_encut": refinement_encut(initial),
        "probe_encut": probe_encut(initial),
        "max_refinements": MAX_REFINEMENTS,
        "acceptance": dict(ACCEPTANCE),
    }


def policy_payload() -> dict[str, Any]:
    """The rule definition (plan digest and methodology records)."""

    return {
        "id": RULE_ID,
        "version": RULE_VERSION,
        "refinement_encut": f"initial ENCUT + {REFINEMENT_INCREMENT_EV} eV",
        "probe_encut": f"max({PROBE_REFERENCE_ENCUT_EV} eV, initial ENCUT + {PROBE_MARGIN_EV} eV)",
        "max_refinements": MAX_REFINEMENTS,
        "acceptance": dict(ACCEPTANCE),
        "vasp_convergence": "relaxation electronic and ionic, probe electronic",
        "magnetic_state_change": "unresolved (never accepted, never refined)",
        "probe_settings": "final-pass settings with ENCUT = probe cutoff, NSW = 0, IBRION = -1",
        "refinement_settings": "final-pass settings with ENCUT = refinement cutoff, from the rejected geometry",
        "kpoints": "frozen: every probe and the refinement use the actual KPOINTS file of the relaxation they follow",
        "magnetic_evidence": "spin-polarised: LORBIT = 11 on the final pass and the adaptive stages; site-resolved moments required",
        "max_adaptive_relaxations_per_workflow": MAX_ADAPTIVE_RELAXATIONS,
        "downstream": {
            "static_encut_floor": "refinement cutoff (on top of the stage's own floor)",
            "phonon_force_encut": "probe cutoff",
        },
        "unresolved": "the workflow stops; downstream stages never start",
    }


def probe_incar_overrides(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"ENCUT": int(record["probe_encut"]), "NSW": 0, "IBRION": -1}


def refinement_incar_overrides(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"ENCUT": int(record["refinement_encut"])}


def stage_convergence_record(stage) -> dict[str, Any] | None:
    value = dict(getattr(stage, "options", None) or {}).get(OPTION_KEY)
    return dict(value) if isinstance(value, Mapping) else None


def validate_record(stage, record: Any, *, cutoff_record: Mapping[str, Any] | None) -> list[str]:
    """Problems with a frozen record on ``stage`` (empty when valid)."""

    if stage.stage_type not in (StageType.RELAX, StageType.PHONON_FORCES):
        return ["only the final relaxation pass and its phonon force stage carry one"]
    if not isinstance(record, Mapping):
        return ["not an object"]
    if cutoff_record is None:
        return ["the stage has no plane-wave cutoff record"]
    if dict(record) != convergence_record(int(cutoff_record["ENCUT"])):
        return ["it does not follow from the stage's plane-wave cutoff and this rule version"]
    return []


# --------------------------------------------------------------------------
# Worst-case stages for the resource estimate
# --------------------------------------------------------------------------


def worst_case_stage_previews(stage_previews: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Additional stage previews of the worst-case adaptive path.

    For every final relaxation pass with a frozen record: the initial probe,
    the refinement relaxation and the final probe, generated from that pass's
    inputs exactly as the controller generates them at run time (on the
    submitted structure, as every preview is). They are used by the resource
    estimate only; the workflow stages themselves are unchanged.
    """

    extra: list[dict[str, Any]] = []
    next_index = max((int(p["index"]) for p in stage_previews), default=0) + 1
    for preview in stage_previews:
        record = stage_convergence_record(preview["stage_spec"])
        if record is None or preview["stage_spec"].stage_type is not StageType.RELAX:
            continue
        for role in WORST_CASE_ROLES:
            if role == ROLE_REFINEMENT:
                updates, stage_type = refinement_incar_overrides(record), StageType.RELAX
            else:
                updates, stage_type = probe_incar_overrides(record), StageType.STATIC
            inputs = preview["input_set"]
            incar = dict(inputs.incar)
            incar.update(updates)
            extra.append(
                {
                    **preview,
                    "index": next_index,
                    "label": f"Geometry convergence: {role.replace('_', ' ')}",
                    "directory": STAGE_DIRECTORY,
                    "phonon_plan": None,
                    "geometry_convergence_role": role,
                    "stage_spec": SimpleNamespace(stage_type=stage_type, theory=preview["stage_spec"].theory),
                    "input_set": SimpleNamespace(
                        incar=incar, poscar=inputs.poscar, kpoints=inputs.kpoints, potcar=inputs.potcar
                    ),
                }
            )
            next_index += 1
    return extra


# --------------------------------------------------------------------------
# Evaluation and decision (pure)
# --------------------------------------------------------------------------


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _magnetisation_problem(reference: Mapping[str, Any], current: Mapping[str, Any]) -> tuple[str | None, dict]:
    """(category, detail) comparing two magnetisation records."""

    spin = bool(reference.get("spin_polarized")) or bool(current.get("spin_polarized"))
    if not spin:
        return None, {"applicable": False}
    a, b = reference.get("magnetisation"), current.get("magnetisation")
    if not isinstance(a, Mapping) or not isinstance(b, Mapping) or not _finite(a.get("total")) or not _finite(b.get("total")):
        return "invalid_evidence", {"applicable": True, "magnetic_state_verified": False, "problem": "magnetisation missing"}
    # The magnetic configuration is assessed from the site-resolved moments
    # (OUTCAR "magnetization (x)", written with LORBIT, which the adaptive
    # stages set; it does not change the Hamiltonian). The total moment alone
    # cannot show a changed configuration (an antiferromagnet has none), so
    # without site moments for both runs the evidence is invalid.
    sites_a, sites_b = list(a.get("sites") or []), list(b.get("sites") or [])
    if not sites_a or not sites_b:
        return "invalid_evidence", {
            "applicable": True,
            "magnetic_state_verified": False,
            "problem": "site-resolved magnetic moments missing (total magnetisation alone is not sufficient)",
        }
    if len(sites_a) != len(sites_b) or not all(_finite(v) for v in sites_a + sites_b):
        return "invalid_evidence", {
            "applicable": True,
            "magnetic_state_verified": False,
            "problem": "site-resolved magnetic moments inconsistent",
        }
    tolerance = ACCEPTANCE["max_magnetisation_change_mub"]
    total_change = abs(float(b["total"]) - float(a["total"]))
    site_change = max(abs(float(x) - float(y)) for x, y in zip(sites_a, sites_b))
    changed = total_change > tolerance or site_change > tolerance
    detail = {
        "applicable": True,
        "magnetic_state_verified": not changed,
        "reference_total_mub": float(a["total"]),
        "total_mub": float(b["total"]),
        "total_change_mub": total_change,
        "max_site_change_mub": site_change,
        "reference_site_moments_mub": [float(v) for v in sites_a],
        "site_moments_mub": [float(v) for v in sites_b],
        "tolerance_mub": tolerance,
    }
    if changed:
        return "magnetic_state_changed", detail
    return None, detail


def evaluate_probe(
    *,
    relaxation: Mapping[str, Any],
    probe: Mapping[str, Any],
    reference_relaxation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one probe of one relaxation. Pure and deterministic.

    ``relaxation`` and ``probe`` are evidence records (see
    ``geometry_convergence_jobs.read_stage_evidence``). ``reference_relaxation``
    is the initial relaxation when ``relaxation`` is the refinement.

    Returns ``{"category": "accepted" | "rejected" | "magnetic_state_changed"
    | "vasp_not_converged" | "invalid_evidence", "checks": [...], "reasons": [...]}``.
    """

    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    def invalid(reason: str) -> dict[str, Any]:
        return {"category": "invalid_evidence", "checks": checks, "reasons": [reason]}

    for name, evidence in (("relaxation", relaxation), ("probe", probe)):
        if not isinstance(evidence, Mapping):
            return invalid(f"{name} evidence missing")
    for name, evidence, keys in (
        ("relaxation", relaxation, ("converged_electronic", "converged_ionic")),
        ("probe", probe, ("converged_electronic",)),
    ):
        for key in keys:
            value = evidence.get(key)
            if not isinstance(value, bool):
                return invalid(f"{name} {key} not recorded")
            checks.append({"name": f"{name}_{key}", "value": value, "required": True, "passed": value})
    if not all(check["passed"] for check in checks):
        failed = [check["name"] for check in checks if not check["passed"]]
        return {"category": "vasp_not_converged", "checks": checks, "reasons": [f"VASP did not converge: {', '.join(failed)}"]}

    limits = (
        ("max_force_ev_a", "max_force_ev_a", "maximum residual force (eV/A)"),
        ("pressure_kbar", "max_abs_pressure_kbar", "|hydrostatic pressure| (kbar)"),
        ("max_stress_kbar", "max_abs_stress_component_kbar", "maximum |stress component| (kbar)"),
    )
    for key, tolerance_key, label in limits:
        value = probe.get(key)
        if not _finite(value):
            return invalid(f"probe {label} not available")
        tolerance = ACCEPTANCE[tolerance_key]
        passed = abs(float(value)) <= tolerance
        checks.append({"name": f"probe_{key}", "value": float(value), "tolerance": tolerance, "passed": passed})
        if not passed:
            reasons.append(f"probe {label} {abs(float(value)):.6g} exceeds {tolerance:g}")

    for name, ref, cur in (
        ("probe_magnetisation", relaxation, probe),
        ("refinement_magnetisation", reference_relaxation, relaxation),
    ):
        if ref is None:
            continue
        category, detail = _magnetisation_problem(ref, cur)
        checks.append({"name": name, **detail, "passed": category is None})
        if category == "invalid_evidence":
            return invalid(f"{name}: {detail['problem']}")
        if category == "magnetic_state_changed":
            return {
                "category": "magnetic_state_changed",
                "checks": checks,
                "reasons": [
                    f"{name} changed by {detail['total_change_mub']:.3g} mu_B (total), "
                    f"{detail['max_site_change_mub']:.3g} mu_B (largest site); tolerance "
                    f"{detail['tolerance_mub']:g} mu_B"
                ],
            }
    return {"category": "accepted" if not reasons else "rejected", "checks": checks, "reasons": reasons}


def decide(record: Mapping[str, Any], *, refinements_done: int, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """The controller's action after one evaluation. Pure and deterministic."""

    category = evaluation["category"]
    reasons = list(evaluation.get("reasons") or [])
    if category == "accepted":
        outcome = OUTCOME_ACCEPTED_WITHOUT_REFINEMENT if refinements_done == 0 else OUTCOME_ACCEPTED_AFTER_REFINEMENT
        return {"action": "accept", "status": STATUS_ACCEPTED, "outcome": outcome, "reasons": reasons}
    if category == "rejected":
        if refinements_done < int(record["max_refinements"]):
            return {"action": "refine", "status": STATUS_IN_PROGRESS, "outcome": None, "reasons": reasons}
        return {
            "action": "stop",
            "status": STATUS_UNRESOLVED,
            "outcome": OUTCOME_UNRESOLVED,
            "reasons": reasons + ["the permitted refinement is exhausted"],
        }
    if category == "magnetic_state_changed":
        return {"action": "stop", "status": STATUS_UNRESOLVED, "outcome": OUTCOME_UNRESOLVED_MAGNETIC, "reasons": reasons}
    if category == "invalid_evidence":
        return {"action": "stop", "status": STATUS_INVALID_EVIDENCE, "outcome": OUTCOME_INVALID_EVIDENCE, "reasons": reasons}
    return {"action": "stop", "status": STATUS_EXECUTION_FAILURE, "outcome": OUTCOME_EXECUTION_FAILURE, "reasons": reasons}


__all__ = [
    "ACCEPTANCE",
    "MAX_REFINEMENTS",
    "OPTION_KEY",
    "RULE_ID",
    "RULE_VERSION",
    "STAGE_DIRECTORY",
    "convergence_record",
    "decide",
    "evaluate_probe",
    "policy_payload",
    "probe_encut",
    "probe_incar_overrides",
    "refinement_encut",
    "refinement_incar_overrides",
    "stage_convergence_record",
    "validate_record",
    "worst_case_stage_previews",
]
