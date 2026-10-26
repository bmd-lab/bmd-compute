"""Automatic execution resource selection (R1).

BMD Compute chooses CPUs, memory and walltime for a calculation whose resources
the user leaves on Automatic. Values the user (browser) or client (machine API)
gives explicitly are kept exactly, subject to the existing validation in
``backend.calculations.resources``. This module is the only place where
automatic resources are decided; the browser and the machine API both reach it
through ``main.build_submission_state_from_structure``.

Policy ``bmd_compute.automatic_resources`` version 2
---------------------------------------------------

CPUs: the existing default of 24 MPI ranks, a conservative initial selection.

Memory: a conservative bracket per stage from the same descriptors as the
walltime (no separate model); the workflow takes the largest stage bracket:

* 128 GB: HSE06, meta-GGA or any non-PBE theory, descriptors unavailable, or
  outside the calibrated descriptor range (atoms, bands, k-points, D);
* 64 GB: within that range, with SOC, a phonon displacement stage, or 33-64
  atoms;
* 32 GB: other PBE stages (including spin, DFT+U and D3) up to 32 atoms.

The brackets assume 24 ranks; each doubling of the CPUs above 24 raises the
bracket one step, never above 128 GB. 128 GB (version 1's fixed request) is
therefore the largest automatic memory. On the 16 October 2026 telemetry every
measured PBE stage stayed below 5 GB of SLURM step MaxRSS, the 64-atom Si
phonon displacements below 7 GB, and only hybrid stages exceeded 8 GB.

Walltime: a conservative, chemistry-independent empirical estimate built from
plan-time computational descriptors of every stage. The descriptors are read
from the same generated stage inputs that the preview, the plan digest and the
runner use (INCAR, KPOINTS, POSCAR and POTCAR symbols):

* electrons: POTCAR valence (ZVAL) per species, or NELECT when set;
* bands: VASP's default NBANDS rule (spin and SOC aware, rounded to the band
  groups of the requested ranks and NCORE), or NBANDS when set;
* k-points: irreducible count of the automatic mesh from spglib, the full mesh
  with SOC, ISYM <= 0 or non-uniform magnetic moments, or the explicit count;
* FFT grid: per-axis PREC=Accurate grid from ENCUT and the lattice lengths,
  rounded up to an FFT-friendly size;
* spin, SOC, hybrid functional, stage type, relaxation degrees of freedom and,
  for phonons, the number of displaced supercells.

The cost descriptor is::

    D = k-points x bands x NG x log2(NG) x (2 for ISPIN=2 or SOC)

For PBE static, relaxation and phonon-displacement stages inside the
calibrated domain the per-SCF-step time is bounded by the upper envelope
fitted on the 16 October 2026 BMD Compute telemetry (all compute nodes,
including the slowest, so node-to-node variability is inside the envelope)::

    t_step <= 1.61e-6 * D**0.75 seconds

and a stage is allowed::

    1.5 * [t_step * (N_scf + 21 + 4 * N_ionic) + 30 s] * N_calculations

with N_scf = 60 for single-point stages; for relaxations
N_ionic = min(NSW, 3 x degrees of freedom) and N_scf = 60 + 20 (N_ionic - 1).
The workflow adds 300 s plus 15 % for Jobflow, Custodian and transfers. In a
leave-one-chemistry-out test this covered 31/31 held-out PBE runs from seven
chemistries, with a median request about 33 times the actual runtime. It is a
reliability-first baseline, not a statistical guarantee.

Outside that domain Compute does not extrapolate silently:

* PBE stages outside the calibrated ranges, PBE DOS/band stages and r2SCAN are
  *extrapolated*: three times the model and at least 72 h for the workflow;
* HSE06 stages use fixed *conservative fallbacks* per stage chosen by D:
  36 h (D <= 2e10, the range of every completed HSE06 run), 7 days
  (D <= 1e11) and 21 days beyond, or with SOC;
* stages whose descriptors cannot be determined get 72 h.

The request is rounded up (15 min below 24 h, whole hours above), never below
1 h and never above the 100-day limit of the ``owner100d`` QOS.
"""

from __future__ import annotations

import bz2
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence

from backend.calculations.models import StageType, Theory
from backend.calculations.resources import (
    MAX_WALLTIME_SECONDS,
    ExecutionResources,
    format_walltime,
    normalize_execution_resources,
)
from backend.config import DEFAULT_ACCOUNT, DEFAULT_PARTITION, DEFAULT_RESOURCES


POLICY_ID = "bmd_compute.automatic_resources"
POLICY_VERSION = 2

SOURCE_AUTOMATIC = "automatic"
SOURCE_MANUAL = "manual"

AUTOMATIC_VALUES = frozenset({"", "auto", "automatic"})

AUTOMATIC_CPUS = DEFAULT_RESOURCES["ntasks"]
# The largest automatic memory bracket (version 1's fixed request).
AUTOMATIC_MEMORY_GB = DEFAULT_RESOURCES["mem_gb"]
MEMORY_BRACKETS_GB = (32, 64, AUTOMATIC_MEMORY_GB)
MEMORY_SMALL_MAX_ATOMS = 32
MEMORY_REFERENCE_CPUS = 24

CALIBRATION = {
    "dataset": "BMD Compute telemetry snapshot 2026-10-16",
    "pbe_stages": 58,
    "chemistries": 7,
    "validation": "leave-one-chemistry-out: 31/31 held-out PBE runs covered",
    "cpus": 24,
}

# Conservative PBE per-SCF-step envelope (seconds) over all observed nodes.
PBE_STEP_ENVELOPE_COEFFICIENT = 1.61e-6
PBE_STEP_ENVELOPE_EXPONENT = 0.75
SAFETY_MARGIN = 1.5
STAGE_FIXED_OVERHEAD_S = 30.0
STAGE_INITIALISATION_STEPS = 21.0
IONIC_STEP_OVERHEAD_STEPS = 4.0
SINGLE_POINT_SCF_STEPS = 60
RELAX_FIRST_IONIC_SCF_STEPS = 60
RELAX_LATER_IONIC_SCF_STEPS = 20
RELAX_IONIC_STEPS_PER_DEGREE_OF_FREEDOM = 3
DEFAULT_NSW = 99
WORKFLOW_OVERHEAD_S = 300.0
WORKFLOW_OVERHEAD_FRACTION = 0.15
NONSCF_NBANDS_FACTOR = 1.2

# Calibrated domain (PBE completed stages in the telemetry snapshot).
CALIBRATED_STAGE_TYPES = frozenset({StageType.STATIC, StageType.RELAX, StageType.PHONON_FORCES})
CALIBRATED_MAX_DESCRIPTOR = 5.5e10
CALIBRATED_MAX_BANDS = 315
CALIBRATED_MAX_ATOMS = 64
CALIBRATED_MAX_KPOINTS = 343

EXTRAPOLATION_FACTOR = 3.0
EXTRAPOLATED_WORKFLOW_FLOOR_S = 72 * 3600

HOUR = 3600
DAY = 24 * HOUR
HYBRID_FALLBACK_TIERS = (
    (2e10, 36 * HOUR),
    (1e11, 7 * DAY),
)
HYBRID_FALLBACK_MAX_S = 21 * DAY
UNKNOWN_STAGE_FALLBACK_S = 72 * HOUR

MIN_WALLTIME_S = HOUR
SHORT_ROUNDING_S = 15 * 60
LONG_ROUNDING_S = HOUR

BASIS_EMPIRICAL = "empirical_estimate"
BASIS_FALLBACK = "conservative_fallback"

STAGE_CALIBRATED = "calibrated"
STAGE_EXTRAPOLATED = "extrapolated"
STAGE_HYBRID_FALLBACK = "hybrid_fallback"
STAGE_UNKNOWN_FALLBACK = "unknown_fallback"

RYDBERG_EV = 13.605693122994
BOHR_ANGSTROM = 0.529177210903
SYMPREC = 1e-5


# ------------------------------------------------------------------ request


@dataclass(frozen=True)
class ResourceRequest:
    """Resources as requested. ``None`` means Automatic."""

    cpus: int | None = None
    memory_gb: int | None = None
    walltime: str | None = None
    queue: str | None = None

    @classmethod
    def from_values(cls, *, cpus=None, memory_gb=None, walltime=None, queue=None) -> "ResourceRequest":
        return cls(
            cpus=_requested(cpus),
            memory_gb=_requested(memory_gb),
            walltime=_requested(walltime),
            queue=_requested(queue),
        )

    @classmethod
    def from_execution_resources(cls, resources: ExecutionResources) -> "ResourceRequest":
        """Every field explicit (the behaviour of callers that pass resources)."""

        resources = normalize_execution_resources(resources)
        return cls(
            cpus=resources.cpus,
            memory_gb=resources.memory_gb,
            walltime=resources.walltime,
            queue=resources.queue,
        )

    def source(self, name: str) -> str:
        return SOURCE_AUTOMATIC if getattr(self, name) is None else SOURCE_MANUAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpus": self.cpus,
            "memory_gb": self.memory_gb,
            "walltime": self.walltime,
            "queue": self.queue,
        }


def _requested(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return None if text.lower() in AUTOMATIC_VALUES else text
    return value


@dataclass(frozen=True)
class ResourceSelection:
    resources: ExecutionResources
    request: ResourceRequest
    walltime_estimate: dict[str, Any]
    record: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ public


def resolve_execution_resources(
    request: ResourceRequest | ExecutionResources | None,
    *,
    stage_previews: Sequence[Mapping[str, Any]],
    potcar_functional: str = "PBE_64",
) -> ResourceSelection:
    """Resolve Automatic fields; keep explicit fields exactly."""

    if request is None:
        request = ResourceRequest()
    elif isinstance(request, ExecutionResources):
        request = ResourceRequest.from_execution_resources(request)

    cpus = request.cpus if request.cpus is not None else AUTOMATIC_CPUS
    memory_gb = request.memory_gb if request.memory_gb is not None else AUTOMATIC_MEMORY_GB
    queue = request.queue if request.queue is not None else DEFAULT_PARTITION
    # Validate explicit CPU/memory/queue before estimating (the estimate uses CPUs).
    partial = ExecutionResources(
        cpus=cpus,
        memory_gb=memory_gb,
        walltime=request.walltime or DEFAULT_RESOURCES["walltime"],
        queue=queue,
        account=DEFAULT_ACCOUNT,
    )
    # Explicit values, normalised by the validation above (e.g. "48" -> 48).
    request = ResourceRequest(
        cpus=partial.cpus if request.cpus is not None else None,
        memory_gb=partial.memory_gb if request.memory_gb is not None else None,
        walltime=partial.walltime if request.walltime is not None else None,
        queue=partial.queue if request.queue is not None else None,
    )
    estimate = estimate_walltime(
        stage_previews,
        cpus=partial.cpus,
        potcar_functional=potcar_functional,
    )
    walltime = request.walltime if request.walltime is not None else estimate["walltime"]
    memory = estimate_memory(estimate["stages"], cpus=partial.cpus)
    resources = ExecutionResources(
        cpus=partial.cpus,
        memory_gb=request.memory_gb if request.memory_gb is not None else memory["memory_gb"],
        walltime=walltime,
        queue=partial.queue,
        account=DEFAULT_ACCOUNT,
    )
    record = selection_record(request, resources, estimate, memory)
    return ResourceSelection(
        resources=resources,
        request=request,
        walltime_estimate=estimate,
        record=record,
    )


def selection_record(
    request: ResourceRequest,
    resources: ExecutionResources,
    estimate: Mapping[str, Any],
    memory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    memory = memory or estimate_memory(estimate.get("stages") or [], cpus=resources.cpus)
    cpus_source = request.source("cpus")
    memory_source = request.source("memory_gb")
    walltime_source = request.source("walltime")
    return {
        "policy": {"id": POLICY_ID, "version": POLICY_VERSION},
        "fields": {
            "cpus": {
                "source": cpus_source,
                "value": resources.cpus,
                "basis": (
                    "BMD Compute default of 24 CPUs (conservative initial selection)"
                    if cpus_source == SOURCE_AUTOMATIC
                    else "set manually"
                ),
            },
            "memory_gb": {
                "source": memory_source,
                "value": resources.memory_gb,
                "recommended": memory["memory_gb"],
                "basis": memory["basis"] if memory_source == SOURCE_AUTOMATIC else "set manually",
                "deciding_stage": memory["deciding_stage"],
            },
            "walltime": {
                "source": walltime_source,
                "value": resources.walltime,
                "recommended": estimate["walltime"],
                "basis": estimate["basis"] if walltime_source == SOURCE_AUTOMATIC else "set manually",
            },
            "queue": {"source": request.source("queue"), "value": resources.queue},
        },
        "walltime_estimate": dict(estimate),
    }


def selection_digest_material(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The part of a selection record that identifies a plan (no prose, no floats)."""

    if not record:
        return None
    fields = record.get("fields") or {}
    estimate = record.get("walltime_estimate") or {}
    return {
        "policy": dict(record.get("policy") or {}),
        "sources": {name: (fields.get(name) or {}).get("source") for name in sorted(fields)},
        "recommended_walltime": (fields.get("walltime") or {}).get("recommended"),
        "recommended_memory_gb": (fields.get("memory_gb") or {}).get("recommended"),
        "walltime_basis": estimate.get("basis"),
        "stage_classes": [stage.get("class") for stage in estimate.get("stages") or []],
    }


def planned_resources_token(resources: ExecutionResources | Mapping[str, Any], record: Mapping[str, Any] | None) -> str:
    """Compact text that binds the resources approved at planning.

    The browser carries it from Build to Prepare and Submit, so a calculation
    planned under one policy can never silently change its resources.
    """

    if isinstance(resources, ExecutionResources):
        values = resources.to_dict()
    else:
        values = dict(resources)
    material = selection_digest_material(record) or {}
    policy = material.get("policy") or {}
    sources = material.get("sources") or {}
    return ";".join(
        [
            f"cpus={values['ntasks']}:{sources.get('cpus', SOURCE_MANUAL)}",
            f"memory_gb={values['mem_gb']}:{sources.get('memory_gb', SOURCE_MANUAL)}",
            f"walltime={values['walltime']}:{sources.get('walltime', SOURCE_MANUAL)}",
            f"policy={policy.get('id', POLICY_ID)}/{policy.get('version', POLICY_VERSION)}",
        ]
    )


# ------------------------------------------------------------------ estimate


def estimate_walltime(
    stage_previews: Sequence[Mapping[str, Any]],
    *,
    cpus: int = AUTOMATIC_CPUS,
    potcar_functional: str = "PBE_64",
) -> dict[str, Any]:
    stages = [
        estimate_stage(preview, cpus=cpus, potcar_functional=potcar_functional)
        for preview in stage_previews
    ]
    modelled = sum(stage["seconds"] for stage in stages if stage["class"] in (STAGE_CALIBRATED, STAGE_EXTRAPOLATED))
    fallback = sum(stage["seconds"] for stage in stages if stage["class"] in (STAGE_HYBRID_FALLBACK, STAGE_UNKNOWN_FALLBACK))
    overhead = WORKFLOW_OVERHEAD_S + WORKFLOW_OVERHEAD_FRACTION * modelled
    total = modelled + fallback + overhead
    floor_applied = False
    if any(stage["class"] in (STAGE_EXTRAPOLATED, STAGE_UNKNOWN_FALLBACK) for stage in stages):
        if total < EXTRAPOLATED_WORKFLOW_FLOOR_S:
            total = float(EXTRAPOLATED_WORKFLOW_FLOOR_S)
            floor_applied = True
    seconds, capped = round_walltime_seconds(total)
    basis = (
        BASIS_EMPIRICAL
        if stages and all(stage["class"] == STAGE_CALIBRATED for stage in stages)
        else BASIS_FALLBACK
    )
    estimate = {
        "policy": {"id": POLICY_ID, "version": POLICY_VERSION},
        "basis": basis,
        "walltime": format_walltime(seconds),
        "walltime_seconds": seconds,
        "estimated_seconds": _rounded(total),
        "workflow_overhead_seconds": _rounded(overhead),
        "minimum_floor_applied": floor_applied,
        "capped_at_qos_limit": capped,
        "stages": stages,
        "calibration": dict(CALIBRATION),
        "assumptions": {
            "cpus": cpus,
            "safety_margin": SAFETY_MARGIN,
            "node_variability": "envelope covers every observed POWER node, including the slowest",
        },
    }
    estimate["explanation"] = walltime_explanation(estimate)
    return estimate


def stage_memory_bracket(stage: Mapping[str, Any]) -> tuple[int, str]:
    """The memory bracket (GB at 24 ranks) of one estimated stage and why."""

    descriptors = stage.get("descriptors")
    if not descriptors:
        return AUTOMATIC_MEMORY_GB, "descriptors unavailable"
    if descriptors.get("hybrid"):
        return AUTOMATIC_MEMORY_GB, "hybrid functional"
    if descriptors.get("meta_gga") or stage.get("theory") != Theory.PBE.value:
        return AUTOMATIC_MEMORY_GB, f"{stage.get('theory')} is not PBE"
    if (
        descriptors["atoms"] > CALIBRATED_MAX_ATOMS
        or descriptors["bands"] > CALIBRATED_MAX_BANDS
        or descriptors["kpoints"] > CALIBRATED_MAX_KPOINTS
        or descriptors["descriptor"] > CALIBRATED_MAX_DESCRIPTOR
    ):
        return AUTOMATIC_MEMORY_GB, "outside the calibrated range"
    if descriptors.get("soc"):
        return MEMORY_BRACKETS_GB[1], "spin-orbit coupling"
    if stage.get("stage_type") == StageType.PHONON_FORCES.value:
        return MEMORY_BRACKETS_GB[1], "phonon displacement supercells"
    if descriptors["atoms"] > MEMORY_SMALL_MAX_ATOMS:
        return MEMORY_BRACKETS_GB[1], f"more than {MEMORY_SMALL_MAX_ATOMS} atoms"
    return MEMORY_BRACKETS_GB[0], f"PBE, at most {MEMORY_SMALL_MAX_ATOMS} atoms"


def estimate_memory(stages: Sequence[Mapping[str, Any]], *, cpus: int = AUTOMATIC_CPUS) -> dict[str, Any]:
    """Automatic memory: the largest stage bracket, escalated for CPUs above 24."""

    brackets = [stage_memory_bracket(stage) for stage in stages] or [
        (AUTOMATIC_MEMORY_GB, "descriptors unavailable")
    ]
    bracket, reason = max(brackets, key=lambda item: item[0])
    deciding = next(
        (stage.get("index") for stage, item in zip(stages, brackets) if item[0] == bracket),
        None,
    )
    steps = math.ceil(math.log2(cpus / MEMORY_REFERENCE_CPUS)) if cpus > MEMORY_REFERENCE_CPUS else 0
    memory_gb = bracket
    for _ in range(steps):
        memory_gb = min(AUTOMATIC_MEMORY_GB, memory_gb * 2)
    basis = f"{bracket} GB bracket: {reason}"
    if memory_gb != bracket:
        basis += f"; raised to {memory_gb} GB for {cpus} CPUs"
    return {
        "memory_gb": memory_gb,
        "bracket_gb": bracket,
        "basis": basis,
        "deciding_stage": deciding,
        "stage_brackets": [item[0] for item in brackets],
    }


def round_walltime_seconds(seconds: float) -> tuple[int, bool]:
    seconds = max(float(seconds), float(MIN_WALLTIME_S))
    step = SHORT_ROUNDING_S if seconds < DAY else LONG_ROUNDING_S
    rounded = int(math.ceil(seconds / step) * step)
    if rounded > MAX_WALLTIME_SECONDS:
        return MAX_WALLTIME_SECONDS, True
    return rounded, False


def walltime_explanation(estimate: Mapping[str, Any]) -> str:
    stages = estimate.get("stages") or []
    counts: dict[str, int] = {}
    for stage in stages:
        counts[stage["class"]] = counts.get(stage["class"], 0) + 1
    if estimate.get("basis") == BASIS_EMPIRICAL:
        text = (
            f"Empirical estimate: {len(stages)} PBE stage(s) inside the range calibrated on "
            "BMD Compute runs, sized from k-points, bands and FFT grid with a 1.5x safety "
            "margin and allowance for the slowest compute nodes."
        )
    else:
        parts = []
        if counts.get(STAGE_CALIBRATED):
            parts.append(f"{counts[STAGE_CALIBRATED]} calibrated PBE stage(s)")
        if counts.get(STAGE_EXTRAPOLATED):
            parts.append(f"{counts[STAGE_EXTRAPOLATED]} stage(s) outside the calibrated range (3x model)")
        if counts.get(STAGE_HYBRID_FALLBACK):
            parts.append(f"{counts[STAGE_HYBRID_FALLBACK]} HSE06 stage(s) on fixed fallback walltimes")
        if counts.get(STAGE_UNKNOWN_FALLBACK):
            parts.append(f"{counts[STAGE_UNKNOWN_FALLBACK]} stage(s) that could not be sized (72 h each)")
        text = "Conservative fallback: " + "; ".join(parts) + "."
        if estimate.get("minimum_floor_applied"):
            text += " A 72 h minimum applies outside the calibrated range."
    if estimate.get("capped_at_qos_limit"):
        text += " Capped at the 100-day QOS limit; review this calculation before submitting."
    return text


def estimate_stage(
    preview: Mapping[str, Any],
    *,
    cpus: int,
    potcar_functional: str,
) -> dict[str, Any]:
    stage = preview["stage_spec"]
    stage_type = stage.stage_type
    record: dict[str, Any] = {
        "index": preview["index"],
        "stage_type": stage_type.value,
        "theory": stage.theory.value,
    }
    try:
        descriptors = stage_descriptors(preview, cpus=cpus, potcar_functional=potcar_functional)
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot be sized"
        record.update(
            {
                "class": STAGE_UNKNOWN_FALLBACK,
                "seconds": float(UNKNOWN_STAGE_FALLBACK_S),
                "reason": f"descriptors unavailable ({type(exc).__name__})",
            }
        )
        return record

    record["descriptors"] = descriptors
    if descriptors["hybrid"]:
        seconds, reason = hybrid_fallback_seconds(descriptors)
        record.update({"class": STAGE_HYBRID_FALLBACK, "seconds": float(seconds), "reason": reason})
        return record

    step_s = pbe_step_upper_seconds(descriptors["descriptor"])
    scf_steps, ionic_steps = step_counts(stage_type, descriptors)
    calculations = descriptors["calculations"]
    seconds = (
        SAFETY_MARGIN
        * (step_s * (scf_steps + STAGE_INITIALISATION_STEPS + IONIC_STEP_OVERHEAD_STEPS * ionic_steps) + STAGE_FIXED_OVERHEAD_S)
        * calculations
    )
    record.update(
        {
            "step_upper_seconds": _rounded(step_s),
            "scf_steps": scf_steps,
            "ionic_steps": ionic_steps,
        }
    )
    reasons = outside_calibration(stage, descriptors)
    if reasons:
        record.update(
            {
                "class": STAGE_EXTRAPOLATED,
                "seconds": _rounded(seconds * EXTRAPOLATION_FACTOR),
                "reason": "; ".join(reasons),
            }
        )
    else:
        record.update({"class": STAGE_CALIBRATED, "seconds": _rounded(seconds)})
    return record


def pbe_step_upper_seconds(descriptor: float) -> float:
    return PBE_STEP_ENVELOPE_COEFFICIENT * float(descriptor) ** PBE_STEP_ENVELOPE_EXPONENT


def step_counts(stage_type: StageType, descriptors: Mapping[str, Any]) -> tuple[int, int]:
    if stage_type is not StageType.RELAX:
        return SINGLE_POINT_SCF_STEPS, 1
    nsw = int(descriptors.get("nsw") or 0)
    if nsw <= 0:
        return SINGLE_POINT_SCF_STEPS, 1
    ionic = min(
        nsw,
        int(math.ceil(RELAX_IONIC_STEPS_PER_DEGREE_OF_FREEDOM * descriptors["degrees_of_freedom"])),
    )
    ionic = max(ionic, 1)
    return RELAX_FIRST_IONIC_SCF_STEPS + RELAX_LATER_IONIC_SCF_STEPS * (ionic - 1), ionic


def hybrid_fallback_seconds(descriptors: Mapping[str, Any]) -> tuple[int, str]:
    if descriptors["soc"]:
        return HYBRID_FALLBACK_MAX_S, "HSE06 with SOC: fixed 21-day fallback (poorly characterised)"
    descriptor = descriptors["descriptor"]
    for limit, seconds in HYBRID_FALLBACK_TIERS:
        if descriptor <= limit:
            return seconds, (
                f"HSE06: fixed {_duration_text(seconds)} fallback for cost descriptor up to {limit:.0e}"
            )
    return HYBRID_FALLBACK_MAX_S, "HSE06 beyond observed sizes: fixed 21-day fallback"


def outside_calibration(stage, descriptors: Mapping[str, Any]) -> list[str]:
    reasons = []
    if stage.theory is not Theory.PBE or descriptors.get("meta_gga"):
        reasons.append(f"{stage.theory.value} is not calibrated")
    if stage.stage_type not in CALIBRATED_STAGE_TYPES:
        reasons.append(f"{stage.stage_type.value} stages are not calibrated")
    if descriptors["descriptor"] > CALIBRATED_MAX_DESCRIPTOR:
        reasons.append("cost descriptor above calibrated range")
    if descriptors["bands"] > CALIBRATED_MAX_BANDS:
        reasons.append("more bands than calibrated")
    if descriptors["atoms"] > CALIBRATED_MAX_ATOMS:
        reasons.append("more atoms than calibrated")
    if descriptors["kpoints"] > CALIBRATED_MAX_KPOINTS:
        reasons.append("more k-points than calibrated")
    return reasons


# ------------------------------------------------------------------ descriptors


def stage_descriptors(
    preview: Mapping[str, Any],
    *,
    cpus: int,
    potcar_functional: str,
) -> dict[str, Any]:
    stage = preview["stage_spec"]
    input_set = preview["input_set"]
    incar = dict(input_set.incar)
    poscar = input_set.poscar
    structure = poscar.structure
    atoms = len(structure)

    soc = bool(incar.get("LSORBIT") or incar.get("LNONCOLLINEAR"))
    ispin = int(incar.get("ISPIN", 1) or 1)
    hybrid = bool(incar.get("LHFCALC"))
    meta_gga = bool(incar.get("METAGGA"))

    nelect = float(incar["NELECT"]) if incar.get("NELECT") else valence_electrons(
        poscar, potcar_symbol_lines(input_set), potcar_functional
    )
    magmom = incar.get("MAGMOM") or []
    if isinstance(magmom, (int, float)):
        magmom = [magmom]
    magnetisation = sum(abs(float(value)) for value in _flatten(magmom))

    ncore = int(incar.get("NCORE", 1) or 1)
    if incar.get("NBANDS"):
        bands = int(incar["NBANDS"])
    else:
        bands = default_bands(
            nelect,
            atoms,
            spin_polarized=ispin == 2 or soc,
            soc=soc,
            magnetisation=magnetisation,
        )
        if stage.stage_type in (StageType.DOS, StageType.BAND_STRUCTURE):
            bands = int(math.ceil(bands * NONSCF_NBANDS_FACTOR))
    bands = round_to_band_groups(bands, cpus=cpus, ncore=ncore)

    encut = float(incar["ENCUT"])
    grid = fft_grid(structure.lattice.abc, encut, prec=str(incar.get("PREC", "Accurate")))
    points = grid[0] * grid[1] * grid[2]

    isym = incar.get("ISYM")
    full_mesh = soc or (isym is not None and int(isym) <= 0)
    if ispin == 2 and not full_mesh and _magnetic_moments_break_symmetry(structure, magmom):
        full_mesh = True
    kpoints = kpoint_count(input_set.kpoints, structure, full_mesh=full_mesh)

    spin_factor = 2 if (soc or ispin == 2) else 1
    descriptor = kpoints * bands * points * math.log2(points) * spin_factor

    calculations = 1
    phonon_plan = preview.get("phonon_plan")
    if phonon_plan is not None:
        calculations = max(1, int(phonon_plan.get("displacement_count") or 1))

    isif = incar.get("ISIF")
    degrees = 3 * atoms + (0 if isif is not None and int(isif) == 2 else 6)
    nsw = incar.get("NSW")
    if stage.stage_type is StageType.RELAX and nsw is None:
        nsw = DEFAULT_NSW

    return {
        "atoms": atoms,
        "electrons": _rounded(nelect),
        "bands": bands,
        "kpoints": kpoints,
        "full_kpoint_mesh": bool(full_mesh),
        "fft_grid": list(grid),
        "encut_ev": encut,
        "spin_polarized": ispin == 2,
        "soc": soc,
        "hybrid": hybrid,
        "meta_gga": meta_gga,
        "dft_u": bool(incar.get("LDAU")),
        "dispersion": bool(incar.get("IVDW")),
        "descriptor": float(f"{descriptor:.6e}"),
        "calculations": calculations,
        "degrees_of_freedom": degrees,
        "nsw": int(nsw) if nsw is not None else 0,
    }


def default_bands(nelect: float, atoms: int, *, spin_polarized: bool, soc: bool, magnetisation: float) -> int:
    """VASP's default NBANDS (before rounding to band groups)."""

    bands = max(int(math.ceil((nelect + 2) / 2)) + max(atoms // 2, 3), int(0.6 * nelect))
    if spin_polarized:
        bands = max(bands, int(0.6 * nelect) + int(magnetisation / 2))
    if soc:
        bands *= 2
    return bands


def round_to_band_groups(bands: int, *, cpus: int, ncore: int) -> int:
    groups = max(1, int(cpus) // max(1, int(ncore)))
    return int(math.ceil(bands / groups) * groups)


def fft_grid(abc: Sequence[float], encut_ev: float, *, prec: str = "Accurate") -> tuple[int, int, int]:
    """Per-axis FFT grid, rounded up to an even 2-3-5-7-smooth size.

    PREC=Accurate needs at least 2 x 2 G_cut |a| / 2 pi points along each axis;
    lower precisions are treated as Accurate (conservative).
    """

    g_cut = math.sqrt(encut_ev / RYDBERG_EV) / BOHR_ANGSTROM
    factor = 4.0 if not str(prec).lower().startswith(("n", "l", "m")) else 3.0
    return tuple(_fft_size(factor * g_cut * length / (2 * math.pi)) for length in abc)  # type: ignore[return-value]


def _fft_size(minimum: float) -> int:
    size = max(2, int(math.ceil(minimum - 1e-9)))
    while True:
        if size % 2 == 0 and _smooth(size):
            return size
        size += 1


def _smooth(value: int) -> bool:
    for prime in (2, 3, 5, 7):
        while value % prime == 0:
            value //= prime
    return value == 1


def kpoint_count(kpoints, structure, *, full_mesh: bool) -> int:
    if kpoints is None:
        raise ValueError("KPOINTS not generated (KSPACING)")
    style = getattr(getattr(kpoints, "style", None), "name", str(getattr(kpoints, "style", ""))).lower()
    num = int(getattr(kpoints, "num_kpts", 0) or 0)
    if style.startswith("line"):
        segments = max(1, len(kpoints.kpts) // 2)
        return max(1, num) * segments
    if num > 0:
        return num
    if style.startswith(("gamma", "monkhorst")):
        mesh = [int(value) for value in kpoints.kpts[0]]
        total = mesh[0] * mesh[1] * mesh[2]
        if full_mesh:
            return total
        shift = [float(value) for value in (getattr(kpoints, "kpts_shift", None) or (0, 0, 0))]
        if style.startswith("monkhorst"):
            # Monkhorst-Pack meshes are half-shifted along even subdivisions.
            is_shift = [1 if n % 2 == 0 else 0 for n in mesh]
        else:
            is_shift = [1 if value else 0 for value in shift]
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            irreducible = SpacegroupAnalyzer(structure, symprec=SYMPREC).get_ir_reciprocal_mesh(
                mesh, is_shift=is_shift
            )
            return max(1, min(total, len(irreducible)))
        except Exception:  # noqa: BLE001 - fall back to the full (larger) mesh
            return total
    raise ValueError(f"unsupported KPOINTS style {style!r}")


def potcar_symbol_lines(input_set) -> list[str]:
    return [line.strip() for line in str(getattr(input_set, "potcar", "") or "").splitlines() if line.strip()]


def valence_electrons(poscar, potcar_symbols: Sequence[str], potcar_functional: str) -> float:
    counts = list(poscar.natoms)
    if len(potcar_symbols) != len(counts):
        raise ValueError("POTCAR symbols do not match POSCAR species")
    table = zval_table(potcar_functional)
    return float(sum(count * table[symbol] for count, symbol in zip(counts, potcar_symbols)))


@lru_cache(maxsize=8)
def zval_table(potcar_functional: str) -> Mapping[str, float]:
    """POTCAR valence by symbol from pymatgen's POTCAR summary statistics."""

    from importlib import resources as importlib_resources

    path = importlib_resources.files("pymatgen.io.vasp") / "potcar-summary-stats.json.bz2"
    with path.open("rb") as handle:
        data = json.loads(bz2.decompress(handle.read()))
    entries = data[potcar_functional]
    table: dict[str, float] = {}
    for variants in entries.values():
        for entry in variants:
            symbol = entry.get("symbol")
            if symbol and "ZVAL" in entry:
                table.setdefault(symbol, float(entry["ZVAL"]))
    return table


def _magnetic_moments_break_symmetry(structure, magmom) -> bool:
    values = list(_flatten(magmom))
    if len(values) != len(structure):
        return bool(values)
    by_species: dict[str, set] = {}
    for site, value in zip(structure, values):
        by_species.setdefault(site.species_string, set()).add(round(float(value), 6))
    return any(len(moments) > 1 for moments in by_species.values())


def _flatten(values):
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _flatten(value)
        else:
            yield value


def _rounded(value: float) -> float:
    return float(f"{float(value):.6g}")


def _duration_text(seconds: int) -> str:
    if seconds % DAY == 0 and seconds >= DAY * 2:
        return f"{seconds // DAY}-day"
    return f"{seconds // HOUR} h"


__all__ = [
    "AUTOMATIC_CPUS",
    "AUTOMATIC_MEMORY_GB",
    "estimate_memory",
    "BASIS_EMPIRICAL",
    "BASIS_FALLBACK",
    "POLICY_ID",
    "POLICY_VERSION",
    "ResourceRequest",
    "ResourceSelection",
    "SOURCE_AUTOMATIC",
    "SOURCE_MANUAL",
    "estimate_stage",
    "estimate_walltime",
    "planned_resources_token",
    "resolve_execution_resources",
    "selection_digest_material",
]
