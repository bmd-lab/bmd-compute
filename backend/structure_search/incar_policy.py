"""INCAR ladder for USPEX structure searches (proposal; needs BMD review, #65).

USPEX copies ``Specific/INCAR_1 .. INCAR_N`` unchanged into every candidate's
calculation, one file per relaxation step. One INCAR set therefore has to work
for any candidate cell, and nothing in it may depend on a particular
structure (no MAGMOM, no NBANDS, a fixed KPAR, ``ISPIN = 1``). The
composition is fixed for a whole search, so values that depend only on the
composition, such as the plane-wave cutoffs, can differ between searches and
are still identical for every candidate.

Lab direction (docs/structure_search.md section 2): start from looser
criteria than BMD standards and tighten step by step, so that the last steps
match BMD standards. The ladder below combines the USPEX 25.0 manual's
recommendations (section 3.1) with BMD's relaxation and static settings
(``docs/methodology.md`` section 3):

====  ==========================  ====  ======  =====  ===  ===================  =====  ======  =====================
step  purpose                     ISIF  IBRION  POTIM  NSW  ENCUT (eV)           EDIFF  EDIFFG  smearing
====  ==========================  ====  ======  =====  ===  ===================  =====  ======  =====================
1     shape, far from a minimum   4     2       0.02   99   early                1e-2   1e-1    MP, SIGMA 0.10
2     shape                       4     1       0.3    99   early                1e-3   1e-2    MP, SIGMA 0.10
3     volume                      3     1       0.3    99   BMD relaxation       1e-4   1e-3    MP, SIGMA 0.05
4     volume, BMD relax standard  3     1       0.3    99   BMD relaxation       1e-6   -0.01   final (Gauss 0.05)
5     single point (ranking)      2     -1      -      0    max(620, relax.)     1e-6   -       final (Gauss 0.05)
====  ==========================  ====  ======  =====  ===  ===================  =====  ======  =====================

Cutoffs, from the search's POTCARs (``search_cutoff_record``):

- "BMD relaxation" is BMD's species-dependent relaxation cutoff,
  ``bmd_compute.plane_wave_cutoff/1``:
  ``ENCUT = ceil(1.3 x max ENMAX / 50 eV) x 50 eV``. It is resolved by the
  same function and the same versioned ENMAX metadata as every BMD Geometry
  Optimisation (``backend.calculations.geometry_relaxation.resolve_cutoff_record``,
  ``potcar_enmax_PBE_64.json``), and the resolved record is part of the
  ladder's provenance record. Like BMD's relaxation protocol, steps 3 and 4
  set no ``ADDGRID``.
- Step 5 uses ``max(620 eV, BMD relaxation cutoff)``, the cutoff of a BMD
  Static Energy after a relaxation, and keeps that stage's ``ADDGRID``.
- "early" is ``ceil(EARLY_STEP_ENCUT_NEEDS_LAB_DECISION x max ENMAX / 50 eV)
  x 50 eV``. With the placeholder multiplier 1.0 it is never below any
  POTCAR's ENMAX.

Examples: Si gives 250 / 350 / 620 eV (early / relaxation / ranking); Li2O
(Li_sv, ENMAX 499.034 eV) gives 500 / 650 / 650 eV.

Reasons:

- The last steps must match BMD standards (lab decision), and BMD's
  relaxation standard is ``bmd_compute.geometry_relaxation/1`` with the
  species-dependent cutoff; the volume-changing steps follow it, which also
  avoids Pulay-stress volume errors. Only the fixed-volume steps use a lower
  cutoff.
- Every step's ENCUT must be at least the largest ENMAX of the search's
  POTCARs (:func:`require_encut_covers_enmax`). With the current rules this
  always holds; the check is a safety net and refuses a search, never
  adjusts it.
- Step 4 uses BMD's pass-1 relaxation convergence (EDIFF 1e-6, forces below
  0.01 eV/A), so the geometry the ranking energy is evaluated on meets BMD
  standards. This relaxes every candidate, including poor ones, to that
  standard and is probably the largest cost of a search; it is listed as a
  lab decision. (BMD's relaxation protocol adds a tighter pass 2, EDIFF
  1e-7 / -0.001 eV/A, which the search does not.)
- Step 5 sets the energy USPEX ranks candidates by (the enthalpy is taken
  from the last step). It uses the BMD Static Energy cutoff rule and EDIFF,
  so steps 4 to 5 are the same relax-then-static sequence as the Relaxed
  Energy workflow.
- Steps 4 and 5 share the final-step smearing so that the geometry and the
  ranking energy use the same occupations. (BMD's relaxation standard itself
  uses Gaussian smearing with SIGMA 0.2 eV.)
- Methfessel-Paxton smearing in steps 1 to 3 follows the manual: many
  sampled structures are metallic.
- No PSTRESS: pressure is set only by ``ExternalPressure`` in INPUT.txt
  (manual section 4.1).
- ``LWAVE`` and ``LCHARG`` are off because a search runs thousands of VASP
  calculations.

Values still awaiting a lab decision are module constants whose names say so
(``*_NEEDS_LAB_DECISION`` in :data:`VALUES_NEEDING_LAB_DECISION`). Phase 0
(#59) should calibrate them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.calculations.geometry_relaxation import (
    CUTOFF_INCREMENT_EV,
    CUTOFF_RULE_ID,
    CUTOFF_RULE_VERSION,
    resolve_cutoff_record,
)
from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import DEFAULT_NCORE
from backend.calculations.vasp_stage_definitions import ENCUT_STATIC_FINAL_DEFAULT
from backend.structure_search.potcar import POTCAR_FUNCTIONAL, bmd_potcar_symbols


INCAR_POLICY_ID = "bmd_compute.structure_search_incar"
INCAR_POLICY_VERSION = 1


# ---------------------------------------------------------------------------
# Values that need a lab decision. Each is a placeholder chosen as described;
# change it here and update the golden INCAR files and docs/methodology.md.
# ---------------------------------------------------------------------------

# Multiplier of the early-step (1 and 2) cutoff rule:
# ENCUT = ceil(multiplier x max ENMAX / 50 eV) x 50 eV. Placeholder 1.0, so
# the early cutoff is the lowest 50 eV step at or above every POTCAR's ENMAX
# (Si: 250 eV). A value below 1 would put steps 1-2 below ENMAX, which
# require_encut_covers_enmax refuses. Replaces the earlier fixed 430 eV,
# which refused every search containing Li (Li_sv, ENMAX 499.034 eV) or He
# (478.896 eV).
EARLY_STEP_ENCUT_NEEDS_LAB_DECISION = 1.0

# PREC of the loose steps 1 and 2. The manual's ladder starts with Low, then
# Normal; Accurate also sets denser FFT grids, so these steps cost more than
# they need to. Kept at BMD's Accurate until the lab decides.
EARLY_STEP_PREC_NEEDS_LAB_DECISION = "Accurate"

# Convergence of step 4. BMD's relaxation standard relaxes every candidate,
# including poor ones that will never be shortlisted (the shortlist is
# re-relaxed by Relaxed Energy anyway), and is probably the main cost of a
# search. The alternative is the manual's final tolerances (EDIFF 1e-4,
# EDIFFG 1e-3) or a value in between; Phase 0 can measure the cost.
STEP_4_CONVERGENCE_NEEDS_LAB_DECISION: Mapping[str, Any] = {"EDIFF": 1e-6, "EDIFFG": -0.01}

# Smearing of steps 4 and 5 (the final geometry and the ranking energy).
# Options under discussion: Methfessel-Paxton (ISMEAR 1) with a small SIGMA,
# or Gaussian (ISMEAR 0) with SIGMA 0.05. Placeholder: Gaussian, SIGMA 0.05.
# BMD's tetrahedron method (ISMEAR -5, used by PBE Static Energy) is not an
# option: candidates with coarse meshes can have fewer k-points than it needs.
FINAL_SMEARING_NEEDS_LAB_DECISION: Mapping[str, Any] = {"ISMEAR": 0, "SIGMA": 0.05}

# KPAR for every step. Because the INCAR is the same for every candidate, KPAR
# must work for any k-point count (rules in validate_parallel_tags). KPAR = 1
# meets every rule for any rank count and any number of k-points.
# KPAR = 4 also meets the two divisibility rules with 48 ranks. The lab's
# notes measured KPAR as worth up to 9x on a k-point-dense cell. But
# KPAR = 4 exceeds NKPTS for candidates with fewer than 4 irreducible
# k-points, which a search does produce (large cells, coarse early
# KresolStart). So the placeholder stays at 1 until the Phase 0 pilot (#59)
# shows the speed-up on real search candidates, and how VASP behaves when
# KPAR > NKPTS.
FIXED_KPAR_NEEDS_LAB_DECISION = 1



# ---------------------------------------------------------------------------
# Fixed parts of the policy.
# ---------------------------------------------------------------------------

# MPI ranks of one relaxation (docs/structure_search.md section 4: 4 x 48 on
# a 192-core node; to be measured in Phase 0). The INCARs do not contain the
# rank count, but KPAR and NCORE are checked against it.
DEFAULT_RANKS_PER_RELAXATION = 48

# NCORE = 4 is BMD's fixed value (backend/calculations/resources.py). In the
# lab's August 2026 leeburton-pool VASP notes it was the best measured value,
# with moderate confidence. Those notes come from a single small cell
# (3-atom Sc2C, 65 irreducible k-points), not from search candidates, so
# they are not general results. NPAR is never set together with NCORE.

# Ionic step limit of every relaxation step: BMD's relaxation value
# (docs/methodology.md section 3, adopted from atomate2).
RELAX_STEP_NSW = 99

# Settings shared by every step: BMD's common stage settings
# (docs/methodology.md section 3) plus what a search must fix for all
# candidates. Steps 3 and 4 remove ADDGRID, as BMD's relaxation protocol
# does; steps 1-2 and 5 keep it (step 5 as BMD's Static Energy).
COMMON_SETTINGS: Mapping[str, Any] = {
    "PREC": "Accurate",
    "LREAL": False,
    "ADDGRID": True,
    "LASPH": True,
    "NELM": 200,
    "ISPIN": 1,
    "NCORE": DEFAULT_NCORE,
    "KPAR": FIXED_KPAR_NEEDS_LAB_DECISION,
    "LWAVE": False,
    "LCHARG": False,
}

# Keys that must never appear: pressure belongs in INPUT.txt; MAGMOM and
# NBANDS depend on the structure and cannot be shared by every candidate; and
# DFT+U, SOC and dispersion corrections are not part of v1 searches.
# NPAR is forbidden because NCORE already sets the band parallelisation and
# the two must not be combined.
FORBIDDEN_KEYS = frozenset({"PSTRESS", "MAGMOM", "NBANDS", "LDAU", "LSORBIT", "IVDW", "NPAR"})

# BMD k-point policy: Gamma-centred meshes of reciprocal density 64 points per
# reciprocal volume (pymatgen automatic_density_by_vol, which includes the 2 pi
# factor). The equivalent spacing is 64**(-1/3) = 0.25 1/A, which in USPEX's
# KresolStart units (2 pi / A) is 0.0398. Rounded meshes from pymatgen give
# 0.040-0.042 in practice; the limit below is that value rounded to 0.04.
# Steps 4 and 5 (BMD-standard steps) must be at least this dense. How USPEX
# rounds a KresolStart value to a mesh is not documented here; Phase 0 should
# compare the KPOINTS USPEX writes with pymatgen's mesh for the same cell.
BMD_RECIPROCAL_DENSITY = 64
BMD_EQUIVALENT_KRESOL = BMD_RECIPROCAL_DENSITY ** (-1.0 / 3.0) / (2.0 * math.pi)
BMD_STANDARD_MAX_KRESOL = 0.04

# Steps (1-based) whose KresolStart must meet BMD_STANDARD_MAX_KRESOL.
BMD_STANDARD_STEPS = (4, 5)

ENCUT_BELOW_ENMAX_CODE = "encut_below_potcar_enmax"
POTCAR_ENMAX_MISMATCH_CODE = "potcar_enmax_differs_from_cutoff_record"
INVALID_CUTOFF_RECORD_CODE = "invalid_plane_wave_cutoff_record"

_STEP_PURPOSES = (
    "fixed-volume relaxation far from a minimum",
    "fixed-volume relaxation",
    "full relaxation including volume",
    "full relaxation to BMD relaxation standards",
    "single point: the energy USPEX ranks candidates by",
)
NUM_STEPS = len(_STEP_PURPOSES)


@dataclass(frozen=True)
class LadderStep:
    """One relaxation step of the ladder: its purpose and its INCAR settings."""

    number: int
    purpose: str
    settings: Mapping[str, Any]


def _mp(sigma: float) -> dict[str, Any]:
    return {"ISMEAR": 1, "SIGMA": sigma}


def validate_parallel_tags(kpar: int, ncore: int, ranks: int) -> None:
    """Check KPAR and NCORE for a relaxation of ``ranks`` MPI ranks.

    Rules (the INCAR is identical for every candidate, so they must hold
    whatever the k-point count):

    - ``ranks`` is divisible by ``KPAR``;
    - ``ranks / KPAR`` is divisible by ``NCORE``.

    The third rule, ``KPAR`` <= NKPTS, depends on each candidate's mesh and
    cannot be checked here. It is why ``FIXED_KPAR_NEEDS_LAB_DECISION`` is
    chosen conservatively. Raises ``ValueError``.
    """

    for label, value in (("KPAR", kpar), ("NCORE", ncore), ("ranks", ranks)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a whole number of at least 1, not {value!r}")
    if ranks % kpar:
        raise ValueError(f"KPAR = {kpar} does not divide the {ranks} MPI ranks of a relaxation")
    if (ranks // kpar) % ncore:
        raise ValueError(
            f"NCORE = {ncore} does not divide the {ranks // kpar} ranks per k-point group "
            f"({ranks} ranks / KPAR {kpar})"
        )


def search_cutoff_record(elements: Iterable[str], *, functional: str = POTCAR_FUNCTIONAL) -> dict[str, Any]:
    """BMD's plane-wave cutoff record for a search of ``elements``.

    The POTCAR symbols are BMD's choice for each element
    (``potcar.bmd_potcar_symbols``), and the record is resolved by
    ``geometry_relaxation.resolve_cutoff_record``, exactly as for a BMD
    Geometry Optimisation: rule ``bmd_compute.plane_wave_cutoff/1``, ENMAX
    from the versioned metadata. Raises that module's validation error if the
    metadata has no ENMAX for a symbol.
    """

    symbols = list(bmd_potcar_symbols(elements).values())
    return resolve_cutoff_record(symbols, functional=functional)


def _require_cutoff_record(record: Any) -> Mapping[str, Any]:
    rule = record.get("rule") if isinstance(record, Mapping) else None
    valid = (
        isinstance(rule, Mapping)
        and rule.get("id") == CUTOFF_RULE_ID
        and rule.get("version") == CUTOFF_RULE_VERSION
        and isinstance(record.get("ENCUT"), int)
        and not isinstance(record.get("ENCUT"), bool)
        and isinstance(record.get("max_enmax_ev"), (int, float))
        and math.isfinite(float(record["max_enmax_ev"]))
        and float(record["max_enmax_ev"]) > 0
    )
    if not valid:
        raise IncarPolicyError(
            "The structure-search INCARs need a resolved BMD plane-wave cutoff record "
            f"({CUTOFF_RULE_ID}/{CUTOFF_RULE_VERSION}); use search_cutoff_record().",
            code=INVALID_CUTOFF_RECORD_CODE,
        )
    return record


def early_step_encut(max_enmax_ev: float) -> int:
    """``ceil(EARLY_STEP_ENCUT_NEEDS_LAB_DECISION x max ENMAX / 50 eV) x 50 eV``."""

    scaled = Decimal(repr(float(max_enmax_ev))) * Decimal(repr(EARLY_STEP_ENCUT_NEEDS_LAB_DECISION))
    steps = (scaled / Decimal(CUTOFF_INCREMENT_EV)).to_integral_value(rounding=ROUND_CEILING)
    return int(steps) * CUTOFF_INCREMENT_EV


def ladder_cutoffs(cutoff_record: Mapping[str, Any]) -> dict[str, int]:
    """The three cutoffs of the ladder for one search (eV).

    ``early`` (steps 1-2), ``relaxation`` (steps 3-4, BMD's resolved
    relaxation cutoff) and ``ranking`` (step 5, ``max(620, relaxation)``).
    """

    record = _require_cutoff_record(cutoff_record)
    relaxation = int(record["ENCUT"])
    return {
        "early": early_step_encut(record["max_enmax_ev"]),
        "relaxation": relaxation,
        "ranking": max(ENCUT_STATIC_FINAL_DEFAULT, relaxation),
    }


def incar_ladder(
    cutoff_record: Mapping[str, Any],
    ranks: int = DEFAULT_RANKS_PER_RELAXATION,
) -> tuple[LadderStep, ...]:
    """The five-step INCAR ladder for one search (see the module docstring).

    ``cutoff_record`` is the search's resolved cutoff record
    (:func:`search_cutoff_record`). ``ranks`` is the MPI rank count of one
    relaxation; it does not change the INCARs and only checks that KPAR and
    NCORE are valid for it.
    """

    validate_parallel_tags(COMMON_SETTINGS["KPAR"], COMMON_SETTINGS["NCORE"], ranks)
    cutoffs = ladder_cutoffs(cutoff_record)

    final_smearing = dict(FINAL_SMEARING_NEEDS_LAB_DECISION)
    step_4_convergence = dict(STEP_4_CONVERGENCE_NEEDS_LAB_DECISION)
    relax = {"NSW": RELAX_STEP_NSW, "ALGO": "Fast"}
    rows = (
        {"ISIF": 4, "IBRION": 2, "POTIM": 0.02, **relax, "ENCUT": cutoffs["early"],
         "EDIFF": 1e-2, "EDIFFG": 1e-1, **_mp(0.10)},
        {"ISIF": 4, "IBRION": 1, "POTIM": 0.3, **relax, "ENCUT": cutoffs["early"],
         "EDIFF": 1e-3, "EDIFFG": 1e-2, **_mp(0.10)},
        {"ISIF": 3, "IBRION": 1, "POTIM": 0.3, **relax, "ENCUT": cutoffs["relaxation"],
         "EDIFF": 1e-4, "EDIFFG": 1e-3, **_mp(0.05)},
        {"ISIF": 3, "IBRION": 1, "POTIM": 0.3, **relax, "ENCUT": cutoffs["relaxation"],
         **step_4_convergence, **final_smearing},
        {"ISIF": 2, "IBRION": -1, "NSW": 0, "ALGO": "Normal",
         "ENCUT": cutoffs["ranking"], "EDIFF": 1e-6, **final_smearing},
    )
    # Per-step overrides of COMMON_SETTINGS. A dict update keeps a key in
    # COMMON_SETTINGS' position, so the INCAR line order is stable; ``None``
    # removes the key.
    overrides = {
        1: {"PREC": EARLY_STEP_PREC_NEEDS_LAB_DECISION},
        2: {"PREC": EARLY_STEP_PREC_NEEDS_LAB_DECISION},
        3: {"ADDGRID": None},
        4: {"ADDGRID": None},
    }
    steps = []
    for index, (purpose, settings) in enumerate(zip(_STEP_PURPOSES, rows), start=1):
        merged = {**settings, **COMMON_SETTINGS, **overrides.get(index, {})}
        steps.append(LadderStep(
            number=index,
            purpose=purpose,
            settings={key: value for key, value in merged.items() if value is not None},
        ))
    return tuple(steps)



# Every value a lab decision is still needed for. NUM_STEPS is listed because
# the step count is a choice too: the manual suggests 4-5 steps, and the
# spec's KresolStart must have exactly NUM_STEPS values
# (require_spec_matches_ladder). Calling that check before INPUT.txt is
# written belongs to the search entry point (#67).
VALUES_NEEDING_LAB_DECISION = {
    "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION": EARLY_STEP_ENCUT_NEEDS_LAB_DECISION,
    "EARLY_STEP_PREC_NEEDS_LAB_DECISION": EARLY_STEP_PREC_NEEDS_LAB_DECISION,
    "STEP_4_CONVERGENCE_NEEDS_LAB_DECISION": dict(STEP_4_CONVERGENCE_NEEDS_LAB_DECISION),
    "FINAL_SMEARING_NEEDS_LAB_DECISION": dict(FINAL_SMEARING_NEEDS_LAB_DECISION),
    "FIXED_KPAR_NEEDS_LAB_DECISION": FIXED_KPAR_NEEDS_LAB_DECISION,
    "NUM_STEPS": NUM_STEPS,
}


class IncarPolicyError(CalculationValidationError):
    """A structure search does not fit the INCAR ladder."""

    def __init__(self, message: str, *, code: str, suggestion: str | None = None, **details):
        super().__init__(
            message,
            suggestion=suggestion,
            diagnostic={
                "code": code,
                "policy_id": INCAR_POLICY_ID,
                "policy_version": INCAR_POLICY_VERSION,
                **details,
            },
        )
        self.code = code


def require_spec_matches_ladder(spec) -> None:
    """Check a search's KresolStart values against the ladder and BMD k-point policy.

    The spec must have one KresolStart value per ladder step, and the
    BMD-standard steps must be at least as dense as BMD's k-point policy.
    """

    kresol = tuple(spec.kresol_start)
    if len(kresol) != NUM_STEPS:
        raise IncarPolicyError(
            f"The search has {len(kresol)} relaxation step(s) (KresolStart values) "
            f"but the BMD INCAR ladder has {NUM_STEPS}.",
            code="step_count_mismatch",
            suggestion=f"Give exactly {NUM_STEPS} KresolStart values, one per step.",
            steps=len(kresol),
        )
    too_coarse = [
        step for step in BMD_STANDARD_STEPS
        if kresol[step - 1] > BMD_STANDARD_MAX_KRESOL + 1e-12
    ]
    if too_coarse:
        raise IncarPolicyError(
            f"KresolStart for step(s) {', '.join(map(str, too_coarse))} is coarser than "
            f"BMD's k-point policy allows ({BMD_STANDARD_MAX_KRESOL} x 2 pi / A).",
            code="kresol_coarser_than_bmd_policy",
            suggestion=f"Use KresolStart of at most {BMD_STANDARD_MAX_KRESOL} for steps "
                       f"{', '.join(map(str, BMD_STANDARD_STEPS))}.",
            steps=too_coarse,
        )


def require_encut_covers_enmax(
    largest_enmax_ev: float,
    cutoff_record: Mapping[str, Any],
    ranks: int = DEFAULT_RANKS_PER_RELAXATION,
) -> None:
    """Refuse a search whose POTCARs need a higher cutoff than some step uses.

    ``largest_enmax_ev`` is the largest ENMAX (eV) of the search's POTCARs,
    for example ``potcar.max_enmax(write_specific_potcars(...))``. Every
    step's ENCUT must be at least that value. With the current cutoff rules
    this always holds; the check is a safety net (for example against a
    multiplier below 1). A failing search is refused with code
    ``ENCUT_BELOW_ENMAX_CODE``; the cutoff is never raised here.
    """

    enmax = float(largest_enmax_ev)
    if not math.isfinite(enmax) or enmax <= 0:
        raise ValueError(f"largest_enmax_ev must be a positive number, not {largest_enmax_ev!r}")
    below = [
        {"step": step.number, "encut": step.settings["ENCUT"]}
        for step in incar_ladder(cutoff_record, ranks)
        if float(step.settings["ENCUT"]) < enmax
    ]
    if below:
        steps = ", ".join(str(row["step"]) for row in below)
        cutoffs = ", ".join(f"{row['encut']} eV" for row in below)
        raise IncarPolicyError(
            f"The plane-wave cutoff of step(s) {steps} ({cutoffs}) is below the largest "
            f"ENMAX of this search's POTCARs ({enmax:g} eV).",
            code=ENCUT_BELOW_ENMAX_CODE,
            largest_enmax_ev=enmax,
            steps=[row["step"] for row in below],
        )


def require_potcars_match_cutoff_record(
    potcar_records: Iterable[Mapping[str, Any]],
    cutoff_record: Mapping[str, Any],
) -> None:
    """Check the copied POTCARs against the ENMAX values the cutoffs were resolved from.

    ``potcar_records`` come from ``potcar.write_specific_potcars`` (ENMAX read
    from the files VASP will use); ``cutoff_record`` from
    :func:`search_cutoff_record` (ENMAX from the versioned metadata). They
    must name the same POTCAR symbols with the same ENMAX, otherwise the
    cutoffs were computed for different POTCARs and the search is refused.
    """

    record = _require_cutoff_record(cutoff_record)
    expected = {str(row["symbol"]): float(row["ENMAX"]) for row in record.get("potcars", [])}
    found = {str(row["potcar_symbol"]): float(row["enmax"]) for row in potcar_records}
    differing = sorted(
        symbol for symbol in set(expected) | set(found)
        if symbol not in expected or symbol not in found
        or abs(expected[symbol] - found[symbol]) > 1e-6
    )
    if differing:
        raise IncarPolicyError(
            "The POTCAR files do not match the ENMAX metadata the plane-wave cutoffs "
            f"were resolved from ({', '.join(differing)}).",
            code=POTCAR_ENMAX_MISMATCH_CODE,
            suggestion="Regenerate the POTCAR ENMAX metadata from the POWER library.",
            symbols=differing,
            expected={symbol: expected.get(symbol) for symbol in differing},
            found={symbol: found.get(symbol) for symbol in differing},
        )


def render_incar(settings: Mapping[str, Any]) -> str:
    """INCAR text with one ``KEY = value`` line per setting, in the given order."""

    forbidden = sorted(FORBIDDEN_KEYS & set(settings))
    if forbidden:
        raise ValueError(f"Structure-search INCARs must not set {', '.join(forbidden)}")
    return "".join(f"{key} = {_incar_value(value)}\n" for key, value in settings.items())


def incar_texts(
    cutoff_record: Mapping[str, Any],
    ranks: int = DEFAULT_RANKS_PER_RELAXATION,
) -> dict[str, str]:
    """``{"INCAR_1": text, ...}`` for every ladder step of one search."""

    return {
        f"INCAR_{step.number}": render_incar(step.settings)
        for step in incar_ladder(cutoff_record, ranks)
    }


def write_incar_files(
    specific_dir: str | Path,
    cutoff_record: Mapping[str, Any],
    ranks: int = DEFAULT_RANKS_PER_RELAXATION,
) -> list[Path]:
    """Write ``INCAR_1 .. INCAR_N`` for one search into ``specific_dir`` (which must exist)."""

    paths = []
    for name, text in incar_texts(cutoff_record, ranks).items():
        path = Path(specific_dir) / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def incar_policy_record(
    cutoff_record: Mapping[str, Any],
    ranks: int = DEFAULT_RANKS_PER_RELAXATION,
) -> dict[str, Any]:
    """JSON-safe record of one search's ladder, for provenance.

    ``plane_wave_cutoff`` is the resolved cutoff record unchanged, in the same
    form BMD freezes into a Geometry Optimisation stage.
    """

    return {
        "policy_id": INCAR_POLICY_ID,
        "policy_version": INCAR_POLICY_VERSION,
        "ranks_per_relaxation": ranks,
        "plane_wave_cutoff": dict(_require_cutoff_record(cutoff_record)),
        "cutoffs": ladder_cutoffs(cutoff_record),
        "steps": [
            {"step": step.number, "purpose": step.purpose, "incar": dict(step.settings)}
            for step in incar_ladder(cutoff_record, ranks)
        ],
        "bmd_standard_steps": list(BMD_STANDARD_STEPS),
        "bmd_standard_max_kresol": BMD_STANDARD_MAX_KRESOL,
        "values_needing_lab_decision": dict(VALUES_NEEDING_LAB_DECISION),
    }


def _incar_value(value) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, float):
        return repr(value)
    return str(value)


__all__ = [
    "BMD_EQUIVALENT_KRESOL",
    "BMD_STANDARD_MAX_KRESOL",
    "BMD_STANDARD_STEPS",
    "COMMON_SETTINGS",
    "DEFAULT_RANKS_PER_RELAXATION",
    "EARLY_STEP_PREC_NEEDS_LAB_DECISION",
    "ENCUT_BELOW_ENMAX_CODE",
    "INVALID_CUTOFF_RECORD_CODE",
    "POTCAR_ENMAX_MISMATCH_CODE",
    "STEP_4_CONVERGENCE_NEEDS_LAB_DECISION",
    "early_step_encut",
    "ladder_cutoffs",
    "require_encut_covers_enmax",
    "require_potcars_match_cutoff_record",
    "search_cutoff_record",
    "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION",
    "FINAL_SMEARING_NEEDS_LAB_DECISION",
    "FIXED_KPAR_NEEDS_LAB_DECISION",
    "FORBIDDEN_KEYS",
    "INCAR_POLICY_ID",
    "INCAR_POLICY_VERSION",
    "IncarPolicyError",
    "LadderStep",
    "NUM_STEPS",
    "RELAX_STEP_NSW",
    "VALUES_NEEDING_LAB_DECISION",
    "incar_ladder",
    "incar_policy_record",
    "incar_texts",
    "render_incar",
    "require_spec_matches_ladder",
    "validate_parallel_tags",
    "write_incar_files",
]
