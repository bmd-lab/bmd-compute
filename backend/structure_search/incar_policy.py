"""INCAR ladder for USPEX structure searches (proposal; needs BMD review, #65).

USPEX copies ``Specific/INCAR_1 .. INCAR_N`` unchanged into every candidate's
calculation, one file per relaxation step. One INCAR set therefore has to work
for any candidate cell, and nothing in it may depend on a particular
structure (no MAGMOM, no NBANDS, a fixed KPAR, ``ISPIN = 1``).

Lab direction (docs/structure_search.md section 2): start from looser
criteria than BMD standards and tighten step by step, so that the last steps
match BMD standards. The ladder below combines the USPEX 25.0 manual's
recommendations (section 3.1) with BMD's relaxation and static settings
(``docs/methodology.md`` section 3):

====  ==========================  ====  ======  =====  ===  ==============  =====  ======  =====================
step  purpose                     ISIF  IBRION  POTIM  NSW  ENCUT (eV)      EDIFF  EDIFFG  smearing
====  ==========================  ====  ======  =====  ===  ==============  =====  ======  =====================
1     shape, far from a minimum   4     2       0.02   99   early (430)     1e-2   1e-1    MP, SIGMA 0.10
2     shape                       4     1       0.3    99   early (430)     1e-3   1e-2    MP, SIGMA 0.10
3     volume                      3     1       0.3    99   volume (580)    1e-4   1e-3    MP, SIGMA 0.05
4     volume, BMD relax standard  3     1       0.3    99   volume (580)    1e-6   -0.01   final (Gauss 0.05)
5     single point (ranking)      2     -1      -      0    620             1e-6   -       final (Gauss 0.05)
====  ==========================  ====  ======  =====  ===  ==============  =====  ======  =====================

Reasons:

- Steps that change the volume (ISIF 3) use BMD's fixed relaxation cutoff
  of 580 eV (``ENCUT_RELAX_DEFAULT``), to avoid Pulay-stress volume errors;
  only the fixed-volume steps (ISIF 4) use a lower cutoff. Since this ladder
  was written, BMD's own relaxations moved to a species-dependent cutoff
  (``bmd_compute.plane_wave_cutoff/1``: ceil(1.3 x max ENMAX / 50 eV) x 50 eV,
  no ADDGRID; ``docs/methodology.md`` section 3 on ``main``). 580 eV is now
  the value for records made before that protocol. Whether the search ladder
  should follow the new rule is listed as a lab decision below.
- Every step's ENCUT must be at least the largest ENMAX of the search's
  POTCARs (:func:`require_encut_covers_enmax`); a search that fails this is
  refused, not adjusted.
- Step 4 uses BMD's relaxation convergence (EDIFF 1e-6, forces below
  0.01 eV/A; BMD's pass-1 values), so the geometry the ranking energy is
  evaluated on meets BMD standards. This relaxes every candidate, including
  poor ones, to that standard and is probably the largest cost of a search;
  it is listed as a lab decision.
- Step 5 sets the energy USPEX ranks candidates by (the enthalpy is taken
  from the last step). It uses the BMD Static Energy cutoff and EDIFF, so
  steps 4 to 5 are the same relax-then-static sequence as the Relaxed
  Energy workflow.
- Steps 4 and 5 share the final-step smearing so that the geometry and the
  ranking energy use the same occupations.
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
from pathlib import Path
from typing import Any, Mapping

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import DEFAULT_NCORE
from backend.calculations.vasp_stage_definitions import (
    ENCUT_RELAX_DEFAULT,
    ENCUT_STATIC_FINAL_DEFAULT,
)


INCAR_POLICY_ID = "bmd_compute.structure_search_incar"
INCAR_POLICY_VERSION = 1


# ---------------------------------------------------------------------------
# Values that need a lab decision. Each is a placeholder chosen as described;
# change it here and update the golden INCAR files and docs/methodology.md.
# ---------------------------------------------------------------------------

# Plane-wave cutoff of the fixed-volume steps (1 and 2). The manual raises the
# cutoff by 30-40 % for the ISIF 3 steps; 580 / 1.35 = 430 eV is the midpoint
# of that range, rounded to 10 eV. It can be below a POTCAR's ENMAX (for
# example Li_sv, ENMAX 499.034 eV in BMD's PBE_64 metadata on main), so
# require_encut_covers_enmax refuses such searches.
EARLY_STEP_ENCUT_NEEDS_LAB_DECISION = 430

# What to do when a step's ENCUT is below the largest ENMAX of a search's
# POTCARs. Implemented: refuse the search (IncarPolicyError,
# ENCUT_BELOW_ENMAX_CODE). The alternative, raising the early cutoff
# automatically for that search (the composition, and so the POTCARs, are
# fixed per search, so every candidate would still get the same INCAR), is a
# lab decision and is not implemented.
ENCUT_BELOW_ENMAX_ACTION_NEEDS_LAB_DECISION = "refuse"

# Plane-wave cutoff of the volume-changing steps 3 and 4: BMD's fixed
# relaxation value 580 eV. BMD's relaxations now use a species-dependent
# cutoff (see the module docstring); whether the ladder should do the same,
# per search, is a lab decision. The value is unchanged until then.
VOLUME_STEP_ENCUT_NEEDS_LAB_DECISION = ENCUT_RELAX_DEFAULT

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
# candidates.
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


def incar_ladder(ranks: int = DEFAULT_RANKS_PER_RELAXATION) -> tuple[LadderStep, ...]:
    """The five-step INCAR ladder (see the module docstring for the reasons).

    ``ranks`` is the MPI rank count of one relaxation. It does not change the
    INCARs; it only checks that KPAR and NCORE are valid for it.
    """

    validate_parallel_tags(COMMON_SETTINGS["KPAR"], COMMON_SETTINGS["NCORE"], ranks)

    early = EARLY_STEP_ENCUT_NEEDS_LAB_DECISION
    volume = VOLUME_STEP_ENCUT_NEEDS_LAB_DECISION
    final_smearing = dict(FINAL_SMEARING_NEEDS_LAB_DECISION)
    step_4_convergence = dict(STEP_4_CONVERGENCE_NEEDS_LAB_DECISION)
    relax = {"NSW": RELAX_STEP_NSW, "ALGO": "Fast"}
    rows = (
        ("fixed-volume relaxation far from a minimum",
         {"ISIF": 4, "IBRION": 2, "POTIM": 0.02, **relax, "ENCUT": early,
          "EDIFF": 1e-2, "EDIFFG": 1e-1, **_mp(0.10)}),
        ("fixed-volume relaxation",
         {"ISIF": 4, "IBRION": 1, "POTIM": 0.3, **relax, "ENCUT": early,
          "EDIFF": 1e-3, "EDIFFG": 1e-2, **_mp(0.10)}),
        ("full relaxation including volume",
         {"ISIF": 3, "IBRION": 1, "POTIM": 0.3, **relax, "ENCUT": volume,
          "EDIFF": 1e-4, "EDIFFG": 1e-3, **_mp(0.05)}),
        ("full relaxation to BMD relaxation standards",
         {"ISIF": 3, "IBRION": 1, "POTIM": 0.3, **relax, "ENCUT": volume,
          **step_4_convergence, **final_smearing}),
        ("single point: the energy USPEX ranks candidates by",
         {"ISIF": 2, "IBRION": -1, "NSW": 0, "ALGO": "Normal",
          "ENCUT": ENCUT_STATIC_FINAL_DEFAULT, "EDIFF": 1e-6, **final_smearing}),
    )
    # Steps 1-2 override PREC; a dict update keeps the key in COMMON_SETTINGS'
    # position, so the INCAR line order is the same for every step.
    early_overrides = {"PREC": EARLY_STEP_PREC_NEEDS_LAB_DECISION}
    return tuple(
        LadderStep(
            number=index,
            purpose=purpose,
            settings={**settings, **COMMON_SETTINGS, **(early_overrides if index <= 2 else {})},
        )
        for index, (purpose, settings) in enumerate(rows, start=1)
    )


NUM_STEPS = len(incar_ladder())

# Every value a lab decision is still needed for. NUM_STEPS is listed because
# the step count is a choice too: the manual suggests 4-5 steps, and the
# spec's KresolStart must have exactly NUM_STEPS values
# (require_spec_matches_ladder). Calling that check before INPUT.txt is
# written belongs to the search entry point (#67).
VALUES_NEEDING_LAB_DECISION = {
    "EARLY_STEP_ENCUT_NEEDS_LAB_DECISION": EARLY_STEP_ENCUT_NEEDS_LAB_DECISION,
    "ENCUT_BELOW_ENMAX_ACTION_NEEDS_LAB_DECISION": ENCUT_BELOW_ENMAX_ACTION_NEEDS_LAB_DECISION,
    "VOLUME_STEP_ENCUT_NEEDS_LAB_DECISION": VOLUME_STEP_ENCUT_NEEDS_LAB_DECISION,
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
    ranks: int = DEFAULT_RANKS_PER_RELAXATION,
) -> None:
    """Refuse a search whose POTCARs need a higher cutoff than some step uses.

    ``largest_enmax_ev`` is the largest ENMAX (eV) of the search's POTCARs,
    for example ``potcar.max_enmax(write_specific_potcars(...))``. Every
    step's ENCUT must be at least that value. A failing search is refused
    with code ``ENCUT_BELOW_ENMAX_CODE``; the cutoff is never raised
    automatically (``ENCUT_BELOW_ENMAX_ACTION_NEEDS_LAB_DECISION``).
    """

    enmax = float(largest_enmax_ev)
    if not math.isfinite(enmax) or enmax <= 0:
        raise ValueError(f"largest_enmax_ev must be a positive number, not {largest_enmax_ev!r}")
    below = [
        {"step": step.number, "encut": step.settings["ENCUT"]}
        for step in incar_ladder(ranks)
        if float(step.settings["ENCUT"]) < enmax
    ]
    if below:
        steps = ", ".join(str(row["step"]) for row in below)
        cutoffs = ", ".join(f"{row['encut']} eV" for row in below)
        raise IncarPolicyError(
            f"The plane-wave cutoff of step(s) {steps} ({cutoffs}) is below the largest "
            f"ENMAX of this search's POTCARs ({enmax:g} eV). Searches whose POTCARs need a "
            "higher cutoff than the ladder uses are not supported yet.",
            code=ENCUT_BELOW_ENMAX_CODE,
            suggestion="Whether the early-step cutoff should be raised for such searches "
                       "is still a lab decision.",
            largest_enmax_ev=enmax,
            steps=[row["step"] for row in below],
        )


def render_incar(settings: Mapping[str, Any]) -> str:
    """INCAR text with one ``KEY = value`` line per setting, in the given order."""

    forbidden = sorted(FORBIDDEN_KEYS & set(settings))
    if forbidden:
        raise ValueError(f"Structure-search INCARs must not set {', '.join(forbidden)}")
    return "".join(f"{key} = {_incar_value(value)}\n" for key, value in settings.items())


def incar_texts(ranks: int = DEFAULT_RANKS_PER_RELAXATION) -> dict[str, str]:
    """``{"INCAR_1": text, ...}`` for every ladder step, checked for ``ranks`` MPI ranks."""

    return {f"INCAR_{step.number}": render_incar(step.settings) for step in incar_ladder(ranks)}


def write_incar_files(
    specific_dir: str | Path,
    ranks: int = DEFAULT_RANKS_PER_RELAXATION,
) -> list[Path]:
    """Write ``INCAR_1 .. INCAR_N`` into ``specific_dir`` (which must exist)."""

    paths = []
    for name, text in incar_texts(ranks).items():
        path = Path(specific_dir) / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def incar_policy_record(ranks: int = DEFAULT_RANKS_PER_RELAXATION) -> dict[str, Any]:
    """JSON-safe record of the ladder, for provenance."""

    return {
        "policy_id": INCAR_POLICY_ID,
        "policy_version": INCAR_POLICY_VERSION,
        "ranks_per_relaxation": ranks,
        "steps": [
            {"step": step.number, "purpose": step.purpose, "incar": dict(step.settings)}
            for step in incar_ladder(ranks)
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
    "ENCUT_BELOW_ENMAX_ACTION_NEEDS_LAB_DECISION",
    "ENCUT_BELOW_ENMAX_CODE",
    "STEP_4_CONVERGENCE_NEEDS_LAB_DECISION",
    "VOLUME_STEP_ENCUT_NEEDS_LAB_DECISION",
    "require_encut_covers_enmax",
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
