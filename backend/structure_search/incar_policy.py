"""INCAR ladder for USPEX structure searches (#65; lab decision of 5 Nov 2026).

USPEX copies ``Specific/INCAR_1 .. INCAR_N`` unchanged into every candidate's
calculation, one file per relaxation step. One INCAR set therefore has to work
for any candidate cell, and nothing in it may depend on a particular
structure (no MAGMOM, no NBANDS, a fixed KPAR, ``ISPIN = 1``). The
composition is fixed for a whole search, so values that depend only on the
composition, such as the plane-wave cutoffs, are set once per search and are
still identical for every candidate.

BMD scientific decision (roeeasher for the BMD Lab, 5 Nov 2026, recorded on
PR #74):

    The final relaxation step must be consistent with the accuracy of BMD's
    Geometry Optimisation already implemented in bmd-compute. Before that,
    follow the USPEX default recommendations (manual section 3.1).

with two follow-up choices: the final relaxation is BMD Geometry
Optimisation pass 1 unchanged, and a ranking single point follows it.

====  =================================  ====  ======  =====  ===  ==================  =====  ======  =================  ========
step  source                             ISIF  IBRION  POTIM  NSW  ENCUT               EDIFF  EDIFFG  smearing           PREC
====  =================================  ====  ======  =====  ===  ==================  =====  ======  =================  ========
1     USPEX manual INCAR_1               4     2       0.02   65   early               1e-2   1e-1    MP, SIGMA 0.10     Low
2     USPEX manual INCAR_2               4     1       0.30   55   early               1e-3   1e-2    MP, SIGMA 0.08     Normal
3     USPEX manual INCAR_3               3     2       0.02   65   relaxation          1e-3   1e-2    MP, SIGMA 0.07     Normal
4     BMD Geometry Optimisation pass 1   3     2       -      99   relaxation          1e-6   -0.01   Gauss, SIGMA 0.2   Accurate
5     ranking single point               2     -1      -      0    max(620, relax.)    1e-6   -       Gauss, SIGMA 0.05  Accurate
====  =================================  ====  ======  =====  ===  ==================  =====  ======  =================  ========

- Steps 1-3 set only what the manual's example INCARs set, plus the
  search-wide tags below: no ALGO, LREAL, LASPH, NELM or ADDGRID.
- Step 4 *is* BMD Geometry Optimisation pass 1
  (``bmd_compute.geometry_relaxation/1``). Its INCAR is the one BMD's own
  input generator writes for that stage for the search's elements
  (:func:`bmd_stage_settings`); it is not retyped here.
- Step 5 gives the energy USPEX ranks candidates by. USPEX ranks on VASP's
  free energy F, not E0 (Si8 pilot: individual 27 has F = -39.702683 eV and
  E0 = -39.702789 eV, and ``Individuals_extended`` shows -39.7027). With
  step 4's SIGMA 0.2, the smearing-entropy term in F would favour metallic
  candidates, so a single point with Gaussian SIGMA 0.05 follows. Its other
  settings are those BMD's generator writes for a Static Energy after a
  relaxation (including ADDGRID), except:

  - ENCUT is ``max(620 eV, relaxation cutoff)``, the rule
    ``docs/methodology.md`` section 3 gives for a static after a relaxation
    without the adaptive geometry record. A search has no adaptive
    refinement, so the managed workflow's ``max(620, E0 + 150)`` does not
    apply;
  - Gaussian ``SIGMA = 0.05`` replaces the tetrahedron method
    (``ISMEAR = -5``), which candidates with coarse meshes can have too few
    k-points for;
  - ``ISIF = 2`` and ``IBRION = -1`` are written explicitly;
  - output-only flags (``OUTPUT_ONLY_STATIC_KEYS``) are dropped: a search
    runs thousands of calculations, and section 9 of the methodology
    excludes output-writing flags from the declared methodology.

Cutoffs, from the search's POTCARs (:func:`search_cutoff_record`):

- "relaxation" is BMD's species-dependent relaxation cutoff,
  ``bmd_compute.plane_wave_cutoff/1``: ``ceil(1.3 x max ENMAX / 50 eV) x
  50 eV`` (the manual's "+30-40 %"). It is resolved by the same function and
  versioned ENMAX metadata as every BMD Geometry Optimisation.
- "early" is ``ceil(EARLY_STEP_ENCUT_MULTIPLIER x max ENMAX / 50 eV) x 50 eV``
  with multiplier 1.0 (the manual's "default cutoff"), so it is never below
  a POTCAR's ENMAX.

Examples: Si gives 250 / 350 / 620 eV (early / relaxation / ranking); Li2O
(Li_sv, ENMAX 499.034 eV) gives 500 / 650 / 650 eV.

Search-wide tags on every step (``SEARCH_WIDE_SETTINGS``): ``ISPIN = 1``,
``NCORE = 4``, ``KPAR = 1`` (still a lab decision), ``LWAVE`` and ``LCHARG``
off. No step sets ``PSTRESS`` (pressure comes from ``ExternalPressure`` in
INPUT.txt, manual section 4.1), ``MAGMOM``, ``NBANDS``, ``NPAR``, ``LDAU``,
``LSORBIT`` or ``IVDW``.

The only value still awaiting a lab decision is
``FIXED_KPAR_NEEDS_LAB_DECISION`` (see :data:`VALUES_NEEDING_LAB_DECISION`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.calculations.geometry_relaxation import (
    CUTOFF_INCREMENT_EV,
    CUTOFF_RULE_ID,
    CUTOFF_RULE_VERSION,
    PROTOCOL_ID,
    PROTOCOL_VERSION,
    resolve_cutoff_record,
)
from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import DEFAULT_NCORE
from backend.calculations.vasp_stage_definitions import ENCUT_STATIC_FINAL_DEFAULT
from backend.structure_search.potcar import POTCAR_FUNCTIONAL, bmd_potcar_symbols


INCAR_POLICY_ID = "bmd_compute.structure_search_incar"
INCAR_POLICY_VERSION = 1

LAB_DECISION = {
    "date": "2026-11-05",
    "decided_by": "roeeasher for the BMD Lab",
    "recorded_in": "https://github.com/bmd-lab/bmd-compute/pull/74#issuecomment-6654619861",
    "rule": (
        "The final relaxation step must be consistent with the accuracy of BMD's "
        "Geometry Optimisation already implemented in bmd-compute. Before that, "
        "follow the USPEX default recommendations (manual section 3.1)."
    ),
}


# ---------------------------------------------------------------------------
# The one value that still needs a lab decision.
# ---------------------------------------------------------------------------

# KPAR for every step. Because the INCAR is the same for every candidate, KPAR
# must work for any k-point count (rules in validate_parallel_tags). KPAR = 1
# meets every rule for any rank count and any number of k-points.
# KPAR = 4 also meets the two divisibility rules with 48 ranks. The lab's
# notes measured KPAR as worth up to 9x on a k-point-dense cell. But
# KPAR = 4 exceeds NKPTS for candidates with fewer than 4 irreducible
# k-points, which a search does produce (large cells, coarse early
# KresolStart). So it stays at 1 until the full-node test (#59) shows the
# speed-up on real search candidates, and how VASP behaves when
# KPAR > NKPTS.
FIXED_KPAR_NEEDS_LAB_DECISION = 1

VALUES_NEEDING_LAB_DECISION = {
    "FIXED_KPAR_NEEDS_LAB_DECISION": FIXED_KPAR_NEEDS_LAB_DECISION,
}


# ---------------------------------------------------------------------------
# Decided parts of the policy.
# ---------------------------------------------------------------------------

# Multiplier of the early-step (1 and 2) cutoff rule:
# ENCUT = ceil(multiplier x max ENMAX / 50 eV) x 50 eV. 1.0 is the manual's
# "default cutoff": the lowest 50 eV step at or above every POTCAR's ENMAX.
EARLY_STEP_ENCUT_MULTIPLIER = 1.0

# MPI ranks of one relaxation (docs/structure_search.md section 4: 4 x 48 on
# a 192-core node; to be measured in Phase 0). The INCARs do not contain the
# rank count, but KPAR and NCORE are checked against it.
DEFAULT_RANKS_PER_RELAXATION = 48

# NCORE = 4 is BMD's fixed value (backend/calculations/resources.py). In the
# lab's August 2026 leeburton-pool VASP notes it was the best measured value,
# with moderate confidence. Those notes come from a single small cell
# (3-atom Sc2C, 65 irreducible k-points), not from search candidates, so
# they are not general results. NPAR is never set together with NCORE.

# Tags every step sets, whatever its source; they override the BMD stage
# values where both set a tag.
SEARCH_WIDE_SETTINGS: Mapping[str, Any] = {
    "ISPIN": 1,
    "NCORE": DEFAULT_NCORE,
    "KPAR": FIXED_KPAR_NEEDS_LAB_DECISION,
    "LWAVE": False,
    "LCHARG": False,
}

# Steps 1-3: exactly the tags of the USPEX 25.0 manual's example INCAR_1..3
# (section 3.1), as given in the lab decision. "early" and "relaxation" are
# replaced by the search's cutoffs.
MANUAL_STEPS: tuple[Mapping[str, Any], ...] = (
    {"PREC": "Low", "ENCUT": "early", "EDIFF": 1e-2, "EDIFFG": 1e-1, "NSW": 65,
     "IBRION": 2, "POTIM": 0.02, "ISIF": 4, "ISMEAR": 1, "SIGMA": 0.10},
    {"PREC": "Normal", "ENCUT": "early", "EDIFF": 1e-3, "EDIFFG": 1e-2, "NSW": 55,
     "IBRION": 1, "POTIM": 0.30, "ISIF": 4, "ISMEAR": 1, "SIGMA": 0.08},
    {"PREC": "Normal", "ENCUT": "relaxation", "EDIFF": 1e-3, "EDIFFG": 1e-2, "NSW": 65,
     "IBRION": 2, "POTIM": 0.02, "ISIF": 3, "ISMEAR": 1, "SIGMA": 0.07},
)

# Step 5 overrides of BMD's static-after-relaxation INCAR.
RANKING_SMEARING: Mapping[str, Any] = {"ISMEAR": 0, "SIGMA": 0.05}
RANKING_IONIC: Mapping[str, Any] = {"ISIF": 2, "IBRION": -1, "NSW": 0}
# Output-writing flags of BMD's Static Energy that the ranking single point
# drops (docs/methodology.md section 9: not part of the declared methodology).
OUTPUT_ONLY_STATIC_KEYS = frozenset({"NEDOS", "LORBIT", "LVTOT", "LAECHG", "LELF", "LVHAR"})

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
# Steps 4 and 5 (the BMD-standard steps) must be at least this dense; steps
# 1-3 use the user's spec values. How USPEX rounds a KresolStart value to a
# mesh is not documented here; Phase 0 should compare the KPOINTS USPEX writes
# with pymatgen's mesh for the same cell.
BMD_RECIPROCAL_DENSITY = 64
BMD_EQUIVALENT_KRESOL = BMD_RECIPROCAL_DENSITY ** (-1.0 / 3.0) / (2.0 * math.pi)
BMD_STANDARD_MAX_KRESOL = 0.04

# Steps (1-based) whose KresolStart must meet BMD_STANDARD_MAX_KRESOL.
BMD_STANDARD_STEPS = (4, 5)

ENCUT_BELOW_ENMAX_CODE = "encut_below_potcar_enmax"
POTCAR_ENMAX_MISMATCH_CODE = "potcar_enmax_differs_from_cutoff_record"
INVALID_CUTOFF_RECORD_CODE = "invalid_plane_wave_cutoff_record"
BMD_STAGE_MISMATCH_CODE = "bmd_stage_inputs_differ_from_search_record"

STEP_SOURCES = (
    "USPEX manual INCAR_1 (section 3.1)",
    "USPEX manual INCAR_2 (section 3.1)",
    "USPEX manual INCAR_3 (section 3.1)",
    f"BMD Geometry Optimisation pass 1 ({PROTOCOL_ID}/{PROTOCOL_VERSION})",
    "ranking single point (BMD Static Energy after a relaxation, Gaussian SIGMA 0.05)",
)
NUM_STEPS = len(STEP_SOURCES)


@dataclass(frozen=True)
class LadderStep:
    """One relaxation step of the ladder: where it comes from and its INCAR settings."""

    number: int
    source: str
    settings: Mapping[str, Any]


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
        and isinstance(record.get("potcars"), list)
        and bool(record["potcars"])
        and isinstance(record.get("functional"), str)
    )
    if not valid:
        raise IncarPolicyError(
            "The structure-search INCARs need a resolved BMD plane-wave cutoff record "
            f"({CUTOFF_RULE_ID}/{CUTOFF_RULE_VERSION}); use search_cutoff_record().",
            code=INVALID_CUTOFF_RECORD_CODE,
        )
    return record


def early_step_encut(max_enmax_ev: float) -> int:
    """``ceil(EARLY_STEP_ENCUT_MULTIPLIER x max ENMAX / 50 eV) x 50 eV``."""

    scaled = Decimal(repr(float(max_enmax_ev))) * Decimal(repr(EARLY_STEP_ENCUT_MULTIPLIER))
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


def _element_of(symbol: str) -> str:
    return str(symbol).split("_", 1)[0]


def _plain(value):
    # pymatgen returns ENCUT as a float (350.0); INCARs here write whole numbers.
    if isinstance(value, float) and value.is_integer() and abs(value) >= 1:
        return int(value)
    return value


@lru_cache(maxsize=32)
def _bmd_stage_incars(symbols: tuple[str, ...], functional: str, metadata_digest: str) -> tuple[dict, dict]:
    """BMD's generated INCARs for pass 1 and the static after it, for these POTCARs.

    Builds a placeholder cell with one atom of each element and asks BMD's
    own input generators (POTCAR.spec mode, no POTCAR file is read) for the
    INCARs of a BMD Standard Geometry Relaxation followed by a PBE Static
    Energy. Only composition-level settings are used from them: every key
    that depends on a structure is either absent for a non-magnetic PBE
    stage or forbidden by FORBIDDEN_KEYS. ``metadata_digest`` only keys the
    cache, so a change of ENMAX metadata regenerates the INCARs.
    """

    from pymatgen.core import Lattice, Structure

    from backend.calculations.geometry_relaxation import (
        freeze_plane_wave_cutoffs,
        protocol_relaxation_stages,
    )
    from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
    from backend.calculations.potcar_record import input_set_potcar_symbols
    from backend.generated_inputs import generated_input_stage_previews

    elements = [_element_of(symbol) for symbol in symbols]
    coords = [[i / len(elements)] * 3 for i in range(len(elements))]
    structure = Structure(Lattice.cubic(4.0 * len(elements)), elements, coords)
    workflow = WorkflowSpec(stages=(
        *protocol_relaxation_stages(),
        StageSpec(stage_type=StageType.STATIC, theory=Theory.PBE),
    ))
    workflow = freeze_plane_wave_cutoffs(structure, workflow, potcar_functional=functional)
    previews = generated_input_stage_previews(structure, workflow, potcar_functional=functional)
    generated_symbols = tuple(input_set_potcar_symbols(previews[0]["input_set"]))
    if generated_symbols != symbols:
        raise IncarPolicyError(
            f"BMD's generator selected POTCARs {list(generated_symbols)}, but the search's "
            f"cutoff record has {list(symbols)}.",
            code=BMD_STAGE_MISMATCH_CODE,
        )
    pass_1 = {key: _plain(value) for key, value in dict(previews[0]["input_set"].incar).items()}
    static = {key: _plain(value) for key, value in dict(previews[2]["input_set"].incar).items()}
    return pass_1, static


def bmd_stage_settings(cutoff_record: Mapping[str, Any]) -> tuple[dict, dict]:
    """(pass-1 INCAR, static-after-relaxation INCAR) BMD writes for this search's POTCARs.

    The pass-1 INCAR must use exactly the search's resolved relaxation cutoff;
    otherwise the two were resolved from different metadata and the search is
    refused (``BMD_STAGE_MISMATCH_CODE``).
    """

    record = _require_cutoff_record(cutoff_record)
    symbols = tuple(str(row["symbol"]) for row in record["potcars"])
    digest = str((record.get("metadata") or {}).get("digest", ""))
    pass_1, static = _bmd_stage_incars(symbols, str(record["functional"]), digest)
    if int(pass_1.get("ENCUT", -1)) != int(record["ENCUT"]):
        raise IncarPolicyError(
            f"BMD's Geometry Optimisation resolved ENCUT {pass_1.get('ENCUT')} eV for these "
            f"POTCARs, but the search's cutoff record has {record['ENCUT']} eV.",
            code=BMD_STAGE_MISMATCH_CODE,
        )
    return dict(pass_1), dict(static)


def _sorted(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {key: settings[key] for key in sorted(settings)}


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

    validate_parallel_tags(SEARCH_WIDE_SETTINGS["KPAR"], SEARCH_WIDE_SETTINGS["NCORE"], ranks)
    cutoffs = ladder_cutoffs(cutoff_record)
    pass_1, static = bmd_stage_settings(cutoff_record)

    manual = [
        {key: (cutoffs[value] if key == "ENCUT" else value) for key, value in step.items()}
        for step in MANUAL_STEPS
    ]
    ranking = {key: value for key, value in static.items() if key not in OUTPUT_ONLY_STATIC_KEYS}
    ranking.update(RANKING_IONIC)
    ranking.update(RANKING_SMEARING)
    ranking["ENCUT"] = cutoffs["ranking"]
    rows = (*manual, _sorted(pass_1), _sorted(ranking))

    steps = []
    for index, (source, settings) in enumerate(zip(STEP_SOURCES, rows), start=1):
        merged = {key: value for key, value in settings.items() if key not in SEARCH_WIDE_SETTINGS}
        merged.update(SEARCH_WIDE_SETTINGS)
        steps.append(LadderStep(number=index, source=source, settings=merged))
    return tuple(steps)


def require_spec_matches_ladder(spec) -> None:
    """Check a search's KresolStart values against the ladder and BMD k-point policy.

    The spec must have one KresolStart value per ladder step, and the
    BMD-standard steps (4 and 5) must be at least as dense as BMD's k-point
    policy. Steps 1-3 use the user's values.
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
    this always holds; the check is a safety net. A failing search is
    refused with code ``ENCUT_BELOW_ENMAX_CODE``; the cutoff is never raised
    here.
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
    form BMD freezes into a Geometry Optimisation stage, and
    ``step_4_protocol`` records that step 4 is that protocol's pass 1.
    """

    return {
        "policy_id": INCAR_POLICY_ID,
        "policy_version": INCAR_POLICY_VERSION,
        "lab_decision": dict(LAB_DECISION),
        "ranks_per_relaxation": ranks,
        "plane_wave_cutoff": dict(_require_cutoff_record(cutoff_record)),
        "cutoffs": ladder_cutoffs(cutoff_record),
        "step_4_protocol": {"id": PROTOCOL_ID, "version": PROTOCOL_VERSION, "pass": 1},
        "steps": [
            {"step": step.number, "source": step.source, "incar": dict(step.settings)}
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
    "BMD_STAGE_MISMATCH_CODE",
    "BMD_STANDARD_MAX_KRESOL",
    "BMD_STANDARD_STEPS",
    "DEFAULT_RANKS_PER_RELAXATION",
    "EARLY_STEP_ENCUT_MULTIPLIER",
    "ENCUT_BELOW_ENMAX_CODE",
    "FIXED_KPAR_NEEDS_LAB_DECISION",
    "FORBIDDEN_KEYS",
    "INCAR_POLICY_ID",
    "INCAR_POLICY_VERSION",
    "INVALID_CUTOFF_RECORD_CODE",
    "IncarPolicyError",
    "LAB_DECISION",
    "LadderStep",
    "MANUAL_STEPS",
    "NUM_STEPS",
    "OUTPUT_ONLY_STATIC_KEYS",
    "POTCAR_ENMAX_MISMATCH_CODE",
    "RANKING_IONIC",
    "RANKING_SMEARING",
    "SEARCH_WIDE_SETTINGS",
    "STEP_SOURCES",
    "VALUES_NEEDING_LAB_DECISION",
    "bmd_stage_settings",
    "early_step_encut",
    "incar_ladder",
    "incar_policy_record",
    "incar_texts",
    "ladder_cutoffs",
    "render_incar",
    "require_encut_covers_enmax",
    "require_potcars_match_cutoff_record",
    "require_spec_matches_ladder",
    "search_cutoff_record",
    "validate_parallel_tags",
    "write_incar_files",
]
