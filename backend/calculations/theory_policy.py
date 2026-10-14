from __future__ import annotations

from enum import Enum

from backend.calculations.models import Purpose, Theory


HSE06_SUPPORTED_PURPOSES = frozenset(
    {
        Purpose.RELAX,
        Purpose.RELAX_STATIC,
        Purpose.STATIC,
    }
)


class CalculationStage(str, Enum):
    RELAX = "relax"
    STATIC = "static"
    DOS = "dos"
    BAND_STRUCTURE = "band_structure"
    PHONON_FORCES = "phonon_forces"


HSE06_SUPPORTED_STAGES = frozenset(
    {
        CalculationStage.RELAX,
        CalculationStage.STATIC,
        CalculationStage.DOS,
        CalculationStage.BAND_STRUCTURE,
    }
)

_THEORY_DEFAULT_POTCAR_FUNCTIONAL = {
    Theory.PBE: "PBE_64",
    Theory.HSE06: "PBE_64",
}

_PURPOSE_STAGES = {
    Purpose.RELAX: CalculationStage.RELAX,
    Purpose.RELAX_STATIC: CalculationStage.STATIC,
    Purpose.DOUBLE_RELAX: CalculationStage.RELAX,
    Purpose.STATIC: CalculationStage.STATIC,
    Purpose.DOS: CalculationStage.DOS,
    Purpose.BAND_STRUCTURE: CalculationStage.BAND_STRUCTURE,
}

_HSE06_BASE_INCAR = {
    "LHFCALC": True,
    "AEXX": 0.25,
    "HFSCREEN": 0.2,
    "GGA": "PE",
    "ALGO": "Damped",
    "TIME": 0.4,
}

_HSE06_STAGE_INCAR = {
    CalculationStage.RELAX: {
        "PRECFOCK": "Fast",
    },
    CalculationStage.STATIC: {
        "PRECFOCK": "Accurate",
        "ISMEAR": 0,
    },
    CalculationStage.DOS: {
        "ALGO": "Normal",
        "PRECFOCK": "Fast",
        "ISMEAR": -5,
    },
    CalculationStage.BAND_STRUCTURE: {
        "ALGO": "Normal",
        "PRECFOCK": "Fast",
        "ISMEAR": 0,
        "SIGMA": 0.01,
    },
}

_HYBRID_PARALLEL_KEYS = ("KPAR", "NPAR")


def theory_default_potcar_functional(theory: Theory | str) -> str:
    normalized = Theory.from_value(theory)
    try:
        return _THEORY_DEFAULT_POTCAR_FUNCTIONAL[normalized]
    except KeyError as exc:
        raise ValueError(
            f"No default POTCAR functional is configured for {normalized.value}."
        ) from exc


def theory_supported_purposes(theory: Theory | str) -> frozenset[Purpose]:
    normalized = Theory.from_value(theory)
    if normalized is Theory.PBE:
        return frozenset(
            {
                Purpose.RELAX,
                Purpose.RELAX_STATIC,
                Purpose.DOUBLE_RELAX,
                Purpose.STATIC,
                Purpose.DOS,
                Purpose.BAND_STRUCTURE,
            }
        )
    if normalized is Theory.HSE06:
        return HSE06_SUPPORTED_PURPOSES

    return frozenset()


def theory_supported_stages(theory: Theory | str) -> frozenset[CalculationStage]:
    normalized = Theory.from_value(theory)
    if normalized is Theory.PBE:
        return frozenset(CalculationStage)
    if normalized is Theory.HSE06:
        return HSE06_SUPPORTED_STAGES

    return frozenset()


def theory_uses_hybrid_functional(theory: Theory | str) -> bool:
    return Theory.from_value(theory) is Theory.HSE06


def theory_stage_from_purpose(purpose: Purpose | str) -> CalculationStage:
    normalized = Purpose.from_value(purpose)
    try:
        return _PURPOSE_STAGES[normalized]
    except KeyError as exc:
        raise ValueError(f"No calculation stage is configured for {normalized.value}.") from exc


def _coerce_calculation_stage(stage_or_purpose) -> CalculationStage:
    if isinstance(stage_or_purpose, CalculationStage):
        return stage_or_purpose

    try:
        return theory_stage_from_purpose(Purpose.from_value(stage_or_purpose))
    except ValueError:
        pass

    normalized = (
        str(stage_or_purpose or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    for stage in CalculationStage:
        if normalized in (stage.value, stage.name.lower()):
            return stage

    raise ValueError(f"Unsupported calculation stage: {stage_or_purpose!r}")


def theory_incar_settings(
    theory: Theory | str,
    stage: CalculationStage | Purpose | str | None = None,
    *,
    purpose: Purpose | str | None = None,
) -> dict:
    normalized_theory = Theory.from_value(theory)
    if stage is None:
        stage = purpose
    normalized_stage = _coerce_calculation_stage(stage)

    if normalized_theory is Theory.HSE06:
        settings = dict(_HSE06_BASE_INCAR)
        settings.update(_HSE06_STAGE_INCAR.get(normalized_stage, {}))
        return settings

    return {}


def apply_theory_incar_settings(
    user_incar: dict | None,
    *,
    theory: Theory | str,
    stage: CalculationStage | Purpose | str | None = None,
    purpose: Purpose | str | None = None,
) -> dict:
    settings = dict(user_incar or {})
    for key, value in theory_incar_settings(theory, stage, purpose=purpose).items():
        settings.setdefault(key, value)

    if theory_uses_hybrid_functional(theory):
        for key in _HYBRID_PARALLEL_KEYS:
            settings.pop(key, None)

    return settings


__all__ = [
    "CalculationStage",
    "HSE06_SUPPORTED_PURPOSES",
    "HSE06_SUPPORTED_STAGES",
    "apply_theory_incar_settings",
    "theory_default_potcar_functional",
    "theory_incar_settings",
    "theory_stage_from_purpose",
    "theory_supported_purposes",
    "theory_supported_stages",
    "theory_uses_hybrid_functional",
]
