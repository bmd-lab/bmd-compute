from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from backend.calculations.models import StageType, Theory
from backend.calculations.resource_policy import stage_allows_automatic_ncore
from backend.calculations.theory_policy import (
    CalculationStage,
    theory_incar_settings,
    theory_supported_stages,
)


"""
Internal executable-stage definitions for BMD Compute.

These definitions describe BMD Compute's current executable choices. They are
not a methodology authority.
BMDex remains the authority for methodology, evidence, and provenance policy.
"""


ENCUT_STATIC_PREP_DEFAULT = 520
ENCUT_RELAX_DEFAULT = 580
ENCUT_STATIC_FINAL_DEFAULT = 620
BAND_STRUCTURE_LINE_DENSITY_DEFAULT = 40
HSE_DOS_RECIPROCAL_DENSITY_DEFAULT = 100
HSE_BAND_STRUCTURE_RECIPROCAL_DENSITY_DEFAULT = 64

_STAGE_DEFINITION_SOURCE = "backend.calculations.vasp_stage_definitions"
_HSE_DOS_BASE_SOURCE = (
    f"{_STAGE_DEFINITION_SOURCE}.apply_hse_dos_base_incar_settings"
)
_HSE_BAND_BASE_SOURCE = (
    f"{_STAGE_DEFINITION_SOURCE}.apply_hse_band_structure_base_incar_settings"
)
_KPOINT_EXTRA_PARAMETER_KEYS = frozenset({"line_density", "reciprocal_density"})


@dataclass(frozen=True)
class Atomate2Selection:
    input_set_generator: str
    maker: str
    generator_mode: str | None = None
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageIncarPolicy:
    executable_defaults: Mapping[str, Any] = field(default_factory=dict)
    compatibility_defaults: Mapping[str, Any] = field(default_factory=dict)
    explicit_user_preserved_defaults: Mapping[str, Any] = field(default_factory=dict)
    final_defaults: Mapping[str, Any] = field(default_factory=dict)
    prep_defaults: Mapping[str, Any] = field(default_factory=dict)
    final_encut_floor: int | None = None
    prep_encut_floor: int | None = None
    compatibility_conditional_sigma_for_tetrahedron: bool = False
    prep_conditional_sigma_for_tetrahedron: bool = False


@dataclass(frozen=True)
class VaspStageDefinition:
    stage_type: StageType
    atomate2: Atomate2Selection
    incar_policy: StageIncarPolicy
    restart_policy: Mapping[str, Any]
    kpoints_policy: Mapping[str, Any]


_STATIC_COMMON_DEFAULTS = {
    "GGA": None,
    "ENAUG": None,
    "LMIXTAU": None,
    "EDIFF": 1e-6,
    "ALGO": "Normal",
}

_STATIC_COMPATIBILITY_DEFAULTS = {
    **_STATIC_COMMON_DEFAULTS,
    "ISMEAR": -5,
    "NEDOS": 3001,
    "LORBIT": 11,
    "LVTOT": True,
    "LAECHG": True,
    "LCHARG": True,
    "LWAVE": True,
    "LELF": True,
}

_STATIC_FINAL_DEFAULTS = {
    **_STATIC_COMMON_DEFAULTS,
    "LWAVE": False,
    "LCHARG": True,
    "ISMEAR": -5,
    "SIGMA": 0.05,
    "NEDOS": 4001,
    "LORBIT": 11,
    "LREAL": False,
    "PREC": "Accurate",
    "ADDGRID": True,
    "LVTOT": True,
    "LAECHG": True,
    "LELF": True,
}

_STATIC_PREP_DEFAULTS = {
    **_STATIC_COMMON_DEFAULTS,
    "LWAVE": True,
    "LCHARG": True,
    "LVTOT": False,
    "LELF": False,
    "LVHAR": False,
    "LAECHG": False,
    "ISMEAR": -5,
    "NEDOS": 3001,
    "LORBIT": 11,
}

_RELAX_COMPATIBILITY_DEFAULTS = {
    "GGA": None,
    "ENAUG": None,
    "LMIXTAU": None,
    "ALGO": "Fast",
    "ADDGRID": True,
    "EDIFFG": -0.01,
}

_RELAX_EXECUTABLE_DEFAULTS = {
    "ENCUT": ENCUT_RELAX_DEFAULT,
    "EDIFF": 1e-6,
    **_RELAX_COMPATIBILITY_DEFAULTS,
}

_RELAX_EXPLICIT_USER_PRESERVED_DEFAULTS = {
    "LCHARG": False,
    "LWAVE": False,
    "LAECHG": None,
    "LVTOT": None,
    "LELF": None,
    "LVHAR": None,
    "LORBIT": None,
}

_RESTART_INCAR_AMENDMENTS = {"ICHARG": 11}

_HSE_BAND_STRUCTURE_STAGE_DEFAULTS = {
    "ENAUG": None,
    "LMIXTAU": None,
    "ADDGRID": True,
    "EDIFF": 1e-6,
    "LORBIT": 11,
    "LREAL": False,
    "PREC": "Accurate",
}

_HSE_DOS_STAGE_DEFAULTS = {
    **_HSE_BAND_STRUCTURE_STAGE_DEFAULTS,
    "NEDOS": 4001,
}


def _atomate2(
    generator: str,
    maker: str,
    mode: str | None = None,
    **extra_parameters: Any,
) -> Atomate2Selection:
    return Atomate2Selection(
        input_set_generator=f"atomate2.vasp.sets.core.{generator}",
        maker=f"atomate2.vasp.jobs.core.{maker}",
        generator_mode=mode,
        extra_parameters=extra_parameters,
    )


def _restart(
    description: str,
    *,
    requires_previous_stage: bool = False,
    incar_amendments: Mapping[str, Any] | None = None,
) -> dict:
    return {
        "requires_previous_stage": requires_previous_stage,
        "incar_amendments": incar_amendments or {},
        "description": description,
    }


def _kpoints(
    mode: str,
    description: str,
    **default_parameters: Any,
) -> dict:
    return {
        "mode": mode,
        "default_parameters": default_parameters,
        "description": description,
    }


_STAGE_DEFINITIONS: dict[StageType, VaspStageDefinition] = {
    StageType.RELAX: VaspStageDefinition(
        stage_type=StageType.RELAX,
        atomate2=_atomate2("RelaxSetGenerator", "RelaxMaker"),
        incar_policy=StageIncarPolicy(
            executable_defaults=_RELAX_EXECUTABLE_DEFAULTS,
            compatibility_defaults=_RELAX_COMPATIBILITY_DEFAULTS,
            explicit_user_preserved_defaults=_RELAX_EXPLICIT_USER_PRESERVED_DEFAULTS,
        ),
        restart_policy=_restart(
            "Relax stages start from the submitted or previous structure."
        ),
        kpoints_policy=_kpoints(
            "pymatgen/atomate2 default",
            "Uses the selected pymatgen/atomate2 relax KPOINTS policy unless BMD stage options request gamma or mesh settings.",
        ),
    ),
    StageType.STATIC: VaspStageDefinition(
        stage_type=StageType.STATIC,
        atomate2=_atomate2("StaticSetGenerator", "StaticMaker"),
        incar_policy=StageIncarPolicy(
            compatibility_defaults=_STATIC_COMPATIBILITY_DEFAULTS,
            final_defaults=_STATIC_FINAL_DEFAULTS,
            prep_defaults=_STATIC_PREP_DEFAULTS,
            final_encut_floor=ENCUT_STATIC_FINAL_DEFAULT,
            prep_encut_floor=ENCUT_STATIC_PREP_DEFAULT,
            compatibility_conditional_sigma_for_tetrahedron=True,
            prep_conditional_sigma_for_tetrahedron=True,
        ),
        restart_policy=_restart(
            "Static stages may consume the previous stage structure and directory when part of a workflow."
        ),
        kpoints_policy=_kpoints(
            "pymatgen/atomate2 default",
            "Uses the selected pymatgen/atomate2 static KPOINTS policy unless BMD stage options request gamma or mesh settings.",
        ),
    ),
    StageType.DOS: VaspStageDefinition(
        stage_type=StageType.DOS,
        atomate2=_atomate2("NonSCFSetGenerator", "NonSCFMaker", "uniform"),
        incar_policy=StageIncarPolicy(),
        restart_policy=_restart(
            "DOS stages consume the preceding Static Energy stage and restart from its converged charge density.",
            requires_previous_stage=True,
            incar_amendments=_RESTART_INCAR_AMENDMENTS,
        ),
        kpoints_policy=_kpoints(
            "uniform",
            "Uses atomate2 NonSCFSetGenerator uniform mode with any selected BMD k-point override.",
        ),
    ),
    StageType.BAND_STRUCTURE: VaspStageDefinition(
        stage_type=StageType.BAND_STRUCTURE,
        atomate2=_atomate2("NonSCFSetGenerator", "NonSCFMaker", "line"),
        incar_policy=StageIncarPolicy(),
        restart_policy=_restart(
            "PBE Band Structure stages consume the preceding Static Energy stage and restart from its converged charge density.",
            requires_previous_stage=True,
            incar_amendments=_RESTART_INCAR_AMENDMENTS,
        ),
        kpoints_policy=_kpoints(
            "line",
            "Uses line-mode high-symmetry k-points.",
            line_density=BAND_STRUCTURE_LINE_DENSITY_DEFAULT,
        ),
    ),
}

_THEORY_STAGE_OVERRIDES = {
    (Theory.HSE06, StageType.DOS): {
        "atomate2": _atomate2(
            "HSEBSSetGenerator",
            "HSEBSMaker",
            "uniform",
            reciprocal_density=HSE_DOS_RECIPROCAL_DENSITY_DEFAULT,
            custodian_policy="backend.calculations.custodian_policy.hse_band_structure_run_vasp_kwargs",
        ),
        "bmd_incar_defaults": _HSE_DOS_STAGE_DEFAULTS,
        "encut_floor": ENCUT_STATIC_FINAL_DEFAULT,
        "restart_policy": _restart(
            "HSE06 DOS stages run a self-consistent hybrid calculation on a uniform weighted k-point mesh. A preceding Static Energy stage supplies the structure and optional starting charge density.",
            requires_previous_stage=True,
        ),
        "source": _HSE_DOS_BASE_SOURCE,
    },
    (Theory.HSE06, StageType.BAND_STRUCTURE): {
        "atomate2": _atomate2(
            "HSEBSSetGenerator",
            "HSEBSMaker",
            "line",
            reciprocal_density=HSE_BAND_STRUCTURE_RECIPROCAL_DENSITY_DEFAULT,
            custodian_policy="backend.calculations.custodian_policy.hse_band_structure_run_vasp_kwargs",
        ),
        "bmd_incar_defaults": _HSE_BAND_STRUCTURE_STAGE_DEFAULTS,
        "encut_floor": ENCUT_STATIC_FINAL_DEFAULT,
        "restart_policy": _restart(
            "HSE06 Band Structure stages run a self-consistent hybrid calculation with a weighted uniform mesh plus zero-weight high-symmetry line path. A preceding HSE06 Static Energy stage supplies the structure and optional starting charge density.",
            requires_previous_stage=True,
        ),
        "source": _HSE_BAND_BASE_SOURCE,
    },
}


def stage_definition(stage_type: StageType | str) -> VaspStageDefinition:
    normalized = StageType.from_value(stage_type)
    try:
        return _STAGE_DEFINITIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"No VASP stage definition is configured for {normalized.value}.") from exc


def list_stage_definitions() -> tuple[dict, ...]:
    stages = sorted(_STAGE_DEFINITIONS, key=lambda item: item.value)
    return tuple(describe_stage(stage_type) for stage_type in stages)


def describe_stage(stage_type: StageType | str, theory: Theory | str | None = None) -> dict:
    definition = stage_definition(stage_type)
    normalized_theory = Theory.from_value(theory) if theory is not None else None
    override = _theory_stage_override_for(definition.stage_type, normalized_theory)
    atomate2 = _atomate2_selection_for(definition.stage_type, normalized_theory)
    calculation_stage = CalculationStage(definition.stage_type.value)
    theory_supported_for_stage = (
        calculation_stage in theory_supported_stages(normalized_theory)
        if normalized_theory is not None
        else None
    )
    theory_amendments = (
        theory_incar_settings(
            normalized_theory,
            calculation_stage,
        )
        if normalized_theory is not None and theory_supported_for_stage
        else {}
    )

    return {
        "scope": "BMD Compute executable implementation, not a methodology authority",
        "stage_type": definition.stage_type.value,
        "theory": normalized_theory.value if normalized_theory else None,
        "theory_supported_for_stage": theory_supported_for_stage,
        "base_bmd_incar_amendments": _base_bmd_incar_amendments(definition),
        "theory_stage_bmd_incar_amendments": _theory_stage_bmd_incar_amendments(override),
        "selected_atomate2": _atomate2_description(atomate2),
        "applicable_theory_amendments": _copy_mapping(theory_amendments),
        "restart_policy": _restart_description(
            override.get("restart_policy", definition.restart_policy)
            if override is not None
            else definition.restart_policy
        ),
        "kpoints_policy": _kpoints_description(definition.kpoints_policy, atomate2),
        "resource_policy": {
            "automatic_ncore_eligible": stage_allows_automatic_ncore(definition.stage_type),
            "source": "backend.calculations.resource_policy.stage_allows_automatic_ncore",
        },
        "implementation_source": {
            "stage_definition": _STAGE_DEFINITION_SOURCE,
            "theory_policy": "backend.calculations.theory_policy",
            "resource_policy": "backend.calculations.resource_policy",
            "workflow_builder": "backend.workflows",
        },
    }


def apply_stage_base_incar_settings(
    stage_type: StageType | str,
    user_incar: Mapping[str, Any] | None,
    *,
    intent: str = "final",
    explicit_user_settings: Mapping[str, Any] | None = None,
) -> dict:
    definition = stage_definition(stage_type)
    settings = dict(user_incar or {})
    explicit = dict(explicit_user_settings or {})
    policy = definition.incar_policy

    if definition.stage_type is StageType.RELAX:
        _apply_defaults(settings, policy.executable_defaults)
        _apply_explicit_user_preserved_defaults(
            settings,
            policy.explicit_user_preserved_defaults,
            explicit,
        )
        return settings

    if definition.stage_type is StageType.STATIC:
        normalized_intent = str(intent or "final").strip().lower()
        if normalized_intent == "prep":
            _apply_defaults(settings, policy.prep_defaults)
            if policy.prep_conditional_sigma_for_tetrahedron:
                _apply_conditional_tetrahedron_sigma(settings)
            _apply_encut_floor(settings, policy.prep_encut_floor)
        else:
            _apply_defaults(settings, policy.final_defaults)
            _apply_encut_floor(settings, policy.final_encut_floor)
        return settings

    return settings


def apply_stage_restart_incar_settings(
    stage_type: StageType | str,
    user_incar: Mapping[str, Any] | None,
) -> dict:
    definition = stage_definition(stage_type)
    settings = dict(user_incar or {})
    settings.update(_copy_mapping(definition.restart_policy["incar_amendments"]))
    return settings


def stage_restart_incar_amendments(
    stage_type: StageType | str,
    theory: Theory | str | None = None,
) -> dict:
    """Return the restart INCAR amendments for a stage type and theory."""

    definition = stage_definition(stage_type)
    normalized_theory = Theory.from_value(theory) if theory is not None else None
    override = _theory_stage_override_for(definition.stage_type, normalized_theory)
    policy = (
        override.get("restart_policy", definition.restart_policy)
        if override is not None
        else definition.restart_policy
    )
    return _copy_mapping(policy["incar_amendments"])


def stage_reads_previous_charge_density(
    stage_type: StageType | str,
    theory: Theory | str | None = None,
) -> bool:
    """Return whether the stage restarts from the previous stage's fixed CHGCAR."""

    return stage_restart_incar_amendments(stage_type, theory).get("ICHARG") == 11


def apply_hse_dos_base_incar_settings(
    user_incar: Mapping[str, Any] | None,
) -> dict:
    settings = dict(user_incar or {})
    override = _theory_stage_override_for(StageType.DOS, Theory.HSE06)
    if override is not None:
        _apply_defaults(settings, override["bmd_incar_defaults"])
        _apply_encut_floor(settings, override["encut_floor"])
    return settings


def apply_hse_band_structure_base_incar_settings(
    user_incar: Mapping[str, Any] | None,
) -> dict:
    settings = dict(user_incar or {})
    override = _theory_stage_override_for(StageType.BAND_STRUCTURE, Theory.HSE06)
    if override is not None:
        _apply_defaults(settings, override["bmd_incar_defaults"])
        _apply_encut_floor(settings, override["encut_floor"])
    return settings


def apply_static_compatibility_incar_settings(
    user_incar: Mapping[str, Any] | None,
) -> dict:
    settings = dict(user_incar or {})
    policy = stage_definition(StageType.STATIC).incar_policy
    _apply_defaults(settings, policy.compatibility_defaults)
    if policy.compatibility_conditional_sigma_for_tetrahedron:
        _apply_conditional_tetrahedron_sigma(settings)
    return settings


def apply_relax_compatibility_incar_settings(
    user_incar: Mapping[str, Any] | None,
    *,
    explicit_user_settings: Mapping[str, Any] | None = None,
) -> dict:
    settings = dict(user_incar or {})
    policy = stage_definition(StageType.RELAX).incar_policy
    _apply_explicit_user_preserved_defaults(
        settings,
        policy.explicit_user_preserved_defaults,
        dict(explicit_user_settings or {}),
    )
    _apply_defaults(settings, policy.compatibility_defaults)
    return settings


def _atomate2_selection_for(
    stage_type: StageType,
    theory: Theory | None,
) -> Atomate2Selection:
    override = _theory_stage_override_for(stage_type, theory)
    if override is not None and override.get("atomate2") is not None:
        return override["atomate2"]
    return stage_definition(stage_type).atomate2


def _theory_stage_override_for(
    stage_type: StageType,
    theory: Theory | None,
) -> Mapping[str, Any] | None:
    if theory is None:
        return None
    return _THEORY_STAGE_OVERRIDES.get((theory, stage_type))


def _theory_stage_bmd_incar_amendments(override: Mapping[str, Any] | None) -> dict:
    if override is None:
        return {}
    payload = {
        "defaults": _copy_mapping(override["bmd_incar_defaults"]),
        "encut_floor": override["encut_floor"],
        "source": override.get("source", _STAGE_DEFINITION_SOURCE),
    }
    return _compact_payload(payload)


def _base_bmd_incar_amendments(definition: VaspStageDefinition) -> dict:
    policy = definition.incar_policy
    payload = {
        "executable_defaults": _copy_mapping(policy.executable_defaults),
        "explicit_user_preserved_defaults": _copy_mapping(
            policy.explicit_user_preserved_defaults
        ),
        "final_defaults": _copy_mapping(policy.final_defaults),
        "prep_defaults": _copy_mapping(policy.prep_defaults),
        "final_encut_floor": policy.final_encut_floor,
        "prep_encut_floor": policy.prep_encut_floor,
    }
    return _compact_payload(payload)


def _atomate2_description(selection: Atomate2Selection) -> dict:
    return {
        "input_set_generator": selection.input_set_generator,
        "maker": selection.maker,
        "generator_mode": selection.generator_mode,
        "extra_parameters": _copy_mapping(selection.extra_parameters),
    }


def _restart_description(policy: Mapping[str, Any]) -> dict:
    return {
        "requires_previous_stage": policy["requires_previous_stage"],
        "incar_amendments": _copy_mapping(policy["incar_amendments"]),
        "description": policy["description"],
    }


def _kpoints_description(policy: Mapping[str, Any], atomate2: Atomate2Selection) -> dict:
    parameters = _copy_mapping(policy["default_parameters"])
    for key in _KPOINT_EXTRA_PARAMETER_KEYS:
        if key in atomate2.extra_parameters:
            parameters[key] = deepcopy(atomate2.extra_parameters[key])
    return {
        "mode": atomate2.generator_mode or policy["mode"],
        "default_parameters": parameters,
        "description": policy["description"],
    }


def _copy_mapping(mapping: Mapping[str, Any]) -> dict:
    return deepcopy(dict(mapping))


def _compact_payload(payload: Mapping[str, Any]) -> dict:
    return {key: value for key, value in payload.items() if value not in ({}, None)}


def _apply_defaults(settings: dict, defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        settings.setdefault(key, deepcopy(value))


def _apply_explicit_user_preserved_defaults(
    settings: dict,
    defaults: Mapping[str, Any],
    explicit_user_settings: Mapping[str, Any],
) -> None:
    for key, value in defaults.items():
        if key not in explicit_user_settings:
            settings[key] = deepcopy(value)


def _apply_conditional_tetrahedron_sigma(settings: dict) -> None:
    if settings.get("ISMEAR", -5) == -5:
        settings.setdefault("SIGMA", None)


def _apply_encut_floor(settings: dict, floor: int | None) -> None:
    if floor is None:
        return
    try:
        encut_now = int(float(settings.get("ENCUT", 0)))
    except Exception:
        encut_now = 0
    settings["ENCUT"] = max(encut_now, floor)
