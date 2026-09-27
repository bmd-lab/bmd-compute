from __future__ import annotations

from typing import Any, Mapping


DISPERSION_OPTION_KEY = "dispersion"
DISPERSION_METHOD_KEY = "method"
DEFAULT_DISPERSION_METHOD = "dftd3-bj"

_DISPERSION_METHODS = (
    {
        "value": "dftd3",
        "label": "DFT-D3",
        "incar_effect": {"IVDW": 11},
    },
    {
        "value": "dftd3-bj",
        "label": "DFT-D3(BJ)",
        "incar_effect": {"IVDW": 12},
    },
)

_METHOD_ALIASES = {
    "d3": "dftd3",
    "dft-d3": "dftd3",
    "dftd3": "dftd3",
    "d3-bj": "dftd3-bj",
    "d3bj": "dftd3-bj",
    "dft-d3-bj": "dftd3-bj",
    "dftd3-bj": "dftd3-bj",
    "dftd3bj": "dftd3-bj",
}


def dispersion_method_options(*, include_incar_effect: bool = False) -> tuple[dict[str, Any], ...]:
    options = []
    for method in _DISPERSION_METHODS:
        payload = {
            "value": method["value"],
            "label": method["label"],
        }
        if include_incar_effect:
            payload["incar_effect"] = dict(method["incar_effect"])
        options.append(payload)
    return tuple(options)


def normalize_dispersion_method(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return DEFAULT_DISPERSION_METHOD

    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "")
    try:
        return _METHOD_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(method["label"] for method in _DISPERSION_METHODS)
        raise ValueError(f"Unsupported dispersion correction {value!r}. Choose {supported}.") from exc


def dispersion_method_from_options(options: Mapping[str, Any] | None) -> str:
    values = dict(options or {})
    dispersion = values.get(DISPERSION_OPTION_KEY)
    if isinstance(dispersion, Mapping):
        return normalize_dispersion_method(dispersion.get(DISPERSION_METHOD_KEY))
    if isinstance(dispersion, str):
        return normalize_dispersion_method(dispersion)
    return DEFAULT_DISPERSION_METHOD


def dispersion_option_payload(method: str | None = None) -> dict[str, dict[str, str]]:
    return {
        DISPERSION_OPTION_KEY: {
            DISPERSION_METHOD_KEY: normalize_dispersion_method(method),
        }
    }


def dispersion_modifier_policy() -> dict[str, Any]:
    return {
        "modifier": "dispersion",
        "label": "Dispersion correction",
        "option_key": DISPERSION_OPTION_KEY,
        "method_key": DISPERSION_METHOD_KEY,
        "default_method": DEFAULT_DISPERSION_METHOD,
        "methods": list(dispersion_method_options(include_incar_effect=True)),
        "upstream_interface": {
            "atomate2_pymatgen_generator_keyword": "vdw",
        },
        "phase_1_support": {
            "theories": ["pbe"],
            "stage_types": ["relax", "static"],
            "blocked_with_modifiers": [],
            "blocked_terminal_stage_types": ["dos", "band_structure"],
        },
    }


__all__ = [
    "DEFAULT_DISPERSION_METHOD",
    "DISPERSION_METHOD_KEY",
    "DISPERSION_OPTION_KEY",
    "dispersion_method_from_options",
    "dispersion_method_options",
    "dispersion_modifier_policy",
    "dispersion_option_payload",
    "normalize_dispersion_method",
]
