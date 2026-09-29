"""POTCAR identity record derived from BMD Compute's executable stage generators.

The POTCAR symbols Compute shows and records must be the ones the stage
generators that actually run will write. They are therefore read from the
same per-stage input sets used for generated-input previews and the
input-reference producer (``generated_input_stage_previews``), in POTCAR.spec
mode so no licensed POTCAR file is read. There is deliberately no separate
BMD POTCAR table and no fallback to another input set: when the executable
generators cannot be evaluated, the record says so instead of guessing.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


POTCAR_SYMBOL_SOURCE = "bmd_compute.executable_stage_generators"
POTCAR_RECORD_RESOLVED = "resolved"
POTCAR_RECORD_UNAVAILABLE = "unavailable"


class PotcarRecordError(ValueError):
    """Raised when a stage's POSCAR species and POTCAR symbols do not pair up."""


def input_set_potcar_symbols(input_set) -> list[str]:
    """POTCAR symbols of a generated input set, in POTCAR (= POSCAR species) order."""

    potcar = getattr(input_set, "potcar", None)
    symbols = getattr(potcar, "symbols", None)
    if symbols is not None:
        return [str(symbol) for symbol in symbols]
    return [line.strip() for line in str(potcar or "").splitlines() if line.strip()]


def input_set_poscar_species(input_set) -> list[str]:
    """POSCAR species blocks of a generated input set, in file order."""

    poscar = getattr(input_set, "poscar", None)
    return [str(symbol) for symbol in (getattr(poscar, "site_symbols", None) or [])]


def stage_potcar_entry(preview: Mapping[str, Any]) -> dict[str, Any]:
    """POTCAR identity for one generated stage preview."""

    input_set = preview["input_set"]
    stage = preview["stage_spec"]
    species = input_set_poscar_species(input_set)
    symbols = input_set_potcar_symbols(input_set)
    if not symbols or len(species) != len(symbols):
        raise PotcarRecordError(
            f"Stage {preview['index']} POSCAR species {species!r} do not pair with "
            f"POTCAR symbols {symbols!r}."
        )
    return {
        "stage_index": int(preview["index"]),
        "stage_type": stage.stage_type.value,
        "theory": stage.theory.value,
        "species": [
            {"species": name, "potcar_symbol": symbol}
            for name, symbol in zip(species, symbols)
        ],
        "symbols": list(symbols),
    }


def aggregate_stage_potcars(stages: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine per-stage POTCAR entries without inventing a single value.

    ``species``/``symbols`` are populated only when every stage uses the same
    species-to-POTCAR mapping; otherwise they are ``None`` and readers must use
    ``stages``.
    """

    stage_list = [dict(stage) for stage in stages]
    if not stage_list:
        raise PotcarRecordError("No executable stages were available for the POTCAR record.")
    first = stage_list[0]
    consistent = all(
        stage["species"] == first["species"] and stage["symbols"] == first["symbols"]
        for stage in stage_list[1:]
    )
    return {
        "status": POTCAR_RECORD_RESOLVED,
        "symbol_source": POTCAR_SYMBOL_SOURCE,
        "consistent_across_stages": consistent,
        "species": [dict(row) for row in first["species"]] if consistent else None,
        "symbols": list(first["symbols"]) if consistent else None,
        "stages": stage_list,
    }


def unavailable_potcar_record(reason: str) -> dict[str, Any]:
    return {
        "status": POTCAR_RECORD_UNAVAILABLE,
        "symbol_source": POTCAR_SYMBOL_SOURCE,
        "consistent_across_stages": None,
        "species": [],
        "symbols": [],
        "stages": [],
        "reason": reason,
    }


def executable_potcar_record(
    structure,
    workflow_spec,
    *,
    potcar_functional: str,
    resources=None,
) -> dict[str, Any]:
    """Resolve the POTCAR identity each stage of ``workflow_spec`` will use."""

    if structure is None:
        return unavailable_potcar_record("No structure was supplied.")
    try:
        from backend.generated_inputs import generated_input_stage_previews

        previews = generated_input_stage_previews(
            structure,
            workflow_spec,
            resources=resources,
            potcar_functional=potcar_functional,
        )
        return aggregate_stage_potcars(stage_potcar_entry(preview) for preview in previews)
    except Exception as exc:  # the record must never fall back to a guess
        return unavailable_potcar_record(
            "The executable stage generators could not be evaluated: "
            f"{type(exc).__name__}: {exc}"
        )


__all__ = [
    "POTCAR_RECORD_RESOLVED",
    "POTCAR_RECORD_UNAVAILABLE",
    "POTCAR_SYMBOL_SOURCE",
    "PotcarRecordError",
    "aggregate_stage_potcars",
    "executable_potcar_record",
    "input_set_poscar_species",
    "input_set_potcar_symbols",
    "stage_potcar_entry",
    "unavailable_potcar_record",
]
