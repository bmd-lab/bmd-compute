from __future__ import annotations

import hashlib
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping



"""
BMD Compute automatic DFT+U policy, version 1.

Trigger: the pinned pymatgen/Materials Project GGA+U rule (O or F is the most
electronegative element and an element with a non-zero MP U value is present),
narrowed by one BMD gate: automatic +U is suppressed only when every
charge-balanced oxidation-state guess from pymatgen places every triggering
element at d0. The gate is decided for the compound as a whole, may only
suppress +U, and falls back to the MP rule when pymatgen offers no guess.

Parameters: once triggered, L/U/J and LDAUTYPE are read unchanged from the
pinned pymatgen ``MPRelaxSet`` configuration (the table atomate2 copies into its
default input sets). BMD Compute does not define its own U values.

The resolved parameters are frozen into the stage options at preparation, so
preview, runtime and the input reference generate +U from the same recorded
values, and runtime refuses to run if the generated INCAR disagrees.
"""


POLICY_ID = "bmd_compute.dft_u"
POLICY_VERSION = 1
CONSIDERATION_ID = "dftu.mp_oxide_fluoride"
RULE_ID = "pymatgen.MPRelaxSet.most_electronegative_oxide_fluoride"
FROZEN_OPTION_KEY = "dft_u"
PARAMETER_SOURCE = "pymatgen.io.vasp.sets.MPRelaxSet (MPRelaxSet.yaml INCAR LDAU*)"

DECISION_APPLY = "apply"
DECISION_SUPPRESS = "suppress"
DECISION_NOT_TRIGGERED = "not_triggered"

GATE_NOT_EVALUATED = "not_evaluated"
GATE_D_ELECTRONS_PRESENT = "d_electrons_present"
GATE_ALL_D0 = "all_d0"
GATE_UNAVAILABLE = "oxidation_states_unavailable"

_HUBBARD_KEYS = ("LDAU", "LDAUTYPE", "LDAUU", "LDAUL", "LDAUJ")
_D0_TOLERANCE = 1e-9
_VALUE_TOLERANCE = 1e-9


class DftUFreezeError(ValueError):
    """Generated +U settings disagree with the frozen BMD parameters."""


def mp_hubbard_table() -> dict[str, Any]:
    """Return the pinned pymatgen MPRelaxSet +U settings (a copy)."""

    from pymatgen.io.vasp.sets import MPRelaxSet

    incar = MPRelaxSet.CONFIG["INCAR"]
    return {key: deepcopy(incar[key]) for key in _HUBBARD_KEYS}


def _anion_keys(table: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(key for key, value in table["LDAUU"].items() if isinstance(value, Mapping))
    )


def _element_symbol(element) -> str:
    return str(getattr(element, "symbol", element))


def _valence_s_plus_d(symbol: str) -> int:
    # Same electron count pymatgen uses for crystal-field d occupancy
    # (Species.get_crystal_field_spin): outer s + d electrons of the neutral atom.
    from pymatgen.core import Element

    structure = Element(symbol).full_electronic_structure
    d_shells = [shell for shell in structure if shell[1] == "d"]
    if not d_shells:
        raise ValueError(f"{symbol} has no d shell")
    n_d, _, d_count = d_shells[-1]
    s_count = sum(count for n, orbital, count in structure if orbital == "s" and n == n_d + 1)
    return int(d_count + s_count)


def evaluate_dft_u(structure) -> dict[str, Any]:
    """Evaluate the v1 automatic DFT+U rule for a parsed structure.

    Returns a JSON-safe record of the trigger inputs, oxidation-state guesses,
    derived d counts, gate result and decision.
    """

    table = mp_hubbard_table()
    composition = getattr(structure, "composition", None)
    elements = (
        sorted(
            (element for element in composition.elements if composition[element] > 0),
            key=lambda element: element.X,
        )
        if composition is not None
        else []
    )
    symbols = sorted({_element_symbol(element) for element in elements})
    most_electronegative = _element_symbol(elements[-1]) if elements else None
    anion_keys = _anion_keys(table)
    anion = most_electronegative if most_electronegative in anion_keys else None
    triggering = (
        sorted(symbol for symbol in symbols if float(table["LDAUU"][anion].get(symbol, 0)) > 0)
        if anion
        else []
    )
    record: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "consideration_id": CONSIDERATION_ID,
        "rule_id": RULE_ID,
        "elements": symbols,
        "most_electronegative_element": most_electronegative,
        "mp_anion_basis": list(anion_keys),
        "deciding_anion": anion,
        "triggering_elements": triggering,
        "mp_rule_triggered": bool(anion and triggering),
        "oxidation_state_guesses": [],
        "d_counts": [],
        "gate": GATE_NOT_EVALUATED,
        "gate_reason": "The pymatgen/Materials Project oxide/fluoride rule did not trigger.",
        "decision": DECISION_NOT_TRIGGERED,
    }
    if not record["mp_rule_triggered"]:
        return record

    try:
        guesses = composition.reduced_composition.oxi_state_guesses()
    except Exception as exc:  # pymatgen may refuse unusual compositions
        guesses = ()
        record["oxidation_state_error"] = f"{type(exc).__name__}: {exc}"
    record["oxidation_state_guesses"] = [
        {str(symbol): float(value) for symbol, value in sorted(dict(guess).items())}
        for guess in guesses
    ]
    record["d_counts"] = [
        {
            symbol: round(_valence_s_plus_d(symbol) - float(guess[symbol]), 6)
            for symbol in triggering
        }
        for guess in record["oxidation_state_guesses"]
    ]

    if not record["d_counts"]:
        record["gate"] = GATE_UNAVAILABLE
        record["gate_reason"] = (
            "pymatgen found no charge-balanced oxidation-state assignment, so the "
            "d0 gate could not be evaluated; the Materials Project rule applies."
        )
        record["decision"] = DECISION_APPLY
    elif all(
        all(abs(value) <= _D0_TOLERANCE for value in counts.values())
        for counts in record["d_counts"]
    ):
        record["gate"] = GATE_ALL_D0
        record["gate_reason"] = (
            "Every charge-balanced oxidation-state assignment places every "
            "triggering element at d0, so automatic DFT+U is suppressed."
        )
        record["decision"] = DECISION_SUPPRESS
    else:
        record["gate"] = GATE_D_ELECTRONS_PRESENT
        record["gate_reason"] = (
            "At least one charge-balanced oxidation-state assignment leaves a "
            "triggering element with d electrons."
        )
        record["decision"] = DECISION_APPLY

    if record["decision"] == DECISION_APPLY:
        record["parameters"] = frozen_parameters(symbols, anion, table=table)
    return record


def frozen_parameters(symbols, anion: str, *, table: Mapping[str, Any] | None = None) -> dict:
    """Frozen per-element L/U/J and LDAUTYPE read unchanged from the MP table."""

    table = table or mp_hubbard_table()
    return {
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "parameter_source": PARAMETER_SOURCE,
        "deciding_anion": anion,
        "LDAUTYPE": table["LDAUTYPE"],
        "species": {
            symbol: {
                "L": table["LDAUL"][anion].get(symbol, 0),
                "U": table["LDAUU"][anion].get(symbol, 0),
                "J": table["LDAUJ"][anion].get(symbol, 0),
            }
            for symbol in sorted(symbols)
        },
    }


def frozen_user_incar(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """INCAR settings that make pymatgen write exactly the frozen parameters."""

    species = frozen["species"]
    return {
        "LDAU": True,
        "LDAUTYPE": frozen["LDAUTYPE"],
        "LDAUU": {symbol: values["U"] for symbol, values in species.items()},
        "LDAUL": {symbol: values["L"] for symbol, values in species.items()},
        "LDAUJ": {symbol: values["J"] for symbol, values in species.items()},
    }


def stage_frozen_dft_u(stage) -> dict[str, Any] | None:
    value = dict(getattr(stage, "options", None) or {}).get(FROZEN_OPTION_KEY)
    return dict(value) if isinstance(value, Mapping) else None


def generated_dft_u_settings(input_set) -> dict[str, Any]:
    """Record the +U settings actually present in a generated input set."""

    incar = _input_file(input_set, "INCAR", "incar")
    poscar = _input_file(input_set, "POSCAR", "poscar")
    return {
        "poscar_symbols": list(poscar.site_symbols),
        "potcar_symbols": _potcar_symbols(input_set),
        "LDAU": incar.get("LDAU"),
        "LDAUTYPE": incar.get("LDAUTYPE"),
        "LDAUL": _as_list(incar.get("LDAUL")),
        "LDAUU": _as_list(incar.get("LDAUU")),
        "LDAUJ": _as_list(incar.get("LDAUJ")),
        "LMAXMIX": incar.get("LMAXMIX"),
    }


def verify_generated_dft_u(input_set, frozen: Mapping[str, Any], expected=None) -> None:
    """Raise DftUFreezeError unless the generated INCAR matches the frozen values."""

    actual = generated_dft_u_settings(input_set)
    problems = []
    species = frozen["species"]
    if actual["LDAU"] is not True:
        problems.append(f"LDAU is {actual['LDAU']!r}, expected True")
    if actual["LDAUTYPE"] != frozen["LDAUTYPE"]:
        problems.append(f"LDAUTYPE is {actual['LDAUTYPE']!r}, expected {frozen['LDAUTYPE']!r}")
    for key, field in (("LDAUU", "U"), ("LDAUL", "L"), ("LDAUJ", "J")):
        try:
            wanted = [species[symbol][field] for symbol in actual["poscar_symbols"]]
        except KeyError as exc:
            problems.append(f"no frozen {field} value for species {exc.args[0]!r}")
            continue
        if not _same_numbers(actual[key], wanted):
            problems.append(f"{key} is {actual[key]!r}, expected {wanted!r}")
    for key in (
        "poscar_symbols",
        "potcar_symbols",
        "LDAU",
        "LDAUTYPE",
        "LDAUL",
        "LDAUU",
        "LDAUJ",
        "LMAXMIX",
    ):
        if not expected or key not in expected:
            continue
        wanted = expected[key]
        if key in ("LDAUL", "LDAUU", "LDAUJ"):
            matches = _same_numbers(actual[key], wanted or [])
        else:
            matches = actual[key] == wanted
        if not matches:
            problems.append(
                f"{key} is {actual[key]!r}, but preparation recorded {wanted!r}"
            )
    if problems:
        raise DftUFreezeError(
            "Generated DFT+U settings disagree with the frozen BMD Compute "
            "parameters: " + "; ".join(problems)
        )


def parameter_source_record() -> dict[str, Any]:
    """Versions and checksum identifying the pinned +U table (preparation-time)."""

    from importlib import metadata

    import pymatgen.io.vasp as pymatgen_vasp

    path = Path(pymatgen_vasp.__file__).resolve().parent / "MPRelaxSet.yaml"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    packages = {}
    for name in ("atomate2", "pymatgen", "pymatgen-core"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        from atomate2.vasp.sets.base import _BASE_VASP_SET

        atomate2_table = {key: _BASE_VASP_SET["INCAR"].get(key) for key in _HUBBARD_KEYS}
        atomate2_matches = atomate2_table == mp_hubbard_table()
    except Exception:
        atomate2_matches = None
    return {
        "parameter_source": PARAMETER_SOURCE,
        "mprelaxset_yaml": "pymatgen/io/vasp/MPRelaxSet.yaml",
        "mprelaxset_yaml_sha256": digest,
        "generator_config": "atomate2.vasp.sets.base._BASE_VASP_SET",
        "atomate2_table_matches_mprelaxset": atomate2_matches,
        "packages": packages,
    }


def frozen_dft_u_generator_class(base_class):
    """Return the verifying subclass of an atomate2 input-set generator."""

    return _frozen_generator_class_for(base_class)


@lru_cache(maxsize=None)
def _frozen_generator_class_for(base_class):
    from dataclasses import dataclass, field

    @dataclass
    class FrozenDftUGenerator(base_class):
        bmd_frozen_dft_u: dict = field(default_factory=dict)

        def get_input_set(self, *args: Any, **kwargs: Any):
            input_set = super().get_input_set(*args, **kwargs)
            frozen = dict(self.bmd_frozen_dft_u or {})
            verify_generated_dft_u(
                input_set,
                frozen["parameters"],
                frozen.get("expected_generated"),
            )
            return input_set

    name = f"FrozenDftU{base_class.__name__}"
    FrozenDftUGenerator.__module__ = __name__
    FrozenDftUGenerator.__name__ = name
    FrozenDftUGenerator.__qualname__ = name
    return FrozenDftUGenerator


def __getattr__(name: str):
    # Resolve serialized ``@class`` names without importing atomate2 eagerly.
    prefix = "FrozenDftU"
    if name.startswith(prefix):
        import atomate2.vasp.sets.core as core

        base = getattr(core, name[len(prefix):], None)
        if base is not None:
            return frozen_dft_u_generator_class(base)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _input_file(input_set, key: str, attribute: str):
    try:
        if key in input_set:
            return input_set[key]
    except TypeError:
        pass
    return getattr(input_set, attribute)


def _potcar_symbols(input_set) -> list[str]:
    for key, attribute in (("POTCAR", "potcar"), ("POTCAR.spec", None)):
        try:
            value = _input_file(input_set, key, attribute) if attribute else input_set[key]
        except (AttributeError, KeyError, TypeError):
            continue
        symbols = getattr(value, "symbols", None)
        if symbols is not None:
            return [str(symbol) for symbol in symbols]
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
    return []


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _same_numbers(left, right) -> bool:
    if left is None or len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= _VALUE_TOLERANCE for a, b in zip(left, right))


__all__ = [
    "CONSIDERATION_ID",
    "DECISION_APPLY",
    "DECISION_NOT_TRIGGERED",
    "DECISION_SUPPRESS",
    "DftUFreezeError",
    "FROZEN_OPTION_KEY",
    "GATE_ALL_D0",
    "GATE_D_ELECTRONS_PRESENT",
    "GATE_NOT_EVALUATED",
    "GATE_UNAVAILABLE",
    "PARAMETER_SOURCE",
    "POLICY_ID",
    "POLICY_VERSION",
    "RULE_ID",
    "evaluate_dft_u",
    "frozen_dft_u_generator_class",
    "frozen_parameters",
    "frozen_user_incar",
    "generated_dft_u_settings",
    "mp_hubbard_table",
    "parameter_source_record",
    "prepared_dft_u_record",
    "stage_frozen_dft_u",
    "verify_generated_dft_u",
]


def prepared_dft_u_record(
    evaluation: Mapping[str, Any] | None,
    structure,
    workflow_spec,
    *,
    resources=None,
    potcar_functional: str = "PBE_64",
) -> dict[str, Any] | None:
    """Freeze the +U record at preparation: evaluation, source and generated stages.

    ``generated_stages`` records, for every stage carrying frozen parameters,
    the +U INCAR tags, LMAXMIX, POSCAR species order and POTCAR symbols that the
    pinned generator produced. Runtime compares its own generated inputs with
    these values and stops on any difference.
    """

    if evaluation is None:
        return None
    record = dict(evaluation)
    record["parameter_source_record"] = parameter_source_record()
    stages = list(getattr(workflow_spec, "stages", ()) or ())
    if not any(stage_frozen_dft_u(stage) for stage in stages):
        record["generated_stages"] = []
        return record

    from backend.generated_inputs import generated_input_stage_previews

    previews = generated_input_stage_previews(
        structure,
        workflow_spec,
        resources=resources,
        potcar_functional=potcar_functional,
    )
    generated = []
    for preview in previews:
        stage = preview["stage_spec"]
        if stage_frozen_dft_u(stage) is None:
            continue
        settings = generated_dft_u_settings(preview["input_set"])
        generated.append(
            {
                "stage_index": int(preview["index"]),
                "stage_type": stage.stage_type.value,
                "theory": stage.theory.value,
                **settings,
            }
        )
    record["generated_stages"] = generated
    return record
