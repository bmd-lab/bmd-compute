"""BMD Standard Geometry Relaxation Protocol v1 (``bmd_compute.geometry_relaxation/1``).

One numerical procedure supplies the equilibrium geometry for every managed
workflow that relaxes a structure (Relaxed Structure, Relaxed Energy,
Electronic DOS, Electronic Band Structure and Phonons). It is two
Geometry Optimisation stages with the same Hamiltonian, POTCARs and
plane-wave cutoff; the second starts from the first's structure and magnetic
moments with a newly constructed plane-wave basis (no WAVECAR or CHGCAR is
carried over):

* pass 1: EDIFF = 1e-6, EDIFFG = -0.01 eV/A;
* pass 2: EDIFF = 1e-7, EDIFFG = -0.001 eV/A.

Both passes use ISMEAR = 0, SIGMA = 0.20 eV, IBRION = 2, ISIF = 3 (ISIF = 2
for the explicit fixed-cell ``ions_only`` modifier), PREC = Accurate,
LREAL = False, LASPH = True, no ADDGRID (as the phonon displacement
calculations), the existing k-point density, and no Custodian
LargeSigmaHandler. Physical modifiers (spin, DFT+U, dispersion, theory) are
carried unchanged by the stages themselves.

Species-dependent plane-wave cutoff (``bmd_compute.plane_wave_cutoff/1``)::

    ENCUT = ceil(1.3 * max(ENMAX of the selected POTCARs) / 50 eV) * 50 eV

The POTCAR symbols are those Compute's executable input generators select for
the structure (read in POTCAR.spec mode, as ``potcar_record`` does); ENMAX
comes from the versioned metadata generated from the POWER library
(``potcar_enmax_metadata``). The resolved record is frozen into the stage
options of every Geometry Optimisation and phonon force stage at planning, so
previews, plan digests, preparation and the POWER run use exactly the same
value, and the run checks it against the POTCAR files VASP reads before any
stage starts. Missing or inconsistent metadata fails planning; no cutoff is
ever guessed.

The same resolved cutoff is used by the phonon displacement force
calculations that consume the final relaxation, so the relaxed structure is a
minimum of the same basis and Hamiltonian as the forces.

Custom workflows keep their explicit stages: an unmarked Custom Geometry
Optimisation keeps its single-stage convergence (pass-1 values, or pass-2
values when it feeds a phonon force stage, as before) and gains only the
species-dependent cutoff, ADDGRID and Custodian treatment above. The canonical
pair can be requested in a Custom workflow with the ``geometry_relaxation``
stage option (see :func:`protocol_option`).

Workflows recorded before this protocol carry no cutoff record and are built
exactly as before.
"""

from __future__ import annotations

import json
import math
from decimal import ROUND_CEILING, Decimal
from functools import lru_cache
from typing import Any, Mapping, Sequence

from backend.calculations.models import Modifier, StageSpec, StageType, WorkflowSpec
from backend.calculations.potcar_enmax_metadata import (
    METADATA_SCHEMA,
    METADATA_SCHEMA_VERSION,
    metadata_digest,
    metadata_path,
    potcar_identity,
    validate_metadata,
)

PROTOCOL_ID = "bmd_compute.geometry_relaxation"
PROTOCOL_VERSION = 1
PROTOCOL_OPTION_KEY = "geometry_relaxation"

CUTOFF_RULE_ID = "bmd_compute.plane_wave_cutoff"
CUTOFF_RULE_VERSION = 1
CUTOFF_OPTION_KEY = "plane_wave_cutoff"
CUTOFF_SAFETY_FACTOR = "1.3"
CUTOFF_INCREMENT_EV = 50

PASS_CONVERGENCE = {
    1: {"EDIFF": 1e-6, "EDIFFG": -0.01},
    2: {"EDIFF": 1e-7, "EDIFFG": -0.001},
}
# Settings shared by every relaxation pass. ``None`` removes the tag
# (ADDGRID is not used by the phonon displacement calculations either).
COMMON_RELAX_SETTINGS = {
    "ISMEAR": 0,
    "SIGMA": 0.2,
    "IBRION": 2,
    "ISIF": 3,
    "PREC": "Accurate",
    "LREAL": False,
    "LASPH": True,
    "ADDGRID": None,
}
from backend.calculations.custodian_policy import (  # noqa: E402
    RELAX_HANDLER_EXCLUSIONS as RELAX_CUSTODIAN_HANDLER_EXCLUSIONS,
)

_CUTOFF_STAGE_TYPES = frozenset({StageType.RELAX, StageType.PHONON_FORCES})


class PlaneWaveCutoffError(ValueError):
    """Invalid ENMAX input to the cutoff arithmetic."""


def _validation_error(message: str, *, suggestion: str, code: str, **diagnostic):
    from backend.calculations.registry import CalculationValidationError

    return CalculationValidationError(
        message,
        suggestion=suggestion,
        diagnostic={"code": code, **diagnostic},
    )


# --------------------------------------------------------------------------
# Protocol stages
# --------------------------------------------------------------------------


def protocol_option(pass_number: int) -> dict[str, Any]:
    """Stage options marking pass ``pass_number`` of the canonical relaxation."""

    return {PROTOCOL_OPTION_KEY: {"protocol": PROTOCOL_ID, "version": PROTOCOL_VERSION, "pass": int(pass_number)}}


def protocol_relaxation_stages(theory=None, modifiers=()) -> list[StageSpec]:
    """The two Geometry Optimisation stages of the canonical relaxation."""

    from backend.calculations.models import Theory

    theory = theory or Theory.PBE
    return [
        StageSpec(StageType.RELAX, theory, frozenset(modifiers), options=protocol_option(1)),
        StageSpec(StageType.RELAX, theory, frozenset(modifiers), options=protocol_option(2)),
    ]


def stage_protocol_pass(stage: StageSpec) -> int | None:
    value = dict(stage.options or {}).get(PROTOCOL_OPTION_KEY)
    if isinstance(value, Mapping):
        number = value.get("pass")
        return number if number in PASS_CONVERGENCE else None
    return None


def stage_cutoff_record(stage: StageSpec) -> dict[str, Any] | None:
    value = dict(stage.options or {}).get(CUTOFF_OPTION_KEY)
    return dict(value) if isinstance(value, Mapping) else None


def _feeds_phonon_forces(workflow: WorkflowSpec, index: int) -> bool:
    stages = workflow.stages
    return (
        stages[index].stage_type is StageType.RELAX
        and index + 1 < len(stages)
        and stages[index + 1].stage_type is StageType.PHONON_FORCES
    )


def relaxation_pass_role(workflow: WorkflowSpec, index: int) -> int:
    """Convergence role of a Geometry Optimisation stage (1 or 2).

    A protocol stage uses its marked pass. An unmarked (Custom) stage keeps
    its single-stage convergence: pass-2 values when it feeds a phonon force
    stage (the former phonon-grade relaxation), pass-1 values otherwise.
    """

    marked = stage_protocol_pass(workflow.stages[index])
    if marked is not None:
        return marked
    return 2 if _feeds_phonon_forces(workflow, index) else 1


def relaxation_incar_settings(workflow: WorkflowSpec, index: int) -> dict[str, Any]:
    """INCAR settings of a cutoff-frozen Geometry Optimisation stage."""

    stage = workflow.stages[index]
    record = stage_cutoff_record(stage)
    if record is None:
        return {}
    settings = dict(COMMON_RELAX_SETTINGS)
    settings.update(PASS_CONVERGENCE[relaxation_pass_role(workflow, index)])
    settings["ENCUT"] = int(record["ENCUT"])
    return settings


def downstream_cutoff_floor(workflow: WorkflowSpec, index: int) -> int | None:
    """Lowest ENCUT a stage downstream of a cutoff-frozen relaxation may use.

    * A Static Energy stage never uses a lower cutoff than the Geometry
      Optimisation that produced its structure (the nearest preceding relax):
      its own floor (620 eV) still applies, so it runs at
      ``max(existing static cutoff, resolved relaxation cutoff)``.
    * A PBE DOS or Band Structure stage that restarts from the preceding
      static's charge density (``ICHARG = 11``) keeps that static's cutoff, so
      the density is read on the basis it was computed with.

    Other stages (HSE06 DOS and Band Structure, which are self-consistent and
    take only the structure and NBANDS from the static) are unaffected, as are
    workflows recorded before the protocol. Returns ``None`` when no floor applies.
    """

    from backend.calculations.vasp_stage_definitions import stage_reads_previous_charge_density

    stages = workflow.stages
    stage = stages[index]
    if stage.stage_type is StageType.STATIC:
        for previous in reversed(stages[:index]):
            if previous.stage_type is StageType.RELAX:
                record = stage_cutoff_record(previous)
                return int(record["ENCUT"]) if record is not None else None
        return None
    if (
        index > 0
        and stages[index - 1].stage_type is StageType.STATIC
        and stage_reads_previous_charge_density(stage.stage_type, stage.theory)
    ):
        return downstream_cutoff_floor(workflow, index - 1)
    return None


def relaxation_stage_payload(workflow: WorkflowSpec, index: int) -> dict[str, Any] | None:
    """Methodology of one cutoff-frozen relaxation stage (plan digest, previews)."""

    stage = workflow.stages[index]
    if stage.stage_type is not StageType.RELAX or stage_cutoff_record(stage) is None:
        return None
    marked = stage_protocol_pass(stage)
    return {
        "protocol": {"id": PROTOCOL_ID, "version": PROTOCOL_VERSION} if marked is not None else None,
        "pass": marked,
        "convergence_role": relaxation_pass_role(workflow, index),
        "convergence": dict(PASS_CONVERGENCE[relaxation_pass_role(workflow, index)]),
        "common_settings": {key: value for key, value in COMMON_RELAX_SETTINGS.items()},
        "plane_wave_cutoff_rule": cutoff_rule_payload(),
        "custodian_excluded_handlers": list(RELAX_CUSTODIAN_HANDLER_EXCLUSIONS),
        "basis": "new plane-wave basis per stage (no WAVECAR or CHGCAR carried over)",
        "magnetic_moments": "inherited from the previous stage's relaxed structure",
    }


def protocol_payload() -> dict[str, Any]:
    """The protocol definition (documentation and methodology records)."""

    return {
        "id": PROTOCOL_ID,
        "version": PROTOCOL_VERSION,
        "passes": {str(number): dict(values) for number, values in PASS_CONVERGENCE.items()},
        "common_settings": dict(COMMON_RELAX_SETTINGS),
        "plane_wave_cutoff_rule": cutoff_rule_payload(),
        "custodian_excluded_handlers": list(RELAX_CUSTODIAN_HANDLER_EXCLUSIONS),
        "stress": "diagnostic only (not a failure criterion)",
    }


def without_relaxation_options(workflow: WorkflowSpec) -> WorkflowSpec:
    """``workflow`` without protocol marks and cutoff records (display and naming only)."""

    return WorkflowSpec(
        stages=[
            StageSpec(
                stage.stage_type,
                stage.theory,
                stage.modifiers,
                label=stage.label,
                options={
                    key: value
                    for key, value in dict(stage.options or {}).items()
                    if key not in (PROTOCOL_OPTION_KEY, CUTOFF_OPTION_KEY)
                },
            )
            for stage in workflow.stages
        ],
        label=workflow.label,
        recipe=workflow.recipe,
    )


# --------------------------------------------------------------------------
# Species-dependent plane-wave cutoff
# --------------------------------------------------------------------------


def cutoff_rule_payload() -> dict[str, Any]:
    return {
        "id": CUTOFF_RULE_ID,
        "version": CUTOFF_RULE_VERSION,
        "safety_factor": float(CUTOFF_SAFETY_FACTOR),
        "increment_ev": CUTOFF_INCREMENT_EV,
        "rule": "ENCUT = ceil(safety_factor * max(ENMAX) / increment) * increment",
    }


def resolved_encut(enmax_values: Sequence[float]) -> int:
    """``ceil(1.3 * max(ENMAX) / 50) * 50`` in exact decimal arithmetic."""

    values = [float(value) for value in enmax_values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise PlaneWaveCutoffError("ENMAX values must be finite and positive.")
    scaled = Decimal(repr(max(values))) * Decimal(CUTOFF_SAFETY_FACTOR)
    steps = (scaled / Decimal(CUTOFF_INCREMENT_EV)).to_integral_value(rounding=ROUND_CEILING)
    return int(steps) * CUTOFF_INCREMENT_EV


def load_enmax_metadata(functional: str) -> dict[str, Any]:
    """The committed ENMAX metadata for ``functional``; fails clearly if absent."""

    path = metadata_path(functional)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise _validation_error(
            f"BMD Compute has no POTCAR ENMAX metadata for {functional}, so it cannot "
            "resolve the species-dependent plane-wave cutoff.",
            suggestion=(
                "Generate the metadata from the POWER POTCAR library with "
                "backend/calculations/potcar_enmax_metadata.py and deploy it."
            ),
            code="potcar_enmax_metadata_unavailable",
            functional=functional,
        ) from None
    return _parsed_metadata(text, functional)


@lru_cache(maxsize=8)
def _parsed_metadata(text: str, functional: str) -> dict[str, Any]:
    try:
        metadata = json.loads(text)
    except ValueError:
        metadata = None
    problems = validate_metadata(metadata, functional=functional)
    if problems:
        raise _validation_error(
            f"The POTCAR ENMAX metadata for {functional} is invalid: " + "; ".join(problems) + ".",
            suggestion="Regenerate the metadata from the POWER POTCAR library.",
            code="potcar_enmax_metadata_invalid",
            functional=functional,
        )
    return metadata


def resolve_cutoff_record(symbols: Sequence[str], *, functional: str) -> dict[str, Any]:
    """The frozen cutoff record for the POTCAR symbols a generator selected."""

    metadata = load_enmax_metadata(functional)
    entries = metadata["potcars"]
    missing = [symbol for symbol in symbols if symbol not in entries]
    if not symbols or missing:
        raise _validation_error(
            "BMD Compute has no ENMAX for the POTCAR dataset(s) "
            + ", ".join(missing or ["(none selected)"])
            + f" ({functional}), so it cannot resolve the plane-wave cutoff.",
            suggestion="Regenerate the POTCAR ENMAX metadata from the POWER library.",
            code="potcar_enmax_missing",
            functional=functional,
            missing_symbols=list(missing),
        )
    potcars = [
        {
            "symbol": symbol,
            "TITEL": entries[symbol]["TITEL"],
            "ENMAX": float(entries[symbol]["ENMAX"]),
            "sha256": entries[symbol]["sha256"],
        }
        for symbol in symbols
    ]
    max_enmax = max(item["ENMAX"] for item in potcars)
    return {
        "rule": {"id": CUTOFF_RULE_ID, "version": CUTOFF_RULE_VERSION},
        "functional": functional,
        "safety_factor": float(CUTOFF_SAFETY_FACTOR),
        "increment_ev": CUTOFF_INCREMENT_EV,
        "potcars": potcars,
        "max_enmax_ev": max_enmax,
        "ENCUT": resolved_encut([item["ENMAX"] for item in potcars]),
        "metadata": {
            "schema": METADATA_SCHEMA,
            "schema_version": METADATA_SCHEMA_VERSION,
            "digest": metadata_digest(metadata),
            "synthetic": bool(metadata.get("synthetic")),
        },
    }


def _stage_potcar_symbols(structure, stage: StageSpec, *, potcar_functional: str) -> list[str]:
    """POTCAR symbols the stage's executable generator selects (POTCAR.spec)."""

    from backend.calculations.potcar_record import input_set_potcar_symbols

    if stage.stage_type is StageType.PHONON_FORCES:
        from backend.workflows import build_phonon_force_input_set_generator_for_stage

        generator = build_phonon_force_input_set_generator_for_stage(stage, potcar_functional=potcar_functional)
    else:
        from backend.workflows import (
            build_relax_input_set_generator,
            dispersion_method_for_stage,
            frozen_dft_u_for_stage,
        )

        generator = build_relax_input_set_generator(
            structure,
            isif=2 if Modifier.IONS_ONLY in stage.modifiers else None,
            theory=stage.theory,
            modifiers=stage.modifiers,
            potcar_functional=potcar_functional,
            dispersion_method=dispersion_method_for_stage(stage),
            frozen_dft_u=frozen_dft_u_for_stage(stage),
        )
    return input_set_potcar_symbols(generator.get_input_set(structure, potcar_spec=True))


def freeze_plane_wave_cutoffs(
    structure,
    workflow: WorkflowSpec,
    *,
    potcar_functional: str,
    only_missing: bool = False,
) -> WorkflowSpec:
    """Return ``workflow`` with the cutoff record on every relax/phonon stage.

    ``only_missing`` keeps records a workflow already carries (for read-only
    tools given a recorded workflow); planning always resolves them afresh.
    """

    stages = []
    changed = False
    for stage in workflow.stages:
        if stage.stage_type not in _CUTOFF_STAGE_TYPES or (
            only_missing and stage_cutoff_record(stage) is not None
        ):
            stages.append(stage)
            continue
        symbols = _stage_potcar_symbols(structure, stage, potcar_functional=potcar_functional)
        options = dict(stage.options or {})
        options[CUTOFF_OPTION_KEY] = resolve_cutoff_record(symbols, functional=potcar_functional)
        stages.append(
            StageSpec(stage.stage_type, stage.theory, stage.modifiers, label=stage.label, options=options)
        )
        changed = True
    if not changed:
        return workflow
    return WorkflowSpec(stages=stages, label=workflow.label, recipe=workflow.recipe)


def require_cutoff_functional(workflow: WorkflowSpec, potcar_functional: str) -> None:
    """Refuse to build stages whose cutoff is missing or was resolved for other POTCARs.

    A BMD Standard Geometry Relaxation stage is only ever built with the cutoff
    resolved at planning; it never falls back to a fixed ENCUT.
    """

    for index, stage in enumerate(workflow.stages):
        record = stage_cutoff_record(stage)
        if record is None and stage_protocol_pass(stage) is not None:
            raise _validation_error(
                f"Stage {index + 1} is part of the BMD Standard Geometry Relaxation but has "
                "no resolved plane-wave cutoff.",
                suggestion="Plan the calculation through BMD Compute, then try again.",
                code="plane_wave_cutoff_missing",
            )
        if record is not None and record.get("functional") != potcar_functional:
            raise _validation_error(
                f"Stage {index + 1}'s plane-wave cutoff was resolved for "
                f"{record.get('functional')} POTCARs, not {potcar_functional}.",
                suggestion="Rebuild the calculation, then try again.",
                code="plane_wave_cutoff_functional_mismatch",
            )


# --------------------------------------------------------------------------
# Validation (called by backend.calculations.registry)
# --------------------------------------------------------------------------

_PROTOCOL_SUGGESTION = (
    "Use the BMD Geometry Relaxation as two consecutive Geometry Optimisation "
    "stages (pass 1, then pass 2) with identical settings, or remove the option."
)


def validate_stage_options(stage: StageSpec) -> None:
    options = dict(stage.options or {})
    if PROTOCOL_OPTION_KEY in options:
        value = options[PROTOCOL_OPTION_KEY]
        if (
            stage.stage_type is not StageType.RELAX
            or not isinstance(value, Mapping)
            or set(value) != {"protocol", "version", "pass"}
            or value.get("protocol") != PROTOCOL_ID
            or value.get("version") != PROTOCOL_VERSION
            or isinstance(value.get("pass"), bool)
            or value.get("pass") not in PASS_CONVERGENCE
        ):
            raise _validation_error(
                "The BMD Geometry Relaxation option is only valid as "
                f"{{'protocol': '{PROTOCOL_ID}', 'version': {PROTOCOL_VERSION}, 'pass': 1 or 2}} "
                "on a Geometry Optimisation stage.",
                suggestion=_PROTOCOL_SUGGESTION,
                code="geometry_relaxation_option_invalid",
            )
    if CUTOFF_OPTION_KEY in options:
        _validate_cutoff_record(stage, options[CUTOFF_OPTION_KEY])


def _validate_cutoff_record(stage: StageSpec, record: Any) -> None:
    def fail(detail: str):
        return _validation_error(
            f"The plane-wave cutoff record of this stage is invalid ({detail}).",
            suggestion="Rebuild the calculation, then try again.",
            code="plane_wave_cutoff_record_invalid",
        )

    if stage.stage_type not in _CUTOFF_STAGE_TYPES:
        raise fail("only Geometry Optimisation and phonon force stages carry one")
    if not isinstance(record, Mapping):
        raise fail("not an object")
    if record.get("rule") != {"id": CUTOFF_RULE_ID, "version": CUTOFF_RULE_VERSION}:
        raise fail("unknown rule")
    if record.get("safety_factor") != float(CUTOFF_SAFETY_FACTOR) or record.get("increment_ev") != CUTOFF_INCREMENT_EV:
        raise fail("rule parameters differ")
    potcars = record.get("potcars")
    if not isinstance(potcars, list) or not potcars:
        raise fail("no POTCAR datasets")
    enmax = []
    for item in potcars:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("symbol"), str)
            or isinstance(item.get("ENMAX"), bool)
            or not isinstance(item.get("ENMAX"), (int, float))
        ):
            raise fail("malformed POTCAR entry")
        enmax.append(float(item["ENMAX"]))
    try:
        expected = resolved_encut(enmax)
    except PlaneWaveCutoffError:
        raise fail("non-positive ENMAX") from None
    if record.get("ENCUT") != expected or record.get("max_enmax_ev") != max(enmax):
        raise fail("ENCUT does not follow from the recorded ENMAX values")


def validate_workflow(workflow: WorkflowSpec) -> None:
    """Protocol pairing and cutoff consistency across stages."""

    stages = workflow.stages
    for index, stage in enumerate(stages):
        marked = stage_protocol_pass(stage)
        if marked == 1:
            following = stages[index + 1] if index + 1 < len(stages) else None
            if following is None or stage_protocol_pass(following) != 2 or not _same_relaxation(stage, following):
                raise _validation_error(
                    f"Stage {index + 1} is pass 1 of the BMD Geometry Relaxation but is not "
                    "directly followed by its pass 2 with identical settings.",
                    suggestion=_PROTOCOL_SUGGESTION,
                    code="geometry_relaxation_pairing_invalid",
                )
        elif marked == 2:
            previous = stages[index - 1] if index > 0 else None
            if previous is None or stage_protocol_pass(previous) != 1:
                raise _validation_error(
                    f"Stage {index + 1} is pass 2 of the BMD Geometry Relaxation but does not "
                    "directly follow its pass 1.",
                    suggestion=_PROTOCOL_SUGGESTION,
                    code="geometry_relaxation_pairing_invalid",
                )
        if stage.stage_type is StageType.PHONON_FORCES and index > 0:
            relax_record = stage_cutoff_record(stages[index - 1])
            force_record = stage_cutoff_record(stage)
            if relax_record != force_record:
                raise _validation_error(
                    "The phonon force calculations must use the same plane-wave cutoff "
                    "record as the Geometry Optimisation that produced their structure.",
                    suggestion="Rebuild the calculation, then try again.",
                    code="phonon_cutoff_mismatch",
                )


def _same_relaxation(first: StageSpec, second: StageSpec) -> bool:
    def comparable(stage: StageSpec):
        options = {key: value for key, value in dict(stage.options or {}).items() if key != PROTOCOL_OPTION_KEY}
        return (stage.theory, frozenset(stage.modifiers), json.dumps(options, sort_keys=True, default=str))

    return comparable(first) == comparable(second)


# --------------------------------------------------------------------------
# Run time: the recorded ENMAX against the POTCAR files VASP will read
# --------------------------------------------------------------------------


def recorded_cutoff_potcars(flow_spec: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Every POTCAR identity recorded by the cutoff records of ``flow_spec``."""

    workflow = dict((flow_spec or {}).get("workflow_spec") or {})
    recorded: dict[str, dict[str, Any]] = {}
    for stage in workflow.get("stages") or ():
        record = dict((stage or {}).get("options") or {}).get(CUTOFF_OPTION_KEY)
        if not isinstance(record, Mapping):
            continue
        for item in record.get("potcars") or ():
            if isinstance(item, Mapping) and isinstance(item.get("symbol"), str):
                recorded[item["symbol"]] = {
                    "functional": record.get("functional"),
                    "TITEL": item.get("TITEL"),
                    "ENMAX": item.get("ENMAX"),
                    "sha256": item.get("sha256"),
                }
    return recorded


def _default_potcar_loader(symbol: str, functional: str):
    from pymatgen.io.vasp.inputs import PotcarSingle

    return PotcarSingle.from_symbol_and_functional(symbol, functional)


def runtime_potcar_check(flow_spec: Mapping[str, Any] | None, *, potcar_loader=None) -> dict[str, Any]:
    """Compare the recorded POTCAR identities with the datasets VASP will read.

    Returns ``{"status": "not_applicable" | "passed" | "failed", "potcars":
    [...], "problems": [...]}``. Workflows without cutoff records (recorded
    before this protocol) are not applicable.
    """

    recorded = recorded_cutoff_potcars(flow_spec)
    if not recorded:
        return {"status": "not_applicable", "potcars": [], "problems": []}
    loader = potcar_loader or globals()["_default_potcar_loader"]
    problems: list[str] = []
    checked = []
    for symbol in sorted(recorded):
        expected = recorded[symbol]
        try:
            actual = potcar_identity(loader(symbol, expected["functional"]))
        except Exception as exc:  # noqa: BLE001 - reported as a parity problem
            problems.append(f"POTCAR {symbol} ({expected['functional']}) could not be read: {type(exc).__name__}: {exc}")
            continue
        differences = [
            key
            for key in ("TITEL", "ENMAX", "sha256")
            if (float(actual[key]) != float(expected[key]) if key == "ENMAX" else actual[key] != expected[key])
        ]
        if differences:
            problems.append(
                f"POTCAR {symbol} differs from the dataset the plane-wave cutoff was resolved "
                f"from ({', '.join(differences)}): recorded ENMAX {expected['ENMAX']}, "
                f"installed ENMAX {actual['ENMAX']}"
            )
        checked.append({"symbol": symbol, **actual})
    return {"status": "failed" if problems else "passed", "potcars": checked, "problems": problems}


__all__ = [
    "COMMON_RELAX_SETTINGS",
    "CUTOFF_OPTION_KEY",
    "CUTOFF_RULE_ID",
    "CUTOFF_RULE_VERSION",
    "PASS_CONVERGENCE",
    "PROTOCOL_ID",
    "PROTOCOL_OPTION_KEY",
    "PROTOCOL_VERSION",
    "RELAX_CUSTODIAN_HANDLER_EXCLUSIONS",
    "downstream_cutoff_floor",
    "freeze_plane_wave_cutoffs",
    "load_enmax_metadata",
    "protocol_option",
    "protocol_payload",
    "protocol_relaxation_stages",
    "relaxation_incar_settings",
    "relaxation_pass_role",
    "relaxation_stage_payload",
    "resolve_cutoff_record",
    "require_cutoff_functional",
    "resolved_encut",
    "runtime_potcar_check",
    "stage_cutoff_record",
    "stage_protocol_pass",
    "validate_stage_options",
    "validate_workflow",
    "without_relaxation_options",
]
