"""Initial SIGMA of self-consistent HSE06 Band Structure stages.

Policy ``bmd_compute.hse_band_structure_sigma`` version 1. BMD Check's C2
electronic-character assessment of the preceding, completed HSE06 Static
Energy stage chooses the band-structure stage's *initial* smearing width:

* ``metallic``: ``ISMEAR = 0``, ``SIGMA = 0.20`` eV;
* ``nonmetallic``, ``uncertain`` or unusable evidence: ``ISMEAR = 0``,
  ``SIGMA = 0.01`` eV (the previous fixed value), with the reason recorded.

BMD Check assesses; BMD Compute owns only this mapping. C2 is called in
process, at run time, when atomate2 generates the band-structure inputs from
its copies of the static stage's ``vasprun.xml``, ``OUTCAR`` and ``OSZICAR``.
Previews and plans, which have no previous stage, show the default
``SIGMA = 0.01`` and describe the rule (:func:`conditional_settings_payload`).

The decision is written beside the INCAR as ``bmd_hse_band_sigma.json``.
Custodian's LargeSigmaHandler stays active and may lower SIGMA later; those
corrections are recorded by Custodian itself (``custodian.json``).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

POLICY_ID = "bmd_compute.hse_band_structure_sigma"
POLICY_VERSION = 1
ISMEAR = 0
METALLIC_SIGMA_EV = 0.2
DEFAULT_SIGMA_EV = 0.01
EVIDENCE_STAGE_TYPE = "static"
EVIDENCE_THEORY = "hse06"
PROVENANCE_FILENAME = "bmd_hse_band_sigma.json"
# Files atomate2 must copy from the static stage: CHGCAR is HSEBSMaker's own
# default; OSZICAR lets C1 establish the static's electronic convergence.
COPIED_PREVIOUS_STAGE_FILES = ("CHGCAR", "OSZICAR")


def conditional_settings_payload() -> dict[str, Any]:
    """The rule, as recorded in plan identity (no runtime outcome)."""

    return {
        "policy": {"id": POLICY_ID, "version": POLICY_VERSION},
        "evidence": {
            "stage": "preceding HSE06 Static Energy",
            "assessment": "bmd_check.electronic_character_assessment (C2)",
        },
        "ISMEAR": ISMEAR,
        "initial_SIGMA_ev": {
            "metallic": METALLIC_SIGMA_EV,
            "nonmetallic": DEFAULT_SIGMA_EV,
            "uncertain": DEFAULT_SIGMA_EV,
        },
        "default_SIGMA_ev": DEFAULT_SIGMA_EV,
    }


def preview_note(evidence_stage_index: int) -> str:
    return (
        f"# SIGMA = {DEFAULT_SIGMA_EV} is the default. At run time it is {METALLIC_SIGMA_EV} eV "
        f"if BMD Check C2 classifies the Stage {evidence_stage_index} HSE06 static as metallic "
        f"({POLICY_ID} v{POLICY_VERSION}); Custodian may lower it during the run."
    )


def select_initial_sigma(directory: str | Path, *, evidence_stage_index: int | None) -> dict[str, Any]:
    """Assess the copied static evidence in ``directory`` and choose SIGMA."""

    directory = Path(directory)
    assessment = None
    try:
        from bmd_agent.resources.electronic_character import (
            assess_stage_directory,
            serialize_electronic_character_assessment,
        )

        assessment = serialize_electronic_character_assessment(
            assess_stage_directory(
                directory,
                stage_index=evidence_stage_index,
                stage_type=EVIDENCE_STAGE_TYPE,
                theory=EVIDENCE_THEORY,
            )
        )
        classification = assessment.get("classification")
        if classification == "metallic":
            sigma, basis = METALLIC_SIGMA_EV, "metallic"
        elif classification == "nonmetallic":
            sigma, basis = DEFAULT_SIGMA_EV, "nonmetallic"
        else:
            codes = [item.get("reason_code") for item in assessment.get("findings") or []]
            sigma, basis = DEFAULT_SIGMA_EV, "fallback: uncertain (" + ", ".join(filter(None, codes)) + ")"
    except Exception as exc:  # noqa: BLE001 - any failure is the conservative default
        classification = None
        sigma, basis = DEFAULT_SIGMA_EV, f"fallback: assessment unavailable ({type(exc).__name__}: {exc})"
    return {
        "policy": {"id": POLICY_ID, "version": POLICY_VERSION},
        "evidence_stage_index": evidence_stage_index,
        "evidence_vasprun_sha256": _sha256(directory / "vasprun.xml"),
        "classification": classification,
        "method_version": (assessment or {}).get("method_version"),
        "ISMEAR": ISMEAR,
        "initial_SIGMA_ev": sigma,
        "basis": basis,
        "custodian": "LargeSigmaHandler may lower SIGMA during the run; see custodian.json",
        "assessment": assessment,
    }


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def hse_band_sigma_generator_class(base_class):
    """The C2-informed subclass of an HSE06 band-structure input generator."""

    return _generator_class_for(base_class)


@lru_cache(maxsize=None)
def _generator_class_for(base_class):
    import copy
    from dataclasses import dataclass, is_dataclass, replace

    class HseBandSigmaGenerator(base_class):
        bmd_sigma_evidence_stage_index: int | None = None

        if not is_dataclass(base_class):  # test doubles of the atomate2 generator

            def __init__(self, *args: Any, bmd_sigma_evidence_stage_index: int | None = None, **kwargs: Any):
                super().__init__(*args, **kwargs)
                self.bmd_sigma_evidence_stage_index = bmd_sigma_evidence_stage_index

        def get_input_set(self, structure=None, prev_dir=None, **kwargs: Any):
            if prev_dir is None:
                # Previews and plans: no completed static stage, default SIGMA.
                return super().get_input_set(structure, **kwargs)
            decision = select_initial_sigma(prev_dir, evidence_stage_index=self.bmd_sigma_evidence_stage_index)
            settings = {**dict(self.user_incar_settings or {}), "SIGMA": decision["initial_SIGMA_ev"]}
            if is_dataclass(self):
                generator = replace(self, user_incar_settings=settings)
            else:
                generator = copy.copy(self)
                generator.user_incar_settings = settings
            input_set = super(HseBandSigmaGenerator, generator).get_input_set(structure, prev_dir=prev_dir, **kwargs)
            incar = input_set["INCAR"]
            if incar.get("ISMEAR") != ISMEAR or incar.get("SIGMA") != decision["initial_SIGMA_ev"]:
                raise RuntimeError(
                    "The generated HSE06 band-structure INCAR does not match the recorded smearing decision."
                )
            input_set[PROVENANCE_FILENAME] = json.dumps(decision, indent=2, sort_keys=True) + "\n"
            return input_set

    if is_dataclass(base_class):
        HseBandSigmaGenerator = dataclass(HseBandSigmaGenerator)
    name = f"HseBandSigma{base_class.__name__}"
    HseBandSigmaGenerator.__module__ = __name__
    HseBandSigmaGenerator.__name__ = name
    HseBandSigmaGenerator.__qualname__ = name
    return HseBandSigmaGenerator


def __getattr__(name: str):
    # Resolve serialized ``@class`` names without importing atomate2 eagerly.
    prefix = "HseBandSigma"
    if name.startswith(prefix):
        from backend.calculations import hse_band_kpoints

        base_name = name[len(prefix):]
        if base_name == hse_band_kpoints.FULL_ZONE_HSE_BAND_GENERATOR_NAME:
            return hse_band_sigma_generator_class(hse_band_kpoints.full_zone_hse_band_structure_generator_class())
        import atomate2.vasp.sets.core as core

        base = getattr(core, base_name, None)
        if base is not None:
            return hse_band_sigma_generator_class(base)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "COPIED_PREVIOUS_STAGE_FILES",
    "DEFAULT_SIGMA_EV",
    "METALLIC_SIGMA_EV",
    "POLICY_ID",
    "POLICY_VERSION",
    "PROVENANCE_FILENAME",
    "conditional_settings_payload",
    "hse_band_sigma_generator_class",
    "preview_note",
    "select_initial_sigma",
]
