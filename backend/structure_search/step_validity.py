"""Validity of one USPEX relaxation step's VASP result as a fitness evaluation.

USPEX reads a candidate's energy and structure from the VASP files each
relaxation step leaves in its CalcFold. A VASP process that exits with status
0 has not thereby converged: VASP stops at ``NELM`` electronic steps or
``NSW`` ionic steps without an error. This module decides, after each step and
before USPEX sees the result, whether the step may count, following the lab's
stage-specific rules (6 Nov 2026, :data:`STEP_VALIDITY`):

* steps 1-3: electronic convergence required; reaching ``NSW`` is allowed;
* step 4: electronic and ionic convergence required;
* step 5: electronic convergence and a valid (finite) ranking energy required.

Every step also needs a finite final energy and readable evidence. Each
result is classified as one of:

``valid``
    VASP exited with status 0 and the evidence meets the step's criteria.
``invalid_convergence``
    VASP completed (status 0) and the evidence is readable, but a criterion
    fails: an unconverged electronic or ionic loop, or a non-finite energy.
``execution_failed``
    VASP did not complete: a non-zero exit status, or the launcher stopped
    it. The evidence is not consulted for the decision.
``invalid_evidence``
    VASP exited with status 0 but the evidence is missing, unreadable,
    truncated, inconsistent or too large to read, or the checker itself
    failed. bmd-compute cannot tell whether the step converged, so it does
    not count (fail closed).

Evidence and definitions (the same as elsewhere in bmd-compute):

* ``vasprun.xml`` is parsed with pymatgen's ``Vasprun`` (no DOS,
  eigenvalues or POTCAR), as in
  ``backend.calculations.geometry_convergence_jobs.read_stage_evidence``.
  Electronic convergence is ``Vasprun.converged_electronic`` (the final ionic
  step's SCF loop ended before ``NELM``) and ionic convergence is
  ``Vasprun.converged_ionic`` (fewer ionic steps than ``NSW``, or a single
  point), the definitions BMD's result parser and its adaptive geometry
  relaxation use. A truncated ``vasprun.xml`` (a killed run) is malformed.
* The energy is the final ionic step's free energy ``e_fr_energy`` (F,
  ``TOTEN``), which is what USPEX ranks on (``incar_policy`` step 5).
* ``OUTCAR`` must contain ``free  energy   TOTEN`` lines; the last must
  agree with ``vasprun.xml`` within :data:`ENERGY_AGREEMENT_EV`, so the
  energy USPEX reads is the one that was checked.
* ``CONTCAR`` must be a readable structure with the same number of atoms
  as ``vasprun.xml``: USPEX takes the relaxed structure from it.

Files larger than :data:`MAX_EVIDENCE_BYTES` are not read (invalid evidence),
so the check is bounded. The check is deterministic: it reads only these
files and uses no clock, randomness or network.

This module is used only by the structure-search launch bridge
(``backend.structure_search.bridge.launcher``); ordinary BMD workflows never
call it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

STEP_VALIDITY_POLICY_ID = "bmd_compute.structure_search_step_validity"
STEP_VALIDITY_POLICY_VERSION = 1

# Lab decision, 6 Nov 2026 (recorded in incar_policy.LAB_DECISIONS).
STEP_VALIDITY = (
    {"step": 1, "electronic_convergence": True, "ionic_convergence": False, "valid_energy": True},
    {"step": 2, "electronic_convergence": True, "ionic_convergence": False, "valid_energy": True},
    {"step": 3, "electronic_convergence": True, "ionic_convergence": False, "valid_energy": True},
    {"step": 4, "electronic_convergence": True, "ionic_convergence": True, "valid_energy": True},
    {"step": 5, "electronic_convergence": True, "ionic_convergence": False, "valid_energy": True},
)

VALID = "valid"
INVALID_CONVERGENCE = "invalid_convergence"
EXECUTION_FAILED = "execution_failed"
INVALID_EVIDENCE = "invalid_evidence"
CLASSIFICATIONS = (VALID, INVALID_CONVERGENCE, EXECUTION_FAILED, INVALID_EVIDENCE)

# Largest evidence file read (bytes). Search candidates are small cells; a
# larger file is treated as unusable evidence rather than parsed.
MAX_EVIDENCE_BYTES = 512 * 1024 * 1024
# OUTCAR's TOTEN and vasprun.xml's e_fr_energy are both written with 8
# decimals; they must agree to this tolerance (eV).
ENERGY_AGREEMENT_EV = 1e-5

# Files USPEX reads after a VASP step. When a step does not count they are
# withheld from USPEX (renamed in the CalcFold, see withhold_outputs), so a
# stale or unconverged energy can never be read as a fitness.
USPEX_READ_FILES = ("OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml")
# Inputs kept with the archived evidence for the record.
RECORDED_INPUT_FILES = ("INCAR", "KPOINTS", "POSCAR")
WITHHELD_SUFFIX = ".bmd_invalid"

CALCFOLD_RECORD_FILE = "bmd_step_validity.jsonl"
ARCHIVE_DIR = "bmd_step_validity"

_TOTEN = re.compile(rb"free\s+energy\s+TOTEN\s*=\s*(\S+)\s*eV")


class _Evidence(Exception):
    """Evidence is missing or malformed (classified as invalid_evidence)."""


def policy_for_step(step: int) -> Mapping[str, Any] | None:
    for row in STEP_VALIDITY:
        if row["step"] == step:
            return row
    return None


def policy_record() -> dict[str, Any]:
    return {
        "policy_id": STEP_VALIDITY_POLICY_ID,
        "policy_version": STEP_VALIDITY_POLICY_VERSION,
        "criteria": [dict(row) for row in STEP_VALIDITY],
        "electronic_convergence": "pymatgen Vasprun.converged_electronic (final ionic step)",
        "ionic_convergence": "pymatgen Vasprun.converged_ionic",
        "energy": "final ionic step e_fr_energy (TOTEN), finite, equal to OUTCAR's last TOTEN",
        "energy_agreement_ev": ENERGY_AGREEMENT_EV,
        "max_evidence_bytes": MAX_EVIDENCE_BYTES,
    }


def _readable_file(calcfold: Path, name: str) -> Path:
    path = calcfold / name
    if not path.is_file():
        raise _Evidence(f"{name} is missing")
    size = path.stat().st_size
    if size == 0:
        raise _Evidence(f"{name} is empty")
    if size > MAX_EVIDENCE_BYTES:
        raise _Evidence(f"{name} is larger than {MAX_EVIDENCE_BYTES} bytes")
    return path


def _last_outcar_toten(path: Path) -> float:
    last = None
    with open(path, "rb") as handle:
        for line in handle:
            match = _TOTEN.search(line)
            if match:
                last = match.group(1)
    if last is None:
        raise _Evidence("OUTCAR has no 'free energy TOTEN' line")
    try:
        return float(last.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        raise _Evidence(f"OUTCAR's last TOTEN is not a number ({last!r})") from None


def read_evidence(calcfold: Path) -> dict[str, Any]:
    """The convergence evidence of the VASP run in ``calcfold``; raises ``_Evidence``."""

    import warnings

    from pymatgen.io.vasp import Poscar, Vasprun

    vasprun_path = _readable_file(calcfold, "vasprun.xml")
    outcar_path = _readable_file(calcfold, "OUTCAR")
    contcar_path = _readable_file(calcfold, "CONTCAR")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vasprun = Vasprun(
                str(vasprun_path), parse_dos=False, parse_eigen=False, parse_potcar_file=False
            )
        steps = list(vasprun.ionic_steps or [])
        if not steps:
            raise _Evidence("vasprun.xml has no ionic steps")
        parameters = dict(vasprun.parameters or {})
        nelm = parameters.get("NELM")
        nsw = parameters.get("NSW")
        if nelm is None or nsw is None:
            raise _Evidence("vasprun.xml does not record NELM and NSW")
        energy = steps[-1].get("e_fr_energy")
        if energy is None:
            raise _Evidence("vasprun.xml has no final free energy (e_fr_energy)")
        evidence = {
            "converged_electronic": bool(vasprun.converged_electronic),
            "converged_ionic": bool(vasprun.converged_ionic),
            "ionic_steps": len(steps),
            "final_electronic_steps": len(steps[-1].get("electronic_steps") or []),
            "nelm": int(nelm),
            "nsw": int(nsw),
            "free_energy_ev": float(energy),
            "natoms": len(vasprun.final_structure),
        }
    except _Evidence:
        raise
    except Exception as exc:  # noqa: BLE001 - any parse failure is malformed evidence
        raise _Evidence(f"vasprun.xml cannot be read: {type(exc).__name__}: {exc}") from None
    evidence["outcar_free_energy_ev"] = _last_outcar_toten(outcar_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            contcar_atoms = len(Poscar.from_file(str(contcar_path), check_for_potcar=False).structure)
    except Exception as exc:  # noqa: BLE001
        raise _Evidence(f"CONTCAR cannot be read: {type(exc).__name__}: {exc}") from None
    if contcar_atoms != evidence["natoms"]:
        raise _Evidence(
            f"CONTCAR has {contcar_atoms} atoms but vasprun.xml has {evidence['natoms']}"
        )
    return evidence


def evaluate_step(step: int, calcfold: Path, *, returncode: int | None, stopped: bool = False) -> dict[str, Any]:
    """Classify one finished step. Never raises; an internal error is invalid evidence."""

    calcfold = Path(calcfold)
    policy = policy_for_step(step)
    result: dict[str, Any] = {
        "policy_id": STEP_VALIDITY_POLICY_ID,
        "policy_version": STEP_VALIDITY_POLICY_VERSION,
        "step": step,
        "criteria": dict(policy) if policy is not None else None,
        "returncode": returncode,
        "classification": None,
        "reasons": [],
        "evidence": None,
    }
    if stopped or returncode is None or returncode != 0:
        result["classification"] = EXECUTION_FAILED
        result["reasons"].append(
            "the launcher stopped the step" if stopped or returncode is None
            else f"VASP exited with status {returncode}"
        )
        return result
    if policy is None:
        result["classification"] = INVALID_EVIDENCE
        result["reasons"].append(f"no validity policy for step {step}")
        return result
    try:
        evidence = read_evidence(calcfold)
    except _Evidence as exc:
        result["classification"] = INVALID_EVIDENCE
        result["reasons"].append(str(exc))
        return result
    except Exception as exc:  # noqa: BLE001 - fail closed
        result["classification"] = INVALID_EVIDENCE
        result["reasons"].append(f"validity check failed: {type(exc).__name__}: {exc}")
        return result
    result["evidence"] = evidence

    if not math.isfinite(evidence["outcar_free_energy_ev"]) or not math.isfinite(evidence["free_energy_ev"]):
        reasons = ["the final energy is not finite"]
        result["classification"] = INVALID_CONVERGENCE
        result["reasons"] = reasons
        return result
    if abs(evidence["outcar_free_energy_ev"] - evidence["free_energy_ev"]) > ENERGY_AGREEMENT_EV:
        result["classification"] = INVALID_EVIDENCE
        result["reasons"].append(
            f"OUTCAR's last TOTEN ({evidence['outcar_free_energy_ev']}) differs from vasprun.xml's "
            f"final free energy ({evidence['free_energy_ev']})"
        )
        return result

    reasons = []
    if policy["electronic_convergence"] and not evidence["converged_electronic"]:
        reasons.append(
            f"electronic loop not converged: the final ionic step used "
            f"{evidence['final_electronic_steps']} of NELM = {evidence['nelm']} electronic steps"
        )
    if policy["ionic_convergence"] and not evidence["converged_ionic"]:
        reasons.append(
            f"ionic relaxation not converged: {evidence['ionic_steps']} ionic steps reached "
            f"NSW = {evidence['nsw']}"
        )
    result["classification"] = INVALID_CONVERGENCE if reasons else VALID
    result["reasons"] = reasons
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def withhold_outputs(calcfold: Path, archive_dir: Path, request_id: str) -> list[dict[str, Any]]:
    """Archive the step's files and withhold the ones USPEX reads.

    Each of :data:`USPEX_READ_FILES` and :data:`RECORDED_INPUT_FILES` present
    in ``calcfold`` is copied unchanged into ``archive_dir``; then each of
    :data:`USPEX_READ_FILES` is renamed in the CalcFold to
    ``<name>.bmd_invalid.<request_id>``. Nothing is deleted or rewritten:
    the original bytes stay both in the CalcFold (renamed) and in the archive.
    """

    calcfold = Path(calcfold)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name in (*USPEX_READ_FILES, *RECORDED_INPUT_FILES):
        source = calcfold / name
        if not source.is_file():
            continue
        shutil.copy2(source, archive_dir / name)
        entry = {"name": name, "bytes": source.stat().st_size, "sha256": _sha256(source)}
        if name in USPEX_READ_FILES:
            withheld = calcfold / f"{name}{WITHHELD_SUFFIX}.{request_id}"
            os.replace(source, withheld)
            entry["withheld_as"] = withheld.name
        files.append(entry)
    return files


def write_json_atomically(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


__all__ = [
    "ARCHIVE_DIR",
    "CALCFOLD_RECORD_FILE",
    "CLASSIFICATIONS",
    "ENERGY_AGREEMENT_EV",
    "EXECUTION_FAILED",
    "INVALID_CONVERGENCE",
    "INVALID_EVIDENCE",
    "MAX_EVIDENCE_BYTES",
    "RECORDED_INPUT_FILES",
    "STEP_VALIDITY",
    "STEP_VALIDITY_POLICY_ID",
    "STEP_VALIDITY_POLICY_VERSION",
    "USPEX_READ_FILES",
    "VALID",
    "WITHHELD_SUFFIX",
    "evaluate_step",
    "policy_for_step",
    "policy_record",
    "read_evidence",
    "withhold_outputs",
    "write_json_atomically",
]
