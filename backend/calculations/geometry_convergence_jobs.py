"""Run-time jobs of BMD Adaptive Geometry Relaxation v1.

The scientific policy (cutoffs, acceptance criteria, decisions) is
``backend.calculations.geometry_convergence``; this module only executes it.

One Jobflow job, :func:`geometry_convergence_controller`, follows the final
pass of the canonical relaxation. It replaces itself (Jobflow's dynamic
``Response(replace=...)``, executed by BMD's runner inside the same SLURM
allocation) with the probe and a decision job, and the decision job, when the
policy asks for it, with the single refinement relaxation, the final probe and
the final decision. The replacement flows expose the decision's output, so the
stages after the controller receive the accepted geometry and the accepted
relaxation's directory, never the initial relaxation's unless it was accepted.

A stop (unresolved geometry, VASP failure, malformed evidence or insufficient
remaining walltime) writes the record and raises
:class:`GeometryConvergenceStopped`; BMD's runner then skips every stage that
depends on the geometry and the run fails. Every completed VASP directory is
kept. The record ``geometry_convergence/geometry_convergence.json`` is written
before every dynamic stage and after every decision, so an interrupted run
still shows which stage was pending.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from jobflow import Flow, Response, job

from backend.calculations import geometry_convergence as policy


class GeometryConvergenceStopped(RuntimeError):
    """The geometry is not released downstream (see the convergence record)."""


class EvidenceError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------
# Evidence from completed VASP directories
# --------------------------------------------------------------------------


def local_directory(dir_name) -> Path:
    """The local path of an atomate2 ``dir_name`` (``host:/path`` or a path)."""

    text = str(dir_name)
    if ":" in text and not os.path.exists(text):
        text = text.split(":", 1)[1]
    return Path(text)


def _first_existing(directory: Path, name: str) -> Path | None:
    for candidate in (directory / name, directory / f"{name}.gz"):
        if candidate.is_file():
            return candidate
    return None


def read_stage_evidence(dir_name) -> dict[str, Any]:
    """Convergence, forces, stress and magnetisation of one completed VASP stage."""

    from pymatgen.io.vasp import Outcar, Vasprun

    from backend.remote_result_parser import geometry_relaxation_diagnostic

    directory = local_directory(dir_name)
    path = _first_existing(directory, "vasprun.xml")
    if path is None:
        raise EvidenceError("missing", f"no vasprun.xml in {directory}")
    try:
        vasprun = Vasprun(str(path), parse_dos=False, parse_eigen=False, parse_potcar_file=False)
        steps = list(vasprun.ionic_steps or [])
        if not steps:
            raise ValueError("no ionic steps")
        forces = steps[-1].get("forces")
        if forces is None:
            raise ValueError("no forces")
        max_force = max((math.sqrt(sum(float(c) ** 2 for c in f)) for f in forces), default=0.0)
        stress = (geometry_relaxation_diagnostic(vasprun).get("residual_stress") or {})
        parameters = dict(vasprun.parameters or {})
        ispin = int(parameters.get("ISPIN", 1) or 1)
        evidence = {
            "directory": str(directory),
            "encut": float(parameters.get("ENCUT")) if parameters.get("ENCUT") is not None else None,
            "converged_electronic": bool(vasprun.converged_electronic),
            "converged_ionic": bool(vasprun.converged_ionic),
            "ionic_steps": len(steps),
            "max_force_ev_a": max_force,
            "pressure_kbar": stress.get("hydrostatic_pressure_kbar"),
            "max_stress_kbar": stress.get("max_abs_component_kbar"),
            "stress_tensor_kbar": stress.get("stress_tensor_kbar"),
            "free_energy_ev": float(steps[-1]["e_fr_energy"]),
            "volume_a3": float(vasprun.final_structure.volume),
            "lattice_abc": [float(v) for v in vasprun.final_structure.lattice.abc],
            "spin_polarized": ispin == 2,
            "magnetisation": None,
        }
    except EvidenceError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as malformed evidence
        raise EvidenceError("malformed", f"{path}: {type(exc).__name__}: {exc}") from None
    if evidence["spin_polarized"]:
        outcar_path = _first_existing(directory, "OUTCAR")
        try:
            outcar = Outcar(str(outcar_path)) if outcar_path is not None else None
            total = None if outcar is None else outcar.total_mag
            sites = [float(site["tot"]) for site in (outcar.magnetization or ())] if outcar is not None else []
            evidence["magnetisation"] = None if total is None else {"total": float(total), "sites": sites}
        except Exception:  # noqa: BLE001 - the policy reports missing magnetisation
            evidence["magnetisation"] = None
    return evidence


# --------------------------------------------------------------------------
# Resource budget
# --------------------------------------------------------------------------

_SQUEUE_TIME = re.compile(r"^(?:(?P<days>\d+)-)?(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)$")


def _parse_squeue_time(text: str) -> float | None:
    text = text.strip()
    if text.upper() == "UNLIMITED":
        return math.inf
    match = _SQUEUE_TIME.match(text)
    if not match:
        return None
    days = int(match.group("days") or 0)
    hours = int(match.group("h") or 0)
    return float(((days * 24 + hours) * 60 + int(match.group("m"))) * 60 + int(match.group("s")))


def remaining_walltime_seconds() -> tuple[float | None, str]:
    """(seconds left in this SLURM allocation or ``None``, source)."""

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None, "not_slurm"
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.strip().isdigit():
        return float(int(end) - time.time()), "SLURM_JOB_END_TIME"
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%L"], capture_output=True, text=True, timeout=60
        ).stdout
        seconds = _parse_squeue_time(out.splitlines()[0]) if out.strip() else None
    except Exception:  # noqa: BLE001 - unknown time left stops the controller
        seconds = None
    return (seconds, "squeue") if seconds is not None else (None, "unknown")


def estimated_stage_seconds(maker, structure, *, stage_type: str, theory: str) -> float:
    """The planning estimator's conservative time for one controller stage."""

    from backend.calculations.automatic_resources import estimate_stage
    from backend.calculations.models import StageType, Theory

    input_set = maker.input_set_generator.get_input_set(structure, potcar_spec=True)
    preview = {
        "index": 0,
        "phonon_plan": None,
        "stage_spec": SimpleNamespace(stage_type=StageType(stage_type), theory=Theory(theory)),
        "input_set": SimpleNamespace(
            incar=dict(input_set.incar), poscar=input_set.poscar, kpoints=input_set.kpoints, potcar=input_set.potcar
        ),
    }
    cpus = int(os.environ.get("SLURM_NTASKS") or 24)
    return float(estimate_stage(preview, cpus=cpus, potcar_functional="PBE_64")["seconds"])


def _budget_check(stages, *, theory: str) -> dict[str, Any]:
    """Whether the remaining allocation covers ``stages`` [(label, maker, structure, type)]."""

    remaining, source = remaining_walltime_seconds()
    check = {"stages": [label for label, *_ in stages], "remaining_s": remaining, "source": source}
    if source == "not_slurm":
        return {**check, "passed": True, "required_s": None, "note": "not running under SLURM"}
    if remaining is None:
        return {**check, "passed": False, "required_s": None, "reason": "remaining walltime could not be determined"}
    try:
        required = sum(
            estimated_stage_seconds(maker, structure, stage_type=stage_type, theory=theory)
            for _, maker, structure, stage_type in stages
        )
    except Exception as exc:  # noqa: BLE001 - an unsizable stage is not started
        return {**check, "passed": False, "required_s": None, "reason": f"stage could not be sized: {type(exc).__name__}: {exc}"}
    passed = remaining >= required
    return {
        **check,
        "remaining_s": None if math.isinf(remaining) else remaining,
        "required_s": required,
        "passed": passed,
        **({} if passed else {"reason": f"{remaining:.0f} s left, {required:.0f} s required by the planning estimate"}),
    }


# --------------------------------------------------------------------------
# Frozen reciprocal-space sampling
# --------------------------------------------------------------------------


def read_kpoints(dir_name):
    """(Kpoints, identity) of the KPOINTS file a completed stage actually used."""

    from pymatgen.io.vasp import Kpoints

    directory = local_directory(dir_name)
    path = _first_existing(directory, "KPOINTS")
    if path is None:
        raise EvidenceError("missing", f"no KPOINTS in {directory} (the k-point mesh cannot be frozen)")
    try:
        kpoints = Kpoints.from_file(str(path))
    except Exception as exc:  # noqa: BLE001 - reported as malformed evidence
        raise EvidenceError("malformed", f"{path}: {type(exc).__name__}: {exc}") from None
    return kpoints, kpoints_identity(kpoints)


def kpoints_identity(kpoints) -> dict[str, Any]:
    text = str(kpoints)
    return {
        "style": str(getattr(kpoints.style, "name", kpoints.style)),
        "kpts": [[float(v) for v in row] for row in kpoints.kpts],
        "shift": [float(v) for v in (kpoints.kpts_shift or (0.0, 0.0, 0.0))],
        "num_kpts": int(kpoints.num_kpts or 0),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _same_mesh(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return all(a.get(key) == b.get(key) for key in ("style", "kpts", "shift", "num_kpts"))


# --------------------------------------------------------------------------
# Controller stages (built from the final relaxation pass's maker)
# --------------------------------------------------------------------------


def _with_incar(maker, updates: Mapping[str, Any], kpoints=None):
    generator = maker.input_set_generator
    settings = dict(generator.user_incar_settings or {})
    settings.update(dict(updates))
    changes = {"user_incar_settings": settings}
    if kpoints is not None:
        # A pymatgen Kpoints object is used verbatim by the input set, so the
        # stage samples exactly the mesh of the relaxation it follows.
        changes["user_kpoints_settings"] = kpoints
    return dataclasses.replace(generator, **changes)


def probe_maker(relaxation_maker, record: Mapping[str, Any], *, name: str, kpoints=None):
    """Single point with the relaxation's settings at the probe cutoff (NSW 0, IBRION -1)."""

    from atomate2.vasp.jobs.core import StaticMaker

    return StaticMaker(
        input_set_generator=_with_incar(relaxation_maker, policy.probe_incar_overrides(record), kpoints),
        name=name,
        run_vasp_kwargs=relaxation_maker.run_vasp_kwargs,
        stop_children_kwargs=relaxation_maker.stop_children_kwargs,
    )


def refinement_maker(relaxation_maker, record: Mapping[str, Any], *, name: str, kpoints=None):
    """The final pass's relaxation at the refinement cutoff."""

    return dataclasses.replace(
        relaxation_maker,
        name=name,
        input_set_generator=_with_incar(relaxation_maker, policy.refinement_incar_overrides(record), kpoints),
    )


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------


def _record_path(record_dir) -> Path:
    return Path(record_dir) / policy.RECORD_FILENAME


def read_record(record_dir) -> dict[str, Any]:
    return json.loads(_record_path(record_dir).read_text(encoding="utf-8"))


def write_record(record_dir, record: Mapping[str, Any]) -> None:
    payload = dict(record)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = _record_path(record_dir)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _new_record(frozen: Mapping[str, Any], downstream_cutoffs) -> dict[str, Any]:
    return {
        "schema": policy.RECORD_SCHEMA,
        "schema_version": policy.RECORD_SCHEMA_VERSION,
        "policy": dict(frozen),
        "status": policy.STATUS_IN_PROGRESS,
        "outcome": None,
        "reasons": [],
        "refinements": 0,
        "accepted_encut": None,
        "accepted_stage": None,
        "accepted_relaxation_dir": None,
        "accepted_files": {},
        "downstream_cutoffs": list(downstream_cutoffs or []),
        "frozen_kpoints": None,
        "pending_stage": None,
        "operations": [],
        "attempts": [],
        "resource_checks": [],
    }


def _start(record_dir, record: dict, operation: str, **detail) -> None:
    """Record ``operation`` as the pending dynamic operation (durably, before it runs)."""

    for entry in record["operations"]:
        if entry["state"] == "started":
            entry["state"] = "completed"
    record["operations"].append({"operation": operation, "state": "started", **detail})
    record["pending_stage"] = {"operation": operation, **detail}
    write_record(record_dir, record)


# Job metadata naming the controller operation a VASP job performs, so the
# BMD runner can record a failure it observes (backend.execution).
OPERATION_METADATA_KEY = "bmd_geometry_convergence_operation"
FAILURE_REASON_MAX = 500


def _observed(job, record_dir, operation: str):
    job.metadata[OPERATION_METADATA_KEY] = {"record_dir": str(record_dir), "operation": operation}
    return job


def record_observed_failure(observer: Mapping[str, Any], exc: BaseException) -> None:
    """Record that the runner observed ``exc`` in a controller operation.

    Called by BMD's runner where the job raised (before it marks the job
    errored); the original exception is then handled exactly as before. A
    controller stop (``GeometryConvergenceStopped``) has already written its
    own outcome and is left unchanged. An operation that never raises here
    but never completes (killed, interrupted) stays ``started``.
    """

    if isinstance(exc, GeometryConvergenceStopped):
        return
    record_dir, operation = observer["record_dir"], str(observer["operation"])
    record = read_record(record_dir)
    reason = f"{operation} failed: {type(exc).__name__}: {exc}"[:FAILURE_REASON_MAX]
    entry = next(
        (item for item in reversed(record.get("operations") or []) if item.get("operation") == operation and item.get("state") == "started"),
        None,
    )
    if entry is None:
        entry = {"operation": operation}
        record.setdefault("operations", []).append(entry)
    entry.update(state="failed", failure=reason)
    record.update(
        status=policy.STATUS_EXECUTION_FAILURE,
        outcome=policy.OUTCOME_EXECUTION_FAILURE,
        reasons=[reason],
        pending_stage=None,
    )
    write_record(record_dir, record)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_accepted(relaxation_dir: Path, record_dir: Path) -> dict[str, str]:
    """Copy the accepted relaxation's key files to ``accepted/`` (originals kept)."""

    import gzip

    target = record_dir / policy.ACCEPTED_DIRECTORY
    target.mkdir(exist_ok=True)
    copied = {}
    for name in policy.ACCEPTED_FILES:
        source = _first_existing(relaxation_dir, name)
        if source is None:
            continue
        # Always the plain file name (the results read CONTCAR, OUTCAR, vasprun.xml).
        if source.suffix == ".gz":
            with gzip.open(source, "rb") as handle:
                (target / name).write_bytes(handle.read())
        else:
            shutil.copy2(source, target / name)
        copied[name] = _sha256(target / name)
    return copied


def _stop(record_dir, record: dict, *, status: str, outcome: str, reasons) -> None:
    for entry in record["operations"]:
        if entry["state"] == "started":
            entry["state"] = "completed" if status != policy.STATUS_EXECUTION_FAILURE else "failed"
    record.update(status=status, outcome=outcome, reasons=list(reasons), pending_stage=None)
    write_record(record_dir, record)
    raise GeometryConvergenceStopped(
        f"BMD geometry convergence stopped ({outcome}): " + "; ".join(record["reasons"])
        + f". The geometry is not released to downstream stages; see {_record_path(record_dir)}."
    )


def _invalid(record_dir, record, reason: str) -> None:
    _stop(
        record_dir,
        record,
        status=policy.STATUS_INVALID_EVIDENCE,
        outcome=policy.OUTCOME_INVALID_EVIDENCE,
        reasons=[reason],
    )


# --------------------------------------------------------------------------
# The controller jobs
# --------------------------------------------------------------------------


def _evidence(dir_name, record_dir, record) -> dict[str, Any]:
    try:
        return read_stage_evidence(dir_name)
    except EvidenceError as exc:
        _invalid(record_dir, record, f"{exc.kind} evidence: {exc}")
        raise  # unreachable


def _mesh(dir_name, record_dir, record):
    try:
        return read_kpoints(dir_name)
    except EvidenceError as exc:
        _invalid(record_dir, record, f"{exc.kind} k-point evidence: {exc}")
        raise  # unreachable


def _budget(record_dir, record, *, stages, theory, operation: str) -> None:
    check = _budget_check(stages, theory=theory)
    record["resource_checks"].append({"before": operation, **check})
    if not check["passed"]:
        _stop(
            record_dir,
            record,
            status=policy.STATUS_RESOURCE_BUDGET,
            outcome=policy.OUTCOME_RESOURCE_BUDGET,
            reasons=[f"not starting {operation}: {check['reason']}"],
        )


@job
def geometry_convergence_checkpoint(structure, dir_name, record_dir: str, frozen_kpoints: dict, final_probe_maker, theory: str = "pbe"):
    """After the refinement, before the final probe: check its mesh and mark the probe pending.

    Transitions (each written durably): on entry the refinement is
    ``executed`` (its VASP job returned; its evidence is not yet verified) and
    this checkpoint ``started``; once the refinement's evidence is verified
    both are ``completed`` and the final probe is ``started``. An exception
    observed by the runner marks this checkpoint ``failed``; an interruption
    leaves it ``started``.
    """

    record = read_record(record_dir)
    for entry in record["operations"]:
        if entry["operation"] == policy.OPERATION_REFINEMENT and entry["state"] == "started":
            entry["state"] = "executed"
    record["operations"].append({"operation": policy.OPERATION_REFINEMENT_CHECKPOINT, "state": "started"})
    record["pending_stage"] = {"operation": policy.OPERATION_REFINEMENT_CHECKPOINT}
    write_record(record_dir, record)
    try:
        _, identity = read_kpoints(dir_name)
    except EvidenceError as exc:
        _invalid(record_dir, record, f"refinement {exc.kind} k-point evidence: {exc}")
    if not _same_mesh(identity, frozen_kpoints):
        _invalid(record_dir, record, "the refinement did not use the frozen k-point mesh")
    for entry in record["operations"]:
        if entry["operation"] == policy.OPERATION_REFINEMENT and entry["state"] == "executed":
            entry["state"] = "completed"
    _budget(
        record_dir,
        record,
        stages=[("final probe", final_probe_maker, structure, "static")],
        theory=theory,
        operation=policy.OPERATION_FINAL_PROBE,
    )
    _start(record_dir, record, policy.OPERATION_FINAL_PROBE, encut=record["policy"]["probe_encut"])
    return {"structure": structure, "dir_name": dir_name}


@job
def geometry_convergence_controller(
    structure,
    relaxation_dir,
    relaxation_maker,
    record: dict,
    theory: str = "pbe",
    downstream_cutoffs: list | None = None,
    state: dict | None = None,
):
    """Probe, decide and (at most once) refine the final relaxation's geometry."""

    frozen = dict(record)
    state = dict(state or {})
    if not state:
        return _initial(structure, relaxation_dir, relaxation_maker, frozen, theory, downstream_cutoffs)
    record_dir = state["record_dir"]
    run_record = read_record(record_dir)
    operation = policy.OPERATION_FINAL_DECISION if int(state["refinements"]) else policy.OPERATION_INITIAL_DECISION
    _start(record_dir, run_record, operation)
    try:
        return _decide(structure, relaxation_dir, relaxation_maker, frozen, theory, downstream_cutoffs, state, record_dir, run_record)
    except GeometryConvergenceStopped:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded, then the run fails
        _stop(
            record_dir,
            run_record,
            status=policy.STATUS_EXECUTION_FAILURE,
            outcome=policy.OUTCOME_EXECUTION_FAILURE,
            reasons=[f"{operation} failed: {type(exc).__name__}: {exc}"],
        )


def _initial(structure, relaxation_dir, relaxation_maker, frozen, theory, downstream_cutoffs):
    # Initial call, in the controller's own stage directory.
    record_dir = str(Path.cwd())
    run_record = _new_record(frozen, downstream_cutoffs)
    write_record(record_dir, run_record)
    kpoints, identity = _mesh(relaxation_dir, record_dir, run_record)
    run_record["frozen_kpoints"] = {"source": str(local_directory(relaxation_dir)), **identity}
    probe = probe_maker(relaxation_maker, frozen, name=policy.OPERATION_INITIAL_PROBE, kpoints=kpoints).make(structure)
    decision = geometry_convergence_controller(
        structure,
        relaxation_dir,
        relaxation_maker,
        frozen,
        theory=theory,
        downstream_cutoffs=downstream_cutoffs,
        state={
            "record_dir": record_dir,
            "refinements": 0,
            "probe_dir": probe.output.dir_name,
            "initial_relaxation_dir": relaxation_dir,
            "frozen_kpoints": identity,
        },
    )
    decision.name = policy.OPERATION_INITIAL_DECISION
    _observed(probe, record_dir, policy.OPERATION_INITIAL_PROBE)
    _budget(
        record_dir,
        run_record,
        stages=[("initial probe", probe.maker, structure, "static")],
        theory=theory,
        operation=policy.OPERATION_INITIAL_PROBE,
    )
    _start(record_dir, run_record, policy.OPERATION_INITIAL_PROBE, encut=frozen["probe_encut"])
    return Response(replace=Flow([probe, decision], output=decision.output))


def _decide(structure, relaxation_dir, relaxation_maker, frozen, theory, downstream_cutoffs, state, record_dir, run_record):
    refinements = int(state["refinements"])
    relaxation = _evidence(relaxation_dir, record_dir, run_record)
    probe = _evidence(state["probe_dir"], record_dir, run_record)
    _, relaxation_mesh = _mesh(relaxation_dir, record_dir, run_record)
    _, probe_mesh = _mesh(state["probe_dir"], record_dir, run_record)
    frozen_mesh = state["frozen_kpoints"]
    reference = None
    if refinements:
        reference = _evidence(state["initial_relaxation_dir"], record_dir, run_record)
    if not (_same_mesh(relaxation_mesh, frozen_mesh) and _same_mesh(probe_mesh, frozen_mesh)):
        evaluation = {
            "category": "invalid_evidence",
            "checks": [{"name": "kpoints_identical", "passed": False}],
            "reasons": ["the probe or relaxation did not use the frozen k-point mesh"],
        }
    else:
        evaluation = policy.evaluate_probe(relaxation=relaxation, probe=probe, reference_relaxation=reference)
        evaluation["checks"].insert(0, {"name": "kpoints_identical", "passed": True})
    decision = policy.decide(frozen, refinements_done=refinements, evaluation=evaluation)
    stage_name = "final pass (stage 2 of the canonical relaxation)" if refinements == 0 else "refinement relaxation"
    run_record["attempts"].append(
        {
            "role": "initial" if refinements == 0 else "refinement",
            "relaxation_stage": stage_name,
            "relaxation_encut": frozen["initial_encut"] if refinements == 0 else frozen["refinement_encut"],
            "probe_encut": frozen["probe_encut"],
            "relaxation_dir": str(local_directory(relaxation_dir)),
            "probe_dir": str(local_directory(state["probe_dir"])),
            "relaxation_kpoints": relaxation_mesh,
            "probe_kpoints": probe_mesh,
            "relaxation": {k: v for k, v in relaxation.items() if k != "directory"},
            "probe": {k: v for k, v in probe.items() if k != "directory"},
            "category": evaluation["category"],
            "checks": evaluation["checks"],
            "reasons": evaluation["reasons"],
            "decision": decision["action"],
        }
    )
    run_record["refinements"] = refinements

    if decision["action"] == "accept":
        accepted_dir = local_directory(relaxation_dir)
        attempt = run_record["attempts"][-1]
        run_record.update(
            outcome=decision["outcome"],
            reasons=decision["reasons"],
            accepted_encut=attempt["relaxation_encut"],
            accepted_stage=attempt["role"],
            accepted_relaxation_dir=str(accepted_dir),
            accepted_files=_copy_accepted(accepted_dir, Path(record_dir)),
        )
        for entry in run_record["operations"]:
            entry["state"] = "completed"
        run_record.update(status=decision["status"], pending_stage=None)
        write_record(record_dir, run_record)
        return {
            "structure": structure,
            "dir_name": relaxation_dir,
            "accepted_encut": run_record["accepted_encut"],
            "accepted_stage": run_record["accepted_stage"],
            "outcome": run_record["outcome"],
            "record": str(_record_path(record_dir)),
        }

    if decision["action"] == "refine":
        # The refinement and both its probes sample the rejected relaxation's mesh.
        kpoints, identity = _mesh(relaxation_dir, record_dir, run_record)
        refinement = refinement_maker(
            relaxation_maker, frozen, name=policy.OPERATION_REFINEMENT, kpoints=kpoints
        ).make(structure)
        final_probe_maker = probe_maker(refinement.maker, frozen, name=policy.OPERATION_FINAL_PROBE, kpoints=kpoints)
        checkpoint = geometry_convergence_checkpoint(
            refinement.output.structure,
            refinement.output.dir_name,
            record_dir,
            identity,
            final_probe_maker,
            theory=theory,
        )
        checkpoint.name = "checkpoint_refinement"
        final_probe = final_probe_maker.make(checkpoint.output.structure)
        final_decision = geometry_convergence_controller(
            checkpoint.output.structure,
            checkpoint.output.dir_name,
            refinement.maker,
            frozen,
            theory=theory,
            downstream_cutoffs=downstream_cutoffs,
            state={
                "record_dir": record_dir,
                "refinements": refinements + 1,
                "probe_dir": final_probe.output.dir_name,
                "initial_relaxation_dir": state["initial_relaxation_dir"],
                "frozen_kpoints": identity,
            },
        )
        final_decision.name = policy.OPERATION_FINAL_DECISION
        _observed(refinement, record_dir, policy.OPERATION_REFINEMENT)
        _observed(checkpoint, record_dir, policy.OPERATION_REFINEMENT_CHECKPOINT)
        _observed(final_probe, record_dir, policy.OPERATION_FINAL_PROBE)
        _budget(
            record_dir,
            run_record,
            stages=[
                ("refinement relaxation", refinement.maker, structure, "relax"),
                ("final probe", final_probe_maker, structure, "static"),
            ],
            theory=theory,
            operation=policy.OPERATION_REFINEMENT,
        )
        _start(record_dir, run_record, policy.OPERATION_REFINEMENT, encut=frozen["refinement_encut"])
        return Response(replace=Flow([refinement, checkpoint, final_probe, final_decision], output=final_decision.output))

    _stop(record_dir, run_record, status=decision["status"], outcome=decision["outcome"], reasons=decision["reasons"])


def controller_job(relaxation_job, record: Mapping[str, Any], *, theory: str, maker=None):
    """The controller following ``relaxation_job`` (the final protocol pass, made by ``maker``)."""

    controller = geometry_convergence_controller(
        relaxation_job.output.structure,
        relaxation_job.output.dir_name,
        maker if maker is not None else relaxation_job.maker,
        dict(record),
        theory=theory,
    )
    controller.name = policy.STAGE_DIRECTORY
    return controller


__all__ = [
    "EvidenceError",
    "GeometryConvergenceStopped",
    "controller_job",
    "estimated_stage_seconds",
    "geometry_convergence_checkpoint",
    "geometry_convergence_controller",
    "kpoints_identity",
    "record_observed_failure",
    "local_directory",
    "probe_maker",
    "read_kpoints",
    "read_record",
    "read_stage_evidence",
    "refinement_maker",
    "remaining_walltime_seconds",
    "write_record",
]
