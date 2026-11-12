"""A fake finished managed USPEX search: the run directory the job leaves on POWER, and SLURM's record of it.

Used by the scientific execution acceptance tests to exercise the POWER
evidence collector and the local analyser without POWER, USPEX or VASP. The
layout and record formats follow the job (``backend.structure_search.job``),
the launcher (journal, done files, step logs, per-CalcFold validity records)
and USPEX 25 (``results1/Individuals``, ``gatheredPOSCARS``) as the validity
probe of 6 Nov 2026 recorded them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from backend.structure_search.bridge import evidence_archive as ea

KIT = Path(__file__).resolve().parents[1] / "reference" / "uspex_pilot"
COLLECTOR = KIT / "execution_validation" / "power_execution_evidence.sh"
NODE_MEMORY_MB = 512000
STEP_MEM = f"{int(NODE_MEMORY_MB * 0.9) // 4}M"
DIAMOND_PRIMITIVE = """{header}
1.0
0.0 2.735 2.735
2.735 0.0 2.735
2.735 2.735 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""
OTHER_STRUCTURE = """{header}
1.0
3.9 0.0 0.0
-1.95 3.3775 0.0
0.0 0.0 {c}
Si
2
Direct
0.0 0.0 0.0
0.3333333 0.6666667 0.5
"""


@dataclass
class Scenario:
    """What the fake run looks like. Defaults: a clean natural completion, diamond the best."""

    generation_sizes: tuple[int, ...] = (10, 11)
    listed_generations: tuple[int, ...] = (1, 2)
    rejected: dict = field(default_factory=dict)  # individual -> (step, classification)
    listed_rejected: dict = field(default_factory=dict)  # individual -> step whose energy is (wrongly) listed
    missing_steps: dict = field(default_factory=dict)  # individual -> step whose run is absent
    energy_offsets: dict = field(default_factory=dict)  # individual -> eV added to the listed energy
    best: int = 13  # the diamond, lowest energy
    diamond: bool = True
    slurm_state: str = "COMPLETED"
    exit_code: str = "0:0"
    outcome: str = "uspex_finished"
    master_status: int = 0
    stopped: bool = False
    unfinished_after: int | None = None  # individuals above this never ran (walltime)
    parallel: int = 4
    pin: str = "disjoint"  # disjoint | shared | none
    alloc_mem: str = "500G"
    # The frozen plan's node request and launch settings (defaults: 4 x 48 on an exclusive node).
    planned_parallel: int = 4  # what the plan and the launcher record; ``parallel`` is what actually ran at once
    ranks: int = 48
    node_memory_mb: int = 512000  # the job's memory as the launcher established it
    memory_source: str = "scontrol AllocTRES"
    cpu_offset: int = 0  # first CPU of the job's cgroup on the node (a shared node: not 0)

    runtime_status: str = "passed"
    timelimit: str = "04:00:00"
    # Identity and energy semantics.
    number_offset: int = 0  # USPEX's number = CalcFold individual + offset (renumbering)
    quantity: str = "per_cell"  # what Individuals' energy is: per_cell | per_atom
    semantics_tables: bool = True  # write Individuals_all / Individuals_extended (energy and energy_per_atom)
    table_inconsistent: bool = False  # Individuals_all's energy is neither energy_per_atom x N nor energy_per_atom
    same_energy: tuple = ()  # CalcFold individuals given exactly the step-5 energy of the first
    contcar_differs: tuple = ()  # surviving CalcFold individuals whose step-5 CONTCAR differs from gatheredPOSCARS
    contcar_swap: tuple = ()  # two surviving CalcFold individuals whose step-5 CONTCARs are exchanged
    no_contcar: bool = False  # surviving step-5 CalcFolds carry no CONTCAR
    listed_e0: bool = False  # USPEX lists E0 (sigma -> 0) instead of F
    deleted_logs: tuple = ()  # CalcFold individuals whose step logs are gone
    pressure: str = "0"
    # The launcher's evidence archive (written with the launcher's own writer).
    archive: bool = True  # False: a run from before the archive existed
    same_structure: tuple = ()  # CalcFold individuals relaxed to exactly the structure of the first
    archive_missing: tuple = ()  # (individual, step) pairs whose record was never written
    archive_tamper: tuple = ()  # (individual, step) pairs whose archived reasons differ from the journal's
    archive_interrupted: bool = False  # a write left unfinished and set aside on a launcher restart
    archive_conflict: bool = False  # a different record offered for a committed step
    survivor_altered: tuple = ()  # surviving CalcFold individuals whose step-5 CONTCAR is not the archived one
    continuity_break: tuple = ()  # (individual, step) pairs that did not start from the previous output
    crash_before_done: tuple = ()  # (individual, step): record committed, launcher died before the done file
    archive_duplicate: bool = False  # the first step's record offered twice, identically
    critical_line: bool = False  # the launcher could not journal an evidence event (stderr line)

    @property
    def ncpus(self) -> int:
        return self.planned_parallel * self.ranks

    @property
    def step_mem(self) -> str:
        return f"{int(self.node_memory_mb * 0.9) // self.planned_parallel}M"


SHARED_48 = dict(parallel=3, planned_parallel=3, ranks=16, node_memory_mb=32768, alloc_mem="32G",
                 memory_source="SLURM_MEM_PER_NODE", cpu_offset=72)


def energy(individual: int, step: int, scenario: Scenario) -> float:
    if scenario.same_energy and individual in scenario.same_energy:
        individual = scenario.same_energy[0]
    base = -10.84 if (individual == scenario.best and scenario.diamond) else -10.2 - 0.011 * individual
    return round(base + 0.05 * (5 - step) + 0.000000123 * individual, 8)


E0_SHIFT = -0.0042  # E0 (sigma -> 0) differs from F by the smearing entropy term


def structure(individual: int, scenario: Scenario, header: str) -> str:
    """The relaxed structure of a CalcFold individual: diamond for the best, otherwise a distinct cell."""

    if scenario.same_structure and individual in scenario.same_structure:
        individual = scenario.same_structure[0]
    if individual == scenario.best and scenario.diamond:
        return DIAMOND_PRIMITIVE.format(header=header)
    return OTHER_STRUCTURE.format(header=header, c=f"{3.0 + 0.05 * individual:.4f}")


def _write(path: Path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)


def build_run(run: Path, *, identity_run_dir: str, attempt: str, job: str, plan_digest: str, scenario: Scenario,
              incars: dict[str, str], job_script: str, acknowledgement: dict) -> dict:
    """Write the run directory; return the expectations and SLURM rows that match it."""

    search = run / "search"
    spool = run / "bridge_spool"
    STEP_MEM = scenario.step_mem
    ranks = scenario.ranks
    journal = [{"event": "started", "time": 1000.0, "max_parallel": scenario.parallel, "ranks": ranks},
               {"event": "master_started", "time": 1000.1, "pid": 4242, "argv": ["singularity", "exec"]}]
    clock = 1001.0
    slots_free = [clock] * scenario.parallel
    step_records = []
    individuals_by_generation = []
    number = 0
    for size in scenario.generation_sizes:
        generation = []
        for _ in range(size):
            number += 1
            generation.append(number)
        individuals_by_generation.append(generation)
    sacct_steps = []
    step_index = 0
    for generation in individuals_by_generation:
        generation_end = max(slots_free)
        slots_free = [max(t, generation_end) for t in slots_free]
        for individual in generation:
            if scenario.unfinished_after is not None and individual > scenario.unfinished_after:
                continue
            slot = min(range(scenario.parallel), key=lambda k: slots_free[k])
            t = slots_free[slot]
            for step in range(1, 6):
                if scenario.missing_steps.get(individual) == step:
                    continue
                rejected = scenario.rejected.get(individual)
                classification = rejected[1] if rejected and rejected[0] == step else "valid"
                status = {"valid": 0, "invalid_convergence": 118, "invalid_evidence": 119,
                          "execution_failed": 7}[classification]
                request_id = f"s{step}-{individual:04d}-{step_index:05d}"
                calcfold = f"{identity_run_dir}/search/Calculation/Calcfold_{individual}_{step}"
                seconds = 30.0
                argv = ["srun", "--exact", "--mpi=pmi2", "-n", str(ranks), f"--mem={STEP_MEM}", "vasp_std"]
                journal.append({"event": "started_step", "id": request_id, "step": step, "slot": slot,
                                "calcfold": calcfold, "argv": argv, "time": t, "pid": 1000 + step_index,
                                "environment_set": {"I_MPI_DEBUG": "4"}})
                reasons = [] if classification == "valid" else ["injected"]
                returncode = 7 if classification == "execution_failed" else 0
                journal.append({"event": "validity", "id": request_id, "step": step, "classification": classification,
                                "reasons": reasons, "time": t + seconds})
                free_energy = energy(individual, step, scenario)
                if scenario.pin == "none":
                    pins = ""
                elif scenario.pin == "partial" and step_index == 0:
                    pins = "[0] MPI startup(): Rank    Pid      Node name  Pin cpu\n" + "".join(
                        f"[0] MPI startup(): {r}       {2000 + r}    compute-0-271  "
                        f"{{{scenario.cpu_offset + ranks * slot + r}}}\n" for r in range(ranks - 8))
                else:
                    base = scenario.cpu_offset + ranks * (0 if scenario.pin == "shared" else slot)
                    pins = "[0] MPI startup(): Rank    Pid      Node name  Pin cpu\n" + "".join(
                        f"[0] MPI startup(): {r}       {2000 + r}    compute-0-271  {{{base + r}}}\n"
                        for r in range(ranks))
                if True:  # written, hashed for the record, then removed if the scenario loses it
                    _write(spool / "logs" / f"{request_id}.log",
                           pins + f"   1 F= {free_energy + 0.1:.8E} E0= {free_energy + 0.1 + E0_SHIFT:.8E}  d E =-.1E+00\n"
                           f"   2 F= {free_energy:.8E} E0= {free_energy + E0_SHIFT:.8E}  d E =-.1E-02\n")
                # The checker's verdict, the CalcFold's line of it, and the archive record (the launcher's writer).
                verdict = {"id": request_id, "calcfold": calcfold, "policy_id": "bmd_compute.structure_search_step_validity", "step": step,
                           "classification": classification, "reasons": reasons, "returncode": returncode,
                           "evidence": None if classification == "execution_failed" else {
                               "free_energy_ev": free_energy, "converged_electronic": True,
                               "converged_ionic": classification == "valid", "ionic_steps": 2}}
                line = json.dumps(verdict, sort_keys=True)
                relaxed = individual
                if step == 5 and individual in scenario.contcar_differs:
                    relaxed = individual + 1000
                if step == 5 and individual in scenario.contcar_swap:
                    relaxed = sum(scenario.contcar_swap) - individual
                output = structure(relaxed, scenario, f"Calcfold_{individual}_{step}").encode()
                given = (structure(individual + 2000, scenario, "random").encode() if step == 1
                         or (individual, step) in scenario.continuity_break else previous_output)
                previous_output = output
                log_path = spool / "logs" / f"{request_id}.log"
                log_facts = {"file": log_path.name, "bytes": log_path.stat().st_size,
                             "sha256": ea.sha256_file(log_path)}
                if individual in scenario.deleted_logs:
                    log_path.unlink()
                evidence_sha = None
                if scenario.archive and (individual, step) not in scenario.archive_missing:
                    in_facts = {"file": "POSCAR", "source": "POSCAR", "present": True, "bytes": len(given),
                                "sha256": ea.sha256_bytes(given), "archived": True}
                    out_facts = {"file": "CONTCAR", "source": "CONTCAR", "present": True, "bytes": len(output),
                                 "sha256": ea.sha256_bytes(output), "archived": True,
                                 "withheld": classification != "valid"}
                    archived_verdict = (dict(verdict, reasons=["something else"])
                                        if (individual, step) in scenario.archive_tamper else verdict)
                    record = ea.build_record(
                        request_id=request_id, step=step, calcfold=Path(calcfold), created=int(t), slot=slot,
                        argv=argv, started=t, finished=t + seconds, returncode=returncode, status=status,
                        reason="; ".join(reasons), validity=archived_verdict, validity_record_line=line,
                        step_log=log_facts, input_structure=in_facts, output_structure=out_facts)
                    files = {"POSCAR": given, "CONTCAR": output}
                    evidence_sha = ea.commit(search / ea.ARCHIVE_DIR, record, files)["record_sha256"]
                    journal.append({"event": "evidence", "id": request_id, "step": step, "outcome": "committed",
                                    "record_sha256": evidence_sha})
                    if scenario.archive_conflict and step_index == 0:
                        ea.commit(search / ea.ARCHIVE_DIR, dict(record, reason="offered again"), files)
                    if scenario.archive_duplicate and step_index == 0:
                        again = ea.commit(search / ea.ARCHIVE_DIR, record, files)
                        journal.append({"event": "evidence", "id": request_id, "step": step,
                                        "outcome": again["outcome"], "record_sha256": again["record_sha256"]})
                elif scenario.archive:
                    journal.append({"event": "evidence_write_failed", "id": request_id, "error": "OSError: full"})
                if (individual, step) in scenario.crash_before_done:
                    # The launcher died after committing the record; the next launcher answered "lost" (121).
                    journal.append({"event": "recovered", "id": request_id, "status": 121})
                    _write(spool / "done" / f"{request_id}.done", f"status=121\nid={request_id}\n"
                           "reason=claimed by an earlier launcher that did not finish it\n")
                    step_records.append({"individual": individual, "step": step, "id": request_id,
                                         "classification": classification, "free_energy_ev": free_energy,
                                         "line": line, "output": output})
                    sacct_steps.append(f"{job}.{step_index}|vasp_std|COMPLETED|0:0|2026-11-20T10:00:00|"
                                       f"2026-11-20T10:00:30|00:00:30|{ranks}|cpu={ranks},mem={STEP_MEM},node=1|")
                    step_index += 1
                    t += seconds
                    break
                journal.append({"event": "finished", "id": request_id, "slot": slot, "step": step,
                                "calcfold": calcfold, "status": status, "returncode": returncode,
                                "seconds": seconds, "validity": classification, "time": t + seconds})
                _write(spool / "done" / f"{request_id}.done",
                       f"status={status}\nid={request_id}\nstep={step}\ncalcfold={calcfold}\n"
                       f"returncode={returncode}\nstarted={t}\nfinished={t + seconds}\nvalidity={classification}\n"
                       + (f"evidence={evidence_sha}\n" if evidence_sha else ""))
                step_records.append({"individual": individual, "step": step, "id": request_id,
                                     "classification": classification, "free_energy_ev": free_energy,
                                     "line": line, "output": output})
                sacct_steps.append(f"{job}.{step_index}|vasp_std|COMPLETED|0:0|2026-11-20T10:00:00|"
                                   f"2026-11-20T10:00:30|00:00:30|{ranks}|cpu={ranks},mem={STEP_MEM},node=1|")
                step_index += 1
                t += seconds
                if classification != "valid":
                    break
            slots_free[slot] = t
    end = max(slots_free) + 1
    journal.append({"event": "master_finished", "returncode": scenario.master_status, "stopped": scenario.stopped,
                    "time": end})
    _write(spool / "launcher.jsonl", "".join(json.dumps(e) + "\n" for e in journal))
    _write(spool / "heartbeat", "1793961000\n")

    # USPEX's listing: complete candidates of the listed generations (or what the scenario says), under
    # USPEX's own numbers (CalcFold individual + number_offset).
    individuals_text = ("generation   number   num_atoms_all    energy   cell_volume   space_group            "
                        "origin   parents   e_above_hull   opt_property   quasi_entropy   a_order   s_order\n")
    all_text = "generation number energy energy_per_atom num_atoms_all expected_origin origin parents\n"
    gathered = []
    for g_index, generation in enumerate(individuals_by_generation, start=1):
        if g_index not in scenario.listed_generations:
            continue
        for individual in generation:
            if individual in scenario.listed_rejected:
                cell = energy(individual, scenario.listed_rejected[individual], scenario)
            elif individual in scenario.rejected or individual in {i for i, _ in scenario.crash_before_done} or (
                    scenario.unfinished_after is not None
                                                      and individual > scenario.unfinished_after):
                continue
            else:
                cell = energy(individual, 5, scenario) + scenario.energy_offsets.get(individual, 0.0)
            if scenario.listed_e0:
                cell += E0_SHIFT
            listed = cell / 2 if scenario.quantity == "per_atom" else cell
            number = individual + scenario.number_offset
            individuals_text += (f"{g_index:10d}{number:9d}             [2]{listed:10.3f}        40.000"
                                 f"             1    Random        []          0.000          0.000           "
                                 f"0.000     1.000     1.000\n")
            table_energy = listed + (0.37 if scenario.table_inconsistent else 0.0)
            all_text += (f"{g_index} {number} {table_energy:.6f} {cell / 2:.6f} [2] Random Heredity "
                         f"[{max(1, number - 2)}, {max(1, number - 1)}]\n")
            header = f"number={number}, energy={cell:.6f}, e_above_hull=0.0, opt_property=0.0"
            gathered.append(structure(individual, scenario, header))
    _write(search / "results1" / "Individuals", individuals_text)
    _write(search / "results1" / "BESTIndividuals", individuals_text)
    if scenario.semantics_tables:
        _write(search / "results1" / "Individuals_all", all_text)
        _write(search / "results1" / "Individuals_extended", all_text)
    _write(search / "results1" / "gatheredPOSCARS", "".join(gathered))

    # Surviving CalcFolds: the last generation's (USPEX removes earlier ones), each with its verdict records.
    verdicts: dict[str, int] = {}
    last_generation = set(individuals_by_generation[-1])
    for record in step_records:
        if record["individual"] not in last_generation:
            continue
        calcfold = search / "Calculation" / f"Calcfold_{record['individual']}_{record['step']}"
        _write(calcfold / "bmd_step_validity.jsonl", record["line"] + "\n")
        _write(calcfold / "OUTCAR", "free  energy   TOTEN  =  -10.8 eV  (VASP output: never copied)\n")
        _write(calcfold / "vasprun.xml", "<modeling>never copied</modeling>\n")
        _write(calcfold / "WAVECAR", b"\0" * 64)
        if record["step"] == 5 and record["classification"] == "valid" and not scenario.no_contcar:
            survivor = record["output"]
            if record["individual"] in scenario.survivor_altered:
                survivor = structure(record["individual"] + 3000, scenario, "altered").encode()
            _write(calcfold / "CONTCAR", survivor)
        verdicts[record["classification"]] = verdicts.get(record["classification"], 0) + 1

    if scenario.archive_interrupted:
        _write(search / ea.ARCHIVE_DIR / "interrupted" / "s1-0001-00000-1-2" / "POSCAR", "unfinished\n")

    for name, text in incars.items():
        _write(search / "Specific" / name, text)
    _write(search / "Specific" / "POTCAR_Si", "PAW_PBE Si 05Jan2001\n VRHFIN =Si: s2p2 (licensed: never copied)\n")
    input_text = (f"USPEX : calculationType\n\n% atomType\nSi\n% EndAtomType\n\n{scenario.pressure} : ExternalPressure\n"
                  "2 : minAt\n2 : maxAt\n\n10 : populationSize\n2 : numGenerations\n")
    _write(search / "INPUT.txt", input_text)
    _write(search / "prepared_inputs.json",
           json.dumps({"files": {"INPUT.txt": hashlib.sha256(input_text.encode()).hexdigest()}}))
    _write(run / "uspex_master.log", "** USPEX v25.0                        11 November 2025 **\nHappy USPEXing!\n")
    _write(run / f"slurm-{job}.out", f"structure-search job {job} for attempt {attempt} on compute-0-271\n"
           + ("BMD-LAUNCHER-CRITICAL: could not journal evidence {\"id\": \"s5-0013\"}: OSError: full\n"
              if scenario.critical_line else ""))
    _write(run / "job_started", f"{job}\n")
    _write(run / "bmd_search_job.sbatch", job_script)
    _write(run / "resource_acknowledgement.json", json.dumps(acknowledgement))
    status = {
        "schema": "bmd_compute.structure_search_job_status", "schema_version": 1, "attempt_id": attempt,
        "plan_digest": plan_digest, "slurm_job_id": job, "finished_at": "2026-11-20T11:00:00Z",
        "execution": {"outcome": scenario.outcome, "master_exit_status": scenario.master_status,
                      "launch": {"mode": "srun_steps", "parallel_relaxations": scenario.planned_parallel,
                                 "ranks_per_relaxation": ranks,
                                 "step_mem": STEP_MEM, "memory_margin_fraction": 0.1,
                                 "node_memory": {"mb": scenario.node_memory_mb, "source": scenario.memory_source,
                                                 "detail": f"mem={scenario.alloc_mem}"}}},
        "scientific_validity": {"status": "not_assessed", "step_validity_verdicts": verdicts},
        "isolation": {"jobflow": False, "atomate2": False},
    }
    _write(run / "search_status.json", json.dumps(status))
    runtime = {"status": scenario.runtime_status, "problems": [],
               "observed": {"slurm_version": "slurm 26.05.5", "i_mpi_pmi_library_exists": True,
                            "banner_version": "25.0"},
               "prepared": {"launch": {"mode": "srun_steps", "parallel_relaxations": scenario.planned_parallel,
                                       "ranks_per_relaxation": ranks}}}
    _write(run / "runtime_check.json", json.dumps(runtime))

    sacct = [f"{job}|bmds-{attempt[:8]}|{scenario.slurm_state}|{scenario.exit_code}|2026-11-20T10:00:00|"
             f"2026-11-20T11:00:00|01:00:00|{scenario.ncpus}|billing={scenario.ncpus},cpu={scenario.ncpus},"
             f"mem={scenario.alloc_mem},node=1|"
             f"{scenario.timelimit}",
             f"{job}.batch|batch|{scenario.slurm_state}|{scenario.exit_code}|2026-11-20T10:00:00|"
             f"2026-11-20T11:00:00|01:00:00|{scenario.ncpus}|cpu={scenario.ncpus},mem={scenario.alloc_mem},node=1|",
             *sacct_steps]
    return {"sacct": sacct, "steps": step_records}


FAKE_TOOL = r'''#!{python}
import json, os, sys
tool = os.path.basename(sys.argv[0])
state = json.load(open(os.environ["FAKE_SLURM_EXECUTION"]))
if tool == "squeue":
    if state.get("queued"):
        print(state["job"] + "|RUNNING")
    sys.exit(0)
print("\n".join(state["sacct"]))
'''


def install_slurm(bin_dir: Path, state_file: Path, *, job: str, sacct: list[str], queued: bool = False) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for tool in ("squeue", "sacct"):
        path = bin_dir / tool
        path.write_text(FAKE_TOOL.replace("{python}", sys.executable))
        path.chmod(0o755)
    state_file.write_text(json.dumps({"job": job, "sacct": sacct, "queued": queued}))


def localised_collector(*, workspace: Path, state_dir: Path, evidence_parent: Path) -> str:
    script = COLLECTOR.read_text()
    for old, new in {"WORKSPACE=/bmd/bmdguest/searches": f"WORKSPACE={workspace}",
                     "STATE_DIR=/bmd-db/guest/logs/structure_search/submission_attempts": f"STATE_DIR={state_dir}",
                     "PY=/bmd/bmdguest/envs/atomate2_remote/bin/python": f"PY={sys.executable}",
                     "EVIDENCE_PARENT=/bmd-db/guest/validation": f"EVIDENCE_PARENT={evidence_parent}"}.items():
        assert old in script, old
        script = script.replace(old, new)
    return script


def run_collector(tmp: Path, *, attempt: str, run_name: str, job: str, workspace: Path, state_dir: Path,
                  sacct: list[str], queued: bool = False) -> tuple[int, str, Path | None]:
    evidence_parent = tmp / "evidence_parent"
    evidence_parent.mkdir(exist_ok=True)
    bin_dir = tmp / "fake_bin"
    state_file = tmp / "fake_slurm_execution.json"
    install_slurm(bin_dir, state_file, job=job, sacct=sacct, queued=queued)
    environment = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                   "FAKE_SLURM_EXECUTION": str(state_file)}
    script = localised_collector(workspace=workspace, state_dir=state_dir, evidence_parent=evidence_parent)
    completed = subprocess.run(["bash", "-s", "--", attempt, run_name, job], input=script.encode(),
                               capture_output=True, env=environment)
    output = completed.stdout.decode() + completed.stderr.decode()
    directories = [line.split(": ", 1)[1] for line in output.splitlines() if line.startswith("EVIDENCE DIRECTORY: ")]
    return completed.returncode, output, Path(directories[-1]) if directories else None


def tree_state(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size, stat.S_IMODE(p.stat().st_mode))
            for p in sorted(root.rglob("*"))}


__all__ = ["Scenario", "build_run", "run_collector", "install_slurm", "localised_collector", "tree_state",
           "energy", "STEP_MEM", "SHARED_48", "shutil"]
