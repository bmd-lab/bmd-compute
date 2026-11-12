# Bounded scientific execution acceptance (design, not yet implemented)

Status: approved (11 Nov 2026); tooling built, not yet run on POWER (see
`README.md`). Four hours is the maximum allocation, not a predicted
completion time. Diamond is recorded (found, and whether lowest in energy),
not required for operational success. The
operational pathway (submit, monitor, cancel) passed on POWER on 10 Nov 2026
(`../staging_validation/RESULTS_20261110_c6edaf94.md`), but that job never
started; this test is the first managed search that executes USPEX and VASP.

## The search

Si, through the existing planner's Advanced settings only:

| Setting | Value | Why |
| --- | --- | --- |
| `min_atoms`, `max_atoms` | 2, 2 | the diamond primitive cell; the smallest cell in which the known ground state exists (1-atom cells cannot hold it, and USPEX's variation operators have little to act on) |
| `num_generations` | 2 | the smallest that exercises generation progression (1 runs only the initial population) |
| `walltime` | `04:00:00` | see the estimate below |

Fixed by policy (not overridable): `populationSize` = `initialPopSize` = 10
(USPEX's 2 × N rounded to 10, BMD floor of 10, for any `max_atoms` ≤ 7);
`stopCrit` = `max_atoms` = 2; the five-step ladder (INCAR_1–5), KresolStart
(0.12, 0.10, 0.08, 0.04, 0.04), POTCAR `Si` (PBE_64), the candidate-validity
rules (electronic convergence and a valid energy at every step, ionic
convergence at step 4), ambient pressure, plain non-magnetic PBE; the
runtime policy (`srun_steps`, 4 relaxations × 48 ranks, bridge step timeout
4 h, `sleepSeconds` 30, `numParallelCalcs` 4); the resource preset (one
exclusive 192-core node, `--mem=0`, `owner100d`). The planner gives plan
digest `sha256:1989c888f58012590920ebbf4b5ca31c2d5aa0dd53449cc6bcbe3951b9b4898e`
for exactly these four overrides (commit `d1ea514`); it differs from the
default Si plan only in those fields.

## Size and time

- Structures: about 10 (generation 1) + 10–11 (generation 2, including the
  kept best, which USPEX recalculates) ≈ 21; VASP runs ≤ 21 × 5 ≈ 105, fewer
  if candidates are rejected (a rejected chain stops at its failed step).
- Measured so far (validity probe, job 23406856: Si8, the pilot's 3-step
  INCARs, 4 ranks, 2 at a time): 26 runs in 43.6 min; per step mean 17 s,
  148 s and 259 s, maximum 745 s. The BMD ladder has never run under USPEX,
  and 48-rank runs of 2-atom cells have never been timed. Steps 4–5
  (`PREC = Accurate`, `EDIFF = 1e-6`, 0.04 k-resolution, 620 eV static) are
  denser in k-points but the cell is 4× smaller.
- Estimate: 5–20 min per chain, about 6 rounds of 4 parallel chains plus
  the idle time at each generation's end and USPEX's 30 s polling: 0.5–2 h.
  `04:00:00` is about twice the pessimistic estimate and bounds the cost at
  768 core-hours. 15 minutes is not enough: one probe relaxation alone took
  12.4 minutes, and generation 1 needs at least three rounds.

## Complete or stopped by the walltime

| Evidence | Natural completion | Walltime |
| --- | --- | --- |
| SLURM state (settled, `sacct`) | `COMPLETED`, exit 0:0, end well before start + 4 h | `TIMEOUT` |
| `search_status.json` | `uspex_finished`, `master_exit_status` 0 | `stopped` (123), or still `running` if SLURM killed it first |
| launcher journal | `master_finished`, returncode 0, `stopped: false` | `stopped: true`, or no `master_finished` |
| USPEX `results1/Individuals` | generations 1 and 2 listed | generation 2 missing or incomplete |
| BMD Compute state | `SEARCH_COMPLETED` | `SEARCH_TIMED_OUT` |

`SEARCH_COMPLETED` means the USPEX search ran to its end; it does not
establish scientific success (`scientific_validity` stays `not_assessed`).

## Evidence required, by item

1. Allocation and compute-node checks: `runtime_check.json` passed on the
   compute node, including the MPI library and SLURM version (the checks the
   login node defers); `search_status.json` host and modules.
2. Memory under `--exclusive --mem=0`: `search_status.json` `launch.node_memory`
   (source `SLURM_MEM_PER_NODE`, or `scontrol` AllocTRES under `--mem=0`), the
   step memory (node memory × 0.9 / 4), compared with `scontrol show job` /
   `sacct` AllocTRES; no step refused for memory.
3. USPEX startup and the bridge: master log banner (USPEX 25.0), no
   start-up error; the managed `INPUT.txt` (including `value : sleepSeconds`)
   accepted; `run_local.sh` calls reaching the bridge (spool requests and
   done files).
4. Parallel VASP and all five steps: `sacct` job steps, each 48 CPUs, at most
   4 overlapping, one per VASP run; `I_MPI_DEBUG=4` pinning showing no core
   shared between concurrent steps; for every candidate that reached step 5,
   one run of each step 1–5 with the plan's INCAR and KresolStart.
5. Validity checks: a verdict for every step (`bmd_step_validity.jsonl`),
   counts in `search_status.json`; every rejected step (118, 119 or VASP's
   status) stops its chain, is not retried, is not listed, and no stale
   energy is used. Failures cannot be injected in the managed path; if none
   occur naturally, the forced cases remain those of the validity probe.
6. Generations and termination: `Individuals` lists generations 1 and 2;
   master exit 0, not stopped.
7. Status and provenance: `search_status.json` (plan digest, acknowledgement
   SHA-256, job ID, modules, launch, verdict counts, isolation), the job
   script and acknowledgement unchanged (hashes), `job_started` = job ID.
8. Monitoring: observations through `RUNNING` to settled `COMPLETED` /
   `SEARCH_COMPLETED`, identity `verified` while queued and `partial` after
   (POWER's accounting keeps no comment), times consistent, binding `sbatch`.
9. Independent inspection: every listed energy equals the step-5 TOTEN of
   that candidate's chain (from OUTCAR/vasprun, read on POWER, hashed, not
   copied); the best structure's symmetry (spglib through pymatgen on POWER;
   diamond, Fd-3m, expected) and energy per atom recorded; master-log error
   lines classified.

## What the implementation can already do

Monitoring and classification need no change: settled `COMPLETED` with
`uspex_finished` and master status 0 gives `SEARCH_COMPLETED`; `TIMEOUT`
gives `SEARCH_TIMED_OUT`; a finished job's `partial` identity is accepted for
an already-bound job. The job records the verdict counts and launch
settings.

## Tooling (built 11 Nov 2026)

1. The kit's `execution` profile (`setup --execution`): the four Advanced
   settings above, the plan digest pinned; `submit` also needs
   `--confirm-walltime 04:00:00`, re-plans, and checks the exact resource
   request, the frozen spec and launch settings, the acknowledgement and the
   rebuilt job script before the one `sbatch`. Cancellation keeps its gate.
2. `power_execution_evidence.sh` (read-only collector) and
   `analyze_execution.py` (local analyser), run by the kit's
   `execution-evidence` command after the job has ended.
3. `backend/structure_search/results.py` still cannot read
   `Individuals_all`; the analyser reads `Individuals` and `gatheredPOSCARS`
   itself (deferred: changing `results.py` would change the runtime package).
4. Hardened before review (11 Nov 2026): candidate identity is established
   from energy, the recorded structure against surviving step-5 CONTCARs and
   a numbering convention only where the survivors confirm it, never assumed;
   the listed quantity (per cell or per atom, F not E0, zero pressure, 2 atoms)
   is established from `Individuals_all`/`_extended` before any comparison;
   retention through the launcher's journal, done files and step logs is
   checked for every listed candidate; incomplete or unreadable MPI pinning
   reports are `missing`.
5. Evidence archive (runtime change, 11 Nov 2026): the launcher commits each
   step's original verdict, input POSCAR and output CONTCAR with their hashes
   to `search/bmd_evidence/` before the done file, so removed CalcFolds no
   longer take them away; the analyser's `evidence_archive` check verifies it
   and `retention` relies on it. Runtime package `92913a28…` (80 files); the
   pinned plan digest is unchanged, but every attempt staged with `fca63b4b…`
   is refused (`runtime_package_changed`) and the test needs a new attempt.

6. Shared node (12 Nov 2026, approved): the kit's `execution_shared`
   profile asks for 3 × 16 ranks (48 CPUs) and `--mem=32G` on a shared node,
   with the same search, ladder and validity policy (plan
   `sha256:0b3b09de…`; INCAR files byte-identical to the 4 × 48 plan's).
   The kit sets the profile's node settings in its own process before
   planning; the job reads them from the frozen plan. The analyser takes
   CPUs, concurrency and job memory from the frozen plan. The 4 × 48
   exclusive profile stays, unchanged, as a later full-node benchmark.

## First contact on POWER (risks this test exists to retire)

- 4 × 48-rank `srun_steps` placement on a full node, and Intel MPI pinning
  (open question in `docs/structure_search.md` §9).
- Allocated-memory detection under `--mem=0` on a real allocation.
- The managed `INPUT.txt` read by real USPEX (the probe wrote its own input;
  the `value : sleepSeconds` form is unconfirmed).
- The BMD five-step ladder under USPEX (the probe used the pilot's three
  steps), including `KPAR = 4` with 48 ranks on 2-atom cells.
- The production USPEX cache used by a job for the first time (the probe
  used a private cache).

- The evidence archive on POWER's file system: its fsyncs and renames per
  step, and whether USPEX writes each next step's POSCAR as the previous
  CONTCAR (the `evidence_archive` check compares them structurally; a USPEX
  rewrite that changes the cell would show as a continuity failure to examine).

Any of these can end the job early with `uspex_failed`, `launch_setup_failed`
or rejected steps; each is a finding to examine, not something to retry
blindly.
