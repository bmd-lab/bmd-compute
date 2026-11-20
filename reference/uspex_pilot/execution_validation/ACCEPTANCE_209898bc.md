# Scientific execution acceptance: attempt 209898bc (PI-accepted with a qualification)

| | |
|---|---|
| Attempt | `209898bc-2793-416c-a53b-061e28e5e87f` |
| SLURM job | `23624369`, shared-node profile (3 × 16 ranks, 48 CPUs, `--mem=32G`), node `compute-0-270` |
| Run | `uspex_search-20261113-121025-209898bc`, 14 Nov 2026 08:48–11:44 UTC, USPEX 25.0, natural completion |
| Plan | `sha256:0b3b09de2b115cc8a72ea43b2e22ac5db2545e0d5012ff7057a0b6cbc8469e79` |
| Runtime package | `b7682918959e3079d31a9862c08ffd9f6b1e3b45905c2e323f99fa88b84c79c2` |
| Executed code | `ab3d264` (job start-up fix) |
| Analyser | `4d33ead` (`analyze_execution.py`, `compare_step_log_inspection.py`) |
| **Independent analyser result** | **INCOMPLETE EVIDENCE**: 19 checks verified, 0 failed, 1 missing (`pinning`) |
| **Decision** | **Accepted by the PI (Lee Burton), 18 Nov 2026**, as a documented exception for this pilot, with the CPU-placement qualification below |
| **What is accepted** | Successful **managed scientific execution**: BMD Compute planned, staged, submitted and ran a complete USPEX search on POWER as specified, and recorded verifiable evidence of it. This is **not** a validation of USPEX's predictive methodology or of the search's scientific results, which remain experimental |

The analyser result is preserved as computed and is not changed by this decision. The decision does
not relax the pinning criterion: the criterion and the analyser are unchanged, and any other attempt
must meet the criterion as written.

## Evidence (unchanged; SHA-256)

| Item | SHA-256 |
|---|---|
| Collected evidence and original analysis, `evidence-209898bc-20261114T123951Z.tgz` | `bfab0d2fb393409dab8ea7964f6562ad2686eb368a681d1959ee16e4261992ed` |
| First read-only step-log inspection (6 samples per log, 17 Nov), `step-log-inspection-209898bc.txt` | `159ecd9a18b795f021439467f61a505849079e87d36ab720509665f8a89ff29a` |
| Second read-only inspection, complete pinning tables (18 Nov, `4d33ead`), `step-log-inspection-209898bc-tables.txt` | `e093aef2d0a0b82ce1c8eaf74b55a6a85d770b2e866cabc8f047595c1ea9d154` |
| Hash comparison of the second inspection with the evidence, `compare-step-log-inspection-209898bc.txt` | `ee5136fda65550e5f92791cd06869c0fff55d1192af7eff7948e9785d568539d` |
| Pinning supplement, `pinning-supplement-209898bc.json` | `80ef69b767bd4a2b63ff8dfba419a7643c5254a29a00a5f960f17b204c174c00` |
| Reanalysis report with the supplement, `report_209898bc_with_pinning.json` | `2e62da0c474248eaf816de8bbabccad0fbf1fa4089725f3b94aec53185c8caf0` |
| Reanalysis driver (expectations from the frozen plan), `reanalyse-209898bc.py` | `f531acfa1944a5fe236838d79f28f92f00ec7d1334068de22b45cc12586f967f` |

The derived files are kept beside the evidence in `Claude outputs/uspex_probe/`. The completed run
directory on POWER was only read (both inspections confirmed it unchanged).

## What the evidence establishes

- **Execution:** the job completed naturally (SLURM `COMPLETED`, USPEX exit 0, both generations
  listed). Steps ran at most 3 at a time, within the 48-CPU allocation and the per-step memory.
- **Step accounting (109 recorded, 108 executed successfully):**

| Count | What |
|---|---|
| **109** | VASP steps **recorded**: 109 launcher steps, 109 SLURM job steps (`sacct`), 109 step logs, 109 archived step records, each 16 MPI ranks |
| **108** | steps in which VASP **exited successfully** (return code 0): 107 judged `valid` and 1 judged `invalid_convergence` (candidate 2, step 4: NSW = 99 reached; it ran correctly and was rejected by the validity rules, as designed) |
| **1** | step that **did not execute successfully**: candidate 22, step 5 (`s5-1794656526-1771089-eb17e47d3b8c`, SLURM step `23624369.108`). VASP exited with status 1 after 29.6 s; it was classified `execution_failed`, and USPEX was answered status 1 |

  The 108 successful steps are the ones the pinning check requires to have an interpretable
  table. The failed step also left a complete 16-rank table (CPUs from 20) and was compared with
  every other step; it does not count toward the 108.
- **Science, operationally:** 22 candidates. 20 completed the five-step ladder. One was rejected at
  step 4 (`invalid_convergence`, NSW reached) and one at step 5 (`execution_failed`).
  - Every listed candidate is associated with exactly one accepted chain at its step-5 F.
  - Every listed candidate's records and final structure are kept in the verified archive.
  - Best: individual 1, −10.259 eV per cell (−5.1295 eV/atom), Imma at 0.1 Å tolerance (C2/m at 0.01).
  - Diamond was not found. That is recorded, not required.
- **Structural continuity:** every step-5 input is step 4's output: 18 byte- or component-identical,
  and 3 in another lattice basis established explicitly (D1).
  - USPEX changed the structure on all 66 intermediate transitions: largest principal strain 2.3 %,
    largest volume change 2.3 %, largest atomic displacement 0.084 Å.
  - These changes are accepted provisionally as experimental methodology (D2). They are reported,
    not validated.
- **Qualifications carried by the analysis:**
  - **`identities`:** used the job-script and acknowledgement hashes from the evidence, because the
    VM ledger was not consulted in the reanalysis; the original ledger-based analysis verified them.
  - **`isolation`:** attributes the recorded `atomate2` import to the documented POTCAR-table
    import; the record cannot prove that.

## CPU placement: the qualification

- **Recovered:** all 109 Intel MPI 2021.10 pinning tables (one per recorded step) were read from the
  original step logs by a read-only inspection.
  - Every log's SHA-256 and size equal the collector's `step_logs.json` and the launcher's archived
    record (109 of 109).
  - The original collector had missed the tables because Intel MPI under `srun` prints bare CPU
    numbers, not `{cpus}`.
- **Interpretable:** all 108 successfully executed steps have a usable table: ranks 0–15 once each,
  distinct PIDs, one host, 48 distinct logical CPUs in total. No step has two ranks on one CPU. The
  109th, failed step's table is also complete and was compared with every other step (no overlap);
  it is not one of the 108.
- **No definite overlap:** among steps sharing CPUs, none is proven to have overlapped in time.
  1,937 pairs are reuse by steps that did not overlap.
- **One unresolved pair:**

| SLURM step | Launcher step | CPUs | sacct start | sacct end |
|---|---|---|---|---|
| `23624369.8` | `s3-1794646940-1553286-17b7d51029bf` (candidate 3, step 3) | 12–19, 108–115 | 09:02:25 | **09:02:59** |
| `23624369.10` | `s4-1794646976-1554753-338358bc1c58` (candidate 2, step 4) | 12–19, 108–115 | **09:02:59** | 09:43:02 |

The two steps used the same 16 CPUs. `.8` ends and `.10` starts in the same second of sacct's
whole-second record, so the analyser cannot establish whether they overlapped. The launcher's own
interval for `.8` ends at 09:03:05.135, after its validity check, so it does not settle it either.
Neither the journal nor the archived record holds the moment `.8`'s `srun` returned.

**Why the evidence strongly supports sequential reuse:**

1. Each step was launched with `srun --exact`, and POWER confines steps with `ConstrainCores=yes`.
   SLURM assigns a new step only CPUs of the job that no running step holds.
2. When `.10` was created, the job's other 32 CPUs were held as follows:
   - **`.9`:** CPUs 20–23, 72–75, 116–119, 168–171, held from 09:02:57.
   - **Free:** CPUs 76–83 and 172–179. No step had held them since 09:01:56, and `.11` took them
     next, at 09:03:32.
3. SLURM nevertheless gave `.10` exactly `.8`'s CPUs, rather than the free set. With `--exact` and
   `ConstrainCores`, that is what happens when `.8`'s CPUs have already been released, that is, when
   `.8` had ended.
4. `.10`'s `srun` started at 09:02:59.684 on the launcher's clock. A real overlap would therefore need
   `.8` to have run past that instant, within the last ~0.3 s of its recorded second. This comparison
   crosses two clocks (compute node and SLURM controller); it supports the reading but does not
   prove it.

**What this does not establish:** the argument rests on SLURM's allocation behaviour and on the
consistency of two clocks. It does not independently establish the sub-second order of `.8`'s end
and `.10`'s start. The frozen criterion asks for that order from the recorded intervals, and they
cannot give it. That is why the analyser's result remains INCOMPLETE EVIDENCE.

## Decision

The PI accepts attempt 209898bc as the scientific-execution acceptance of the shared-node USPEX pilot,
with the CPU-placement qualification above, as an exception specific to this attempt:

- **The analyser result stands:** INCOMPLETE EVIDENCE (`pinning` missing for one pair). It is
  recorded alongside the decision, not replaced by it.
- **No relaxation:** the pinning criterion is not relaxed. Future attempts must show disjoint CPU
  placement for every genuinely concurrent pair from recorded evidence. The proposed `srun`
  timestamps below are intended to make that possible.
- **Evidence preserved:** the evidence, the original analysis, both inspections, the comparison, the
  supplement and the reanalysis report are kept unchanged at the hashes above.
- **Scope:** this acceptance covers managed scientific execution of the pilot as specified: that
  the system ran the search correctly and recorded verifiable evidence. It is not general validation
  of USPEX's predictive methodology, and makes no claim that the search finds ground states. The
  USPEX search methodology remains experimental:
  - the five-step ladder and defaults stay provisional (methodology §10);
  - the intermediate USPEX transformations are accepted provisionally (D2);
  - the resource request stays a provisional estimate.

## Proposed diagnostic for future runs (not implemented)

Record when each `srun` step started and returned, separately from the validity check that follows.

- **What:** in `backend/structure_search/bridge/launcher.py`, the launcher already journals
  `started_step` before `Popen` and `finished` after the validity check and evidence archive. Add one
  journal event, `srun_returned`, and the same field in the archived record's `launch` block:
  - the step's ID;
  - the return code;
  - the wall-clock time (`time.time()`) taken immediately after the `srun` process is reaped,
    before validity checking starts;
  - the monotonic duration since the `Popen` call.
- **Unchanged:** nothing in the `srun` command line, the environment, CPU binding, VASP settings or
  USPEX parameters. The existing Intel MPI pinning-table collection stays.
- **Effect:** the analyser can bound each SLURM step by `[started_step, srun_returned]` on one clock,
  with no cross-clock assumption, instead of `[started_step, finished]`, which includes the validity
  check (up to 73 s here). Same-second sacct boundaries like `.8`/`.10` are then settled when the
  launcher intervals are disjoint.
- **Analyser side:** use `srun_returned` when present. Older evidence keeps today's behaviour.
- **Cost:** it changes the runtime package digest. It therefore needs a reviewed commit, a new frozen
  plan, and the usual acceptance before use.
