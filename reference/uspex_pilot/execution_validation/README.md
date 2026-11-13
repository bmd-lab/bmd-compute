# Scientific execution acceptance: one small managed Si search, run to its natural end

The first managed search that executes USPEX and VASP on POWER. The design and
its estimates are in [`DESIGN.md`](DESIGN.md). The operational pathway
(submit, monitor, cancel) passed on 10 Nov 2026
(`../staging_validation/RESULTS_20261110_c6edaf94.md`).

| | |
| --- | --- |
| Search | Si; `min_atoms` 2, `max_atoms` 2, `num_generations` 2 (the planner's Advanced settings); population 10 and `stopCrit` 2 by policy |
| Profiles | `--execution-shared` (**approved, 12 Nov 2026**): 3 parallel relaxations × 16 ranks = 48 CPUs on a shared node, `--mem=32G`, no `--exclusive`; `--execution`: 4 × 48 = 192 CPUs on one exclusive node (job 23590056's profile) |
| Plan digest (pinned in the kit) | shared: `sha256:0b3b09de2b115cc8a72ea43b2e22ac5db2545e0d5012ff7057a0b6cbc8469e79`; exclusive: `sha256:1989c888f58012590920ebbf4b5ca31c2d5aa0dd53449cc6bcbe3951b9b4898e` |
| Resources (shared) | `-p leeburton-pool --account=power-leeburton-users_v2 --qos=owner100d --time=04:00:00 --nodes=1 --ntasks=48 --mem=32G --exclude=compute-0-[332-333]`; request hash `sha256:d4eaedfd…` |
| Launch (shared) | `srun_steps`: 3 parallel relaxations × 16 ranks, per-step memory = 90 % of 32 GB / 3 = 9830M; no rank reserved for the USPEX master and launcher (batch step); no core numbers set. Concurrent placement and memory on a shared POWER node are still to be validated by this run (`parallel`, `pinning`, `memory`, step `MaxRSS`) |
| Science | the five-step ladder, KresolStart, Si POTCAR and validity rules, unchanged |
| Walltime | 4 h is the maximum allocation, not a prediction (estimate 0.5–2 h) |

## Files

| File | Role |
| --- | --- |
| `power_execution_evidence.sh` | read-only POWER collector, run over SSH by the kit after the job has ended |
| `analyze_execution.py` | local analyser of the fetched evidence (standard library; pymatgen for symmetry) |

The kit commands live in `../staging_validation/run_staging_validation.py`
(`setup --execution-shared` or `--execution`, `run`, `promote`, `evidence`, `submit
--confirm-attempt ID --confirm-walltime 04:00:00`, `observe`, `cancel`,
`execution-evidence`, `archive`), with the same isolated root, ledger, token
and principal as every other part of the kit.

## What the collector takes, and what it never takes

Copied (each with its size and SHA-256 in `manifest.json`, bounded, regular
files only): `search_status.json`, `runtime_check.json`, `job_started`, the
job script and resource acknowledgement, the USPEX master log, the SLURM
output, the job's start-up log (`job_setup.log`), the launcher journal, done files and heartbeat, `INPUT.txt`, the
frozen plan and preparation manifest, `Specific/INCAR_*`, USPEX's
`results*/{Individuals, BESTIndividuals, Individuals_all, Individuals_extended,
gatheredPOSCARS, BESTgatheredPOSCARS, Parameters.txt, generation_properties}`,
the surviving CalcFolds' `bmd_step_validity.jsonl` and `bmd_launch.jsonl`, the
step-5 `CONTCAR` of surviving CalcFolds (a relaxed cell, 1 MiB cap), and the archived
`validity.json` of rejected steps; the launcher's evidence archive
(`search/bmd_evidence/`: `index.jsonl` and each record's `record.json`,
`POSCAR` and `CONTCAR`, 1 MiB cap each; the names of unfinished writes); the remote attempt state; `sacct` for the
job and every job step (UTC); the clocks. For every VASP step log only a
summary: size, SHA-256, final F and E0 and ionic-step count (VASP's own
`F=`/`E0=` lines), Intel MPI pinning rows, error lines.

Never copied: any POTCAR, WAVECAR, CHGCAR, OUTCAR, `vasprun.xml` or other VASP
output, the step logs themselves, credentials. It refuses to run while the
job is queued, and checks that the run directory is unchanged afterwards.

## What the analysis checks

Each check is `verified`, `failed` or `missing` (evidence absent):

| Check | Requirement |
| --- | --- |
| `collection` | allowlisted records only; run directory unchanged |
| `slurm_outcome` | SLURM's settled state and exit code (a fact, not the verdict) |
| `allocation` | the frozen plan's CPUs (concurrency × ranks: 48 shared, 192 exclusive), one node, time limit 04:00:00 |
| `runtime_check` | passed on the compute node, MPI library and SLURM version included, 4 × 48 |
| `memory` | job memory recorded, equal to SLURM's AllocTRES and (explicit `--mem`) to the plan's, per-step memory = 90 % / concurrency |
| `uspex_startup` | USPEX 25.0 banner, master started, `INPUT.txt` the managed one |
| `bridge` | every bridge request ran once and was answered; no rejected, lost or abandoned steps |
| `vasp_steps` | every step an `srun` of 48 ranks with the per-step memory, a SLURM step of 48 CPUs, VASP printed a final F |
| `parallel` | between 2 steps and the plan's concurrency at a time |
| `pinning` | every step's Intel MPI report lists ranks 0–47 once, on one host, each with cores; concurrent steps on disjoint cores. Overlap fails; an absent, partial or unreadable report is `missing`, never success |
| `validity_decisions` | one decision per step, the same in journal, done file and status; done status 0 / 118 / 119 / non-zero |
| `chains` | each candidate ran steps 1, 2, … in order, once each, and stopped at its first step not accepted; a step is accepted only with completion evidence: the `finished` event and the done file USPEX read both say `valid`, status 0 (an archived verdict alone is not delivery) |
| `energy_semantics` | the quantity USPEX lists is established from its own tables, not assumed (below) |
| `listed_energies` | every listed candidate is associated with exactly one accepted chain (five valid steps), at that chain's step-5 F (below) |
| `evidence_archive` | every step the journal started has one committed, indexed archive record whose files match their hashes and whose identity, outcome and verdict equal the journal, done file and step log; a surviving CalcFold's verdict line and step-5 CONTCAR are the archived ones byte for byte; the checker's F equals VASP's printed F; each step started from its predecessor's output structure. Contradictions and conflicts fail; absent (`evidence_write_failed`), unindexed, interrupted, unverifiable or undelivered records (committed, but no `finished` event: the launcher died before the done file) and `BMD-LAUNCHER-CRITICAL` lines in the SLURM output are `missing`, never a scientific rejection |
| `retention` | every listed candidate's five steps are verified in the archive and its final relaxed structure kept, including candidates whose CalcFolds USPEX removed; each bound to its generation through USPEX's listing |
| `generations` | generations 1 and 2 listed, one results folder, master exited 0, not stopped |
| `search_status` | this attempt, job and plan; finished; jobflow and atomate2 not imported |
| `identities` | `job_started`, job script, acknowledgement and INCAR ladder are the frozen ones |

Classification: `natural_completion` (SLURM `COMPLETED` 0:0 **and**
`uspex_finished` with master status 0 **and** every generation listed),
`stopped_by_walltime` (SLURM `TIMEOUT`) or `failed_or_incomplete`. Result:
`PASSED` only for a natural completion with nothing failed or missing;
`FAILED` for anything failed or any other ending; `INCOMPLETE EVIDENCE`
otherwise. Findings (recorded, not required): candidates, rejections by step
and reason, step durations, the best candidate's energy, energy per atom and
space group (two tolerances), and whether a diamond-like structure appears
and is the lowest in energy.

## Candidate identity, energy semantics and evidence retention

**Identity.** USPEX's individual number is not assumed to be the CalcFold
number (survivors of `keptBest` are renumbered, for one). Each listed row is
associated with an accepted chain (a CalcFold individual whose five steps were
all `valid`) from evidence:

1. its energy, normalised to the cell, against every accepted chain's step-5 F
   (`gatheredPOSCARS`' 6-decimal value, ±1e-4 eV, else `Individuals`' 3-decimal
   value, ±1.5 meV, scaled with the normalisation);
2. the structure USPEX recorded (`gatheredPOSCARS`) against each accepted
   chain's archived step-5 `CONTCAR` (a surviving CalcFold's only where the
   archive has none; same cell, same atoms);
3. the naming convention, used only where the survivors confirm it (structure,
   energy and number agree for at least one, and contradict it for none).

Structure and energy agreeing on one chain, or energy matching exactly one
chain when every accepted chain has a known F, establishes the association.
Two listed rows on one chain, a chain of a later generation started before
one of an earlier generation, no matching accepted energy (with the stale
match to a rejected chain's earlier step named), or a structure that
contradicts the chain the energy points to, is `failed`. More than one chain
and nothing to decide, or chains whose F is unknown, is `missing` (`INCOMPLETE
EVIDENCE`).

**Energy semantics.** USPEX 25 writes `energy` (per cell, eV; enthalpy at the
run's pressure) beside `energy_per_atom` and `num_atoms_all` in
`Individuals_all`/`Individuals_extended` (si_bridge sample; the validity probe
matched listed energies to the cell's F). The quantity is established only if
every row there shows `energy` = `energy_per_atom` × atoms (per cell) or =
`energy_per_atom` (per atom); atoms per cell equal the plan's 2 (also in
`Individuals` and `gatheredPOSCARS`); `INPUT.txt` has `ExternalPressure` 0 (so
enthalpy is energy) and `minAt` = `maxAt` = 2; `Individuals` and
`gatheredPOSCARS` agree with those tables. A listing that equals VASP's E0
rather than F (they differ by the smearing term) fails. A numerical match is
the comparison, not the proof: the quantity must be established first, or
nothing is compared and the result is `INCOMPLETE EVIDENCE`.

**Retention.** USPEX keeps only the last generation's CalcFolds. The launcher
commits each step's record (original verdict, input and output structures,
hashes) to `search/bmd_evidence/` before USPEX can remove the CalcFold
(`docs/structure_search.md`, scientific evidence archive). The analysis
verifies that archive against the launcher's journal, done files and VASP's
step logs and the surviving CalcFolds, and uses its step-5 CONTCARs as the
recorded structures of every accepted chain, so identity by structure holds
for every generation, not only the last. Generation is not known to the
launcher (USPEX does not pass it): archived records say `generation: null`.
The analysis reports the generation USPEX listed for each associated
candidate separately, as `inferred_generation_association`, never as part of
execution identity; a candidate whose association is ambiguous gets none and
makes the result `INCOMPLETE EVIDENCE`.

## Procedure (VM, bash; stops before each gate)

A new, isolated attempt in a new validation root: never an earlier attempt
or root (the kit refuses `5ead0670`, `14bfbbac` and `c6edaf94`, and every
attempt staged with an earlier runtime package is refused by the service with
`runtime_package_changed`). Earlier roots, attempts and POWER evidence are
left as they are. The reviewed commit reaches the VM as a Git bundle (compare
its SHA-256 first; cloning reads and checks it).

```bash
# E0. The live service's interpreter (read only) and a separate clone of the reviewed commit.
LIVE_PID=$(pgrep -f 'uvicorn main:app' | head -1)
BMD_PY=$(readlink -f /proc/$LIVE_PID/exe)
"$BMD_PY" -c "import httpx, paramiko, starlette.testclient; print('client libraries present')"
BUNDLE=~/bmd-compute-<commit12>.bundle                  # copied from Windows; SHA-256 as reported
sha256sum "$BUNDLE"
COMMIT=<the reviewed commit>
git clone --quiet "$BUNDLE" ~/bmd-execution-<commit12>
git -C ~/bmd-execution-<commit12> checkout --quiet --detach "$COMMIT"
cd ~/bmd-execution-<commit12> && git log -1 --oneline && git status --porcelain --untracked-files=no
K=reference/uspex_pilot/staging_validation/run_staging_validation.py

# E1. A new private execution root (Si 2/2 atoms, 2 generations, population 10, 04:00:00, plan digest pinned).
mkdir -p -m 700 ~/bmd-validation
VROOT=~/bmd-validation/uspex-execution-$(date -u +%Y%m%dT%H%M%SZ)
env -u BMD_API_STATE_DIR -u BMD_API_TOKENS_FILE "$BMD_PY" $K setup "$VROOT" --execution-shared

# E2. Read-only POWER preflight.                      Expect: PREFLIGHT PASSED
"$BMD_PY" $K preflight "$VROOT"

# E3. Plan, prepare, stage, read back, re-verify.     Expect: plan_digest sha256:1989c888…, stage HTTP 200,
#     fresh re-verification on POWER: passed. Refuses any plan but the pinned digest.
"$BMD_PY" $K run "$VROOT" 2>&1 | tee "$VROOT/run.log"
python3 -c "import json,glob,sys; s=json.load(open(sorted(glob.glob(sys.argv[1]+'/evidence/*-summary.json'))[-1])); print(s['plan_digest'], s['runtime_package_manifest_sha256'])" "$VROOT"
#     Expect: sha256:1989c888f58012590920ebbf4b5ca31c2d5aa0dd53449cc6bcbe3951b9b4898e  <the reviewed runtime digest>

# E4. Acknowledge the resource request and promote.   Expect the #SBATCH lines of the table above
#     (--time=04:00:00 --nodes=1 --ntasks=48 --mem=32G --exclude=compute-0-[332-333], no --exclusive) and
#     the job script SHA-256. Any 409 stops.
"$BMD_PY" $K promote "$VROOT" 2>&1 | tee -a "$VROOT/run.log"

# E5. Independent, read-only POWER evidence.           Expect: EVIDENCE CHECKS PASSED, remote state
#     SEARCH_PREPARED, the job script and acknowledgement those of the promotion, no job started,
#     the run directory unchanged.
"$BMD_PY" $K evidence "$VROOT"
ATTEMPT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['attempt_id'])" "$VROOT/evidence/attempt.json")
echo "$ATTEMPT"

# E6. Read-only SLURM checks (as bmdguest): partition, nodes, association, QOS, queue. Nothing is changed.
ssh powerslurm-bmdguest 'date -u +%FT%TZ;
  scontrol show partition leeburton-pool | grep -E -o "(MaxTime|DefaultTime|State|TotalCPUs|TotalNodes)=[^ ]+";
  sinfo -p leeburton-pool -N -h -o "%N %T %c %m";
  sacctmgr -n -P show assoc where user=$USER format=account,partition,qos;
  sacctmgr -n -P show qos owner100d format=name,maxwall,maxtrespu,maxjobspu;
  squeue -u $USER -h -o "%i %j %T %M %l %R"' | tee "$VROOT/e6_slurm_readonly.txt"
#     Report: MaxTime at least 04:00:00; the account power-leeburton-users_v2 with QOS owner100d
#     associated; which nodes other than compute-0-[332-333] have at least 48 idle cores; no other bmds-* job queued.
#     Anything else is a finding to report before Gate 1, not something to work around.
```

**Stop here.** Report E3 (attempt, run name, plan and runtime digests), E4
(`#SBATCH` lines, job script SHA-256), E5 and E6. Nothing is submitted
before Gate 1, and `archive` is not run yet (it runs once, at the end).

**Gate 1:** written authorisation of one `sbatch` of this attempt with the
4-hour, 48-core shared-node allocation above.

```bash
"$BMD_PY" $K submit "$VROOT" --confirm-attempt "$ATTEMPT" --confirm-walltime 04:00:00 2>&1 | tee -a "$VROOT/run.log"
"$BMD_PY" $K observe "$VROOT"      # repeat (at least 60 s apart) until SEARCH_COMPLETED or another terminal state
```

Cancellation is a separate decision (**Gate 2**) and only if something goes
wrong: `cancel --confirm-attempt … --confirm-job …`.

```bash
"$BMD_PY" $K execution-evidence "$VROOT"   # after the terminal state is recorded; 0 PASSED, 1 FAILED, 2 INCOMPLETE
"$BMD_PY" $K archive "$VROOT"
```

Stop conditions: any `STOP:`/`VALIDATION FAILED:` line, a non-200 status, an
uncertain submission, a contradiction in an observation, a collector failure,
or an analysis that is not `PASSED`. Keep everything; nothing is retried or
removed.
