# Validity probe on POWER: SLURM job 23406856 (6 Nov 2026)

USPEX integration step 2. The real USPEX 25.0 ran through the production
launch bridge, launcher and post-step validity check
(`backend/structure_search/step_validity.py`), with two injected failures.
**Result: USPEX rejected both failed candidates.** Their chains stopped at the
failed step, they were not run again, they were not listed in `Individuals`,
and no energy VASP had produced for them was reused.

## Run

| | |
| --- | --- |
| Job | 23406856, `compute-0-271`, account `power-leeburton-users_v2`, partition `leeburton-pool`, QOS `owner100d` |
| Time | 10:36:22–11:19:43 UTC (elapsed 00:43:37) |
| Source | probe bundle `bmd-compute-uspex-probe-20261106d` (manifest SHA-256 `8f2b1581…6b09`): commit `e236782` plus the step-2 files of the commit that adds this directory, identical file for file |
| USPEX | 25.0 (banner), payload SHA-256 `fc4a08e6…95c8c`, sandbox `/bmd/shared/uspex25_sandbox`, `whichCluster = -1` |
| Runtime check | passed (`runtime_check.json`); runtime record SHA-256 `62a77293…7e341` |
| VASP | 6.4.1 (`vasp/rocky8-intel-6.4.1`), `srun --exact --mpi=pmi2 -n 4 --mem=8G vasp_std`, 2 relaxations at a time |
| Python | `atomate2_remote` 3.12.13, pymatgen 2026.7.16; jobflow and atomate2 not imported |
| System | Si8, population 4, 2 generations, the si_bridge pilot's 3-step INCARs (pilot settings, not BMD policy) |
| POTCAR | PBE_64 `Si`, dataset SHA-256 `2a6df1c1…252c` (the same dataset as BMD's methodology records); no POTCAR content here |

## Injected failures

| | Step-2 electronic non-convergence | Step-3 execution failure |
| --- | --- | --- |
| Injection | `NELM = 3` in the first step-2 relaxation | first completed step-3 relaxation reported as exit 7 |
| Candidate | individual 1, `Calcfold_1_2` | individual 2, `Calcfold_2_3` |
| VASP / wrapper status | 0 / 0 | 0 / 7 |
| Bridge verdict | `invalid_convergence`: "the final ionic step used 3 of NELM = 3 electronic steps" | `execution_failed`: "VASP exited with status 7" |
| Status USPEX received | 118 | 7 |
| Files withheld | OUTCAR, OSZICAR, CONTCAR, vasprun.xml (archived under `bmd_step_validity/<id>/`) | same |
| USPEX master log | `Task failed on step '2' for system=1` | `Task failed on step '3' for system=2` |
| Later steps run | none | none |
| Runs of the failed step | 1 (no retry) | 1 (no retry) |
| Listed by USPEX | no | no |
| Stale-energy reuse | no (VASP energies −40.858, −38.585 eV not listed) | no (−35.514, −35.449, −15.623 eV not listed) |

## Search as a whole

- USPEX master exit status 0, not stopped; generations 1 and 2 listed.
- 26 VASP calls, each CalcFold step run once, no repeated POSCAR.
- `Individuals`: generation 1 lists only individuals 3 and 4 (the two valid
  candidates; USPEX did not replace the rejected ones), generation 2 lists
  5–9. Best: individual 6, −40.596 eV per 8 atoms.
- Done-file statuses seen: 0, 118, 7.
- Master-log lines matching error patterns: the two "Task failed" lines, one
  `dateutil` zoneinfo warning from the unpacked program, and one USPEX
  "Permutation failed to produce a valid structure" warning (a variation
  operator, unrelated to the injections).

## Not observed

- Status 119 (evidence missing or malformed): not injected.
- The optional `status_only` run (validity check off, exit 7 with the files
  left in place) was not made, so it is not known whether the exit status
  alone would be enough. The bridge keeps withholding the files.
- Whether a generation whose candidates are mostly rejected is refilled
  later: here it was not.

## Files

The small text evidence collected from the work directory
`/bmd/bmdguest/validation/uspex_validity_probe_20261106_a` after the job
(paths inside the files are POWER paths of that validation area). No
POTCAR, OUTCAR, `vasprun.xml`, WAVECAR, CHGCAR or cache was collected.

| File | Content |
| --- | --- |
| `probe_report.json` | `analyze_probe.py`'s report (what the tables above summarise) |
| `runtime_check.json` | the job-start runtime check |
| `probe_state/calls.tsv` | one line per VASP call: CalcFold, step, POSCAR SHA-256, injection, VASP and reported status, TOTEN |
| `uspex_master.log` | the USPEX master's log |
| `spool/launcher.jsonl`, `spool/done/*.done` | the launcher journal and one done file per step |
| `search/bmd_step_validity/*/validity.json` | the bridge's verdicts on the two failed steps |
| `search/Calculation/Calcfold_*/bmd_step_validity.jsonl` | per-CalcFold verdict log (CalcFolds still present at the end) |
| `search/results1/{Individuals,Individuals_all,BESTIndividuals}` | USPEX's listings |
| `search/INPUT.txt`, `search/Specific/INCAR_{1,2,3}` | the search input and the pilot INCARs |
| `uspex_validity_probe_20261106_a.slurm-23406856.out` | the job log |
| `SHA256SUMS` | SHA-256 of each file above |

Transferred as `probe_evidence_23406856.tar.gz`, SHA-256
`3146dce29a6b161e6f98eaae9fe220e0a0a37fa3d44cb229e5c3ae69fd0f3ab0`
(33 467 bytes).
