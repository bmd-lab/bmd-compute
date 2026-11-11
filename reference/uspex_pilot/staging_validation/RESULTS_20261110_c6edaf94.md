# Submission, monitoring and cancellation acceptance on POWER: attempt c6edaf94 (10 Nov 2026)

USPEX integration steps 4D-3 and 4D-4. The first live managed submission: a
new Si search was planned with a 15-minute allocation, staged, promoted,
submitted once through the machine API, monitored, cancelled through the
machine API and observed to its settled end, with part D of `README.md`, as
`bmdguest`, behind two separate written approvals (Gate 1 `sbatch`, Gate 2
`scancel`). **Result: operational acceptance passed.** The job never
started, so no USPEX or VASP ran: this validates the operational pathway,
not scientific execution.

## Identity

| | |
| --- | --- |
| Code | `364712ad22bedaf6028be288360113f93227605d` (the acceptance kit, on step 4D-4 `0bf9009`), cloned on the VM from a bundle |
| Runtime package | manifest SHA-256 `fca63b4b55c572e47c8744b2073288ad82fd3214ab1f5248838110bfa8a6b95c` (unchanged since step 4D-2) |
| Attempt | `c6edaf94-a2ab-4be7-a817-ce52f3d14852`, its own validation state directory and token (not the production ledger) |
| Request | formula `Si`, search defaults except the planner's Advanced walltime override `00:15:00` |
| Plan digest | `sha256:4c85cec6eb6286ae085b62ee113dc42cddb6630d83d15c85271a7ccc0d42f993` (differs from the default Si plan only in its walltime) |
| Job script | `bmd_search_job.sbatch`, SHA-256 `71aa6125fdfd97be4b58d3c54a313e9c522e4343405758692c2a7925399a6547`; its only time directive `#SBATCH --time=00:15:00` |
| SLURM job | `23576570` |

## Results

| Check | Result |
| --- | --- |
| Submission (`PUT …/submission`) | one `sbatch`, HTTP 200; job ID from `sbatch`'s own answer |
| Job identity while queued | verified through `squeue` (attempt comment, job name, run directory, submission time) |
| Allocation | SLURM's time limit 15 minutes, as acknowledged |
| Monitoring | three append-only observations, hash-chained |
| Cancellation (`PUT …/cancellation`) | `scancel` of the exact job ID after a fresh identity check; returned successfully |
| Settled outcome | SLURM `CANCELLED`; local state `SEARCH_CANCELLED`, cancellation observed |
| Execution | the job never started: no `job_started`, no USPEX, no VASP |
| UTC/local `sacct` diagnostic | consistent with POWER's offset |
| Attempt comment | visible in `squeue`; **absent from `sacct`** (accounting does not keep job comments) |
| Independent acceptance evidence | passed, zero diagnostics flagged for review |

Consequence of the comment finding, as designed: after a job leaves the
queue its identity can only be `partial` (name, run directory and submission
time, no comment). That is enough to follow an already-bound job to its
terminal state, never to adopt an uncertain submission or to cancel.

## Evidence

| Where | What |
| --- | --- |
| POWER `/bmd-db/guest/validation/uspex_acceptance_4d_20261110T210329Z_c6edaf94_1267255/` | clocks, SLURM configuration, queue and accounting records (UTC and local), remote state, run-directory listing, job script copy, checks |
| VM `~/bmd-validation/uspex-4d-acceptance-20261110T202009Z.tgz` | requests (without headers) and responses, the local records, the observation evidence chain, the summaries, the POWER reports |
| Archive SHA-256 | `8f3ebf5b4c25d11b77ab02f623dbd1e1e5817eb7989309544d141630b74a41c7` |

Neither holds a POTCAR, POTCAR text, a token or other credential. The
cancelled run directory and its attempt state stay on POWER as evidence.
