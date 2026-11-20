# Promotion validation on POWER: attempt 14bfbbac (10 Nov 2026)

USPEX integration step 4D-2. One Si structure search was planned, prepared,
staged and promoted to submission-ready on POWER through the authenticated
machine API, as `bmdguest`, with parts B and C of `README.md`, and stopped at
remote `SEARCH_PREPARED`. **Result: promotion validated.** The job script and
resource acknowledgement were uploaded and verified, the staged run verified
afresh, and nothing was submitted or executed.

## Identity

| | |
| --- | --- |
| Code | `b7bd184309f6482840a9bc2927ecbaf4756c929a` (step 4D-2), cloned on the VM from a bundle of the branch |
| Runtime package | 79 files, manifest SHA-256 `fca63b4b55c572e47c8744b2073288ad82fd3214ab1f5248838110bfa8a6b95c` (computed from the commit) |
| Attempt | `14bfbbac-8599-48a6-bcc2-cfea917fd967`, its own validation state directory and token (not the production ledger) |
| Run | `uspex_search-20261110-120854-14bfbbac` in `/bmd/bmdguest/searches/` |
| Remote state | `SEARCH_PREPARED` |
| Request | formula `Si`, search defaults (policy `bmd_compute.structure_search_defaults` v1, provisional) |
| Plan digest | `sha256:014ef52c3c737e146eb096779de04817357df36decd4a3e70e394e4f74f0ebbc` (the same plan as the step 4C validation; reproduced from the committed code) |
| Job script | `bmd_search_job.sbatch`, schema `bmd_compute.structure_search_job_script` v1, SHA-256 `a52bee7eff638e21b3f0d6bb3ee8384c363aaf158c321c44aa0e69c91876619f` |
| Resource request | the plan's provisional exclusive-node request, explicitly acknowledged (`resource_acknowledgement.json`, bound to the attempt and plan) |

## Results

| Check | Result |
| --- | --- |
| Staging, then fresh read-only re-verification | `STAGED`, passed |
| Promotion `PUT …/promotion` | local `PREPARED`, remote `SEARCH_PREPARED` |
| Job script on POWER | the one the promotion record names (SHA-256), parsed by `bash -n` |
| Resource acknowledgement on POWER | the one the promotion record names; its own SHA-256 verified |
| Bound bridge, `INPUT.txt`, INCARs, plan, manifests | passed BMD's verification and the independent checks |
| Staged Si POTCAR | identity equal to the plan; no POTCAR content in the evidence |
| Login-node USPEX readiness check | passed |
| Job started (`job_started`, `search_status.json`, SLURM output) | none |
| SLURM jobs, USPEX or VASP processes using the run directory | none |
| Evidence collection | did not modify the run |

## Evidence

| Where | What |
| --- | --- |
| POWER `/bmd-db/guest/validation/uspex_staging_4c_20261110T110411Z_14bfbbac/` | listing, hashes, before/after snapshots, remote state, copies of the non-licensed records including the job script and acknowledgement, SLURM and process listings, `checks.txt` |
| VM `~/bmd-validation/uspex-4d2-20261110T100445Z.tgz` | requests (without headers) and responses, the local records, the summaries, the local preparation, `run.log`, the preflight and evidence reports |
| Archive SHA-256 | `083627d6c7dde9d71763cef2c2e11113dbff91ad578e24b97457a43e7d1cb98b` |

Neither holds a POTCAR, POTCAR text, a token or other credential. The
promoted run and its attempt state stay on POWER as evidence. Submitting it
is step 4D-3, which verifies it again first.
