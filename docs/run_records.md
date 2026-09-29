# Run Record Contracts

BMD Compute writes two JSON records on POWER that later readers use to find and
describe a run: BMD Agent, BMD Compute's own Resume path, and people. Each
record carries a top-level `schema` and `schema_version`. The machine-readable
definitions and validators live in `backend/run_records.py`.

| Record | Location | Schema |
| --- | --- | --- |
| Submission specification | `<run_dir>/submission.json` | `bmd_compute.submission`, version 1 |
| Job record | `<logs_dir>/job_<JOB_ID>.json` | `bmd_compute.job_record`, version 1 |

The two schemas are versioned independently. `provenance.schema_version`
inside `submission.json` versions the provenance block separately (see
`provenance.md`).

## Authority

- `submission.json` is the **Prepare-time submission specification**: what BMD
  Compute prepared and asked the cluster to run. It is written during Prepare,
  before `sbatch`. A prepared-but-not-submitted attempt may be prepared again
  (only with identical calculation metadata), which rewrites the file. Once the
  attempt is `SUBMITTING` or `SUBMITTED`, Prepare is rejected before anything is
  written, so `submission.json` and the rest of the run's execution package
  (uploaded `backend/` runtime, `run_job.py`, sbatch script) are immutable.
  Running the calculation again means a new submission attempt with its own
  run directory.
- `job_<JOB_ID>.json` is a run-resolution record. It is written once, after
  `sbatch` returns a job ID, and links that job ID to the run directory and
  submission attempt. Writing it is best-effort: a run may have no job record.
- Status and state fields in these records (`status`, `submission.ready`,
  `submission.submitted`, `submission.reason`, the job record's `status`, and
  the submission-attempt `state`) describe BMD Compute's own preparation or
  submission step when the file was written. They are
  **not scheduler lifecycle authority**. SLURM accounting and the VASP artifacts remain authoritative for
  what actually executed and how it ended.
- Paths in these records point readers at artifacts. A path that no longer
  resolves means the artifact is unavailable; it is never evidence that the
  artifact existed.

## `bmd_compute.submission` v1

Required unless marked optional:

| Field | Meaning |
| --- | --- |
| `schema`, `schema_version` | `"bmd_compute.submission"`, `1` |
| `run_name` | run name, also the run directory name |
| `created_at` | run timestamp chosen at Prepare |
| `submission.attempt_id` | submission attempt UUID |
| `submission.attempt_state` | path of the submission-attempt state file |
| `flow_spec.workflow_spec.stages[]` | the resolved executable workflow, in canonical stage order; each stage has `stage_type`, `theory`, `modifiers` (list), `options` (object) and `label` (string or null) |
| `flow_spec.structure` | optional; the submitted structure representation |
| `paths.run_dir`, `paths.result_dir` | run directory and final-result directory |
| `paths.stage_dirs` | object mapping stage identifiers to directories (below) |
| `paths.log_out`, `log_err`, `slurm_out`, `slurm_err` | optional log paths |
| `cluster.partition`, `cluster.account` | requested SLURM partition and account |
| `resources.nodes`, `ntasks`, `mem_gb`, `walltime` | requested resources |
| `environment.VASP_CMD`, `PMG_VASP_PSP_DIR`, `JOBFLOW_CONFIG_FILE` | optional requested environment policy |
| `provenance` | optional; versioned by its own `schema_version` |

### Stage directories

`workflow_spec.stages[]` defines stage order. `paths.stage_dirs` keys are
explicit stage identifiers: `stage_NN`, or `relax_NN` for the PBE
double-relaxation workflow, where `NN` is the 1-based stage index (at least two
digits). A single-stage workflow has an empty `stage_dirs` and runs in
`result_dir`; a multi-stage workflow names every stage exactly once. JSON object
insertion order is not contractual: readers map identifiers to stages and never
infer stage order from key order.

## `bmd_compute.job_record` v1

| Field | Meaning |
| --- | --- |
| `schema`, `schema_version` | `"bmd_compute.job_record"`, `1` |
| `job_id` | the scheduler job ID returned by `sbatch` |
| `run_name`, `run_dir` | the run this job belongs to; `run_dir/submission.json` is the submission specification |
| `attempt_id` | submission attempt UUID; equals `submission.attempt_id` in `submission.json` |
| `submitted_at` | optional; BMD Compute's clock when the record was created, not a scheduler time |

## Non-contractual fields

Both files contain more than the contract. Those fields are BMD Compute
internals and may change or disappear without a schema change. Examples:
`status`, `label`, `modules`, `runner`, `potcar`, `preflight`, SSH connection
details under `cluster`, `submission` fields other than `attempt_id` and
`attempt_state`, `paths` entries not listed above, `flow_spec.workflow`,
`flow_spec.calculation_spec`, and in the job record `status`, `raw_output`,
`remote_script`, `log_paths`, `cluster`, `resources`, `submission_spec` and
`remote_state_path`. BMD Compute's Resume path still reads the job record's
embedded `submission_spec` internally.

## Versioning and compatibility

- Adding optional fields keeps `schema_version` 1. Removing a contractual field
  or changing its meaning requires a new `schema_version`.
- Records written before these contracts have no `schema`. They remain valid
  legacy records and are not rewritten.
- Readers should treat a record with a known `schema` and an unsupported
  `schema_version` as unsupported, not as legacy and not as malformed.

## Atomic writes

Both records are written to a hidden temporary file in the destination
directory and moved into place with an SFTP POSIX rename, so readers see either
the previous complete file or the new complete file. If the rename fails, the
temporary file is removed and the destination is left unchanged.

## Canonical fixtures

`tests/fixtures/run_records/v1/` holds generated examples of both records for a
single-stage static run, a PBE relax followed by HSE06 static with SOC, and a
PBE double relaxation. `v1/SHA256SUMS` records the SHA-256 of every fixture
file and is the fixtures' identity. The BMD Compute commit recorded inside each
fixture is generation provenance only; it may not be reachable from `main`
(for example after a squash merge) and is never by itself a reason to
regenerate. Regenerate with `python tests/run_record_fixtures.py` only when the
fixture content should change; that also rewrites `SHA256SUMS`. Consumers may
vendor the files with `SHA256SUMS` and verify them by hash; there is no runtime
dependency between repositories.
