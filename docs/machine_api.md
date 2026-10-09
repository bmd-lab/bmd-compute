# Machine API (v1)

bmd-compute serves a small authenticated JSON API for machine clients such as
bmd-run:

| Route | Scope | Purpose |
| --- | --- | --- |
| `GET /api/v1/identity` | any valid token | Service identity, API and schema versions, the caller's principal and scopes |
| `POST /api/v1/plans` | `plan` | Resolve a structure and workflow request into the authoritative calculation plan |
| `PUT /api/v1/attempts/{attempt_id}` | `prepare` (prepare only) or `submit` (prepare and submit) | Prepare, or prepare and submit, one calculation; idempotent per attempt ID |
| `GET /api/v1/attempts/{attempt_id}` | `read` | The attempt's authoritative state, job identity and scheduler state |

`identity` and `plans` never contact POWER. The attempt routes use Compute's
existing remote preparation, submission and monitoring machinery with
server-owned connection profiles; see [Execution](#execution). There is no
machine route for cancellation or result retrieval. The browser routes are
unchanged and do not use this authentication. The API is not listed in
`/openapi.json`.

## Shared scientific resolution

`POST /api/v1/plans` calls `main.plan_calculation_request`, which runs the same
module-level functions, in the same order, as the browser's Build route:
`workflow_spec_from_form`, `execution_resources_from_form`, `parse_structure`,
`resolve_workflow_for_structure` (automatic treatments),
`method_considerations_for_workflow_state`, and
`build_submission_state_from_structure` (new-calculation admission, generated
inputs, submission specification). The API adds no methodology. Its only
additions are request-shape validation and classification of two input errors
the models raise as plain exceptions (unknown stage/theory/modifier names, and
CIF parser failures), so machine callers receive a classified 422 instead of a
generic failure.

The response projection reads the server-built submission specification and
regenerates per-stage inputs with `generated_input_stage_previews`, the same
generator the browser's multi-stage preview uses.

## Transport and network

The intended transport is an SSH tunnel from POWER that terminates on the
service host, for example forwarding a POWER loopback port to
`127.0.0.1:8000` on the VM. By default the API refuses any client whose address
is not loopback (`127.0.0.1` or `::1`), before reading credentials. Do not send
bearer tokens over plain HTTP across the network. Setting
`BMD_API_ALLOW_NON_LOOPBACK=1` disables the loopback check; use it only behind
HTTPS.

Query-string credentials are not accepted, and any request with a query string
is refused. Never put a token in a URL: a web server or proxy may log the full
URL before the application sees and rejects the request, so refusing it does
not keep a token out of access logs. Send tokens only in the `Authorization`
header.

## Authentication

Every request needs `Authorization: Bearer <token>`. Tokens have the form
`bmdc1.<token_id>.<secret>` (32 random bytes). The service reads the token
store named by `BMD_API_TOKENS_FILE` on every request:

```json
{
  "schema": "bmd_compute.api_tokens",
  "schema_version": 1,
  "principals": [
    {
      "principal": "lee-benchmark",
      "token_id": "<16 hex characters>",
      "verifier": "sha256:<64 hex characters>",
      "scopes": ["plan"],
      "enabled": true
    }
  ]
}
```

- The file must be an absolute path outside the repository, a regular file of
  at most 64 KiB, owned by the service account and not readable or writable by
  group or others (POSIX). Otherwise the API answers 503 `api_not_configured`.
- Only the SHA-256 verifier is stored. Verifiers are compared in constant time.
- Scope vocabulary: `read`, `plan`, `prepare`, `submit`. An unknown scope makes
  the whole file invalid. `plan` never grants execution: `prepare`, `submit`
  and `read` are separate scopes (see [Execution](#execution)).
- Setting `enabled` to `false`, or removing an entry, revokes a token on the
  next request.
- Generate a token with `python -m compute_api.tokens --principal NAME --scope
  plan`. It prints the token once and the entry to add to the file; it writes
  nothing.

Order of checks: query string, loopback, token store, token, scope, then the
request body. A missing, malformed, unknown, tampered or disabled token gets
401 `unauthenticated`; a valid token without the route's scope gets 403
`insufficient_scope`. Tokens and Authorization headers are never logged or
returned.

## Plan request

```json
{
  "structure": {"format": "poscar", "text": "<POSCAR or CIF text>"},
  "workflow": {"desired_output": "energy_only"},
  "resources": {"cpus": 48, "memory_gb": 128, "walltime": "24:00:00"}
}
```

- `structure.format`: `poscar` or `cif` (case-insensitive). `structure.text`:
  at most 2 MiB. The whole body: at most 3 MiB, `application/json`.
- `workflow`: exactly one of `desired_output` (a BMD Compute Desired Output
  identifier) or `custom` (`{"stages": [{"stage_type", "theory", "modifiers",
  "options"}]}`, at most 16 stages). Custom workflows go through
  `validate_user_workflow_spec`, which rejects free-form INCAR/KPOINTS and
  BMD-generated options.
- `resources` (optional; each key optional): `cpus`, `memory_gb`, `walltime`
  (`HH:MM:SS`), `queue`. Values are validated by the existing
  `ExecutionResources` allowlists. Nodes, account and partition are not
  accepted.
- Every object is closed: unknown fields (for example `cluster`, `paths`, SSH
  settings, scripts, commands) are rejected with 422 `invalid_request`, naming
  the field but never echoing its value.

## Plan response

`schema: "bmd_compute.api.plan"`, `schema_version: 1`, `api_version: "v1"`:

- `plan_digest`, `plan_digest_version`;
- `request`: structure format, workflow mode and Desired Output;
- `structure`: formula, reduced formula, atom count, space group, crystal
  system, volume and the canonical-structure SHA-256;
- `workflow`: recipe and ordered stages (stage type, theory, modifiers, stage
  options including frozen DFT+U parameters, VASP executable);
- `automatic_treatments`: the record stored in `submission.json` (applied,
  omitted, not applicable and advisory treatments, prepared DFT+U record), or
  `null` for Custom workflows;
- `method_considerations`: policy version and, per consideration, its id,
  method, modifier, status, selection state, automatic-application state
  (`applied`, `omitted_unsupported_combination`, `not_applicable`, `advisory`), trigger
  elements and classes, applicable stage types, policy source, and the stages
  it was applied to;
- `resources`: nodes, CPUs, memory, walltime, partition and account;
- `scientific_inputs`: POTCAR functional and, per stage, INCAR settings and
  their text hash, a KPOINTS summary and text hash, POTCAR symbols and a POSCAR
  summary and hash;
- `software`: loaded modules and the parity-critical package versions;
- `policies`: admission, automatic DFT+U, method considerations, Custodian,
  runtime parity and plan-digest versions;
- `capability_schema_version`.

The response never contains submission identity tokens, attempt IDs or
fingerprints, SSH profiles, remote or environment paths, runner settings,
SLURM scripts, provenance blocks or raw exception text.

Errors: `{"api_version": "v1", "error": {"code", "message", ...}}` with codes
`query_not_allowed` (400), `invalid_json` (400), `unauthenticated` (401),
`loopback_required` (403), `insufficient_scope` (403), `request_too_large`
(413), `unsupported_media_type` (415), `invalid_request` (422, with `fields`),
`structure_invalid` (422), `calculation_invalid` (422, with `suggestion` and
`diagnostic_code` when Compute provides one, for example
`hse06_soc_not_supported_for_new_calculations`), `plan_failed` (500, generic)
and `api_not_configured` (503).

## Plan digest

`plan_digest` (`bmd_compute.plan_digest` v1) is the SHA-256 of canonical JSON
built from the server-built plan. It identifies what would be calculated and
with which resources, and complements the submission-attempt fingerprint, which
it does not replace.

Covered: the parsed structure (lattice, species, fractional coordinates, site
properties); the resolved workflow (stages, theories, modifiers, options); the
automatic-treatment record; per stage the VASP executable, INCAR settings,
KPOINTS text, the SHA-256 of the generated POSCAR text and POTCAR symbols; the
POTCAR functional and symbol record;
nodes, MPI tasks, memory, walltime, partition and account; loaded modules and
the VASP command template; parity-critical package versions; and the versions
of the admission, automatic DFT+U, method-consideration, Custodian,
runtime-parity and plan-digest policies.

Not covered: run timestamp, run name, label and creation time; attempt ID,
fingerprint and identity token; remote paths, POTCAR repository location,
runner and log paths; SSH profile; the SLURM script text; the Git commit and
runtime-package manifest. A POSCAR comment line or whitespace that parses to
the same structure gives the same digest. The authoritative definition is the
docstring of `compute_api/plan_digest.py`.

## Execution

### Configuration

Execution additionally needs `BMD_API_STATE_DIR`: an absolute path to an
existing directory outside the repository, owned by the service account and
not accessible to group or others (mode `0700`). It holds the attempt ledger
(one JSON file per attempt under `attempts/`, plus `ledger.lock`). Without it,
or if it is unsafe, the attempt routes answer 503 `execution_not_configured`.
Planning does not need it.

The `plan` scope never grants execution. A token needs `prepare` for
`submit: false` and `submit` for `submit: true`, and `read` to look an attempt
up. A token holding neither `prepare` nor `submit` is refused before the body
is read.

### Request: `PUT /api/v1/attempts/{attempt_id}`

`attempt_id` is chosen by the client and must be a canonical lowercase RFC 4122
UUID (for example from `uuid.uuid4()`). The body is a plan request (same fields
and rules as `POST /plans`) plus:

- `expected_plan_digest` (required): the `plan_digest` the client obtained from
  `POST /plans` and intends to run;
- `submit` (required boolean): `false` prepares only, `true` prepares if
  needed and submits;
- `labels` (optional): `campaign` and `cell`, short identifiers
  (`[A-Za-z0-9][A-Za-z0-9._:-]{0,63}`), recorded for provenance.

SSH settings, remote paths, job IDs, run timestamps, account, partition and
node counts are rejected like every other unknown field.

### Response

`schema: "bmd_compute.api.attempt"`, `schema_version: 1`: `attempt_id`,
`plan_digest`, `state`, `labels`, `created_at` (UTC), effective `resources`,
`submission` (`requested`, `job_id`, `submitted_at_local` as recorded by
Compute) and `scheduler` (`null` until a job exists; then `available`,
`summary` of `PENDING`, `RUNNING`, `SUCCESS`, `FAILURE` or `UNKNOWN`, the
normalized SLURM `state` such as `COMPLETED` or `TIMEOUT`, `exit_code`,
`elapsed`, `started_at`, `ended_at`, `terminal` and `checked_at`).

`state` is one of:

| State | Meaning |
| --- | --- |
| `registered` | Bound to this principal and plan; not yet prepared on POWER |
| `prepared` | PREPARED on POWER; not submitted |
| `submitted` | `sbatch` returned a job ID, recorded on POWER |
| `submission_uncertain` | The remote attempt is SUBMITTING: a submission is in progress or its `sbatch` outcome is unknown. It is never resubmitted automatically. |

`GET /api/v1/attempts/{attempt_id}` returns the same projection after reading
the remote attempt state and, when a job exists, monitoring it through the
authenticated job record (`job_<id>.json` must name this attempt). A job ID is
never accepted from the client.

### Attempt binding and idempotency

The first PUT for an attempt ID resolves the plan, checks that its digest
equals `expected_plan_digest`, applies the [execution limits](#execution-limits)
and then atomically creates a ledger record binding the attempt ID to:

- the authenticated principal;
- the SHA-256 of the validated request (structure, workflow, resources,
  labels);
- the plan digest;
- the run timestamp used to build the submission specification (unique among
  ledger attempts, so attempts never share a run directory);
- the resulting submission-attempt fingerprint.

Every later PUT for that ID must come from the same principal (otherwise 403
`attempt_forbidden`) with the identical request (otherwise 409
`attempt_request_mismatch`). It rebuilds the submission specification with the
recorded run timestamp and attempt ID, so the run directory, run name and
fingerprint are unchanged, and it must still resolve to the recorded plan
digest (409 `plan_changed`) and fingerprint (409
`attempt_fingerprint_mismatch`).

The remote attempt state on POWER stays the authority. The PUT then:

1. reads it through a server-owned connection; a state whose attempt or
   fingerprint differs is never touched (409 `attempt_fingerprint_mismatch`);
2. if there is no remote state, prepares with `prepare_remote_submission`;
3. if it is PREPARED: refuses (409 `runtime_package_changed`) when the
   prepared runtime package differs from the running service's, because a
   prepared attempt is never re-prepared or replaced; returns `prepared` when
   `submit` is false; otherwise reserves a submission against the caps and
   calls `submit_remote_workflow`;
4. if it is SUBMITTED: returns the recorded job, without submitting again;
5. if it is SUBMITTING: answers 409 `submission_uncertain`.

Compute's existing remote `mkdir` lock and PREPARED -> SUBMITTING -> SUBMITTED
transitions serialize concurrent duplicate PUTs, so one attempt produces at
most one `sbatch`. After a lost response or timeout the client repeats the
same PUT or calls GET; it never needs a new attempt ID to find out what
happened.

| Outcome of a submission call | Result |
| --- | --- |
| Job submitted (now or earlier) | 200 `submitted` with the job ID |
| `sbatch` ran and refused the job | 502 `submit_failed`; remains `prepared`; the reservation is released |
| `sbatch` did not run (busy, lost connection, another request holds the lock) | 503 `remote_busy` or 409 `submission_not_started`; the reservation is kept until a retry resolves it |
| `sbatch` outcome unknown | 409 `submission_uncertain`; never resubmitted |
| Outcome could not be read back | 503 `submission_outcome_unconfirmed`; GET or repeat the PUT |

### Execution limits

Initial API limits (not the POWER allocation), enforced on the effective
resources of every attempt, before any remote activity:

- at most 96 CPUs, 128 GB memory and a requested walltime of 72:00:00;
- one node; partition and account are always Compute's own.

Violations are 422 `resource_limit_exceeded`. Per principal, from the durable
ledger:

- at most 2 active jobs (reserved, submitted or uncertain and not known to be
  finished; before refusing, Compute refreshes the principal's other jobs from
  SLURM): 429 `active_job_cap_exceeded`;
- at most 5 submissions in any rolling 24 hours: 429 `submission_cap_exceeded`;
- at most 20 new attempts in any rolling 24 hours: 429 `attempt_cap_exceeded`.

Caps are decided under an exclusive file lock and survive restarts.
Repeating a PUT for an attempt that already holds a reservation does not
count again.

### Execution errors

In addition to the planning errors: 403 `attempt_forbidden`; 404
`attempt_not_found`; 409 `plan_digest_mismatch` (with the resolved
`plan_digest`), `attempt_request_mismatch`, `plan_changed`,
`attempt_fingerprint_mismatch`, `runtime_package_changed`,
`attempt_in_progress`, `submission_not_started`, `submission_uncertain` (with
the attempt projection) and `remote_state_invalid`; 422 `invalid_attempt_id`
and `resource_limit_exceeded`; 429 cap errors; 502 `prepare_failed` and
`submit_failed`; 503 `execution_not_configured`, `remote_busy`,
`remote_unavailable`, `remote_state_unconfirmed` and
`submission_outcome_unconfirmed`. No error contains remote output, paths,
tracebacks or credentials.
