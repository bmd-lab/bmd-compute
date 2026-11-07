# Machine API (v1)

bmd-compute serves a small authenticated JSON API for machine clients such as
bmd-run:

| Route | Scope | Purpose |
| --- | --- | --- |
| `GET /api/v1/identity` | any valid token | Service identity, API and schema versions, the caller's principal and scopes |
| `POST /api/v1/plans` | `plan` | Resolve a structure and workflow request into the authoritative calculation plan |
| `POST /api/v1/structure-search-plans` | `plan` | Plan a formula-based USPEX structure search (New structure); read-only, no preparation or submission |
| `PUT /api/v1/structure-search-attempts/{attempt_id}` | `prepare` | Prepare a search plan locally in the API state root (`PREPARED_LOCALLY`); idempotent per attempt ID; no remote staging or submission |
| `GET /api/v1/structure-search-attempts/{attempt_id}` | `read` | Read a structure-search attempt back |
| `PUT /api/v1/structure-search-attempts/{attempt_id}/staging` | `prepare` | Stage a `PREPARED_LOCALLY` attempt on POWER and verify it there (`STAGED`); idempotent; never submits |
| `PUT /api/v1/attempts/{attempt_id}` | `prepare` (prepare only) or `submit` (prepare and submit) | Prepare, or prepare and submit, one calculation; idempotent per attempt ID |
| `GET /api/v1/attempts/{attempt_id}` | `read` | The attempt's authoritative state, job identity and scheduler state |

`identity`, `plans` and `structure-search-plans` never contact POWER. The attempt routes use Compute's
existing remote preparation, submission and monitoring machinery with
server-owned connection profiles; see [Execution](#execution). There is no
machine route for cancellation or result retrieval. The browser routes are
unchanged and do not use this authentication. The API is not listed in
`/openapi.json`.

## Shared scientific resolution

`POST /api/v1/plans` calls `main.plan_calculation_request`, which runs the same
module-level functions, in the same order, as the browser's Build route:
`workflow_spec_from_form`, `resource_request_from_form`, `parse_structure`,
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
  identifier: `relaxed_energy`, `relaxed_structure`, `electronic_dos`,
  `electronic_band_structure`, `phonons`, or the legacy single-point
  `energy_only`, which keeps its fixed-geometry meaning although the browser
  no longer offers it; see `methodology.md` section 1) or `custom`
  (bmd-run does not yet request `relaxed_energy`: milestone E1b must add it to
  bmd-run's Desired Output vocabularies, output schemas and CLI; bmd-run keeps
  its `energy_only` default pending a separate decision) (`{"stages": [{"stage_type", "theory", "modifiers",
  "options"}]}`, at most 16 stages). Custom workflows go through
  `validate_user_workflow_spec`, which rejects free-form INCAR/KPOINTS and
  BMD-generated options.
- `resources` (optional; each key optional): `cpus`, `memory_gb`, `walltime`
  (`H:MM:SS`, up to four hour digits), `queue`. An omitted key means
  Automatic: BMD Compute resolves it with the same estimator as the browser
  (`docs/automatic_resources.md`). Explicit values are kept exactly and
  validated by the existing `ExecutionResources` allowlists. Nodes, account,
  partition and QOS are not accepted.
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
- `resources`: nodes, CPUs, memory, walltime, partition, account and QOS;
- `resource_selection`: automatic-resource policy, the source (`automatic` or
  `manual`) of each resource, and the walltime recommendation with its basis
  (`empirical_estimate` or `conservative_fallback`), explanation and per-stage
  classes and allowances;
- `scientific_inputs`: POTCAR functional and, per stage, INCAR settings and
  their text hash, a KPOINTS summary and text hash, POTCAR symbols and a POSCAR
  summary and hash;
- `software`: loaded modules and the parity-critical package versions;
- `policies`: admission, automatic DFT+U, method considerations, Custodian,
  runtime parity, automatic resources and plan-digest versions;
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

## Structure-search plans

`POST /api/v1/structure-search-plans` plans a New structure request: a USPEX
search for low-energy structures of a chemical formula. It is not a Desired
Output of `/plans` (which needs a structure): `/plans` and `PUT /attempts`
refuse `desired_output: "new_structure"` with 422
(`workflow.desired_output: use_structure_search_plans`) before any planning
or ledger access. The route and the browser's `/predict` preview call the
same entry point, `backend.structure_search.planning.plan_structure_search`,
which builds the frozen `StructureSearchPlan` (`docs/structure_search.md`).

Request (`application/json`, closed objects):

```json
{
  "formula": "SiO2",
  "search": {"min_atoms": 3, "max_atoms": 12, "num_generations": 30, "walltime": "168:00:00"},
  "expected_plan_digest": "sha256:..."
}
```

- `formula` (required): at most 64 characters; read by the same formula
  reader as the browser. `search` and every key in it are optional (omitted:
  the provisional search default). `expected_plan_digest` (optional): the
  request is refused with 409 `plan_changed` (carrying the current
  `plan_digest`) if it now plans differently.
- Settings v1 fixes by policy (pressure, theory, spin, `kresol_start`,
  population sizes, INCAR/POTCAR keys, ...) are refused with
  `fixed_by_bmd_policy`; `structure`, `workflow` and `resources` with
  `not_accepted_for_structure_search`; anything else with `unknown_field`.
- An unreadable formula, or a composition or setting outside the search scope
  (spin-polarised, DFT+U or SOC compositions, too many elements, cells or
  generations), gives 422 `calculation_invalid` with the same message and
  `diagnostic_code` the browser shows.

Response (`schema: "bmd_compute.structure_search_plan_response"`,
`schema_version: 1`): `request_kind: "structure_search"`,
`desired_output: "new_structure"`, `plan_schema` and `plan_schema_version`,
`plan_digest` (the scientific search-plan digest; it excludes provenance and
is distinct from the calculation plan digest), `formula` (`entered`,
`display`, `formula_unit_as_entered`, `canonical`, `reduced_from_entered`,
`reordered_from_entered`, `formula_atoms`), `search` (method, theory,
`energy_grade: "search-grade"`, `equivalent_to_bmd_adaptive_relaxation:
false`, cell sizes, population, generations, walltime, defaults status),
`finalist_refinement` (`status: "planned, not yet available"`), `resources`
(`status: "provisional_estimate"`, `approved: false`, the exclusive-node
request), `preparation_available: false`, `submission_available: false`, and
`plan`: the complete frozen plan document (record, `plan_digest`,
`provenance`). Equivalent formulas (`SiO2`, `O2Si`) give the same digest; the
entered formula is kept in `formula` and `plan.provenance`.

## Structure-search attempts (local preparation)

`PUT /api/v1/structure-search-attempts/{attempt_id}` (scope `prepare`; the
attempt ID is a canonical lowercase UUID) takes the `structure-search-plans`
body with `expected_plan_digest` required; `submit` is refused
(`submission_not_available`). Implemented in
`compute_api.structure_search_execution`, in the same service and with the
same ledger mechanisms (validated reads, atomic `O_EXCL` create and
rename-replace, the state root's `ledger.lock`) as calculation attempts.

State root layout (`BMD_API_STATE_DIR`, directories mode 0700):

| Path | Content |
| --- | --- |
| `attempts/<id>.json` | calculation attempts, unchanged (v1.2.0 layout and schema `bmd_compute.api_attempt` v1) |
| `ledger.lock` | shared by both kinds |
| `structure_search/attempts/<id>.json` | search attempt records, schema `bmd_compute.api_structure_search_attempt_record` v1 |
| `structure_search/preparations/<id>/preparation-<n>/` | prepared search inputs |

Search records are never written to `attempts/`, so calculation scans, by
this release or by v1.2.0 after a rollback, see only calculation records;
no migration is needed either way. Attempt IDs are unique across the two
kinds: under the shared lock each kind refuses an ID the other uses (409
`attempt_kind_mismatch`), and if both exist (possible only if an older
release created one) both kinds' routes refuse it.

Sequence of a `PUT`:

1. An attempt already `PREPARED_LOCALLY` for the same principal and request
   is returned as stored (its files are checked first), without re-planning,
   even if scientific policy has changed since.
2. Otherwise (a new attempt, or one left `PREPARING`) the request is planned
   with today's policy; a digest other than `expected_plan_digest` gives 409
   `plan_changed` before anything is written.
3. Under the lock the record is created (`PREPARING`) or its principal (403)
   and request (409 `attempt_request_mismatch`) are checked. The request
   identity is the formula as entered, the search settings and the expected
   digest.
4. `search_plan.prepare_search_directory` writes and verifies the inputs;
   the record becomes `PREPARED_LOCALLY` with the plan digest, the files'
   SHA-256 and the manifest's SHA-256. No POTCAR is read or written.

The launch-bridge path on POWER is not bound yet: `INPUT.txt` names
`search_plan.UNBOUND_BRIDGE_PATH` and the manifest records `bound: false`;
`verify_execution_binding` refuses such a directory. Remote staging (below)
binds and verifies the real path on POWER; the local preparation is never
modified.

Response (`schema: "bmd_compute.api_structure_search_attempt"`, v1):
`attempt_id`, `attempt_kind`, `plan_digest`, `state` (`PREPARING` or
`PREPARED_LOCALLY`, `STAGING` or `STAGED`), `created_at`, `updated_at`,
`preparation` (`files`, `manifest_sha256`, `bridge_path_bound: false`,
`contains_potcar_contents: false`, `interrupted_preparations`, `integrity`),
`remote_staging_available` (true in `PREPARED_LOCALLY` and `STAGING`),
`staging` (null until staging starts; see below), `submission_available:
false`. No local or remote paths are returned.

`GET` (scope `read`) returns the same body without planning or writing;
`integrity` is `verified`, `damaged` (the files no longer match the record)
or `incomplete` (`PREPARING`).

Interruptions: a record left `PREPARING` is completed by the next `PUT`,
after step 2: a preparation directory that verifies is adopted, a missing
one is prepared, and one that exists but does not verify is kept, recorded
as abandoned, and preparation continues in `preparation-<n+1>`. A damaged
`PREPARED_LOCALLY` attempt gives 409 `preparation_damaged` and is not
replaced. Nothing is ever overwritten or deleted, and there is no automatic
clean-up yet.

## Structure-search staging (remote, step 4C)

`PUT /api/v1/structure-search-attempts/{attempt_id}/staging` (scope
`prepare`) takes exactly `{"expected_plan_digest": "sha256:..."}` and stages
a `PREPARED_LOCALLY` attempt of the same principal on POWER as `bmdguest`,
through the configured connection profile and the shared remote-operation
limiter. It never submits, claims a SLURM job or runs USPEX or VASP.

POWER layout (all under `bmdguest`):

| Path | Content |
| --- | --- |
| `/bmd/bmdguest/searches/<run_name>/` (`BMD_SEARCH_WORKSPACE_DIR`) | fresh per attempt; `run_name` is `uspex_search-<UTC timestamp>-<first 8 of attempt ID>` |
| `…/<run_name>/submission.json` | the search submission spec (schema `bmd_compute.structure_search_submission`; POTCAR identities, no contents) |
| `…/<run_name>/bmd_backend_runtime/` | the versioned runtime package, installed by the runtime-package guard; the launch bridge is its `structure_search/bridge/bmd_bridge.sh` |
| `…/<run_name>/search/` | bound `INPUT.txt` and `prepared_inputs.json`, the other prepared inputs, and `Specific/POTCAR_<El>` (mode 0600) |
| `/bmd-db/guest/logs/structure_search/submission_attempts/<id>.json` (`BMD_SEARCH_LOGS_DIR`) | remote attempt state (`STAGING` → `STAGED`) and `<id>.lock`, separate from calculation attempt states |
| `/bmd/bmdguest/uspex_cache/` (`BMD_USPEX_CACHE_DIR`) | production USPEX unpack cache; its ready marker must exist |

Sequence:

1. Local checks before contacting POWER: the attempt exists and is
   `PREPARED_LOCALLY` (409 `attempt_not_prepared` while `PREPARING`), the
   digest matches (409 `plan_digest_mismatch`), the local preparation still
   verifies (409 `preparation_damaged`), the plan is current (409
   `plan_changed`) and the bridge in today's runtime package has the hash in
   the plan (409 `bridge_changed`).
2. The record becomes `STAGING` with the run name, the submission spec and
   the runtime package's manifest digest, all fixed for every later retry.
3. On POWER, under the attempt's remote claim (409 `staging_in_progress` if
   held): a new attempt refuses an existing run directory (409
   `run_directory_exists`; it is never reused), records remote `STAGING`,
   which reserves the run name for this attempt, and then creates the
   directory exclusively. A stand-alone, read-only inventory of the run
   directory (kind, mode and SHA-256 of every entry; no contents) decides
   what is uploaded (see *Interrupted staging* below). Each file is written
   to its own temporary name and moved into place atomically.
4. A finalize step runs on POWER from the uploaded package after the
   runtime-package guard has verified it: it reloads the plan and checks its
   digest, checks the cache ready marker (409 `uspex_cache_not_ready`), copies
   each licensed POTCAR from the POWER library into `search/Specific/`
   without replacing any file, checks TITEL, ENMAX and SHA-256 of source and
   copy against the plan (409 `potcar_unavailable`,
   `potcar_identity_mismatch`), and runs `verify_execution_binding`: the
   whole search directory must be exactly what the plan renders with the
   bridge path bound, and the bridge must be a regular file with the planned
   SHA-256.
5. Only when the finalize report matches what the service expects, and a
   final inventory shows every file present, nothing extra and no temporary
   left, is the remote state set to `STAGED`, then the local record.
   Anything else gives 409 `staging_unconfirmed`, `staging_incomplete` (or
   the reported code); success is never claimed for a partial staging.

The bound `INPUT.txt` and manifest are rendered by the service from the
frozen plan and verified byte for byte on POWER by the uploaded code.
POTCAR contents never pass through the API, the ledger or a response.

`staging` in the response: `location` (`power`), `run_name`,
`remote_state`, `started_at`, `staged_at`, `bridge_path_bound`,
`bridge_sha256`, `runtime_package_manifest_sha256`, `files`, `potcars`
(element, symbol, TITEL, ENMAX, dataset SHA-256), `contains_potcar_contents:
false`.

Idempotency and recovery: a `STAGED` attempt is returned from the ledger
without contacting POWER (`GET` likewise). A `STAGING` attempt is resumed
in the same run directory with the same spec; if the runtime package
changed meanwhile it gives 409 `runtime_package_changed`. If POWER is
already `STAGED` (the service stopped before recording it), the staging is
re-verified read-only and adopted. Other errors: 503 `remote_busy` (limiter),
409 `remote_attempt_mismatch` (fingerprint), 503 `staging_unavailable`
(transport).

### Interrupted staging

Every file staging writes on POWER (uploads, and the POTCAR copies made by
the finalize step) goes first to a temporary sibling named for one attempt
and one upload operation, then is moved into place without replacing
anything:

```
.<target name>.bmdstage-<attempt id>-<first 16 hex of the content SHA-256>.part
```

(for a POTCAR, the planned dataset SHA-256). A retry of the same attempt,
under the same remote claim and after the fingerprint check, applies these
rules to the inventory:

| Found | Action |
| --- | --- |
| an expected file with the expected SHA-256 | kept, never rewritten |
| an expected file that differs, a symbolic link, an unexpected file or directory | 409 `staging_damaged`, nothing touched |
| this attempt's temporary for a missing target, same content tag, complete bytes and mode 0640 | moved into place |
| the same, but partial or without its final mode | rewritten in place (superseded), then moved |
| this attempt's POTCAR temporary for a missing copy | rewritten in place, linked into place, then removed |
| this attempt's POTCAR temporary that is the same file as the completed copy (interrupted after the link) | removed |
| a temporary of another attempt, of other content, for an unknown target, beside a completed target, an unattributable `.tmp`, or a POTCAR temporary that is not the completed copy | 409 `staging_ambiguous`, nothing touched |
| the remote `STAGING` state but no run directory (interrupted after the reservation) | the directory is created |

Completed files, the local preparation and the attempt states are never
modified or deleted. A request that dies while holding the claim leaves
`submission_attempts/<id>.lock`: later requests get 409
`staging_in_progress` (a live request cannot be told from a dead one).
After checking that no staging request is running, an operator may remove
that empty lock directory; nothing else needs repair. A refused attempt is
kept for inspection; use a new attempt ID.

### Finalization timeout and timings

The finalize step (staging and `reverify_staged_search`) runs as one SSH
command with a dedicated 300 s timeout (`FINALIZE_TIMEOUT_S`); every other
remote command, and ordinary calculations, keep the runner's 60 s default.
A timeout leaves the attempt `STAGING` (503 `staging_unavailable`) and is
recovered by repeating the request. The service logs each phase's duration
and, from POWER, where finalization spent its time (guard and imports,
package re-check, plan load, POTCAR copy and identity, staged-input
verification); the timers wrap the verified package's functions in the
remote process and change neither the package nor the checks. Measured on
POWER for Si (7 Nov 2026): finalization 74.9 s, of which about 37 s POTCAR
identity and about 20 s staged-input verification
(`reference/uspex_pilot/staging_validation/RESULTS_20261107_5ead0670.md`).

### What `STAGED` means

`STAGED` (and `staging.remote_integrity: "last_verified"`,
`staging.last_verified_at`) records what POWER verified at that time, not a
guarantee that the run directory is still intact: files on POWER can change
afterwards, and reads (`GET`, a repeated staging `PUT`) never contact POWER.
`compute_api.structure_search_execution.reverify_staged_search` performs a
fresh, read-only verification (local preparation, current plan, same
runtime package, remote state and fingerprint, the guard-verified package,
plan digest, bound bridge, POTCAR identities and a complete inventory) and
changes nothing. Step 4D must run it inside the same remote claim as
submission, before an attempt can become submission-ready; until then
`submission_available` is always false.

`STAGED` is not the calculation `PREPARED` state: the existing claim and
`sbatch` paths refuse it. Submission is a later step.

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
nodes, MPI tasks, memory, walltime, partition, account and QOS; the resource
selection (sources, recommended walltime, basis and stage classes); loaded
modules and the VASP command template; parity-critical package versions; and
the versions of the admission, automatic DFT+U, method-consideration,
Custodian, runtime-parity, automatic-resource and plan-digest policies
(plan-digest version 2).

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
equals `expected_plan_digest`, and then atomically creates a ledger record
binding the attempt ID to:

- the authenticated principal;
- the SHA-256 of the validated request (structure, workflow, resources,
  labels);
- the plan digest;
- the run timestamp used to build the submission specification (unique among
  ledger attempts, so attempts never share a run directory);
- the resulting submission-attempt fingerprint.

Every later PUT for that ID must come from the same principal (otherwise 403
`attempt_forbidden`) with the identical request (otherwise 409
`attempt_request_mismatch`).

An exact retry of an attempt whose submission has already started (ledger
submission status reserved, uncertain or submitted, and `expected_plan_digest`
equal to the recorded plan digest) is recognised from the ledger alone, before
the current planning policy is applied, and is reconciled from the remote
attempt state: SUBMITTED returns the recorded job and SUBMITTING answers
`submission_uncertain`, without replanning or resubmitting. This keeps retries
idempotent for attempts registered under an earlier plan-digest version or
resource policy.

Any other PUT for an existing attempt, including a reserved attempt whose
remote state is still PREPARED, rebuilds the submission specification with the
recorded run timestamp and attempt ID, so the run directory, run name and
fingerprint are unchanged, and it must still resolve to the recorded plan
digest (409 `plan_changed`) and fingerprint (409
`attempt_fingerprint_mismatch`). An attempt registered under an earlier policy
that still needs preparation or submission therefore fails closed and needs a
new attempt ID.

The remote attempt state on POWER stays the authority. The PUT then:

1. reads it through a server-owned connection; a state whose attempt or
   fingerprint differs is never touched (409 `attempt_fingerprint_mismatch`);
2. if there is no remote state, prepares with `prepare_remote_submission`;
3. if it is PREPARED: refuses (409 `runtime_package_changed`) when the
   prepared runtime package differs from the running service's, because a
   prepared attempt is never re-prepared or replaced; returns `prepared` when
   `submit` is false; otherwise atomically records submission intent and
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

### Resource policy

Machine plans and attempts use the same automatic resource selection and the
same `ExecutionResources` validation as the browser. CPU and memory values must be among the browser's offered
choices, walltime uses the same accepted value, and partition and account are
always Compute's server-owned values. The API does not add separate CPU,
memory, walltime, active-job, submission-rate or attempt-rate ceilings; POWER
and SLURM remain responsible for allocation limits, scheduling and fair share.

The ledger still records submission intent under its exclusive lock. This is
part of retry and exactly-once submission semantics, not a job quota.

### Execution errors

In addition to the planning errors: 403 `attempt_forbidden`; 404
`attempt_not_found`; 409 `plan_digest_mismatch` (with the resolved
`plan_digest`), `attempt_request_mismatch`, `plan_changed`,
`attempt_fingerprint_mismatch`, `runtime_package_changed`,
`attempt_in_progress`, `submission_not_started`, `submission_uncertain` (with
the attempt projection) and `remote_state_invalid`; 422 `invalid_attempt_id`;
502 `prepare_failed` and `submit_failed`; 503 `execution_not_configured`, `remote_busy`,
`remote_unavailable`, `remote_state_unconfirmed` and
`submission_outcome_unconfirmed`. No error contains remote output, paths,
tracebacks or credentials.
