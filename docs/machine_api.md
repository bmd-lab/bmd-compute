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
| `PUT /api/v1/structure-search-attempts/{attempt_id}/promotion` | `submit` | Acknowledge the resource request and promote a `STAGED` attempt to submission-ready (`PREPARED` / `SEARCH_PREPARED`); idempotent; never submits |
| `PUT /api/v1/structure-search-attempts/{attempt_id}/submission` | `submit` | Submit a `PREPARED` attempt to SLURM at most once (`SUBMITTED` / `SEARCH_SUBMITTED`), or reconcile an uncertain submission read-only; never resubmits |
| `POST /api/v1/structure-search-attempts/{attempt_id}/observations` | `read` | Observe a submitted search once on POWER (read-only) and append the evidence; rate-limited |
| `PUT /api/v1/structure-search-attempts/{attempt_id}/cancellation` | `cancel` | Request cancellation of a submitted search's job; `scancel` by exact job ID after verifying its identity; repeatable |
| `PUT /api/v1/structure-search-attempts/{attempt_id}/submission/remote-record` | `submit` | Complete a lost `SEARCH_SUBMITTED` record for a `SUBMITTED` attempt after verifying its job against SLURM; idempotent; never submits |
| `PUT /api/v1/attempts/{attempt_id}` | `prepare` (prepare only) or `submit` (prepare and submit) | Prepare, or prepare and submit, one calculation; idempotent per attempt ID |
| `GET /api/v1/attempts/{attempt_id}` | `read` | The attempt's authoritative state, job identity and scheduler state |

`identity`, `plans` and `structure-search-plans` never contact POWER. The attempt routes use Compute's
existing remote preparation, submission and monitoring machinery with
server-owned connection profiles; see [Execution](#execution). There is no
machine route for cancelling a calculation or retrieving results (structure
searches have their own cancellation route, scope `cancel`). The browser routes are
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
- Scope vocabulary: `read`, `plan`, `prepare`, `submit`, `cancel`. An unknown
  scope makes the whole file invalid. `plan` never grants execution: `prepare`,
  `submit` and `read` are separate scopes (see [Execution](#execution)).
  `cancel` is separate from `submit`: a token that may submit cannot cancel
  without `cancel`, and `cancel` grants nothing else. Cancellation is still
  limited to the attempt's owner and its own recorded job ID.
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
  "search": {"min_atoms": 3, "max_atoms": 12, "num_generations": 30, "walltime": "168:00:00",
             "random_seed": 20261126, "excluded_nodes": "compute-0-270"},
  "expected_plan_digest": "sha256:..."
}
```

- `formula` (required): at most 64 characters; read by the same formula
  reader as the browser. `search` and every key in it are optional (omitted:
  the provisional search default). `expected_plan_digest` (optional): the
  request is refused with 409 `plan_changed` (carrying the current
  `plan_digest`) if it now plans differently.
- Two optional `search` fields have **no default**; omitted (or `null`), the
  request, its identity and the plan are exactly as before they existed:
  - `random_seed`: an integer from 1 to 2147483647, written to USPEX
    `INPUT.txt` as `<n> : fixRndSeed`. The plan records the seed requested,
    the line written and `effective_rng_determinism` (not verified: whether
    USPEX 25.0 reproduces a search with the same seed has not been tested);
    `prepared_inputs.json` records the seed written. Otherwise refused with
    `must_be_integer_1_to_2147483647`.
  - `excluded_nodes`: a SLURM host list (`compute-0-270`,
    `compute-0-[270,271]`, at most 64 nodes) excluded **in addition to** the
    deployment's `BMD_SEARCH_EXCLUDED_NODES`, which it can never remove or
    change. The effective `--exclude` is the deployment default followed by
    the new nodes; the resource request also records
    `exclude_deployment_default` and `exclude_additional`. It is covered by
    the plan digest, the resource acknowledgement and the job script. A
    non-string value is refused with `must_be_slurm_hostlist_string`; a string
    that is not a usable host list gives 422 `calculation_invalid`.
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

## Structure-search execution setting

Every structure-search route that writes to POWER or submits (staging,
promotion, submission, completing a lost remote record) answers 503
`structure_search_execution_disabled` and changes nothing unless the service
was started with `BMD_STRUCTURE_SEARCH_EXECUTION=enabled` (exactly that value).
The check comes after authentication and the scope check. Production leaves
it unset: USPEX searches stay experimental and the browser page is a preview.
Planning, local preparation, observation and cancellation are not gated, so
that a search already submitted can always be observed and stopped. The
validation kit (`reference/uspex_pilot/staging_validation/`) enables it only
for its own in-process service.

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
submission, before an attempt can become submission-ready.
`submission_available` is true only for a `PREPARED` attempt.

`STAGED` is not the calculation `PREPARED` state: the existing claim and
`sbatch` paths refuse it. Submission is a later step.

## Structure-search promotion (step 4D-2)

`PUT /api/v1/structure-search-attempts/{attempt_id}/promotion` (scope
`submit`, the attempt's owner) takes a `STAGED` attempt to `PREPARED`:
submission-ready, freshly verified on POWER. It never runs `sbatch`, `srun`,
USPEX or VASP; submission is the separate route below.

Body, exactly:

```json
{"expected_plan_digest": "sha256:...",
 "resource_acknowledgement": {"acknowledged": true, "resources_sha256": "sha256:..."}}
```

`resources_sha256` is the `resources.resources_sha256` that `GET` reports for
a `STAGED` or `PREPARED` attempt: the SHA-256 of the acknowledgement statement
and the plan's exact provisional resource request (`status`, `approved`,
`request`, `sbatch_directives`; `job.resource_request_view`). Echoing it is
the user's explicit acknowledgement of exactly that request. `submit` is
refused (`submission_not_available`).

Sequence:

1. Local checks, without contacting POWER: owner (403), plan digest (409
   `plan_digest_mismatch`), `STAGED` (409 `attempt_not_staged`; a
   `PREPARED` attempt returns its stored result), intact local preparation,
   plan still current (409 `plan_changed`), today's runtime package equal to
   the staged one (409 `runtime_package_changed`), and the echoed resource
   hash (409 `resources_changed`, with the current `resources_sha256`).
2. The first request freezes the promotion in the local record: the
   resource acknowledgement (schema
   `bmd_compute.structure_search_resource_acknowledgement` v1; principal from
   the token, server time, SHA-256 over the record), the VASP module snapshot
   (`backend.config.MODULES`) and the job script's SHA-256. The record stays
   `STAGED` until POWER confirms; retries rebuild the same script from the
   frozen inputs (409 `job_script_changed` if they cannot) and a different
   acknowledgement is refused (409 `acknowledgement_mismatch`).
3. On POWER, under the attempt's claim (409 `promotion_in_progress`) and the
   operation limiter, with the remote fingerprint checked: from `STAGED`, the
   run directory is reconciled as in staging with `bmd_search_job.sbatch` and
   `resource_acknowledgement.json` added to the expected files (an unfinished
   upload of this attempt is completed; a differing script, a `job_started`
   or any other foreign file gives 409 `staging_damaged` and is kept), the two
   files are uploaded only where absent (mode 0640), then a fresh read-only
   verification: runtime package (guard), frozen plan, bound bridge, POTCAR
   identities, staged inputs, the job script's SHA-256 and `bash -n`, the USPEX
   readiness check on the login node (payload SHA-256 and version, cache
   ready marker and unpacked program, cache writability, Singularity, runtime
   settings; not the compute-node-only MPI library and SLURM `srun` version,
   which the job checks on its compute node; it runs no SLURM command; 409
   `runtime_check_failed` with its `problems`), and a complete
   inventory. Only then is the remote state replaced atomically with
   `SEARCH_PREPARED` and a promotion record (schema
   `bmd_compute.structure_search_promotion` v1: job script name, SHA-256 and
   schema, acknowledgement and resource hashes, modules, runtime-package and
   plan digests, verification time, readiness check, verified files and
   POTCAR identities). A remote `SEARCH_PREPARED` with this attempt's frozen
   script and acknowledgement (an earlier request finished on POWER) is
   adopted after the same verification; other identities give 409
   `promotion_mismatch`; any other remote state 409 `transition_not_allowed`.
4. The local record becomes `PREPARED`.

Remote search attempt states (`STAGING`, `STAGED`, `SEARCH_PREPARED`) are
written atomically: to an attempt-bound temporary file beside the state file,
then renamed over it, so an interruption leaves the previous complete state.
Calculation attempt states are written as before.

The response adds `promotion` (`remote_state`, `started_at`, `verified_at`,
`remote_integrity: "last_verified"`, `job_script`, `acknowledgement`,
`modules`, `runtime_check`, `in_progress`), `promotion_available` and
`submission_ready`. `PREPARED`, like `STAGED`, is what POWER last verified,
not current integrity: reads never contact POWER, and submission (step 4D-3)
verifies again under its own claim. The job's own start (on the compute node)
repeats the input and runtime checks independently.

Other errors are as for staging; a transport failure is 503
`promotion_unavailable`, after which the same request resumes.

## Structure-search submission (step 4D-3)

`PUT /api/v1/structure-search-attempts/{attempt_id}/submission` (scope
`submit`, the attempt's owner) submits a `PREPARED` attempt to SLURM at most
once: locally `PREPARED → SUBMITTING → SUBMITTED`, on POWER `SEARCH_PREPARED
→ SEARCH_SUBMITTING → SEARCH_SUBMITTED`. It never cancels, monitors, starts
finalist refinement or interprets results. Ordinary calculation submission
(`PUT /api/v1/attempts/{id}`) is unchanged and refuses search states.

Body, exactly:

```json
{"expected_plan_digest": "sha256:...",
 "job_script_sha256": "<64 hex: promotion.job_script.sha256>",
 "resource_acknowledgement": {"acknowledged": true, "resources_sha256": "sha256:..."}}
```

Sequence:

1. Local checks, without contacting POWER: owner (403), plan digest (409
   `plan_digest_mismatch`), state (409 `attempt_not_prepared` unless
   `PREPARED`, or a `SUBMITTING`/`SUBMISSION_UNCERTAIN`/`SUBMITTED` attempt
   being repeated), the echoed job script (409 `job_script_mismatch`) and
   resource request (409 `acknowledgement_mismatch`) equal to the promotion,
   intact local preparation, current plan (409 `plan_changed`), the staged
   runtime package (409 `runtime_package_changed`), and the job script
   reproduced byte for byte from the frozen inputs (409
   `job_script_changed`). A `SUBMITTED` attempt returns its stored result.
   The record then becomes `SUBMITTING`.
2. On POWER, under the attempt's claim (409 `submission_in_progress`) and the
   operation limiter, with the fingerprint and the promotion identities
   checked (409 `promotion_mismatch`):
   - `SEARCH_PREPARED`: a fresh read-only verification of the whole run
     directory (as at promotion, including the job script, acknowledgement and
     login-node readiness check). Then `SEARCH_SUBMITTING` is written
     atomically, and only then `cd <run_dir> && sbatch --parsable
     bmd_search_job.sbatch` runs, once, with no command-line options: the
     resources are the script's frozen `#SBATCH` lines, the acknowledged
     request. A refusal before `sbatch` (verification, readiness) returns the
     attempt to `PREPARED` and leaves POWER at `SEARCH_PREPARED`.
   - `SEARCH_SUBMITTING` (an earlier outcome is unknown): read-only
     reconciliation, never `sbatch`.
   - `SEARCH_SUBMITTED`: adopted as recorded.
3. Outcomes of `sbatch`:

| Outcome | POWER | Local | Response |
| --- | --- | --- | --- |
| A job ID | `SEARCH_SUBMITTED` | `SUBMITTED` (recorded locally as soon as `sbatch` answers) | 200 with `submission.job_id` |
| Definite refusal (`sbatch: error`, or `sbatch` not found; no job ID) | `SEARCH_PREPARED` with `last_submission_error` | `PREPARED` | 502 `submit_rejected` with `sbatch_error` |
| Anything else (lost connection, timeout, exit 255, unreadable output) | `SEARCH_SUBMITTING` | `SUBMISSION_UNCERTAIN` | 409 `submission_uncertain` |
| A job ID, but the `SEARCH_SUBMITTED` write failed (or the service stopped) | `SEARCH_SUBMITTING` | `SUBMITTED`, `remote_record_pending: true` | 200; complete with `…/remote-record` |

An uncertain attempt is never submitted again automatically. Repeating the
request only reconciles: `squeue` and `sacct -X` (this account, last 7 days)
are read, records are grouped by job ID, and each job concerning the attempt
(its comment, its run directory, or its job name with no directory) is
classed:

- **verified**: the SLURM comment `bmd_attempt:<attempt_id>` (the primary
  identity), the job name `bmds-<attempt_id[:8]>` and the working directory
  (the run directory) are all reported and all the attempt's;
- **conflict**: something contradicts the identity (two values for one field
  of one job, this comment with another name or directory, this directory or
  name under another comment);
- **unverified**: nothing contradicts it, but a field (usually the comment,
  which accounting may not keep) is not reported. It is never adopted.

| Reconciliation | Result |
| --- | --- |
| Exactly one verified job, no other candidate | adopted: `SEARCH_SUBMITTED`, local `SUBMITTED`, `reconciled: true` |
| No job (accounting may lag) | 409 `submission_uncertain`; repeat later |
| Only unverified candidates | 409 `submission_unverified` |
| Several candidates, at least one verified | 409 `submission_ambiguous` |
| Any conflict | 409 `submission_conflict` |
| `squeue` or `sacct` did not answer | 409 `submission_uncertain` |

Another attempt's job is never a match: its comment and run directory differ,
so it is not a candidate even when its job name (the first eight characters of
the UUID) is the same; reported without a directory under that name, it is a
conflict. Job IDs are compared whole (an array task or range is its own
candidate). A job ID on its own, supplied or recorded, is never evidence: the
job must carry the full identity.

Each refusal keeps `SUBMISSION_UNCERTAIN` locally and `SEARCH_SUBMITTING` on
POWER and carries the `reconciliation` summary (`matches`, `unverified`,
`conflicts`, `checked_at`). An empty comment is never read as "no job". If the
record is uncertain but POWER shows `SEARCH_PREPARED`, nothing is submitted
(409 `submission_uncertain`): only an operator could have reset it.

### Completing a lost remote record

`PUT /api/v1/structure-search-attempts/{attempt_id}/submission/remote-record`
(scope `submit`, the owner), body exactly `{"expected_plan_digest":
"sha256:...", "job_id": "<the recorded job ID>"}`, for a `SUBMITTED` attempt
with `remote_record_pending: true` (`remote_record_recovery_available`). It
never runs `sbatch`. Locally: owner, plan digest, `SUBMITTED` (409
`attempt_not_submitted`) and the echoed job ID (409 `job_id_mismatch`); with
nothing pending it returns the stored response without contacting POWER. On
POWER, under the attempt's claim:

1. The remote state must be `SEARCH_SUBMITTING` (or `SEARCH_SUBMITTED`) with
   this promotion and this attempt's frozen job identity (attempt, comment,
   name, directory, job script), and no other job ID (409
   `promotion_mismatch`, `submission_conflict`; any other remote state 409
   `transition_not_allowed`).
2. SLURM must report the recorded job ID as the one verified job for this
   identity (as above). Otherwise nothing is written: 409 `job_not_visible`
   (not reported yet; retry later), `job_identity_unverified` (reported
   without its comment, name or directory), `submission_ambiguous`,
   `submission_conflict` (another job ID, or contradictory records); 503
   `scheduler_unavailable`.
3. `SEARCH_SUBMITTED` is written atomically, keeping the original
   submission record (request and verification times), the job ID and its
   `submitted_at`, and adding `remote_record_recovery` (time, principal,
   scheduler summary). An existing `SEARCH_SUBMITTED` naming the same job is
   adopted as is. The local record clears `remote_record_pending`.

### Uncertain submissions: future operator procedure (not implemented)

There is no automatic reset or retry, and no route for either. An attempt in
`SUBMISSION_UNCERTAIN` stays there until a later, separately reviewed
operator step. The intended procedure, for that step:

1. Wait beyond any accounting delay, then repeat the submission request
   (reconciliation only) and keep its `reconciliation` summary.
2. Read-only on POWER, as `bmdguest`: `squeue -u bmdguest -o
   '%i|%j|%k|%Z|%T'`, `sacct -X -P -u bmdguest -S <requested_at> -o
   JobID,JobName,Comment,WorkDir,State,Submit`, `scontrol show job <id>` for
   any candidate, and the run directory (`job_started`, `search_status.json`,
   `slurm-*.out`). Record what is found with the attempt ID.
3. If exactly one job is this attempt's, an operator route (to be designed)
   would adopt it by explicit job ID, recording who decided and on what
   evidence.
4. Only if no job exists, with evidence (no queue or accounting record over
   the window, no job output, no `job_started`), would an operator route
   reset the attempt to submission-ready, with an audit record, the remote
   state moved back explicitly, and a fresh verification before any new
   `sbatch`. Until then the attempt is not submitted; a new search needs a
   new attempt ID.

The response adds `submission`: `status`, `job_id`, `job_name`,
`attempt_comment`, `requested_at`, `submitted_at`, `reconciled`, `adopted`,
`remote_record_pending`, `remote_record_recovery_available`,
`remote_record_recovery`, `uncertain`, `uncertain_reason`, `last_error`,
`reconciliation`, and three outcomes kept apart: `scheduler` (state, not
refreshed by this release), `execution` (outcome, not read), and
`scientific_validity: "not_assessed"`; `finalists: "not_started"`. Reads never
contact POWER. A transport failure is 503 `submission_unavailable`; a held
claim 409 `submission_in_progress` (the operation limiter, 503
`remote_busy`).

The submission is service-side code only: the runtime package is unchanged,
so an attempt promoted by step 4D-2 (such as `14bfbbac`) can be submitted.

## Structure-search monitoring and cancellation (step 4D-4)

Service-side only (`compute_api/structure_search_monitoring.py`): the runtime
package is unchanged. Reads (`GET`) never contact POWER; there is no
background poller.

### Observations

`POST /api/v1/structure-search-attempts/{attempt_id}/observations` (scope
`read`, the owner; no body) for an attempt in `SUBMITTED`,
`CANCEL_REQUESTED` or a terminal state (409 `attempt_not_submitted`; 409
`submission_uncertain` for an uncertain submission, which has no bound job).
Within `MIN_OBSERVATION_INTERVAL_SECONDS` (60 s) of the last observation it
returns that one (`observation.performed: false`, `retry_after_s`) without
contacting POWER.

One read-only SSH session through the operation limiter, without the
attempt's claim (nothing is written on POWER): `squeue -h -u "$(id -un)" -o
'%i|%j|%k|%Z|%T|%r|%V'` (all of this account's queued jobs), `sacct -X -n -P
-j <job> -o JobID,JobName,Comment,WorkDir,State,ExitCode,Submit,Start,End`,
POWER's clock (`date +%s; date +%Y-%m-%dT%H:%M:%S`), the remote attempt
state (fingerprint checked), and, through the contained reader (no symbolic
links, bounded), `job_started`, `search_status.json` and
`bridge_spool/heartbeat` in the run directory. The SLURM output file is never
read.

Time stamps: both SLURM commands run with `TZ=UTC SLURM_TIME_FORMAT=standard`,
so every time is ISO 8601 UTC, and POWER's clock is read in the same session
(`date -u +%s`, `date -u +%Y-%m-%dT%H:%M:%S`, which must agree, and `date
+%z`, POWER's local offset, kept as evidence), bracketed by the service's
clock to measure the skew between the hosts. A job's submission time must lie
inside the attempt's submission window: from the attempt's `requested_at`
(recorded before verification and `sbatch`) to the moment the job was bound
(`sbatch`'s answer, or the verified reconciliation), shifted by the measured
skew, ±60 s plus the skew's measurement uncertainty (whole-second rounding and
clock drift; reused job IDs are months apart). It is `consistent`;
`contradicted` when it lies outside by more than any time-zone offset (14 h),
which means another job under a reused ID; or `unverified` — never guessed —
when a time is not in the expected UTC form, lies after POWER's own clock, is
out of order, lies outside the window by less than a time-zone offset (the
reason names POWER's offset when the difference matches it), POWER's clock was
not readable, or the skew exceeds 300 s.

Identity: the job must be the one the attempt bound (`binding`: `sbatch`'s
own answer, a verified reconciliation, or POWER's record of either;
otherwise `job_binding_not_established`), reported with the attempt's exact
job name and run directory, and submitted inside the window. A different
non-empty comment, any other field that differs, two values for one field,
another job carrying this attempt's comment or directory, a submission time
`contradicted` as above, `job_started` or `search_status.json` naming another
job, a remote record naming another job or identity, or a remote state other
than `SEARCH_SUBMITTING`/`SEARCH_SUBMITTED`/`SEARCH_CANCEL_REQUESTED` block any
state change and are recorded for operator review. With the comment reported
the identity is `verified`; with no comment reported (accounting may not keep
it) it is `partial`, which is enough to follow an already-bound job to its
terminal state but never enough to adopt an uncertain submission (step 4D-3)
or to cancel; a missing name or directory, or `unverified` times, leave it
`unverified` (no state change).

Scheduler state: `squeue` while the job is queued or running, otherwise
`sacct`; `UNKNOWN` when neither reports it. A terminal state is recorded only
when accounting reports it, `squeue` does not report a different one, and the
job ended at least 120 s (`SETTLE_SECONDS`) before POWER's own clock;
otherwise `classification.blocked` is `not_terminal`, `terminal_settling`
(with `remaining_s`), `job_not_reported`, `identity_not_established` or
`contradiction`. The terminal state is `lifecycle.classify_job`:
`SEARCH_COMPLETED` only for SLURM `COMPLETED` with `search_status.json`
recording `uspex_finished` and master exit status 0; `CANCELLED` →
`SEARCH_CANCELLED`; `TIMEOUT`/`DEADLINE` → `SEARCH_TIMED_OUT`; anything else,
or a missing, foreign or unreadable record, `SEARCH_FAILED`. Once recorded, a
terminal state never changes; a later observation that disagrees is added to
`monitoring.contradictions` (`operator_review: true`).

Evidence: each observation is a new file
`<state root>/structure_search/observations/<attempt>/<NNNNNN>.json` (mode
0600, linked into place so an existing file is never replaced, naming the
previous file's SHA-256): commands, POWER's clock, every scheduler record,
identity assessment, execution record summary (with the status file's
SHA-256), heartbeat age, remote state, classification and classifier
version. The local record keeps a summary (`monitoring`: `observation_count`,
`last`, `terminal`, `contradictions`). An interrupted record update is applied
from the evidence by the next request (503 `observation_unavailable` until
then). Calculation records and the calculation ledger are untouched.

The response adds `observation`, `observation_available`, `monitoring`, and
fills `submission.scheduler` (`state`, `source`, `observed_at`, `settled`) and
`submission.execution` (`outcome`, `source: search_status.json`, `sha256`)
from the last observation; `scientific_validity` stays `not_assessed`.

### Cancellation

`PUT /api/v1/structure-search-attempts/{attempt_id}/cancellation` (scope
`cancel`, the owner), body exactly `{"expected_plan_digest": "sha256:...",
"job_id": "<digits>", "confirm": true}` (422 otherwise). Local checks: plan
digest, the recorded job ID (409 `job_id_mismatch`), `SUBMITTED` or
`CANCEL_REQUESTED` (409 `attempt_terminal`, `submission_uncertain`,
`attempt_not_submitted`), and a complete remote record (409
`remote_record_pending`).

1. **Requested**: the record becomes `CANCEL_REQUESTED` (who, when) before
   POWER is contacted.
2. **Remote request recorded**: on POWER, under the attempt's claim (409
   `cancellation_in_progress`), the remote state must be `SEARCH_SUBMITTED`
   or `SEARCH_CANCEL_REQUESTED` with this promotion, identity and job ID
   (409 `submission_conflict`, `transition_not_allowed`). A fresh identity
   check follows, every time: `squeue` must report the job queued or running
   with a `verified` identity (comment, name, directory, submission window)
   and no other job carrying it (409 `job_identity_unverified`,
   `submission_conflict`, `job_not_visible`; 503 `scheduler_unavailable`).
   Then `SEARCH_CANCEL_REQUESTED` is written atomically (once), and the local
   record notes it before anything is sent; if it cannot, nothing is sent.
3. **Command attempted**: `scancel <job ID>`, one numeric ID, never a name,
   user or other option: `command_sent`, `command_found_job_ended`,
   `command_failed` or `command_outcome_unknown`. A job no longer queued gets
   no command (`not_sent_job_not_active`).
4. **Observed**: only an observation of SLURM's settled terminal state ends
   it (`cancellation.observed`, `outcome: cancelled`, or
   `job_ended_before_cancellation_took_effect`; the state is what SLURM
   reports).

Where a request stops (`cancellation.last_refusal` gives the code and stage):

| Stopped | Local state | `cancellation.status` |
| --- | --- | --- |
| Refused or failed before any remote cancellation request exists (including a lost connection) | `SUBMITTED` (cancellation available again) | `refused_before_remote_request` |
| The remote write itself interrupted (it may or may not have landed) | `CANCEL_REQUESTED` | `remote_request_uncertain` |
| Remote request recorded, `scancel` not attempted (or a later refusal) | `CANCEL_REQUESTED` | `remote_recorded_command_pending` |
| `scancel` attempted, answer lost | `CANCEL_REQUESTED` | `command_outcome_unknown` |
| Settled terminal state observed | terminal (SLURM's) | `observed` |

Pending states are resumed only by repeating the request
(`resume_by_repeating`), which checks the identity afresh before any
`scancel`; observations never send anything. If an earlier request's remote
record is missing locally, the next observation adopts it
(`adopted_from_power`, `remote_recorded_command_pending`). Every command is
recorded (`cancellation.command`); earlier requests are kept
(`earlier_requests`). Nothing is resubmitted or restarted; a repeated
submission request returns the stored attempt.

New states, defined service-side: local `CANCEL_REQUESTED` (`SUBMITTED →
CANCEL_REQUESTED → terminal`, or back to `SUBMITTED` when refused before any
remote request) and remote `SEARCH_CANCEL_REQUESTED`
(`SEARCH_SUBMITTED → SEARCH_CANCEL_REQUESTED`). The calculation path refuses
them like every other search state.

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
