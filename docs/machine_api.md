# Machine API (v1, planning only)

bmd-compute serves a small authenticated JSON API for machine clients such as
bmd-run. Phase 1A provides **planning only**:

| Route | Scope | Purpose |
| --- | --- | --- |
| `GET /api/v1/identity` | any valid token | Service identity, API and schema versions, the caller's principal and scopes |
| `POST /api/v1/plans` | `plan` | Resolve a structure and workflow request into the authoritative calculation plan |

There is no machine route for Prepare, Submit, Monitor, Resume or results, and
no route here contacts POWER, opens SSH, calls SLURM or writes files. The
browser routes are unchanged and do not use this authentication. The API is
not listed in `/openapi.json`.

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
  the whole file invalid. `prepare` and `submit` are reserved: no route accepts
  them yet.
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
