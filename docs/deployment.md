# Deployment And Operations

This document describes the current BMD Compute operating model. It is documentation of the present deployment policy, not a complete production runbook.

## Current Model

```text
authorized student
  -> TAU VPN / university network
  -> BMD Compute on TAU-hosted VM
  -> Paramiko as constrained shared bmdguest identity
  -> PowerSLURM leeburton-pool partition
  -> VASP
```

BMD Compute is currently intended to run as a lab-internal service reachable only from the TAU VPN or university network.

Do not expose the Uvicorn port directly to the public Internet.

Source-code visibility is not an access control. A public repository does not
change this network-bound deployment requirement.

## Network And Identity

The web app does not currently provide app-level login, SSO, CSRF protection, or private per-user ownership. The access boundary is the TAU network/VPN plus the constrained shared remote identity.

Resume by Job ID is not a private authorization boundary. Jobs visible through this path should be treated as shared service/group calculations.

If the app is later exposed outside the current VPN-bound model, authentication, authorization, CSRF posture, auditing, and per-user job ownership need to be redesigned before deployment.

## SSH Host Trust

Remote SSH connections fail closed. The service account running BMD Compute
must trust the POWER login host in an OpenSSH-compatible known-hosts file.
The connection layer reads the resolved `UserKnownHostsFile`, the service
account's standard `~/.ssh/known_hosts`, and system SSH known-hosts files.
An operator may set `BMD_SSH_KNOWN_HOSTS_FILE` to an explicit deployment-local
file. Unknown or changed keys are rejected; never replace this with automatic
host-key acceptance.

For stable browser Prepare/Submit forms across service restarts or multiple
workers, set `BMD_SUBMISSION_IDENTITY_SECRET` to the same high-entropy private
value for every process. Without it, BMD Compute uses a process-local ephemeral
key, which is secure but invalidates already-rendered forms after a restart.
Neither value belongs in the repository.

## Uvicorn Process

The app currently assumes a single BMD Compute Python/Uvicorn process unless an operator deliberately changes that deployment.

Important process-local state:

- remote-operation semaphore
- completed-result cache
- in-memory UI/runtime state

Running multiple workers multiplies process-local limits and caches. Do not treat the current semaphore or result cache as cross-process controls.

## Remote Operation Limits

Current defaults:

```text
BMD_MAX_CONCURRENT_REMOTE_OPERATIONS = 4
BMD_REMOTE_OPERATION_SLOT_TIMEOUT_S = 5
```

These settings limit simultaneous SSH-backed operations within one Python process. If all slots are busy for the timeout, the operation should fail fast with a user-facing busy response rather than creating unbounded SSH connections.

## Result Cache

Completed results are cached in process to avoid repeated remote parsing of the same completed job. The cache is bounded and non-durable.

Implications:

- restarting Uvicorn clears the cache
- clearing the cache does not affect cluster jobs
- Resume and Refresh Queue Status can still recover job state from SLURM/job records
- Load Results may parse remotely again after restart

## Submission Idempotency

Submission idempotency is enforced with remote-side state under the BMD logs area, including submission attempt records and lock state.

This is designed to prevent duplicate SLURM jobs from repeated submit requests with the same attempt id. It is not a substitute for a durable application database or per-user job ledger.

## Resource Policy

Current default resources:

```text
partition = leeburton-pool
nodes = 1
ntasks = 24
memory = 128G
walltime = 72:00:00
account = power-leeburton-users_v2
```

CPU counts, memory values, and queue are allow-listed in the backend. The account is fixed backend policy and is not user-editable.

## Scientific Runtime Environment

`constraints/scientific-runtime.txt` is the single specification of the
scientific stack for both environments:

- the web/preparation environment (`environment.yml`, whose pip section installs
  the constraints file), and
- the POWER execution environment `/bmd/bmdguest/envs/atomate2_remote`, which
  must hold exactly these versions.

The supported repository deployment path is `environment.yml` -> pip ->
`-r constraints/scientific-runtime.txt`. The constraints file defines the
package/version contract; which installer put a package in place is
deliberately not part of runtime parity, only the installed versions are.

It lists two tiers:

- **Parity-critical** (atomate2, pymatgen, pymatgen-core, custodian, emmet-core,
  jobflow, spglib): they generate the inputs, symmetry and k-points, run the
  workflow, correct VASP errors, or decide whether a stage converged. Preparation
  records their exact versions in `submission.json` (`runtime_parity`); the POWER
  runner reads its own before building the workflow and stops, without starting
  VASP, on any missing or different version.
- **Supporting** (monty, numpy, scipy, pydantic, pydantic-settings, maggma,
  ruamel.yaml): pinned for reproducible installs and recorded at run time, but a
  difference does not stop a run.

Web-only, SSH, scheduler and test packages stay out of the constraints file.
Upgrading any pinned package is a deliberate change: update the constraints file,
install it in both environments, and re-run the test suite.

The runner also stops if atomate2's `VASP_INCAR_UPDATES` is non-empty or
`VASP_INHERIT_INCAR` is enabled (environment variable or `~/.atomate2.yaml`),
because either would change BMD's generated INCARs at run time. An atomate2
stage that is unsuccessful (not converged) always fails the workflow: BMD passes
`stop_children_kwargs={"handle_unsuccessful": "error"}` to every maker instead of
relying on atomate2's `VASP_HANDLE_UNSUCCESSFUL` setting.

## Future Production Work

Before broader production exposure, decide and document:

- systemd or equivalent process supervision
- reverse proxy and HTTPS termination
- authentication and authorization
- CSRF policy
- persistent job database or job history
- cross-process remote-operation limits if multiple workers are used
- log retention and incident response
