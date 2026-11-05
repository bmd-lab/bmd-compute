# Deployment And Operations

This document describes the current bmd-compute operating model. It is documentation of the present deployment policy, not a complete production runbook.

## Current Model

```text
authorized student
  -> TAU VPN / university network
  -> bmd-compute on TAU-hosted VM
  -> Paramiko as constrained shared bmdguest identity
  -> PowerSLURM leeburton-pool partition
  -> VASP
```

bmd-compute is currently intended to run as a lab-internal service reachable only from the TAU VPN or university network.

Do not expose the Uvicorn port directly to the public Internet.

Source-code visibility is not an access control. A public repository does not
change this network-bound deployment requirement.

## Network And Identity

The web app does not currently provide app-level login, SSO, CSRF protection, or private per-user ownership. The access boundary is the TAU network/VPN plus the constrained shared remote identity.

Resume by Job ID is not a private authorization boundary. Jobs visible through this path should be treated as shared service/group calculations.

If the app is later exposed outside the current VPN-bound model, authentication, authorization, CSRF posture, auditing, and per-user job ownership need to be redesigned before deployment.

## SSH Host Trust

Remote SSH connections fail closed. The service account running bmd-compute
must trust the POWER login host in an OpenSSH-compatible known-hosts file.
The connection layer reads the resolved `UserKnownHostsFile`, the service
account's standard `~/.ssh/known_hosts`, and system SSH known-hosts files.
An operator may set `BMD_SSH_KNOWN_HOSTS_FILE` to an explicit deployment-local
file. Unknown or changed keys are rejected; never replace this with automatic
host-key acceptance.

For stable browser Prepare/Submit forms across service restarts or multiple
workers, set `BMD_SUBMISSION_IDENTITY_SECRET` to the same high-entropy private
value for every process. Without it, bmd-compute uses a process-local ephemeral
key, which is secure but invalidates already-rendered forms after a restart.
Neither value belongs in the repository.

## Uvicorn Process

The app currently assumes a single bmd-compute Python/Uvicorn process unless an operator deliberately changes that deployment.

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

## Uploaded Structure-Search Results

Imports of USPEX results (`structure_search_import.md`) are the app's first
server-side uploads. They are written to `uploads/structure_search/` in the
checkout (git-ignored). Each import keeps only the USPEX result files, at
most 512 MB, and is deleted when a later import finds it older than 7 days
or beyond the newest 20. The web host therefore needs a few GB free in the
checkout. The folder may be deleted at any time; calculations built from an
import keep their origin in the submission record.

Structure origin tokens are signed with a key derived from
`BMD_SUBMISSION_IDENTITY_SECRET`. Without it, a restart invalidates origins
that were handed off but not yet built.

## Submission Idempotency

Submission idempotency is enforced with remote-side state under the BMD logs area, including submission attempt records and lock state.

This is designed to prevent duplicate SLURM jobs from repeated submit requests with the same attempt id. It is not a substitute for a durable application database or per-user job ledger.

## Resource Policy

Resources left on Automatic (the default) are chosen by the backend: 24 CPUs,
a 32, 64 or 128 GB memory bracket and a walltime estimated from the structure and workflow; see
`docs/automatic_resources.md`.

```text
partition = leeburton-pool
nodes = 1
ntasks = 24                 (Automatic)
memory = 128G               (Automatic)
walltime = estimated        (Automatic; explicit callers default to 72:00:00)
account = power-leeburton-users_v2
qos = owner100d
```

CPU counts, memory values, and queue are allow-listed in the backend. The account and QOS are fixed backend policy and are not user-editable.

Deploying a new automatic-resource policy changes the plan digest and attempt
fingerprint of recomputed plans. Machine-API attempts whose submission had
already started before the deployment stay retryable with the identical
request (the existing job, or `submission_uncertain`, is returned without
resubmitting); attempts that were registered or prepared but not yet submitted
fail closed with `plan_changed` and need a new attempt ID. See
`docs/machine_api.md`.

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
  workflow, correct VASP errors, or decide whether a stage converged.
- **Parity-critical for phonon workflows only** (phonopy, seekpath, "Tier 1P"):
  required at exactly the pinned versions, at preparation and on POWER, for any
  submission whose `workflow_spec` contains a `phonon_forces` stage. The
  required set is derived from the submission's own workflow on both sides.
  For every other submission they are recorded as supporting packages, so
  existing PBE/HSE06/DFT+U submissions keep running before they are installed. Preparation
  records their exact versions in `submission.json` (`runtime_parity`); the POWER
  runner reads its own before building the workflow and stops, without starting
  VASP, on any missing or different version.
- **Supporting** (monty, numpy, scipy, pydantic, pydantic-settings, maggma,
  ruamel.yaml): pinned for reproducible installs and recorded at run time, but a
  difference does not stop a run.

Deployment transition: install phonopy and seekpath at the pinned versions in
both the preparation environment and the POWER environment before preparing a
phonon calculation (`docs/phonon_workflow.md`). Until then, phonon preparation
fails with a parity error and every other workflow is unaffected.

Web-only, SSH, scheduler and test packages stay out of the constraints file.
Upgrading any pinned package is a deliberate change: update the constraints file,
install it in both environments, and re-run the test suite.

The runner also stops if atomate2's `VASP_INCAR_UPDATES` is non-empty or
`VASP_INHERIT_INCAR` is enabled (environment variable or `~/.atomate2.yaml`),
because either would change BMD's generated INCARs at run time. An atomate2
stage that is unsuccessful (not converged) always fails the workflow: BMD passes
`stop_children_kwargs={"handle_unsuccessful": "error"}` to every maker instead of
relying on atomate2's `VASP_HANDLE_UNSUCCESSFUL` setting.

## Structure Searches (USPEX)

Structure searches are being added (#58; design in
`docs/structure_search.md`). The settings, resources and checks below exist,
but no search can be submitted yet: preparation and `run_submission` are not
wired to them (#67, part 2).

### Settings

Each setting is in `backend/config.py` with a POWER default and can be
overridden with the environment variable named here, on the machine that
prepares submissions. Preparation records the values in `submission.json`
(`structure_search.runtime`), and the job on POWER checks against the recorded
values, not against its own environment.

| Setting | Variable | POWER default |
|---|---|---|
| USPEX 25.0 sandbox (unpacked image, not the `.sif`) | `BMD_USPEX_SANDBOX_DIR` | `/bmd/shared/uspex25_sandbox` |
| Shared unpack cache, used as `XDG_CACHE_HOME` for the USPEX command only | `BMD_USPEX_CACHE_DIR` | `/bmd/shared/uspex_cache` |
| Expected payload SHA-256 of `/opt/uspex/uspex` | `BMD_USPEX_PAYLOAD_SHA256` | `fc4a08e6…e87e95c8c` (USPEX 25.0, 11 Nov 2025) |
| Singularity executable | `BMD_SINGULARITY_EXECUTABLE` | `/powerapps/share/rocky9/mamba/mamba-1.5.8/envs/singularity-3.8.6/bin/singularity` |
| `I_MPI_PMI_LIBRARY` for `srun --mpi=pmi2` steps | `BMD_SEARCH_I_MPI_PMI_LIBRARY` | `/usr/lib64/libpmi2.so` |
| Launch mode (`srun_steps` or `mpirun_pinned`) | `BMD_SEARCH_LAUNCH_MODE` | `srun_steps` |
| Relaxations in parallel (USPEX `numParallelCalcs`) | `BMD_SEARCH_PARALLEL_RELAXATIONS` | `4` |
| MPI ranks per relaxation | `BMD_SEARCH_RANKS_PER_RELAXATION` | `48` |
| Bridge timeout, seconds (one relaxation step, queueing included) | `BMD_SEARCH_BRIDGE_TIMEOUT_S` | `14400` (provisional) |
| Sockets per node | `BMD_SEARCH_NODE_SOCKETS` | `2` |
| Cores per socket | `BMD_SEARCH_CORES_PER_SOCKET` | `96` |
| Nodes searches exclude (SLURM host list; empty for none) | `BMD_SEARCH_EXCLUDED_NODES` | `compute-0-[332-333]` |

`compute-0-332` and `compute-0-333` are excluded because they use a different
NUMA layout (NPS4, 8 × 24 cores), so the 2 × 96 layout does not hold there.
On the normal nodes, socket 0 holds CPUs 0–95 and socket 1 holds CPUs 96–191.
The bridge timeout is provisional, about twice the slowest whole relaxation
chain estimated in `docs/structure_search.md` §7; Phase 0 (#59) should
replace it with a measured value.

### Resources

A search uses its own preset, `StructureSearchResources` in
`backend/calculations/resources.py`. Ordinary workflows do not use it and are
unchanged.

```text
partition = leeburton-pool
nodes = 1, --exclusive
ntasks = relaxations in parallel x ranks per relaxation   (4 x 48 = 192)
memory = --mem=0           (the whole node; the partition default is 1 GB per job)
walltime = explicit        (required, at most 100 days; no automatic estimate)
account = power-leeburton-users_v2
qos = owner100d
exclude = compute-0-[332-333]
```

### Environment setup: unpack USPEX once

`/opt/uspex/uspex` in the image is a self-extracting program. It unpacks
itself (2 GB, about a minute on `/bmd`) into
`<cache>/uspex_sfx/<payload sha256>/` and writes a `.ready` marker. A search
job never unpacks it: the job-start check stops the run if `.ready` is missing.
After a new cache or a new USPEX build, unpack it once, as an operator, on a
compute node:

```bash
srun -p leeburton-pool -N 1 -n 1 -t 00:30:00 \
  python3 -m backend.structure_search.runtime unpack
```

Run it from a bmd-compute checkout (Python 3.9 or later; the module uses only
the standard library) with the same `BMD_USPEX_*` and
`BMD_SINGULARITY_EXECUTABLE` values as the web service. It does nothing if
`.ready` already exists. It refuses to start if a partial
`uspex_sfx/<sha>*` folder exists (the extractor would nest a fresh copy inside
it and fail); remove that folder after checking that no other unpack is
running, and run it again. `<cache>` and `<cache>/uspex_sfx` must be
group-writable with the setgid bit (`chmod g+ws`), because every USPEX start
writes a lock file there; if `<cache>` is not writable, USPEX silently unpacks
2 GB into `/tmp` instead.

`python3 -m backend.structure_search.runtime check` runs the job-start check
by hand and prints the result. `... record` prints the provenance record.

### Job-start check

For a structure-search submission, the POWER runtime-environment check (the
same step that checks package parity) also checks, before USPEX or VASP
starts:

- the sandbox exists and contains `/opt/uspex/uspex`;
- the `PAYLOAD_SHA256=` line in that file's header equals the prepared value,
  and its version banner, if present, says 25.0 (the sandbox itself is not
  hashed: the payload SHA-256 identifies the program);
- `<cache>/uspex_sfx/<sha>/.ready` and the unpacked `uspex_core/uspex` exist;
- `<cache>` and `<cache>/uspex_sfx` are writable by the job's user;
- the Singularity executable exists;
- in `srun_steps` mode, the PMI library exists and SLURM is 21.08 or later
  (`srun --exact`).

Every problem is reported at once and the run stops. The result is written to
`runtime_environment.json` under `structure_search_runtime`, with the version,
payload SHA-256, paths, the extractor file's size and modification time, and
the SLURM version.

## Future Production Work

Before broader production exposure, decide and document:

- systemd or equivalent process supervision
- reverse proxy and HTTPS termination
- authentication and authorization
- CSRF policy
- persistent job database or job history
- cross-process remote-operation limits if multiple workers are used
- log retention and incident response
