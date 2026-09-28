# BMD Compute Architecture

BMD Compute is a browser-based scientific workbench for introductory VASP workflows on TAU PowerSLURM. It is designed for BMD group project students and beginning computational users who need a guided path through DFT/VASP concepts before HPC operation becomes the main task.

Experienced computational researchers should still use normal SSH, SLURM, and scripting when they need full control. BMD Compute deliberately exposes a constrained, reviewed workflow surface.

## System Flow

```text
Browser
  -> FastAPI / Jinja
  -> stage-first WorkflowSpec
  -> pymatgen / atomate2 / jobflow
  -> Paramiko RemoteRunner
  -> PowerSLURM sbatch
  -> VASP / Custodian
  -> remote pymatgen result parser
  -> Results UI
```

The browser never talks directly to the cluster. FastAPI coordinates requests and renders templates. Scientific policy lives in backend modules. Infrastructure modules own SSH, SLURM, remote packaging, and result transfer.

## Cross-Repository Authority

BMD Compute is the authoritative implementation of the core BMD VASP generation pipeline: constructing, validating, executing, and provenancing BMD VASP calculations. It owns BMD's executable calculation methodology: the workflows, stages, settings and treatments it actually implements. BMDex supplies curated supporting scientific evidence, validation records, datasets, and non-core scientific tools outside that pipeline; it does not define BMD Compute methodology. BMD Agent observes exposed contracts, evidence, and infrastructure observations without duplicating their authority. Scientific validation and adoption of any methodology remain human judgment.

If a capability determines how BMD generates a VASP calculation, its authoritative implementation belongs in BMD Compute. If it provides supporting scientific data or tooling but is not part of the core VASP data-generation pipeline, it belongs in BMDex.

## Main Layers

### Browser

The browser is the product interface. It collects structure input, displays structure summaries, lets the user select ordered scientific stages, shows generated input previews, submits jobs, refreshes monitoring, and renders results.

The UI should not contain VASP policy. It may serialize form state, but it should not decide INCAR, KPOINTS, POTCAR, executable, resource, or restart behavior.

### FastAPI

FastAPI receives forms, normalizes them into backend objects, calls backend services, and renders `templates/index.html`.

The current `main.py` still owns more orchestration than the long-term ideal, especially around page state. New scientific or infrastructure behavior should still be added in backend modules rather than route handlers.

### Calculation Backend

The calculation backend owns:

- `CalculationSpec` compatibility objects
- `StageSpec` and `WorkflowSpec`
- registry validation and capability checks
- theory/stage INCAR policy
- resource policy
- generated input previews
- atomate2/jobflow construction
- workflow summaries and stage directory naming

The stage-first model is now the source of truth for new work. Recommended workflows and custom workflows both produce the same ordered `WorkflowSpec`.

### Remote Infrastructure

Remote operations use `RemoteRunner`, with Paramiko as the production implementation.

Remote preparation writes generated files as file data through SFTP rather than embedding input contents into shell arguments. This is required for large files such as HSE06 band-structure KPOINTS.

Every UI-triggered remote operation should open, use, and close its SSH/SFTP resources deterministically. A process-local bounded semaphore limits simultaneous remote operations. The default limit is 4, with a default remote-slot wait of 5 seconds.

### PowerSLURM Execution

The submitted sbatch script starts the BMD remote runner script rather than launching VASP directly from the preview. The remote runner reconstructs the workflow, configures atomate2/Custodian with the resolved VASP command, and executes stages in order.

`VASP_CMD` defaults to:

```text
mpirun -n $SLURM_NTASKS vasp_std
```

The runtime command resolver expands `$SLURM_NTASKS` from the actual environment or submission resources before passing argv to Custodian, so the executed command preserves the requested MPI task count.

SOC/non-collinear stages are routed stage-locally to `vasp_ncl`; ordinary stages use `vasp_std`.

### Monitoring And Results

Refresh Queue Status is monitoring-only. It queries scheduler state and renders the current status. Completed jobs show or retain the Load Results control.

Load Results is explicit. It invokes remote-side pymatgen parsing and returns a compact JSON result payload instead of transferring raw `vasprun.xml` through the web process. Result parsing has a bounded timeout and a bounded in-process completed-result cache.

Workflow-specific results are rendered through a generic visualization payload system. DOS and Band Structure currently use interactive Plotly panes with client-side PNG export.

## Data Flow

```text
Structure text/file
  -> pymatgen Structure
  -> StructureSummary
  -> WorkflowSpec
  -> GeneratedInputPreview
  -> SubmissionSpec
  -> Remote preparation state
  -> SLURM JobRecord
  -> Monitoring result
  -> ResultSummary and workflow visualizations
```

Every layer should carry structured state forward rather than re-parsing display text.

## Security And Access Model

Current deployment policy:

```text
authorized student
  -> TAU VPN / university network
  -> BMD Compute
  -> constrained shared bmdguest identity
  -> PowerSLURM
```

There is currently no app-level authentication, SSO, CSRF protection, or private per-user job ownership model. This is acceptable only for a VPN-bound lab service. It is not safe to expose BMD Compute directly to the public Internet.

Resume by SLURM job ID is a shared-service convenience, not an authorization boundary. Jobs submitted through the app should be treated as service/group calculations under the constrained shared identity.

## Operational Hardening

Current operational safeguards include:

- allow-listed CPU counts, memory values, and queue
- fixed backend account policy rather than a user-editable account field
- server-side submission idempotency state on the remote filesystem
- deterministic SSH/SFTP cleanup
- process-local bounded remote-operation concurrency
- explicit completed-result loading
- remote-side result parsing with timeout and compact payload
- production traceback hiding unless `BMD_DEBUG` is enabled
- structured submission provenance in `submission.json`

Known limitations:

- the SSH concurrency limit is process-local; multiple Uvicorn workers would multiply the effective limit
- the completed-result cache is in-process and non-durable
- app-level authentication and per-user job ownership are future work if the service moves beyond the VPN-bound shared model
- a durable application database is not yet implemented

## Design Rule

When adding a feature, prefer this path:

```text
validated scientific behavior
  -> reusable backend module
  -> route/controller integration
  -> browser presentation
```

Do not migrate notebook logic directly into FastAPI, and do not scatter VASP policy through templates or JavaScript.
