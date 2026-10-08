# bmd-compute Roadmap

This roadmap tracks complete scientific and operational capabilities. It should reflect the current repository, not the original prototype state.

## Done

### Project Foundation

- [x] FastAPI/Jinja browser application
- [x] backend package split for structures, calculations, workflows, submission, monitoring, results, and remote infrastructure
- [x] AGENTS.md architecture guidance
- [x] reference notebook retained as historical scientific evidence
- [x] pytest configuration with cache provider disabled for the Windows/Codex sandbox

### Structure And Workflow Construction

- [x] POSCAR and CIF input parsing through pymatgen
- [x] structure summary
- [x] stage-first `WorkflowSpec`
- [x] reusable `StageSpec`
- [x] `CalculationSpec` compatibility layer
- [x] recommended workflow selector
- [x] custom ordered stage cards
- [x] generated VASP input previews
- [x] generated SLURM/script policy preview
- [x] stage-local theory/modifier selection

### Scientific Workflows

- [x] PBE Geometry Optimisation
- [x] PBE Static Energy
- [x] PBE Double Geometry Optimisation
- [x] PBE Geometry Optimisation -> Static Energy
- [x] PBE Density of States
- [x] PBE Band Structure
- [x] HSE06 Geometry Optimisation
- [x] HSE06 Static Energy
- [x] HSE06 Geometry Optimisation -> Static Energy
- [x] HSE06 Band Structure with HSE06 Static Energy precursor
- [x] HSE06 Density of States with HSE06 Static Energy precursor
- [x] Desired Outputs: Energy only, Relaxed structure, Electronic density of states, Electronic band structure
- [x] Spin Polarised modifier, applied automatically in Desired Outputs by composition screen
- [x] DFT-D3 / DFT-D3(BJ) van der Waals correction, DFT-D3(BJ) applied automatically for two-dimensional bonded connectivity
- [x] SOC for PBE/HSE06 Static and HSE06 DOS/Band Structure, applied automatically for heavy elements
- [x] DFT+U: explicit in Custom workflows; automatic MP/pymatgen oxide/fluoride policy `bmd_compute.dft_u` v1 in Desired Outputs
- [x] declared v1 executable methodology (`docs/methodology.md`) with regression tests

### Execution Integrity

- [x] versioned run-record contracts (`bmd_compute.submission` v1, `bmd_compute.job_record` v1) with atomic remote writes
- [x] POTCAR records derived from the executable stage generators
- [x] submitted runs immutable: Prepare cannot rewrite a submitted attempt
- [x] shared scientific-runtime constraints for the preparation VM and POWER
- [x] runtime scientific-stack parity check and `runtime_environment.json` record
- [x] explicit unsuccessful-stage policy on every stage
- [x] stage options restricted to supported treatments (no free-form INCAR/KPOINTS)

### Remote Preparation, Submission, And Monitoring

- [x] Paramiko-backed `RemoteRunner`
- [x] OpenSSH config resolution for host aliases, user, port, identity files, and proxy commands
- [x] SFTP-based generated input transfer
- [x] large generated input transfer without shell argument expansion
- [x] PowerSLURM sbatch preparation
- [x] VASP command resolution preserving `SLURM_NTASKS`
- [x] stage-local `vasp_std` / `vasp_ncl` routing
- [x] server-side submission idempotency
- [x] monitoring through scheduler/accounting queries
- [x] Resume Existing Calculation by SLURM job ID
- [x] Refresh Queue Status as monitoring-only
- [x] explicit Load Results action

### Results And Visualization

- [x] final structure and total energy summary
- [x] remote-side pymatgen result parsing
- [x] JSON-safe remote parser boundary
- [x] bounded completed-result cache
- [x] workflow-specific result payloads
- [x] interactive DOS Plotly visualization
- [x] interactive Band Structure Plotly visualization
- [x] high-symmetry label rendering
- [x] spin-aware band-structure legends
- [x] default Band Structure `[-10, 10]` eV viewport
- [x] Download PNG controls for Plotly scientific visualizations

### UI And Operational Polish

- [x] Burton Materials Discovery Lab-aligned light theme
- [x] full-width stage-first Scientific Specification panel
- [x] Execution Resources above Scientific Specification
- [x] backend-defined CPU and memory selectors
- [x] fixed non-editable account policy
- [x] collapsed Structure Input after resume/monitor/results actions
- [x] production traceback hiding unless debug mode is enabled
- [x] deterministic SSH cleanup
- [x] bounded process-local SSH concurrency
- [x] structured submission provenance

## Current Accepted Deployment Policy

- [x] TAU VPN/university-network bounded access model
- [x] shared constrained `bmdguest` identity for PowerSLURM operations
- [x] no public exposure of the Uvicorn port
- [x] Resume by Job ID treated as shared-service recovery, not private ownership

This is acceptable for the current lab on-ramp deployment. It is not a public Internet security model.

## Operational Candidates (post-v1)

- [ ] document the production service wrapper used on the TAU VM when finalized
- [ ] decide whether to add a reverse proxy and HTTPS termination in front of Uvicorn
- [ ] decide whether app-level authentication is required if access moves beyond the current VPN-bound model
- [ ] decide whether CSRF protection is required for the eventual deployment topology
- [ ] add durable job history if students need a persistent dashboard
- [ ] add cancellation support when operationally needed
- [ ] add download/archive controls for selected output files
- [ ] record POTCAR hashes if the lab decides that provenance should include them

## Post-v1 Candidates

These are possible future directions, not commitments; none is part of bmd-compute v1.

- [ ] continuing a workflow across SLURM allocations or reusing completed stages
- [ ] r2SCAN policy selection and validation
- [ ] Dielectric/optics
- [ ] broader SOC workflows, only after separate review
- [ ] GW
- [ ] Elastic constants
- [ ] Equation of state
- [ ] Phonons
- [ ] NEB
- [ ] LOBSTER / bonding analysis

## Architectural Debt To Watch

- [ ] extract more route orchestration out of `main.py` into backend application services
- [ ] decide how non-linear jobflow detours/additions should map to stage directories before exposing workflows that need them
- [ ] replace process-local cache/limits with durable or cross-process mechanisms before running multiple Uvicorn workers
- [ ] add a durable application database only when state requirements justify it
