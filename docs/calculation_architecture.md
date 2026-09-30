# Calculation Architecture

BMD Compute represents calculations as ordered scientific stages. The user chooses a scientific goal; the backend turns that into a validated stage plan and the corresponding pymatgen/atomate2 inputs.

The v1 executable methodology itself (Desired Outputs, automatic treatments, stage settings, k-points, POTCARs and chaining) is declared in `methodology.md`. This page describes the objects and rules that implement it.

The current architecture is stage-first. Legacy `CalculationSpec` objects remain for compatibility, but new multi-stage behavior should be expressed as `WorkflowSpec`.

## Core Objects

### CalculationSpec

`CalculationSpec` is the compatibility representation for a single scientific purpose:

```python
CalculationSpec(
    purpose=Purpose.STATIC,
    theory=Theory.PBE,
    modifiers=frozenset(),
)
```

It should not contain INCAR templates, KPOINTS templates, SLURM resources, or atomate2 maker classes.

### StageSpec

`StageSpec` is the unit of stage-first calculation intent:

```python
StageSpec(
    stage_type=StageType.STATIC,
    theory=Theory.HSE06,
    modifiers=frozenset(),
)
```

Each stage owns its type, theory, modifiers, label, and stage-local options.

### WorkflowSpec

`WorkflowSpec` is an ordered list of stages:

```python
WorkflowSpec(
    stages=[
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.HSE06),
        StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
    ],
)
```

Recommended workflows and custom workflows both serialize to this same object. That keeps workflow construction, generated previews, remote reconstruction, and results handling on one shared path.

## Supported Stage Types

Current stage types are:

- Geometry Optimisation (`relax`)
- Static Energy (`static`)
- Density of States (`dos`)
- Band Structure (`band_structure`)

DOS and Band Structure are terminal analysis stages. Validation requires them to follow a converged Static Energy stage using a compatible theory.

## Theories

Current theory enum values are:

- `pbe`
- `hse06`
- `r2scan`

PBE and HSE06 have implemented support. r2SCAN is represented as future intent but is not currently enabled.

HSE06 support is stage-specific:

- Geometry Optimisation: supported, with `PRECFOCK = Fast`
- Static Energy: supported, with `PRECFOCK = Accurate` and hybrid-compatible smearing
- Band Structure: supported through atomate2 HSE band primitives when preceded by HSE06 Static Energy
- DOS: supported through atomate2 HSE uniform-mode primitives when preceded by HSE06 Static Energy (the Desired Output) or PBE Static Energy (Custom)

## Modifiers

Current modifiers are:

- Spin Polarised
- DFT+U
- SOC
- van der Waals correction (DFT-D3 or DFT-D3(BJ), PBE Relax/Static only)
- Gamma-only
- Ions-only

Modifiers are validated by stage and theory. They are not free-form INCAR fragments.

Important rules:

- SOC is available for PBE and HSE06 Static Energy stages and HSE06 DOS and Band Structure stages, and uses `vasp_ncl` with Custodian `auto_gamma` disabled.
- SOC and van der Waals correction may be combined on PBE Static Energy.
- In BMD-managed Desired Output workflows Spin Polarised is applied automatically to every stage when the spin composition screen triggers, and DFT-D3(BJ) to PBE Relax/Static stages when two-dimensional bonded connectivity is detected (see `methodology.md`).
- A stage that restarts from the previous fixed charge density (`ICHARG = 11`), or whose generator sizes `NBANDS` from the previous run (PBE and HSE06 DOS/Band Structure), must use the same SOC setting as that stage.
- HSE06 DOS + SOC keeps the automatic uniform Gamma mesh; with `ISYM = 0` VASP expands it over the full zone.
- In BMD-managed Desired Output workflows SOC is applied automatically to every non-relaxation stage when the heavy-element SOC policy triggers; relaxations remain non-SOC.
- In BMD-managed Desired Output workflows DFT+U follows automatic policy `bmd_compute.dft_u` v1: the pinned pymatgen/Materials Project GGA+U oxide/fluoride rule (O or F is the most electronegative element and a Co, Cr, Fe, Mn, Mo, Ni, V or W is present) with the unchanged MP L/U/J/LDAUTYPE values. It is suppressed only when every charge-balanced pymatgen oxidation-state guess puts every triggering element at d0; with no guess the MP rule applies. It is placed on PBE Relax and PBE Static stages only (Static Energy: the PBE Static; Relaxed Structure: both PBE relaxations; DOS/Band Structure: the PBE relaxation, never the HSE06 stages). It does not change the spin decision and may coexist with SOC and D3 on PBE stages. Parameters are frozen at preparation and verified at run time.
- A stage that restarts from the previous fixed charge density (`ICHARG = 11`) must use the same DFT+U setting as that stage.
- In Custom workflows DFT+U is explicit and is applied only when selected; it is never added or removed automatically.
- Spin polarization is supported where the stage registry allows it.
- Ions-only is a PBE relax-stage compatibility modifier.

## Registry And Capability Model

`backend/calculations/registry.py` answers whether a requested `CalculationSpec`, `StageSpec`, or `WorkflowSpec` is supported.

The registry owns:

- supported purpose/theory/modifier combinations
- supported stage/theory/modifier combinations
- stage ordering rules
- compatible precursor requirements
- legacy-to-stage mapping
- user-facing display names and validation messages

The registry should not generate INCAR or KPOINTS settings.

## Theory Policy

`backend/calculations/theory_policy.py` owns centralized theory/stage INCAR amendments.

HSE06 policy is applied by theory plus calculation stage. This avoids workflow-builder special cases and lets mixed workflows such as PBE relax -> HSE06 static -> HSE06 band structure stay stage-local.

Current HSE06 functional policy includes:

```text
LHFCALC = True
AEXX = 0.25
HFSCREEN = 0.2
GGA = PE
```

Stage-specific HSE06 amendments include:

```text
Relax:          ALGO = Damped, TIME = 0.4, PRECFOCK = Fast
Static:         ALGO = Damped, TIME = 0.4, PRECFOCK = Accurate, ISMEAR = 0
DOS:            ALGO = Normal, PRECFOCK = Fast, ISMEAR = -5
Band Structure: ALGO = Normal, PRECFOCK = Fast, ISMEAR = 0, SIGMA = 0.01
```

## Resource Policy

Execution resources are modeled separately from scientific theory in `backend/calculations/resources.py`.

Current allow-lists:

- CPUs: `24, 48, 72, 96, 120, 144, 168, 192`
- Memory GB: `32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512`
- Queue: `leeburton-pool`

Defaults:

```text
nodes = 1
ntasks = 24
memory = 128 GB
walltime = 72:00:00
queue = leeburton-pool
account = power-leeburton-users_v2
```

The account is fixed backend policy and is not user-editable.

Automatic NCORE is resource-derived and stage-specific. It currently applies to Relax, Static, and DOS stages. Band Structure stages omit automatic NCORE until parallel band-structure performance is separately benchmarked.

## Generated Input Previews

Generated inputs are pre-submission policy previews. They show the INCAR, KPOINTS, POSCAR, POTCAR symbols, and SLURM/script policy BMD Compute intends to use before remote preparation.

Previews are generated for every stage from the submitted structure, without any previous-stage output. At runtime, stage 2 onwards is generated from the stage before it: its relaxed structure, its final magnetic moments (spin-polarised stages without SOC), k-points and band paths regenerated for the relaxed cell, `NBANDS` derived from it, the HSE06 DOS `SIGMA` that follows its band gap, and, for PBE DOS/Band Structure, its charge density. These are expected differences, not methodology deviations; see `methodology.md` section 7.

Preview generation and remote execution should share the same stage builders for policy-sensitive inputs. New modifiers and theory amendments should include tests comparing preview and reconstructed execution paths.

## Stage Directories

Multi-stage workflows preserve every stage output in separate directories.

Examples:

```text
stage_01/
stage_02/
stage_03/
```

The legacy Double Geometry Optimisation workflow keeps its historical names:

```text
relax_01/
relax_02/
```

These names are internal execution/result details. The UI presents the scientific calculation plan rather than implementation-level job names.

## Submission And Execution

Submission state includes the serialized `WorkflowSpec`, resources, environment, cluster policy, remote paths, and provenance.

Remote execution reconstructs the workflow from `submission.json`, configures atomate2/Custodian, and runs the stages in order. The sbatch allocation controls `SLURM_NTASKS`; the VASP command resolver expands the runtime task count before Custodian receives argv.

A workflow runs in one SLURM allocation. The runner first checks that the POWER scientific stack matches the versions recorded at preparation and stops before any VASP work if it does not (`runtime_environment.json`). An unsuccessful (unconverged) stage fails the workflow and no dependent stage runs. Reaching the walltime (`TIMEOUT`) or any other failure ends the run as failed. Results are loaded only for a completed run, from its final stage; partial results are not loaded. Running again is a new submission attempt in a new run directory, and completed stages are not reused; continuing a workflow across allocations is not part of v1.

Submission idempotency is enforced by server-side state in the remote logs area. Repeated submit attempts with the same attempt id should not create duplicate SLURM jobs once a submission has reached the protected state.

## Results

Generic results include final structure and total energy information where available.

Workflow-specific scientific visualizations are plugged into a generic result-rendering path:

- Density of States: pymatgen-parsed DOS Plotly visualization
- Band Structure: pymatgen-parsed band structure Plotly visualization with high-symmetry labels, spin-aware legends, Fermi-level alignment, and default `[-10, 10]` eV viewport

Result parsing runs remotely where possible and returns JSON-safe compact payloads to the web process.

## Current And Future Boundary

Supported now:

- the four Desired Outputs: Energy only, Relaxed structure, Electronic density of states and Electronic band structure (see `methodology.md`)
- Custom PBE relax/static/relax-static/double-relax/DOS/band-structure workflows
- HSE06 relax/static/relax-static stages and workflows
- HSE06 DOS after PBE or HSE06 Static Energy, and HSE06 band structure after HSE06 Static Energy
- Spin Polarised, DFT-D3/DFT-D3(BJ), SOC and DFT+U, applied automatically in Desired Output workflows and by selection in Custom workflows

Not part of v1 (deliberately unsupported, or post-v1 candidates rather than commitments):

- SOC relaxation and PBE DOS/Band Structure + SOC
- r2SCAN
- Dielectric/optics
- GW
- arbitrary user INCAR or KPOINTS editing (rejected at validation)
- continuing a workflow across SLURM allocations, or reusing completed stages
- non-linear jobflow directory semantics beyond the current linear stage chains

## Executable Capability JSON

BMD Compute exposes its executable stage capability description through a small read-only JSON producer:

```bash
python -m backend.calculations.capabilities
```

The command serializes the existing stage introspection layer (`list_stage_definitions()` and `describe_stage()`) and does not create a second capability registry. The payload is versioned with `schema_version = 1`, includes producer provenance when Git information is available, and is intended for internal BMD ecosystem consumers such as BMD Agent.

The payload describes BMD Compute's executable calculation methodology: the stages, theories, settings and treatments this checkout implements and can execute. BMD Compute is the authority for that executable methodology. BMDex supplies curated supporting evidence, validation records, datasets and tools; it does not define BMD Compute methodology. The payload does not claim that any capability has been scientifically validated or adopted; that remains human judgment, recorded separately.

Consumers establish compatibility from the machine-readable fields `schema_version` and `source.repository` (`"bmd_compute"`). The `scope` strings are human-readable descriptions and may be reworded without a schema change; consumers must not compare them for equality.

Provenance inspection runs `git` with `--no-optional-locks` (and `GIT_OPTIONAL_LOCKS=0`), so invoking the producer never refreshes or rewrites the checkout's Git index.
