# Calculation Architecture

BMD Compute represents calculations as ordered scientific stages. The user chooses a scientific goal; the backend turns that into a validated stage plan and the corresponding pymatgen/atomate2 inputs.

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
- DOS: not supported

## Modifiers

Current modifiers are:

- Spin Polarised
- explicit DFT+U
- SOC
- Gamma-only
- Ions-only

Modifiers are validated by stage and theory. They are not free-form INCAR fragments.

Important rules:

- SOC is available for PBE and HSE06 Static Energy stages and HSE06 DOS and Band Structure stages, and uses `vasp_ncl` with Custodian `auto_gamma` disabled.
- SOC and van der Waals correction may be combined on PBE Static Energy.
- A stage that restarts from the previous fixed charge density (`ICHARG = 11`) must use the same SOC setting as that stage.
- In BMD-managed Desired Output workflows SOC is applied automatically to every non-relaxation stage when the heavy-element SOC policy triggers; relaxations remain non-SOC.
- DFT+U is explicit and is applied only when selected and when reviewed U values are available for the structure.
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
Relax:          PRECFOCK = Fast
Static:         PRECFOCK = Accurate, ISMEAR = 0
Band Structure: atomate2 HSE band path with HSE-compatible stage settings
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

For downstream stages after a relaxation, the preview cannot know the future relaxed structure. At runtime, stage chaining uses the previous stage output as the input structure.

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

Submission idempotency is enforced by server-side state in the remote logs area. Repeated submit attempts with the same attempt id should not create duplicate SLURM jobs once a submission has reached the protected state.

## Results

Generic results include final structure and total energy information where available.

Workflow-specific scientific visualizations are plugged into a generic result-rendering path:

- Density of States: pymatgen-parsed DOS Plotly visualization
- Band Structure: pymatgen-parsed band structure Plotly visualization with high-symmetry labels, spin-aware legends, Fermi-level alignment, and default `[-10, 10]` eV viewport

Result parsing runs remotely where possible and returns JSON-safe compact payloads to the web process.

## Current And Future Boundary

Supported now:

- PBE relax/static/relax-static/double-relax/DOS/band-structure workflows
- HSE06 relax/static/relax-static stages and workflows
- HSE06 band structure with an HSE06 static electronic precursor
- SOC on PBE/HSE06 Static and HSE06 DOS/Band Structure stages, applied automatically in Desired Output workflows
- explicit DFT+U when available from the input set

Future or deliberately unsupported:

- SOC relaxation and PBE DOS/Band Structure + SOC
- r2SCAN
- Dielectric/optics
- GW
- arbitrary user INCAR editing
- non-linear jobflow directory semantics beyond the current linear stage chains
## Executable Capability JSON

BMD Compute exposes its executable stage capability description through a small read-only JSON producer:

```bash
python -m backend.calculations.capabilities
```

The command serializes the existing stage introspection layer (`list_stage_definitions()` and `describe_stage()`) and does not create a second capability registry. The payload is versioned with `schema_version = 1`, includes producer provenance when Git information is available, and is intended for internal BMD ecosystem consumers such as BMD Agent.

The contract describes what BMD Compute currently implements and can execute. It is not a scientific-methodology authority and does not claim that a capability is validated, adopted, or a BMD standard. Scientific validation status, evidence, and adopted methodology belong outside BMD Compute, currently intended for BMDex.
