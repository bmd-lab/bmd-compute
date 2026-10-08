# atomate2 / pymatgen Capability Status

The declared v1 methodology values live in `methodology.md`; this page maps bmd-compute stages to the upstream atomate2/pymatgen primitives they use.

This document records how the currently installed atomate2/pymatgen VASP surface maps to bmd-compute. It is not a promise that every upstream maker is exposed in the UI.

bmd-compute exposes only reviewed scientific workflows. Upstream availability alone is not sufficient for student-facing support.

## Status Categories

| Status | Meaning |
| --- | --- |
| Implemented and validated | Implemented in bmd-compute and validated by tests plus at least one reviewed/live calculation path where noted. |
| Implemented, tests only | Implemented and covered locally, but not independently PowerSLURM-validated as a separate scientific claim. |
| Deliberately unsupported | Upstream support may exist, but bmd-compute blocks it because the scientific or operational policy is not reviewed. |
| Future candidate | Upstream primitives exist and are plausible future work. |
| Unavailable | No clear installed native atomate2/pymatgen path has been identified. |

## Implemented Capabilities

### PBE Geometry Optimisation

Status: implemented and validated through the core BMD workflow path.

Primary primitives:

- `atomate2.vasp.sets.core.RelaxSetGenerator`
- `atomate2.vasp.jobs.core.RelaxMaker`

BMD policy keeps the Burton Lab relax settings, including the 580 eV ENCUT policy and relaxation convergence settings.

### PBE Static Energy

Status: implemented and validated through the core BMD workflow path.

Primary primitives:

- `atomate2.vasp.sets.core.StaticSetGenerator`
- `atomate2.vasp.jobs.core.StaticMaker`

BMD policy applies the final/static settings, including the 620 eV ENCUT policy, charge-density output, ELF for ordinary collinear PBE static calculations, and automatic `NCORE = 8` where the stage is eligible and resources have passed validation.

### Double Geometry Optimisation

Status: implemented.

Primary primitives:

- two ordered PBE relax stages
- preserved historical stage directories `relax_01/` and `relax_02/`

### PBE Relax -> Static

Status: implemented.

Primary primitives:

- ordered relax and static stages
- runtime static stage consumes the relaxed structure from the preceding relax stage

### PBE Density of States

Status: implemented with workflow-specific results.

Primary primitives:

- preceding Static Energy stage
- `NonSCFSetGenerator` / `NonSCFMaker` style non-SCF DOS path
- pymatgen parsing from `vasprun.xml`
- Plotly DOS visualization

The DOS stage inherits grid- and basis-defining settings from the preceding static stage where required for CHGCAR compatibility.

### PBE Band Structure

Status: implemented with workflow-specific results.

Primary primitives:

- preceding Static Energy stage
- line-mode band-structure stage using atomate2/pymatgen high-symmetry path generation
- pymatgen parsing from `vasprun.xml` plus KPOINTS
- Plotly band visualization

The result renderer includes high-symmetry label conversion, Fermi-level alignment, spin-aware legend behavior, a default `[-10, 10]` eV viewport, and PNG export.

### HSE06 Geometry Optimisation

Status: implemented.

Primary primitives:

- HSE relax stage through atomate2/pymatgen HSE relax support
- centralized HSE06 theory/stage policy

BMD policy:

```text
LHFCALC = True
AEXX = 0.25
HFSCREEN = 0.2
GGA = PE
PRECFOCK = Fast
```

### HSE06 Static Energy

Status: implemented and specifically benchmarked for resource policy.

Primary primitives:

- HSE static stage through atomate2/pymatgen HSE static support
- centralized HSE06 theory/stage policy

BMD policy:

```text
LHFCALC = True
AEXX = 0.25
HFSCREEN = 0.2
GGA = PE
PRECFOCK = Accurate
ISMEAR = 0
```

Automatic NCORE remains independent of theory. Eligible stages receive the current fixed `NCORE = 8` policy after resource validation; a 24-rank HSE06 static benchmark showed strong benefit from that value.

### HSE06 Relax -> Static

Status: implemented as an ordered stage-first workflow.

The static stage consumes the relaxed structure from the HSE06 relax stage. Theory policy remains stage-specific, so relax uses `PRECFOCK = Fast` and static uses `PRECFOCK = Accurate`.

### HSE06 Band Structure

Status: implemented and reviewed through the stage-first path.

Supported sequence:

```text
Geometry Optimisation - PBE
Static Energy         - HSE06
Band Structure        - HSE06
```

The PBE relax supplies geometry. bmd-compute requires HSE06 Static Energy before HSE06 Band Structure. The band stage takes its structure, `NBANDS` and starting moments from that stage and recomputes the HSE06 ground state self-consistently. Atomate2 0.1.5 may physically copy `CHGCAR` from the previous directory, but the generated INCAR requests neither a fixed-charge-density nor a WAVECAR restart (`ICHARG` and `ISTART` are absent).

Primary primitives:

- `HSEBSSetGenerator(mode="line")`
- `HSEBSMaker`

The generated KPOINTS file may contain a combined weighted uniform mesh plus zero-weight high-symmetry line path. Remote preparation must transfer this as file data, not as shell command text.

### HSE06 Density of States

Status: implemented.

Supported sequences:

```text
Geometry Optimisation - PBE
Static Energy         - HSE06
Density of States     - HSE06
```

This is the Electronic density of states Desired Output. HSE06 DOS may also follow a PBE Static Energy stage, a supported Custom option:

```text
Static Energy         - PBE
Density of States     - HSE06
```

Primary primitives:

- `HSEBSSetGenerator(mode="uniform")` with `reciprocal_density = 100`
- `HSEBSMaker`

The DOS stage is self-consistent on its uniform mesh (`ISMEAR = -5`, `NEDOS = 4001`); like HSE06 Band Structure it takes the structure, `NBANDS` and starting moments from the preceding Static Energy stage. Atomate2 0.1.5 may physically copy `CHGCAR`, but the generated INCAR requests neither a fixed-charge-density nor a WAVECAR restart (`ICHARG` and `ISTART` are absent).

### Spin Polarised Calculations

Status: implemented for supported PBE/HSE06 stages where the registry allows the modifier. In Desired Output workflows it is applied automatically to every stage when the structure contains an element in the spin composition screen (see `methodology.md`).

bmd-compute preserves pymatgen/atomate2 magnetic initialization where appropriate. SOC stages convert initial moments into vector `MAGMOM`.

### Van der Waals Correction (DFT-D3)

Status: implemented for PBE Geometry Optimisation and PBE Static Energy through the atomate2/pymatgen `vdw` generator keyword: DFT-D3 (`IVDW = 11`) or DFT-D3(BJ) (`IVDW = 12`). In Desired Output workflows DFT-D3(BJ) is applied automatically to those stages when two-dimensional bonded connectivity is detected; Custom workflows use it only when selected.

### DFT+U

Status: implemented for supported PBE stages when the selected input set provides active U values.

bmd-compute does not silently inherit DFT+U into ordinary PBE. If DFT+U is requested and no active U values are available, validation fails clearly.

In BMD-managed Desired Output workflows, automatic DFT+U (policy `bmd_compute.dft_u` v1) adopts the pinned pymatgen `MPRelaxSet` GGA+U oxide/fluoride trigger and L/U/J/LDAUTYPE values unchanged (atomate2's generator table is the same table), adds a compound-level d0 gate over pymatgen oxidation-state guesses, and applies +U to PBE Relax/Static stages only. The values are passed to the atomate2 generator explicitly and the generated INCAR is checked against them, so upstream table changes cannot silently alter a prepared run. `LMAXMIX` is left to pymatgen's rule and recorded and verified rather than set.

### Spin-Orbit Coupling

Status: implemented for PBE and HSE06 Static Energy stages and HSE06 DOS and Band Structure stages. In BMD-managed Desired Output workflows SOC is applied automatically to every non-relaxation stage when the heavy-element SOC policy triggers; Custom workflows remain manual.

Validated examples include:

- PBE Static -> PBE Static + SOC on Si
- PBE + DFT+U Static -> PBE + DFT+U + SOC Static on Fe2O3

SOC policy includes:

```text
LSORBIT = True
LNONCOLLINEAR = True
GGA_COMPAT = False
ISYM = 0
SAXIS = 0 0 1
ISPIN omitted
LELF omitted
vector MAGMOM (zero unless the structure is in the spin screen or the stage is Spin Polarised)
vasp_ncl executable, Custodian auto_gamma disabled
```

HSE06 Band Structure + SOC keeps `HSEBSSetGenerator`'s zero-weight high-symmetry path and `reciprocal_density = 64`, but replaces its symmetry-reduced weighted SCF points with every point of the same mesh (equal weights) because `ISYM = 0` makes VASP use the listed points as the complete sampling. The expansion is checked against the generator's own reduced mesh and refuses to proceed on any mismatch. HSE06 DOS + SOC keeps the automatic uniform mesh.

## Deliberately Unsupported Or Unvalidated

### SOC Relaxation, PBE SOC DOS, PBE SOC Band Structure

Status: deliberately unsupported.

These require separate scientific review before being exposed. PBE DOS and Band Structure restart from a fixed charge density, and a SOC precursor cannot feed a non-SOC fixed-density stage.

### r2SCAN

Status: future candidate.

Potential upstream primitives include MP2024/r2SCAN-oriented relax/static makers and pymatgen input sets, but bmd-compute has not selected or validated a lab policy.

### Dielectric And Optics

Status: future candidate.

Potential upstream primitives include `DielectricMaker`, `PolarizationMaker`, and `OpticsMaker`. These are not exposed.

### GW, Elastic, EOS, NEB, Phonon, MD, LOBSTER, AMSET

Status: future candidates or unavailable depending on optional dependencies.

These remain outside the introductory workflow surface and should not be added without separate scientific and operational review.

## Optional Dependency Notes

Some atomate2 modules may require optional packages such as phonopy, seekpath, pymatgen-analysis-diffusion, or defect-analysis packages. bmd-compute should not expose workflows that require missing optional dependencies until the production environment is deliberately updated and tested.

## Implementation Guidance

Prefer native atomate2/pymatgen primitives over local VASP templates.

For new workflows:

1. identify the native maker/input-set path
2. validate precursor and restart requirements
3. add centralized theory/stage/resource policy only where needed
4. add generated-preview and remote-reconstruction tests
5. add result rendering only through the workflow-specific visualization framework
6. document whether the capability is implemented, cluster-validated, or deliberately blocked
