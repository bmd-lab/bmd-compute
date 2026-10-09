# bmd-compute v1 Executable Methodology

This page declares the executable calculation methodology bmd-compute v1
implements and runs. bmd-compute is the authority for this executable
methodology. Scientific validation and adoption of any choice remain human
judgment and are not claimed here.

Values below are what the pinned scientific stack
(`constraints/scientific-runtime.txt`) generates through bmd-compute's stage
builders today. Some are set by bmd-compute itself; others come from the pinned
atomate2/pymatgen input sets and are adopted here deliberately, so that a
future dependency change has to reproduce them on purpose rather than by
accident. `tests/test_declared_methodology.py` checks the scientifically
meaningful values against generated inputs.

Related pages: `calculation_architecture.md` (objects and validation rules),
`atomate2_capabilities.md` (upstream mapping), `input_reference_contract.md`,
`provenance.md` and `run_records.md`.

## 1. Workflows

### Desired Outputs

| UI choice | Key | Stages |
| --- | --- | --- |
| Energy only (Static Energy) | `energy_only` | PBE Static Energy |
| Relaxed structure | `relaxed_structure` | PBE Geometry Optimisation -> PBE Geometry Optimisation |
| Electronic density of states | `electronic_dos` | PBE Geometry Optimisation -> HSE06 Static Energy -> HSE06 Density of States |
| Electronic band structure | `electronic_band_structure` | PBE Geometry Optimisation -> HSE06 Static Energy -> HSE06 Band Structure |

Desired Outputs are resolved on the server from the structure: the automatic
treatments in section 2 are added to the stages above and the result is frozen
at preparation.

### Custom workflows

A Custom workflow is an ordered list of stages the user chooses. Each stage's
type, theory and Advanced Options (Spin Polarised, DFT+U, SOC, van der Waals
correction) must be a combination the stage registry supports, and chaining
rules (section 7) apply. Custom workflows are never changed automatically; the
automatic rules below are only offered as advice. Stage options may carry only
supported treatment settings; free-form INCAR or KPOINTS values are rejected.

### Structure constraints

No workflow, Desired Output or Custom, runs with user-supplied atomic
constraints. A structure carrying VASP selective-dynamics flags (a POSCAR
`Selective dynamics` block with at least one `F`, or a `selective_dynamics` site
property) is rejected with diagnostic code `selective_dynamics_unsupported`
before any VASP input is generated; the flags are never stripped or passed on.
An all-`T` block constrains nothing and is discarded by pymatgen when the POSCAR
is parsed, so it is accepted as an ordinary structure.

## 2. Automatic treatments (Desired Outputs only)

| Treatment | Trigger | Stages that receive it | Settings |
| --- | --- | --- | --- |
| Spin polarisation | the structure contains Ti, V, Cr, Mn, Fe, Co, Ni, Mo, Tc, Ru, Rh, Re, Os, Ir, a lanthanide Ce-Yb, or U, Np, Pu, Am, Cm, Bk, Cf | every stage | `ISPIN = 2` with the starting moments of section 6 |
| van der Waals correction | pymatgen CrystalNN bonding plus Larsen dimensionality finds two-dimensional bonded connectivity | PBE Geometry Optimisation and PBE Static Energy | DFT-D3(BJ), `IVDW = 12` |
| Spin-orbit coupling | the structure contains a 4d (Y-Cd) or 5d (Hf-Hg) transition metal, a lanthanide (La-Lu), an actinide (Ac-Lr), or Tl, Pb, Bi, Po | every PBE stage except Geometry Optimisation; never HSE06 (omitted with a red warning, see below) | non-collinear, see below |
| DFT+U | policy `bmd_compute.dft_u` v1: O or F is the most electronegative element and Co, Cr, Fe, Mn, Mo, Ni, V or W is present, unless every charge-balanced pymatgen oxidation-state guess puts every such element at d0 | PBE Geometry Optimisation and PBE Static Energy; never HSE06 | pinned Materials Project/pymatgen GGA+U values unchanged (Dudarev, `LDAUTYPE = 2`, U on d states, J = 0), frozen at preparation and verified at run time |

Treatments are decided independently: +U never switches on spin, and SOC, D3
and +U may share a PBE stage. The triggers are composition or connectivity
screens, not proof that a treatment is required.

If the dimensionality observation returns `analysis_failed`, a managed Desired
Output fails closed before generated-input preview, preparation or submission.
bmd-compute cannot safely decide whether automatic DFT-D3(BJ) is required in
that state. The structured calculation error retains the failed observation and
reason. Structure analysis remains observational and non-blocking, and Custom
workflows remain user-managed and are not rejected by this automatic-treatment
policy.

SOC stages run `vasp_ncl` with `LSORBIT = True`, `LNONCOLLINEAR = True`,
`ISYM = 0`, `SAXIS = 0 0 1`, `GGA_COMPAT = False` and no `ISPIN`; Custodian's
`auto_gamma` is disabled so the executable cannot switch to `vasp_gam`.

HSE06 + SOC is not supported for new calculations (admission policy
`bmd_compute.new_calculation_admission` v1). This is a product-support
limitation: the combination can be exceptionally computationally expensive
and may need system-specific convergence and resource settings. It is not a
statement that HSE06 + SOC is scientifically invalid. When the heavy-element
SOC trigger fires for Electronic density of states or Electronic band
structure, BMD keeps the HSE06 workflow unchanged, omits SOC from its HSE06
stages, records the omission in `automatic_treatments.omitted_treatments`, and
shows a red warning before submission: SOC may significantly affect the
predicted electronic structure of heavy-element materials, including band
ordering and band gaps. No PBE + SOC DOS or Band Structure stage is
substituted. Energy only keeps automatic PBE + SOC Static Energy. Explicit
HSE06 + SOC requests (Custom workflows, legacy form fields) are rejected.
Records of earlier HSE06 + SOC runs remain readable and keep their own
runtime behaviour.

## 3. Stage methodology

Values marked (adopted) come from the pinned atomate2/pymatgen input sets
rather than from bmd-compute's own stage settings. They are declared v1
methodology all the same: a dependency change must reproduce them
deliberately.

Common to every stage: PBE_64 POTCARs (section 5), `EDIFF = 1e-6`,
`PREC = Accurate`, `LREAL = False`, `ADDGRID = True`, `LASPH = True` (adopted)
and at most `NELM = 200` electronic steps (adopted). On Geometry Optimisation
stages `PREC` and `LREAL` are also adopted values. Stages without spin
polarisation or SOC set `ISPIN = 1` explicitly.

### PBE Geometry Optimisation

| Setting | Value |
| --- | --- |
| Plane-wave cutoff | `ENCUT = 580` eV |
| Ionic relaxation | conjugate gradient (`IBRION = 2`, adopted), ions and cell (`ISIF = 3`, adopted; BMD sets `ISIF = 2` for the Custom ions-only option) |
| Force convergence | `EDIFFG = -0.01` eV/A |
| Ionic step limit | `NSW = 99` (adopted) |
| Electronic algorithm | `ALGO = Fast` |
| Smearing | Gaussian, `ISMEAR = 0`, `SIGMA = 0.2` eV (adopted) |

The smearing is the pymatgen choice for a system whose band gap is not yet
known; BMD adopts it for all relaxations.

### PBE Static Energy

| Setting | Value |
| --- | --- |
| Plane-wave cutoff | `ENCUT = 620` eV |
| Electronic algorithm | `ALGO = Normal` |
| Smearing | tetrahedron with Bloechl corrections, `ISMEAR = -5` (`SIGMA = 0.05`) |
| DOS sampling | `NEDOS = 4001`, `LORBIT = 11` |

### PBE Density of States and Band Structure (Custom only)

Non-self-consistent (`ICHARG = 11`) from the preceding PBE Static Energy
charge density, `ENCUT = 620` eV. DOS uses `ISMEAR = -5`; Band Structure uses
`ISMEAR = 0`, `SIGMA = 0.05` and `ISYM = 0` on the high-symmetry path.

### HSE06

All HSE06 stages use `LHFCALC = True`, `AEXX = 0.25`, `HFSCREEN = 0.2` and
`GGA = PE`.

| Stage | Settings |
| --- | --- |
| Geometry Optimisation (Custom) | relaxation settings above with `ALGO = Damped`, `TIME = 0.4`, `PRECFOCK = Fast` |
| Static Energy | `ENCUT = 620` eV, `ALGO = Damped`, `TIME = 0.4`, `PRECFOCK = Accurate`, `ISMEAR = 0`, `SIGMA = 0.05` |
| Density of States | self-consistent on a uniform mesh, `ENCUT = 620` eV, `ALGO = Normal`, `PRECFOCK = Fast`, `ISMEAR = -5`, `NEDOS = 4001`, `NELMIN = 5` (adopted) |
| Band Structure | self-consistent on a uniform mesh plus a zero-weight path, `ENCUT = 620` eV, `ALGO = Normal`, `PRECFOCK = Fast`, `ISMEAR = 0`, `SIGMA = 0.01`, `NELMIN = 5` (adopted) |

HSE06 DOS may follow PBE or HSE06 Static Energy; HSE06 Band Structure requires
HSE06 Static Energy. Both are self-consistent. Atomate2 0.1.5's `HSEBSMaker`
may physically copy `CHGCAR` when given a previous-stage directory, but BMD's
generated HSE06 INCAR requests neither a fixed-charge-density nor a WAVECAR
restart (`ICHARG` and `ISTART` are absent). The preceding run still supplies the
structure, starting moments and `NBANDS` context described in section 7.

## 4. K-points

All meshes are Gamma-centred automatic meshes defined by a density per
reciprocal-lattice volume (pymatgen `automatic_density_by_vol`):

| Stage | Sampling |
| --- | --- |
| Geometry Optimisation, Static Energy (PBE and HSE06) | reciprocal density 64 (adopted) |
| PBE Density of States | reciprocal density 100 (adopted) |
| PBE Band Structure | high-symmetry path, line density 40 |
| HSE06 Density of States | reciprocal density 100 |
| HSE06 Band Structure | uniform reciprocal density 64 plus a zero-weight high-symmetry path at line density 40 |

For historical HSE06 + SOC records (no longer accepted for new calculations),
HSE06 Band Structure lists every point of its uniform mesh with equal weight
(because `ISYM = 0`), and HSE06 DOS keeps its automatic mesh, which VASP
expands over the full zone.

## 5. POTCARs

The PBE_64 POTCAR set with the symbol choices of the pinned atomate2
generator configuration (`_BASE_VASP_SET`), not pymatgen's `MPRelaxSet`
choices. Examples: Ni -> `Ni`, Fe -> `Fe`, Ti -> `Ti_sv`, W -> `W_sv`,
Mo -> `Mo_sv`, V -> `V_sv`, Bi -> `Bi_d`, O -> `O`. Every submission records
the species -> POTCAR mapping each stage's generator resolves (`potcar.stages`
in `submission.json`), so the table itself is not reproduced here.

## 6. Starting magnetic moments

Spin-polarised stages start from the pymatgen/Materials Project default moments
of the pinned input set (for example 5 uB on Fe, Mn and Ni, 0.6 uB for
elements without a listed value). SOC stages use vector moments along `SAXIS`:
those same moments when the structure contains an element in the spin screen
of section 2 or the stage is Spin Polarised, otherwise zero.

## 7. Stage chaining

- Each stage after the first starts from the previous stage's final
  structure.
- PBE DOS and Band Structure read the previous stage's charge density
  (`ICHARG = 11`) and must match its SOC and DFT+U settings.
- PBE and HSE06 DOS and Band Structure take `NBANDS` from the previous stage
  (1.2 x its `NBANDS`) and must match its SOC setting.
- DOS and Band Structure must directly follow a Static Energy stage:

  | Analysis stage | after PBE Static Energy | after HSE06 Static Energy |
  | --- | --- | --- |
  | PBE DOS / PBE Band Structure | yes | no |
  | HSE06 DOS | yes | yes |
  | HSE06 Band Structure | no | yes |

  PBE DOS and Band Structure need a PBE Static Energy because they read its
  charge density; that is the only electronic hand-off between stages. HSE06
  DOS and Band Structure are self-consistent on their own meshes and take only
  the structure, `NBANDS` and starting moments from the Static Energy stage.
  The Electronic density of states Desired Output uses HSE06 Static Energy;
  PBE Static Energy -> HSE06 DOS is a supported Custom option.
- No WAVECAR is carried between stages.

### What differs from the pre-execution reference

Previews and input references are generated before execution, for every stage
from the submitted structure and without any previous-stage output. At run
time, stage 2 onwards is generated with the outputs of the stage before it, so
these fields can legitimately differ from the reference:

- the structure (the previous stage's relaxed structure);
- starting magnetic moments of spin-polarised stages without SOC (the
  previous stage's final moments); SOC stages keep the section 6 moments;
- k-points and band-structure paths, regenerated for the relaxed cell;
- `NBANDS`, derived from the previous stage;
- `SIGMA` on HSE06 DOS, which follows the previous stage's band gap and has no
  effect with `ISMEAR = -5`;
- PBE DOS/Band Structure starting from the previous charge density.

Such differences are expected consequences of this methodology, not
methodology deviations. Custodian error corrections at run time are recorded
separately by Custodian.

## 8. Success, failure and the execution model

- A stage is successful only if atomate2/emmet judge it converged
  (electronically, and ionically for relaxations, within `NELM` and `NSW`).
  An unsuccessful stage fails the workflow and no dependent stage runs; BMD
  sets this explicitly on every stage rather than inheriting atomate2's
  configurable default.
- A workflow runs in one SLURM allocation. Reaching the walltime (`TIMEOUT`)
  or any failure ends the run as failed.
- Results are loaded only for a completed run, from its final stage. Partial
  results of a failed run are not loaded, although completed stage directories
  remain on disk.
- Running again is a new submission attempt in a new run directory; completed
  stages are not reused. Continuing a workflow across allocations is not part
  of v1.

## 9. Handled by the frozen stack and records

These affect generated inputs but are fixed by the pinned stack and recorded
rather than declared individually: pymatgen's band-gap-dependent smearing
mechanics, symmetry precision, `LMAXMIX` (set by pymatgen's rule and verified
for +U stages), the Custodian handler set (recorded per stage), the fixed
automatic `NCORE = 8` policy on eligible stages for validated resource
selections, and the runtime parity check that stops a run whose parity-critical
packages or input-altering atomate2 settings differ from preparation.

Output-writing flags (for example `LCHARG`, `LWAVE`, `LVTOT`, `LAECHG`, `LELF`)
are not part of the declared methodology.
