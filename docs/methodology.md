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
| Relaxed Energy | `relaxed_energy` | BMD Standard Geometry Relaxation -> PBE Static Energy |
| Relaxed structure | `relaxed_structure` | BMD Standard Geometry Relaxation |
| Electronic density of states | `electronic_dos` | BMD Standard Geometry Relaxation -> HSE06 Static Energy -> HSE06 Density of States |
| Electronic band structure | `electronic_band_structure` | BMD Standard Geometry Relaxation -> HSE06 Static Energy -> HSE06 Band Structure |
| Phonons | `phonons` | BMD Standard Geometry Relaxation -> PBE Phonon Force Calculations (`phonon_workflow.md`) |

The BMD Standard Geometry Relaxation (`bmd_compute.geometry_relaxation/1`,
section 3) is two PBE Geometry Optimisation stages (pass 1, pass 2).
| Energy only (legacy; not in the browser menu) | `energy_only` | PBE Static Energy |

Desired Outputs are resolved on the server from the structure: the automatic
treatments in section 2 are added to the stages above and the result is frozen
at preparation.

The energy-producing choices differ as follows:

- **Relaxed structure** relaxes the cell and atoms twice and reports the second
  relaxation's structure and energy. It has no separate single-point stage.
- **Relaxed Energy** runs the same two relaxations and then an accurate PBE
  Static Energy (`NSW = 0`, tetrahedron `ISMEAR = -5`) on the structure the
  second relaxation produced. Its stages are exactly the Relaxed structure
  stages followed by the Energy only stage: no INCAR value is new. The
  reported total energy is the static's; the reported structure is the
  second relaxation's (the static does not move atoms). All three stages run
  in one allocation (section 8). Automatic treatments follow section 2 per
  stage: spin on every stage, D3 and DFT+U on all three stages, and SOC on the
  final static only, so for heavy-element structures the geometry is relaxed
  without SOC and the energy is evaluated with it.
- **Energy only** (`energy_only`) is the original fixed-geometry single point
  on the submitted structure. It keeps exactly this meaning for the machine
  API, BMD Run and historical records, but is no longer offered in the
  browser's Desired Output menu; a browser request that names it (a legacy
  form post or a historical record) still builds it and shows it as
  "Energy only (legacy)". A fixed-geometry static remains available to
  students as a one-stage Custom workflow.
- **Custom** runs exactly the stages and Advanced Options the user chooses
  (below).

### Custom workflows

A Custom workflow is an ordered list of stages the user chooses. Each stage's
type, theory and Advanced Options (Spin Polarised, DFT+U, SOC, van der Waals
correction) must be a combination the stage registry supports, and chaining
rules (section 7) apply. Custom workflows are never changed automatically; the
automatic rules below are only offered as advice.

A workflow is managed or Custom by its selection, never by its stages: the
browser's Desired Output selector (carried unchanged by the Prepare, Submit
and Monitor forms) and the machine API's `desired_output`/`custom` choice are
authoritative. A Custom workflow whose stages match a Desired Output (a single
static, relax -> relax, relax -> relax -> static, the DOS, band-structure or
phonon sequences) stays Custom through Build, Update Calculation, Prepare,
Submit and execution, and receives no automatic treatments. Its executable
specification carries no client-supplied descriptive metadata: a posted
workflow `recipe`, workflow `label` or stage `label` is dropped (set to null)
before planning, so no managed Desired Output identifier can be recorded on a
Custom workflow or in its provenance.

Monitoring and Resume show the workflow recorded in BMD Compute's job record:
a managed Desired Output is identified by its recorded automatic-treatment
resolution (which must agree with the workflow's recipe), and a workflow with
neither is Custom. Posted workflow fields are not used, identity is never
inferred from the stage sequence, and a workflow the record cannot identify
is not shown. Stage options may carry only
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
substituted. Energy only, and the final static of Relaxed Energy, keep
automatic PBE + SOC Static Energy. Explicit
HSE06 + SOC requests (Custom workflows, legacy form fields) are rejected.
Records of earlier HSE06 + SOC runs remain readable and keep their own
runtime behaviour.

## 3. Stage methodology

Values marked (adopted) come from the pinned atomate2/pymatgen input sets
rather than from bmd-compute's own stage settings. They are declared v1
methodology all the same: a dependency change must reproduce them
deliberately.

Common to every stage: PBE_64 POTCARs (section 5), `EDIFF = 1e-6`,
`PREC = Accurate`, `LREAL = False`, `ADDGRID = True` (not on Geometry
Optimisation, see below), `LASPH = True` (adopted) and at most `NELM = 200`
electronic steps (adopted). On Geometry Optimisation
stages `PREC` and `LREAL` are also adopted values. Stages without spin
polarisation or SOC set `ISPIN = 1` explicitly.

### PBE Geometry Optimisation: BMD Standard Geometry Relaxation v1

Protocol `bmd_compute.geometry_relaxation/1`
(`backend/calculations/geometry_relaxation.py`). Every Desired Output that
relaxes a structure runs two Geometry Optimisation stages with the same
Hamiltonian (theory and modifiers), POTCARs and cutoff; pass 2 starts from
pass 1's structure and magnetic moments with a newly constructed plane-wave
basis (no WAVECAR or CHGCAR is carried over).

| Setting | Pass 1 | Pass 2 |
| --- | --- | --- |
| Plane-wave cutoff | species-dependent (below) | the same value |
| Electronic convergence | `EDIFF = 1e-6` | `EDIFF = 1e-7` |
| Force convergence | `EDIFFG = -0.01` eV/A | `EDIFFG = -0.001` eV/A |
| Ionic relaxation | `IBRION = 2`, `ISIF = 3` (`ISIF = 2` for the Custom ions-only option) | same |
| Ionic step limit | `NSW = 99` (adopted) | same |
| Electronic algorithm | `ALGO = Fast` | same |
| Smearing | Gaussian, `ISMEAR = 0`, `SIGMA = 0.2` eV | same |
| Precision | `PREC = Accurate`, `LREAL = False`, `LASPH = True`, no `ADDGRID` | same |
| Custodian | BMD handlers without LargeSigmaHandler | same |

Plane-wave cutoff (`bmd_compute.plane_wave_cutoff/1`):
`ENCUT = ceil(1.3 x max ENMAX / 50 eV) x 50 eV` over the POTCAR datasets the
executable input generator selects for the structure. ENMAX is read from the
versioned metadata `backend/calculations/potcar_enmax_PBE_64.json`, generated
from the POWER PBE_64 library by `backend/calculations/potcar_enmax_metadata.py`
(header values and dataset SHA-256 only, never POTCAR contents). The resolved
record (POTCAR symbols, TITEL, ENMAX, SHA-256, rule version and ENCUT) is
frozen into each Geometry Optimisation and phonon force stage at planning, so
previews, the plan digest, preparation and the POWER run use the same value;
before any stage runs, the run checks the recorded values against the POTCAR
files VASP reads (`runtime_environment.json`, `plane_wave_cutoff_potcars`).
Missing or inconsistent metadata fails planning; no cutoff is guessed.

No LargeSigmaHandler and no `ADDGRID`, so the relaxed structure is a minimum
of the same smearing and Hamiltonian as the phonon displacement forces that
may consume it.

Cell convergence: VASP's ionic convergence (the `EDIFFG` force criterion) is
reported as VASP records it. The final stage's stress tensor, hydrostatic
pressure, largest stress component, volume and lattice-parameter changes and
ionic step count are recorded as an independent diagnostic
(`geometry_relaxation` in the results), classified against indicative levels
(|P| <= 1 kbar and every component <= 2 kbar; review up to 5 kbar; final-pass
volume change above 0.5 %). In v1 they never fail a calculation.

Adaptive geometry convergence (`bmd_compute.geometry_convergence/1`,
`backend/calculations/geometry_convergence.py`; run-time jobs in
`geometry_convergence_jobs.py`). After pass 2, inside the same SLURM
allocation, a single point of the relaxed geometry at the probe cutoff
`Ep = max(700 eV, E0 + 300 eV)` (pass-2 settings, `NSW = 0`, `IBRION = -1`) is
evaluated: the relaxation converged electronically and ionically, the probe
converged electronically, maximum residual force <= 0.001 eV/A,
|P| <= 1 kbar, largest |stress component| <= 2 kbar, and (spin-polarised) total
magnetisation, and site moments when written, within 0.1 mu_B of the
relaxation's. Spin-polarised final passes and every adaptive stage set
`LORBIT = 11` (output only) so VASP writes the site moments; without them, or
with only the total moment, the evidence is invalid and never accepted. Every
probe and the refinement use the actual `KPOINTS` file of the relaxation they
follow (frozen, never regenerated from the changed cell); the record keeps its
identity, and a stage on another mesh is invalid evidence. One BMD Geometry
Relaxation per workflow. Accepted: the geometry is released. Rejected: one refinement
relaxation at `E1 = E0 + 150 eV` (pass-2 settings, from the rejected geometry)
and one final probe at `Ep`; a second rejection is *unresolved*. A magnetic
state change is unresolved without refinement; missing, malformed or
inconsistent evidence is invalid evidence; unconverged VASP or a crashed stage
is an execution failure; a stage whose planning estimate exceeds the
remaining walltime is not started (resource budget exhausted). Any stop raises
in the run, so no downstream stage starts; every VASP directory is kept. The
cutoffs, tolerances and refinement limit are frozen into the final pass (and
the phonon force stage) at planning, covered by the plan digest, and the
walltime and memory request covers the worst-case path (probe, refinement,
probe); the run may finish earlier. The record
`geometry_convergence/geometry_convergence.json` holds every cutoff, probe
result, check, decision, stage directory and remaining-walltime check, and
`geometry_convergence/accepted/` copies the accepted relaxation's outputs
(reported as the relaxation result, labelled as the refinement when it
came from the refinement). The record names each dynamic operation before it
runs; for a FAILED job, authenticated monitoring shows a bounded summary of
it (cutoffs, forces, stress, decisions and the reason; no file paths). k-points, smearing, magnetism, DFT+U,
dispersion and POTCARs are those of the relaxation. These are engineering
acceptance criteria (the geometry is stationary, to these tolerances, for the
probe-cutoff Hamiltonian), not a proof of complete basis-set convergence.
Plans recorded with the protocol but without this record build without it.

Downstream cutoff (plan-fixed): a Static Energy stage runs at
`max(620 eV, E1)`, the highest relaxation cutoff the controller may accept
(`max(620 eV, resolved relaxation cutoff)` for a relaxation without the
adaptive record); a PBE DOS or Band Structure that restarts from that static's
charge density (`ICHARG = 11`) keeps the static's cutoff. HSE06 DOS and Band
Structure (self-consistent; they take only the structure and `NBANDS` from the
static) keep their own 620 eV. Phonon displacement forces use `Ep`. The record
lists the cutoff each downstream stage uses.

Custom workflows keep their explicit stages. A Custom Geometry Optimisation
uses the species-dependent cutoff, no `ADDGRID` and no LargeSigmaHandler, with
pass-1 convergence, or pass-2 convergence when it directly feeds a phonon
force stage. The canonical pair can be requested by marking two consecutive,
identical Geometry Optimisation stages with the stage option
`{"geometry_relaxation": {"protocol": "bmd_compute.geometry_relaxation", "version": 1, "pass": 1}}`
(then `"pass": 2`); recipe `geometry_relaxation` lists it, and the browser's
Custom builder adds the pair with **Add BMD Geometry Relaxation (two passes)**
(the two passes keep identical theory and Advanced Options; removing either
removes both). **Add Stage** still adds a single, unmarked stage.

Workflows recorded before this protocol carry no cutoff record and are
rebuilt exactly as recorded (`ENCUT = 580` eV, `EDIFF = 1e-6`,
`EDIFFG = -0.01`, `ADDGRID = True`; 700 eV / `1e-7` / `-0.001` for the
relaxation feeding phonons).

The smearing is the pymatgen choice for a system whose band gap is not yet
known; BMD adopts it for all relaxations and sets it explicitly, so it never
follows previous-stage band-gap evidence.

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
HSE06 Static Energy. The Band Structure's `SIGMA = 0.01` is its default initial
value: when BMD Check's C2 assessment classifies the completed HSE06 static as
metallic, the stage starts from `SIGMA = 0.20` eV instead (`ISMEAR = 0` in every
case; uncertain or unusable evidence keeps 0.01, with the reason recorded in
`bmd_hse_band_sigma.json`; policy `bmd_compute.hse_band_structure_sigma` v1).
Custodian's LargeSigmaHandler may lower SIGMA during the run. Both are self-consistent. Atomate2 0.1.5's `HSEBSMaker`
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
- the initial `SIGMA` on HSE06 Band Structure (0.20 eV when BMD Check C2
  classifies the preceding HSE06 static as metallic, section 3);
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
automatic `NCORE = 4` policy on eligible stages for validated resource
selections, and the runtime parity check that stops a run whose parity-critical
packages or input-altering atomate2 settings differ from preparation.

Output-writing flags (for example `LCHARG`, `LWAVE`, `LVTOT`, `LAECHG`, `LELF`)
are not part of the declared methodology.

## 10. Structure searches (USPEX) — proposal, not yet in use

Status: **proposal awaiting BMD scientific review and lab sign-off (#65).**
No structure search can be started from bmd-compute yet; the design is in
`structure_search.md`. The values here are implemented in
`backend/structure_search/incar_policy.py` and `potcar.py` and pinned by
golden files in `tests/fixtures/structure_search/`.

USPEX copies the same `INCAR_1 .. INCAR_5` into every candidate's
calculation, so nothing in them can depend on a particular structure. The
ladder starts looser than BMD standards and tightens so that the last two
steps meet them:

| Step | Purpose | ISIF | IBRION / POTIM | NSW | ENCUT (eV) | EDIFF | EDIFFG | Smearing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cell shape, far from a minimum | 4 | 2 / 0.02 | 99 | 430 * | 1e-2 | 0.1 | MP (`ISMEAR = 1`), `SIGMA = 0.10` |
| 2 | cell shape | 4 | 1 / 0.3 | 99 | 430 * | 1e-3 | 0.01 | MP, `SIGMA = 0.10` |
| 3 | cell shape and volume | 3 | 1 / 0.3 | 99 | 580 * | 1e-4 | 0.001 | MP, `SIGMA = 0.05` |
| 4 | BMD relaxation standard | 3 | 1 / 0.3 | 99 | 580 * | 1e-6 * | -0.01 * | final * |
| 5 | single point: ranking energy | 2 | -1 | 0 | 620 | 1e-6 | — | final * |

\* needs a lab decision (below).

Every step also sets `PREC = Accurate` (* on steps 1–2), `LREAL = False`, `ADDGRID = True`,
`LASPH = True`, `NELM = 200`, `ISPIN = 1`, `NCORE = 4`, `KPAR = 1` *,
`LWAVE = False` and `LCHARG = False`. `NPAR` is never set alongside `NCORE`.
KPAR and NCORE are checked in code against the rank count of a relaxation
(48 by default, a parameter). The parallelisation numbers come from the lab's
VASP notes on a single 3-atom Sc₂C cell, so they are not general results.
No step sets `PSTRESS` (pressure is set
only by `ExternalPressure` in `INPUT.txt`), `MAGMOM`, `NBANDS`, DFT+U, SOC or
a dispersion correction.

Reasons:

- The step pattern (ISIF 4, 4, 3, 3, then a single point), the
  loose-to-tight EDIFF/EDIFFG of steps 1–3, IBRION 2 with POTIM 0.02 far from
  a minimum and IBRION 1 with POTIM 0.3 after it, and Methfessel–Paxton
  smearing for the early steps follow the USPEX 25.0 manual (section 3.1).
- Steps that change the volume use BMD's fixed relaxation cutoff of 580 eV
  to avoid Pulay-stress volume errors; only fixed-volume steps use a lower
  one. BMD's own relaxations now use the species-dependent cutoff of
  section 3 (`bmd_compute.plane_wave_cutoff/1`, without `ADDGRID`), and
  580 eV is the value for records made before that protocol. Whether the
  search ladder follows the new rule is a lab decision (below).
- Every step's ENCUT must be at least the largest ENMAX of the search's
  POTCARs. A search that fails this is refused (code
  `encut_below_potcar_enmax`), never adjusted. For example, a search
  containing Li (`Li_sv`, ENMAX 499.034 eV) is refused, because steps 1–2
  use 430 eV.
- Step 4 uses the BMD relaxation convergence (EDIFF 1e-6, forces below
  0.01 eV/Å), and step 5 the BMD Static Energy cutoff and EDIFF, so steps
  4–5 repeat the relax-then-static pattern of Relaxed Energy. USPEX ranks
  candidates by the energy of the last step.
- Steps 4 and 5 share one smearing so the geometry and the ranking energy
  use the same occupations.

Values that need a lab decision (named `*_NEEDS_LAB_DECISION` in the code):

| Value | Placeholder | Notes |
| --- | --- | --- |
| ENCUT of steps 1–2 | 430 eV | 580 eV / 1.35, the middle of the manual's 30–40 % increase |
| ENCUT below a POTCAR's ENMAX | refuse the search | alternative: raise the early cutoff for that search. The composition, and so the POTCARs, are fixed per search, so every candidate would still get the same INCAR |
| ENCUT of steps 3–4 | 580 eV | alternative: BMD's species-dependent relaxation cutoff of section 3, computed per search |
| `PREC` of steps 1–2 | `Accurate` | the manual starts with `Low`, then `Normal`; `Accurate` also sets denser FFT grids |
| Convergence of step 4 | `EDIFF = 1e-6`, `EDIFFG = -0.01` | relaxes every candidate to BMD standards, probably the largest cost of a search; alternative: the manual's final tolerances (`1e-4` / `1e-3`) or a value in between |
| Number of steps | 5 | the manual suggests 4–5; the spec's `KresolStart` must have one value per step (checked before `INPUT.txt` is written, #67) |
| Smearing of steps 4–5 | Gaussian, `SIGMA = 0.05` | alternative: Methfessel–Paxton with a small SIGMA. The tetrahedron method of PBE Static Energy is not used because candidates with coarse meshes can have too few k-points for it |
| KPAR | 1 | must divide the rank count of a relaxation (48 by default), leave a quotient NCORE divides, and not exceed NKPTS for any candidate. KPAR = 4 meets the first two rules with 48 ranks and was worth up to 9x on one k-point-dense test cell, but exceeds NKPTS for candidates with fewer than 4 k-points; Phase 0 (#59) decides |

K-points: USPEX builds each mesh from `KresolStart` (one value per step,
units 2π/Å). BMD's k-point policy (reciprocal density 64, section 4)
corresponds to a spacing of 64^(−1/3) = 0.25 Å⁻¹, that is 0.0398 × 2π/Å;
pymatgen's rounded meshes give 0.040–0.042. Steps 4 and 5 must therefore use
`KresolStart` ≤ 0.04; steps 1–3 may be coarser. How USPEX rounds a
`KresolStart` value to a mesh has not been compared with pymatgen yet
(Phase 0).

POTCARs: the same PBE_64 files and symbol choices as every other BMD
calculation (section 5), read from the pinned atomate2 relaxation generator
and copied from `<POTCAR repository>/POT_PAW_PBE_64/<symbol>/POTCAR` to
`Specific/POTCAR_<El>`. Each file's header must name the expected symbol
before it is copied; its ENMAX (read from the PSCTR parameter block, as
pymatgen reads it) and SHA-256 are recorded.
