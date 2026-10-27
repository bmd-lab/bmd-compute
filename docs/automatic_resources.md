# Automatic execution resources (R1)

BMD Compute chooses CPUs, memory and walltime for every resource the user
leaves on **Automatic**. This is the default for new calculations in the
browser, and an omitted resource in a machine-API request means Automatic.
Explicit values are kept exactly, subject to the existing validation.

The estimator lives in `backend/calculations/automatic_resources.py`. Policy
`bmd_compute.automatic_resources`, version 2. The browser and the machine API
both reach it through `main.resolve_resources_for_workflow`, which is called
from `build_submission_state_from_structure`, so there is one authority. BMD Run
does not estimate resources.

The objective is a conservative, workload-dependent request that a student
without HPC experience can rely on. It is not an optimal prediction and it
carries no statistical guarantee.

## CPUs and memory

Automatic CPUs are 24, a conservative initial selection rather than a
performance prediction. Automatic NCORE is unchanged (`NCORE = 4` on eligible
stages).

Automatic memory (version 2) is a conservative bracket, not a memory model. Each
stage is assigned one from the walltime descriptors below, and the workflow
requests the largest:

| Bracket | Stage |
|---|---|
| 128 GB | HSE06, meta-GGA or other non-PBE theory; descriptors unavailable; outside the calibrated range (more than 64 atoms, 315 bands or 343 k-points, or D above 5.5e10) |
| 64 GB | inside that range with SOC, a phonon displacement stage, or 33-64 atoms |
| 32 GB | other PBE stages, including spin, DFT+U and D3, up to 32 atoms |

The brackets assume 24 CPUs. Each doubling of the CPUs above 24 raises the
bracket one step, never above 128 GB, so no automatic request exceeds version
1's fixed 128 GB. The record holds the bracket's basis and the deciding stage.

The margins are wide. The 2026-10-16 telemetry recorded SLURM step MaxRSS,
sampled every 30 s, which is a lower bound on the job's peak:

- every measured PBE stage stayed below 5 GB;
- the 64-atom Si phonon displacements stayed below 7 GB;
- only hybrid stages went above 8 GB, reaching 29 GB for HSE06+SOC Bi2Se3 on 24 CPUs.

A calculation that exceeds its bracket ends as OUT_OF_MEMORY. Rebuild it with a manual memory value.

## Walltime

### Descriptors

For every stage, the estimator reads the stage inputs that Compute generates
for preview, digest and execution: INCAR, KPOINTS, POSCAR and POTCAR symbols.

| Quantity | Plan-time source |
|---|---|
| electrons | `NELECT`, or POTCAR valence (ZVAL, pymatgen POTCAR summary statistics) per species |
| bands | `NBANDS`, or VASP's default rule (spin and SOC aware, using the INCAR MAGMOM), x1.2 for DOS and band-structure stages, rounded up to the band groups of the CPUs and NCORE |
| k-points | irreducible k-points of the automatic mesh from spglib (symprec 1e-5). The full mesh is used with SOC, with ISYM <= 0, or with moments that differ within a species. Explicit and line-mode lists are counted. |
| FFT grid | per axis, `4 G_cut abs(a) / 2 pi` from ENCUT and the lattice lengths, rounded up to an even 2-3-5-7-smooth size |
| other | spin, SOC, hybrid functional, DFT+U, dispersion, stage type, relaxation degrees of freedom and NSW, phonon displacement count |

The cost descriptor is:

```text
D = k-points x bands x NG x log2(NG) x (2 for ISPIN = 2 or SOC)
```

Validation against the 2026-10-16 telemetry:

- Electron counts matched the OUTCAR for 13 of 13 compositions.
- The band rule reproduced 78 of 81 stages.
- The descriptors reproduce the recorded grids:
  - Si2: 20 k-points, 32^3 FFT grid;
  - Si64 phonon supercell: 6 k-points, 168 bands, 96^3 FFT grid.

### Calibrated PBE model

The model applies to PBE static, relaxation and phonon-displacement stages inside the calibrated domain:

- D <= 5.5e10;
- at most 315 bands;
- at most 64 atoms;
- at most 343 k-points.

```text
t_step   <= 1.61e-6 x D^0.75 s          (upper envelope over all observed nodes)
stage     = 1.5 x [t_step x (N_scf + 21 + 4 N_ionic) + 30 s] x N_calculations
static    : N_scf = 60, N_ionic = 1
relax     : N_ionic = min(NSW, 3 x degrees of freedom), N_scf = 60 + 20 (N_ionic - 1)
phonons   : N_calculations = Phonopy displacement count
workflow  = sum(stages) + 300 s + 15 %
```

The envelope covers the slowest compute nodes seen in the telemetry (270 and 271).
Node-to-node variability is therefore inside the estimate and no node is assumed equivalent.

In a leave-one-chemistry-out test the model covered 31 of 31 held-out PBE runs from
seven chemistries. The median request was about 33 times the actual runtime.
Relaxations dominate the overestimate, because the number of ionic steps cannot be predicted at plan time.

### Outside the calibrated domain

Stages outside the calibrated domain are clearly identified as conservative fallbacks:

| Class | Applies to | Allowance |
|---|---|---|
| `extrapolated` | PBE outside the ranges above, PBE DOS or band-structure stages, meta-GGA | 3 x the model; the workflow gets at least 72 h |
| `hybrid_fallback` | HSE06, D <= 2e10 (the range of every completed HSE06 run) | 36 h per stage |
| `hybrid_fallback` | HSE06, D <= 1e11 | 7 days per stage |
| `hybrid_fallback` | HSE06 with SOC, or beyond | 21 days per stage |
| `unknown_fallback` | descriptors could not be determined | 72 h per stage; the workflow gets at least 72 h |

HSE06 is not sized with the PBE model. Its per-step cost did not collapse onto the PBE descriptor in the telemetry.

Not every failure in the telemetry was a walltime failure. Several HSE06+SOC runs were stopped by Custodian's frozen-job handler before they converged. These are numerical or Custodian failures, and a longer walltime does not fix them.

### Request

The estimate is rounded up:

- in 15-minute steps below 24 h, and in whole hours above;
- never below 1 h;
- never above the 100-day limit of the `owner100d` QOS. When this cap applies it is reported.

Requests above 72 h are allowed when the estimate justifies them.

## QOS

Every submission under the BMD Compute account (`power-leeburton-users_v2`) uses
`--qos=owner100d`. It appears in the `#SBATCH` header, on the `sbatch` command
line and in the standalone script example. A submission specification with any
other account carries no QOS. There is no user-facing QOS selector.

Manual walltimes are accepted as `H:MM:SS` with up to four hour digits, up to
`2400:00:00` (100 days).

## Reproducibility

The submission specification records `flow_spec.resource_selection`. It holds the
policy id and version, and for each field its source (`automatic` or `manual`), its value and its basis.
For the walltime it also holds the full estimate: per-stage descriptors, class, allowance and reason.

The plan digest (version 2) covers:

- the QOS;
- the source of each field;
- the recommended walltime and its basis;
- the recommended memory;
- the per-stage classes;
- the automatic-resource policy version.

The browser carries a `planned_resources` value from Build to Prepare and Submit.
If the resources resolved at preparation differ from the ones approved at Build, preparation stops and asks for a rebuild.
For example, this happens when a new policy version is deployed in between. The machine API gets the same guarantee from the ledger-bound plan digest.

Deployment note: attempts registered before this change keep their recorded
plan digest (version 1) and fingerprint. Repeating the identical machine-API
request for such an attempt still works once its submission has started: a
submitted attempt returns its existing job, and an uncertain one is reported as
uncertain and never resubmitted. A historical attempt that has not yet been
submitted (registered or prepared only, or reserved without reaching `sbatch`)
cannot be prepared or submitted under the new policy and fails closed with 409
`plan_changed`; plan it again and use a new attempt ID. A changed request can
never reuse an existing attempt ID.

Policy 2, which adds the memory brackets, works the same way. Every automatic plan gets a new digest, even where memory stays at 128 GB. Policy-1 attempts that have not yet been submitted are refused: the browser asks for a rebuild, and the machine API returns 409 `plan_changed`. They are never resubmitted with different memory.

## Deferred

The following are deliberately left for later:

- iteration-count and convergence models;
- broader calibration across chemistries;
- CPU scaling and automatic CPU selection;
- empirical memory prediction;
- node characterisation;
- HSE06 and SOC cost models;
- uncertainty calibration and prospective validation.
