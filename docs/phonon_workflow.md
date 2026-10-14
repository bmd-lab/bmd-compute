# Phonon Workflow (Milestone A)

Status: executable integration. Workflow construction, admission, preview and a
simulated end-to-end run (fake VASP) are tested. No live POWER calculation has
validated the scientific results yet. The phonon results UI is Milestone B.

## Workflow

`phonons` Desired Output (browser and machine API, `--desired-output phonons`):

1. **Stage 1 — PBE Geometry Optimisation** (`stage_01`): the existing BMD relax
   (RelaxSetGenerator, ISIF 3, automatic NCORE) with phonon-grade settings
   `ENCUT = 700` (matching the force calculations), `EDIFF = 1e-7`,
   `EDIFFG = -1e-3`. Applied only to a relax that directly feeds a phonon
   stage; ordinary relaxations keep ENCUT 580, EDIFF 1e-6, EDIFFG -0.01.
2. **Stage 2 — Phonon Force Calculations** (`stage_02`): atomate2 0.1.5
   `atomate2.vasp.flows.phonons.PhononMaker` on the stage-1 output structure,
   with `bulk_relax_maker`, `static_energy_maker` and `born_maker` set to
   `None`:
   - `get_supercell_size` — automatic supercell, minimum image length 10 Å,
     `prefer_90_degrees=True` (Si: 2×2×2 conventional cell, 64 atoms,
     1 displacement). 10 Å is an initial practical default (atomate2's own
     default is 20 Å), not a claim of force-constant convergence; check
     supercell convergence per material;
   - `generate_phonon_displacements` — Phonopy, 0.01 Å, `symprec = 1e-5`;
   - `run_phonon_displacements` — replaced at run time by one
     `PhononDisplacementMaker` VASP calculation per displacement;
   - `generate_frequencies_eigenvectors` — Phonopy force constants
     (`FORCE_CONSTANTS`, `phonopy.yaml`), band structure on the seekpath path,
     DOS. No LO–TO correction.

Force calculations use atomate2's `PhononDisplacementMaker` input set
(ENCUT 700, EDIFF 1e-7, reciprocal density 100, LREAL False, ALGO Normal) with
BMD amendments: `GGA`, `ENAUG`, `LMIXTAU` removed (PBE, as in every BMD stage;
atomate2's base set is PBEsol), `ISPIN = 1` with `auto_ispin` off, single-point
Gaussian-smeared forces (`ISMEAR = 0`, `SIGMA = 0.2`, `IBRION = -1`, `NSW = 0`,
`ISIF` unset — overriding atomate2 0.1.5's `IBRION = 2`/`ISIF = 3` and the
static set's `ISMEAR = -5`), and NCORE never set. No WAVECAR/CHGCAR is carried from the relax.

Phonopy, called by atomate2, is the authority for the supercell, displacements
and force constants. `backend/phonons/plan.py` remains an independent
cross-check of the same displacements (tested), not an execution input.

Settings live in `backend/phonons/workflow.py`; amplitude and symprec are read
from `backend/phonons/policy.py`.

## Scope and admission

- PBE only; the phonon stage must be last and directly follow a full
  (cell + ions) PBE Geometry Optimisation.
- Excluded: spin polarisation, SOC, DFT+U, HSE06, Gamma-only. A `phonons`
  Desired Output fails closed when BMD's automatic treatments would apply spin,
  SOC or DFT+U (`phonons_structure_requires_unsupported_treatment`); Custom
  workflows are refused by the registry and admission
  (`phonon_forces_unsupported_modifier`, `phonons_require_nonmagnetic_pbe`).
- DFT-D3 / D3(BJ) is supported and must match on the relax and the force stage;
  automatic dispersion applies to both.

## Execution directories

`backend/execution.py` assigns each top-level stage its directory as before. A
nested stage flow gives each of its jobs a named subdirectory, and jobs created
by `Response.replace/detour/addition` get named subdirectories of the job that
created them:

```
stage_01/
stage_02/get_supercell_size/
stage_02/generate_phonon_displacements/
stage_02/run_phonon_displacements/phonon_static_1_of_N/ ...
stage_02/generate_frequencies_eigenvectors/   # phonopy.yaml, FORCE_CONSTANTS, ...
```

All displacements run sequentially inside the one SLURM allocation, each with
the full VASP command.

## Preview

The Build preview and the machine-API plan show, for the phonon stage, the
supercell matrix, atom count and displacement count computed by the same
atomate2/Phonopy job functions from the *submitted* structure, plus the
force-calculation INCAR/KPOINTS/POSCAR of the first displaced supercell. At run
time they are recomputed from the relaxed structure.

## Runtime

phonopy 4.8.1 and seekpath 2.1.0 are parity-critical for phonon submissions
only (`docs/deployment.md`, "Tier 1P"). Install both in the preparation and
POWER environments before preparing a phonon calculation; other workflows do
not depend on them.
