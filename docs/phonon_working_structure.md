# Phonon Working Structure (Phonopy M2b)

Status: provisional and not executable. Nothing in BMD Compute runs this yet.
M2c will call it after Stage 1 completes.

Under R2, the phonon displacement plan derives from the actual output of the
prerequisite relaxation. `backend/phonons/working_structure.py` defines the
deterministic boundary between that output and the structure M1 plans with:

```
A  original user structure        kept in the submission; not an input here
B  actual Stage-1 structure       -> incoming identity
C  BMD phonon working structure   -> working identity -> M1 DisplacementPlan
```

`prepare_phonon_working_structure(stage1_structure)` reads B, never modifies
it, and returns a `bmd_compute.phonon_working_structure` v1 record. C is never
presented as the Stage-1 output.

## Identities

| Identity | Definition | Answers |
| --- | --- | --- |
| `incoming.sha256` | canonical hash of `lattice` (rows, Angstrom), `species` (per site `{element, oxidation_state, spin}`), fractional `coords` exactly as given, `coords_type`, and the supported `site_properties` (`magmom`, `velocities`) | what exact structure Stage 1 handed over |
| `working.sha256` | M1 `structure_record` hash: lattice, element symbols, fractional coordinates | what exact structure BMD handed to Phonopy; equal to `plan.working_structure.sha256` |
| `record_sha256` | canonical hash of the whole record except itself | which transformation record |

Both structure identities describe the exact representation. The same crystal
with different site order, origin or unwrapped coordinates is a different
identity. Site labels are not part of either identity.

## Site, species and structure policy

The policy is closed. Anything not listed is rejected, never repaired.

| Input | Treatment |
| --- | --- |
| `magmom` (scalar, 3-vector or `Magmom`) | exactly zero on every site: recorded in the incoming identity, then dropped (`drop_zero_magmom`). Nonzero, non-finite or unreadable: rejected (`magnetic`). |
| `velocities` (VASP CONTCAR block) | exactly zero: recorded, then dropped (`drop_zero_velocities`). Otherwise rejected. |
| `selective_dynamics` | rejected, as in every managed BMD workflow |
| `predictor_corrector`, `forces`, any other site property | rejected |
| `Species` oxidation state | recorded in the incoming identity; reduced to the element symbol (`reduce_species_to_elements`) |
| `Species` spin | zero: recorded and reduced; nonzero or unreadable: rejected (`magnetic`) |
| dummy species, disordered or partially occupied sites | rejected |
| structure charge, structure-level `properties`, non-periodic directions | rejected |
| non-finite values, singular or left-handed lattice | rejected |

"Zero" means exactly zero (`-0.0` counts as zero). This is the same exact-zero
rule M1 applies to spin and `magmom`. A Stage-1 calculation with residual
moments such as `0.001` is therefore refused; see the open questions below.

## Symmetry idealization

Applied as `symmetry_idealization`, always:

1. spglib analyses B with `symprec_angstrom = 1e-5`,
   `angle_tolerance_degrees = -1.0` and `hall_number = 0`. All three are passed
   explicitly.
2. The lattice becomes spglib's idealized standard lattice mapped back into the
   input basis and Cartesian frame: rows `P^T L_std R`, where `P` is
   `transformation_matrix` and `R` is `std_rotation_matrix`.
3. Each position becomes the average of its symmetry images under the detected
   operations, which is the projection onto the symmetric configuration.
4. The working structure must give the same detected space group and operation
   count as the incoming structure, and no site may move more than `symprec`.
   Otherwise the structure is refused (`symmetry_unstable`).

This keeps the atom count, atom order, lattice basis, origin and handedness.
The Cartesian frame is kept: a rotated input gives the same rotation of the
output. Free coordinates such as a polar axis are not moved, and nothing is
wrapped or rounded. Averaging can change the last bits of coordinates that were
already exact. That change is deterministic and visible in the record
(`idealization.max_site_shift_angstrom`).

Full spglib standardization (`standardize_cell`, refined or primitive cells) is
not used. On the reference structures it changes the atom count (diamond Si
goes from 2 to 8 atoms). On GaN rounded to 4 decimals it rotates and reflects
the frame and shifts the origin. Choosing a conventional or primitive cell, or
a frame rotation, would be unapproved methodology.

### Tolerance

The idealization tolerance is the policy's own versioned value. It equals the
approved M1 Phonopy `symprec` (1e-5), and the two are not coupled in code.
Idealization therefore only makes exact what Phonopy would already treat as
symmetric. It never raises the symmetry Phonopy sees:

| GaN input | Detected | Idealization | Displacements (3x3x2) |
| --- | --- | --- | --- |
| exact | P6_3mc | none needed | 4 |
| `1/3` rounded to 6 decimals | P6_3mc | sites moved by 1.8e-6 Angstrom | 4 |
| `1/3` rounded to 4 decimals | Cmc2_1 | sub-tolerance only | 8 |

Recovering P6_3mc for the last case needs a looser tolerance. That is an
unapproved scientific decision, and M2b does not make it.

## Record: `bmd_compute.phonon_working_structure` v1

| Field | Meaning |
| --- | --- |
| `schema`, `schema_version` | `"bmd_compute.phonon_working_structure"`, `1` |
| `policy` | `policy_id`, `policy_version` (1), `status` (`provisional_not_executable`), `method` (`spglib_symmetry_idealization_in_input_setting`), `symprec_angstrom`, `angle_tolerance_degrees`; registered values only |
| `software` | `spglib`, `numpy` versions |
| `incoming` | B, with its `sha256` |
| `working` | C as an M1 `structure_record`, with its `sha256` |
| `transformations` | applied kinds, in the fixed order `drop_zero_magmom`, `drop_zero_velocities`, `reduce_species_to_elements`, `symmetry_idealization` |
| `symmetry.incoming`, `symmetry.working` | `international`, `number`, `operations`; must be equal |
| `idealization` | `max_site_shift_angstrom`, `max_lattice_change_angstrom`, recomputed in plain Python by the validator |
| `record_sha256` | canonical hash of every other field |

The record contains no supercell, primitive matrix, displacements, tasks,
INCAR, KPOINTS, scheduler data, timestamps or hosts. Canonical JSON and
hashing are M1's (`backend/phonons/records.py`).

## Validation

- **Structural:** `validate_working_structure_record`, which
  `PhononWorkingStructure` also runs. It rejects:
  - unknown keys;
  - a wrong schema, version or policy;
  - an incoming hash that does not match its content;
  - working species that are not the incoming elements;
  - a transformation list that differs from the one the incoming structure
    implies;
  - different incoming and working symmetry;
  - idealization magnitudes that do not match the two structures;
  - a hash mismatch.
- **Authority:** `verify_phonon_working_structure(record, stage1_structure)`
  re-derives the record from the actual Stage-1 structure and requires
  identical canonical JSON. It fails with:
  - `incoming_mismatch` for another Stage-1 structure;
  - `working_mismatch` for a consistently forged working structure;
  - `software_mismatch` when spglib or NumPy differ. In that case it does not
    guess.

Exact floats mean that hosts whose linear algebra differs in the last bit
derive different working structures. Re-verification then fails closed, as for
M1 plans.

## Chain to M1 and M2a

`PhononWorkingStructure.working_structure()` returns a new pymatgen structure
for `build_displacement_plan`, and the plan's `working_structure` equals the
record's `working` exactly. The M2a task set records the Stage-1 identity as
`upstream_structure.sha256`; see `stage_task_sets.md`.
`verify_phonon_force_task_set_chain` checks, in order:
1. the parent stage;
2. the re-derivation of the working-structure record from the actual Stage-1
   structure;
3. the upstream identity;
4. the M1 plan rebuild from the working structure;
5. the task list.

## Open questions (not decided in M2b)

- **Stage-1 eligibility:** whether a relaxation that changed the space group
  should be refused. M2b records both groups but does not judge them.
- **Recovering symmetry lost beyond 1e-5:** a looser idealization tolerance,
  or standardizing before Stage 1.
- **Residual moments:** whether Stage 1 for non-magnetic phonons runs without
  spin polarization, or a residual-moment tolerance is approved. Today any
  nonzero `magmom` is refused.
- **Phonopy parity:** whether spglib and NumPy must match exactly between
  preparation and execution for re-verification on POWER. Phonopy parity is an
  M3 requirement.
