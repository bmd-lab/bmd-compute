# Phonon Displacement Plan (M1 foundation)

`backend/phonons` builds the deterministic Phonopy displacement plan that future
phonon execution will consume. It is **not** an executable workflow: there is
no `phonons` Desired Output, no stage type, no Prepare/Submit path and no force
calculation. The planning policy is provisional and is not BMD methodology.

```python
from backend.phonons import build_displacement_plan

plan = build_displacement_plan(structure, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
plan.plan_sha256, plan.task_count, plan.tasks
```

## Stage versus task

In the eventual R2 workflow, phonons are three scientific stages: a phonon-grade
geometry optimisation, one PBE phonon force-calculation stage, and phonon
analysis. The plan describes the task instances of the force stage. Each
displacement is a task (`task_kind = "phonon_displacement"`, ids `disp_001`,
`disp_002`, ...), not a workflow stage. Tasks carry no stage type, theory,
modifiers, INCAR or KPOINTS: those belong to the common stage methodology,
which is shown once. Task order is Phonopy's dataset order. It defines task
identity and the order in which forces must be supplied. It is not an
execution order, so sequential, concurrent or array execution can all use the
same plan.

Under R2 the plan is built from the relaxed structure after the prerequisite
relaxation. Build and Prepare therefore cannot know the final task count, and
nothing here assumes they do.

## Inputs and policy

- `structure`: an ordered, non-magnetic pymatgen `Structure`. A nonzero `magmom`
  site property or a nonzero `Species.spin` is refused. The structure is not
  standardized, wrapped or symmetrized; standardization is not approved
  methodology yet. Its identity in the plan is the lattice, element symbols and
  fractional coordinates as given. Oxidation-state decorations and other site
  properties (for example `selective_dynamics`) are not part of that identity.
  How other site properties should be treated is an open policy question to
  settle before execution.
- `supercell_matrix`: an explicit 3x3 integer matrix with positive determinant.
  Automatic supercell selection is not approved and is not provided.
- `PhononPolicy` (`bmd_compute.phonon_displacement_planning` v1, status
  `provisional_not_executable`): displacement 0.01 Å, `is_plusminus="auto"`,
  `is_diagonal=True`, `symprec=1e-5`, `primitive_matrix="auto"`. A policy that
  claims this id and version must carry exactly these values. Changing one,
  `symprec` included, needs a new registered version.

Every Phonopy planning argument is passed explicitly and recorded in
`phonopy_arguments`. That includes `primitive_matrix="auto"` (Phonopy's default
changed in v4), `use_SNF_supercell=False` (it changes supercell atom order),
`distinguish_symbol_index=False`, `calculator="vasp"` and `lang="C"`. A silent
backend fallback is refused. The primitive matrix Phonopy resolves is recorded
explicitly.

## Record: `bmd_compute.phonon_displacement_plan` v1

| Field | Meaning |
| --- | --- |
| `schema`, `schema_version` | `"bmd_compute.phonon_displacement_plan"`, `1` |
| `policy` | the full policy, including id, version and status |
| `software` | `phonopy` and `spglib` versions |
| `phonopy_arguments` | `construct` (to `Phonopy(...)`) and `displacements` (to `generate_displacements(...)`) |
| `working_structure` | lattice (Å, row vectors), element symbols, fractional coordinates, `sha256` |
| `symmetry` | detected international symbol and number, and the `symprec` used |
| `supercell_matrix`, `primitive_matrix`, `primitive_natom` | the explicit and resolved transformations |
| `supercell` | Phonopy's supercell: lattice, symbols, Cartesian coordinates (Å), `sha256` |
| `dataset` | Phonopy's displacement dataset (`natom`, `first_atoms[number, displacement]`) |
| `task_kind`, `tasks[]` | `task_id`, `dataset_index`, `atom_index`, `displacement` (Cartesian Å), `structure_sha256` |
| `plan_sha256` | SHA-256 of the canonical JSON of every other field |

A task's `structure_sha256` identifies its displaced supercell: the supercell
with the displacement added to one atom's Cartesian position. That is how
Phonopy 4.8.1 builds displaced cells, so any reader can recompute it from the
record alone.

## Canonicalization

The canonical JSON of a record uses only plain JSON types (NumPy values are
rejected, not coerced), with sorted keys, no whitespace and ASCII escapes.
Floats are written with Python's shortest round-trip `repr`, so the exact
binary64 value is kept and nothing is rounded. `-0.0` is written as `0.0`, and
NaN or infinity is rejected. The plan therefore does not depend on dictionary
order, object identity, paths, locale or process. It contains no paths,
hostnames or timestamps. Because floats are exact, two hosts whose linear
algebra differs in the last bit produce different hashes, so a future
cross-host check should compare rebuilt plans structurally.

## Validation and authority

`validate_displacement_plan_record` (and `DisplacementPlan.from_dict`/`from_json`)
rejects malformed or inconsistent records and never repairs them. It checks key
sets, types, matrices, counts, task ids and order, dataset indices,
displacement lengths, every structure hash and the plan hash. A valid record is
only internally consistent. It cannot show that the dataset is complete, so a
supplied plan is accepted as scientific authority only through
`verify_displacement_plan`, which rebuilds the plan and requires an exact match.

## Runtime isolation

`import backend.phonons` imports neither Phonopy nor pymatgen. Phonopy is
imported only inside `build_displacement_plan`/`verify_displacement_plan`, and
when it is missing they raise `PhonopyUnavailableError`. `phonopy==4.8.1` is a
supporting (Tier 2) pin: it is recorded at run time, and its absence on POWER
does not stop any existing workflow. It must become parity-critical before
phonon execution depends on it.
