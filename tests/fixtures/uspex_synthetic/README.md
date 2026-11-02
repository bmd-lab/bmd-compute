# Synthetic USPEX results (provisional)

**These files are synthetic. They were not produced by USPEX.** They exist only
to test `backend/structure_search/results.py` (issue #62) before real output is
available.

When they were written, the column layout of `Individuals` / `BESTIndividuals`,
the comment line of the `*POSCARS` files, and the way USPEX marks a missing
enthalpy had not been observed on real USPEX 25.0 output (see
`docs/structure_search.md`, §9). The files follow the provisional choices
recorded in `COLUMN_ALIASES`, `POSCAR_ID_PATTERN` and `MISSING_ENTHALPY_TOKENS`
in `backend/structure_search/results.py`, nothing more.

**They must be replaced by, or checked against, a trimmed `results1/` from the
Phase 0 pilot (#59) before the import UI (#63) is merged.** If real files
differ, update the three definitions above and these fixtures together.

## Contents

`mgo_two_restarts/` imitates a fixed-composition MgO search that was restarted
once (`results1/`, then `results2/`). All numbers are made up:

| Folder | Generations | IDs | Notes |
|---|---|---|---|
| `results1` | 1–3 | 1–7 | generation 2 (IDs 4, 5) has no enthalpies (`NaN`), as for failed relaxations |
| `results2` | 4–5 | 8–10 | lowest enthalpy per atom is ID 8 |

- Structures are idealised rock-salt (Mg4O4) and simple hexagonal (Mg2O2)
  cells built with pymatgen; IDs 3 and 10 have one atom slightly displaced.
- Enthalpies were chosen by hand (about −5.85 eV/atom). They are not DFT
  results and must not be used as reference values.
- `gatheredPOSCARS` holds every individual with an enthalpy;
  `goodStructures_POSCARS` holds IDs 1, 6 (`results1`) and 8, 10 (`results2`).
- `Parameters.txt` contains only the `% atomType` block that the parser reads.

Cases that need damaged or growing files (truncated last line, incomplete
POSCAR block, unknown header, missing element line) are built from these files
inside the tests, in a temporary directory.

No research data is included.
