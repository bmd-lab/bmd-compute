# Sample USPEX 25.0 output: Si8 bridge pilot

Real output of the Si8 pilot (`../`, #59) run on POWER on 4 Nov 2026
(job with `whichCluster = -1`, launch bridge from #66, `srun_steps`, 2
relaxations of 4 ranks at a time). Si8 at 0 GPa, population 8, 4
generations, 3 relaxation steps with the pilot INCARs in `../Specific/`
(pilot settings, not BMD policy). Toy system, no research data.

Copied unchanged from `search/results1/`, except that `uspex.output` (a
copy of the master log) and `gatheredPOSCARS_unrelaxed_all` (identical to
`gatheredPOSCARS_unrelaxed`) are left out. No POTCAR content is included.

What the files show (USPEX 25.0, 11 Nov 2025 build):

- File names differ from the 25.0 manual: `good_structures` and
  `good_structures_POSCARS` (not `goodStructures_POSCARS`), plus
  `Individuals_all`, `Individuals_extended`, `best_structure(_POSCAR)`,
  `BESTgatheredPOSCARS`, `generation_properties`, `enthalpies_*.dat`.
- `Individuals`: whitespace columns `generation number num_atoms_all energy
  cell_volume space_group origin parents e_above_hull opt_property
  quasi_entropy a_order s_order`. `energy` is per cell in eV (enthalpy at
  the run's pressure; 0 GPa here). `num_atoms_all` and `parents` are Python
  lists (`[8]`, `[2, 8]`, `[]`), and a list may contain spaces. `e_above_hull`
  is per cell, relative to the best structure.
- `Individuals_extended` adds `energy_per_atom`, `full_formula`,
  `space_group_symbol`, `density` and more.
- Survivors are recalculated: they reappear in the next generation as new
  individuals with origin `keptBest` and a new number (17, 26, 35 here).
  35 individuals x 3 steps = 105 relaxations.
- POSCAR blocks in `gatheredPOSCARS`, `good_structures_POSCARS` and
  `best_structure_POSCAR` start with
  `number=<id>, energy=<eV per cell>, e_above_hull=..., opt_property=...`;
  in `BESTgatheredPOSCARS` with `Gen=<g>, generation=<g>, number=<id>,
  energy=...`. There is no `EA<id>` tag.
- Space groups are under-determined: the best structure (number 35,
  -43.2799 eV, 20.46 A^3/atom) is listed as space group 1, but is diamond
  Si (Fd-3m, 227) at spglib symprec >= 0.05 and matches ideal diamond
  with pymatgen's StructureMatcher (RMS 0.0004).
- `Calculation/` keeps only the last generation's CalcFolds
  (`Calcfold_<individual>_<step>`, here 27-35); earlier ones are removed.
