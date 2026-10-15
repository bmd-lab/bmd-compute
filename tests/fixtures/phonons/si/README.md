# Si phonon result fixtures

Written by Phonopy 4.8.1 through the same atomate2 0.1.5 analysis path the
phonon workflow runs (seekpath path from pymatgen `KPathSeek`, Phonopy
`write_yaml_band_structure` and `write_total_dos`), so the file formats are
those of `stage_02/generate_frequencies_eigenvectors/` on POWER.

- `phonon_band_structure.yaml`: 6 path segments (Γ–X, X–U | K–Γ, Γ–L, L–W,
  W–X) with 5 q-points each instead of atomate2's 101, to keep the fixture
  small. The path has one discontinuity (U|K).
- `phonon_dos.yaml`: the full total-DOS file (tetrahedron method).
- `phonopy.yaml`: only the header and primitive cell (first 37 lines); the
  supercell block is omitted.

The force constants come from the harmonic spring model used by
`tests/test_phonon_end_to_end_fake.py`, not from VASP, so the frequencies are
not those of PBE silicon (optical Γ ≈ 15.23 THz here; 15.02 THz in the live
POWER run).
