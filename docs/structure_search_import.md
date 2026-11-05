# Importing USPEX Results Run by Hand

Issue #63, part of the structure-search work tracked in #58 (design:
`structure_search.md`). This page describes what is implemented: how a user
brings the results of a USPEX search they ran themselves into bmd-compute,
and how candidates reach the existing Desired Outputs.

Running USPEX from bmd-compute (search spec, INCAR ladder, launch bridge,
POWER resources) is separate work and is not part of this feature.

## What the user does

1. On the workbench, open **Import Structure Search Results** and upload a
   `.zip` or `.tar.gz` of either the USPEX calculation folder (the one that
   holds `results1/`) or of one or more `resultsN/` folders.
2. The **Structure Search Candidates** stage shows:
   - a table of the ranked candidates: rank, individual ID, results folder,
     generation, USPEX origin (`Heredity`, `keptBest`, ...), composition,
     USPEX energy per cell and per atom, volume per atom, the space group
     USPEX reports, and the space group pymatgen/spglib finds at the
     parser's default tolerance (`SYMMETRY_SYMPREC`, 0.1 Å);
   - a plot of USPEX energy per atom against generation (best per
     generation, and every listed candidate), drawn by the same Plotly
     renderer as the results pages;
   - every warning the parser recorded, and a note if the search pressure
     is not zero or could not be found.
3. Tick one or more candidates, choose a Desired Output and press **Use
   Selected Candidates**. Each candidate is listed with its POSCAR and a
   **Build** button. Build posts the POSCAR to the ordinary
   `/build-calculation` route, exactly as if it had been pasted, so it passes
   through `parse_structure()`, automatic treatments, new-calculation
   admission, resource selection, Prepare and Submit unchanged. With several
   candidates each Build opens in its own tab.
4. The built calculation shows a **Structure origin** panel, and the USPEX
   record is stored with the submission (see Provenance below).

USPEX energies are search-grade: they come from USPEX's own relaxation
ladder (the `energy` column of `Individuals`, eV per cell; the enthalpy at
the search pressure). They help choose candidates. The final ranking comes
from the BMD calculation each candidate is sent to.

The Structure Summary of a built candidate can report a lower symmetry than
the candidate table. It uses pymatgen's default tolerance (0.01 Å); the
table uses 0.1 Å. For the Si8 pilot's best structure (individual 35) the
table shows Fd-3m (227) and the Structure Summary P2₁ (4). USPEX itself lists
space group 1. The structure handed off is the same in all three cases. Only
the tolerance differs.

## Archive contents and safety

Minimum content: in at least one `resultsN/` folder, `Individuals` plus
`gatheredPOSCARS` or `good_structures_POSCARS`. Candidates are written from
`gatheredPOSCARS` when present (every relaxed individual), otherwise from
`good_structures_POSCARS`. A candidate with no POSCAR block cannot be
selected; the page says so.

`backend/structure_search/archive.py` checks the archive. Only standard
library code is used, and nothing in the archive is executed.

- The format is detected from the file content (`PK..` zip, `1f 8b` gzip),
  not from the file name. Uncompressed `.tar` and other files get a clear
  message.
- The whole archive is rejected if any member has an absolute path, a `..`
  component, a backslash, a drive letter or a NUL character, or is a
  symbolic link, hard link, device file, FIFO or other special file, or is
  encrypted.
- Only the result files the parser reads are written (`Individuals`,
  `BESTIndividuals`, `gatheredPOSCARS`, `good_structures_POSCARS`,
  `BESTgatheredPOSCARS`, `best_structure_POSCAR`, `Parameters.txt` inside a
  `resultsN/` folder, and `INPUT.txt`). Everything else (CalcFolds, INCARs,
  scripts) is skipped. Files are written with mode 0600.
- Limits (safety limits for the web server, defined at the top of
  `archive.py`): 200 MB upload, 20,000 members, 256 MB per kept file,
  512 MB kept in total, 4 GB decompressed from a `.tar.gz` stream. Sizes
  are counted from the decompressed bytes, not from sizes the archive
  claims.
- An archive holding results of more than one calculation is refused.

## Storage and lifetime

Before this feature the app had no server-side upload handling: structures
are pasted text, and `uploads/` was an empty, git-ignored folder. Imports
now live in `uploads/structure_search/<import id>/`:

- `import.json`: the archive's original file name, format, size and SHA-256,
  the calculation folder inside the archive and the results folders read;
- `candidates.json`: the candidate table shown on the page;
- `search/`: the kept result files.

The uploaded archive itself is deleted once unpacked. Each new import first
deletes imports older than 7 days and keeps at most 20
(`IMPORT_RETENTION_SECONDS`, `MAX_STORED_IMPORTS` in
`backend/structure_search/imports.py`). An import can be reopened at
`/structure-search/<import id>` while it exists. Deleting an import does not
affect calculations already built from it, because their origin travels
with the submission (next section).

## Provenance

The hand-off gives each candidate a signed structure origin token
(`backend/structure_origin.py`). It holds the origin record and the SHA-256
of the POSCAR text, signed with a key derived from
`BMD_SUBMISSION_IDENTITY_SECRET`. Build, Prepare and Submit verify it
against the posted structure text. A token that does not verify, or that
belongs to a different (edited) structure, stops the request with a message.
Without a configured secret, tokens stop verifying when the server restarts
(as submission identity tokens do); the user hands the candidate off again.

The verified record is stored as `flow_spec.structure.origin` and copied to
`provenance.structure_source.origin` (see `provenance.md`). For a USPEX
candidate it contains:

| Field | Meaning |
|---|---|
| `schema`, `source` | `bmd-compute.structure-origin.uspex-import.1`, `uspex_import` |
| `archive_sha256`, `archive_filename`, `archive_format` | the uploaded archive |
| `import_id`, `imported_at` | the import on this server |
| `calculation_path_in_archive`, `results_folder`, `poscars_file` | where the structure was read |
| `individual_id`, `generation`, `origin`, `parents`, `lineage_ids` | the USPEX individual |
| `uspex_energy_ev`, `uspex_energy_ev_per_atom`, `uspex_energy_note` | USPEX's energy, as read from `Individuals` (three decimals per cell) |
| `uspex_space_group`, `uspex_rank` | as reported by USPEX / the parser's ranking |
| `formula`, `natoms` | composition |
| `external_pressure_gpa` | `ExternalPressure` from `Parameters.txt`/`INPUT.txt`, or null |
| `parser_schema` | version of the parser summary used |

## Known gaps

- The Relaxed Energy workflow has no pressure setting. Candidates of a search
  at non-zero pressure are relaxed at zero pressure; the page warns about
  this (see `structure_search.md`, section 6).
- Equivalent structures with different IDs are not merged (parser
  limitation; see `results.py`).
- Only USPEX 25.0 output has been seen (one single-element search). An
  unknown column header stops the import with a message that names it.
