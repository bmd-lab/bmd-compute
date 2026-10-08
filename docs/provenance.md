# Submission Provenance

bmd-compute records a structured provenance block in `submission.json` when a calculation is prepared for remote execution.

The file-level contract for `submission.json` and the job record is described in [run_records.md](run_records.md).

The provenance block supplements the actual executed VASP files. It does not replace INCAR, KPOINTS, POSCAR, POTCAR, OUTCAR, `vasprun.xml`, Custodian logs, or SLURM accounting.

## Schema

Current schema version:

```text
schema_version = 1
```

The top-level provenance sections are:

- `bmd_compute`
- `python_environment`
- `vasp`
- `potcar`
- `execution`

All values are intended to be JSON-safe.

## bmd-compute Source

The source section records best-effort Git metadata:

- commit hash when available
- dirty/clean state when available
- dirty path count when available
- unavailable/unknown status when Git metadata cannot be read

The provenance code uses a one-shot `safe.directory` argument for Git inspection where needed. It does not modify global Git configuration. Git inspection runs with optional locking disabled (`--no-optional-locks`, `GIT_OPTIONAL_LOCKS=0`), so recording provenance never rewrites the checkout's index or leaves `.git/index.lock` behind.

## Runtime Package

`bmd_compute.runtime_source` records the packaged runtime that Prepare uploads
to `<run_dir>/backend/`: every `backend/**/*.py` file outside `tests/` and cache
directories, with its SHA-256 (`manifest`, keyed by package-relative POSIX
path). Files are uploaded byte for byte, so the manifest is the hash of exactly
what lands on POWER.

The manifest is enforced, not only recorded:

- Prepare refuses to upload if the package it is about to upload no longer
  matches the recorded manifest.
- The generated `run_job.py` carries the same manifest and the verification
  code itself (embedded from `backend/runtime_package_guard.py`, never imported
  from the uploaded copy). Before importing anything from `backend/` it requires
  the directory to hold exactly the manifest files: each a regular file with the
  recorded SHA-256, and no missing, extra or symbolic-link entries
  (`__pycache__` directories are ignored). Every `backend` module is then loaded
  from source bytes that are hashed again at load time; bytecode caches are
  never read or written for it.
- Before its own standard-library imports, `run_job.py` removes the run
  directory from `sys.path`, so a file left there cannot shadow a module the
  bootstrap imports before the check.
- The Prepare runtime preflight runs that same `run_job.py` in preflight mode,
  so a prepared attempt has already passed the check once on POWER.

A mismatch stops the run before any package code executes and before VASP
starts, with a `BMD_RUNTIME_PACKAGE_VERIFICATION_FAILED` report in the runner
stderr log listing each missing, modified or unexpected file. The remedy is a
new submission attempt; files in a prepared run directory are not meant to be
edited.

Attempts prepared before this check existed keep the `run_job.py` they were
prepared with and run unverified, as before.

## Python Environment

The preparation environment records:

- Python version
- Python implementation
- local versions of the scientific packages in `constraints/scientific-runtime.txt` (atomate2, pymatgen, pymatgen-core, custodian, emmet-core, jobflow, spglib, and the supporting monty, numpy, scipy, pydantic, pydantic-settings, maggma, ruamel.yaml and phonopy)

`submission.json` also records `runtime_parity`: the exact versions of the parity-critical packages at preparation. The POWER runner reads its own versions before any VASP work, writes them to `<run_dir>/runtime_environment.json` (see `run_records.md`) and stops if a parity-critical package differs or an input-altering atomate2 setting is active. `python_environment.remote_execution` in the provenance block points to that record; at submission time it is still marked deferred because the runtime record does not exist yet. Runs prepared before this mechanism have no runtime record; their runner log is the only runtime evidence.

## VASP Execution

The VASP section records:

- global VASP command template
- per-stage executable policy
- per-stage command template
- per-stage Custodian `VaspJob` options set by BMD (SOC stages record `auto_gamma: false`)
- deferred VASP version/build information

Per-stage executable provenance distinguishes ordinary `vasp_std` stages from SOC/non-collinear `vasp_ncl` stages.

The actual argv passed to Custodian is still determined at runtime from the sbatch environment and submission resources.

## Workflow And Resources

The execution section records:

- serialized `WorkflowSpec`
- for BMD-managed Desired Output workflows, the automatic treatment record (`automatic_treatments`): base recipe, resolved workflow, which treatments were applied to which stages, and any consideration with no applicable stage
- when the automatic DFT+U rule triggers, `automatic_treatments.dft_u`: policy id/version, rule id, deciding anion, triggering elements, every pymatgen oxidation-state guess and the derived d counts, the d0 gate result and reason, the decision, the frozen L/U/J/LDAUTYPE parameters and their source, the `MPRelaxSet.yaml` SHA-256, the `atomate2`/`pymatgen`/`pymatgen-core` versions, and, per +U stage, the generated `LDAU`/`LDAUTYPE`/`LDAUL`/`LDAUU`/`LDAUJ`/`LMAXMIX` with POSCAR species order and POTCAR symbols. Runtime regenerates each +U stage from the frozen values and stops if any of these differ
- stage order
- per-stage Custodian policy (`execution.custodian.stages`), including the unsuccessful-stage policy `stop_children_kwargs = {"handle_unsuccessful": "error"}` that makes an unconverged stage fail the workflow
- selected resources
- partition/account policy
- module load policy
- environment policy such as `VASP_CMD`, `JOBFLOW_CONFIG_FILE`, and `PMG_VASP_PSP_DIR`

This makes the prepared scientific and operational intent visible without requiring a user to infer it from UI text.

## POTCAR Identity

The POTCAR section records:

- functional
- species
- symbols
- symbol source
- repository/path policy where available

Species and symbols are read from the executable stage generators, the same
per-stage input sets used for generated-input previews and the input-reference
producer, in POTCAR.spec mode (`symbol_source` is
`bmd_compute.executable_stage_generators`). There is no separate BMD POTCAR table
and no fallback to another input set. `stages` lists every stage's
species-to-POTCAR mapping in POSCAR order. When all stages agree,
`consistent_across_stages` is true and `species`/`symbols` repeat that mapping;
when they differ, `species`/`symbols` are null and `stages` is the record. If the
generators cannot be evaluated, `status` is `unavailable` with a `reason`.

Records written before this change took `species`/`symbols` from pymatgen's
`MPRelaxSet` (`symbol_source` `pymatgen`) and can disagree with the POTCARs that
actually ran (for example `Ni_pv` recorded where `Ni` executed). For those runs,
the executed `POTCAR` in the run directory is authoritative.

POTCAR hashes are not currently recorded. Raw POTCAR contents are not stored in provenance.

If the lab later requires POTCAR hash provenance, add it deliberately and document how hashes are computed from the remote/shared POTCAR repository.

## Limitations

Provenance is best-effort metadata captured at preparation time. It does not prove that a remote executable, module, POTCAR repository, or cluster environment remained unchanged after submission.

For scientific reproducibility, retain the actual remote stage directories and VASP/Custodian outputs alongside the provenance block.
