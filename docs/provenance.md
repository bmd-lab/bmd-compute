# Submission Provenance

BMD Compute records a structured provenance block in `submission.json` when a calculation is prepared for remote execution.

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

## BMD Compute Source

The source section records best-effort Git metadata:

- commit hash when available
- dirty/clean state when available
- dirty path count when available
- unavailable/unknown status when Git metadata cannot be read

The provenance code uses a one-shot `safe.directory` argument for Git inspection where needed. It does not modify global Git configuration.

## Python Environment

The preparation environment records:

- Python version
- Python implementation
- local scientific package versions for `pymatgen`, `atomate2`, `jobflow`, and `custodian`

Remote execution package versions are marked as deferred at submission time because the remote job environment is the source of truth when the calculation actually runs.

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
- stage order
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

POTCAR hashes are not currently recorded. Raw POTCAR contents are not stored in provenance.

If the lab later requires POTCAR hash provenance, add it deliberately and document how hashes are computed from the remote/shared POTCAR repository.

## Limitations

Provenance is best-effort metadata captured at preparation time. It does not prove that a remote executable, module, POTCAR repository, or cluster environment remained unchanged after submission.

For scientific reproducibility, retain the actual remote stage directories and VASP/Custodian outputs alongside the provenance block.
