# Stage Task Sets (Phonopy M2a)

Status: representation only. Nothing in bmd-compute builds, prepares, submits or
executes a stage task set yet.

## Stages and tasks

A workflow is an ordered sequence of stages (`WorkflowSpec` of `StageSpec`). A
stage describes scientific methodology. A *task* is one concrete calculation of
a stage. `backend/calculations/stage_materialization.py` records, per stage
type, how a stage becomes tasks:

| Materialization | Meaning | Stage types |
| --- | --- | --- |
| `single` | the stage is exactly one calculation | `relax`, `static`, `dos`, `band_structure` |
| `derived_task_set` | one methodology applied to tasks derived at runtime | `phonon_forces` |

Existing stages keep the implicit one-stage, one-calculation model. Their
`WorkflowSpec`, `StageSpec`, `bmd_compute.submission` v1 records, stage
directories, generated inputs, run records and result discovery are unchanged,
and they never get a task-set record.

This is materialization, not scheduling. Nothing here says whether tasks run
sequentially, concurrently or as an array job.

`StageSpec` is unchanged. Task IDs, task counts, directories, results and
structures are not part of it, and the task count is not known at Build time.

## `phonon_forces` is not executable

`StageType.PHONON_FORCES` lets a workflow describe the phonon force stage as one
stage holding N displacement tasks. It is listed in
`NON_EXECUTABLE_STAGE_TYPES`. The calculation registry refuses it in
`validate_stage_spec` (diagnostic code `stage_type_not_executable`), before any
other check. That function guards Build, Prepare, Submit, input previews, the
input reference and POWER-side flow reconstruction. The stage type is not in
the public options catalogue, and there is no `phonons` Desired Output.

## `bmd_compute.stage_task_set` v1

`backend/stage_task_set.py` defines the authoritative record of the tasks
materialized for one `derived_task_set` stage:

| Field | Meaning |
| --- | --- |
| `schema`, `schema_version` | `"bmd_compute.stage_task_set"`, `1` |
| `parent.submission_attempt_id` | attempt whose immutable recipe holds the stage |
| `parent.workflow_sha256` | canonical hash of that attempt's `WorkflowSpec` |
| `parent.stage_index` | 1-based stage position (`stage_02` is index 2) |
| `parent.stage_type` | a `derived_task_set` stage type |
| `parent.stage_sha256` | canonical hash of the parent `StageSpec` (its methodology) |
| `upstream_structure.stage_index` | earlier stage whose output structure the source was derived from |
| `upstream_structure.sha256` | identity of that stage's actual output structure, as the materializer defines it (phonons: the M2b incoming Stage-1 identity, never the working-structure hash) |
| `source.materializer` | fixed by the parent stage type |
| `source.schema`, `schema_version`, `sha256` | the record that defines the tasks, by its own canonical hash |
| `task_order` | `"source_canonical"` |
| `tasks[]` | `{"task_id", "input_sha256"}` in the source's canonical order |
| `task_set_sha256` | canonical hash of every other field |

Canonical JSON and hashing reuse the M1 rules in `backend/phonons/records.py`.
The record holds no timestamps, hosts, absolute paths, directories or
scheduler state, so its identity depends only on scientific identity.

A task identifies itself, its parent stage (through the record) and its
scientific input (by hash). It carries no theory, modifiers, INCAR, KPOINTS,
resources, job ID, retry state or path. Methodology belongs to the parent
`StageSpec`, and its generated inputs are shown once for the stage, never per
task. A stage-level input that can only be fixed at runtime, such as a k-point
mesh for the materialized supercell, is future stage-level materialized input
provenance. It does not belong in task records.

### Canonical order is not execution order

`tasks` are listed in the source's canonical association order. For phonons
this is Phonopy dataset order, which later force assembly relies on. The order
is part of the record's identity: reordering is rejected. It is never an
execution order. Tasks may later run in any order, or concurrently.

### Validation

- **Structural:** `validate_stage_task_set_record` (also run by `StageTaskSet`)
  rejects the following:
  - wrong schema or version;
  - missing or extra keys at any level;
  - non-canonical data;
  - a malformed attempt ID or hash;
  - a single-calculation parent stage type;
  - a materializer or source schema that does not match the parent stage type;
  - an upstream stage that is not earlier than the parent;
  - an empty task list;
  - task IDs that are invalid, duplicated or out of canonical sequence;
  - duplicated task inputs;
  - a hash mismatch.

  A record that passes is internally consistent, not authoritative.
- **Parent re-verification:** `verify_stage_task_set_parent` requires that the
  attempt, the workflow hash, and the parent stage's type and methodology hash
  match. A task set therefore cannot be reused for another attempt, workflow or
  stage.
- **Source re-verification** is the materializer's job. For phonons,
  `backend/phonons/task_set.py::verify_phonon_force_task_set` requires that the
  bound `plan_sha256` match, and that the task IDs and displaced-structure
  hashes equal the M1 `DisplacementPlan`'s, in dataset order. The plan itself
  is verified against its structure by M1's `verify_displacement_plan`.

## Phonon binding to M1

The M1 `DisplacementPlan` remains the only authority for the displacement set:
displacement vectors, dataset indices, displaced structures, supercell,
primitive matrix, Phonopy policy and version. The task set refers to it by
`plan_sha256` and lists only `disp_NNN` IDs and displaced-structure hashes.

The upstream identity is the actual Stage-1 output, not the structure Phonopy
saw. Three identities, each answering one question, are bridged by the M2b
working-structure record (`phonon_working_structure.md`):

| Identity | Question |
| --- | --- |
| `task_set.upstream_structure.sha256` = `working.incoming_sha256` | what Stage 1 produced |
| `working.working_sha256` = `plan.working_structure.sha256` | what BMD handed to Phonopy |
| `task_set.source.sha256` = `plan.plan_sha256` | which displacement plan |

`build_phonon_force_task_set` takes the working-structure record, not a raw
hash, and requires the plan to have been built from exactly its working
structure. `verify_phonon_force_task_set_chain` checks every link from the
actual Stage-1 structure: parent stage, working-structure re-derivation,
upstream identity, M1 plan rebuild and task list.

## Task directories

Task directories are derived, never recorded. They are not stages:

```
stage_02/
    tasks/
        disp_001/
        disp_002/
```

`stage_task_dir(stage_root, task_id)` returns `<stage_root>/tasks/<task_id>`
for a validated task ID and a normalized absolute stage root, and cannot
escape `tasks/`. `paths.stage_dirs` keeps listing stage roots only. Directory
names are not scientific authority; the task set is. No directory is created in
M2a. The location of a persisted task-set file is decided when it is first
written (M2c).

## Requirements recorded for later milestones

- **M2b (done):** the phonon working-structure boundary; see
  `phonon_working_structure.md`.
- **M2c:** runtime materialization: write and re-verify the task set for an
  attempt on POWER, and decide its file location and submission integration.
- **M2d / M3: result discovery.** `backend/results.py` assumes the final stage
  is a VASP directory with `CONTCAR`, `OUTCAR` and `vasprun.xml`. Phonon
  workflows need stage-specific result discovery. Existing assumptions stay as
  they are until then.
- **M3: runtime parity.** Phonopy must become parity-critical before any remote
  phonon materialization or execution. It is recorded-only today.
- **M3: resources.** Concurrent displacement tasks cannot each inherit the
  full-allocation VASP command and resources without oversubscribing the
  allocation. Initial execution may run tasks sequentially. Resource
  partitioning is an execution concern and does not belong in task identity.
- **M3: barrier.** The analysis stage starts only when every task of the force
  stage is complete. Completion and forces will be execution records keyed by
  task ID; they do not belong in the task set.
