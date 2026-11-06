# Validity probe: how USPEX 25 treats a failed relaxation step

USPEX integration step 2. **Run on POWER on 6 Nov 2026 (SLURM job
23406856, `MODE=validity`): USPEX 25.0 rejected both injected failures.**
Evidence and summary: [`results_23406856/`](results_23406856/RESULTS.md).
The optional `status_only` run has not been made.

## Question

The launch bridge reports a relaxation step that does not count as a fitness
evaluation as a failure (`backend/structure_search/step_validity.py`):

| Case | Status USPEX receives |
| --- | --- |
| VASP exited 0, a convergence or energy criterion failed | 118 |
| VASP exited 0, evidence missing or malformed | 119 |
| VASP failed | VASP's own status (here 7) |

In each case the launcher also withholds the files USPEX reads (`OUTCAR`,
`OSZICAR`, `CONTCAR`, `vasprun.xml`: archived under
`search/bmd_step_validity/<id>/`, renamed `<name>.bmd_invalid.<id>` in the
CalcFold). What USPEX 25.0 then does has never been observed. The probe
distinguishes:

- **rejection**: the candidate's chain stops at the failed step, it is not
  run again, and USPEX lists it without a usable energy (or not at all);
- **bounded retry**: USPEX runs the same candidate (same POSCAR, or same
  individual and step) again, and how often;
- **stale-energy reuse**: USPEX lists the candidate with an energy VASP
  produced earlier for it (an earlier step, or the withheld step);
- **search termination**: the USPEX master stops, crashes or stalls.

## What runs

- The real USPEX 25.0 program: `singularity exec --cleanenv -B /bmd --pwd
  <search> /bmd/shared/uspex25_sandbox /opt/uspex/uspex`, built by
  `backend/structure_search/bridge/master.py`. Sandbox, unpack cache, payload
  SHA-256 and Singularity path are `backend.config`'s structure-search
  values; the job-start runtime check (`backend/structure_search/runtime.py`)
  must pass before USPEX starts (`runtime_check.json`), and is also run on
  the login node beforehand (`preflight.sh`).
- The production launch bridge, launcher and validity check, unchanged. The
  only probe-specific piece is `inject_vasp.sh`, the launcher's command
  template, which wraps `srun --exact --mpi=pmi2 -n 4 --mem=8G vasp_std`.
- Si8, the si_bridge pilot's 3-step INCARs (pilot settings, not BMD policy),
  population 4, 2 generations: about 10–20 relaxations of 10–60 s, 2 at a
  time of 4 ranks.
- Python: `/bmd/bmdguest/envs/atomate2_remote/bin/python`, fixed in the job
  script (it has pymatgen; the runner refuses any other interpreter).

## Injections

Each is applied to exactly one step of one candidate per run. `mkdir` in
`probe_state/` claims it atomically, so concurrent steps cannot both take it
and a retry of the same candidate is not injected again.

| MODE | Injection | Expected bridge status |
| --- | --- | --- |
| `validity` (default) | first step-2 relaxation runs with `NELM = 3` (genuine electronic non-convergence; the INCAR is restored afterwards, the injected copy kept) | 118 |
| `validity` | first step-3 relaxation that VASP completes is reported as exit 7 | 7 |
| `status_only` (optional second run; validity check off) | step-3 exit 7 only, files left in place | 7 |

## Private USPEX cache

bmdguest can read the shared USPEX payload but cannot write in
`/bmd/shared/uspex_cache/uspex_sfx`, where the extractor creates a lock file
on every start. The probe (and only the probe) therefore uses
`/bmd/bmdguest/validation/uspex_cache_private` as `XDG_CACHE_HOME`: its
settings are production `SearchRuntimeSettings.from_config()` with only
`cache_dir` replaced; `master.py`, the bind and the production settings are
unchanged. No `BMD_*` variable is read: the three scripts clear them and the
runner refuses any that remain. The cache is filled once by
`init_private_cache.sh`, which checks the program's payload SHA-256 and
version banner and then runs the production `unpack_once` (the real
extractor writes its own `.ready`; `unpack_once` creates `uspex_sfx` inside
the private cache with mode 2775, nothing else changes). The preflight prints
`runtime_record_sha256`; the job is given it as `PROBE_RUNTIME_SHA256` and
refuses to run with any other settings.

## Isolation and safety

- Everything stays under `/bmd`, the only path the production Singularity
  bind (`-B /bmd`, unchanged) shows USPEX. The source tree is
  `/bmd/bmdguest/validation/bmd-compute-uspex-probe-<tag>` and the work
  directory a fresh `/bmd/bmdguest/validation/uspex_validity_probe_<tag>`,
  directly beneath `/bmd/bmdguest/validation` (bmdguest can write in
  `/bmd/bmdguest` on POWER's NFSv4 `/bmd`, but not in `/bmd/validation`). The runner (and `preflight.sh`, through
  `run_validity_probe.py --check-paths`) refuses `..`, `.`, `//`, trailing
  `/`, any symlinked component, an existing path, and anything in `/bmd-db`
  (including `/bmd-db/guest/flows`, `logs` and `validation`), `/bmd/shared`,
  and everything in `/bmd/bmdguest` except `/bmd/bmdguest/validation`
  (environments, tools and existing files stay protected). The work directory is made with an exclusive `mkdir`;
  `/bmd/bmdguest/validation` is used as it is, or made with a plain `mkdir`
  if missing. No permissions are changed. The job log goes
  to `$WORK.slurm-<jobid>.out`. Earlier pilots and validations are never
  touched.
- No jobflow, atomate2, MongoDB, BMD Compute workspace (`/bmd-db/guest`),
  submission ledger or API is used. The job drops every `BMD_*`,
  `JOBFLOW_CONFIG_FILE` and `ATOMATE2_CONFIG_FILE` variable, and the runner
  records (and fails on) any import of jobflow or atomate2.
- Reads only: the POTCAR (copied), the USPEX sandbox and cache, the Python
  environment. The USPEX extractor creates and removes its lock file in the
  shared cache, as every search does.
- Account `power-leeburton-users_v2`, partition `leeburton-pool`, QOS
  `owner100d`, 1 node, 8 tasks, 24 GB, 1:30:00, nodes compute-0-332/333
  excluded.

## Files

| File | Role |
| --- | --- |
| `init_private_cache.sh` | unpack USPEX once into the private cache (compute node, `srun`) |
| `preflight.sh` | read-only login-node checks before submission; prints the runtime record SHA-256 |
| `validity_probe.sbatch` | the job |
| `run_validity_probe.py` | writes the search, runs the runtime check, starts USPEX and the launcher |
| `inject_vasp.sh` | the two injections and the per-call log `probe_state/calls.tsv` |
| `analyze_probe.py` | the report after the job (`probe_report.json`) |

## Procedure

Run as `bmdguest` (ssh host alias `powerslurm-bmdguest`). Check first that
`/bmd/bmdguest/validation` is a real directory bmdguest can write to (it is
created with a plain `mkdir` if missing); the probe never changes permissions.
The exact commands, with the bundle's checksums, are given with each
transfer bundle. In outline:

1. copy the bundle to `/bmd/bmdguest/validation/` and unpack it into the new
   directory `/bmd/bmdguest/validation/bmd-compute-uspex-probe-<tag>`;
2. verify every file against `MANIFEST.sha256`;
3. run `preflight.sh` (read-only);
4. after approval, submit `validity_probe.sbatch` with
   `WORK=/bmd/bmdguest/validation/uspex_validity_probe_<tag>`;
5. run `analyze_probe.py "$WORK"`.

## Reading the result

| Observation in `probe_report.json` | Meaning | Next step |
| --- | --- | --- |
| Both injected candidates: chain stopped, not retried, not listed with a usable energy; master status 0, both generations listed | The bridge's failure report is sufficient | Proceed to search submission |
| `stale_energy_reuse` true for an injected candidate | USPEX ignored the failure | Blocker: signalling must change before real searches |
| `retried` true | USPEX retries failed steps | Record the retry bound; check every retry was also judged |
| Master status non-zero, a generation missing, or errors in the master log after an injection | Withheld files break USPEX | Blocker: find a failure form USPEX accepts |
| status_only run: failure honoured with files in place | Exit status alone is enough; withholding is a safety margin | Keep both |
| status_only run: energy listed despite status 7 | Withholding is required | Keep withholding (current design) |
