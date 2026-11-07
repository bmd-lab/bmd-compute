# USPEX step 4C: staging-only POWER validation (one Si search)

Validates managed remote staging end to end on POWER for one Si structure
search and **stops at remote `STAGED`**. Nothing is submitted: no `sbatch`,
no USPEX, no VASP. Production workflows and the production ledger are not
used: the run has its own state directory, token and principal
(`uspex-4c-validation`), and drives the machine API in-process through its
authenticated router.

What it proves:

- planning, local preparation and staging through the authenticated API;
- the frozen plan (digest checked on POWER by the uploaded code);
- the runtime package uploaded with the attempt, verified by the guard, and
  the bridge bound from it (path and SHA-256 equal to the plan);
- the real Si POTCAR copied from `/bmd-db/potcars` into the private search
  directory, with TITEL, ENMAX and dataset SHA-256 equal to the plan, mode
  0600, and no POTCAR text in any response or evidence file;
- read-back without contacting POWER, and a fresh read-only re-verification
  on POWER (the check step 4D will require);
- independent checks on POWER that do not use BMD's code, and that no SLURM
  job, USPEX or VASP process touched the run directory.

Locations used on POWER (all as `bmdguest`):

| Path | Use |
| --- | --- |
| `/bmd/bmdguest/searches/uspex_search-<UTC>-<attempt[:8]>/` | the staged run (new; kept) |
| `/bmd-db/guest/logs/structure_search/submission_attempts/<attempt>.json` | remote attempt state (kept) |
| `/bmd/bmdguest/uspex_cache/` | read only (its ready marker must exist) |
| `/bmd-db/potcars/POT_PAW_PBE_64/Si/POTCAR` | read only |
| `/bmd-db/guest/validation/uspex_staging_4c_<UTC>_<attempt[:8]>/` | evidence written by `power_evidence.sh` (new) |

## Preconditions (stop if any is not met)

1. Step 4C is committed, and that commit is available to the VM's clone (how
   it gets there is Lee's decision).
2. The production USPEX cache `/bmd/bmdguest/uspex_cache` is initialised
   (part A). `preflight` checks its ready marker.

## A. Initialise the production USPEX cache (once)

This is environment setup, separate from the validation. It uses BMD
Compute's own mechanism (`python -m backend.structure_search.runtime unpack`,
`runtime.unpack_once`): it checks the program's payload SHA-256, starts the
USPEX self-extractor once in an empty folder inside the cache, on a compute
node through `srun`, and stops it as soon as the extractor's own `.ready`
marker appears. It does not run a search, VASP or a batch job, and does
nothing if `.ready` already exists. It refuses a partial unpack instead of
overwriting it.

On the VM (bash), put exactly the committed `backend/` on POWER, in the
validation area:

```bash
cd ~/bmd-4c-validation          # the separate clone of the commit (step B1 below creates it)
CODE=/bmd/bmdguest/validation/bmd-compute-4c-$(git rev-parse --short=12 HEAD)
git archive --format=tar HEAD backend | ssh powerslurm-bmdguest "mkdir '$CODE' && tar -x -C '$CODE'"
echo "$CODE"
```

On POWER (bash, as `bmdguest`, login node):

```bash
CODE=/bmd/bmdguest/validation/bmd-compute-4c-<12-character commit>
PY=/bmd/bmdguest/envs/atomate2_remote/bin/python
CACHE=/bmd/bmdguest/uspex_cache
ls -ld "$CACHE" || mkdir "$CACHE"        # must exist, be a real directory and writable by bmdguest
# The settings this commit uses, with no BMD_ override from the shell:
bash -c 'for n in $(env | sed -n "s/^\(BMD_[A-Za-z0-9_]*\)=.*/\1/p"); do unset "$n"; done
         cd "$0" && "$1" -E -m backend.structure_search.runtime record' "$CODE" "$PY" \
  | grep -E '"(cache_dir|payload_sha256|ready_marker|sandbox_dir)"'
#   expect cache_dir /bmd/bmdguest/uspex_cache and payload_sha256 fc4a08e6...e87e95c8c
srun -p leeburton-pool --account=power-leeburton-users_v2 --qos=owner100d -N 1 -n 1 -c 2 --mem=8G \
  -t 00:45:00 --exclude=compute-0-332,compute-0-333 \
  bash -c 'for n in $(env | sed -n "s/^\(BMD_[A-Za-z0-9_]*\)=.*/\1/p"); do unset "$n"; done
           cd "$0" && "$1" -E -m backend.structure_search.runtime unpack' "$CODE" "$PY"
ls -l "$CACHE/uspex_sfx/fc4a08e65ffc5f4fc3b45cb058de2cd54731ab2c191123aac8a3c69e87e95c8c/.ready"
ls -ld "$CACHE" "$CACHE/uspex_sfx"
```

Expected: `{"status": "unpacked", ...}` (or `"already_unpacked"`) and the
`.ready` file listed. If it reports a partial unpack in the way, stop and
report it; do not remove anything without checking that no other unpack is
running. `unpack` creates `uspex_sfx/` with mode 2775 (the documented
cache layout).

## B. Staging-only validation (VM, bash)

Run as the user that runs the live service (the one whose SSH config has
`powerslurm-bmdguest`). Do not stop or restart the live service; nothing
below touches it. Every step goes through the driver, which refuses to run
unless the validation root is its own (made by `setup`, private, holding
only this validation's attempt and token), no `BMD_API_*` variable is set,
and no running BMD Compute service uses the same state or token paths. It
never uses the shell's submission identity secret.

```bash
# B0. The live service's checkout and interpreter (read only), and the commit.
LIVE_PID=$(pgrep -f 'uvicorn main:app' | head -1)
BMD_REPO=$(readlink -f /proc/$LIVE_PID/cwd)
BMD_PY=$(readlink -f /proc/$LIVE_PID/exe)
COMMIT=<the step 4C commit>
"$BMD_PY" -c "import httpx, paramiko, starlette.testclient; print('client libraries present')"
# If that import fails, stop and report it: do not install anything into the service environment.

# B1. A separate clone of exactly that commit (the live checkout is not changed).
git clone --quiet --no-hardlinks "$BMD_REPO" ~/bmd-4c-validation
git -C ~/bmd-4c-validation checkout --quiet --detach "$COMMIT"
cd ~/bmd-4c-validation && git log -1 --oneline
K=reference/uspex_pilot/staging_validation/run_staging_validation.py

# B2. A new private validation root, outside every repository.
mkdir -p -m 700 ~/bmd-validation
VROOT=~/bmd-validation/uspex-4c-$(date -u +%Y%m%dT%H%M%SZ)
env -u BMD_API_STATE_DIR -u BMD_API_TOKENS_FILE "$BMD_PY" $K setup "$VROOT"

# B3. Read-only POWER preflight. Stop unless it ends with "PREFLIGHT PASSED".
"$BMD_PY" $K preflight "$VROOT"

# B4. Plan, prepare, stage, read back, freshly re-verify. Ends at STAGED.
"$BMD_PY" $K run "$VROOT" 2>&1 | tee "$VROOT/run.log"

# B5. Independent, read-only POWER evidence (attempt ID and run name come from the
#     validation's own ledger, are checked against each other and are quoted for SSH).
"$BMD_PY" $K evidence "$VROOT"

# B6. The archive (never the token, the token file, a POTCAR or POTCAR text).
"$BMD_PY" $K archive "$VROOT"
```

Expected: `PREFLIGHT PASSED`; `stage: HTTP 200` and `fresh
re-verification on POWER: passed`; `EVIDENCE CHECKS PASSED`, including
"the run directory is unchanged"; and the archive's SHA-256.

If B4 reports HTTP 503 (a transport interruption), run B4 again with
`tee -a`: it resumes the same attempt in the same run directory, which is
itself part of the validation. Any 409, or any `FAIL` line, is a finding:
stop, keep everything, and report the code and the evidence. Never remove
or edit anything in the run directory or the attempt state. If a request
died while holding the claim (`staging_in_progress` on every retry), report
it before removing anything. Outputs are never overwritten: `preflight`,
`evidence` and `archive` refuse to run twice into the same root.

## What the evidence contains

- VM, `$VROOT.tgz` (mode 0600): `evidence/` (every request without its
  headers, every response, the redacted local record, the summary,
  SHA256SUMS), the local preparation (`POTCAR.spec` lists symbols only),
  `run.log`, `power_preflight.txt`, `power_evidence.txt`. Not included:
  `client_token`, `tokens.json`, the raw ledger records. Every member is
  checked for the token, its secret, the verifier, private keys,
  `Authorization`/`Bearer` text and POTCAR dataset text before writing.
- POWER, the new `/bmd-db/guest/validation/uspex_staging_4c_<UTC>_<id>/`
  (mode 0700): listing, SHA-256 of every staged file (hashes only, including
  the POTCAR's), before/after snapshots of the run directory, the remote
  state, copies of `submission.json`, the bound `INPUT.txt`, the manifest,
  the plan, `POTCAR.spec` and the INCARs, SLURM and process listings, and
  `checks.txt`. No POTCAR is copied.

## Report back

- the commit, attempt ID, run name and plan digest (from the summary);
- the last lines of `run.log`, `power_preflight.txt` and
  `power_evidence.txt`;
- the archive's SHA-256 and the POWER evidence directory's path.

Nothing is cleaned up automatically. The staged run stays in the search
workspace as evidence; it is `STAGED` and cannot be submitted by this
release.
