# USPEX steps 4C and 4D: staging, promotion and submission acceptance on POWER (one Si search)

Parts B and C validate managed remote staging and promotion end to end on
POWER for one Si structure search and **stop at remote `STAGED` /
`SEARCH_PREPARED`**: nothing is submitted, no `sbatch`, no USPEX, no VASP.
Part D is the controlled acceptance of submission, monitoring and
cancellation with a new attempt, behind two separate approval gates. Production workflows and the production ledger are not
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

## Validation record

Run on 7 Nov 2026 for attempt `5ead0670-911a-4c52-9587-9f3ae9539269`
(commits `2bbd1ba` + `b13f01f`): staging validated, nothing submitted.
Results, recovery, timings and evidence checksums:
[`RESULTS_20261107_5ead0670.md`](RESULTS_20261107_5ead0670.md).

Promotion (part C) run on 10 Nov 2026 for attempt
`14bfbbac-8599-48a6-bcc2-cfea917fd967` (commit `b7bd184`): stopped at
`SEARCH_PREPARED`, nothing submitted. Record:
[`RESULTS_20261110_14bfbbac.md`](RESULTS_20261110_14bfbbac.md).

Submission-and-cancellation acceptance (part D) run on 10 Nov 2026 for attempt
`c6edaf94-a2ab-4be7-a817-ce52f3d14852`, SLURM job `23576570` (commit
`364712a`, 15-minute allocation): submitted once, monitored, cancelled,
settled at `CANCELLED`; the job never started. Record:
[`RESULTS_20261110_c6edaf94.md`](RESULTS_20261110_c6edaf94.md).

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

## C. Promotion-only validation (step 4D-2; VM, bash)

Stops at remote `SEARCH_PREPARED`: the job script is uploaded and verified,
nothing is submitted (there is no submission route yet, and the kit never
runs `sbatch`, `srun`, USPEX or VASP). It needs a new attempt staged with the
promotion code: a new validation root, steps B0–B4 with the 4D-2 commit, then:

```bash
# C1. Acknowledge the attempt's provisional resource request and promote it.
#     It prints the #SBATCH lines being acknowledged. Stop on any 409.
"$BMD_PY" $K promote "$VROOT" 2>&1 | tee -a "$VROOT/run.log"

# C2. Independent, read-only POWER evidence: remote state SEARCH_PREPARED, the job
#     script and acknowledgement match the promotion record, bash can parse the
#     script, no job has started, and the run directory is unchanged.
"$BMD_PY" $K evidence "$VROOT"

# C3. The archive.
"$BMD_PY" $K archive "$VROOT"
```

Expected: `promote` prints the attempt, run name and job script SHA-256;
`EVIDENCE CHECKS PASSED` with `remote state:  SEARCH_PREPARED`. A 503 is
retried by running C1 again (it resumes the same promotion); any 409 is a
finding. The validation token has `submit` scope for promotion (and, since the
dedicated cancellation scope, `cancel` for part D); the kit calls no submission
route here. The staged and promoted run stays on POWER as evidence;
it can only be submitted by a later step, after a fresh verification.

## D. Controlled submission and cancellation acceptance (steps 4D-3 and 4D-4; VM, bash)

The first live managed submission. It uses a **new** Si attempt made by a new
validation root (never `5ead0670` or `14bfbbac`; the kit refuses them), runs
`sbatch` once and `scancel` at most for that one job, each only after its own
written approval. The kit never runs `sbatch` or `scancel` itself: it calls the
service's routes in-process with the validation's own ledger and token, and
the service runs them after its own fresh verification. Every kit step stops
(non-zero, `STOP:` or `VALIDATION FAILED:`) on an unexpected status, an
identity that is not verified, inconsistent times, a contradiction, an
uncertain submission or a broken evidence chain. On any stop: keep
everything, run nothing else, report.

The acceptance search asks for a **15-minute** allocation, not the default
168 hours. It is set once, at `setup --acceptance-walltime 00:15:00`, and
reaches the plan only through the planner's own Advanced walltime override
(`search.walltime` in the planning request). Nothing else in the plan
differs from the default Si plan: the INCAR ladder, POTCARs, validity rules
and USPEX settings are the same, and the runtime package is the commit's.
The walltime is therefore part of the plan digest, the acknowledged
resource request, the frozen job script (`#SBATCH --time=00:15:00`, its only
time directive) and, through the job script's SHA-256, the remote submission
record. `submit` refuses a root without it (the kit never submits a 168-hour
allocation) and, before the request, re-plans to confirm the digest is the
planner's for 15 minutes (and differs from the default's), and checks the
acknowledgement and the rebuilt job script; any mismatch stops with nothing
sent. `sbatch` runs with no options, so SLURM enforces 15 minutes whenever
the job starts; `acceptance-evidence` reads the job's time limit back from
`squeue` and `sacct`.

The reviewed commit reaches the VM as a Git bundle when it is not in the
VM's clones: compare the bundle's SHA-256 with the one reported, then make a
new clone from the bundle (cloning reads and checks the bundle; `git bundle
verify` works only inside an existing repository) and check out the exact
commit, detached, before anything else.

```bash
# D0. New acceptance root and attempt, with the reviewed commit: B0-B1, then this setup instead
#     of B2, then B3-B4 and C1-C3. Stop unless C2 ends with EVIDENCE CHECKS PASSED and
#     "remote state:  SEARCH_PREPARED", and C1 printed "#SBATCH --time=00:15:00".
VROOT=~/bmd-validation/uspex-4d-acceptance-$(date -u +%Y%m%dT%H%M%SZ)
env -u BMD_API_STATE_DIR -u BMD_API_TOKENS_FILE "$BMD_PY" $K setup "$VROOT" --acceptance-walltime 00:15:00
#     ... B3, B4 and C1-C3 with this VROOT ...
ATTEMPT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['attempt_id'])" "$VROOT/evidence/attempt.json")
echo "$ATTEMPT"

# D1. Read-only checks on POWER (as bmdguest): accounting configuration, UTC printing, the queue.
ssh powerslurm-bmdguest 'scontrol show config | grep -i -E "^(AccountingStoreFlags|MinJobAge)";
  date -u +%FT%T; date +%z;
  TZ=UTC SLURM_TIME_FORMAT=standard sacct -X -n -P -j 23406856 -o JobID,Submit,End;
  SLURM_TIME_FORMAT=standard sacct -X -n -P -j 23406856 -o JobID,Submit,End;
  sinfo -p leeburton-pool; squeue -u bmdguest' | tee "$VROOT/d1_readonly.txt"
#     Diagnostic only: record the UTC and local sacct times and the date +%z offset as printed. If the
#     difference is not the offset, flag it for review (do not assume a relationship); it is not a stop.
```

**Gate 1, submission.** Lee's written approval for one `sbatch` of `$ATTEMPT`
with a 15-minute allocation (the job script SHA-256 and `#SBATCH` lines are
those C1 printed and C2 verified). Without it, stop here.

```bash
# D2. The one submission. Verifies the 15-minute allocation (digest, acknowledgement, job script),
#     prints the job script SHA-256 and #SBATCH lines, then sends one request.
#     Expect a job ID. "STOP: UNCERTAIN SUBMISSION": do not repeat, do not run sbatch; report.
"$BMD_PY" $K submit "$VROOT" --confirm-attempt "$ATTEMPT" 2>&1 | tee -a "$VROOT/run.log"
JOB=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['job_id'])" "$VROOT/evidence/job.json")

# D3. Observe at once (queued or running): identity "verified", times "consistent",
#     binding "sbatch", scheduler source squeue. Repeat after 60 s if rate-limited.
"$BMD_PY" $K observe "$VROOT" 2>&1 | tee -a "$VROOT/run.log"

# D4. Independent read-only POWER evidence while the job is queued or running:
#     "RESULT comment while queued: present" is the comment-persistence check in the queue.
"$BMD_PY" $K acceptance-evidence "$VROOT"
```

**Gate 2, cancellation.** Lee's written approval for one `scancel` of `$JOB`
for `$ATTEMPT`. (If the job has already ended, skip D5 and go to D6.)

```bash
# D5. Cancellation: a fresh identity check by the service, then scancel <JOB>.
#     Expect "command_sent" (or "not_sent_job_not_active" if it ended first).
"$BMD_PY" $K cancel "$VROOT" --confirm-attempt "$ATTEMPT" --confirm-job "$JOB" 2>&1 | tee -a "$VROOT/run.log"

# D6. Observe until settled (at least 120 s after the job ended): expect SEARCH_CANCELLED with
#     cancellation "observed" (or SLURM's own terminal state if the job ended first).
"$BMD_PY" $K observe "$VROOT" 2>&1 | tee -a "$VROOT/run.log"

# D7. Independent read-only POWER evidence after the end: "RESULT comment in accounting:
#     present|absent" is the comment-persistence check in accounting; "RESULT UTC/local diagnostic"
#     records the submit time in UTC and in local time with POWER's offset (a mismatch is a REVIEW
#     line to report, not a failure).
"$BMD_PY" $K acceptance-evidence "$VROOT"

# D8. The archive (evidence, observation evidence chain, POWER reports; never a token or POTCAR).
"$BMD_PY" $K archive "$VROOT"
```

Expected: `submit` prints "Short allocation verified: 00:15:00 ..." and one job ID; the
acceptance-evidence reports show the time limit `00:15:00` (`15:00` in `squeue`); every `observe` shows identity
`verified` (`partial` only once the job has left the queue and accounting
keeps no comment), times `consistent` and binding `sbatch`; `cancel` shows
`command_sent` with a verified identity; the final observation shows
`SEARCH_CANCELLED` (or SLURM's terminal state); both acceptance-evidence
reports end with `ACCEPTANCE EVIDENCE CHECKS PASSED`.

Stop conditions (each is a finding): any `STOP:`/`VALIDATION FAILED:` line;
any HTTP status other than 200; `submission_uncertain`/`unverified`/
`ambiguous`/`conflict`; identity `contradicted`, `unverified` or
`not_reported` (after a wait for accounting); `operator_review`; a `FAIL` in
an acceptance-evidence report; more than one job or run directory for the
attempt. Nothing is reset or removed; the run directory and the job's
records stay on POWER as evidence.

Report back: the commit, attempt ID, job ID, the `submit`, `observe` and
`cancel` outputs, both acceptance-evidence reports (in particular the RESULT
lines on comment persistence, the UTC/local diagnostic and `AccountingStoreFlags`, and any
`REVIEW` lines), and the
archive's SHA-256.

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
