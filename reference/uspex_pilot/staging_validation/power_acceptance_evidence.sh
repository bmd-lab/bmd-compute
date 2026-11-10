#!/usr/bin/env bash
# Read-only POWER evidence for the controlled submission acceptance (USPEX step 4D, part D).
#
# Run from the VM through the validation driver, which reads the attempt ID,
# run name and job ID from the validation's own ledger, checks them against
# each other and quotes them:
#   python reference/uspex_pilot/staging_validation/run_staging_validation.py acceptance-evidence ROOT
# (equivalent to: ssh powerslurm-bmdguest 'bash -s -- ATTEMPT_ID RUN_NAME JOB_ID WALLTIME' < power_acceptance_evidence.sh)
#
# It only reads: the clocks, the SLURM configuration, the queue and accounting
# (in UTC and, for comparison, in POWER's local time), the remote attempt
# state, the run directory's listing, job_started and search_status.json.
# It never reads the SLURM output (only its size) or any POTCAR, writes one new
# evidence directory under /bmd-db/guest/validation/ (refusing an existing one,
# never inside the run), and submits, cancels or starts nothing.
set -uo pipefail
umask 077

if [ "$#" -ne 4 ]; then echo "usage: power_acceptance_evidence.sh ATTEMPT_ID RUN_NAME JOB_ID WALLTIME (got $# arguments)"; exit 2; fi
ATTEMPT=$1
RUN_NAME=$2
JOB=$3
WALLTIME=$4
[[ $ATTEMPT =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { echo "malformed attempt ID"; exit 2; }
[[ $RUN_NAME =~ ^uspex_search-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$ ]] || { echo "malformed run name"; exit 2; }
[ "${RUN_NAME: -8}" = "${ATTEMPT:0:8}" ] || { echo "run name does not belong to the attempt"; exit 2; }
[[ $JOB =~ ^[0-9]{1,20}$ ]] || { echo "malformed job ID"; exit 2; }
[[ $WALLTIME =~ ^[0-9]{2,4}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "malformed walltime"; exit 2; }

# -- locations (POWER) --
WORKSPACE=/bmd/bmdguest/searches
STATE_DIR=/bmd-db/guest/logs/structure_search/submission_attempts
PY=/bmd/bmdguest/envs/atomate2_remote/bin/python
EVIDENCE_PARENT=/bmd-db/guest/validation
# -- end of locations --
RUN=$WORKSPACE/$RUN_NAME
# The run directory as SLURM and the attempt records name it (the job's working directory). On POWER it is $RUN.
IDENTITY_RUN_DIR=/bmd/bmdguest/searches/$RUN_NAME
STATE=$STATE_DIR/$ATTEMPT.json
OUT=$EVIDENCE_PARENT/uspex_acceptance_4d_$(date -u +%Y%m%dT%H%M%SZ)_${ATTEMPT:0:8}_$$

[ -d "$RUN" ] && [ ! -L "$RUN" ] || { echo "no run directory $RUN"; exit 2; }
[ -f "$STATE" ] && [ ! -L "$STATE" ] || { echo "no attempt state $STATE"; exit 2; }
case "$OUT/" in "$RUN"/*) echo "evidence would be inside the run directory"; exit 2;; esac
mkdir "$OUT" || { echo "cannot create $OUT (it must be new and its parent must exist)"; exit 2; }
exec > >(tee "$OUT/checks.txt") 2>&1

echo "== USPEX 4D acceptance evidence for $ATTEMPT (job $JOB) on $(hostname) at $(date -u +%FT%TZ) as $(id -un)"
echo "run directory: $RUN"
echo "evidence:      $OUT"

# 1. Clocks: UTC (epoch and ISO) and POWER's local offset.
{ date -u +%s; date -u +%Y-%m-%dT%H:%M:%S; date +%z; } > "$OUT/clock.txt"
# 2. SLURM configuration that decides what accounting keeps.
scontrol show config 2>&1 | grep -i -E '^(AccountingStoreFlags|MinJobAge|SLURM_VERSION|ClusterName)' \
    > "$OUT/slurm_config.txt" || true
# 3. The queue and accounting, in UTC; the job's accounting also in local time, for comparison.
env TZ=UTC SLURM_TIME_FORMAT=standard squeue -h -u "$(id -un)" -o '%i|%j|%k|%Z|%T|%r|%V' \
    > "$OUT/squeue_utc.txt" 2> "$OUT/squeue.err"
echo $? > "$OUT/squeue.rc"
env TZ=UTC SLURM_TIME_FORMAT=standard sacct -X -n -P -j "$JOB" \
    -o JobID,JobName,Comment,WorkDir,State,ExitCode,Submit,Start,End > "$OUT/sacct_job_utc.txt" 2> "$OUT/sacct.err"
echo $? > "$OUT/sacct.rc"
env SLURM_TIME_FORMAT=standard sacct -X -n -P -j "$JOB" -o JobID,Submit > "$OUT/sacct_job_local.txt" 2>&1 || true
# The job's time limit as SLURM enforces it (queue and accounting).
squeue -h -j "$JOB" -o '%i|%l' > "$OUT/squeue_timelimit.txt" 2>/dev/null || true
sacct -X -n -P -j "$JOB" -o JobID,Timelimit > "$OUT/sacct_timelimit.txt" 2>&1 || true
env TZ=UTC SLURM_TIME_FORMAT=standard sacct -X -n -P -u "$(id -un)" -S now-7days \
    -o JobID,JobName,Comment,WorkDir,State > "$OUT/sacct_7days_utc.txt" 2>&1 || true
# 4. The attempt: remote state, run-directory listing, job_started, search_status.json, SLURM output size only.
cp -p "$STATE" "$OUT/remote_state.json"
find "$RUN" -maxdepth 1 -printf '%M %u:%g %10s %TY-%Tm-%TdT%TH:%TM %P\n' | sort -k5 > "$OUT/listing.txt"
for name in job_started search_status.json bmd_search_job.sbatch; do
    if [ -f "$RUN/$name" ] && [ ! -L "$RUN/$name" ]; then cp -p "$RUN/$name" "$OUT/$name"; fi
done
find "$RUN" -maxdepth 1 -name 'slurm-*.out' -printf '%s %f\n' > "$OUT/slurm_output_sizes.txt"
ls -1 "$WORKSPACE" | grep -F -- "-${ATTEMPT:0:8}" > "$OUT/run_directories_for_attempt.txt" || true

# 5. Checks (independent of BMD's code). RESULT lines are findings to report, not failures.
"$PY" -I - "$OUT" "$ATTEMPT" "$IDENTITY_RUN_DIR" "$JOB" "$WALLTIME" <<'PY'
import hashlib, json, os, re, sys
from datetime import datetime, timedelta, timezone

out, attempt, run, job, walltime = sys.argv[1:6]
comment, name = f"bmd_attempt:{attempt}", f"bmds-{attempt[:8]}"
failures = 0


def check(what, ok):
    global failures
    print(("PASS  " if ok else "FAIL  ") + what)
    failures += 0 if ok else 1


def result(what):
    print("RESULT " + what)


reviews = 0


def review(what):
    """A diagnostic that does not match expectations: recorded for review, not a failure."""

    global reviews
    print("REVIEW " + what)
    reviews += 1


def read(file):
    path = os.path.join(out, file)
    return open(path).read() if os.path.exists(path) else ""


def rows(file, width):
    return [line.split("|") for line in read(file).splitlines() if line.count("|") == width - 1]


def utc(text):
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


clock = read("clock.txt").split()
power_now = utc(clock[1]) if len(clock) == 3 else None
check("POWER's UTC clock is readable (epoch and ISO agree)",
      power_now is not None and abs(power_now.timestamp() - int(clock[0])) <= 1)
offset = clock[2] if len(clock) == 3 else "?"
result(f"POWER's local offset: {offset}")
flags = read("slurm_config.txt")
result("AccountingStoreFlags: " + (re.search(r"AccountingStoreFlags\s*=\s*(.*)", flags).group(1).strip()
                                   if re.search(r"AccountingStoreFlags\s*=", flags) else "not reported"))

state = json.load(open(os.path.join(out, "remote_state.json")))
output = json.loads(state.get("output") or "{}")
check("the remote state names this attempt and run directory",
      state.get("attempt_id") == attempt and state.get("run_dir") == run)
check("the remote state is SEARCH_SUBMITTED or SEARCH_CANCEL_REQUESTED",
      state.get("state") in ("SEARCH_SUBMITTED", "SEARCH_CANCEL_REQUESTED"))
submission = output.get("submission") or {}
check("the remote submission record names this job and identity",
      submission.get("job_id") == job and submission.get("comment") == comment
      and submission.get("job_name") == name and submission.get("workdir") == run.rstrip("/"))
print(f"      remote state: {state.get('state')}")


def seconds(text):
    """A SLURM duration ([D-]HH:MM:SS, MM:SS or M) in seconds; None if not a finite duration."""

    match = re.fullmatch(r"(?:([0-9]+)-)?(?:([0-9]+):)?([0-9]+):([0-9]+)", text.strip())
    if not match:
        return None
    days, hours, minutes, secs = (int(group or 0) for group in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + secs


limit = seconds(walltime)
script_path = os.path.join(out, "bmd_search_job.sbatch")
script = open(script_path, "rb").read() if os.path.exists(script_path) else b""
promoted = (output.get("promotion") or {}).get("job_script") or {}
check("the job script on POWER is the one the remote record names",
      bool(script) and hashlib.sha256(script).hexdigest() == promoted.get("sha256")
      == submission.get("job_script_sha256"))
times = [line.strip() for line in script.decode("utf-8", "replace").splitlines()
         if line.lstrip().startswith("#SBATCH") and "--time" in line]
check(f"the job script asks for exactly --time={walltime}", times == [f"#SBATCH --time={walltime}"])
queue_limits = [line.split("|")[1] for line in read("squeue_timelimit.txt").splitlines() if line.startswith(job + "|")]
account_limits = [line.split("|")[1] for line in read("sacct_timelimit.txt").splitlines() if line.startswith(job + "|")]
check("SLURM reports the job's time limit", bool(queue_limits or account_limits))
for source, limits in (("squeue", queue_limits), ("sacct", account_limits)):
    for value in limits:
        check(f"{source}: the time limit {value} is the acceptance walltime {walltime}", seconds(value) == limit)
result(f"time limit: squeue {queue_limits or '-'}, sacct {account_limits or '-'} (requested {walltime})")

check("squeue answered", read("squeue.rc").strip() == "0")
check("sacct answered", read("sacct.rc").strip() == "0")
queued = [r for r in rows("squeue_utc.txt", 7) if r[0] == job]
accounted = [r for r in rows("sacct_job_utc.txt", 9) if r[0] == job]
check("SLURM reports the job (queue or accounting)", bool(queued or accounted))
for source, records in (("squeue", queued), ("sacct", accounted)):
    for r in records:
        check(f"{source}: job name is {name}", r[1] == name)
        check(f"{source}: working directory is the run directory", r[3].rstrip("/") == run.rstrip("/"))
        check(f"{source}: no different comment", r[2] in ("", "(null)", comment))
result("comment while queued: " + ("job not queued now" if not queued else
                                   "present" if queued[0][2] == comment else "absent"))
result("comment in accounting: " + ("job not in accounting yet" if not accounted else
                                    "present" if accounted[0][2] == comment else "absent"))
if accounted:
    submit, start, end = utc(accounted[0][6]), utc(accounted[0][7]), utc(accounted[0][8])
    check("accounting submit time is ISO 8601 (UTC request honoured)", submit is not None)
    if submit is not None and power_now is not None:
        check("accounting submit time is not after POWER's UTC clock", submit <= power_now + timedelta(seconds=5))
    # Diagnostic only: the same submit time printed with TZ=UTC and in POWER's local time. The values are
    # recorded; a difference other than POWER's current offset (date +%z) is flagged for review, not failed
    # (the offset in force at submission may differ, and nothing is inferred from it).
    local = [line.split("|") for line in read("sacct_job_local.txt").splitlines() if line.startswith(job + "|")]
    local_submit = local[0][1] if local and len(local[0]) > 1 else None
    shown = utc(local_submit) if local_submit else None
    match = re.fullmatch(r"([+-])([0-9]{2})([0-9]{2})", offset)
    difference = (shown - submit).total_seconds() if shown is not None and submit is not None else None
    result(f"UTC/local diagnostic: submit {accounted[0][6]} with TZ=UTC, {local_submit or '-'} in local time, "
           f"difference {difference if difference is not None else '-'} s, POWER's offset now {offset}")
    expected = ((int(match.group(2)) * 3600 + int(match.group(3)) * 60) * (1 if match.group(1) == "+" else -1)
                if match else None)
    if difference is None or expected is None or difference != expected:
        review("the UTC/local difference is not POWER's current offset (or could not be read); review the values")
    result(f"accounting: state {accounted[0][4]}, exit {accounted[0][5]}, submit {accounted[0][6]}, "
           f"start {accounted[0][7]}, end {accounted[0][8]} (UTC)")
if queued:
    result(f"queue: state {queued[0][4]}, reason {queued[0][5]}, submit {queued[0][6]} (UTC)")

others = [r for r in rows("squeue_utc.txt", 7) + rows("sacct_7days_utc.txt", 5)
          if r[0] != job and (r[2] == comment or r[3].rstrip("/") == run.rstrip("/"))]
check("no other job carries this attempt's comment or run directory", not others)
check("exactly one run directory exists for this attempt",
      read("run_directories_for_attempt.txt").split() == [os.path.basename(run)])
started = read("job_started").strip()
if started:
    check("job_started names this job", started == job)
result("job_started: " + (started or "absent (the job has not started)"))
status = json.loads(read("search_status.json") or "null")
if status is not None:
    check("search_status.json names this attempt", status.get("attempt_id") == attempt)
    check("search_status.json names this job", str(status.get("slurm_job_id")) in (job, "None"))
    result(f"execution outcome (search_status.json): {(status.get('execution') or {}).get('outcome')}")
else:
    result("search_status.json: absent")
open(os.path.join(out, "review_count.txt"), "w").write(f"{reviews}\n")
sys.exit(1 if failures else 0)
PY
failures=$?

(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name checks.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
if [ "$failures" -ne 0 ]; then
    echo "== ACCEPTANCE EVIDENCE CHECKS FAILED. Evidence kept in $OUT"
    exit 1
fi
REVIEWS=$(cat "$OUT/review_count.txt" 2>/dev/null || echo "?")
echo "== ACCEPTANCE EVIDENCE CHECKS PASSED ($REVIEWS diagnostic(s) flagged for review). Evidence kept in $OUT"
