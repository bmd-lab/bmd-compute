#!/usr/bin/env bash
# Read-only inspection of the VASP step logs of one finished managed structure search on POWER:
# CPU-placement (Intel MPI pinning) diagnostics and each log's SHA-256, for comparison with the
# archived evidence (step_logs.json and the launcher's per-step records).
#
# Authorised for attempt 209898bc-2793-416c-a53b-061e28e5e87f / job 23624369: first run 17 Nov 2026
# (counts and 6 samples per log); this version also prints every log's complete Intel MPI pinning table
# and all its "MPI startup()" lines, for a second read-only run that needs its own authorisation.
# Run from the VM:
#   ssh -o BatchMode=yes powerslurm-bmdguest 'bash -s -- ATTEMPT_ID RUN_NAME JOB_ID' \
#       < power_step_log_inspection.sh > step-log-inspection-<attempt8>.txt
#
# It only reads. It writes nothing on POWER (no files, no directories) and prints to standard output:
# one JSON line per step log (name, size, SHA-256, counts of Intel MPI startup, version, pinning and
# SLURM cpu-bind lines, at most 6 sample lines; the parsed pinning table, rank, PID, host and CPUs, in
# either Intel MPI row format, at most 512 rows; every "MPI startup()" line, at most 64; lines at most
# 200 printable characters; never the rest of the log), one JSON line per problem, and a summary. It refuses to run while the job is queued, reads
# only regular files directly inside bridge_spool/logs (no links, no subdirectories), handles any file
# name (names are printed JSON-encoded), and checks at the end that the log directory did not change.
# Exit status: 0 complete, 1 incomplete (any log unreadable, skipped, or the directory changed),
# 2 refused (arguments, locations), 3 job still queued.
set -uo pipefail

if [ "$#" -ne 3 ]; then echo "REFUSED usage: power_step_log_inspection.sh ATTEMPT_ID RUN_NAME JOB_ID"; exit 2; fi
ATTEMPT=$1
RUN_NAME=$2
JOB=$3
[[ $ATTEMPT =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { echo "REFUSED malformed attempt ID"; exit 2; }
[[ $RUN_NAME =~ ^uspex_search-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$ ]] || { echo "REFUSED malformed run name"; exit 2; }
[ "${RUN_NAME: -8}" = "${ATTEMPT:0:8}" ] || { echo "REFUSED run name does not belong to the attempt"; exit 2; }
[[ $JOB =~ ^[0-9]{1,20}$ ]] || { echo "REFUSED malformed job ID"; exit 2; }

# -- locations (POWER) --
WORKSPACE=/bmd/bmdguest/searches
PY=/bmd/bmdguest/envs/atomate2_remote/bin/python
# -- end of locations --
RUN=$WORKSPACE/$RUN_NAME
LOGS=$RUN/bridge_spool/logs

[ -d "$RUN" ] && [ ! -L "$RUN" ] || { echo "REFUSED no run directory $RUN"; exit 2; }
[ -d "$LOGS" ] && [ ! -L "$LOGS" ] || { echo "REFUSED no step-log directory $LOGS"; exit 2; }
[ -x "$PY" ] || { echo "REFUSED no Python at $PY"; exit 2; }
QUEUED=$(env TZ=UTC SLURM_TIME_FORMAT=standard squeue -h -j "$JOB" -o '%i|%T' 2>/dev/null | grep -c "^$JOB|")
[ "${QUEUED:-0}" -eq 0 ] || { echo "REFUSED job $JOB is still in the queue"; exit 3; }

echo "== step-log inspection for $ATTEMPT (job $JOB) on $(hostname) at $(date -u +%FT%TZ) as $(id -un)"
"$PY" -I - "$LOGS" <<'PY'
import hashlib, json, os, re, stat, sys

logs = sys.argv[1]
# Both Intel MPI row formats: CPUs in braces ("{0,1}") or bare at the end of the line ("12"), as Intel MPI
# 2021.10 printed them on POWER under srun. The same pattern as power_execution_evidence.sh.
PIN = re.compile(rb"MPI startup\(\):\s+(\d+)\s+(\d+)\s+(\S+)\s+(?:\{([0-9,\-\s]+)\}|([0-9]+(?:[,\-][0-9]+)*))\s*$")
PIN_HEADER = re.compile(rb"MPI startup\(\):\s+Rank\s+Pid\s+Node name\s+Pin cpu")
MAX_PIN_ROWS, MAX_STARTUP_LINES = 512, 64
MPI_STARTUP = re.compile(rb"MPI startup\(\)")
MPI_VERSION = re.compile(rb"Intel\(R\) MPI Library[^\r\n]*")
CPU_BIND = re.compile(rb"cpu[-_]bind")
SAMPLE = re.compile(rb"MPI startup\(\)|Intel\(R\) MPI Library|cpu[-_]bind|I_MPI_|[Pp]in cpu")
MAX_SAMPLES, SAMPLE_CHARS, MIB = 6, 200, 1024 * 1024


def printable(line):
    text = line.decode("utf-8", "replace").strip()
    return "".join(c if c.isprintable() else "?" for c in text)[:SAMPLE_CHARS]


def emit(record):
    print(json.dumps(record, sort_keys=True, ensure_ascii=True), flush=True)


def state(directory):
    entries = {}
    with os.scandir(directory) as scan:
        for entry in scan:
            info = entry.stat(follow_symlinks=False)
            entries[entry.name] = (info.st_mode, info.st_size, info.st_mtime_ns)
    return entries


problems = 0
try:
    before = state(logs)
except OSError as exc:
    emit({"problem": "cannot list the step-log directory", "error": f"{type(exc).__name__}: {exc}"})
    print("== INSPECTION INCOMPLETE")
    sys.exit(1)
read = skipped = 0
for name in sorted(before):
    path = os.path.join(logs, name)
    mode = before[name][0]
    if not stat.S_ISREG(mode):
        kind = "symbolic link" if stat.S_ISLNK(mode) else "directory" if stat.S_ISDIR(mode) else "not a regular file"
        emit({"problem": "skipped", "name": name, "reason": kind})
        skipped += 1
        continue
    if not name.endswith(".log"):
        emit({"note": "ignored", "name": name, "reason": "not a .log file"})
        continue
    summary = {"log": name[:-4], "bytes": None, "sha256": None, "mpi_startup_lines": 0, "pin_rows": 0,
               "intel_mpi_version": None, "cpu_bind_lines": 0, "samples": [], "pin_header_lines": 0,
               "pin_table": [], "startup_lines": []}
    try:
        digest = hashlib.sha256()
        size = 0
        # Opened without following a link created since the listing, and checked to be the same file.
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != before[name][1]:
                raise OSError("the file changed between listing and reading")
            for line in handle:
                digest.update(line)
                size += len(line)
                if MPI_STARTUP.search(line):
                    summary["mpi_startup_lines"] += 1
                    if len(summary["startup_lines"]) < MAX_STARTUP_LINES:
                        summary["startup_lines"].append(printable(line))
                if PIN_HEADER.search(line):
                    summary["pin_header_lines"] += 1
                pin = PIN.search(line)
                if pin:
                    summary["pin_rows"] += 1
                    if len(summary["pin_table"]) < MAX_PIN_ROWS:
                        summary["pin_table"].append([int(pin.group(1)), int(pin.group(2)),
                                                     printable(pin.group(3)),
                                                     (pin.group(4) or pin.group(5)).decode().replace(" ", "")])
                if CPU_BIND.search(line):
                    summary["cpu_bind_lines"] += 1
                version = MPI_VERSION.search(line)
                if version and summary["intel_mpi_version"] is None:
                    summary["intel_mpi_version"] = printable(version.group(0))
                if SAMPLE.search(line) and len(summary["samples"]) < MAX_SAMPLES:
                    summary["samples"].append(printable(line))
        summary["bytes"], summary["sha256"] = size, digest.hexdigest()
        emit(summary)
        read += 1
    except (OSError, ValueError) as exc:
        emit({"problem": "unreadable", "name": name, "error": f"{type(exc).__name__}: {exc}"})
        problems += 1
try:
    unchanged = state(logs) == before
except OSError:
    unchanged = False
if not unchanged:
    emit({"problem": "the step-log directory changed during the inspection"})
    problems += 1
emit({"summary": {"entries": len(before), "logs_read": read, "unreadable_or_changed": problems,
                  "skipped": skipped, "directory_unchanged": unchanged}})
complete = problems == 0 and skipped == 0 and read > 0
print("== INSPECTION COMPLETE" if complete else "== INSPECTION INCOMPLETE")
sys.exit(0 if complete else 1)
PY
rc=$?
if [ "$rc" -gt 1 ]; then echo "== INSPECTION FAILED (Python exit $rc)"; exit 1; fi
exit "$rc"
