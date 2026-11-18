#!/usr/bin/env bash
# Read-only POWER evidence of one finished managed structure search (USPEX scientific execution acceptance).
#
# Run from the VM through the validation driver, which reads the attempt ID,
# run name and job ID from the validation's own ledger, checks them against
# each other and quotes them:
#   python reference/uspex_pilot/staging_validation/run_staging_validation.py execution-evidence ROOT
# (equivalent to: ssh powerslurm-bmdguest 'bash -s -- ATTEMPT_ID RUN_NAME JOB_ID' < power_execution_evidence.sh)
#
# It only reads. From the run directory it copies small text records from a
# fixed list (never a POTCAR, WAVECAR, CHGCAR, OUTCAR, vasprun.xml or any other
# VASP output), each with its size and SHA-256, and for every VASP step log it
# writes only a summary (its size, SHA-256, final free energy F, ionic-step
# count, Intel MPI pinning rows, error lines, and bounded CPU-placement
# diagnostics: counts of Intel MPI startup, version and SLURM cpu-bind lines and
# a few short sample lines, so that "Intel MPI printed nothing" can be told
# apart from "printed in a format the pinning pattern does not read"). It refuses to run while the
# job is still queued, and checks at the end that the run directory did not
# change. Everything goes into one new directory under /bmd-db/guest/validation
# (refusing an existing one, never inside the run). It submits, cancels or
# starts nothing.
set -uo pipefail
umask 077

if [ "$#" -ne 3 ]; then echo "usage: power_execution_evidence.sh ATTEMPT_ID RUN_NAME JOB_ID (got $# arguments)"; exit 2; fi
ATTEMPT=$1
RUN_NAME=$2
JOB=$3
[[ $ATTEMPT =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { echo "malformed attempt ID"; exit 2; }
[[ $RUN_NAME =~ ^uspex_search-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$ ]] || { echo "malformed run name"; exit 2; }
[ "${RUN_NAME: -8}" = "${ATTEMPT:0:8}" ] || { echo "run name does not belong to the attempt"; exit 2; }
[[ $JOB =~ ^[0-9]{1,20}$ ]] || { echo "malformed job ID"; exit 2; }

# -- locations (POWER) --
WORKSPACE=/bmd/bmdguest/searches
STATE_DIR=/bmd-db/guest/logs/structure_search/submission_attempts
PY=/bmd/bmdguest/envs/atomate2_remote/bin/python
EVIDENCE_PARENT=/bmd-db/guest/validation
# -- end of locations --
RUN=$WORKSPACE/$RUN_NAME
STATE=$STATE_DIR/$ATTEMPT.json
OUT=$EVIDENCE_PARENT/uspex_execution_4d_$(date -u +%Y%m%dT%H%M%SZ)_${ATTEMPT:0:8}_$$

[ -d "$RUN" ] && [ ! -L "$RUN" ] || { echo "no run directory $RUN"; exit 2; }
[ -f "$STATE" ] && [ ! -L "$STATE" ] || { echo "no attempt state $STATE"; exit 2; }
case "$OUT/" in "$RUN"/*) echo "evidence would be inside the run directory"; exit 2;; esac

# Post-run only: a job still in the queue may still be writing.
QUEUED=$(env TZ=UTC SLURM_TIME_FORMAT=standard squeue -h -j "$JOB" -o '%i|%T' 2>/dev/null | grep -c "^$JOB|")
[ "${QUEUED:-0}" -eq 0 ] || { echo "job $JOB is still in the queue; collect the evidence after it has ended"; exit 3; }

mkdir "$OUT" || { echo "cannot create $OUT (it must be new and its parent must exist)"; exit 2; }
exec > >(tee "$OUT/collector.txt") 2>&1
echo "== USPEX execution evidence for $ATTEMPT (job $JOB) on $(hostname) at $(date -u +%FT%TZ) as $(id -un)"
echo "run directory: $RUN"

snapshot() { find "$RUN" -printf '%M %s %T@ %P\n' | LC_ALL=C sort -k4; }
snapshot > "$OUT/snapshot_before.txt"

{ date -u +%s; date -u +%Y-%m-%dT%H:%M:%S; date +%z; } > "$OUT/clock.txt"
cp -p "$STATE" "$OUT/remote_state.json"
# The job and every job step, in UTC: state, exit code, times, CPUs and allocated resources.
env TZ=UTC SLURM_TIME_FORMAT=standard sacct -n -P -j "$JOB" \
    -o JobID,JobName,State,ExitCode,Start,End,Elapsed,NCPUS,AllocTRES,Timelimit > "$OUT/sacct_steps.txt" 2> "$OUT/sacct.err"
echo $? > "$OUT/sacct.rc"

"$PY" -I - "$RUN" "$OUT" "$JOB" <<'PY'
import hashlib, json, os, re, stat, sys

run, out, job = sys.argv[1:4]
files_dir = os.path.join(out, "files")
os.mkdir(files_dir)
manifest = {"copied": [], "summarised_logs": 0, "refused": [], "missing": []}
FORBIDDEN = re.compile(r"(^|/)(POTCAR[^/]*|WAVECAR|CHGCAR|CHG|OUTCAR[^/]*|vasprun\.xml[^/]*|PROCAR|DOSCAR|EIGENVAL|"
                       r"LOCPOT|ELFCAR|AECCAR\d)$")
MIB = 1024 * 1024


def regular(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    return info if stat.S_ISREG(info.st_mode) else False


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(relative, cap=8 * MIB, keep_tail=None):
    """Copy one allowlisted regular file (no links), bounded; a larger file keeps its head and tail."""

    if FORBIDDEN.search(relative):
        manifest["refused"].append(relative)
        return
    source = os.path.join(run, relative)
    info = regular(source)
    if info is None:
        manifest["missing"].append(relative)
        return
    if info is False:
        manifest["refused"].append(relative)
        return
    target = os.path.join(files_dir, relative)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    truncated = info.st_size > cap
    with open(source, "rb") as handle, open(target, "wb") as copy_handle:
        if not truncated:
            copy_handle.write(handle.read(cap + 1)[:cap])
        else:
            tail = keep_tail or cap // 2
            copy_handle.write(handle.read(cap - tail))
            copy_handle.write(b"\n[... truncated by the evidence collector ...]\n")
            handle.seek(max(0, info.st_size - tail))
            copy_handle.write(handle.read(tail))
    manifest["copied"].append({"path": relative, "bytes": info.st_size, "sha256": sha256(source),
                               "truncated": truncated})


for name in ("search_status.json", "runtime_check.json", "job_started", "bmd_search_job.sbatch",
             "resource_acknowledgement.json"):
    copy(name, cap=4 * MIB)
copy("uspex_master.log", cap=16 * MIB, keep_tail=12 * MIB)
copy(f"slurm-{job}.out", cap=4 * MIB, keep_tail=3 * MIB)
copy("job_setup.log", cap=4 * MIB, keep_tail=3 * MIB)  # the job shell's system/module set-up output
copy("bridge_spool/launcher.jsonl", cap=64 * MIB)
copy("bridge_spool/heartbeat", cap=4096)
for name in ("INPUT.txt", "search_plan.json", "prepared_inputs.json"):
    copy(f"search/{name}", cap=4 * MIB)


def listing(relative):
    path = os.path.join(run, relative)
    if not os.path.isdir(path) or os.path.islink(path):
        return []
    return sorted(os.listdir(path))


for name in listing("search/Specific"):
    if re.fullmatch(r"INCAR_[0-9]+", name):
        copy(f"search/Specific/{name}", cap=65536)
for name in listing("bridge_spool/done"):
    if name.endswith(".done"):
        copy(f"bridge_spool/done/{name}", cap=65536)
RESULT_FILES = ("Individuals", "BESTIndividuals", "Individuals_all", "Individuals_extended", "gatheredPOSCARS",
                "BESTgatheredPOSCARS", "Parameters.txt", "generation_properties")
for folder in listing("search"):
    if re.fullmatch(r"results[0-9]+", folder):
        for name in RESULT_FILES:
            if name in listing(f"search/{folder}"):
                copy(f"search/{folder}/{name}", cap=32 * MIB)
for calcfold in listing("search/Calculation"):
    # The relaxed structure of a surviving final step (a structure, not licensed): structural identity.
    names = ("bmd_step_validity.jsonl", "bmd_launch.jsonl") + (("CONTCAR",) if re.fullmatch(
        r"Calcfold_[0-9]+_5", calcfold) else ())
    for name in names:
        if name in listing(f"search/Calculation/{calcfold}"):
            copy(f"search/Calculation/{calcfold}/{name}", cap=1 * MIB if name == "CONTCAR" else 4 * MIB)
for record in listing("search/bmd_step_validity"):
    if "validity.json" in listing(f"search/bmd_step_validity/{record}"):
        copy(f"search/bmd_step_validity/{record}/validity.json", cap=4 * MIB)

# The launcher's scientific evidence archive (record, input POSCAR, output CONTCAR per step; text, small).
EVIDENCE = "search/bmd_evidence"
copy(f"{EVIDENCE}/index.jsonl", cap=16 * MIB)
for area in ("steps", "conflicts", "interrupted"):
    for entry in listing(f"{EVIDENCE}/{area}"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}", entry):
            manifest["refused"].append(f"{EVIDENCE}/{area}/{entry}")
            continue
        present = listing(f"{EVIDENCE}/{area}/{entry}")
        for name in ("record.json", "POSCAR", "CONTCAR"):
            if name in present:
                copy(f"{EVIDENCE}/{area}/{entry}/{name}", cap=1 * MIB)
manifest["evidence_staging"] = listing(f"{EVIDENCE}/.staging")

# VASP step logs: a summary only (the logs themselves stay on POWER).
F_LINE = re.compile(rb"^\s*\d+\s+F=\s*(\S+)\s+E0=\s*(\S+)")
# Intel MPI's pinning table: "Rank Pid Node name Pin cpu", then one row per rank. Two row formats: the CPU
# set in braces ("{0,1}") and, as Intel MPI 2021.10 printed it on POWER under srun (attempt 209898bc), a
# bare CPU or list ("12", "12,13", "12-13") ending the line.
PIN = re.compile(rb"MPI startup\(\):\s+(\d+)\s+(\d+)\s+(\S+)\s+(?:\{([0-9,\-\s]+)\}|([0-9]+(?:[,\-][0-9]+)*))\s*$")
PIN_HEADER = re.compile(rb"MPI startup\(\):\s+Rank\s+Pid\s+Node name\s+Pin cpu")
ERROR = re.compile(rb"(?i)(error|fatal|abort|killed|segmentation|out of memory|oom|cancelled)")
# CPU-placement diagnostics (counted; a few sample lines kept, short and printable only). Reading them
# changes nothing: binding, the srun command and VASP are untouched.
MPI_STARTUP = re.compile(rb"MPI startup\(\)")
MPI_VERSION = re.compile(rb"Intel\(R\) MPI Library[^\r\n]*")
CPU_BIND = re.compile(rb"cpu[-_]bind")
SAMPLE = re.compile(rb"MPI startup\(\)|Intel\(R\) MPI Library|cpu[-_]bind|I_MPI_|[Pp]in cpu")
MAX_SAMPLES, SAMPLE_CHARS = 8, 200


def printable(line):
    text = line.decode("utf-8", "replace").strip()
    return "".join(c if c.isprintable() else "?" for c in text)[:SAMPLE_CHARS]


logs = {}
for name in listing("bridge_spool/logs"):
    path = os.path.join(run, "bridge_spool", "logs", name)
    info = regular(path)
    if not name.endswith(".log") or not info:
        continue
    summary = {"bytes": info.st_size, "sha256": sha256(path), "ionic_steps": 0, "final_free_energy_ev": None,
               "final_e0_ev": None, "pin_rows": [], "error_lines": [], "mpi_startup_lines": 0,
               "intel_mpi_version": None, "cpu_bind_lines": 0, "placement_samples": [], "pin_header_lines": 0}
    with open(path, "rb") as handle:
        for line in handle:
            match = F_LINE.match(line)
            if match:
                summary["ionic_steps"] += 1
                for key, group in (("final_free_energy_ev", 1), ("final_e0_ev", 2)):
                    try:
                        summary[key] = float(match.group(group).decode().replace("D", "E"))
                    except ValueError:
                        summary[key] = None
                continue
            if MPI_STARTUP.search(line):
                summary["mpi_startup_lines"] += 1
            if CPU_BIND.search(line):
                summary["cpu_bind_lines"] += 1
            version = MPI_VERSION.search(line)
            if version and summary["intel_mpi_version"] is None:
                summary["intel_mpi_version"] = printable(version.group(0))
            if SAMPLE.search(line) and len(summary["placement_samples"]) < MAX_SAMPLES:
                summary["placement_samples"].append(printable(line))
            if PIN_HEADER.search(line):
                summary["pin_header_lines"] += 1
            pin = PIN.search(line)
            if pin and len(summary["pin_rows"]) < 512:
                summary["pin_rows"].append([int(pin.group(1)), int(pin.group(2)),
                                            pin.group(3).decode("utf-8", "replace"),
                                            (pin.group(4) or pin.group(5)).decode().replace(" ", "")])
            elif ERROR.search(line) and len(summary["error_lines"]) < 20:
                summary["error_lines"].append(line.decode("utf-8", "replace").strip()[:300])
    logs[name[:-4]] = summary
with open(os.path.join(out, "step_logs.json"), "w") as handle:
    json.dump(logs, handle, indent=1, sort_keys=True)
manifest["summarised_logs"] = len(logs)
manifest["calcfolds_present"] = listing("search/Calculation")
with open(os.path.join(out, "manifest.json"), "w") as handle:
    json.dump(manifest, handle, indent=1, sort_keys=True)
print(f"copied {len(manifest['copied'])} records, summarised {len(logs)} step logs, "
      f"refused {len(manifest['refused'])}, missing {len(manifest['missing'])}")
PY
rc=$?

snapshot > "$OUT/snapshot_after.txt"
if cmp -s "$OUT/snapshot_before.txt" "$OUT/snapshot_after.txt"; then
    echo "PASS  the run directory is unchanged (modes, sizes, modification times)"
else
    echo "FAIL  the run directory changed while the evidence was collected"
    rc=1
fi
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name collector.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "EVIDENCE DIRECTORY: $OUT"
[ "$rc" -eq 0 ] && echo "== EVIDENCE COLLECTED" || echo "== EVIDENCE COLLECTION FAILED"
exit "$rc"
