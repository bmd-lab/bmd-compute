#!/usr/bin/env bash
# Collect and independently check the POWER evidence of one staged search (USPEX step 4C).
#
# Run from the VM through the validation driver, which reads the attempt ID
# and run name from the validation's own ledger, checks them and quotes them:
#   python reference/uspex_pilot/staging_validation/run_staging_validation.py evidence ROOT
# (equivalent to: ssh powerslurm-bmdguest 'bash -s -- ATTEMPT_ID RUN_NAME' < power_evidence.sh)
#
# Read-only for the staged run: it only reads the run directory and the
# attempt state, and checks at the end that the run directory is unchanged.
# It writes one new evidence directory under /bmd-db/guest/validation/
# (refusing an existing one, never inside the run), never copies a POTCAR
# (only its identity and hashes), and submits or starts nothing.
set -uo pipefail
umask 077

if [ "$#" -ne 2 ]; then echo "usage: power_evidence.sh ATTEMPT_ID RUN_NAME (got $# arguments)"; exit 2; fi
ATTEMPT=$1
RUN_NAME=$2
[[ $ATTEMPT =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { echo "malformed attempt ID"; exit 2; }
[[ $RUN_NAME =~ ^uspex_search-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$ ]] || { echo "malformed run name"; exit 2; }
[ "${RUN_NAME: -8}" = "${ATTEMPT:0:8}" ] || { echo "run name does not belong to the attempt"; exit 2; }

# -- locations (POWER) --
WORKSPACE=/bmd/bmdguest/searches
STATE_DIR=/bmd-db/guest/logs/structure_search/submission_attempts
LIBRARY=/bmd-db/potcars/POT_PAW_PBE_64
PY=/bmd/bmdguest/envs/atomate2_remote/bin/python
EVIDENCE_PARENT=/bmd-db/guest/validation
# -- end of locations --
RUN=$WORKSPACE/$RUN_NAME
STATE=$STATE_DIR/$ATTEMPT.json
OUT=$EVIDENCE_PARENT/uspex_staging_4c_$(date -u +%Y%m%dT%H%M%SZ)_${ATTEMPT:0:8}

[ -d "$RUN" ] && [ ! -L "$RUN" ] || { echo "no run directory $RUN"; exit 2; }
[ -f "$STATE" ] && [ ! -L "$STATE" ] || { echo "no attempt state $STATE"; exit 2; }
case "$OUT/" in "$RUN"/*) echo "evidence would be inside the run directory"; exit 2;; esac
mkdir "$OUT" || { echo "cannot create $OUT (it must be new and its parent must exist)"; exit 2; }
exec > >(tee "$OUT/checks.txt") 2>&1
snapshot() { find "$RUN" -printf '%M %u:%g %s %T@ %P\n' | sort -k5; }
snapshot > "$OUT/snapshot_before.txt"

failures=0
check() {
    local what=$1; shift
    if "$@" >/dev/null 2>&1; then echo "PASS  $what"; else echo "FAIL  $what"; failures=$((failures + 1)); fi
}

echo "== USPEX 4C staging evidence for $ATTEMPT on $(hostname) at $(date -u +%FT%TZ) as $(id -un)"
echo "run directory: $RUN"
echo "evidence:      $OUT"

# 1. Inventory: every entry with mode and size; SHA-256 of every file (hashes only).
find "$RUN" -printf '%M %u:%g %10s %P\n' | sort -k4 > "$OUT/listing.txt"
(cd "$RUN" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$OUT/sha256.txt"

# 2. Copies of the non-licensed records (never a POTCAR).
mkdir "$OUT/files" "$OUT/files/search" "$OUT/files/search/Specific"
cp -p "$STATE" "$OUT/remote_state.json"
cp -p "$RUN/submission.json" "$OUT/files/"
for name in INPUT.txt prepared_inputs.json search_plan.json POTCAR.spec; do
    cp -p "$RUN/search/$name" "$OUT/files/search/"
done
cp -p "$RUN"/search/Specific/INCAR_* "$OUT/files/search/Specific/"

# 3. Independent checks (not using BMD's runtime package).
check "remote attempt state is STAGED for this attempt" "$PY" -I -c "
import json, sys; s = json.load(open('$STATE'))
sys.exit(not (s['state'] == 'STAGED' and s['attempt_id'] == '$ATTEMPT' and s['run_dir'] == '$RUN'))"
check "no claim left behind" test ! -e "${STATE%.json}.lock"
check "no hidden or temporary file anywhere in the run directory" \
    bash -c "[ -z \"\$(find '$RUN' -name '.*' ! -path '*/__pycache__/*')\" ]"
check "no symbolic link in the run directory" bash -c "[ -z \"\$(find '$RUN' -type l)\" ]"
"$PY" -I - "$RUN" "$LIBRARY" <<'PY'
import hashlib, json, os, stat, sys
from pymatgen.io.vasp.inputs import PotcarSingle

run, library = sys.argv[1], sys.argv[2]
search = os.path.join(run, "search")
plan = json.load(open(os.path.join(search, "search_plan.json")))
record = plan  # the plan document is the record plus its digest and provenance
manifest = json.load(open(os.path.join(search, "prepared_inputs.json")))
bridge = os.path.join(run, "backend", "structure_search", "bridge", "bmd_bridge.sh")
failures = 0

def check(what, ok):
    global failures
    print(("PASS  " if ok else "FAIL  ") + what)
    failures += 0 if ok else 1

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

check("bridge is a regular file", stat.S_ISREG(os.lstat(bridge).st_mode))
check("bridge SHA-256 equals the plan's", sha(bridge) == record["execution"]["bridge"]["sha256"])
check("manifest binds this bridge", manifest["bindings"] == {"bridge_path": bridge, "bound": True})
check("INPUT.txt names this bridge", bridge in open(os.path.join(search, "INPUT.txt")).read())
check("INPUT.txt is not left unbound", "UNBOUND-bridge-path" not in open(os.path.join(search, "INPUT.txt")).read())
for name, digest in manifest["files"].items():
    check(f"manifest hash of {name}", sha(os.path.join(search, name)) == digest)
for row in record["potcar"]["elements"]:
    staged = os.path.join(search, row["staged_at_execution_as"])
    source = os.path.join(library, row["symbol"], "POTCAR")
    potcar = PotcarSingle.from_file(staged)
    titel = potcar.keywords.get("TITEL", "").strip()
    enmax = float(potcar.keywords["ENMAX"])
    dataset = hashlib.sha256(str(potcar.data).encode("utf-8")).hexdigest()
    print(f"      {row['element']}: TITEL {titel!r} ENMAX {enmax} dataset sha256 {dataset} file sha256 {sha(staged)}")
    check(f"{row['element']} POTCAR TITEL matches the plan", titel == str(row["TITEL"]).strip())
    check(f"{row['element']} POTCAR ENMAX matches the plan", abs(enmax - float(row["ENMAX"])) <= 1e-6)
    check(f"{row['element']} POTCAR dataset SHA-256 matches the plan", dataset == row["dataset_sha256"])
    check(f"{row['element']} staged POTCAR is byte-identical to the POWER library", sha(staged) == sha(source))
    check(f"{row['element']} staged POTCAR is private (0600)", stat.S_IMODE(os.lstat(staged).st_mode) == 0o600)
sys.exit(1 if failures else 0)
PY
[ $? -eq 0 ] || failures=$((failures + 1))

# 4. Nothing was submitted or started for this run.
sacct -X -u bmdguest -S now-7days -o JobID,JobName%40,State,WorkDir%160 -P > "$OUT/sacct_7days.txt" 2>&1
squeue -u bmdguest -o '%i %j %T %Z' > "$OUT/squeue.txt" 2>&1
check "no SLURM job (last 7 days) ran in the run directory" bash -c "! grep -F '$RUN' '$OUT/sacct_7days.txt'"
check "no queued SLURM job uses the run directory" bash -c "! grep -F '$RUN' '$OUT/squeue.txt'"
check "no results or USPEX output in the search directory" \
    bash -c "[ -z \"\$(find '$RUN/search' -maxdepth 1 \( -name 'results*' -o -name 'CalcFold*' -o -name '*.out' \))\" ]"
pgrep -u bmdguest -af 'uspex|vasp' > "$OUT/processes.txt" 2>&1 || true
check "no USPEX or VASP process uses the run directory" bash -c "! grep -F '$RUN' '$OUT/processes.txt'"

snapshot > "$OUT/snapshot_after.txt"
check "the run directory is unchanged (modes, owners, sizes, modification times)" \
    cmp -s "$OUT/snapshot_before.txt" "$OUT/snapshot_after.txt"
check "the run directory's file hashes are unchanged" \
    bash -c "cd '$RUN' && sha256sum --quiet -c '$OUT/sha256.txt'"

(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name checks.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
if [ "$failures" -ne 0 ]; then
    echo "== EVIDENCE CHECKS FAILED ($failures). Evidence kept in $OUT"
    exit 1
fi
echo "== EVIDENCE CHECKS PASSED. Evidence kept in $OUT"
