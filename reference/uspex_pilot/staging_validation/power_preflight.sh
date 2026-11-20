#!/usr/bin/env bash
# Read-only POWER preflight for the USPEX step 4C staging validation.
#
# Run from the VM through the validation driver:  python reference/uspex_pilot/staging_validation/run_staging_validation.py preflight ROOT
# (equivalent to: ssh powerslurm-bmdguest 'bash -s' < power_preflight.sh)
#
# Writes nothing, submits nothing, starts nothing. Exits non-zero if staging
# cannot succeed (it then must not be attempted).
set -uo pipefail

WORKSPACE=/bmd/bmdguest/searches
LOGS=/bmd-db/guest/logs
SEARCH_LOGS=$LOGS/structure_search
CACHE=/bmd/bmdguest/uspex_cache
PAYLOAD=fc4a08e65ffc5f4fc3b45cb058de2cd54731ab2c191123aac8a3c69e87e95c8c
READY=$CACHE/uspex_sfx/$PAYLOAD/.ready
POTCAR=/bmd-db/potcars/POT_PAW_PBE_64/Si/POTCAR
PY=/bmd/bmdguest/envs/atomate2_remote/bin/python

failures=0
check() {  # check "description" command...
    local what=$1; shift
    if "$@" >/dev/null 2>&1; then echo "PASS  $what"; else echo "FAIL  $what"; failures=$((failures + 1)); fi
}

echo "== USPEX 4C staging preflight on $(hostname) at $(date -u +%FT%TZ) as $(id -un)"
check "running as bmdguest" test "$(id -un)" = bmdguest
check "search workspace is a real directory: $WORKSPACE" bash -c "[ -d '$WORKSPACE' ] && [ ! -L '$WORKSPACE' ]"
check "search workspace is writable by bmdguest" test -w "$WORKSPACE"
check "logs directory is writable: $LOGS" test -w "$LOGS"
check "USPEX cache is initialised: $READY" test -e "$READY"
check "Si POTCAR is a readable regular file" bash -c "[ -f '$POTCAR' ] && [ ! -L '$POTCAR' ] && [ -r '$POTCAR' ]"
check "remote Python runs" test -x "$PY"
check "remote Python imports pymatgen" "$PY" -I -c "import pymatgen.io.vasp.inputs"

echo "-- for the record"
ls -ld "$WORKSPACE" "$CACHE" "$LOGS" 2>&1
ls -ld "$SEARCH_LOGS" 2>/dev/null || echo "$SEARCH_LOGS does not exist yet (staging creates it)"
echo "existing run directories in $WORKSPACE: $(find "$WORKSPACE" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
"$PY" -I -c "import pymatgen.core as p; print('pymatgen', p.__version__)" 2>&1
"$PY" -I - "$POTCAR" <<'PY' 2>&1
import hashlib, sys
from pymatgen.io.vasp.inputs import PotcarSingle
potcar = PotcarSingle.from_file(sys.argv[1])
print("Si POTCAR TITEL:", potcar.keywords.get("TITEL", "").strip(), "ENMAX:", potcar.keywords.get("ENMAX"),
      "dataset sha256:", hashlib.sha256(str(potcar.data).encode("utf-8")).hexdigest())
PY
echo "bmdguest jobs in the queue now: $(squeue -h -u bmdguest 2>/dev/null | wc -l)"

if [ "$failures" -ne 0 ]; then
    echo "== PREFLIGHT FAILED ($failures). Do not stage."
    exit 1
fi
echo "== PREFLIGHT PASSED"
