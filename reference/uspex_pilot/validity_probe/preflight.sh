#!/bin/bash
# preflight.sh: read-only checks on a POWER login node before the validity
# probe is submitted. Starts no job, no USPEX and no VASP.
#
# Usage (same variables as validity_probe.sbatch, plus EXPECTED_MANIFEST_SHA256,
# the SHA-256 of the source tree's MANIFEST.sha256 given with the transfer):
#   REPO=... WORK=... POTCAR_SI=... EXPECTED_MANIFEST_SHA256=<sha> bash preflight.sh
#
# Exits non-zero at the first failed check. The only write is the runtime
# check's writability test of the shared USPEX cache (a temporary file, removed
# at once), the same test every structure search makes at job start.

set -u
PYTHON=/bmd/bmdguest/envs/atomate2_remote/bin/python
# Same as validity_probe.sbatch and init_private_cache.sh: no BMD_* variable
# from the calling shell may change the probe's settings.
for name in $(env | sed -n 's/^\(BMD_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$name"; done
unset JOBFLOW_CONFIG_FILE ATOMATE2_CONFIG_FILE
ok() { echo "ok   $*"; }
fail() { echo "FAIL $*" >&2; exit 1; }

[ -n "${REPO:-}" ] && [ -n "${WORK:-}" ] && [ -n "${POTCAR_SI:-}" ] && [ -n "${EXPECTED_MANIFEST_SHA256:-}" ] \
  || fail "set REPO, WORK, POTCAR_SI and EXPECTED_MANIFEST_SHA256"

# 1. The source tree is the reviewed one, unmodified: its manifest is the one
#    given with the transfer, and every file matches the manifest.
case "$REPO" in /bmd/bmdguest/validation/bmd-compute-uspex-probe-*) ;; *) fail "REPO must be /bmd/bmdguest/validation/bmd-compute-uspex-probe-<tag>";; esac
[ -f "$REPO/MANIFEST.sha256" ] || fail "REPO=$REPO has no MANIFEST.sha256"
manifest_sha=$(sha256sum "$REPO/MANIFEST.sha256" | cut -c1-64)
[ "$manifest_sha" = "$EXPECTED_MANIFEST_SHA256" ] || fail "MANIFEST.sha256 is $manifest_sha, expected $EXPECTED_MANIFEST_SHA256"
(cd "$REPO" && sha256sum --check --quiet --strict MANIFEST.sha256) || fail "files in REPO differ from MANIFEST.sha256"
ok "source tree $REPO matches MANIFEST.sha256 ($manifest_sha)"

# 2. Interpreter and libraries.
[ -x "$PYTHON" ] || fail "$PYTHON is not executable"
"$PYTHON" -c 'import pymatgen.io.vasp, pymatgen.core; print("     pymatgen", pymatgen.core.__version__)' \
  || fail "pymatgen is not importable with $PYTHON"
ok "python $PYTHON"

# 3. Work and source paths, checked by the runner's own validation (creates
#    nothing): WORK fresh and directly beneath /bmd/bmdguest/validation, named
#    uspex_validity_probe_*, no symlinked components, outside production areas;
#    /bmd/bmdguest/validation present and writable (or creatable) for this user.
"$PYTHON" "$REPO/reference/uspex_pilot/validity_probe/run_validity_probe.py" --check-paths --work "$WORK" \
  || fail "the work or source path is refused (see above)"
for log in "$WORK".slurm-*; do [ -e "$log" ] && fail "a job log for WORK already exists: $log"; done
ok "WORK=$WORK is acceptable and unused"

# 4. POTCAR is a PBE Si dataset.
grep -q 'TITEL *= *PAW_PBE Si ' "$POTCAR_SI" || fail "POTCAR_SI=$POTCAR_SI is not a PAW_PBE Si POTCAR"
ok "POTCAR $(grep -m1 TITEL "$POTCAR_SI" | sed 's/^ *//')"

# 5. The USPEX installation, with exactly the settings the job will use:
#    production settings with the private cache (run_validity_probe.py
#    --check-runtime): sandbox, payload SHA-256 and version banner, the
#    private cache's .ready and unpacked program, cache writability,
#    Singularity, PMI library, SLURM version. Prints the runtime record's
#    SHA-256, which the job must reproduce (PROBE_RUNTIME_SHA256).
"$PYTHON" "$REPO/reference/uspex_pilot/validity_probe/run_validity_probe.py" --check-runtime \
  || fail "USPEX runtime check failed (see the problems above); has the private cache been initialised?"
ok "USPEX runtime check passed with the private cache; pass the runtime_record_sha256 above to sbatch"

# 6. Account, partition and QOS association.
assoc=$(sacctmgr -nP show assoc user="$USER" account=power-leeburton-users_v2 format=account,partition,qos 2>/dev/null)
echo "$assoc" | sed 's/^/     /'
echo "$assoc" | grep -q 'power-leeburton-users_v2' || fail "$USER has no association with power-leeburton-users_v2"
echo "$assoc" | grep -q 'owner100d' || fail "the association does not list QOS owner100d"
ok "account power-leeburton-users_v2 with QOS owner100d"

# 7. The job script is syntactically valid and asks for the approved resources.
bash -n "$REPO/reference/uspex_pilot/validity_probe/validity_probe.sbatch" || fail "sbatch script syntax"
for line in '#SBATCH -p leeburton-pool' '#SBATCH --account=power-leeburton-users_v2' '#SBATCH --qos=owner100d' \
            '#SBATCH -n 8' '#SBATCH --mem=24G' '#SBATCH -t 01:30:00'; do
  grep -qxF "$line" "$REPO/reference/uspex_pilot/validity_probe/validity_probe.sbatch" || fail "sbatch script lacks: $line"
done
ok "sbatch script requests leeburton-pool, power-leeburton-users_v2, owner100d, 8 tasks, 24G, 1:30:00"
echo "preflight passed"
