#!/bin/bash
# init_private_cache.sh: unpack USPEX 25.0 once into the probe's private cache,
# /bmd/bmdguest/validation/uspex_cache_private, with BMD Compute's own
# mechanism (backend.structure_search.runtime.unpack_once): it checks the
# program's payload SHA-256 (and the probe checks the version banner), starts
# the real extractor with XDG_CACHE_HOME set to the private cache and waits for
# the extractor's own .ready marker, then runs the job-start runtime check.
# Nothing is fabricated; a partial unpack is refused, not overwritten.
#
# Run on a compute node, not the login node (the extractor unpacks about 2 GB):
#   REPO=... srun -p leeburton-pool --account=power-leeburton-users_v2 --qos=owner100d \
#     -N 1 -n 1 -c 2 --mem=8G -t 00:45:00 --exclude=compute-0-332,compute-0-333 \
#     bash "$REPO/reference/uspex_pilot/validity_probe/init_private_cache.sh"

set -u
PYTHON=/bmd/bmdguest/envs/atomate2_remote/bin/python
for name in $(env | sed -n 's/^\(BMD_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$name"; done
unset JOBFLOW_CONFIG_FILE ATOMATE2_CONFIG_FILE

[ -n "${REPO:-}" ] || { echo "REPO is not set" >&2; exit 1; }
case "$REPO" in /bmd/bmdguest/validation/bmd-compute-uspex-probe-*) ;; *) echo "REPO must be /bmd/bmdguest/validation/bmd-compute-uspex-probe-<tag>" >&2; exit 1;; esac
echo "== $(date +%T) unpacking USPEX into the private cache on $(hostname)"
"$PYTHON" "$REPO/reference/uspex_pilot/validity_probe/run_validity_probe.py" --init-cache
status=$?
echo "== $(date +%T) finished with status $status"
exit "$status"
