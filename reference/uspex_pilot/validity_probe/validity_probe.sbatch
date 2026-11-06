#!/bin/bash
# USPEX validity probe (integration step 2): how USPEX 25 treats a relaxation
# step the launch bridge reports as failed. Si8, 2 relaxations x 4 ranks in an
# 8-core job. See README.md for the full procedure.
#
# Required: PROBE_RUNTIME_SHA256 (the runtime_record_sha256 preflight.sh
# printed; the job refuses other runtime settings), REPO (the transferred source tree,
# /bmd/bmdguest/validation/bmd-compute-uspex-probe-<tag>), WORK (a new directory
# /bmd/bmdguest/validation/uspex_validity_probe_<tag>; run_validity_probe.py validates
# it fully), POTCAR_SI (the PBE_64 Si POTCAR). Optional: MODE=validity
# (default) or status_only. Everything stays under /bmd, which the production
# Singularity bind (-B /bmd) shows USPEX.
# Submit with an explicit log path next to WORK, for example
#   sbatch -o "$WORK.slurm-%j.out" validity_probe.sbatch
#
# Every setup step prints a line before it runs and stops with a message if
# it fails, so the job log always says how far the job got.
#SBATCH -p leeburton-pool
#SBATCH --account=power-leeburton-users_v2
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-validity-probe

say() { echo "== $(date +%T) $*"; }
die() { echo "!! $(date +%T) STOPPED: $*" >&2; exit 1; }

# The interpreter is fixed: the BMD Compute POWER runtime environment, which
# has pymatgen (needed by the validity check). It is not taken from the
# submitting shell.
PYTHON=/bmd/bmdguest/envs/atomate2_remote/bin/python
MODE=${MODE:-validity}

say "job ${SLURM_JOB_ID:-?} on $(hostname), account ${SLURM_JOB_ACCOUNT:-?}, partition ${SLURM_JOB_PARTITION:-?}, qos ${SLURM_JOB_QOS:-?}"
[ -n "${REPO:-}" ] || die "REPO is not set (the bmd-compute checkout under /bmd)"
[ -n "${WORK:-}" ] || die "WORK is not set (a new directory under /bmd)"
[ -n "${POTCAR_SI:-}" ] || die "POTCAR_SI is not set (the PBE_64 Si POTCAR)"
[ -n "${PROBE_RUNTIME_SHA256:-}" ] || die "PROBE_RUNTIME_SHA256 is not set (from preflight.sh)"
[ -f "$POTCAR_SI" ] || die "POTCAR_SI=$POTCAR_SI is not a file"
[ -f "$REPO/backend/structure_search/step_validity.py" ] || die "no step-validity code under REPO=$REPO"
[ ! -e "$WORK" ] && [ ! -L "$WORK" ] || die "WORK=$WORK already exists; choose a new directory"
case "$WORK" in /bmd/bmdguest/validation/uspex_validity_probe_*) ;; *) die "WORK must be /bmd/bmdguest/validation/uspex_validity_probe_<tag>";; esac
case "$REPO" in /bmd/bmdguest/validation/bmd-compute-uspex-probe-*) ;; *) die "REPO must be /bmd/bmdguest/validation/bmd-compute-uspex-probe-<tag>";; esac
case "$MODE" in validity|status_only) ;; *) die "MODE must be validity or status_only, not $MODE" ;; esac
[ -x "$PYTHON" ] || die "$PYTHON is not executable"
say "REPO=$REPO WORK=$WORK POTCAR_SI=$POTCAR_SI MODE=$MODE"
say "python: $PYTHON ($("$PYTHON" --version 2>&1))"
"$PYTHON" -c 'import pymatgen.io.vasp' || die "pymatgen is not importable with $PYTHON"

# Nothing from the submitting shell may redirect the probe into BMD Compute's
# workspace, jobflow store or USPEX settings: drop every BMD_* variable and the
# jobflow/atomate2 configuration variables. The USPEX settings then come from
# backend.config's defaults, the values structure searches use.
for name in $(env | sed -n 's/^\(BMD_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$name"; done
unset JOBFLOW_CONFIG_FILE ATOMATE2_CONFIG_FILE

say "making the module command available"
if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile >/dev/null 2>&1
fi
type module >/dev/null 2>&1 || die "the 'module' command is not available in the job"

say "loading the VASP modules (as BMD Compute does)"
module load intel/rocky8-oneAPI-2023 vasp/rocky8-intel-6.4.1 || die "module load of the VASP modules failed"
command -v vasp_std >/dev/null || die "vasp_std not on PATH after module load"
say "vasp_std: $(command -v vasp_std)"
unset I_MPI_PIN_DOMAIN
say "SLURM $(srun --version)"

say "starting the validity probe (MODE=$MODE)"
"$PYTHON" "$REPO/reference/uspex_pilot/validity_probe/run_validity_probe.py" \
  --work "$WORK" --potcar "$POTCAR_SI" --mode "$MODE" --runtime-sha256 "$PROBE_RUNTIME_SHA256"
status=$?
say "probe finished with status $status"
exit "$status"
