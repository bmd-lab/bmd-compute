#!/bin/bash
# USPEX bridge pilot (Phase 0, #59): Si8 at 0 GPa through the launch bridge.
# Default: 2 relaxations at a time of 4 ranks each, in an 8-core job, so it
# starts quickly on a busy partition. For a bigger run, override both the
# allocation and the layout, for example
#   PARALLEL=4 RANKS=12 STEP_MEM=12G sbatch -n 48 --mem=64G ... si_bridge.sbatch
# See README.md here.
#
# Every setup step prints a line before it runs and stops with a message if
# it fails, so the job log always says how far the job got.
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-si-pilot
#SBATCH -o si_pilot-%j.out

say() { echo "== $(date +%T) $*"; }
die() { echo "!! $(date +%T) STOPPED: $*" >&2; exit 1; }

say "job ${SLURM_JOB_ID:-?} on $(hostname)"
[ -n "${REPO:-}" ] || die "REPO is not set (the bmd-compute checkout with the #66 bridge code)"
[ -n "${WORK:-}" ] || die "WORK is not set (a new directory under /bmd)"
[ -n "${POTCAR_SI:-}" ] || die "POTCAR_SI is not set (the PBE_64 Si POTCAR)"
[ -f "$POTCAR_SI" ] || die "POTCAR_SI=$POTCAR_SI is not a file"
[ -f "$REPO/backend/structure_search/bridge/launcher.py" ] || die "no bridge code under REPO=$REPO"
[ ! -e "$WORK" ] || die "WORK=$WORK already exists; choose a new directory"
PYTHON=${PYTHON:-/usr/bin/python3}
IMAGE=${IMAGE:-/bmd/shared/uspex25_sandbox}
CACHE=${CACHE:-/bmd/shared/uspex_cache}
# whichCluster -1: the probe job (#59, 4 Nov) showed USPEX 25 runs the
# command as "sh run_local.sh" in each CalcFold in this mode; in mode 0 it
# never ran the command at all.
WHICH_CLUSTER=${WHICH_CLUSTER:--1}
PARALLEL=${PARALLEL:-2}
RANKS=${RANKS:-4}
STEP_MEM=${STEP_MEM:-8G}
need=$((PARALLEL * RANKS))
[ "${SLURM_NTASKS:-$need}" -ge "$need" ] || die "PARALLEL x RANKS = $need is more than the job's ${SLURM_NTASKS} tasks"
say "REPO=$REPO WORK=$WORK POTCAR_SI=$POTCAR_SI WHICH_CLUSTER=$WHICH_CLUSTER PARALLEL=$PARALLEL RANKS=$RANKS STEP_MEM=$STEP_MEM"
say "python: $("$PYTHON" --version 2>&1)" ; "$PYTHON" --version >/dev/null 2>&1 || die "PYTHON=$PYTHON does not run"

say "making the module command available"
if ! type module >/dev/null 2>&1; then
  # A batch shell may not have the module function; /etc/profile defines it.
  # shellcheck disable=SC1091
  source /etc/profile >/dev/null 2>&1
fi
type module >/dev/null 2>&1 || die "the 'module' command is not available in the job"

say "finding singularity (without loading uspex-25 into this shell)"
if [ -z "${SINGULARITY:-}" ]; then
  # tail -1: module scripts may print messages to stdout before the path.
  SINGULARITY=$(bash -c 'source /etc/profile >/dev/null 2>&1; module load uspex-25 >/dev/null 2>&1; command -v singularity' | tail -1)
fi
[ -n "$SINGULARITY" ] && [ -x "$SINGULARITY" ] || die "singularity not found (got '$SINGULARITY'); set SINGULARITY to its full path"
say "singularity: $SINGULARITY"
[ -d "$IMAGE" ] || [ -f "$IMAGE" ] || die "IMAGE=$IMAGE not found"

say "loading the VASP modules"
module load intel/rocky8-oneAPI-2023 vasp/rocky8-intel-6.4.1 || die "module load of the VASP modules failed"
command -v vasp_std >/dev/null || die "vasp_std not on PATH after module load"
say "vasp_std: $(command -v vasp_std)"

# srun --mpi=pmi2 with Intel MPI needs SLURM's PMI2 library; the lab's
# benchmarks used mpirun, so this path is part of what the pilot tests.
for lib in /usr/lib64/libpmi2.so /usr/lib64/slurm/libpmi2.so; do
  if [ -e "$lib" ]; then export I_MPI_PMI_LIBRARY=$lib; break; fi
done
unset I_MPI_PIN_DOMAIN
say "SLURM $(srun --version)  I_MPI_PMI_LIBRARY=${I_MPI_PMI_LIBRARY:-unset}  srun mpi types: $(srun --mpi=list 2>&1 | tr '\n' ' ')"

say "starting the pilot"
"$PYTHON" "$REPO/reference/uspex_pilot/si_bridge/run_pilot.py" \
  --work "$WORK" --singularity "$SINGULARITY" --image "$IMAGE" --cache "$CACHE" \
  --potcar "$POTCAR_SI" --parallel "$PARALLEL" --ranks "$RANKS" --mem "$STEP_MEM" --launch-mode srun_steps --which-cluster "$WHICH_CLUSTER"
status=$?
say "pilot finished with status $status"
exit "$status"
