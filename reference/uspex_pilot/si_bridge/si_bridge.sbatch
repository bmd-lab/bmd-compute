#!/bin/bash
# USPEX bridge pilot (Phase 0, #59): Si8 at 0 GPa, 4 parallel relaxations of
# 12 ranks each, in a 48-core non-exclusive job. See README.md here.
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-si-pilot
#SBATCH -o si_pilot-%j.out

set -eu
: "${REPO:?set REPO to the bmd-compute checkout with the #66 bridge code}"
: "${WORK:?set WORK to a new directory under /bmd}"
: "${POTCAR_SI:?set POTCAR_SI to the PBE_64 Si POTCAR}"
PYTHON=${PYTHON:-python3}
IMAGE=${IMAGE:-/bmd/shared/uspex25_sandbox}
CACHE=${CACHE:-/bmd/shared/uspex_cache}

# Singularity's full path, found without loading uspex-25 into this shell
# (that module also activates conda, which would leak into the VASP steps).
SINGULARITY=$(bash -lc 'module load uspex-25 >/dev/null 2>&1; command -v singularity')

module load intel/rocky8-oneAPI-2023 vasp/rocky8-intel-6.4.1 >/dev/null 2>&1
# srun --mpi=pmi2 with Intel MPI needs SLURM's PMI2 library; the lab's
# benchmarks used mpirun, so this path is part of what the pilot tests.
for lib in /usr/lib64/libpmi2.so /usr/lib64/slurm/libpmi2.so; do
  if [ -e "$lib" ]; then export I_MPI_PMI_LIBRARY=$lib; break; fi
done
unset I_MPI_PIN_DOMAIN || true

echo "== node $(hostname)  SLURM $(srun --version)  python $($PYTHON --version 2>&1)"
echo "== singularity $SINGULARITY  image $IMAGE  I_MPI_PMI_LIBRARY=${I_MPI_PMI_LIBRARY:-unset}"
echo "== vasp_std $(command -v vasp_std)  start $(date -Is)"

"$PYTHON" "$REPO/reference/uspex_pilot/si_bridge/run_pilot.py" \
  --work "$WORK" --singularity "$SINGULARITY" --image "$IMAGE" --cache "$CACHE" \
  --potcar "$POTCAR_SI" --parallel 4 --ranks 12 --mem 12G --launch-mode srun_steps

echo "== end $(date -Is)"
