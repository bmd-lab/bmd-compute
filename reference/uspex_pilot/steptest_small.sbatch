#!/bin/bash
# Small version of steptest.sbatch: 8 cores and 16 GB instead of a whole
# exclusive node, so it starts quickly on a busy partition. It answers the
# same SLURM questions (do concurrent `srun --exact` steps run in parallel,
# does each need --mem, may a job call sbatch) but not 48-rank placement.
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -J steptest-small
#SBATCH -o steptest-small-%j.out

echo "== SLURM: $(srun --version)  node: $(hostname)"
echo "== A: 4 concurrent 2-task steps, no --mem  $(date +%T)"
for i in 1 2 3 4; do srun --exact -n 2 ./probe.sh > A_$i.log 2>&1 & done; wait
echo "== B: 4 concurrent 2-task steps, --mem=3G  $(date +%T)"
for i in 1 2 3 4; do srun --exact -n 2 --mem=3G ./probe.sh > B_$i.log 2>&1 & done; wait
echo "== C: sbatch from inside a job (--test-only)"
sbatch --test-only -p leeburton-pool -n 1 -t 00:01:00 --wrap hostname 2>&1
echo "== done $(date +%T)"
