#!/bin/bash
# Phase 0 step test for the USPEX launch bridge (docs/structure_search.md, #59).
# A: four concurrent 48-task steps without --mem
# B: four concurrent 48-task steps with --mem=200G
# C: whether a job may call sbatch (--test-only submits nothing)
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 00:15:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J steptest
#SBATCH -o steptest-%j.out

echo "== SLURM: $(srun --version)  node: $(hostname)"

echo "== A: 4 concurrent steps, no --mem  $(date +%T)"
for i in 1 2 3 4; do
  srun --exact -n 48 ./probe.sh > A_$i.log 2>&1 &
done
wait

echo "== B: 4 concurrent steps, --mem=200G  $(date +%T)"
for i in 1 2 3 4; do
  srun --exact -n 48 --mem=200G ./probe.sh > B_$i.log 2>&1 &
done
wait

echo "== C: sbatch from inside the allocation (--test-only, submits nothing)"
sbatch --test-only -p leeburton-pool -n 1 -t 00:01:00 --wrap hostname 2>&1

echo "== done $(date +%T)"
