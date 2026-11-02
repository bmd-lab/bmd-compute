#!/bin/bash
# Run by every task of a test job step. Prints which step and task it is,
# when it started, and which CPUs SLURM allows it, then sleeps so that
# concurrent steps overlap in time.
cpus=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)
echo "step=$SLURM_STEP_ID task=$SLURM_PROCID start=$(date +%T) host=$(hostname) cpus=$cpus"
sleep 30
