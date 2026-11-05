#!/bin/bash
# USPEX restart test (#59, #69): how does USPEX 25 continue a search when it
# is rerun in the same folder after being stopped mid-search?
#
#   Phase A  start the Si8 pilot search; after STOP_AFTER_GEN generations are
#            complete (and WAIT_AFTER seconds into the next one, so some
#            relaxations are running), send SIGTERM, as SLURM does at the end
#            of an allocation. The pilot stops the USPEX master and its VASP
#            steps.
#   Phase B  rerun USPEX in the same search folder with a fresh bridge spool
#            (run_pilot.py --resume r1) and let it finish.
#   Report   which results*/ folders exist, whether individual numbers and
#            generations continue, and how the resumed run ended.
#
# Same defaults as si_bridge.sbatch (8-core job, 2 relaxations x 4 ranks).
#
# Every setup step prints a line before it runs and stops with a message if
# it fails, so the job log always says how far the job got.
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-restart
#SBATCH -o uspex_restart-%j.out

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
STOP_AFTER_GEN=${STOP_AFTER_GEN:-2}
WAIT_AFTER=${WAIT_AFTER:-60}
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

PILOT=(--work "$WORK" --singularity "$SINGULARITY" --image "$IMAGE" --cache "$CACHE"
  --potcar "$POTCAR_SI" --parallel "$PARALLEL" --ranks "$RANKS" --mem "$STEP_MEM"
  --launch-mode srun_steps --which-cluster "$WHICH_CLUSTER")

completed_generations() { # data rows in results1/generation_properties
  awk 'NR > 1 && $1 ~ /^[0-9]+$/ {n++} END {print n + 0}' "$WORK/search/results1/generation_properties" 2>/dev/null || echo 0
}

say "phase A: starting the search; will stop it after $STOP_AFTER_GEN generation(s) + ${WAIT_AFTER}s"
"$PYTHON" "$REPO/reference/uspex_pilot/si_bridge/run_pilot.py" "${PILOT[@]}" > "$WORK.phaseA.out" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if [ "$(completed_generations)" -ge "$STOP_AFTER_GEN" ]; then
    say "phase A: $(completed_generations) generation(s) complete; waiting ${WAIT_AFTER}s, then SIGTERM"
    sleep "$WAIT_AFTER"
    kill -TERM "$pid" 2>/dev/null
    break
  fi
  sleep 10
done
wait "$pid"; statusA=$?
say "phase A ended with status $statusA (123 = stopped as intended)"
mv "$WORK.phaseA.out" "$WORK/phaseA.out" 2>/dev/null
say "after phase A: results folders: $(cd "$WORK/search" && ls -d results* 2>/dev/null | tr '\n' ' ')"
echo "   phase A relaxation statuses (123 = stopped by the SIGTERM): $(cat "$WORK"/spool/done/* 2>/dev/null | grep status= | sort | uniq -c | tr '\n' ' ')"
cp "$WORK/search/results1/Individuals" "$WORK/Individuals_after_phaseA" 2>/dev/null
tail -3 "$WORK/search/results1/Individuals" 2>/dev/null | sed 's/^/   /'

say "phase B: rerunning USPEX in the same folder (fresh spool spool_r1)"
"$PYTHON" "$REPO/reference/uspex_pilot/si_bridge/run_pilot.py" "${PILOT[@]}" --resume r1
statusB=$?
say "phase B ended with status $statusB"

say "report"
cd "$WORK/search" || die "no search folder"
echo "   results folders: $(ls -d results* | tr '\n' ' ')"
for r in results*; do
  echo "   --- $r: $(awk 'NR > 1' "$r/Individuals" 2>/dev/null | wc -l) individuals"
  echo "   first rows:"; head -3 "$r/Individuals" 2>/dev/null | sed 's/^/     /'
  echo "   last rows:";  tail -3 "$r/Individuals" 2>/dev/null | sed 's/^/     /'
  echo "   generation_properties:"; sed 's/^/     /' "$r/generation_properties" 2>/dev/null
  echo "   best_structure:"; sed 's/^/     /' "$r/best_structure" 2>/dev/null; echo
done
echo "   relaxations: phase A $(ls "$WORK/spool/done" | wc -l), phase B $(ls "$WORK/spool_r1/done" 2>/dev/null | wc -l); statuses B: $(cat "$WORK"/spool_r1/done/* 2>/dev/null | grep status= | sort | uniq -c | tr '\n' ' ')"
echo "   'Task failed' lines: phase A $(grep -c 'Task failed' "$WORK/uspex_master.log"), phase B $(grep -c 'Task failed' "$WORK/uspex_master_r1.log")"
echo "   resumed master log, lines about restart/continuation:"
grep -i -E "restart|resum|continu|pickup|pick up|load|generation [0-9]+ started|results[0-9]|done" "$WORK/uspex_master_r1.log" | grep -v -i "Loading may take" | head -25 | sed 's/^/     /'
echo "   resumed master log, last 5 lines:"; tail -5 "$WORK/uspex_master_r1.log" | sed 's/^/     /'
say "done"
