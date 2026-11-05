#!/bin/bash
# Resume options test (#59, #69). The restart tests showed that USPEX 25
# starts a new search both on a plain rerun and with pickUpGen 3 /
# pickUpFolder 1. This job tries the two remaining options, one after the
# other, on the stopped search in WORK (results1/ = generations 1-2):
#
#   Part 1  rerun in WORK with pickUpGen = 2 (the last complete generation)
#           and pickUpFolder = 1.
#   Part 2  a new search in WORK2, seeded (manual section 6.5) with WORK's
#           results1/good_structures_POSCARS as Seeds/POSCARS.
#
# Report: for each part, the new results folder, whether it starts at
# generation 1 or continues, individual numbering, the best energy, and
# (part 2) Seeds_history. Same 8-core layout as si_bridge.sbatch.
#
# Usage: WORK must hold the stopped search; WORK2 must not exist.
#   REPO=... WORK=/bmd/<user>/uspex_restart_1 WORK2=/bmd/<user>/uspex_seeded_1 POTCAR_SI=... \
#     sbatch --export=ALL resume_options.sbatch
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 03:30:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-resume-opts
#SBATCH -o uspex_resume_opts-%j.out

say() { echo "== $(date +%T) $*"; }
die() { echo "!! $(date +%T) STOPPED: $*" >&2; exit 1; }

say "job ${SLURM_JOB_ID:-?} on $(hostname)"
[ -n "${REPO:-}" ] || die "REPO is not set (the bmd-compute checkout with the #66 bridge code)"
[ -n "${WORK:-}" ] || die "WORK is not set (a new directory under /bmd)"
[ -n "${POTCAR_SI:-}" ] || die "POTCAR_SI is not set (the PBE_64 Si POTCAR)"
[ -f "$POTCAR_SI" ] || die "POTCAR_SI=$POTCAR_SI is not a file"
[ -f "$REPO/backend/structure_search/bridge/launcher.py" ] || die "no bridge code under REPO=$REPO"
[ -f "$WORK/search/results1/good_structures_POSCARS" ] || die "WORK=$WORK has no stopped search (results1/good_structures_POSCARS)"
[ -n "${WORK2:-}" ] || die "WORK2 is not set (a new directory for the seeded search)"
[ ! -e "$WORK2" ] || die "WORK2=$WORK2 already exists"
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
RESUME_TAG=${RESUME_TAG:-r1}
PICKUP_GEN=${PICKUP_GEN:-}
PICKUP_FOLDER=${PICKUP_FOLDER:-}
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

COMMON=(--singularity "$SINGULARITY" --image "$IMAGE" --cache "$CACHE"
  --potcar "$POTCAR_SI" --parallel "$PARALLEL" --ranks "$RANKS" --mem "$STEP_MEM"
  --launch-mode srun_steps --which-cluster "$WHICH_CLUSTER")

report_new_folder() { # search_dir folders_before master_log
  local dir=$1 before=$2 log=$3 r
  for r in $(cd "$dir" && ls -d results* 2>/dev/null); do
    case " $before " in *" $r "*) continue;; esac
    echo "   new folder $r: $(awk 'NR > 1' "$dir/$r/Individuals" 2>/dev/null | wc -l) individuals"
    echo "   first rows:"; head -3 "$dir/$r/Individuals" 2>/dev/null | sed 's/^/     /'
    echo "   last rows:"; tail -2 "$dir/$r/Individuals" 2>/dev/null | sed 's/^/     /'; echo
    echo "   generation_properties:"; sed 's/^/     /' "$dir/$r/generation_properties" 2>/dev/null
    echo "   best_structure:"; sed 's/^/     /' "$dir/$r/best_structure" 2>/dev/null; echo
    [ -e "$dir/$r/Seeds_history" ] && { echo "   Seeds_history:"; head -20 "$dir/$r/Seeds_history" | sed 's/^/     /'; }
  done
  echo "   first generation line in the master log: $(grep -m1 -E 'Generation [0-9]+ started' "$log")"
  echo "   master log lines about pickUp, seeds, restart:"
  grep -i -E "pick|seed|restart|resum|continu" "$log" | head -15 | sed 's/^/     /'
  echo "   'Task failed' lines: $(grep -c 'Task failed' "$log")"
}

say "part 1: rerun in $WORK with pickUpGen 2 / pickUpFolder 1"
before=$(cd "$WORK/search" && ls -d results* | tr '\n' ' ')
"$PYTHON" "$REPO/reference/uspex_pilot/si_bridge/run_pilot.py" --work "$WORK" "${COMMON[@]}" \
  --resume r3 --pick-up-gen 2 --pick-up-folder 1
say "part 1 ended with status $?"
echo "   pickUp lines: $(grep pickUp "$WORK/search/INPUT.txt" | tr '\n' ' ')"
report_new_folder "$WORK/search" "$before" "$WORK/uspex_master_r3.log"

say "part 2: new search in $WORK2 seeded with $WORK/search/results1/good_structures_POSCARS"
echo "   seed structures: $(grep -c '^number=' "$WORK/search/results1/good_structures_POSCARS")"
"$PYTHON" "$REPO/reference/uspex_pilot/si_bridge/run_pilot.py" --work "$WORK2" "${COMMON[@]}" \
  --seeds "$WORK/search/results1/good_structures_POSCARS"
say "part 2 ended with status $?"
report_new_folder "$WORK2/search" "" "$WORK2/uspex_master.log"
echo "   seeds folder now: $(ls "$WORK2/search/Seeds" 2>/dev/null | tr '\n' ' ')"
echo "   origins in generation 1: $(awk 'NR > 1 && $1 == 1 {print $7}' "$WORK2/search/results1/Individuals" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
say "done"
