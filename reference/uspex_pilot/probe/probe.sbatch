#!/bin/bash
# One small job that answers the remaining launch questions for the USPEX
# bridge (#59) without a whole node:
#
#   Part 1 (USPEX, no VASP): does USPEX 25 actually run commandExecutable,
#     and how (working directory, environment)? The command is a dummy that
#     only records how it was called and exits. Three experiments, each
#     limited to 3 minutes:
#       A  whichCluster -1
#       B  whichCluster  0
#       C  whichCluster  0, plus a host loop that marks run_local.sh executable
#   Part 2 (VASP, no USPEX): do 4-rank VASP runs start with 4 real ranks
#     through  srun --exact --mpi=pmi2  and through plain  mpirun?
#
# Usage (WORK must not exist):
#   WORK=/bmd/<user>/uspex_probe_1 POTCAR_SI=/bmd-db/potcars/POT_PAW_PBE_64/Si/POTCAR \
#     sbatch --export=ALL probe.sbatch
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-probe
#SBATCH -o uspex_probe-%j.out

say() { echo "== $(date +%T) $*"; }
die() { echo "!! $(date +%T) STOPPED: $*"; exit 1; }

[ -n "${WORK:-}" ] || die "WORK is not set"
[ -n "${POTCAR_SI:-}" ] && [ -f "$POTCAR_SI" ] || die "POTCAR_SI is not a file"
[ ! -e "$WORK" ] || die "WORK=$WORK already exists"
case $WORK in /bmd/*) ;; *) die "WORK must be under /bmd (the container sees only /bmd)";; esac
IMAGE=${IMAGE:-/bmd/shared/uspex25_sandbox}
CACHE=${CACHE:-/bmd/shared/uspex_cache}
mkdir -p "$WORK" || die "cannot create $WORK"
say "job ${SLURM_JOB_ID:-?} on $(hostname); WORK=$WORK"

type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
SINGULARITY=${SINGULARITY:-$(bash -c 'source /etc/profile >/dev/null 2>&1; module load uspex-25 >/dev/null 2>&1; command -v singularity' | tail -1)}
[ -x "$SINGULARITY" ] || die "singularity not found (got '$SINGULARITY')"
say "singularity: $SINGULARITY"

# ---------------------------------------------------------------- part 1
# The dummy relaxation command (one per experiment) records how it was
# called, then fails so USPEX moves on quickly. Its log has an absolute path,
# so it is found whatever directory USPEX runs it from. BMD_PROBE_MARK is set
# for the master: if the log says MISSING, USPEX starts commands with a
# reduced environment.
write_probe() { # tag
  local script="$WORK/probe_cmd_$1.sh" log="$WORK/probe_calls_$1.log"
  {
    echo '#!/bin/sh'
    echo '{'
    echo '  echo "== called $(date +%T) args: $* pid $$"'
    echo '  echo "pwd: $(pwd)"'
    echo '  echo "parent: $(tr "\000" " " < /proc/$PPID/cmdline 2>/dev/null)"'
    echo '  echo "BMD_PROBE_MARK=${BMD_PROBE_MARK:-MISSING}"'
    echo '  env | grep -E "^(LC_|LANG|PATH|XDG|PYTHON|OMP)" | sort'
    echo '  ls -la'
    echo "} >> \"$log\" 2>&1"
    echo 'exit 1'
  } > "$script"
  chmod 755 "$script"
}

write_input() { # dir whichCluster tag
  cat > "$1/INPUT.txt" <<EOF
PARAMETERS EVOLUTIONARY ALGORITHM
USPEX : calculationMethod
300 : calculationType

% atomType
Si
% EndAtomType

% numSpecies
8
% EndNumSpecies

0.0 : ExternalPressure
4 : populationSize
4 : initialPopSize
1 : numGenerations
1 : stopCrit

% abinitioCode
1
% ENDabinit

% KresolStart
0.2
% Kresolend

% commandExecutable
sh $WORK/probe_cmd_$3.sh 1
% EndExecutable

$2 : whichCluster
2 : numParallelCalcs
1 : coresPerJob
EOF
}

run_uspex() { # tag whichCluster [watch]
  local tag=$1 wc=$2 watch=${3:-no} dir="$WORK/uspex_$1"
  mkdir -p "$dir/Specific"
  printf 'SYSTEM = probe\nNSW = 0\n' > "$dir/Specific/INCAR_1"
  cp "$POTCAR_SI" "$dir/Specific/POTCAR_Si"
  write_probe "$tag"
  write_input "$dir" "$wc" "$tag"
  say "part 1 $tag: whichCluster $wc (watcher: $watch)"
  local watcher=
  if [ "$watch" = yes ]; then
    ( while :; do find "$dir" -name run_local.sh ! -perm -u+x -exec chmod 755 {} + 2>/dev/null; sleep 0.02; done ) &
    watcher=$!
  fi
  ( cd "$dir" && env -i PATH=/usr/bin:/bin LC_ALL=C.UTF-8 \
      SINGULARITYENV_XDG_CACHE_HOME="$CACHE" SINGULARITYENV_LC_ALL=C.UTF-8 \
      SINGULARITYENV_LANG=C.UTF-8 SINGULARITYENV_PYTHONIOENCODING=utf-8 \
      SINGULARITYENV_BMD_PROBE_MARK=passed \
      timeout 180 "$SINGULARITY" exec --cleanenv -B /bmd --pwd "$dir" "$IMAGE" /opt/uspex/uspex \
      > "$dir/master.log" 2>&1 )
  echo "   master exit status: $? (124 = stopped by the 3-minute limit)"
  [ -n "$watcher" ] && kill "$watcher" 2>/dev/null
  echo "   --- dummy command calls:"
  if [ -f "$WORK/probe_calls_$tag.log" ]; then
    echo "   $(grep -c '^== called' "$WORK/probe_calls_$tag.log") call(s); first one:"
    sed -n '1,/^total /p' "$WORK/probe_calls_$tag.log" | head -14 | sed 's/^/   /'
  else
    echo "   NONE: USPEX never ran the command"
  fi
  echo "   --- run_local.sh files:"
  find "$dir" -name run_local.sh -exec ls -l {} \; 2>/dev/null | head -3 | sed 's/^/   /'
  local f; f=$(find "$dir" -name run_local.sh | head -1); [ -n "$f" ] && sed 's/^/   | /' "$f"
  echo "   --- master log lines about launching or failing:"
  grep -i -E "launch|submit|fail|error|denied|Traceback|run_local|command" "$dir/master.log" | grep -v zoneinfo | head -15 | sed 's/^/   /'
  echo "   --- master log, last 6 lines:"
  tail -6 "$dir/master.log" | sed 's/^/   /'
}

run_uspex A -1
run_uspex B 0
run_uspex C 0 yes

# ---------------------------------------------------------------- part 2
say "loading the VASP modules"
module load intel/rocky8-oneAPI-2023 vasp/rocky8-intel-6.4.1 >/dev/null 2>&1 || die "module load failed"
command -v vasp_std >/dev/null || die "vasp_std not found"
for lib in /usr/lib64/libpmi2.so /usr/lib64/slurm/libpmi2.so; do
  if [ -e "$lib" ]; then export I_MPI_PMI_LIBRARY=$lib; break; fi
done
unset I_MPI_PIN_DOMAIN
export I_MPI_DEBUG=4

vasp_case() { # tag command...
  local tag=$1; shift
  local d="$WORK/vasp_$tag"; mkdir -p "$d"
  printf '%s\n' "Si8 diamond (conventional cell)" "5.47" "1 0 0" "0 1 0" "0 0 1" "Si" "8" "Direct" \
    "0.00 0.00 0.00" "0.00 0.50 0.50" "0.50 0.00 0.50" "0.50 0.50 0.00" \
    "0.25 0.25 0.25" "0.25 0.75 0.75" "0.75 0.25 0.75" "0.75 0.75 0.25" > "$d/POSCAR"
  printf 'SYSTEM = probe\nPREC = Normal\nENCUT = 250\nNSW = 0\nISMEAR = 0\nSIGMA = 0.05\nISPIN = 1\nNCORE = 2\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n' > "$d/INCAR"
  printf 'Gamma\n0\nGamma\n2 2 2\n' > "$d/KPOINTS"
  cp "$POTCAR_SI" "$d/POTCAR"
  say "part 2 $tag: $*"
  ( cd "$d" && timeout 300 "$@" > vasp.out 2>&1 )
  echo "   exit status: $?"
  grep -h -E "running on|mpi-ranks|distrk" "$d/OUTCAR" 2>/dev/null | head -3 | sed 's/^/   /'
  grep -E "F=" "$d/OSZICAR" 2>/dev/null | tail -1 | sed 's/^/   /'
  grep -A6 "MPI startup(): Rank" "$d/vasp.out" | head -7 | sed 's/^/   /'
  grep -i -E "error|abort|fatal" "$d/vasp.out" | head -5 | sed 's/^/   /'
}

vasp_case srun srun --exact --mpi=pmi2 -n 4 --mem=8G vasp_std
vasp_case mpirun mpirun -n 4 vasp_std

say "done"
