#!/bin/bash
# Full-node launch and KPAR test for USPEX searches (#59, #75). Performance
# tuning only: correctness of the launch path is already shown by the probe
# and the Si8 pilot. Nothing waits on this job.
#
#   Part 1: four concurrent 48-rank VASP runs on one node, launched as
#     A  srun_steps     srun --exact --mpi=pmi2 -n 48 --mem=200G vasp_std
#     B  mpirun_pinned  mpirun -n 48, I_MPI_PIN_PROCESSOR_LIST per slot
#                       (0-47, 96-143, 48-95, 144-191), I_MPI_HYDRA_BOOTSTRAP=fork
#     C  baseline       one 192-rank run (mpirun -n 192)
#     Reports wall time per run, the throughput of A and B relative to C, and
#     whether any two concurrent runs share a CPU.
#   Part 2: KPAR 1 against KPAR 4 for one 48-rank run of a small, k-point
#     dense cell (Si8, 6x6x6 mesh), as a search's final steps would use.
#
# Timings on this cluster vary by up to 2x with node state (lab VASP notes);
# this is one pass, so treat the numbers as indicative, not as a benchmark.
#
# Usage (WORK must not exist):
#   WORK=/bmd/<user>/uspex_fullnode_1 POTCAR_SI=/bmd-db/potcars/POT_PAW_PBE_64/Si/POTCAR \
#     sbatch --export=ALL fullnode.sbatch
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 192
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 01:30:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-fullnode
#SBATCH -o uspex_fullnode-%j.out

say() { echo "== $(date +%T) $*"; }
die() { echo "!! $(date +%T) STOPPED: $*"; exit 1; }

[ -n "${WORK:-}" ] || die "WORK is not set"
[ -n "${POTCAR_SI:-}" ] && [ -f "$POTCAR_SI" ] || die "POTCAR_SI is not a file"
[ ! -e "$WORK" ] || die "WORK=$WORK already exists"
mkdir -p "$WORK" || die "cannot create $WORK"
say "job ${SLURM_JOB_ID:-?} on $(hostname)"
lscpu | grep -E "Model name|Socket|NUMA node[0-9]" | sed 's/^/   /'

type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
module load intel/rocky8-oneAPI-2023 vasp/rocky8-intel-6.4.1 >/dev/null 2>&1 || die "module load failed"
command -v vasp_std >/dev/null || die "vasp_std not found"
for lib in /usr/lib64/libpmi2.so /usr/lib64/slurm/libpmi2.so; do
  if [ -e "$lib" ]; then export I_MPI_PMI_LIBRARY=$lib; break; fi
done
unset I_MPI_PIN_DOMAIN
export I_MPI_DEBUG=4

si8_positions="0.00 0.00 0.00
0.00 0.50 0.50
0.50 0.00 0.50
0.50 0.50 0.00
0.25 0.25 0.25
0.25 0.75 0.75
0.75 0.25 0.75
0.75 0.75 0.25"

# Si64: 2x2x2 supercell of the 8-atom diamond cell, a fixed number of
# electronic steps so every run does the same work.
make_si64() { # dir
  local d=$1 i j k; mkdir -p "$d"
  {
    echo "Si64 diamond 2x2x2"; echo "10.94"; echo "1 0 0"; echo "0 1 0"; echo "0 0 1"
    echo "Si"; echo "64"; echo "Direct"
    for i in 0 1; do for j in 0 1; do for k in 0 1; do
      echo "$si8_positions" | awk -v i=$i -v j=$j -v k=$k '{printf "%.4f %.4f %.4f\n", ($1+i)/2, ($2+j)/2, ($3+k)/2}'
    done; done; done
  } > "$d/POSCAR"
  printf 'SYSTEM = fullnode test\nPREC = Normal\nENCUT = 300\nNSW = 0\nISMEAR = 0\nSIGMA = 0.05\nISPIN = 1\nNELM = 12\nNELMIN = 12\nEDIFF = 1E-9\nNCORE = 4\nKPAR = 1\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n' > "$d/INCAR"
  printf 'Gamma\n0\nGamma\n2 2 2\n' > "$d/KPOINTS"
  cp "$POTCAR_SI" "$d/POTCAR"
}

report_runs() { # dirs...
  local d t
  for d in "$@"; do
    t=$(cat "$d/elapsed" 2>/dev/null)
    echo "   $(basename "$d"): ${t:-?} s, $(grep -h -o 'running *[0-9]* mpi-ranks' "$d/OUTCAR" 2>/dev/null | head -1), first pins: $(grep 'MPI startup(): [0-9]' "$d/vasp.out" | head -3 | awk '{print $NF}' | tr -d '{}' | tr '\n' ' ')"
  done
  local shared
  shared=$(for d in "$@"; do grep 'MPI startup(): [0-9]' "$d/vasp.out" | awk '{print $NF}' | tr -d '{}' | sort -u; done | sort | uniq -d | wc -l)
  echo "   CPUs used by more than one concurrent run: $shared"
}

timed() { # dir command...
  local d=$1; shift
  ( cd "$d" && s=$(date +%s.%N) && timeout 900 "$@" > vasp.out 2>&1; e=$(date +%s.%N); awk -v s="$s" -v e="$e" 'BEGIN{printf "%.1f\n", e-s}' > elapsed )
}

# ------------------------------------------------------------------ part 1
say "part 1 A: 4 x srun --exact --mpi=pmi2 -n 48 --mem=200G"
dirs=(); for i in 0 1 2 3; do make_si64 "$WORK/A$i"; dirs+=("$WORK/A$i"); done
start=$(date +%s.%N)
for d in "${dirs[@]}"; do timed "$d" srun --exact --mpi=pmi2 -n 48 --mem=200G vasp_std & done; wait
tA=$(awk -v s="$start" -v e="$(date +%s.%N)" 'BEGIN{printf "%.1f", e-s}')
report_runs "${dirs[@]}"; echo "   all four done in $tA s"

say "part 1 B: 4 x mpirun -n 48, pinned per slot, Hydra fork bootstrap"
lists=("0-47" "96-143" "48-95" "144-191")
dirs=(); for i in 0 1 2 3; do make_si64 "$WORK/B$i"; dirs+=("$WORK/B$i"); done
start=$(date +%s.%N)
for i in 0 1 2 3; do
  timed "${dirs[$i]}" env I_MPI_PIN_PROCESSOR_LIST="${lists[$i]}" I_MPI_HYDRA_BOOTSTRAP=fork mpirun -n 48 vasp_std &
done; wait
tB=$(awk -v s="$start" -v e="$(date +%s.%N)" 'BEGIN{printf "%.1f", e-s}')
report_runs "${dirs[@]}"; echo "   all four done in $tB s"

say "part 1 C: 1 x mpirun -n 192 (baseline)"
make_si64 "$WORK/C0"
timed "$WORK/C0" mpirun -n 192 vasp_std
tC=$(cat "$WORK/C0/elapsed")
report_runs "$WORK/C0"
awk -v a="$tA" -v b="$tB" -v c="$tC" 'BEGIN{printf "   throughput vs one 192-rank run: A %.2fx, B %.2fx (4 runs in the time shown vs 1 run in %s s)\n", 4*c/a, 4*c/b, c}'

# ------------------------------------------------------------------ part 2
for kpar in 1 4; do
  d="$WORK/K$kpar"; mkdir -p "$d"
  { echo "Si8 diamond"; echo "5.47"; echo "1 0 0"; echo "0 1 0"; echo "0 0 1"; echo "Si"; echo "8"; echo "Direct"; echo "$si8_positions"; } > "$d/POSCAR"
  printf 'SYSTEM = kpar test\nPREC = Accurate\nENCUT = 580\nNSW = 0\nISMEAR = 0\nSIGMA = 0.05\nISPIN = 1\nNELM = 15\nNELMIN = 15\nEDIFF = 1E-9\nNCORE = 4\nKPAR = %s\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n' "$kpar" > "$d/INCAR"
  printf 'Gamma\n0\nGamma\n6 6 6\n' > "$d/KPOINTS"
  cp "$POTCAR_SI" "$d/POTCAR"
  say "part 2: Si8 6x6x6, 48 ranks, KPAR $kpar"
  timed "$d" srun --exact --mpi=pmi2 -n 48 --mem=200G vasp_std
  echo "   $(cat "$d/elapsed") s; $(grep -h -m1 NKPTS "$d/OUTCAR" | awk '{print $3, $4}'); $(grep -h -m1 distrk "$d/OUTCAR")"
done

say "done"
