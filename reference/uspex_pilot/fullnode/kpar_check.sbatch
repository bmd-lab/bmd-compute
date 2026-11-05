#!/bin/bash
# What does VASP 6.4.1 do when KPAR exceeds the number of irreducible
# k-points? A search shares one INCAR across all candidates, and some
# candidates have fewer than 4 k-points (#74, #59). Si8 at 8 ranks with
# KPAR 4 and NCORE 2, on meshes giving 1 k-point (Gamma) and a few k-points,
# each compared with KPAR 1 for time and energy.
#
# Usage (WORK must not exist):
#   WORK=/bmd/<user>/uspex_kpar_1 POTCAR_SI=/bmd-db/potcars/POT_PAW_PBE_64/Si/POTCAR \
#     sbatch --export=ALL kpar_check.sbatch
#SBATCH -p leeburton-pool
#SBATCH --qos=owner100d
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --exclude=compute-0-332,compute-0-333
#SBATCH -J uspex-kpar
#SBATCH -o uspex_kpar-%j.out

say() { echo "== $(date +%T) $*"; }
die() { echo "!! $(date +%T) STOPPED: $*"; exit 1; }

[ -n "${WORK:-}" ] || die "WORK is not set"
[ -n "${POTCAR_SI:-}" ] && [ -f "$POTCAR_SI" ] || die "POTCAR_SI is not a file"
[ ! -e "$WORK" ] || die "WORK=$WORK already exists"
mkdir -p "$WORK" || die "cannot create $WORK"
say "job ${SLURM_JOB_ID:-?} on $(hostname)"

type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
module load intel/rocky8-oneAPI-2023 vasp/rocky8-intel-6.4.1 >/dev/null 2>&1 || die "module load failed"
for lib in /usr/lib64/libpmi2.so /usr/lib64/slurm/libpmi2.so; do
  if [ -e "$lib" ]; then export I_MPI_PMI_LIBRARY=$lib; break; fi
done
unset I_MPI_PIN_DOMAIN

run_case() { # tag mesh kpar
  local tag=$1 mesh=$2 kpar=$3 d="$WORK/$1"
  mkdir -p "$d"
  printf '%s\n' "Si8 diamond" "5.47" "1 0 0" "0 1 0" "0 0 1" "Si" "8" "Direct" \
    "0.00 0.00 0.00" "0.00 0.50 0.50" "0.50 0.00 0.50" "0.50 0.50 0.00" \
    "0.25 0.25 0.25" "0.25 0.75 0.75" "0.75 0.25 0.75" "0.75 0.75 0.25" > "$d/POSCAR"
  printf 'SYSTEM = kpar check\nPREC = Normal\nENCUT = 250\nNSW = 0\nISMEAR = 0\nSIGMA = 0.05\nISPIN = 1\nNELM = 15\nNELMIN = 15\nEDIFF = 1E-9\nNCORE = 2\nKPAR = %s\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n' "$kpar" > "$d/INCAR"
  printf 'Gamma\n0\nGamma\n%s\n' "$mesh" > "$d/KPOINTS"
  cp "$POTCAR_SI" "$d/POTCAR"
  local s e
  s=$(date +%s.%N)
  ( cd "$d" && timeout 600 srun --exact --mpi=pmi2 -n 8 --mem=8G vasp_std > vasp.out 2>&1 )
  local rc=$?
  e=$(date +%s.%N)
  printf '   %-10s mesh %-6s KPAR %s: exit %s, %5.1f s, NKPTS %s, %s, %s\n' "$tag" "$mesh" "$kpar" "$rc" \
    "$(awk -v s="$s" -v e="$e" 'BEGIN{print e-s}')" \
    "$(grep -m1 -o 'NKPTS = *[0-9]*' "$d/OUTCAR" 2>/dev/null | grep -o '[0-9]*$')" \
    "$(grep -m1 'distrk' "$d/OUTCAR" 2>/dev/null | sed 's/^ *//')" \
    "$(grep 'F=' "$d/OSZICAR" 2>/dev/null | tail -1 | awk '{print "F=" $3}')"
  grep -i -E "error|warning|kpar|abort" "$d/vasp.out" "$d/OUTCAR" 2>/dev/null | grep -v -i "warning: underlay" | sort -u | head -5 | sed 's/^/      | /'
}

say "Gamma only (1 k-point)"
run_case g_kpar1 "1 1 1" 1
run_case g_kpar4 "1 1 1" 4
say "2x2x2 mesh (few k-points)"
run_case m2_kpar1 "2 2 2" 1
run_case m2_kpar4 "2 2 2" 4
say "4x4x4 mesh (more k-points than KPAR)"
run_case m4_kpar1 "4 4 4" 1
run_case m4_kpar4 "4 4 4" 4
say "done"
