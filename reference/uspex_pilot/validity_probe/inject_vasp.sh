#!/bin/sh
# inject_vasp.sh STEP COMMAND...: run one VASP step for the validity probe.
#
# Used as the launcher's command template (run_validity_probe.py):
#   inject_vasp.sh {step} srun --exact --mpi=pmi2 -n {ranks} --mem={mem} vasp_std
#
# Every call is logged to $BMD_PROBE_STATE/calls.tsv (one line per VASP
# step: CalcFold, step, SHA-256 and first line of the POSCAR it started from,
# the injection applied, VASP's exit status, the status reported, and the
# last TOTEN in OUTCAR). That log survives USPEX removing old CalcFolds and
# shows retries (the same POSCAR run again at the same step).
#
# Injections, listed in $BMD_PROBE_INJECTIONS (comma-separated), each applied
# to exactly one step of one candidate per probe run. A call claims an
# injection by creating $BMD_PROBE_STATE/claimed_<name> with the shell's own
# exclusive create (noclobber: open with O_CREAT|O_EXCL), so concurrent calls
# cannot both take it, and a later retry of the same candidate is not injected
# again. Not "mkdir": the uutils (Rust) mkdir of newer Ubuntu releases reports
# success when a concurrent mkdir has just created the directory, and two calls
# both claimed exit7_step3 on GitHub Actions (PR #83, run 44600410654):
#
#   nelm_step2   the first step-2 request runs with "NELM = 3" in its INCAR:
#                a genuine electronic non-convergence. VASP exits 0; the
#                launcher's validity check should report 118. The CalcFold's
#                original INCAR is restored after VASP finishes (the injected
#                one is kept as INCAR.bmd_probe_injected and in
#                $BMD_PROBE_STATE), so a retry in the same CalcFold runs the
#                unmodified INCAR.
#   exit7_step3  the first step-3 request that VASP completes with status 0
#                is reported with status 7 (a VASP failure as USPEX sees it).
#
# Probe use only; never part of a real search.

set -u
step=$1
shift
state=${BMD_PROBE_STATE:?BMD_PROBE_STATE is not set}
injections=",${BMD_PROBE_INJECTIONS:-},"

wants() { case $injections in *",$1,"*) return 0 ;; esac; return 1; }
claim() { ( set -C; : > "$state/claimed_$1" ) 2>/dev/null; }

calcfold=$(pwd -P)
name=$(basename "$calcfold")
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
poscar_sha=$(sha256sum POSCAR 2>/dev/null | cut -c1-64)
poscar_head=$(head -n 1 POSCAR 2>/dev/null | tr '\t' ' ')
injected=none

if [ "$step" = 2 ] && wants nelm_step2 && claim nelm_step2; then
    injected=nelm_step2
    cp -p INCAR INCAR.bmd_probe_original
    sed -i '/^[[:space:]]*NELM[[:space:]]*=/d' INCAR
    echo "NELM = 3" >> INCAR
    cp -p INCAR "$state/INCAR.injected.$name"
fi

"$@"
vasp_status=$?
reported=$vasp_status

if [ "$injected" = nelm_step2 ]; then
    mv INCAR INCAR.bmd_probe_injected
    mv INCAR.bmd_probe_original INCAR
fi
if [ "$step" = 3 ] && [ "$vasp_status" -eq 0 ] && wants exit7_step3 && claim exit7_step3; then
    injected=exit7_step3
    reported=7
fi

toten=$(grep -a 'free  energy   TOTEN' OUTCAR 2>/dev/null | tail -n 1 | awk '{print $5}')
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$started" "$finished" "$name" "$step" "$poscar_sha" "$injected" \
    "$vasp_status" "$reported" "${toten:-}" "$poscar_head" >> "$state/calls.tsv"
if [ "$injected" != none ]; then
    printf '%s\t%s\t%s\t%s\n' "$finished" "$name" "$step" "$injected" >> "$state/injections.tsv"
fi
exit "$reported"
