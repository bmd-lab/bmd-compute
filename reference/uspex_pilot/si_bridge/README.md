# Si8 bridge pilot (Phase 0)

A tiny USPEX 25.0 search for Si8 at 0 GPa (the answer should be diamond Si,
space group 227) run through the launch bridge described in
`docs/structure_search.md` §4. Part of #59; uses the bridge code from #66.

What it tests: USPEX in the shared sandbox, the bridge script as
`commandExecutable`, VASP started by the host launcher as `srun --exact`
steps (by default 2 at once, 4 ranks each, in an 8-core job that starts
quickly; `PARALLEL`, `RANKS`, `STEP_MEM` and `sbatch -n` change the layout), and USPEX reading the results. It does **not** test BMD
methodology: the INCARs in `Specific/` are pilot settings (Si ENMAX is
about 245 eV; steps follow the shape recommended in the USPEX manual §3.1).
It also does not test 4 × 48-rank placement on a full node.

## Run it

The checkout must be under `/bmd`, because the container only sees `/bmd`.
Until #66 is merged, combine its branch with this directory:

```bash
git clone https://github.com/bmd-lab/bmd-compute.git /bmd/<user>/bmd-compute-pilot
cd /bmd/<user>/bmd-compute-pilot
git fetch origin claude/uspex-launch-bridge claude/confident-ride-qqpxp5
git checkout -b pilot origin/claude/uspex-launch-bridge
git checkout origin/claude/confident-ride-qqpxp5 -- reference/uspex_pilot/si_bridge
```

Find the Si POTCAR in the shared PBE_64 set:

```bash
find /bmd-db/potcars -path '*64*' -path '*/Si/POTCAR' 2>/dev/null | head
```

Submit (`WORK` must not exist yet; `PYTHON` defaults to `/usr/bin/python3`,
the system Python, because a personal `PATH` may put another `python3` first;
the bridge code was tested on Python 3.9 and 3.12). If the job log says
Singularity was not found, add `SINGULARITY=<full path>` (from
`module load uspex-25; command -v singularity`):

```bash
REPO=/bmd/<user>/bmd-compute-pilot WORK=/bmd/<user>/uspex_si_pilot_1 POTCAR_SI=<path from find> \
  sbatch --export=ALL reference/uspex_pilot/si_bridge/si_bridge.sbatch
```

## What to check afterwards

USPEX runs with `whichCluster = -1` by default: the probe job
(`../probe/probe.sbatch`, #59) showed that USPEX 25 runs the command as
`sh run_local.sh` in each CalcFold in that mode, and never runs it in mode 0.


USPEX runs each relaxation command through its own `run_local.sh` in the
CalcFold and does not keep the command's output. The pilot therefore points
`commandExecutable` at `search/bmd_bridge_pilot.sh`, a wrapper that sets the
bridge settings explicitly and appends its arguments, environment and the
bridge's error output to `bmd_bridge_debug.log` in each CalcFold. Read that
file first if relaxations fail.

```bash
W=/bmd/<user>/uspex_si_pilot_1
cat si_pilot-<jobid>.out
tail -30 $W/uspex_master.log
ls $W/search/results1/
head -20 $W/search/results1/Individuals
grep -h '"slot"' $W/search/CalcFold*/bmd_launch.jsonl | head
grep -l -i "error\|abort" $W/spool/logs/*.log | head
grep -h "running on\|mpi-ranks" $W/search/CalcFold*/OUTCAR | sort | uniq -c
grep -h "MPI startup(): Rank" -A13 $(ls $W/spool/logs/*.log | head -1) | head -16
```

- The master log should show generations progressing, and `results1/`
  should hold `Individuals`, `BESTIndividuals` and `goodStructures_POSCARS`.
  Note which file carries the enthalpies for `goodStructures_POSCARS`, and
  whether survivors of a generation get new CalcFold runs (#59).
- `OUTCAR` should report 12 MPI ranks per run. If it reports 1 rank (twelve
  copies of a single-rank VASP), `srun --mpi=pmi2` is not wiring Intel MPI
  up; check `I_MPI_PMI_LIBRARY` in the job output.
- The `I_MPI_DEBUG=4` map shows which cores each run used; four concurrent
  runs should not share cores.
- The lowest-enthalpy structure should be diamond Si (space group 227).
- Whether `whichCluster = 0` core control works in the container (if USPEX
  fails before the first CalcFold, retry with `-1 : whichCluster`).

## Restart test

`restart_test.sbatch` (same 8-core layout) starts the pilot search, sends it
SIGTERM about a minute after `STOP_AFTER_GEN` generations (default 2) are
complete, as SLURM does at the end of an allocation, then reruns USPEX in the
same folder with a fresh bridge spool (`run_pilot.py --resume r1`). The job
log reports which `results*/` folders exist afterwards, whether individual
numbers and generations continue, and how the resumed run ended. It answers
how a "resume search" attempt (#69) behaves.

```bash
REPO=/bmd/<user>/bmd-compute-pilot WORK=/bmd/<user>/uspex_restart_1 POTCAR_SI=<path> \
  sbatch --export=ALL reference/uspex_pilot/si_bridge/restart_test.sbatch
```
