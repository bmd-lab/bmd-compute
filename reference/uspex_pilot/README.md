# USPEX pilot scripts (Phase 0)

Manual tests on POWER for the structure-search design
(`docs/structure_search.md`, tracking issue #58, Phase 0 issue #59). These
scripts are not part of the running application.

## Step test

Checks that the launch bridge's mechanism works: whether several
`srun --exact` job steps started together inside one exclusive allocation
run at the same time on separate cores, whether each step needs an explicit
`--mem`, and whether a job may call `sbatch`.

Run from group storage, not from home (home quotas are small):

```bash
mkdir -p /bmd/<user>/uspex_checks && cp probe.sh steptest.sbatch /bmd/<user>/uspex_checks/
cd /bmd/<user>/uspex_checks && chmod +x probe.sh && sbatch steptest.sbatch
```

When the job has finished (replace `<jobid>`):

```bash
sacct -j <jobid> --format=JobID%22,Start,End,Elapsed,NCPUS,ReqMem,State
cat steptest-<jobid>.out; grep -h -i -E 'step creation|busy|error' A_*.log B_*.log
grep -h 'task=0 ' A_*.log B_*.log
cat B_*.log | grep -o 'cpus=[0-9,-]*' | sort | uniq -d | wc -l
```

How to read it:

- **B** (with `--mem`): all four steps should start within a second or two of
  each other and finish in about 30 s. This is the result the bridge needs.
- **A** (no `--mem`): if steps started about 30 s apart, each step needs an
  explicit `--mem` under `--mem=0`.
- The last command counts CPUs used by more than one step in B; 0 is good.
  If every task shows `cpus=0-191`, SLURM does not bind tasks to cores and
  VASP placement must be checked with `I_MPI_DEBUG=4` in the bridge pilot.
- **C**: "Job … to start at …" means jobs may submit jobs; an error means
  they may not (this does not affect the recommended design).

Record the result in `docs/structure_search.md` (§9) and in #59.
