# bmd-compute

bmd-compute is a browser-based on-ramp for Burton Materials Discovery Lab
students and beginning computational users at Tel Aviv University to build,
submit, monitor, and inspect VASP calculations on TAU PowerSLURM.

bmd-compute's canonical ecosystem role is the core VASP data-generation pipeline. It owns the authoritative implementation of decisions required to construct, validate, execute, and provenance BMD VASP calculations.

If a capability determines how BMD generates a VASP calculation, its authoritative implementation belongs in bmd-compute. If it provides supporting scientific data or tooling but is not part of the core VASP data-generation pipeline, it belongs in bmd-store. bmd-check consumes exposed capabilities and evidence to inspect, diagnose, explain, and advise without duplicating their authority.

It is intentionally not a replacement for normal SSH/SLURM cluster access for experienced computational researchers. Its job is to make the first scientific workflow visible and teachable: structure in, reviewed calculation plan, generated VASP inputs, remote submission, monitoring, and scientific results.

## Architecture

```text
Browser
  -> FastAPI / Jinja
  -> stage-first WorkflowSpec
  -> pymatgen / atomate2 / jobflow
  -> Paramiko
  -> PowerSLURM
  -> VASP
  -> remote pymatgen result parsing
  -> Results UI
```

FastAPI should stay a thin controller. Scientific behavior belongs in `backend/`; infrastructure concerns such as SSH, SLURM submission, provenance, and remote result parsing are kept separate from calculation policy where practical.

## Current Capabilities

Students choose a Desired Output, or build a Custom workflow:

- Energy only: PBE Static Energy
- Relaxed structure: PBE Geometry Optimisation -> PBE Geometry Optimisation
- Electronic density of states: PBE Geometry Optimisation -> HSE06 Static Energy -> HSE06 Density of States
- Electronic band structure: PBE Geometry Optimisation -> HSE06 Static Energy -> HSE06 Band Structure
- Custom: an ordered list of supported PBE/HSE06 stages and Advanced Options, never changed automatically

Desired Outputs add spin polarisation, DFT-D3(BJ), SOC and DFT+U automatically when their rules trigger. If structure dimensionality analysis fails, a Desired Output fails closed before preview or remote work because BMD cannot safely decide whether DFT-D3(BJ) is required. Custom workflows use Spin Polarised, van der Waals correction, SOC and DFT+U only when selected.

The authoritative declaration of bmd-compute's v1 executable methodology (workflows, automatic treatments, stage settings, k-points, POTCARs, starting moments, stage chaining and the execution model) is [`docs/methodology.md`](docs/methodology.md).

Validated or specifically reviewed examples include PBE Static + SOC on Si, PBE + DFT+U Static -> PBE + DFT+U + SOC Static on Fe2O3, and HSE06 Band Structure through the stage-first HSE static precursor path.

Deliberately unavailable or unreviewed combinations remain blocked by validation. Examples include r2SCAN, Dielectric, GW, general SOC relaxation, PBE DOS/Band Structure + SOC, and arbitrary free-form INCAR or KPOINTS editing.

## Scientific Policy Notes

bmd-compute uses pymatgen and atomate2 as the default scientific implementation layer and applies small Burton Materials Discovery Lab policies centrally.

Important current policies:

- Generated inputs are pre-submission previews generated from the submitted structure. From stage 2 onwards, execution uses the previous stage's outputs (relaxed structure, final moments, regenerated k-points, NBANDS, charge density where applicable); these expected differences are described in `docs/methodology.md`.
- Spin polarisation is applied automatically in Desired Output workflows to every stage when the structure contains an element in the spin composition screen, and DFT-D3(BJ) (`IVDW = 12`) to PBE Relax/Static stages when two-dimensional bonded connectivity is detected.
- PBE relax stages use the Burton Lab relax policy, including ENCUT 580 eV.
- Static/final electronic stages use the Burton Lab final policy, including ENCUT 620 eV where applicable.
- HSE06 policy is stage-specific: relax uses `PRECFOCK = Fast`, static uses `PRECFOCK = Accurate`, and HSE06 band structure uses the reviewed atomate2 HSE band path.
- DFT+U in the standard Desired Output workflows follows the pinned pymatgen/Materials Project GGA+U oxide/fluoride rule and values (Co, Cr, Fe, Mn, Mo, Ni, V, W with O or F as the most electronegative element), suppressed only when every charge-balanced pymatgen oxidation-state guess puts every triggering element at d0. It is applied to PBE Relax/Static stages only, never to HSE06 stages, and does not turn on spin polarisation. Values are frozen at preparation and verified at run time. These are standard empirical MP values, not values fitted to a given material. Custom workflows use DFT+U only when selected, and bmd-compute never silently inherits Hubbard U into plain PBE.
- SOC/non-collinear stages route to `vasp_ncl` (Custodian `auto_gamma` is disabled for them so the command cannot be swapped for `vasp_gam`), keep `ISYM = 0`, suppress incompatible `LELF`, and omit `ISPIN`.
- SOC starting moments: vector `MAGMOM` keeps the pymatgen/Materials Project starting moments when the structure contains an element in the spin method-consideration screen, or when the stage is explicitly Spin Polarised; otherwise SOC starts from zero vector moments.
- In the standard Desired Output workflows SOC is BMD methodology, not advice: when the heavy-element SOC policy triggers, SOC is applied to every non-relaxation stage (PBE Static for Static Energy; HSE06 Static and HSE06 DOS or Band Structure for DOS and Band Structure). Relaxations stay non-SOC. Custom workflows are never changed automatically.
- HSE06 Band Structure + SOC keeps the atomate2 zero-weight high-symmetry path and `reciprocal_density = 64`, but replaces pymatgen's symmetry-reduced weighted SCF points with every point of the same mesh, because `ISYM = 0` makes VASP treat the listed points as the complete sampling. Non-SOC HSE06 Band Structure is unchanged.
- Stages that consume the previous stage's electronic data must match its SOC setting: PBE DOS and Band Structure restart from the fixed charge density (`ICHARG = 11`), and PBE and HSE06 DOS/Band Structure size `NBANDS` from the previous run. Atomate2 may physically copy `CHGCAR` into an HSE06 DOS/Band Structure stage, but BMD requests neither a fixed-charge-density nor a WAVECAR restart (`ICHARG` and `ISTART` are absent), so the HSE06 calculation remains self-consistent.
- An unsuccessful (unconverged) stage fails the workflow; BMD sets this on every stage instead of relying on atomate2's configurable default. A workflow runs in one SLURM allocation: reaching the walltime is a failure, results are loaded only from the final stage of a completed run, and running again starts a new attempt without reusing completed stages. Continuation across allocations is not part of v1.
- The POWER runner checks its scientific package versions against those recorded at preparation and stops before any VASP work on a mismatch (`constraints/scientific-runtime.txt`, `runtime_environment.json`).
- NCORE is an execution-resource policy, not a theory policy. For validated resource selections, eligible Relax, Static, and DOS stages automatically receive the current fixed `NCORE = 8` policy; Band Structure omits automatic NCORE pending separate benchmarking.

## Operational Safety

The current deployment model is:

```text
authorized student
  -> TAU VPN / university network
  -> bmd-compute
  -> constrained shared bmdguest identity
  -> PowerSLURM
```

There is no application-level login, SSO, CSRF protection, or per-user job ownership boundary in the current app. That is an accepted VPN-bound lab deployment policy, not a public Internet security model. Do not expose the Uvicorn port directly to the public Internet. Resume by SLURM job ID is a convenience for shared service/group calculations, not a private authorization boundary.

Current operational hardening includes allow-listed CPU/memory/queue values, fixed backend account policy, bounded process-local SSH concurrency, deterministic SSH cleanup, server-side submission idempotency, explicit Load Results for completed calculations, remote-side pymatgen result parsing, production traceback hiding, and structured submission provenance.

## Local Development

```bash
git clone https://github.com/bmd-lab/bmd-compute.git
cd bmd-compute
conda env create -f environment.yml
conda activate bmd-compute
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. Building and previewing calculations is local;
POWER preparation and submission require deployment-local SSH, cluster, and
licensed POTCAR configuration that is intentionally not stored here.

`environment.yml` covers the web application and development tooling. The scientific stack for both this environment and the POWER runtime is defined once, in `constraints/scientific-runtime.txt` (exact versions taken from POWER). The POWER runner compares its parity-critical packages with the versions recorded at preparation and refuses to start VASP on any difference; see `docs/deployment.md`.

Typical local checks:

```bash
python -m pytest tests -q
python -m compileall backend tests
git diff --check
```

On the Windows/Codex development machine, use the intended environment, for example:

```bash
conda run -n bmd-compute python -m pytest tests -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow
and the list of material that must never be committed.

## License and third-party software

bmd-compute's repository-owned source code and documentation are released
under the [MIT License](LICENSE). This does not license or redistribute VASP,
POTCAR/PAW datasets, or third-party dependencies; those remain subject to their
own licenses and access requirements.

Publishing this source repository is not a security boundary for the deployed
service. The current web application assumes the institutional/VPN network
boundary described above and is not suitable for direct public-Internet
exposure without separate authentication, authorization, and CSRF work.

