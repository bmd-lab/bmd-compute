# BMD Compute

BMD Compute is a browser-based on-ramp for Burton Materials Design Lab students and beginning computational users to build, submit, monitor, and inspect VASP calculations on TAU PowerSLURM.

BMD Compute's canonical ecosystem role is the core VASP data-generation pipeline. It owns the authoritative implementation of decisions required to construct, validate, execute, and provenance BMD VASP calculations.

If a capability determines how BMD generates a VASP calculation, its authoritative implementation belongs in BMD Compute. If it provides supporting scientific data or tooling but is not part of the core VASP data-generation pipeline, it belongs in BMDex. BMD Agent consumes and coordinates these capabilities without duplicating their authority.

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

Supported student-facing workflows include:

- Geometry Optimisation
- Static Energy
- Double Geometry Optimisation
- Geometry Optimisation -> Static Energy
- Density of States
- Band Structure

Supported scientific controls include:

- PBE
- HSE06 for reviewed stages and stage-first workflows
- Spin Polarised calculations where enabled by the stage registry
- explicit DFT+U, only when the selected input set has reviewed U values
- SOC for PBE and HSE06 Static Energy stages and HSE06 DOS and Band Structure stages

Validated or specifically reviewed examples include PBE Static + SOC on Si, PBE + DFT+U Static -> PBE + DFT+U + SOC Static on Fe2O3, and HSE06 Band Structure through the stage-first HSE static precursor path.

Deliberately unavailable or unreviewed combinations remain blocked by validation. Examples include r2SCAN, Dielectric, GW, general SOC relaxation, PBE DOS/Band Structure + SOC, and arbitrary free-form INCAR editing.

## Scientific Policy Notes

BMD Compute uses pymatgen and atomate2 as the default scientific implementation layer and applies small Burton Lab policies centrally.

Important current policies:

- Generated inputs are pre-submission policy previews. In downstream stages after relaxation, the runtime structure comes from the previous completed stage.
- PBE relax stages use the Burton Lab relax policy, including ENCUT 580 eV.
- Static/final electronic stages use the Burton Lab final policy, including ENCUT 620 eV where applicable.
- HSE06 policy is stage-specific: relax uses `PRECFOCK = Fast`, static uses `PRECFOCK = Accurate`, and HSE06 band structure uses the reviewed atomate2 HSE band path.
- DFT+U is explicit. BMD Compute does not silently inherit Hubbard U into plain PBE calculations.
- SOC/non-collinear stages route to `vasp_ncl` (Custodian `auto_gamma` is disabled for them so the command cannot be swapped for `vasp_gam`), keep `ISYM = 0`, suppress incompatible `LELF`, and omit `ISPIN`.
- SOC starting moments: vector `MAGMOM` keeps the pymatgen/Materials Project starting moments when the structure contains an element in the spin method-consideration screen, or when the stage is explicitly Spin Polarised; otherwise SOC starts from zero vector moments.
- In the standard Desired Output workflows SOC is BMD methodology, not advice: when the heavy-element SOC policy triggers, SOC is applied to every non-relaxation stage (PBE Static for Static Energy; HSE06 Static and HSE06 DOS or Band Structure for DOS and Band Structure). Relaxations stay non-SOC. Custom workflows are never changed automatically.
- HSE06 Band Structure + SOC keeps the atomate2 zero-weight high-symmetry path and `reciprocal_density = 64`, but replaces pymatgen's symmetry-reduced weighted SCF points with every point of the same mesh, because `ISYM = 0` makes VASP treat the listed points as the complete sampling. Non-SOC HSE06 Band Structure is unchanged.
- Stages that consume the previous stage's electronic data must match its SOC setting: PBE DOS and Band Structure restart from the fixed charge density (`ICHARG = 11`), and PBE and HSE06 DOS/Band Structure size `NBANDS` from the previous run. HSE06 DOS/Band Structure do not read the copied static CHGCAR (no `ICHARG`/`ISTART`, no WAVECAR carried forward).
- NCORE is an execution-resource policy, not a theory policy. Automatic NCORE is stage-specific and currently applies to Relax, Static, and DOS stages; Band Structure omits automatic NCORE pending separate benchmarking.

## Operational Safety

The current deployment model is:

```text
authorized student
  -> TAU VPN / university network
  -> BMD Compute
  -> constrained shared bmdguest identity
  -> PowerSLURM
```

There is no application-level login, SSO, CSRF protection, or per-user job ownership boundary in the current app. That is an accepted VPN-bound lab deployment policy, not a public Internet security model. Do not expose the Uvicorn port directly to the public Internet. Resume by SLURM job ID is a convenience for shared service/group calculations, not a private authorization boundary.

Current operational hardening includes allow-listed CPU/memory/queue values, fixed backend account policy, bounded process-local SSH concurrency, deterministic SSH cleanup, server-side submission idempotency, explicit Load Results for completed calculations, remote-side pymatgen result parsing, production traceback hiding, and structured submission provenance.

## Local Development

```bash
git clone https://github.com/bmd-lab/bmd_compute.git
cd bmd_compute
conda env create -f environment.yml
conda activate bmd-compute
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. Building and previewing calculations is local;
POWER preparation and submission require deployment-local SSH, cluster, and
licensed POTCAR configuration that is intentionally not stored here.

The environment file is intentionally broad and currently not a lock file. It describes the main conda packages needed by the app, but exact production reproducibility still depends on the maintained `bmd-compute` environment.

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

BMD Compute's repository-owned source code and documentation are released
under the [MIT License](LICENSE). This does not license or redistribute VASP,
POTCAR/PAW datasets, or third-party dependencies; those remain subject to their
own licenses and access requirements.

Publishing this source repository is not a security boundary for the deployed
service. The current web application assumes the institutional/VPN network
boundary described above and is not suitable for direct public-Internet
exposure without separate authentication, authorization, and CSRF work.

