# Reference Material

This directory contains historical material used during the development of bmd-compute. These files are not part of the running application.

## workflow-selector-macandwindows.ipynb

This notebook was the primary Atomate2/Jobflow/PowerSLURM interface before bmd-compute. It remains an important source of validated scientific behavior, especially for the original Burton Lab PBE relax/static workflow policy.

bmd-compute has now evolved beyond a direct notebook clone. The current application should be read as:

- preserved notebook policy where it remains valid
- intentionally changed policy where implementation and PowerSLURM testing exposed a better or safer rule
- newly added policy for capabilities that were not part of the original notebook interface

Examples:

- Preserved: Burton Lab PBE relax/final ENCUT policy, with 580 eV for relax stages and 620 eV for final/static-style stages.
- Preserved: browser-generated VASP previews should reflect the policy that will be used before submission.
- Changed after implementation/testing: stage-first workflows now represent mixed-theory sequences explicitly instead of relying on a single global workflow label.
- Changed after PowerSLURM validation: PBE Static -> SOC no longer propagates WAVECAR by default, because the copied collinear WAVECAR was not a reliable restart artifact for the validated SOC transition.
- Added after notebook development: HSE06 stage policy, HSE06 Band Structure, explicit SOC executable routing, remote-side result parsing, submission idempotency, and structured provenance.

When migrating or revising notebook behavior:

1. understand the notebook behavior
2. compare it with current PowerSLURM validation and bmd-compute policy
3. extract or update reusable backend modules
4. test the backend independently
5. connect it to the browser

Do not copy notebook code directly into FastAPI route handlers.

