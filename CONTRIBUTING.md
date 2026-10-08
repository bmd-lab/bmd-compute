# Contributing to bmd-compute

bmd-compute is maintained as a focused research-software project. Keep changes
small, reviewable, and supported by tests.

## Typical contribution

```bash
git clone https://github.com/bmd-lab/bmd-compute.git
cd bmd-compute
conda env create -f environment.yml
conda activate bmd-compute
git switch -c your-name/short-change-name
```

Make one focused change, then run:

```bash
python -m pytest tests -q
python -m compileall backend tests
git diff --check
```

Commit the change, push the branch, and open a pull request. Explain the reason
for the change, the files affected, and the validation performed. Scientific
or execution-policy changes need focused regression tests and appropriate BMD
review before deployment.

## Never commit

- API keys, tokens, passwords, or other credentials
- SSH private keys
- `POTCAR` files or licensed VASP potential contents
- deployment-local credentials or configuration
- private or unpublished research data unless its release is explicitly approved

If a credential is committed accidentally, report it immediately so it can be
revoked and the incident handled. Removing it in a later commit is not enough.

Production SSH trust, cluster access, POTCAR repositories, and service
configuration remain deployment-local and are not supplied by this repository.
