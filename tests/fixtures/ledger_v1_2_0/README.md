# v1.2.0 attempt ledger (test fixture)

`ledger.py` is `compute_api/ledger.py` exactly as released in BMD Compute
v1.2.0 (`git show v1.2.0:compute_api/ledger.py`). The structure-search
attempt tests load it to show that a v1.2.0 service, after a rollback, still
reads, scans and writes calculation attempts in a state root that also holds
structure-search attempts. It is not imported by the application.
