"""Test-wide fixtures.

The production POTCAR ENMAX metadata (``backend/calculations/
potcar_enmax_PBE_64.json``) is generated on POWER from the licensed POTCAR
library. Tests never read it: every test plans against clearly labelled
SYNTHETIC metadata (``"synthetic": true``, ENMAX = 400 eV for every
selectable symbol, so the resolved cutoff is 550 eV). Tests that need other
values write their own metadata file and point the resolver at it.
"""

from __future__ import annotations

import hashlib
import json

import pytest

SYNTHETIC_ENMAX_EV = 400.0


def synthetic_enmax_metadata(enmax_by_symbol=None, *, default_enmax=SYNTHETIC_ENMAX_EV) -> dict:
    from backend.calculations.potcar_enmax_metadata import (
        METADATA_SCHEMA,
        METADATA_SCHEMA_VERSION,
        generator_selectable_symbols,
    )

    values = dict(enmax_by_symbol or {})
    symbols = sorted(set(generator_selectable_symbols()) | set(values))
    return {
        "schema": METADATA_SCHEMA,
        "schema_version": METADATA_SCHEMA_VERSION,
        "functional": "PBE_64",
        "synthetic": True,
        "source": {"note": "SYNTHETIC test metadata; not derived from any POTCAR library"},
        "potcars": {
            symbol: {
                "TITEL": f"SYNTHETIC {symbol}",
                "ENMAX": float(values.get(symbol, default_enmax)),
                "sha256": hashlib.sha256(f"synthetic:{symbol}".encode()).hexdigest(),
            }
            for symbol in symbols
        },
    }


def use_enmax_metadata(monkeypatch, path, metadata) -> None:
    """Point the cutoff resolver at ``metadata`` written to ``path``."""

    import backend.calculations.geometry_relaxation as geometry_relaxation

    path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(geometry_relaxation, "metadata_path", lambda functional: path)


_ORIGINAL_METADATA_PATH = None
_ORIGINAL_POTCAR_LOADER = None


def synthetic_potcar_loader(symbol, functional):
    """A stand-in for the POTCAR a run would read, matching the synthetic metadata.

    No POTCAR file exists in the test environment; simulated runs check the
    recorded identities against these SYNTHETIC datasets.
    """

    from types import SimpleNamespace

    entry = synthetic_enmax_metadata()["potcars"][symbol]
    return SimpleNamespace(
        keywords={"TITEL": entry["TITEL"], "ENMAX": entry["ENMAX"]},
        data=f"synthetic:{symbol}",
    )


def pytest_configure(config):
    # Installed at configuration time, before collection, because several test
    # modules plan calculations at import time; module-, session- and
    # function-scoped fixtures then all see the synthetic metadata.
    global _ORIGINAL_METADATA_PATH
    import tempfile
    from pathlib import Path

    import backend.calculations.geometry_relaxation as geometry_relaxation

    directory = Path(tempfile.mkdtemp(prefix="bmd-synthetic-enmax-"))
    path = directory / "potcar_enmax_PBE_64.json"
    path.write_text(json.dumps(synthetic_enmax_metadata(), sort_keys=True), encoding="utf-8")
    _ORIGINAL_METADATA_PATH = geometry_relaxation.metadata_path
    geometry_relaxation.metadata_path = lambda functional: path
    global _ORIGINAL_POTCAR_LOADER
    _ORIGINAL_POTCAR_LOADER = geometry_relaxation._default_potcar_loader
    geometry_relaxation._default_potcar_loader = synthetic_potcar_loader


def pytest_unconfigure(config):
    import backend.calculations.geometry_relaxation as geometry_relaxation

    if _ORIGINAL_METADATA_PATH is not None:
        geometry_relaxation.metadata_path = _ORIGINAL_METADATA_PATH
    if _ORIGINAL_POTCAR_LOADER is not None:
        geometry_relaxation._default_potcar_loader = _ORIGINAL_POTCAR_LOADER


def planned(structure, workflow, potcar_functional="PBE_64"):
    """``workflow`` with its plane-wave cutoffs resolved, as planning does."""

    from backend.calculations.geometry_relaxation import freeze_plane_wave_cutoffs

    return freeze_plane_wave_cutoffs(structure, workflow, potcar_functional=potcar_functional)
