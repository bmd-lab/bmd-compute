"""Tests for backend/structure_origin.py: the signed structure origin record."""

from __future__ import annotations

import pytest

from backend import submission as submission_module
from backend.calculations.registry import CalculationValidationError
from backend.structure_origin import (
    StructureOriginError,
    create_structure_origin_token,
    verify_structure_origin,
)


POSCAR = """Si2
1.0
3.8 0.0 0.0
0.0 3.8 0.0
0.0 0.0 3.8
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""
ORIGIN = {"source": "uspex_import", "individual_id": 35, "uspex_energy_ev": -43.28}


def test_round_trip():
    token = create_structure_origin_token(ORIGIN, POSCAR)
    assert verify_structure_origin(token, POSCAR) == ORIGIN


def test_browser_line_endings_and_trailing_spaces_still_match():
    token = create_structure_origin_token(ORIGIN, POSCAR)
    posted = "\r\n".join(line + "  " for line in POSCAR.splitlines()) + "\r\n"
    assert verify_structure_origin(token, posted) == ORIGIN


def test_no_token_means_no_origin():
    assert verify_structure_origin(None, POSCAR) is None
    assert verify_structure_origin("  ", POSCAR) is None


def test_edited_structure_is_rejected():
    token = create_structure_origin_token(ORIGIN, POSCAR)
    with pytest.raises(StructureOriginError, match="changed after it was taken"):
        verify_structure_origin(token, POSCAR.replace("0.25 0.25 0.25", "0.3 0.25 0.25"))


def test_altered_record_is_rejected():
    token = create_structure_origin_token(ORIGIN, POSCAR)
    other = create_structure_origin_token({**ORIGIN, "uspex_energy_ev": -99.0}, POSCAR)
    forged = other.split(".")[0] + "." + token.split(".")[1]
    with pytest.raises(StructureOriginError, match="could not be verified"):
        verify_structure_origin(forged, POSCAR)
    with pytest.raises(StructureOriginError, match="could not be read"):
        verify_structure_origin("not-a-token", POSCAR)


def test_token_from_another_secret_is_rejected(monkeypatch):
    monkeypatch.setenv(submission_module.SUBMISSION_IDENTITY_SECRET_ENV, "first")
    token = create_structure_origin_token(ORIGIN, POSCAR)
    monkeypatch.setenv(submission_module.SUBMISSION_IDENTITY_SECRET_ENV, "second")
    with pytest.raises(StructureOriginError, match="could not be verified"):
        verify_structure_origin(token, POSCAR)


def test_submission_identity_token_does_not_verify_as_origin():
    identity = submission_module.create_submission_identity_token("20261105-120000", "a" * 32)
    with pytest.raises(StructureOriginError):
        verify_structure_origin(identity, POSCAR)


def test_error_is_a_calculation_validation_error():
    # The workbench routes already show CalculationValidationError messages.
    assert issubclass(StructureOriginError, CalculationValidationError)
