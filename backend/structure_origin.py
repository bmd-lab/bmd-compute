"""
Where a structure came from, carried safely from page to page.

The browser workbench sends the structure text back to the server on every
step (Build, Prepare, Submit). When a structure was not typed by the user but
taken from another source, for example a candidate of an uploaded USPEX
search (``backend.structure_search.imports``), that origin must reach the
submission's provenance unchanged and must belong to exactly that structure.

A *structure origin token* does this. It holds

* the origin record (a JSON object written by the source module), and
* the SHA-256 of the structure text it describes,

signed with a key derived from the submission identity secret
(``backend.submission.derived_signing_key``). :func:`verify_structure_origin`
accepts the token only if the signature is valid and the posted structure
text is the one the token was made for, so an origin can neither be forged
nor attached to an edited structure.

Like the submission identity token, a token signed with the per-process
ephemeral secret (no ``BMD_SUBMISSION_IDENTITY_SECRET`` configured) stops
verifying after a server restart; the user then hands the candidate off again.

The verified record is stored in the submission as
``flow_spec.structure.origin`` and copied into the provenance block as
``provenance.structure_source`` (see ``docs/provenance.md``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re

from backend.calculations.registry import CalculationValidationError
from backend.submission import derived_signing_key


STRUCTURE_ORIGIN_TOKEN_VERSION = 1
_SIGNING_PURPOSE = "structure-origin-token-v1"
MAX_TOKEN_LENGTH = 16 * 1024

ORIGIN_RETRY_SUGGESTION = (
    "Open the imported structure-search results again and hand the candidate "
    "off again, or paste the structure as a new calculation without an origin."
)


class StructureOriginError(CalculationValidationError):
    """The structure origin token is missing a valid signature or does not match."""


def normalized_structure_text(text: str) -> str:
    """
    Structure text in the form that is hashed.

    Browsers send textarea content with CRLF line endings and may drop
    trailing spaces, so line endings and trailing whitespace are normalised.
    Nothing else is changed.
    """

    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    return "\n".join(line.rstrip() for line in lines)


def structure_text_sha256(text: str) -> str:
    return hashlib.sha256(normalized_structure_text(text).encode("utf-8")).hexdigest()


def create_structure_origin_token(origin: dict, structure_text: str) -> str:
    """Sign ``origin`` (a JSON-safe dict) for this exact structure text."""

    if not isinstance(origin, dict) or not origin:
        raise ValueError("A structure origin must be a non-empty dict.")
    payload = {
        "version": STRUCTURE_ORIGIN_TOKEN_VERSION,
        "structure_sha256": structure_text_sha256(structure_text),
        "origin": origin,
    }
    encoded = _b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    signature = hmac.new(
        derived_signing_key(_SIGNING_PURPOSE), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    token = f"{encoded}.{_b64encode(signature)}"
    if len(token) > MAX_TOKEN_LENGTH:
        raise ValueError("The structure origin record is too large to carry in a form.")
    return token


def verify_structure_origin(token: str | None, structure_text: str) -> dict | None:
    """
    Return the origin record for ``structure_text``, or None if no token was sent.

    Raises :class:`StructureOriginError` (a ``CalculationValidationError``,
    so the workbench shows it like any other calculation error) when the
    token is malformed, its signature is invalid, or it was made for a
    different structure text.
    """

    value = str(token or "").strip()
    if not value:
        return None
    if len(value) > MAX_TOKEN_LENGTH or value.count(".") != 1:
        raise StructureOriginError(
            "The record of where this structure came from could not be read.",
            suggestion=ORIGIN_RETRY_SUGGESTION,
        )
    encoded, supplied = value.split(".", 1)
    expected = hmac.new(
        derived_signing_key(_SIGNING_PURPOSE), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    try:
        signature_ok = hmac.compare_digest(_b64decode(supplied), expected)
    except ValueError:
        signature_ok = False
    if not signature_ok:
        raise StructureOriginError(
            "The record of where this structure came from could not be verified. "
            "This happens after the server restarts, or if the page was altered.",
            suggestion=ORIGIN_RETRY_SUGGESTION,
        )
    try:
        payload = json.loads(_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise StructureOriginError(
            "The record of where this structure came from could not be read.",
            suggestion=ORIGIN_RETRY_SUGGESTION,
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != STRUCTURE_ORIGIN_TOKEN_VERSION
        or not isinstance(payload.get("origin"), dict)
    ):
        raise StructureOriginError(
            "The record of where this structure came from has an unsupported version.",
            suggestion=ORIGIN_RETRY_SUGGESTION,
        )
    if payload.get("structure_sha256") != structure_text_sha256(structure_text):
        raise StructureOriginError(
            "The structure was changed after it was taken from its source, so its "
            "recorded origin no longer applies.",
            suggestion=(
                "Hand the original candidate off again, or paste the edited structure "
                "as a new calculation in the Structure input."
            ),
        )
    return payload["origin"]


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value or ""):
        raise ValueError("Invalid base64url value.")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


__all__ = [
    "STRUCTURE_ORIGIN_TOKEN_VERSION",
    "StructureOriginError",
    "create_structure_origin_token",
    "normalized_structure_text",
    "structure_text_sha256",
    "verify_structure_origin",
]
