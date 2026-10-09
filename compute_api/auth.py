"""Bearer-token authentication for the machine API.

Token material never lives in the repository. The service reads a small JSON
file named by ``BMD_API_TOKENS_FILE``. Each entry names a principal, a token
identifier, the SHA-256 verifier of the full token and the principal's scopes.
The plaintext token is shown once, when it is generated, and is never stored.

Tokens have the form ``bmdc1.<token_id>.<secret>``, where ``secret`` is 32
random bytes (base64url). Because tokens are high-entropy random values, an
unsalted SHA-256 verifier is sufficient; a slow password hash is not needed.
Verifiers are compared in constant time, and an unknown token identifier still
performs a comparison so that the response time does not reveal which
identifiers exist.

Everything fails closed: no configured file, an unreadable or malformed file,
an unsafe file location or permission, an unknown scope, a missing, malformed,
unknown or disabled token, and a principal without the required scope are all
refused before any request body is read.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path


TOKENS_FILE_ENV = "BMD_API_TOKENS_FILE"
TOKENS_FILE_SCHEMA = "bmd_compute.api_tokens"
TOKENS_FILE_SCHEMA_VERSION = 1
TOKENS_FILE_MAX_BYTES = 64 * 1024
TOKEN_PREFIX = "bmdc1"

SCOPE_READ = "read"
SCOPE_PLAN = "plan"
SCOPE_PREPARE = "prepare"
SCOPE_SUBMIT = "submit"
# The complete scope vocabulary. "plan" never grants execution; "prepare" and
# "submit" are checked independently by the attempt routes.
KNOWN_SCOPES = frozenset({SCOPE_READ, SCOPE_PLAN, SCOPE_PREPARE, SCOPE_SUBMIT})

PRINCIPAL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
TOKEN_ID_PATTERN = re.compile(r"^[a-z0-9]{8,32}$")
SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")
VERIFIER_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_AUTHORIZATION_HEADER_CHARS = 256

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DUMMY_VERIFIER = "sha256:" + "0" * 64


class ApiAuthError(Exception):
    """Authentication or authorization refusal. Messages never contain token material."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Principal:
    principal: str
    token_id: str
    scopes: frozenset[str]

    def require(self, scope: str) -> None:
        if scope not in KNOWN_SCOPES:
            raise ApiAuthError(500, "server_misconfigured", "The requested scope is not defined.")
        if scope not in self.scopes:
            raise ApiAuthError(
                403,
                "insufficient_scope",
                f"This token does not have the '{scope}' scope.",
            )


@dataclass(frozen=True)
class _TokenEntry:
    principal: str
    token_id: str
    verifier: str
    scopes: frozenset[str]
    enabled: bool


def _not_configured() -> ApiAuthError:
    return ApiAuthError(
        503,
        "api_not_configured",
        "The BMD Compute machine API is not configured on this service.",
    )


def tokens_file_path(environ=None) -> Path:
    environ = os.environ if environ is None else environ
    value = str(environ.get(TOKENS_FILE_ENV) or "").strip()
    if not value:
        raise _not_configured()
    path = Path(value)
    if not path.is_absolute():
        raise _not_configured()
    return path


def _require_safe_location(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise _not_configured() from None
    if resolved == _REPOSITORY_ROOT or _REPOSITORY_ROOT in resolved.parents:
        # Token verifiers must never sit in (or be committed from) the checkout.
        raise _not_configured()
    try:
        info = resolved.stat()
    except OSError:
        raise _not_configured() from None
    if not stat.S_ISREG(info.st_mode) or info.st_size > TOKENS_FILE_MAX_BYTES:
        raise _not_configured()
    if os.name == "posix":
        if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise _not_configured()
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise _not_configured()
    return resolved


def load_token_entries(path: Path) -> dict[str, _TokenEntry]:
    resolved = _require_safe_location(path)
    try:
        raw = resolved.read_bytes()
    except OSError:
        raise _not_configured() from None
    if len(raw) > TOKENS_FILE_MAX_BYTES:
        raise _not_configured()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _not_configured() from None
    return _parse_token_document(document)


def _parse_token_document(document) -> dict[str, _TokenEntry]:
    if not isinstance(document, dict) or set(document) != {"schema", "schema_version", "principals"}:
        raise _not_configured()
    if document["schema"] != TOKENS_FILE_SCHEMA or document["schema_version"] != TOKENS_FILE_SCHEMA_VERSION:
        raise _not_configured()
    principals = document["principals"]
    if not isinstance(principals, list):
        raise _not_configured()

    entries: dict[str, _TokenEntry] = {}
    for item in principals:
        if not isinstance(item, dict) or set(item) != {"principal", "token_id", "verifier", "scopes", "enabled"}:
            raise _not_configured()
        principal = item["principal"]
        token_id = item["token_id"]
        verifier = item["verifier"]
        scopes = item["scopes"]
        enabled = item["enabled"]
        if not (isinstance(principal, str) and PRINCIPAL_PATTERN.fullmatch(principal)):
            raise _not_configured()
        if not (isinstance(token_id, str) and TOKEN_ID_PATTERN.fullmatch(token_id)):
            raise _not_configured()
        if not (isinstance(verifier, str) and VERIFIER_PATTERN.fullmatch(verifier)):
            raise _not_configured()
        if not isinstance(enabled, bool):
            raise _not_configured()
        if (
            not isinstance(scopes, list)
            or not all(isinstance(scope, str) for scope in scopes)
            or len(set(scopes)) != len(scopes)
            or not set(scopes) <= KNOWN_SCOPES
        ):
            # An unknown scope is a configuration error, never silently ignored.
            raise _not_configured()
        if token_id in entries:
            raise _not_configured()
        entries[token_id] = _TokenEntry(
            principal=principal,
            token_id=token_id,
            verifier=verifier,
            scopes=frozenset(scopes),
            enabled=enabled,
        )
    return entries


def token_verifier(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()


def _unauthenticated() -> ApiAuthError:
    return ApiAuthError(
        401,
        "unauthenticated",
        "A valid BMD Compute API bearer token is required.",
    )


def _bearer_token(authorization_values: list[str]) -> str:
    if len(authorization_values) != 1:
        raise _unauthenticated()
    value = authorization_values[0]
    if len(value) > MAX_AUTHORIZATION_HEADER_CHARS:
        raise _unauthenticated()
    scheme, separator, token = value.partition(" ")
    if separator != " " or scheme.lower() != "bearer":
        raise _unauthenticated()
    return token.strip()


def _split_token(token: str) -> tuple[str, str] | None:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        return None
    token_id, secret = parts[1], parts[2]
    if not TOKEN_ID_PATTERN.fullmatch(token_id) or not SECRET_PATTERN.fullmatch(secret):
        return None
    return token_id, secret


def authenticate(authorization_values: list[str], *, environ=None) -> Principal:
    """Return the authenticated principal or raise :class:`ApiAuthError`.

    The token store is read on every call, so editing the file (for example
    setting ``enabled`` to false) takes effect on the next request.
    """

    entries = load_token_entries(tokens_file_path(environ))
    token = _bearer_token(authorization_values)
    parts = _split_token(token)
    token_id = parts[0] if parts is not None else None
    entry = entries.get(token_id) if token_id is not None else None
    supplied = token_verifier(token) if parts is not None else _DUMMY_VERIFIER
    expected = entry.verifier if entry is not None else _DUMMY_VERIFIER
    matches = hmac.compare_digest(supplied.encode("ascii"), expected.encode("ascii"))
    if parts is None or entry is None or not matches or not entry.enabled:
        raise _unauthenticated()
    return Principal(principal=entry.principal, token_id=entry.token_id, scopes=entry.scopes)


def generate_token() -> tuple[str, str, str]:
    """Return ``(token, token_id, verifier)`` for a new token. Nothing is stored."""

    token_id = secrets.token_hex(8)
    secret = secrets.token_urlsafe(32)
    token = f"{TOKEN_PREFIX}.{token_id}.{secret}"
    return token, token_id, token_verifier(token)
