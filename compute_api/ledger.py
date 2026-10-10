"""Durable VM-side registry of machine-API submission attempts.

Remote attempt state on POWER (``logs/submission_attempts/<uuid>.json``) stays
the authority for whether an attempt is prepared, being submitted or
submitted. That record cannot by itself say *who* created an attempt through
the API, which plan digest the attempt was bound to, which run timestamp must
be reused to rebuild an identical submission specification, or how many jobs
a principal has submitted. This ledger records exactly those facts, one small
JSON file per attempt, in a deployment-local directory outside the
repository (``BMD_API_STATE_DIR``).

* An attempt record is created once, atomically (``O_CREAT | O_EXCL``), before
  any remote side effect. A concurrent duplicate request loses the race and
  re-reads the winner's record.
* Every change is written to a temporary file and atomically renamed.
* Read-modify-write updates and cap decisions run under an exclusive
  ``flock`` on ``ledger.lock`` (plus an in-process lock), so they hold across
  threads, worker processes and restarts.

The ledger never stores tokens, credentials or structure text. It does store
the server-built remote attempt-state path, which is never returned to clients.
"""

from __future__ import annotations

import json
import os
import re
import stat
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


STATE_DIR_ENV = "BMD_API_STATE_DIR"
ATTEMPT_RECORD_SCHEMA = "bmd_compute.api_attempt"
ATTEMPT_RECORD_SCHEMA_VERSION = 1
ATTEMPTS_DIRNAME = "attempts"
LOCK_FILENAME = "ledger.lock"
MAX_RECORD_BYTES = 256 * 1024

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_THREAD_LOCK = threading.RLock()


class LedgerUnavailable(Exception):
    """The execution ledger is not configured, unsafe or unreadable."""


def canonical_attempt_id(value: str) -> str | None:
    """Return ``value`` if it is a canonical lowercase RFC 4122 UUID, else None."""

    if not isinstance(value, str) or not _CANONICAL_UUID.fullmatch(value):
        return None
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    if str(parsed) != value or parsed.variant != uuid.RFC_4122:
        return None
    return value


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_isoformat(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


class AttemptLedger:
    def __init__(self, root: Path):
        self.root = root
        self.attempts_dir = root / ATTEMPTS_DIRNAME

    # ------------------------------------------------------------ configuration

    @classmethod
    def from_environment(cls, environ=None) -> "AttemptLedger":
        environ = os.environ if environ is None else environ
        value = str(environ.get(STATE_DIR_ENV) or "").strip()
        if not value or not Path(value).is_absolute():
            raise LedgerUnavailable("not configured")
        try:
            root = Path(value).resolve(strict=True)
        except (OSError, RuntimeError):
            raise LedgerUnavailable("missing") from None
        if root == _REPOSITORY_ROOT or _REPOSITORY_ROOT in root.parents:
            raise LedgerUnavailable("inside repository")
        cls._require_private_directory(root)
        ledger = cls(root)
        try:
            ledger.attempts_dir.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            raise LedgerUnavailable("cannot create attempts directory") from None
        cls._require_private_directory(ledger.attempts_dir)
        return ledger

    @staticmethod
    def _require_private_directory(path: Path) -> None:
        try:
            info = path.stat()
        except OSError:
            raise LedgerUnavailable("unreadable") from None
        if not stat.S_ISDIR(info.st_mode):
            raise LedgerUnavailable("not a directory")
        if os.name == "posix":
            if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise LedgerUnavailable("group or world accessible")
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                raise LedgerUnavailable("wrong owner")

    # ------------------------------------------------------------------ locking

    @contextmanager
    def locked(self) -> Iterator[None]:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - the service runs on Linux
            raise LedgerUnavailable("file locking unavailable") from None
        with _THREAD_LOCK:
            fd = os.open(self.root / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # ---------------------------------------------------------------- records

    def _path(self, attempt_id: str) -> Path:
        if canonical_attempt_id(attempt_id) is None:
            raise ValueError("attempt id must be a canonical UUID")
        return self.attempts_dir / f"{attempt_id}.json"

    def read(self, attempt_id: str) -> dict[str, Any] | None:
        path = self._path(attempt_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            raise LedgerUnavailable("unreadable record") from None
        if len(raw) > MAX_RECORD_BYTES:
            raise LedgerUnavailable("record too large")
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise LedgerUnavailable("malformed record") from None
        if (
            not isinstance(record, dict)
            or record.get("schema") != ATTEMPT_RECORD_SCHEMA
            or record.get("schema_version") != ATTEMPT_RECORD_SCHEMA_VERSION
            or record.get("attempt_id") != attempt_id
        ):
            raise LedgerUnavailable("unexpected record")
        return record

    def create(self, record: dict[str, Any]) -> bool:
        """Create a record atomically. Return False if it already exists."""

        path = self._path(record["attempt_id"])
        data = self._encode(record)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        self._fsync_directory()
        return True

    def replace(self, record: dict[str, Any]) -> None:
        """Atomically replace an existing record. Call while holding ``locked()``."""

        path = self._path(record["attempt_id"])
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(self._encode(record))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        self._fsync_directory()

    def records(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.attempts_dir.glob("*.json")):
            attempt_id = path.stem
            if canonical_attempt_id(attempt_id) is None:
                continue
            record = self.read(attempt_id)
            if record is not None:
                records.append(record)
        return records

    @staticmethod
    def _encode(record: dict[str, Any]) -> bytes:
        data = json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        if len(data) > MAX_RECORD_BYTES:
            raise LedgerUnavailable("record too large")
        return data

    def _fsync_directory(self) -> None:
        if os.name != "posix":
            return
        try:
            fd = os.open(self.attempts_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
