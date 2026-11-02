"""File protocol shared by ``bmd_bridge.sh`` and the host-side launcher.

``bmd_bridge.sh`` (run by USPEX inside the Singularity image) and
``launcher.py`` (run on the host, inside the SLURM allocation) talk only
through files in one spool directory. The full protocol is described at the
top of ``bmd_bridge.sh``; this module holds the directory names, exit codes
and request format so the Python side has one definition of them. A test
checks that the shell script uses the same exit codes.

Spool layout::

    <spool>/
        tmp/        files being written; renamed into place when complete
        requests/   <id>.req   published by the bridge, not yet claimed
        claimed/    <id>.req   claimed by the launcher (queued or running)
        done/       <id>.done  "status=<n>" plus details, written by the launcher
        cancel/     <id>       the bridge gave up on a claimed request
        withdrawn/  <id>.req   the bridge gave up before the launcher claimed it
        rejected/   requests the launcher refused (malformed or duplicate)
        logs/       <id>.log   stdout and stderr of each VASP step
        heartbeat   Unix time of the launcher's last check-in
        stopped     present once the launcher has stopped
        launcher.jsonl  one JSON object per launcher event
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


TMP_DIR = "tmp"
REQUESTS_DIR = "requests"
CLAIMED_DIR = "claimed"
DONE_DIR = "done"
CANCEL_DIR = "cancel"
WITHDRAWN_DIR = "withdrawn"
REJECTED_DIR = "rejected"
LOGS_DIR = "logs"
SPOOL_SUBDIRS = (
    TMP_DIR,
    REQUESTS_DIR,
    CLAIMED_DIR,
    DONE_DIR,
    CANCEL_DIR,
    WITHDRAWN_DIR,
    REJECTED_DIR,
    LOGS_DIR,
)
HEARTBEAT_FILE = "heartbeat"
STOPPED_FILE = "stopped"
JOURNAL_FILE = "launcher.jsonl"

REQUEST_SUFFIX = ".req"
DONE_SUFFIX = ".done"

# Exit codes. The same numbers appear in bmd_bridge.sh.
EXIT_USAGE = 120
EXIT_LAUNCHER_LOST = 121
EXIT_REJECTED = 122
EXIT_STOPPED = 123
EXIT_TIMEOUT = 124
EXIT_SIGNAL = 143

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RequestError(ValueError):
    """A request file is malformed."""


@dataclass(frozen=True)
class StepRequest:
    """One relaxation step that USPEX asked to run, as read from a request file."""

    request_id: str
    step: int
    calcfold: Path
    created: int


def parse_request(text: str, *, expected_id: str | None = None) -> StepRequest:
    """Parse the ``key=value`` lines that ``bmd_bridge.sh`` writes.

    Raises ``RequestError`` if a field is missing or invalid, or if the ID
    inside the file does not match ``expected_id`` (the file name).
    """

    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise RequestError(f"line without '=': {line!r}")
        if key in fields:
            raise RequestError(f"repeated field {key!r}")
        fields[key] = value

    missing = [key for key in ("id", "step", "calcfold", "created") if key not in fields]
    if missing:
        raise RequestError(f"missing fields: {', '.join(missing)}")

    request_id = fields["id"]
    if not REQUEST_ID_PATTERN.match(request_id):
        raise RequestError(f"invalid request id {request_id!r}")
    if expected_id is not None and request_id != expected_id:
        raise RequestError(f"request id {request_id!r} does not match file name {expected_id!r}")
    if not fields["step"].isdigit() or int(fields["step"]) < 1:
        raise RequestError(f"invalid step {fields['step']!r}")
    if not fields["created"].isdigit():
        raise RequestError(f"invalid created time {fields['created']!r}")
    calcfold = fields["calcfold"]
    if not calcfold.startswith("/"):
        raise RequestError(f"CalcFold must be an absolute path, got {calcfold!r}")

    return StepRequest(
        request_id=request_id,
        step=int(fields["step"]),
        calcfold=Path(calcfold),
        created=int(fields["created"]),
    )


def request_id_from_filename(name: str, suffix: str = REQUEST_SUFFIX) -> str | None:
    """Return the request ID for ``<id><suffix>``, or ``None`` if the name is not one."""

    if not name.endswith(suffix):
        return None
    request_id = name[: -len(suffix)]
    return request_id if REQUEST_ID_PATTERN.match(request_id) else None


def write_atomically(spool: Path, target: Path, text: str) -> None:
    """Write ``text`` to ``target`` so readers see either nothing or the whole file.

    The text goes to a temporary file in ``<spool>/tmp`` (same file system),
    is flushed to disk, and is then renamed onto ``target``.
    """

    tmp = spool / TMP_DIR / f".launcher-{os.getpid()}-{target.name}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


__all__ = [
    "CANCEL_DIR",
    "CLAIMED_DIR",
    "DONE_DIR",
    "DONE_SUFFIX",
    "EXIT_LAUNCHER_LOST",
    "EXIT_REJECTED",
    "EXIT_SIGNAL",
    "EXIT_STOPPED",
    "EXIT_TIMEOUT",
    "EXIT_USAGE",
    "HEARTBEAT_FILE",
    "JOURNAL_FILE",
    "LOGS_DIR",
    "REJECTED_DIR",
    "REQUESTS_DIR",
    "REQUEST_ID_PATTERN",
    "REQUEST_SUFFIX",
    "RequestError",
    "SPOOL_SUBDIRS",
    "STOPPED_FILE",
    "StepRequest",
    "TMP_DIR",
    "WITHDRAWN_DIR",
    "parse_request",
    "request_id_from_filename",
    "write_atomically",
]
