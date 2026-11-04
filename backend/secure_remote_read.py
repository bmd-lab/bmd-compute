"""Contained, bounded read of one regular file under a trusted directory.

Used only for the adaptive geometry-convergence record of a failed job
(``backend.monitoring.geometry_convergence_diagnostic``). It is deliberately
narrow and does not change the general-purpose remote file methods.

The source of this module is self-contained (standard library only) so the
same code runs on the remote host (``ParamikoRemoteRunner.read_contained_file``
sends it with ``run_python``) and in local tests.

The whole absolute path is walked from ``/``: every directory component (of
``root`` and below it) is opened relative to its already-opened parent with
``O_NOFOLLOW | O_DIRECTORY``, and the file with ``O_NOFOLLOW``, so a symbolic
link anywhere in the path - an ancestor of ``root``, ``root`` itself, a
directory below it or the file - is refused. Only safe component names are
accepted. The file must be a regular file of at most ``max_bytes`` bytes, and
the content is read from the descriptor that was checked - a path replaced
after it was opened cannot substitute its content, and growth during the read
is refused. Without the Linux flags the read fails closed.
"""

from __future__ import annotations

import os
import stat

SAFE_COMPONENT_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class ContainedReadError(Exception):
    """The file is absent, not contained, not regular, or too large."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _valid_component(name: str) -> bool:
    return bool(name) and name not in (".", "..") and set(name) <= SAFE_COMPONENT_CHARACTERS


def read_contained_regular_file(root: str, parts, max_bytes: int, _after_open=None) -> bytes:
    """``root/parts...`` as bytes, or raise :class:`ContainedReadError`.

    ``_after_open`` is a test hook called with the checked path after the
    final descriptor is opened and verified.
    """

    parts = tuple(parts)
    if not parts or not all(isinstance(part, str) and _valid_component(part) for part in parts):
        raise ContainedReadError("invalid_path")
    if not isinstance(root, str) or not root.startswith("/"):
        raise ContainedReadError("invalid_root")
    root_parts = tuple(root[1:].split("/")) if root != "/" else ()
    if not all(_valid_component(part) for part in root_parts):
        # Empty (//, trailing /), ".", ".." or unsafe names: never normalised.
        raise ContainedReadError("invalid_root")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag or os.open not in getattr(os, "supports_dir_fd", set()):
        raise ContainedReadError("unsupported_platform")
    descriptors = []
    try:
        try:
            current = os.open("/", os.O_RDONLY | directory_flag | nofollow)
        except OSError:
            raise ContainedReadError("root_unavailable") from None
        descriptors.append(current)
        for part in root_parts:
            # Each ancestor and the root itself, relative to its verified parent.
            try:
                current = os.open(part, os.O_RDONLY | directory_flag | nofollow, dir_fd=current)
            except OSError:
                raise ContainedReadError("root_unavailable") from None
            descriptors.append(current)
        for part in parts[:-1]:
            try:
                current = os.open(part, os.O_RDONLY | directory_flag | nofollow, dir_fd=current)
            except FileNotFoundError:
                raise ContainedReadError("missing") from None
            except OSError:
                raise ContainedReadError("not_contained") from None
            descriptors.append(current)
        nonblock = getattr(os, "O_NONBLOCK", 0)  # never block opening a FIFO
        try:
            handle = os.open(parts[-1], os.O_RDONLY | nofollow | nonblock, dir_fd=current)
        except FileNotFoundError:
            raise ContainedReadError("missing") from None
        except OSError:
            # ELOOP: the final component is a symbolic link.
            raise ContainedReadError("not_regular") from None
        descriptors.append(handle)
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode):
            raise ContainedReadError("not_regular")
        if info.st_size > max_bytes:
            raise ContainedReadError("too_large")
        if _after_open is not None:
            _after_open(os.path.join(root, *parts))
        chunks, total = [], 0
        while True:
            chunk = os.read(handle, min(65536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                # Grew after the size check: reject rather than truncate.
                raise ContainedReadError("too_large")
        return b"".join(chunks)
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def remote_source(root: str, parts, max_bytes: int) -> str:
    """Python source that performs the read remotely and prints one JSON line."""

    import inspect
    import json
    import sys

    module_source = inspect.getsource(sys.modules[__name__])
    call = json.dumps({"root": root, "parts": list(parts), "max_bytes": int(max_bytes)})
    return (
        module_source
        + "\n\nif __name__ == '__main__':\n"
        + "    import base64 as _b64, json as _json\n"
        + f"    _args = _json.loads({call!r})\n"
        + "    try:\n"
        + "        _data = read_contained_regular_file(_args['root'], _args['parts'], _args['max_bytes'])\n"
        + "        print(_json.dumps({'ok': True, 'data': _b64.b64encode(_data).decode('ascii')}))\n"
        + "    except ContainedReadError as _exc:\n"
        + "        print(_json.dumps({'ok': False, 'code': _exc.code}))\n"
    )


def parse_remote_output(stdout: str, max_bytes: int) -> bytes:
    """Bytes from :func:`remote_source` output, or raise :class:`ContainedReadError`."""

    import base64
    import json

    lines = [line for line in (stdout or "").splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise ContainedReadError("remote_failure") from None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        code = payload.get("code") if isinstance(payload, dict) else None
        raise ContainedReadError(str(code) if code else "remote_failure")
    try:
        data = base64.b64decode(str(payload.get("data") or ""), validate=True)
    except ValueError:
        raise ContainedReadError("remote_failure") from None
    if len(data) > max_bytes:
        raise ContainedReadError("too_large")
    return data


__all__ = ["ContainedReadError", "parse_remote_output", "read_contained_regular_file", "remote_source"]
