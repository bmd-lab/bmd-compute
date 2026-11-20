"""Remote staging of a locally prepared structure search on POWER (USPEX step 4C).

Two halves:

* **Service side** (:func:`staging_layout`, :func:`build_search_submission_spec`,
  :func:`staged_search_files`, :func:`build_finalize_script`): where a staged
  search lives, its submission record, the bound search inputs to upload and
  the script that finishes staging on POWER. Used by
  ``compute_api.structure_search_execution``.
* **POWER side** (:func:`finalize_staged_search`): runs on the login node as
  ``bmdguest``, from the runtime package uploaded with the attempt, after the
  embedded guard has verified that package against its manifest. It copies
  the licensed POTCARs from the POWER library into the private search
  directory and verifies everything; it returns identities and hashes only,
  never POTCAR contents. It does not start USPEX, VASP or SLURM.

Layout of one staged attempt (``STRUCTURE_SEARCH_WORKSPACE_DIR``, under ``/bmd``)::

    /bmd/bmdguest/searches/<run_name>/
        submission.json        the search submission record (attempt identity, plan digest)
        backend/               the runtime package uploaded with this attempt
            structure_search/bridge/bmd_bridge.sh    the bound launch bridge
        search/                the USPEX run directory
            INPUT.txt          bound to the bridge above
            Specific/INCAR_1..5, Specific/POTCAR_<El> (copied on POWER, mode 0600)
            POTCAR.spec, search_plan.json, prepared_inputs.json (bound)

and its remote attempt state, with BMD's logs but apart from calculations::

    /bmd-db/guest/logs/structure_search/submission_attempts/<attempt_id>.json   (STAGING, STAGED)

``STAGED`` is not ``PREPARED``: the existing claim and ``sbatch`` path
(``ParamikoRemoteRunner``) accepts only ``PREPARED``, and refuses to re-prepare
an attempt in any other state, so a staged search can be neither submitted
nor overwritten by it. Submission-ready preparation is step 4D.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import posixpath
import re
import stat
from pathlib import Path
from typing import Any, Callable, Mapping

SEARCH_SUBMISSION_SCHEMA = "bmd_compute.structure_search_submission"
SEARCH_SUBMISSION_SCHEMA_VERSION = 1
RUN_LABEL = "uspex_search"
SEARCH_SUBDIR = "search"
BRIDGE_RELATIVE_PATH = "structure_search/bridge/bmd_bridge.sh"
REMOTE_STATE_STAGING = "STAGING"
REMOTE_STATE_STAGED = "STAGED"
RESULT_MARKER = "BMD_SEARCH_STAGING_RESULT="
REQUEST_MARKER = "BMD_SEARCH_STAGING_REQUEST_B64 = "
STAGED_POTCAR_MODE = 0o600
UPLOAD_MODE = 0o640
INVENTORY_MARKER = "BMD_SEARCH_STAGING_INVENTORY="
INVENTORY_DIR_MARKER = "BMD_SEARCH_STAGING_INVENTORY_DIR = "
TEMPORARY_TAG = "bmdstage"
_TEMPORARY_RE = re.compile(
    r"^\.(?P<target>.+)\.bmdstage-(?P<attempt>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"-(?P<tag>[0-9a-f]{16})\.part$"
)
# What STAGED means: the state POWER last verified, not a live guarantee.
STAGED_REMOTE_INTEGRITY = "last_verified"


class StagingError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# -- service side ------------------------------------------------------------------


def staging_layout(run_name: str, *, workspace_dir: str | None = None, logs_dir: str | None = None,
                   package_dir_name: str = "backend") -> dict[str, str]:
    from backend import config
    from backend.structure_search.bridge.master import DEFAULT_BIND

    workspace = workspace_dir or config.STRUCTURE_SEARCH_WORKSPACE_DIR
    logs = logs_dir or config.STRUCTURE_SEARCH_LOGS_DIR
    if not posixpath.isabs(workspace) or not (workspace + "/").startswith(DEFAULT_BIND + "/"):
        raise StagingError("workspace_not_under_bind", f"The search workspace must be under {DEFAULT_BIND}.")
    if "/" in run_name or run_name in {"", ".", ".."}:
        raise StagingError("invalid_run_name", "Invalid run name.")
    run_dir = posixpath.join(workspace, run_name)
    package_dir = posixpath.join(run_dir, package_dir_name)
    return {
        "workspace_dir": workspace,
        "logs_dir": logs,
        "run_dir": run_dir,
        "package_dir": package_dir,
        "search_dir": posixpath.join(run_dir, SEARCH_SUBDIR),
        "bridge_path": posixpath.join(package_dir, BRIDGE_RELATIVE_PATH),
        "submission_json": posixpath.join(run_dir, "submission.json"),
    }


def build_search_submission_spec(*, plan, attempt_id: str, run_timestamp: str, created_at: str,
                                 runtime_manifest: Mapping[str, str]) -> dict[str, Any]:
    """The search's submission record, shaped as BMD's attempt-state helpers expect.

    ``initialize_submission_attempt`` places the attempt state under
    ``<logs_dir>/submission_attempts/`` (here the search logs directory) and
    computes the attempt fingerprint, as for calculations.
    """

    from backend.config import NOTEBOOK_DEFAULTS
    from backend.submission import initialize_submission_attempt, sanitize_label

    run_name = f"{sanitize_label(RUN_LABEL)}-{run_timestamp}-{attempt_id[:8]}"
    layout = staging_layout(run_name)
    record = plan.record
    spec: dict[str, Any] = {
        "schema": SEARCH_SUBMISSION_SCHEMA,
        "schema_version": SEARCH_SUBMISSION_SCHEMA_VERSION,
        "label": RUN_LABEL,
        "run_name": run_name,
        "created_at": created_at,
        "flow_spec": {"desired_output": record["desired_output"], "structure_search_plan_digest": plan.digest},
        "paths": {
            "run_dir": layout["run_dir"],
            "flows_dir": layout["workspace_dir"],
            "logs_dir": layout["logs_dir"],
            "search_dir": layout["search_dir"],
            "remote_script": None,
            "result_dir": None,
            "stage_dirs": {},
        },
        "cluster": {
            key: NOTEBOOK_DEFAULTS.get(key)
            for key in ("ssh_config_host", "remote_host", "username", "port")
        },
        "resources": dict(record["resources"]["request"]),
        "environment": {},
        "modules": [],
        "runner": {"python": NOTEBOOK_DEFAULTS["remote_python"], "backend_package_dir": "backend"},
        "potcar": {
            "functional": record["potcar"]["functional"],
            "library": NOTEBOOK_DEFAULTS["PMG_VASP_PSP_DIR"],
            "elements": [
                {key: row[key] for key in ("element", "symbol", "TITEL", "ENMAX", "dataset_sha256")}
                for row in record["potcar"]["elements"]
            ],
            "contains_potcar_contents": False,
        },
        "runtime_parity": None,
        "structure_search": {
            "plan_digest": plan.digest,
            "runtime": record["runtime"],
            "bridge_path": layout["bridge_path"],
            "bridge_sha256": record["execution"]["bridge"]["sha256"],
            "search_dir": layout["search_dir"],
            "state": REMOTE_STATE_STAGED,
            "submission_available": False,
        },
        "provenance": {"bmd_compute": {"runtime_source": {"manifest": dict(runtime_manifest)}}},
    }
    return initialize_submission_attempt(spec, attempt_id=attempt_id)


def staged_search_files(plan, local_manifest: Mapping[str, Any], *, bridge_path: str) -> dict[str, bytes]:
    """The search inputs as staged: the local preparation, re-rendered bound to ``bridge_path``.

    The INCARs, POTCAR.spec and the frozen plan are the same bytes as the
    local preparation; INPUT.txt and prepared_inputs.json differ only in the
    binding. The local preparation itself is never changed.
    """

    from backend.structure_search.search_plan import render_prepared_files

    return render_prepared_files(plan, bridge_path=bridge_path, generator=local_manifest.get("generator") or {})


def build_finalize_script(request: Mapping[str, Any], *, runtime_sources: Mapping[str, str],
                          guard_module: str) -> str:
    """Python run on POWER: verify the uploaded package, then finish staging with it."""

    payload = base64.b64encode(json.dumps(dict(request), sort_keys=True).encode("utf-8")).decode("ascii")
    return "\n".join(
        [
            "import sys",
            'if sys.path and sys.path[0] == "":',
            "    del sys.path[0]",
            "import base64, json",
            f"{REQUEST_MARKER}{payload!r}",
            "REQUEST = json.loads(base64.b64decode(BMD_SEARCH_STAGING_REQUEST_B64))",
            "# BMD Compute runtime package guard (embedded).",
            runtime_sources[guard_module].strip(),
            "# End of embedded runtime package guard.",
            "sys.path.insert(0, REQUEST['run_dir'])",
            "bmd_install_verified_runtime_package(REQUEST['package_dir'], REQUEST['manifest'])",
            "from backend.structure_search.staging import finalize_staged_search_main",
            "finalize_staged_search_main(REQUEST)",
        ]
    ) + "\n"


def parse_finalize_output(output: str) -> dict[str, Any]:
    lines = [line for line in (output or "").splitlines() if line.startswith(RESULT_MARKER)]
    if len(lines) != 1:
        raise StagingError("staging_unconfirmed", "Staging on POWER did not report a result.")
    try:
        result = json.loads(lines[0][len(RESULT_MARKER):])
    except ValueError:
        raise StagingError("staging_unconfirmed", "Staging on POWER reported an unreadable result.") from None
    if not isinstance(result, dict):
        raise StagingError("staging_unconfirmed", "Staging on POWER reported an unreadable result.")
    return result


def request_from_script(script: str) -> dict[str, Any]:
    """The request embedded in a finalize script (for test doubles of POWER)."""

    for line in script.splitlines():
        if line.startswith(REQUEST_MARKER):
            import ast

            return json.loads(base64.b64decode(ast.literal_eval(line[len(REQUEST_MARKER):])))
    raise ValueError("no staging request in script")


# -- interrupted uploads ---------------------------------------------------------
#
# Every file staging writes goes first to a temporary sibling whose name binds
# it to one attempt and one upload operation (the target and the first 16 hex
# digits of the content's SHA-256), then is moved into place:
#
#     .<target name>.bmdstage-<attempt id>-<sha256[:16]>.part
#
# After an interruption the next request, holding the same remote claim, can
# therefore tell its own unfinished uploads from anything else: a temporary of
# this attempt for a missing target and the same content is completed (renamed
# when its bytes are complete, rewritten in place when they are not); anything
# it cannot attribute unambiguously is refused and left untouched.


def staging_temporary_name(target_name: str, attempt_id: str, sha256_hex: str) -> str:
    return f".{target_name}.{TEMPORARY_TAG}-{attempt_id}-{sha256_hex[:16]}.part"


def staging_temporary_path(remote_path: str, attempt_id: str, sha256_hex: str) -> str:
    directory, name = posixpath.split(remote_path)
    return posixpath.join(directory, staging_temporary_name(name, attempt_id, sha256_hex))


def parse_staging_temporary(name: str) -> dict[str, str] | None:
    match = _TEMPORARY_RE.match(name)
    return match.groupdict() if match else None


def _looks_temporary(name: str) -> bool:
    return name.startswith(".") and (
        name.endswith((".tmp", ".part")) or TEMPORARY_TAG in name or ".staging-" in name
    )


def reconcile_staging(
    inventory: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    attempt_id: str,
    potcars: Mapping[str, str],
    complete: bool = False,
) -> dict[str, Any]:
    """Decide, from an inventory of a run directory, what staging may still do.

    ``expected`` maps run-directory-relative paths of uploaded files to their
    SHA-256; ``potcars`` maps the relative paths of the POTCARs the POWER side
    copies to the first 16 hex digits of the planned dataset SHA-256. Returns
    ``{"present": [...], "missing": {path: {"temporary": name or None,
    "temporary_complete": bool}}}``. Raises :class:`StagingError`:
    ``staging_damaged`` for a completed file that differs, a symbolic link or
    anything else not part of this staging, and ``staging_ambiguous`` for a
    temporary file that cannot be attributed to exactly one of this attempt's
    upload operations. With ``complete=True`` every expected file and POTCAR
    must be present and no temporary may remain.
    """

    entries = dict(inventory.get("entries") or {})
    allowed_dirs = {""}
    for relative in list(expected) + list(potcars):
        parent = posixpath.dirname(relative)
        while parent and parent not in allowed_dirs:
            allowed_dirs.add(parent)
            parent = posixpath.dirname(parent)

    present: list[str] = []
    temporaries: dict[str, dict[str, Any]] = {}
    for relative, entry in sorted(entries.items()):
        kind = entry.get("kind")
        directory, name = posixpath.split(relative)
        if kind == "dir":
            if relative not in allowed_dirs:
                raise StagingError("staging_damaged", f"Unexpected directory {relative} in the run directory.")
            continue
        if kind != "file":
            raise StagingError("staging_damaged", f"{relative} is not a regular file.")
        if relative in expected:
            if entry.get("sha256") != expected[relative]:
                raise StagingError(
                    "staging_damaged",
                    f"{relative} differs from what this attempt uploads; it is not replaced.",
                )
            present.append(relative)
            continue
        if relative in potcars:
            continue  # identity is checked on POWER
        parsed = parse_staging_temporary(name)
        if parsed is None:
            if _looks_temporary(name):
                raise StagingError("staging_ambiguous",
                                   f"Temporary file {relative} does not belong to a known staging upload.")
            raise StagingError("staging_damaged", f"Unexpected file {relative} in the run directory.")
        target = posixpath.join(directory, parsed["target"])
        if parsed["attempt"] != attempt_id:
            raise StagingError("staging_ambiguous", f"Temporary file {relative} belongs to another attempt.")
        if target in potcars:
            if parsed["tag"] != potcars[target]:
                raise StagingError("staging_ambiguous", f"Temporary file {relative} is not for the planned POTCAR.")
            if complete:
                raise StagingError("staging_incomplete", f"A POTCAR copy for {target} is unfinished.")
            continue  # completed or superseded by the POWER side
        if target not in expected or parsed["tag"] != expected[target][:16]:
            raise StagingError("staging_ambiguous", f"Temporary file {relative} is not one of this attempt's uploads.")
        if target in entries:
            raise StagingError(
                "staging_ambiguous",
                f"{target} is complete but a temporary upload of it remains; it is left for inspection.",
            )
        temporaries[target] = {
            "temporary": relative,
            "temporary_complete": entry.get("sha256") == expected[target]
            and entry.get("mode") == UPLOAD_MODE,
        }

    missing = {
        relative: temporaries.get(relative, {"temporary": None, "temporary_complete": False})
        for relative in sorted(expected)
        if relative not in present
    }
    if complete:
        absent = sorted(missing) + sorted(path for path in potcars if path not in entries)
        if absent:
            raise StagingError("staging_incomplete", f"{len(absent)} staged file(s) are missing: {absent[0]} ...")
    return {"present": present, "missing": missing}


def staging_inventory(run_dir):
    """Every entry below ``run_dir``: kind, mode and SHA-256 of regular files.

    Self-contained (standard library only): it is sent to POWER as source and
    run there before the runtime package is trusted. It follows no symbolic
    link and returns no file contents. ``__pycache__`` directories are skipped
    (the runtime package guard ignores them too).
    """

    import hashlib as _hashlib
    import os as _os
    import stat as _stat

    entries = {}
    pending = [("", str(run_dir))]
    while pending:
        prefix, directory = pending.pop()
        with _os.scandir(directory) as iterator:
            items = sorted(iterator, key=lambda item: item.name)
        for item in items:
            relative = prefix + item.name
            info = item.stat(follow_symlinks=False)
            mode = _stat.S_IMODE(info.st_mode)
            if _stat.S_ISLNK(info.st_mode):
                entries[relative] = {"kind": "symlink"}
            elif _stat.S_ISDIR(info.st_mode):
                if item.name == "__pycache__":
                    continue
                entries[relative] = {"kind": "dir", "mode": mode}
                pending.append((relative + "/", item.path))
            elif _stat.S_ISREG(info.st_mode):
                digest = _hashlib.sha256()
                with open(item.path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                entries[relative] = {"kind": "file", "mode": mode, "size": info.st_size,
                                     "sha256": digest.hexdigest()}
            else:
                entries[relative] = {"kind": "other"}
    return {"entries": entries}


def build_inventory_script(run_dir: str) -> str:
    import inspect

    return "\n".join(
        [
            "import sys",
            'if sys.path and sys.path[0] == "":',
            "    del sys.path[0]",
            "import json",
            f"{INVENTORY_DIR_MARKER}{run_dir!r}",
            inspect.getsource(staging_inventory).strip(),
            f"print({INVENTORY_MARKER!r} + json.dumps(staging_inventory({INVENTORY_DIR_MARKER.split(' =')[0]}), "
            "sort_keys=True))",
        ]
    ) + "\n"


def inventory_dir_from_script(script: str) -> str:
    """The run directory an inventory script lists (for test doubles of POWER)."""

    import ast

    for line in script.splitlines():
        if line.startswith(INVENTORY_DIR_MARKER):
            return ast.literal_eval(line[len(INVENTORY_DIR_MARKER):])
    raise ValueError("no inventory directory in script")


def parse_inventory_output(output: str) -> dict[str, Any]:
    lines = [line for line in (output or "").splitlines() if line.startswith(INVENTORY_MARKER)]
    if len(lines) != 1:
        raise StagingError("staging_unconfirmed", "POWER did not list the run directory.")
    try:
        inventory = json.loads(lines[0][len(INVENTORY_MARKER):])
    except ValueError:
        raise StagingError("staging_unconfirmed", "POWER listed the run directory unreadably.") from None
    if not isinstance(inventory, dict) or not isinstance(inventory.get("entries"), dict):
        raise StagingError("staging_unconfirmed", "POWER listed the run directory unreadably.")
    return inventory


# -- POWER side --------------------------------------------------------------------


def _on(root: str, path: str) -> Path:
    return Path(root) / str(path).lstrip("/")


def _copy_exclusive(source: Path, destination: Path, temporary: Path, mode: int) -> None:
    """Copy ``source`` to ``destination`` without ever replacing an existing file.

    ``temporary`` is this attempt's own temporary name for this POTCAR (see
    :func:`staging_temporary_name`). A leftover of it from an interrupted copy
    is superseded in place; the finished copy is linked into place, which
    fails if ``destination`` exists, and only then is the temporary removed.
    """

    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    if os.path.lexists(temporary):
        if not stat.S_ISREG(os.lstat(temporary).st_mode):
            raise StagingError("staging_ambiguous", f"{temporary.name} is not a regular file.")
        flags |= os.O_TRUNC
    else:
        flags |= os.O_EXCL
    fd = os.open(temporary, flags, mode)
    with os.fdopen(fd, "wb") as handle, open(source, "rb") as reader:
        for chunk in iter(lambda: reader.read(1 << 20), b""):
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.link(temporary, destination)  # fails if destination exists
    os.unlink(temporary)


def _finish_interrupted_link(destination: Path, temporary: Path) -> None:
    """A copy interrupted after its link and before removing the temporary."""

    if not os.path.lexists(temporary):
        return
    if not os.path.samefile(temporary, destination) or os.path.islink(temporary):
        raise StagingError(
            "staging_ambiguous",
            f"{temporary.name} remains beside a different {destination.name}; both are left for inspection.",
        )
    os.unlink(temporary)  # the same file as the completed destination


def finalize_staged_search(
    request: Mapping[str, Any],
    *,
    filesystem_root: str = "/",
    potcar_identity: Callable[[Any], Mapping[str, Any]] | None = None,
    runtime_check: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy and verify POTCARs, then verify the whole staged search. Never runs USPEX or VASP.

    Optional checks for promotion (step 4D-2), both read-only: ``job_script``
    (``{"name", "sha256"}``: a regular file in the run directory with that
    SHA-256 that ``bash -n`` accepts) and ``runtime_check`` (true: the
    login-node USPEX readiness check of the frozen plan's runtime record,
    :func:`promotion_runtime_check`; the job repeats the full check on its
    compute node).

    ``request``: ``run_dir``, ``package_dir``, ``search_dir``, ``bridge_path``,
    ``manifest`` (runtime package), ``plan_digest``, ``potcar_library``,
    ``functional_dir``, ``cache_ready_marker``, ``attempt_id`` and
    ``copy_potcars`` (False for a read-only re-verification, which changes
    nothing). Returns ``{"ok": True, ...}`` or
    ``{"ok": False, "code", "message"}``; it never returns file contents.
    """

    from backend.runtime_package_guard import bmd_runtime_package_problems
    from backend.structure_search.search_plan import (
        ContentAddressedPotcarIdentity,
        SearchPlanError,
        load_plan_text,
        potcar_identity_problems,
        verify_execution_binding,
    )

    # One identity per distinct POTCAR content within this operation only: the
    # library file and its staged copy are byte-identical, and the staged copy is
    # identified again by the staged-input verification below.
    identify = potcar_identity or ContentAddressedPotcarIdentity()
    try:
        problems = bmd_runtime_package_problems(str(_on(filesystem_root, request["package_dir"])), request["manifest"])
        if problems:
            raise StagingError("runtime_package_mismatch", "The uploaded runtime package does not match its manifest.")
        search_dir = _on(filesystem_root, request["search_dir"])
        plan = load_plan_text((search_dir / "search_plan.json").read_text(encoding="utf-8"))
        if plan.digest != request["plan_digest"]:
            raise StagingError("plan_changed", "The staged plan is not the approved plan.")
        if not os.path.lexists(_on(filesystem_root, request["cache_ready_marker"])):
            raise StagingError("uspex_cache_not_ready", "The USPEX cache has not been initialised.")

        identities = {}
        for row in plan.record["potcar"]["elements"]:
            source = _on(filesystem_root, posixpath.join(
                request["potcar_library"], request["functional_dir"], row["symbol"], "POTCAR"))
            info = os.lstat(source)
            if not stat.S_ISREG(info.st_mode):
                raise StagingError("potcar_unavailable", f"The POTCAR for {row['element']} is not a regular file.")
            if potcar_identity_problems({"potcar": {"elements": [row]}}, {row["element"]: dict(identify(source))}):
                raise StagingError(
                    "potcar_identity_mismatch",
                    f"The POWER POTCAR for {row['element']} is not the dataset the plan was made for.",
                )
            destination = search_dir / row["staged_at_execution_as"]
            temporary = destination.with_name(
                staging_temporary_name(destination.name, request["attempt_id"], row["dataset_sha256"])
            )
            if os.path.lexists(destination):
                if not stat.S_ISREG(os.lstat(destination).st_mode):
                    raise StagingError("staging_damaged", f"{row['staged_at_execution_as']} is not a regular file.")
                if request.get("copy_potcars"):
                    _finish_interrupted_link(destination, temporary)
            elif request.get("copy_potcars"):
                _copy_exclusive(source, destination, temporary, STAGED_POTCAR_MODE)
            identities[row["element"]] = dict(identify(destination))
        wrong = potcar_identity_problems(plan.record, identities)
        if wrong:
            raise StagingError("potcar_identity_mismatch",
                               f"Staged POTCAR(s) for {', '.join(wrong)} are not the planned datasets.")

        manifest = verify_execution_binding(
            search_dir,
            bridge_path=request["bridge_path"],
            filesystem_root=filesystem_root,
            require_current=False,
            potcars="staged",
            potcar_identity=identify,
        )
        job_script = _verify_job_script(request, filesystem_root) if request.get("job_script") else None
        readiness = None
        if request.get("runtime_check"):
            readiness = dict((runtime_check or promotion_runtime_check)(plan.record["runtime"]))
            if readiness.get("status") != "passed":
                return {"ok": False, "code": "runtime_check_failed",
                        "message": "The USPEX runtime readiness check failed on POWER.",
                        "problems": [str(problem) for problem in readiness.get("problems") or []]}
    except StagingError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message}
    except SearchPlanError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message}
    except (OSError, ValueError, KeyError) as exc:
        return {"ok": False, "code": "staging_failed", "message": f"{type(exc).__name__} during staging."}
    return {
        "ok": True,
        "plan_digest": plan.digest,
        "files": dict(manifest["files"]),
        "bindings": dict(manifest["bindings"]),
        "potcars": [
            {
                "element": row["element"],
                "symbol": row["symbol"],
                "TITEL": identities[row["element"]]["TITEL"],
                "ENMAX": identities[row["element"]]["ENMAX"],
                "dataset_sha256": identities[row["element"]]["sha256"],
                "verified": True,
            }
            for row in plan.record["potcar"]["elements"]
        ],
        "contains_potcar_contents": False,
        **({"job_script": job_script} if job_script is not None else {}),
        **({"runtime_check": _readiness_summary(readiness)} if readiness is not None else {}),
    }


def _verify_job_script(request: Mapping[str, Any], filesystem_root: str) -> dict[str, Any]:
    """The uploaded job script is a regular file with the expected SHA-256 that bash can parse."""

    import subprocess

    expected = request["job_script"]
    name = str(expected["name"])
    if "/" in name or name.startswith("."):
        raise StagingError("job_script_invalid", "The job script name is not a plain file name.")
    path = _on(filesystem_root, posixpath.join(request["run_dir"], name))
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise StagingError("job_script_missing", "The job script is not in the run directory.") from None
    if not stat.S_ISREG(info.st_mode):
        raise StagingError("job_script_invalid", "The job script is not a regular file.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise StagingError("job_script_mismatch", "The job script on POWER is not the one this attempt froze.")
    checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=60, check=False)
    if checked.returncode != 0:
        raise StagingError("job_script_invalid", "bash cannot parse the job script.")
    return {"name": name, "sha256": digest, "syntax": "ok"}


def promotion_runtime_check(runtime_record: Mapping[str, Any]) -> dict[str, Any]:
    """The USPEX readiness check at promotion, on the login node (scope ``login_node``).

    The payload identity, cache, Singularity and runtime settings; not the
    compute-node-only MPI library or SLURM capability, which the job checks
    when it starts on its compute node. Runs no SLURM command.
    """

    from backend.structure_search.runtime import search_runtime_check

    return search_runtime_check(runtime_record, scope="login_node")


def _readiness_summary(check: Mapping[str, Any]) -> dict[str, Any]:
    observed = check.get("observed") or {}
    return {
        "status": check.get("status"),
        "payload_sha256": observed.get("payload_sha256"),
        "banner_version": observed.get("banner_version"),
        "ready_marker_exists": observed.get("ready_marker_exists"),
        "unpacked_program_exists": observed.get("unpacked_program_exists"),
        "singularity_executable": (observed.get("singularity") or {}).get("executable"),
        "scope": check.get("scope"),
        "deferred_to_job_start": list(observed.get("deferred_to_job_start") or []),
    }


def finalize_staged_search_main(request: Mapping[str, Any]) -> None:
    result = finalize_staged_search(request)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "INVENTORY_MARKER",
    "REMOTE_STATE_STAGED",
    "STAGED_REMOTE_INTEGRITY",
    "build_inventory_script",
    "parse_inventory_output",
    "parse_staging_temporary",
    "reconcile_staging",
    "staging_inventory",
    "staging_temporary_name",
    "staging_temporary_path",
    "REMOTE_STATE_STAGING",
    "RESULT_MARKER",
    "StagingError",
    "build_finalize_script",
    "build_search_submission_spec",
    "finalize_staged_search",
    "parse_finalize_output",
    "request_from_script",
    "staged_search_files",
    "staging_layout",
]
