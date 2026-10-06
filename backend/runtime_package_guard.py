"""Pre-import verification of the uploaded BMD Compute runtime package.

Prepare uploads the ``backend/`` package next to ``run_job.py`` on POWER and
records its sha256 manifest in the attempt's provenance. This module is the
check that the code POWER is about to run is exactly that package.

It is bootstrap source, not a library: BMD Compute embeds the text of this
file into the generated ``run_job.py`` and the remote preflight, together with
the expected manifest, so the check runs before anything under ``backend/`` is
imported. The uploaded copy of this file is never what performs the check on
POWER; that would let the package verify itself.

Because the text is embedded into another script, this file must stay
standard-library only, must not use ``from __future__`` imports, and keeps every
top-level name prefixed with ``_bmd``/``Bmd`` so it cannot collide with the
script it is embedded in. Its imports happen once, at the top, before the
finder is installed: an import inside the finder would re-enter it.

The check has two parts:

* Before the first import, the whole package directory must match the manifest:
  every manifest file present as a regular file with the recorded sha256, and
  no symbolic links or other files. ``__pycache__`` directories are ignored
  because bytecode is never used (below).
* After that, a finder placed first on ``sys.meta_path`` serves every
  ``backend`` module. It only resolves names that are in the manifest, hashes
  the source bytes again when it loads them, and compiles those bytes directly,
  so neither a file changed after the check nor stale bytecode can be executed.
"""

import hashlib as _bmd_hashlib
import importlib.machinery as _bmd_importlib_machinery
import json as _bmd_json
import os as _bmd_os
import stat as _bmd_stat
import sys as _bmd_sys

BMD_RUNTIME_PACKAGE_VERIFICATION_FAILED = "BMD_RUNTIME_PACKAGE_VERIFICATION_FAILED"
_BMD_RUNTIME_PACKAGE_MAX_REPORTED_PROBLEMS = 40


class BmdRuntimePackageVerificationError(ImportError):
    """The uploaded runtime package does not match the attempt's manifest."""


def bmd_runtime_package_manifest_digest(manifest):
    """Return the sha256 of the canonical JSON form of ``manifest``."""

    payload = _bmd_json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"))
    return _bmd_hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bmd_sha256_file(path):
    digest = _bmd_hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bmd_runtime_package_problems(package_dir, manifest):
    """Return sorted problems that make ``package_dir`` differ from ``manifest``.

    ``manifest`` maps package-relative POSIX paths to sha256 hex digests. An
    empty list means the directory holds exactly the manifest files.
    """

    try:
        info = _bmd_os.lstat(package_dir)
    except OSError:
        return ["missing package directory"]
    if not _bmd_stat.S_ISDIR(info.st_mode):
        return ["package directory is not a real directory"]

    problems = []
    accounted = set()
    pending = [("", package_dir)]
    while pending:
        prefix, directory = pending.pop()
        try:
            with _bmd_os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            problems.append("unreadable directory: " + (prefix or "."))
            continue
        for entry in entries:
            relative_path = prefix + entry.name
            if entry.is_symlink():
                problems.append("symbolic link: " + relative_path)
                accounted.add(relative_path)
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name != "__pycache__":
                    pending.append((relative_path + "/", entry.path))
                continue
            accounted.add(relative_path)
            if relative_path not in manifest:
                problems.append("unexpected file: " + relative_path)
                continue
            if not entry.is_file(follow_symlinks=False):
                problems.append("not a regular file: " + relative_path)
                continue
            try:
                actual = _bmd_sha256_file(entry.path)
            except OSError:
                problems.append("unreadable file: " + relative_path)
                continue
            if actual != manifest[relative_path]:
                problems.append("modified file: " + relative_path)

    for relative_path in set(manifest) - accounted:
        problems.append("missing file: " + relative_path)
    return sorted(problems)


def bmd_runtime_package_failure_message(package_dir, manifest, problems, *, detail):
    shown = list(problems)[:_BMD_RUNTIME_PACKAGE_MAX_REPORTED_PROBLEMS]
    lines = [
        BMD_RUNTIME_PACKAGE_VERIFICATION_FAILED
        + ": the BMD Compute runtime package does not match the manifest "
        "recorded when this submission attempt was prepared. " + detail,
        "  package directory: " + str(package_dir),
        "  expected manifest sha256: "
        + bmd_runtime_package_manifest_digest(manifest)
        + " (" + str(len(manifest)) + " files)",
    ]
    lines.extend("  - " + problem for problem in shown)
    if len(problems) > len(shown):
        lines.append("  - ... and " + str(len(problems) - len(shown)) + " more")
    lines.append(
        "Prepare a new submission attempt in BMD Compute. Do not add, edit or "
        "remove files in the run directory's runtime package."
    )
    return "\n".join(lines)


class _BmdVerifiedSourceLoader:
    def __init__(self, package_dir, manifest, relative_path, path):
        self.package_dir = package_dir
        self.manifest = manifest
        self.relative_path = relative_path
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.path, "rb") as handle:
            data = handle.read()
        if _bmd_hashlib.sha256(data).hexdigest() != self.manifest[self.relative_path]:
            raise BmdRuntimePackageVerificationError(
                bmd_runtime_package_failure_message(
                    self.package_dir,
                    self.manifest,
                    ["modified file: " + self.relative_path],
                    detail="A module changed after the package was verified.",
                )
            )
        code = compile(data, self.path, "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _BmdVerifiedPackageFinder:
    def __init__(self, package_name, package_dir, manifest):
        self.package_name = package_name
        self.package_dir = package_dir
        self.manifest = manifest

    def find_spec(self, fullname, path=None, target=None):
        name = self.package_name
        if fullname != name and not fullname.startswith(name + "."):
            return None
        parts = fullname.split(".")[1:]
        package_path = "/".join(parts + ["__init__.py"])
        module_path = "/".join(parts) + ".py" if parts else None
        if package_path in self.manifest:
            relative_path, is_package = package_path, True
        elif module_path in self.manifest:
            relative_path, is_package = module_path, False
        else:
            raise ModuleNotFoundError(
                "No module named " + repr(fullname)
                + " in the verified BMD Compute runtime package",
                name=fullname,
            )
        origin = _bmd_os.path.join(self.package_dir, *relative_path.split("/"))
        spec = _bmd_importlib_machinery.ModuleSpec(
            fullname,
            _BmdVerifiedSourceLoader(self.package_dir, self.manifest, relative_path, origin),
            origin=origin,
            is_package=is_package,
        )
        if is_package:
            spec.submodule_search_locations = [_bmd_os.path.dirname(origin)]
        spec.has_location = True
        return spec


def bmd_install_verified_runtime_package(package_dir, manifest, *, package_name="backend"):
    """Verify ``package_dir`` against ``manifest`` and serve imports from it.

    Raises BmdRuntimePackageVerificationError, before any package code runs,
    when the directory differs from the manifest. Returns the manifest digest.
    """

    manifest = dict(manifest)
    already_loaded = sorted(
        name
        for name in _bmd_sys.modules
        if name == package_name or name.startswith(package_name + ".")
    )
    if already_loaded:
        raise BmdRuntimePackageVerificationError(
            bmd_runtime_package_failure_message(
                package_dir,
                manifest,
                ["imported before verification: " + name for name in already_loaded],
                detail="Nothing further was run.",
            )
        )
    problems = bmd_runtime_package_problems(package_dir, manifest)
    if problems:
        raise BmdRuntimePackageVerificationError(
            bmd_runtime_package_failure_message(
                package_dir,
                manifest,
                problems,
                detail="No package code was run and VASP was not started.",
            )
        )
    _bmd_sys.meta_path.insert(0, _BmdVerifiedPackageFinder(package_name, package_dir, manifest))
    return bmd_runtime_package_manifest_digest(manifest)
