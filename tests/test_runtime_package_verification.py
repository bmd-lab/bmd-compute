"""The POWER bootstrap verifies the uploaded runtime package before importing it.

Prepare uploads ``backend/`` next to ``run_job.py``. These tests materialize a
run directory exactly as Prepare uploads it and then run the generated
bootstrap, so they exercise the same code that runs on POWER.
"""

import ast
import hashlib
import importlib.util
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

from backend import runtime_package_guard
from backend.runtime_package import (
    build_runtime_package_manifest,
    runtime_package_manifest_digest,
    runtime_package_manifest_for_sources,
    runtime_package_root,
)
from backend.runtime_package_guard import (
    BMD_RUNTIME_PACKAGE_VERIFICATION_FAILED,
    BmdRuntimePackageVerificationError,
    bmd_install_verified_runtime_package,
    bmd_runtime_package_problems,
)
from backend.submission import (
    REMOTE_RUNTIME_PREFLIGHT_ARGUMENT,
    RUNTIME_PACKAGE_GUARD_MODULE,
    build_backend_module_sources,
    build_remote_runtime_preflight_source,
    build_run_job_script,
    create_submission_spec,
    remote_preparation_file_groups,
)


POSCAR = """Si
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
direct
0.0 0.0 0.0
0.25 0.25 0.25
"""

SENTINEL = "UNVERIFIED_CODE_RAN"
PAYLOAD = (
    "import pathlib, os\n"
    f"pathlib.Path(os.environ['BMD_TEST_SENTINEL_DIR'], {SENTINEL!r}).write_text('x')\n"
)


def _submission_spec(run_dir: Path) -> dict:
    spec = create_submission_spec(
        {
            "workflow": "static",
            "potcar_functional": "PBE_64",
            "kpoints": None,
            "incar": {},
            "structure": {"type": "pasted_text", "format": "poscar", "text": POSCAR},
        },
        label="Si runtime package verification",
        timestamp="20261006-120000",
        env={},
    )
    spec["paths"]["run_dir"] = run_dir.as_posix()
    return spec


def _prepare_run_dir(tmp_path: Path) -> tuple[Path, dict]:
    """Write every Prepare upload that lives in the run directory."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _submission_spec(run_dir)
    for group in remote_preparation_file_groups(spec):
        for item in group["files"]:
            path = Path(item["path"])
            if run_dir not in path.parents:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            # newline="" writes the uploaded text byte for byte.
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(item["text"])
    return run_dir, spec


def _environment(tmp_path: Path) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("BMD_SUBMISSION_SPEC", None)
    env["BMD_TEST_SENTINEL_DIR"] = str(tmp_path)
    return env


def _run_preflight(tmp_path: Path, run_dir: Path, spec: dict):
    # Exactly how remote preparation runs it: `cd run_dir && python - <<PY`.
    return subprocess.run(
        [sys.executable, "-"],
        input=build_remote_runtime_preflight_source(spec),
        cwd=run_dir,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_job(tmp_path: Path, run_dir: Path, spec: dict):
    # Exactly how the sbatch script runs it, including BMD_SUBMISSION_SPEC.
    env = _environment(tmp_path)
    env["BMD_SUBMISSION_SPEC"] = str(run_dir / spec["runner"]["submission_spec_name"])
    return subprocess.run(
        [sys.executable, "-u", "run_job.py"],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_rejected(tmp_path: Path, completed, problem: str) -> None:
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert BMD_RUNTIME_PACKAGE_VERIFICATION_FAILED in completed.stderr
    assert f"  - {problem}\n" in completed.stderr
    assert "No package code was run and VASP was not started." in completed.stderr
    assert "Prepare a new submission attempt in BMD Compute." in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / SENTINEL).exists()


# --- the guard is bootstrap source ---------------------------------------------------


def test_guard_module_is_standard_library_only_and_embeddable():
    source = Path(runtime_package_guard.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "__future__"
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"hashlib", "importlib", "json", "os", "stat", "sys"}
    top_level_names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top_level_names.append(node.name)
        elif isinstance(node, ast.Import):
            top_level_names.extend(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            top_level_names.extend(target.id for target in node.targets)
    assert top_level_names
    assert all(name.lower().startswith(("bmd", "_bmd")) for name in top_level_names)
    # The finder must not import anything itself: that would re-enter it.
    finder = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_BmdVerifiedPackageFinder"
    )
    loader = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_BmdVerifiedSourceLoader"
    )
    for node in (*ast.walk(finder), *ast.walk(loader)):
        assert not isinstance(node, (ast.Import, ast.ImportFrom))


def test_run_job_embeds_the_uploaded_guard_and_never_imports_it_from_the_package():
    sources = build_backend_module_sources()
    script = build_run_job_script(runtime_package_sources=sources)

    assert sources[RUNTIME_PACKAGE_GUARD_MODULE].strip() in script
    assert not re.search(r"^\s*(from|import)\s+backend(\.|\s+import\s+)runtime_package_guard", script, re.M)
    verification = script.index("bmd_install_verified_runtime_package(\n")
    assert verification < script.index("from backend.execution import run_submission")
    assert verification < script.index("import backend.execution")


def test_expected_manifest_is_the_recorded_provenance_of_byte_exact_uploads(tmp_path):
    run_dir, spec = _prepare_run_dir(tmp_path)
    recorded = spec["provenance"]["bmd_compute"]["runtime_source"]["manifest"]

    uploaded = {
        path.relative_to(run_dir / "backend").as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (run_dir / "backend").rglob("*.py")
    }
    assert uploaded == recorded == build_runtime_package_manifest()
    assert runtime_package_manifest_for_sources(build_backend_module_sources()) == recorded
    run_job = (run_dir / "run_job.py").read_text(encoding="utf-8")
    assert runtime_package_manifest_digest(recorded) in _run_job_digest_line(tmp_path, run_dir, spec)
    assert all(digest in run_job for digest in recorded.values())


def _run_job_digest_line(tmp_path: Path, run_dir: Path, spec: dict) -> str:
    completed = _run_preflight(tmp_path, run_dir, spec)
    assert completed.returncode == 0, completed.stderr
    return next(
        line for line in completed.stdout.splitlines()
        if line.startswith("[runner] runtime package verified:")
    )


def test_package_files_with_crlf_lines_are_uploaded_byte_for_byte():
    sources = build_backend_module_sources()
    root = runtime_package_root()
    crlf_files = [
        name for name in sources if b"\r\n" in (root / name).read_bytes()
    ]
    for name in crlf_files:
        assert sources[name].encode("utf-8") == (root / name).read_bytes()


def test_prepare_refuses_when_recorded_provenance_does_not_describe_the_upload(tmp_path):
    spec = _submission_spec(tmp_path)
    manifest = spec["provenance"]["bmd_compute"]["runtime_source"]["manifest"]
    manifest["execution.py"] = "0" * 64

    with pytest.raises(RuntimeError, match="recorded provenance no longer describes"):
        remote_preparation_file_groups(spec)


def test_preflight_command_stays_small_and_carries_no_package_payload(tmp_path):
    spec = _submission_spec(tmp_path)
    source = build_remote_runtime_preflight_source(spec)

    assert len(source) < 1024
    assert REMOTE_RUNTIME_PREFLIGHT_ARGUMENT in source
    assert "runpy.run_path" in source


# --- the untampered package ----------------------------------------------------------


def test_untampered_upload_passes_preflight_through_the_bootstrap(tmp_path):
    run_dir, spec = _prepare_run_dir(tmp_path)

    completed = _run_preflight(tmp_path, run_dir, spec)

    assert completed.returncode == 0, completed.stderr
    assert "[runner] runtime package verified:" in completed.stdout
    assert "BMD_RUNTIME_PREFLIGHT_OK=backend.execution,backend.workflows" in completed.stdout
    # Bytecode is neither read nor written for the verified package.
    assert not list((run_dir / "backend").rglob("__pycache__"))


def test_untampered_upload_runs_the_submission_through_the_plain_bootstrap(tmp_path):
    """`python -u run_job.py` as sbatch runs it reaches run_submission.

    The submission.json lacks a prepared runtime parity record, so the first
    thing run_submission does (the runtime parity check) stops it before any
    calculation is built; reaching that error proves the verified package
    imported and ran.
    """

    run_dir, spec = _prepare_run_dir(tmp_path)
    (run_dir / spec["runner"]["submission_spec_name"]).write_text(
        '{"flow_spec": {}}', encoding="utf-8"
    )

    completed = _run_job(tmp_path, run_dir, spec)

    assert completed.returncode == 1
    assert "[runner] runtime package verified:" in completed.stdout
    assert "[runner] submission spec:" in completed.stdout
    assert BMD_RUNTIME_PACKAGE_VERIFICATION_FAILED not in completed.stderr
    assert "RecursionError" not in completed.stderr
    assert "RuntimeParityError" in completed.stderr
    assert not list((run_dir / "backend").rglob("__pycache__"))


SHADOWING_MODULES = ("json", "traceback", "runpy", "hashlib", "stat", "importlib")


def _plant_shadowing_modules(run_dir: Path) -> None:
    for name in SHADOWING_MODULES:
        (run_dir / f"{name}.py").write_text(PAYLOAD, encoding="utf-8")


def test_bootstrap_does_not_import_modules_planted_in_the_run_directory(tmp_path):
    run_dir, spec = _prepare_run_dir(tmp_path)
    _plant_shadowing_modules(run_dir)

    preflight = _run_preflight(tmp_path, run_dir, spec)
    assert preflight.returncode == 0, preflight.stderr
    assert "BMD_RUNTIME_PREFLIGHT_OK" in preflight.stdout
    assert not (tmp_path / SENTINEL).exists()

    (run_dir / spec["runner"]["submission_spec_name"]).write_text(
        '{"flow_spec": {}}', encoding="utf-8"
    )
    job = _run_job(tmp_path, run_dir, spec)
    assert "[runner] runtime package verified:" in job.stdout
    assert "RuntimeParityError" in job.stderr
    assert not (tmp_path / SENTINEL).exists()


def test_run_job_rejects_unexpected_arguments(tmp_path):
    run_dir, spec = _prepare_run_dir(tmp_path)
    env = _environment(tmp_path)
    completed = subprocess.run(
        [sys.executable, "run_job.py", "--something-else"],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 2
    assert "unexpected arguments" in completed.stderr


# --- tampered packages ---------------------------------------------------------------


def _append_payload(run_dir: Path, relative_path: str) -> None:
    with (run_dir / "backend" / relative_path).open("a", encoding="utf-8") as handle:
        handle.write("\n" + PAYLOAD)


def _tamper_modified(run_dir):
    _append_payload(run_dir, "execution.py")
    return "modified file: execution.py"


def _tamper_modified_init(run_dir):
    _append_payload(run_dir, "__init__.py")
    return "modified file: __init__.py"


def _tamper_missing(run_dir):
    (run_dir / "backend" / "calculations" / "resource_policy.py").unlink()
    return "missing file: calculations/resource_policy.py"


def _tamper_extra_module(run_dir):
    (run_dir / "backend" / "planted.py").write_text(PAYLOAD, encoding="utf-8")
    return "unexpected file: planted.py"


def _tamper_extra_nested_module(run_dir):
    (run_dir / "backend" / "calculations" / "planted.py").write_text(PAYLOAD, encoding="utf-8")
    return "unexpected file: calculations/planted.py"


def _tamper_extra_subpackage(run_dir):
    package = run_dir / "backend" / "plugins"
    package.mkdir()
    (package / "__init__.py").write_text(PAYLOAD, encoding="utf-8")
    return "unexpected file: plugins/__init__.py"


def _tamper_sourceless_bytecode(run_dir):
    planted = run_dir / "backend" / "planted.py"
    planted.write_text(PAYLOAD, encoding="utf-8")
    py_compile.compile(str(planted), cfile=str(run_dir / "backend" / "planted.pyc"))
    planted.unlink()
    return "unexpected file: planted.pyc"


def _tamper_extension_module(run_dir):
    (run_dir / "backend" / "planted.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    return "unexpected file: planted.cpython-312-x86_64-linux-gnu.so"


def _tamper_stale_editor_file(run_dir):
    (run_dir / "backend" / "execution.py.orig").write_text("old", encoding="utf-8")
    return "unexpected file: execution.py.orig"


def _tamper_symlinked_module(run_dir):
    module = run_dir / "backend" / "execution.py"
    copy = run_dir / "execution_copy.py"
    copy.write_bytes(module.read_bytes())
    module.unlink()
    module.symlink_to(copy)
    return "symbolic link: execution.py"


def _tamper_symlinked_package(run_dir):
    backend = run_dir / "backend"
    backend.rename(run_dir / "backend_real")
    backend.symlink_to(run_dir / "backend_real", target_is_directory=True)
    return "package directory is not a real directory"


def _tamper_missing_package(run_dir):
    (run_dir / "backend").rename(run_dir / "backend_moved")
    return "missing package directory"


TAMPERING = [
    _tamper_modified,
    _tamper_modified_init,
    _tamper_missing,
    _tamper_extra_module,
    _tamper_extra_nested_module,
    _tamper_extra_subpackage,
    _tamper_sourceless_bytecode,
    _tamper_extension_module,
    _tamper_stale_editor_file,
    _tamper_symlinked_module,
    _tamper_symlinked_package,
    _tamper_missing_package,
]


@pytest.mark.parametrize("tamper", TAMPERING, ids=lambda tamper: tamper.__name__)
def test_run_job_refuses_a_package_that_differs_from_the_manifest(tmp_path, tamper):
    run_dir, spec = _prepare_run_dir(tmp_path)
    problem = tamper(run_dir)

    _assert_rejected(tmp_path, _run_job(tmp_path, run_dir, spec), problem)


@pytest.mark.parametrize("tamper", TAMPERING, ids=lambda tamper: tamper.__name__)
def test_preflight_refuses_a_package_that_differs_from_the_manifest(tmp_path, tamper):
    run_dir, spec = _prepare_run_dir(tmp_path)
    problem = tamper(run_dir)

    _assert_rejected(tmp_path, _run_preflight(tmp_path, run_dir, spec), problem)


def test_rejection_is_deterministic_and_lists_every_problem(tmp_path):
    run_dir, spec = _prepare_run_dir(tmp_path)
    _tamper_extra_module(run_dir)
    _tamper_modified(run_dir)
    _tamper_missing(run_dir)

    first = _run_job(tmp_path, run_dir, spec)
    second = _run_job(tmp_path, run_dir, spec)

    assert first.stderr == second.stderr
    problems = [line for line in first.stderr.splitlines() if line.startswith("  - ")]
    assert problems == [
        "  - missing file: calculations/resource_policy.py",
        "  - modified file: execution.py",
        "  - unexpected file: planted.py",
    ]


def test_stale_bytecode_for_a_verified_module_is_never_executed(tmp_path):
    run_dir, spec = _prepare_run_dir(tmp_path)
    module = run_dir / "backend" / "execution.py"
    original = module.read_bytes()
    # Bytecode for different code, made to look current for the original source:
    # same size and same mtime, which is all CPython's default pyc check compares.
    impostor = PAYLOAD.encode("utf-8")
    impostor += b"#" * (len(original) - len(impostor))
    module.write_bytes(impostor)
    py_compile.compile(str(module), cfile=importlib.util.cache_from_source(str(module)))
    impostor_stat = module.stat()
    module.write_bytes(original)
    os.utime(module, ns=(impostor_stat.st_atime_ns, impostor_stat.st_mtime_ns))

    control = subprocess.run(
        [sys.executable, "-c", "import backend.execution"],
        cwd=run_dir,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (tmp_path / SENTINEL).exists(), "control: a plain import uses the stale bytecode"
    (tmp_path / SENTINEL).unlink()

    completed = _run_preflight(tmp_path, run_dir, spec)

    assert control.returncode == 0
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / SENTINEL).exists()


# --- the guard in-process ------------------------------------------------------------


@pytest.fixture
def probe_package(tmp_path):
    name = "bmd_guard_probe_pkg"
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'package'\n", encoding="utf-8")
    (package / "module.py").write_text("VALUE = 'module'\n", encoding="utf-8")
    manifest = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*.py")
    }
    meta_path = list(sys.meta_path)
    yield name, package, manifest
    sys.meta_path[:] = meta_path
    for module_name in [m for m in sys.modules if m.split(".")[0] == name]:
        del sys.modules[module_name]


def test_guard_serves_verified_modules_from_the_package_directory(probe_package):
    name, package, manifest = probe_package

    digest = bmd_install_verified_runtime_package(str(package), manifest, package_name=name)
    module = importlib.import_module(f"{name}.module")

    assert digest == runtime_package_manifest_digest(manifest)
    assert module.VALUE == "module"
    assert Path(module.__file__) == package / "module.py"
    assert sys.modules[name].__path__ == [str(package)]
    assert not (package / "__pycache__").exists()


def test_guard_rejects_a_module_changed_after_verification(probe_package):
    name, package, manifest = probe_package
    bmd_install_verified_runtime_package(str(package), manifest, package_name=name)
    (package / "module.py").write_text("VALUE = 'changed'\n", encoding="utf-8")

    with pytest.raises(BmdRuntimePackageVerificationError, match="modified file: module.py"):
        importlib.import_module(f"{name}.module")


def test_guard_does_not_import_files_added_after_verification(probe_package):
    name, package, manifest = probe_package
    bmd_install_verified_runtime_package(str(package), manifest, package_name=name)
    (package / "added.py").write_text("VALUE = 'added'\n", encoding="utf-8")

    with pytest.raises(ModuleNotFoundError, match="verified BMD Compute runtime package"):
        importlib.import_module(f"{name}.added")


def test_guard_refuses_a_package_imported_before_verification(probe_package):
    name, package, manifest = probe_package
    sys.modules[name] = type(sys)(name)

    with pytest.raises(BmdRuntimePackageVerificationError, match="imported before verification"):
        bmd_install_verified_runtime_package(str(package), manifest, package_name=name)


def test_guard_ignores_bytecode_caches_and_reports_nothing_for_an_exact_tree(probe_package):
    _name, package, manifest = probe_package
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"stale")

    assert bmd_runtime_package_problems(str(package), manifest) == []
