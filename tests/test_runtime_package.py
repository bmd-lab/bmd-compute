import json
import os
import subprocess
import sys

from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.runtime_package import (
    build_runtime_package_manifest,
    runtime_package_relative_paths,
)
from backend.submission import (
    build_backend_module_sources,
    build_remote_runtime_preflight_source,
    create_submission_spec,
    json_dumps_for_remote_file,
)


POSCAR = '''Si
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
direct
0.0 0.0 0.0
0.25 0.25 0.25
'''


def _write_runtime_bundle(tmp_path, sources=None):
    sources = dict(build_backend_module_sources() if sources is None else sources)
    for relative_path, source in sources.items():
        path = tmp_path / "backend" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _run_isolated(tmp_path, source):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    env.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_runtime_package_includes_transitive_backend_python_sources():
    sources = build_backend_module_sources()
    names = set(sources)

    assert "execution.py" in names
    assert "workflows.py" in names
    assert "calculations/resource_policy.py" in names
    assert "calculations/vasp_stage_definitions.py" in names
    assert "runtime_package.py" in names
    assert all(name.endswith(".py") for name in names)
    assert all("__pycache__" not in name for name in names)
    assert all(not name.startswith("tests/") for name in names)
    assert tuple(sorted(runtime_package_relative_paths())) == runtime_package_relative_paths()


def test_runtime_package_manifest_is_deterministic_and_hashes_packaged_sources():
    manifest = build_runtime_package_manifest()

    assert list(manifest) == sorted(manifest)
    assert set(manifest) == set(build_backend_module_sources())
    assert manifest["calculations/resource_policy.py"]
    assert all(len(digest) == 64 for digest in manifest.values())


def test_isolated_runtime_bundle_imports_execution_workflows_and_specs(tmp_path):
    _write_runtime_bundle(tmp_path)
    script = '''
import json
import sys

import backend.execution
import backend.workflows
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import validate_workflow_spec

workflows = [
    WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)]),
    WorkflowSpec([
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.HSE06),
        StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
    ], recipe="custom"),
    WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC})]),
]
for workflow in workflows:
    validate_workflow_spec(workflow)

print(json.dumps({
    "ok": True,
    "fastapi_imported": "fastapi" in sys.modules,
    "paramiko_imported": "paramiko" in sys.modules,
}, sort_keys=True))
'''
    completed = _run_isolated(tmp_path, script)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "fastapi_imported": False,
        "ok": True,
        "paramiko_imported": False,
    }


def test_isolated_runtime_bundle_fails_when_transitive_module_is_missing(tmp_path):
    sources = build_backend_module_sources()
    sources.pop("calculations/resource_policy.py")
    _write_runtime_bundle(tmp_path, sources)

    completed = _run_isolated(tmp_path, "import backend.workflows")

    assert completed.returncode != 0
    assert "backend.calculations.resource_policy" in completed.stderr


def test_remote_runtime_preflight_uses_uploaded_bundle_and_submission_json(tmp_path):
    _write_runtime_bundle(tmp_path)
    workflow_spec = WorkflowSpec([
        StageSpec(StageType.RELAX, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.HSE06),
        StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
    ], recipe="custom")
    flow_spec = {
        "workflow": "custom_workflow",
        "workflow_spec": workflow_spec.to_dict(),
        "potcar_functional": "PBE_64",
        "kpoints": None,
        "incar": {},
        "structure": {
            "type": "pasted_text",
            "format": "poscar",
            "text": POSCAR,
        },
    }
    spec = create_submission_spec(
        flow_spec,
        label="Si isolated preflight",
        timestamp="20260821-120000",
        env={},
    )
    spec["paths"]["run_dir"] = tmp_path.as_posix()
    spec["runner"]["submission_spec_name"] = "submission.json"
    (tmp_path / "submission.json").write_text(
        json_dumps_for_remote_file(spec),
        encoding="utf-8",
    )

    completed = _run_isolated(tmp_path, build_remote_runtime_preflight_source(spec))

    assert completed.returncode == 0, completed.stderr
    assert "BMD_RUNTIME_PREFLIGHT_OK" in completed.stdout
