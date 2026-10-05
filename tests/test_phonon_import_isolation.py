"""backend.phonons must not make Phonopy a requirement for existing workflows.

The runtime package ships every backend Python file to POWER, where Phonopy may
be absent. These tests run in fresh interpreters so module caches cannot hide
an eager import.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from backend.runtime_package import runtime_package_relative_paths
from backend.submission import build_backend_module_sources


REPO_ROOT = Path(__file__).resolve().parents[1]
PHONONS_DIR = REPO_ROOT / "backend" / "phonons"

BLOCK_PHONOPY = """
import sys
class _BlockPhonopy:
    def find_spec(self, name, path=None, target=None):
        if name == "phonopy" or name.startswith("phonopy."):
            raise ImportError("phonopy is not installed (blocked by test)")
        return None
sys.meta_path.insert(0, _BlockPhonopy())
"""


def _run(source: str, *, pythonpath: Path, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    env.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-c", source], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )


def test_importing_backend_phonons_does_not_import_phonopy(tmp_path):
    completed = _run(
        "import sys, json\n"
        "import backend.phonons\n"
        "from backend.phonons import PROVISIONAL_PHONON_POLICY\n"
        "PROVISIONAL_PHONON_POLICY.phonopy_arguments([[2,0,0],[0,2,0],[0,0,2]])\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {'phonopy','pymatgen','numpy'})))",
        pythonpath=REPO_ROOT,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_existing_entry_points_do_not_import_phonopy(tmp_path):
    completed = _run(
        "import sys\n"
        "import main, backend.execution, backend.workflows, backend.submission, backend.results\n"
        "print('phonopy' in sys.modules)",
        pythonpath=REPO_ROOT,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_shipped_runtime_bundle_works_without_phonopy(tmp_path):
    for relative_path, source in build_backend_module_sources().items():
        path = tmp_path / "backend" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    assert "phonons/plan.py" in runtime_package_relative_paths()

    script = BLOCK_PHONOPY + """
import json
import backend.execution, backend.workflows
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import validate_workflow_spec
validate_workflow_spec(WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)]))

import backend.phonons
from backend.phonons import PhonopyUnavailableError, build_displacement_plan
from pymatgen.core import Lattice, Structure
structure = Structure(Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
try:
    build_displacement_plan(structure, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    outcome = "built"
except PhonopyUnavailableError as exc:
    outcome = type(exc).__name__
print(json.dumps({"outcome": outcome, "phonopy_imported": "phonopy" in sys.modules}))
"""
    completed = _run(script, pythonpath=tmp_path, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"outcome": "PhonopyUnavailableError", "phonopy_imported": False}


def test_plan_records_validate_without_phonopy(tmp_path):
    import pytest

    pytest.importorskip("phonopy")
    from backend.phonons import build_displacement_plan
    from pymatgen.core import Lattice, Structure

    plan = build_displacement_plan(
        Structure(Lattice([[0, 2.845, 2.845], [2.845, 0, 2.845], [2.845, 2.845, 0]]), ["Na", "Cl"],
                  [[0, 0, 0], [0.5, 0.5, 0.5]]),
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    )
    record_path = tmp_path / "plan.json"
    record_path.write_text(plan.to_json(), encoding="utf-8")

    script = BLOCK_PHONOPY + f"""
import json
from pathlib import Path
from backend.phonons import DisplacementPlan
plan = DisplacementPlan.from_json(Path({str(record_path)!r}).read_text(encoding="utf-8"))
print(json.dumps({{"sha": plan.plan_sha256, "tasks": plan.task_count,
                   "phonopy_imported": "phonopy" in sys.modules}}))
"""
    completed = _run(script, pythonpath=REPO_ROOT, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "sha": plan.plan_sha256,
        "tasks": plan.task_count,
        "phonopy_imported": False,
    }


def test_phonons_modules_have_no_module_level_heavy_imports():
    heavy = {"phonopy", "pymatgen", "numpy", "spglib"}
    for path in sorted(PHONONS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in heavy, f"{path.name} imports {name} at module level"
