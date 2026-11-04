"""Shell scripts in the verified runtime package (the USPEX launch bridge).

``backend/structure_search/bridge/bmd_bridge.sh`` (PR #73) is not on main
yet, so these tests build a temporary package tree with a stand-in script.
"""

from pathlib import Path

import pytest

import backend.runtime_package as runtime_package
from backend.runtime_package import (
    build_runtime_package_manifest,
    build_runtime_package_sources,
    is_runtime_package_source,
    runtime_package_manifest_for_sources,
    runtime_package_relative_paths,
)
from backend.runtime_package_guard import bmd_runtime_package_problems


BRIDGE = "structure_search/bridge/bmd_bridge.sh"


@pytest.mark.parametrize(
    ("relative_path", "included"),
    [
        ("execution.py", True),
        ("structure_search/bridge/protocol.py", True),
        (BRIDGE, True),
        ("structure_search/bridge/other.sh", True),
        ("structure_search/bridge/nested/run.sh", False),
        ("structure_search/run.sh", False),
        ("deploy.sh", False),
        ("structure_search/bridge/notes.txt", False),
        ("structure_search/bridge/__pycache__/bmd_bridge.sh", False),
        ("tests/structure_search/bridge/bmd_bridge.sh", False),
    ],
)
def test_only_python_and_bridge_shell_scripts_are_packaged(relative_path, included):
    assert is_runtime_package_source(relative_path) is included


@pytest.fixture
def package_tree(tmp_path, monkeypatch):
    root = tmp_path / "backend"
    (root / "structure_search" / "bridge").mkdir(parents=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "execution.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "structure_search" / "__init__.py").write_text('"""x"""\n', encoding="utf-8")
    (root / "structure_search" / "bridge" / "__init__.py").write_text("", encoding="utf-8")
    (root / BRIDGE).write_text('#!/bin/sh\necho "step $1"\nexit 0\n', encoding="utf-8")
    (root / "deploy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(runtime_package, "runtime_package_root", lambda: root)
    return root


def _upload(sources, destination: Path) -> None:
    for relative_path, text in sources.items():
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def test_bridge_script_is_uploaded_and_hashed(package_tree):
    sources = build_runtime_package_sources()
    manifest = build_runtime_package_manifest()

    assert BRIDGE in runtime_package_relative_paths()
    assert "deploy.sh" not in sources
    assert sources[BRIDGE] == (package_tree / BRIDGE).read_text(encoding="utf-8")
    assert manifest == runtime_package_manifest_for_sources(sources)
    assert set(manifest) == {
        "__init__.py",
        "execution.py",
        "structure_search/__init__.py",
        "structure_search/bridge/__init__.py",
        BRIDGE,
    }


def test_uploaded_bridge_script_is_verified_on_power(package_tree, tmp_path):
    sources = build_runtime_package_sources()
    manifest = runtime_package_manifest_for_sources(sources)
    uploaded = tmp_path / "run" / "backend"
    _upload(sources, uploaded)

    assert bmd_runtime_package_problems(str(uploaded), manifest) == []

    (uploaded / BRIDGE).write_text('#!/bin/sh\necho "changed"\n', encoding="utf-8")
    assert bmd_runtime_package_problems(str(uploaded), manifest) == [f"modified file: {BRIDGE}"]

    (uploaded / BRIDGE).unlink()
    assert bmd_runtime_package_problems(str(uploaded), manifest) == [f"missing file: {BRIDGE}"]
