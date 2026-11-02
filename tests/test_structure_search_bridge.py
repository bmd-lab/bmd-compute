"""Tests for bmd_bridge.sh and the bridge file protocol (issue #66).

See docs/structure_search.md §4. Covered here: the shell script's
portability (POSIX sh, only tools the USPEX image has), its usage errors and
exit codes, and request parsing. The launcher, the master command and the
end-to-end behaviour have their own test_structure_search_bridge_*.py files.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.runtime_package import runtime_package_relative_paths
from backend.structure_search.bridge import protocol
from backend.structure_search.bridge.master import bridge_script_path


BRIDGE = bridge_script_path()


# -- bmd_bridge.sh ---------------------------------------------------------------


def test_bridge_script_parses_as_posix_sh():
    for shell in ("sh", "dash"):
        if shutil.which(shell):
            completed = subprocess.run([shell, "-n", str(BRIDGE)], capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr


def test_bridge_script_avoids_bash_only_syntax_and_missing_tools():
    code = "\n".join(
        line for line in BRIDGE.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    for bashism in ("[[", "function ", "local ", "$RANDOM", "<<<", "$(<", "source ", "pipefail"):
        assert bashism not in code, bashism
    # Not available in the USPEX image (docs/structure_search.md §5.3).
    for tool in ("python", "srun", "hostname", "mktemp --", "realpath"):
        assert not re.search(rf"(^|[\s;|(]){re.escape(tool)}", code), tool


def test_bridge_exit_codes_match_protocol_module():
    text = BRIDGE.read_text()
    shell_codes = dict(re.findall(r"^(EXIT_[A-Z_]+)=(\d+)$", text, flags=re.MULTILINE))
    assert shell_codes == {
        name: str(getattr(protocol, name))
        for name in ("EXIT_USAGE", "EXIT_LAUNCHER_LOST", "EXIT_STOPPED", "EXIT_TIMEOUT", "EXIT_SIGNAL")
    }
    assert f"#   {protocol.EXIT_REJECTED}    the launcher rejected" in text


def _run_bridge(args, cwd, env):
    return subprocess.run(["sh", str(BRIDGE), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=30)


def test_bridge_usage_errors_exit_120_without_writing_a_request(tmp_path):
    spool = tmp_path / "spool"
    for sub in protocol.SPOOL_SUBDIRS:
        (spool / sub).mkdir(parents=True)
    good = {"PATH": "/usr/bin:/bin", "BMD_BRIDGE_SPOOL": str(spool), "BMD_BRIDGE_TIMEOUT_SECONDS": "5"}
    cases = [
        ([], good),
        (["0"], good),
        (["x"], good),
        (["1", "2"], good),
        (["1"], {**good, "BMD_BRIDGE_SPOOL": "relative/spool"}),
        (["1"], {key: value for key, value in good.items() if key != "BMD_BRIDGE_TIMEOUT_SECONDS"}),
        (["1"], {**good, "BMD_BRIDGE_POLL_SECONDS": "fast"}),
        (["1"], {**good, "BMD_BRIDGE_HEARTBEAT_SECONDS": "0"}),
    ]
    for args, env in cases:
        completed = _run_bridge(args, tmp_path, env)
        assert completed.returncode == protocol.EXIT_USAGE, (args, env, completed.stderr)
        assert "bmd_bridge:" in completed.stderr
    assert list((spool / protocol.REQUESTS_DIR).iterdir()) == []


def test_bridge_without_spool_directories_reports_launcher_lost(tmp_path):
    env = {
        "PATH": "/usr/bin:/bin",
        "BMD_BRIDGE_SPOOL": str(tmp_path / "no-spool"),
        "BMD_BRIDGE_TIMEOUT_SECONDS": "5",
    }
    completed = _run_bridge(["1"], tmp_path, env)
    assert completed.returncode == protocol.EXIT_LAUNCHER_LOST
    assert "is the launcher running" in completed.stderr


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The verified runtime package uploads only .py files (backend/runtime_package.py). "
        "Issue #67 must add bmd_bridge.sh to it; remove this xfail then."
    ),
)
def test_bridge_script_is_in_the_verified_runtime_package():
    assert "structure_search/bridge/bmd_bridge.sh" in runtime_package_relative_paths()


def test_launcher_modules_are_in_the_verified_runtime_package():
    paths = runtime_package_relative_paths()
    for name in ("__init__.py", "protocol.py", "master.py"):
        assert f"structure_search/bridge/{name}" in paths


def test_structure_search_package_init_is_one_line():
    init = Path(__file__).resolve().parents[1] / "backend" / "structure_search" / "__init__.py"
    assert init.read_text() == '"""USPEX structure search (see docs/structure_search.md)."""\n'


# -- request parsing ---------------------------------------------------------------


def test_parse_request_round_trip():
    request = protocol.parse_request(
        "id=s1-1-2-ab\nstep=2\ncalcfold=/bmd/x/CalcFold1\ncreated=17\n", expected_id="s1-1-2-ab"
    )
    assert (request.request_id, request.step, str(request.calcfold), request.created) == (
        "s1-1-2-ab", 2, "/bmd/x/CalcFold1", 17,
    )


@pytest.mark.parametrize(
    "text",
    [
        "id=a\nstep=1\ncalcfold=/x\n",
        "id=a\nstep=0\ncalcfold=/x\ncreated=1\n",
        "id=a\nstep=1\ncalcfold=rel\ncreated=1\n",
        "id=a\nid=a\nstep=1\ncalcfold=/x\ncreated=1\n",
        "id=../a\nstep=1\ncalcfold=/x\ncreated=1\n",
        "id=b\nstep=1\ncalcfold=/x\ncreated=1\n",
        "garbage\n",
    ],
)
def test_parse_request_rejects_malformed_text(text):
    with pytest.raises(protocol.RequestError):
        protocol.parse_request(text, expected_id="a")
