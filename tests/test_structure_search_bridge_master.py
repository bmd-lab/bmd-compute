"""Tests for the USPEX master command line and the commandExecutable lines (issue #66)."""

from __future__ import annotations

import pytest

from backend.structure_search.bridge.master import build_master_command, uspex_command_executable


# -- USPEX master command ------------------------------------------------------------


def test_master_command_line_and_explicit_environment():
    command = build_master_command(
        singularity="/opt/singularity/bin/singularity",
        image="/bmd/shared/uspex25_sandbox",
        search_dir="/bmd/user/search1",
        xdg_cache_home="/bmd/shared/uspex_cache",
        spool_dir="/bmd/user/search1/.bmd_bridge",
        bridge_timeout_seconds=86400,
        host_environment={"SINGULARITY_TMPDIR": "/scratch/tmp"},
    )
    assert command.argv == (
        "/opt/singularity/bin/singularity", "exec", "--cleanenv", "-B", "/bmd",
        "--pwd", "/bmd/user/search1", "/bmd/shared/uspex25_sandbox", "/opt/uspex/uspex",
    )
    assert dict(command.env) == {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "SINGULARITY_TMPDIR": "/scratch/tmp",
        "SINGULARITYENV_XDG_CACHE_HOME": "/bmd/shared/uspex_cache",
        "SINGULARITYENV_BMD_BRIDGE_SPOOL": "/bmd/user/search1/.bmd_bridge",
        "SINGULARITYENV_BMD_BRIDGE_TIMEOUT_SECONDS": "86400",
        "SINGULARITYENV_BMD_BRIDGE_POLL_SECONDS": "2",
        "SINGULARITYENV_BMD_BRIDGE_HEARTBEAT_SECONDS": "300",
    }
    # XDG_CACHE_HOME reaches only the container, never the host side.
    assert "XDG_CACHE_HOME" not in command.env
    assert str(command.cwd) == "/bmd/user/search1"
    assert command.shell_line().startswith("cd /bmd/user/search1 && env -i ")


def test_master_command_rejects_unsafe_paths():
    base = dict(
        singularity="/usr/bin/singularity",
        image="/bmd/shared/uspex_25.sif",
        search_dir="/bmd/user/search1",
        xdg_cache_home="/bmd/shared/cache",
        spool_dir="/bmd/user/search1/spool",
        bridge_timeout_seconds=0,
    )
    bad = [
        dict(search_dir="relative"),
        dict(search_dir="/home/user/search1"),
        dict(xdg_cache_home="/home/user/.cache"),
        dict(spool_dir="/tmp/spool"),
        dict(search_dir="/bmd/../etc"),
        dict(bridge_timeout_seconds=-1),
        dict(bridge_heartbeat_seconds=0),
        dict(uspex_path="uspex"),
        dict(host_environment={"SINGULARITYENV_X": "1"}),
        dict(host_environment={"PATH": "/x"}),
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            build_master_command(**{**base, **overrides})


def test_uspex_command_executable_lines():
    assert uspex_command_executable("/bmd/run/backend/structure_search/bridge/bmd_bridge.sh", 3) == [
        "sh /bmd/run/backend/structure_search/bridge/bmd_bridge.sh 1",
        "sh /bmd/run/backend/structure_search/bridge/bmd_bridge.sh 2",
        "sh /bmd/run/backend/structure_search/bridge/bmd_bridge.sh 3",
    ]
    assert uspex_command_executable("/bmd/a b/bmd_bridge.sh", 1) == ["sh '/bmd/a b/bmd_bridge.sh' 1"]
    with pytest.raises(ValueError):
        uspex_command_executable("/x", 0)
