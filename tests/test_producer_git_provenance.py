"""Producer Git provenance must never write to the BMD Compute checkout.

BMD Agent invokes the capability and input-reference producers against the
live, protected BMD Compute checkout. Plain ``git status`` may opportunistically
refresh ``.git/index`` and create ``.git/index.lock``; producer provenance must
therefore run Git with optional locks disabled.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from backend.calculations import capabilities
from backend.calculations.capabilities import build_capability_payload, git_provenance
from backend.calculations.input_reference import build_input_reference_payload


SI_POSCAR = """Si
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

STATIC_REQUEST = {
    "structure": {"type": "pasted_text", "format": "poscar", "text": SI_POSCAR},
    "workflow_spec": {
        "stages": [
            {
                "stage_type": "static",
                "theory": "pbe",
                "modifiers": [],
                "label": None,
                "options": {},
            }
        ],
        "label": None,
        "recipe": None,
    },
}


def _is_git_command(command) -> bool:
    if isinstance(command, (str, bytes)) or not command:
        return False
    return Path(str(command[0])).name.lower() in {"git", "git.exe"}


def _record_git_calls(monkeypatch) -> list[tuple[tuple[str, ...], dict | None]]:
    calls: list[tuple[tuple[str, ...], dict | None]] = []
    real_run = subprocess.run

    def spy(command, *args, **kwargs):
        if _is_git_command(command):
            calls.append((tuple(str(item) for item in command), kwargs.get("env")))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    return calls


def _assert_no_optional_locks(calls) -> None:
    assert calls, "expected producer provenance to invoke git"
    for command, env in calls:
        assert "--no-optional-locks" in command, command
        # The option is a global git option and must precede the subcommand.
        assert command.index("--no-optional-locks") < command.index("-C"), command
        assert env is not None and env.get("GIT_OPTIONAL_LOCKS") == "0", command


def test_git_command_disables_optional_locks_before_subcommand():
    command = capabilities._git_command(("status", "--porcelain"), Path("/checkout"))

    assert command[:2] == ("git", "--no-optional-locks")
    assert command[-2:] == ("status", "--porcelain")
    assert capabilities._git_environment()["GIT_OPTIONAL_LOCKS"] == "0"


def test_capability_producer_invokes_git_without_optional_locks(monkeypatch):
    calls = _record_git_calls(monkeypatch)

    payload = build_capability_payload()

    _assert_no_optional_locks(calls)
    subcommands = {command[command.index("-C") + 2] for command, _ in calls}
    assert {"rev-parse", "status"} <= subcommands
    assert payload["source"]["repository"] == "bmd_compute"


def test_input_reference_producer_invokes_git_without_optional_locks(monkeypatch):
    calls = _record_git_calls(monkeypatch)

    payload = build_input_reference_payload(STATIC_REQUEST)

    assert payload["status"] == "ok"
    _assert_no_optional_locks(calls)
    assert payload["producer"]["repository"] == "bmd_compute"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "BMD Test",
            "GIT_AUTHOR_EMAIL": "bmd-test@example.invalid",
            "GIT_COMMITTER_NAME": "BMD Test",
            "GIT_COMMITTER_EMAIL": "bmd-test@example.invalid",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(repo.parent),
        }
    )
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _repository_with_stale_index(root: Path) -> Path:
    """Create a clean repository whose index stat data no longer matches.

    The tracked file keeps identical content but gets a new mtime, so a plain
    ``git status`` reports the tree clean while wanting to refresh the index.
    """

    repo = root / "checkout"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    tracked = repo / "tracked.txt"
    tracked.write_text("unchanged\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "initial")
    future = time.time() + 3600
    os.utime(tracked, (future, future))
    return repo


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_provenance_does_not_rewrite_the_checkout_index(tmp_path):
    control = _repository_with_stale_index(tmp_path / "control")
    control_index = control / ".git" / "index"
    control_before = control_index.read_bytes()
    _git(control, "status", "--porcelain")
    if control_index.read_bytes() == control_before:
        pytest.skip("this git version does not refresh the index during status")

    repo = _repository_with_stale_index(tmp_path / "producer")
    index = repo / ".git" / "index"
    before = index.read_bytes()

    provenance = git_provenance(repo_root=repo)

    assert provenance["provenance_available"] is True
    assert provenance["dirty"] is False
    assert index.read_bytes() == before
    assert not (repo / ".git" / "index.lock").exists()
