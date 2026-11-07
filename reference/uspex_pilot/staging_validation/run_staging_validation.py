"""Staging-only POWER validation of one Si structure search (USPEX step 4C).

Run on the BMD Compute VM, from a separate checkout of the reviewed commit
(never the live deployment directory), with the interpreter of the service
environment. Five steps, each a subcommand taking the same new ROOT:

    setup ROOT       create a new, private validation root (it must not exist)
    preflight ROOT   read-only POWER checks (power_preflight.sh over SSH)
    run ROOT         plan -> prepare -> stage -> read back -> fresh re-verification
    evidence ROOT    independent read-only POWER evidence (power_evidence.sh over SSH)
    archive ROOT     ROOT.tgz with the evidence, checked to hold no secret or POTCAR text

Validation root (all private to the user)::

    ROOT/VALIDATION_ROOT.json   marks the root as this validation's
    ROOT/state/                 the only BMD_API_STATE_DIR used (0700)
    ROOT/tokens.json            the only BMD_API_TOKENS_FILE used: one principal,
                                scopes plan, prepare, read (0600)
    ROOT/client_token           that principal's bearer token (0600; never archived)
    ROOT/evidence/              requests and responses (no token, no headers)

``run`` drives the machine API in-process, through its authenticated router
(loopback client, bearer token) with this validation's own ledger. It stops
at remote ``STAGED``: it never calls a calculation route, never asks for
submission and never runs sbatch, USPEX or VASP. Re-running it after an
interruption resumes the same attempt. It refuses to start if any
``BMD_API_*`` variable is set, if the root is not one ``setup`` made, if the
state directory holds anything but this validation's own attempt, or if a
running BMD Compute service on this machine uses the same state or token
paths. The submission identity secret of the shell, if any, is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

PRINCIPAL = "uspex-4c-validation"
SCOPES = ("plan", "prepare", "read")
FORMULA = "Si"
LOOPBACK = ("127.0.0.1", 50000)
SSH_HOST = "powerslurm-bmdguest"
ROOT_KIND = "bmd_compute.uspex_4c_staging_validation_root"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ATTEMPT_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
RUN_NAME_RE = re.compile(r"^uspex_search-[0-9]{8}-[0-9]{6}-(?P<prefix>[0-9a-f]{8})$")
SUBMISSION_IDENTITY_SECRET_ENV = "BMD_SUBMISSION_IDENTITY_SECRET"
# Text that must never be archived: POTCAR dataset text, keys, credentials.
FORBIDDEN_TEXT = ("VRHFIN", "POMASS", "PSCTR", "LEXCH", "End of Dataset", "PRIVATE KEY", "Authorization",
                  "Bearer ", "bmdc1.", '"verifier"')
REDACTED = "<redacted: HMAC submission identity, not archived>"


class ValidationFailed(SystemExit):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailed(f"VALIDATION FAILED: {message}")


def _private_file(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- setup ---------------------------------------------------------------------------------


def setup(root: Path) -> None:
    sys.path.insert(0, str(REPO))
    from compute_api.auth import TOKENS_FILE_SCHEMA, TOKENS_FILE_SCHEMA_VERSION, generate_token

    root = Path(os.path.abspath(root))
    _require(root == root.resolve(), f"{root} goes through a symbolic link.")
    _require(REPO not in root.parents and root != REPO, f"the validation root must be outside {REPO}.")
    root.mkdir(mode=0o700, parents=False)  # refuses an existing root
    for name in ("state", "evidence"):
        (root / name).mkdir(mode=0o700)
    token, token_id, verifier = generate_token()
    tokens = {
        "schema": TOKENS_FILE_SCHEMA,
        "schema_version": TOKENS_FILE_SCHEMA_VERSION,
        "principals": [{"principal": PRINCIPAL, "token_id": token_id, "verifier": verifier,
                        "scopes": sorted(SCOPES), "enabled": True}],
    }
    _private_file(root / "tokens.json", json.dumps(tokens, indent=2, sort_keys=True) + "\n")
    _private_file(root / "client_token", token + "\n")
    _private_file(root / "VALIDATION_ROOT.json",
                  json.dumps({"kind": ROOT_KIND, "created_at": _now(), "principal": PRINCIPAL}, indent=2) + "\n")
    print(f"Validation root ready: {root}")


# -- isolation -------------------------------------------------------------------------------


def live_service_paths(proc: Path = Path("/proc")) -> set[Path]:
    """BMD_API_STATE_DIR and BMD_API_TOKENS_FILE of running BMD Compute services this user can see."""

    found: set[Path] = set()
    for entry in proc.iterdir() if proc.is_dir() else ():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            if b"uvicorn" not in command and b"main:app" not in command:
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        for item in environment:
            name, _, value = item.partition(b"=")
            if name in (b"BMD_API_STATE_DIR", b"BMD_API_TOKENS_FILE") and value:
                found.add(Path(os.path.abspath(value.decode("utf-8", "replace"))))
    return found


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def check_isolation(root: Path, *, proc: Path = Path("/proc")) -> dict:
    """Refuse anything that could make this validation use production state or credentials."""

    inherited = sorted(name for name in os.environ if name.startswith("BMD_API_"))
    if inherited:
        raise ValidationFailed(f"{', '.join(inherited)} set in this shell; unset it. "
                               f"The validation uses only {root}.")
    _require(root == Path(os.path.abspath(root)).resolve(), f"{root} goes through a symbolic link.")
    try:
        marker = json.loads((root / "VALIDATION_ROOT.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValidationFailed(f"{root} was not made by 'setup'.") from None
    _require(marker.get("kind") == ROOT_KIND, f"{root} was not made by 'setup'.")
    state, tokens_file, evidence = root / "state", root / "tokens.json", root / "evidence"
    for path in (root, state, evidence):
        _require(path.is_dir() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o700,
                 f"{path} must be a private (0700) directory.")
    for path in (tokens_file, root / "client_token"):
        _require(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600,
                 f"{path} must be a private (0600) file.")
    principals = json.loads(tokens_file.read_text(encoding="utf-8"))["principals"]
    _require([(p["principal"], sorted(p["scopes"])) for p in principals] == [(PRINCIPAL, sorted(SCOPES))],
             "the token file must hold only the validation principal.")

    # The state directory holds this validation's own attempt and nothing else.
    _require(not list((state / "attempts").glob("*")), "the state directory holds calculation attempts.")
    attempt_file = evidence / "attempt.json"
    own = {json.loads(attempt_file.read_text())["attempt_id"]} if attempt_file.exists() else set()
    search_records = {p.stem for p in (state / "structure_search" / "attempts").glob("*.json")}
    _require(search_records <= own, "the state directory holds attempts this validation did not make.")
    unexpected = sorted(p.name for p in state.iterdir()
                        if p.name not in {"structure_search", "attempts", "ledger.lock"})
    _require(not unexpected, f"unexpected entries in the state directory: {unexpected}.")

    for live in live_service_paths(proc):
        for ours in (state, tokens_file):
            _require(not _overlaps(ours, live), f"a running BMD Compute service uses {live}.")
    return {"state": state, "tokens_file": tokens_file, "evidence": evidence}


# -- run ---------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def redact(value):
    """A copy without submission identity signatures (HMAC values; not evidence)."""

    if isinstance(value, dict):
        return {key: (REDACTED if key == "identity_token" else redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class Recorder:
    def __init__(self, evidence: Path):
        self.evidence = evidence
        existing = sorted(evidence.glob("[0-9][0-9][0-9]-*.json"))
        self.index = int(existing[-1].name[:3]) + 1 if existing else 1

    def save(self, step: str, payload) -> Path:
        path = self.evidence / f"{self.index:03d}-{step}.json"
        self.index += 1
        _private_file(path, json.dumps(redact(payload), indent=2, sort_keys=True) + "\n")
        return path

    def exchange(self, client, step: str, method: str, url: str, token: str, body=None):
        response = client.request(method, url, headers={"Authorization": f"Bearer {token}"}, json=body)
        try:
            payload = response.json()
        except ValueError:
            payload = {"unparsed": response.text}
        # The request is recorded without its headers: the token is never written.
        self.save(step, {"request": {"method": method, "url": url, "body": body},
                         "status": response.status_code, "response": payload})
        print(f"{step}: HTTP {response.status_code}")
        return response.status_code, payload


def _check_no_potcar_contents(payload) -> None:
    """Identity (TITEL, ENMAX, SHA-256) is reported on purpose; dataset text never is."""

    text = json.dumps(payload)
    for marker in ("VRHFIN", "POMASS", "PSCTR", "LEXCH"):
        _require(marker not in text, f"a response contains POTCAR text ({marker}).")


def run(root: Path, *, allow_dirty: bool = False, runner_factory=None, proc: Path = Path("/proc")) -> dict:
    root = Path(os.path.abspath(root))
    paths = check_isolation(root, proc=proc)
    state, tokens_file, evidence = paths["state"], paths["tokens_file"], paths["evidence"]
    token = (root / "client_token").read_text(encoding="utf-8").strip()

    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        raise ValidationFailed("The checkout has uncommitted changes; validate a reviewed commit.")

    os.environ.pop(SUBMISSION_IDENTITY_SECRET_ENV, None)  # never sign with a production secret
    os.environ["BMD_API_STATE_DIR"] = str(state)
    os.environ["BMD_API_TOKENS_FILE"] = str(tokens_file)
    os.chdir(REPO)  # the app resolves its templates relative to the checkout
    sys.path.insert(0, str(REPO))
    import warnings

    try:
        from starlette.testclient import TestClient
    except ImportError as exc:  # httpx missing: report it, never install into the service environment
        raise ValidationFailed(f"The in-process API client is unavailable ({exc}); stop and report.") from None

    import main
    from compute_api import execution
    from compute_api.ledger import STATE_DIR_ENV
    from compute_api.structure_search_execution import reverify_staged_search

    _require(os.environ[STATE_DIR_ENV] == str(state), "the ledger is not the validation's own.")
    if runner_factory is not None:
        execution.RUNNER_FACTORY = runner_factory
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = TestClient(main.app, client=LOOPBACK)

    recorder = Recorder(evidence)
    attempt_file = evidence / "attempt.json"
    if not attempt_file.exists():
        recorder.save("environment", {"commit": commit, "dirty": bool(dirty), "python": sys.version,
                                      "formula": FORMULA, "principal": PRINCIPAL, "state_dir": str(state)})

    status, _ = recorder.exchange(client, "identity", "GET", "/api/v1/identity", token)
    _require(status == 200, f"identity returned {status}.")

    if attempt_file.exists():
        attempt = json.loads(attempt_file.read_text(encoding="utf-8"))
        attempt_id, digest = attempt["attempt_id"], attempt["plan_digest"]
        print(f"Resuming attempt {attempt_id}.")
    else:
        status, plan = recorder.exchange(client, "plan", "POST", "/api/v1/structure-search-plans", token,
                                         {"formula": FORMULA})
        _require(status == 200, f"planning returned {status}.")
        digest = plan["plan_digest"]
        attempt_id = str(uuid.uuid4())
        _private_file(attempt_file, json.dumps({"attempt_id": attempt_id, "plan_digest": digest}, indent=2) + "\n")

    route = f"/api/v1/structure-search-attempts/{attempt_id}"
    status, prepared = recorder.exchange(client, "prepare", "PUT", route, token,
                                         {"formula": FORMULA, "expected_plan_digest": digest})
    _require(status == 200, f"local preparation returned {status}: {prepared}.")
    _require(prepared["state"] in {"PREPARED_LOCALLY", "STAGING", "STAGED"}, f"unexpected state {prepared['state']}.")

    status, staged = recorder.exchange(client, "stage", "PUT", f"{route}/staging", token,
                                       {"expected_plan_digest": digest})
    if status != 200:
        code = (staged.get("error") or {}).get("code")
        raise ValidationFailed(
            f"Staging returned {status} ({code}). Evidence is in {evidence}. A transport interruption (503) may "
            "be retried by running 'run' again; any other code is a finding: stop and report it."
        )
    _require(staged["state"] == "STAGED", "staging did not end at STAGED.")
    staging = staged["staging"]
    _require(staging["remote_state"] == "STAGED" and staging["bridge_path_bound"] is True, "bridge not bound.")
    _require(staging["remote_integrity"] == "last_verified", "STAGED must be reported as last verified.")
    _require(staged["submission_available"] is False, "submission must not be available.")
    _require(staging["contains_potcar_contents"] is False, "POTCAR contents flag.")
    _require([row["element"] for row in staging["potcars"]] == [FORMULA], "unexpected POTCAR set.")
    _require(all(row["verified"] for row in staging["potcars"]), "POTCAR identity not verified.")
    _check_no_potcar_contents(staged)

    status, read_back = recorder.exchange(client, "read_back", "GET", route, token)
    _require(status == 200 and read_back == staged, "read-back differs from the staging response.")

    verification = reverify_staged_search(principal=PRINCIPAL, attempt_id=attempt_id)
    recorder.save("fresh_reverification", verification)
    print("fresh re-verification on POWER: passed")
    _check_no_potcar_contents(verification)

    record = json.loads((state / "structure_search" / "attempts" / f"{attempt_id}.json").read_text(encoding="utf-8"))
    spec = record["staging"]["spec"]
    summary = {
        "result": "STAGED and freshly re-verified; nothing submitted",
        "commit": commit,
        "attempt_id": attempt_id,
        "plan_digest": digest,
        "run_name": record["staging"]["run_name"],
        "run_dir": spec["paths"]["run_dir"],
        "search_dir": spec["structure_search"]["search_dir"],
        "bridge_path": spec["structure_search"]["bridge_path"],
        "bridge_sha256": spec["structure_search"]["bridge_sha256"],
        "remote_state_file": f"{spec['paths']['logs_dir']}/submission_attempts/{attempt_id}.json",
        "runtime_package_manifest_sha256": record["staging"]["runtime_manifest_digest"],
        "potcars": staging["potcars"],
        "calculation_ledger_entries": sorted(p.name for p in (state / "attempts").glob("*.json")),
    }
    _require(summary["calculation_ledger_entries"] == [], "a calculation attempt was created.")
    recorder.save("local_record", record)  # identity signature redacted
    recorder.save("summary", summary)
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
            for p in sorted(evidence.iterdir()) if p.is_file() and p.name != "SHA256SUMS"]
    sums_path = evidence / "SHA256SUMS"
    sums_path.unlink(missing_ok=True)
    _private_file(sums_path, "\n".join(sums) + "\n")
    print(json.dumps({key: summary[key] for key in ("attempt_id", "run_name", "plan_digest")}, indent=2))
    return summary


# -- POWER scripts over SSH ----------------------------------------------------------------------


def evidence_arguments(root: Path) -> tuple[str, str]:
    """The attempt ID and run name of this validation, checked against each other and the ledger."""

    attempt = json.loads((root / "evidence" / "attempt.json").read_text(encoding="utf-8"))["attempt_id"]
    record = json.loads((root / "state" / "structure_search" / "attempts" / f"{attempt}.json").read_text())
    run_name = record["staging"]["run_name"]
    match = RUN_NAME_RE.match(str(run_name))
    _require(bool(ATTEMPT_RE.match(str(attempt))), f"malformed attempt ID {attempt!r}.")
    _require(bool(match) and match["prefix"] == attempt[:8], f"run name {run_name!r} does not match the attempt.")
    _require(record["state"] == "STAGED", f"the attempt is {record['state']}, not STAGED.")
    return attempt, run_name


def remote_command(*arguments: str) -> str:
    """The command the remote login shell runs: ``bash -s -- ARG...`` with every argument quoted."""

    return "bash -s -- " + " ".join(shlex.quote(argument) for argument in arguments) if arguments else "bash -s"


def run_power_script(root: Path, script: str, output_name: str, arguments: tuple[str, ...] = ()) -> int:
    """Run one kit script on POWER as bmdguest; record its output beside the evidence (never overwritten)."""

    output = root / output_name
    _require(not output.exists(), f"{output} exists; it is not overwritten.")
    command = ["ssh", "-o", "BatchMode=yes", SSH_HOST, remote_command(*arguments)]
    with open(HERE / script, "rb") as stdin:
        completed = subprocess.run(command, stdin=stdin, capture_output=True)
    text = completed.stdout.decode("utf-8", "replace") + completed.stderr.decode("utf-8", "replace")
    _private_file(output, text)
    sys.stdout.write(text)
    return completed.returncode


def preflight(root: Path, *, proc: Path = Path("/proc")) -> int:
    check_isolation(Path(os.path.abspath(root)), proc=proc)
    return run_power_script(Path(os.path.abspath(root)), "power_preflight.sh", "power_preflight.txt")


def evidence(root: Path, *, proc: Path = Path("/proc")) -> int:
    root = Path(os.path.abspath(root))
    check_isolation(root, proc=proc)
    attempt, run_name = evidence_arguments(root)
    return run_power_script(root, "power_evidence.sh", "power_evidence.txt", (attempt, run_name))


# -- archive -----------------------------------------------------------------------------------


def archive_members(root: Path) -> list[Path]:
    """What the archive holds: evidence, the local preparation and the run logs. Never a token."""

    members = sorted(p for p in (root / "evidence").rglob("*") if p.is_file())
    members += sorted(p for p in (root / "state" / "structure_search" / "preparations").rglob("*") if p.is_file())
    members += [root / name for name in ("run.log", "power_preflight.txt", "power_evidence.txt")
                if (root / name).is_file()]
    return members


def archive(root: Path) -> Path:
    root = Path(os.path.abspath(root))
    check_isolation(root)
    secret = (root / "client_token").read_text(encoding="utf-8").strip()
    verifier = json.loads((root / "tokens.json").read_text())["principals"][0]["verifier"]
    target = root.with_name(root.name + ".tgz")
    _require(not target.exists(), f"{target} exists; it is not overwritten.")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in archive_members(root):
            relative = path.relative_to(root)
            _require(not path.is_symlink(), f"{relative} is a symbolic link.")
            _require(path.name not in {"client_token", "tokens.json"} and not re.match(r"^POTCAR(_.*)?$", path.name),
                     f"{relative} must not be archived.")
            text = path.read_bytes().decode("utf-8", "replace")
            for forbidden in (secret, secret.split(".")[-1], verifier, *FORBIDDEN_TEXT):
                _require(forbidden not in text, f"{relative} contains text that must not be archived.")
            signatures = re.findall(r'"identity_token":\s*"([^"]*)"', text)
            _require(all(value == REDACTED for value in signatures) and text.count("identity_token") == len(signatures),
                     f"{relative} holds a submission identity signature.")
            tar.add(path, arcname=f"{root.name}/{relative}", recursive=False)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(buffer.getvalue())
    print(f"{hashlib.sha256(buffer.getvalue()).hexdigest()}  {target}")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "preflight", "run", "evidence", "archive"):
        sub.add_parser(name).add_argument("root", type=Path)
    sub.choices["run"].add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup(args.root)
        return 0
    if args.command == "run":
        run(args.root, allow_dirty=args.allow_dirty)
        return 0
    if args.command == "archive":
        archive(args.root)
        return 0
    return {"preflight": preflight, "evidence": evidence}[args.command](args.root)


if __name__ == "__main__":
    raise SystemExit(main())
