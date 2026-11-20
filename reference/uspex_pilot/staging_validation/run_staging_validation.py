"""Staging, promotion and submission acceptance on POWER for one Si structure search (USPEX steps 4C, 4D).

Run on the BMD Compute VM, from a separate checkout of the reviewed commit
(never the live deployment directory), with the interpreter of the service
environment. Each step is a subcommand taking the same new ROOT:

    setup ROOT [--acceptance-walltime 00:15:00 | --execution | --execution-shared]
                     create a new, private validation root (it must not exist); --acceptance-walltime: the
                     operational acceptance (10-15 minutes); --execution: the scientific execution acceptance
                     (Si, 2 atoms, 2 generations, 04:00:00, pinned plan digest) on one exclusive 192-core node
                     (4 x 48 ranks); --execution-shared: the same search on 48 cores of a shared node
                     (3 x 16 ranks, --mem=32G), all via the planner
    preflight ROOT   read-only POWER checks (power_preflight.sh over SSH)
    run ROOT         plan -> prepare -> stage -> read back -> fresh re-verification
    promote ROOT     acknowledge the resource request -> promote to SEARCH_PREPARED (no sbatch)
    evidence ROOT    independent read-only POWER evidence (power_evidence.sh over SSH)
    submit ROOT --confirm-attempt ID [--confirm-walltime 04:00:00]
                     the one submission request of this validation (one sbatch, by the service); the
                     execution profile also needs its walltime confirmed
    observe ROOT     one read-only observation (no submission, no cancellation)
    cancel ROOT --confirm-attempt ID --confirm-job JOB
                     request cancellation of exactly that job (scancel, by the service, after a fresh check)
    acceptance-evidence ROOT
                     independent read-only POWER evidence of the job (power_acceptance_evidence.sh over SSH)
    execution-evidence ROOT
                     after an execution-profile job has ended: collect its POWER evidence read-only
                     (execution_validation/power_execution_evidence.sh over SSH), fetch it and analyse it
                     (execution_validation/analyze_execution.py)
    archive ROOT     ROOT.tgz with the evidence, checked to hold no secret or POTCAR text

Validation root (all private to the user)::

    ROOT/VALIDATION_ROOT.json   marks the root as this validation's
    ROOT/state/                 the only BMD_API_STATE_DIR used (0700)
    ROOT/tokens.json            the only BMD_API_TOKENS_FILE used: one principal,
                                scopes plan, prepare, read, submit (0600)
    ROOT/client_token           that principal's bearer token (0600; never archived)
    ROOT/evidence/              requests and responses (no token, no headers)

``run`` and ``promote`` drive the machine API in-process, through its
authenticated router (loopback client, bearer token) with this validation's
own ledger. ``run`` stops at remote ``STAGED``; ``promote`` stops at
``SEARCH_PREPARED``. Neither calls a calculation or submission route, nor runs
sbatch, srun, USPEX or VASP. Re-running either after an interruption resumes
the same attempt.

Acceptance (step 4D, part D of README.md), each command a separate decision:
``submit`` sends this validation's one submission request, only with
``--confirm-attempt`` equal to the attempt ID, only for a ``PREPARED``
attempt this root made (never a historical validation attempt), and never a
second time (a marker is written before the request; any outcome other than
a clean ``SUBMITTED`` stops, an uncertain one with "do not repeat").
``observe`` only observes. ``cancel`` needs ``--confirm-attempt`` and
``--confirm-job`` equal to the recorded attempt and job, and only accepts a
command sent or a job already ended. Every one stops (non-zero) on an
unexpected status, an identity that is not verified (``partial`` only for a
job no longer queued), inconsistent times, contradictions or a broken
evidence chain. The kit itself never runs `sbatch`/`scancel`: the service
does, through its routes. It refuses to start if any
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
# submit: promotion (step 4D-2) and the acceptance's one submission request; cancel: its cancellation
# (step 4D). A root set up before the cancel scope existed holds a token file without it and is refused
# by the root check: set up a new root.
SCOPES = ("cancel", "plan", "prepare", "read", "submit")
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
# The 4C and 4D-2 validation attempts are evidence and are never submitted.
# Earlier validations (4D-2 promotion, 4D-2 staging, the 4D acceptance of 10 Nov 2026): kept as evidence,
# never submitted, observed or cancelled again by the kit.
HISTORICAL_ATTEMPTS = frozenset({"5ead0670-911a-4c52-9587-9f3ae9539269", "14bfbbac-8599-48a6-bcc2-cfea917fd967",
                                 "c6edaf94-a2ab-4be7-a817-ce52f3d14852"})
JOB_ID_RE = re.compile(r"^[0-9]{1,20}$")
TERMINAL_STATES = ("SEARCH_COMPLETED", "SEARCH_FAILED", "SEARCH_CANCELLED", "SEARCH_TIMED_OUT")
UNCERTAIN_CODES = ("submission_uncertain", "submission_unverified", "submission_ambiguous", "submission_conflict")
ACCEPTED_CANCELLATION = ("command_sent", "not_sent_job_not_active", "command_found_job_ended")
RESUMABLE_CANCELLATION = ("remote_request_uncertain", "remote_recorded_command_pending")
# The acceptance's allocation: a short walltime set through the planner's Advanced override (never by editing a
# plan or a job script), carried by the plan digest, the acknowledgement, the job script and the remote record.
ACCEPTANCE_WALLTIME_DEFAULT = "00:15:00"
ACCEPTANCE_WALLTIME_RANGE_S = (10 * 60, 15 * 60)
# The scientific execution acceptance (execution_validation/DESIGN.md): a fixed, small Si search, set only through
# the planner's Advanced settings. Its plan digest is pinned: the kit refuses any other plan for this profile.
EXECUTION_SEARCH = {"min_atoms": 2, "max_atoms": 2, "num_generations": 2, "walltime": "04:00:00"}
EXECUTION_WALLTIME = "04:00:00"
EXECUTION_PLAN_DIGEST = "sha256:1989c888f58012590920ebbf4b5ca31c2d5aa0dd53449cc6bcbe3951b9b4898e"
EXECUTION_DIRECTIVES = (
    "#SBATCH -p leeburton-pool", "#SBATCH --account=power-leeburton-users_v2", "#SBATCH --qos=owner100d",
    "#SBATCH --time=04:00:00", "#SBATCH --nodes=1", "#SBATCH --ntasks=192", "#SBATCH --exclusive",
    "#SBATCH --mem=0", "#SBATCH --exclude=compute-0-[332-333]",
)
EXECUTION_SPEC = {"min_atoms": 2, "max_atoms": 2, "num_generations": 2, "population_size": 10,
                  "initial_population_size": 10, "stop_criterion": 2, "walltime": "04:00:00"}
EXECUTION_LAUNCH = {"mode": "srun_steps", "parallel_relaxations": 4, "ranks_per_relaxation": 48}
EXECUTION_STEPS = 5
EXECUTION_DIR = HERE.parent / "execution_validation"
# Every scientific execution profile: the same search and ladder; only the node request and the launch
# settings differ. ``configuration`` is applied to backend.config in the kit's own process before the in-process
# service plans anything (the live service and its configuration are never touched); the job on POWER reads
# these values from the frozen plan, never from its environment. Each profile's plan digest and resource
# request hash are pinned: the kit refuses anything else.
EXECUTION_PROFILES = {
    "execution": {
        "flag": "--execution",
        "note": "scientific execution acceptance: a 2-atom, two-generation Si search on one exclusive node",
        "plan_digest": EXECUTION_PLAN_DIGEST,
        "resources_sha256": "sha256:bd5ebc2aa303f062fc84066b8bde8e17a26aaacdcdd5c4000595ac1a1821664a",
        "directives": EXECUTION_DIRECTIVES,
        "request": {"nodes": 1, "ntasks": 192, "exclusive": True, "mem": "0"},
        "launch": EXECUTION_LAUNCH,
        "configuration": {},
    },
    "execution_shared": {
        "flag": "--execution-shared",
        "note": "scientific execution acceptance: the same Si search on 48 cores of a shared node (3 x 16 ranks)",
        "plan_digest": "sha256:0b3b09de2b115cc8a72ea43b2e22ac5db2545e0d5012ff7057a0b6cbc8469e79",
        "resources_sha256": "sha256:d4eaedfd95cdb2c797dbf637c7abb1264c82a0185ad6652051a413c3b6bea56f",
        "directives": (
            "#SBATCH -p leeburton-pool", "#SBATCH --account=power-leeburton-users_v2", "#SBATCH --qos=owner100d",
            "#SBATCH --time=04:00:00", "#SBATCH --nodes=1", "#SBATCH --ntasks=48", "#SBATCH --mem=32G",
            "#SBATCH --exclude=compute-0-[332-333]",
        ),
        "request": {"nodes": 1, "ntasks": 48, "exclusive": False, "mem": "32G"},
        "launch": {"mode": "srun_steps", "parallel_relaxations": 3, "ranks_per_relaxation": 16},
        "configuration": {"STRUCTURE_SEARCH_LAUNCH_MODE": "srun_steps", "STRUCTURE_SEARCH_PARALLEL_RELAXATIONS": 3,
                          "STRUCTURE_SEARCH_RANKS_PER_RELAXATION": 16, "STRUCTURE_SEARCH_EXCLUSIVE_NODE": False,
                          "STRUCTURE_SEARCH_JOB_MEMORY": "32G"},
    },
}
_CONFIGURATION_RESTORE: dict = {}


def apply_profile_configuration(profile: dict | None) -> dict:
    """Set the profile's node and launch settings in this process's backend.config (nothing else; never the
    environment). Returns what was set. Settings an earlier call made are restored first."""

    sys.path.insert(0, str(REPO))
    from backend import config

    restore_profile_configuration()
    values = dict(EXECUTION_PROFILES[profile["profile"]]["configuration"]) if profile and \
        profile["profile"] in EXECUTION_PROFILES else {}
    for name, value in values.items():
        _require(hasattr(config, name), f"backend.config has no setting {name}.")
        _CONFIGURATION_RESTORE[name] = getattr(config, name)
        setattr(config, name, value)
    return values


def restore_profile_configuration() -> None:
    from backend import config

    for name, value in _CONFIGURATION_RESTORE.items():
        setattr(config, name, value)
    _CONFIGURATION_RESTORE.clear()


def execution_profile(profile: dict | None) -> dict | None:
    """The definition of a scientific execution profile, or None for the other kinds of root."""

    return EXECUTION_PROFILES.get(profile["profile"]) if profile else None
POWER_EVIDENCE_PARENT = "/bmd-db/guest/validation"


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


def canonical_acceptance_walltime(value: str) -> str:
    """The acceptance walltime in the planner's canonical form, within 10-15 minutes, or a refusal."""

    sys.path.insert(0, str(REPO))
    from backend.calculations.resources import walltime_seconds
    from backend.structure_search.spec import canonical_walltime

    try:
        walltime = canonical_walltime(value)
    except Exception as exc:  # noqa: BLE001 - the planner's own validation message
        raise ValidationFailed(f"VALIDATION FAILED: walltime {value!r} refused by the planner: {exc}") from None
    low, high = ACCEPTANCE_WALLTIME_RANGE_S
    _require(low <= walltime_seconds(walltime) <= high,
             f"the acceptance walltime must be 10-15 minutes, not {walltime}.")
    return walltime


def setup(root: Path, *, acceptance_walltime: str | None = None, execution: bool | str = False) -> None:
    sys.path.insert(0, str(REPO))
    from compute_api.auth import TOKENS_FILE_SCHEMA, TOKENS_FILE_SCHEMA_VERSION, generate_token

    execution = "execution" if execution is True else (execution or None)
    _require(execution is None or execution in EXECUTION_PROFILES, f"unknown execution profile {execution!r}.")
    _require(not (execution and acceptance_walltime is not None),
             "choose one profile: --acceptance-walltime (operational), --execution or --execution-shared "
             "(scientific execution).")
    walltime = canonical_acceptance_walltime(acceptance_walltime) if acceptance_walltime is not None else None
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
    marker = {"kind": ROOT_KIND, "created_at": _now(), "principal": PRINCIPAL}
    if walltime is not None:
        marker["acceptance"] = {"walltime": walltime, "search": {"walltime": walltime},
                                "note": "submission-and-cancellation acceptance: short allocation via the planner"}
    if execution:
        definition = EXECUTION_PROFILES[execution]
        marker["acceptance"] = {"profile": execution, "walltime": EXECUTION_WALLTIME,
                                "search": dict(EXECUTION_SEARCH), "plan_digest": definition["plan_digest"],
                                "request": dict(definition["request"]), "launch": dict(definition["launch"]),
                                "note": definition["note"]}
        if execution == "execution":  # the 4 x 48 roots made before the shared profile existed carry no more
            marker["acceptance"].pop("request")
            marker["acceptance"].pop("launch")
            marker["acceptance"]["note"] = "scientific execution acceptance: a 2-atom, two-generation Si search"
    _private_file(root / "VALIDATION_ROOT.json", json.dumps(marker, indent=2) + "\n")
    print(f"Validation root ready: {root}" + (f" (acceptance, walltime {walltime})" if walltime else "")
          + (f" ({execution} acceptance, walltime {EXECUTION_WALLTIME})" if execution else ""))


def acceptance_profile(root: Path) -> dict | None:
    """The root's acceptance profile: None (staging/promotion only), ``operational`` or ``execution``."""

    marker = json.loads((root / "VALIDATION_ROOT.json").read_text(encoding="utf-8"))
    acceptance = marker.get("acceptance")
    if acceptance is None:
        return None
    if acceptance.get("profile") in EXECUTION_PROFILES:
        name = acceptance["profile"]
        definition = EXECUTION_PROFILES[name]
        _require(acceptance.get("walltime") == EXECUTION_WALLTIME and acceptance.get("search") == EXECUTION_SEARCH
                 and acceptance.get("plan_digest") == definition["plan_digest"]
                 and (name == "execution" or (acceptance.get("request") == definition["request"]
                                              and acceptance.get("launch") == definition["launch"])),
                 f"the validation root's {name} profile is not the approved one.")
        return {"profile": name, "walltime": EXECUTION_WALLTIME, "search": dict(EXECUTION_SEARCH),
                "plan_digest": definition["plan_digest"]}
    _require("profile" not in acceptance, f"unknown acceptance profile {acceptance.get('profile')!r}.")
    walltime = canonical_acceptance_walltime(acceptance.get("walltime"))
    _require(acceptance == {**acceptance, "walltime": walltime, "search": {"walltime": walltime}},
             "the validation root's acceptance settings are inconsistent.")
    return {"profile": "operational", "walltime": walltime, "search": {"walltime": walltime}, "plan_digest": None}


def acceptance_walltime(root: Path) -> str | None:
    profile = acceptance_profile(root)
    return profile["walltime"] if profile else None


def search_body(root: Path) -> dict:
    """The planning request: Si with the search defaults, or with the profile's Advanced settings."""

    profile = acceptance_profile(root)
    return {"formula": FORMULA, **({"search": dict(profile["search"])} if profile else {})}


def check_allocation(resources: dict, walltime: str, where: str) -> None:
    """A resource request (plan, acknowledgement or response) that asks for exactly ``walltime``, once."""

    _require(resources["request"]["walltime"] == walltime,
             f"{where}: the requested walltime is {resources['request']['walltime']}, not {walltime}.")
    times = [line for line in resources["sbatch_directives"] if "--time" in line]
    _require(times == [f"#SBATCH --time={walltime}"], f"{where}: the time directives are {times}, not --time={walltime}.")


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


def _log_staging_diagnostics() -> None:
    """Show the service's staging timings (phases, finalize on POWER) on stderr, once per process."""

    import logging

    logger = logging.getLogger("compute_api.structure_search_execution")
    if not any(getattr(handler, "_bmd_validation", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
        handler._bmd_validation = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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


def _open_api(root: Path, *, allow_dirty: bool, runner_factory, proc: Path) -> dict:
    """The machine API in-process, with this validation's own ledger and token only."""

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
    # The validation's own in-process service only: production keeps structure-search execution off.
    os.environ["BMD_STRUCTURE_SEARCH_EXECUTION"] = "enabled"
    os.chdir(REPO)  # the app resolves its templates relative to the checkout
    sys.path.insert(0, str(REPO))
    import warnings

    configured = apply_profile_configuration(acceptance_profile(root))  # before anything plans

    try:
        from starlette.testclient import TestClient
    except ImportError as exc:  # httpx missing: report it, never install into the service environment
        raise ValidationFailed(f"The in-process API client is unavailable ({exc}); stop and report.") from None

    import main
    from compute_api import execution
    from compute_api.ledger import STATE_DIR_ENV
    from compute_api.structure_search_execution import reverify_staged_search

    _require(os.environ[STATE_DIR_ENV] == str(state), "the ledger is not the validation's own.")
    _log_staging_diagnostics()
    if runner_factory is not None:
        execution.RUNNER_FACTORY = runner_factory
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = TestClient(main.app, client=LOOPBACK)
    return {"client": client, "token": token, "state": state, "evidence": evidence, "commit": commit,
            "dirty": dirty, "reverify_staged_search": reverify_staged_search, "configuration": configured}


def _write_sums(evidence: Path) -> None:
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
            for p in sorted(evidence.iterdir()) if p.is_file() and p.name != "SHA256SUMS"]
    sums_path = evidence / "SHA256SUMS"
    sums_path.unlink(missing_ok=True)
    _private_file(sums_path, "\n".join(sums) + "\n")


def run(root: Path, *, allow_dirty: bool = False, runner_factory=None, proc: Path = Path("/proc")) -> dict:
    root = Path(os.path.abspath(root))  # before _open_api changes the working directory
    api = _open_api(root, allow_dirty=allow_dirty, runner_factory=runner_factory, proc=proc)
    client, token, state, evidence = api["client"], api["token"], api["state"], api["evidence"]
    commit, dirty, reverify_staged_search = api["commit"], api["dirty"], api["reverify_staged_search"]
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
                                         search_body(root))
        _require(status == 200, f"planning returned {status}.")
        digest = plan["plan_digest"]
        definition = execution_profile(acceptance_profile(root))
        if definition:
            _require(digest == definition["plan_digest"],
                     f"the planner gives {digest} for the execution search, not the approved "
                     f"{definition['plan_digest']}; nothing was prepared.")
        attempt_id = str(uuid.uuid4())
        _private_file(attempt_file, json.dumps({"attempt_id": attempt_id, "plan_digest": digest}, indent=2) + "\n")

    definition = execution_profile(acceptance_profile(root))
    if definition:
        _require(digest == definition["plan_digest"], "this attempt's plan is not the approved execution plan.")
    route = f"/api/v1/structure-search-attempts/{attempt_id}"
    status, prepared = recorder.exchange(client, "prepare", "PUT", route, token,
                                         {**search_body(root), "expected_plan_digest": digest})
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
    if acceptance_walltime(root):
        check_allocation(staged["resources"], acceptance_walltime(root), "the staged plan")
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
    _write_sums(evidence)
    print(json.dumps({key: summary[key] for key in ("attempt_id", "run_name", "plan_digest")}, indent=2))
    return summary


def promote(root: Path, *, allow_dirty: bool = False, runner_factory=None, proc: Path = Path("/proc")) -> dict:
    """Promote this validation's STAGED attempt to submission-ready (SEARCH_PREPARED). Never submits.

    Reads the attempt's provisional resource request, acknowledges exactly
    that request, and promotes. Repeating it after an interruption resumes
    the same promotion; on an attempt already PREPARED it returns the stored
    result. It calls no submission route (there is none) and starts nothing.
    """

    root = Path(os.path.abspath(root))  # before _open_api changes the working directory
    api = _open_api(root, allow_dirty=allow_dirty, runner_factory=runner_factory, proc=proc)
    client, token, state, evidence = api["client"], api["token"], api["state"], api["evidence"]
    attempt_file = evidence / "attempt.json"
    _require(attempt_file.is_file(), "no attempt: run 'run' first.")
    attempt = json.loads(attempt_file.read_text(encoding="utf-8"))
    attempt_id, digest = attempt["attempt_id"], attempt["plan_digest"]
    route = f"/api/v1/structure-search-attempts/{attempt_id}"
    recorder = Recorder(evidence)

    status, current = recorder.exchange(client, "before_promotion", "GET", route, token)
    _require(status == 200 and current["state"] in {"STAGED", "PREPARED"},
             f"the attempt is {current.get('state')}, not STAGED or PREPARED.")
    resources = current["resources"]
    _require(resources["status"] == "provisional_estimate" and resources["approved"] is False,
             "unexpected resource status.")
    if acceptance_walltime(root):
        check_allocation(resources, acceptance_walltime(root), "the plan being acknowledged")
    print(f"Acknowledging the provisional resource request {resources['resources_sha256']}:")
    print("  " + " ".join(resources["sbatch_directives"]))

    body = {"expected_plan_digest": digest,
            "resource_acknowledgement": {"acknowledged": True, "resources_sha256": resources["resources_sha256"]}}
    status, promoted = recorder.exchange(client, "promote", "PUT", f"{route}/promotion", token, body)
    if status != 200:
        code = (promoted.get("error") or {}).get("code")
        raise ValidationFailed(
            f"Promotion returned {status} ({code}). Evidence is in {evidence}. A transport interruption (503) "
            "may be retried by running 'promote' again; any other code is a finding: stop and report it."
        )
    promotion = promoted["promotion"]
    _require(promoted["state"] == "PREPARED" and promoted["submission_ready"] is True, "not submission-ready.")
    _require(promoted["submission_available"] is True, "not ready to submit (the kit never submits it).")
    _require(promotion["remote_state"] == "SEARCH_PREPARED", "POWER did not record SEARCH_PREPARED.")
    _require(promotion["remote_integrity"] == "last_verified", "PREPARED must be reported as last verified.")
    _require((promotion["runtime_check"] or {}).get("status") == "passed", "the readiness check did not pass.")
    _check_no_potcar_contents(promoted)
    status, read_back = recorder.exchange(client, "after_promotion", "GET", route, token)
    _require(status == 200 and read_back == promoted, "read-back differs from the promotion response.")

    record = json.loads((state / "structure_search" / "attempts" / f"{attempt_id}.json").read_text(encoding="utf-8"))
    summary = {
        "result": "SEARCH_PREPARED (submission-ready); nothing submitted",
        "commit": api["commit"],
        "attempt_id": attempt_id,
        "plan_digest": digest,
        "run_name": record["staging"]["run_name"],
        "job_script": promotion["job_script"],
        "acknowledgement_sha256": promotion["acknowledgement"]["acknowledgement_sha256"],
        "resources_sha256": promotion["acknowledgement"]["resources_sha256"],
        "modules": promotion["modules"],
        "verified_at": promotion["verified_at"],
        "runtime_check": promotion["runtime_check"],
        "calculation_ledger_entries": sorted(p.name for p in (state / "attempts").glob("*.json")),
    }
    _require(summary["calculation_ledger_entries"] == [], "a calculation attempt was created.")
    recorder.save("promoted_record", record)
    recorder.save("promotion_summary", summary)
    _write_sums(evidence)
    print(json.dumps({key: summary[key] for key in ("attempt_id", "run_name", "job_script")}, indent=2))
    return summary


# -- acceptance: submit, observe, cancel (step 4D, part D) ---------------------------------------


def _own_attempt(evidence: Path) -> tuple[str, str]:
    attempt_file = evidence / "attempt.json"
    _require(attempt_file.is_file(), "no attempt: run 'run' and 'promote' first.")
    attempt = json.loads(attempt_file.read_text(encoding="utf-8"))
    attempt_id, digest = attempt["attempt_id"], attempt["plan_digest"]
    _require(bool(ATTEMPT_RE.match(str(attempt_id))), f"malformed attempt ID {attempt_id!r}.")
    _require(attempt_id not in HISTORICAL_ATTEMPTS, f"{attempt_id} is a historical validation attempt; "
                                                    "it is never submitted, observed or cancelled by the kit.")
    return attempt_id, digest


def _own_job(evidence: Path, attempt_id: str) -> dict:
    job_file = evidence / "job.json"
    _require(job_file.is_file(), "no job: this validation has not submitted (run 'submit' first).")
    job = json.loads(job_file.read_text(encoding="utf-8"))
    _require(job.get("attempt_id") == attempt_id and bool(JOB_ID_RE.match(str(job.get("job_id")))),
             "job.json does not belong to this validation's attempt.")
    return job


def _stop_on_error(status: int, payload: dict, step: str, evidence: Path, advice: str) -> None:
    if status == 200:
        return
    error = payload.get("error") or {}
    raise ValidationFailed(f"STOP: {step} returned {status} ({error.get('code')}): {error.get('message')}. "
                           f"Evidence is in {evidence}. {advice}")


def submit(root: Path, *, confirm_attempt: str, confirm_walltime: str | None = None, allow_dirty: bool = False,
           runner_factory=None, proc: Path = Path("/proc")) -> dict:
    """The one submission request of this validation (one `sbatch`, by the service), never repeated by the kit."""

    root = Path(os.path.abspath(root))  # before _open_api changes the working directory
    api = _open_api(root, allow_dirty=allow_dirty, runner_factory=runner_factory, proc=proc)
    client, token, evidence = api["client"], api["token"], api["evidence"]
    attempt_id, digest = _own_attempt(evidence)
    _require(confirm_attempt == attempt_id,
             f"--confirm-attempt must be exactly this validation's attempt ID ({attempt_id}); nothing was sent.")
    marker = evidence / "submission_requested.json"
    _require(not marker.exists(), f"this validation already sent its one submission request ({marker}); it is "
                                  "never sent again. Use 'observe', and report anything unexpected.")
    route = f"/api/v1/structure-search-attempts/{attempt_id}"
    recorder = Recorder(evidence)
    status, current = recorder.exchange(client, "before_submission", "GET", route, token)
    _require(status == 200 and current.get("state") == "PREPARED" and current.get("submission_available") is True,
             f"the attempt is {current.get('state')}, not a submission-ready PREPARED attempt; nothing was sent.")
    promotion = current["promotion"]
    _require(promotion["remote_state"] == "SEARCH_PREPARED", "POWER did not last record SEARCH_PREPARED.")
    profile = acceptance_profile(root)
    _require(profile is not None, "this root was not set up for an acceptance ('setup --acceptance-walltime', "
                                  "'--execution' or '--execution-shared'); the kit never submits a default "
                                  "(168-hour) allocation. Nothing was sent.")
    walltime = profile["walltime"]
    if execution_profile(profile):
        _require(confirm_walltime == walltime,
                 f"--confirm-walltime must be exactly {walltime} for the execution acceptance; nothing was sent.")
    else:
        _require(confirm_walltime in (None, walltime), f"--confirm-walltime must be {walltime}; nothing was sent.")
    allocation = verify_short_allocation(api, recorder, attempt_id=attempt_id, digest=digest, walltime=walltime,
                                         current=current, profile=profile)
    body = {"expected_plan_digest": digest, "job_script_sha256": promotion["job_script"]["sha256"],
            "resource_acknowledgement": {"acknowledged": True,
                                         "resources_sha256": promotion["acknowledgement"]["resources_sha256"]}}
    print(f"Submitting attempt {attempt_id} once, walltime {walltime}, job script {promotion['job_script']['sha256']}:")
    print("  " + " ".join(current["resources"]["sbatch_directives"]))
    _private_file(marker, json.dumps({"attempt_id": attempt_id, "requested_at": _now(), "body": body},
                                     indent=2, sort_keys=True) + "\n")  # before the request: never twice
    status, submitted = recorder.exchange(client, "submit", "PUT", f"{route}/submission", token, body)
    if status != 200:
        code = (submitted.get("error") or {}).get("code")
        if code in UNCERTAIN_CODES or status == 503:
            raise ValidationFailed(f"STOP: UNCERTAIN SUBMISSION ({status} {code}). Do not repeat 'submit' and do not "
                                   f"run sbatch. Keep everything; report it. Evidence is in {evidence}.")
        _stop_on_error(status, submitted, "submission", evidence,
                       "The kit never submits again; report it (a refusal by SLURM means no job exists).")
    sub = submitted["submission"]
    _require(submitted["state"] == "SUBMITTED" and sub["status"] == "submitted",
             f"submission ended at {submitted.get('state')}; stop and report.")
    _require(bool(JOB_ID_RE.match(str(sub["job_id"]))), f"malformed job ID {sub.get('job_id')!r}; stop and report.")
    _require(sub["reconciled"] is False and sub["adopted"] is False and sub["uncertain"] is False,
             "the job was not bound by sbatch's own answer; stop and report.")
    _require(sub["remote_record_pending"] is False,
             "POWER's SEARCH_SUBMITTED record is pending; stop and report (do not submit again).")
    _require(sub["attempt_comment"] == f"bmd_attempt:{attempt_id}" and sub["job_name"] == f"bmds-{attempt_id[:8]}",
             "the recorded job identity is not this attempt's; stop and report.")
    _check_no_potcar_contents(submitted)
    record = json.loads((api["state"] / "structure_search" / "attempts" / f"{attempt_id}.json").read_text())
    job = {"attempt_id": attempt_id, "job_id": sub["job_id"], "job_name": sub["job_name"],
           "attempt_comment": sub["attempt_comment"], "run_name": record["staging"]["run_name"],
           "run_dir": record["staging"]["spec"]["paths"]["run_dir"], "requested_at": sub["requested_at"],
           "submitted_at": sub["submitted_at"], "walltime": walltime, "profile": profile["profile"],
           "job_script_sha256": allocation["job_script_sha256"],
           "acknowledgement_sha256": allocation["acknowledgement_sha256"]}
    _private_file(evidence / "job.json", json.dumps(job, indent=2, sort_keys=True) + "\n")
    recorder.save("submission_summary", {**job, "commit": api["commit"], "state": submitted["state"]})
    _write_sums(evidence)
    print(json.dumps({key: job[key] for key in ("attempt_id", "job_id", "run_name")}, indent=2))
    return job


def rebuilt_job_script(state: Path, attempt_id: str) -> str:
    """The job script the service would submit, rebuilt from the frozen inputs (its SHA-256 checked by the service)."""

    return frozen_inputs(state, attempt_id)[4]


def frozen_inputs(state: Path, attempt_id: str):
    """(plan, local manifest, sources, manifest, job script) rebuilt by the service from the frozen inputs."""

    from compute_api.execution import ExecutionError
    from compute_api.structure_search_execution import SearchAttemptLedger
    from compute_api.structure_search_submission import _inputs

    ledger = SearchAttemptLedger.from_environment_read_only()
    _require(ledger.root == state / "structure_search", "the ledger is not the validation's own.")
    inputs = _inputs(ledger, ledger.read(attempt_id))
    if isinstance(inputs, ExecutionError):
        raise ValidationFailed(f"VALIDATION FAILED: the frozen inputs cannot be rebuilt ({inputs.code}); nothing was "
                               "sent.")
    return inputs


def verify_execution_plan(api: dict, attempt_id: str, digest: str, current: dict, profile: dict | None = None) -> dict:
    """The execution profile's exact plan: pinned digest, resource request and its hash, spec and launch settings."""

    profile = profile or {"profile": "execution"}
    definition = EXECUTION_PROFILES[profile["profile"]]
    pinned = definition["plan_digest"]
    _require(digest == pinned, f"the attempt's plan {digest} is not the approved execution plan {pinned}; nothing "
                               "was sent.")
    resources = current["resources"]
    _require(tuple(resources["sbatch_directives"]) == definition["directives"],
             f"the resource request is {resources['sbatch_directives']}, not the approved one; nothing was sent.")
    request = resources["request"]
    expected = definition["request"]
    _require({key: request.get(key) for key in expected} == expected,
             f"the resource request is not {expected}: {request}; nothing was sent.")
    plan = frozen_inputs(api["state"], attempt_id)[0]
    _require(plan.digest == pinned, "the frozen plan is not the approved execution plan.")
    from backend.structure_search.job import resource_request_sha256

    _require(resource_request_sha256(plan) == definition["resources_sha256"],
             "the frozen plan's resource request is not the approved one (hash); nothing was sent.")
    spec = plan.record["search"]["spec"]
    _require({key: spec.get(key) for key in EXECUTION_SPEC} == EXECUTION_SPEC,
             f"the frozen search settings are not the approved ones: {spec}; nothing was sent.")
    launch = plan.record["runtime"]["launch"]
    _require({key: launch.get(key) for key in definition["launch"]} == definition["launch"],
             f"the frozen launch settings are not {definition['launch']}: {launch}; nothing was sent.")
    _require(plan.record["execution"]["input_txt"]["num_parallel_calcs"] == definition["launch"]["parallel_relaxations"],
             "USPEX's numParallelCalcs is not the profile's concurrency; nothing was sent.")
    _require(len(plan.record["incar_files"]) == EXECUTION_STEPS, "the frozen plan is not the five-step ladder.")
    return {"spec": {key: spec.get(key) for key in EXECUTION_SPEC}, "launch": dict(definition["launch"]),
            "request": dict(expected), "resources_sha256": definition["resources_sha256"],
            "incar_sha256": {name: row["sha256"] for name, row in sorted(plan.record["incar_files"].items())}}


def verify_short_allocation(api: dict, recorder, *, attempt_id: str, digest: str, walltime: str, current: dict,
                            profile: dict | None = None) -> dict:
    """Before the one sbatch: the profile's walltime (and, for ``execution``, the whole frozen plan) in the plan
    digest, the acknowledgement and the job script."""

    from backend.calculations.resources import walltime_seconds

    profile = profile or {"profile": "operational", "search": {"walltime": walltime}}
    execution = None
    if execution_profile(profile):
        execution = verify_execution_plan(api, attempt_id, digest, current, profile)
    else:
        low, high = ACCEPTANCE_WALLTIME_RANGE_S
        _require(low <= walltime_seconds(walltime) <= high, f"walltime {walltime} is outside 10-15 minutes.")
    check_allocation(current["resources"], walltime, "the attempt's resource request")
    client, token = api["client"], api["token"]
    status, replanned = recorder.exchange(client, "replan_acceptance", "POST", "/api/v1/structure-search-plans",
                                          token, {"formula": FORMULA, "search": dict(profile["search"])})
    _require(status == 200 and replanned["plan_digest"] == digest,
             "the plan digest is not the planner's digest for this profile's settings; nothing was sent.")
    status, default = recorder.exchange(client, "replan_default", "POST", "/api/v1/structure-search-plans", token,
                                        {"formula": FORMULA})
    _require(status == 200 and default["plan_digest"] != digest,
             "the plan digest does not depend on the walltime; nothing was sent.")
    record = json.loads((api["state"] / "structure_search" / "attempts" / f"{attempt_id}.json").read_text())
    acknowledgement = record["promotion"]["acknowledgement"]
    _require(acknowledgement["plan_digest"] == digest and acknowledgement["attempt_id"] == attempt_id,
             "the resource acknowledgement is for another plan or attempt; nothing was sent.")
    check_allocation(acknowledgement["resources"], walltime, "the resource acknowledgement")
    script = rebuilt_job_script(api["state"], attempt_id)
    times = [line.strip() for line in script.splitlines() if line.lstrip().startswith("#SBATCH") and "--time" in line]
    _require(times == [f"#SBATCH --time={walltime}"],
             f"the job script's time directives are {times}, not --time={walltime}; nothing was sent.")
    script_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
    _require(script_sha256 == record["promotion"]["job_script"]["sha256"],
             "the rebuilt job script is not the promoted one; nothing was sent.")
    if execution is not None:
        directives = tuple(line.strip() for line in script.splitlines() if line.lstrip().startswith("#SBATCH")
                           and not line.strip().startswith(("#SBATCH --job-name", "#SBATCH --comment",
                                                            "#SBATCH --chdir", "#SBATCH --output",
                                                            "#SBATCH --no-requeue")))
        definition = EXECUTION_PROFILES[profile["profile"]]
        _require(directives == definition["directives"],
                 f"the job script's resource directives are {list(directives)}, not the approved ones; nothing was "
                 "sent.")
        check_allocation(acknowledgement["resources"], walltime, "the resource acknowledgement")
        _require(tuple(acknowledgement["resources"]["sbatch_directives"]) == definition["directives"],
                 "the resource acknowledgement does not cover exactly the approved request; nothing was sent.")
        _require(current["promotion"]["acknowledgement"].get("resources_sha256") == definition["resources_sha256"],
                 "the resource acknowledgement's request hash is not the approved one; nothing was sent.")
    summary = {"profile": profile["profile"], "walltime": walltime, "plan_digest": digest,
               "default_plan_digest": default["plan_digest"],
               "acknowledgement_sha256": acknowledgement["acknowledgement_sha256"],
               "job_script_sha256": script_sha256, "job_script_time_directives": times,
               **({"execution": execution} if execution else {})}
    recorder.save("short_allocation_check" if execution is None else "execution_allocation_check", summary)
    print(f"Allocation verified ({profile['profile']}): {walltime} in the plan digest, the acknowledgement and the "
          "job script.")
    return summary


def verify_observation_chain(state: Path, attempt_id: str) -> int:
    """The observation evidence files: consecutive, private, each naming the previous one's SHA-256."""

    directory = state / "structure_search" / "observations" / attempt_id
    files = sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")) if directory.is_dir() else []
    previous = None
    for index, path in enumerate(files, start=1):
        _require(path.name == f"{index:06d}.json", f"observation evidence {path.name} is out of sequence.")
        _require(not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600,
                 f"observation evidence {path.name} is not a private file.")
        document = json.loads(path.read_text(encoding="utf-8"))
        _require(document.get("sequence") == index and document.get("previous_sha256") == previous
                 and document.get("attempt_id") == attempt_id,
                 f"observation evidence {path.name} does not continue the chain.")
        previous = hashlib.sha256(path.read_bytes()).hexdigest()
    return len(files)


def observe(root: Path, *, allow_dirty: bool = False, runner_factory=None, proc: Path = Path("/proc")) -> dict:
    """One read-only observation of this validation's job. Never submits or cancels."""

    api = _open_api(root, allow_dirty=allow_dirty, runner_factory=runner_factory, proc=proc)
    client, token, evidence = api["client"], api["token"], api["evidence"]
    attempt_id, _ = _own_attempt(evidence)
    job = _own_job(evidence, attempt_id)
    recorder = Recorder(evidence)
    route = f"/api/v1/structure-search-attempts/{attempt_id}"
    status, observed = recorder.exchange(client, "observe", "POST", f"{route}/observations", token)
    _stop_on_error(status, observed, "observation", evidence, "Report it; nothing was changed.")
    count = verify_observation_chain(api["state"], attempt_id)
    _require(observed["submission"]["job_id"] == job["job_id"], "the attempt's job ID changed; stop and report.")
    _require(observed["state"] in ("SUBMITTED", "CANCEL_REQUESTED", *TERMINAL_STATES),
             f"unexpected state {observed['state']}; stop and report.")
    if not observed["observation"]["performed"]:
        print(f"Not observed again yet (rate limit): retry in {observed['observation']['retry_after_s']} s.")
        return {"performed": False, "state": observed["state"]}
    last = observed["monitoring"]["last"]
    identity, scheduler = last["identity"], last["scheduler"]
    summary = {"state": observed["state"], "sequence": last["sequence"], "evidence_files": count,
               "scheduler": {key: scheduler.get(key) for key in ("state", "source", "settled", "submit", "start",
                                                                  "end", "exit_code")},
               "identity": identity["status"], "binding": identity.get("binding"),
               "times": (identity.get("times") or {}).get("status"), "notes": identity.get("notes"),
               "execution": last["execution"].get("outcome"), "blocked": last["classification"]["blocked"],
               "terminal": observed["monitoring"]["terminal"], "cancellation": (observed.get("cancellation") or {}
                                                                                ).get("status")}
    recorder.save("observation_summary", summary)
    _write_sums(evidence)
    print(json.dumps(summary, indent=2))
    _require(not observed["monitoring"]["operator_review"] and not identity["problems"],
             f"STOP: contradiction for operator review: {identity['problems']}.")
    _require(identity.get("binding") == "sbatch", f"STOP: the job is bound by {identity.get('binding')}, not sbatch.")
    _require(identity["status"] in ("verified", "partial"),
             f"STOP: identity {identity['status']} (times: {identity.get('times')}).")
    _require(identity["status"] == "verified" or scheduler.get("source") != "squeue",
             "STOP: squeue reports the queued job without its comment.")
    _require((identity.get("times") or {}).get("status") == "consistent",
             f"STOP: SLURM times not consistent: {identity.get('times')}.")
    _require(last["classification"]["blocked"] in (None, "not_terminal", "terminal_settling"),
             f"STOP: observation blocked ({last['classification']['blocked']}).")
    return summary


def cancel(root: Path, *, confirm_attempt: str, confirm_job: str, allow_dirty: bool = False, runner_factory=None,
           proc: Path = Path("/proc")) -> dict:
    """Request cancellation of exactly this validation's job (`scancel` by the service, after a fresh check)."""

    api = _open_api(root, allow_dirty=allow_dirty, runner_factory=runner_factory, proc=proc)
    client, token, evidence = api["client"], api["token"], api["evidence"]
    attempt_id, digest = _own_attempt(evidence)
    job = _own_job(evidence, attempt_id)
    _require(confirm_attempt == attempt_id and confirm_job == job["job_id"],
             f"--confirm-attempt and --confirm-job must be exactly {attempt_id} and {job['job_id']}; nothing was "
             "sent.")
    recorder = Recorder(evidence)
    route = f"/api/v1/structure-search-attempts/{attempt_id}"
    status, current = recorder.exchange(client, "before_cancellation", "GET", route, token)
    _require(status == 200 and current["submission"]["job_id"] == job["job_id"], "the attempt's job changed; stop.")
    if current["state"] == "CANCEL_REQUESTED":
        _require((current["cancellation"] or {}).get("status") in RESUMABLE_CANCELLATION,
                 f"cancellation is {current['cancellation'].get('status')}: nothing to resume; use 'observe'.")
    else:
        _require(current["state"] == "SUBMITTED", f"the attempt is {current['state']}; nothing to cancel.")
    print(f"Requesting cancellation of job {job['job_id']} (attempt {attempt_id}).")
    body = {"expected_plan_digest": digest, "job_id": job["job_id"], "confirm": True}
    status, cancelled = recorder.exchange(client, "cancel", "PUT", f"{route}/cancellation", token, body)
    _stop_on_error(status, cancelled, "cancellation", evidence,
                   "Report it with the attempt's cancellation record (GET); observe before any new decision.")
    cancellation = cancelled["cancellation"]
    summary = {"state": cancelled["state"], "status": cancellation["status"],
               "command": cancellation["command"]["last"], "check": cancellation.get("last_check")}
    recorder.save("cancellation_summary", summary)
    _write_sums(evidence)
    print(json.dumps(summary, indent=2))
    _require(cancellation["status"] in ACCEPTED_CANCELLATION,
             f"STOP: cancellation {cancellation['status']}; report it and observe (do not repeat without a decision).")
    _require((cancellation.get("last_check") or {}).get("identity") == "verified",
             "STOP: the job's identity was not verified before the command.")
    return summary


# -- POWER scripts over SSH ----------------------------------------------------------------------


def evidence_arguments(root: Path, *, states=frozenset({"STAGED", "PREPARED"})) -> tuple[str, str]:
    """The attempt ID and run name of this validation, checked against each other and the ledger."""

    attempt = json.loads((root / "evidence" / "attempt.json").read_text(encoding="utf-8"))["attempt_id"]
    record = json.loads((root / "state" / "structure_search" / "attempts" / f"{attempt}.json").read_text())
    run_name = record["staging"]["run_name"]
    match = RUN_NAME_RE.match(str(run_name))
    _require(bool(ATTEMPT_RE.match(str(attempt))), f"malformed attempt ID {attempt!r}.")
    _require(bool(match) and match["prefix"] == attempt[:8], f"run name {run_name!r} does not match the attempt.")
    _require(record["state"] in states, f"the attempt is {record['state']}, not one of {sorted(states)}.")
    return attempt, run_name


def remote_command(*arguments: str) -> str:
    """The command the remote login shell runs: ``bash -s -- ARG...`` with every argument quoted."""

    return "bash -s -- " + " ".join(shlex.quote(argument) for argument in arguments) if arguments else "bash -s"


def acceptance_arguments(root: Path) -> tuple[str, str, str, str]:
    """The attempt ID, run name and job ID of this validation, checked against each other and the ledger."""

    attempt, run_name = evidence_arguments(root, states={"SUBMITTED", "CANCEL_REQUESTED", *TERMINAL_STATES})
    job = _own_job(root / "evidence", attempt)
    record = json.loads((root / "state" / "structure_search" / "attempts" / f"{attempt}.json").read_text())
    _require(record["submission"]["job_id"] == job["job_id"], "job.json and the ledger name different jobs.")
    _require(attempt not in HISTORICAL_ATTEMPTS, "a historical validation attempt.")
    walltime = acceptance_walltime(root)
    _require(walltime is not None and job.get("walltime") == walltime, "job.json and the root name different walltimes.")
    return attempt, run_name, job["job_id"], walltime


def run_power_script(root: Path, script, output_name: str, arguments: tuple[str, ...] = ()) -> int:
    """Run one kit script on POWER as bmdguest; record its output beside the evidence (never overwritten)."""

    output = root / output_name
    _require(not output.exists(), f"{output} exists; it is not overwritten.")
    command = ["ssh", "-o", "BatchMode=yes", SSH_HOST, remote_command(*arguments)]
    with open(script if isinstance(script, Path) else HERE / script, "rb") as stdin:
        completed = subprocess.run(command, stdin=stdin, capture_output=True)
    text = completed.stdout.decode("utf-8", "replace") + completed.stderr.decode("utf-8", "replace")
    _private_file(output, text)
    sys.stdout.write(text)
    return completed.returncode


def preflight(root: Path, *, proc: Path = Path("/proc")) -> int:
    check_isolation(Path(os.path.abspath(root)), proc=proc)
    return run_power_script(Path(os.path.abspath(root)), "power_preflight.sh", "power_preflight.txt")


def acceptance_evidence(root: Path, *, proc: Path = Path("/proc")) -> int:
    """Read-only POWER evidence of the submitted job; a new numbered output each time (never overwritten)."""

    root = Path(os.path.abspath(root))
    check_isolation(root, proc=proc)
    arguments = acceptance_arguments(root)
    index = 1 + len(list(root.glob("power_acceptance_evidence_*.txt")))
    return run_power_script(root, "power_acceptance_evidence.sh", f"power_acceptance_evidence_{index:02d}.txt",
                            arguments)


# -- scientific execution evidence (execution_validation/) -----------------------------------------------------


def execution_arguments(root: Path) -> tuple[str, str, str]:
    """Attempt ID, run name and job ID of an execution-profile validation whose job has ended."""

    profile = acceptance_profile(root)
    definition = execution_profile(profile)
    _require(definition is not None,
             "this root was not set up for an execution acceptance ('setup --execution' or '--execution-shared').")
    attempt, run_name = evidence_arguments(root, states=set(TERMINAL_STATES))
    job = _own_job(root / "evidence", attempt)
    record = json.loads((root / "state" / "structure_search" / "attempts" / f"{attempt}.json").read_text())
    _require(record["submission"]["job_id"] == job["job_id"], "job.json and the ledger name different jobs.")
    _require(record["plan_digest"] == definition["plan_digest"],
             "the attempt's plan is not the approved execution plan.")
    return attempt, run_name, job["job_id"]


def execution_expected(root: Path, attempt: str, job_id: str) -> dict:
    """What the evidence must show, from the validation's own frozen records (never from the evidence)."""

    record = json.loads((root / "state" / "structure_search" / "attempts" / f"{attempt}.json").read_text())
    plans = sorted((root / "state" / "structure_search" / "preparations" / attempt).rglob("search_plan.json"))
    _require(len(plans) == 1, "the frozen plan of this attempt was not found exactly once.")
    plan = json.loads(plans[0].read_text(encoding="utf-8"))
    plan_record = plan.get("record", plan)
    incars = plan_record.get("incar_files") or {}
    _require(len(incars) == EXECUTION_STEPS, "the frozen plan does not hold the five ladder INCARs.")
    job = _own_job(root / "evidence", attempt)
    # Concurrency, ranks, CPUs and memory from the frozen plan itself (pinned by its digest), not from constants.
    launch = plan_record["runtime"]["launch"]
    request = plan_record["resources"]["request"]
    definition = execution_profile(acceptance_profile(root))
    _require(definition is not None and {key: launch.get(key) for key in definition["launch"]} == definition["launch"]
             and {key: request.get(key) for key in definition["request"]} == definition["request"],
             "the frozen plan's launch settings or resource request are not the profile's.")
    _require(request["ntasks"] == launch["parallel_relaxations"] * launch["ranks_per_relaxation"],
             "the frozen plan's tasks are not its concurrency x ranks.")
    return {
        "attempt_id": attempt, "job_id": job_id, "plan_digest": record["plan_digest"],
        "walltime": request["walltime"], "parallel": launch["parallel_relaxations"],
        "ranks": launch["ranks_per_relaxation"], "ncpus": request["ntasks"], "exclusive": request["exclusive"],
        "job_memory": request["mem"], "steps": EXECUTION_STEPS,
        "generations": EXECUTION_SEARCH["num_generations"],
        # One cell size (min_atoms = max_atoms = 2) and zero pressure (admission policy): USPEX's
        # listed energy is then F per cell, or per atom, never an enthalpy.
        "cell_atoms": EXECUTION_SEARCH["min_atoms"], "pressure_gpa": 0.0,
        "job_script_sha256": record["promotion"]["job_script"]["sha256"],
        "acknowledgement_sha256": record["promotion"]["acknowledgement"]["acknowledgement_sha256"],
        "incar_sha256": {name: row["sha256"] for name, row in sorted(incars.items())},
        "job_json": job,
    }


def fetch_power_directory(remote_dir: str, target: Path) -> None:
    """Copy one evidence directory from POWER (``tar`` over SSH) into a new local directory, links refused."""

    parent, _, leaf = remote_dir.rpartition("/")
    _require(parent == POWER_EVIDENCE_PARENT and re.fullmatch(r"uspex_execution_4d_[0-9TZ]+_[0-9a-f]{8}_[0-9]+", leaf)
             is not None, f"unexpected evidence directory {remote_dir!r}.")
    parent, name = remote_dir.rsplit("/", 1)
    command = ["ssh", "-o", "BatchMode=yes", SSH_HOST, f"tar -C {shlex.quote(parent)} -czf - {shlex.quote(name)}"]
    completed = subprocess.run(command, capture_output=True)
    _require(completed.returncode == 0, f"fetching {remote_dir} failed: {completed.stderr.decode()[-300:]}")
    target.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            _require(member.isfile() or member.isdir(), f"{member.name} is not a file or directory.")
            _require(parts and parts[0] == name and ".." not in parts and not Path(member.name).is_absolute(),
                     f"{member.name} is outside the evidence directory.")
            destination = target.joinpath(*parts[1:]) if len(parts) > 1 else target
            if member.isdir():
                destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = tar.extractfile(member)
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(source.read())


def load_analyser():
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze_execution", EXECUTION_DIR / "analyze_execution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execution_evidence(root: Path, *, proc: Path = Path("/proc")) -> int:
    """Collect the POWER evidence of the ended execution job (read-only), fetch it, and analyse it locally.

    Returns 0 for PASSED, 1 for FAILED, 2 for INCOMPLETE EVIDENCE (or a collection failure).
    """

    root = Path(os.path.abspath(root))
    check_isolation(root, proc=proc)
    attempt, run_name, job_id = execution_arguments(root)
    expected = execution_expected(root, attempt, job_id)
    index = 1 + len(list(root.glob("power_execution_evidence_*.txt")))
    output = f"power_execution_evidence_{index:02d}.txt"
    returncode = run_power_script(root, EXECUTION_DIR / "power_execution_evidence.sh", output,
                                  (attempt, run_name, job_id))
    text = (root / output).read_text(encoding="utf-8", errors="replace")
    found = re.findall(r"^EVIDENCE DIRECTORY: (\S+)$", text, flags=re.M)
    if not found:
        print("STOP: the collector made no evidence directory (see the output above).")
        return 2
    target = root / f"execution_evidence_{index:02d}"
    fetch_power_directory(found[-1], target)
    analyser = load_analyser()
    report = analyser.analyse(target, {key: value for key, value in expected.items() if key != "job_json"})
    report["collector_returncode"] = returncode
    report["power_evidence_directory"] = found[-1]
    _private_file(root / f"execution_analysis_{index:02d}.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\n".join(analyser.summary_lines(report)))
    if returncode != 0:
        print("STOP: the collector reported a failure; the analysis above is not a pass.")
        return 1
    return {"PASSED": 0, "FAILED": 1}.get(report["result"], 2)


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
    members += sorted(p for p in (root / "state" / "structure_search" / "observations").rglob("*") if p.is_file())
    members += [root / name for name in ("run.log", "power_preflight.txt", "power_evidence.txt")
                if (root / name).is_file()]
    members += sorted(root.glob("power_acceptance_evidence_*.txt"))
    members += sorted(root.glob("power_execution_evidence_*.txt")) + sorted(root.glob("execution_analysis_*.json"))
    for directory in sorted(root.glob("execution_evidence_*")):
        members += sorted(p for p in directory.rglob("*") if p.is_file())
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
    for name in ("setup", "preflight", "run", "promote", "evidence", "submit", "observe", "cancel",
                 "acceptance-evidence", "execution-evidence", "archive"):
        sub.add_parser(name).add_argument("root", type=Path)
    for name in ("run", "promote", "submit", "observe", "cancel"):
        sub.choices[name].add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    for name in ("submit", "cancel"):
        sub.choices[name].add_argument("--confirm-attempt", required=True, metavar="ATTEMPT_ID")
    sub.choices["cancel"].add_argument("--confirm-job", required=True, metavar="JOB_ID")
    sub.choices["setup"].add_argument("--acceptance-walltime", metavar="HH:MM:SS",
                                      help="a submission acceptance root: this 10-15 minute walltime (planner override)")
    sub.choices["setup"].add_argument("--execution", action="store_true",
                                      help="a scientific execution acceptance root (2-atom, two-generation Si, 4 h, "
                                           "one exclusive 192-core node)")
    sub.choices["setup"].add_argument("--execution-shared", action="store_true",
                                      help="the same acceptance on 48 cores of a shared node (3 x 16 ranks, 32G)")
    sub.choices["submit"].add_argument("--confirm-walltime", metavar="HH:MM:SS",
                                       help="required for the execution profile: exactly its walltime")
    args = parser.parse_args(argv)
    if args.command == "submit":
        submit(args.root, confirm_attempt=args.confirm_attempt, confirm_walltime=args.confirm_walltime,
               allow_dirty=args.allow_dirty)
        return 0
    if args.command == "observe":
        observe(args.root, allow_dirty=args.allow_dirty)
        return 0
    if args.command == "cancel":
        cancel(args.root, confirm_attempt=args.confirm_attempt, confirm_job=args.confirm_job,
               allow_dirty=args.allow_dirty)
        return 0
    if args.command == "acceptance-evidence":
        return acceptance_evidence(args.root)
    if args.command == "execution-evidence":
        return execution_evidence(args.root)
    if args.command == "setup":
        _require(not (args.execution and args.execution_shared), "choose --execution or --execution-shared.")
        setup(args.root, acceptance_walltime=args.acceptance_walltime,
              execution="execution_shared" if args.execution_shared else args.execution)
        return 0
    if args.command == "run":
        run(args.root, allow_dirty=args.allow_dirty)
        return 0
    if args.command == "promote":
        promote(args.root, allow_dirty=args.allow_dirty)
        return 0
    if args.command == "archive":
        archive(args.root)
        return 0
    return {"preflight": preflight, "evidence": evidence}[args.command](args.root)


if __name__ == "__main__":
    raise SystemExit(main())
