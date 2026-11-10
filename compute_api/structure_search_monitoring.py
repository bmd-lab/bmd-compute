"""Monitoring and cancellation of submitted structure searches (USPEX step 4D-4).

Service-side only: nothing under ``backend/`` changes, so the runtime package
(and every attempt's staged copy of it) is unchanged.

Observation: ``POST /api/v1/structure-search-attempts/{id}/observations``
(scope ``read``, the attempt's owner). One read-only SSH session through the
shared operation limiter (no claim: nothing is written on POWER) reads

* ``squeue`` for this account (every queued job, so another job carrying
  this attempt's identity is seen) and ``sacct -X -j <job>``, both with
  ``TZ=UTC SLURM_TIME_FORMAT=standard`` so every SLURM time is ISO 8601 UTC;
* POWER's clock in UTC (``date -u``, epoch and ISO, which must agree) and its
  local offset (``date +%z``, kept as evidence), bracketed by the service's
  clock so the skew between the two hosts is measured, not assumed;
* the remote attempt state, and from the run directory, through the
  contained reader (no symbolic links, bounded): ``job_started``,
  ``search_status.json`` and the launcher heartbeat. The SLURM output is
  never read.

Identity: the job must be the one this attempt bound (from ``sbatch``'s own
answer, a verified reconciliation, or POWER's record of either), reported
with the attempt's exact job name and run directory, with no other job
carrying the attempt's comment or directory, and submitted inside this
attempt's submission window (``check_times``). A different non-empty
comment, any other differing field, a submission outside the window, a
``job_started`` or ``search_status.json`` naming another job, or a remote
record naming another job is a contradiction: recorded, and nothing changes.
A comment SLURM does not report (accounting may not keep it) leaves the
identity ``partial``, which is enough to follow an already-bound job but
never enough to adopt an uncertain submission or to cancel. Timestamps that
cannot be interpreted reliably leave the identity ``unverified``: nothing is
guessed.

Three answers are kept apart and each carries its source and time: the
scheduler state (``squeue`` while queued, ``sacct`` otherwise; ``UNKNOWN``
when neither reports the job, never terminal), the execution outcome
(``search_status.json``, with its SHA-256), and scientific validity
(``not_assessed``). A terminal state is recorded only when accounting reports
it, ``squeue`` does not contradict it, and the job ended at least
``SETTLE_SECONDS`` before POWER's current UTC time; classification is
``lifecycle.classify_job``. A terminal state is never changed afterwards; a
later observation that disagrees is kept as evidence for an operator.

Every observation is a new, write-once evidence file
(``<state root>/structure_search/observations/<attempt>/<sequence>.json``,
linked into place exclusively, each naming the previous one's SHA-256); the
local record keeps a summary. If the record update is interrupted, the next
request applies the evidence first.

Cancellation: ``PUT …/cancellation`` (scope ``submit``, the owner) with
``{"expected_plan_digest", "job_id", "confirm": true}``:

1. *requested*: the record becomes ``CANCEL_REQUESTED`` before POWER;
2. under the attempt's claim, POWER's record must name this job and identity,
   and ``squeue`` must report the job queued or running with its full
   identity (comment, name, directory, submission window) and no other job
   carrying it; then ``SEARCH_CANCEL_REQUESTED`` is written atomically and the
   local record notes it (*remote request recorded*), and only then
   ``scancel <job ID>`` runs, never by name or user (*command attempted*);
3. *observed*: only a later observation of SLURM's settled terminal state.

Where it stops decides what happens to the record:

* refused or failed before any remote cancellation request exists (local
  record and POWER agree): back to ``SUBMITTED``, the refusal kept;
* the remote write itself interrupted: ``CANCEL_REQUESTED``,
  ``remote_request_uncertain``;
* remote request recorded, ``scancel`` not attempted:
  ``CANCEL_REQUESTED``, ``remote_recorded_command_pending``;
* ``scancel`` attempted: ``CANCEL_REQUESTED`` with its outcome
  (``command_outcome_unknown`` when the answer was lost).

The pending states are resumed only by a repeated request, which checks the
job's identity afresh before any ``scancel``. Nothing is cancelled
automatically, and nothing is resubmitted or restarted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from compute_api.execution import ExecutionError, _authorize_record
from compute_api.ledger import isoformat, utc_now
from compute_api.structure_search_execution import (
    LOGGER,
    SearchAttemptLedger,
    _calculation_id_conflict,
    _claimed_runner,
    _private_directory,
    _staging_failed,
    _write_search_state,
    search_attempt_response,
)
from compute_api.structure_search_promotion import _remote_record
from compute_api.structure_search_submission import (
    STATE_SUBMISSION_UNCERTAIN,
    STATE_SUBMITTED,
    _identity_matches,
    _promotion_matches,
    job_identity,
    reconcile_jobs,
)

STATE_CANCEL_REQUESTED = "CANCEL_REQUESTED"
REMOTE_SEARCH_SUBMITTING = "SEARCH_SUBMITTING"
REMOTE_SEARCH_SUBMITTED = "SEARCH_SUBMITTED"
REMOTE_SEARCH_CANCEL_REQUESTED = "SEARCH_CANCEL_REQUESTED"

OBSERVATION_SCHEMA = "bmd_compute.structure_search_observation"
OBSERVATION_SCHEMA_VERSION = 1
CLASSIFIER_VERSION = 2
OBSERVATIONS_DIRNAME = "observations"

SETTLE_SECONDS = 120  # a terminal state is recorded only this long after the job's end (POWER's UTC clock)
MIN_OBSERVATION_INTERVAL_SECONDS = 60  # requests within this interval return the last observation
# The submission window: SLURM's submit time (POWER's clock, whole seconds) must lie between the attempt's
# request (recorded before verification and sbatch) and the moment the job was bound (sbatch's answer, or the
# verified reconciliation), both on the service's clock and shifted by the measured skew. The slack covers whole-
# second rounding, the uncertainty of the skew measurement (added separately) and clock drift between the
# submission and the observation under NTP (well under a second a day): a minute is generous for that and far
# below the months that separate reused job IDs.
SUBMIT_WINDOW_SLACK_SECONDS = 60
MAX_CLOCK_SKEW_SECONDS = 300  # beyond this the two clocks are not trusted to compare times at all
# A submission time outside the window by less than the largest UTC offset could be a time printed in a local
# time zone rather than UTC: unverified, not a contradiction. Further out (a reused job ID is months away), it is.
MAX_ZONE_OFFSET_SECONDS = 14 * 3600
FUTURE_SLACK_SECONDS = 5  # a SLURM time later than POWER's own clock means it is not UTC (or not reliable)
MAX_STATUS_BYTES = 256 * 1024
MAX_SMALL_FILE_BYTES = 4096
MAX_KEPT = 20  # contradictions, commands and earlier requests kept in the record (all are in the evidence)
_JOB_ID = re.compile(r"^[0-9]{1,20}$")
_UTC_ENV = "TZ=UTC SLURM_TIME_FORMAT=standard"

SQUEUE_FIELDS = ("job_id", "job_name", "comment", "workdir", "state", "reason", "submit")
SACCT_FIELDS = ("job_id", "job_name", "comment", "workdir", "state", "exit_code", "submit", "start", "end")
SQUEUE_COMMAND = f"{_UTC_ENV} squeue -h -u \"$(id -un)\" -o '%i|%j|%k|%Z|%T|%r|%V'"
CLOCK_COMMAND = "date -u +%s; date -u +%Y-%m-%dT%H:%M:%S; date +%z"


def _terminal_states() -> tuple[str, ...]:
    from backend.structure_search.lifecycle import TERMINAL_STATES

    return TERMINAL_STATES


def _sacct_command(job_id: str) -> str:
    return (f"{_UTC_ENV} sacct -X -n -P -j {shlex.quote(job_id)} "
            "-o JobID,JobName,Comment,WorkDir,State,ExitCode,Submit,Start,End")


def _scancel_command(job_id: str) -> str:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("scancel takes one numeric job ID")
    return f"scancel {job_id}"  # exactly one numeric job ID: never a name, a user or any other option


# -- local transitions (service-side; the shared table in lifecycle covers SUBMITTED -> terminal) ---------


def _transition(record: dict, target: str) -> None:
    from backend.structure_search import lifecycle

    current = record["state"]
    extra = {(STATE_SUBMITTED, STATE_CANCEL_REQUESTED),
             (STATE_CANCEL_REQUESTED, STATE_SUBMITTED)} | {  # refused before any remote request
        (STATE_CANCEL_REQUESTED, state) for state in _terminal_states()}
    if (current, target) not in extra:
        lifecycle.require_local_transition(current, target)
    record["state"] = target
    record["updated_at"] = isoformat(utc_now())


# -- parsing ----------------------------------------------------------------------------------------------


def _clean(value: str) -> str:
    value = value.strip()
    return "" if value in ("(null)", "None", "Unknown", "N/A") else value


def parse_scheduler_output(text: str, fields: tuple[str, ...], source: str) -> list[dict[str, str]]:
    """``|``-separated SLURM output; job steps skipped, IDs kept whole, malformed lines ignored."""

    records = []
    for line in (text or "").splitlines():
        values = line.split("|")
        if len(values) != len(fields):
            continue
        record = {field: _clean(value) for field, value in zip(fields, values)}
        job_id = record["job_id"]
        if not job_id or "." in job_id or not job_id[0].isdigit():
            continue
        record["workdir"] = record["workdir"].rstrip("/")
        record["source"] = source
        records.append(record)
    return records


def _slurm_utc(value: str | None) -> datetime | None:
    """A SLURM time printed with ``TZ=UTC SLURM_TIME_FORMAT=standard``: exactly ``YYYY-MM-DDTHH:MM:SS``."""

    text = str(value or "")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", text):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _service_utc(value: str | None) -> datetime | None:
    """A service timestamp (``…Z``), timezone-aware; None unless it states its offset."""

    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


# -- timestamps (pure) -------------------------------------------------------------------------------------


def time_context(record, clock: Mapping[str, Any]) -> dict[str, Any]:
    """What the submission window and settling checks need: POWER's UTC clock and its skew from ours."""

    submission = record["submission"]
    reconciliation = submission.get("reconciliation") or {}
    bound_at = submission.get("submitted_at") or reconciliation.get("checked_at") or (
        (submission.get("remote_record_recovery") or {}).get("recovered_at"))
    return {
        "requested_at": submission.get("requested_at"),
        "bound_at": bound_at,
        "power_now": clock.get("now"),
        "skew_s": clock.get("skew_s"),
        "skew_uncertainty_s": clock.get("skew_uncertainty_s"),
        "power_offset": clock.get("offset"),
    }


def check_times(times: Mapping[str, str | None], context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Submit/start/end of one job against the attempt's submission window. Never guesses.

    Returns ``consistent``; ``contradicted`` (reliable timestamps outside the
    window: another job under a reused ID, or before this attempt); or
    ``unverified`` with a reason (a timestamp not in the expected UTC form, a
    time after POWER's own clock, out of order, no readable clock, or a skew
    too large to compare).
    """

    if context is None:
        return {"status": "unverified", "reason": "no clock context"}
    raw = {key: times.get(key) or "" for key in ("submit", "start", "end")}
    parsed = {key: _slurm_utc(value) for key, value in raw.items()}
    if not raw["submit"]:
        return {"status": "unverified", "reason": "submission time not reported"}
    unreadable = [key for key, value in raw.items() if value and parsed[key] is None]
    if unreadable:
        return {"status": "unverified", "reason": f"unreadable {', '.join(unreadable)} time: "
                                                  f"{[raw[key] for key in unreadable]}"}
    now, skew, uncertainty = context.get("power_now"), context.get("skew_s"), context.get("skew_uncertainty_s")
    requested = _service_utc(context.get("requested_at"))
    bound = _service_utc(context.get("bound_at"))
    if now is None or skew is None or uncertainty is None:
        return {"status": "unverified", "reason": "POWER's clock was not readable"}
    if abs(skew) > MAX_CLOCK_SKEW_SECONDS:
        return {"status": "unverified", "reason": f"clock skew {skew:.0f} s is too large to compare times"}
    if requested is None or bound is None:
        return {"status": "unverified", "reason": "the attempt's submission times are not recorded"}
    offset = _offset_seconds(context.get("power_offset"))
    for key in ("submit", "end"):
        ahead = (parsed[key] - now).total_seconds() if parsed[key] is not None else None
        if ahead is not None and ahead > FUTURE_SLACK_SECONDS:
            return {"status": "unverified", "reason": f"{key} time {raw[key]} is {ahead:.0f} s after POWER's UTC "
                                                      f"clock{_zone_hint(ahead, offset)}: not interpreted"}
    order = [parsed[key] for key in ("submit", "start", "end") if parsed[key] is not None]
    if order != sorted(order):
        return {"status": "unverified", "reason": "submit, start and end times are out of order"}
    submitted = parsed["submit"].timestamp() - skew  # on the service's clock
    slack = SUBMIT_WINDOW_SLACK_SECONDS + uncertainty
    earliest, latest = requested.timestamp() - slack, bound.timestamp() + slack
    if not earliest <= submitted <= latest:
        deviation = submitted - (earliest if submitted < earliest else latest)
        if abs(deviation) <= MAX_ZONE_OFFSET_SECONDS + slack:
            return {"status": "unverified",
                    "reason": f"submitted at {raw['submit']} UTC, {deviation:+.0f} s outside this attempt's submission "
                              f"window{_zone_hint(deviation, offset)}: within a time-zone offset, so not interpreted"}
        return {"status": "contradicted",
                "reason": f"submitted at {raw['submit']} UTC, outside this attempt's submission window "
                          f"({context['requested_at']} to {context['bound_at']}, ±{slack:.0f} s, skew {skew:.0f} s)"}
    return {"status": "consistent", "window": [context["requested_at"], context["bound_at"]],
            "slack_s": round(slack, 1), "skew_s": round(skew, 1)}


def _offset_seconds(text: str | None) -> int | None:
    match = re.fullmatch(r"([+-])([0-9]{2})([0-9]{2})", str(text or ""))
    if not match:
        return None
    seconds = int(match.group(2)) * 3600 + int(match.group(3)) * 60
    return seconds if match.group(1) == "+" else -seconds


def _zone_hint(deviation: float, offset: int | None) -> str:
    if offset and abs(deviation - offset) <= SUBMIT_WINDOW_SLACK_SECONDS + 5:
        return f" (matching POWER's local offset {offset / 3600:+.1f} h: printed in local time, not UTC?)"
    return ""


def job_binding(record) -> str | None:
    """How this attempt's job ID was established, or None if it was not established reliably."""

    submission = record.get("submission") or {}
    if submission.get("status") != "submitted" or not submission.get("job_id"):
        return None
    if submission.get("reconciled"):
        return "verified_reconciliation" if (submission.get("reconciliation") or {}).get(
            "outcome") == "matched" else None
    if submission.get("adopted"):
        return "power_record"  # POWER's SEARCH_SUBMITTED, written only after sbatch or a verified reconciliation
    return "sbatch" if submission.get("submitted_at") else None


# -- assessment (pure) -----------------------------------------------------------------------------------


def assess_job(records: list[Mapping[str, str]], *, identity: Mapping[str, str], job_id: str,
               times: Mapping[str, Any] | None) -> dict[str, Any]:
    """The bound job's identity and scheduler state from ``squeue`` and ``sacct`` records. Decides nothing.

    ``verified``: comment, name and directory reported and the attempt's,
    timestamps consistent. ``partial``: the same with no comment reported.
    ``unverified``: name or directory not reported, or timestamps not
    interpretable. ``contradicted``: anything that names another identity.
    """

    from backend.structure_search.lifecycle import SCHEDULER_ACTIVE, SCHEDULER_TERMINAL, normalized_scheduler_state

    ours = [record for record in records if record["job_id"] == job_id]
    problems: list[str] = []
    notes: list[str] = []
    others = reconcile_jobs([record for record in records if record["job_id"] != job_id], identity)
    other_ids = sorted(set(others["matches"]) | {job["job_id"] for job in others["unverified"] + others["conflicts"]})
    if other_ids:
        problems.append(f"other jobs carry this attempt's identity: {', '.join(other_ids)}")
    reported: dict[str, set[str]] = {field: {r[field] for r in ours if r.get(field)}
                                     for field in ("comment", "job_name", "workdir")}
    for field, values in reported.items():
        if values - {identity[field]}:  # a different non-empty value always contradicts (the comment included)
            problems.append(f"{field} reported as {sorted(values)}, expected {identity[field]!r}")
    if not reported["comment"]:
        notes.append("comment_not_reported")
    complete = bool(reported["job_name"]) and bool(reported["workdir"])
    if not complete:
        notes.append("name_or_directory_not_reported")

    times_reported = {}
    for key in ("submit", "start", "end"):
        values = sorted({r[key] for r in ours if r.get(key)})
        if len(values) > 1:
            problems.append(f"{key} times disagree: {values}")
        times_reported[key] = values[0] if len(values) == 1 else None
    time_check = check_times(times_reported, times) if ours else {"status": "unverified",
                                                                  "reason": "job not reported"}
    if time_check["status"] == "contradicted":
        problems.append(time_check["reason"])
    elif time_check["status"] == "unverified":
        notes.append("times_unverified")

    if not ours:
        status = "not_reported"
    elif problems:
        status = "contradicted"
    elif not complete or time_check["status"] != "consistent":
        status = "unverified"
    elif reported["comment"]:
        status = "verified"
    else:
        status = "partial"

    queued = [r for r in ours if r["source"] == "squeue"]
    accounted = [r for r in ours if r["source"] == "sacct"]
    queue_state = normalized_scheduler_state(queued[0]["state"]) if queued else None
    accounting_states = sorted({normalized_scheduler_state(r["state"]) for r in accounted})
    if len(accounting_states) > 1:
        problems.append(f"accounting reports several states: {accounting_states}")
        status = "contradicted"
    accounting_state = accounting_states[0] if accounting_states else None
    if queue_state in SCHEDULER_ACTIVE:
        state, source = queue_state, "squeue"
    elif accounting_state:
        state, source = accounting_state, "sacct"
    elif queue_state:
        state, source = queue_state, "squeue"
    else:
        state, source = "UNKNOWN", None
    if queue_state in SCHEDULER_TERMINAL and accounting_state and accounting_state != queue_state:
        problems.append(f"squeue reports {queue_state}, accounting {accounting_state}")
        status = "contradicted"
    terminal_reported = (accounting_state in SCHEDULER_TERMINAL
                         and (queue_state is None or queue_state == accounting_state))
    row = accounted[0] if accounted else {}
    return {
        "identity": {
            "status": status,
            "problems": problems,
            "notes": notes,
            "other_jobs": other_ids,
            "times": time_check,
        },
        "scheduler": {
            "state": state,
            "source": source,
            "queue_state": queue_state,
            "accounting_state": accounting_state,
            "terminal_reported": bool(terminal_reported),
            "reason": queued[0].get("reason") if queued else None,
            "exit_code": row.get("exit_code") or None,
            "submit": times_reported["submit"],
            "start": times_reported["start"],
            "end": times_reported["end"],
            "time_zone": "UTC",
        },
    }


def _settled(scheduler: Mapping[str, Any], power_now: datetime | None) -> dict[str, Any]:
    end = _slurm_utc(scheduler.get("end"))
    if not scheduler.get("terminal_reported"):
        return {"settled": False, "reason": "no terminal state in accounting"}
    if end is None or power_now is None:
        return {"settled": False, "reason": "end time or POWER's clock not readable"}
    elapsed = (power_now - end).total_seconds()
    if elapsed < SETTLE_SECONDS:
        return {"settled": False, "reason": "settling", "remaining_s": round(SETTLE_SECONDS - elapsed)}
    return {"settled": True, "elapsed_s": round(elapsed)}


# -- reading POWER (read-only) ---------------------------------------------------------------------------


def _read_small(runner, run_dir: str, parts: list[str], max_bytes: int) -> dict[str, Any]:
    from backend.secure_remote_read import ContainedReadError

    try:
        data = runner.read_contained_file(run_dir, parts, max_bytes=max_bytes)
    except ContainedReadError as exc:
        return {"present": False if exc.code == "missing" else None, "error": None if exc.code == "missing"
                else exc.code}
    return {"present": True, "bytes": data, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def read_clock(runner) -> dict[str, Any]:
    """POWER's UTC clock, its local offset (evidence), and the skew from the service's clock, measured."""

    before = time.time()
    result = runner.run(CLOCK_COMMAND, check=False, modules=False, export_env=False)
    after = time.time()
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()] if result.ok else []
    unreadable = {"epoch": None, "utc": None, "offset": None, "now": None, "skew_s": None,
                  "skew_uncertainty_s": None}
    if len(lines) != 3 or not lines[0].isdigit():
        return {**unreadable, "problem": "POWER's clock did not answer as expected"}
    epoch = int(lines[0])
    utc = _slurm_utc(lines[1])
    if utc is None or abs(utc.timestamp() - epoch) > 1:
        return {**unreadable, "problem": "POWER's epoch and UTC time disagree"}
    middle = (before + after) / 2
    return {"epoch": epoch, "utc": lines[1], "offset": lines[2] if re.fullmatch(r"[+-][0-9]{4}", lines[2])
            else None, "now": utc, "skew_s": epoch - middle, "skew_uncertainty_s": (after - before) / 2 + 1.0,
            "problem": None}


def _read_power(runner, *, record) -> dict[str, Any]:
    """Everything one observation needs from POWER, read-only. Raises only for a failed connection."""

    from backend.structure_search.bridge.protocol import HEARTBEAT_FILE
    from backend.structure_search.job import SPOOL_DIRNAME, STARTED_MARKER, STATUS_FILE

    spec = record["staging"]["spec"]
    run_dir = spec["paths"]["run_dir"]
    job_id = record["submission"]["job_id"]
    squeue = runner.run(SQUEUE_COMMAND, check=False, modules=False, export_env=False)
    sacct = runner.run(_sacct_command(job_id), check=False, modules=False, export_env=False)
    clock = read_clock(runner)
    state = runner._read_submission_attempt_state(spec)
    if state:
        runner._ensure_submission_attempt_matches(spec, state)
    return {
        "commands": [SQUEUE_COMMAND, _sacct_command(job_id), CLOCK_COMMAND],
        "squeue": {"answered": squeue.ok,
                   "records": parse_scheduler_output(squeue.stdout, SQUEUE_FIELDS, "squeue") if squeue.ok else []},
        "sacct": {"answered": sacct.ok,
                  "records": parse_scheduler_output(sacct.stdout, SACCT_FIELDS, "sacct") if sacct.ok else []},
        "clock": clock,
        "remote_state": state,
        "job_started": _read_small(runner, run_dir, [STARTED_MARKER], MAX_SMALL_FILE_BYTES),
        "status": _read_small(runner, run_dir, [STATUS_FILE], MAX_STATUS_BYTES),
        "heartbeat": _read_small(runner, run_dir, [SPOOL_DIRNAME, HEARTBEAT_FILE], MAX_SMALL_FILE_BYTES),
    }


# -- one observation ---------------------------------------------------------------------------------------


def _execution(status_read: Mapping[str, Any], record) -> tuple[dict[str, Any], dict | None, list[str]]:
    problems: list[str] = []
    document = None
    if status_read.get("present"):
        try:
            document = json.loads(status_read["bytes"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            document = None
    job_id = record["submission"]["job_id"]
    recorded_job = document.get("slurm_job_id") if isinstance(document, dict) else None
    if recorded_job not in (None, "", job_id):
        problems.append(f"search_status.json was written by job {recorded_job}, not {job_id}")
    execution = document.get("execution") if isinstance(document, dict) else None
    summary = {
        "source": "search_status.json",
        "present": status_read.get("present"),
        "read_error": status_read.get("error"),
        "sha256": status_read.get("sha256"),
        "size": status_read.get("size"),
        "attempt_id": document.get("attempt_id") if isinstance(document, dict) else None,
        "slurm_job_id": recorded_job,
        "outcome": execution.get("outcome") if isinstance(execution, dict) else None,
        "master_exit_status": execution.get("master_exit_status") if isinstance(execution, dict) else None,
        "finished_at": document.get("finished_at") if isinstance(document, dict) else None,
    }
    return summary, document, problems


def build_observation(power: Mapping[str, Any], *, record, principal: str) -> dict[str, Any]:
    """The evidence of one observation and the state it supports (pure, apart from the clock)."""

    from backend.structure_search.lifecycle import classify_job

    attempt_id = record["attempt_id"]
    submission = record["submission"]
    job_id = submission["job_id"]
    identity_expected = job_identity(attempt_id, record["staging"]["spec"]["paths"]["run_dir"])
    clock = power["clock"]
    records = power["squeue"]["records"] + power["sacct"]["records"]
    assessed = assess_job(records, identity=identity_expected, job_id=job_id, times=time_context(record, clock))
    identity, scheduler = dict(assessed["identity"]), assessed["scheduler"]
    problems = list(identity["problems"])
    binding = job_binding(record)
    answered = power["squeue"]["answered"] and power["sacct"]["answered"]
    if not answered:
        identity["notes"] = identity["notes"] + ["scheduler_did_not_answer"]
    settled = _settled(scheduler, clock["now"])

    started = power["job_started"]
    started_value = started["bytes"].decode("utf-8", "replace").strip() if started.get("present") else None
    if started_value is not None and started_value != job_id:
        problems.append(f"job_started names job {started_value!r}, not {job_id}")
    execution, document, status_problems = _execution(power["status"], record)
    problems += status_problems
    heartbeat_age = None
    if power["heartbeat"].get("present") and clock["epoch"] is not None:
        text = power["heartbeat"]["bytes"].decode("utf-8", "replace").strip()
        heartbeat_age = clock["epoch"] - int(text) if text.isdigit() else None

    remote = power["remote_state"] or {}
    remote_output = _remote_record(remote.get("output")) or {}
    remote_submission = remote_output.get("submission") or {}
    remote_state = remote.get("state")
    if remote_state not in (REMOTE_SEARCH_SUBMITTING, REMOTE_SEARCH_SUBMITTED, REMOTE_SEARCH_CANCEL_REQUESTED):
        problems.append(f"POWER holds {remote_state} for a submitted attempt")
    elif remote_submission and not _identity_matches(remote_submission, record):
        problems.append("POWER's submission record names another job identity")
    elif remote_submission.get("job_id") not in (None, job_id):
        problems.append(f"POWER's record names job {remote_submission.get('job_id')}, not {job_id}")
    if problems:
        identity["status"] = "contradicted"

    derived = None
    if problems:
        blocked = "contradiction"
    elif binding is None:
        blocked = "job_binding_not_established"
    elif identity["status"] == "not_reported":
        blocked = "job_not_reported"
    elif identity["status"] not in ("verified", "partial") or not answered:
        blocked = "identity_not_established"
    elif not settled["settled"]:
        blocked = "not_terminal" if not scheduler["terminal_reported"] else "terminal_settling"
    else:
        blocked = None
        derived = classify_job(scheduler["accounting_state"], document, attempt_id=attempt_id)["state"]
    return {
        "schema": OBSERVATION_SCHEMA,
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "job_id": job_id,
        "observed_at": isoformat(utc_now()),
        "observed_by": principal,
        "commands": power["commands"],
        "power_clock": {key: clock.get(key) for key in ("epoch", "utc", "offset", "skew_s", "skew_uncertainty_s",
                                                        "problem")},
        "scheduler": {**scheduler, **settled, "squeue_answered": power["squeue"]["answered"],
                      "sacct_answered": power["sacct"]["answered"], "records": records},
        "identity": {**identity, "binding": binding, "expected": identity_expected, "problems": problems},
        "execution": {**execution, "job_started": started_value, "job_started_present": started.get("present"),
                      "heartbeat_age_s": heartbeat_age},
        "remote": {"state": remote_state, "job_id": remote_submission.get("job_id"),
                   "cancellation": remote_output.get("cancellation")},
        "scientific_validity": {"status": "not_assessed"},
        "finalists": {"status": "not_started"},
        "classification": {"classifier_version": CLASSIFIER_VERSION, "derived_state": derived,
                           "blocked": blocked},
    }


# -- evidence (write-once files) ---------------------------------------------------------------------------


def _observations_dir(ledger, attempt_id: str) -> Path:
    return ledger.root / OBSERVATIONS_DIRNAME / attempt_id


def _evidence_files(ledger, attempt_id: str) -> list[Path]:
    directory = _observations_dir(ledger, attempt_id)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if re.fullmatch(r"[0-9]{6}\.json", path.name))


def _write_evidence(ledger, attempt_id: str, observation: dict) -> tuple[int, str]:
    """Write the next evidence file exclusively (never replacing one). Call while holding the ledger lock."""

    _private_directory(ledger.root / OBSERVATIONS_DIRNAME)
    directory = _observations_dir(ledger, attempt_id)
    _private_directory(directory)
    files = _evidence_files(ledger, attempt_id)
    sequence = int(files[-1].stem) + 1 if files else 1
    previous = hashlib.sha256(files[-1].read_bytes()).hexdigest() if files else None
    document = {**observation, "sequence": sequence, "previous_sha256": previous}
    data = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    final = directory / f"{sequence:06d}.json"
    temporary = directory / f".{sequence:06d}.{os.getpid()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, final)  # fails if the name exists: evidence is never overwritten
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return sequence, hashlib.sha256(data).hexdigest()


# -- applying evidence to the record (idempotent) ---------------------------------------------------------


def _summary(observation: Mapping[str, Any], sequence: int, sha256: str) -> dict[str, Any]:
    scheduler = observation["scheduler"]
    execution = observation["execution"]
    return {
        "sequence": sequence,
        "sha256": sha256,
        "observed_at": observation["observed_at"],
        "observed_by": observation["observed_by"],
        "power_clock": observation["power_clock"],
        "scheduler": {key: scheduler.get(key) for key in (
            "state", "source", "queue_state", "accounting_state", "terminal_reported", "settled", "reason",
            "remaining_s", "exit_code", "submit", "start", "end", "time_zone")},
        "identity": {key: observation["identity"].get(key) for key in ("status", "binding", "problems", "notes",
                                                                      "other_jobs", "times")},
        "execution": {key: execution.get(key) for key in (
            "source", "present", "read_error", "sha256", "outcome", "master_exit_status", "finished_at",
            "job_started", "heartbeat_age_s")},
        "remote_state": observation["remote"]["state"],
        "classification": observation["classification"],
    }


def _apply(record: dict, observation: Mapping[str, Any], sequence: int, sha256: str) -> None:
    monitoring = dict(record.get("monitoring") or {"observation_count": 0, "terminal": None,
                                                   "contradictions": [], "contradiction_count": 0})
    if sequence <= int(monitoring.get("last_sequence") or 0):
        return  # already applied
    summary = _summary(observation, sequence, sha256)
    monitoring.update(observation_count=int(monitoring.get("observation_count") or 0) + 1,
                      last_sequence=sequence, last=summary)
    derived = observation["classification"]["derived_state"]
    contradictions = list(monitoring.get("contradictions") or [])

    def contradiction(kind: str, detail: Any) -> None:
        monitoring["contradiction_count"] = int(monitoring.get("contradiction_count") or 0) + 1
        contradictions.append({"sequence": sequence, "observed_at": observation["observed_at"], "kind": kind,
                               "detail": detail})

    if observation["identity"]["problems"]:
        contradiction("identity", observation["identity"]["problems"])
    remote_cancel = observation["remote"]["state"] == REMOTE_SEARCH_CANCEL_REQUESTED
    if remote_cancel and record["state"] == STATE_SUBMITTED and not observation["identity"]["problems"]:
        # POWER records a cancellation request the local record does not: adopt the request (never a command)
        _transition(record, STATE_CANCEL_REQUESTED)
        remote = observation["remote"].get("cancellation") or {}
        _start_request(record, requested_at=remote.get("requested_at"), requested_by=remote.get("requested_by"),
                       job_id=record["submission"]["job_id"])
        record["cancellation"].update(status="remote_recorded_command_pending", remote_recorded=True,
                                      adopted_from_power=True)
    if derived is not None:
        if record["state"] in _terminal_states():
            if derived != record["state"]:
                contradiction("terminal_state", {"recorded": record["state"], "observed": derived})
        else:
            scheduler = observation["scheduler"]
            cancelling = record["state"] == STATE_CANCEL_REQUESTED
            _transition(record, derived)
            monitoring["terminal"] = {
                "state": derived, "recorded_at": isoformat(utc_now()), "observation_sequence": sequence,
                "scheduler_state": scheduler.get("accounting_state"), "exit_code": scheduler.get("exit_code"),
                "end": scheduler.get("end"), "execution_outcome": observation["execution"].get("outcome"),
                "master_exit_status": observation["execution"].get("master_exit_status"),
                "identity": observation["identity"]["status"], "binding": observation["identity"].get("binding"),
            }
            if cancelling:
                ended = scheduler.get("accounting_state")
                record["cancellation"] = {
                    **record["cancellation"], "status": "observed",
                    "observed": {"observed_at": observation["observed_at"], "observation_sequence": sequence,
                                 "scheduler_state": ended,
                                 "outcome": "cancelled" if ended == "CANCELLED"
                                 else "job_ended_before_cancellation_took_effect"},
                }
    monitoring["contradictions"] = contradictions[-MAX_KEPT:]
    record["monitoring"] = monitoring
    record["updated_at"] = isoformat(utc_now())


def _catch_up(ledger, attempt_id: str) -> dict | None:
    """Apply evidence files the record has not seen (an interrupted update). Holds the ledger lock."""

    with ledger.locked():
        record = ledger.read(attempt_id)
        last = int((record.get("monitoring") or {}).get("last_sequence") or 0)
        pending = [path for path in _evidence_files(ledger, attempt_id) if int(path.stem) > last]
        if not pending:
            return record
        for path in pending:
            data = path.read_bytes()
            _apply(record, json.loads(data.decode("utf-8")), int(path.stem), hashlib.sha256(data).hexdigest())
        ledger.replace(record)
        LOGGER.info("structure-search %s: applied %d stored observation(s)", attempt_id, len(pending))
        return record


# -- requests ----------------------------------------------------------------------------------------------


def _load(attempt_id: str, principal: str):
    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No structure-search attempt with this ID exists.")
    _authorize_record(record, principal)
    return ledger, record


def _response(record, *, observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    response = search_attempt_response(record)
    if observation is not None:
        response["observation"] = dict(observation)
    return response


def observe_search_attempt(*, principal: str, attempt_id: str) -> dict[str, Any]:
    """Observe a submitted search once on POWER (read-only) and record the evidence."""

    from backend.remote_runtime import connected_remote_runner, connection_profile_from_cluster
    from compute_api import execution

    ledger, record = _load(attempt_id, principal)
    if record["state"] == STATE_SUBMISSION_UNCERTAIN:
        raise ExecutionError(409, "submission_uncertain",
                             "This attempt's submission is uncertain: there is no bound job to observe. Repeat the "
                             "submission request to reconcile it.")
    if record["state"] not in (STATE_SUBMITTED, STATE_CANCEL_REQUESTED, *_terminal_states()):
        raise ExecutionError(409, "attempt_not_submitted", "Only a submitted attempt can be observed.")
    if not (record.get("submission") or {}).get("job_id"):
        raise ExecutionError(409, "attempt_not_submitted", "This attempt has no job ID.")
    record = _catch_up(ledger, attempt_id)
    last = (record.get("monitoring") or {}).get("last")
    if last:
        age = (utc_now() - datetime.fromisoformat(last["observed_at"].replace("Z", "+00:00"))).total_seconds()
        if age < MIN_OBSERVATION_INTERVAL_SECONDS:
            return _response(record, observation={"performed": False, "reason": "rate_limited",
                                                  "retry_after_s": round(MIN_OBSERVATION_INTERVAL_SECONDS - age),
                                                  "sequence": last["sequence"]})

    spec = record["staging"]["spec"]
    with connected_remote_runner(profile=connection_profile_from_cluster(spec["cluster"]),
                                 runner_factory=execution.RUNNER_FACTORY) as runner:
        power = _read_power(runner, record=record)
    observation = build_observation(power, record=record, principal=principal)
    with ledger.locked():
        current = ledger.read(attempt_id)
        sequence, sha256 = _write_evidence(ledger, attempt_id, {**observation, "local_state": current["state"]})
    record = _catch_up(ledger, attempt_id)  # applies exactly the new evidence (and anything missed)
    LOGGER.info("structure-search %s observed (#%d): scheduler %s, identity %s, state %s", attempt_id, sequence,
                observation["scheduler"]["state"], observation["identity"]["status"], record["state"])
    return _response(record, observation={"performed": True, "sequence": sequence, "sha256": sha256})


# -- cancellation ------------------------------------------------------------------------------------------


def _start_request(record: dict, *, requested_at, requested_by, job_id) -> None:
    previous = record.get("cancellation")
    earlier = list((previous or {}).get("earlier_requests") or [])
    if previous:
        earlier.append({key: previous.get(key) for key in ("requested_at", "requested_by", "status",
                                                          "last_refusal", "command_count")})
    record["cancellation"] = {"status": "requested", "requested_at": requested_at, "requested_by": requested_by,
                              "job_id": job_id, "remote_recorded": False, "commands": [], "command_count": 0,
                              "observed": None, "last_refusal": None, "earlier_requests": earlier[-MAX_KEPT:]}


def _scancel_outcome(result) -> str:
    text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    if result.ok:
        return "sent"
    if "already completing or completed" in text or "invalid job id" in text:
        return "job_already_ended"
    return "failed"


def _cancel_remote(*, record, progress: dict, on_remote_recorded) -> dict[str, Any]:
    """Verify the job afresh on POWER under the claim, record the request there, then scancel by job ID.

    ``progress`` tells the caller how far this got if it stops: ``remote_state``
    (as read), ``remote_write_started``, ``remote_recorded``.
    """

    from backend.structure_search.lifecycle import SCHEDULER_ACTIVE

    spec = record["staging"]["spec"]
    job_id = record["submission"]["job_id"]
    cancellation = record["cancellation"]
    with _claimed_runner(spec, busy_code="cancellation_in_progress") as runner:
        state = runner._read_submission_attempt_state(spec)
        if not state:
            raise _staging_failed("remote_state_missing", "POWER holds no state for this attempt.")
        runner._ensure_submission_attempt_matches(spec, state)
        current = state.get("state")
        progress["remote_state"] = current
        output = _remote_record(state.get("output")) or {}
        if current not in (REMOTE_SEARCH_SUBMITTED, REMOTE_SEARCH_CANCEL_REQUESTED):
            raise _staging_failed("transition_not_allowed",
                                  f"Cancellation needs POWER to record the submitted job; it holds {current}.")
        if not _promotion_matches(output.get("promotion"), record["promotion"]):
            raise _staging_failed("promotion_mismatch", "POWER records another promotion of this attempt.")
        remote_submission = output.get("submission") or {}
        if not _identity_matches(remote_submission, record) or remote_submission.get("job_id") != job_id:
            raise ExecutionError(409, "submission_conflict",
                                 "POWER's submission record does not name this job and identity; nothing was sent.")
        remote_cancel = output.get("cancellation")
        if remote_cancel and remote_cancel.get("job_id") != job_id:
            raise ExecutionError(409, "submission_conflict", "POWER records a cancellation of another job.")

        # A fresh identity check, every time, before any scancel.
        squeue = runner.run(SQUEUE_COMMAND, check=False, modules=False, export_env=False)
        sacct = runner.run(_sacct_command(job_id), check=False, modules=False, export_env=False)
        clock = read_clock(runner)
        if not squeue.ok:
            raise ExecutionError(503, "scheduler_unavailable", "squeue did not answer; nothing was sent. Retry.")
        records = parse_scheduler_output(squeue.stdout, SQUEUE_FIELDS, "squeue") + (
            parse_scheduler_output(sacct.stdout, SACCT_FIELDS, "sacct") if sacct.ok else [])
        identity = job_identity(record["attempt_id"], spec["paths"]["run_dir"])
        assessed = assess_job(records, identity=identity, job_id=job_id, times=time_context(record, clock))
        found, scheduler = assessed["identity"], assessed["scheduler"]
        evidence = {"identity": found["status"], "problems": found["problems"], "notes": found["notes"],
                    "times": found["times"], "queue_state": scheduler["queue_state"],
                    "accounting_state": scheduler["accounting_state"], "checked_at": isoformat(utc_now()),
                    "power_clock": {key: clock.get(key) for key in ("utc", "offset", "skew_s", "problem")}}
        if found["status"] == "contradicted":
            raise ExecutionError(409, "submission_conflict",
                                 "SLURM's records contradict this attempt's job; nothing was sent.", evidence=evidence)
        if scheduler["queue_state"] not in SCHEDULER_ACTIVE:
            if scheduler["queue_state"] or scheduler["accounting_state"]:
                return {"command": None, "outcome": "not_sent_job_not_active", "evidence": evidence}
            raise ExecutionError(409, "job_not_visible",
                                 f"SLURM does not report job {job_id}; nothing was sent. Observe or retry later.",
                                 evidence=evidence)
        if found["status"] != "verified":  # comment, name, directory and times: partial is never enough here
            raise ExecutionError(409, "job_identity_unverified",
                                 f"SLURM reports job {job_id} without a fully verified identity; nothing was sent.",
                                 evidence=evidence)

        if current == REMOTE_SEARCH_SUBMITTED:
            progress["remote_write_started"] = True
            _write_search_state(runner, spec, REMOTE_SEARCH_CANCEL_REQUESTED, output=json.dumps(
                {**output, "cancellation": {"job_id": job_id, "requested_at": cancellation["requested_at"],
                                            "requested_by": cancellation["requested_by"],
                                            "recorded_at": isoformat(utc_now())}}, sort_keys=True))
        progress["remote_recorded"] = True
        on_remote_recorded()  # the local record knows before scancel; if it cannot, nothing is sent
        command = _scancel_command(job_id)
        attempted_at = isoformat(utc_now())
        try:
            result = runner.run(command, check=False, modules=False, export_env=False)
        except Exception as exc:  # noqa: BLE001 - scancel is safe to repeat; the observation decides
            return {"command": command, "attempted_at": attempted_at, "outcome": "outcome_unknown",
                    "error": type(exc).__name__, "evidence": evidence}
        return {"command": command, "attempted_at": attempted_at, "outcome": _scancel_outcome(result),
                "returncode": result.returncode, "stderr": (result.stderr or "").strip()[-300:],
                "evidence": evidence}


_COMMAND_STATUS = {"sent": "command_sent", "job_already_ended": "command_found_job_ended",
                   "failed": "command_failed", "outcome_unknown": "command_outcome_unknown",
                   "not_sent_job_not_active": "not_sent_job_not_active"}


def cancel_search_attempt(*, principal: str, attempt_id: str, shape) -> dict[str, Any]:
    """Request cancellation of a submitted search's job (recorded first), then scancel it by job ID."""

    ledger, record = _load(attempt_id, principal)
    if record["plan_digest"] != shape.expected_plan_digest:
        raise ExecutionError(409, "plan_digest_mismatch", "This attempt was prepared for a different plan digest.")
    submission = record.get("submission") or {}
    if record["state"] in _terminal_states():
        raise ExecutionError(409, "attempt_terminal", f"This search has already ended ({record['state']}).",
                             state=record["state"])
    if record["state"] == STATE_SUBMISSION_UNCERTAIN:
        raise ExecutionError(409, "submission_uncertain",
                             "This attempt's submission is uncertain: there is no verified job to cancel.")
    if record["state"] not in (STATE_SUBMITTED, STATE_CANCEL_REQUESTED):
        raise ExecutionError(409, "attempt_not_submitted", "Only a submitted search can be cancelled.")
    if shape.job_id != submission.get("job_id"):
        raise ExecutionError(409, "job_id_mismatch", "This is not the job ID recorded for this attempt.",
                             job_id=submission.get("job_id"))
    if submission.get("remote_record_pending"):
        raise ExecutionError(409, "remote_record_pending",
                             "POWER's record of this submission is incomplete; complete it first "
                             "(PUT …/submission/remote-record).")
    if job_binding(record) is None:
        raise ExecutionError(409, "job_binding_not_established", "This attempt's job ID is not established.")

    with ledger.locked():  # 1. requested (local, before POWER)
        record = ledger.read(attempt_id)
        if record["state"] == STATE_SUBMITTED:
            _transition(record, STATE_CANCEL_REQUESTED)
            _start_request(record, requested_at=isoformat(utc_now()), requested_by=principal, job_id=shape.job_id)
            ledger.replace(record)
        elif record["state"] != STATE_CANCEL_REQUESTED:
            raise ExecutionError(409, "attempt_terminal", f"This search has already ended ({record['state']}).",
                                 state=record["state"])

    def remote_recorded() -> None:
        with ledger.locked():
            current = ledger.read(attempt_id)
            current["cancellation"] = {**current["cancellation"], "remote_recorded": True,
                                       "status": "remote_recorded_command_pending"}
            ledger.replace(current)

    progress = {"remote_state": None, "remote_write_started": False, "remote_recorded": False}
    try:  # 2. on POWER, after a fresh verification
        result = _cancel_remote(record=record, progress=progress, on_remote_recorded=remote_recorded)
    except Exception as exc:
        _after_failure(ledger, attempt_id, progress, exc)
        raise

    with ledger.locked():
        record = ledger.read(attempt_id)
        cancellation = dict(record["cancellation"])
        if result["command"] is not None:
            cancellation["command_count"] = int(cancellation.get("command_count") or 0) + 1
            cancellation["commands"] = (list(cancellation.get("commands") or []) + [
                {key: result.get(key) for key in ("command", "attempted_at", "outcome", "returncode", "stderr",
                                                  "error")}])[-MAX_KEPT:]
            cancellation["remote_recorded"] = True
        if cancellation.get("status") != "observed":
            cancellation["status"] = _COMMAND_STATUS[result["outcome"]]
        cancellation["last_check"] = result["evidence"]
        cancellation["last_refusal"] = None
        record["cancellation"] = cancellation
        ledger.replace(record)
    LOGGER.info("structure-search %s: cancellation of job %s: %s", attempt_id, shape.job_id, result["outcome"])
    return _response(record)


def _after_failure(ledger, attempt_id: str, progress: Mapping[str, Any], exc: Exception) -> None:
    """Record where a cancellation request stopped; return to SUBMITTED only if no remote request can exist."""

    code = getattr(exc, "code", None) or type(exc).__name__
    try:
        with ledger.locked():
            current = ledger.read(attempt_id)
            cancellation = dict(current.get("cancellation") or {})
            if current["state"] != STATE_CANCEL_REQUESTED or not cancellation:
                return
            commands = int(cancellation.get("command_count") or 0)
            remote_known = bool(cancellation.get("remote_recorded") or progress["remote_recorded"]
                                or progress["remote_state"] == REMOTE_SEARCH_CANCEL_REQUESTED)
            refusal = {"at": isoformat(utc_now()), "code": code,
                       "stage": "remote_write" if progress["remote_write_started"] and not progress["remote_recorded"]
                       else "after_remote_request" if remote_known else "before_remote_request"}
            if progress["remote_write_started"] and not progress["remote_recorded"]:
                cancellation["status"] = "remote_request_uncertain"  # the atomic write may or may not have landed
            elif remote_known and not commands:
                cancellation.update(status="remote_recorded_command_pending", remote_recorded=True)
            elif commands:
                pass  # an earlier scancel was attempted: its status stands
            else:
                # Refused or failed before any remote cancellation request: nothing exists on POWER to resume.
                # (If an earlier request's remote record was lost locally, the next observation adopts it.)
                cancellation["status"] = "refused_before_remote_request"
                current["cancellation"] = {**cancellation, "last_refusal": refusal}
                _transition(current, STATE_SUBMITTED)
                ledger.replace(current)
                return
            current["cancellation"] = {**cancellation, "last_refusal": refusal}
            ledger.replace(current)
    except Exception:  # noqa: BLE001 - the request's own error is what the caller reports
        LOGGER.error("structure-search %s: could not record where cancellation stopped", attempt_id, exc_info=True)


__all__ = [
    "REMOTE_SEARCH_CANCEL_REQUESTED",
    "STATE_CANCEL_REQUESTED",
    "assess_job",
    "build_observation",
    "cancel_search_attempt",
    "check_times",
    "job_binding",
    "observe_search_attempt",
    "parse_scheduler_output",
    "read_clock",
]
