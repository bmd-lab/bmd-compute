"""Promotion of a STAGED structure search to submission-ready (USPEX step 4D-2).

``PUT /api/v1/structure-search-attempts/{id}/promotion`` (scope ``submit``,
the attempt's owner) takes a ``STAGED`` attempt to ``PREPARED`` locally and
``SEARCH_PREPARED`` on POWER. It never runs ``sbatch``, ``srun``, USPEX or
VASP: submission is step 4D-3.

Sequence:

1. Local checks, without contacting POWER: the attempt, its owner, the plan
   digest, the intact local preparation, the plan still current, today's
   runtime package equal to the staged one, and the client's
   ``resources_sha256`` equal to the plan's provisional resource request
   (``job.resource_request_sha256``).
2. On the first request, under the ledger lock, the promotion inputs are
   frozen in the record: the resource acknowledgement (principal from the
   token, server time), the VASP module snapshot, and the job script's
   SHA-256. The record stays ``STAGED`` until POWER confirms. Every retry
   rebuilds the same script from the frozen inputs and refuses if it differs.
3. On POWER, under the attempt's claim and the shared operation limiter, with
   the remote fingerprint checked:

   * ``STAGED``: the run directory is reconciled as in staging, with the job
     script and ``resource_acknowledgement.json`` added to the expected files
     (an unfinished upload of this attempt is completed; anything else,
     including a differing script, is refused and kept); the two files are
     uploaded only where absent; then a fresh read-only verification (runtime
     package guard, frozen plan, bound bridge, POTCAR identities, staged
     inputs, the job script's SHA-256 and ``bash -n``, the USPEX runtime
     readiness check) and a complete inventory. Only then is the remote state
     replaced atomically with ``SEARCH_PREPARED`` and its promotion record.
   * ``SEARCH_PREPARED`` with this attempt's frozen script and acknowledgement
     (an earlier request finished on POWER): adopted after the same read-only
     verification. Any other identities are refused.
   * anything else is refused.

4. The local record becomes ``PREPARED``. A repeated request returns it.

``PREPARED`` records what POWER last verified, not current integrity; reads
never contact POWER, and submission (4D-3) verifies again under its own claim.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from compute_api.execution import ExecutionError, _authorize_record
from compute_api.ledger import isoformat, utc_now
from compute_api.structure_search_execution import (
    LOGGER,
    STATE_PREPARED,
    STATE_STAGED,
    SearchAttemptLedger,
    _calculation_id_conflict,
    _claimed_runner,
    _PhaseTimer,
    _RemoteStaging,
    _staging_failed,
    _staging_inputs,
    _stored,
    _write_search_state,
    search_attempt_response,
    search_resources,
)

ACKNOWLEDGEMENT_FILE = "resource_acknowledgement.json"
PROMOTION_SCHEMA = "bmd_compute.structure_search_promotion"
PROMOTION_SCHEMA_VERSION = 1


def acknowledgement_text(acknowledgement: Mapping[str, Any]) -> str:
    return json.dumps(dict(acknowledgement), indent=2, sort_keys=True) + "\n"


def _job_script(record, plan, sources, manifest) -> str:
    from backend.structure_search.job import build_job_script

    promotion = record["promotion"]
    return build_job_script(spec=record["staging"]["spec"], plan=plan, acknowledgement=promotion["acknowledgement"],
                            runtime_sources=sources, manifest=manifest, modules=promotion["modules"])


def _remote_record(output: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(output) if isinstance(output, str) and output else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _promote_remote(*, record, plan, local_manifest, sources, manifest, script: str) -> dict[str, Any]:
    """Promote one staged attempt on POWER. Never runs sbatch, srun, USPEX or VASP."""

    from backend.structure_search import lifecycle
    from backend.structure_search.job import JOB_SCRIPT_NAME, JOB_SCRIPT_SCHEMA, JOB_SCRIPT_SCHEMA_VERSION
    from backend.runtime_package import runtime_package_manifest_digest

    promotion = record["promotion"]
    spec = record["staging"]["spec"]
    acknowledgement = promotion["acknowledgement"]
    remote = _RemoteStaging(
        spec=spec, plan=plan, local_manifest=local_manifest, sources=sources, manifest=manifest,
        extra_files={JOB_SCRIPT_NAME: script, ACKNOWLEDGEMENT_FILE: acknowledgement_text(acknowledgement)},
    )
    options = {"job_script": {"name": JOB_SCRIPT_NAME, "sha256": promotion["job_script"]["sha256"]},
               "runtime_check": True}
    phase = _PhaseTimer(remote.attempt_id)
    outcome = "failed"

    def summary(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": PROMOTION_SCHEMA,
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "plan_digest": plan.digest,
            "job_script": {"name": JOB_SCRIPT_NAME, "sha256": promotion["job_script"]["sha256"],
                           "schema": JOB_SCRIPT_SCHEMA, "schema_version": JOB_SCRIPT_SCHEMA_VERSION},
            "acknowledgement_sha256": acknowledgement["acknowledgement_sha256"],
            "resources_sha256": promotion["resources_sha256"],
            "modules": list(promotion["modules"]),
            "runtime_package_manifest_sha256": runtime_package_manifest_digest(manifest),
            "verified_at": isoformat(utc_now()),
            "runtime_check": result.get("runtime_check"),
            "files": result["files"],
            "potcars": result["potcars"],
        }

    try:
        with _claimed_runner(spec, busy_code="promotion_in_progress") as runner:
            phase.phases["connect_and_claim"] = round(time.monotonic() - phase.started, 3)
            state = runner._read_submission_attempt_state(spec)
            if not state:
                raise _staging_failed("remote_state_missing", "POWER holds no state for this attempt.")
            runner._ensure_submission_attempt_matches(spec, state)
            current = state.get("state")
            if current == lifecycle.REMOTE_SEARCH_PREPARED:
                existing = _remote_record(state.get("output")) or {}
                if ((existing.get("job_script") or {}).get("sha256") != promotion["job_script"]["sha256"]
                        or existing.get("acknowledgement_sha256") != acknowledgement["acknowledgement_sha256"]):
                    raise _staging_failed("promotion_mismatch",
                                          "POWER holds a different promotion of this attempt; nothing was changed.")
                with phase("reverify"):
                    result = remote.verify(runner, options=options)  # adopted only after verification
                outcome = "adopted after re-verification"
                return {**summary(result), "remote_state": lifecycle.REMOTE_SEARCH_PREPARED}
            if current != lifecycle.REMOTE_STAGED:  # promotion starts only from STAGED
                raise lifecycle.LifecycleError(
                    "transition_not_allowed", f"Promotion starts from STAGED; POWER holds {current or 'no state'}.")
            lifecycle.require_remote_transition(current, lifecycle.REMOTE_SEARCH_PREPARED)

            with phase("inventory"):
                uploads = remote.reconcile(runner)
            with phase(f"uploads ({len(uploads['missing'])} file(s))"):
                remote.upload_missing(runner, uploads)
            with phase("fresh_verification"):
                result = remote.verify(runner, options=options)
            record_out = summary(result)
            _write_search_state(runner, spec, lifecycle.REMOTE_SEARCH_PREPARED,
                                output=json.dumps(record_out, sort_keys=True))
            outcome = "promoted"
            return {**record_out, "remote_state": lifecycle.REMOTE_SEARCH_PREPARED}
    finally:
        phase.summary(outcome)


def promote_search_attempt(*, principal: str, attempt_id: str, shape) -> dict[str, Any]:
    """Promote a STAGED search attempt to PREPARED (idempotent; never submits)."""

    from backend.config import MODULES
    from backend.structure_search import lifecycle
    from backend.structure_search.job import (
        JOB_SCRIPT_NAME,
        JOB_SCRIPT_SCHEMA,
        JOB_SCRIPT_SCHEMA_VERSION,
        JobDefinitionError,
        build_resource_acknowledgement,
        job_script_sha256,
        resource_request_sha256,
    )

    ledger = SearchAttemptLedger.from_environment_read_only()
    if ledger.calculation_exists(attempt_id):
        raise _calculation_id_conflict()
    record = ledger.read(attempt_id)
    if record is None:
        raise ExecutionError(404, "attempt_not_found", "No structure-search attempt with this ID exists.")
    _authorize_record(record, principal)
    if record["plan_digest"] != shape.expected_plan_digest:
        raise ExecutionError(409, "plan_digest_mismatch", "This attempt was prepared for a different plan digest.")
    if record["state"] not in (STATE_STAGED, STATE_PREPARED):
        raise ExecutionError(409, "attempt_not_staged", "Only a STAGED attempt can be promoted.")
    stored = _stored(ledger, record)  # the local preparation must still be intact
    frozen = record.get("promotion")
    if frozen and frozen["resources_sha256"] != shape.resources_sha256:
        raise ExecutionError(409, "acknowledgement_mismatch",
                             "This attempt's promotion was acknowledged for a different resource request.")
    if record["state"] == STATE_PREPARED:
        return stored

    plan, local_manifest, sources, manifest, manifest_digest = _staging_inputs(ledger, record)
    if manifest_digest != record["staging"]["runtime_manifest_digest"]:
        raise ExecutionError(409, "runtime_package_changed",
                             "BMD Compute's runtime package changed since this attempt was staged. "
                             "Use a new attempt ID.")
    resources_sha256 = resource_request_sha256(plan)
    if shape.resources_sha256 != resources_sha256:
        raise ExecutionError(409, "resources_changed",
                             "The acknowledged resource request is not this attempt's; read the attempt again.",
                             resources_sha256=resources_sha256)

    with ledger.locked():
        record = ledger.read(attempt_id)
        _authorize_record(record, principal)
        if record["state"] == STATE_PREPARED:
            return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))
        if record["state"] != STATE_STAGED:
            raise ExecutionError(409, "attempt_not_staged", "Only a STAGED attempt can be promoted.")
        if not record.get("promotion"):
            started = isoformat(utc_now())
            try:
                acknowledgement = build_resource_acknowledgement(plan, attempt_id=attempt_id, principal=principal,
                                                                 acknowledged_at=started)
            except JobDefinitionError as exc:
                raise ExecutionError(409, exc.code, exc.message) from None
            record["promotion"] = {
                "acknowledgement": acknowledgement,
                "resources_sha256": resources_sha256,
                "modules": list(MODULES),
                "job_script": {"name": JOB_SCRIPT_NAME, "schema": JOB_SCRIPT_SCHEMA,
                               "schema_version": JOB_SCRIPT_SCHEMA_VERSION, "sha256": None},
                "started_at": started,
                "verified_at": None,
                "result": None,
            }
            try:
                record["promotion"]["job_script"]["sha256"] = job_script_sha256(
                    _job_script(record, plan, sources, manifest))
            except JobDefinitionError as exc:
                raise ExecutionError(409, exc.code, exc.message) from None
            record["updated_at"] = started
            ledger.replace(record)
        elif record["promotion"]["resources_sha256"] != shape.resources_sha256:
            raise ExecutionError(409, "acknowledgement_mismatch",
                                 "This attempt's promotion was acknowledged for a different resource request.")

    try:
        script = _job_script(record, plan, sources, manifest)
    except JobDefinitionError as exc:
        raise ExecutionError(409, exc.code, exc.message) from None
    if job_script_sha256(script) != record["promotion"]["job_script"]["sha256"]:
        raise ExecutionError(409, "job_script_changed",
                             "The job script this attempt froze can no longer be reproduced; use a new attempt ID.")

    result = _promote_remote(record=record, plan=plan, local_manifest=local_manifest, sources=sources,
                             manifest=manifest, script=script)

    with ledger.locked():
        record = ledger.read(attempt_id)
        if record["state"] != STATE_PREPARED:
            lifecycle.require_local_transition(record["state"], STATE_PREPARED)
            record["state"] = STATE_PREPARED
            record["promotion"] = {
                **record["promotion"],
                "verified_at": result["verified_at"],
                "result": {
                    "remote_state": result["remote_state"],
                    "runtime_check": result.get("runtime_check"),
                    "runtime_package_manifest_sha256": result["runtime_package_manifest_sha256"],
                },
            }
            record["updated_at"] = isoformat(utc_now())
            ledger.replace(record)
    LOGGER.info("structure-search %s promoted to PREPARED (job script %s)", attempt_id,
                record["promotion"]["job_script"]["sha256"][:12])
    return search_attempt_response(record, integrity="verified", resources=search_resources(ledger, record))


__all__ = ["ACKNOWLEDGEMENT_FILE", "acknowledgement_text", "promote_search_attempt"]
