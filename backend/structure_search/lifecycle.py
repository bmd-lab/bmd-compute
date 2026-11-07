"""Life cycle of a managed structure-search attempt (USPEX step 4D).

Two records describe an attempt, and each has its own states:

* the **remote attempt state** on POWER
  (``<search logs>/submission_attempts/<attempt id>.json``), which is what
  exactly-once submission relies on, and
* the **local record** in BMD Compute's search ledger, which is what the API
  reports.

Remote states are deliberately disjoint from the calculation states
(``PREPARED``, ``SUBMITTING``, ``SUBMITTED``): the ordinary calculation claim
and ``sbatch`` path accepts only ``PREPARED`` and refuses to re-prepare any
other state, so a structure search can never enter it.

Once a job exists, three separate questions are answered separately, never
inferred from one another:

* **scheduler**: what SLURM says about the job (``COMPLETED`` only means the
  batch script exited 0 as far as SLURM is concerned);
* **execution**: what the job recorded in ``search_status.json`` (did the
  inputs verify, did the runtime check pass, did the USPEX master finish);
* **scientific validity**: whether the search's results may be used. Step 4D
  does not assess it (``not_assessed``); every relaxation step is still held
  to the candidate-validity rules by the launcher while the search runs.

``SEARCH_COMPLETED`` means the USPEX search ran to its end. It does not mean
a structure has been determined: finalist selection and the adaptive
relaxation of finalists are a later stage (``finalists.status``
``not_started``).

Uncertain submissions: a submission whose ``sbatch`` outcome is unknown stays
``SEARCH_SUBMITTING`` remotely (``SUBMISSION_UNCERTAIN`` locally) and is never
submitted again automatically. Reconciliation may only move it to
``SEARCH_SUBMITTED``, and only when exactly one job carries this attempt's
identity. Returning it to ``SEARCH_PREPARED`` needs an operator's explicit
confirmation that no job exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# -- remote attempt states ------------------------------------------------------------------

REMOTE_STAGING = "STAGING"
REMOTE_STAGED = "STAGED"
REMOTE_SEARCH_PREPARED = "SEARCH_PREPARED"
REMOTE_SEARCH_SUBMITTING = "SEARCH_SUBMITTING"
REMOTE_SEARCH_SUBMITTED = "SEARCH_SUBMITTED"
REMOTE_STATES = (
    REMOTE_STAGING,
    REMOTE_STAGED,
    REMOTE_SEARCH_PREPARED,
    REMOTE_SEARCH_SUBMITTING,
    REMOTE_SEARCH_SUBMITTED,
)
# The calculation path's states; a structure search never uses them.
CALCULATION_REMOTE_STATES = ("PREPARED", "SUBMITTING", "SUBMITTED")

# -- local states -----------------------------------------------------------------------------

PREPARING = "PREPARING"
PREPARED_LOCALLY = "PREPARED_LOCALLY"
STAGING = "STAGING"
STAGED = "STAGED"
PREPARED = "PREPARED"  # submission-ready: freshly verified on POWER, job script verified
SUBMITTING = "SUBMITTING"
SUBMISSION_UNCERTAIN = "SUBMISSION_UNCERTAIN"
SUBMITTED = "SUBMITTED"
SEARCH_COMPLETED = "SEARCH_COMPLETED"
SEARCH_FAILED = "SEARCH_FAILED"
SEARCH_CANCELLED = "SEARCH_CANCELLED"
SEARCH_TIMED_OUT = "SEARCH_TIMED_OUT"
TERMINAL_STATES = (SEARCH_COMPLETED, SEARCH_FAILED, SEARCH_CANCELLED, SEARCH_TIMED_OUT)
LOCAL_STATES = (
    PREPARING,
    PREPARED_LOCALLY,
    STAGING,
    STAGED,
    PREPARED,
    SUBMITTING,
    SUBMISSION_UNCERTAIN,
    SUBMITTED,
    *TERMINAL_STATES,
)


class LifecycleError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Transition:
    source: str | None
    target: str
    condition: str
    automatic: bool = True


REMOTE_TRANSITIONS = (
    Transition(None, REMOTE_STAGING, "a new attempt reserves its run name, then creates the run directory"),
    Transition(REMOTE_STAGING, REMOTE_STAGED, "finalization verified the complete staged run on POWER"),
    Transition(REMOTE_STAGED, REMOTE_SEARCH_PREPARED,
               "under the attempt claim: a fresh read-only verification of the staged run passed and the "
               "versioned job script was uploaded and verified"),
    Transition(REMOTE_SEARCH_PREPARED, REMOTE_SEARCH_PREPARED,
               "re-verified under the attempt claim (records the new verification time)"),
    Transition(REMOTE_SEARCH_PREPARED, REMOTE_SEARCH_SUBMITTING,
               "under the attempt claim, after a fresh verification and with a matching resource "
               "acknowledgement; written before sbatch is run"),
    Transition(REMOTE_SEARCH_SUBMITTING, REMOTE_SEARCH_SUBMITTED,
               "sbatch returned a job ID, or reconciliation found exactly one job with this attempt's identity"),
    Transition(REMOTE_SEARCH_SUBMITTING, REMOTE_SEARCH_PREPARED,
               "sbatch ran and definitively refused the job, so no job exists"),
    Transition(REMOTE_SEARCH_SUBMITTING, REMOTE_SEARCH_PREPARED,
               "an operator confirmed that no job exists for an uncertain submission", automatic=False),
)

LOCAL_TRANSITIONS = (
    Transition(None, PREPARING, "a new attempt is recorded"),
    Transition(PREPARING, PREPARED_LOCALLY, "the frozen plan's inputs were prepared and verified locally"),
    Transition(PREPARED_LOCALLY, STAGING, "staging started; run name, spec and runtime package are fixed"),
    Transition(STAGING, STAGED, "POWER verified the staged run"),
    Transition(STAGED, PREPARED, "the remote attempt became SEARCH_PREPARED"),
    Transition(PREPARED, PREPARED, "re-verified before submission"),
    Transition(PREPARED, SUBMITTING, "the remote attempt became SEARCH_SUBMITTING"),
    Transition(SUBMITTING, SUBMITTED, "sbatch returned a job ID"),
    Transition(SUBMITTING, PREPARED, "sbatch definitively refused the job"),
    Transition(SUBMITTING, SUBMISSION_UNCERTAIN, "the sbatch outcome is unknown; never submitted again automatically"),
    Transition(SUBMISSION_UNCERTAIN, SUBMITTED, "reconciliation found exactly one job with this attempt's identity"),
    Transition(SUBMISSION_UNCERTAIN, PREPARED, "an operator confirmed that no job exists", automatic=False),
    *(Transition(SUBMITTED, state, "the job ended; see classify_job") for state in TERMINAL_STATES),
)


def _find(transitions, current, target) -> Transition | None:
    for transition in transitions:
        if transition.source == current and transition.target == target:
            return transition
    return None


def require_remote_transition(current: str | None, target: str, *, operator_confirmed: bool = False) -> Transition:
    """The allowed remote transition from ``current`` to ``target``, or :class:`LifecycleError`."""

    for state in (current, target):
        if state in CALCULATION_REMOTE_STATES:
            raise LifecycleError("calculation_state", f"{state} is a calculation state; searches never use it.")
    candidates = [t for t in REMOTE_TRANSITIONS if t.source == current and t.target == target]
    allowed = [t for t in candidates if t.automatic or operator_confirmed]
    if not allowed:
        raise LifecycleError(
            "transition_not_allowed",
            f"A structure-search attempt cannot go from {current or 'no state'} to {target}"
            + (" without an operator's confirmation." if candidates else "."),
        )
    return allowed[0]


def require_local_transition(current: str | None, target: str, *, operator_confirmed: bool = False) -> Transition:
    transition = _find(LOCAL_TRANSITIONS, current, target)
    if transition is None or (not transition.automatic and not operator_confirmed):
        raise LifecycleError(
            "transition_not_allowed",
            f"A structure-search attempt cannot go from {current or 'no state'} to {target}"
            + (" without an operator's confirmation." if transition is not None else "."),
        )
    return transition


def resubmission_allowed(remote_state: str | None) -> bool:
    """Whether sbatch may be run for an attempt in ``remote_state``: only from SEARCH_PREPARED."""

    return remote_state == REMOTE_SEARCH_PREPARED


# -- scheduler, execution and scientific validity ----------------------------------------------

SCHEDULER_ACTIVE = ("PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING")
SCHEDULER_TERMINAL = (
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
)

# Execution outcomes the job records in search_status.json.
EXECUTION_IN_PROGRESS = ("verifying_inputs", "checking_runtime", "running")
EXECUTION_FINISHED = "uspex_finished"
EXECUTION_OUTCOMES = (
    *EXECUTION_IN_PROGRESS,
    EXECUTION_FINISHED,
    "uspex_failed",
    "stopped",
    "refused_inputs",
    "runtime_check_failed",
    "launch_setup_failed",
    "runtime_package_refused",
    "setup_failed",
)
SCIENTIFIC_NOT_ASSESSED = "not_assessed"
FINALISTS_NOT_STARTED = "not_started"


def normalized_scheduler_state(value: Any) -> str:
    token = str(value or "").strip().split(" ", 1)[0].upper().rstrip("+")
    return token if token in SCHEDULER_ACTIVE + SCHEDULER_TERMINAL else "UNKNOWN"


def classify_job(scheduler_state: Any, status: Mapping[str, Any] | None, *, attempt_id: str) -> dict[str, Any]:
    """The local state of a submitted search from SLURM's state and the job's own status record.

    A search is ``SEARCH_COMPLETED`` only when SLURM reports ``COMPLETED`` and
    the job recorded that the USPEX master finished with status 0. SLURM's
    ``COMPLETED`` with any other, a missing or a foreign execution record is a
    failure. Cancellation and walltime are reported as such whatever the job
    managed to record. Scientific validity is never inferred here.
    """

    scheduler = normalized_scheduler_state(scheduler_state)
    execution = _execution(status, attempt_id)
    result = {
        "scheduler": {"state": scheduler, "terminal": scheduler in SCHEDULER_TERMINAL},
        "execution": execution,
        "scientific_validity": {"status": SCIENTIFIC_NOT_ASSESSED},
        "finalists": {"status": FINALISTS_NOT_STARTED},
    }
    if scheduler not in SCHEDULER_TERMINAL:
        result["state"] = SUBMITTED
    elif scheduler == "CANCELLED":
        result["state"] = SEARCH_CANCELLED
    elif scheduler in ("TIMEOUT", "DEADLINE"):
        result["state"] = SEARCH_TIMED_OUT
    elif scheduler == "COMPLETED" and execution["outcome"] == EXECUTION_FINISHED and execution.get(
            "master_exit_status") == 0:
        result["state"] = SEARCH_COMPLETED
    else:
        result["state"] = SEARCH_FAILED
    return result


def _execution(status: Mapping[str, Any] | None, attempt_id: str) -> dict[str, Any]:
    if not isinstance(status, Mapping):
        return {"outcome": "not_recorded"}
    if status.get("attempt_id") != attempt_id:
        return {"outcome": "foreign_record"}
    execution = status.get("execution")
    if not isinstance(execution, Mapping) or execution.get("outcome") not in EXECUTION_OUTCOMES:
        return {"outcome": "unreadable_record"}
    return {
        "outcome": execution["outcome"],
        "master_exit_status": execution.get("master_exit_status"),
    }


__all__ = [
    "CALCULATION_REMOTE_STATES",
    "LOCAL_STATES",
    "LOCAL_TRANSITIONS",
    "LifecycleError",
    "REMOTE_STATES",
    "REMOTE_TRANSITIONS",
    "TERMINAL_STATES",
    "Transition",
    "classify_job",
    "normalized_scheduler_state",
    "require_local_transition",
    "require_remote_transition",
    "resubmission_allowed",
]
