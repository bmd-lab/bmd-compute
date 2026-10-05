"""Phonon displacement planning (M1 foundation; not an executable workflow).

Importing this package does not import Phonopy. Phonopy is imported only when
``build_displacement_plan`` or ``verify_displacement_plan`` is called, so
existing BMD Compute workflows keep working where Phonopy is absent.
"""

from backend.phonons.plan import (
    DisplacementPlan,
    DisplacementTask,
    PhonopyUnavailableError,
    build_displacement_plan,
    verify_displacement_plan,
)
from backend.phonons.policy import (
    PHONON_PLANNING_POLICY_ID,
    PHONON_PLANNING_POLICY_STATUS,
    PHONON_PLANNING_POLICY_VERSION,
    PROVISIONAL_PHONON_POLICY,
    PhononPolicy,
    PhononPolicyError,
)
from backend.phonons.records import (
    DISPLACEMENT_PLAN_SCHEMA,
    DISPLACEMENT_PLAN_SCHEMA_VERSION,
    TASK_KIND,
    PhononPlanContractError,
    canonical_json,
    canonical_sha256,
    validate_displacement_plan_record,
)

__all__ = [
    "DISPLACEMENT_PLAN_SCHEMA",
    "DISPLACEMENT_PLAN_SCHEMA_VERSION",
    "DisplacementPlan",
    "DisplacementTask",
    "PHONON_PLANNING_POLICY_ID",
    "PHONON_PLANNING_POLICY_STATUS",
    "PHONON_PLANNING_POLICY_VERSION",
    "PROVISIONAL_PHONON_POLICY",
    "PhononPlanContractError",
    "PhononPolicy",
    "PhononPolicyError",
    "PhonopyUnavailableError",
    "TASK_KIND",
    "build_displacement_plan",
    "canonical_json",
    "canonical_sha256",
    "validate_displacement_plan_record",
    "verify_displacement_plan",
]
