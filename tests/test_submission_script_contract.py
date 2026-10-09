import hashlib
import json
from pathlib import Path

import pytest

import backend.submission as submission
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.resources import ExecutionResources
from backend.config import DEFAULT_ACCOUNT, DEFAULT_PARTITION, MODULES
from backend.submission import (
    build_sbatch_script,
    build_standalone_slurm_example,
    build_submission_summary,
    build_submission_script_artifact,
    remote_preparation_file_groups,
)
from backend.workflows import vasp_command_for_modifiers, vasp_executable_for_modifiers
from main import build_submission_state


POSCAR = """Si
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

TEMPLATE_SOURCE = Path("templates/index.html").read_text(encoding="utf-8")


def _submission_state(workflow_spec: WorkflowSpec):
    return build_submission_state(
        structure_text=POSCAR,
        fmt="poscar",
        workflow_spec=workflow_spec,
        execution_resources=ExecutionResources(
            nodes=1,
            cpus=48,
            memory_gb=256,
            walltime="12:00:00",
            queue=DEFAULT_PARTITION,
            account=DEFAULT_ACCOUNT,
        ),
        timestamp="20260922-120000",
        submission_attempt_id="12345678-1234-5678-1234-567812345678",
    )


def _historical_submission_spec(workflow_spec: WorkflowSpec) -> dict:
    """A record prepared before HSE06 + SOC was closed to new calculations.

    New requests are refused at ``main.build_submission_state_from_structure``;
    an existing record's submission spec is rendered below that boundary.
    """

    from backend.parser import parse_structure

    return submission.create_submission_spec(
        {
            "workflow_spec": workflow_spec.to_dict(),
            "potcar_functional": "PBE_64",
            "structure": {"type": "pasted_text", "format": "poscar", "text": POSCAR},
        },
        structure=parse_structure(POSCAR, "poscar"),
        label="historical",
        timestamp="20260922-120000",
        ntasks=48,
        mem_gb=256,
        walltime="12:00:00",
        env={},
        submission_attempt_id="12345678-1234-5678-1234-567812345678",
    )


def _uploaded_script(submission_spec: dict) -> str:
    group = next(
        group
        for group in remote_preparation_file_groups(submission_spec)
        if group["step"] == "Submission script written"
    )
    assert len(group["files"]) == 1
    return group["files"][0]["text"]


@pytest.mark.parametrize("theory", [Theory.PBE, Theory.HSE06])
def test_single_stage_pedagogical_script_uses_resolved_vasp_std(theory):
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, theory)])
    _, _, generated_inputs, submission_spec = _submission_state(workflow)

    script = generated_inputs["slurm_script"]
    assert script == build_standalone_slurm_example(submission_spec)["text"]
    assert script.startswith("#!/bin/bash\n")
    assert f"#SBATCH -p {DEFAULT_PARTITION}" in script
    assert f"#SBATCH --account={DEFAULT_ACCOUNT}" in script
    expected_job_name = "vasp_run_hse_static" if theory is Theory.HSE06 else "vasp_run_static"
    assert f"#SBATCH -J {expected_job_name}" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=48" in script
    assert "#SBATCH --mem=256G" in script
    for module_name in MODULES:
        assert f"module load {module_name}\n" in script
    assert "mpirun -n $SLURM_NTASKS vasp_std\n" in script
    assert generated_inputs["slurm_script_kind"] == "standalone"
    assert generated_inputs["slurm_script_description"] == (
        "Equivalent standalone script for running this calculation directly on POWER."
    )


def test_pedagogical_script_omits_bmd_execution_plumbing():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    _, _, generated_inputs, submission_spec = _submission_state(workflow)
    script = generated_inputs["slurm_script"]

    for internal_text in (
        "run_job.py",
        "submission.json",
        "BMD_SUBMISSION_ATTEMPT_ID",
        "BMD_RUNTIME",
        submission_spec["runner"]["python"],
        submission_spec["paths"]["log_out"],
    ):
        assert internal_text not in script
    assert "export VASP_CMD=" not in script
    assert script != generated_inputs["exact_slurm_script"]


def test_submission_script_digest_covers_exact_displayed_and_uploaded_bytes():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    _, _, generated_inputs, submission_spec = _submission_state(workflow)
    displayed = generated_inputs["exact_slurm_script"]
    uploaded = _uploaded_script(submission_spec)
    artifact = build_submission_script_artifact(submission_spec)
    provenance = submission_spec["provenance"]["execution"]["submission_script"]
    expected_digest = hashlib.sha256(displayed.encode("utf-8")).hexdigest()

    assert artifact["text"] == displayed == uploaded
    assert artifact["sha256"] == expected_digest
    assert artifact["size_bytes"] == len(displayed.encode("utf-8"))
    assert provenance == {
        "path": submission_spec["paths"]["remote_script"],
        "builder": "backend.submission.build_sbatch_script",
        "encoding": "utf-8",
        "hash_algorithm": "sha256",
        "sha256": expected_digest,
        "size_bytes": len(displayed.encode("utf-8")),
    }


def test_mixed_workflow_keeps_stage_local_executable_provenance():
    workflow = WorkflowSpec([
        StageSpec(StageType.STATIC, Theory.PBE),
        StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC}),
    ])
    _, _, generated_inputs, submission_spec = _submission_state(workflow)

    stages = submission_spec["provenance"]["vasp"]["stages"]
    assert [stage["executable"] for stage in stages] == ["vasp_std", "vasp_ncl"]
    assert [row["executable"] for row in generated_inputs["vasp_executables"]] == [
        "vasp_std",
        "vasp_ncl",
    ]
    assert [
        stage["executable"]
        for stage in generated_inputs["submission_summary"]["execution"]["stages"]
    ] == ["vasp_std", "vasp_ncl"]
    pedagogical = generated_inputs["slurm_script"]
    assert generated_inputs["slurm_script_kind"] == "illustrative_workflow"
    assert "not a standalone runnable multi-stage workflow" in pedagogical
    assert "# Stage 1 - PBE Static Energy" in pedagogical
    assert "# Stage 2 - PBE Static Energy + Spin-Orbit Coupling (SOC)" in pedagogical
    assert "mpirun -n $SLURM_NTASKS vasp_std" in pedagogical
    assert "mpirun -n $SLURM_NTASKS vasp_ncl" in pedagogical
    assert "manages structure and result transfer between stages" in pedagogical
    assert generated_inputs["exact_slurm_script"] == _uploaded_script(submission_spec)


def test_submission_summary_is_structured_from_submission_and_provenance():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    _, _, generated_inputs, submission_spec = _submission_state(workflow)
    summary = generated_inputs["submission_summary"]

    assert summary == build_submission_summary(submission_spec)
    assert summary["job_name"] == submission_spec["run_name"]
    assert summary["resources"] == {
        "partition": submission_spec["cluster"]["partition"],
        "account": submission_spec["cluster"]["account"],
        "walltime": submission_spec["resources"]["walltime"],
        "nodes": submission_spec["resources"]["nodes"],
        "tasks": submission_spec["resources"]["ntasks"],
        "memory_gb": submission_spec["resources"]["mem_gb"],
    }
    assert summary["execution"]["run_directory"] == submission_spec["paths"]["run_dir"]
    assert summary["execution"]["runner_script"] == submission_spec["runner"]["script_name"]
    assert summary["environment"]["modules"] == submission_spec["modules"]["load"]
    assert summary["execution"]["stages"] == [{
        "index": 1,
        "display_name": "PBE Static Energy",
        "stage_type": "static",
        "theory": "pbe",
        "modifiers": [],
        "executable": "vasp_std",
    }]
    assert summary["artifact"]["sha256"] == submission_spec["provenance"]["execution"][
        "submission_script"
    ]["sha256"]
    assert "#!/usr/bin/env bash" not in json.dumps(summary)
    assert summary != _uploaded_script(submission_spec)


def test_soc_submission_summary_reports_vasp_ncl():
    workflow = WorkflowSpec([
        StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC}),
    ])
    _, _, generated_inputs, _ = _submission_state(workflow)
    stage = generated_inputs["submission_summary"]["execution"]["stages"][0]

    assert stage["display_name"] == "PBE Static Energy + Spin-Orbit Coupling (SOC)"
    assert stage["executable"] == "vasp_ncl"


def test_hse06_soc_submission_summary_uses_authoritative_stage_provenance():
    # Historical HSE06 + SOC records still summarise from their own provenance.
    workflow = WorkflowSpec([
        StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
    ])
    summary = build_submission_summary(_historical_submission_spec(workflow))
    stage = summary["execution"]["stages"][0]

    assert stage["display_name"] == "HSE06 Static Energy + Spin-Orbit Coupling (SOC)"
    assert stage["executable"] == "vasp_ncl"


@pytest.mark.parametrize(
    ("theory", "modifiers", "expected_executable"),
    [
        (Theory.PBE, set(), "vasp_std"),
        (Theory.HSE06, set(), "vasp_std"),
        (Theory.PBE, {Modifier.SOC}, "vasp_ncl"),
        (Theory.HSE06, {Modifier.SOC}, "vasp_ncl"),
    ],
)
def test_pedagogical_script_uses_authoritative_stage_executable(
    theory, modifiers, expected_executable
):
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, theory, modifiers)])
    if theory is Theory.HSE06 and Modifier.SOC in modifiers:
        # Closed to new calculations; historical records still render.
        submission_spec = _historical_submission_spec(workflow)
    else:
        _, _, _, submission_spec = _submission_state(workflow)
    submission_spec["environment"]["VASP_CMD"] = "mpirun -n $SLURM_NTASKS wrong_global"

    script = build_standalone_slurm_example(submission_spec)["text"]

    assert f"mpirun -n $SLURM_NTASKS {expected_executable}" in script
    assert "wrong_global" not in script


def test_exact_script_emits_each_environment_export_once():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)])
    _, _, generated_inputs, _ = _submission_state(workflow)
    script = generated_inputs["exact_slurm_script"]

    assert script.count("set -e -o pipefail") == 1
    for variable in (
        "VASP_CMD",
        "PMG_VASP_PSP_DIR",
        "JOBFLOW_CONFIG_FILE",
        "CUSTODIAN_NO_GZIP",
        "ATOMATE2_VASP_ZIP_FILES",
    ):
        assert script.count(f"export {variable}=") == 1


def test_template_distinguishes_summary_from_exact_script_artifact():
    assert 'for="input-tab-submission-summary">Submission Summary</label>' in TEMPLATE_SOURCE
    assert 'for="input-tab-slurm">SLURM Script</label>' in TEMPLATE_SOURCE
    assert 'for="input-tab-slurm">Exact SLURM Script</label>' not in TEMPLATE_SOURCE
    assert "Equivalent standalone script for running this calculation directly on POWER." in (
        TEMPLATE_SOURCE
    )
    assert "generated_inputs.submission_summary.resources.partition" in TEMPLATE_SOURCE
    assert "generated_inputs.submission_summary.execution.stages" in TEMPLATE_SOURCE
    assert "generated_inputs.submission_summary.artifact.sha256" in TEMPLATE_SOURCE
    assert "{{ generated_inputs.slurm_script }}" in TEMPLATE_SOURCE
    assert "<summary>Exact BMD Submission Script</summary>" in TEMPLATE_SOURCE
    assert "BMD Compute's internal execution artifact" in TEMPLATE_SOURCE
    assert "{{ generated_inputs.exact_slurm_script }}" in TEMPLATE_SOURCE


@pytest.mark.parametrize("theory", [Theory.PBE, Theory.HSE06])
def test_soc_stage_executable_routing_remains_vasp_ncl(theory):
    stage = StageSpec(StageType.STATIC, theory, {Modifier.SOC})

    assert vasp_executable_for_modifiers(stage.modifiers) == "vasp_ncl"
    assert vasp_command_for_modifiers(
        stage.modifiers,
        base_command="mpirun -n $SLURM_NTASKS vasp_std",
    ) == "mpirun -n '$SLURM_NTASKS' vasp_ncl"


def test_parallel_simplified_slurm_preview_builder_no_longer_exists():
    assert not hasattr(submission, "build_slurm_preview_script")
