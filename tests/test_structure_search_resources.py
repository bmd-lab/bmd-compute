"""The exclusive-node resource preset for structure searches only."""

import pytest

from backend.calculations.registry import CalculationValidationError
from backend.calculations.resources import (
    ExecutionResources,
    StructureSearchResources,
    normalize_execution_resources,
    structure_search_resources,
)
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.config import DEFAULT_ACCOUNT, DEFAULT_PARTITION
from backend.submission import build_sbatch_script


def test_search_preset_is_one_exclusive_node_with_all_memory():
    resources = structure_search_resources("240:00:00")

    assert resources.sbatch_directives() == [
        f"#SBATCH -p {DEFAULT_PARTITION}",
        f"#SBATCH --account={DEFAULT_ACCOUNT}",
        "#SBATCH --qos=owner100d",
        "#SBATCH --time=240:00:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=192",
        "#SBATCH --exclusive",
        "#SBATCH --mem=0",
        "#SBATCH --exclude=compute-0-[332-333]",
    ]
    assert resources.to_dict() == {
        "preset": "structure_search_exclusive_node",
        "nodes": 1,
        "ntasks": 192,
        "exclusive": True,
        "mem": "0",
        "walltime": "240:00:00",
        "partition": DEFAULT_PARTITION,
        "account": DEFAULT_ACCOUNT,
        "qos": "owner100d",
        "exclude": "compute-0-[332-333]",
    }


@pytest.mark.parametrize("walltime", [None, "", "auto", "Automatic"])
def test_search_needs_an_explicit_walltime(walltime):
    with pytest.raises(CalculationValidationError, match="explicit walltime"):
        StructureSearchResources(walltime=walltime)


@pytest.mark.parametrize("walltime", ["2400:00:00", "1:00:00"])
def test_search_walltime_up_to_100_days_is_accepted(walltime):
    assert StructureSearchResources(walltime=walltime).walltime == walltime


@pytest.mark.parametrize("walltime", ["2400:00:01", "0:00:00", "10 days", "1-00:00:00"])
def test_search_walltime_over_100_days_or_malformed_is_rejected(walltime):
    with pytest.raises(CalculationValidationError):
        StructureSearchResources(walltime=walltime)


def test_search_tasks_must_fit_on_one_node():
    assert StructureSearchResources(walltime="1:00:00", ntasks=96, node_cores=192).ntasks == 96
    with pytest.raises(CalculationValidationError, match="193 tasks, but a node has 192 cores"):
        StructureSearchResources(walltime="1:00:00", ntasks=193, node_cores=192)


def test_search_exclusion_can_be_empty_and_must_be_a_host_list():
    resources = StructureSearchResources(walltime="1:00:00", exclude="")
    assert not any("--exclude" in line for line in resources.sbatch_directives())
    assert (
        StructureSearchResources(walltime="1:00:00", exclude="compute-0-[332-333],compute-0-7").exclude
        == "compute-0-[332-333],compute-0-7"
    )
    for unsafe in ("compute-0-1; rm -rf /", "compute-0-1\n#SBATCH --qos=normal", "a b"):
        with pytest.raises(CalculationValidationError, match="not a SLURM host list"):
            StructureSearchResources(walltime="1:00:00", exclude=unsafe)


def test_search_queue_is_allow_listed():
    with pytest.raises(CalculationValidationError, match="Queue must be one of"):
        StructureSearchResources(walltime="1:00:00", queue="other-pool")


def test_search_preset_is_not_accepted_as_ordinary_resources():
    with pytest.raises(CalculationValidationError):
        normalize_execution_resources(structure_search_resources("1:00:00"))


# -- ordinary workflows are unchanged ---------------------------------------------


def test_ordinary_resource_defaults_are_unchanged():
    assert ExecutionResources().to_dict() == {
        "nodes": 1,
        "ntasks": 24,
        "mem_gb": 128,
        "walltime": "72:00:00",
        "partition": DEFAULT_PARTITION,
        "account": DEFAULT_ACCOUNT,
    }


def test_ordinary_sbatch_script_has_no_search_directives():
    from backend.parser import parse_structure
    from backend.submission import create_submission_spec

    poscar = "Si\n5.43\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\nSi\n2\ndirect\n0 0 0\n0.25 0.25 0.25\n"
    spec = create_submission_spec(
        {
            "workflow_spec": WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)]).to_dict(),
            "potcar_functional": "PBE_64",
            "structure": {"type": "pasted_text", "format": "poscar", "text": poscar},
        },
        structure=parse_structure(poscar, "poscar"),
        label="ordinary",
        timestamp="20261104-120000",
        ntasks=24,
        mem_gb=128,
        walltime="12:00:00",
        env={},
        submission_attempt_id="12345678-1234-5678-1234-567812345678",
    )
    script = build_sbatch_script(spec)

    assert "#SBATCH --ntasks=24" in script
    assert "#SBATCH --mem=128G" in script
    assert "--exclusive" not in script
    assert "--exclude" not in script
    assert "--mem=0" not in script
    assert "uspex" not in script.lower()
