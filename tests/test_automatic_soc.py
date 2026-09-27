from __future__ import annotations

import json

import pytest
from pymatgen.io.vasp.inputs import Kpoints
from starlette.requests import Request

import main
from backend.calculations.default_treatments import (
    AUTOMATIC_APPLICATION_ADVISORY,
    AUTOMATIC_APPLICATION_APPLIED,
    AUTOMATIC_APPLICATION_NOT_APPLICABLE,
    DISPERSION_CONSIDERATION_ID,
    SOC_CONSIDERATION_ID,
    SPIN_CONSIDERATION_ID,
    resolve_default_treatments,
)
from backend.calculations.hse_band_kpoints import (
    FULL_ZONE_KPOINTS_ADJUSTMENT,
    FullZoneKpointsError,
    automatic_mesh_divisions,
    replace_weighted_kpoints_with_full_zone,
)
from backend.calculations.input_reference import build_input_reference_payload
from backend.calculations.models import Modifier, StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import (
    CalculationValidationError,
    desired_output_workflow_spec,
    validate_workflow_spec,
)
from backend.calculations.vasp_stage_definitions import stage_reads_previous_charge_density
from backend.generated_inputs import preview_generated_inputs
from backend.parser import parse_structure, structure_from_spec
from backend.workflows import (
    build_atomate2_flow_from_spec,
    build_band_structure_input_set_generator,
    run_vasp_kwargs_for_modifiers,
    vasp_job_kwargs_for_modifiers,
)


BI_POSCAR = """Bi
3.3
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Bi
1
direct
0.0 0.0 0.0
"""

PT_POSCAR = """Pt
3.92
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Pt
1
direct
0.0 0.0 0.0
"""

FEPT_POSCAR = """FePt
1.0
3.86 0.0 0.0
0.0 3.86 0.0
0.0 0.0 3.72
Fe Pt
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

BI2SE3_POSCAR = """Bi2Se3
1.0
4.143 0.0 0.0
-2.0715 3.5878 0.0
0.0 0.0 28.636
Bi Se
6 9
direct
0.0 0.0 0.1
0.0 0.0 0.2
0.333333 0.666667 0.433333
0.333333 0.666667 0.533333
0.666667 0.333333 0.766667
0.666667 0.333333 0.866667
0.0 0.0 0.3
0.0 0.0 0.4
0.333333 0.666667 0.633333
0.333333 0.666667 0.733333
0.666667 0.333333 0.966667
0.666667 0.333333 0.066667
0.0 0.0 0.5
0.333333 0.666667 0.833333
0.666667 0.333333 0.166667
"""

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


def request(path: str = "/build-workflow") -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def build_route(poscar: str, *, workflow: str, workflow_spec: WorkflowSpec | None = None):
    return main.build_workflow(
        request(),
        structure=poscar,
        fmt="poscar",
        purpose=None,
        theory=None,
        modifiers=None,
        cpus=None,
        memory_gb=None,
        walltime=None,
        queue=None,
        workflow_spec_json=(
            json.dumps(workflow_spec.to_dict(), sort_keys=True)
            if workflow_spec is not None
            else None
        ),
        workflow=workflow,
        method=None,
    )


def resolve(desired_output: str, poscar: str):
    return resolve_default_treatments(
        parse_structure(poscar),
        desired_output_workflow_spec(desired_output),
        desired_output=desired_output,
    )


def signature(workflow: WorkflowSpec) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (
            stage.stage_type.value,
            stage.theory.value,
            tuple(sorted(modifier.value for modifier in stage.modifiers)),
        )
        for stage in workflow.stages
    ]


def stage_sections(text: str) -> list[str]:
    if "# Stage " not in text:
        return [text]
    return ["# Stage " + section for section in text.split("# Stage ")[1:]]


def incar_value(incar_text: str, key: str) -> str | None:
    for line in incar_text.splitlines():
        if line.startswith(f"{key} = "):
            return line.removeprefix(f"{key} = ")
    return None


def magmom_components(incar_text: str) -> list[float]:
    value = incar_value(incar_text, "MAGMOM")
    assert value is not None, "MAGMOM must be written for SOC stages"
    components: list[float] = []
    for token in value.split():
        if "*" in token:
            count, number = token.split("*", 1)
            components.extend([float(number)] * int(count))
        else:
            components.append(float(token))
    return components


def considerations_by_id(context: dict) -> dict:
    return {
        consideration["id"]: consideration
        for consideration in context["method_considerations"]["considerations"]
    }


def runtime_jobs(submission_spec: dict):
    runtime_structure = structure_from_spec(submission_spec["flow_spec"]["structure"])
    flow = build_atomate2_flow_from_spec(
        runtime_structure,
        submission_spec["flow_spec"],
        run_name=submission_spec["run_name"],
        resources=submission_spec["resources"],
    )
    return runtime_structure, list(flow.jobs)


def job_maker(job):
    maker = getattr(getattr(job, "function", None), "__self__", None)
    assert maker is not None, "expected an atomate2 maker-bound job"
    return maker


def runtime_input_set(submission_spec: dict, stage_index: int):
    runtime_structure, jobs = runtime_jobs(submission_spec)
    generator = job_maker(jobs[stage_index]).input_set_generator
    return generator.get_input_set(runtime_structure, potcar_spec=True)


# --- Static Energy: Bi and Pt -------------------------------------------------


@pytest.mark.parametrize(("poscar", "symbol"), [(BI_POSCAR, "Bi"), (PT_POSCAR, "Pt")])
def test_heavy_element_static_energy_runs_pbe_static_soc_with_zero_moments(poscar, symbol):
    response = build_route(poscar, workflow="energy_only")
    context = response.context

    assert response.status_code == 200
    stages = context["selected_workflow"]["stages"]
    assert len(stages) == 1
    assert stages[0]["stage_type"] == "static"
    assert stages[0]["theory"] == "pbe"
    assert "soc" in stages[0]["modifiers"]
    assert "spin_polarized" not in stages[0]["modifiers"]

    soc = considerations_by_id(context)[SOC_CONSIDERATION_ID]
    assert soc["automatic_application_state"] == AUTOMATIC_APPLICATION_APPLIED
    assert soc["automatic_application"]["stage_indices"] == [1]
    rendered = response.template.render(context)
    assert "Spin-Orbit Coupling (SOC) applied" in rendered
    assert f"{symbol} detected. Spin-orbit coupling (SOC) has been included automatically" in rendered
    assert "Suggested to activate the Spin-Orbit Coupling" not in rendered

    generated = context["generated_inputs"]
    assert generated["vasp_executable"] == "vasp_ncl"
    incar = generated["incar"]
    assert "LSORBIT = True" in incar
    assert "LNONCOLLINEAR = True" in incar
    assert "ISYM = 0" in incar
    assert "ISPIN =" not in incar
    assert magmom_components(incar) == [0.0, 0.0, 0.0]

    submission_spec = context["submission_spec"]
    assert sorted(submission_spec["flow_spec"]["workflow_spec"]["stages"][0]["modifiers"]) == sorted(
        stages[0]["modifiers"]
    )
    vasp_stages = submission_spec["provenance"]["vasp"]["stages"]
    assert [stage["executable"] for stage in vasp_stages] == ["vasp_ncl"]
    assert vasp_stages[0]["custodian_vasp_job_kwargs"] == {"auto_gamma": False}

    runtime_incar = str(runtime_input_set(submission_spec, 0).incar)
    assert magmom_components(runtime_incar) == magmom_components(incar)
    assert "LSORBIT = True" in runtime_incar


def test_magnetic_heavy_element_structure_keeps_nonzero_soc_starting_moments():
    resolution = resolve("energy_only", FEPT_POSCAR)
    assert signature(resolution.resolved_workflow) == [
        ("static", "pbe", ("soc", "spin_polarized")),
    ]

    preview = preview_generated_inputs(
        parse_structure(FEPT_POSCAR),
        resolution.resolved_workflow,
        resources={"ntasks": 24},
    )
    components = magmom_components(preview["incar"])
    vectors = [components[index:index + 3] for index in range(0, len(components), 3)]
    assert len(vectors) == 2
    assert all(vector[0] == 0.0 and vector[1] == 0.0 for vector in vectors)
    # Fe keeps the Materials Project starting moment along SAXIS.
    assert vectors[0][2] == 5.0
    assert all(vector[2] > 0 for vector in vectors)
    assert "ISPIN =" not in preview["incar"]


def test_nonmagnetic_custom_soc_uses_zero_moments_and_spin_choice_restores_them():
    structure = parse_structure(SI_POSCAR)
    soc_only = preview_generated_inputs(
        structure,
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC})], recipe="custom"),
    )
    soc_spin = preview_generated_inputs(
        structure,
        WorkflowSpec(
            [StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC, Modifier.SPIN_POLARIZED})],
            recipe="custom",
        ),
    )

    assert magmom_components(soc_only["incar"]) == [0.0] * 6
    assert magmom_components(soc_spin["incar"]) == [0.0, 0.0, 0.6] * 2


# --- SOC + D3 ------------------------------------------------------------------


def test_layered_heavy_element_static_energy_combines_soc_and_d3_on_one_stage():
    resolution = resolve("energy_only", BI2SE3_POSCAR)

    assert signature(resolution.resolved_workflow) == [
        ("static", "pbe", ("dispersion", "soc")),
    ]
    assert resolution.resolved_workflow.stages[0].options == {"dispersion": {"method": "dftd3-bj"}}
    assert [treatment.consideration_id for treatment in resolution.applied_treatments] == [
        DISPERSION_CONSIDERATION_ID,
        SOC_CONSIDERATION_ID,
    ]

    preview = preview_generated_inputs(
        parse_structure(BI2SE3_POSCAR),
        resolution.resolved_workflow,
        resources={"ntasks": 24},
    )
    assert preview["vasp_executable"] == "vasp_ncl"
    assert "IVDW = 12" in preview["incar"]
    assert "LSORBIT = True" in preview["incar"]
    assert magmom_components(preview["incar"]) == [0.0] * (3 * 15)

    validate_workflow_spec(
        WorkflowSpec(
            [
                StageSpec(
                    StageType.STATIC,
                    Theory.PBE,
                    {Modifier.SOC, Modifier.DISPERSION},
                    options={"dispersion": {"method": "dftd3"}},
                )
            ],
            recipe="custom",
        )
    )


# --- DOS and Band Structure: Bi2Se3 --------------------------------------------


@pytest.mark.parametrize(
    ("desired_output", "terminal"),
    [("electronic_dos", "dos"), ("electronic_band_structure", "band_structure")],
)
def test_bi2se3_dos_and_band_apply_soc_to_hse_stages_only(desired_output, terminal):
    resolution = resolve(desired_output, BI2SE3_POSCAR)

    assert signature(resolution.resolved_workflow) == [
        ("relax", "pbe", ("dispersion",)),
        ("static", "hse06", ("soc",)),
        (terminal, "hse06", ("soc",)),
    ]
    soc_treatment = next(
        treatment
        for treatment in resolution.applied_treatments
        if treatment.consideration_id == SOC_CONSIDERATION_ID
    )
    assert soc_treatment.stage_indices == (2, 3)
    assert resolution.advisory_consideration_ids == ()

    response = build_route(BI2SE3_POSCAR, workflow=desired_output)
    context = response.context
    assert response.status_code == 200
    assert [stage["modifiers"] for stage in context["selected_workflow"]["stages"]] == [
        ["dispersion"],
        ["soc"],
        ["soc"],
    ]

    generated = context["generated_inputs"]
    assert [item["executable"] for item in generated["vasp_executables"]] == [
        "vasp_std",
        "vasp_ncl",
        "vasp_ncl",
    ]
    relax, static, analysis = stage_sections(generated["incar"])
    assert "LSORBIT" not in relax
    assert "ISPIN = 1" in relax
    assert "IVDW = 12" in relax
    for section in (static, analysis):
        assert "# VASP executable - vasp_ncl" in section
        assert "LHFCALC = True" in section
        assert "LSORBIT = True" in section
        assert "ISYM = 0" in section
        assert "ISPIN =" not in section
        assert "IVDW" not in section
        assert magmom_components(section) == [0.0] * (3 * 15)

    soc = considerations_by_id(context)[SOC_CONSIDERATION_ID]
    assert soc["automatic_application_state"] == AUTOMATIC_APPLICATION_APPLIED
    rendered = response.template.render(context)
    assert "in the HSE06 Static Energy and HSE06" in rendered

    submission_spec = context["submission_spec"]
    flow_spec = submission_spec["flow_spec"]
    assert flow_spec["automatic_treatments"]["resolved_workflow"] == flow_spec["workflow_spec"]
    assert submission_spec["provenance"]["execution"]["automatic_treatments"] == flow_spec["automatic_treatments"]
    vasp_stages = submission_spec["provenance"]["vasp"]["stages"]
    assert [stage["executable"] for stage in vasp_stages] == ["vasp_std", "vasp_ncl", "vasp_ncl"]
    assert [stage["custodian_vasp_job_kwargs"] for stage in vasp_stages] == [
        {},
        {"auto_gamma": False},
        {"auto_gamma": False},
    ]

    runtime_structure, jobs = runtime_jobs(submission_spec)
    run_kwargs = [job_maker(job).run_vasp_kwargs for job in jobs]
    assert "vasp_cmd" not in run_kwargs[0]
    assert "vasp_job_kwargs" not in run_kwargs[0]
    for kwargs in run_kwargs[1:]:
        assert kwargs["vasp_cmd"].endswith("vasp_ncl")
        assert kwargs["vasp_job_kwargs"] == {"auto_gamma": False}

    # The terminal stage's structure is only known at run time; its SOC
    # starting moments must still be written, and must match the preview.
    terminal_runtime = job_maker(jobs[2]).input_set_generator.get_input_set(
        runtime_structure,
        potcar_spec=True,
    )
    assert magmom_components(str(terminal_runtime.incar)) == magmom_components(analysis)
    assert "LSORBIT = True" in str(terminal_runtime.incar)


def test_relaxed_structure_keeps_relaxations_non_soc_and_reports_it():
    resolution = resolve("relaxed_structure", BI_POSCAR)
    assert all(Modifier.SOC not in stage.modifiers for stage in resolution.resolved_workflow.stages)
    assert resolution.advisory_consideration_ids == ()
    assert [item["consideration_id"] for item in resolution.not_applicable_considerations] == [
        SOC_CONSIDERATION_ID,
    ]

    response = build_route(BI_POSCAR, workflow="relaxed_structure")
    soc = considerations_by_id(response.context)[SOC_CONSIDERATION_ID]
    assert soc["automatic_application_state"] == AUTOMATIC_APPLICATION_NOT_APPLICABLE
    rendered = response.template.render(response.context)
    assert "BMD Compute keeps geometry optimisations non-SOC" in rendered
    assert "Suggested to activate the Spin-Orbit Coupling" not in rendered
    assert "vasp_ncl" not in json.dumps(response.context["generated_inputs"]["vasp_executables"])


def test_custom_workflow_is_not_given_automatic_soc():
    workflow = WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE)], recipe="custom")
    response = build_route(BI_POSCAR, workflow="custom", workflow_spec=workflow)

    assert response.context["selected_workflow"]["stages"][0]["modifiers"] == []
    soc = considerations_by_id(response.context)[SOC_CONSIDERATION_ID]
    assert soc["automatic_application_state"] == AUTOMATIC_APPLICATION_ADVISORY
    assert "LSORBIT" not in response.context["generated_inputs"]["incar"]
    assert "automatic_treatments" not in response.context["submission_spec"]["flow_spec"]


def test_spin_and_soc_compose_on_hse_stages_for_magnetic_heavy_elements():
    resolution = resolve("electronic_band_structure", FEPT_POSCAR)
    stage_modifiers = [set(item[2]) for item in signature(resolution.resolved_workflow)]
    assert "soc" not in stage_modifiers[0]
    assert "spin_polarized" in stage_modifiers[0]
    assert stage_modifiers[1] == {"soc", "spin_polarized"}
    assert stage_modifiers[2] == {"soc", "spin_polarized"}
    applied = {treatment.consideration_id for treatment in resolution.applied_treatments}
    assert {SPIN_CONSIDERATION_ID, SOC_CONSIDERATION_ID} <= applied


# --- Custodian auto_gamma --------------------------------------------------------


def test_soc_stages_disable_custodian_auto_gamma_and_others_do_not():
    assert vasp_job_kwargs_for_modifiers({Modifier.SOC}) == {"auto_gamma": False}
    assert vasp_job_kwargs_for_modifiers({Modifier.SOC, Modifier.GAMMA_ONLY}) == {"auto_gamma": False}
    assert vasp_job_kwargs_for_modifiers({Modifier.SPIN_POLARIZED}) == {}
    assert vasp_job_kwargs_for_modifiers(()) == {}

    soc_kwargs = run_vasp_kwargs_for_modifiers({Modifier.SOC, Modifier.GAMMA_ONLY})
    assert soc_kwargs["vasp_cmd"].endswith("vasp_ncl")
    assert soc_kwargs["vasp_job_kwargs"] == {"auto_gamma": False}
    assert "vasp_job_kwargs" not in run_vasp_kwargs_for_modifiers(())
    # Each call returns an independent dict: atomate2 mutates vasp_job_kwargs.
    soc_kwargs["vasp_job_kwargs"]["auto_npar"] = False
    assert run_vasp_kwargs_for_modifiers({Modifier.SOC})["vasp_job_kwargs"] == {"auto_gamma": False}


# --- Full-zone HSE06 Band + SOC k-points -----------------------------------------


def _weighted_and_zero(kpoints):
    weighted = [
        (tuple(point), weight)
        for point, weight in zip(kpoints.kpts, kpoints.kpts_weights)
        if weight > 0
    ]
    zero = [
        (tuple(point), label)
        for point, weight, label in zip(kpoints.kpts, kpoints.kpts_weights, kpoints.labels)
        if weight == 0
    ]
    return weighted, zero


def _grid_key(point):
    return tuple(round(float(value) % 1.0, 6) % 1.0 for value in point)


@pytest.mark.parametrize("poscar", [SI_POSCAR, BI2SE3_POSCAR])
def test_hse_band_soc_weighted_mesh_spans_full_zone_and_keeps_zero_weight_path(poscar):
    structure = parse_structure(poscar)
    plain = build_band_structure_input_set_generator(structure, theory=Theory.HSE06).get_input_set(
        structure,
        potcar_spec=True,
    )
    soc = build_band_structure_input_set_generator(
        structure,
        theory=Theory.HSE06,
        modifiers={Modifier.SOC},
    ).get_input_set(structure, potcar_spec=True)

    mesh = automatic_mesh_divisions(plain.poscar.structure, 64)
    mesh_size = mesh[0] * mesh[1] * mesh[2]
    plain_weighted, plain_zero = _weighted_and_zero(plain.kpoints)
    soc_weighted, soc_zero = _weighted_and_zero(soc.kpoints)

    # Non-SOC HSE06 band structure is untouched: a symmetry-reduced mesh.
    assert sum(weight for _, weight in plain_weighted) == mesh_size
    assert len(plain_weighted) < mesh_size

    # SOC: every mesh point once, equal weights, weighted points first.
    assert len(soc_weighted) == mesh_size
    assert {weight for _, weight in soc_weighted} == {1}
    assert len({_grid_key(point) for point, _ in soc_weighted}) == mesh_size
    assert all(weight > 0 for weight in soc.kpoints.kpts_weights[:mesh_size])
    assert {_grid_key(point) for point, _ in plain_weighted} <= {
        _grid_key(point) for point, _ in soc_weighted
    }
    assert soc_zero == plain_zero
    assert len(soc_zero) > 0
    assert soc.kpoints.num_kpts == len(soc.kpoints.kpts)

    assert soc.incar["ISYM"] == 0
    assert soc.incar["LSORBIT"] is True
    assert "NCORE" not in soc.incar


def test_hse_band_soc_preview_runtime_and_reference_share_full_zone_kpoints():
    response = build_route(BI2SE3_POSCAR, workflow="electronic_band_structure")
    submission_spec = response.context["submission_spec"]

    runtime_kpoints = runtime_input_set(submission_spec, 2).kpoints
    terminal_preview = stage_sections(response.context["generated_inputs"]["kpoints"])[2]
    assert "BMD full-zone weighted mesh for SOC" in str(runtime_kpoints)
    assert "BMD full-zone weighted mesh for SOC" in terminal_preview
    for line in str(runtime_kpoints).splitlines()[1:]:
        assert line in terminal_preview

    payload = build_input_reference_payload(
        {
            "structure": {"type": "pasted_text", "format": "poscar", "text": BI2SE3_POSCAR},
            "workflow_spec": submission_spec["flow_spec"]["workflow_spec"],
            "resources": {"ntasks": 24, "mem_gb": 128},
            "potcar_functional": "PBE_64",
        },
        include_provenance=False,
    )
    band_stage = payload["reference"]["stages"][2]
    assert band_stage["generator"]["bmd_kpoints_adjustment"] == FULL_ZONE_KPOINTS_ADJUSTMENT
    assert "bmd_kpoints_adjustment" not in payload["reference"]["stages"][1]["generator"]


def _synthetic_band_kpoints(weights):
    return Kpoints(
        comment="synthetic",
        num_kpts=6,
        style=Kpoints.supported_modes.Reciprocal,
        kpts=[
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
        ],
        kpts_weights=weights,
        coord_type="Reciprocal",
        labels=[None, None, None, None, "\\Gamma", "X"],
    )


def test_full_zone_expansion_of_a_reduced_mesh_is_exact():
    expanded = replace_weighted_kpoints_with_full_zone(
        _synthetic_band_kpoints([1, 3, 3, 1, 0, 0]),
        mesh=(2, 2, 2),
    )

    assert expanded.kpts_weights == [1] * 8 + [0, 0]
    assert expanded.labels[8:] == ["\\Gamma", "X"]
    assert [tuple(point) for point in expanded.kpts[8:]] == [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    assert {_grid_key(point) for point in expanded.kpts[:8]} == {
        _grid_key((x, y, z))
        for x in (0.0, 0.5)
        for y in (0.0, 0.5)
        for z in (0.0, 0.5)
    }


@pytest.mark.parametrize(
    ("weights", "mesh"),
    [
        ([1, 3, 3, 2, 0, 0], (2, 2, 2)),  # weights do not sum to the mesh size
        ([1, 3, 3, 1, 0, 0], (4, 2, 1)),  # right size, but points are not on that mesh
        ([0, 0, 0, 0, 0, 0], (2, 2, 2)),  # nothing weighted to expand
    ],
)
def test_full_zone_expansion_refuses_inconsistent_input(weights, mesh):
    with pytest.raises(FullZoneKpointsError):
        replace_weighted_kpoints_with_full_zone(_synthetic_band_kpoints(weights), mesh=mesh)


# --- Fixed-charge-density (ICHARG=11) SOC chaining --------------------------------


def test_only_pbe_analysis_stages_read_the_previous_fixed_charge_density():
    assert stage_reads_previous_charge_density(StageType.DOS, Theory.PBE)
    assert stage_reads_previous_charge_density(StageType.BAND_STRUCTURE, Theory.PBE)
    assert not stage_reads_previous_charge_density(StageType.DOS, Theory.HSE06)
    assert not stage_reads_previous_charge_density(StageType.BAND_STRUCTURE, Theory.HSE06)
    assert not stage_reads_previous_charge_density(StageType.STATIC, Theory.PBE)
    assert not stage_reads_previous_charge_density(StageType.RELAX, Theory.PBE)


@pytest.mark.parametrize("terminal", [StageType.DOS, StageType.BAND_STRUCTURE])
def test_soc_charge_density_cannot_feed_a_non_soc_icharg11_stage(terminal):
    workflow = WorkflowSpec(
        [
            StageSpec(StageType.RELAX, Theory.PBE),
            StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC}),
            StageSpec(terminal, Theory.PBE),
        ],
        recipe="custom",
    )
    with pytest.raises(CalculationValidationError) as excinfo:
        validate_workflow_spec(workflow)
    assert "fixed charge density" in excinfo.value.message
    assert "Spin-Orbit Coupling (SOC)" in excinfo.value.message


@pytest.mark.parametrize(
    "workflow",
    [
        # Non-SOC fixed-density chain is unchanged.
        WorkflowSpec([StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.DOS, Theory.PBE)]),
        # Static -> Static+SOC shares only the structure.
        WorkflowSpec(
            [StageSpec(StageType.STATIC, Theory.PBE), StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC})]
        ),
        # A non-SOC static between SOC and a PBE DOS restores a valid chain.
        WorkflowSpec(
            [
                StageSpec(StageType.STATIC, Theory.PBE, {Modifier.SOC}),
                StageSpec(StageType.STATIC, Theory.PBE),
                StageSpec(StageType.DOS, Theory.PBE),
            ]
        ),
        # HSE06 analysis stages are self-consistent and only inherit the structure.
        WorkflowSpec(
            [StageSpec(StageType.STATIC, Theory.HSE06), StageSpec(StageType.DOS, Theory.HSE06, {Modifier.SOC})]
        ),
        WorkflowSpec(
            [
                StageSpec(StageType.STATIC, Theory.HSE06, {Modifier.SOC}),
                StageSpec(StageType.BAND_STRUCTURE, Theory.HSE06),
            ]
        ),
    ],
)
def test_valid_custom_chains_remain_allowed(workflow):
    validate_workflow_spec(workflow)
