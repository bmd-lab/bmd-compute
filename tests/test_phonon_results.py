"""Phonon results discovery, parsing and presentation (Milestone B).

Fixtures in tests/fixtures/phonons/si were written by Phonopy through the
atomate2 analysis path (see its README). Nothing here contacts POWER.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from backend.remote_result_parser import (
    PHONON_IMAGINARY_REPORT_THRESHOLD_THZ,
    _phonon_result_payload,
    _result_has_final_relaxation_stage,
)
from backend.results import load_results_for_completed_job, phonon_result_directories
from backend.workflow_results import PHONON_RESULT_FILES

import test_results as result_cases


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phonons" / "si"
RUN_DIR = result_cases.RUN_DIR
PHONON_WORKFLOW_SPEC = {
    "stages": [
        {"stage_type": "relax", "theory": "pbe", "modifiers": [], "label": None, "options": {}},
        {"stage_type": "phonon_forces", "theory": "pbe", "modifiers": [], "label": None, "options": {}},
    ],
    "label": None,
    "recipe": "phonons",
}
PHONON_SUBMISSION_SPEC = {
    **result_cases.submission_spec,
    "flow_spec": {"workflow_spec": PHONON_WORKFLOW_SPEC, "workflow": "custom_workflow", "potcar_functional": "PBE_64"},
    "paths": {
        "run_dir": RUN_DIR,
        "stage_dirs": {"stage_01": f"{RUN_DIR}/stage_01", "stage_02": f"{RUN_DIR}/stage_02"},
        # As prepared by Milestone A: the result directory is the phonon stage.
        "result_dir": f"{RUN_DIR}/stage_02",
    },
}
ANALYSIS_DIR = f"{RUN_DIR}/stage_02/generate_frequencies_eigenvectors"


def fixture_paths(directory: Path = FIXTURES) -> dict:
    return {key: str(directory / filename) for key, filename in PHONON_RESULT_FILES.items()}


def payload(directory: Path = FIXTURES):
    summary, visualizations = _phonon_result_payload(fixture_paths(directory))
    return summary, {item["id"]: item for item in visualizations}


# --- 1. discovery -----------------------------------------------------------------------


class PhononResultsRunner(result_cases.ResultsRunner):
    def __init__(self, *, phonon_files_present=True):
        super().__init__()
        self.phonon_files_present = phonon_files_present

    def is_file(self, remote_path):
        if remote_path.startswith(ANALYSIS_DIR + "/"):
            self.checked_paths.append(remote_path)
            return self.phonon_files_present
        return super().is_file(remote_path)


def capture_parser(store):
    def parser(files, context):
        store["files"] = {key: value["path"] for key, value in files.items()}
        store["context"] = context
        vasp_files = {key: value for key, value in files.items() if key not in PHONON_RESULT_FILES}
        return result_cases.fake_parser(vasp_files, context)

    return parser


def test_phonon_results_read_the_relax_vasp_outputs_and_the_nested_analysis_files():
    assert phonon_result_directories(RUN_DIR, PHONON_SUBMISSION_SPEC, None) is None
    captured = {}
    runner = PhononResultsRunner()
    result = load_results_for_completed_job(
        result_cases.monitoring_success,
        submission_spec=PHONON_SUBMISSION_SPEC,
        runner_factory=lambda: runner,
        parser=capture_parser(captured),
    )
    assert result["status"] == "success"
    assert result["workdir"] == f"{RUN_DIR}/stage_01"
    assert captured["files"] == {
        "contcar": f"{RUN_DIR}/stage_01/CONTCAR",
        "outcar": f"{RUN_DIR}/stage_01/OUTCAR",
        "vasprun": f"{RUN_DIR}/stage_01/vasprun.xml",
        "phonon_band_structure": f"{ANALYSIS_DIR}/phonon_band_structure.yaml",
        "phonon_dos": f"{ANALYSIS_DIR}/phonon_dos.yaml",
        "phonopy_yaml": f"{ANALYSIS_DIR}/phonopy.yaml",
    }
    # The phonon stage directory is never checked for VASP outputs.
    assert not any(path.startswith(f"{RUN_DIR}/stage_02/") and path.endswith(("CONTCAR", "OUTCAR", "vasprun.xml"))
                   for path in runner.checked_paths)


def test_missing_phonon_outputs_do_not_fail_the_relaxation_results():
    captured = {}
    result = load_results_for_completed_job(
        result_cases.monitoring_success,
        submission_spec=PHONON_SUBMISSION_SPEC,
        runner_factory=lambda: PhononResultsRunner(phonon_files_present=False),
        parser=capture_parser(captured),
    )
    assert result["status"] == "success"
    assert set(captured["files"]) == {"contcar", "outcar", "vasprun"}


def test_relax_convergence_is_reported_for_phonon_workflows():
    assert _result_has_final_relaxation_stage(PHONON_WORKFLOW_SPEC) is True
    assert _result_has_final_relaxation_stage({"stages": [{"stage_type": "static"}, {"stage_type": "phonon_forces"}]}) is False


# --- 2-4. frequencies, modes, path, signs ------------------------------------------------


def test_frequencies_and_mode_counts_are_read_from_phonopy_output():
    summary, plots = payload()
    bands = summary["band_structure"]
    assert summary["available"] is True
    assert summary["phonopy_version"] == "4.8.1"
    assert summary["symmetry_tolerance"] == 1e-5
    assert bands["branches"] == 6  # 3 x 2 atoms
    assert bands["qpoints"] == 30
    gamma = bands["gamma_frequencies_thz"]
    assert len(gamma) == 6
    assert all(abs(value) < 1e-4 for value in gamma[:3])  # acoustic
    assert gamma[3:] == pytest.approx([15.233209] * 3, abs=1e-6)  # optical, degenerate
    assert bands["max_frequency_thz"] == pytest.approx(16.808794, abs=1e-6)
    traces = plots["phonon_band_structure"]["plot"]["traces"]
    assert len(traces) == 6  # every branch, including degenerate ones


def test_high_symmetry_labels_segments_and_discontinuity():
    summary, plots = payload()
    plot = plots["phonon_band_structure"]["plot"]
    assert plot["ticktext"] == ["Γ", "X", "U|K", "Γ", "L", "W", "X"]
    assert plot["vertical_lines"] == plot["tickvals"]
    assert plot["tickvals"] == sorted(plot["tickvals"])
    assert summary["band_structure"]["path_labels"] == ["Γ-X", "X-U", "K-Γ", "Γ-L", "L-W", "W-X"]
    assert summary["band_structure"]["discontinuities"] == 1
    # 30 q-points plus one gap at the U|K jump, aligned for every branch.
    assert len(plot["x"]) == 31
    gap = plot["x"].index(None)
    assert gap == 10
    assert all(len(trace["y"]) == 31 and trace["y"][gap] is None for trace in plot["traces"])
    assert plot["x"][gap - 1] == plot["x"][gap + 1]  # Phonopy distances are kept as written
    assert plot["yaxis_title"] == "Frequency (THz)"
    assert plot["reference_axis"] == "y" and plot["energy_reference"] == "zero"
    assert "yaxis_range" not in plot  # negative frequencies are never clipped


def _rewrite_band_frequencies(tmp_path: Path, transform) -> Path:
    target = tmp_path / "si"
    shutil.copytree(FIXTURES, target)
    path = target / "phonon_band_structure.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.strip().startswith("frequency:"):
            prefix, value = line.split("frequency:")
            line = f"{prefix}frequency: {transform(float(value)):.10f}"
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


def test_small_negative_acoustic_values_are_kept_and_reported_as_small():
    summary, plots = payload()
    bands = summary["band_structure"]
    assert bands["min_frequency_thz"] < 0  # signed values preserved (about -2e-6 THz)
    assert bands["negative_frequency_values"] > 0
    assert bands["imaginary_status"] == "small"
    assert "numerical" in bands["imaginary_note"]
    assert min(value for trace in plots["phonon_band_structure"]["plot"]["traces"] for value in trace["y"] if value is not None) < 0


def test_substantial_imaginary_modes_are_reported_without_a_stability_verdict(tmp_path):
    # Shift every frequency down by 0.5 THz: acoustic branches become clearly imaginary.
    directory = _rewrite_band_frequencies(tmp_path, lambda value: value - 0.5)
    summary, plots = payload(directory)
    bands = summary["band_structure"]
    assert bands["imaginary_status"] == "significant"
    assert bands["min_frequency_thz"] == pytest.approx(-0.5, abs=1e-3)
    assert bands["significant_imaginary_qpoints"]
    assert bands["significant_imaginary_qpoints"][0]["label"] == "Γ"
    note = bands["imaginary_note"].lower()
    assert "may indicate" in note and "unstable" not in note and "stable" not in note.replace("instability", "")
    lowest = min(v for t in plots["phonon_band_structure"]["plot"]["traces"] for v in t["y"] if v is not None)
    assert lowest == pytest.approx(-0.5, abs=1e-3)
    assert PHONON_IMAGINARY_REPORT_THRESHOLD_THZ == 0.1


def test_no_negative_frequencies(tmp_path):
    directory = _rewrite_band_frequencies(tmp_path, lambda value: abs(value) + 0.01)
    bands = payload(directory)[0]["band_structure"]
    assert bands["imaginary_status"] == "none"
    assert bands["negative_frequency_values"] == 0


# --- 5. DOS -------------------------------------------------------------------------------


def test_phonon_dos_is_parsed_with_units_and_signed_frequencies():
    summary, plots = payload()
    dos = summary["dos"]
    assert dos["points"] == 201
    assert dos["method"] == "Tetrahedron method"
    assert dos["units"] == "states/THz per primitive cell"
    assert dos["integrated_modes"] == pytest.approx(6.0, abs=0.05)  # 3N modes, N = 2
    assert dos["frequency_min_thz"] < 0 < dos["frequency_max_thz"]
    plot = plots["phonon_dos"]["plot"]
    assert plot["xaxis_title"] == "Frequency (THz)"
    assert plot["yaxis_title"] == "DOS (states/THz per primitive cell)"
    assert plot["reference_axis"] == "x" and plot["energy_reference"] == "zero"
    assert len(plot["x"]) == len(plot["traces"][0]["y"]) == 201


# --- 6. plot payload and download ---------------------------------------------------------


def test_visualizations_reuse_the_line_plot_contract_and_png_download():
    _, plots = payload()
    assert list(plots) == ["phonon_band_structure", "phonon_dos"]
    for item in plots.values():
        assert item["kind"] == "line_plot"
        assert item["download_filename"].endswith(".png")
        assert item["element_id"].startswith("workflow-visualization-")
        json.dumps(item, allow_nan=False)  # what the remote parser emits


def test_template_draws_a_zero_reference_for_phonons_and_keeps_the_fermi_reference():
    source = Path("templates/index.html").read_text(encoding="utf-8")
    assert 'plot.energy_reference === "fermi" || plot.energy_reference === "zero"' in source
    assert "results_summary.phonons" in source
    assert "results_summary.files.phonon_band_structure" in source
    # Downloads stay the shared line-plot PNG export.
    assert source.count('data-download-plot="{{ visualization.element_id }}"') == 1


# --- 8. missing or incomplete files --------------------------------------------------------


def test_missing_and_incomplete_phonon_files_are_reported(tmp_path):
    summary, visualizations = _phonon_result_payload({})
    assert summary["available"] is False and visualizations == []
    assert len(summary["problems"]) == 3
    assert any(problem.startswith("phonopy.yaml was not found") for problem in summary["problems"])

    directory = tmp_path / "si"
    shutil.copytree(FIXTURES, directory)
    text = (directory / "phonon_band_structure.yaml").read_text(encoding="utf-8")
    (directory / "phonon_band_structure.yaml").write_text(text[: len(text) // 2], encoding="utf-8")
    summary, plots = payload(directory)
    assert summary["band_structure"]["available"] is False
    assert any("phonon_band_structure.yaml could not be read" in problem for problem in summary["problems"])
    assert summary["dos"]["available"] is True and "phonon_dos" in plots

    (directory / "phonon_dos.yaml").write_text("# Tetrahedron method\n", encoding="utf-8")
    summary, _ = payload(directory)
    assert summary["dos"]["available"] is False


def test_remote_parser_presents_phonons_alongside_relaxation_results(tmp_path):
    paths, _ = result_cases.fixture_result_files(tmp_path)
    paths.update(fixture_paths())
    context = {**result_cases.monitoring_success, "workflow_spec": PHONON_WORKFLOW_SPEC}
    with result_cases.fake_pymatgen_results_modules():
        result = result_cases.execute_remote_parser_source(paths, context)
    assert result["status"] == "success"
    assert result["phonons"]["available"] is True
    ids = [item["id"] for item in result["visualizations"]]
    assert ids == ["phonon_band_structure", "phonon_dos"]
    assert "dos" not in result and "band_structure" not in result  # no electronic payloads


def test_non_finite_band_values_are_rejected_without_losing_the_dos(tmp_path):
    for index, bad in enumerate(("nan", ".inf", "-.inf")):
        directory = tmp_path / f"case_{index}"
        shutil.copytree(FIXTURES, directory)
        band = directory / "phonon_band_structure.yaml"
        band.write_text(
            band.read_text(encoding="utf-8").replace("frequency:   15.2332087512", f"frequency: {bad}", 1),
            encoding="utf-8",
        )
        summary, plots = payload(directory)
        assert summary["band_structure"]["available"] is False
        assert any("non-finite" in problem for problem in summary["problems"])
        assert summary["dos"]["available"] is True and list(plots) == ["phonon_dos"]
        json.dumps(summary, allow_nan=False)
        json.dumps(list(plots.values()), allow_nan=False)


def test_non_finite_dos_values_are_rejected_without_losing_the_dispersion(tmp_path):
    directory = tmp_path / "si"
    shutil.copytree(FIXTURES, directory)
    dos = directory / "phonon_dos.yaml"
    lines = dos.read_text(encoding="utf-8").splitlines()
    lines[5] = "        5.0000000000        nan"
    dos.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary, plots = payload(directory)
    assert summary["dos"]["available"] is False
    assert any("phonon_dos.yaml DOS values contains non-finite" in problem for problem in summary["problems"])
    assert summary["band_structure"]["available"] is True and list(plots) == ["phonon_band_structure"]
    json.dumps(list(plots.values()), allow_nan=False)


def test_unreadable_phonopy_yaml_reports_missing_metadata_without_assuming_values(tmp_path):
    directory = tmp_path / "si"
    shutil.copytree(FIXTURES, directory)
    (directory / "phonopy.yaml").write_bytes(b"\xff\xfe\x00 not utf-8")
    summary, plots = payload(directory)
    assert summary["phonopy_version"] is None
    assert summary["symmetry_tolerance"] is None  # no assumed tolerance
    assert any("phonopy.yaml could not be read: UnicodeDecodeError" in problem for problem in summary["problems"])
    bands = summary["band_structure"]
    # Frequencies and the path are still presented, with unlabeled separators.
    assert bands["available"] is True and bands["labels_resolved"] is False
    # Unlabeled separators; the U|K discontinuity keeps its "|" marker.
    assert plots["phonon_band_structure"]["plot"]["ticktext"] == ["", "", "|", "", "", "", ""]
    assert plots["phonon_band_structure"]["plot"]["tickvals"] == payload()[1]["phonon_band_structure"]["plot"]["tickvals"]
    assert summary["dos"]["available"] is True


def test_remote_parser_keeps_relaxation_results_when_phonon_files_are_corrupt(tmp_path):
    directory = tmp_path / "si"
    shutil.copytree(FIXTURES, directory)
    (directory / "phonopy.yaml").write_bytes(b"\xff\xfe")
    band = directory / "phonon_band_structure.yaml"
    band.write_text(band.read_text(encoding="utf-8").replace("frequency:   15.2332087512", "frequency: nan", 1), encoding="utf-8")
    vasp_dir = tmp_path / "relax"
    vasp_dir.mkdir()
    paths, _ = result_cases.fixture_result_files(vasp_dir)
    paths.update(fixture_paths(directory))
    context = {**result_cases.monitoring_success, "workflow_spec": PHONON_WORKFLOW_SPEC}
    with result_cases.fake_pymatgen_results_modules():
        result = result_cases.execute_remote_parser_source(paths, context)
    assert result["status"] == "success"
    assert result["phonons"]["band_structure"]["available"] is False
    assert result["phonons"]["dos"]["available"] is True
    assert [item["id"] for item in result["visualizations"]] == ["phonon_dos"]
