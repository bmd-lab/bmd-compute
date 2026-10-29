"""Fermi reference of non-self-consistent PBE band structures.

A path-only (ICHARG = 11) band structure cannot determine the Fermi energy, so
results use the preceding PBE static's Fermi energy, read with pymatgen's
``Outcar``, whenever it is a finite number. Only a missing, unreadable or
invalid static value keeps the run's own Fermi energy, with the reason recorded.
A path-only run does not establish band filling, so the Fermi-dependent band gap
and metallicity are reported as unavailable (null, with a reason), never as a
number; eigenvalues and the aligned plot are kept. Band structures are
real pymatgen ``BandStructureSymmLine`` objects built from controlled bands.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin

from backend import remote_result_parser, workflow_results
from backend.calculations.models import StageSpec, StageType, Theory, WorkflowSpec
from backend.calculations.registry import desired_output_workflow_spec
from backend.remote_result_parser import PRECEDING_STATIC_OUTCAR_KEY
from backend.results import preceding_static_outcar_path

LATTICE = Lattice.cubic(5.43)
LABELS = {"\\Gamma": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0]}
KPOINTS = np.array([[0.5 * i / 10, 0.0, 0.0] for i in range(11)])
X = np.linspace(0.0, 1.0, len(KPOINTS))
DEEP = -5.0 + 0.0 * X


class PathVasprun:
    """The parts of pymatgen's Vasprun the result parser reads."""

    def __init__(self, bands, efermi, icharg=11):
        self.bands = np.array(bands)
        self.efermi = efermi
        self.incar = {"ICHARG": icharg} if icharg is not None else {}

    def get_band_structure(self, kpoints_filename=None, line_mode=False, efermi=None):
        return BandStructureSymmLine(
            KPOINTS,
            {Spin.up: self.bands},
            LATTICE.reciprocal_lattice,
            self.efermi if efermi is None else efermi,
            LABELS,
        )


@pytest.fixture
def static_outcar(monkeypatch, tmp_path):
    """The preceding static's OUTCAR, with pymatgen's Outcar reporting ``efermi``."""

    path = tmp_path / "OUTCAR"

    def write(efermi):
        path.write_text("static OUTCAR\n", encoding="utf-8")
        from pymatgen.io.vasp import outputs

        monkeypatch.setattr(outputs, "Outcar", lambda _path: type("O", (), {"efermi": efermi})())
        return {PRECEDING_STATIC_OUTCAR_KEY: str(path)}

    return write


def _payload(vasprun, local_paths, module=remote_result_parser):
    summary, visualization = module._band_structure_result_payload(vasprun, {}, dict(local_paths))
    return summary, visualization


# Metal: one band crossing the Fermi level everywhere on the path.
METAL = [DEEP, -1.0 + 2.0 * X]
# Gapped semiconductor: valence maximum 0.0 at Gamma, conduction minimum 1.0.
SEMICONDUCTOR = [DEEP, -1.0 * X, 1.0 + 0.5 * X]
# Semimetal: valence rises to +0.05, conduction falls to -0.05 (overlap).
SEMIMETAL = [DEEP, -0.5 + 0.55 * X, 0.5 - 0.55 * X]
# Very small gap: valence maximum 0.0, conduction minimum 0.02.
SMALL_GAP = [DEEP, -1.0 * X, 0.02 + X]


@pytest.mark.parametrize("module", [remote_result_parser, workflow_results])
def test_metal_uses_the_preceding_static_fermi_energy(static_outcar, module):
    summary, visualization = _payload(PathVasprun(METAL, efermi=0.0), static_outcar(0.3), module)

    assert summary["fermi_level_ev"] == pytest.approx(0.3)
    assert summary["fermi_level_source"] == "preceding_static"
    assert summary["band_structure_run_fermi_level_ev"] == pytest.approx(0.0)
    assert summary["preceding_static_fermi_level_ev"] == pytest.approx(0.3)
    assert summary["fermi_level_fallback_reason"] is None
    _assert_path_properties_withheld(summary)
    # Plotted energies are relative to the reference, not to VASP's path value.
    assert visualization["plot"]["fermi_level_ev"] == pytest.approx(0.3)
    # The crossing band starts at -1.0 eV at Gamma: plotted at -1.3 eV.
    assert any(trace["y"][0] == pytest.approx(-1.3) for trace in visualization["plot"]["traces"])


def _assert_path_properties_withheld(summary):
    assert (summary["band_gap_ev"], summary["is_metal"], summary["direct_gap"]) == (None, None, None)
    assert summary["band_gap_unavailable_reason"] == remote_result_parser.PATH_ONLY_BAND_GAP_REASON


def _gap(bands, efermi):
    return remote_result_parser._band_gap_summary(PathVasprun(bands, efermi=efermi).get_band_structure())


# A metal by the path run's own value whose static reference falls between bands.
METAL_FALSE_GAP = [DEEP, -1.0 + 0.9 * X, 0.1 + 0.9 * X]


@pytest.mark.parametrize(
    ("bands", "run_efermi", "static_efermi"),
    [
        (METAL_FALSE_GAP, -0.5, 0.0),  # path value says metal; static reference says gapped
        (SEMICONDUCTOR, 0.5, 0.0),  # static reference exactly at the valence edge
        (SEMICONDUCTOR, 0.5, -0.03),  # static reference below the path's valence top
        (SEMIMETAL, 0.0, 0.6),  # semimetal; static reference above both overlapping bands
        (SMALL_GAP, 0.01, 0.03),  # narrow gap; static reference inside the conduction band
    ],
    ids=["metal", "insulator-edge", "insulator-below-edge", "semimetal", "narrow-gap"],
)
def test_static_reference_is_authoritative_even_when_pymatgen_classification_changes(
    static_outcar, bands, run_efermi, static_efermi
):
    by_run, by_static = _gap(bands, run_efermi), _gap(bands, static_efermi)
    # Each case is one where the two Fermi values disagree on metallicity or gap.
    assert (by_run["is_metal"], by_run["band_gap_ev"]) != (by_static["is_metal"], by_static["band_gap_ev"])

    summary, visualization = _payload(PathVasprun(bands, efermi=run_efermi), static_outcar(static_efermi))

    assert summary["fermi_level_source"] == "preceding_static"
    assert summary["fermi_level_fallback_reason"] is None
    assert summary["fermi_level_ev"] == pytest.approx(static_efermi)
    assert summary["preceding_static_fermi_level_ev"] == pytest.approx(static_efermi)
    assert summary["band_structure_run_fermi_level_ev"] == pytest.approx(run_efermi)
    # Neither value's Fermi-dependent gap or metallicity is presented as a property.
    _assert_path_properties_withheld(summary)
    assert visualization["plot"]["fermi_level_ev"] == pytest.approx(static_efermi)


@pytest.mark.parametrize("bands", [SEMICONDUCTOR, SEMIMETAL, SMALL_GAP], ids=["insulator", "semimetal", "narrow-gap"])
def test_static_reference_inside_the_path_gap_or_overlap_is_used(static_outcar, bands):
    summary, _ = _payload(PathVasprun(bands, efermi=0.01), static_outcar(0.015))
    assert summary["fermi_level_source"] == "preceding_static"
    assert summary["fermi_level_ev"] == pytest.approx(0.015)
    _assert_path_properties_withheld(summary)


@pytest.mark.parametrize("module", [remote_result_parser, workflow_results])
def test_genuine_gap_with_static_fermi_at_the_vbm_is_not_misreported(static_outcar, module):
    # A real 1.0 eV gap along the path; VASP's static Fermi energy sits exactly
    # at the valence-band maximum (0.0 eV at Gamma). pymatgen would call the
    # VBM state unoccupied and report a wrong gap; no gap number is reported.
    assert _gap(SEMICONDUCTOR, 0.0)["band_gap_ev"] != pytest.approx(1.0)
    summary, visualization = _payload(PathVasprun(SEMICONDUCTOR, efermi=0.5), static_outcar(0.0), module)

    _assert_path_properties_withheld(summary)
    assert summary["fermi_level_ev"] == pytest.approx(0.0)
    assert summary["band_structure_run_fermi_level_ev"] == pytest.approx(0.5)
    assert summary["preceding_static_fermi_level_ev"] == pytest.approx(0.0)
    # Raw eigenvalues are kept and plotted relative to the static reference:
    # the valence top at 0.0 eV and the conduction bottom at 1.0 eV.
    traces = [trace["y"] for trace in visualization["plot"]["traces"]]
    assert any(y[0] == pytest.approx(0.0) and y[-1] == pytest.approx(-1.0) for y in traces)
    assert any(y[0] == pytest.approx(1.0) for y in traces)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "not a number"])
def test_invalid_static_fermi_energy_keeps_the_run_value(static_outcar, value):
    summary, _ = _payload(PathVasprun(METAL, efermi=0.0), static_outcar(value))

    _assert_path_properties_withheld(summary)
    assert summary["fermi_level_source"] == "band_structure_run"
    assert summary["fermi_level_ev"] == pytest.approx(0.0)
    assert summary["preceding_static_fermi_level_ev"] is None
    assert summary["fermi_level_fallback_reason"] in {
        "preceding static Fermi energy is not a finite number",
        "preceding static OUTCAR records no Fermi energy",
    }


def test_missing_unreadable_or_empty_static_outcar_keeps_the_run_value(tmp_path, static_outcar):
    run = PathVasprun(METAL, efermi=0.0)
    garbage = tmp_path / "garbage_OUTCAR"
    garbage.write_text("not an OUTCAR\n", encoding="utf-8")
    cases = {
        "not available": ({}, None),
        "could not be read": ({PRECEDING_STATIC_OUTCAR_KEY: str(garbage)}, None),
    }
    for reason, (paths, _) in cases.items():  # real pymatgen Outcar on unreadable input
        summary, _ = _payload(run, paths)
        assert summary["fermi_level_source"] == "band_structure_run"
        assert summary["fermi_level_ev"] == pytest.approx(0.0)
        assert reason in summary["fermi_level_fallback_reason"]
    _assert_path_properties_withheld(summary)  # missing static evidence: still withheld
    summary, _ = _payload(run, static_outcar(None))
    assert summary["fermi_level_fallback_reason"] == "preceding static OUTCAR records no Fermi energy"
    assert summary["band_structure_run_fermi_level_ev"] == pytest.approx(0.0)


@pytest.mark.parametrize("icharg", [None, 0, 2])
def test_self_consistent_band_structures_are_unchanged(static_outcar, icharg):
    # HSE06 band structures are self-consistent: no reference record, VASP's value.
    vasprun = PathVasprun(METAL, efermi=0.0, icharg=icharg)
    summary, visualization = _payload(vasprun, static_outcar(0.3))

    assert summary["fermi_level_ev"] == pytest.approx(0.0)
    assert not any(
        key.startswith(("fermi_level_source", "band_structure_run", "preceding_static", "band_gap_unavailable"))
        for key in summary
    )
    # Self-consistent runs keep pymatgen's gap and metallicity exactly as before.
    expected = _gap(METAL, 0.0)
    assert (summary["band_gap_ev"], summary["is_metal"], summary["direct_gap"]) == (
        expected["band_gap_ev"],
        expected["is_metal"],
        expected["direct_gap"],
    )
    assert visualization["plot"]["fermi_level_ev"] == pytest.approx(0.0)


def test_both_parsers_agree(static_outcar):
    for bands, run_efermi, static_efermi in [
        (METAL, 0.0, 0.3),
        (METAL_FALSE_GAP, -0.5, 0.0),
        (SEMICONDUCTOR, 0.5, 0.0),
        (SEMIMETAL, 0.0, 0.6),
        (SMALL_GAP, 0.01, 0.03),
    ]:
        paths = static_outcar(static_efermi)
        remote = _payload(PathVasprun(bands, efermi=run_efermi), paths, remote_result_parser)
        local = _payload(PathVasprun(bands, efermi=run_efermi), paths, workflow_results)
        assert remote == local


def test_only_pbe_band_structures_after_a_static_read_the_static_outcar():
    def spec(*stages):
        return WorkflowSpec([StageSpec(stage_type, theory) for stage_type, theory in stages])

    pbe_band = spec((StageType.STATIC, Theory.PBE), (StageType.BAND_STRUCTURE, Theory.PBE))
    relaxed_pbe_band = spec(
        (StageType.RELAX, Theory.PBE), (StageType.STATIC, Theory.PBE), (StageType.BAND_STRUCTURE, Theory.PBE)
    )
    assert preceding_static_outcar_path("/run", pbe_band) == "/run/stage_01/OUTCAR"
    assert preceding_static_outcar_path("/run/", relaxed_pbe_band) == "/run/stage_02/OUTCAR"
    for workflow in (
        desired_output_workflow_spec("electronic_band_structure"),  # HSE06, self-consistent
        desired_output_workflow_spec("electronic_dos"),
        desired_output_workflow_spec("relaxed_energy"),
        spec((StageType.STATIC, Theory.PBE), (StageType.DOS, Theory.PBE)),
        None,
    ):
        assert preceding_static_outcar_path("/run", workflow) is None


# ------------------------------------------------------------- results page


@pytest.mark.parametrize(
    ("band_structure", "shown", "absent"),
    [
        (
            {"available": True, "band_gap_ev": None, "band_gap_unavailable_reason": "path-only reason"},
            'title="path-only reason">Not reported</span>',
            "Unknown",
        ),
        ({"available": True, "band_gap_ev": 0.0}, "0.0 eV", "Not reported"),  # a genuine zero gap
        ({"available": True, "band_gap_ev": None}, "Unknown", "Not reported"),
    ],
    ids=["withheld", "zero-gap", "unknown"],
)
def test_results_page_distinguishes_withheld_gap_from_zero_and_unknown(monkeypatch, band_structure, shown, absent):
    import main
    from test_monitor_route_latency import response_html, resume_request

    monkeypatch.setattr(
        main,
        "monitor_job",
        lambda job_id: {"status": "success", "job_id": job_id, "slurm_state": "COMPLETED", "summary": "SUCCESS"},
    )
    monkeypatch.setattr(
        main,
        "load_results_for_completed_job",
        lambda monitoring_result: {
            "status": "success",
            "final_formula": "Si",
            "visualizations": [],
            "viewer": {},
            "files": {},
            "diagnostics": {},
            "band_structure": band_structure,
        },
    )
    html = response_html(main.resume_existing_calculation(resume_request(), job_id="20791016", load_results="true"))
    gap_callout = html[html.index('<div class="callout-title">Band Gap</div>'):]
    gap_callout = gap_callout[: gap_callout.index("</div>\n                    </div>")]

    assert shown in gap_callout
    assert absent not in gap_callout
