"""USPEX INPUT.txt writer: golden files, fixed execution settings and refusals."""

from pathlib import Path

import pytest

from backend.structure_search.admission import StructureSearchAdmissionError
from backend.structure_search.input_writer import (
    INPUT_FILENAME,
    SearchExecutionSettings,
    SearchInputError,
    command_executable_lines,
    read_input_keywords,
    render_input_text,
    write_input_file,
)
from backend.structure_search.spec import StructureSearchSpec


FIXTURES = Path(__file__).parent / "fixtures" / "structure_search"
# Placeholder: the real bridge path is defined by the launch bridge (#66).
BRIDGE = "/bmd/placeholder/bmd_bridge.sh"
SETTINGS = SearchExecutionSettings(bridge_path=BRIDGE, num_parallel_calcs=4, sleep_seconds=30)

# Every keyword bmd-compute relies on; none may be left to a USPEX default.
REQUIRED_KEYWORDS = {
    "calculationType", "atomType", "numSpecies", "ExternalPressure", "minAt", "maxAt",
    "populationSize", "initialPopSize", "numGenerations", "stopCrit", "abinitioCode",
    "KresolStart", "commandExecutable", "whichCluster", "numParallelCalcs",
    "coresPerJob", "sleepSeconds",
}


def _mg2sio4():
    return StructureSearchSpec(
        elements=("Mg", "Si", "O"), counts=(2, 1, 4), min_atoms=7, max_atoms=14,
        num_generations=40, stop_criterion=15,
        kresol_start=(0.12, 0.10, 0.08, 0.06, 0.04), walltime="120:00:00",
        external_pressure_gpa=10,
    )


def _si8():
    return StructureSearchSpec(
        elements=("Si",), counts=(8,), min_atoms=8, max_atoms=8,
        num_generations=20, stop_criterion=8, kresol_start=(0.1, 0.08, 0.06, 0.04),
        walltime="48:00:00", population_size=12, initial_population_size=24,
    )


@pytest.mark.parametrize(
    ("spec", "golden"),
    [(_mg2sio4, "INPUT_Mg2SiO4_10GPa.txt"), (_si8, "INPUT_Si8_0GPa.txt")],
)
def test_input_matches_golden_file(spec, golden):
    assert render_input_text(spec(), SETTINGS) == (FIXTURES / golden).read_text(encoding="utf-8")


def test_input_is_complete_and_uses_fixed_execution_settings():
    spec = _mg2sio4()
    keywords = read_input_keywords(render_input_text(spec, SETTINGS))
    assert set(keywords) == REQUIRED_KEYWORDS
    assert keywords["calculationType"] == ["300"]
    assert keywords["whichCluster"] == ["0"]
    assert keywords["coresPerJob"] == ["1"]
    assert keywords["numParallelCalcs"] == ["4"]
    assert keywords["abinitioCode"] == ["1 1 1 1 1"]
    assert keywords["commandExecutable"] == [f"sh {BRIDGE} {n}" for n in range(1, 6)]
    assert keywords["atomType"] == ["Mg Si O"]
    assert keywords["numSpecies"] == ["2 1 4"]
    assert keywords["ExternalPressure"] == ["10"]
    assert keywords["populationSize"] == keywords["initialPopSize"] == ["30"]


def test_one_entry_per_step_for_every_per_step_keyword():
    for steps in (1, 3, 6):
        spec = StructureSearchSpec(
            elements=("C",), counts=(4,), min_atoms=4, max_atoms=8, num_generations=5,
            stop_criterion=3, kresol_start=tuple(0.1 - 0.01 * i for i in range(steps)),
            walltime="1:00:00",
        )
        keywords = read_input_keywords(render_input_text(spec, SETTINGS))
        assert len(keywords["abinitioCode"][0].split()) == steps
        assert len(keywords["KresolStart"][0].split()) == steps
        assert len(keywords["commandExecutable"]) == steps


def test_block_markers_are_the_manual_spellings():
    text = render_input_text(_si8(), SETTINGS)
    for opening, closing in [
        ("atomType", "EndAtomType"), ("numSpecies", "EndNumSpecies"),
        ("abinitioCode", "ENDabinit"), ("KresolStart", "Kresolend"),
        ("commandExecutable", "EndExecutable"),
    ]:
        lines = text.splitlines()
        assert lines.count(f"% {opening}") == 1
        assert lines.index(f"% {closing}") > lines.index(f"% {opening}")


def test_pressure_only_as_external_pressure():
    text = render_input_text(_mg2sio4(), SETTINGS)
    assert "PSTRESS" not in text.upper()
    assert "10 : ExternalPressure" in text.splitlines()


def test_inadmissible_spec_is_never_written(tmp_path):
    spec = StructureSearchSpec(
        elements=("Fe", "O"), counts=(2, 3), min_atoms=5, max_atoms=10, num_generations=5,
        stop_criterion=3, kresol_start=(0.1,), walltime="1:00:00",
    )
    with pytest.raises(StructureSearchAdmissionError):
        write_input_file(tmp_path, spec, SETTINGS)
    assert not (tmp_path / INPUT_FILENAME).exists()


def test_write_input_file(tmp_path):
    path = write_input_file(tmp_path, _si8(), SETTINGS)
    assert path == tmp_path / "INPUT.txt"
    assert path.read_text(encoding="utf-8") == (FIXTURES / "INPUT_Si8_0GPa.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "bridge_path",
    ["", "relative/bridge.sh", "/bmd/with space.sh", "/bmd/x.sh; rm -rf /", "/bmd/$HOME.sh", "/bmd/../etc/x.sh"],
)
def test_unsafe_bridge_paths_rejected(bridge_path):
    with pytest.raises(SearchInputError) as excinfo:
        SearchExecutionSettings(bridge_path=bridge_path, num_parallel_calcs=4, sleep_seconds=30)
    assert excinfo.value.code == "invalid_bridge_path"


@pytest.mark.parametrize("field", ["num_parallel_calcs", "sleep_seconds"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, "4", None])
def test_execution_settings_must_be_explicit_positive_integers(field, value):
    values = {"bridge_path": BRIDGE, "num_parallel_calcs": 4, "sleep_seconds": 30, field: value}
    with pytest.raises(SearchInputError):
        SearchExecutionSettings(**values)


def test_command_executable_lines():
    assert command_executable_lines("/b/x.sh", 2) == ["sh /b/x.sh 1", "sh /b/x.sh 2"]


def test_reader_rejects_malformed_text():
    with pytest.raises(ValueError):
        read_input_keywords("% atomType\nSi\n")
    with pytest.raises(ValueError):
        read_input_keywords("1 : minAt\n2 : minAt\n")
    with pytest.raises(ValueError):
        read_input_keywords("no separator here\n")
