"""v1.3.1 search plans pinned: requests without the optional random seed and node exclusion.

The values were computed with the released v1.3.1 code (commit 91cc857), the repository's ENMAX metadata and the
default configuration, with the placeholder bridge path below and no generator provenance. A request that does not
use ``random_seed`` or ``excluded_nodes`` must keep exactly these plan digests, resource acknowledgements and
``INPUT.txt`` files. (The two execution-acceptance digests are also pinned in tests/test_uspex_execution_kit.py.)
"""

from __future__ import annotations


BRIDGE = "/bmd/placeholder/bmd_bridge.sh"
SHARED = {"STRUCTURE_SEARCH_PARALLEL_RELAXATIONS": 3, "STRUCTURE_SEARCH_RANKS_PER_RELAXATION": 16,
          "STRUCTURE_SEARCH_EXCLUSIVE_NODE": False, "STRUCTURE_SEARCH_JOB_MEMORY": "32G",
          "STRUCTURE_SEARCH_LAUNCH_MODE": "srun_steps"}
SI8_SPEC = dict(elements=["Si"], counts=[8], min_atoms=8, max_atoms=8, num_generations=20, stop_criterion=8,
                kresol_start=[0.1, 0.08, 0.06, 0.04, 0.04], walltime="48:00:00", population_size=12,
                initial_population_size=24)
# The frozen five-step ladder for Si (Diagnostic A, Experiment E and the Si8 benchmark design).
SI_LADDER_SHA256 = {
    "INCAR_1": "6e4749b2427f50073aab33a5d1e5b19e4cd2222a182b2ddd258ba13ffdab50d5",
    "INCAR_2": "3fdfff7bde3712b3185ae2a21c10b36c5eddfce284e0662eaafc101150238b09",
    "INCAR_3": "48fe4194602f06dc5813d67efb60611315a541d56e532bf9336495648283c166",
    "INCAR_4": "e0c6e6f3fa870ed266dae9f31b813a3734f4a3376f1ceb0ab86b03cef3f2fe38",
    "INCAR_5": "d0c740ca0955a357ad5d2bd6d674b7cb01d156549dc821b388fd313bc447b060",
}

PINS = [
    ("exclusive", "Si", {},
     "sha256:014ef52c3c737e146eb096779de04817357df36decd4a3e70e394e4f74f0ebbc",
     "sha256:3a4d25afc55bc485543b3561567795e13add6a5e81f9319ff08d239a28b6ecaf",
     "eb209b8172ad851d24458ef456b7a9ec68da17cb6f0febe7453411151bacfdc9"),
    ("exclusive", "SiO2", {"max_atoms": 12, "num_generations": 30},
     "sha256:728e2708fbef44b37f914fb44c16f708f98aa589d2097b5b513cde9d1565256f",
     "sha256:3a4d25afc55bc485543b3561567795e13add6a5e81f9319ff08d239a28b6ecaf",
     "b03e356affc3d2fa76be9449684daa95684af4f00dcc9c7174863caec9839587"),
    ("shared", "Si", {"min_atoms": 8, "max_atoms": 8, "num_generations": 30, "walltime": "168:00:00"},
     "sha256:039fcdc7d44be420c4bc7bb3c415a4e4d643db70b006b96ae6a95ff1a907f094",
     "sha256:b2f9585aab7a4588b207f4c0c92e310dea890c55f2c9b993375b09cdc0cf2f28",
     "a763d28ca1da1c0ce47bd00c4070acc5f4367b4d006c7881c9af4ea2e3576b7c"),
    ("shared", "Mg2SiO4", {},
     "sha256:d39132264c32eefeb7d7ee02914cdf9b812d1918726c8f8a4f2d487e5add41c4",
     "sha256:b2f9585aab7a4588b207f4c0c92e310dea890c55f2c9b993375b09cdc0cf2f28",
     "ba2b1d03e5ff049eded83109c34b32aa80fd0195f12e733344609d884104db60"),
]


_PROBE = r"""
import hashlib, json, sys, warnings
warnings.simplefilter("ignore")
from backend import config
from backend.structure_search import job
from backend.structure_search.prediction_plan import AdvancedSettings
from backend.structure_search.search_plan import build_search_plan, build_search_plan_from_formula, render_prepared_files
from backend.structure_search.spec import StructureSearchSpec
cases = json.loads(sys.argv[1])
defaults = {name: getattr(config, name) for name in cases["shared"]}
out = []
for case in cases["plans"]:
    for name, value in (cases["shared"] if case["profile"] == "shared" else defaults).items():
        setattr(config, name, value)
    if case["kind"] == "spec":
        spec = StructureSearchSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in case["spec"].items()})
        plan = build_search_plan(spec, generator={}, **case["extra"])
    else:
        plan = build_search_plan_from_formula(case["formula"], AdvancedSettings(**case["advanced"], **case["extra"]),
                                              generator={})
    files = render_prepared_files(plan, bridge_path=cases["bridge"], generator={})
    out.append([plan.digest, job.resource_request_sha256(plan), hashlib.sha256(files["INPUT.txt"]).hexdigest(),
                {name: entry["sha256"] for name, entry in plan.record["incar_files"].items()}])
print(json.dumps(out))
"""


def _clean_process(plans):
    """Plan in a clean process (no test configuration, no BMD_ environment), as the execution-kit pins do."""

    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    environment = {key: value for key, value in os.environ.items() if not key.startswith("BMD_")}
    cases = json.dumps({"plans": plans, "shared": SHARED, "bridge": BRIDGE})
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", _PROBE, cases], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], env=environment, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


BLANKS = ({}, {"random_seed": None, "excluded_nodes": None}, {"random_seed": "", "excluded_nodes": ""})


def test_v131_plans_are_unchanged_when_the_new_fields_are_omitted_or_blank():
    plans, expected = [], []
    for profile, formula, advanced, digest, acknowledgement, input_sha256 in PINS:
        for extra in BLANKS:
            plans.append({"kind": "formula", "profile": profile, "formula": formula, "advanced": advanced,
                          "extra": extra})
            expected.append([digest, acknowledgement, input_sha256])
    for extra in ({}, {"excluded_nodes": None}, {"excluded_nodes": ""}):
        plans.append({"kind": "spec", "profile": "exclusive", "spec": SI8_SPEC, "extra": extra})
        expected.append(["sha256:02700235296a368b3c5dac6f9bcf53883cd70b34a1081234fdc62b98bbb87dda",
                         "sha256:c40c7fc1c020845b26997a7a85a2a5fa1d012e48b0ca394ea2d47ef34cc63086",
                         "7282f24b597f69af2df2525c9e8e85f00a8237cbec222f4bd6c780d3c9cc264a"])
    found = _clean_process(plans)
    assert [row[:3] for row in found] == expected


def test_the_benchmark_request_keeps_the_frozen_si_ladder():
    advanced = {"min_atoms": 8, "max_atoms": 8, "num_generations": 30, "walltime": "168:00:00"}
    plans = [{"kind": "formula", "profile": "shared", "formula": "Si", "advanced": advanced, "extra": extra}
             for extra in ({}, {"random_seed": 20261126, "excluded_nodes": "compute-0-270"})]
    for row in _clean_process(plans):
        assert row[3] == SI_LADDER_SHA256
