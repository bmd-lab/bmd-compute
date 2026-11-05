"""Fake USPEX master for the launch-bridge tests.

Behaves like USPEX 25 with whichCluster = 0 as far as the bridge can tell:
it creates CalcFold<i> directories, and for each structure runs the
relaxation steps in order, calling ``sh bmd_bridge.sh <step>`` in the
CalcFold (the commandExecutable) and waiting for it. Up to --parallel
structures are relaxed at once (numParallelCalcs). After each step it reads
OSZICAR and CONTCAR, and copies CONTCAR to POSCAR for the next step, as
USPEX does. A step with a non-zero exit status ends that structure's chain.

Writes <search dir>/fake_uspex_results.json:
    {"CalcFold1": {"exit_codes": [0, 0], "energies": [...], "final_poscar": "..."}}
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


POSCAR = """{name}
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Si
1
Direct
0.0 0.0 0.0
"""


def relax(folder: Path, bridge: str, steps: int) -> dict:
    result = {"exit_codes": [], "energies": [], "final_poscar": None}
    for step in range(1, steps + 1):
        (folder / "INCAR").write_text(f"STEP = {step}\n")
        for name in ("OSZICAR", "CONTCAR"):
            (folder / name).unlink(missing_ok=True)
        completed = subprocess.run(["sh", bridge, str(step)], cwd=folder)
        result["exit_codes"].append(completed.returncode)
        if completed.returncode != 0:
            break
        last = (folder / "OSZICAR").read_text().splitlines()[-1]
        result["energies"].append(float(last.split("E0=")[1].split()[0]))
        (folder / "POSCAR").write_text((folder / "CONTCAR").read_text())
    result["final_poscar"] = (folder / "POSCAR").read_text().splitlines()[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True, type=Path)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--structures", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--parallel", type=int, required=True)
    args = parser.parse_args()

    folders = []
    for index in range(1, args.structures + 1):
        folder = args.search_dir / f"CalcFold{index}"
        folder.mkdir(exist_ok=True)
        (folder / "POSCAR").write_text(POSCAR.format(name=folder.name))
        folders.append(folder)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = dict(
            zip(
                (folder.name for folder in folders),
                pool.map(lambda folder: relax(folder, args.bridge, args.steps), folders),
            )
        )

    out = args.search_dir / "fake_uspex_results.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1, sort_keys=True))
    os.replace(tmp, out)


main()
