"""Fake USPEX master with USPEX 25's CalcFold naming, for the validity-probe harness test.

Each individual i is relaxed in steps 1..N, step s in
``Calculation/Calcfold_<i>_<s>`` (as USPEX 25 names them), starting from the
previous step's CONTCAR; the POSCAR's first line is ``number=<i>``. A step
with a non-zero exit status ends that individual's chain, and the individual
is left out of ``results1/Individuals`` (the "rejection" outcome). Up to
--parallel individuals at a time. Writes ``results1/Individuals`` in USPEX
25's column layout with the last step's OSZICAR energy.
"""

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

POSCAR = "number={number}\n1.0\n3 0 0\n0 3 0\n0 0 3\nSi\n1\nDirect\n0 0 0\n"
HEADER = ("generation   number   num_atoms_all    energy   cell_volume   space_group            origin"
          "   parents   e_above_hull   opt_property   quasi_entropy   a_order   s_order")


def individuals_row(number: int, energy: float) -> str:
    """One Individuals row, padded to the header's width as USPEX writes them."""

    row = (f"         1        {number}             [1]   {energy:.3f}       27.000"
           "             1   RandSymFirstGen        []          0.000          0.000"
           "           0.000     1.000     1.000")
    return row.rjust(len(HEADER))


def relax(search: Path, bridge: str, number: int, steps: int):
    previous = None
    for step in range(1, steps + 1):
        folder = search / "Calculation" / f"Calcfold_{number}_{step}"
        folder.mkdir(parents=True)
        (folder / "INCAR").write_text(f"STEP = {step}\nNELM = 60\n")
        if previous is None:
            (folder / "POSCAR").write_text(POSCAR.format(number=number))
        else:
            shutil.copyfile(previous / "CONTCAR", folder / "POSCAR")
        completed = subprocess.run(["sh", bridge, str(step)], cwd=folder)
        if completed.returncode != 0:
            return None
        previous = folder
    last = (previous / "OSZICAR").read_text().splitlines()[-1]
    return float(last.split("F=")[1].split()[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True, type=Path)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--individuals", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--parallel", type=int, required=True)
    args = parser.parse_args()
    numbers = list(range(1, args.individuals + 1))
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        energies = list(pool.map(lambda n: relax(args.search_dir, args.bridge, n, args.steps), numbers))
    results = args.search_dir / "results1"
    results.mkdir()
    rows = [HEADER]
    for number, energy in zip(numbers, energies):
        if energy is None:
            continue
        rows.append(individuals_row(number, energy))
    (results / "Individuals").write_text("\n".join(rows))
    (results / "Parameters.txt").write_text("% atomType\nSi\n% EndAtomType\n")


if __name__ == "__main__":
    main()
