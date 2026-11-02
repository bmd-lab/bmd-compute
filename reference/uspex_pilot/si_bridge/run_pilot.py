"""Bridge pilot: a tiny USPEX 25 search for Si8 through the launch bridge.

Phase 0 (#59) test of the design in docs/structure_search.md §4, using the
bridge code from #66 (backend/structure_search/bridge/). It checks mechanics,
not methodology: the INCARs in Specific/ are pilot settings, not BMD policy.

Run from si_bridge.sbatch, inside a SLURM job, with the VASP modules loaded.
The VASP steps inherit this process's environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from backend.structure_search.bridge.launcher import (  # noqa: E402
    LauncherSettings,
    install_stop_signal_handlers,
    run_search,
)
from backend.structure_search.bridge.master import (  # noqa: E402
    bridge_script_path,
    build_master_command,
    uspex_command_executable,
)

STEPS = 3


def write_input(search_dir: Path, bridge: Path, num_parallel: int) -> None:
    executable = "\n".join(uspex_command_executable(bridge, STEPS))
    text = f"""PARAMETERS EVOLUTIONARY ALGORITHM
USPEX : calculationMethod
300 : calculationType

% atomType
Si
% EndAtomType

% numSpecies
8
% EndNumSpecies

0.0 : ExternalPressure
8 : populationSize
8 : initialPopSize
4 : numGenerations
3 : stopCrit

% abinitioCode
1 1 1
% ENDabinit

% KresolStart
0.16 0.10 0.07
% Kresolend

% commandExecutable
{executable}
% EndExecutable

0 : whichCluster
{num_parallel} : numParallelCalcs
1 : coresPerJob
"""
    (search_dir / "INPUT.txt").write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path, help="pilot work directory under /bmd")
    parser.add_argument("--singularity", required=True)
    parser.add_argument("--image", required=True, help="SIF or sandbox")
    parser.add_argument("--cache", required=True, help="XDG_CACHE_HOME holding uspex_sfx/")
    parser.add_argument("--potcar", required=True, type=Path, help="Si POTCAR (PBE_64)")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=12)
    parser.add_argument("--mem", default="12G", help="memory per srun step")
    parser.add_argument("--launch-mode", default="srun_steps")
    parser.add_argument("--bridge-timeout", type=int, default=3600)
    args = parser.parse_args()

    work = args.work.resolve()
    search_dir = work / "search"
    spool = work / "spool"
    search_dir.mkdir(parents=True, exist_ok=False)
    specific = search_dir / "Specific"
    shutil.copytree(HERE / "Specific", specific)
    shutil.copyfile(args.potcar, specific / "POTCAR_Si")

    bridge = bridge_script_path()
    if not str(bridge).startswith("/bmd/"):
        # USPEX runs the bridge inside the container, which only sees /bmd.
        raise SystemExit(f"the bmd-compute checkout must be under /bmd (bridge is {bridge})")
    write_input(search_dir, bridge, args.parallel)

    master = build_master_command(
        singularity=args.singularity,
        image=args.image,
        search_dir=search_dir,
        xdg_cache_home=args.cache,
        spool_dir=spool,
        bridge_timeout_seconds=args.bridge_timeout,
    )
    (work / "master_command.txt").write_text(master.shell_line() + "\n")

    settings = LauncherSettings(
        spool_dir=spool,
        search_dir=search_dir,
        max_parallel=args.parallel,
        ranks=args.ranks,
        environment=dict(os.environ),
        launch_mode=args.launch_mode,
        mem=args.mem,
    )
    stop = threading.Event()
    install_stop_signal_handlers(stop)
    status = run_search(
        master.argv,
        master.env,
        master.cwd,
        settings,
        master_log=work / "uspex_master.log",
        stop_event=stop,
    )
    print(f"USPEX master finished with status {status}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
