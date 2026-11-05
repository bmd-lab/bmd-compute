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


def write_input(search_dir: Path, bridge: Path, num_parallel: int, which_cluster: int = 0) -> None:
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

{which_cluster} : whichCluster
{num_parallel} : numParallelCalcs
1 : coresPerJob
"""
    (search_dir / "INPUT.txt").write_text(text)


def write_bridge_wrapper(search_dir: Path, bridge: Path, spool: Path, timeout: int) -> Path:
    """Pilot-only wrapper that USPEX runs instead of the bridge itself.

    The first pilot run showed every relaxation command failing within a
    second, before the bridge wrote a request, and USPEX keeps the command's
    output to itself. The wrapper sets the bridge settings explicitly (in case
    USPEX starts commands with a reduced environment) and records its
    arguments, environment and the bridge's error output in
    bmd_bridge_debug.log in each CalcFold.
    """

    wrapper = search_dir / "bmd_bridge_pilot.sh"
    wrapper.write_text(
        f"""#!/bin/sh
log="$(pwd)/bmd_bridge_debug.log"
{{
  echo "== start $(date +%T) pid $$ args: $*"
  echo "pwd: $(pwd)  id: $(id)"
  env | sort
}} >> "$log" 2>&1
export BMD_BRIDGE_SPOOL={spool}
export BMD_BRIDGE_TIMEOUT_SECONDS={timeout}
export BMD_BRIDGE_POLL_SECONDS=2
export BMD_BRIDGE_HEARTBEAT_SECONDS=300
sh {bridge} "$@" 2>> "$log"
status=$?
echo "== exit $status $(date +%T)" >> "$log"
exit "$status"
"""
    )
    wrapper.chmod(0o755)
    return wrapper


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
    parser.add_argument("--which-cluster", type=int, choices=(-1, 0), default=-1,
                        help="USPEX whichCluster: 0 = local with core control, -1 = local without")
    parser.add_argument("--resume", metavar="TAG",
                        help="rerun USPEX in an existing WORK/search (restart test): keep INPUT.txt "
                             "and Specific/, use a fresh spool WORK/spool_TAG and log uspex_master_TAG.log")
    parser.add_argument("--pick-up-gen", type=int,
                        help="with --resume: write 'N : pickUpGen' into INPUT.txt (manual section 4.7)")
    parser.add_argument("--pick-up-folder", type=int,
                        help="with --resume: write 'N : pickUpFolder' into INPUT.txt (results<N>/)")
    args = parser.parse_args()

    work = args.work.resolve()
    search_dir = work / "search"
    if args.resume:
        # USPEX continues a search when it is rerun in the same folder (manual
        # section 4.7). Only the bridge spool is new, as for a resumed attempt.
        if not (search_dir / "INPUT.txt").is_file():
            raise SystemExit(f"--resume needs an existing search in {search_dir}")
        spool = work / f"spool_{args.resume}"
        if spool.exists():
            raise SystemExit(f"{spool} already exists; choose a new --resume tag")
        master_log = work / f"uspex_master_{args.resume}.log"
    else:
        spool = work / "spool"
        master_log = work / "uspex_master.log"
        search_dir.mkdir(parents=True, exist_ok=False)
        specific = search_dir / "Specific"
        shutil.copytree(HERE / "Specific", specific)
        shutil.copyfile(args.potcar, specific / "POTCAR_Si")

    bridge = bridge_script_path()
    if not str(bridge).startswith("/bmd/"):
        # USPEX runs the bridge inside the container, which only sees /bmd.
        raise SystemExit(f"the bmd-compute checkout must be under /bmd (bridge is {bridge})")
    # On resume the wrapper is rewritten in place (INPUT.txt refers to its
    # path), so it points at the new spool; INPUT.txt itself is not touched.
    wrapper = write_bridge_wrapper(search_dir, bridge, spool, args.bridge_timeout)
    if not args.resume:
        write_input(search_dir, wrapper, args.parallel, args.which_cluster)
    elif args.pick_up_gen is not None or args.pick_up_folder is not None:
        # A plain rerun in the same folder started a new search (restart test,
        # 5 Nov 2026), so ask USPEX to pick up from a given generation and
        # results folder instead. Earlier pickUp lines are replaced.
        input_path = search_dir / "INPUT.txt"
        lines = [line for line in input_path.read_text().splitlines()
                 if not line.strip().endswith((": pickUpGen", ": pickUpFolder"))]
        if args.pick_up_gen is not None:
            lines.append(f"{args.pick_up_gen} : pickUpGen")
        if args.pick_up_folder is not None:
            lines.append(f"{args.pick_up_folder} : pickUpFolder")
        input_path.write_text("\n".join(lines) + "\n")

    master = build_master_command(
        singularity=args.singularity,
        image=args.image,
        search_dir=search_dir,
        xdg_cache_home=args.cache,
        spool_dir=spool,
        bridge_timeout_seconds=args.bridge_timeout,
    )
    # USPEX prints non-ASCII (Russian) messages; with whichCluster -1 it
    # crashed with UnicodeEncodeError under the ASCII locale that LC_ALL=C and
    # --cleanenv give the container. Give it a UTF-8 locale (C.UTF-8 is built
    # into glibc 2.35 in the Ubuntu 22.04 image).
    master_env = dict(master.env)
    master_env.update({
        "LC_ALL": "C.UTF-8",
        "SINGULARITYENV_LC_ALL": "C.UTF-8",
        "SINGULARITYENV_LANG": "C.UTF-8",
        "SINGULARITYENV_PYTHONIOENCODING": "utf-8",
    })
    (work / ("master_command.txt" if not args.resume else f"master_command_{args.resume}.txt")).write_text(
        master.shell_line() + "\n# plus, set by the pilot: "
        + " ".join(f"{k}={v}" for k, v in sorted(master_env.items()) if k not in master.env or master.env[k] != v)
        + "\n"
    )

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
        master_env,
        master.cwd,
        settings,
        master_log=master_log,
        stop_event=stop,
    )
    print(f"USPEX master finished with status {status}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
