"""Fake ``srun ... vasp_std`` and ``mpirun ... vasp_std`` for the launch-bridge tests.

Installed as ``srun`` and as ``mpirun`` on the PATH of the VASP step
environment; it checks the command line for whichever name it was called by. It checks the
command line the launcher built, records what it saw, and writes the two
files USPEX reads after a VASP relaxation: OSZICAR (last line holds the
energy) and CONTCAR (the relaxed structure; here, POSCAR with a marker).

Environment:
    FAKE_SRUN_RECORDS  directory; one JSON file per call, rewritten at exit
    FAKE_SRUN_CONTROL  JSON file: {"sleep": seconds,
                                   "fail": {"CalcFold2:2": exit_code},
                                   "hang": ["CalcFold1:1"],
                                   "ignore_term": ["CalcFold3:1"]}
"hang" sleeps until SIGTERM; "ignore_term" ignores SIGTERM and needs SIGKILL.
The step number comes from the "STEP = n" line the fake USPEX writes in INCAR.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


def main():
    cwd = Path.cwd()
    argv = sys.argv[1:]
    control = json.loads(Path(os.environ["FAKE_SRUN_CONTROL"]).read_text())
    step = 0
    for line in (cwd / "INCAR").read_text().splitlines():
        if line.startswith("STEP ="):
            step = int(line.split("=")[1])
    key = f"{cwd.name}:{step}"
    record_path = Path(os.environ["FAKE_SRUN_RECORDS"]) / f"{os.getpid()}-{time.time_ns()}.json"
    tool = Path(sys.argv[0]).name
    record = {
        "tool": tool,
        "argv": argv,
        "cwd": str(cwd),
        "key": key,
        "pid": os.getpid(),
        "start": time.time(),
        "end": None,
        "terminated": False,
        "I_MPI_DEBUG": os.environ.get("I_MPI_DEBUG"),
        "I_MPI_PIN_DOMAIN": os.environ.get("I_MPI_PIN_DOMAIN"),
        "I_MPI_PIN_PROCESSOR_LIST": os.environ.get("I_MPI_PIN_PROCESSOR_LIST"),
        "I_MPI_HYDRA_BOOTSTRAP": os.environ.get("I_MPI_HYDRA_BOOTSTRAP"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    }

    def save():
        tmp = record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, record_path)

    def on_term(signum, frame):
        record["end"] = time.time()
        record["terminated"] = True
        save()
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)
    save()

    assert argv[-1] == "vasp_std", argv
    if tool == "srun":
        assert argv[:2] == ["--exact", "--mpi=pmi2"], argv
    else:
        assert argv[0] == "-n", argv

    if key in control.get("ignore_term", []):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    if key in control.get("hang", []):
        while True:
            time.sleep(1)
    time.sleep(control.get("sleep", 0.2))

    code = int(control.get("fail", {}).get(key, 0))
    if code == 0:
        energy = -10.0 * step - len(cwd.name)
        (cwd / "OSZICAR").write_text(
            "       N       E                     dE             d eps       ncg     rms          rms(c)\n"
            f"   1 F= {energy:.8E} E0= {energy:.8E}  d E =-.1E-03\n"
        )
        poscar = (cwd / "POSCAR").read_text().splitlines()
        poscar[0] = f"{poscar[0]} relaxed-step-{step}"
        (cwd / "CONTCAR").write_text("\n".join(poscar) + "\n")
    record["end"] = time.time()
    record["exit"] = code
    save()
    sys.exit(code)


main()
