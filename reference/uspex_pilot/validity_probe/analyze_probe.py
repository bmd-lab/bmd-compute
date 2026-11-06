"""Summarise a finished validity probe: what USPEX did with each injected failure.

    /bmd/bmdguest/envs/atomate2_remote/bin/python analyze_probe.py WORK [--out REPORT.json]

Reads only files under WORK (written by run_validity_probe.py, the launcher
and USPEX) and writes one JSON report (default WORK/probe_report.json) plus a
short text summary on stdout. It changes nothing else.

For each injected failure it records the evidence needed to tell apart:

* candidate rejection: the candidate's chain stopped at the failed step, it
  was not run again, and USPEX lists it without a usable energy (absent, or
  a penalty-sized value) - the outcome the bridge relies on;
* bounded retry: the same candidate (same POSCAR or same individual and
  step) was run again, and how many times;
* stale-energy reuse: USPEX lists the candidate with an energy equal to one
  VASP produced earlier for it (an earlier step, or the withheld step itself);
* search termination: the USPEX master's exit status, whether every
  generation appeared, and error lines in the master log.

These are observations for a person to read; the report does not decide
the outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

CALLS_COLUMNS = (
    "started", "finished", "calcfold", "step", "poscar_sha256", "injected",
    "vasp_status", "reported_status", "outcar_toten", "poscar_head",
)
INDIVIDUAL_FILES = ("Individuals", "Individuals_all", "BESTIndividuals")
_CALCFOLD = re.compile(r"[Cc]alc[Ff]old_?(\d+)_(\d+)$")
_NUMBER = re.compile(r"number\s*=\s*(\d+)")
# Individuals energies have 3 decimals (eV per cell); TOTEN has 8.
ENERGY_MATCH_EV = 2e-3
PENALTY_MAGNITUDE_EV = 1e3
NUM_GENERATIONS = 2


def _float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_calls(state: Path) -> list[dict]:
    path = state / "calls.tsv"
    calls = []
    if not path.is_file():
        return calls
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < len(CALLS_COLUMNS) - 1:
            continue
        row = dict(zip(CALLS_COLUMNS, fields + [""] * (len(CALLS_COLUMNS) - len(fields))))
        row["step"] = int(row["step"]) if row["step"].isdigit() else None
        row["outcar_toten"] = _float(row["outcar_toten"])
        match = _CALCFOLD.search(row["calcfold"])
        number = _NUMBER.search(row["poscar_head"])
        row["individual"] = int(match.group(1)) if match else (int(number.group(1)) if number else None)
        calls.append(row)
    return calls


def read_done(spool: Path) -> dict[str, dict]:
    done = {}
    for path in sorted((spool / "done").glob("*.done")) if (spool / "done").is_dir() else ():
        fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        done[path.stem] = fields
    return done


def read_validity(search: Path) -> list[dict]:
    records = []
    archive = search / "bmd_step_validity"
    if archive.is_dir():
        for path in sorted(archive.glob("*/validity.json")):
            records.append(json.loads(path.read_text()))
    return records


def read_individuals(search: Path) -> dict[str, list[dict]]:
    from backend.structure_search.results import find_results_folders, read_individuals_file

    tables: dict[str, list[dict]] = {}
    for folder in find_results_folders(str(search)):
        folder = Path(folder).name
        for name in INDIVIDUAL_FILES:
            warnings: list[str] = []
            try:
                rows = read_individuals_file(str(search / folder / name), folder, warnings)
            except Exception as exc:  # noqa: BLE001 - keep the raw text instead
                tables[f"{folder}/{name}"] = [{"parse_error": f"{type(exc).__name__}: {exc}"}]
                continue
            if rows is None:
                continue
            tables[f"{folder}/{name}"] = [
                {"generation": row.generation, "id": row.id, "origin": row.origin,
                 "energy_ev": row.energy_ev, "parents": list(row.parents or ())}
                for row in rows
            ] + ([{"warnings": warnings}] if warnings else [])
    return tables


def master_log_facts(work: Path) -> dict:
    path = work / "uspex_master.log"
    if not path.is_file():
        return {"present": False}
    lines = path.read_text(errors="replace").splitlines()
    errors = [line for line in lines if re.search(r"Traceback|Error|error|Exception|failed|FAILED", line)]
    return {"present": True, "lines": len(lines), "error_lines": errors[-40:], "tail": lines[-40:]}


def launcher_facts(spool: Path) -> dict:
    path = spool / "launcher.jsonl"
    events = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    finished = [event for event in events if event.get("event") == "master_finished"]
    return {
        "events": len(events),
        "master_finished": finished[-1] if finished else None,
        "validity_events": [event for event in events if event.get("event") == "validity"],
    }


def analyse(work: Path) -> dict:
    search, spool, state = work / "search", work / "spool", work / "probe_state"
    calls = read_calls(state)
    done = read_done(spool)
    validity = read_validity(search)
    individuals = read_individuals(search)
    manifest = json.loads((work / "probe_manifest.json").read_text()) if (work / "probe_manifest.json").is_file() else None

    by_candidate_step = defaultdict(list)
    by_poscar_step = defaultdict(list)
    for call in calls:
        by_candidate_step[(call["individual"], call["step"])].append(call)
        by_poscar_step[(call["poscar_sha256"], call["step"])].append(call)
    repeats = {
        f"individual {key[0]} step {key[1]}": len(group)
        for key, group in by_candidate_step.items() if key[0] is not None and len(group) > 1
    }
    repeats.update({
        f"POSCAR {key[0][:12]} step {key[1]}": len(group)
        for key, group in by_poscar_step.items() if key[0] and len(group) > 1
    })

    listed = defaultdict(list)
    generations = set()
    for table, rows in individuals.items():
        for row in rows:
            if "id" in row:
                listed[row["id"]].append({"table": table, "energy_ev": row["energy_ev"], "origin": row["origin"]})
                generations.add(row["generation"])

    injections = []
    for call in calls:
        if call["injected"] in ("", "none"):
            continue
        candidate = call["individual"]
        same = [c for c in calls if candidate is not None and c["individual"] == candidate]
        later_steps = sorted({c["step"] for c in same if c["step"] and c["step"] > call["step"] and c["started"] >= call["started"]})
        runs_of_failed_step = [c for c in same if c["step"] == call["step"]]
        produced = sorted({c["outcar_toten"] for c in same if c["outcar_toten"] is not None})
        entries = listed.get(candidate, []) if candidate is not None else []
        stale = [
            entry for entry in entries
            if entry["energy_ev"] is not None
            and any(abs(entry["energy_ev"] - energy) <= ENERGY_MATCH_EV for energy in produced)
        ]
        penalty = [e for e in entries if e["energy_ev"] is not None and abs(e["energy_ev"]) >= PENALTY_MAGNITUDE_EV]
        bridge = [
            fields for fields in done.values()
            if fields.get("calcfold", "").rstrip("/").endswith("/" + call["calcfold"])
            and fields.get("step") == str(call["step"])
        ]
        record = [v for v in validity if v.get("calcfold", "").endswith("/" + call["calcfold"]) and v.get("step") == call["step"]]
        injections.append({
            "injection": call["injected"],
            "calcfold": call["calcfold"],
            "individual": candidate,
            "step": call["step"],
            "vasp_status": call["vasp_status"],
            "reported_status": call["reported_status"],
            "bridge_status_to_uspex": [fields.get("status") for fields in bridge],
            "bridge_validity": [fields.get("validity") for fields in bridge],
            "validity_records": [{"classification": v.get("classification"), "reasons": v.get("reasons"),
                                  "files": [f.get("name") for f in v.get("files", [])]} for v in record],
            "runs_of_this_step": len(runs_of_failed_step),
            "later_steps_run": later_steps,
            "energies_vasp_produced_for_candidate": produced,
            "uspex_listing": entries,
            "listed_energy_matches_a_vasp_energy": bool(stale),
            "listed_with_penalty_sized_energy": bool(penalty),
            "observations": {
                "chain_stopped_at_failed_step": not later_steps,
                "retried": len(runs_of_failed_step) > 1,
                "listed_by_uspex": bool(entries),
                "listed_with_usable_energy": any(
                    e["energy_ev"] is not None and abs(e["energy_ev"]) < PENALTY_MAGNITUDE_EV for e in entries
                ),
                "stale_energy_reuse": bool(stale),
            },
        })

    launcher = launcher_facts(spool)
    master_log = master_log_facts(work)
    return {
        "work": str(work),
        "manifest": manifest,
        "runtime_check_status": (json.loads((work / "runtime_check.json").read_text()).get("status")
                                 if (work / "runtime_check.json").is_file() else None),
        "calls": len(calls),
        "injections": injections,
        "repeated_runs": repeats,
        "done_statuses": sorted({fields.get("status", "?") for fields in done.values()}),
        "validity_archive": [{"id": v.get("id"), "step": v.get("step"), "classification": v.get("classification"),
                              "reasons": v.get("reasons")} for v in validity],
        "individual_tables": individuals,
        "search": {
            "master_status": (manifest or {}).get("master_status"),
            "master_finished_event": launcher["master_finished"],
            "generations_listed": sorted(generations),
            "all_generations_listed": set(range(1, NUM_GENERATIONS + 1)) <= generations,
            "master_log": master_log,
        },
        "isolation": {
            "jobflow_imported": (manifest or {}).get("jobflow_imported"),
            "atomate2_imported": (manifest or {}).get("atomate2_imported"),
        },
    }


def summary_lines(report: dict) -> list[str]:
    lines = [f"probe: {report['work']}", f"VASP calls: {report['calls']}",
             f"runtime check: {report['runtime_check_status']}",
             f"master status: {report['search']['master_status']}, generations listed: "
             f"{report['search']['generations_listed']}"]
    for item in report["injections"]:
        o = item["observations"]
        lines.append(
            f"- {item['injection']}: {item['calcfold']} (individual {item['individual']}, step {item['step']}), "
            f"VASP {item['vasp_status']}, wrapper {item['reported_status']}, bridge to USPEX "
            f"{item['bridge_status_to_uspex']} {item['bridge_validity']}; "
            f"chain stopped: {o['chain_stopped_at_failed_step']}; retried: {o['retried']} "
            f"({item['runs_of_this_step']} runs); listed: {o['listed_by_uspex']}; usable energy listed: "
            f"{o['listed_with_usable_energy']}; stale energy: {o['stale_energy_reuse']}"
        )
    if report["repeated_runs"]:
        lines.append(f"repeated runs: {report['repeated_runs']}")
    lines.append(f"isolation: {report['isolation']}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    work = args.work.resolve()
    if not (work / "probe_state").is_dir():
        raise SystemExit(f"{work} is not a validity probe directory (no probe_state/)")
    report = analyse(work)
    out = args.out or work / "probe_report.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print("\n".join(summary_lines(report)))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
