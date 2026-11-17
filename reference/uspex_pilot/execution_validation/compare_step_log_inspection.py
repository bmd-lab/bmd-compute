"""Compare a read-only step-log inspection (power_step_log_inspection.sh output) with the archived evidence.

    python compare_step_log_inspection.py INSPECTION_TXT EVIDENCE_DIR

EVIDENCE_DIR is the collector's evidence directory (``step_logs.json``, ``files/search/bmd_evidence``).
Every log must appear once in the inspection with the SHA-256 and size that the collector recorded and
that the launcher's archived record names (``step_log``); anything else is reported. Standard library.
Exit status 0 only when the inspection is complete and every hash agrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_inspection(text: str) -> dict:
    logs, problems, summary, complete = {}, [], None, False
    for line in text.splitlines():
        if line.startswith("== INSPECTION COMPLETE"):
            complete = True
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append({"problem": "unreadable output line", "line": line[:200]})
            continue
        if "log" in row:
            if row["log"] in logs:
                problems.append({"problem": "log listed twice", "name": row["log"]})
            logs[row["log"]] = row
        elif "summary" in row:
            summary = row["summary"]
        elif "problem" in row:
            problems.append(row)
    return {"logs": logs, "problems": problems, "summary": summary, "complete": complete}


def compare(inspection: dict, evidence: Path) -> dict:
    collected = json.loads((evidence / "step_logs.json").read_text())
    archived = {}
    steps = evidence / "files" / "search" / "bmd_evidence" / "steps"
    for record in sorted(steps.glob("*/record.json")) if steps.is_dir() else []:
        try:
            archived[record.parent.name] = (json.loads(record.read_text()).get("step_log") or {}).get("sha256")
        except ValueError:
            archived[record.parent.name] = None
    seen = inspection["logs"]
    matched, mismatched = [], []
    for name, summary in sorted(collected.items()):
        row = seen.get(name)
        if row is None:
            continue
        agree = row["sha256"] == summary["sha256"] and row["bytes"] == summary["bytes"]
        if name in archived:
            agree = agree and archived[name] == row["sha256"]
        (matched if agree else mismatched).append(name)
    absent = sorted(set(collected) - set(seen))
    extra = sorted(set(seen) - set(collected))
    not_archived = sorted(set(collected) - set(archived))
    rows = list(seen.values())
    placement = {
        "logs": len(rows),
        "with_intel_mpi_version_line": sum(1 for r in rows if r.get("intel_mpi_version")),
        "with_mpi_startup_lines": sum(1 for r in rows if r.get("mpi_startup_lines")),
        "with_pinning_rows": sum(1 for r in rows if r.get("pin_rows")),
        "with_slurm_cpu_bind_lines": sum(1 for r in rows if r.get("cpu_bind_lines")),
        "versions": sorted({r["intel_mpi_version"] for r in rows if r.get("intel_mpi_version")})[:5],
        "samples": [s for r in rows[:3] for s in r.get("samples") or []][:18],
    }
    ok = (inspection["complete"] and not inspection["problems"] and not mismatched and not absent and not extra
          and len(matched) == len(collected))
    return {"result": "HASHES AGREE" if ok else "NOT VERIFIED", "inspection_complete": inspection["complete"],
            "inspection_problems": inspection["problems"][:20], "inspection_summary": inspection["summary"],
            "collected_logs": len(collected), "matched": len(matched), "mismatched": mismatched[:20],
            "absent_from_inspection": absent[:20], "not_in_evidence": extra[:20],
            "collected_but_not_archived": not_archived[:20], "placement": placement}


if __name__ == "__main__":
    report = compare(read_inspection(Path(sys.argv[1]).read_text(errors="replace")), Path(sys.argv[2]))
    print(json.dumps(report, indent=1, sort_keys=True))
    raise SystemExit(0 if report["result"] == "HASHES AGREE" else 1)
