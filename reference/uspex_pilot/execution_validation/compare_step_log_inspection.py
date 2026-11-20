"""Compare a read-only step-log inspection (power_step_log_inspection.sh output) with the archived evidence.

    python compare_step_log_inspection.py INSPECTION_TXT EVIDENCE_DIR [--supplement OUT.json]

EVIDENCE_DIR is the collector's evidence directory (``step_logs.json``, ``files/search/bmd_evidence``).
Every log must appear once in the inspection with the SHA-256 and size that the collector recorded and
that the launcher's archived record names (``step_log``); anything else is reported. Standard library.
Exit status 0 only when the inspection is complete and every hash agrees.

``--supplement`` writes the pinning tables of the hash-verified logs as a pinning supplement for
``analyze_execution.py`` (which checks the hashes again itself), only when every hash agrees. The
original evidence directory is never changed.
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
        "with_pinning_header": sum(1 for r in rows if r.get("pin_header_lines")),
        "with_full_pinning_table_collected": sum(1 for r in rows if r.get("pin_table")),
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


SUPPLEMENT_SCHEMA = "bmd_compute.uspex_pinning_supplement"


def pinning_supplement(inspection: dict, report: dict, *, source_sha256: str) -> dict:
    """The pinning tables of an inspection whose every log hash agrees with the evidence (else refused)."""

    if report["result"] != "HASHES AGREE":
        raise ValueError("the inspection does not agree with the evidence: no supplement")
    missing = sorted(name for name, row in inspection["logs"].items() if "pin_table" not in row)
    if missing:
        raise ValueError(f"the inspection carries no pinning tables (e.g. {missing[:3]}): an older inspection")
    return {"schema": SUPPLEMENT_SCHEMA, "schema_version": 1, "inspection_sha256": source_sha256,
            "logs": {name: {"sha256": row["sha256"], "bytes": row["bytes"], "pin_rows": row["pin_table"],
                            "pin_rows_total": row.get("pin_rows"), "pin_header_lines": row.get("pin_header_lines"),
                            "mpi_startup_lines": row.get("mpi_startup_lines"),
                            "startup_lines": row.get("startup_lines")}
                     for name, row in sorted(inspection["logs"].items())}}


if __name__ == "__main__":
    import hashlib

    raw = Path(sys.argv[1]).read_bytes()
    inspection = read_inspection(raw.decode(errors="replace"))
    report = compare(inspection, Path(sys.argv[2]))
    print(json.dumps(report, indent=1, sort_keys=True))
    if len(sys.argv) == 5 and sys.argv[3] == "--supplement":
        target = Path(sys.argv[4])
        if target.exists():
            raise SystemExit(f"{target} exists: refusing to overwrite")
        try:
            supplement = pinning_supplement(inspection, report, source_sha256=hashlib.sha256(raw).hexdigest())
        except ValueError as exc:
            print(f"SUPPLEMENT REFUSED: {exc}")
            raise SystemExit(1) from None
        target.write_text(json.dumps(supplement, indent=1, sort_keys=True))
        print(f"SUPPLEMENT WRITTEN: {target} ({len(supplement['logs'])} logs)")
    raise SystemExit(0 if report["result"] == "HASHES AGREE" else 1)
