"""Local analysis of the POWER evidence of one managed structure search (scientific execution acceptance).

Reads only the directory ``power_execution_evidence.sh`` wrote (fetched to the
VM) and the expectations taken from the validation's own ledger, and returns
a report that keeps three things apart:

* **checks**, each ``verified`` (the evidence shows it), ``failed`` (the
  evidence contradicts it) or ``missing`` (the evidence needed is absent);
* the **classification** of how the job ended (``natural_completion``,
  ``stopped_by_walltime`` or ``failed_or_incomplete``), never from SLURM's
  ``COMPLETED`` alone;
* **scientific findings** (candidates, rejections, energies, the best
  structure and whether diamond-like symmetry appears), which are recorded,
  not required.

Only a candidate whose five ladder steps all ran once, in order, and were all
judged ``valid`` may have a usable final energy. Each candidate USPEX lists
must be associated, from evidence (energy, the recorded structure against
surviving CONTCARs, and the CalcFold numbering only where the survivors
confirm it), with exactly one such chain, at its step-5 free energy F as VASP
printed it (independent of BMD's own checker), after the listed quantity (per
cell or per atom) has been established from USPEX's own tables. Insufficient
evidence for an association or for the quantity is ``missing``; evidence
that contradicts one is ``failed``.

The launcher's evidence archive (``search/bmd_evidence/``: each step's
original verdict and its input and output structures, committed before USPEX
could remove the CalcFold) is verified against the journal, done files, VASP's
step logs and the surviving CalcFolds; its step-5 CONTCARs are the recorded
structures of every accepted chain, in every generation. The result is ``PASSED`` only when the job completed
naturally, no check failed and no required evidence is missing.

This module uses the standard library; pymatgen, if importable, is used only
for the symmetry of the recorded structures.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

VERIFIED, FAILED, MISSING = "verified", "failed", "missing"
CLASSIFICATIONS = ("valid", "invalid_convergence", "execution_failed", "invalid_evidence")
DONE_STATUS = {"valid": {0}, "invalid_convergence": {118}, "invalid_evidence": {119}}
INDIVIDUALS_TOLERANCE_EV = 0.0015  # Individuals prints energies with 3 decimals
GATHERED_TOLERANCE_EV = 1e-4  # gatheredPOSCARS prints them with 6
MEMORY_MARGIN_FRACTION = 0.10  # the launcher's default (backend.structure_search.bridge.launcher)
_CALCFOLD = re.compile(r"Calcfold_(\d+)_(\d+)$")


class Report:
    def __init__(self):
        self.checks: list[dict[str, Any]] = []

    def add(self, check: str, what: str, status: str, detail: Any = None) -> None:
        self.checks.append({"check": check, "what": what, "status": status, "detail": detail})


# -- reading the evidence ------------------------------------------------------------------------------------


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError):
        return None


def _json(path: Path):
    text = _text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line in (_text(path) or "").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _key_values(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _tokens(line: str) -> list[str]:
    """Whitespace columns, with a bracketed list (``[3, 4]``, ``[2]``) kept as one column."""

    tokens, depth, current = [], 0, ""
    for part in line.split():
        current = f"{current} {part}" if current else part
        depth += part.count("[") - part.count("]")
        if depth <= 0:
            tokens.append(current)
            current, depth = "", 0
    if current:
        tokens.append(current)
    return tokens


def _atoms(value: str | None) -> int | None:
    match = re.fullmatch(r"\[\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\]", str(value or "").strip())
    return sum(int(v) for v in match.group(1).split(",")) if match else None


def read_table(text: str | None) -> list[dict[str, Any]] | None:
    """A USPEX ``Individuals``-type table as rows of strings keyed by header; None if not a table."""

    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    header = lines[0].split()
    if not {"generation", "number"} <= set(header):
        return None
    rows = []
    for line in lines[1:]:
        values = _tokens(line)
        if len(values) != len(header):
            return None
        rows.append(dict(zip(header, values)))
    return rows


def _float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def read_individuals(text: str | None) -> list[dict[str, Any]] | None:
    """``Individuals``: generation, number, the listed energy, atoms per cell; None if unreadable."""

    table = read_table(text)
    if table is None or not table or "energy" not in table[0]:
        return None if table is None or (table and "energy" not in table[0]) else []
    rows = []
    for row in table:
        try:
            rows.append({"generation": int(row["generation"]), "number": int(row["number"]),
                         "energy": float(row["energy"]), "natoms": _atoms(row.get("num_atoms_all")),
                         "origin": row.get("origin")})
        except (TypeError, ValueError):
            return None
    return rows


def read_gathered_poscars(text: str | None) -> dict[int, dict[str, Any]]:
    """``gatheredPOSCARS``: per individual number, the energy USPEX recorded and the structure text."""

    blocks: dict[int, dict[str, Any]] = {}
    lines = (text or "").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("number=")]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        header = dict(part.strip().split("=", 1) for part in lines[start].split(",") if "=" in part)
        try:
            number = int(header["number"])
            energy = float(header["energy"])
        except (KeyError, ValueError):
            continue
        blocks[number] = {"energy": energy, "poscar": "\n".join(lines[start:end]) + "\n"}
    return blocks


def parse_poscar(text: str | None) -> dict[str, Any] | None:
    """Lattice (Å, scaled), species counts and fractional coordinates of a VASP 5 POSCAR/CONTCAR; None if unreadable."""

    lines = [line.strip() for line in (text or "").splitlines()]
    try:
        scale = float(lines[1].split()[0])
        lattice = [[scale * float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)]
        species = lines[5].split()
        counts = [int(v) for v in lines[6].split()]
        mode = lines[7].lower()
        if not mode.startswith("d"):
            return None
        coords = [[float(v) for v in lines[8 + i].split()[:3]] for i in range(sum(counts))]
    except (IndexError, ValueError):
        return None
    if scale <= 0 or len(species) != len(counts):
        return None
    return {"lattice": lattice, "species": species, "counts": counts, "coords": coords}


def same_structure(a: dict | None, b: dict | None, *, lattice_tol: float = 1e-3, coord_tol: float = 1e-3) -> bool | None:
    """True if two parsed structures are the same cell with the same atoms (as USPEX copies a CONTCAR); None if unknown."""

    if a is None or b is None:
        return None
    if a["species"] != b["species"] or a["counts"] != b["counts"]:
        return False
    if any(abs(x - y) > lattice_tol for row_a, row_b in zip(a["lattice"], b["lattice"]) for x, y in zip(row_a, row_b)):
        return False

    def wrapped(x):
        return x - round(x)

    unused = list(range(len(b["coords"])))
    for coord in a["coords"]:
        match = next((j for j in unused if all(abs(wrapped(p - q)) <= coord_tol for p, q in zip(coord, b["coords"][j]))),
                     None)
        if match is None:
            return False
        unused.remove(match)
    return True


def energy_semantics(individuals: list[dict], tables: dict[str, list[dict] | None], gathered: dict[int, dict],
                     *, cell_atoms: int, pressure_gpa: float) -> dict[str, Any]:
    """What USPEX's listed ``energy`` is: per cell or per atom, from its own tables; never assumed.

    USPEX 25 writes ``energy`` (per cell, eV; enthalpy at the run's pressure)
    beside ``energy_per_atom`` in ``Individuals_all``/``Individuals_extended``
    (si_bridge sample and validity probe). The quantity is established only if
    those tables show ``energy`` = ``energy_per_atom`` × atoms (per cell) or
    ``energy`` = ``energy_per_atom`` (per atom) for every numeric row, the
    atom counts agree with the plan's cell size and the recorded structures,
    the pressure is zero (enthalpy = energy) and ``Individuals`` agrees with
    those tables.
    """

    result: dict[str, Any] = {"quantity": None, "basis": [], "problems": []}
    if pressure_gpa != 0:
        result["problems"].append(f"external pressure {pressure_gpa} GPa: USPEX lists enthalpy, not F")
        return {**result, "status": MISSING}
    per_cell = per_atom = 0
    reference: dict[int, float] = {}
    for name, table in tables.items():
        for row in table or []:
            energy, per, atoms = _float(row.get("energy")), _float(row.get("energy_per_atom")), _atoms(
                row.get("num_atoms_all"))
            if energy is None or per is None or atoms is None:
                continue
            if atoms != cell_atoms:
                result["problems"].append(f"{name}: individual {row.get('number')} has {atoms} atoms, not {cell_atoms}")
            tolerance = 5e-5 * atoms + 1e-4
            cell_like = abs(energy - per * atoms) <= tolerance
            atom_like = abs(energy - per) <= 1e-4
            if cell_like and not atom_like:
                per_cell += 1
            elif atom_like and not cell_like:
                per_atom += 1
            elif not cell_like and not atom_like:
                result["problems"].append(f"{name}: individual {row.get('number')}: energy {energy} is neither "
                                          f"{per} x {atoms} nor {per}")
            try:
                reference[int(row["number"])] = energy
            except (KeyError, ValueError):
                pass
    if result["problems"]:
        return {**result, "status": FAILED}
    if per_cell and per_atom:
        result["problems"].append("USPEX's tables report energy both per cell and per atom")
        return {**result, "status": FAILED}
    if not per_cell and not per_atom:
        result["problems"].append("no USPEX table gives energy with energy_per_atom and atoms per cell")
        return {**result, "status": MISSING}
    quantity = "per_cell" if per_cell else "per_atom"
    result["basis"].append(f"{per_cell or per_atom} rows of Individuals_all/_extended: energy = "
                           + ("energy_per_atom x atoms" if per_cell else "energy_per_atom"))
    factor = 1 if quantity == "per_cell" else cell_atoms
    # gatheredPOSCARS' "energy=" is checked separately: the same quantity, or per cell beside per-atom tables.
    gathered_fits = {1: 0, cell_atoms: 0} if cell_atoms != 1 else {1: 0}
    compared = 0
    for row in individuals:
        if row["natoms"] is not None and row["natoms"] != cell_atoms:
            result["problems"].append(f"Individuals: individual {row['number']} has {row['natoms']} atoms")
        if row["number"] in reference and abs(reference[row["number"]] - row["energy"]) > 0.0015:
            result["problems"].append(f"Individuals and Individuals_all disagree for individual {row['number']}")
        gathered_row = gathered.get(row["number"])
        if gathered_row is not None:
            structure = parse_poscar(gathered_row["poscar"])
            if structure is not None and sum(structure["counts"]) != cell_atoms:
                result["problems"].append(f"gatheredPOSCARS: individual {row['number']} has "
                                          f"{sum(structure['counts'])} atoms")
            compared += 1
            for gathered_factor in gathered_fits:
                if abs(gathered_row["energy"] * gathered_factor - row["energy"] * factor) <= 0.0015 * factor:
                    gathered_fits[gathered_factor] += 1
    if result["problems"]:
        return {**result, "status": FAILED}
    consistent = [f for f, n in gathered_fits.items() if compared and n == compared]
    if compared and len(consistent) != 1:
        result["problems"].append("gatheredPOSCARS energies are not one consistent quantity alongside Individuals")
        return {**result, "status": FAILED}
    result["quantity"] = quantity
    result["factor_to_cell"] = factor
    result["gathered_factor_to_cell"] = consistent[0] if consistent else None
    result["gathered_quantity"] = (None if not consistent else "per_cell" if consistent[0] == 1 else "per_atom")
    return {**result, "status": VERIFIED}


def associate(listed: list[dict], complete: dict[int, dict], every_step: dict[int, list[dict]],
              surviving_structures: dict[int, dict], gathered: dict[int, dict], *, factor: int,
              gathered_factor: int | None) -> dict[str, Any]:
    """Associate each listed candidate with exactly one accepted (complete, valid) step-5 chain.

    Evidence used, never the bare assumption that USPEX's number is the
    CalcFold's: the listed energy (normalised to the cell) against each
    chain's step-5 F; the structure USPEX recorded (``gatheredPOSCARS``)
    against the step-5 CONTCAR of surviving CalcFolds; and the CalcFold
    naming convention, used only where the surviving generation confirms it
    (structure, energy and number agree, and nothing contradicts it).
    Outcomes per candidate: ``established`` (with its method), ``ambiguous``
    or ``contradicted``.
    """

    unknown = sorted(m for m, chain in complete.items() if chain["free_energy_ev"] is None)
    rows = []
    for row in listed:
        number = row["number"]
        gathered_row = gathered.get(number)
        precise = gathered_row["energy"] if gathered_row is not None and gathered_factor else None
        energy = precise * gathered_factor if precise is not None else row["energy"] * factor
        tolerance = GATHERED_TOLERANCE_EV * gathered_factor if precise is not None else INDIVIDUALS_TOLERANCE_EV * factor
        by_energy = sorted(m for m, chain in complete.items()
                           if chain["free_energy_ev"] is not None and abs(chain["free_energy_ev"] - energy) <= tolerance)
        structure = parse_poscar(gathered_row["poscar"]) if gathered_row is not None else None
        by_structure = sorted(m for m, contcar in surviving_structures.items() if same_structure(structure, contcar))
        stale = sorted({(m, s["step"]) for m, chain in every_step.items() for s in chain if m not in complete
                        and s["free_energy_ev"] is not None and abs(s["free_energy_ev"] - energy) <= tolerance})
        by_e0 = sorted(m for m, chain in complete.items() if chain.get("e0_ev") is not None
                       and abs(chain["e0_ev"] - energy) <= tolerance and m not in by_energy)
        rows.append({"generation": row["generation"], "number": number, "energy_cell_ev": energy, "by_e0": by_e0,
                     "tolerance_ev": tolerance, "by_energy": by_energy, "by_structure": by_structure,
                     "structure_known": structure is not None, "stale_matches": stale})

    confirmed = [r["number"] for r in rows if r["number"] in r["by_structure"] and r["number"] in r["by_energy"]]
    counter = [r["number"] for r in rows if r["by_structure"] and r["number"] not in r["by_structure"]
               and set(r["by_structure"]) & set(r["by_energy"])]
    convention = bool(confirmed) and not counter
    for r in rows:
        number, energy_set, structure_set = r["number"], set(r["by_energy"]), set(r["by_structure"])
        both = structure_set & energy_set
        if structure_set:
            if len(both) == 1:
                r.update(outcome="established", chain=next(iter(both)), method="structure and energy")
            elif both and convention and number in both:
                r.update(outcome="established", chain=number, method="structure, energy and confirmed numbering")
            elif both:
                r.update(outcome="ambiguous", reason=f"structure and energy match chains {sorted(both)}")
            elif structure_set & set(unknown):
                r.update(outcome="ambiguous", reason=f"structure matches chain(s) {sorted(structure_set)}, whose "
                                                     "step-5 F VASP's log does not give")
            else:
                r.update(outcome="contradicted", reason=f"structure matches chain(s) {sorted(structure_set)} but no "
                                                        "accepted step-5 energy of them matches")
            continue
        if number in surviving_structures and r["structure_known"]:
            if energy_set == {number}:
                r.update(outcome="contradicted", reason=f"the energy matches chain {number}, whose step-5 structure "
                                                        "differs from the one USPEX recorded")
                continue
        if not energy_set and unknown:
            r.update(outcome="ambiguous", reason=f"no known step-5 F matches, and chains {unknown[:10]} have none "
                                                 "recorded")
        elif not energy_set:
            r.update(outcome="contradicted", reason="no accepted step-5 energy matches"
                     + (f" (it matches rejected or incomplete chains {r['stale_matches']}: stale)" if r["stale_matches"]
                        else "") + (f" (it equals E0, not F, of chains {r['by_e0']})" if r["by_e0"] else ""))
        elif convention and number in energy_set:
            r.update(outcome="established", chain=number, method="energy and numbering confirmed on the survivors")
        elif len(energy_set) == 1 and not unknown:
            r.update(outcome="established", chain=next(iter(energy_set)), method="unique energy")
        elif len(energy_set) == 1:
            r.update(outcome="ambiguous", reason=f"energy matches chain {next(iter(energy_set))} only among chains with "
                                                 f"a recorded F; chains {unknown[:10]} have none")
        else:
            r.update(outcome="ambiguous", reason=f"energy matches chains {sorted(energy_set)} and nothing decides")
    chains_used: dict[int, list[int]] = {}
    for r in rows:
        if r["outcome"] == "established":
            chains_used.setdefault(r["chain"], []).append(r["number"])
    for chain, numbers in chains_used.items():
        if len(numbers) > 1:
            for r in rows:
                if r["number"] in numbers:
                    r.update(outcome="contradicted", reason=f"chain {chain} is associated with individuals {numbers}")
    # Generation order: every chain of generation g started before any chain of generation g + 1.
    established = [r for r in rows if r["outcome"] == "established"]
    for r in established:
        later = [o for o in established if o["generation"] > r["generation"]]
        if any(complete[o["chain"]]["chain_start"] < complete[r["chain"]]["chain_start"] for o in later):
            r.update(outcome="contradicted", reason="its chain started after a chain of a later generation")
    return {"rows": rows, "numbering_confirmed": convention, "confirmed_by_structure": confirmed,
            "numbering_counterexamples": counter}


ARCHIVE = Path("search") / "bmd_evidence"
F_AGREEMENT_EV = 1e-4  # the checker's vasprun F against VASP's printed F (8 significant decimals)


def verify_archive(files: Path, manifest: dict, steps: list[dict], chains: dict[int, list[dict]],
                   complete: dict[int, dict], started: dict, finished: dict, validity_events: dict, done: dict,
                   logs: dict, journal: list[dict]) -> dict[str, Any]:
    """Check the launcher's evidence archive against the journal, done files, step logs and surviving CalcFolds.

    Each step the journal started must have one committed record whose files match their hashes, whose
    SHA-256 is in the archive index and equals the journal's ``evidence`` event and the done file's
    ``evidence=``, and whose identity (request, step, CalcFold, slot, command), outcome (return code,
    status) and verdict (classification, reasons) equal the launcher's other records. A surviving
    CalcFold's ``bmd_step_validity.jsonl`` line must be the archived verdict byte for byte, and its
    step-5 CONTCAR the archived one. The checker's F must equal VASP's printed F; each step must start
    from the structure its predecessor ended with. Contradictions fail; absent, interrupted or
    unverifiable records are missing evidence.
    """

    root = files / ARCHIVE
    truncated = {c["path"] for c in manifest.get("copied") or [] if c.get("truncated")}
    index_lines, torn = [], 0
    for line in (_text(root / "index.jsonl") or "").splitlines():
        try:
            entry = json.loads(line)
            index_lines.append((entry["id"], entry["record_sha256"]))
        except (ValueError, KeyError, TypeError):
            torn += 1
    indexed = set(index_lines)
    records: dict[str, dict] = {}
    unverifiable: list[str] = []
    for folder in sorted((root / "steps").glob("*")) if (root / "steps").is_dir() else []:
        record_path = folder / "record.json"
        relative = str(record_path.relative_to(files))
        if not record_path.is_file() or relative in truncated:
            unverifiable.append(folder.name)
            continue
        data = record_path.read_bytes()
        try:
            record = json.loads(data)
        except ValueError:
            unverifiable.append(folder.name)
            continue
        own = []
        for name, key in (("POSCAR", "input_structure"), ("CONTCAR", "output_structure")):
            facts = record.get(key) or {}
            path = folder / name
            if facts.get("archived"):
                if not path.is_file() or str(path.relative_to(files)) in truncated:
                    own.append(f"{name} absent from the evidence")
                elif hashlib.sha256(path.read_bytes()).hexdigest() != facts.get("sha256"):
                    own.append(f"{name} does not match its recorded hash")
        records[folder.name] = {"record": record, "sha256": hashlib.sha256(data).hexdigest(), "problems": own}

    evidence_events = {e.get("id"): e for e in journal if e.get("event") == "evidence"}
    write_failures = sorted({e.get("id") for e in journal if e.get("event") == "evidence_write_failed"})
    contradictions: dict[str, list[str]] = {}
    absent: list[str] = []
    undelivered: list[dict] = []
    verified: set[str] = set()

    def contradict(request_id: str, problem: str) -> None:
        contradictions.setdefault(request_id, []).append(problem)

    surviving_lines: dict[str, str] = {}
    for path in sorted((files / "search" / "Calculation").glob("*/bmd_step_validity.jsonl")):
        if str(path.relative_to(files)) in truncated:
            continue
        for line in (_text(path) or "").splitlines():
            try:
                surviving_lines[json.loads(line)["id"]] = line
            except (ValueError, KeyError, TypeError):
                pass
    for s_ in steps:
        request_id = s_["id"]
        entry = records.get(request_id)
        if entry is None:
            absent.append(request_id)  # never committed (evidence_write_failed) or lost: incomplete, not rejected
            continue
        record, digest = entry["record"], entry["sha256"]
        for problem in entry["problems"]:
            contradict(request_id, problem)
        if request_id not in finished:
            # Committed, then the launcher stopped before finishing the step: the done file USPEX read (if
            # any) is a later launcher's "lost" answer, not this record. Not a contradiction, not delivered.
            undelivered.append({"id": request_id, "done_status": done.get(request_id, {}).get("status"),
                                "archived_status": (record.get("launch") or {}).get("status")})
            continue
        if (request_id, digest) not in indexed:
            absent.append(request_id)  # committed but not indexed: not yet recovered by a launcher start
        event = evidence_events.get(request_id) or {}
        if event and event.get("record_sha256") != digest:
            contradict(request_id, "the journal's evidence event names another record")
        if done.get(request_id, {}).get("evidence") not in (None, digest):
            contradict(request_id, "the done file names another record")
        start, end = started.get(request_id, {}), finished.get(request_id, {})
        launch = record.get("launch") or {}
        verdict = record.get("validity") or {}
        events = validity_events.get(request_id) or [{}]
        pairs = (("id", record.get("id"), request_id), ("step", record.get("step"), start.get("step")),
                 ("calcfold", record.get("calcfold"), start.get("calcfold")),
                 ("slot", launch.get("slot"), start.get("slot")), ("argv", launch.get("argv"), start.get("argv")),
                 ("returncode", launch.get("returncode"), end.get("returncode")),
                 ("status", launch.get("status"), end.get("status")),
                 ("classification", verdict.get("classification"), end.get("validity")),
                 ("journal verdict", verdict.get("classification"), events[0].get("classification")),
                 ("reasons", verdict.get("reasons"), events[0].get("reasons")),
                 ("done verdict", verdict.get("classification"), done.get(request_id, {}).get("validity")))
        for name, archived, other in pairs:
            if archived != other:
                contradict(request_id, f"{name}: archive {archived!r}, launcher {other!r}")
        line = surviving_lines.get(request_id)
        if line is not None:
            if hashlib.sha256((line + "\n").encode()).hexdigest() != record.get("validity_record_line_sha256"):
                contradict(request_id, "the surviving CalcFold's verdict line is not the archived one")
            elif json.loads(line) != verdict:
                contradict(request_id, "the archived verdict differs from the CalcFold's")
        log = logs.get(request_id)
        if log is None:
            absent.append(request_id)
        else:
            if (record.get("step_log") or {}).get("sha256") != log.get("sha256"):
                contradict(request_id, "the step log is not the one the record names")
            archived_f = (verdict.get("evidence") or {}).get("free_energy_ev")
            if (archived_f is not None and log.get("final_free_energy_ev") is not None
                    and abs(archived_f - log["final_free_energy_ev"]) > F_AGREEMENT_EV):
                contradict(request_id, f"checker F {archived_f} differs from VASP's printed F "
                                       f"{log['final_free_energy_ev']}")
        if request_id not in contradictions and request_id not in absent:
            verified.add(request_id)

    continuity = []
    for individual, chain in chains.items():
        ordered = sorted(chain, key=lambda s_: s_["step"] or 0)
        for before, after in zip(ordered, ordered[1:]):
            a, b = records.get(before["id"]), records.get(after["id"])
            if not a or not b:
                continue
            out_sha = (a["record"].get("output_structure") or {}).get("sha256")
            in_sha = (b["record"].get("input_structure") or {}).get("sha256")
            if out_sha and in_sha and out_sha != in_sha:
                same = same_structure(parse_poscar(_text(root / "steps" / before["id"] / "CONTCAR")),
                                      parse_poscar(_text(root / "steps" / after["id"] / "POSCAR")))
                if same is not True:
                    continuity.append([before["id"], after["id"]])
                    contradict(after["id"], "it did not start from its predecessor's output structure"
                               if same is False else "its input cannot be compared with its predecessor's output")

    finals: dict[int, dict] = {}
    for individual, last in complete.items():
        entry = records.get(last["id"])
        if not entry:
            continue
        record = entry["record"]
        if not record.get("final_relaxed_structure"):
            contradict(last["id"], "a valid step 5 without an archived final structure")
            continue
        finals[individual] = {"id": last["id"], "structure": parse_poscar(_text(root / "steps" / last["id"] / "CONTCAR")),
                              "sha256": record["output_structure"]["sha256"]}
        survivor = files / "search" / "Calculation" / Path(record.get("calcfold", "")).name / "CONTCAR"
        if survivor.is_file() and hashlib.sha256(survivor.read_bytes()).hexdigest() != finals[individual]["sha256"]:
            contradict(last["id"], "the surviving CalcFold's CONTCAR is not the archived final structure")

    conflicts = sorted(p.name for p in (root / "conflicts").glob("*")) if (root / "conflicts").is_dir() else []
    interrupted = sorted(p.name for p in (root / "interrupted").glob("*")) if (root / "interrupted").is_dir() else []
    staging = manifest.get("evidence_staging") or []
    for name in conflicts:
        contradict(name.rsplit("-", 1)[0], "a different record was offered for this committed step")
    critical = [line.strip()[:300] for path in sorted(files.glob("slurm-*.out"))
                for line in (_text(path) or "").splitlines() if line.startswith("BMD-LAUNCHER-CRITICAL")]
    status = (FAILED if contradictions else
              MISSING if (absent or unverifiable or interrupted or staging or torn or write_failures or undelivered
                          or critical or not records)
              else VERIFIED)
    return {"status": status, "verified": verified, "finals": finals, "detail": {
        "steps": len(steps), "records": len(records), "verified_steps": len(verified),
        "missing_records": sorted(set(absent))[:20], "write_failures": write_failures[:20],
        "archived_but_not_delivered": undelivered[:20], "launcher_critical_lines": critical[:20],
        "unverifiable": unverifiable[:20], "interrupted": interrupted[:20], "unfinished_writes": staging[:20],
        "conflicts": conflicts[:20], "torn_index_lines": torn, "continuity_breaks": continuity[:20],
        "contradictions": {k: v for k, v in list(contradictions.items())[:20]},
        "final_structures": len(finals),
        "generation": "not part of execution identity: the launcher records none (USPEX does not pass it)"}}


def _sacct(text: str | None, job: str) -> dict[str, Any]:
    rows = [line.split("|") for line in (text or "").splitlines() if line.count("|") == 9]
    main = next((r for r in rows if r[0] == job), None)
    steps = [r for r in rows if re.fullmatch(rf"{re.escape(job)}\.\d+", r[0])]
    return {"main": main, "steps": steps, "batch": next((r for r in rows if r[0] == f"{job}.batch"), None)}


def _memory_mb(tres: str) -> int | None:
    match = re.search(r"(?:^|,)mem=([0-9.]+)([KMGTP]?)", tres or "")
    if not match:
        return None
    factor = {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}[match.group(2)]
    return int(float(match.group(1)) * factor)


def _cpus(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-", 1)
            cpus.update(range(int(low), int(high) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def _symmetry(poscar: str) -> dict[str, Any]:
    """Space group at two tolerances and a diamond match; pymatgen only, None if unavailable."""

    try:
        import warnings

        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.vasp import Poscar
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        return {"available": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            structure = Poscar.from_str(poscar).structure  # the "number=..." line is the comment line
        except Exception as exc:  # noqa: BLE001
            return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
        result: dict[str, Any] = {"available": True, "natoms": len(structure),
                                  "volume_per_atom": round(structure.volume / len(structure), 4)}
        for symprec in (0.01, 0.1):
            try:
                analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
                result[f"space_group_{symprec}"] = [analyzer.get_space_group_number(),
                                                    analyzer.get_space_group_symbol()]
            except Exception:  # noqa: BLE001
                result[f"space_group_{symprec}"] = None
        half = 5.47 / 2  # primitive diamond Si (Fd-3m), written out: building it from the space group is slow
        diamond = Structure(Lattice([[0, half, half], [half, 0, half], [half, half, 0]]), ["Si", "Si"],
                            [[0, 0, 0], [0.25, 0.25, 0.25]])
        result["matches_diamond"] = bool(StructureMatcher(scale=True).fit(structure, diamond))
        result["diamond_like"] = result["matches_diamond"] or (result.get("space_group_0.1") or [None])[0] == 227
        return result


# -- the analysis ----------------------------------------------------------------------------------------------


def analyse(evidence: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """The report for one evidence directory. ``expected`` comes from the validation's own records."""

    evidence = Path(evidence)
    files = evidence / "files"
    job, attempt = str(expected["job_id"]), expected["attempt_id"]
    report = Report()
    findings: dict[str, Any] = {}

    # Collection itself.
    manifest = _json(evidence / "manifest.json") or {}
    collector = _text(evidence / "collector.txt") or ""
    report.add("collection", "the collector copied only allowlisted records and left the run directory unchanged",
               VERIFIED if manifest and "PASS  the run directory is unchanged" in collector
               and not manifest.get("refused") else FAILED if manifest else MISSING,
               {"refused": manifest.get("refused"), "missing": manifest.get("missing")})

    # SLURM outcome and allocation.
    sacct = _sacct(_text(evidence / "sacct_steps.txt"), job)
    main = sacct["main"]
    slurm_state = main[2].split()[0] if main else None
    if not main:
        report.add("slurm_outcome", "SLURM's settled outcome of the job", MISSING, "sacct reported no job row")
    else:
        report.add("slurm_outcome", "SLURM's settled outcome of the job",
                   VERIFIED if slurm_state == "COMPLETED" and main[3] == "0:0" else FAILED,
                   {"state": main[2], "exit_code": main[3], "start": main[4], "end": main[5], "elapsed": main[6]})
        report.add("allocation", "the job held one exclusive 192-CPU node for the acknowledged walltime",
                   VERIFIED if main[7] == "192" and main[9] == expected["walltime"] and "node=1" in main[8]
                   else FAILED, {"ncpus": main[7], "alloc_tres": main[8], "timelimit": main[9]})

    # Status, runtime check, memory.
    status = _json(files / "search_status.json")
    execution = (status or {}).get("execution") or {}
    launch = execution.get("launch") or {}
    runtime = _json(files / "runtime_check.json")
    observed = (runtime or {}).get("observed") or {}
    prepared = ((runtime or {}).get("prepared") or {}).get("launch") or {}
    if runtime is None:
        report.add("runtime_check", "the compute-node runtime check", MISSING)
    else:
        report.add("runtime_check", "the compute-node runtime check passed, including MPI library and SLURM version",
                   VERIFIED if runtime.get("status") == "passed" and observed.get("slurm_version")
                   and observed.get("i_mpi_pmi_library_exists") is True
                   and prepared.get("parallel_relaxations") == expected["parallel"]
                   and prepared.get("ranks_per_relaxation") == expected["ranks"] else FAILED,
                   {"status": runtime.get("status"), "problems": runtime.get("problems"),
                    "slurm_version": observed.get("slurm_version"), "banner_version": observed.get("banner_version")})
    node_memory = launch.get("node_memory") or {}
    alloc_mb = _memory_mb(main[8]) if main else None
    if not node_memory:
        report.add("memory", "the allocated node memory and the per-step memory", MISSING,
                   "search_status.json records no node memory")
    else:
        step_mem = f"{int(node_memory.get('mb', 0) * (1 - MEMORY_MARGIN_FRACTION)) // expected['parallel']}M"
        consistent = (node_memory.get("mb", 0) > 0 and launch.get("step_mem") == step_mem
                      and launch.get("mode") == "srun_steps")
        status_value = (FAILED if not consistent or (alloc_mb is not None and alloc_mb != node_memory["mb"])
                        else MISSING if alloc_mb is None else VERIFIED)
        report.add("memory", "the allocated node memory and the per-step memory", status_value,
                   {"node_memory": node_memory, "step_mem": launch.get("step_mem"), "expected_step_mem": step_mem,
                    "sacct_alloc_mem_mb": alloc_mb})

    # The launcher journal, done files and step logs.
    journal = _jsonl(files / "bridge_spool" / "launcher.jsonl")
    started = {e["id"]: e for e in journal if e.get("event") == "started_step"}
    finished = {e["id"]: e for e in journal if e.get("event") == "finished"}
    validity = {}
    for event in journal:
        if event.get("event") == "validity":
            validity.setdefault(event["id"], []).append(event)
    done = {}
    for path in sorted((files / "bridge_spool" / "done").glob("*.done")):
        values = _key_values(_text(path) or "")
        done[values.get("id", path.stem)] = values
    logs = _json(evidence / "step_logs.json") or {}
    master_text = _text(files / "uspex_master.log")
    master_started = [e for e in journal if e.get("event") == "master_started"]
    master_finished = [e for e in journal if e.get("event") == "master_finished"]

    banner = bool(master_text and re.search(r"USPEX v?25\.0", master_text))
    prepared_inputs = _json(files / "search" / "prepared_inputs.json") or {}
    input_text = (files / "search" / "INPUT.txt")
    input_sha = hashlib.sha256(input_text.read_bytes()).hexdigest() if input_text.is_file() else None
    input_expected = (prepared_inputs.get("files") or {}).get("INPUT.txt")
    if master_text is None or not journal:
        report.add("uspex_startup", "USPEX 25.0 started on the managed INPUT.txt", MISSING)
    else:
        report.add("uspex_startup", "USPEX 25.0 started on the managed INPUT.txt",
                   VERIFIED if banner and master_started and input_sha and input_sha == input_expected else FAILED,
                   {"banner": banner, "master_started": bool(master_started), "input_sha256": input_sha,
                    "manifest_input_sha256": input_expected})

    anomalies = [e for e in journal if e.get("event") in ("rejected", "start_failed", "recovered", "abandoned",
                                                          "validity_record_failed", "launch_record_failed")]
    unmatched = sorted(set(started) ^ set(finished))
    done_mismatch = sorted(i for i in started if i not in done or (i in finished and str(finished[i].get("status"))
                                                                    != done[i].get("status")))
    if not started:
        report.add("bridge", "every bridge request ran once and was answered", MISSING, "no steps in the journal")
    else:
        report.add("bridge", "every bridge request ran once and was answered",
                   VERIFIED if not anomalies and not unmatched and not done_mismatch and len(done) == len(started)
                   else FAILED, {"steps": len(started), "done_files": len(done), "unmatched": unmatched[:20],
                                 "done_mismatch": done_mismatch[:20],
                                 "anomalies": [e.get("event") for e in anomalies][:20]})

    # The steps: individual, step, timing, status, validity, energy.
    steps = []
    for request_id, event in started.items():
        match = _CALCFOLD.search(str(event.get("calcfold", "")))
        end = finished.get(request_id, {})
        log = logs.get(request_id, {})
        steps.append({
            "id": request_id, "individual": int(match.group(1)) if match else None,
            "step": event.get("step"), "calcfold_step": int(match.group(2)) if match else None,
            "start": event.get("time"), "end": end.get("time"), "seconds": end.get("seconds"),
            "returncode": end.get("returncode"), "status": end.get("status"),
            "classification": end.get("validity"), "argv": event.get("argv") or [],
            "free_energy_ev": log.get("final_free_energy_ev"), "e0_ev": log.get("final_e0_ev"),
            "ionic_steps": log.get("ionic_steps"),
            "pin_rows": log.get("pin_rows"), "log_errors": log.get("error_lines"), "has_log": request_id in logs,
            # Accepted only with completion evidence: the launcher's finished event and the done file USPEX
            # read both say valid, status 0. An archived verdict alone is not delivery.
            "delivered_status": done.get(request_id, {}).get("status"),
            "accepted": (end.get("validity") == "valid" and str(end.get("status")) == "0"
                         and done.get(request_id, {}).get("status") == "0"
                         and done.get(request_id, {}).get("validity") == "valid"),
        })
    steps.sort(key=lambda s: (s["start"] or 0, s["id"]))

    srun_steps = [s for s in steps if "srun" in [str(a).rsplit("/", 1)[-1] for a in s["argv"]]]
    ranks_ok = all(str(expected["ranks"]) in s["argv"][s["argv"].index("-n") + 1:s["argv"].index("-n") + 2]
                   for s in srun_steps if "-n" in s["argv"])
    mem_ok = all(f"--mem={launch.get('step_mem')}" in s["argv"] for s in srun_steps) if launch.get("step_mem") else False
    ran = [s for s in steps if s["returncode"] == 0]
    no_log = [s["id"] for s in ran if not s["has_log"]]
    no_energy = [s["id"] for s in ran if s["has_log"] and (s["ionic_steps"] in (None, 0) or s["free_energy_ev"] is None)]
    sacct_steps = sacct["steps"]
    if not steps or not main:
        report.add("vasp_steps", "VASP ran as 48-rank SLURM steps with the per-step memory", MISSING)
    else:
        sound = (srun_steps and len(srun_steps) == len(steps) and ranks_ok and mem_ok and not no_energy
                 and len(sacct_steps) >= len(srun_steps) and all(r[7] == str(expected["ranks"]) for r in sacct_steps))
        report.add("vasp_steps", "VASP ran as 48-rank SLURM steps with the per-step memory",
                   FAILED if not sound else MISSING if no_log else VERIFIED,
                   {"steps": len(steps), "srun_steps": len(srun_steps), "sacct_steps": len(sacct_steps),
                    "ranks_ok": ranks_ok, "mem_ok": mem_ok, "steps_without_energy": no_energy[:20],
                    "steps_without_log": no_log[:20]})

    intervals = [(s["start"], s["end"]) for s in steps if s["start"] is not None and s["end"] is not None]
    concurrency = 0
    for start, _ in intervals:
        concurrency = max(concurrency, sum(1 for a, b in intervals if a <= start < b))
    if not intervals:
        report.add("parallel", "VASP steps ran in parallel, never more than the plan's limit", MISSING)
    else:
        report.add("parallel", "VASP steps ran in parallel, never more than the plan's limit",
                   VERIFIED if 2 <= concurrency <= expected["parallel"] else FAILED,
                   {"max_concurrent": concurrency, "limit": expected["parallel"]})

    # Pinning: a step's report is interpretable only if it lists each of its 48 ranks once, on one host.
    interpretable, uninterpretable, own_overlap = [], [], []
    for s_ in ran:
        rows = s_["pin_rows"] or []
        ranks = sorted(r[0] for r in rows)
        hosts = {r[2] for r in rows}
        try:
            cpu_sets = [_cpus(r[3]) for r in rows]
        except ValueError:
            cpu_sets = None
        if ranks != list(range(expected["ranks"])) or len(hosts) != 1 or not cpu_sets or not all(cpu_sets):
            uninterpretable.append(s_["id"])
            continue
        union = set().union(*cpu_sets)
        if sum(len(c) for c in cpu_sets) != len(union):
            own_overlap.append(s_["id"])
        interpretable.append({**s_, "cpus": union, "host": next(iter(hosts))})
    overlap = []
    for i, a_ in enumerate(interpretable):
        for b_ in interpretable[i + 1:]:
            if a_["start"] < b_["end"] and b_["start"] < a_["end"] and a_["host"] == b_["host"] and a_["cpus"] & b_["cpus"]:
                overlap.append([a_["id"], b_["id"], sorted(a_["cpus"] & b_["cpus"])[:8]])
    report.add("pinning", "concurrent 48-rank VASP steps used disjoint cores (Intel MPI pinning report)",
               FAILED if overlap or own_overlap else MISSING if uninterpretable or not ran else VERIFIED,
               {"steps_ran": len(ran), "interpretable": len(interpretable), "uninterpretable": uninterpretable[:10],
                "overlaps": overlap[:10], "ranks_sharing_a_core": own_overlap[:10]})

    # Validity decisions.
    bad_decisions = []
    for s in steps:
        events = validity.get(s["id"], [])
        classes = [e.get("classification") for e in events]
        done_status = done.get(s["id"], {}).get("status")
        allowed = DONE_STATUS.get(s["classification"])
        if (len(events) != 1 or classes[0] not in CLASSIFICATIONS or classes[0] != s["classification"]
                or done.get(s["id"], {}).get("validity") != s["classification"]
                or done_status is None or (allowed is not None and int(done_status) not in allowed)
                or (s["classification"] == "execution_failed" and int(done_status) == 0)):
            bad_decisions.append(s["id"])
    surviving = {}
    for path in sorted((files / "search" / "Calculation").glob("*/bmd_step_validity.jsonl")):
        for row in _jsonl(path):
            surviving[row.get("classification")] = surviving.get(row.get("classification"), 0) + 1
    recorded = ((status or {}).get("scientific_validity") or {}).get("step_validity_verdicts")
    if not steps:
        report.add("validity_decisions", "one validity decision per step, consistent everywhere", MISSING)
    else:
        report.add("validity_decisions", "one validity decision per step, consistent everywhere",
                   VERIFIED if not bad_decisions and (recorded is None or recorded == surviving) else FAILED,
                   {"inconsistent_steps": bad_decisions[:20], "status_verdicts": recorded,
                    "surviving_calcfold_verdicts": surviving})

    # Chains: per individual, steps 1..k in order, once each, stopping at the first non-valid step.
    chains: dict[int, list[dict]] = {}
    for s in steps:
        chains.setdefault(s["individual"], []).append(s)
    complete, rejected, broken = {}, {}, {}
    for individual, chain in sorted(chains.items(), key=lambda item: (item[0] is None, item[0] or 0)):
        numbers = [s["step"] for s in chain]
        problems = []
        if individual is None or any(s["step"] != s["calcfold_step"] for s in chain):
            problems.append("step and CalcFold disagree")
        if numbers != list(range(1, len(numbers) + 1)):
            problems.append(f"steps ran as {numbers}")
        first_bad = next((i for i, s in enumerate(chain) if not s["accepted"]), None)
        if first_bad is not None and first_bad != len(chain) - 1:
            problems.append("steps ran after a rejected step")
        if problems:
            broken[individual] = problems
        elif first_bad is not None:
            rejected[individual] = {"step": chain[first_bad]["step"],
                                    "classification": chain[first_bad]["classification"] or "not_delivered",
                                    "delivered_status": chain[first_bad]["delivered_status"]}
        elif len(chain) == expected["steps"]:
            complete[individual] = {**chain[-1], "chain_start": chain[0]["start"] or 0}
        else:
            broken[individual] = [f"stopped after step {len(chain)} without a rejection"]
    report.add("chains", "every candidate ran its ladder in order, once per step, stopping at a rejection",
               VERIFIED if chains and not broken else FAILED if broken else MISSING,
               {"candidates": len(chains), "complete": len(complete), "rejected": len(rejected),
                "broken": {str(k): v for k, v in list(broken.items())[:20]}})

    # The launcher's evidence archive, verified against everything else the launcher and VASP left.
    archive = verify_archive(files, manifest, steps, chains, complete, started, finished, validity, done, logs,
                             journal)
    report.add("evidence_archive", "every step's archived record agrees with the journal, done file, step log "
               "and surviving CalcFolds", archive["status"], archive["detail"])

    # What USPEX lists, what it means, and which accepted chain each listed candidate is.
    results_dirs = sorted(p.name for p in (files / "search").glob("results*") if p.is_dir())
    results = files / "search" / "results1"
    individuals = read_individuals(_text(results / "Individuals"))
    gathered = read_gathered_poscars(_text(results / "gatheredPOSCARS"))
    semantics: dict[str, Any] | None = None
    association: dict[str, Any] | None = None
    # Final structures: the archived step-5 CONTCAR of every accepted chain; a surviving CalcFold's
    # CONTCAR only where the archive has none (it is checked against the archive above).
    recorded_structures = {individual: final["structure"] for individual, final in archive["finals"].items()}
    for path in sorted((files / "search" / "Calculation").glob("Calcfold_*_5/CONTCAR")):
        match = _CALCFOLD.search(path.parent.name)
        if match and int(match.group(1)) in complete and int(match.group(1)) not in recorded_structures:
            recorded_structures[int(match.group(1))] = parse_poscar(_text(path))
    if individuals is None:
        report.add("energy_semantics", "the quantity USPEX lists (per cell or per atom, eV) is established", MISSING)
        report.add("listed_energies", "every listed candidate is unambiguously one accepted chain, at its step-5 F",
                   MISSING, "results1/Individuals missing or unreadable")
        report.add("retention", "every listed candidate's steps and final structure are kept after CalcFold clean-up",
                   MISSING)
        report.add("generations", "USPEX listed every generation and ended normally", MISSING)
    else:
        input_values = dict((m.group(2), m.group(1)) for m in re.finditer(r"^\s*(\S+)\s*:\s*(\w+)\s*$",
                                                                           _text(files / "search" / "INPUT.txt") or "",
                                                                           re.MULTILINE))
        pressure = _float(input_values.get("ExternalPressure"))
        atoms_range = (_float(input_values.get("minAt")), _float(input_values.get("maxAt")))
        semantics = energy_semantics(individuals, {name: read_table(_text(results / name)) for name in
                                                   ("Individuals_all", "Individuals_extended")}, gathered,
                                     cell_atoms=expected["cell_atoms"], pressure_gpa=expected["pressure_gpa"])
        if pressure is None or atoms_range[0] is None:
            semantics = {**semantics, "status": MISSING if semantics["status"] != FAILED else FAILED,
                         "problems": semantics["problems"] + ["INPUT.txt gives no ExternalPressure or minAt/maxAt"]}
        elif pressure != expected["pressure_gpa"] or atoms_range != (expected["cell_atoms"], expected["cell_atoms"]):
            semantics = {**semantics, "status": FAILED,
                         "problems": semantics["problems"] + [f"INPUT.txt: ExternalPressure {pressure}, atoms "
                                                              f"{atoms_range}, not the plan's"]}
        if semantics["status"] != VERIFIED:
            semantics.pop("factor_to_cell", None)
        association = (associate(individuals, complete, chains, recorded_structures, gathered,
                                factor=semantics.get("factor_to_cell") or 1,
                                gathered_factor=semantics.get("gathered_factor_to_cell"))
                       if semantics["status"] == VERIFIED else None)
        # VASP's F and E0 differ by the smearing term: a listing of E0 is the wrong quantity, not a stray value.
        e0_matches = [row["number"] for row in (association or {}).get("rows", [])
                      if row["outcome"] == "contradicted" and row["by_e0"]]
        if e0_matches:
            semantics = {**semantics, "status": FAILED,
                         "problems": semantics["problems"] + [f"listed energies equal E0, not F: {e0_matches}"]}
        report.add("energy_semantics", "the quantity USPEX lists (per cell or per atom, eV) is established",
                   semantics["status"], {key: semantics.get(key) for key in ("quantity", "gathered_quantity", "basis",
                                                                             "problems")})
        if association is None:
            report.add("listed_energies", "every listed candidate is unambiguously one accepted chain, at its step-5 F",
                       MISSING, "not compared: the listed quantity is not established")
        else:
            outcomes = [r["outcome"] for r in association["rows"]]
            report.add("listed_energies", "every listed candidate is unambiguously one accepted chain, at its step-5 F",
                       FAILED if "contradicted" in outcomes else MISSING if "ambiguous" in outcomes else VERIFIED,
                       {"listed": len(outcomes), "established": outcomes.count("established"),
                        "ambiguous": [{k: r.get(k) for k in ("number", "reason")} for r in association["rows"]
                                      if r["outcome"] == "ambiguous"][:20],
                        "contradicted": [{k: r.get(k) for k in ("number", "reason")} for r in association["rows"]
                                         if r["outcome"] == "contradicted"][:20],
                        "methods": sorted({r["method"] for r in association["rows"] if r.get("method")}),
                        "numbering_confirmed": association["numbering_confirmed"],
                        "numbering_counterexamples": association["numbering_counterexamples"]})
        # Retention: every listed candidate's five steps in the verified archive, its final structure kept,
        # whether or not USPEX has removed its CalcFolds; its generation bound through the listing.
        traced, untraced, bound = [], [], []
        present = set(manifest.get("calcfolds_present") or [])
        for row in (association or {}).get("rows", []):
            if row["outcome"] != "established":
                untraced.append(row["number"])  # not associated with one chain: nothing to trace it to
                continue
            chain = chains.get(row["chain"], [])
            ok = (len(chain) == expected["steps"] and all(s_["id"] in archive["verified"] for s_ in chain)
                  and row["chain"] in archive["finals"])
            (traced if ok else untraced).append(row["number"])
            if ok:
                bound.append({"number": row["number"], "listed_generation": row["generation"], "chain": row["chain"],
                              "final_record": archive["finals"][row["chain"]]["id"],
                              "final_structure_sha256": archive["finals"][row["chain"]]["sha256"]})
        deleted = sorted(r["chain"] for r in (association or {}).get("rows", []) if r.get("chain") is not None
                         and f"Calcfold_{r['chain']}_5" not in present)
        report.add("retention", "every listed candidate's steps and final structure are kept after CalcFold clean-up",
                   MISSING if association is None or untraced or not traced else VERIFIED,
                   {"traced": len(traced), "untraced": untraced[:20], "chains_whose_calcfolds_were_removed": deleted,
                    "sources": "bmd_evidence archive (verdict, input and output structures, hashes), checked "
                               "against the journal, done files and VASP step logs",
                    "inferred_generation_association": {
                        "source": "USPEX's Individuals listing through the post-run identity association; "
                                  "inferred, not execution identity (archived records carry generation null)",
                        "rows": bound[:50]}})
        generations = sorted({r["generation"] for r in individuals})
        ended = (master_finished and master_finished[-1].get("returncode") == 0
                 and master_finished[-1].get("stopped") is False)
        report.add("generations", "USPEX listed every generation and ended normally",
                   VERIFIED if generations == list(range(1, expected["generations"] + 1)) and ended
                   and results_dirs == ["results1"] else FAILED,
                   {"generations_listed": generations, "per_generation": {
                       str(g): sum(1 for r in individuals if r["generation"] == g) for g in generations},
                    "master_finished": master_finished[-1] if master_finished else None,
                    "results_folders": results_dirs})

    # Search status and identities.
    if status is None:
        report.add("search_status", "search_status.json for this attempt, job and plan", MISSING)
    else:
        report.add("search_status", "search_status.json for this attempt, job and plan",
                   VERIFIED if status.get("attempt_id") == attempt and str(status.get("slurm_job_id")) == job
                   and status.get("plan_digest") == expected["plan_digest"] and status.get("finished_at")
                   and not any((status.get("isolation") or {}).values()) else FAILED,
                   {"outcome": execution.get("outcome"), "master_exit_status": execution.get("master_exit_status"),
                    "finished_at": status.get("finished_at"), "isolation": status.get("isolation")})
    script = files / "bmd_search_job.sbatch"
    ack = _json(files / "resource_acknowledgement.json")
    ack_sha = None
    if isinstance(ack, dict):
        claimed = ack.pop("acknowledgement_sha256", None)
        own = hashlib.sha256(json.dumps(ack, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                             .encode()).hexdigest()
        ack_sha = claimed if claimed == own else None
    incars = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((files / "search" / "Specific").glob("INCAR_*"))}
    identity = {
        "job_started": (_text(files / "job_started") or "").strip() == job,
        "job_script": script.is_file() and hashlib.sha256(script.read_bytes()).hexdigest()
        == expected["job_script_sha256"],
        "acknowledgement": ack_sha == expected["acknowledgement_sha256"],
        "incars": incars == expected["incar_sha256"],
    }
    report.add("identities", "the job, job script, acknowledgement and INCAR ladder are the frozen ones",
               VERIFIED if all(identity.values()) else FAILED, identity)

    # Classification of the ending, from several sources (never SLURM's COMPLETED alone).
    generations_ok = any(c["check"] == "generations" and c["status"] == VERIFIED for c in report.checks)
    if slurm_state == "TIMEOUT":
        classification = "stopped_by_walltime"
    elif (slurm_state == "COMPLETED" and main and main[3] == "0:0" and execution.get("outcome") == "uspex_finished"
          and execution.get("master_exit_status") == 0 and generations_ok):
        classification = "natural_completion"
    else:
        classification = "failed_or_incomplete"

    # Scientific findings (recorded, not required).
    by_class: dict[str, int] = {}
    for s in steps:
        by_class[s["classification"] or "none"] = by_class.get(s["classification"] or "none", 0) + 1
    durations: dict[str, list] = {}
    for s in steps:
        if s["seconds"] is not None:
            durations.setdefault(str(s["step"]), []).append(s["seconds"])
    findings["steps"] = {"total": len(steps), "by_classification": by_class,
                         "seconds_by_step": {k: {"n": len(v), "min": min(v), "max": max(v),
                                                 "mean": round(sum(v) / len(v), 1)}
                                             for k, v in sorted(durations.items())}}
    findings["candidates"] = {"started": len(chains), "completed_ladder": len(complete),
                              "rejected": {str(k): v for k, v in rejected.items()}}
    if individuals:
        best = min(individuals, key=lambda r: r["energy"])
        structures = {}
        for row in individuals:
            if row["number"] in gathered:
                structures[row["number"]] = _symmetry(gathered[row["number"]]["poscar"])
        best_symmetry = structures.get(best["number"])
        natoms = (best_symmetry or {}).get("natoms")
        diamond_like = sorted(n for n, s in structures.items() if s.get("diamond_like"))
        factor = (semantics or {}).get("factor_to_cell") if individuals is not None else None
        cell_energy = round(best["energy"] * factor, 6) if factor else None
        chain_of_best = next((r.get("chain") for r in (association or {}).get("rows", [])
                              if r["number"] == best["number"] and r["outcome"] == "established"), None)
        findings["best"] = {"number": best["number"], "generation": best["generation"],
                            "listed_energy": best["energy"], "listed_quantity": (semantics or {}).get("quantity"),
                            "energy_ev_per_cell": cell_energy,
                            "energy_per_atom_ev": round(cell_energy / natoms, 5) if natoms and cell_energy is not None
                            else None,
                            "accepted_chain": chain_of_best,
                            "step5_free_energy_ev": complete[chain_of_best]["free_energy_ev"] if chain_of_best is not None
                            else None,
                            "symmetry": best_symmetry}
        findings["diamond"] = {
            "required": False,
            "found": bool(diamond_like) if structures else None,
            "individuals": diamond_like,
            "is_lowest_energy": (best["number"] in diamond_like) if structures else None,
            "note": "Recorded, not a condition of operational success.",
        }

    failed = [c["check"] for c in report.checks if c["status"] == FAILED]
    missing = [c["check"] for c in report.checks if c["status"] == MISSING]
    if classification != "natural_completion" or failed:
        result = "FAILED"
    elif missing:
        result = "INCOMPLETE EVIDENCE"
    else:
        result = "PASSED"
    return {
        "schema": "bmd_compute.uspex_execution_acceptance_report",
        "schema_version": 1,
        "attempt_id": attempt,
        "job_id": job,
        "result": result,
        "classification": classification,
        "verified": [c["check"] for c in report.checks if c["status"] == VERIFIED],
        "failed": failed,
        "missing": missing,
        "checks": report.checks,
        "findings": findings,
        "scientific_validity": "not assessed by this report beyond the recorded findings",
    }


def summary_lines(report: dict[str, Any]) -> list[str]:
    lines = [f"== {report['result']} ({report['classification']})"]
    for check in report["checks"]:
        lines.append(f"{check['status'].upper():9} {check['check']}: {check['what']}")
    best = report["findings"].get("best")
    if best:
        symmetry = best.get("symmetry") or {}
        lines.append(f"FINDING   best individual {best['number']} (generation {best['generation']}): "
                     f"{best['energy_ev_per_cell']} eV per cell, {best['energy_per_atom_ev']} eV/atom, "
                     f"space group {symmetry.get('space_group_0.1')}; accepted chain {best['accepted_chain']} "
                     f"(step-5 F {best['step5_free_energy_ev']} eV)")
    diamond = report["findings"].get("diamond")
    if diamond:
        lines.append(f"FINDING   diamond-like structure found: {diamond['found']}; lowest energy: "
                     f"{diamond['is_lowest_energy']} (recorded, not required)")
    candidates = report["findings"].get("candidates") or {}
    lines.append(f"FINDING   candidates {candidates.get('started')}, completed the ladder "
                 f"{candidates.get('completed_ladder')}, rejected {len(candidates.get('rejected') or {})}")
    return lines


if __name__ == "__main__":  # python analyze_execution.py EVIDENCE_DIR EXPECTED_JSON
    import sys

    result = analyse(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()))
    print("\n".join(summary_lines(result)))
    raise SystemExit(0 if result["result"] == "PASSED" else 1)
