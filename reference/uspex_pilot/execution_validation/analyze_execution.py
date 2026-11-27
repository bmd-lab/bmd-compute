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
JOURNAL_TIME_ROUNDING_S = 0.0005  # the journal's own "time" is rounded to milliseconds
WALL_MONOTONIC_TOLERANCE_S = 0.5  # srun_returned's wall-clock and monotonic durations must agree this well
MONOTONIC_ROUNDING_S = 1e-5  # the launcher rounds monotonic times to microseconds
RETURNED_FIELDS = ("observed_at", "observed_monotonic", "monotonic_seconds", "returncode", "stop_requested",
                   "killed", "launcher_instance")


_FIX_RND_SEED_LINE = re.compile(r"^\s*(-?[0-9]+)\s*:\s*fixRndSeed\s*$", re.MULTILINE)


def _seed_lines(text: str | None) -> list[int] | None:
    return None if text is None else [int(value) for value in _FIX_RND_SEED_LINE.findall(text)]


def _random_seed_check(manifest: dict, input_text: str | None, parameters_text: str | None, *,
                       planned: Any = None) -> tuple[str, dict]:
    """(status, detail): planned seed vs prepared_inputs.json vs INPUT.txt vs USPEX's copy (results1/Parameters.txt).

    ``planned`` is the seed in the frozen plan's spec (search_plan.json, digest-covered); without it the
    manifest's ``requested`` is used. VERIFIED means all of them agree and the seed was written once and read
    back from USPEX's copy; it says nothing about whether USPEX's random number generator reproduces the search
    (that needs a repeat run).
    """

    requested = planned if planned is not None else manifest.get("requested")
    written, read_back = _seed_lines(input_text), _seed_lines(parameters_text)
    detail = {"planned": planned, "prepared_inputs_requested": manifest.get("requested"),
              "prepared_inputs_written": manifest.get("written_in_input_txt"),
              "input_txt": written, "parameters_txt": read_back,
              "effective_rng_determinism": "not assessed by this analysis"}
    if type(requested) is not int or manifest.get("requested") != requested:
        return FAILED, detail
    if written is None or read_back is None:
        return MISSING, detail
    if not read_back:
        # How USPEX 25.0 writes the seed in its copy is not yet confirmed on a real run.
        detail["note"] = "no 'value : fixRndSeed' line found in results1/Parameters.txt"
        return MISSING, detail
    ok = (type(requested) is int and manifest.get("written_in_input_txt") == requested
          and written == [requested] and read_back == [requested])
    return (VERIFIED if ok else FAILED), detail


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


# -- how USPEX changed a structure between two relaxation steps (stdlib; reported, never judged) ----------------

# Below these a structure is treated as unchanged: the precision to which VASP and USPEX write
# structures (about 1e-6 in lattice and fractional coordinates), not a scientific tolerance.
WRITE_PRECISION_STRAIN = 1e-5
WRITE_PRECISION_DISPLACEMENT_A = 1e-4


def _det(m) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _inverse(m) -> list[list[float]]:
    d = _det(m)
    cof = [[m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3]
            - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3] for j in range(3)] for i in range(3)]
    return [[cof[i][j] / d for j in range(3)] for i in range(3)]


def _matmul(a, b) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(m) -> list[list[float]]:
    return [list(row) for row in zip(*m)]


def _symmetric_eigenvalues(c) -> list[float]:
    """Eigenvalues of a real symmetric 3x3 matrix, ascending (closed form)."""

    import math

    p1 = c[0][1] ** 2 + c[0][2] ** 2 + c[1][2] ** 2
    q = (c[0][0] + c[1][1] + c[2][2]) / 3
    if p1 <= 1e-30:
        return sorted([c[0][0], c[1][1], c[2][2]])
    p2 = sum((c[i][i] - q) ** 2 for i in range(3)) + 2 * p1
    p = math.sqrt(p2 / 6)
    b = [[(c[i][j] - (q if i == j else 0)) / p for j in range(3)] for i in range(3)]
    r = max(-1.0, min(1.0, _det(b) / 2))
    phi = math.acos(r) / 3
    high = q + 2 * p * math.cos(phi)
    low = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    return sorted([low, 3 * q - high - low, high])


def _wrapped(v):
    return [x - round(x) for x in v]


def _norm(v) -> float:
    return sum(x * x for x in v) ** 0.5


def _atom_displacements(a: dict, b_frac: list, b_species: list, lattice) -> tuple[float, float, float] | None:
    """Largest and RMS displacement (Å) of matched atoms with the rigid translation removed, and that
    translation (Å); None if the atoms cannot be matched. Ties go to the smallest translation."""

    a_species = [s for s, n in zip(a["species"], a["counts"]) for _ in range(n)]
    best = None
    first = a_species[0]
    for j in [j for j, s in enumerate(b_species) if s == first]:
        shift = [p - q for p, q in zip(b_frac[j], a["coords"][0])]
        unused = set(range(len(b_frac)))
        deltas = []
        for i, coord in enumerate(a["coords"]):
            options = [(k, _wrapped([p - q - t for p, q, t in zip(b_frac[k], coord, shift)])) for k in unused
                       if b_species[k] == a_species[i]]
            if not options:
                deltas = None
                break
            k, delta = min(options, key=lambda o: _norm(_matmul([o[1]], lattice)[0]))
            unused.discard(k)
            deltas.append(delta)
        if deltas is None:
            continue
        mean = [sum(d[x] for d in deltas) / len(deltas) for x in range(3)]
        lengths = [_norm(_matmul([_wrapped([d[x] - mean[x] for x in range(3)])], lattice)[0]) for d in deltas]
        translation = _norm(_matmul([_wrapped([t + m for t, m in zip(shift, mean)])], lattice)[0])
        candidate = (round(max(lengths), 9), round((sum(v * v for v in lengths) / len(lengths)) ** 0.5, 9),
                     round(translation, 9))
        if best is None or candidate < best:
            best = candidate
    return best


def structure_transformation(a: dict | None, b: dict | None, *, max_coefficient: int = 2) -> dict[str, Any]:
    """How structure ``b`` (a step's input) differs from ``a`` (its predecessor's output): measured, not judged.

    The cells are put in correspondence from the lattices alone: the unimodular integer combination
    of ``b``'s lattice vectors (coefficients up to ``max_coefficient``) whose metric is closest to
    ``a``'s. Reported: the volume change; the principal Green–Lagrange strains of that correspondence
    (rotation-free, so a rewritten or reoriented but otherwise identical cell has zero strain); and the
    displacements of matched atoms of the same species in the corresponding fractional coordinates,
    converted with ``b``'s cell, after removing the rigid translation (a uniform strain alone moves no
    atom). Ties go to the smaller displacement, then the correspondence keeping the cell's handedness,
    then the smaller translation. ``comparable`` is False, with a reason, when a structure is
    unreadable, the composition differs or no correspondence is found. The split between strain and
    displacement depends on the correspondence (least strain here; pymatgen's StructureMatcher
    minimises atomic RMS instead): see the README, "Structural continuity between ladder steps".
    """

    if a is None or b is None:
        return {"comparable": False, "reason": "a structure is unreadable"}
    composition = {"species": a["species"], "counts": a["counts"]}
    if a["species"] != b["species"] or a["counts"] != b["counts"]:
        return {"comparable": False, "reason": "the composition differs", "composition_same": False,
                "before": composition, "after": {"species": b["species"], "counts": b["counts"]}}
    la, lb = a["lattice"], b["lattice"]
    volume_a, volume_b = abs(_det(la)), abs(_det(lb))
    lengths_a = [_norm(row) for row in la]
    range_ = range(-max_coefficient, max_coefficient + 1)
    vectors = [(n, _matmul([list(n)], lb)[0]) for n in ((i, j, k) for i in range_ for j in range_ for k in range_)
               if n != (0, 0, 0)]
    per_axis = [sorted(vectors, key=lambda v: abs(_norm(v[1]) - length))[:12] for length in lengths_a]
    inverse_a = _inverse(la)
    candidates = []
    for c0 in per_axis[0]:
        for c1 in per_axis[1]:
            for c2 in per_axis[2]:
                m = [list(c0[0]), list(c1[0]), list(c2[0])]
                if round(abs(_det(m))) != 1:
                    continue
                lb_m = [c0[1], c1[1], c2[1]]
                c = _matmul(_matmul(inverse_a, _matmul(lb_m, _transpose(lb_m))), _transpose(inverse_a))
                strains = [0.5 * (x - 1) for x in _symmetric_eigenvalues(c)]
                candidates.append((max(abs(s) for s in strains), strains, m, lb_m))
    if not candidates:
        return {"comparable": False, "reason": "no lattice correspondence found", "composition_same": True}
    least = min(c[0] for c in candidates)
    b_species = [s for s, n in zip(b["species"], b["counts"]) for _ in range(n)]
    best = None
    for strain, strains, m, lb_m in candidates:
        if strain > least + 1e-9:
            continue
        inverse_m = _inverse(m)
        b_frac = [_matmul([coord], inverse_m)[0] for coord in b["coords"]]
        moved = _atom_displacements(a, b_frac, b_species, lb_m)
        # Ties (M and -M have the same metric) go to the correspondence that keeps the cell's handedness.
        improper = int((_det(lb_m) > 0) != (_det(la) > 0))
        # Order: displacement, then handedness, then the size of the rigid translation.
        if moved is not None and (best is None or (moved[0], moved[1], improper, moved[2]) < best[0]):
            best = ((moved[0], moved[1], improper, moved[2]), strains, m)
    if best is None:
        return {"comparable": False, "reason": "atoms could not be matched", "composition_same": True}
    (largest, rms, _, translation), strains, m = best
    return {"comparable": True, "composition_same": True, "composition": composition,
            "volume_before_a3": round(volume_a, 6), "volume_after_a3": round(volume_b, 6),
            "volume_change_relative": round((volume_b - volume_a) / volume_a, 8),
            "principal_strains": [round(s, 8) for s in strains],
            "max_abs_principal_strain": round(max(abs(s) for s in strains), 8),
            "max_atomic_displacement_a": round(largest, 6), "rms_atomic_displacement_a": round(rms, 6),
            "rigid_translation_a": round(translation, 6),
            "lattice_correspondence": [[int(round(x)) for x in row] for row in m],
            # A mirror image has the same metric: whether the correspondence keeps the handedness of the cell.
            "orientation_preserving": (_det(_matmul(m, lb)) > 0) == (_det(la) > 0)}


def _changed(transition: dict) -> bool | None:
    """Whether USPEX changed the structure (beyond write precision); None if not quantified."""

    if transition.get("identical_bytes"):
        return False
    if not transition.get("comparable"):
        return None
    return (transition["max_abs_principal_strain"] > WRITE_PRECISION_STRAIN
            or transition["max_atomic_displacement_a"] > WRITE_PRECISION_DISPLACEMENT_A)


def same_cell_in_another_basis(a: dict | None, b: dict | None,
                               *, tolerance_a: float = WRITE_PRECISION_DISPLACEMENT_A) -> dict[str, Any]:
    """Strict continuity: is ``b`` exactly ``a``, possibly written in another lattice basis?

    Established explicitly, never inferred from a match up to symmetry:

    1. the basis change is the integer matrix M = round(L_a L_b^-1); it must be unimodular
       (|det M| = 1), and M L_b must equal L_a vector by vector in the same Cartesian frame to within
       ``tolerance_a`` (Å). A strained, rotated or mirrored cell fails here, whatever its metric;
    2. every atom of ``a`` must have its own atom of ``b`` (same species, one to one), with the
       fractional coordinates of ``b`` taken into the basis of ``a`` (f_b M^-1) and differences
       reduced only by whole lattice vectors (periodic wrapping), within ``tolerance_a`` in Cartesian
       distance. No translation is removed, so an origin shift fails; so does any symmetry-equivalent
       structure whose atoms are elsewhere.
    """

    if a is None or b is None:
        return {"same": None, "reason": "a structure is unreadable"}
    if a["species"] != b["species"] or a["counts"] != b["counts"]:
        return {"same": False, "reason": "the composition differs"}
    la, lb = a["lattice"], b["lattice"]
    if abs(_det(lb)) < 1e-12 or abs(_det(la)) < 1e-12:
        return {"same": None, "reason": "a degenerate cell"}
    real = _matmul(la, _inverse(lb))
    m = [[round(x) for x in row] for row in real]
    if abs(_det(m)) != 1:
        return {"same": False, "reason": "the cell is not a unimodular basis change of its predecessor's",
                "basis_change": m}
    mapped = _matmul(m, lb)
    lattice_residual = max(abs(x - y) for row_m, row_a in zip(mapped, la) for x, y in zip(row_m, row_a))
    result = {"basis_change": m, "lattice_residual_a": round(lattice_residual, 9)}
    if lattice_residual > tolerance_a:
        return {**result, "same": False, "reason": "the cell differs from its predecessor's in the same Cartesian "
                                                   "frame (strained, rotated or mirrored)"}
    inverse_m = _inverse(m)
    b_frac = [_matmul([coord], inverse_m)[0] for coord in b["coords"]]
    b_species = [sp for sp, n in zip(b["species"], b["counts"]) for _ in range(n)]
    a_species = [sp for sp, n in zip(a["species"], a["counts"]) for _ in range(n)]
    unused, worst = set(range(len(b_frac))), 0.0
    for coord, species in zip(a["coords"], a_species):
        distances = sorted((_norm(_matmul([_wrapped([p - q for p, q in zip(b_frac[k], coord)])], la)[0]), k)
                           for k in unused if b_species[k] == species)
        if not distances or distances[0][0] > tolerance_a:
            return {**result, "same": False, "reason": "an atom has no counterpart at the same position (moved, "
                                                       "shifted with the origin, or elsewhere by symmetry)",
                    "max_atom_residual_a": round(distances[0][0], 6) if distances else None}
        worst = max(worst, distances[0][0])
        unused.discard(distances[0][1])
    return {**result, "same": True, "max_atom_residual_a": round(worst, 9),
            "reason": "the same cell and atoms" + ("" if m == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                                                    else ", written in another lattice basis")}


def _statistics(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"n": len(ordered), "min": ordered[0], "median": round(median, 8), "max": ordered[-1]}


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


INTERMEDIATE_NOTE = ("USPEX changed the structure between intermediate relaxation steps. The changes are "
                     "measured here; they are not assessed as physically negligible and are not validated "
                     "by this report. Accepted provisionally (decision D2, 17 Nov 2026) as part of the "
                     "experimental USPEX search methodology; not an independently validated USPEX algorithm.")
MAX_TRANSITION_ROWS = 2000


def intermediate_findings(transformations: list[dict]) -> dict[str, Any]:
    """The measured USPEX changes between intermediate steps, by transition, with per-transition rows."""

    intermediate = [t for t in transformations if not t["final_transition"]]
    by_pair: dict[str, Any] = {}
    for pair in sorted({tuple(t["steps"]) for t in intermediate}):
        rows = [t for t in intermediate if tuple(t["steps"]) == pair]
        quantified = [t for t in rows if t.get("comparable")]
        by_pair[f"{pair[0]}->{pair[1]}"] = {
            "transitions": len(rows), "quantified": len(quantified),
            "changed": sum(1 for t in rows if _changed(t)),
            "identical_bytes": sum(1 for t in rows if t["identical_bytes"]),
            "abs_volume_change_relative": _statistics([abs(t["volume_change_relative"]) for t in quantified]),
            "max_abs_principal_strain": _statistics([t["max_abs_principal_strain"] for t in quantified]),
            "max_atomic_displacement_a": _statistics([t["max_atomic_displacement_a"] for t in quantified]),
            "rigid_translation_a": _statistics([t["rigid_translation_a"] for t in quantified]),
        }
    return {"note": INTERMEDIATE_NOTE, "transitions": len(intermediate),
            "changed": sum(1 for t in intermediate if _changed(t)),
            "not_quantified": sum(1 for t in intermediate if not t.get("comparable") and not t["identical_bytes"]),
            "composition_changed": sum(1 for t in intermediate if t.get("composition_same") is False),
            "by_transition": by_pair, "rows": intermediate[:MAX_TRANSITION_ROWS],
            "rows_total": len(intermediate),
            "definitions": {
                "lattice_correspondence": "unimodular integer combination of the later cell's vectors closest "
                                          "in metric to the earlier cell (coefficients up to 2)",
                "principal_strains": "Green-Lagrange strain of that correspondence (rotation-free)",
                "atomic_displacement": "matched atoms of the same species, corresponding fractional coordinates "
                                       "converted with the later cell, rigid translation removed (a uniform "
                                       "strain alone moves no atom)",
                "changed": f"strain above {WRITE_PRECISION_STRAIN} or displacement above "
                           f"{WRITE_PRECISION_DISPLACEMENT_A} A (write precision, not a scientific tolerance)"}}


ARCHIVE = Path("search") / "bmd_evidence"
F_AGREEMENT_EV = 1e-4  # the checker's vasprun F against VASP's printed F (8 significant decimals)


def verify_archive(files: Path, manifest: dict, steps: list[dict], chains: dict[int, list[dict]],
                   complete: dict[int, dict], started: dict, finished: dict, validity_events: dict, done: dict,
                   logs: dict, journal: list[dict], *, final_step: int = 5) -> dict[str, Any]:
    """Check the launcher's evidence archive against the journal, done files, step logs and surviving CalcFolds.

    Each step the journal started must have one committed record whose files match their hashes, whose
    SHA-256 is in the archive index and equals the journal's ``evidence`` event and the done file's
    ``evidence=``, and whose identity (request, step, CalcFold, slot, command), outcome (return code,
    status) and verdict (classification, reasons) equal the launcher's other records. A surviving
    CalcFold's ``bmd_step_validity.jsonl`` line must be the archived verdict byte for byte, and its
    step-5 CONTCAR the archived one. The checker's F must equal VASP's printed F. The final step must
    start from exactly the structure its predecessor ended with; an intermediate step must keep its
    predecessor's candidate and composition, and how USPEX changed the structure is measured
    (``transformations``) and reported, never treated as negligible or as validated. Contradictions
    fail; absent, interrupted, unverifiable or unquantifiable records are missing evidence.
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
    returned_events: dict[str, list[dict]] = {}
    for e in journal:
        if e.get("event") == "srun_returned":
            returned_events.setdefault(e.get("id"), []).append(e)
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
        if (any(key in launch for key in ("popen", "popen_monotonic", "returned"))
                or any(key in start for key in ("popen_time", "popen_monotonic")) or request_id in returned_events):
            # A launcher with srun_returned: the archived launch timing must be the journal's, both or neither.
            journal_returned = [{k: e.get(k) for k in RETURNED_FIELDS} for e in returned_events.get(request_id, [])]
            pairs += (("popen", launch.get("popen"), start.get("popen_time")),
                      ("popen_monotonic", launch.get("popen_monotonic"), start.get("popen_monotonic")),
                      ("srun_returned", launch.get("returned"),
                       journal_returned[0] if len(journal_returned) == 1 else journal_returned or None))
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

    # Structural continuity. The final step (the one whose F is the candidate's energy) must start from
    # exactly the structure its predecessor ended with. Between intermediate steps USPEX may change the
    # structure (attempt 209898bc: strained cells and moved atoms on every 1->2, 2->3 and 3->4 transition):
    # there the candidate and composition must be kept, and the change is measured and reported, not judged.
    continuity, transformations, uncomparable, basis_rewrites = [], [], [], []
    for individual, chain in chains.items():
        ordered = sorted(chain, key=lambda s_: s_["step"] or 0)
        for before, after in zip(ordered, ordered[1:]):
            a, b = records.get(before["id"]), records.get(after["id"])
            if not a or not b:
                continue
            final = after["step"] == final_step
            identity = [_CALCFOLD.search(str(entry["record"].get("calcfold", ""))) for entry in (a, b)]
            # The step order itself is the ``chains`` check's; here, the archived records' candidate.
            if None in identity or int(identity[0].group(1)) != int(identity[1].group(1)):
                contradict(after["id"], "its archived candidate is not its predecessor's")
            out_sha = (a["record"].get("output_structure") or {}).get("sha256")
            in_sha = (b["record"].get("input_structure") or {}).get("sha256")
            output = parse_poscar(_text(root / "steps" / before["id"] / "CONTCAR"))
            given = parse_poscar(_text(root / "steps" / after["id"] / "POSCAR"))
            identical = bool(out_sha and in_sha and out_sha == in_sha)
            measured = structure_transformation(output, given)
            transformations.append({"individual": individual, "steps": [before["step"], after["step"]],
                                    "before": before["id"], "after": after["id"], "final_transition": final,
                                    "identical_bytes": identical, **measured})
            if final:
                same = True if identical else same_structure(output, given)
                if same is not True:
                    strict = same_cell_in_another_basis(output, given)
                    transformations[-1]["strict_continuity"] = strict
                    if strict["same"] is True:
                        same = True
                        basis_rewrites.append([before["id"], after["id"], strict["basis_change"]])
                    elif strict["same"] is None and same is False:
                        same = None
                if same is not True:
                    continuity.append([before["id"], after["id"]])
                    contradict(after["id"], "the final step did not start from its predecessor's output structure"
                               if same is False else "the final step's input cannot be compared with its "
                               "predecessor's output")
            elif measured.get("composition_same") is False:
                contradict(after["id"], "its input has a different composition from its predecessor's output")
            elif not identical and not measured["comparable"]:
                uncomparable.append([before["id"], after["id"], measured["reason"]])

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
                          or critical or uncomparable or not records)
              else VERIFIED)
    intermediate = [t for t in transformations if not t["final_transition"]]
    step_log_sha256 = {request_id: (entry["record"].get("step_log") or {}).get("sha256")
                       for request_id, entry in records.items()}
    return {"status": status, "verified": verified, "finals": finals, "transformations": transformations,
            "step_log_sha256": step_log_sha256,
            "detail": {
        "steps": len(steps), "records": len(records), "verified_steps": len(verified),
        "missing_records": sorted(set(absent))[:20], "write_failures": write_failures[:20],
        "archived_but_not_delivered": undelivered[:20], "launcher_critical_lines": critical[:20],
        "unverifiable": unverifiable[:20], "interrupted": interrupted[:20], "unfinished_writes": staging[:20],
        "conflicts": conflicts[:20], "torn_index_lines": torn,
        "final_step_continuity_breaks": continuity[:20], "final_step_continuity_breaks_total": len(continuity),
        "final_step_transitions": len(transformations) - len(intermediate),
        # The same cell written in another lattice basis (no strain, atom movement or shift beyond write
        # precision): attempt 209898bc had three such final-step inputs.
        "final_step_basis_rewrites": basis_rewrites[:20], "final_step_basis_rewrites_total": len(basis_rewrites),
        "intermediate_transitions": len(intermediate),
        "intermediate_transitions_changed": sum(1 for t in intermediate if _changed(t)),
        "intermediate_transitions_not_quantified": uncomparable[:20],
        "intermediate_transitions_not_quantified_total": len(uncomparable),
        "contradicted_steps_total": len(contradictions),
        "contradictions": {k: v for k, v in list(contradictions.items())[:20]},
        "final_structures": len(finals),
        "generation": "not part of execution identity: the launcher records none (USPEX does not pass it)"}}


# The job records, per package, whether it was imported in the job process (search_status.json "isolation").
REQUIRED_ISOLATION_KEYS = ("jobflow",)
# Imports that the job is documented to make: the package, and where and why.
DOCUMENTED_IMPORTS = {
    "atomate2": "backend/structure_search/potcar.py (_relax_generator_potcar_table) reads the POTCAR symbol "
                "table from atomate2.vasp.sets.core.RelaxSetGenerator, so that searches and BMD's relaxations "
                "cannot use different POTCARs; the job reaches it while checking its prepared inputs "
                "(verify_execution_binding -> render_input_text -> admission)",
}


def isolation_check(status: dict | None) -> tuple[str, dict[str, Any]]:
    """No workflow machinery in the job process; a documented import is reported, not counted as a failure.

    The record is one boolean per package: whether the package was imported at all. It does not say
    which of the package's modules were loaded or by what, so a documented import is accepted as
    attributed to its documented source, never as proven to be only that.
    """

    if status is None:
        return MISSING, {"reason": "no search_status.json"}
    isolation = status.get("isolation")
    if not isinstance(isolation, dict) or any(not isinstance(isolation.get(k), bool) for k in REQUIRED_ISOLATION_KEYS):
        return MISSING, {"recorded": isolation, "reason": f"the record does not state {list(REQUIRED_ISOLATION_KEYS)}"}
    unexpected = sorted(k for k, loaded in isolation.items() if loaded is not False and k not in DOCUMENTED_IMPORTS)
    documented = {k: DOCUMENTED_IMPORTS[k] for k, loaded in sorted(isolation.items())
                  if loaded is True and k in DOCUMENTED_IMPORTS}
    malformed = sorted(k for k in DOCUMENTED_IMPORTS if k in isolation and not isinstance(isolation[k], bool))
    detail = {"recorded": isolation, "unexpected_workflow_machinery": unexpected,
              "documented_imports_recorded": documented,
              "limitation": "one boolean per package: whether it was imported at all, not which modules or why; "
                            "a documented import is attributed to its documented source, not proven"}
    return (FAILED if unexpected or malformed else VERIFIED), detail


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


def _utc_epoch(text: str | None) -> float | None:
    from datetime import datetime, timezone

    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _maximum_matching(edges: dict[str, list[str]], *, removed_steps: frozenset = frozenset(),
                      removed_slurm: frozenset = frozenset()) -> dict[str, str]:
    """A maximum matching of launcher steps to SLURM steps (augmenting paths), without the removed ones."""

    owner: dict[str, str] = {}

    def augment(step: str, seen: set) -> bool:
        for candidate in edges[step]:
            if candidate in removed_slurm or candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in owner or augment(owner[candidate], seen):
                owner[candidate] = step
                return True
        return False

    for step in edges:
        if step not in removed_steps:
            augment(step, set())
    return {step: candidate for candidate, step in owner.items()}


def map_slurm_steps(steps: list[dict], sacct_steps: list[list[str]], *, tolerance_s: float = 2.0) -> dict[str, Any]:
    """Which SLURM steps (sacct) the launcher steps can have run as, over every feasible matching.

    A SLURM step is a candidate for a launcher step if its interval (whole seconds, UTC) lies inside the
    launcher's (to ``tolerance_s``): the launcher starts before srun and finishes after srun exits and the
    step's validity is checked. The feasible matchings are the maximum one-to-one matchings of launcher
    steps to candidates. Returned: ``edges`` and ``size`` (of a maximum matching) for joint queries,
    ``intervals`` per SLURM step, and ``candidates``: for each launcher step matched in *every* maximum
    matching, the SLURM steps it is matched to in at least one. A step missing from ``candidates`` has
    no SLURM step in some feasible matching.
    """

    intervals = {}
    for row in sacct_steps:
        start, end = _utc_epoch(row[4]), _utc_epoch(row[5])
        if start is not None and end is not None:
            intervals[row[0]] = (start, end)
    edges: dict[str, list[str]] = {}
    for s_ in sorted((s_ for s_ in steps if s_["start"] is not None and s_["end"] is not None),
                     key=lambda s_: (s_["start"], s_["id"])):
        edges[s_["id"]] = [job_step for _, job_step in sorted(
            (abs(start - s_["start"]), job_step) for job_step, (start, end) in intervals.items()
            if s_["start"] - 1 - tolerance_s <= start <= s_["end"] and end <= s_["end"] + tolerance_s)]
    return feasible_mapping(edges, intervals)


def feasible_mapping(edges: dict[str, list[str]], intervals: dict[str, tuple]) -> dict[str, Any]:
    """The mapping structure of ``map_slurm_steps`` from candidate edges and SLURM intervals."""

    size = len(_maximum_matching(edges))
    candidates: dict[str, list[str]] = {}
    for step in edges:
        if len(_maximum_matching(edges, removed_steps=frozenset([step]))) != size - 1:
            continue  # some maximum matching leaves this step without a SLURM step
        kept = [c for c in edges[step] if _jointly_feasible(edges, size, {step: c})]
        if kept:
            candidates[step] = kept
    return {"edges": edges, "size": size, "intervals": intervals, "candidates": candidates}


def _jointly_feasible(edges: dict[str, list[str]], size: int, forced: dict[str, str]) -> bool:
    """Whether some maximum matching assigns every step in ``forced`` to its SLURM step."""

    if len(set(forced.values())) != len(forced) or any(c not in edges.get(s, ()) for s, c in forced.items()):
        return False
    rest = _maximum_matching(edges, removed_steps=frozenset(forced), removed_slurm=frozenset(forced.values()))
    return len(rest) == size - len(forced)


def _overlap_verdict(mapping: dict[str, Any], first: str, second: str) -> str:
    """``separated``, ``overlapping`` (by at least a second) or ``unresolved`` for two launcher steps.

    Decided over every feasible matching: only pairings of SLURM steps that some maximum matching uses
    for both steps at once are considered, and the verdict must be the same for all of them. With whole
    seconds, ``ea < sb`` proves the steps separated and ``ea > sb`` with ``eb > sa`` proves an overlap;
    equal seconds settle nothing.
    """

    candidates = mapping["candidates"]
    if first not in candidates or second not in candidates:
        return "unresolved"
    verdicts = set()
    for job_a in candidates[first]:
        for job_b in candidates[second]:
            if not _jointly_feasible(mapping["edges"], mapping["size"], {first: job_a, second: job_b}):
                continue
            (sa, ea), (sb, eb) = mapping["intervals"][job_a], mapping["intervals"][job_b]
            verdicts.add("separated" if ea < sb or eb < sa else "unresolved" if ea == sb or eb == sa
                         else "overlapping")
    return verdicts.pop() if len(verdicts) == 1 else "unresolved"


def pinning_table(rows: list | None, ranks: int) -> tuple[dict | None, str | None, dict | None]:
    """One step's Intel MPI table: ``(placement, None, overlap)`` if interpretable, else ``(None, reason, None)``.

    Interpretable: ranks 0..ranks-1 each exactly once, distinct PIDs, one host, every rank with a
    non-empty CPU set. ``overlap`` lists CPUs given to more than one rank of the step (None if none).
    """

    rows = rows or []
    if not rows:
        return None, "no pinning table", None
    numbers = [r[0] for r in rows]
    if len(numbers) != len(set(numbers)):
        return None, "a rank is listed more than once", None
    if sorted(numbers) != list(range(ranks)):
        return None, f"ranks {sorted(numbers)[:8]}{'...' if len(numbers) > 8 else ''} are not 0..{ranks - 1}", None
    if len({r[1] for r in rows}) != len(rows):
        return None, "two ranks have the same PID", None
    hosts = {r[2] for r in rows}
    if len(hosts) != 1:
        return None, f"ranks on {len(hosts)} hosts", None
    try:
        cpu_sets = {r[0]: _cpus(str(r[3])) for r in rows}
    except ValueError:
        return None, "an unreadable CPU set", None
    if not all(cpu_sets.values()):
        return None, "a rank without CPUs", None
    owners: dict[int, list[int]] = {}
    for rank, cpus in cpu_sets.items():
        for cpu in cpus:
            owners.setdefault(cpu, []).append(rank)
    shared = {cpu: sorted(r) for cpu, r in owners.items() if len(r) > 1}
    placement = {"host": next(iter(hosts)), "cpus": set(owners), "by_rank": cpu_sets}
    return placement, None, ({"cpus": sorted(shared)[:8], "ranks": sorted({x for r in shared.values() for x in r})[:8]}
                             if shared else None)


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def launch_timing(journal: list[dict], started: dict, finished: dict) -> tuple[dict[str, dict[str, Any]], dict]:
    """Each step's ``srun`` timing on the launcher's own clocks, where the evidence is complete and consistent.

    From ``started_step`` (``popen_time``, ``popen_monotonic``: just before ``srun`` was started) and the
    ``srun_returned`` event (``observed_at``, ``observed_monotonic``: when the launcher observed ``srun``'s
    exit, before the validity check), both naming the same ``launcher_instance``. This bounds the step's MPI
    tasks only if ``srun`` exited normally: an ``srun`` the launcher asked to stop or killed, or that died from
    a signal, may return while its tasks still run, so such steps get no timing. It is the launcher's
    observation of ``srun`` returning, not the moment SLURM released the step's CPUs.

    Steps from launchers that predate the event have none of these fields and get no timing (the analysis
    then uses only the earlier evidence). Anything missing or inconsistent makes the step's timing unusable
    and never makes a verdict more favourable: a duplicate or missing event, another launcher instance, a
    negative duration, a monotonic duration that is not ``observed_monotonic - popen_monotonic``, wall-clock
    and monotonic durations that disagree by more than ``WALL_MONOTONIC_TOLERANCE_S``, a wall-clock time
    outside the step's journal interval, or the event before ``started_step`` or after the step's verdict or
    ``finished`` in the journal. Returns ``(usable, unusable)``: ``usable`` maps a step to its timing
    (wall-clock and monotonic times, launcher instance, journal positions); ``unusable`` to the reason.
    """

    positions: dict[str, dict[str, list[int]]] = {}
    returned: dict[str, list[dict]] = {}
    for index, event in enumerate(journal):
        name = event.get("event")
        if name in ("started_step", "srun_returned", "validity", "finished"):
            positions.setdefault(event.get("id"), {}).setdefault(name, []).append(index)
        if name == "srun_returned":
            returned.setdefault(event.get("id"), []).append(event)
    usable: dict[str, dict[str, Any]] = {}
    unusable: dict[str, str] = {}
    for request_id in sorted(set(started) | set(returned)):
        start, end, events = started.get(request_id), finished.get(request_id), returned.get(request_id, [])
        if not events and not any(key in (start or {}) for key in ("popen_time", "popen_monotonic")):
            continue  # a launcher from before srun_returned: no timing evidence, the earlier treatment applies
        start = start or {}
        event = events[0] if len(events) == 1 else {}
        popen, popen_mono = start.get("popen_time"), start.get("popen_monotonic")
        observed, observed_mono = event.get("observed_at"), event.get("observed_monotonic")
        duration, returncode = event.get("monotonic_seconds"), event.get("returncode")
        instance = start.get("launcher_instance")
        where = positions.get(request_id, {})
        later = [where[name][0] for name in ("validity", "finished") if name in where]
        if request_id not in started:
            reason = "srun_returned without started_step"
        elif len(events) != 1:
            reason = f"{len(events)} srun_returned events"
        elif not all(_number(v) for v in (popen, popen_mono, start.get("time"))):
            reason = "started_step has no numeric popen times"
        elif not all(_number(v) for v in (observed, observed_mono, duration)):
            reason = "srun_returned has no numeric times"
        elif not (isinstance(instance, str) and instance and event.get("launcher_instance") == instance):
            reason = "started_step and srun_returned are not from one launcher instance"
        elif end is None:
            reason = "no finished event"
        elif not isinstance(returncode, int) or isinstance(returncode, bool) or returncode < 0:
            reason = "srun died from a signal: its tasks may have outlived it"
        elif event.get("killed") is not False or event.get("stop_requested") is not False:
            reason = "the launcher stopped or killed the step: srun's return need not end its tasks"
        elif returncode != end.get("returncode"):
            reason = "return code differs from the finished event"
        elif duration < 0 or observed_mono < popen_mono:
            reason = "negative monotonic duration"
        elif abs(duration - (observed_mono - popen_mono)) > MONOTONIC_ROUNDING_S:
            reason = "monotonic duration is not observed_monotonic - popen_monotonic"
        elif abs((observed - popen) - duration) > WALL_MONOTONIC_TOLERANCE_S:
            reason = "wall-clock and monotonic durations disagree"
        elif not (popen <= start["time"] + JOURNAL_TIME_ROUNDING_S and popen <= observed):
            reason = "popen_time is after the step started or after srun returned"
        elif not (_number(end.get("time")) and observed <= end["time"] + JOURNAL_TIME_ROUNDING_S):
            reason = "srun_returned is after the step finished"
        elif not later or not (where["started_step"][0] < where["srun_returned"][0]
                               and all(where["srun_returned"][0] < i for i in later)):
            reason = "srun_returned is not between started_step and the step's verdict in the journal"
        else:
            usable[request_id] = {"popen": float(popen), "observed": float(observed), "popen_monotonic":
                                  float(popen_mono), "observed_monotonic": float(observed_mono), "instance": instance,
                                  "started_index": where["started_step"][0],
                                  "returned_index": where["srun_returned"][0]}
            continue
        unusable[request_id] = reason
    return usable, unusable


def launcher_order(first: dict | None, second: dict | None) -> str | None:
    """Whether the launcher's own records show one step's ``srun`` returning before the other's started.

    ``separated`` needs all three, for one of the two orders: the same launcher instance and the earlier
    step's ``observed_monotonic`` strictly before the later step's ``popen_monotonic`` (one process, one
    monotonic clock, which a wall-clock adjustment cannot move), the same order on the wall clock, and the
    same order in the journal (the launcher journals ``srun_returned`` before it starts any further step).
    ``disagree`` if only some of them say so (for example after a wall-clock step); None if none do, or if
    the steps lack usable timing or come from different launcher instances. Only ``separated`` settles
    anything.
    """

    if not first or not second or first["instance"] != second["instance"]:
        return None
    verdicts = set()
    for a, b in ((first, second), (second, first)):
        votes = (a["observed_monotonic"] < b["popen_monotonic"], a["observed"] < b["popen"],
                 a["returned_index"] < b["started_index"])
        if all(votes):
            return "separated"
        if any(votes):
            verdicts.add("disagree")
    return "disagree" if verdicts else None


def assess_pinning(steps: list[dict], sacct_steps: list[list[str]], *, ranks: int) -> tuple[str, dict[str, Any]]:
    """Whether VASP steps that ran at the same time used disjoint CPUs, from Intel MPI's tables.

    Every step that ran (return code 0) needs an interpretable table; a step that failed but left one
    is compared too. CPUs shared within a step fail. Two steps sharing CPUs on one host fail only if their
    SLURM steps genuinely overlapped: never if the launcher intervals are disjoint or the SLURM intervals
    are separated; an overlap that cannot be established at sacct's one-second resolution, or for steps
    that cannot be mapped to SLURM steps, is ``missing``. Reuse of CPUs by steps that did not overlap is
    expected and never rejected. Disjoint launcher intervals prove disjoint SLURM steps on the launcher's
    own clock (each SLURM step runs inside its srun, which runs inside the launcher's interval); otherwise
    the verdict must hold under every feasible matching of launcher steps to SLURM steps.

    Where both steps carry usable ``process`` timing (``launch_timing``) and ``launcher_order`` finds one
    step's ``srun`` seen to return normally strictly before the other's ``srun`` was started (on the same
    launcher's monotonic clock, its wall clock and its journal), the pair is sequential reuse: the earlier
    step's tasks had ended before the later step existed. That is an inference about when the processes ran,
    not about when SLURM released the CPUs. If SLURM's intervals nevertheless show the two steps overlapping by
    a second or more, the pair is unresolved. If the launcher's measures disagree, launcher timing settles
    nothing for the pair (it is counted) and the earlier treatment decides. Without usable timing the earlier
    treatment applies unchanged.
    """

    required = [s_ for s_ in steps if s_["returncode"] == 0]
    placements, uninterpretable, reasons, own_overlap = {}, [], {}, []
    for s_ in steps:
        placement, reason, shared = pinning_table(s_.get("pin_rows"), ranks)
        if placement is None:
            if s_["returncode"] == 0:
                uninterpretable.append(s_["id"])
                reasons[reason] = reasons.get(reason, 0) + 1
            continue
        if shared:
            own_overlap.append([s_["id"], shared])
        placements[s_["id"]] = {**placement, "start": s_["start"], "end": s_["end"], "process": s_.get("process")}
    mapping = map_slurm_steps(steps, sacct_steps)
    conflicts, unresolved, reused, reused_by_return, disagreements = [], [], 0, 0, []
    ids = sorted(placements, key=lambda i: (placements[i]["start"] or 0, i))
    for index, first in enumerate(ids):
        a_ = placements[first]
        for second in ids[index + 1:]:
            b_ = placements[second]
            common = a_["cpus"] & b_["cpus"]
            if a_["host"] != b_["host"] or not common:
                continue
            if None in (a_["start"], a_["end"], b_["start"], b_["end"]):
                unresolved.append([first, second, "no launcher interval"])
                continue
            if a_["end"] <= b_["start"] or b_["end"] <= a_["start"]:
                reused += 1  # the launcher intervals are disjoint: SLURM's lie inside them
                continue
            order = launcher_order(a_["process"], b_["process"])
            if order == "disagree":
                disagreements.append([first, second])
            elif order == "separated":
                if (first in mapping["candidates"] and second in mapping["candidates"]
                        and _overlap_verdict(mapping, first, second) == "overlapping"):
                    unresolved.append([first, second, "srun_returned separates the steps but SLURM's intervals "
                                                      "overlap: the timings disagree"])
                    continue
                reused += 1  # one srun was seen to return before the other was started
                reused_by_return += 1
                continue
            if first not in mapping["candidates"] or second not in mapping["candidates"]:
                unresolved.append([first, second, "not mapped to SLURM steps in every feasible matching"])
                continue
            verdict = _overlap_verdict(mapping, first, second)
            if verdict == "separated":
                reused += 1
            elif verdict == "unresolved":
                unresolved.append([first, second, "SLURM intervals not established as overlapping or separated "
                                                  "(one-second resolution or several possible SLURM steps)"])
            else:
                conflicts.append([first, second, sorted(common)[:8],
                                  [mapping["candidates"][first], mapping["candidates"][second]]])
    status = (FAILED if conflicts or own_overlap else
              MISSING if uninterpretable or unresolved or not required else VERIFIED)
    timed = [s_ for s_ in steps if s_.get("process") is not None or s_.get("process_unusable")]
    extra = {} if not timed else {"srun_returned": {
        "steps_with_process_interval": sum(1 for s_ in steps if s_.get("process") is not None),
        "steps_with_unusable_timing": sum(1 for s_ in steps if s_.get("process_unusable")),
        "unusable_reasons": _counts(s_.get("process_unusable") for s_ in steps if s_.get("process_unusable")),
        "reuse_settled_by_srun_returned": reused_by_return,
        "pairs_whose_launcher_timings_disagree": len(disagreements),
        "basis": "the launcher's observation that srun returned (before the validity check), not the moment "
                 "SLURM released the step's CPUs"}}
    return status, {**extra,
        "steps_ran": len(required), "interpretable": sum(1 for s_ in required if s_["id"] in placements),
        "uninterpretable": uninterpretable[:10], "uninterpretable_total": len(uninterpretable),
        "uninterpretable_reasons": reasons,
        "failed_steps_compared": sorted(s_["id"] for s_ in steps if s_["returncode"] != 0 and s_["id"] in placements),
        "slurm_steps_mapped": sum(1 for c in mapping["candidates"].values() if len(c) == 1),
        "slurm_steps_with_several_candidates": sum(1 for c in mapping["candidates"].values() if len(c) > 1),
        "launcher_steps_not_always_matched": len(mapping["edges"]) - len(mapping["candidates"]),
        "slurm_matching": "every maximum one-to-one matching of launcher steps to SLURM steps inside their "
                          "intervals; a verdict holds only if it is the same under all of them",
        "overlaps": conflicts[:10], "overlaps_total": len(conflicts),
        "unresolved": unresolved[:10], "unresolved_total": len(unresolved),
        "cpu_reuse_between_non_overlapping_steps": reused,
        "ranks_sharing_a_core": own_overlap[:10],
        "hosts": sorted({p["host"] for p in placements.values()}),
        "distinct_cpus": len(set().union(*[p["cpus"] for p in placements.values()])) if placements else 0}


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def merge_pinning_supplement(steps: list[dict], logs: dict, archived_sha256: dict, supplement: dict | None) -> dict:
    """Use a supplement's pinning table for a step only if its log hash and size equal the collector's and
    the archived record's; a step's own collected table is kept, and a supplement that differs from it is
    reported."""

    used, refused, disagreements = [], [], []
    collected = sum(1 for s_ in steps if s_.get("pin_rows"))
    if supplement:
        if supplement.get("schema") != "bmd_compute.uspex_pinning_supplement":
            return {"collector": collected, "supplement": 0, "refused": ["not a pinning supplement"],
                    "disagreements": [], "supplement_inspection_sha256": None}
        rows = supplement.get("logs") or {}
        for s_ in steps:
            entry, log = rows.get(s_["id"]), logs.get(s_["id"])
            if entry is None:
                continue
            if (log is None or entry.get("sha256") != log.get("sha256") or entry.get("bytes") != log.get("bytes")
                    or archived_sha256.get(s_["id"]) != entry.get("sha256")):
                refused.append(s_["id"])
                continue
            table = [list(r) for r in entry.get("pin_rows") or []]
            if s_.get("pin_rows"):
                if [list(r) for r in s_["pin_rows"]] != table:
                    disagreements.append(s_["id"])
                continue
            s_["pin_rows"] = table
            used.append(s_["id"])
        refused += sorted(set(rows) - {s_["id"] for s_ in steps})
    return {"collector": collected, "supplement": len(used), "refused": refused[:20], "refused_total": len(refused),
            "disagreements": disagreements[:20],
            "supplement_inspection_sha256": (supplement or {}).get("inspection_sha256")}


def placement_diagnostics(summaries: list[dict | None]) -> dict[str, Any]:
    """What the step logs show about CPU placement beyond the pinning rows (collector fields, if any).

    Evidence collected before these fields existed (attempt 209898bc) cannot tell whether Intel MPI
    printed nothing or printed in a format the pinning pattern does not read; that is stated, not guessed.
    """

    present = [s for s in summaries if s is not None]
    with_fields = [s for s in present if "mpi_startup_lines" in s]
    if not with_fields:
        return {"collected": False, "steps": len(present),
                "reason": "the collector recorded only pinning rows that matched its pattern; whether Intel MPI "
                          "printed nothing or printed in another format cannot be told from this evidence"}
    def count(key):
        return sum(1 for s in with_fields if s.get(key))
    rows = [s for s in with_fields if s.get("pin_rows")]
    other = [s for s in with_fields if s.get("mpi_startup_lines") and not s.get("pin_rows")]
    return {"collected": True, "steps": len(present), "steps_with_fields": len(with_fields),
            "steps_with_intel_mpi_version_line": count("intel_mpi_version"),
            "steps_with_mpi_startup_lines": count("mpi_startup_lines"),
            "steps_with_pinning_rows": len(rows),
            "steps_with_mpi_startup_lines_but_no_pinning_rows": len(other),
            "steps_with_slurm_cpu_bind_lines": count("cpu_bind_lines"),
            "interpretation": ("no Intel MPI startup output in the step logs" if not count("mpi_startup_lines")
                               else "Intel MPI startup output not in the recognised pinning format" if other
                               else "Intel MPI pinning rows present"),
            "samples": [line for s in with_fields for line in (s.get("placement_samples") or [])][:12]}


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


def analyse(evidence: Path, expected: dict[str, Any], *, pinning_supplement: dict | None = None) -> dict[str, Any]:
    """The report for one evidence directory. ``expected`` comes from the validation's own records.

    ``pinning_supplement`` (from ``compare_step_log_inspection.py --supplement``): Intel MPI pinning tables
    read later from the original step logs on POWER; a table is used only for a log whose SHA-256 and size
    equal the collector's and the archived record's.
    """

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
        ncpus = expected.get("ncpus", expected["parallel"] * expected["ranks"])  # from the frozen plan
        report.add("allocation", "the job held the frozen plan's CPUs on one node for the acknowledged walltime",
                   VERIFIED if main[7] == str(ncpus) and main[9] == expected["walltime"] and "node=1" in main[8]
                   and ncpus == expected["parallel"] * expected["ranks"] else FAILED,
                   {"ncpus": main[7], "expected_ncpus": ncpus, "alloc_tres": main[8], "timelimit": main[9],
                    "exclusive_requested": expected.get("exclusive")})

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
        planned_mb = _memory_mb(f"mem={expected.get('job_memory') or '0'}")
        consistent = (node_memory.get("mb", 0) > 0 and launch.get("step_mem") == step_mem
                      and launch.get("mode") == "srun_steps"
                      and launch.get("parallel_relaxations") in (None, expected["parallel"])
                      # an explicit --mem: the launcher's amount is exactly the plan's
                      and (not planned_mb or node_memory.get("mb") == planned_mb))
        status_value = (FAILED if not consistent or (alloc_mb is not None and alloc_mb != node_memory["mb"])
                        else MISSING if alloc_mb is None else VERIFIED)
        report.add("memory", "the allocated node memory and the per-step memory", status_value,
                   {"node_memory": node_memory, "step_mem": launch.get("step_mem"), "expected_step_mem": step_mem,
                    "sacct_alloc_mem_mb": alloc_mb, "planned_job_memory_mb": planned_mb or None})

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

    # The USPEX random seed: only for searches planned or prepared with one (older evidence gets no such check).
    seed_manifest = prepared_inputs.get("random_seed")
    stored_plan = _json(files / "search" / "search_plan.json") or {}
    plan_seed = ((stored_plan.get("search") or {}).get("spec") or {}).get("random_seed")
    if isinstance(seed_manifest, dict) or plan_seed is not None:
        report.add("random_seed", "the planned fixRndSeed was written to INPUT.txt and reached USPEX's "
                   "Parameters.txt (reproducibility of the search itself is not assessed here)",
                   *_random_seed_check(seed_manifest if isinstance(seed_manifest, dict) else {},
                                       _text(input_text) if input_text.is_file() else None,
                                       _text(files / "search" / "results1" / "Parameters.txt"),
                                       planned=plan_seed))

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
    process, process_unusable = launch_timing(journal, started, finished)
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
            "process": process.get(request_id), "process_unusable": process_unusable.get(request_id),
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
    ncpus_planned = expected.get("ncpus", expected["parallel"] * expected["ranks"])
    steps_what = (f"VASP ran as SLURM steps of {expected['ranks']} MPI ranks each (up to {expected['parallel']} at "
                  f"once in the {ncpus_planned}-CPU job allocation) with the per-step memory")
    if not steps or not main:
        report.add("vasp_steps", steps_what, MISSING)
    else:
        sound = (srun_steps and len(srun_steps) == len(steps) and ranks_ok and mem_ok and not no_energy
                 and len(sacct_steps) >= len(srun_steps) and all(r[7] == str(expected["ranks"]) for r in sacct_steps))
        report.add("vasp_steps", steps_what,
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
                             journal, final_step=expected["steps"])
    report.add("evidence_archive", "every step's archived record agrees with the journal, done file, step log "
               "and surviving CalcFolds; the final step started from its predecessor's output structure; "
               "intermediate steps kept their candidate and composition", archive["status"], archive["detail"])
    findings["intermediate_transformations"] = intermediate_findings(archive["transformations"])

    # Pinning: Intel MPI's table for every step, from the collector or from a hash-verified supplement.
    pin_source = merge_pinning_supplement(steps, logs, archive["step_log_sha256"], pinning_supplement)
    pinning_status, pinning_detail = assess_pinning(steps, sacct["steps"], ranks=expected["ranks"])
    pinning_detail["sources"] = pin_source
    pinning_detail["placement_diagnostics"] = placement_diagnostics([logs.get(s_["id"]) for s_ in ran])
    if pin_source["disagreements"]:
        pinning_status = FAILED if pinning_status == FAILED else MISSING
    report.add("pinning", f"concurrent VASP steps ({expected['ranks']} MPI ranks each, up to {expected['parallel']} "
               f"at once in the {ncpus_planned}-CPU allocation) used disjoint cores (Intel MPI pinning report)",
               pinning_status, pinning_detail)

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
                   and status.get("plan_digest") == expected["plan_digest"] and status.get("finished_at") else FAILED,
                   {"outcome": execution.get("outcome"), "master_exit_status": execution.get("master_exit_status"),
                    "finished_at": status.get("finished_at")})
    isolation_status, isolation_detail = isolation_check(status)
    report.add("isolation", "the job process loaded no workflow machinery (jobflow or any other recorded package); "
               "a recorded atomate2 import is attributed to the documented POTCAR-table import, not proven by the "
               "record", isolation_status, isolation_detail)
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
    moved = report["findings"].get("intermediate_transformations") or {}
    if moved.get("transitions"):
        worst = {key: max(((p.get(key) or {}).get("max") or 0) for p in moved["by_transition"].values())
                 for key in ("max_abs_principal_strain", "abs_volume_change_relative", "max_atomic_displacement_a")}
        lines.append(f"FINDING   USPEX changed the structure between intermediate steps in {moved['changed']} of "
                     f"{moved['transitions']} transitions (largest principal strain "
                     f"{worst['max_abs_principal_strain']:.3g}, volume change {worst['abs_volume_change_relative']:.3g}, "
                     f"atomic displacement {worst['max_atomic_displacement_a']:.3g} A; not quantified "
                     f"{moved['not_quantified']}): measured, not judged negligible or validated; accepted "
                     f"provisionally as experimental methodology")
    candidates = report["findings"].get("candidates") or {}
    lines.append(f"FINDING   candidates {candidates.get('started')}, completed the ladder "
                 f"{candidates.get('completed_ladder')}, rejected {len(candidates.get('rejected') or {})}")
    return lines


if __name__ == "__main__":  # python analyze_execution.py EVIDENCE_DIR EXPECTED_JSON [PINNING_SUPPLEMENT_JSON]
    import sys

    supplement = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None
    result = analyse(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), pinning_supplement=supplement)
    print("\n".join(summary_lines(result)))
    raise SystemExit(0 if result["result"] == "PASSED" else 1)
