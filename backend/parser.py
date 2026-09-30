from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure


POSCAR_ELEMENT_ERROR = (
    "Could not determine the chemical elements from the POSCAR."
)
POSCAR_ELEMENT_SUGGESTION = (
    "Please ensure the POSCAR contains a valid element-symbol line "
    "(e.g. Mg O) immediately before the atom counts."
)
INVALID_POSCAR_ERROR = (
    "The uploaded POSCAR is not valid. Please check that it follows the "
    "standard VASP POSCAR format."
)


class StructureValidationError(ValueError):
    """
    User-correctable structure input error.
    """

    def __init__(self, message: str, *, suggestion: str | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion


def parse_structure(text: str, fmt: str = "poscar") -> "Structure":
    """
    Parse a structure from text.

    Parameters
    ----------
    text
        POSCAR or CIF text.

    fmt
        "poscar" or "cif"

    Returns
    -------
    pymatgen.Structure
    """

    text = text.strip()

    from pymatgen.core import Structure

    if fmt.lower() == "poscar":
        try:
            validate_poscar_element_symbols(text)
            return Structure.from_str(text, fmt=fmt)
        except StructureValidationError:
            raise
        except Exception as exc:
            raise StructureValidationError(INVALID_POSCAR_ERROR) from exc

    return Structure.from_str(text, fmt=fmt)


def validate_poscar_element_symbols(text: str) -> None:
    """
    Ensure a POSCAR explicitly contains valid element symbols before counts.

    pymatgen supports VASP 4-style POSCARs without an element line by falling
    back to inferred placeholder species such as H, He and Li. BMD Compute
    rejects those inputs because they can silently launch the wrong chemistry.
    """

    from pymatgen.core.periodic_table import Element

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 7:
        raise StructureValidationError(
            POSCAR_ELEMENT_ERROR,
            suggestion=POSCAR_ELEMENT_SUGGESTION,
        )

    symbol_lines = []
    counts_start = None

    for offset, line in enumerate(lines[5:15], start=5):
        tokens = line.split()
        if _all_ints(tokens):
            counts_start = offset
            break

        symbol_lines.append(tokens)

    if not symbol_lines or counts_start is None:
        raise StructureValidationError(
            POSCAR_ELEMENT_ERROR,
            suggestion=POSCAR_ELEMENT_SUGGESTION,
        )

    count_lines = lines[counts_start : counts_start + len(symbol_lines)]
    if len(count_lines) != len(symbol_lines):
        raise StructureValidationError(
            POSCAR_ELEMENT_ERROR,
            suggestion=POSCAR_ELEMENT_SUGGESTION,
        )

    counts = []
    for line in count_lines:
        tokens = line.split()
        if not _all_ints(tokens):
            raise StructureValidationError(
                POSCAR_ELEMENT_ERROR,
                suggestion=POSCAR_ELEMENT_SUGGESTION,
            )
        counts.extend(tokens)

    symbols = [
        _poscar_symbol_token_to_element(token)
        for line in symbol_lines
        for token in line
    ]

    if len(symbols) != len(counts):
        raise StructureValidationError(
            POSCAR_ELEMENT_ERROR,
            suggestion=POSCAR_ELEMENT_SUGGESTION,
        )

    if not all(Element.is_valid_symbol(symbol) for symbol in symbols):
        raise StructureValidationError(
            POSCAR_ELEMENT_ERROR,
            suggestion=POSCAR_ELEMENT_SUGGESTION,
        )


def _all_ints(tokens: list[str]) -> bool:
    if not tokens:
        return False

    try:
        for token in tokens:
            int(token)
        return True
    except ValueError:
        return False


def _poscar_symbol_token_to_element(token: str) -> str:
    return token.split("/")[0].split("_")[0]


def structure_from_spec(structure_spec: dict) -> "Structure":
    """
    Reconstruct a pymatgen Structure from the serialized structure portion of a
    SubmissionSpec.

    The web workflow always submits ``pasted_text``; the input-reference
    producer also accepts ``builder``. The ``path`` and ``mp`` (Materials
    Project fetch) types are retained for hand-built specifications only: the
    web workflow does not use them and the input-reference producer rejects
    them.
    """

    from pymatgen.core import Lattice, Structure

    kind = structure_spec.get("type")
    if kind == "path":
        return Structure.from_file(structure_spec["path"])

    if kind == "pasted_text":
        fmt = structure_spec.get("format") or "poscar"
        return parse_structure(structure_spec["text"], fmt=fmt)

    if kind == "builder":
        import numpy as np

        lattice_spec = structure_spec.get("lattice", {}) or {}
        lattice = Lattice.from_parameters(
            float(lattice_spec.get("a", 3.84)),
            float(lattice_spec.get("b", 3.84)),
            float(lattice_spec.get("c", 3.84)),
            float(lattice_spec.get("alpha", 120.0)),
            float(lattice_spec.get("beta", 90.0)),
            float(lattice_spec.get("gamma", 60.0)),
        )
        coords = structure_spec["coords"]
        if structure_spec.get("coord_kind", "frac") == "cart":
            inv = np.linalg.inv(lattice.matrix.T)
            coords = [list(inv.dot(np.array(coord, float))) for coord in coords]
        return Structure(lattice, structure_spec["species"], coords)

    if kind == "mp":
        from mp_api.client import MPRester

        key = os.environ.get("MP_API_KEY")
        if not key:
            raise RuntimeError("MP_API_KEY not set")

        with MPRester(key) as mpr:
            result = mpr.materials.summary.search(material_ids=[structure_spec["query"]])
            if not result:
                raise RuntimeError(f"MP-ID not found: {structure_spec['query']}")
            structure = result[0].structure

        if structure_spec.get("conventional", True):
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            structure = SpacegroupAnalyzer(
                structure,
                symprec=1e-3,
            ).get_conventional_standard_structure(international_monoclinic=True)
        if structure_spec.get("symmetrize", False):
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            structure = SpacegroupAnalyzer(structure, symprec=1e-3).get_refined_structure()
        return structure

    raise RuntimeError(f"Unsupported structure spec: {structure_spec}")


__all__ = [
    "INVALID_POSCAR_ERROR",
    "POSCAR_ELEMENT_ERROR",
    "POSCAR_ELEMENT_SUGGESTION",
    "parse_structure",
    "structure_from_spec",
    "StructureValidationError",
    "validate_poscar_element_symbols",
]
