"""POTCAR choice for structure searches.

A search must use the same POTCAR for each element as every other BMD
calculation (``docs/methodology.md`` section 5): the PBE_64 set with the
symbol choices of the pinned atomate2 relaxation generator
(``RelaxSetGenerator``, whose POTCAR table is atomate2's ``_BASE_VASP_SET``).
This module reads that table instead of keeping a separate BMD copy, so the
two can never drift apart.

The symbol lookup reads no POTCAR file. :func:`write_specific_potcars`
copies the chosen files from the shared POTCAR repository into a search's
``Specific/`` folder as ``POTCAR_<El>``, which is where USPEX looks for one
POTCAR per element. POTCAR files are licensed and are never committed; the
tests use placeholder files.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


POTCAR_FUNCTIONAL = "PBE_64"
# Folder of the PBE_64 set inside the POTCAR repository, as pymatgen (and so
# every BMD stage generator) looks it up: <PMG_VASP_PSP_DIR>/POT_PAW_PBE_64/<symbol>/POTCAR.
POTCAR_FUNCTIONAL_DIR = "POT_PAW_PBE_64"
POTCAR_SYMBOL_SOURCE = "atomate2.vasp.sets.core.RelaxSetGenerator.config_dict['POTCAR']"


@lru_cache(maxsize=1)
def _relax_generator_potcar_table() -> Mapping[str, str]:
    from atomate2.vasp.sets.core import RelaxSetGenerator

    table = RelaxSetGenerator().config_dict["POTCAR"]
    return {str(element): str(symbol) for element, symbol in table.items()}


def bmd_potcar_symbol(element: str) -> str | None:
    """POTCAR symbol BMD uses for ``element``, or ``None`` if BMD has no mapping."""

    return _relax_generator_potcar_table().get(str(element))


def bmd_potcar_symbols(elements: Iterable[str]) -> dict[str, str]:
    """Element -> POTCAR symbol for every element; raises ``KeyError`` for an unmapped one."""

    mapping: dict[str, str] = {}
    for element in elements:
        symbol = bmd_potcar_symbol(element)
        if symbol is None:
            raise KeyError(element)
        mapping[str(element)] = symbol
    return mapping


def unmapped_elements(elements: Iterable[str]) -> list[str]:
    """Elements, in input order, that have no BMD POTCAR mapping."""

    return [str(element) for element in elements if bmd_potcar_symbol(element) is None]


class SpecificPotcarError(ValueError):
    """A POTCAR for a search cannot be found or is not the expected one."""


# ENMAX is read the way pymatgen's PotcarSingle reads POTCAR keywords:
# ``KEY = value`` up to ``;`` inside the PSCTR parameter block
# (for example ``   ENMAX  =  499.034; ENMIN  =  374.276 eV``).
_PSCTR_BLOCK = re.compile(r"(?s)parameters from PSCTR are:(.*?)END of PSCTR-controll parameters")
_ENMAX = re.compile(r"^\s*ENMAX\s*=\s*([0-9]+(?:\.[0-9]*)?)\s*(?=;|$)", re.MULTILINE)


def read_potcar_enmax(text: str) -> float:
    """ENMAX (eV) from the text of one POTCAR dataset; raises ``SpecificPotcarError``."""

    block = _PSCTR_BLOCK.search(text)
    values = _ENMAX.findall(block.group(1)) if block else []
    if len(values) != 1:
        raise SpecificPotcarError(
            "Could not read exactly one ENMAX value from the POTCAR's PSCTR parameter block."
        )
    return float(values[0])


def max_enmax(records: Iterable[Mapping[str, Any]]) -> float:
    """Largest ENMAX among records returned by :func:`write_specific_potcars`."""

    values = [float(record["enmax"]) for record in records]
    if not values:
        raise ValueError("No POTCAR records")
    return max(values)


def specific_potcar_name(element: str) -> str:
    """File name USPEX expects in ``Specific/`` for ``element``: ``POTCAR_<El>``."""

    return f"POTCAR_{element}"


def potcar_source_path(psp_dir: str | Path, element: str) -> Path:
    """Path of BMD's POTCAR for ``element`` inside the POTCAR repository ``psp_dir``."""

    symbol = bmd_potcar_symbol(element)
    if symbol is None:
        raise SpecificPotcarError(f"bmd-compute has no POTCAR mapping for {element}.")
    return Path(psp_dir) / POTCAR_FUNCTIONAL_DIR / symbol / "POTCAR"


def write_specific_potcars(
    elements: Iterable[str],
    psp_dir: str | Path,
    specific_dir: str | Path,
) -> list[dict[str, Any]]:
    """Copy BMD's POTCAR for each element to ``specific_dir/POTCAR_<El>``.

    Before copying, each source file's first line must name the expected
    POTCAR symbol (a real POTCAR starts with, for example,
    ``PAW_PBE Ti_sv 26Sep2005``), so a wrong or misplaced file is caught
    before any VASP run, and its ENMAX must be readable. Returns one record
    per element with the symbol, ENMAX (eV), the source path and the SHA-256
    of the copied file, for provenance. The copy is byte-identical to the
    source, so the ENMAX read from the source is the copy's. Nothing is
    written if any element fails.

    Then check the records against the search's cutoffs with
    ``incar_policy.require_potcars_match_cutoff_record`` (the ENMAX values
    must be those the cutoffs were resolved from) and
    ``incar_policy.require_encut_covers_enmax(max_enmax(records), ...)``.
    """

    elements = [str(element) for element in elements]
    sources = []
    for element in elements:
        source = potcar_source_path(psp_dir, element)
        symbol = bmd_potcar_symbol(element)
        if not source.is_file():
            raise SpecificPotcarError(f"POTCAR for {element} ({symbol}) not found at {source}.")
        text = source.read_text(encoding="utf-8", errors="replace")
        first_line = text.split("\n", 1)[0]
        if symbol not in first_line.split():
            raise SpecificPotcarError(
                f"{source} does not start with POTCAR symbol {symbol}: {first_line.strip()!r}."
            )
        try:
            enmax = read_potcar_enmax(text)
        except SpecificPotcarError as exc:
            raise SpecificPotcarError(f"{source}: {exc}") from exc
        sources.append((element, symbol, source, enmax))

    records = []
    for element, symbol, source, enmax in sources:
        target = Path(specific_dir) / specific_potcar_name(element)
        shutil.copyfile(source, target)
        records.append({
            "element": element,
            "potcar_symbol": symbol,
            "enmax": enmax,
            "functional": POTCAR_FUNCTIONAL,
            "source": str(source),
            "target": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    return records


__all__ = [
    "POTCAR_FUNCTIONAL",
    "POTCAR_FUNCTIONAL_DIR",
    "SpecificPotcarError",
    "max_enmax",
    "read_potcar_enmax",
    "potcar_source_path",
    "specific_potcar_name",
    "write_specific_potcars",
    "POTCAR_SYMBOL_SOURCE",
    "bmd_potcar_symbol",
    "bmd_potcar_symbols",
    "unmapped_elements",
]
