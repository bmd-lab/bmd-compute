"""POTCAR choice for structure searches.

A search must use the same POTCAR for each element as every other BMD
calculation (``docs/methodology.md`` section 5): the PBE_64 set with the
symbol choices of the pinned atomate2 relaxation generator
(``RelaxSetGenerator``, whose POTCAR table is atomate2's ``_BASE_VASP_SET``).
This module reads that table instead of keeping a separate BMD copy, so the
two can never drift apart.

No POTCAR file is read here: only symbols (for example ``Ti -> Ti_sv``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping


POTCAR_FUNCTIONAL = "PBE_64"
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


__all__ = [
    "POTCAR_FUNCTIONAL",
    "POTCAR_SYMBOL_SOURCE",
    "bmd_potcar_symbol",
    "bmd_potcar_symbols",
    "unmapped_elements",
]
