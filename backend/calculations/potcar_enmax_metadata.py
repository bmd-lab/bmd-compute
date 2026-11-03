"""ENMAX metadata of the POTCAR datasets BMD Compute's generators can select.

BMD Compute plans on a host without POTCAR files, so the species-dependent
plane-wave cutoff (``backend.calculations.geometry_relaxation``) reads ENMAX
from a small, versioned metadata file generated once from the POTCAR library
that VASP actually reads on POWER. The file records, per POTCAR symbol, only
header values (TITEL, ENMAX) and the SHA-256 of the dataset text; it never
contains potential data.

This module is deliberately self-contained (standard library, pymatgen and
atomate2 only) so that it can be copied to POWER and run there to produce the
file::

    python potcar_enmax_metadata.py --psp-dir /bmd-db/potcars \\
        --output potcar_enmax_PBE_64.json

The output is then committed as
``backend/calculations/potcar_enmax_PBE_64.json``. At run time the same
values are checked again against the POTCAR files VASP will read
(``backend.calculations.geometry_relaxation.runtime_potcar_problems``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

METADATA_SCHEMA = "bmd_compute.potcar_enmax"
METADATA_SCHEMA_VERSION = 1
SUPPORTED_FUNCTIONALS = ("PBE_64",)
# Library subdirectories that may hold the PBE_64 datasets: pymatgen's own
# layout (POT_PAW_PBE_64, read by atomate2 at run time) and the plain name.
FUNCTIONAL_SUBDIRECTORIES = {"PBE_64": ("POT_PAW_PBE_64", "PBE_64")}


class PotcarMetadataError(RuntimeError):
    """The POTCAR library cannot produce consistent ENMAX metadata."""


def metadata_path(functional: str) -> Path:
    return Path(__file__).resolve().parent / f"potcar_enmax_{functional}.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def metadata_digest(metadata: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()


def potcar_text_sha256(text: str) -> str:
    """SHA-256 of one POTCAR dataset's text (``PotcarSingle.data``)."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def potcar_identity(potcar_single) -> dict[str, Any]:
    """The header values BMD records for one ``PotcarSingle``."""

    keywords = getattr(potcar_single, "keywords", {}) or {}
    return {
        "TITEL": str(keywords.get("TITEL", "")).strip(),
        "ENMAX": float(keywords["ENMAX"]),
        "sha256": potcar_text_sha256(potcar_single.data),
    }


def generator_selectable_symbols() -> list[str]:
    """Every POTCAR symbol BMD Compute's atomate2 input generators can select.

    The relaxation, static, non-self-consistent, HSE06 and phonon displacement
    generators all use atomate2's base POTCAR mapping; the union is taken so
    that a future divergence is still covered.
    """

    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
    from atomate2.vasp.sets.core import (
        HSEBSSetGenerator,
        NonSCFSetGenerator,
        RelaxSetGenerator,
        StaticSetGenerator,
    )

    generators = (
        RelaxSetGenerator(),
        StaticSetGenerator(),
        NonSCFSetGenerator(),
        HSEBSSetGenerator(),
        PhononDisplacementMaker().input_set_generator,
    )
    symbols: set[str] = set()
    for generator in generators:
        mapping = dict((generator.config_dict or {}).get("POTCAR") or {})
        for value in mapping.values():
            symbols.add(value["symbol"] if isinstance(value, Mapping) else str(value))
    return sorted(symbols)


def _candidate_files(psp_dir: Path, functional: str, symbol: str) -> list[Path]:
    from monty.os.path import zpath

    found = []
    for subdirectory in FUNCTIONAL_SUBDIRECTORIES[functional]:
        for relative in (Path(subdirectory) / symbol / "POTCAR", Path(subdirectory) / f"POTCAR.{symbol}"):
            path = Path(zpath(str(psp_dir / relative)))
            if path.is_file():
                found.append(path)
    return found


def build_enmax_metadata(
    psp_dir: str | os.PathLike,
    *,
    functional: str = "PBE_64",
    symbols: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Read TITEL, ENMAX and the dataset SHA-256 for every selectable symbol.

    Every symbol must exist under at least one of the functional's library
    subdirectories; when it exists under more than one, all copies must be
    identical. Anything else raises :class:`PotcarMetadataError`.
    """

    from pymatgen.io.vasp.inputs import PotcarSingle

    if functional not in SUPPORTED_FUNCTIONALS:
        raise PotcarMetadataError(f"Unsupported POTCAR functional {functional!r}.")
    root = Path(psp_dir)
    wanted = sorted(set(symbols if symbols is not None else generator_selectable_symbols()))
    entries: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for symbol in wanted:
        files = _candidate_files(root, functional, symbol)
        if not files:
            problems.append(f"{symbol}: no POTCAR under {', '.join(FUNCTIONAL_SUBDIRECTORIES[functional])}")
            continue
        identities = []
        for path in files:
            single = PotcarSingle.from_file(str(path))
            if getattr(single, "symbol", symbol) != symbol:
                problems.append(f"{symbol}: {path} holds {single.symbol!r}")
            identities.append((path, potcar_identity(single)))
        distinct = {canonical_json(identity) for _, identity in identities}
        if len(distinct) != 1:
            problems.append(
                f"{symbol}: copies differ between "
                + ", ".join(str(path.relative_to(root)) for path, _ in identities)
            )
            continue
        identity = dict(identities[0][1])
        identity["library_paths"] = sorted(str(path.relative_to(root)) for path, _ in identities)
        entries[symbol] = identity
    if problems:
        raise PotcarMetadataError("POTCAR ENMAX metadata cannot be built:\n- " + "\n- ".join(problems))

    from importlib import metadata as package_metadata

    def _version(name: str) -> str | None:
        try:
            return package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            return None

    return {
        "schema": METADATA_SCHEMA,
        "schema_version": METADATA_SCHEMA_VERSION,
        "functional": functional,
        "synthetic": False,
        "source": {
            "library_root": str(root),
            "subdirectories": list(FUNCTIONAL_SUBDIRECTORIES[functional]),
            "symbol_selection": "union of atomate2 generator POTCAR mappings",
            "generated_by": "backend.calculations.potcar_enmax_metadata.build_enmax_metadata",
            "packages": {name: _version(name) for name in ("atomate2", "pymatgen", "pymatgen-core")},
        },
        "potcars": entries,
    }


def validate_metadata(metadata: Any, *, functional: str) -> list[str]:
    """Structural problems of a metadata document (empty list when valid)."""

    problems: list[str] = []
    if not isinstance(metadata, Mapping):
        return ["metadata is not a JSON object"]
    if metadata.get("schema") != METADATA_SCHEMA or metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        problems.append("unknown metadata schema or version")
    if metadata.get("functional") != functional:
        problems.append(f"metadata is for {metadata.get('functional')!r}, not {functional!r}")
    potcars = metadata.get("potcars")
    if not isinstance(potcars, Mapping) or not potcars:
        problems.append("metadata lists no POTCAR datasets")
        return problems
    for symbol, entry in potcars.items():
        if not isinstance(entry, Mapping):
            problems.append(f"{symbol}: entry is not an object")
            continue
        enmax = entry.get("ENMAX")
        if isinstance(enmax, bool) or not isinstance(enmax, (int, float)) or not enmax > 0:
            problems.append(f"{symbol}: ENMAX must be a positive number")
        if not isinstance(entry.get("TITEL"), str) or not entry.get("TITEL"):
            problems.append(f"{symbol}: TITEL is missing")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            problems.append(f"{symbol}: sha256 is missing")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--psp-dir", required=True, help="POTCAR library root (PMG_VASP_PSP_DIR)")
    parser.add_argument("--functional", default="PBE_64", choices=SUPPORTED_FUNCTIONALS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        metadata = build_enmax_metadata(args.psp_dir, functional=args.functional)
    except PotcarMetadataError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"{len(metadata['potcars'])} POTCAR datasets; {metadata_digest(metadata)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
