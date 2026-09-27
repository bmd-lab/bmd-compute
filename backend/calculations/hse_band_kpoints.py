from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Any, Sequence


"""
Full-zone weighted k-points for HSE06 Band Structure + SOC stages.

atomate2's ``HSEBSSetGenerator`` (line mode) writes one explicit KPOINTS file
that combines a weighted SCF mesh with a zero-weight high-symmetry path. The
weighted part comes from pymatgen as a symmetry-reduced irreducible set whose
weights are the star multiplicities. That is correct when VASP symmetrises the
density (the non-SOC HSE06 default ``ISYM=3``), but BMD SOC stages keep
``ISYM=0``, in which case VASP uses the listed points as the complete sampling.

For SOC this module keeps everything the generator produced except the
weighted part, which is replaced by every point of the same mesh with equal
weights. The mesh is recovered from the generator's own reduced points and is
verified before anything is replaced, so no second k-point policy is
introduced: the reciprocal density, mesh, grid offset, zero-weight path and
labels all remain the generator's.
"""


FULL_ZONE_HSE_BAND_GENERATOR_NAME = "FullZoneHSEBSSetGenerator"
FULL_ZONE_KPOINTS_ADJUSTMENT = "full_zone_weighted_scf_mesh_for_soc"
_COORDINATE_TOLERANCE = 1e-6


class FullZoneKpointsError(ValueError):
    pass


def uses_full_zone_weighted_kpoints(stage_type, theory, modifiers) -> bool:
    """Return whether a stage replaces the reduced weighted mesh with the full zone.

    Only HSE06 Band Structure stages with SOC do so; every other stage,
    including non-SOC HSE06 Band Structure, keeps the generator's k-points.
    """

    from backend.calculations.models import Modifier, StageType, Theory

    return (
        StageType.from_value(stage_type) is StageType.BAND_STRUCTURE
        and Theory.from_value(theory) is Theory.HSE06
        and Modifier.SOC in {Modifier.from_value(item) for item in (modifiers or ())}
    )


def full_zone_mesh_points(
    mesh: Sequence[int],
    offset: Sequence[float],
) -> list[tuple[float, float, float]]:
    """Return every point of an ``n1 x n2 x n3`` mesh in fractional coordinates.

    ``offset`` is the grid offset in units of the grid spacing (0 for a
    Gamma-centred axis, 0.5 for an even Monkhorst-Pack axis). Coordinates are
    folded into (-0.5, 0.5].
    """

    divisions = [int(value) for value in mesh]
    if len(divisions) != 3 or any(value < 1 for value in divisions):
        raise FullZoneKpointsError(f"Invalid k-point mesh: {tuple(mesh)!r}")
    axes = []
    for count, shift in zip(divisions, offset):
        axes.append([_fold((index + float(shift)) / count) for index in range(count)])
    return [tuple(point) for point in product(*axes)]


def replace_weighted_kpoints_with_full_zone(kpoints, *, mesh: Sequence[int]):
    """Return a copy of an explicit HSE band KPOINTS with a full-zone weighted mesh.

    ``kpoints`` must be the explicit (weighted + zero-weight) KPOINTS produced
    by ``HSEBSSetGenerator``; ``mesh`` is the automatic mesh it was reduced
    from. The weighted points must be a symmetry-reduced subset of that mesh
    whose weights sum to the number of mesh points; otherwise this raises
    rather than guessing.
    """

    from pymatgen.io.vasp.inputs import Kpoints

    points = [tuple(float(value) for value in point) for point in kpoints.kpts]
    weights = list(kpoints.kpts_weights or [])
    if len(weights) != len(points):
        raise FullZoneKpointsError(
            "HSE06 band-structure KPOINTS must list one weight per k-point."
        )
    labels = list(kpoints.labels or [None] * len(points))
    if len(labels) != len(points):
        labels = [None] * len(points)

    weighted = [point for point, weight in zip(points, weights) if float(weight) > 0]
    zero_weight = [
        (point, label)
        for point, weight, label in zip(points, weights, labels)
        if float(weight) == 0
    ]
    if not weighted:
        raise FullZoneKpointsError(
            "HSE06 band-structure KPOINTS has no weighted SCF k-points to expand."
        )
    if not zero_weight:
        raise FullZoneKpointsError(
            "HSE06 band-structure KPOINTS has no zero-weight path k-points."
        )

    divisions = tuple(int(value) for value in mesh)
    mesh_size = divisions[0] * divisions[1] * divisions[2]
    weight_total = sum(float(weight) for weight in weights if float(weight) > 0)
    if abs(weight_total - mesh_size) > _COORDINATE_TOLERANCE:
        raise FullZoneKpointsError(
            "The weighted HSE06 k-points do not form a symmetry-reduced "
            f"{divisions[0]}x{divisions[1]}x{divisions[2]} mesh "
            f"(weights sum to {weight_total:g}, expected {mesh_size})."
        )

    full_zone = _matching_full_zone_mesh(weighted, divisions)
    return Kpoints(
        comment=f"{kpoints.comment} | BMD full-zone weighted mesh for SOC (ISYM=0)",
        num_kpts=len(full_zone) + len(zero_weight),
        style=kpoints.style,
        kpts=[*full_zone, *(point for point, _ in zero_weight)],
        kpts_weights=[1] * len(full_zone) + [0] * len(zero_weight),
        coord_type=kpoints.coord_type,
        labels=[None] * len(full_zone) + [label for _, label in zero_weight],
    )


def automatic_mesh_divisions(structure, reciprocal_density, *, force_gamma=False):
    from pymatgen.io.vasp.inputs import Kpoints

    automatic = Kpoints.automatic_density_by_vol(
        structure,
        int(reciprocal_density),
        force_gamma,
    )
    return tuple(int(value) for value in automatic.kpts[0])


def full_zone_hse_band_structure_generator_class():
    """Return the SOC-only full-zone subclass of the installed HSEBSSetGenerator."""

    from atomate2.vasp.sets.core import HSEBSSetGenerator

    return _full_zone_generator_class_for(HSEBSSetGenerator)


@lru_cache(maxsize=None)
def _full_zone_generator_class_for(base_class):
    class FullZoneHSEBSSetGenerator(base_class):
        """HSEBSSetGenerator whose weighted SCF k-points span the full zone."""

        def get_input_set(self, *args: Any, **kwargs: Any):
            input_set = super().get_input_set(*args, **kwargs)
            structure = _input_set_structure(input_set)
            mesh = automatic_mesh_divisions(
                structure,
                self.reciprocal_density,
                force_gamma=bool(getattr(self, "force_gamma", False)),
            )
            full_zone = replace_weighted_kpoints_with_full_zone(
                _input_set_kpoints(input_set),
                mesh=mesh,
            )
            _set_input_set_kpoints(input_set, full_zone)
            return input_set

    FullZoneHSEBSSetGenerator.__module__ = __name__
    FullZoneHSEBSSetGenerator.__name__ = FULL_ZONE_HSE_BAND_GENERATOR_NAME
    FullZoneHSEBSSetGenerator.__qualname__ = FULL_ZONE_HSE_BAND_GENERATOR_NAME
    return FullZoneHSEBSSetGenerator


def __getattr__(name: str):
    # Lets serialized generators (``@module``/``@class``) resolve the lazily
    # created subclass without importing atomate2 at module import time.
    if name == FULL_ZONE_HSE_BAND_GENERATOR_NAME:
        return full_zone_hse_band_structure_generator_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _matching_full_zone_mesh(weighted, divisions):
    # The reduced points are grid points of the mesh the generator used. Find
    # the grid offset (Gamma-centred or half-shifted per axis) that contains
    # every one of them, so the expansion reproduces that exact grid.
    candidate_offsets = product(*[(0.0, 0.5) for _ in range(3)])
    matches = []
    for offset in candidate_offsets:
        grid = full_zone_mesh_points(divisions, offset)
        grid_keys = {_point_key(point) for point in grid}
        if all(_point_key(point) in grid_keys for point in weighted):
            matches.append(grid)
    if len(matches) != 1:
        raise FullZoneKpointsError(
            "Could not identify a unique "
            f"{divisions[0]}x{divisions[1]}x{divisions[2]} grid containing the "
            "weighted HSE06 k-points."
        )
    return matches[0]


def _fold(value: float) -> float:
    folded = value - round(value)
    if folded <= -0.5 + _COORDINATE_TOLERANCE:
        folded += 1.0
    return folded + 0.0


def _point_key(point) -> tuple[int, int, int]:
    scale = 1e6
    return tuple(int(round((_fold(float(value)) % 1.0) * scale)) % int(scale) for value in point)


def _has_input_file(input_set, key: str) -> bool:
    # pymatgen's VaspInput is a mapping of file names; older atomate2 input
    # sets are plain objects with incar/kpoints/poscar attributes.
    try:
        return key in input_set
    except TypeError:
        return False


def _input_set_kpoints(input_set):
    if _has_input_file(input_set, "KPOINTS"):
        return input_set["KPOINTS"]
    return input_set.kpoints


def _set_input_set_kpoints(input_set, kpoints) -> None:
    if _has_input_file(input_set, "KPOINTS"):
        input_set["KPOINTS"] = kpoints
        return
    input_set.kpoints = kpoints


def _input_set_structure(input_set):
    if _has_input_file(input_set, "POSCAR"):
        return input_set["POSCAR"].structure
    return input_set.poscar.structure


__all__ = [
    "FULL_ZONE_HSE_BAND_GENERATOR_NAME",
    "FULL_ZONE_KPOINTS_ADJUSTMENT",
    "FullZoneKpointsError",
    "automatic_mesh_divisions",
    "full_zone_hse_band_structure_generator_class",
    "full_zone_mesh_points",
    "replace_weighted_kpoints_with_full_zone",
    "uses_full_zone_weighted_kpoints",
]
