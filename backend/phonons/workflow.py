"""Executable finite-displacement phonon stage (Milestone A).

A ``PHONON_FORCES`` stage runs atomate2's VASP ``PhononMaker`` on the relaxed
structure of the immediately preceding PBE Geometry Optimisation stage:

1. ``get_supercell_size`` - atomate2/Phonopy automatic supercell (minimum
   image length ``PHONON_SUPERCELL_MIN_LENGTH_ANGSTROM``);
2. ``generate_phonon_displacements`` - Phonopy displacements;
3. ``run_phonon_displacements`` - replaced at run time by one VASP force
   calculation per displacement (``PhononDisplacementMaker``);
4. ``generate_frequencies_eigenvectors`` - Phonopy force constants, phonon
   band structure (seekpath path) and DOS.

Phonopy, called by atomate2, is the scientific authority for the supercell,
the displacements and the force constants. BMD Compute does not generate
displacements, collect forces or analyse them itself; the BMD displacement
plan (``backend.phonons.plan``) remains available as provenance and as an
independent cross-check of the same displacements.

The displacement amplitude and symmetry tolerance are taken from the BMD
phonon planning policy (``backend.phonons.policy``) so the executable stage and
the cross-check plan cannot drift apart. The force-calculation INCAR and
KPOINTS are atomate2's phonon-grade ``PhononDisplacementMaker`` settings, with
only the BMD amendments listed in ``PHONON_FORCE_BMD_INCAR_AMENDMENTS``.

Like the rest of ``backend.phonons``, this module imports no heavy package at
module level: atomate2's phonon modules import Phonopy, and existing workflows
must keep working where Phonopy is absent.
"""

from __future__ import annotations

from typing import Any

from backend.phonons.policy import PROVISIONAL_PHONON_POLICY

PHONON_WORKFLOW_ID = "bmd_compute.phonons.atomate2_finite_displacement"
PHONON_WORKFLOW_VERSION = 1

PHONON_DISPLACEMENT_ANGSTROM = PROVISIONAL_PHONON_POLICY.displacement_distance_angstrom
PHONON_SYMPREC = PROVISIONAL_PHONON_POLICY.symprec
# atomate2/Phonopy automatic supercell selection: the smallest supercell whose
# minimum image distance is at least this length, preferring right angles.
# 10 Angstrom is an initial practical default for Milestone A (atomate2's own
# default is 20 Angstrom). It is not a claim of force-constant convergence;
# supercell convergence must be checked per material.
PHONON_SUPERCELL_MIN_LENGTH_ANGSTROM = 10.0
PHONON_SUPERCELL_PREFER_90_DEGREES = True
PHONON_KPATH_SCHEME = "seekpath"
PHONON_USE_SYMMETRIZED_STRUCTURE = None
PHONON_SYM_REDUCE = True

# Tightened convergence for the Geometry Optimisation stage that feeds a
# phonon force stage: residual forces must be far below the forces produced by
# a 0.01 Angstrom displacement, and the cell is relaxed at the same plane-wave
# cutoff as the displacement-force calculations (atomate2
# PhononDisplacementMaker ENCUT 700) so the relaxed structure is a minimum of
# the same basis. Ordinary relaxations keep the BMD default (ENCUT 580).
PHONON_PRECURSOR_RELAX_INCAR = {"ENCUT": 700, "EDIFF": 1e-7, "EDIFFG": -1e-3}

# BMD amendments on top of atomate2's PhononDisplacementMaker input set:
# - atomate2's base VASP set selects PBEsol (GGA = PS) with ENAUG/LMIXTAU;
#   BMD Compute runs PBE, so these are removed exactly as for BMD Geometry
#   Optimisation and Static Energy stages (backend.calculations.
#   vasp_stage_definitions). Forces must use the same functional as the relax
#   that produced the structure.
# - phonon stages are non-magnetic only, so spin is fixed off explicitly
#   instead of atomate2's auto_ispin.
# - single-point force calculation: atomate2 0.1.5's PhononDisplacementMaker
#   sets IBRION = 2 and ISIF = 3 (with NSW = 0), and its StaticSetGenerator
#   sets ISMEAR = -5 (tetrahedron), which pymatgen keeps whenever the
#   supercell k-mesh has enough points (e.g. the 64-atom Si cell, 2x2x2).
#   Tetrahedron smearing gives no variational forces, so forces use Gaussian
#   smearing (ISMEAR = 0, SIGMA = 0.2, the value atomate2/pymatgen fall back
#   to), no ionic algorithm (IBRION = -1, NSW = 0), and ISIF is left to the
#   VASP default.
# NCORE is deliberately left unset for force stages.
PHONON_FORCE_BMD_INCAR_AMENDMENTS = {
    "GGA": None,
    "ENAUG": None,
    "LMIXTAU": None,
    "ISPIN": 1,
    "ISMEAR": 0,
    "SIGMA": 0.2,
    "IBRION": -1,
    "NSW": 0,
    "ISIF": None,
}
PHONON_FORCE_AUTO_ISPIN = False

# Phonopy output files written by generate_frequencies_eigenvectors.
PHONON_FORCE_CONSTANTS_FILENAME = "FORCE_CONSTANTS"
PHONON_STORE_FORCE_CONSTANTS = True
PHONON_ANALYSIS_KWARGS = {
    "create_force_constants_file": True,
    "force_constants_filename": PHONON_FORCE_CONSTANTS_FILENAME,
    "calculate_pdos": False,
}

PHONON_JOB_NAMES = (
    "get_supercell_size",
    "generate_phonon_displacements",
    "run_phonon_displacements",
    "generate_frequencies_eigenvectors",
)


def _phonon_force_handler_exclusions() -> tuple[str, ...]:
    from backend.calculations.custodian_policy import PHONON_FORCE_HANDLER_EXCLUSIONS

    return PHONON_FORCE_HANDLER_EXCLUSIONS


def phonon_force_settings_payload() -> dict[str, Any]:
    """Settings of the phonon stage, independent of any structure."""

    return {
        "workflow_id": PHONON_WORKFLOW_ID,
        "workflow_version": PHONON_WORKFLOW_VERSION,
        "maker": "atomate2.vasp.flows.phonons.PhononMaker",
        "force_maker": "atomate2.vasp.jobs.phonons.PhononDisplacementMaker",
        "displacement_angstrom": PHONON_DISPLACEMENT_ANGSTROM,
        "symprec": PHONON_SYMPREC,
        "sym_reduce": PHONON_SYM_REDUCE,
        "supercell": {
            "selection": "atomate2 get_supercell_size (automatic)",
            "min_length_angstrom": PHONON_SUPERCELL_MIN_LENGTH_ANGSTROM,
            "prefer_90_degrees": PHONON_SUPERCELL_PREFER_90_DEGREES,
        },
        "kpath_scheme": PHONON_KPATH_SCHEME,
        "use_symmetrized_structure": PHONON_USE_SYMMETRIZED_STRUCTURE,
        "bulk_relax_maker": None,
        "static_energy_maker": None,
        "born_maker": None,
        "precursor_relax_incar": dict(PHONON_PRECURSOR_RELAX_INCAR),
        "force_bmd_incar_amendments": dict(PHONON_FORCE_BMD_INCAR_AMENDMENTS),
        "force_auto_ispin": PHONON_FORCE_AUTO_ISPIN,
        # One SIGMA for every displacement (see backend.calculations.custodian_policy).
        "force_custodian_handler_exclusions": list(_phonon_force_handler_exclusions()),
        "automatic_ncore": False,
        "store_force_constants": PHONON_STORE_FORCE_CONSTANTS,
        "analysis_kwargs": dict(PHONON_ANALYSIS_KWARGS),
        "outputs": [
            "phonopy.yaml",
            PHONON_FORCE_CONSTANTS_FILENAME,
            "phonon_band_structure.yaml",
            "phonon_dos.yaml",
        ],
    }


def build_phonon_force_input_set_generator(
    *,
    incar: dict | None = None,
    potcar_functional: str = "PBE_64",
    dispersion_vdw: str | None = None,
):
    """atomate2's phonon-grade static generator with the BMD amendments."""

    from dataclasses import replace

    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker

    base = PhononDisplacementMaker().input_set_generator
    user_incar = dict(base.user_incar_settings or {})
    user_incar.update(incar or {})
    user_incar.update(PHONON_FORCE_BMD_INCAR_AMENDMENTS)
    # Never inherit NCORE from global settings for force stages.
    user_incar.pop("NCORE", None)
    return replace(
        base,
        user_incar_settings=user_incar,
        user_potcar_functional=potcar_functional,
        vdw=dispersion_vdw,
        auto_ispin=PHONON_FORCE_AUTO_ISPIN,
    )


def build_phonon_maker(
    *,
    force_input_set_generator,
    name: str,
    run_vasp_kwargs: dict | None = None,
    stop_children_kwargs: dict | None = None,
):
    from atomate2.vasp.flows.phonons import PhononMaker
    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker

    displacement_kwargs: dict[str, Any] = {"input_set_generator": force_input_set_generator}
    if run_vasp_kwargs:
        displacement_kwargs["run_vasp_kwargs"] = dict(run_vasp_kwargs)
    if stop_children_kwargs:
        displacement_kwargs["stop_children_kwargs"] = dict(stop_children_kwargs)

    return PhononMaker(
        name=name,
        sym_reduce=PHONON_SYM_REDUCE,
        symprec=PHONON_SYMPREC,
        displacement=PHONON_DISPLACEMENT_ANGSTROM,
        min_length=PHONON_SUPERCELL_MIN_LENGTH_ANGSTROM,
        prefer_90_degrees=PHONON_SUPERCELL_PREFER_90_DEGREES,
        use_symmetrized_structure=PHONON_USE_SYMMETRIZED_STRUCTURE,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        phonon_displacement_maker=PhononDisplacementMaker(**displacement_kwargs),
        kpath_scheme=PHONON_KPATH_SCHEME,
        store_force_constants=PHONON_STORE_FORCE_CONSTANTS,
        generate_frequencies_eigenvectors_kwargs=dict(PHONON_ANALYSIS_KWARGS),
    )


def build_phonon_flow(structure, maker):
    """Phonon flow for ``structure`` (normally the previous relax output).

    ``prev_dir`` is deliberately ``None``: displacement supercells differ from
    the relaxed cell, so no WAVECAR or CHGCAR is carried into them.
    """

    flow = maker.make(structure, prev_dir=None)
    # atomate2 returns an unnamed Flow; name it after the stage.
    flow.name = maker.name
    return flow


def phonon_planning_preview(structure) -> dict[str, Any]:
    """Supercell and displacement count for ``structure`` using atomate2/Phonopy.

    This calls the undecorated functions of the same atomate2 jobs the
    workflow runs (``get_supercell_size`` and ``generate_phonon_displacements``)
    with the same arguments, so it does not re-implement Phonopy planning. At
    run time the jobs act on the *relaxed* structure, so the supercell and the
    number of displacements can differ if relaxation changes the lattice or
    the symmetry.
    """

    import warnings

    from atomate2.common.jobs.phonons import (
        generate_phonon_displacements,
        get_supercell_size,
    )

    supercell_matrix = get_supercell_size.original(
        structure=structure,
        min_length=PHONON_SUPERCELL_MIN_LENGTH_ANGSTROM,
        max_length=None,
        prefer_90_degrees=PHONON_SUPERCELL_PREFER_90_DEGREES,
        allow_orthorhombic=False,
    )
    with warnings.catch_warnings():
        # atomate2 always warns that initial magnetic moments are ignored;
        # phonon stages are non-magnetic by admission.
        warnings.simplefilter("ignore", UserWarning)
        displacements = generate_phonon_displacements.original(
            structure=structure,
            supercell_matrix=supercell_matrix,
            displacement=PHONON_DISPLACEMENT_ANGSTROM,
            sym_reduce=PHONON_SYM_REDUCE,
            symprec=PHONON_SYMPREC,
            use_symmetrized_structure=PHONON_USE_SYMMETRIZED_STRUCTURE,
            kpath_scheme=PHONON_KPATH_SCHEME,
            code="vasp",
        )
    matrix = [[int(round(float(value))) for value in row] for row in supercell_matrix]
    return {
        "basis": "submitted_structure",
        "note": (
            "Computed by atomate2/Phonopy from the submitted structure. The "
            "workflow recomputes the supercell and displacements from the "
            "relaxed structure at run time."
        ),
        "supercell_matrix": matrix,
        "supercell_atom_count": len(displacements[0]) if displacements else None,
        "displacement_count": len(displacements),
        "first_displaced_supercell": displacements[0] if displacements else None,
        "settings": phonon_force_settings_payload(),
    }


__all__ = [
    "PHONON_ANALYSIS_KWARGS",
    "PHONON_DISPLACEMENT_ANGSTROM",
    "PHONON_FORCE_AUTO_ISPIN",
    "PHONON_FORCE_BMD_INCAR_AMENDMENTS",
    "PHONON_FORCE_CONSTANTS_FILENAME",
    "PHONON_JOB_NAMES",
    "PHONON_KPATH_SCHEME",
    "PHONON_PRECURSOR_RELAX_INCAR",
    "PHONON_STORE_FORCE_CONSTANTS",
    "PHONON_SUPERCELL_MIN_LENGTH_ANGSTROM",
    "PHONON_SYMPREC",
    "PHONON_WORKFLOW_ID",
    "PHONON_WORKFLOW_VERSION",
    "build_phonon_flow",
    "build_phonon_force_input_set_generator",
    "build_phonon_maker",
    "phonon_force_settings_payload",
    "phonon_planning_preview",
]
