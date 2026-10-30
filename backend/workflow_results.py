from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.calculations.models import CalculationSpec, Purpose, StageType, WorkflowSpec
from backend.calculations.registry import workflow_spec_from_calculation_spec


BAND_STRUCTURE_INITIAL_YAXIS_RANGE = [-10, 10]


@dataclass(frozen=True)
class WorkflowResultPayload:
    summaries: dict[str, dict]
    visualizations: list[dict]


class WorkflowResultRenderer:
    purpose: Purpose
    required_file_keys: tuple[str, ...] = ()
    parse_dos: bool = False
    parse_eigenvalues: bool = False

    def render(
        self,
        *,
        vasprun,
        files: dict,
        local_paths: dict | None = None,
    ) -> WorkflowResultPayload:
        raise NotImplementedError


class DensityOfStatesResultRenderer(WorkflowResultRenderer):
    purpose = Purpose.DOS
    required_file_keys = ("doscar",)
    parse_dos = True

    def render(
        self,
        *,
        vasprun,
        files: dict,
        local_paths: dict | None = None,
    ) -> WorkflowResultPayload:
        del local_paths
        summary, visualization = _dos_result_payload(vasprun, files)
        visualizations = [visualization] if visualization is not None else []
        return WorkflowResultPayload(
            summaries={"dos": summary},
            visualizations=visualizations,
        )


class BandStructureResultRenderer(WorkflowResultRenderer):
    purpose = Purpose.BAND_STRUCTURE
    required_file_keys = ("kpoints",)
    parse_eigenvalues = True

    def render(
        self,
        *,
        vasprun,
        files: dict,
        local_paths: dict | None = None,
    ) -> WorkflowResultPayload:
        summary, visualization = _band_structure_result_payload(
            vasprun,
            files,
            local_paths or {},
        )
        visualizations = [visualization] if visualization is not None else []
        return WorkflowResultPayload(
            summaries={"band_structure": summary},
            visualizations=visualizations,
        )


_WORKFLOW_RESULT_RENDERERS: dict[Purpose, WorkflowResultRenderer] = {
    Purpose.DOS: DensityOfStatesResultRenderer(),
    Purpose.BAND_STRUCTURE: BandStructureResultRenderer(),
}


def workflow_result_renderers(
    spec: CalculationSpec | WorkflowSpec | None,
    *,
    available_file_keys: set[str] | None = None,
) -> tuple[WorkflowResultRenderer, ...]:
    stage_types = _workflow_stage_types(spec)
    if stage_types:
        renderers = []
        if StageType.DOS in stage_types:
            renderers.append(_WORKFLOW_RESULT_RENDERERS[Purpose.DOS])
        if StageType.BAND_STRUCTURE in stage_types:
            renderers.append(_WORKFLOW_RESULT_RENDERERS[Purpose.BAND_STRUCTURE])
        return tuple(renderers)

    if available_file_keys and "doscar" in available_file_keys:
        return (_WORKFLOW_RESULT_RENDERERS[Purpose.DOS],)

    if available_file_keys and "kpoints" in available_file_keys:
        return (_WORKFLOW_RESULT_RENDERERS[Purpose.BAND_STRUCTURE],)

    return ()


def workflow_result_file_keys(
    spec: CalculationSpec | WorkflowSpec | None,
) -> tuple[str, ...]:
    keys: list[str] = []
    for renderer in workflow_result_renderers(spec):
        for key in renderer.required_file_keys:
            if key not in keys:
                keys.append(key)

    return tuple(keys)


# Phonopy outputs of the phonon analysis job, read from the phonon stage's
# nested analysis directory. They are optional result files: a missing or
# incomplete file is reported in the phonon summary and never turns the
# relaxation results of the same workflow into a failure.
PHONON_RESULT_FILES = {
    "phonon_band_structure": "phonon_band_structure.yaml",
    "phonon_dos": "phonon_dos.yaml",
    "phonopy_yaml": "phonopy.yaml",
}


def workflow_has_phonon_results(spec: CalculationSpec | WorkflowSpec | None) -> bool:
    stage_types = _workflow_stage_types(spec)
    return bool(stage_types) and stage_types[-1] is StageType.PHONON_FORCES


def workflow_result_parse_dos(
    spec: CalculationSpec | WorkflowSpec | None,
    *,
    available_file_keys: set[str] | None = None,
) -> bool:
    return any(
        renderer.parse_dos
        for renderer in workflow_result_renderers(
            spec,
            available_file_keys=available_file_keys,
        )
    )


def workflow_result_parse_eigenvalues(
    spec: CalculationSpec | WorkflowSpec | None,
    *,
    available_file_keys: set[str] | None = None,
) -> bool:
    return any(
        renderer.parse_eigenvalues
        for renderer in workflow_result_renderers(
            spec,
            available_file_keys=available_file_keys,
        )
    )


def render_workflow_results(
    spec: CalculationSpec | WorkflowSpec | None,
    *,
    vasprun,
    files: dict,
    local_paths: dict | None = None,
) -> WorkflowResultPayload:
    summaries: dict[str, dict] = {}
    visualizations: list[dict] = []
    renderers = workflow_result_renderers(
        spec,
        available_file_keys=set(files),
    )

    for renderer in renderers:
        payload = renderer.render(
            vasprun=vasprun,
            files=files,
            local_paths=local_paths,
        )
        summaries.update(payload.summaries)
        visualizations.extend(payload.visualizations)

    return WorkflowResultPayload(
        summaries=summaries,
        visualizations=visualizations,
    )


def _workflow_stage_types(
    spec: CalculationSpec | WorkflowSpec | None,
) -> tuple[StageType, ...]:
    if isinstance(spec, WorkflowSpec):
        return tuple(stage.stage_type for stage in spec.stages)

    if isinstance(spec, CalculationSpec):
        try:
            workflow_spec = workflow_spec_from_calculation_spec(spec)
        except Exception:
            return ()
        return tuple(stage.stage_type for stage in workflow_spec.stages)

    return ()


def _dos_result_payload(vasprun, files: dict) -> tuple[dict, dict | None]:
    total_dos = _total_dos_from_vasprun(vasprun)
    energies = _as_float_list(getattr(total_dos, "energies", None))
    densities = getattr(total_dos, "densities", None) or {}
    efermi = _float_or_none(
        getattr(total_dos, "efermi", None)
        if total_dos is not None
        else None
    )
    if efermi is None:
        efermi = _float_or_none(getattr(vasprun, "efermi", None))

    summary = {
        "available": total_dos is not None,
        "source": "vasprun.xml",
        "doscar": files.get("doscar", {}).get("path"),
        "energy_points": len(energies) if energies else None,
        "energy_min_ev": _round_float(min(energies)) if energies else None,
        "energy_max_ev": _round_float(max(energies)) if energies else None,
        "spin_channels": len(densities) if densities else None,
        "fermi_level_ev": _round_float(efermi),
    }
    if total_dos is None or not energies or not densities:
        return summary, None

    x_values = [
        _round_float(energy - efermi) if efermi is not None else _round_float(energy)
        for energy in energies
    ]
    traces = []
    for spin, values in densities.items():
        y_values = _as_float_list(values)
        if not y_values:
            continue
        multiplier = -1.0 if _is_spin_down(spin) else 1.0
        traces.append(
            {
                "name": _spin_label(spin),
                "y": [_round_float(multiplier * value) for value in y_values],
            }
        )

    if not traces:
        return summary, None

    visualization = {
        "id": "dos",
        "element_id": "workflow-visualization-dos",
        "kind": "line_plot",
        "title": "Density of States",
        "subtitle": "Total DOS from vasprun.xml",
        "download_filename": "density_of_states.png",
        "plot": {
            "x": x_values,
            "traces": traces,
            "xaxis_title": "Energy - E_F (eV)" if efermi is not None else "Energy (eV)",
            "yaxis_title": "Density of States",
            "x_hover_unit": " eV",
            "y_hover_unit": "",
            "energy_reference": "fermi" if efermi is not None else "absolute",
            "reference_axis": "x" if efermi is not None else None,
            "fermi_level_ev": _round_float(efermi),
            "source": "vasprun.xml",
        },
    }
    return summary, visualization


def _band_structure_result_payload(
    vasprun,
    files: dict,
    local_paths: dict,
) -> tuple[dict, dict | None]:
    from backend.remote_result_parser import band_structure_with_fermi_reference

    band_structure, fermi_reference = band_structure_with_fermi_reference(vasprun, local_paths)
    bands = getattr(band_structure, "bands", None) or {}
    kpoint_count = _band_kpoint_count(band_structure, bands)
    band_count = _band_count(bands)
    efermi = _float_or_none(getattr(band_structure, "efermi", None))
    gap = _band_gap_summary(band_structure)

    summary = {
        "available": band_structure is not None and bool(bands),
        "source": "vasprun.xml",
        "kpoints": files.get("kpoints", {}).get("path"),
        "bands": band_count,
        "kpoints_count": kpoint_count,
        "spin_channels": len(bands) if bands else None,
        "fermi_level_ev": _round_float(efermi),
        "band_gap_ev": gap["band_gap_ev"],
        "is_metal": gap["is_metal"],
        "direct_gap": gap["direct_gap"],
        # Path-only runs: the reference record withholds the Fermi-dependent properties.
        **(fermi_reference or {}),
    }
    if band_structure is None or not bands or not kpoint_count:
        return summary, None

    x_values = _band_distances(band_structure, kpoint_count)
    ticks = _band_ticks(band_structure, x_values)
    traces = []
    spin_polarized = len(bands) > 1
    for spin, matrix in bands.items():
        spin_label = _spin_label(spin) if spin_polarized else "Bands"
        spin_down = _is_spin_down(spin)
        rows = _matrix_rows(matrix)
        for band_index, row in enumerate(rows, start=1):
            if len(row) != len(x_values):
                continue
            if efermi is not None:
                y_values = [_round_float(value - efermi) for value in row]
            else:
                y_values = [_round_float(value) for value in row]
            traces.append(
                {
                    "name": spin_label,
                    "legendgroup": spin_label,
                    "showlegend": spin_polarized and band_index == 1,
                    "color": "#c94001" if spin_down else "#1a6aff",
                    "dash": "dash" if spin_down else "solid",
                    "y": y_values,
                }
            )

    if not traces:
        return summary, None

    visualization = {
        "id": "band_structure",
        "element_id": "workflow-visualization-band-structure",
        "kind": "line_plot",
        "title": "Band Structure",
        "subtitle": "Line-mode band path from vasprun.xml",
        "download_filename": "band_structure.png",
        "plot": {
            "x": x_values,
            "traces": traces,
            "xaxis_title": "K-point path",
            "yaxis_title": "Energy - E_F (eV)" if efermi is not None else "Energy (eV)",
            "yaxis_range": BAND_STRUCTURE_INITIAL_YAXIS_RANGE,
            "x_hover_unit": "",
            "y_hover_unit": " eV",
            "tickvals": ticks["tickvals"],
            "ticktext": ticks["ticktext"],
            "vertical_lines": ticks["tickvals"],
            "energy_reference": "fermi" if efermi is not None else "absolute",
            "reference_axis": "y" if efermi is not None else None,
            "fermi_level_ev": _round_float(efermi),
            "source": "vasprun.xml",
        },
    }
    return summary, visualization


def _total_dos_from_vasprun(vasprun):
    complete_dos = getattr(vasprun, "complete_dos", None)
    return getattr(vasprun, "tdos", None) or getattr(complete_dos, "tdos", None)


def _band_structure_from_vasprun(vasprun, local_paths: dict):
    existing = getattr(vasprun, "band_structure", None)
    if existing is not None:
        return existing

    get_band_structure = getattr(vasprun, "get_band_structure", None)
    if not callable(get_band_structure):
        return None

    kpoints_path = local_paths.get("kpoints")
    if kpoints_path is not None:
        return get_band_structure(
            kpoints_filename=str(kpoints_path),
            line_mode=True,
        )

    return get_band_structure(line_mode=True)


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []

    try:
        iterable = list(values)
    except TypeError:
        return []

    converted = []
    for value in iterable:
        numeric = _float_or_none(value)
        if numeric is not None:
            converted.append(numeric)

    return converted


def _matrix_rows(values: Any) -> list[list[float]]:
    if values is None:
        return []

    if hasattr(values, "tolist"):
        values = values.tolist()

    try:
        rows = list(values)
    except TypeError:
        return []

    converted = []
    for row in rows:
        if hasattr(row, "tolist"):
            row = row.tolist()
        converted_row = _as_float_list(row)
        if converted_row:
            converted.append(converted_row)

    return converted


def _band_kpoint_count(band_structure, bands: dict) -> int | None:
    kpoints = getattr(band_structure, "kpoints", None)
    if kpoints:
        return len(kpoints)

    for matrix in bands.values():
        rows = _matrix_rows(matrix)
        if rows:
            return len(rows[0])

    return None


def _band_count(bands: dict) -> int | None:
    counts = [
        len(_matrix_rows(matrix))
        for matrix in bands.values()
    ]
    return max(counts) if counts else None


def _band_distances(band_structure, kpoint_count: int) -> list[float]:
    distances = _as_float_list(getattr(band_structure, "distance", None))
    if len(distances) == kpoint_count:
        return [_round_float(value) for value in distances]

    return [float(index) for index in range(kpoint_count)]


def _band_ticks(band_structure, x_values: list[float]) -> dict:
    tickvals: list[float] = []
    ticktext: list[str] = []
    kpoints = getattr(band_structure, "kpoints", None) or []

    for index, kpoint in enumerate(kpoints):
        if index >= len(x_values):
            break
        label = _format_kpoint_label(getattr(kpoint, "label", None))
        if not label:
            continue
        value = x_values[index]
        if tickvals and abs(tickvals[-1] - value) < 1e-8:
            labels = ticktext[-1].split("|")
            if label not in labels:
                ticktext[-1] = ticktext[-1] + "|" + label
            continue
        tickvals.append(value)
        ticktext.append(label)

    return {
        "tickvals": tickvals,
        "ticktext": ticktext,
    }


def _band_gap_summary(band_structure) -> dict:
    if band_structure is None:
        return {
            "band_gap_ev": None,
            "is_metal": None,
            "direct_gap": None,
        }

    band_gap = None
    direct_gap = None
    try:
        gap = band_structure.get_band_gap()
        band_gap = _round_float(_float_or_none(gap.get("energy")))
        direct_gap = bool(gap.get("direct")) if gap.get("direct") is not None else None
    except Exception:
        pass

    is_metal = None
    is_metal_fn = getattr(band_structure, "is_metal", None)
    if callable(is_metal_fn):
        try:
            is_metal = bool(is_metal_fn())
        except Exception:
            is_metal = None

    return {
        "band_gap_ev": band_gap,
        "is_metal": is_metal,
        "direct_gap": direct_gap,
    }


def _format_kpoint_label(label) -> str:
    text = str(label or "").strip()
    if not text:
        return ""

    return "|".join(
        _format_single_kpoint_label(part)
        for part in text.split("|")
    )


def _format_single_kpoint_label(label: str) -> str:
    text = label.strip().replace("$", "")
    gamma_tokens = {
        "\\Gamma",
        "\\gamma",
        "Gamma",
        "GAMMA",
        "gamma",
        "Γ",
    }
    return "Γ" if text in gamma_tokens else text


def _spin_label(spin) -> str:
    name = str(getattr(spin, "name", "") or "").strip().lower()
    text = str(spin or "").strip().lower()
    value = getattr(spin, "value", None)

    if value == -1 or name == "down" or "down" in text:
        return "Spin down"
    if value == 1 or name == "up" or "up" in text:
        return "Spin up"

    return str(spin)


def _is_spin_down(spin) -> bool:
    name = str(getattr(spin, "name", "") or "").strip().lower()
    text = str(spin or "").strip().lower()
    return getattr(spin, "value", None) == -1 or name == "down" or "down" in text


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BandStructureResultRenderer",
    "DensityOfStatesResultRenderer",
    "WorkflowResultPayload",
    "WorkflowResultRenderer",
    "render_workflow_results",
    "PHONON_RESULT_FILES",
    "workflow_has_phonon_results",
    "workflow_result_file_keys",
    "workflow_result_parse_dos",
    "workflow_result_parse_eigenvalues",
    "workflow_result_renderers",
]
