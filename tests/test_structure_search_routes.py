"""
Browser routes for importing USPEX results and handing candidates off (#63).

The tests post the real Si8 pilot output (tests/fixtures/uspex_si8_pilot/)
as an archive, read the forms out of the rendered page with an HTML parser
and post them on, as a browser would: import -> hand-off -> Build -> Prepare.
Remote work (SSH) is replaced by a stub in the Prepare test.
"""

from __future__ import annotations

import html
import io
import json
import tarfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from backend.structure_search import imports


PILOT = Path(__file__).parent / "fixtures" / "uspex_si8_pilot"


def pilot_tar_gz() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(PILOT / "results1", arcname="si8_search/results1")
    return buffer.getvalue()


class _Forms(HTMLParser):
    """Collects every form's action and its named fields."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms: list[dict] = []
        self._form = None
        self._textarea = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self._form = {"action": attrs.get("action"), "target": attrs.get("target"), "fields": []}
            self.forms.append(self._form)
        elif self._form is not None and tag == "input" and attrs.get("name"):
            if attrs.get("type") == "checkbox" and "checked" not in attrs:
                return
            self._form["fields"].append((attrs["name"], attrs.get("value", "")))
        elif self._form is not None and tag == "textarea" and attrs.get("name"):
            self._textarea = [attrs["name"], ""]
        elif self._form is not None and tag == "select" and attrs.get("name"):
            self._select = attrs["name"]
        elif self._form is not None and tag == "option" and "selected" in attrs:
            self._form["fields"].append((self._select, attrs.get("value", "")))

    def handle_data(self, data):
        if self._textarea is not None:
            self._textarea[1] += data

    def handle_endtag(self, tag):
        if tag == "textarea" and self._textarea is not None:
            self._form["fields"].append(tuple(self._textarea))
            self._textarea = None
        elif tag == "form":
            self._form = None


def form_data(fields) -> dict:
    """Form fields as httpx expects them (repeated names become lists)."""

    data: dict = {}
    for name, value in fields:
        data.setdefault(name, []).append(value)
    return {name: values[0] if len(values) == 1 else values for name, values in data.items()}


def forms(page: str, action: str) -> list[dict]:
    parser = _Forms()
    parser.feed(page)
    return [form for form in parser.forms if form["action"] == action]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(imports, "DEFAULT_UPLOADS_ROOT", tmp_path / "uploads")
    return TestClient(main.app)


def upload(client, data: bytes, filename: str = "si8.tar.gz"):
    return client.post(
        "/structure-search/import",
        files={"archive": (filename, data, "application/octet-stream")},
    )


@pytest.fixture
def imported(client):
    response = upload(client, pilot_tar_gz())
    assert response.status_code == 200, response.text
    return response


# --- Candidate page ----------------------------------------------------------


def test_workbench_offers_the_import(client):
    page = client.get("/").text
    assert 'action="/structure-search/import"' in page
    assert 'enctype="multipart/form-data"' in page
    assert "Structure Search Candidates" not in page


def test_candidate_page_renders(imported):
    page = imported.text
    assert "Structure Search Candidates" in page
    assert "si8.tar.gz · results1 · 35 individuals · 32 candidates" in page
    row = page[page.index('data-candidate="results1:35"') :]
    row = row[: row.index("</tr>")]
    cells = [html.unescape(cell.split(">", 1)[1].split("<", 1)[0]).strip()
             for cell in row.split("<td")[1:]]
    assert cells[1:] == [
        "1", "35", "results1", "4", "keptBest", "Si8",
        "-43.280", "-5.4100", "20.46", "1", "Fd-3m (227)",
    ]
    assert "search-grade" in page
    assert "Notes from reading the results" not in page  # the pilot has no warnings


def test_candidate_plot_uses_the_shared_renderer(imported):
    page = imported.text
    assert "https://cdn.plot.ly/plotly-2.35.2.min.js" in page
    data = page.split('id="workflow-result-visualizations">', 1)[1].split("</script>", 1)[0]
    (visualization,) = json.loads(data)
    assert visualization["element_id"] == "structure-search-energy-plot"
    assert visualization["kind"] == "line_plot"
    assert 'id="structure-search-energy-plot"' in page
    assert "renderLinePlot(visualization)" in page


def test_import_can_be_reopened(client, imported):
    (form,) = forms(imported.text, "/structure-search/hand-off")
    import_id = dict(form["fields"])["import_id"]
    again = client.get(f"/structure-search/{import_id}")
    assert again.status_code == 200
    assert 'data-candidate="results1:35"' in again.text
    missing = client.get("/structure-search/" + "0" * 32)
    assert missing.status_code == 400
    assert "no longer available" in missing.text


def test_parser_warnings_are_shown(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("results1/Individuals", (PILOT / "results1" / "Individuals").read_text()[:-20])
        archive.writestr("results1/gatheredPOSCARS", (PILOT / "results1" / "gatheredPOSCARS").read_bytes())
    response = upload(client, buffer.getvalue(), "cut.zip")
    assert response.status_code == 200
    assert "Notes from reading the results" in response.text
    assert "results1/Individuals: last line" in response.text


# --- Clear messages for bad uploads -----------------------------------------


def test_missing_file(client):
    response = client.post("/structure-search/import", data={})
    assert response.status_code == 400
    assert "Import Failed" in response.text
    assert "No file was uploaded" in response.text


def test_unknown_file_format(client):
    response = upload(client, b"generation number num_atoms_all energy\n", "Individuals")
    assert response.status_code == 400
    assert "not a .zip or .tar.gz archive" in response.text
    assert "Upload a .zip or .tar.gz of the USPEX calculation folder" in response.text


def test_unsafe_archive_is_refused(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../results1/Individuals", b"x")
    response = upload(client, buffer.getvalue(), "evil.zip")
    assert response.status_code == 400
    assert "&#39;..&#39; path component" in response.text


def test_archive_without_poscars(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("results1/Individuals", (PILOT / "results1" / "Individuals").read_bytes())
    response = upload(client, buffer.getvalue(), "only.zip")
    assert response.status_code == 400
    assert "gatheredPOSCARS (or good_structures_POSCARS)" in response.text


# --- Hand-off ----------------------------------------------------------------


def hand_off(client, imported, selections, workflow="relaxed_energy"):
    (form,) = forms(imported.text, "/structure-search/hand-off")
    fields = [(name, value) for name, value in form["fields"] if name not in ("candidates", "workflow")]
    fields += [("candidates", value) for value in selections] + [("workflow", workflow)]
    return client.post("/structure-search/hand-off", data=form_data(fields))


def test_hand_off_without_selection(client, imported):
    response = hand_off(client, imported, [])
    assert response.status_code == 400
    assert "No candidate was selected" in response.text
    assert 'data-candidate="results1:35"' in response.text  # the table stays


def test_hand_off_rejects_unknown_desired_output(client, imported):
    response = hand_off(client, imported, ["results1:35"], workflow="custom")
    assert response.status_code == 400
    assert "Unknown Desired Output" in response.text


def test_hand_off_lists_one_build_form_per_candidate(client, imported):
    response = hand_off(client, imported, ["results1:35", "results1:20"], workflow="phonons")
    assert response.status_code == 200
    builds = forms(response.text, "/build-calculation")
    handoff_builds = [form for form in builds if "structure_origin_token" in dict(form["fields"])]
    assert len(handoff_builds) == 2
    assert all(form["target"] == "_blank" for form in handoff_builds)
    assert all(dict(form["fields"])["workflow"] == "phonons" for form in handoff_builds)
    assert "Build Phonons" in response.text


def test_hand_off_of_candidate_35_builds_relaxed_energy_with_uspex_provenance(
    client, imported, monkeypatch
):
    response = hand_off(client, imported, ["results1:35"])
    assert response.status_code == 200
    (build_form,) = [
        form
        for form in forms(response.text, "/build-calculation")
        if "structure_origin_token" in dict(form["fields"])
    ]
    assert build_form["target"] is None
    fields = dict(build_form["fields"])
    assert fields["workflow"] == "relaxed_energy"
    assert fields["structure"].startswith("USPEX individual 35 (results1)")

    built = client.post("/build-calculation", data=form_data(build_form["fields"]))
    assert built.status_code == 200, built.text
    submission_spec = built.context["submission_spec"]
    flow_spec = submission_spec["flow_spec"]
    # A normal Relaxed Energy build of a pasted POSCAR ...
    assert flow_spec["workflow_spec"]["recipe"] == "relaxed_energy"
    assert [stage["stage_type"] for stage in flow_spec["workflow_spec"]["stages"]] == [
        "relax", "relax", "static",
    ]
    assert flow_spec["structure"]["type"] == "pasted_text"
    assert flow_spec["structure"]["text"] == fields["structure"]
    # ... whose provenance carries the USPEX record.
    source = submission_spec["provenance"]["structure_source"]
    assert source["type"] == "pasted_text"
    origin = source["origin"]
    assert origin == flow_spec["structure"]["origin"]
    page = imported.context["structure_search"]
    assert origin["archive_sha256"] == page["archive"]["sha256"]
    assert origin["results_folder"] == "results1"
    assert origin["individual_id"] == 35
    assert origin["uspex_energy_ev"] == pytest.approx(-43.28)
    assert origin["origin"] == "keptBest"
    assert "Structure origin" in built.text
    assert page["archive"]["sha256"] in built.text

    # Prepare rebuilds the submission from the posted form; the origin must
    # survive that step too. SSH is replaced by a stub.
    captured = {}

    def fake_prepare(spec, **kwargs):
        captured["spec"] = spec
        return main.remembered_successful_preparation(spec)

    monkeypatch.setattr(main, "prepare_remote_submission", fake_prepare)
    (prepare_form,) = forms(built.text, "/prepare-remote")
    assert "structure_origin_token" in dict(prepare_form["fields"])
    (submit_form,) = forms(built.text, "/submit")
    assert "structure_origin_token" in dict(submit_form["fields"])
    prepared = client.post("/prepare-remote", data=form_data(prepare_form["fields"]))
    assert prepared.status_code == 200, prepared.text
    prepared_spec = captured["spec"]
    assert prepared_spec["provenance"]["structure_source"]["origin"] == origin
    assert prepared_spec["submission"]["attempt_id"] == submission_spec["submission"]["attempt_id"]


def test_edited_structure_loses_its_origin(client, imported):
    response = hand_off(client, imported, ["results1:35"])
    (build_form,) = [
        form
        for form in forms(response.text, "/build-calculation")
        if "structure_origin_token" in dict(form["fields"])
    ]
    fields = [
        (name, value.replace("USPEX individual 35", "edited") if name == "structure" else value)
        for name, value in build_form["fields"]
    ]
    built = client.post("/build-calculation", data=form_data(fields))
    assert built.status_code == 400
    assert "changed after it was taken from its source" in built.text
    assert 'name="structure_origin_token"' not in built.text


def test_pasted_structure_has_no_origin(client):
    poscar = (PILOT / "results1" / "best_structure_POSCAR").read_text()
    built = client.post(
        "/build-calculation",
        data={"structure": poscar, "fmt": "poscar", "workflow": "relaxed_energy"},
    )
    assert built.status_code == 200, built.text
    source = built.context["submission_spec"]["provenance"]["structure_source"]
    assert source == {"type": "pasted_text", "format": "poscar", "origin": None}
    assert "Structure origin" not in built.text
