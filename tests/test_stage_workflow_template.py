from __future__ import annotations

from pathlib import Path


source = Path("templates/index.html").read_text(encoding="utf-8")

assert 'name="workflow_spec_json"' in source
assert 'id="desired-output-select"' in source
assert 'id="workflow-stage-list"' in source
assert 'data-workflow-stage' in source
assert 'data-stage-type' in source
assert 'data-stage-theory' in source
assert 'data-stage-modifier' in source
assert 'data-stage-dispersion-method' not in source
assert 'data-dispersion-control' not in source
assert 'id="add-workflow-stage"' in source
assert "Desired Output" in source
assert "Recommended Workflow" not in source
assert "desired_outputs" in source
assert "Custom Workflow" in source
assert "BMD Compute workflow" in source
assert "Calculation Type" in source
assert "Level of Theory" in source
assert "Advanced Options" in source
assert "default_dispersion_method" not in source
assert "dispersion_methods" not in source
assert "DFT-D3" not in source
assert 'id="structure-input-details"' in source
structure_details_block = source[
    source.index('id="structure-input-details"'):
    source.index('<summary>Structure Input</summary>')
]
assert "details-wide" in structure_details_block
assert "{% if not collapse_structure_input %}open{% endif %}" in structure_details_block
calculation_definition = source[source.index('<form\n                id="calculation-review-form"'):]
# Workflow choice, Execution Resources, the Build button, and only then the
# Method Considerations of the built calculation (outside the form).
assert calculation_definition.index("<h3>Scientific Specification</h3>") < calculation_definition.index("<h3>Execution Resources</h3>")
assert calculation_definition.index("<h3>Execution Resources</h3>") < calculation_definition.index("data-build-calculation-button")
assert calculation_definition.index("</form>") < calculation_definition.index("<h3>Method Considerations</h3>")
resource_panel = calculation_definition[
    calculation_definition.index("<h3>Execution Resources</h3>"):
    calculation_definition.index("data-build-calculation-button")
]
assert resource_panel.index('name="cpus"') < resource_panel.index('name="memory_gb"')
assert resource_panel.index('name="memory_gb"') < resource_panel.index('name="walltime"')
assert resource_panel.index('name="walltime"') < resource_panel.index('name="queue"')
memory_label_index = resource_panel.index("<label>Memory (GB)</label>")
memory_select_index = resource_panel.index('<select name="memory_gb">')
memory_select_end = resource_panel.index("</select>", memory_select_index)
memory_select_block = resource_panel[memory_select_index:memory_select_end]
assert memory_label_index < memory_select_index
assert "selected_resources.allowed_memory_gb" in memory_select_block
assert 'value="{{ memory_gb }}"' in memory_select_block
assert "{% if selected_resources.memory_gb == memory_gb %}selected{% endif %}" in memory_select_block
assert "{{ memory_gb }} GB" in memory_select_block
queue_label_index = resource_panel.index("<label>Queue</label>")
queue_select_index = resource_panel.index('<select name="queue">')
queue_select_end = resource_panel.index("</select>", queue_select_index)
queue_select_block = resource_panel[queue_select_index:queue_select_end]
assert queue_label_index < queue_select_index
assert "selected_resources.allowed_queues" in queue_select_block
assert 'value="{{ queue }}"' in queue_select_block
assert "{% if selected_resources.queue == queue %}selected{% endif %}" in queue_select_block
assert 'type="number"' not in resource_panel
assert 'step="1"' not in resource_panel
assert "grid-template-columns: minmax(0, 1fr);" in source
assert "grid-template-columns: repeat(4, minmax(140px, 1fr));" in source
assert "workflow_spec_json" in source[source.index('<form action="/prepare-remote"'):]
assert "workflow_spec_json" in source[source.index('<form action="/submit"'):]
prepare_form = source[source.index('<form action="/prepare-remote"'):]
submit_form = source[source.index('<form action="/submit"'):]
prepare_form_only = prepare_form[:prepare_form.index("</form>")]
assert 'name="submission_attempt_id"' in prepare_form
assert 'name="submission_attempt_id"' in submit_form
assert 'name="submission_identity_token"' in prepare_form
assert 'name="submission_identity_token"' in submit_form
assert "submission_spec.submission.attempt_id" in prepare_form
assert "submission_spec.submission.attempt_id" in submit_form
assert "submission_spec.submission.identity_token" in prepare_form
assert "submission_spec.submission.identity_token" in submit_form
assert 'name="created_at"' not in prepare_form
assert 'name="created_at"' not in submit_form
assert "data-submit-calculation-button" in submit_form
assert 'form[action="/submit"]' in source
assert 'button.textContent = "Submitting..."' in source
assert "data-prepare-remote-button" in prepare_form
assert 'form[action="/prepare-remote"]' in source
assert 'form.dataset.preparing === "true"' in source
assert 'button.textContent = "Preparing..."' in source
assert "Preparation Time" in source
assert "Files Uploaded" in source
assert "Data Transferred" in source
assert "remote_preparation.diagnostics.total_preparation_s" in source
# Prepare Remote stays available for unsubmitted runs and is disabled only once
# the run has been submitted (its execution package is then immutable).
assert prepare_form_only.count("disabled") == 1
assert "{% if submission_result and submission_result.status == 'success' %}disabled" in prepare_form_only
monitor_form = source[source.index('<form action="/monitor"'):]
assert "workflow_spec_json" in monitor_form
assert "monitor_state_json" in monitor_form
assert "stageOptions.dispersion" not in source
assert "updateDispersionControls" not in source
assert "options: {}" in source
assert "desiredOutputs" in source
assert "recipeSelect" not in source
assert "recipes[index]" not in source
assert "desiredOutputSelect.value !== \"custom\"" in source
resume_monitoring_block = source[source.index('{% elif resume_job_id %}\n            <form action="/resume"'):]
assert 'name="load_results" value="true"' in resume_monitoring_block
assert "Load Results" in resume_monitoring_block
assert "Remote source bytes" in source
assert "Compact result bytes" in source
assert "remote_parse_elapsed_s" in source

print("stage workflow template smoke test passed")
