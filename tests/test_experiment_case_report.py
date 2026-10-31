import pytest

from vidfit import VideoAnalyzer
from vidfit.core.profiling import summarize
from vidfit.evaluation.reports import render_html


def _summary_table(markup):
    advanced = markup.split("<summary>Advanced measurements</summary>", 1)[1]
    return advanced.split("<tbody>", 1)[1].split("</tbody>", 1)[0]


def test_experiment_report_includes_complete_request_costs(short_video):
    result = VideoAnalyzer(backend="inspect").compare(
        short_video, "Describe", [1, 2], execution_mode="single"
    )
    result.settings["execution_mode"] = "selection_experiment"
    selected, uniform = result.configurations
    selected.trials[0].metrics.update(
        end_to_end_s=3.25,
        input_preparation_s=0.5,
        probe_s=0.125,
        visual_change_scan_s=0.075,
        visual_change_capture_s=0.05,
    )
    uniform.trials[0].metrics.update(end_to_end_s=4.5, input_preparation_s=0.625, probe_s=0.125)
    for config in result.configurations:
        config.summary = summarize(config.trials)

    markup = render_html(result)
    table = _summary_table(markup)
    assert "<th>Complete request time</th><td>3.2500 s" in table
    assert "<td>4.5000 s" in table
    assert "<th>Input preparation (included in total)</th><td>0.5000 s" in table
    assert "<th>Metadata probe + hash</th><td>0.1250 s" in table
    assert "<th>Visual-change scan (within decode + sample)</th><td>0.0750 s" in table
    capture_row = table.split("<th>Visual-change frame capture (within decode + sample)</th>")[1]
    capture_row = capture_row.split("</tr>", 1)[0]
    assert "0.0500 s" in capture_row
    assert "Unavailable" in capture_row

    boundaries = markup.split("<h2>Measurement boundaries</h2>", 1)[1]
    assert "Complete request time includes the metadata probe and hash" in boundaries
    assert "all source decoding, frame selection, lossless PNG" in boundaries
    assert "device transfer" in boundaries
    assert "output text decoding, and final CUDA synchronization" in boundaries
    assert "do not add these subtotals again" in boundaries
    assert "Model/runtime startup, the experiment's two global warmups" in boundaries
    assert "report export are outside complete request time" in boundaries
    assert "initial metadata probe" not in boundaries
    assert "Warmups and the first invocation are excluded." not in markup
    assert "Inspection only" in markup


@pytest.mark.parametrize("mode", ["benchmark", "single", "selection_experiment"])
def test_optional_experiment_costs_are_absent_without_measurements(short_video, mode):
    result = VideoAnalyzer(backend="inspect").compare(
        short_video, "Describe", [2], execution_mode="single"
    )
    result.settings["execution_mode"] = mode
    result.configurations[0].summary["metrics"]["end_to_end_s"] = None
    markup = render_html(result)
    table = _summary_table(markup)
    for label in (
        "Complete request time",
        "Input preparation (included in total)",
        "Metadata probe + hash",
        "Visual-change scan (within decode + sample)",
        "Visual-change frame capture (within decode + sample)",
    ):
        assert f"<th>{label}</th>" not in table
    assert "<th>Pipeline latency</th>" in table
    if mode != "selection_experiment":
        assert "Pipeline latency starts before decoding" in markup
        assert "the initial metadata probe, initial gallery extraction" in markup
        assert "Warmups and the first invocation are excluded." in markup
        assert "two global warmups" not in markup
