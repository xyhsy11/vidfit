"""Sampling trace UI callbacks stay usable without model or UI dependencies."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from vidfit import SelectionSpec
from vidfit.ui import UIController


class TraceAnalyzer:
    def __init__(self):
        self.calls = []

    def trace_sampling(self, video, selection, **options):
        self.calls.append((video, selection, options))
        return {"run_id": "trace-only", "video": {"filename": "测试.mp4"}}

    def compare(self, *args, **kwargs):
        raise AssertionError("sampling traces must not invoke inference")

    def preview(self, *args, **kwargs):
        raise AssertionError("sampling traces must not depend on an inference preview")


@pytest.fixture
def trace_renderer(monkeypatch):
    calls = []
    module = ModuleType("vidfit.evaluation.sampling_trace")

    def render(trace, *, standalone):
        calls.append((trace, standalone))
        return "<html>standalone trace</html>" if standalone else "<section>inline trace</section>"

    module.render_sampling_trace = render
    monkeypatch.setitem(sys.modules, "vidfit.evaluation.sampling_trace", module)
    return calls


@pytest.mark.parametrize("mode,expected_fps", [("matched", None), ("default", None), ("fps", 1.5)])
def test_trace_uses_selection_a_and_exports_separate_artifacts(
    tmp_path, trace_renderer, mode, expected_fps
):
    analyzer = TraceAnalyzer()
    controller = UIController(analyzer, tmp_path)
    markup, html_path, json_path, status = controller.trace(
        "example.mp4", "manual", 4, "1, 3", mode, 1.5
    )
    assert analyzer.calls == [
        (
            "example.mp4",
            SelectionSpec.manual([1, 3]),
            {"native_mode": mode, "native_fps": expected_fps},
        )
    ]
    assert markup == "<section>inline trace</section>"
    assert Path(html_path).name == "sampling-trace.html"
    assert Path(html_path).read_text(encoding="utf-8") == "<html>standalone trace</html>"
    assert Path(json_path).name == "sampling-trace.json"
    assert (
        json.loads(Path(json_path).read_text(encoding="utf-8"))["video"]["filename"] == "测试.mp4"
    )
    assert [standalone for _, standalone in trace_renderer] == [False, True]
    assert "No model weights or inference" in status


@pytest.mark.parametrize(
    "video,method,timestamps,message",
    [(None, "uniform", "", "Choose a video"), ("example.mp4", "manual", "invalid", "timestamps")],
)
def test_trace_rejects_invalid_inputs_before_analysis(tmp_path, video, method, timestamps, message):
    analyzer = TraceAnalyzer()
    controller = UIController(analyzer, tmp_path)
    with pytest.raises(ValueError, match=message):
        controller.trace(video, method, 4, timestamps)
    assert not analyzer.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.ui
@pytest.mark.skipif(os.getenv("VIDFIT_RUN_UI_TEST") != "1", reason="opt-in Gradio component check")
def test_trace_controls_clear_failures_and_invalidate_pending_results(monkeypatch, tmp_path):
    from vidfit import VideoAnalyzer
    from vidfit.ui import build_app
    from vidfit.ui_worker import LocalWorker

    calls = []

    def unavailable(self, *args):
        calls.append(args)
        raise ImportError("Install the optional sampling dependencies")

    monkeypatch.setattr(LocalWorker, "trace", unavailable)
    app = build_app(VideoAnalyzer(backend="inspect"), output_root=tmp_path)
    try:
        trace_id, trace_fn = next(
            (key, fn) for key, fn in app.fns.items() if fn.api_name == "trace_sampling"
        )
        args = ("example.mp4", "uniform", 4, "", "matched", 2)
        results = list(trace_fn.fn(*args))
        assert calls == [args]
        assert results[0][:3] == ("", None, None)
        assert results[-1] == (
            "",
            None,
            None,
            "Trace failed: Install the optional sampling dependencies",
        )
        assert len(trace_fn.inputs) == 12
        native_mode, native_fps = trace_fn.inputs[4:6]
        assert native_mode.value == "matched"
        assert [value for _, value in native_mode.choices] == ["matched", "default", "fps"]
        assert native_fps.visible is False
        output_ids = [component._id for component in trace_fn.outputs]
        dependencies = app.config["dependencies"]
        for component in trace_fn.inputs:
            target = (component._id, "change")
            assert any(target in d["targets"] and d["outputs"] == output_ids for d in dependencies)
            assert any(target in d["targets"] and trace_id in d["cancels"] for d in dependencies)
        clear_fn = next(fn for fn in app.fns.values() if fn.name == "invalidate_trace")
        assert clear_fn.fn()[:3] == ("", None, None)
    finally:
        app.close()
        app.vidfit_worker.close()
