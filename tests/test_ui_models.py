"""Model controls bind each request to its own settings without loading model libraries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vidfit import VideoAnalyzer
from vidfit.backends.registry import MODEL_PRESETS
from vidfit.ui import (
    UIController,
    model_settings_from_controls,
    native_trace_capability,
    result_card,
)

CONTROLS = ["uniform", 2, "", "manual", 2, "0, 0.5"]


def settings(model="example/model-a", **changes):
    return {
        "model": model,
        "backend": "inspect",
        "revision": "requested-tag",
        "dtype": "float32",
        "enable_thinking": False,
        "max_pixels": 262144,
        **changes,
    }


def test_request_models_preserve_baseline_and_label_both_results(short_video, tmp_path):
    baseline = VideoAnalyzer(backend="inspect", model="example/baseline", device="cpu")
    original = baseline.options.copy()
    controller = UIController(baseline, tmp_path)
    first_settings = settings()
    first_preview, *_ = controller.preview(
        short_video, "Answer once", *CONTROLS, model_settings=first_settings
    )
    first = controller.run(
        short_video,
        "Describe",
        "Answer once",
        CONTROLS,
        first_preview,
        model_settings=first_settings,
    )
    second_settings = settings("example/model-b", revision="other-tag", enable_thinking=True)
    second_preview, *_ = controller.preview(
        short_video, "Answer once", *CONTROLS, model_settings=second_settings
    )
    second = controller.run(
        short_video,
        "Describe",
        "Answer once",
        CONTROLS,
        second_preview,
        first[0],
        model_settings=second_settings,
    )
    assert baseline.options == original
    assert baseline.backend_name == "inspect"
    assert first_preview["model_settings"]["model"] == "example/model-a"
    assert second_preview["model_settings"]["model"] == "example/model-b"
    assert "example/model-b" in second[1] and "other-tag" in second[1]
    assert "example/model-a" in second[3] and "requested-tag" in second[3]
    assert "thinking requested: on" in second[1]
    assert "Inspection only; requested" in second[1]
    exported = json.loads(Path(second[6]).read_text(encoding="utf-8"))
    assert exported["settings"]["model"] == "example/model-b"
    assert exported["settings"]["device"] == "cpu"
    assert exported["settings"]["enable_thinking"] is True
    assert first_settings == settings()


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "example/model-b"},
        {"backend": "qwen"},
        {"revision": "other"},
        {"dtype": "bfloat16"},
        {"enable_thinking": True},
        {"max_pixels": 1048576},
    ],
)
def test_every_model_setting_invalidates_preview_before_inference(short_video, tmp_path, changed):
    controller = UIController(VideoAnalyzer(backend="inspect"), tmp_path / "exports")
    snapshot, *_ = controller.preview(
        short_video, "Answer once", *CONTROLS, model_settings=settings()
    )
    with pytest.raises(ValueError, match="Model settings changed.*Preview"):
        controller.run(
            short_video,
            "Describe",
            "Answer once",
            CONTROLS,
            snapshot,
            model_settings=settings(**changed),
        )
    assert not controller.output_root.exists()


def test_trace_uses_request_checkpoint_and_never_changes_baseline(monkeypatch, tmp_path):
    from vidfit.evaluation import sampling_trace

    calls = []

    def trace(analyzer, video, selection, **kwargs):
        calls.append((analyzer.backend_name, analyzer.options.copy(), kwargs))
        return {"run_id": "request-trace"}

    monkeypatch.setattr(VideoAnalyzer, "trace_sampling", trace)
    monkeypatch.setattr(sampling_trace, "render_sampling_trace", lambda *args, **kwargs: "trace")
    baseline = VideoAnalyzer(backend="inspect")
    controller = UIController(baseline, tmp_path)
    request = settings("example/custom-qwen", backend="qwen", revision="commit")
    controller.trace("clip.mp4", "uniform", 4, "", model_settings=request)
    assert calls[0][0] == "qwen"
    assert calls[0][1]["model"] == "example/custom-qwen"
    assert calls[0][1]["revision"] == "commit"
    assert baseline.backend_name == "inspect"
    assert baseline.options["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert (tmp_path / "request-trace/sampling-trace.json").is_file()


def test_generic_model_trace_is_explicitly_unavailable_before_loading(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported native trace must not load a checkpoint")

    monkeypatch.setattr(VideoAnalyzer, "trace_sampling", forbidden)
    preset = next(item for item in MODEL_PRESETS if not item["supports_native_trace"])
    controller = UIController(VideoAnalyzer(backend="inspect"), tmp_path)
    with pytest.raises(ValueError, match="Native sampling trace is unavailable"):
        controller.trace(
            "clip.mp4", "uniform", 4, "", model_settings=settings(preset["model"], backend="auto")
        )
    assert native_trace_capability("Qwen/Qwen3-VL-2B-Instruct", "transformers")[0] is False
    available, notice = native_trace_capability("example/custom-vlm", "auto")
    assert available and "checks checkpoint compatibility" in notice


@pytest.mark.parametrize("pixels", [None, True, float("nan"), 1.5])
def test_model_controls_reject_invalid_pixel_limits(pixels):
    with pytest.raises(ValueError, match="Pixel limit"):
        model_settings_from_controls("example/model", "auto", "main", "float16", False, pixels)


def test_result_model_label_escapes_checkpoint_text(short_video):
    result = VideoAnalyzer(backend="inspect").compare(
        short_video, "Describe", [2], execution_mode="single"
    )
    result.model["identifier"] = "<script>model</script>"
    result.model["resolved_revision"] = "<img>"
    markup = result_card(result)
    assert "<script>" not in markup and "<img>" not in markup
    assert "&lt;script&gt;model" in markup and "&lt;img&gt;" in markup


def test_worker_model_settings_cross_process_without_changing_defaults(short_video, tmp_path):
    from vidfit.ui_worker import LocalWorker

    controller = UIController(VideoAnalyzer(backend="inspect", model="example/baseline"), tmp_path)
    worker = LocalWorker(controller)
    try:
        request = settings("example/worker-model")
        snapshot, *_ = worker.preview(short_video, "Answer once", *CONTROLS, model_settings=request)
        result, *_ = worker.run(
            short_video, "Describe", "Answer once", CONTROLS, snapshot, model_settings=request
        )
        assert result.settings["model"] == "example/worker-model"
        assert controller.analyzer.options["model"] == "example/baseline"
        baseline, *_ = worker.preview(short_video, "Answer once", *CONTROLS)
        assert baseline["model_settings"]["model"] == "example/baseline"
    finally:
        worker.close()


def test_model_controls_and_worker_import_without_model_libraries():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vidfit.ui; import vidfit.ui_worker; "
            "vidfit.ui.native_trace_capability('example/custom', 'auto'); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.ui
@pytest.mark.skipif(os.getenv("VIDFIT_RUN_UI_TEST") != "1", reason="opt-in Gradio component check")
def test_model_ui_forwards_controls_and_clears_stale_work(monkeypatch, tmp_path):
    import gradio as gr

    from vidfit.ui import CUSTOM_MODEL, build_app
    from vidfit.ui_worker import LocalWorker

    calls = []

    def preview(self, *args, **kwargs):
        calls.append(("preview", args, kwargs))
        return {"preview": True}, [], "Preview ready"

    def trace(self, *args, **kwargs):
        calls.append(("trace", args, kwargs))
        return "trace", None, None, "Trace ready"

    def run(self, *args, **kwargs):
        calls.append(("run", args, kwargs))
        return None, "answer", [], "previous", [], None, None, "Completed"

    monkeypatch.setattr(LocalWorker, "preview", preview)
    monkeypatch.setattr(LocalWorker, "trace", trace)
    monkeypatch.setattr(LocalWorker, "run", run)
    app = build_app(
        VideoAnalyzer(backend="inspect", revision="old-pinned-commit"), output_root=tmp_path
    )
    try:
        functions = {fn.api_name: (key, fn) for key, fn in app.fns.items()}
        preview_id, preview_fn = functions["preview"]
        trace_id, trace_fn = functions["trace_sampling"]
        run_id, run_fn = functions["run"]
        model_components = preview_fn.inputs[-6:]
        assert [component.label for component in model_components] == [
            "Hugging Face model ID",
            "Backend",
            "Model revision",
            "Model precision",
            "Request thinking (when supported)",
            "Maximum pixels per frame",
        ]
        assert {value for _, value in model_components[1].choices} >= {
            "auto",
            "transformers",
            "inspect",
        }
        assert trace_fn.inputs[-6:] == run_fn.inputs[-6:] == model_components
        values = ["example/request-model", "auto", "pinned", "bfloat16", True, 1048576]
        assert (
            list(preview_fn.fn("clip.mp4", "Answer once", *CONTROLS, *values))[-1][2]
            == "Preview ready"
        )
        assert (
            list(trace_fn.fn("clip.mp4", "uniform", 4, "", "matched", 2, *values))[-1][3]
            == "Trace ready"
        )
        assert (
            list(
                run_fn.fn(
                    "clip.mp4",
                    "Describe",
                    "Answer once",
                    *CONTROLS,
                    {"preview": True},
                    None,
                    False,
                    0,
                    1,
                    None,
                    None,
                    *values,
                )
            )[-1][-1]
            == "Completed"
        )
        for _, _, kwargs in calls:
            assert kwargs["model_settings"] == model_settings_from_controls(*values)

        preset_control = next(
            item
            for item in app.blocks.values()
            if isinstance(item, gr.Dropdown) and item.label == "Model preset"
        )
        assert CUSTOM_MODEL in [value for _, value in preset_control.choices]
        select = next(fn for fn in app.fns.values() if fn.name == "select_preset")
        preset = next(item for item in MODEL_PRESETS if not item["supports_native_trace"])
        selected = select.fn(preset["model"])
        assert selected == (
            preset["model"],
            preset["backend"],
            preset["dtype"],
            preset["max_pixels"],
            "main",
            False,
        )
        assert select.outputs[-2:] == [model_components[2], model_components[4]]
        capability = next(fn for fn in app.fns.values() if fn.name == "update_trace_capability")
        button, notice = capability.fn(preset["model"], "auto")
        assert button["interactive"] is False and "unavailable" in notice

        invalidate = next(fn for fn in app.fns.values() if fn.name == "invalidate_model")
        assert invalidate.fn()[0] is None
        assert invalidate.fn()[3:] == ("", None, None, None, None)
        assert {component.label for component in invalidate.outputs[-4:]} == {
            "Sampling trace HTML",
            "Sampling trace JSON",
            "Offline HTML report",
            "Structured JSON",
        }
        for component in [preset_control, *model_components]:
            target = (component._id, "change")
            assert any(
                target in dependency["targets"]
                and dependency["outputs"] == [item._id for item in invalidate.outputs]
                for dependency in app.config["dependencies"]
            )
            assert any(
                target in dependency["targets"]
                and {preview_id, trace_id, run_id} <= set(dependency["cancels"])
                for dependency in app.config["dependencies"]
            )
    finally:
        app.close()
        app.vidfit_worker.close()
