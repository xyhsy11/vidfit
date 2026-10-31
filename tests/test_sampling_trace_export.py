"""Public trace entry points and offline rendering stay separate from inference."""

import json

import pytest

from vidfit import SelectionSpec, VideoAnalyzer
from vidfit.cli import main
from vidfit.evaluation.sampling_trace import export_sampling_trace, render_sampling_trace


@pytest.fixture
def trace_record():
    selection = {
        "selection": {"method": "uniform"},
        "selected_frames": 1,
        "distinct_frames": 1,
        "duplicate_frame_indices": [],
        "frame_indices": [0],
        "frames": [
            {
                "position": 0,
                "frame_index": 0,
                "timestamp_s": 0.0,
                "fps_timestamp_s": 0.0,
                "target_timestamp_s": 0.25,
                "image_png_base64": "fixture",
            }
        ],
        "gallery": {"shown": 1, "total": 1},
        "temporal_pairs": [
            {
                "frame_indices": [0, 0],
                "fps_timestamp_s": 0,
                "prompt_timestamp_s": "0.0",
                "padding_positions": [1],
            }
        ],
    }
    return {
        "run_id": "test-trace",
        "created_at_utc": "2026-10-31T00:00:00Z",
        "video": {
            "filename": "sample.mp4",
            "frame_count": 2,
            "duration_s": 1,
            "sha256": "test-hash",
            "nominal_fps": 2,
        },
        "processor": {"settings": {"fps": 2, "min_frames": 4, "max_frames": 768}},
        "request": {"native_mode": "matched"},
        "vidfit": selection,
        "native": selection,
        "comparison": {"shared_frame_indices": [0]},
        "notes": [],
    }


def test_offline_trace_keeps_source_times_and_projected_groups(trace_record, tmp_path):
    markup, data = export_sampling_trace(trace_record, tmp_path)
    text = markup.read_text(encoding="utf-8")
    assert "VidFit selection" in text and "Native Qwen selection" in text
    assert "Decoded time" in text and "Index / FPS time" in text
    assert "Derived temporal groups" in text and "Projected" in text
    assert "No model weights loaded and no inference run" in text
    assert "data:image/png;base64,fixture" in text
    assert "<script" not in text and 'src="http' not in text and 'href="http' not in text
    assert json.loads(data.read_text(encoding="utf-8")) == trace_record
    assert "<!doctype html>" not in render_sampling_trace(trace_record, standalone=False)


def test_trace_escapes_video_and_processor_text(trace_record):
    trace_record["video"]["filename"] = '<script>alert("clip")</script>'
    trace_record["processor"]["model"] = '<img src=x onerror="bad()">'
    trace_record["notes"] = ['<svg onload="bad()">']
    markup = render_sampling_trace(trace_record)
    assert "<script>" not in markup and "<img src=x" not in markup and "<svg" not in markup
    assert "&lt;script&gt;" in markup and "&lt;svg" in markup


def test_trace_renders_observed_request_without_assuming_native_algorithm(trace_record):
    trace_record["processor"]["algorithm"] = "Installed custom sampler <source>"
    trace_record["native"]["selection"] = {
        "method": "qwen_native",
        "mode": "matched",
        "requested_count": 4,
        "returned_count": 1,
        "count_before_bounds": None,
        "count_note": "Internal count derivation and bounds are not instrumented.",
    }
    markup = render_sampling_trace(trace_record)
    assert "Installed custom sampler &lt;source&gt;" in markup
    assert "Requested count: 4 → returned 1" in markup
    assert "Internal count derivation and bounds are not instrumented." in markup
    assert "FPS limits do not apply" not in markup
    assert "ties-to-even" not in markup
    assert "bypasses" not in markup


def test_trace_does_not_guess_an_algorithm_for_older_records(trace_record):
    markup = render_sampling_trace(trace_record)
    assert "returned source indices are authoritative" in markup
    assert "ties-to-even" not in markup


def test_analyzer_trace_forwards_only_sampling_options(monkeypatch):
    from vidfit import native_trace

    calls = []
    monkeypatch.setattr(
        native_trace,
        "create_sampling_trace",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"trace": True},
    )
    analyzer = VideoAnalyzer(backend="inspect", model="example/model", revision="abc")
    selection = SelectionSpec.manual([0, 1])
    assert analyzer.trace_sampling("clip.mp4", selection, native_mode="fps", native_fps=1) == {
        "trace": True
    }
    assert calls == [
        (
            ("clip.mp4", selection),
            {
                "model": "example/model",
                "revision": "abc",
                "native_mode": "fps",
                "native_fps": 1,
            },
        )
    ]


def test_trace_cli_exports_without_question_or_inference(trace_record, monkeypatch, tmp_path):
    calls = []

    def trace(self, video, selection, **kwargs):
        calls.append((selection, kwargs))
        return trace_record

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling trace must not run inference")

    monkeypatch.setattr(VideoAnalyzer, "trace_sampling", trace)
    monkeypatch.setattr(VideoAnalyzer, "compare", forbidden)
    assert (
        main(
            [
                "trace",
                "--video",
                "clip.mp4",
                "--timestamps",
                "0",
                "0.5",
                "--native-mode",
                "fps",
                "--native-fps",
                "0.5",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls == [(SelectionSpec.manual([0, 0.5]), {"native_mode": "fps", "native_fps": 0.5})]
    assert (tmp_path / "sampling-trace.html").is_file()
    assert (tmp_path / "sampling-trace.json").is_file()


def test_trace_cli_dependency_failure_is_actionable(monkeypatch, tmp_path, capsys):
    def missing(*args, **kwargs):
        raise RuntimeError('Native sampling trace requires: pip install ".[qwen]"')

    monkeypatch.setattr(VideoAnalyzer, "trace_sampling", missing)
    assert main(["trace", "--video", "clip.mp4", "--output-dir", str(tmp_path)]) == 2
    assert "[qwen]" in capsys.readouterr().err
    assert not (tmp_path / "sampling-trace.json").exists()
