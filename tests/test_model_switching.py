"""Frontend model routing remains dependency-light and preserves requested settings."""

import sys
from types import SimpleNamespace

import pytest

from vidfit import VideoAnalyzer
from vidfit.backends.inspection import InspectionBackend
from vidfit.cli import make_analyzer, parser


@pytest.mark.parametrize("backend", ["qwen", "qwen3vl"])
def test_backend_alias_forwards_model_revision_and_thinking(short_video, monkeypatch, backend):
    calls = []

    def construct(**options):
        calls.append(options)
        return InspectionBackend()

    monkeypatch.setitem(
        sys.modules, "vidfit.backends.qwen3vl", SimpleNamespace(QwenBackend=construct)
    )
    analyzer = VideoAnalyzer(
        backend=backend,
        model="Qwen/Qwen3.5-2B",
        revision="chosen-revision",
        dtype="bfloat16",
        enable_thinking=True,
    )
    result = analyzer.compare(short_video, "Describe", [2], execution_mode="single")
    assert result.complete
    assert len(calls) == 1
    assert calls[0]["model"] == "Qwen/Qwen3.5-2B"
    assert calls[0]["revision"] == "chosen-revision"
    assert calls[0]["dtype"] == "bfloat16"
    assert calls[0]["enable_thinking"] is True
    assert result.settings["generation"]["enable_thinking"] is True


@pytest.mark.parametrize("thinking", [[], ["--thinking"], ["--no-thinking"]])
def test_model_switch_cli_keeps_explicit_runtime_options(thinking):
    args = parser().parse_args(
        [
            "ui",
            "--model",
            "Qwen/Qwen3.5-2B",
            "--dtype",
            "bfloat16",
            *thinking,
        ]
    )
    analyzer = make_analyzer(args)
    assert analyzer.backend_name == "auto"
    assert analyzer.options["model"] == "Qwen/Qwen3.5-2B"
    assert analyzer.options["dtype"] == "bfloat16"
    assert analyzer.options["enable_thinking"] is (thinking == ["--thinking"])


def test_experiment_defaults_to_family_aware_backend():
    args = parser().parse_args(
        [
            "experiment",
            "--collection",
            "collection.json",
            "--video-dir",
            "videos",
            "--output-dir",
            "results",
            "--model",
            "Qwen/Qwen3.5-2B",
        ]
    )
    analyzer = make_analyzer(args)
    assert analyzer.backend_name == "auto"
    assert analyzer.options["enable_thinking"] is False


@pytest.mark.parametrize("value", [1, "false", None])
def test_thinking_option_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        VideoAnalyzer(enable_thinking=value)


def test_worker_startup_keeps_model_dependencies_optional(monkeypatch):
    from vidfit import ui_worker

    options = []
    monkeypatch.setitem(
        sys.modules,
        "vidfit.backends.qwen3vl",
        SimpleNamespace(preload_dependencies=lambda: pytest.fail("worker must not preload models")),
    )
    monkeypatch.setattr(
        ui_worker,
        "ProcessPoolExecutor",
        lambda **kwargs: options.append(kwargs) or SimpleNamespace(shutdown=lambda **kwargs: None),
    )
    worker = ui_worker.LocalWorker(SimpleNamespace(analyzer=VideoAnalyzer(backend="qwen")))
    assert "initializer" not in options[0]
    worker.close()
