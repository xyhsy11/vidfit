"""Shared routing must preserve model identity without forcing runtime imports in previews."""

import json
import sys
from types import SimpleNamespace

import pytest

from vidfit import SelectionSpec, VideoAnalyzer
from vidfit.backends import registry
from vidfit.backends.inspection import InspectionBackend
from vidfit.cli import main, make_analyzer, parser


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_5", "smolvlm", "llava_onevision"])
def test_auto_routes_by_configuration_and_preserves_requested_revision(monkeypatch, family):
    from vidfit.backends import qwen_runtime

    calls = []
    config = SimpleNamespace(model_type=family, _commit_hash="a" * 40)

    def configure(model, **options):
        calls.append(("config", model, options))
        return config

    def construct(**options):
        calls.append(("backend", options))
        return SimpleNamespace(info={"loading_s": 3, "requested_revision": options["revision"]})

    monkeypatch.setattr(qwen_runtime, "transformers_version", lambda: "5.5.4")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=configure)),
    )
    monkeypatch.setitem(
        sys.modules, "vidfit.backends.qwen3vl", SimpleNamespace(QwenBackend=construct)
    )
    monkeypatch.setitem(
        sys.modules,
        "vidfit.backends.transformers_vlm",
        SimpleNamespace(TransformersVLMBackend=construct),
    )
    clock = iter([100.0, 112.0])
    monkeypatch.setattr(registry, "perf_counter", lambda: next(clock))
    backend = registry.create_backend("auto", model="custom/repository", revision="my-tag")
    assert calls[0] == (
        "config",
        "custom/repository",
        {"revision": "my-tag", "trust_remote_code": False},
    )
    assert backend.info["requested_revision"] == "my-tag"
    assert backend.info["loading_s"] == 12
    routed = calls[1][1]
    if family.startswith("qwen"):
        assert routed["revision"] == "a" * 40
        assert "config" not in routed
    else:
        assert routed["config"] is config
        assert routed["transformers_version"] == "5.5.4"


def test_inspection_needs_no_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert isinstance(registry.create_backend("inspect"), InspectionBackend)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        registry.create_backend("unknown")


@pytest.mark.parametrize("backend", ["auto", "transformers"])
def test_non_qwen_preview_accepts_single_frame_without_loading_runtime(
    short_video, monkeypatch, backend
):
    monkeypatch.setitem(sys.modules, "transformers", None)
    analyzer = VideoAnalyzer(backend=backend, model="custom/image-model")
    _, frames, _ = analyzer.preview(short_video, SelectionSpec.uniform(1))
    assert len(frames) == 1


def test_generic_compare_uses_shared_factory_and_allows_single_frame(short_video, monkeypatch):
    calls = []

    def create(name, **options):
        calls.append((name, options))
        return InspectionBackend()

    monkeypatch.setattr("vidfit.analyzer.create_backend", create)
    result = VideoAnalyzer(backend="transformers", model="custom/image-model").compare(
        short_video, "Describe", selections=[SelectionSpec.uniform(1)], execution_mode="single"
    )
    assert result.complete
    assert calls[0][0] == "transformers"
    assert calls[0][1]["model"] == "custom/image-model"


def test_image_adapter_has_no_native_sampling_trace():
    with pytest.raises(ValueError, match="only for supported Qwen"):
        VideoAnalyzer(backend="transformers").trace_sampling("unused", SelectionSpec.uniform(4))


def test_model_catalog_and_custom_cli_selection(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert main(["models", "--json"]) == 0
    presets = json.loads(capsys.readouterr().out)
    assert {item["model"] for item in presets} >= {
        "Qwen/Qwen3.5-2B",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    }
    args = parser().parse_args(["ui", "--backend", "transformers", "--model", "custom/model"])
    analyzer = make_analyzer(args)
    assert analyzer.backend_name == "transformers"
    assert analyzer.options["model"] == "custom/model"
