"""Runtime dispatch doubles never import model libraries or load model weights."""

import importlib.metadata
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vidfit.backends import qwen3vl, qwen_runtime
from vidfit.core.video import probe_video, sample_video


def runtime_double(monkeypatch, family="qwen3_vl", version="4.57.6"):
    calls = []
    config = SimpleNamespace(model_type=family, _commit_hash="resolved-commit")

    class AutoConfig:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append(("config", model, kwargs))
            return config

    class ModelDouble:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append(("weights", model, kwargs))
            return cls()

        def to(self, device):
            calls.append(("device", device.type))
            return self

        def eval(self):
            return self

    class Qwen3ModelDouble(ModelDouble):
        pass

    class Qwen35ModelDouble(ModelDouble):
        pass

    class VideoProcessor:
        temporal_patch_size = merge_size = 2

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append(("video_processor", model, kwargs))
            return cls()

        def sample_frames(self):
            return []

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append(("processor", model, kwargs))
            return SimpleNamespace(
                video_processor=VideoProcessor(),
                apply_chat_template=lambda messages, **kwargs: (
                    "thinking" if kwargs["enable_thinking"] else "direct"
                ),
            )

    module = SimpleNamespace(
        AutoConfig=AutoConfig,
        Qwen3VLForConditionalGeneration=Qwen3ModelDouble,
        Qwen3_5ForConditionalGeneration=Qwen35ModelDouble,
        Qwen3VLVideoProcessor=VideoProcessor,
        AutoProcessor=AutoProcessor,
    )
    torch = SimpleNamespace(
        device=lambda name: SimpleNamespace(type=name, index=None),
        float32="float32",
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setattr(qwen_runtime.importlib.metadata, "version", lambda name: version)
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(qwen3vl, "preload_dependencies", lambda: (torch, AutoProcessor))
    return config, module, calls


def test_runtime_import_does_not_import_model_libraries():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'numpy'):
            raise AssertionError('heavy runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
import vidfit.backends.qwen_runtime
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("family", "version", "class_name"),
    [
        ("qwen3_vl", "4.57.6", "Qwen3VLForConditionalGeneration"),
        ("qwen3_vl", "5.5.4", "Qwen3VLForConditionalGeneration"),
        ("qwen3_5", "5.5.4", "Qwen3_5ForConditionalGeneration"),
    ],
)
def test_config_and_trace_dispatch_without_weights(monkeypatch, family, version, class_name):
    config, module, calls = runtime_double(monkeypatch, family, version)
    loaded, installed = qwen_runtime.load_config("test/checkpoint", "requested-tag")
    assert loaded is config and installed == version
    assert qwen_runtime.load_model_class(config, version) is getattr(module, class_name)
    processor = qwen_runtime.load_video_processor(config, "test/checkpoint", config._commit_hash)
    assert processor.temporal_patch_size == processor.merge_size == 2
    assert calls == [
        (
            "config",
            "test/checkpoint",
            {"revision": "requested-tag", "trust_remote_code": False},
        ),
        (
            "video_processor",
            "test/checkpoint",
            {"revision": "resolved-commit", "trust_remote_code": False},
        ),
    ]


@pytest.mark.parametrize("version", ["4.57.5", "4.57.6rc1", "6.0.0", "6.0.0.dev1"])
def test_outside_runtime_range_fails_before_config(monkeypatch, version):
    _, _, calls = runtime_double(monkeypatch, version=version)
    with pytest.raises(RuntimeError, match="requires Transformers"):
        qwen_runtime.load_config("test/checkpoint", "main")
    assert calls == []


@pytest.mark.parametrize("family", ["qwen3_vl_moe", "qwen3_5_moe", "qwen3_5_text", "llama"])
def test_unsupported_family_never_loads_weights(monkeypatch, family):
    _, _, calls = runtime_double(monkeypatch, family, "5.5.4")
    with pytest.raises(ValueError, match="dense Qwen3"):
        qwen3vl.QwenBackend("test/checkpoint", "cpu", "float32", 262144, 64, "main")
    assert [call[0] for call in calls] == ["config"]


def test_missing_runtime_and_missing_native_class_have_actionable_errors(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(qwen_runtime.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match=r"install.*qwen35"):
        qwen_runtime.load_config("test/checkpoint", "main")

    _, module, calls = runtime_double(monkeypatch, "qwen3_5", "5.5.4")
    del module.Qwen3_5ForConditionalGeneration
    with pytest.raises(RuntimeError, match=r"Qwen3_5ForConditionalGeneration.*qwen35"):
        qwen3vl.QwenBackend("test/checkpoint", "cpu", "float32", 262144, 64, "main")
    assert [call[0] for call in calls] == ["config"]


@pytest.mark.parametrize("autoconfig_rejects", [False, True])
def test_qwen35_on_old_runtime_fails_with_upgrade_instruction(monkeypatch, autoconfig_rejects):
    _, module, calls = runtime_double(monkeypatch, "qwen3_5", "4.57.6")
    if autoconfig_rejects:

        def reject(*args, **kwargs):
            raise ValueError("model type qwen3_5 is not recognized by Transformers")

        module.AutoConfig.from_pretrained = reject
    with pytest.raises(RuntimeError, match="qwen35"):
        qwen3vl.QwenBackend("test/checkpoint", "cpu", "float32", 262144, 64, "main")
    assert not any(call[0] == "weights" for call in calls)


def test_unpinned_checkpoint_and_incompatible_video_processor_are_rejected(monkeypatch):
    config, module, calls = runtime_double(monkeypatch)
    config._commit_hash = None
    with pytest.raises(ValueError, match="commit"):
        qwen_runtime.load_config("test/checkpoint", "main")
    config._commit_hash = "resolved-commit"
    module.Qwen3VLVideoProcessor.temporal_patch_size = 4
    with pytest.raises(ValueError, match="temporal patch"):
        qwen_runtime.load_video_processor(config, "test/checkpoint", config._commit_hash)
    assert not any(call[0] == "weights" for call in calls)


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_5"])
@pytest.mark.parametrize("thinking", [False, True])
def test_backend_loads_correct_class_at_pinned_revision(monkeypatch, family, thinking):
    config, module, calls = runtime_double(monkeypatch, family, "5.5.4")
    backend = qwen3vl.QwenBackend(
        "test/checkpoint", "cpu", "float32", 262144, 64, "requested-tag", thinking
    )
    expected = (
        module.Qwen3VLForConditionalGeneration
        if family == "qwen3_vl"
        else module.Qwen3_5ForConditionalGeneration
    )
    assert isinstance(backend.model, expected)
    assert backend.info["model_type"] == family
    assert backend.info["transformers_version"] == "5.5.4"
    assert backend.info["enable_thinking"] is thinking
    assert backend.info["thinking"]["effective"] is thinking
    assert backend.info["thinking"]["template_controlled"] is True
    assert backend.info["requested_revision"] == "requested-tag"
    assert backend.info["resolved_revision"] == backend.info["processor_resolved_revision"]
    for kind, _, kwargs in (call for call in calls if call[0] in ("processor", "weights")):
        assert kwargs["revision"] == config._commit_hash
        assert kwargs["trust_remote_code"] is False
        if kind == "weights":
            assert kwargs["config"] is config
            assert kwargs["attn_implementation"] == "sdpa"
    assert qwen3vl.Qwen3VLBackend is qwen3vl.QwenBackend


@pytest.mark.parametrize("thinking", [False, True])
def test_thinking_is_explicit_in_template_and_preserves_input_checks(
    monkeypatch, short_video, thinking
):
    from test_processor import MetadataDouble, ProcessorDouble

    monkeypatch.setitem(
        sys.modules, "transformers.video_utils", SimpleNamespace(VideoMetadata=MetadataDouble)
    )
    metadata = probe_video(short_video)
    frames, _ = sample_video(short_video, metadata, 2)

    class TemplateProcessor(ProcessorDouble):
        def apply_chat_template(self, messages, **kwargs):
            assert isinstance(kwargs["enable_thinking"], bool)
            assert messages[0]["content"][1] == {"type": "text", "text": "Question only"}
            return super().apply_chat_template(messages, **kwargs) + str(kwargs["enable_thinking"])

    processor = TemplateProcessor()
    _, processing, _ = qwen3vl.prepare_video_inputs(
        processor, frames, metadata, "Question only", 262144, enable_thinking=thinking
    )
    assert processing["enable_thinking"] is thinking
    assert processing["input_frame_hashes"] == [f.sha256_rgb for f in frames]
    assert processing["validated"] is True
    with pytest.raises(ValueError, match="boolean"):
        qwen3vl.prepare_video_inputs(
            processor, frames, metadata, "Question only", 262144, enable_thinking="false"
        )


def test_uncontrolled_thinking_is_never_silently_claimed(monkeypatch):
    _, module, calls = runtime_double(monkeypatch)
    processor = SimpleNamespace(
        video_processor=module.Qwen3VLVideoProcessor(),
        apply_chat_template=lambda messages, **kwargs: "fixed native template <think>",
    )
    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    backend = qwen3vl.QwenBackend("test/checkpoint", "cpu", "float32", 262144, 64, "main")
    assert backend.info["thinking"]["requested"] is False
    assert backend.info["thinking"]["effective"] is None
    assert backend.info["thinking"]["template_controlled"] is False
    calls.clear()
    with pytest.raises(ValueError, match="does not support enable_thinking"):
        qwen3vl.QwenBackend(
            "test/checkpoint", "cpu", "float32", 262144, 64, "main", enable_thinking=True
        )
    assert not any(call[0] == "weights" for call in calls)


def test_bfloat16_support_is_checked_on_requested_gpu(monkeypatch):
    _, module, calls = runtime_double(monkeypatch, "qwen3_5", "5.5.4")
    active = []

    @contextmanager
    def device_context(device):
        active.append(device.index)
        try:
            yield
        finally:
            active.pop()

    def supported():
        assert active == [1]
        return False

    torch = SimpleNamespace(
        device=lambda name: SimpleNamespace(type="cuda", index=int(name.split(":")[1])),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            device=device_context,
            is_bf16_supported=supported,
        ),
    )
    monkeypatch.setattr(qwen3vl, "preload_dependencies", lambda: (torch, module.AutoProcessor))
    with pytest.raises(ValueError, match="bfloat16 is not supported"):
        qwen3vl.QwenBackend("test/checkpoint", "cuda:1", "bfloat16", 262144, 64, "main")
    assert active == []
    assert not any(call[0] == "weights" for call in calls)
