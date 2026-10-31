"""Generic image-chat integrity checks, without model libraries or model weights."""

import base64
import hashlib
import io
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vidfit.backends import transformers_vlm as vlm
from vidfit.core.schemas import FrameEvidence


def frames(size=(64, 32)):
    result = []
    for index, color in enumerate(("red", "blue")):
        image = Image.new("RGB", size, color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result.append(
            FrameEvidence(
                target_timestamp_s=index / 2,
                timestamp_s=index / 2,
                presentation_timestamp_s=index / 2,
                pts=index,
                time_base="1/2",
                frame_index=index,
                sha256_rgb=hashlib.sha256(image.tobytes()).hexdigest(),
                width=size[0],
                height=size[1],
                image_png_base64=base64.b64encode(buffer.getvalue()).decode(),
            )
        )
    return result


class TensorDouble:
    def __init__(self, shape, *, floating=False, values=None):
        self.shape, self.floating, self.values = shape, floating, values
        self.moves = []

    def is_floating_point(self):
        return self.floating

    def to(self, **kwargs):
        self.moves.append(kwargs)
        return self

    def __getitem__(self, item):
        values = self.values[item]
        return TensorDouble(values.shape, values=values)


class ProcessorDouble:
    def __init__(self, layout="batch", side=32):
        self.layout, self.side = layout, side
        self.image_processor = SimpleNamespace(max_image_size={"longest_edge": 512})
        self.calls, self.messages = [], []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return "".join(
            part["text"] if part["type"] == "text" else "<image>" for part in messages[0]["content"]
        )

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert len(kwargs["images"]) == len(kwargs["text"]) == 1
        count = len(kwargs["images"][0])
        tail = (3, self.side, self.side)
        shape = {
            "flat": (count, *tail),
            "batch": (1, count, *tail),
            "crops": (count, 2, *tail),
            "batch_crops": (1, count, 2, *tail),
            "onevision": (count, 1, *tail),
            "opaque": (count * 5, 1536),
            "unmapped": (count * 3, *tail),
        }[self.layout]
        return {
            "pixel_values": TensorDouble(shape, floating=True),
            "input_ids": TensorDouble((1, 3), values=np.array([[1, 2, 3]])),
            "attention_mask": TensorDouble((1, 3)),
        }

    def batch_decode(self, output, **kwargs):
        self.decoded = output
        return ["Complete answer, including any generated reasoning"]


@pytest.mark.parametrize(
    "layout, views", [("flat", 1), ("batch", 1), ("crops", 2), ("batch_crops", 2), ("onevision", 1)]
)
def test_timestamped_images_preserve_identity_and_nested_batch(layout, views):
    source = frames()
    processor = ProcessorDouble(layout)
    _, processing = vlm.prepare_image_inputs(processor, source, "What changed?", 4096)
    assert processing["input_representation"] == "timestamped_images"
    assert processing["native_video_input"] is False
    assert processing["input_frame_hashes"] == [frame.sha256_rgb for frame in source]
    assert processing["resized_input_frame_hashes"] == processing["input_frame_hashes"]
    assert processing["actual_selected_timestamps_s"] == [0, 0.5]
    assert processing["image_tiling"]["views_per_source_frame"] == views
    assert processing["image_tiling"]["budget_verified"] is True
    prompt = processor.calls[0]["text"][0]
    assert prompt.index("Frame 1 at 0.000000") < prompt.index("Frame 2 at 0.500000")
    assert prompt.endswith("What changed?") and prompt.count("<image>") == 2
    assert all(image.mode == "RGB" for image in processor.calls[0]["images"][0])
    assert "videos" not in processor.calls[0]


def test_explicit_downsize_records_both_hash_stages():
    source = frames((128, 64))
    processor = ProcessorDouble()
    _, processing = vlm.prepare_image_inputs(processor, source, "Describe", 2048)
    assert processing["resized_input_sizes"] == [{"width": 64, "height": 32}] * 2
    assert processing["input_frame_hashes"] == [frame.sha256_rgb for frame in source]
    assert processing["resized_input_frame_hashes"] != processing["input_frame_hashes"]
    assert all(image.width * image.height <= 2048 for image in processor.calls[0]["images"][0])


@pytest.mark.parametrize("corruption", ["hash", "order", "duplicate", "timestamp"])
def test_corrupted_or_nonchronological_source_never_reaches_processor(corruption):
    source = frames()
    if corruption == "hash":
        source[0] = replace(source[0], sha256_rgb="0" * 64)
    elif corruption == "order":
        source.reverse()
    elif corruption == "duplicate":
        source[1] = source[0]
    else:
        source[0] = replace(source[0], timestamp_s=float("nan"))
    processor = ProcessorDouble()
    with pytest.raises(ValueError):
        vlm.prepare_image_inputs(processor, source, "Describe", 4096)
    assert processor.calls == []


@pytest.mark.parametrize(
    "layout, side", [("batch", 128), ("crops", 64), ("opaque", 32), ("unmapped", 32)]
)
def test_processor_upscale_tiling_and_opaque_budget_fail_explicitly(layout, side):
    processor = ProcessorDouble(layout, side)
    with pytest.raises(ValueError, match="pixel|budget|layout"):
        vlm.prepare_image_inputs(processor, frames(), "Describe", 4096)


def test_smol_fixed_resolution_policy_disables_tiling_and_rejects_smaller_cap():
    processor = ProcessorDouble(side=512)
    _, processing = vlm.prepare_image_inputs(
        processor, frames(), "Describe", 262144, model_type="smolvlm"
    )
    assert processor.calls[0]["images_kwargs"] == {
        "do_image_splitting": False,
        "size": {"longest_edge": 512},
    }
    assert processing["image_tiling"]["processed_pixels_per_source_frame"] == 262144
    with pytest.raises(ValueError, match="at least 262144"):
        vlm.prepare_image_inputs(processor, frames(), "Describe", 65536, model_type="smolvlm")


def backend_double(monkeypatch, *, encoder_decoder=False, model_type="test_vlm"):
    calls = []

    class Config:
        _commit_hash = "pinned-commit"
        _attn_implementation = "eager"

    config = Config()
    config.model_type, config.is_encoder_decoder = model_type, encoder_decoder
    processor = ProcessorDouble()

    class AutoConfig:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append(("config", kwargs))
            return config

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append(("processor", kwargs))
            return processor

    class AutoModel:
        _model_mapping = {Config: object}

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append(("weights", kwargs))
            instance = cls()
            instance.config = config
            return instance

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            values = np.array([[7, 8, 9]] if encoder_decoder else [[1, 2, 3, 7, 8, 9]])
            return TensorDouble(values.shape, values=values)

    torch = SimpleNamespace(
        device=lambda name: SimpleNamespace(type="cpu", index=None),
        float32="float32",
        version=SimpleNamespace(cuda=None),
        inference_mode=nullcontext,
        OutOfMemoryError=type("OOM", (Exception,), {}),
    )
    monkeypatch.setattr(vlm, "_dependencies", lambda: (torch, AutoConfig, AutoProcessor, AutoModel))
    monkeypatch.setattr(vlm.importlib.metadata, "version", lambda name: "5.5.4")
    return config, processor, AutoModel, calls


@pytest.mark.parametrize("encoder_decoder", [False, True])
def test_backend_pinning_timing_dtype_output_and_unknown_visual_tokens(
    monkeypatch, encoder_decoder
):
    config, processor, _, calls = backend_double(monkeypatch, encoder_decoder=encoder_decoder)
    backend = vlm.TransformersVLMBackend("org/model", "cpu", "float32", 4096, 64, "branch")
    assert backend.info["input_representation"] == "timestamped_images"
    assert backend.info["thinking"]["effective"] is None
    for kind, kwargs in calls:
        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == ("branch" if kind == "config" else "pinned-commit")
    clock = iter([0, 2, 2, 3, 3, 7, 7, 8])
    monkeypatch.setattr(vlm, "perf_counter", lambda: next(clock))
    outcome = backend.run(frames(), {"accepted_answers": ["SECRET"]}, "Question only")
    metrics = outcome["metrics"]
    assert [
        metrics[k]
        for k in ("preprocessing_s", "device_transfer_s", "generation_s", "output_decode_s")
    ] == [2, 1, 4, 1]
    assert metrics["visual_tokens"] is None and metrics["generated_tokens"] == 3
    assert "visual_tokens" in outcome["unavailable_reasons"]
    assert processor.decoded.values.tolist() == [[7, 8, 9]]
    assert "generated reasoning" in outcome["answer"]
    assert "SECRET" not in str(processor.calls) and "SECRET" not in str(processor.messages)
    generated_inputs = calls[-1][1]
    assert generated_inputs["pixel_values"].moves[0]["dtype"] == "float32"
    assert "dtype" not in generated_inputs["input_ids"].moves[0]
    assert generated_inputs["do_sample"] is False and generated_inputs["num_beams"] == 1


def test_registry_config_is_reused_and_text_only_mapping_rejected_before_weights(monkeypatch):
    config, _, auto_model, calls = backend_double(monkeypatch)
    vlm.TransformersVLMBackend(
        "org/model",
        "cpu",
        "float32",
        4096,
        64,
        "branch",
        config=config,
        transformers_version="5.5.4",
    )
    assert not any(kind == "config" for kind, _ in calls)
    calls.clear()
    auto_model._model_mapping = {}
    with pytest.raises(ValueError, match="AutoModelForImageTextToText"):
        vlm.TransformersVLMBackend("org/model", "cpu", "float32", 4096, 64, "main")
    assert not any(kind == "weights" for kind, _ in calls)


def test_unsupported_thinking_and_smol_low_cap_fail_before_weights(monkeypatch):
    _, _, _, calls = backend_double(monkeypatch)
    with pytest.raises(ValueError, match="does not support enable_thinking"):
        vlm.TransformersVLMBackend(
            "org/model", "cpu", "float32", 4096, 64, "main", enable_thinking=True
        )
    assert not any(kind == "weights" for kind, _ in calls)
    _, _, _, calls = backend_double(monkeypatch, model_type="smolvlm")
    with pytest.raises(ValueError, match="at least 262144"):
        vlm.TransformersVLMBackend("org/model", "cpu", "float32", 4096, 64, "main")
    assert not any(kind == "weights" for kind, _ in calls)
