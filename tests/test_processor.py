"""Bookkeeping doubles exercise failure checks; they are never exported as model results."""

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from vidfit.backends.qwen3vl import prepare_video_inputs
from vidfit.core.video import probe_video, sample_video


@dataclass
class MetadataDouble:
    total_num_frames: int
    fps: float
    duration: float
    width: int
    height: int
    video_backend: str
    frames_indices: list


class BatchFeatureDouble(dict):
    def convert_to_tensors(self, tensor_type):
        assert "video_metadata" not in self
        assert tensor_type == "pt"
        return self


class ProcessorDouble:
    video_processor = SimpleNamespace(temporal_patch_size=2, patch_size=16, merge_size=2)
    video_token_id = 42

    def __init__(self, mode="ok"):
        self.mode = mode

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"][0] == {"type": "video"}
        return "<video>" + messages[0]["content"][1]["text"]

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        assert kwargs["do_sample_frames"] is False
        assert "error_text" not in kwargs["video_metadata"][0].__dict__
        m = kwargs["video_metadata"][0]
        n = len(kwargs["videos"][0])
        if n % 2:
            m.frames_indices.append(m.frames_indices[-1])
        t = (n + n % 2) // 2
        if self.mode == "resample":
            m.frames_indices[0] += 1
        times = [
            (m.frames_indices[i] + m.frames_indices[i + 1]) / (2 * m.fps)
            for i in range(0, len(m.frames_indices), 2)
        ]
        self.tokenizer = SimpleNamespace(
            decode=lambda *a, **k: "".join(f"<{ts:.1f} seconds>" for ts in times)
        )
        tokens = t * 4 - (1 if self.mode == "tokens" else 0)
        return BatchFeatureDouble(
            {
                "video_metadata": [m],
                "video_grid_thw": np.array([[t, 4, 4]]),
                "pixel_values_videos": np.zeros((t * 16, 1536)),
                "input_ids": np.array([[42] * tokens]),
            }
        )


@pytest.mark.parametrize("count", [3, 4])
def test_metadata_padding_and_native_path(monkeypatch, demo, count):
    monkeypatch.setitem(
        sys.modules, "transformers.video_utils", SimpleNamespace(VideoMetadata=MetadataDouble)
    )
    meta = probe_video(demo[0])
    frames, _ = sample_video(demo[0], meta, count)
    processor = ProcessorDouble()
    _, bookkeeping, tokens = prepare_video_inputs(processor, frames, meta, "What appeared?", 262144)
    assert bookkeeping["model_input_frames"] == count + count % 2
    assert bookkeeping["padding"]["duplicated_final_frames"] == count % 2
    assert bookkeeping["source_metadata"]["frames_indices"] == [f.frame_index for f in frames]
    assert tokens == ((count + count % 2) // 2) * 4
    assert processor.kwargs["size"]["longest_edge"] == count * 262144


@pytest.mark.parametrize("mode", ["resample", "tokens"])
def test_incompatible_processor_rejected(monkeypatch, demo, mode):
    monkeypatch.setitem(
        sys.modules, "transformers.video_utils", SimpleNamespace(VideoMetadata=MetadataDouble)
    )
    meta = probe_video(demo[0])
    frames, _ = sample_video(demo[0], meta, 4)
    with pytest.raises(ValueError):
        prepare_video_inputs(ProcessorDouble(mode), frames, meta, "What appeared?", 262144)


def test_vfr_explicitly_unsupported(monkeypatch, demo):
    monkeypatch.setitem(
        sys.modules, "transformers.video_utils", SimpleNamespace(VideoMetadata=MetadataDouble)
    )
    meta = probe_video(demo[0])
    frames, _ = sample_video(demo[0], meta, 4)
    meta["constant_frame_rate"] = False
    with pytest.raises(ValueError, match="constant-frame-rate"):
        prepare_video_inputs(ProcessorDouble(), frames, meta, "Describe", 262144)
