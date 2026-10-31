"""Trace bookkeeping uses CPU doubles; the native implementation check is explicit and opt-in."""

import base64
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from vidfit import SelectionSpec
from vidfit import native_trace as trace_module


@pytest.fixture
def native_double(monkeypatch):
    class Sampler:
        fps = 2

        def __init__(self):
            self.indices = [0, 1]
            self.calls = []

        def sample_frames(self, metadata, **arguments):
            self.calls.append((metadata, arguments))
            return self.indices

    sampler = Sampler()
    provenance = {
        "model": "test-double",
        "resolved_revision": "test-double",
        "class": "SamplerDouble",
        "settings": {
            "fps": 2,
            "min_frames": 4,
            "max_frames": 768,
            "temporal_patch_size": 2,
            "merge_size": 2,
        },
    }
    monkeypatch.setattr(
        trace_module,
        "_load_native_sampler",
        lambda model, revision: (sampler, provenance, SimpleNamespace),
    )
    return sampler


def test_actual_source_indices_and_midpoint_comparison(demo, native_double):
    native_double.indices = [0, 160, 319, 479]
    trace = trace_module.create_sampling_trace(demo[0], SelectionSpec.uniform(4))
    assert native_double.calls[0][1] == {"num_frames": 4}
    assert native_double.calls[0][0].total_num_frames == 480
    assert native_double.calls[0][0].fps == 8
    assert trace["vidfit"]["frame_indices"] == [60, 180, 300, 420]
    assert trace["native"]["frame_indices"] == [0, 160, 319, 479]
    assert [row["timestamp_s"] for row in trace["native"]["frames"]] == [0, 20, 39.875, 59.875]
    assert trace["vidfit"]["frames"][0]["target_timestamp_s"] == 7.5
    assert trace["comparison"]["shared_frame_indices"] == []
    assert trace["native"]["temporal_pairs"][0]["fps_timestamp_s"] == 10
    assert trace["native"]["selection"]["returned_count"] == 4
    assert trace["native"]["selection"]["requested_count"] == 4
    assert trace["native"]["selection"]["count_before_bounds"] is None
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize("mode,fps,arguments", [("default", None, {}), ("fps", 0.5, {"fps": 0.5})])
def test_native_default_and_fps_are_passed_without_count_override(
    short_video, native_double, mode, fps, arguments
):
    trace = trace_module.create_sampling_trace(
        short_video, SelectionSpec.uniform(2), native_mode=mode, native_fps=fps
    )
    assert native_double.calls[0][1] == arguments
    assert trace["native"]["selection"]["effective_sampling_fps"] == (fps or 2)
    assert trace["native"]["selection"]["count_before_bounds"] is None
    assert "not instrumented" in trace["native"]["selection"]["count_note"]
    assert trace["comparison"]["shared_frame_indices"] == [0, 1]


def test_manual_comparison_and_projected_padding(demo, native_double):
    native_double.indices = [0, 240, 479]
    trace = trace_module.create_sampling_trace(demo[0], SelectionSpec.manual([52.5, 24.375, 7.5]))
    assert trace["vidfit"]["frame_indices"] == [60, 195, 420]
    assert trace["native"]["selected_frames"] == 3
    last_pair = trace["native"]["temporal_pairs"][-1]
    assert last_pair["frame_indices"] == [479, 479]
    assert last_pair["padding_positions"] == [3]
    assert last_pair["prompt_timestamp_s"] == "59.9"
    assert "Derived" in trace["native"]["temporal_pairs_basis"]


@pytest.mark.parametrize(
    "indices,fps,expected", [([11, 16], 30, "0.4"), ([1420, 1580], 30000 / 1001, "50.0")]
)
def test_projected_prompt_rounding_matches_native_operation_order(indices, fps, expected):
    assert trace_module._temporal_pairs(indices, fps, 2)[0]["prompt_timestamp_s"] == expected


def test_gallery_limit_preserves_all_selected_indices_and_pts(demo, native_double):
    native_double.indices = list(range(100))
    trace = trace_module.create_sampling_trace(
        demo[0], SelectionSpec.uniform(4), native_mode="default"
    )
    native = trace["native"]
    assert native["selected_frames"] == len(native["frames"]) == 100
    assert native["frames"][-1]["timestamp_s"] == 99 / 8
    assert native["gallery"]["positions"][0] == 0
    assert native["gallery"]["positions"][-1] == 99
    images = [row for row in native["frames"] if "image_png_base64" in row]
    assert len(images) == native["gallery"]["shown"] == 32
    for row in images:
        with Image.open(io.BytesIO(base64.b64decode(row["image_png_base64"]))) as image:
            assert max(image.size) <= 384
        assert len(row["sha256_rgb"]) == 64


def test_duplicates_remain_visible_in_native_trace(short_video, native_double):
    native_double.indices = [0, 0, 1]
    trace = trace_module.create_sampling_trace(short_video, SelectionSpec.uniform(2))
    assert trace["native"]["frame_indices"] == [0, 0, 1]
    assert trace["native"]["duplicate_frame_indices"] == [0]
    assert trace["native"]["selected_frames"] == 3
    assert trace["native"]["distinct_frames"] == 2


def test_duplicate_native_indices_do_not_bypass_thumbnail_limit(short_video, native_double):
    native_double.indices = [0] * 40 + [1] * 40
    trace = trace_module.create_sampling_trace(short_video, SelectionSpec.uniform(2))
    native = trace["native"]
    assert len(native["frames"]) == 80
    assert native["duplicate_frame_indices"] == [0, 1]
    image_positions = [row["position"] for row in native["frames"] if "image_png_base64" in row]
    assert image_positions == native["gallery"]["positions"]
    assert len(image_positions) == 32


def test_large_vidfit_selection_never_encodes_full_source_pngs(demo, native_double, monkeypatch):
    from vidfit.core import video as video_module

    def full_capture_forbidden(*args):
        raise AssertionError("trace must not retain the ordinary full-size PNG evidence")

    encoded_sizes = []
    original_save = Image.Image.save

    def bounded_save(image, *args, **kwargs):
        encoded_sizes.append(image.size)
        assert max(image.size) <= 384
        return original_save(image, *args, **kwargs)

    monkeypatch.setattr(video_module, "_capture", full_capture_forbidden)
    monkeypatch.setattr(Image.Image, "save", bounded_save)
    native_double.indices = [0, 479]
    trace = trace_module.create_sampling_trace(
        demo[0], SelectionSpec.uniform(80), native_mode="default"
    )
    vidfit = trace["vidfit"]
    assert len(vidfit["frames"]) == vidfit["selected_frames"] == 80
    assert vidfit["frame_indices"] == list(range(3, 480, 6))
    assert len([row for row in vidfit["frames"] if "image_png_base64" in row]) == 32
    assert len(encoded_sizes) == 34  # 32 VidFit thumbnails, 2 native thumbnails


@pytest.mark.parametrize("indices", [[], [-1, 1], [0, 2], [1, 0], [0.0, 1.0]])
def test_invalid_native_indices_are_rejected(short_video, native_double, indices):
    native_double.indices = indices
    with pytest.raises(ValueError, match="native sampler returned"):
        trace_module.create_sampling_trace(short_video, SelectionSpec.uniform(2))


@pytest.mark.parametrize("native_fps", [0, -1, True, float("nan"), float("inf"), "2"])
def test_invalid_fps_fails_before_loading(short_video, native_double, native_fps):
    with pytest.raises(ValueError, match="finite positive"):
        trace_module.create_sampling_trace(
            short_video, SelectionSpec.uniform(2), native_mode="fps", native_fps=native_fps
        )
    assert not native_double.calls


def test_vfr_rejected_without_guessing_fps(short_video, native_double, monkeypatch):
    original = trace_module.probe_video
    monkeypatch.setattr(
        trace_module, "probe_video", lambda video: {**original(video), "constant_frame_rate": False}
    )
    with pytest.raises(ValueError, match="constant-frame-rate.*no fallback"):
        trace_module.create_sampling_trace(short_video, SelectionSpec.uniform(2))
    assert not native_double.calls


def test_video_change_is_detected(short_video, native_double, monkeypatch):
    monkeypatch.setattr(trace_module, "file_sha256", lambda path: "changed")
    with pytest.raises(ValueError, match="video changed"):
        trace_module.create_sampling_trace(short_video, SelectionSpec.uniform(2))


def test_missing_dependencies_are_actionable(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(trace_module.qwen_runtime.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match=r'install ".\[qwen\]" or ".\[qwen35\]"'):
        trace_module._load_native_sampler("unused", "main")


@pytest.mark.parametrize("version", ["0.0.0", "4.56.0", "6.0.0"])
def test_unsupported_transformers_version_rejected(monkeypatch, version):
    monkeypatch.setattr(
        trace_module.qwen_runtime.importlib.metadata, "version", lambda name: version
    )
    with pytest.raises(RuntimeError, match="requires Transformers >=4.57.6,<6"):
        trace_module._load_native_sampler("unused", "main")


@pytest.fixture
def sampler_runtime(monkeypatch):
    state = {"family": "qwen3_vl", "version": "5.5.4", "calls": []}

    class Config:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state["calls"].append(("config", model, kwargs))
            return SimpleNamespace(model_type=state["family"], _commit_hash="resolved-commit")

    class Processor:
        fps = 2
        min_frames = 4
        max_frames = 768
        temporal_patch_size = 2
        merge_size = 2
        do_sample_frames = True

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state["calls"].append(("processor", model, kwargs))
            return cls()

        def sample_frames(self, metadata, num_frames=None, fps=None):
            return [0, metadata.total_num_frames - 1]

    Processor.__module__ = "transformers.models.qwen3_vl.video_processing_qwen3_vl"
    Processor.sample_frames.__module__ = Processor.__module__

    def runtime_class(name, version):
        assert version == state["version"]
        # Loading a model class, even without constructing it, is not part of a trace.
        assert name in ("AutoConfig", "Qwen3VLVideoProcessor")
        return {"AutoConfig": Config, "Qwen3VLVideoProcessor": Processor}[name]

    monkeypatch.setattr(trace_module.qwen_runtime, "_runtime_class", runtime_class)
    monkeypatch.setattr(
        trace_module.qwen_runtime.importlib.metadata, "version", lambda name: state["version"]
    )
    monkeypatch.setitem(
        sys.modules, "transformers.video_utils", SimpleNamespace(VideoMetadata=SimpleNamespace)
    )
    state["processor_class"] = Processor
    return state


@pytest.mark.parametrize(
    "family,version", [("qwen3_vl", "4.57.6"), ("qwen3_vl", "5.5.4"), ("qwen3_5", "5.5.4")]
)
def test_native_sampler_resolves_family_and_actual_installed_source(
    sampler_runtime, family, version
):
    sampler_runtime.update(family=family, version=version)
    processor, provenance, metadata_type = trace_module._load_native_sampler(
        "example/model", "main"
    )
    assert sampler_runtime["calls"] == [
        ("config", "example/model", {"revision": "main", "trust_remote_code": False}),
        ("processor", "example/model", {"revision": "resolved-commit", "trust_remote_code": False}),
    ]
    assert provenance["model_type"] == family
    assert provenance["transformers_version"] == version
    assert provenance["class_module"] == processor.__module__
    assert provenance["callable_module"] == processor.sample_frames.__module__
    assert provenance["source"].startswith(
        f"https://github.com/huggingface/transformers/blob/v{version}/"
        "src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py#L"
    )
    assert (
        provenance["callable_source_sha256"]
        == hashlib.sha256(inspect.getsource(processor.sample_frames).encode("utf-8")).hexdigest()
    )
    assert "returned indices are authoritative" in provenance["algorithm"]
    assert "linspace" not in provenance["algorithm"]
    assert provenance["weights_loaded"] is False
    assert metadata_type is SimpleNamespace


def test_native_sampler_rejects_unsupported_family_before_processor_loading(sampler_runtime):
    sampler_runtime["family"] = "qwen3_5_moe"
    with pytest.raises(ValueError, match="Unsupported model_type"):
        trace_module._load_native_sampler("example/model", "main")
    assert len(sampler_runtime["calls"]) == 1


def test_native_sampler_requires_compatible_sampling_signature(sampler_runtime):
    sampler_runtime["processor_class"].sample_frames = lambda self, metadata: [0]
    with pytest.raises(RuntimeError, match="inspectable sample_frames"):
        trace_module._load_native_sampler("example/model", "main")


@pytest.mark.parametrize("setting", ["temporal_patch_size", "merge_size"])
def test_trace_rejects_unverified_temporal_groups(short_video, native_double, setting):
    _, provenance, _ = trace_module._load_native_sampler("unused", "main")
    provenance["settings"][setting] = 3
    with pytest.raises(ValueError, match="temporal_patch_size=2 and merge_size=2"):
        trace_module.create_sampling_trace(short_video, SelectionSpec.uniform(2))


def test_import_keeps_model_dependencies_optional():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vidfit.native_trace; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.native_trace
@pytest.mark.skipif(
    os.getenv("VIDFIT_RUN_NATIVE_TRACE_TEST") != "1",
    reason="opt-in installed native processor check",
)
def test_real_native_processor_without_weights(demo):
    trace = trace_module.create_sampling_trace(demo[0], SelectionSpec.uniform(4))
    assert trace["native"]["frame_indices"] == [0, 160, 319, 479]
    assert trace["vidfit"]["frame_indices"] == [60, 180, 300, 420]
    assert trace["processor"]["class"] == "Qwen3VLVideoProcessor"
    assert trace["processor"]["weights_loaded"] is False
    assert trace["processor"]["transformers_version"] == importlib.metadata.version("transformers")
    assert trace["processor"]["resolved_revision"]
    assert len(trace["processor"]["callable_source_sha256"]) == 64


@pytest.mark.native_trace
@pytest.mark.skipif(
    os.getenv("VIDFIT_RUN_NATIVE_TRACE_TEST") != "1",
    reason="opt-in installed native processor check",
)
def test_real_native_fps_defaults_limits_and_rounding(demo):
    processor, provenance, metadata_type = trace_module._load_native_sampler(
        "Qwen/Qwen3-VL-2B-Instruct", "main"
    )
    assert provenance["settings"]["fps"] == 2
    short = metadata_type(total_num_frames=2, fps=2)
    assert processor.sample_frames(short).tolist() == [0, 1]
    long = metadata_type(total_num_frames=30000, fps=30)
    assert len(processor.sample_frames(long)) == provenance["settings"]["max_frames"]
    assert processor.sample_frames(
        metadata_type(total_num_frames=4, fps=2), num_frames=3
    ).tolist() == [
        0,
        2,
        3,
    ]
    for mode, rate, expected_count in [("default", None, 120), ("fps", 0.5, 30)]:
        trace = trace_module.create_sampling_trace(
            demo[0], SelectionSpec.uniform(4), native_mode=mode, native_fps=rate
        )
        native = trace["native"]
        assert native["selected_frames"] == expected_count
        assert native["frame_indices"][0] == 0 and native["frame_indices"][-1] == 479
        assert native["gallery"]["shown"] == min(32, expected_count)
        assert len([r for r in native["frames"] if "image_png_base64" in r]) == min(
            32, expected_count
        )
