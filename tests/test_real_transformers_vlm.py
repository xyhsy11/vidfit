"""Opt-in trained non-Qwen image-chat regression; ordinary test runs load no weights."""

import os

import pytest

import vidfit.analyzer as analyzer_module
from vidfit import SelectionSpec, VideoAnalyzer


@pytest.mark.model
@pytest.mark.skipif(
    os.getenv("VIDFIT_RUN_VLM_TEST") != "1",
    reason="set VIDFIT_RUN_VLM_TEST=1 to download and run a real Hugging Face VLM on CUDA",
)
def test_real_timestamped_images_single_and_multiple_frames(short_video, monkeypatch):
    model = os.getenv("VIDFIT_TEST_VLM_MODEL", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    max_pixels = int(os.getenv("VIDFIT_TEST_VLM_MAX_PIXELS", "262144"))
    analyzer = VideoAnalyzer(
        backend="auto",
        model=model,
        device="cuda:0",
        dtype=os.getenv("VIDFIT_TEST_VLM_DTYPE", "bfloat16"),
        revision=os.getenv("VIDFIT_TEST_VLM_REVISION", "main"),
        max_pixels=max_pixels,
        max_new_tokens=32,
        enable_thinking=False,
    )
    create_backend = analyzer_module.create_backend
    loads = []

    def counted_backend(*args, **kwargs):
        backend = create_backend(*args, **kwargs)
        loads.append(backend)
        return backend

    monkeypatch.setattr(analyzer_module, "create_backend", counted_backend)
    selections = [SelectionSpec.uniform(1), SelectionSpec.uniform(2)]
    result = analyzer.compare(
        short_video,
        "Describe the colors shown in these selected frames.",
        selections=selections,
        execution_mode="single",
    )
    assert result.complete, [config.error for config in result.configurations]
    assert len(loads) == 1, "Both selections must reuse one loaded model"
    assert result.model["identifier"] == model
    assert result.model["resolved_revision"]
    assert result.model["resolved_revision"] == result.model["processor_resolved_revision"]
    assert result.model["input_representation"] == "timestamped_images"
    for selection, config in zip(selections, result.configurations, strict=True):
        assert config.status == "ok" and len(config.trials) == 1
        trial = config.trials[0]
        processing, metrics = trial.processing, trial.metrics
        assert trial.status == "ok" and trial.error is None and trial.phase == "measured"
        assert isinstance(trial.answer, str) and trial.answer.strip()
        assert len(config.evidence) == metrics["model_input_frames"] == selection.count
        hashes = [frame.sha256_rgb for frame in config.evidence]
        assert trial.evidence_hashes == hashes == processing["input_frame_hashes"]
        # The 64x64 fixture is below every allowed budget; processor resizing is separate.
        assert processing["resized_input_frame_hashes"] == hashes
        assert processing["source_frame_indices"] == [
            frame.frame_index for frame in config.evidence
        ]
        assert processing["actual_selected_timestamps_s"] == [
            frame.timestamp_s for frame in config.evidence
        ]
        assert processing["input_representation"] == "timestamped_images"
        assert processing["native_video_input"] is False and processing["validated"] is True
        budget = processing["image_tiling"]
        assert budget["budget_verified"] is True
        assert 0 < budget["processed_pixels_per_source_frame"] <= max_pixels
        assert budget["processed_image_views"] == selection.count * budget["views_per_source_frame"]
        assert metrics["visual_tokens"] is None
        assert trial.unavailable_reasons["visual_tokens"]
        assert metrics["input_tokens"] > 0 and metrics["generated_tokens"] > 0
        stages = (
            "decode_sampling_s",
            "preprocessing_s",
            "device_transfer_s",
            "generation_s",
            "output_decode_s",
        )
        assert all(metrics[stage] >= 0 for stage in stages)
        assert metrics["generation_s"] > 0
        assert sum(metrics[stage] for stage in stages) <= metrics["pipeline_s"] + 1e-6
