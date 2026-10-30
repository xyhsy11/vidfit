import os

import pytest

from vidfit import VideoAnalyzer


@pytest.mark.model
@pytest.mark.skipif(
    os.getenv("VIDFIT_RUN_MODEL_TEST") != "1",
    reason="set VIDFIT_RUN_MODEL_TEST=1 to download and run Qwen",
)
def test_real_qwen(demo):
    result = VideoAnalyzer(backend="qwen3vl", max_new_tokens=64).compare(
        demo[0],
        "What error message appeared?",
        [4, 8, 16],
        warmup=0,
        repeats=1,
    )
    assert result.complete, [c.error for c in result.configurations]
    assert result.model["resolved_revision"]
    for config in result.configurations:
        trial = config.trials[-1]
        assert trial.processing["validated"]
        assert trial.metrics["model_input_frames"] == config.requested_frames
        assert trial.metrics["visual_tokens"] > 0
        assert isinstance(trial.answer, str)


@pytest.mark.model
@pytest.mark.skipif(os.getenv("VIDFIT_RUN_MODEL_TEST") != "1", reason="opt-in real Qwen")
def test_real_manual_single_and_padding(demo):
    from vidfit import SelectionSpec

    analyzer = VideoAnalyzer(max_new_tokens=64)
    selections = [
        SelectionSpec.manual([7.5, 24.375, 37.5, 52.5]),
        SelectionSpec.manual([7.5, 24.375, 52.5]),
    ]
    result = analyzer.compare(
        demo[0], "What error message appeared?", selections=selections, execution_mode="single"
    )
    assert result.complete, [c.error for c in result.configurations]
    for spec, config in zip(selections, result.configurations, strict=True):
        assert len(config.trials) == 1
        trial = config.trials[0]
        assert trial.phase == "measured" and trial.processing["validated"]
        assert trial.metrics["model_input_frames"] == spec.count + spec.count % 2
        assert trial.evidence_hashes == [e.sha256_rgb for e in config.evidence]
        assert isinstance(trial.answer, str)
