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
