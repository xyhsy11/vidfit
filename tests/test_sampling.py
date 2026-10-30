import math

import numpy as np
import pytest

from vidfit.core.sampling import midpoint_targets
from vidfit.core.video import evidence_image, probe_video, sample_video
from vidfit.demo import generate_demo


def test_midpoints():
    assert midpoint_targets(60, 4) == [7.5, 22.5, 37.5, 52.5]
    assert 24.375 in midpoint_targets(60, 16)
    assert midpoint_targets(1, 1) == [0.5]


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_budgets(budget):
    with pytest.raises(ValueError, match="positive integer"):
        midpoint_targets(60, budget)


@pytest.mark.parametrize("duration", [0, -1, math.nan, math.inf])
def test_invalid_durations(duration):
    with pytest.raises(ValueError):
        midpoint_targets(duration, 4)


def test_real_demo_coverage_and_pixels(demo):
    video, _ = demo
    metadata = probe_video(video)
    assert metadata["frame_count"] == 480
    assert metadata["duration_s"] == 60
    for budget in (4, 8, 16):
        frames, metrics = sample_video(video, metadata, budget)
        event = [f for f in frames if 24 <= f.timestamp_s < 26]
        # Verify actual decoded pixels, not just the midpoint arithmetic or manifest.
        red = []
        for frame in frames:
            rgb = np.asarray(evidence_image(frame))
            region = rgb[148:285, 22:618].astype(int)
            red.append(
                np.mean(
                    (region[:, :, 0] > region[:, :, 1] * 2)
                    & (region[:, :, 0] > region[:, :, 2] * 1.5)
                )
                > 0.5
            )
        assert any(red) == bool(event) == (budget == 16)
        assert metrics["decoded_frames"] == 480
        assert len({f.frame_index for f in frames}) == budget
        assert all(abs(f.timestamp_s - f.target_timestamp_s) <= 1 / 16 + 1e-9 for f in frames)
        assert all(
            f.presentation_timestamp_s
            == f.pts * float(f.time_base.split("/")[0]) / float(f.time_base.split("/")[1])
            for f in frames
        )


def test_short_and_tie_chooses_earlier(short_video):
    metadata = probe_video(short_video)
    frames, _ = sample_video(short_video, metadata, 2)
    assert [f.frame_index for f in frames] == [0, 1]
    assert frames[0].target_timestamp_s == 0.25
    assert frames[0].timestamp_s == 0.0
    with pytest.raises(ValueError, match="distinct frames"):
        sample_video(short_video, metadata, 3)


def test_missing_unreadable_empty(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        probe_video(tmp_path / "missing.mp4")
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="cannot decode"):
        probe_video(path)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="cannot decode"):
        probe_video(path)


def test_demo_reproducible(demo, tmp_path):
    generated, _ = generate_demo(tmp_path / "another.mp4")
    assert generated.read_bytes() == demo[0].read_bytes()
    with pytest.raises(ValueError, match="exists"):
        generate_demo(generated)
