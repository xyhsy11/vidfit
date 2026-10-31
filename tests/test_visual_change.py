"""Exercise visual-change choices on decoded pixels, not reference annotations."""

from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image

from vidfit import SelectionSpec
from vidfit.core.video import evidence_image, probe_video, sample_video


def write_colors(path, colors, fps=4, pts=None):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 36, "bgr0"
        for index, color in enumerate(colors):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 36), color))
            if pts is not None:
                frame.time_base = Fraction(1, fps)
                frame.pts = pts[index]
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_visual_change_rejects_invalid_budgets(budget):
    with pytest.raises(ValueError, match="positive integer"):
        SelectionSpec.visual_change(budget)


def test_visual_change_spec_is_explicit_and_cannot_guess_targets():
    selection = SelectionSpec.visual_change(4)
    assert selection.method == "visual_change" and selection.count == 4
    assert selection.to_dict()["visual_change"]["question_or_annotation_used"] is False
    assert selection.to_dict()["visual_change"]["algorithm"] == "temporal-bin-rgb-difference-v1"
    with pytest.raises(ValueError, match="scanning"):
        selection.targets(60)
    with pytest.raises(ValueError, match="manual timestamps"):
        SelectionSpec("visual_change", 4, (1,))


def test_visual_change_picks_largest_change_per_time_bin(tmp_path):
    colors = [
        "black",
        "black",
        "white",
        "white",
        "white",
        "black",
        "black",
        "black",
        "black",
        "black",
        "white",
        "white",
        "white",
        "white",
        "white",
        "black",
    ]
    path = write_colors(tmp_path / "changes.mkv", colors)
    metadata = probe_video(path)
    frames, metrics = sample_video(path, metadata, SelectionSpec.visual_change(4))
    assert [frame.frame_index for frame in frames] == [2, 5, 10, 15]
    assert [frame.timestamp_s for frame in frames] == [0.5, 1.25, 2.5, 3.75]
    assert all(frame.target_timestamp_s == frame.timestamp_s for frame in frames)
    assert [int(np.asarray(evidence_image(frame)).mean()) for frame in frames] == [255, 0, 255, 0]
    details = metrics["selection_trace"]
    assert [row["reason"] for row in details["selected"]] == ["largest_visual_change"] * 4
    assert [row["score"] for row in details["selected"]] == [1.0] * 4
    assert metrics["decoded_frames"] == 32
    assert metrics["visual_change_scan_decoded_frames"] == 16
    assert metrics["visual_change_capture_decoded_frames"] == 16
    assert (
        metrics["decode_sampling_s"]
        >= (metrics["visual_change_scan_s"] + metrics["visual_change_capture_s"])
        > 0
    )


def test_static_bins_choose_midpoints(tmp_path):
    path = write_colors(tmp_path / "static.mkv", ["black"] * 16)
    metadata = probe_video(path)
    frames, metrics = sample_video(path, metadata, SelectionSpec.visual_change(4))
    assert [frame.frame_index for frame in frames] == [2, 6, 10, 14]
    assert all(row["score"] == 0 for row in metrics["selection_trace"]["selected"])
    assert all(
        row["reason"] == "static_bin_midpoint" for row in metrics["selection_trace"]["selected"]
    )


def test_positive_score_and_static_midpoint_ties_choose_earlier(tmp_path):
    path = write_colors(tmp_path / "tie.mkv", ["black", "white", "black", "black"])
    frames, _ = sample_video(path, probe_video(path), SelectionSpec.visual_change(1))
    assert frames[0].frame_index == 1
    static_path = write_colors(tmp_path / "static-tie.mkv", ["black"] * 2, fps=2)
    frames, _ = sample_video(static_path, probe_video(static_path), SelectionSpec.visual_change(2))
    assert [frame.frame_index for frame in frames] == [0, 1]


def test_vfr_empty_bins_fill_distinct_source_indices(tmp_path):
    path = write_colors(
        tmp_path / "gaps.mkv", ["black", "white", "black", "white"], fps=10, pts=[0, 1, 2, 30]
    )
    metadata = probe_video(path)
    assert not metadata["constant_frame_rate"]
    frames, metrics = sample_video(path, metadata, SelectionSpec.visual_change(4))
    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert len({frame.frame_index for frame in frames}) == 4
    assert "empty_bin_source_index_anchor" in {
        row["reason"] for row in metrics["selection_trace"]["selected"]
    }


def test_capture_hook_preserved_and_trace_is_repeatable(short_video, monkeypatch):
    from vidfit.core import video as video_module

    metadata = probe_video(short_video)
    captured = []

    def capture(frame, index, target, origin):
        captured.append((index, target, float(frame.pts * frame.time_base) - origin))
        return SimpleNamespace(frame_index=index)

    def forbidden(*args, **kwargs):
        raise AssertionError("lightweight capture must not generate full source PNGs")

    monkeypatch.setattr(video_module, "_capture", forbidden)
    for _ in range(2):
        frames, _ = sample_video(
            short_video, metadata, SelectionSpec.visual_change(2), capture_frame=capture
        )
        assert [frame.frame_index for frame in frames] == [0, 1]
    assert captured[:2] == captured[2:]
    assert all(target == actual for _, target, actual in captured)


def test_excessive_budget_and_unrelated_metadata(short_video):
    metadata = probe_video(short_video)
    with pytest.raises(ValueError, match="distinct frames"):
        sample_video(short_video, metadata, SelectionSpec.visual_change(3))
    normal, _ = sample_video(short_video, metadata, SelectionSpec.visual_change(2))
    extra, _ = sample_video(
        short_video,
        {**metadata, "question": "SECRET", "annotation": {"intervals_s": [[0.1, 0.2]]}},
        SelectionSpec.visual_change(2),
    )
    assert [(f.frame_index, f.sha256_rgb) for f in normal] == [
        (f.frame_index, f.sha256_rgb) for f in extra
    ]
