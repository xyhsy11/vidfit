"""A fixed, question-independent visual-change baseline with bounded pixel memory."""

from time import perf_counter

import av
import numpy as np

from vidfit.core.sampling import VISUAL_CHANGE_SETTINGS, midpoint_targets


def _scan(path, metadata, budget):
    duration = metadata["duration_s"]
    midpoints = midpoint_targets(duration, budget)
    origin = metadata["first_presentation_timestamp_s"]
    source_count = metadata["frame_count"]
    # Distinct anchors guarantee a fill even when VFR gaps leave duration bins empty.
    anchor_indices = {((2 * i + 1) * source_count) // (2 * budget) for i in range(budget)}
    anchors = {}
    best, nearest = [None] * budget, [None] * budget
    previous, last_timestamp, count = None, None, 0
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            if frame.pts is None or frame.time_base is None:
                raise ValueError("visual-change selection requires presentation timestamps")
            timestamp = float(frame.pts * frame.time_base) - origin
            if (
                timestamp < 0
                or timestamp >= duration
                or (last_timestamp is not None and timestamp <= last_timestamp)
            ):
                raise ValueError("video timeline changed during visual-change scan")
            last_timestamp = timestamp
            pixels = (
                frame.reformat(
                    width=VISUAL_CHANGE_SETTINGS["thumbnail_width"],
                    height=VISUAL_CHANGE_SETTINGS["thumbnail_height"],
                    format=VISUAL_CHANGE_SETTINGS["thumbnail_format"],
                    interpolation=av.video.reformatter.Interpolation.BILINEAR,
                )
                .to_ndarray()
                .astype(np.int16)
            )
            score = float(np.abs(pixels - previous).mean() / 255) if previous is not None else 0.0
            previous = pixels
            bin_index = min(int(timestamp / duration * budget), budget - 1)
            row = {"frame_index": index, "timestamp_s": timestamp, "score": score}
            if index in anchor_indices:
                anchors[index] = row
            candidate = nearest[bin_index]
            distance = abs(timestamp - midpoints[bin_index])
            if (
                candidate is None
                or distance < abs(candidate["timestamp_s"] - midpoints[bin_index]) - 1e-12
            ):
                nearest[bin_index] = row
            candidate = best[bin_index]
            if score > 0 and (candidate is None or score > candidate["score"]):
                best[bin_index] = row
    if count != source_count or len(anchors) != budget:
        raise ValueError("video frame count changed during visual-change scan")
    selected = []
    missing_bins = []
    for bin_index in range(budget):
        candidate = best[bin_index] or nearest[bin_index]
        if candidate is None:
            missing_bins.append(bin_index)
            continue
        selected.append(
            {
                **candidate,
                "bin_index": bin_index,
                "bin_interval_s": [
                    bin_index * duration / budget,
                    (bin_index + 1) * duration / budget,
                ],
                "reason": "largest_visual_change"
                if best[bin_index] is not None
                else "static_bin_midpoint",
            }
        )
    used = {row["frame_index"] for row in selected}
    for bin_index in missing_bins:
        candidate = min(
            (row for index, row in anchors.items() if index not in used),
            key=lambda row: (abs(row["timestamp_s"] - midpoints[bin_index]), row["frame_index"]),
        )
        selected.append(
            {
                **candidate,
                "bin_index": bin_index,
                "bin_interval_s": [
                    bin_index * duration / budget,
                    (bin_index + 1) * duration / budget,
                ],
                "reason": "empty_bin_source_index_anchor",
            }
        )
        used.add(candidate["frame_index"])
    return sorted(selected, key=lambda row: row["frame_index"]), count


def sample_visual_change(path, metadata, budget, capture_frame):
    """Scan low-resolution pixels, then capture exact selected source frames.

    Timings include every scan/capture pass. ``selection_trace`` is provenance,
    not a numeric metric, and callers must separate it before metric aggregation.
    No question, reference answer or annotated interval is accepted by this API.
    """
    started = perf_counter()
    try:
        selected, scan_count = _scan(path, metadata, budget)
        scan_s = perf_counter() - started
        capture_started = perf_counter()
        wanted = {row["frame_index"]: row for row in selected}
        evidence, capture_count = [], 0
        origin = metadata["first_presentation_timestamp_s"]
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                capture_count += 1
                if index not in wanted:
                    continue
                row = wanted[index]
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("video timestamps changed during visual-change capture")
                if abs(float(frame.pts * frame.time_base) - origin - row["timestamp_s"]) > 1e-12:
                    raise ValueError("video timestamps changed during visual-change capture")
                evidence.append(capture_frame(frame, index, row["timestamp_s"], origin))
        if capture_count != metadata["frame_count"] or len(evidence) != budget:
            raise ValueError("video frame count changed during visual-change capture")
        if len({frame.frame_index for frame in evidence}) != budget:
            raise ValueError("visual-change selection did not produce distinct source frames")
        capture_s = perf_counter() - capture_started
    except (av.FFmpegError, OSError) as exc:
        raise ValueError(f"visual-change sampling failed: {type(exc).__name__}") from exc
    return evidence, {
        "decoded_frames": scan_count + capture_count,
        "decode_sampling_s": perf_counter() - started,
        "visual_change_scan_s": scan_s,
        "visual_change_capture_s": capture_s,
        "visual_change_scan_decoded_frames": scan_count,
        "visual_change_capture_decoded_frames": capture_count,
        "selection_trace": {
            **VISUAL_CHANGE_SETTINGS,
            "selected": selected,
            "scan_decoded_frames": scan_count,
            "capture_decoded_frames": capture_count,
            "score_first_source_frame": 0.0,
            "timing": (
                "decode_sampling_s includes full scan, selection and capture; sub-times overlap it"
            ),
        },
    }
