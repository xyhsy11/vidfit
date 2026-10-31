"""Inspect the installed Qwen sampler without loading model weights or running inference."""

import base64
import hashlib
import inspect
import io
import math
import operator
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import av
from PIL import Image

from vidfit.backends import qwen_runtime
from vidfit.core.sampling import SelectionSpec
from vidfit.core.video import file_sha256, probe_video, sample_video

GALLERY_LIMIT = 32


def _load_native_sampler(model, revision):
    """Return the real video processor, provenance, and native metadata type; never a model."""
    config, version = qwen_runtime.load_config(model, revision)
    resolved = config._commit_hash
    processor = qwen_runtime.load_video_processor(config, model, resolved)
    try:
        from transformers.video_utils import VideoMetadata
    except ImportError as exc:
        raise RuntimeError(
            'Native trace requires a compatible ".[qwen]" or ".[qwen35]" runtime profile'
        ) from exc
    try:
        settings = {
            name: getattr(processor, name)
            for name in (
                "fps",
                "min_frames",
                "max_frames",
                "temporal_patch_size",
                "merge_size",
                "do_sample_frames",
            )
        }
        signature = inspect.signature(processor.sample_frames)
        signature.bind(SimpleNamespace(), num_frames=2)
        signature.bind(SimpleNamespace(), fps=2.0)
        source_lines, first_line = inspect.getsourcelines(processor.sample_frames)
        source_text = "".join(source_lines)
        module = processor.sample_frames.__module__
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError(
            "native trace requires an inspectable sample_frames(metadata, num_frames=..., fps=...) "
            "method and video sampling settings"
        ) from exc
    source_url = (
        f"https://github.com/huggingface/transformers/blob/v{version}/"
        f"src/{module.replace('.', '/')}.py#L{first_line}-L{first_line + len(source_lines) - 1}"
    )
    return (
        processor,
        {
            "model": model,
            "requested_revision": revision,
            "resolved_revision": resolved,
            "model_type": config.model_type,
            "transformers_version": version,
            "class": type(processor).__name__,
            "class_module": type(processor).__module__,
            "method": "sample_frames",
            "callable_module": module,
            "settings": settings,
            "source": source_url,
            "source_note": (
                "Upstream release-tag link; the callable hash identifies the actual installed "
                "implementation, including local changes."
            ),
            "callable_source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "algorithm": (
                "Executed the installed sample_frames implementation; returned indices are "
                "authoritative. Internal count derivation and limits are not instrumented."
            ),
            "selection_inputs": (
                "video metadata and sampling settings supplied; no pixels or question"
            ),
            "weights_loaded": False,
        },
        VideoMetadata,
    )


def _gallery_positions(count):
    shown = min(count, GALLERY_LIMIT)
    if shown == 1:
        return [0]
    return [round(i * (count - 1) / (shown - 1)) for i in range(shown)]


def _thumbnail(image):
    image = image.copy()
    image.thumbnail((384, 384), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _source_row(frame, index, origin, fps, include_image):
    if frame.pts is None or frame.time_base is None:
        raise ValueError("video timestamps changed during native sampling trace")
    pts_s = float(frame.pts * frame.time_base)
    row = {
        "frame_index": index,
        "timestamp_s": pts_s - origin,
        "presentation_timestamp_s": pts_s,
        "pts": frame.pts,
        "time_base": str(frame.time_base),
        "fps_timestamp_s": index / fps,
    }
    if abs(row["timestamp_s"] - row["fps_timestamp_s"]) > 1e-4:
        raise ValueError("source indices/fps do not represent actual PTS; cannot trace Qwen timing")
    if include_image:
        image = frame.to_image().convert("RGB")
        row["sha256_rgb"] = hashlib.sha256(image.tobytes()).hexdigest()
        row["image_png_base64"] = _thumbnail(image)
    return row


def _capture_vidfit(video, metadata, selection):
    """Use the ordinary mapper while retaining only bounded thumbnails, never source PNGs."""
    gallery_positions = set(_gallery_positions(selection.count))
    position = 0

    def capture(frame, index, target, origin):
        nonlocal position
        row = _source_row(
            frame, index, origin, metadata["nominal_fps"], position in gallery_positions
        )
        row.update(position=position, target_timestamp_s=target)
        position += 1
        return SimpleNamespace(frame_index=index, row=row)

    frames, _ = sample_video(video, metadata, selection, capture_frame=capture)
    return [frame.row for frame in frames]


def _capture_indices(video, metadata, indices, gallery_positions):
    """Map selected original indices to actual PTS; decode only displayed frames to RGB."""
    wanted = set(indices)
    gallery_indices = {indices[position] for position in gallery_positions}
    captured = {}
    origin = metadata["first_presentation_timestamp_s"]
    fps = metadata["nominal_fps"]
    try:
        with av.open(str(video)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index not in wanted:
                    continue
                row = _source_row(frame, index, origin, fps, index in gallery_indices)
                captured[index] = row
                if len(captured) == len(wanted):
                    break
    except (av.FFmpegError, OSError) as exc:
        raise ValueError(f"native frame capture failed: {type(exc).__name__}") from exc
    if set(captured) != wanted:
        raise ValueError(
            "native sampler selected unavailable source frames; video may have changed"
        )
    rows = []
    gallery_positions = set(gallery_positions)
    for position, index in enumerate(indices):
        row = dict(captured[index], position=position)
        if position not in gallery_positions:
            row.pop("image_png_base64", None)
            row.pop("sha256_rgb", None)
        rows.append(row)
    return rows


def _temporal_pairs(indices, fps, temporal_patch_size):
    if temporal_patch_size != 2:
        raise ValueError("native trace requires the verified Qwen temporal patch size of 2")
    padded = indices + ([indices[-1]] if len(indices) % 2 else [])
    pairs = []
    for start in range(0, len(padded), 2):
        # Preserve upstream operation order: it affects rounding at decimal ties.
        timestamp = (padded[start] / fps + padded[start + 1] / fps) / 2
        pairs.append(
            {
                "positions": [start, start + 1],
                "frame_indices": padded[start : start + 2],
                "fps_timestamp_s": timestamp,
                "prompt_timestamp_s": f"{timestamp:.1f}",
                "padding_positions": [i for i in (start, start + 1) if i >= len(indices)],
            }
        )
    return pairs


def _selection_record(selection, rows, fps, temporal_patch_size):
    indices = [row["frame_index"] for row in rows]
    positions = _gallery_positions(len(rows))
    return {
        "selection": selection,
        "frame_indices": indices,
        "frames": rows,
        "selected_frames": len(rows),
        "distinct_frames": len(set(indices)),
        "duplicate_frame_indices": [
            index for index, count in Counter(indices).items() if count > 1
        ],
        "temporal_pairs": _temporal_pairs(indices, fps, temporal_patch_size),
        "temporal_pairs_basis": (
            "Derived from selected indices and source fps using Qwen's temporal-pair rule. "
            "Not an observed processor tensor or model attention trace."
        ),
        "gallery": {
            "limit": GALLERY_LIMIT,
            "positions": positions,
            "shown": len(positions),
            "total": len(rows),
            "description": "Resized source thumbnails, evenly spread through selected positions",
        },
    }


def create_sampling_trace(
    video,
    selection: SelectionSpec,
    *,
    model="Qwen/Qwen3-VL-2B-Instruct",
    revision="main",
    native_mode="matched",
    native_fps=None,
):
    """Compare real native sampling to VidFit's selection; return a JSON-ready trace.

    ``matched`` passes VidFit's selected count as ``num_frames``; ``default`` calls
    the native sampler without overrides; ``fps`` supplies an explicit sampling fps.
    No question, annotations, model weights, or inference participate in this trace.
    """
    if not isinstance(selection, SelectionSpec):
        raise ValueError("selection must be a SelectionSpec")
    if native_mode not in ("matched", "default", "fps"):
        raise ValueError("native mode must be matched, default, or fps")
    if native_mode == "fps":
        if (
            isinstance(native_fps, bool)
            or not isinstance(native_fps, (int, float))
            or not math.isfinite(native_fps)
            or native_fps <= 0
        ):
            raise ValueError("native fps must be a finite positive number")
    elif native_fps is not None:
        raise ValueError("native fps is only used with native mode fps")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must not be empty")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must not be empty")
    started = perf_counter()
    metadata = probe_video(video)
    fps = metadata["nominal_fps"]
    if not metadata["constant_frame_rate"] or fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            "native sampling trace requires constant-frame-rate video with valid source fps; "
            "no fallback fps was applied"
        )
    vidfit_rows = _capture_vidfit(video, metadata, selection)
    processor, provenance, metadata_type = _load_native_sampler(model, revision)
    if (
        provenance["settings"]["temporal_patch_size"] != 2
        or provenance["settings"]["merge_size"] != 2
    ):
        raise ValueError("native trace requires verified temporal_patch_size=2 and merge_size=2")
    native_metadata = metadata_type(
        total_num_frames=metadata["frame_count"],
        fps=fps,
        duration=metadata["duration_s"],
        width=metadata["width"],
        height=metadata["height"],
        video_backend="vidfit.pyav",
    )
    arguments = (
        {"num_frames": len(vidfit_rows)}
        if native_mode == "matched"
        else ({"fps": float(native_fps)} if native_mode == "fps" else {})
    )
    # Selection is delegated to the installed implementation, never reproduced here.
    raw_indices = processor.sample_frames(native_metadata, **arguments)
    try:
        indices = [operator.index(index) for index in raw_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError("native sampler returned invalid frame indices") from exc
    if (
        not indices
        or any(i < 0 or i >= metadata["frame_count"] for i in indices)
        or indices != sorted(indices)
    ):
        raise ValueError("native sampler returned empty, out-of-range, or unordered frame indices")
    if native_metadata.fps != fps:
        raise ValueError("native sampler changed source fps unexpectedly")
    native_rows = _capture_indices(video, metadata, indices, _gallery_positions(len(indices)))
    if file_sha256(Path(video)) != metadata["sha256"]:
        raise ValueError("video changed during sampling trace; run the trace again")
    effective_fps = (
        None
        if native_mode == "matched"
        else (float(native_fps) if native_mode == "fps" else processor.fps)
    )
    native_selection = {
        "method": "qwen_native",
        "mode": native_mode,
        "arguments": arguments,
        "effective_sampling_fps": effective_fps,
        "requested_count": arguments.get("num_frames"),
        "count_before_bounds": None,
        "returned_count": len(indices),
        "count_note": (
            "Only request arguments and returned indices are observed. Internal count "
            "derivation and bounds are not instrumented."
        ),
    }
    temporal_patch_size = provenance["settings"]["temporal_patch_size"]
    vidfit = _selection_record(selection.to_dict(), vidfit_rows, fps, temporal_patch_size)
    native = _selection_record(native_selection, native_rows, fps, temporal_patch_size)
    native_set, vidfit_set = set(indices), set(vidfit["frame_indices"])
    notes = [
        "This trace calls the native sample_frames method with metadata and settings only. "
        "The question and visual content do not affect that method's frame choices.",
        "VidFit inference uses its own selected frames with do_sample_frames=False. "
        "Tracing does not change these inputs.",
        "No model weights, inference, visual-token measurement, or attention analysis was run.",
        "Temporal pairs and rounded prompt times are derived bookkeeping, not measured model use.",
    ]
    if min(len(vidfit_rows), len(native_rows)) < 2:
        notes.append(
            "A one-frame sampling trace is possible, but VidFit Qwen inference needs two frames."
        )
    return {
        "schema_version": "vidfit.native_sampling_trace.v1",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": metadata,
        "processor": provenance,
        "request": {
            "native_mode": native_mode,
            "native_fps": native_fps,
            "vidfit_selection": selection.to_dict(),
        },
        "vidfit": vidfit,
        "native": native,
        "comparison": {
            "shared_frame_indices": sorted(native_set & vidfit_set),
            "only_native_frame_indices": sorted(native_set - vidfit_set),
            "only_vidfit_frame_indices": sorted(vidfit_set - native_set),
        },
        "notes": notes,
        "setup_time_s": perf_counter() - started,
    }
