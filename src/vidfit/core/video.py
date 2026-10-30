"""Streaming FFmpeg/PyAV decoding. No full-video pixel cache or random seek assumptions."""

import base64
import hashlib
import io
from pathlib import Path
from time import perf_counter
from typing import Any

import av
from PIL import Image

from vidfit.core.sampling import midpoint_targets
from vidfit.core.schemas import FrameEvidence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path.name}")
    start = perf_counter()
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("file contains no video stream")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            count, first, last, delta, frame_duration = 0, None, None, None, None
            cfr = True
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("video has missing presentation timestamps")
                ts = float(frame.pts * frame.time_base)
                if first is None:
                    first = ts
                if last is not None:
                    if ts <= last:
                        raise ValueError("video presentation timestamps must increase strictly")
                    gap = ts - last
                    if delta is not None and abs(gap - delta) > 1e-5:
                        cfr = False
                    delta = gap
                last = ts
                frame_duration = float(frame.duration * frame.time_base) if frame.duration else None
                count += 1
            if not count:
                raise ValueError("video contains no decodable frames")
            tail = frame_duration or delta or (1 / fps if fps else None)
            if tail is None or tail <= 0:
                raise ValueError("cannot determine duration of the last frame")
            duration = last - first + tail
            return {
                "filename": path.name,
                "sha256": file_sha256(path),
                "duration_s": duration,
                "first_presentation_timestamp_s": first,
                "frame_count": count,
                "nominal_fps": fps,
                "constant_frame_rate": cfr,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "codec": stream.codec_context.name,
                "probe_decoded_frames": count,
                "probe_time_s": perf_counter() - start,
                "duration_definition": "last PTS minus first PTS plus final frame duration",
            }
    except (av.FFmpegError, OSError) as exc:
        raise ValueError(f"cannot decode {path.name}: {type(exc).__name__}") from exc


def _capture(frame: av.VideoFrame, index: int, target: float, origin: float) -> FrameEvidence:
    image = frame.to_image().convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    ts = float(frame.pts * frame.time_base)
    return FrameEvidence(
        target,
        ts - origin,
        ts,
        frame.pts,
        str(frame.time_base),
        index,
        hashlib.sha256(image.tobytes()).hexdigest(),
        image.width,
        image.height,
        base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def evidence_image(frame: FrameEvidence) -> Image.Image:
    with Image.open(io.BytesIO(base64.b64decode(frame.image_png_base64))) as image:
        return image.convert("RGB")


def sample_video(path: str | Path, metadata: dict, budget: int) -> tuple[list[FrameEvidence], dict]:
    targets = midpoint_targets(metadata["duration_s"], budget)
    if budget > metadata["frame_count"]:
        raise ValueError(
            f"requested {budget} frames but video has {metadata['frame_count']} distinct frames"
        )
    start = perf_counter()
    evidence = []
    origin = metadata["first_presentation_timestamp_s"]
    previous, previous_index, count, target_index = None, 0, 0, 0
    try:
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                count += 1
                ts = float(frame.pts * frame.time_base) - origin
                while target_index < len(targets) and ts >= targets[target_index]:
                    target = targets[target_index]
                    chosen, chosen_index = frame, index
                    if previous is not None:
                        prev_ts = float(previous.pts * previous.time_base) - origin
                        if target - prev_ts <= ts - target + 1e-12:
                            chosen, chosen_index = previous, previous_index
                    evidence.append(_capture(chosen, chosen_index, target, origin))
                    target_index += 1
                previous, previous_index = frame, index
            while target_index < len(targets) and previous is not None:
                evidence.append(_capture(previous, previous_index, targets[target_index], origin))
                target_index += 1
    except (av.FFmpegError, OSError) as exc:
        raise ValueError(f"sampling failed: {type(exc).__name__}") from exc
    if len(evidence) != budget or len({e.frame_index for e in evidence}) != budget:
        raise ValueError("midpoint targets map to too few distinct frames; reduce the frame budget")
    return evidence, {"decoded_frames": count, "decode_sampling_s": perf_counter() - start}
