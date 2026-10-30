"""Deterministic silent synthetic recording and a separate ground-truth manifest."""

import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from vidfit.core.video import file_sha256

EVENT = (24.0, 26.0)
ERROR_TEXT = "Error E104: Connection timed out."


def draw_frame(timestamp: float) -> Image.Image:
    image = Image.new("RGB", (640, 360), "#101827")
    draw = ImageDraw.Draw(image)
    small = ImageFont.load_default(size=16)
    normal = ImageFont.load_default(size=22)
    large = ImageFont.load_default(size=30)
    draw.rectangle((0, 0, 640, 52), fill="#1d2b40")
    draw.text((22, 16), "SYNTHETIC  /  Application monitor", font=normal, fill="#e6edf8")
    draw.text((24, 77), "Data synchronization", font=large, fill="white")
    draw.text((24, 124), "Workspace: demo   |   Local test session", font=small, fill="#adbed6")
    draw.rounded_rectangle((24, 167, 616, 255), radius=10, fill="#1d2b40")
    draw.text((42, 185), "Processing records...", font=normal, fill="#bdcce1")
    draw.rectangle((42, 228, 596, 236), fill="#34445e")
    draw.rectangle((42, 228, 42 + int(554 * timestamp / 60), 236), fill="#46d9b0")
    if EVENT[0] <= timestamp < EVENT[1]:
        draw.rounded_rectangle(
            (18, 144, 622, 288), radius=12, fill="#7f1827", outline="#ff7584", width=3
        )
        draw.text((40, 163), "Error E104:", font=large, fill="white")
        draw.text((40, 210), "Connection timed out.", font=large, fill="white")
        draw.text((40, 259), "Retry the operation.", font=small, fill="#ffced3")
    draw.text(
        (24, 318),
        f"Synthetic fixture   |   Elapsed {timestamp:06.3f} s",
        font=small,
        fill="#adbed6",
    )
    return image


def generate_demo(output: str | Path) -> tuple[Path, Path]:
    output = Path(output)
    manifest = output.with_suffix(".truth.json")
    if output.exists() or manifest.exists():
        raise ValueError("demo output or truth manifest exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        stream.options = {"crf": "18", "preset": "medium", "threads": "1"}
        for index in range(480):
            for packet in stream.encode(av.VideoFrame.from_image(draw_frame(index / 8))):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    truth = {
        "fixture_schema": "vidfit.synthetic.v1",
        "synthetic": True,
        "video_sha256": file_sha256(output),
        "duration_s": 60.0,
        "fps": 8,
        "frames": 480,
        "event_interval_s": list(EVENT),
        "event_interval_semantics": "start inclusive, end exclusive",
        "error_text": ERROR_TEXT,
        "note": "Ground truth for report annotation only. Never passed to the model.",
    }
    manifest.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return output, manifest
