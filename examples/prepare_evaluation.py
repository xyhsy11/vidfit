"""Build the five-clip starter collection. No weights, reports or media go into Git."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import av
from PIL import Image, ImageDraw

from vidfit import SelectionSpec
from vidfit.core.video import evidence_image, file_sha256, probe_video, sample_video
from vidfit.demo import generate_demo

SOURCES = {
    "cat": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/79/Cat_relaxing_on_a_sofa.webm",
        "sha256": "7e15ecdc1549f5ed3a5b957e2080c7f1cd32ee61c8d308632896826b6ae5b726",
    },
    "earth": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/94/DSCOVR_EPIC_Earth_Rotation.webm",
        "sha256": "28f28456eb4b9cb4428524e1371b24cef544a8be5b7e0f84bc72299af43b6692",
    },
}


def encode(path, images, width, height):
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "18", "preset": "medium", "threads": "1"}
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def synthetic(kind):
    for index in range(48):
        image = Image.new("RGB", (320, 180), "#142735" if kind == "motion" else "#168249")
        draw = ImageDraw.Draw(image)
        if kind == "motion":
            x = 12 + round(index * 240 / 47)
            draw.rectangle((x, 65, x + 40, 105), fill="#56a8ff")
        elif 16 <= index < 18:
            draw.rectangle((0, 0, 319, 179), fill="#f19b24")
        yield image


def prepare(output, cache):
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if not (output / "clip-01.mp4").exists():
        generate_demo(output / "clip-01.mp4")
    for number, kind in ((2, "motion"), (3, "flash")):
        path = output / f"clip-{number:02d}.mp4"
        if not path.exists():
            encode(path, synthetic(kind), 320, 180)
    for number, name in ((4, "cat"), (5, "earth")):
        source = SOURCES[name]
        original = cache / f"{name}.webm"
        if not original.exists():
            request = Request(
                source["url"],
                headers={
                    "User-Agent": (
                        "VidFit/0.2 (https://github.com/xyhsy11/vidfit; evaluation starter set)"
                    )
                },
            )
            with urlopen(request, timeout=60) as response:
                original.write_bytes(response.read())
        if file_sha256(original) != source["sha256"]:
            raise ValueError(f"source checksum mismatch: {name}; do not use this download")
        path = output / f"clip-{number:02d}.mp4"
        if not path.exists():
            metadata = probe_video(original)
            # Fixed 8 fps, selected from real source PTS; no audio. Limit cat to 8 seconds.
            count = int(min(metadata["duration_s"], 8) * 8)
            frames, _ = sample_video(
                original, metadata, SelectionSpec.manual([i / 8 for i in range(count)])
            )
            size = (640, 360) if name == "cat" else (480, 480)
            encode(
                path,
                (evidence_image(f).resize(size, Image.Resampling.LANCZOS) for f in frames),
                *size,
            )
    hashes = {p.name: file_sha256(p) for p in sorted(output.glob("clip-*.mp4"))}
    print(json.dumps(hashes, indent=2))
    contract = Path(__file__).parent / "evaluation" / "starter.json"
    if contract.exists():
        annotations = json.loads(contract.read_text(encoding="utf-8"))["annotations"]
        for item in annotations:
            if hashes[item["video"]["filename"]] != item["video"]["sha256"]:
                raise ValueError(
                    "generated checksum differs from reviewed collection; use the pinned "
                    "environment or visually review and version your own annotations"
                )
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/generated/evaluation"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/evaluation"))
    args = parser.parse_args()
    prepare(args.output_dir, args.cache_dir)
