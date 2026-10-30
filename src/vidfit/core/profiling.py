"""Measurement helpers; importing this module does not import torch."""

import importlib.metadata
import platform
import statistics
from pathlib import Path

import av

INFERENCE_METRICS = (
    "preprocessing_s",
    "device_transfer_s",
    "generation_s",
    "output_decode_s",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "visual_tokens",
    "input_tokens",
    "generated_tokens",
    "model_input_frames",
)


def environment() -> dict:
    packages = {}
    for name in ("vidfit", "av", "Pillow", "numpy", "torch", "torchvision", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "packages": packages,
        "ffmpeg_libraries": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
    }


def summarize(trials: list) -> dict:
    measured = [t for t in trials if t.phase == "measured" and t.status == "ok"]
    keys = sorted({k for t in trials for k in t.metrics})
    summary = {"measured_trials": len(measured), "metrics": {}}
    for key in keys:
        values = [t.metrics[key] for t in measured if t.metrics.get(key) is not None]
        summary["metrics"][key] = (
            {
                "n": len(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else None,
            }
            if values
            else None
        )
    return summary


def describe_error(exc: Exception) -> str:
    """Keep actionable errors while removing home/workspace paths from shareable reports."""
    message = str(exc)
    for path, label in ((Path.home(), "<home>"), (Path.cwd(), "<workspace>")):
        for variant in (str(path), path.as_posix()):
            message = message.replace(variant, label)
    return f"{type(exc).__name__}: {message}"
