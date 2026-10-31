"""Explicit Qwen runtime capabilities; importing this module loads no model libraries."""

import importlib
import importlib.metadata
import re

_FAMILIES = {
    "qwen3_vl": ("Qwen3VLForConditionalGeneration", (4, 57, 6)),
    "qwen3_5": ("Qwen3_5ForConditionalGeneration", (5, 2, 0)),
}


def _release(version):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?([^+]*)", version)
    if match is None:
        raise RuntimeError(f"Cannot interpret installed Transformers version: {version}")
    release = tuple(int(part or 0) for part in match.groups()[:3])
    prerelease = bool(re.search(r"dev|a|b|rc", match.group(4)))
    return release, prerelease


def transformers_version():
    """Check the supported release range without importing Transformers or PyTorch."""
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            'Qwen dependencies missing; install ".[qwen]" or ".[qwen35]" in your environment'
        ) from exc
    release, prerelease = _release(version)
    if release < (4, 57, 6) or release >= (6, 0, 0) or (release == (4, 57, 6) and prerelease):
        raise RuntimeError(
            f"Qwen requires Transformers >=4.57.6,<6; found {version}. "
            'Install the project ".[qwen]" or ".[qwen35]" runtime profile.'
        )
    return version


def _validate_family(config, version):
    family = getattr(config, "model_type", None)
    if family not in _FAMILIES:
        raise ValueError(
            f"Unsupported model_type {family!r}; select a dense Qwen3-VL (qwen3_vl) "
            "or dense Qwen3.5 (qwen3_5) multimodal checkpoint"
        )
    _, minimum = _FAMILIES[family]
    release, prerelease = _release(version)
    if release < minimum or release >= (6, 0, 0) or (release == minimum and prerelease):
        raise RuntimeError(
            f"{family} requires Transformers >={'.'.join(map(str, minimum))},<6 "
            f"and its native model class; found {version}. "
            'Install the project ".[qwen35]" runtime profile for Qwen3.5.'
        )
    return family


def _runtime_class(name, version):
    try:
        return getattr(importlib.import_module("transformers"), name)
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            f"Transformers {version} cannot load {name}. "
            'Install a compatible complete runtime (".[qwen35]" for Qwen3.5), '
            f"including its PyTorch dependencies. Original error: {exc}"
        ) from exc


def load_config(model, revision):
    """Load and validate checkpoint configuration only; never load model weights."""
    version = transformers_version()
    auto_config = _runtime_class("AutoConfig", version)
    try:
        config = auto_config.from_pretrained(model, revision=revision, trust_remote_code=False)
    except ValueError as exc:
        # Old AutoConfig cannot construct this family, so its normal family check is unreachable.
        if "qwen3_5" in str(exc) and _release(version)[0] < (5, 2, 0):
            raise RuntimeError(
                f"This Qwen3.5 checkpoint is unsupported by Transformers {version}; "
                'install the project ".[qwen35]" runtime profile before loading weights.'
            ) from exc
        raise
    _validate_family(config, version)
    if not getattr(config, "_commit_hash", None):
        raise ValueError("model must resolve to a Hugging Face commit for reproducibility")
    return config, version


def load_model_class(config, version=None):
    """Resolve only the supported family's native class; do not instantiate it."""
    version = version or transformers_version()
    family = _validate_family(config, version)
    return _runtime_class(_FAMILIES[family][0], version)


def load_video_processor(config, model, revision):
    """Both supported families use the native Qwen3-VL video processor, without weights."""
    version = transformers_version()
    _validate_family(config, version)
    processor_class = _runtime_class("Qwen3VLVideoProcessor", version)
    processor = processor_class.from_pretrained(model, revision=revision, trust_remote_code=False)
    if processor.temporal_patch_size != 2 or processor.merge_size != 2:
        raise ValueError("unsupported processor temporal patch or spatial merge configuration")
    if not callable(getattr(processor, "sample_frames", None)):
        raise ValueError("native video processor does not expose frame sampling")
    return processor
