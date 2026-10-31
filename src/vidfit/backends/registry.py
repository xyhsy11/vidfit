"""Lazy model routing shared by the UI, CLI, API and experiment runner.

The catalog is advisory: any native Transformers image-text checkpoint can be
requested by ID. Its processor still has to satisfy the adapter's input checks.
"""

import importlib
from time import perf_counter

BACKEND_NAMES = ("inspect", "auto", "qwen", "qwen3vl", "transformers")

MODEL_PRESETS = [
    dict(
        name="Qwen3-VL · 2B",
        model="Qwen/Qwen3-VL-2B-Instruct",
        backend="auto",
        dtype="float16",
        max_pixels=262144,
        supports_native_trace=True,
    ),
    dict(
        name="Qwen3.5 · 2B",
        model="Qwen/Qwen3.5-2B",
        backend="auto",
        dtype="bfloat16",
        max_pixels=262144,
        supports_native_trace=True,
    ),
    dict(
        name="SmolVLM2 · 500M",
        model="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        backend="auto",
        dtype="bfloat16",
        max_pixels=262144,
        supports_native_trace=False,
    ),
    dict(
        name="SmolVLM2 · 2.2B",
        model="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        backend="auto",
        dtype="bfloat16",
        max_pixels=262144,
        supports_native_trace=False,
    ),
    dict(
        name="LLaVA-OneVision · 0.5B",
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        backend="auto",
        dtype="float16",
        max_pixels=1048576,
        supports_native_trace=False,
    ),
]


def backend_names():
    return BACKEND_NAMES


def create_backend(name, **options):
    """Construct one adapter on demand; configuration determines automatic routing."""
    if name not in BACKEND_NAMES:
        raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(BACKEND_NAMES)}")
    if name == "inspect":
        from vidfit.backends.inspection import InspectionBackend

        return InspectionBackend()
    if name in ("qwen", "qwen3vl"):
        from vidfit.backends.qwen3vl import QwenBackend

        return QwenBackend(**options)
    if name == "transformers":
        from vidfit.backends.transformers_vlm import TransformersVLMBackend

        return TransformersVLMBackend(**options)

    # Do not identify architecture by a repository name: custom IDs and forks work.
    from vidfit.backends.qwen_runtime import transformers_version

    started = perf_counter()
    version = transformers_version()
    runtime = importlib.import_module("transformers")
    try:
        config = runtime.AutoConfig.from_pretrained(
            options["model"], revision=options.get("revision", "main"), trust_remote_code=False
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot load the checkpoint's native Transformers configuration: {exc}. "
            'Install the compatible runtime with pip install -e ".[ui,vlm]". '
            "Checkpoints requiring remote Python code are not supported."
        ) from exc
    if not getattr(config, "_commit_hash", None):
        raise ValueError("model must resolve to a Hugging Face commit for reproducibility")
    if config.model_type in ("qwen3_vl", "qwen3_5"):
        from vidfit.backends.qwen3vl import QwenBackend

        # Pin the resolved commit across the dispatch and adapter initialization.
        backend = QwenBackend(**{**options, "revision": config._commit_hash})
        backend.info["requested_revision"] = options.get("revision", "main")
    else:
        from vidfit.backends.transformers_vlm import TransformersVLMBackend

        backend = TransformersVLMBackend(**options, config=config, transformers_version=version)
    backend.info["loading_s"] = perf_counter() - started
    return backend
