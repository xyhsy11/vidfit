"""Compatible Hugging Face image-chat models, given chronological timestamped images.

This is not a native video path. Processor compatibility and pixel budgets are checked
explicitly; arbitrary VLM architectures and opaque image layouts are not assumed safe.
"""

import hashlib
import importlib.metadata
import math
from time import perf_counter

from PIL import Image

from vidfit.backends.qwen3vl import _format_chat_prompt
from vidfit.core.video import evidence_image

REPRESENTATION = "timestamped_images"
VISUAL_TOKEN_REASON = (
    "No verified mapping from this generic image processor's placeholders to model visual tokens; "
    "image views and tensor shapes are reported instead."
)


def _dependencies():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            'The Hugging Face VLM backend requires the project ".[vlm]" runtime profile. '
            f"Original import error: {exc}"
        ) from exc
    return torch, AutoConfig, AutoProcessor, AutoModelForImageTextToText


def _image_policy(processor, model_type, max_pixels):
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("This checkpoint has no supported image processor")
    if model_type == "smolvlm":
        # Smol's native image default upsizes and tiles. Its no-splitting path still
        # resizes to max_image_size, which must retain the checkpoint's token geometry.
        # transformers/v5.5.4/models/smolvlm/image_processing_smolvlm.py
        size = getattr(image_processor, "max_image_size", {})
        side = size.get("longest_edge") if isinstance(size, dict) else None
        if not isinstance(side, int) or isinstance(side, bool) or side <= 0:
            raise ValueError("SmolVLM processor does not expose its fixed image resolution")
        if side * side > max_pixels:
            raise ValueError(
                f"This SmolVLM checkpoint requires {side}x{side} pixels per image; "
                f"set --max-pixels to at least {side * side}. Input resizing cannot lower "
                "its fixed vision-encoder resolution."
            )
        return {"images_kwargs": {"do_image_splitting": False, "size": {"longest_edge": side}}}
    return {}


def _bound_image(image, max_pixels):
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
    if new_width * new_height > max_pixels:
        if new_width >= new_height:
            new_width = max(1, max_pixels // new_height)
        else:
            new_height = max(1, max_pixels // new_width)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _image_messages(timestamp_labels, question):
    content = [
        {
            "type": "text",
            "text": (
                "Selected video frames in chronological order; times are seconds from the start.\n"
            ),
        }
    ]
    for label in timestamp_labels:
        content.extend([{"type": "text", "text": label + "\n"}, {"type": "image"}])
    content.append({"type": "text", "text": "\n" + question})
    return [{"role": "user", "content": content}]


def _audit_image_pixels(inputs, count, max_pixels):
    """Conservatively count allocated crops/padding for recognized dense image layouts."""
    image_keys = [name for name in inputs if name.startswith("pixel_values")]
    if image_keys != ["pixel_values"]:
        raise ValueError(
            "Cannot verify this processor's image pixel budget: expected one dense pixel_values "
            "tensor. This model needs a dedicated adapter for its image representation."
        )
    shape = tuple(int(size) for size in inputs["pixel_values"].shape)
    if len(shape) < 4 or shape[-3] != 3 or min(shape) < 1:
        raise ValueError(f"Cannot verify image pixel budget for tensor layout {shape}")
    if len(shape) == 4 and shape[0] == count:
        views, layout = 1, "images_channels_height_width"
    elif len(shape) == 5 and shape[:2] == (1, count):
        views, layout = 1, "batch_images_channels_height_width"
    elif len(shape) == 5 and shape[0] == count:
        views, layout = shape[1], "images_crops_channels_height_width"
    elif len(shape) == 6 and shape[:2] == (1, count):
        views, layout = shape[2], "batch_images_crops_channels_height_width"
    else:
        raise ValueError(
            f"Cannot map image tensor layout {shape} to {count} selected frames. "
            "Per-frame crop budgets require a dedicated adapter for this processor."
        )
    height, width = shape[-2:]
    pixels_per_frame = views * height * width
    if pixels_per_frame > max_pixels:
        raise ValueError(
            f"Processor produced {views} view(s) of {width}x{height} per source frame "
            f"({pixels_per_frame} pixels), exceeding --max-pixels {max_pixels}. "
            "Its native resizing/tiling overrides the bounded source images; increase the "
            "explicit pixel budget or select a processor with compatible image controls."
        )
    return {
        "layout": layout,
        "views_per_source_frame": views,
        "processed_image_views": count * views,
        "view_height": height,
        "view_width": width,
        "processed_pixels_per_source_frame": pixels_per_frame,
        "budget_verified": True,
        "accounting": "Allocated dense image views, including padding, counted conservatively",
    }


def prepare_image_inputs(
    processor, frames, question, max_pixels, *, model_type=None, enable_thinking=False
):
    """Validate source identity, bound image inputs, then audit actual processor tensors."""
    if not isinstance(enable_thinking, bool):
        raise ValueError("enable_thinking must be a boolean")
    if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    if not frames:
        raise ValueError("At least one selected source frame is required")
    indices = [frame.frame_index for frame in frames]
    timestamps = [frame.timestamp_s for frame in frames]
    if any(not math.isfinite(ts) or ts < 0 for ts in timestamps):
        raise ValueError("Source timestamps must be finite and nonnegative")
    if any(a >= b for a, b in zip(indices, indices[1:], strict=False)) or any(
        a >= b for a, b in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("Selected source frames must be distinct and chronological")
    images, source_hashes, resized_hashes, resized_sizes = [], [], [], []
    for frame in frames:
        image = evidence_image(frame)
        source_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if image.mode != "RGB" or image.size != (frame.width, frame.height):
            raise ValueError("Selected preview has unexpected RGB mode or dimensions")
        if source_hash != frame.sha256_rgb:
            raise ValueError("Source previews and image-processor input pixels differ")
        image = _bound_image(image, max_pixels)
        images.append(image)
        source_hashes.append(source_hash)
        resized_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        resized_sizes.append({"width": image.width, "height": image.height})
    labels = [f"Frame {i + 1} at {timestamp:.6f} seconds" for i, timestamp in enumerate(timestamps)]
    messages = _image_messages(labels, question)
    prompt, thinking = _format_chat_prompt(processor, messages, enable_thinking)
    policy = _image_policy(processor, model_type, max_pixels)
    # Nested images mean one multi-image sample. Flat lists can trigger a different
    # single-image tiling path in processors such as LLaVA-OneVision.
    inputs = processor(text=[prompt], images=[images], return_tensors="pt", **policy)
    if "input_ids" not in inputs or len(inputs["input_ids"].shape) != 2:
        raise ValueError("Image chat processor must return a batched input_ids tensor")
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Expected one image-chat sample")
    tiling = _audit_image_pixels(inputs, len(frames), max_pixels)
    processing = {
        "input_representation": REPRESENTATION,
        "native_video_input": False,
        "selected_frames": len(frames),
        "model_input_frames": len(frames),
        "source_frame_indices": indices,
        "actual_selected_timestamps_s": timestamps,
        "prompt_timestamp_labels": labels,
        "timestamp_format": "Text labels rounded to six decimal places; actual source PTS retained",
        "input_frame_hashes": source_hashes,
        "resized_input_frame_hashes": resized_hashes,
        "input_hash_definition": "Original RGB evidence verified before explicit source resizing",
        "resized_input_sizes": resized_sizes,
        "source_resize_method": "PIL LANCZOS, preserve aspect ratio, downsize only",
        "requested_max_pixels_per_frame": max_pixels,
        "processor_overrides": policy,
        "image_tiling": tiling,
        "tensor_shapes": {
            name: list(value.shape) for name, value in inputs.items() if hasattr(value, "shape")
        },
        "enable_thinking": enable_thinking,
        "thinking": thinking,
        "visual_token_method": None,
        "validated": True,
        "validation_scope": (
            "Source RGB identity, chronological image submission and dense tensor pixel budget. "
            "This generic path does not claim native video sampling or temporal-token equivalence."
        ),
    }
    return inputs, processing


class TransformersVLMBackend:
    def __init__(
        self,
        model,
        device,
        dtype,
        max_pixels,
        max_new_tokens,
        revision,
        enable_thinking=False,
        *,
        config=None,
        transformers_version=None,
    ):
        if not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean")
        if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels < 1:
            raise ValueError("max_pixels must be a positive integer")
        if dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        start = perf_counter()
        torch, auto_config, auto_processor, auto_model = _dependencies()
        version = transformers_version or importlib.metadata.version("transformers")
        if config is None:
            config = auto_config.from_pretrained(model, revision=revision, trust_remote_code=False)
        resolved = getattr(config, "_commit_hash", None)
        if not resolved:
            raise ValueError("Model must resolve to a Hugging Face commit for reproducibility")
        if type(config) not in auto_model._model_mapping:
            raise ValueError(
                f"Model type {getattr(config, 'model_type', None)!r} is not supported by installed "
                "AutoModelForImageTextToText; choose a compatible multimodal chat checkpoint"
            )
        self.torch, self.device = torch, torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; install a CUDA PyTorch build or select CPU")
            if self.device.index is None or self.device.index >= torch.cuda.device_count():
                raise ValueError("Select an available explicit CUDA device, such as cuda:0")
            if dtype == "bfloat16":
                with torch.cuda.device(self.device):
                    if not torch.cuda.is_bf16_supported():
                        raise ValueError("Requested bfloat16 is not supported by this GPU")
        elif self.device.type != "cpu" or dtype != "float32":
            raise ValueError("CPU inference requires explicit --device cpu --dtype float32")
        self.dtype = getattr(torch, dtype)
        self.max_pixels, self.max_new_tokens = max_pixels, max_new_tokens
        self.enable_thinking, self.config = enable_thinking, config
        try:
            self.processor = auto_processor.from_pretrained(
                model, revision=resolved, trust_remote_code=False
            )
        except ImportError as exc:
            raise RuntimeError(
                f"The checkpoint's processor needs an optional dependency: {exc}. "
                'Install ".[vlm]" and the checkpoint\'s documented dependencies.'
            ) from exc
        policy = _image_policy(self.processor, config.model_type, max_pixels)
        _, thinking = _format_chat_prompt(
            self.processor, _image_messages(["Frame 1 at 0.000000 seconds"], ""), enable_thinking
        )
        self.model = (
            auto_model.from_pretrained(
                model, revision=resolved, config=config, dtype=self.dtype, trust_remote_code=False
            )
            .to(self.device)
            .eval()
        )
        self.synchronize()
        cuda = self.device.type == "cuda"
        self.info = {
            "identifier": model,
            "model_type": config.model_type,
            "model_class": type(self.model).__name__,
            "transformers_version": version,
            "input_representation": REPRESENTATION,
            "processor_identifier": model,
            "processor_class": type(self.processor).__name__,
            "image_processor_class": type(self.processor.image_processor).__name__,
            "processor_overrides": policy,
            "requested_revision": revision,
            "resolved_revision": resolved,
            "processor_resolved_revision": resolved,
            "loading_s": perf_counter() - start,
            "device": str(self.device),
            "dtype": dtype,
            "enable_thinking": enable_thinking,
            "thinking": thinking,
            "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device) if cuda else None,
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(self.device).total_memory if cuda else None
            ),
        }

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def run(self, frames, metadata, question):
        self.synchronize()
        start = perf_counter()
        inputs, processing = prepare_image_inputs(
            self.processor,
            frames,
            question,
            self.max_pixels,
            model_type=self.config.model_type,
            enable_thinking=self.enable_thinking,
        )
        self.synchronize()
        preprocessing_s = perf_counter() - start
        input_tokens = int(inputs["input_ids"].shape[1])
        start = perf_counter()
        # Cast only floating-point inputs: token IDs and image masks retain integer types.
        inputs = {
            name: value.to(
                device=self.device,
                **({"dtype": self.dtype} if value.is_floating_point() else {}),
            )
            if hasattr(value, "to") and hasattr(value, "is_floating_point")
            else value
            for name, value in inputs.items()
        }
        self.synchronize()
        transfer_s = perf_counter() - start
        start = perf_counter()
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=False,
                )
            self.synchronize()
        except self.torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory; select fewer frames, a smaller compatible pixel budget, "
                "or a smaller model. No automatic fallback was applied."
            ) from exc
        generation_s = perf_counter() - start
        start = perf_counter()
        output = (
            generated
            if getattr(self.config, "is_encoder_decoder", False)
            else generated[:, input_tokens:]
        )
        answer = self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        self.synchronize()
        output_decode_s = perf_counter() - start
        cuda = self.device.type == "cuda"
        metrics = {
            "preprocessing_s": preprocessing_s,
            "device_transfer_s": transfer_s,
            "generation_s": generation_s,
            "output_decode_s": output_decode_s,
            "input_tokens": input_tokens,
            "generated_tokens": int(output.shape[1]),
            "visual_tokens": None,
            "model_input_frames": len(frames),
            "peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device)
            if cuda
            else None,
            "peak_cuda_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device)
            if cuda
            else None,
        }
        unavailable = {"visual_tokens": VISUAL_TOKEN_REASON}
        if not cuda:
            unavailable.update(
                {
                    key: "CPU device selected; CUDA memory metrics unavailable"
                    for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")
                }
            )
        return {
            "answer": answer,
            "metrics": metrics,
            "processing": processing,
            "unavailable_reasons": unavailable,
        }
