"""Native Qwen3-VL video path, verified against Transformers 4.57.6.

Model dependencies are imported only on explicit backend construction. The processor
receives only selected RGB evidence, source temporal metadata, and the question.
"""

import hashlib
import importlib.metadata
import re
from dataclasses import asdict
from time import perf_counter

import numpy as np

from vidfit.core.video import evidence_image


def prepare_video_inputs(processor, frames, metadata, question, max_pixels):
    from transformers.video_utils import VideoMetadata

    count = len(frames)
    if count < 2:
        raise ValueError(
            "Qwen3-VL requires at least 2 selected frames; use inspect for single frames"
        )
    fps = metadata["nominal_fps"]
    if not metadata["constant_frame_rate"] or not fps:
        raise ValueError(
            "this Qwen adapter requires constant-frame-rate video to preserve timestamps"
        )
    if any(abs(f.frame_index / fps - f.timestamp_s) > 1e-4 for f in frames):
        raise ValueError(
            "source frame indices/fps cannot represent actual PTS; refusing altered timestamps"
        )
    pixels = np.stack([np.asarray(evidence_image(f)) for f in frames])
    hashes = [hashlib.sha256(p.tobytes()).hexdigest() for p in pixels]
    if hashes != [f.sha256_rgb for f in frames]:
        raise ValueError("source previews and processor-input pixels differ")
    indices = [f.frame_index for f in frames]
    temporal = VideoMetadata(
        total_num_frames=metadata["frame_count"],
        fps=fps,
        duration=metadata["duration_s"],
        width=metadata["width"],
        height=metadata["height"],
        video_backend="vidfit.pyav",
        frames_indices=indices.copy(),
    )
    before = asdict(temporal)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Native processor's size uses TOTAL pixels across T, not pixels per frame.
    inputs = processor(
        text=[text],
        videos=[pixels],
        video_metadata=[temporal],
        do_sample_frames=False,
        return_metadata=True,
        return_tensors="pt",
        size={"shortest_edge": 4096 * count, "longest_edge": max_pixels * count},
    )
    returned = inputs.pop("video_metadata")
    if len(returned) != 1:
        raise ValueError("processor returned unexpected video metadata count")
    grid = inputs["video_grid_thw"].tolist()
    if len(grid) != 1:
        raise ValueError("expected one processed video")
    t, h, w = grid[0]
    vp = processor.video_processor
    temporal_patch, patch, merge = vp.temporal_patch_size, vp.patch_size, vp.merge_size
    if temporal_patch != 2 or merge != 2:
        raise ValueError("unsupported processor temporal patch or spatial merge configuration")
    padded_count = t * temporal_patch
    expected_count = count + count % 2
    expected_indices = indices + ([indices[-1]] if count % 2 else [])
    if padded_count != expected_count or list(returned[0].frames_indices) != expected_indices:
        raise ValueError("processor changed selected frame count or temporal indices unexpectedly")
    if returned[0].fps != fps or returned[0].total_num_frames != metadata["frame_count"]:
        raise ValueError("processor changed source temporal metadata")
    shape = list(inputs["pixel_values_videos"].shape)
    if shape != [t * h * w, 3 * temporal_patch * patch * patch]:
        raise ValueError("processed video tensor shape disagrees with grid metadata")
    visual_tokens = int((inputs["input_ids"] == processor.video_token_id).sum().item())
    grid_tokens = t * h * w // (merge * merge)
    if visual_tokens != grid_tokens:
        raise ValueError("native video placeholder count disagrees with processed grid")
    effective_height, effective_width = h * patch, w * patch
    if effective_height * effective_width > max_pixels:
        raise ValueError("processor exceeded requested per-frame pixel cap; increase max_pixels")
    prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    # Native Qwen timestamps are averages within each temporal pair, rounded to 0.1 s.
    expected_times = [
        (expected_indices[i] + expected_indices[i + 1]) / (2 * fps)
        for i in range(0, padded_count, 2)
    ]
    prompt_times = re.findall(r"<([0-9]+\.[0-9]+) seconds>", prompt)
    if prompt_times != [f"{ts:.1f}" for ts in expected_times]:
        raise ValueError("processor prompt timestamps disagree with selected temporal pairs")
    processing = {
        "native_video_input": True,
        "do_sample_frames": False,
        "requested_max_pixels_per_frame": max_pixels,
        "processor_total_pixel_limits": {"min": 4096 * count, "max": max_pixels * count},
        "selected_frames": count,
        "model_input_frames": padded_count,
        "padding": {
            "duplicated_final_frames": padded_count - count,
            "source_frame_index": indices[-1] if padded_count > count else None,
        },
        "source_metadata": before,
        "returned_metadata": asdict(returned[0]),
        "actual_selected_timestamps_s": [f.timestamp_s for f in frames],
        "temporal_pair_timestamps_s": expected_times,
        "prompt_timestamps_s_rounded": prompt_times,
        "input_frame_hashes": hashes,
        "input_array_shape_thwc": list(pixels.shape),
        "effective_height": effective_height,
        "effective_width": effective_width,
        "effective_pixels_per_frame": effective_height * effective_width,
        "video_grid_thw": grid,
        "patch_size": patch,
        "temporal_patch_size": temporal_patch,
        "spatial_merge_size": merge,
        "tensor_shapes": {k: list(v.shape) for k, v in inputs.items()},
        "visual_token_method": (
            "count native video token IDs; cross-check video_grid_thw / merge_size^2"
        ),
        "validated": True,
    }
    return inputs, processing, visual_tokens


class Qwen3VLBackend:
    def __init__(self, model, device, dtype, max_pixels, max_new_tokens, revision):
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen dependencies missing; install vidfit[qwen] in your virtualenv"
            ) from exc
        if importlib.metadata.version("transformers") != "4.57.6":
            raise RuntimeError("this adapter requires the verified transformers==4.57.6 processor")
        self.torch, self.device = torch, torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; install a compatible CUDA PyTorch wheel or explicitly "
                    "use --device cpu --dtype float32"
                )
            if self.device.index >= torch.cuda.device_count():
                raise ValueError("requested CUDA device does not exist")
            if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("requested bfloat16 is not supported by this GPU")
        elif dtype != "float32":
            raise ValueError(
                "CPU inference requires explicit --dtype float32 in this proof of concept"
            )
        self.max_pixels, self.max_new_tokens = max_pixels, max_new_tokens
        start = perf_counter()
        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
        if config.model_type != "qwen3_vl":
            raise ValueError("model must be a dense Qwen3-VL checkpoint")
        resolved = config._commit_hash
        if not resolved:
            raise ValueError("model must resolve to a Hugging Face commit for reproducibility")
        self.processor = AutoProcessor.from_pretrained(
            model, revision=resolved, trust_remote_code=False
        )
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model,
                revision=resolved,
                config=config,
                dtype=getattr(torch, dtype),
                attn_implementation="sdpa",
                trust_remote_code=False,
            )
            .to(self.device)
            .eval()
        )
        self.synchronize()
        self.info = {
            "identifier": model,
            "processor_identifier": model,
            "requested_revision": revision,
            "resolved_revision": resolved,
            "processor_resolved_revision": resolved,
            "loading_s": perf_counter() - start,
            "device": str(self.device),
            "dtype": dtype,
            "attention_implementation": "sdpa",
            "processor_class": type(self.processor).__name__,
            "video_processor_class": type(self.processor.video_processor).__name__,
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
            ),
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(self.device).total_memory
                if self.device.type == "cuda"
                else None
            ),
        }

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def run(self, frames, metadata, question):
        torch = self.torch
        self.synchronize()
        start = perf_counter()
        inputs, processing, visual_tokens = prepare_video_inputs(
            self.processor,
            frames,
            metadata,
            question,
            self.max_pixels,
        )
        self.synchronize()
        preprocessing_s = perf_counter() - start
        input_tokens = int(inputs["input_ids"].shape[1])
        start = perf_counter()
        inputs = inputs.to(self.device)
        self.synchronize()
        transfer_s = perf_counter() - start
        start = perf_counter()
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                )
            self.synchronize()
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory; rerun with explicit smaller --max-pixels or --frame-budgets; "
                "no automatic fallback was applied"
            ) from exc
        generation_s = perf_counter() - start
        start = perf_counter()
        output = generated[:, input_tokens:]
        answer = self.processor.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
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
            "visual_tokens": visual_tokens,
            "model_input_frames": processing["model_input_frames"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if cuda
            else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
            if cuda
            else None,
        }
        unavailable = (
            {}
            if cuda
            else {
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
