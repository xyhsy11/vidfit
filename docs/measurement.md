# Measurement and evidence contract

Schema `1.0.0` is represented by dataclasses in `core/schemas.py`. Exported JSON
includes every trial, full source PNGs, RGB hashes, source PTS/time bases, requested
timestamps, software versions, model/processor revisions, settings, summaries,
and validation results. Times use seconds, memory uses bytes, and token/frame
counts are integers. Null means unavailable; trial `unavailable_reasons` explains
why. The HTML uses MiB for memory (1 MiB = 1,048,576 bytes).

## Decode and sampling

A streaming metadata pass counts all decoded frames and establishes a timeline
from the first PTS through the last PTS plus its frame duration. A missing final
duration falls back to the preceding frame interval, or nominal fps for a single
frame. Non-increasing or missing PTS are rejected. No audio is used.

Targets are `(i + 0.5) * duration / frame_budget`, relative to the first PTS.
A second streaming pass selects the nearest actual decoded PTS; ties choose the
earlier frame. Targets beyond the last PTS map to the final frame. The sampler
keeps only two decoder frame references plus selected PNG evidence. It does not
retain all decoded pixels. Each trial repeats the decode and evidence capture.
If two targets map to one source frame, the configuration fails; it never silently
duplicates evidence. Frame indices are zero-based presentation-order indices.

An initial gallery extraction and the metadata probe are outside trials. Their
work can warm filesystem caches. Every trial's RGB hashes must match the gallery.
Metadata probe counts/time are recorded separately. `decoded_frames` counts the
frames decoded within that trial, not the requested budget or metadata estimate.

## Native video processing

The adapter follows the [official Qwen3-VL processing path](https://github.com/QwenLM/Qwen3-VL)
with `AutoProcessor`, `videos`, `video_metadata`, and native video placeholders.
Already selected RGB arrays go directly to the processor with
`do_sample_frames=False`. Returned video metadata is retained for validation,
removed from the BatchFeature, and only then are inputs converted to tensors.
There is no second decoder and no image-input fallback.
`qwen-vl-utils` is unnecessary because VidFit has already decoded and sampled.

The adapter is pinned to Transformers 4.57.6 and checks its actual outputs.
The native implementation's `size.longest_edge` is a total pixel cap across
frames. VidFit multiplies `--max-pixels` by the selected count to express a
consistent per-frame cap. The minimum is 4096 pixels per selected frame.
The processor rounds spatial dimensions to multiples of 32. Effective dimensions,
pixel counts, grid, and tensor shapes are recorded; exceeding the requested cap
is rejected. The effective dimensions can vary with source aspect ratio and rounding.

This processor represents time through `frames_indices / fps`, then averages
pairs and formats prompt timestamps to one decimal place. VidFit retains exact
source PTS alongside these effective temporal values and verifies the generated
timestamp text. The adapter requires constant-frame-rate video and checks each
selected index/fps against actual relative PTS. VFR is available in inspection
mode; Qwen rejects it rather than misrepresenting timestamps. A single selected
frame is unsupported by the native video resizer. Odd counts duplicate the final
frame for temporal patch size 2; that duplication is checked and reported
separately from distinct selected source frames. The gallery shows the source
frames, not normalized model tensor pixels.

Visual tokens are the actual count of native video-token IDs in `input_ids`,
cross-checked against `product(video_grid_thw) / spatial_merge_size²`. They are
not inferred from raw frame dimensions. Input-token count includes text,
timestamps, control tokens, and video tokens. Generated-token count includes
any terminal token produced by generation. Answers are decoded without special
tokens. `max_new_tokens=128`, `do_sample=False`, `num_beams=1`, and `use_cache=True`
are fixed across configurations unless the output limit is explicitly changed.

## Timing and memory

| Field | Boundary |
| --- | --- |
| `model.loading_s` | Config/processor/weight loading, device placement, final CUDA sync; may include network downloads |
| `decode_sampling_s` | Open/decode all video frames, select, capture lossless PNGs, and hash RGB pixels |
| `preprocessing_s` | Convert selected PNGs to RGB arrays, native processor, temporal/grid/token validation |
| `device_transfer_s` | Move model input tensors to the explicit device and synchronize |
| `generation_s` | `model.generate` through synchronization; includes vision encoding, prefill, and decoding |
| `output_decode_s` | Decode output tokens to text and synchronize |
| `pipeline_s` | Per-trial decode through answer and final synchronization, including bookkeeping overhead |

Model construction is outside trials. One first invocation is retained separately
for the comparison, followed by one warmup and three measured trials per budget
by default. Even `--warmup 0` retains that separate first invocation. The first
call for later budgets uses an already loaded, previously invoked model; it is
not a cold start. No fresh-process cold-start claim is made. The public API
constructs a new backend per `compare` call; reuse is within that comparison.

All GPU configurations run sequentially on one explicit device. CUDA is
synchronized at stage boundaries and peak counters reset before each trial.
This follows [PyTorch's CUDA guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html).
Peak allocated memory includes resident tensors such as weights. Peak reserved
memory includes allocator-managed cached blocks; the allocator cache is retained
between trials. Neither metric measures total device use, other processes, or
driver overhead. CPU mode has null CUDA memory metrics. No monetary cost estimate
is made.

Summaries use successful `measured` trials only: count, median, min, max, and sample
standard deviation (null for fewer than two observations). Failures remain in the
trial list, mark the configuration incomplete, and cause a nonzero CLI exit even
if other configurations succeed. A partial summary must not be treated as a
complete experiment.

## Reproducibility and privacy

Use a resolved model SHA, exact dependencies, video SHA, and exported settings to
repeat an experiment. The generator is byte-reproducible in the same pinned
Pillow/PyAV/FFmpeg environment; cross-version encoder/font output is not promised.
Greedy decoding does not guarantee bit-identical results across GPU architectures,
drivers, or numerical implementations. Evidence coverage is deterministic; model
correctness remains an experimental observation.

Only explicitly supplied fixture manifests are read. They must match the video
SHA-256. Event overlap uses actual selected timestamps in the known half-open
event interval; it is not a general accuracy metric. Fixture text never enters
the processor input. Reports use the video basename and omit host/user names.
Reports embed source video frames and can contain sensitive content; share them
intentionally. Dependency/model downloads need network access, but decoding and
inference run locally.
