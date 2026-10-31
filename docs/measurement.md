# Measurement and evidence contract

Schema `1.1.0` is represented by dataclasses in `core/schemas.py`. Exported JSON
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

Uniform targets are `(i + 0.5) * duration / frame_budget`, relative to the first PTS.
A second streaming pass selects the nearest actual decoded PTS; ties choose the
earlier frame. Targets beyond the last PTS map to the final frame. The sampler
keeps only two decoder frame references plus selected PNG evidence. It does not
retain all decoded pixels. Each trial repeats the decode and evidence capture.
If two targets map to one source frame, the configuration fails; it never silently
duplicates evidence. Frame indices are zero-based presentation-order indices.

In standard `compare` runs, initial gallery extraction and the metadata probe are
outside trials. Their work can warm filesystem caches. Every trial's RGB hashes
must match the gallery.
Metadata probe counts/time are recorded separately. `decoded_frames` counts the
frames decoded within that trial, not the requested budget or metadata estimate.

### Visual-change selection and complete-request experiments

`SelectionSpec.visual_change(K)` uses the fixed `temporal-bin-rgb-difference-v1`
algorithm. Each source frame is bilinearly resized to 64×36 RGB; its score is the
mean absolute difference from the preceding frame divided by 255. The largest
positive score in each of K equal-duration bins wins, with earlier source frames
winning score ties. A static bin uses the nearest in-bin midpoint. Empty VFR bins
use unused, evenly spaced source-index anchors; the Qwen backend still rejects VFR.
Output is K distinct chronological source frames, without question or annotation
inputs. No score threshold is tuned to the evaluation collection.

The selector uses one full scan and one capture pass. Both are measured inside
`decode_sampling_s`, including source PNG capture and hashing. Scan and capture
sub-times are nested, not additive with that total. Numeric scan/capture frame
counts and a separate `selection_trace` record preserve selected scores, indices,
reasons, bin boundaries and fixed algorithm settings. Standard comparison reports
store setup provenance in `configuration.selection.trace` and per-trial provenance
in `trial.processing.selection_trace`; numeric metric summaries exclude that object.

The separate `experiment` runner compares visual-change 4, uniform midpoint 4 and
uniform midpoint 16 with one shared loaded backend. It records one warmup at each
input count, then cycles through the six method-order permutations across cases
and repeats. Every measured `end_to_end_s` starts before source metadata probing
and ends after final synchronization: probe/hash, selection scan, source capture,
model preprocessing, device transfer, vision encoding, prompt processing, answer
generation and text decoding are included. `input_preparation_s` sums probe,
decode/sampling, model preprocessing and transfer. Loading, warmups, reference
review, report exports and UI transport are excluded from per-request latency and
recorded separately where applicable. `startup_s` includes runtime imports and
model construction. `run_elapsed_s` covers setup, warmups, measured requests and
per-case exports, but excludes later answer review and summary rendering.
`decoded_frames` counts selection/capture passes; the additional metadata decode
is timed in `probe_s` and included in `end_to_end_s`.

All per-trial answers are retained, including failures. Review packets hide method
and selection metadata and deduplicate only identical answers to the same question.
Explicit judgments are bound to question/answer hashes. Summaries show supported
correctness, appropriate abstention on unsupported questions, partial answers and
failures separately; unresolved judgments produce unavailable accuracy. Repeated
trials remain repeated observations of the same underlying question. Existing
`compare` timing and invocation contracts are unchanged.

## Model routing and representations

`auto` resolves the checkpoint configuration on demand. Dense `qwen3_vl` and
`qwen3_5` use the native video adapter; other compatible checkpoints use the generic
Transformers image adapter. `qwen`/`qwen3vl` require the native Qwen path, while
`transformers` explicitly requests images. Routing never substitutes a different
checkpoint after a failure. Model type/class, resolved revision, runtime version
and input representation distinguish these paths in exported records.

The generic path requires a native `AutoModelForImageTextToText` class, a usable
processor/chat template and a supported image tensor layout. Native upstream
registration alone does not establish adapter compatibility. Remote checkpoint
code is disabled. The supplied `vlm` profile pins Transformers 5.5.4; the preset
catalog is advisory, and actual validation is recorded in [validation](validation.md).

## Native Qwen video processing

The adapter follows the [official Qwen3-VL processing path](https://github.com/QwenLM/Qwen3-VL)
with `AutoProcessor`, `videos`, `video_metadata`, and native video placeholders.
Already selected RGB arrays go directly to the processor with
`do_sample_frames=False`. Returned video metadata is retained for validation,
removed from the BatchFeature, and only then are inputs converted to tensors.
There is no second decoder and no image-input fallback.
`qwen-vl-utils` is unnecessary because VidFit has already decoded and sampled.

Dependency profiles pin Transformers 4.57.6 (legacy) or 5.5.4 (modern). The adapter
selects the native dense Qwen3-VL or Qwen3.5 class from the checkpoint configuration
and checks actual processor outputs. Unsupported families fail before weights load.
Model and processor share a resolved commit; runtime version, model type/class and
requested precision are recorded. See [model switching](model-switching.md).
The native implementation's `size.longest_edge` is a total pixel cap across
frames. VidFit multiplies `--max-pixels` by the selected count to express a
consistent per-frame cap. The minimum is 4096 pixels per selected frame.
The processor rounds spatial dimensions to multiples of 32. Effective dimensions,
pixel counts, grid, and tensor shapes are recorded; exceeding the requested cap
is rejected. The effective dimensions can vary with source aspect ratio and rounding.

Thinking is requested off by default. The adapter checks whether the checkpoint's
chat template responds to the flag and records requested/effective mode and whether
it is controllable. An uncontrolled template retains its native behavior with unknown
effective mode; requesting thinking on then fails before weight loading. All generated
text and token costs are retained, including reasoning if present. Thinking requests
and output budgets must be controlled in model comparisons.

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

## Generic timestamped-image processing

The generic adapter decodes the same selected source evidence and verifies its
RGB hashes. It creates bounded RGB copies using aspect-preserving LANCZOS
downsampling, then supplies one multi-image conversation: chronological frames,
text labels containing frame ordinal and source PTS rounded to six decimals, and
the question. Actual source times, original/resized hashes and resized dimensions
are retained. The processor can perform additional resizing, padding or tiling.
Original gallery frames are not claimed to equal normalized processor tensors.

The representation is `timestamped_images`, with `native_video_input=false`.
There is no native video sampler, Qwen-style temporal pairing or duplicated-final-
frame requirement. Single-frame and variable-rate inputs are allowed when the
processor accepts them. The reported validation scope covers source RGB identity,
chronological image submission and the actual dense tensor pixel budget; it does
not establish native-video equivalence or temporal-token semantics.

Pixel accounting recognizes dense image tensors whose dimensions map to the
selected frames, with optional batch and crop axes. The allocated crop count times
height times width for each source frame must not exceed `--max-pixels`. Padded
views are counted conservatively. Opaque layouts and budgets exceeded by processor
resizing/tiling fail before generation. The cap is not a limit on total GPU memory,
prompt length or intermediate CPU preprocessing allocations.

SmolVLM uses an explicit no-splitting image policy and preserves the checkpoint's
fixed image size. The 500M checkpoint produces one 512×512 view per selected frame,
requiring a pixel cap of at least 262144 even for smaller source images. Processor
overrides are included in the report. Other processors retain their native image
settings and must pass the actual tensor audit.

`model_input_frames` is the number of submitted source images on this path;
`image_tiling.processed_image_views` separately counts processor views. Generic
`visual_tokens` remains null with an explanation: no cross-model visual-token
mapping is inferred from image placeholders. `input_tokens` counts the returned
text token sequence; vision embeddings added elsewhere by a model may not appear
in that count. Tensor shapes, view dimensions and image budgets remain available.
Encoder-decoder outputs are decoded directly; decoder-only outputs exclude the
input prompt. Thinking control and generation timing follow the same explicit
request policy as the native adapter.

## Native sampling traces

The separate `VideoAnalyzer.trace_sampling` API, `vidfit trace` command and UI trace
panel call the chosen checkpoint's installed native Qwen video processor's actual
`sample_frames` method on source metadata. They never construct a model or call
generation, and do not alter the existing preview/inference selection. The model
configuration is resolved to a revision and the video processor is loaded from that
revision; effective runtime defaults are recorded even when the configuration file
omits them. Optional Qwen dependencies load only when a trace is explicitly requested.
The trace is unavailable for the generic image adapter and non-Qwen checkpoints;
it never silently traces a different model.

`matched` passes VidFit's selected count as `num_frames`. `default` leaves sampling
arguments at the loaded processor defaults. `fps` passes an explicit positive,
finite rate. Requested arguments and the actual returned count are recorded;
internal clamping/intermediate counts are not inferred. The verified 4.57.6 sampler
uses endpoint linspace with NumPy rounding and FPS-based count limits, but the trace
does not assume all runtime versions implement those rules identically. Provenance
includes the actual processor module, installed version, source URL and callable
source hash. Returned indices are decoded directly, not remapped through
VidFit's nearest-timestamp sampler. Neither the question nor reference annotations
are inputs to this operation.

Trace rows distinguish actual decoded time relative to the first PTS from native
index/FPS time. Raw PTS/time base and source identity are retained. This release
requires a constant-rate source whose index/FPS times match its actual timeline;
it rejects missing or incompatible timing rather than guessing a frame rate.
Derived temporal groups and rounded prompt timestamps are labeled as projections;
they do not prove that the video processor or model accepted the inputs. Padding
is separate from selected source frames. No answer, attention attribution or
measured visual-token count is claimed by the trace.

The independent trace schema and HTML/JSON exports retain all selected positions,
effective processor settings, sampler arguments, shared/unique indices and duplicate
indices. Only a bounded set of resized source thumbnails is embedded. The HTML is
self-contained. Trace setup is independent of benchmark measurements, and existing
comparison schema 1.1 and execution counts remain unchanged.

## Timing and memory

| Field | Boundary |
| --- | --- |
| `model.loading_s` | Config/processor/weight loading, device placement, final CUDA sync; may include network downloads |
| `decode_sampling_s` | Open/decode all video frames, select, capture lossless PNGs, and hash RGB pixels |
| `preprocessing_s` | Convert selected PNGs to RGB inputs, format prompts, apply the processor and validate its outputs; includes generic input resizing and pixel-budget checks |
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


## v0.2 schema and execution modes

Schema 1.1.0 adds per-configuration `selection` and `setup`, per-trial `evaluation`,
optional top-level `annotation`, and `settings.execution_mode`. Existing frame,
trial and metric fields retain their meaning. `Comparison.from_dict` also reads
1.0.0 records with the new fields defaulted. The original uniform API and benchmark
invocation counts remain compatible. `settings.sampling` is `specified` when manual
selections are present; use each configuration's selection method for interpretation.

`SelectionSpec.uniform(n)` preserves midpoint sampling.
`SelectionSpec.manual([...])` preserves the requested timestamp list; its validated,
sorted targets are retained in `selection.target_timestamps_s`. Timestamps must be
finite and inside the half-open range `[0, duration)`. Both methods use the same
nearest-source-PTS mapper and earlier-frame tie rule. Different requests that map
to the same frame fail instead of duplicating source inputs. Trials compare frame
indices, actual/requested timestamps and RGB hashes with the initial gallery.

`execution_mode="single"` has **exactly one measured invocation per configuration**,
no first-invocation call and no warmups. Effective warmup/repeat settings are 0/1,
even when callers leave benchmark defaults in place. The first single run may pay
initial kernel/runtime costs; it is not a steady-state benchmark. `benchmark`
retains one first invocation across the comparison, then configured warmups and
repetitions per selection. Only successful measured trials enter summaries.

The UI defaults to single mode in both Answer once and Compare settings. Its
benchmark checkbox explicitly opts into repeated calls. A model is constructed per
request and reused across that request's configurations. Preview metadata and
frame extraction precede inference and are excluded from pipeline latency.
Configuration `setup` stores initial decode/sampling costs; `video.probe_time_s`
records metadata work. The UI also requires a matching preview snapshot, model
settings and video hash. The trial decode remains measured; no preview pixels are
substituted for it.

References and rubrics live in an optional annotation file separate from synthetic
fixture manifests. Only source frames, the question and source temporal metadata
are supplied to the backend. Annotation content is validated separately and
attached to exported results after inference. No reference answer, rubric, review
status or relevant interval is placed in model messages or processor metadata.

Each trial starts with `evaluation.verdict="not_evaluated"` and
`method="unreviewed"`. An explicit judgment records a verdict, method, rationale,
reviewer/check identifier, timestamp and answer hash. Methods are `human`,
`automated` and `agent`; VidFit does not infer or certify human authorship.
`unanswerable` means a judgment that the answer appropriately recognizes the
question cannot be answered from the video. It is not a synonym for runtime failure.
The collection's `answerability` separately describes the question/reference.

Coverage tests whether **any** actual selected timestamp falls within any annotated
half-open interval. It does not measure comprehensive temporal coverage. With no
applicable intervals it is null/Not annotated. Correctness, evidence overlap,
completion, and answer agreement are independent. Agent-reviewed starter references
are not human-reviewed labels, and the starter set does not establish general
accuracy. See [evaluation](evaluation.md).


The UI worker starts without importing model dependencies or loading weights.
Backend imports and model construction occur when an answer is requested; native
tracing separately loads only its configuration/processor dependencies. Model
loading and process/import startup are outside measured inference. The loading
metric covers adapter construction; the experiment's `startup_s` additionally
wraps runtime dispatch and imports.
With `selections=...`, those specifications replace the legacy `frame_budgets`
argument. CLI exit codes remain 0 for complete runs, 1 for failed/incomplete
configurations, and 2 for invalid top-level input or launch errors.


UI callbacks dispatch to one persistent local worker process, avoiding event-loop
contention during decoding, imports and model work. It executes the same analyzer
methods; no inference service is used. A single worker and Gradio's shared queue
serialize jobs. IPC, PNG gallery transport and UI rendering are outside core trial
latency. Model weights are loaded per request and reused only across that request's
configurations. Per-request model settings are copied into a new analyzer and the
preview snapshot is checked against them. Model-setting edits invalidate the
preview, trace and downloads, and cancel stale callback delivery; an already
executing worker job may finish. Previous results retain their own model labels.
