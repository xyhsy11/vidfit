# VidFit

**Ask a video. Inspect the inputs. Compare the result.**

Video models can miss a brief event because the relevant frame never reached them.
VidFit gives you a local interface to ask a question, see the answer beside the
frames supplied to the model, edit those frames, and compare another run. It also
records latency and GPU allocator memory, plus verified visual-token counts on
the native Qwen path, for reproducible experiments.

![VidFit local interface with real Qwen answers and selected input frames](docs/assets/interface.jpg)

The screenshot records a real run before final width/caption polish; see
[validation limits](docs/validation.md#v02-local-interface-release-2026-10-30).

v0.2 includes a Gradio interface, uniform and manual timestamp selection, an
experimental visual-change selector, an optional five-clip / ten-question
evaluation starter set, and offline HTML / JSON exports.

## Install and launch

Python **3.10–3.13** is supported; **3.12** is the verified local version.
The real-model path was verified on Windows with an **RTX 3080 Laptop GPU, 16 GB**,
PyTorch 2.8.0+cu128, Transformers 4.57.6 and cached
`Qwen/Qwen3-VL-2B-Instruct`. This is a tested configuration, not a minimum GPU
requirement. CPU tests need neither PyTorch nor weights.

Windows PowerShell, from a new checkout:

```powershell
git clone https://github.com/xyhsy11/vidfit.git
cd vidfit
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[ui,vlm]"
.\.venv\Scripts\vidfit.exe ui --backend auto
```

A single local worker handles requests; model dependencies and weights load on Run.
Open **http://127.0.0.1:7860**. The server binds to loopback, disables Gradio
analytics and sharing, and serializes inference requests. Processing and inference
stay on your computer. Initial package/model downloads require internet access;
model downloads can be several GB. Reports contain your selected frames, so review
them before sharing.

On Linux, create/activate a Python 3.12 environment, install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -e ".[ui,vlm]"
vidfit ui --backend auto
```

macOS supports the inspection path; Apple MPS inference is not implemented.
Linux setup is supported by the package; GPU/UI validation here was on Windows.

| Install | Purpose |
| --- | --- |
| `python -m pip install .` | Hardware-free inspection API / CLI; no model or UI imports |
| `python -m pip install ".[ui]"` | Local interface with `vidfit ui --backend inspect` |
| `python -m pip install ".[qwen]"` | Legacy Qwen3-VL runtime, Transformers 4.57.6 |
| `python -m pip install ".[qwen35]"` | Qwen3-VL and Qwen3.5 runtime, Transformers 5.5.4 |
| `python -m pip install ".[vlm]"` | Modern Qwen runtime plus compatible Transformers image-chat models |
| `python -m pip install -e ".[dev]"` | Tests, Ruff and package build |
| `python -m pip install -e ".[ui,vlm,dev]"` | Full modern development environment |

The UI extra pins Gradio 6.0.2 and supports both runtime dependency ranges. Choose
one model profile per environment. The legacy `qwen` pin conflicts with `qwen35`
and `vlm`; use a separate environment to preserve old benchmarks. No PyPI
publication is claimed; install from this checkout.

For a hardware-free first look:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\vidfit.exe demo --output examples/generated/brief-error.mp4
.\.venv\Scripts\vidfit.exe ui --backend inspect --video examples/generated/brief-error.mp4
```

Inspection shows real decoded frames and exports, with **no model answer** and
unavailable inference costs. The demo generator refuses to overwrite existing
assets; reuse an existing file or choose a new path.

## Switch models

Install `.[ui,vlm]`, then choose a preset or enter a Hugging Face model ID in the
interface. Preview again after changing model settings, then run without restarting
the server. The same choice is available in the CLI through `--backend auto`:

```powershell
vidfit models
vidfit ui --backend auto --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct --dtype bfloat16
vidfit ui --backend auto --model Qwen/Qwen3.5-2B --dtype bfloat16
vidfit ui --backend auto --model Qwen/Qwen3-VL-2B-Instruct --dtype float16
```

Automatic routing uses native video for supported dense Qwen checkpoints. Other
compatible `AutoModelForImageTextToText` checkpoints receive chronological selected
images with timestamp labels. Custom IDs are allowed; processors must satisfy
the adapter's input checks. This does not promise support for every Hugging Face
model. SmolVLM2 500M and LLaVA-OneVision 0.5B completed real 4- and 16-frame
integration checks; this is not an accuracy claim. Untested presets are identified in the
[model switching guide](docs/model-switching.md).

The generic adapter reports visual tokens as unavailable, and verifies pixel
budgets against actual processed views, including crops and padding. SmolVLM2 500M
has a fixed 512×512 minimum per image. Native Qwen tracing and temporal-token checks
remain specific to the native Qwen adapter. Existing `qwen`/`qwen3vl` backend
commands keep that behavior; `transformers` explicitly selects the image adapter.

Use a separate virtual environment to retain the original benchmark runtime.
The [model switching guide](docs/model-switching.md) gives complete Windows setup,
CLI/API examples, offline use and validation limits. The same `--model`,
`--revision`, and `--dtype` options work for comparisons and selection experiments.
The native Qwen sampling trace uses the chosen checkpoint's processor without weights.

Thinking is requested off by default; opt in with `--thinking`. Reports distinguish
the requested setting from whether the checkpoint's chat template can control it.
A fixed template keeps its native behavior. Unsupported model classes, missing
dependencies and incompatible processor outputs fail explicitly. Model weights are
downloaded on first inference if not cached. No other model is substituted.

## Use your own video

1. Upload a video and use the player to inspect it. H.264 MP4 is the verified
   browser playback format. The core uses PyAV; no system FFmpeg executable is
   needed for that path. Audio is not sent to the model.
2. Choose a model and enter a question. Choose **Answer once**, a uniform frame
   budget, then click **Preview selected frames** and **Run selected settings**.
3. Read **Model answer** beside the timestamped input gallery. Click a frame to
   enlarge it. Captions distinguish requested and actual source timestamps.
4. Switch selection A to **manual**, enter seconds separated by commas or spaces,
   preview again, and rerun. Your previous result remains next to the current one.
5. Choose **Compare settings** to run A and B sequentially with the same question,
   model and generation settings. Download the offline HTML report and structured
   JSON below the results.

An edit to the video, selection or model settings invalidates the preview. Manual
timestamps must be finite, inside `[0, duration)` and map to distinct source frames. Output frames
are chronological. Mapping uses nearest decoded presentation timestamps, with the
earlier frame winning a tie. Qwen requires at least two distinct frames and a
constant-frame-rate video; inspection and compatible generic image processors can
also use one-frame and VFR sources.
Native Qwen temporal padding is recorded separately from selected source frames.

## Inspect how Qwen picks frames

Open **How Qwen picks frames** in the interface and click **Trace native sampling**.
The trace compares **Selection A** with the actual Qwen3-VL video processor sampler:

- **Same frame count as Selection A** isolates the difference in frame placement.
- **Processor default FPS** uses the loaded processor's sampling settings.
- **Custom FPS** lets you change the native sampling rate and see its effect.

The trace shows source-frame thumbnails, every selected index and decoded timestamp,
shared frames, effective settings and processor provenance. It also shows derived
temporal groups and the index/FPS timestamps Qwen would use. Those are projections
of the grouping rule; video preprocessing and model inference have not been run.
Thumbnails are bounded and resized; all selected positions remain in the tables
and downloadable JSON. Download the separate offline HTML to keep the comparison.

Native sampling uses video metadata and sampling settings, not the question or
visual content. VidFit's chosen selection still supplies the frames for answering;
tracing does not change it. The trace requires a compatible `[qwen]`, `[qwen35]`
or `[vlm]` profile; Qwen3.5 requires a modern profile. It downloads only configuration
files if they are not cached. It loads no model weights and needs no GPU. Base inspection remains
usable without Qwen dependencies.

From the command line, with an existing video:

```powershell
.\.venv\Scripts\vidfit.exe trace --video examples/generated/brief-error.mp4 --frames 4 --output-dir reports/native-trace
.\.venv\Scripts\vidfit.exe trace --video examples/generated/brief-error.mp4 --timestamps 7.5 24.375 37.5 52.5 --native-mode default --output-dir reports/native-default
.\.venv\Scripts\vidfit.exe trace --video examples/generated/brief-error.mp4 --frames 4 --native-mode fps --native-fps 1 --output-dir reports/native-1fps
```

For API use, call `analyzer.trace_sampling(video, SelectionSpec.uniform(4))`.
The result is a JSON-ready dictionary, separate from answer-comparison records.
See [native sampling traces](docs/measurement.md#native-sampling-traces) for the
sampling and timestamp contract.

## Execution and result costs

**Answer once and Compare settings each run exactly one inference per selected
configuration by default.** Enable the explicit benchmark checkbox for one
separate first invocation, then the configured warmups and measured repetitions.
The model is loaded once per request and reused across its configurations; loading,
metadata probing, previews and exports are outside measured inference.

Use `vidfit ui --max-pixels 131072 --max-new-tokens 64` to choose smaller explicit
limits. There is no silent model, frame, precision or memory fallback. Unavailable
CUDA, unsupported videos, invalid selections and run failures are surfaced in the
interface or report. A single UI server serializes runs; separate CLI/server
processes are the operator's responsibility.

## Compare a visual-change selector with uniform sampling

Choose **Visual changes** for Selection A or B in the UI, or use
`SelectionSpec.visual_change(4)` in the API. This fixed baseline compares consecutive
64×36 RGB thumbnails and selects the largest change in each equal-duration time
bin. A static bin uses its midpoint. The selector receives neither your question
nor evaluation references, and preserves the selected source frames and timestamps.
It can select distracting motion; a large pixel change is not proof of relevance.

For a complete experiment on the prepared starter collection:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\vidfit.exe experiment --collection examples/evaluation/starter.json --video-dir examples/generated/evaluation --backend qwen3vl --repeats 2 --max-pixels 262144 --max-new-tokens 64 --output-dir reports/visual-change-run
```

Use a new output directory for each run. The command compares visual-change 4 with
**VidFit uniform midpoint 4 and 16**, not Qwen's native endpoint sampler. It loads
one model, records two warmups, rotates method order and measures each request from
metadata probing through the answer, including the selector and all preprocessing.
Model startup and whole-experiment wall time are separate. All answers, frames,
costs and failures remain in the reports.

`summary.html` and `summary.json` initially leave correctness unreviewed.
`answers-for-review.json` deduplicates identical answers within each question and
hides method, timing and frame-selection information. Explicit review records need
`answer_id`, `verdict`, `method`, `reviewer` and `rationale`. Apply them with:

```powershell
.\.venv\Scripts\vidfit.exe review-experiment --output-dir reports/visual-change-run --judgments reports/visual-change-run/review.json
```

Raw comparison records are retained. Strict success means a correct supported
answer or appropriate abstention on an unsupported question; partial answers are
reported separately. Repeated runs are not additional independent questions. This
small starter collection supports an exploratory comparison, not general accuracy
claims or independent human certification.

The [real Qwen pilot results](docs/visual-change-results.md) report 60%, 60%, and
70% strict success for visual-change 4, uniform 4, and uniform 16, with mean complete
request times of 2.528, 2.576, and 3.574 seconds. This five-clip pilot found no overall
accuracy gain over uniform 4; all preprocessing costs are included.

## Interpret results

- **Run status** says whether processing completed or failed. It is not a quality score.
- **Evidence coverage** says whether selected timestamps overlap an optional
  annotated interval. With no applicable annotation it is unavailable.
- **Answer correctness** is **Not evaluated** until an explicit judgment is recorded.
  Human, automated and agent judgments are labeled separately.
- Displayed frames are **inputs provided to the model**, not model-verified
  citations or explanations. Answer agreement is not accuracy.
- **Latency** is elapsed work. Longer generated answers can take more time.
  **Visual tokens** count processed visual input; **GPU memory** measures PyTorch
  allocator peaks including resident weights, not total device use.

Every benchmark trial retains its own answer, phase, measurements and evaluation.
The offline HTML embeds frames and styling; JSON retains source indices, PTS,
hashes, requested timestamps, selection method, settings and unavailable reasons.
See the [measurement contract](docs/measurement.md) for precise boundaries and
[schema 1.1 compatibility notes](docs/measurement.md#v02-schema-and-execution-modes).

## Reproduce the brief-event comparison

The synthetic 60-second application clip contains an error only in `[24,26)`.
Four uniform midpoints miss it. Four manual inputs can include it with the same
frame count:

```powershell
.\.venv\Scripts\vidfit.exe compare --video examples/generated/brief-error.mp4 --question "What error message appeared?" --frame-budgets 4 --timestamps 7.5 24.375 37.5 52.5 --backend qwen3vl --execution-mode benchmark --warmup 1 --repeats 3 --fixture-manifest examples/generated/brief-error.truth.json --output-dir reports/comparison
```

Use `--execution-mode single` for one invocation per configuration, or
`--backend inspect` to examine inputs without weights. With only `--timestamps`,
the CLI runs only manual selection; explicit `--frame-budgets` adds uniform
configurations. `--sampling manual` requires timestamps. The original uniform CLI
and `VideoAnalyzer.compare(..., frame_budgets=[...])` defaults remain compatible.
CLI comparison still defaults to **benchmark**: one first invocation, one warmup
and three measured trials per configuration. Even `--warmup 0` keeps that first
invocation unless single mode is selected.

Observed on the reference GPU, with identical 262144-pixel caps, greedy generation
and 128-token limits:

| Four-frame selection | Event overlap | Agent answer judgment | Visual tokens | Output tokens | Median generation / pipeline | Peak allocated / reserved |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | No | Incorrect: says no error appeared | 440 | 23 | 18.132 / 19.462 s | 4193.9 / 4260.0 MiB |
| Manual | Yes | Partially correct: includes error plus unrelated status text | 440 | 14 | 11.937 / 13.725 s | 4193.9 / 4260.0 MiB |

These are observations on one synthetic clip, not evidence of general accuracy or
speed improvement. Answer lengths differ and timings vary with runtime/system
load. [Actual answers, settings and validation](docs/v0.2-results.md) include the
existing downloaded real-world sample. No accuracy improvement is required for
the software to function.

## Python API

```python
from vidfit import SelectionSpec, VideoAnalyzer

analyzer = VideoAnalyzer(backend="inspect")  # use auto with a compatible model for inference
manual = SelectionSpec.manual([7.5, 24.375, 37.5, 52.5])
metadata, preview_frames, setup_costs = analyzer.preview("video.mp4", manual)
result = analyzer.compare(
    "video.mp4",
    "What error message appeared?",
    selections=[SelectionSpec.uniform(4), manual],
    execution_mode="single",
)
result.export_html("reports/comparison.html")
result.export_json("reports/comparison.json")
assert result.complete  # completion only; does not assert answer correctness
```

## Starter evaluation and development

The [evaluation guide](docs/evaluation.md) documents five short clips, ten
questions, download/generation checksums, source licenses, visual-review provenance,
reference isolation, and explicit answer judgments. It includes real footage,
synthetic sustained actions, brief events and unsupported questions. It is a
starter collection for exercising the workflow, not a representative benchmark.

```powershell
.\.venv\Scripts\python.exe -m pip install av==16.0.1 Pillow==12.0.0 numpy==2.2.6
.\.venv\Scripts\python.exe examples/prepare_evaluation.py
.\.venv\Scripts\vidfit.exe evaluate --collection examples/evaluation/starter.json --video-dir examples/generated/evaluation --backend inspect --output-dir reports/starter
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m build
```

Qwen native-video model checks are separate and opt-in, using the local cache:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:VIDFIT_RUN_MODEL_TEST = "1"
.\.venv\Scripts\python.exe -m pytest -q -m model
```

On Unix, use `VIDFIT_RUN_MODEL_TEST=1 python -m pytest -q -m model` and the
corresponding `export` commands for offline cache variables. CPU CI covers Windows
and Linux on Python 3.10 and 3.12; see [validation](docs/validation.md) for actual
local checks and limits. [CONTRIBUTING.md](CONTRIBUTING.md) explains the workflow.

## Limits and roadmap

Native video supports dense Qwen3-VL and dense Qwen3.5. The generic adapter supports
compatible Transformers image-chat checkpoints and recognized dense image layouts;
its selected-image representation is not equivalent to native video. Inference uses
one explicit CUDA device or explicit float32 CPU. It does not support audio
reasoning, MPS, VFR Qwen inputs,
quantization, streaming, automatic OOM recovery or question-aware frame selection.
Long videos require repeated full decodes. Embedded PNG exports can be large.
Gradio/browser playback support is narrower than PyAV decoding support.

Next milestones: semantic/question-aware frame selection, broader independently
human-reviewed evaluation, additional backends, and budget recommendations.
These are future work. Contributions that improve reproducibility, accessibility,
CPU tests or annotation review are welcome.

VidFit source is [MIT](LICENSE). Model weights and downloaded clips retain their
own licenses; the [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
identifies the model's Apache-2.0 license. Large videos, generated reports, weights
and caches are excluded from Git.
