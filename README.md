# VidFit

**Ask a video. Inspect the inputs. Compare the result.**

Video models can miss a brief event because the relevant frame never reached them.
VidFit gives you a local interface to ask a question, see the answer beside the
frames supplied to the model, edit those frames, and compare another run. It also
records processed visual tokens, latency and GPU allocator memory for reproducible
experiments.

![VidFit local interface with real Qwen answers and selected input frames](docs/assets/interface.jpg)

The screenshot records a real run before final width/caption polish; see
[validation limits](docs/validation.md#v02-local-interface-release-2026-10-30).

v0.2 includes a Gradio interface, uniform and manual timestamp selection, an
optional five-clip / ten-question evaluation starter set, and offline HTML / JSON
exports. Automatic frame selection is a follow-up milestone.

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
.\.venv\Scripts\python.exe -m pip install -e ".[ui,qwen]"
.\.venv\Scripts\vidfit.exe ui
```

A single local worker imports the Qwen runtime at startup; weights load after Run.
Open **http://127.0.0.1:7860**. The server binds to loopback, disables Gradio
analytics and sharing, and serializes inference requests. Processing and inference
stay on your computer. Initial package/model downloads require internet access;
the model download is several GB. Reports contain your selected frames, so review
them before sharing.

On Linux, create/activate a Python 3.12 environment, install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -e ".[ui,qwen]"
vidfit ui
```

macOS supports the inspection path; Apple MPS inference is not implemented.
Linux setup is supported by the package; GPU/UI validation here was on Windows.

| Install | Purpose |
| --- | --- |
| `python -m pip install .` | Hardware-free inspection API / CLI; no model or UI imports |
| `python -m pip install ".[ui]"` | Local interface with `vidfit ui --backend inspect` |
| `python -m pip install ".[qwen]"` | Native Qwen3-VL inference |
| `python -m pip install -e ".[dev]"` | Tests, Ruff and package build |
| `python -m pip install -e ".[ui,qwen,dev]"` | Full development environment |

The UI extra pins Gradio 6.0.2 and a Hugging Face Hub range compatible with the
pinned Qwen adapter. No PyPI publication is claimed; install from this checkout.

For a hardware-free first look:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\vidfit.exe demo --output examples/generated/brief-error.mp4
.\.venv\Scripts\vidfit.exe ui --backend inspect --video examples/generated/brief-error.mp4
```

Inspection shows real decoded frames and exports, with **no model answer** and
unavailable inference costs. The demo generator refuses to overwrite existing
assets; reuse an existing file or choose a new path.

## Use your own video

1. Upload a video and use the player to inspect it. H.264 MP4 is the verified
   browser playback format. The core uses PyAV; no system FFmpeg executable is
   needed for that path. Audio is not sent to the model.
2. Enter a question. Choose **Answer once**, a uniform frame budget, then click
   **Preview selected frames** and **Run selected settings**.
3. Read **Model answer** beside the timestamped input gallery. Click a frame to
   enlarge it. Captions distinguish requested and actual source timestamps.
4. Switch selection A to **manual**, enter seconds separated by commas or spaces,
   preview again, and rerun. Your previous result remains next to the current one.
5. Choose **Compare settings** to run A and B sequentially with the same question,
   model and generation settings. Download the offline HTML report and structured
   JSON below the results.

An edit to the video or selection invalidates the preview. Manual timestamps must
be finite, inside `[0, duration)` and map to distinct source frames. Output frames
are chronological. Mapping uses nearest decoded presentation timestamps, with the
earlier frame winning a tie. Qwen requires at least two distinct frames and a
constant-frame-rate video; inspection also supports one-frame and VFR sources.
Native Qwen temporal padding is recorded separately from selected source frames.

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

analyzer = VideoAnalyzer(backend="inspect")  # use qwen3vl for inference
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

Real-model checks are separate and opt-in, using the local cache:

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

The adapter supports dense Qwen3-VL, one explicit CUDA device or explicit float32
CPU inference. It does not support audio reasoning, MPS, VFR Qwen inputs,
quantization, streaming, automatic OOM recovery or automatic frame selection.
Long videos require repeated full decodes. Embedded PNG exports can be large.
Gradio/browser playback support is narrower than PyAV decoding support.

Next milestones: automatic/scene-aware frame selection, broader independently
human-reviewed evaluation, additional backends, and budget recommendations.
These are future work. Contributions that improve reproducibility, accessibility,
CPU tests or annotation review are welcome.

VidFit source is [MIT](LICENSE). Model weights and downloaded clips retain their
own licenses; the [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
identifies the model's Apache-2.0 license. Large videos, generated reports, weights
and caches are excluded from Git.
