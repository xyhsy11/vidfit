# VidFit

**Inspect the frames. Compare the evidence. Measure the run.**

VidFit is an open-source, local-first Python workbench for inspecting and comparing
video-language model inputs and inference costs.

A model can miss a brief event because it never received the relevant frame.
VidFit makes that evidence visible alongside the answer, processed visual tokens,
latency, and CUDA allocator memory. This first proof of concept compares uniform
frame budgets through an offline HTML report and structured JSON.

## Current status

Implemented: an installable package, CLI and Python API, streaming PyAV decoding,
deterministic midpoint sampling, a synthetic demo, hardware-free inspection,
a native Qwen3-VL adapter, repeated trials, and self-contained exports.
The full 4/8/16-frame Qwen comparison ran successfully on a 16 GB RTX 3080
Laptop GPU. CPU tests and CI require no model weights or PyTorch. See
[validation](docs/validation.md) for actual checks and remaining validation gaps.

The demo is a 60-second **synthetic** application recording. An error appears only
during `[24.0, 26.0)` seconds. Four selected frames miss it; sixteen include it.
This evidence difference is verified in the decoded pixels. A correct model
answer is an experimental result, not a promised consequence of a larger budget.

## Quick start: inspect without a GPU

Use Python **3.10–3.13**; Python **3.12** is the reference version. Python 3.14 is
outside this milestone's dependency range. No system FFmpeg executable is needed:
PyAV wheels include the FFmpeg libraries used for decoding and demo encoding.

Windows PowerShell:

```powershell
git clone https://github.com/xyhsy11/vidfit.git
cd vidfit
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\vidfit.exe demo --output examples/generated/brief-error.mp4
.\.venv\Scripts\vidfit.exe compare --video examples/generated/brief-error.mp4 --question "What error message appeared?" --frame-budgets 4 8 16 --sampling uniform --backend inspect --fixture-manifest examples/generated/brief-error.truth.json --output-dir reports/inspection
```

Linux/macOS:

```sh
git clone https://github.com/xyhsy11/vidfit.git
cd vidfit
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
vidfit demo --output examples/generated/brief-error.mp4
vidfit compare --video examples/generated/brief-error.mp4 --question "What error message appeared?" --frame-budgets 4 8 16 --sampling uniform --backend inspect --fixture-manifest examples/generated/brief-error.truth.json --output-dir reports/inspection
```

For a smaller install, use `pip install .` without `[dev]`. No PyPI release is
claimed. If `py` is unavailable on Windows,
use the full path to an installed Python 3.12 interpreter. Activation is optional;
the explicit PowerShell paths avoid changing execution policy.

Open `reports/inspection/comparison.html` in a browser. Its styling and source
frames are embedded; it needs no server, internet connection, or external assets.
Click an evidence frame to enlarge it. The corresponding `comparison.json`
retains every trial, timestamp, hash, setting, and measurement.

Inspection mode clearly states **“Inspection only — model inference was not run.”**
Answers, generation times, visual tokens, and GPU memory are `null` with reasons.
Only genuine decode, sampling, evidence, and pipeline measurements are recorded.

The generator refuses to overwrite existing assets. Use a new output path for a
second fixture. Generated videos, truth manifests, reports, weights, and caches
are ignored by Git.

## Run Qwen3-VL locally

The model is `Qwen/Qwen3-VL-2B-Instruct`. Model dependencies are optional and are
loaded only when `--backend qwen3vl` is explicitly requested. Initial weights and
processor downloads require network access and several GB of disk space.
Decoding, processing, and inference run locally. No inference service receives
the video. Reports embed its selected frames; share them intentionally.

The verified integration uses **Transformers 4.57.6**, **PyTorch 2.8.0**,
**torchvision 0.23.0**, and **hf-xet 1.7.1** for model downloads. Install a
PyTorch wheel appropriate for your hardware from
the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then
install the optional extra. The reference CUDA setup uses CUDA 12.8 wheels:

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[qwen]"
vidfit compare --video examples/generated/brief-error.mp4 --question "What error message appeared?" --frame-budgets 4 8 16 --sampling uniform --backend qwen3vl --device cuda:0 --dtype float16 --max-pixels 262144 --max-new-tokens 128 --warmup 1 --repeats 3 --fixture-manifest examples/generated/brief-error.truth.json --output-dir reports/qwen3vl
```

On PowerShell, use `.\.venv\Scripts\python.exe` and
`.\.venv\Scripts\vidfit.exe` if the environment is not activated. Optionally place
downloads in the ignored project cache before the real run:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
```

```sh
export HF_HOME="$PWD/.cache/huggingface"
```

The default device is explicitly `cuda:0`, with `float16` weights and SDPA
attention. There is no automatic device, model, precision, or budget fallback.
CPU inference must be requested with `--device cpu --dtype float32`; it is not
the reference performance path. Apple MPS, quantization, Flash Attention, and
multi-GPU placement are outside scope. GPU memory depends on the frame budget,
resolution, token limit, and runtime; see measured hardware in the validation
record rather than treating a single GPU as a minimum requirement.

`--max-pixels` is a per-frame processed pixel cap. The native video processor
rounds spatial dimensions and combines frames into temporal pairs. VidFit
records effective dimensions and tensor shapes, verifies visual-token counts,
disables secondary sampling, and explicitly records final-frame padding for odd
budgets. Selected source previews are distinguished from normalized processed
inputs. Qwen currently requires constant-frame-rate sources and at least two
selected frames; inspection can also handle VFR and one-frame budgets.

Generation uses greedy decoding (`do_sample=False`, `num_beams=1`) and a default
128-new-token limit. The model and processor resolve to the same commit SHA,
which is recorded in the report. Use `--revision <resolved-sha>` when reproducing
a run. Greedy decoding does not ensure identical floating-point behavior across
different hardware or drivers.

## Sampling and the synthetic event

Targets are equal-bin midpoints:

```text
target_timestamp(i) = (i + 0.5) * duration / frame_budget
```

| Budget | Target examples for 60 seconds | Fixture event evidence |
| --- | --- | --- |
| 4 | 7.5, 22.5, 37.5, 52.5 s | Absent |
| 8 | 3.75, 11.25, 18.75, 26.25, … s | Absent |
| 16 | 1.875, 5.625, …, **24.375**, … s | Present |

Each target maps to the nearest actual decoded presentation timestamp, choosing
the earlier frame for a tie. Requested timestamps are not labeled as exact
decoded timestamps. Reports retain both, plus original PTS, time base,
presentation-order frame index, and SHA-256 of decoded RGB pixels. Targets are
relative to the first presentation timestamp.

Invalid budgets, empty/unreadable files, and insufficient distinct source frames
produce explicit failures. No source frame is silently duplicated to meet the
sampling budget. Native processor padding is a separate, reported operation.

`--fixture-manifest` optionally adds fixture-based evidence coverage after
verifying its video SHA-256. The truth manifest is separate from model inputs;
its error text never enters the question, input filename, or processor metadata.
Coverage is not a general accuracy metric or a claim that 16 frames must improve
model accuracy.

## Python API

```python
from vidfit import VideoAnalyzer

analyzer = VideoAnalyzer(backend="inspect")
results = analyzer.compare(
    video="examples/generated/brief-error.mp4",
    question="What error message appeared?",
    frame_budgets=[4, 8, 16],
    sampling="uniform",
    warmup=1,
    repeats=3,
    fixture_manifest="examples/generated/brief-error.truth.json",
)
results.export_html("reports/comparison.html")
results.export_json("reports/comparison.json")
assert results.complete
```

For inference, construct `VideoAnalyzer(model="Qwen/Qwen3-VL-2B-Instruct",
backend="qwen3vl", device="cuda:0", dtype="float16", max_pixels=262144)`.
`results.configurations` contains typed dataclass records; `results.to_dict()`
returns the JSON-ready experiment. Importing `VideoAnalyzer` does not import
PyTorch or download anything.

The CLI preserves successful configurations when another fails and exports
partial results. Exit status is 0 for a complete experiment, 1 for incomplete
configurations, and 2 for invalid top-level input. Use `vidfit compare --help`
for all options. Treat a partial summary as incomplete.

## Profiling and reproducibility

By default, a comparison records one separate first invocation, then one warmup
and three measured repetitions per configuration. Warmup and first-invocation
results are retained but excluded from medians. Even `--warmup 0` retains the
first invocation separately. Summaries include min, max, count, and sample
standard deviation. Every measured trial has its own answer and costs.

Each trial repeats streaming decode, selection, PNG capture, hashing,
preprocessing, transfer, generation, and output text decoding. Pipeline latency
covers those stages through final synchronization. Model loading is separate;
the initial metadata pass, initial gallery extraction, and report writing are
outside trial latency. The model and processor are reused across budgets;
filesystem and CUDA allocator caches can be warm. The first call for a new
budget is not described as a fresh-process cold start.

CUDA stage boundaries are synchronized and peak memory counters reset before
each trial. Allocated and reserved memory measure different PyTorch allocator
quantities, include resident model weights, and do not measure total device use.
Visual-token counts come from actual native processed IDs checked against the
video grid, never an estimate from raw frame count. Missing metrics remain null
with reasons. All timing fields use seconds and JSON memory fields use bytes.

See [the measurement contract](docs/measurement.md) for exact boundaries, native
processor checks, schema semantics, caching, and limitations. The integration
follows the [official Qwen3-VL implementation](https://github.com/QwenLM/Qwen3-VL)
and [PyTorch CUDA timing/memory guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Development and tests

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The CI workflow is configured to run CPU tests, lint, wheel builds, and
installed-wheel API/CLI smoke checks on Windows and Linux with Python 3.10 and 3.12. It does not download weights or
install model dependencies. macOS setup is documented but is not in the CI matrix.

Opt into the real-model integration test only after installing `[qwen]`:

```powershell
$env:VIDFIT_RUN_MODEL_TEST = "1"
.\.venv\Scripts\python.exe -m pytest -q -m model
```

```sh
VIDFIT_RUN_MODEL_TEST=1 python -m pytest -q -m model
```

This test runs real inference on all three budgets and validates input accounting
without asserting that any answer must be correct. It uses a shorter 64-token
limit and one measured repetition; use the CLI command above for the full
default experiment. Ordinary processor doubles test bookkeeping and rejection
paths only; they do not establish model correctness.

## Architecture and limits

`VideoAnalyzer` coordinates a small pipeline. `core/video.py` streams PyAV frames
and captures evidence; `core/sampling.py` defines midpoint targets;
`core/schemas.py` holds results; `core/profiling.py` summarizes trials.
`backends/inspection.py` records unavailable inference metrics;
`backends/qwen3vl.py` validates native video inputs and runs the model.
`evaluation/reports.py` produces escaped, offline HTML; `demo.py` creates the fixture.

This is a CLI proof of concept. Large videos require a metadata decode plus
fresh decodes for each trial. Reports embed full PNG evidence and can become
large. No audio, batch concurrency, automatic OOM recovery, cross-version
processor compatibility, or general answer-quality evaluation is included.
Qwen VFR support requires a verified native timestamp representation and remains
deferred. The synthetic demo is reproducible within the same dependency/FFmpeg
environment; byte-identical encoding across library versions is not promised.

Future work: Gradio/React interfaces, scene-change and manual sampling, budget
recommendations, additional backends, streaming, and learned frame selection.
These are not implemented in this milestone.

## Contributing and licensing

See [CONTRIBUTING.md](CONTRIBUTING.md). VidFit source is licensed under
[MIT](LICENSE). Model weights are separate: the
[Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
identifies their Apache-2.0 license. Downloading weights does not change their
license or make them part of VidFit's MIT source distribution.
