# Switching models

Choose a preset or enter a Hugging Face model ID in the interface, then preview
and run. The CLI uses the same routing through `--backend auto --model MODEL_ID`.
Changing models needs no source edit or interface restart. Dispatch uses the
checkpoint configuration's `model_type`, not its name. Inspection still needs no
model libraries.

| Dependency extra | Transformers | Supported model types |
| --- | --- | --- |
| `qwen` | 4.57.6 | Dense Qwen3-VL (`qwen3_vl`), original baseline |
| `qwen35` | 5.5.4 | Dense Qwen3-VL and dense Qwen3.5 (`qwen3_5`) |
| `vlm` | 5.5.4 | Both Qwen adapters plus compatible Transformers image-chat models |

All profiles pin PyTorch 2.8.0, torchvision 0.23.0 and hf-xet 1.7.1. `vlm` also
includes `num2words`, required by the SmolVLM processor. Other checkpoints may need
their own optional processor dependencies. Gradio 6.0.2 supports both Hub dependency
ranges. Extras are additive: choose one model profile per environment, such as
`ui,vlm`. Never combine legacy `qwen` with `qwen35` or `vlm`: their Transformers pins
conflict. Missing dependencies or incompatible models fail without substituting
another checkpoint.

| Backend | Behavior |
| --- | --- |
| `auto` | Native video for dense Qwen3-VL/Qwen3.5; generic timestamped images for other compatible checkpoints |
| `qwen` / `qwen3vl` | Requires a supported dense Qwen native-video checkpoint; the old name remains an alias |
| `transformers` | Explicitly uses the generic timestamped-image adapter |
| `inspect` | Decodes evidence without model answers |

Custom IDs are not restricted to the presets. Generic compatibility requires an
installed native `AutoModelForImageTextToText` class, a usable image-chat template
and processor, and a verifiable image tensor layout. Remote Python code is not
enabled. Gated repositories require access through your existing Hugging Face setup.
A preset is a convenience, not certification that every revision has been tested.
`vidfit models` lists the catalog without downloads; `vidfit models --json` returns
the same presets as JSON.

## Windows setup alongside the original environment

From the project directory, use a separate environment to keep old measurements
reproducible. These commands use the tested CUDA 12.8 wheel combination:

```powershell
py -3.12 -m venv .local/venv-vlm
.\.local\venv-vlm\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.local\venv-vlm\Scripts\python.exe -m pip install -e ".[ui,vlm]"
.\.local\venv-vlm\Scripts\python.exe -m pip check
$env:HF_HOME = "$PWD/.cache/huggingface"
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
.\.local\venv-vlm\Scripts\vidfit.exe ui --backend auto
```

Use BF16 explicitly for Qwen3.5 and SmolVLM2. GPU support is checked on the requested
device. There is no silent precision, quantization, CPU
offload or model fallback. Larger checkpoints may exceed available GPU memory.
CPU inference requires explicit `--device cpu --dtype float32` and may be slow.
On Linux, create/activate a Python environment, install the appropriate PyTorch
CUDA wheels, then install `.[ui,vlm]` and use the same `vidfit` arguments. The CLI
keeps its existing FP16 default: changing only `--model` does not change precision.

The first inference downloads uncached checkpoint weights. The setup above clears
offline variables so a new model can be fetched. Once cached, set `HF_HUB_OFFLINE`
and `TRANSFORMERS_OFFLINE` to `1` for offline use. Both environments can share the
cache. Use `--revision <commit>` for a fixed checkpoint; successful inference
reports record the actual resolved model and processor commit.

## CLI and Python

Activate the chosen environment, or use its full executable path. Switching back
to Qwen3-VL works in the modern profile; use the original environment when you need
the exact old processor version as well.

```powershell
vidfit ui --backend auto --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct --dtype bfloat16
vidfit compare --backend auto --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct --dtype bfloat16 --video examples/generated/evaluation/clip-01.mp4 --question "What error message appeared?" --sampling visual_change --frame-budgets 4 --execution-mode single --output-dir reports/smolvlm-check
vidfit experiment --backend auto --model Qwen/Qwen3.5-2B --dtype bfloat16 --collection examples/evaluation/starter.json --video-dir examples/generated/evaluation --repeats 2 --max-new-tokens 64 --output-dir reports/qwen35-experiment
vidfit trace --model Qwen/Qwen3.5-2B --video examples/generated/evaluation/clip-01.mp4 --frames 4 --output-dir reports/qwen35-trace
```

```python
from vidfit import SelectionSpec, VideoAnalyzer

analyzer = VideoAnalyzer(
    backend="auto",
    model="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    dtype="bfloat16",
    enable_thinking=False,
    max_new_tokens=64,
)
result = analyzer.compare(
    "examples/generated/evaluation/clip-01.mp4",
    "What error message appeared?",
    selections=[SelectionSpec.visual_change(4), SelectionSpec.uniform(16)],
    execution_mode="single",
)
result.export_html("reports/smolvlm-comparison.html")
```

The UI offers presets and a custom model ID. A preset supplies the adapter,
precision and pixel budget, and resets revision to `main` and thinking to off.
Custom selection leaves explicit controls available. Model-setting changes clear
the preview and downloads; preview again before running. Each result retains its
own model label, revision and settings after a switch. `qwen3vl` remains a native
Qwen backend alias for existing scripts.

SmolVLM2 500M and LLaVA-OneVision 0.5B completed real 4- and 16-frame inference
through the generic path. Those initial smoke-test answers were unreviewed; a later
[four-model selection comparison](validation.md#four-model-selection-comparison-and-llava-diagnosis-2026-10-31)
evaluated ten questions with explicit agent judgments. The SmolVLM2 2.2B preset has
not been locally verified.

Start with four frames for an initial check. The earlier LLaVA 16-frame smoke
request took about 143 seconds and reported about 19.47 GiB peak allocation on the
reference 16 GB GPU. Subsequent diagnostics identified avoidable costs in the
timestamped-image representation and attention dispatch: on one saved 16-frame
example, changing dispatch alone reduced generation from 108.80 to 2.33 seconds
with the identical answer. This was an isolated diagnostic, not an application
optimization or a full benchmark rerun. The current adapter still uses timestamped
images and the runtime's default attention dispatch. See the linked validation
section for timing boundaries, native-video results and limits.

## Native video and timestamped images

The Qwen adapter supplies selected frames to the native video processor without
resampling. It checks temporal pairing, timestamp tokens, effective resolution
and visual-token counts. It requires constant-rate video and at least two distinct
frames. Native sampling traces remain available only for supported Qwen processors.

The generic adapter submits chronological images with source-time labels and the
question. It does not claim native-video equivalence, even when the checkpoint
also supports video. It can accept variable-rate sources and a single frame when
the chosen processor supports them. Original RGB identity, resized copies, source
times and processor overrides are recorded.

`--max-pixels` bounds allocated processor image views per selected source frame,
including crops and padding, checked against actual tensors before generation.
Small inputs may still be upscaled: SmolVLM2 500M requires 512×512 pixels per image,
so its minimum is 262144. VidFit disables SmolVLM image splitting while preserving
the fixed encoder resolution. A smaller budget or an unrecognized tensor layout
fails explicitly. This cap does not bound total GPU memory or preprocessing memory.

Generic `visual_tokens` is unavailable with an explanation, because placeholder
counts are not a verified cross-model visual-token measure. Reports retain
input/output tokens, image views, tensor shapes, timing and GPU allocator peaks.
Read `input_representation` and validation scope before comparing costs.

## Thinking and measurement compatibility

The default is `--no-thinking` / `enable_thinking=False`. Use `--thinking` only for
an intentional reasoning run and allow a suitable output budget. The adapter
checks whether the checkpoint's chat template responds to the flag. When it does,
reports record requested and effective template modes. When it does not, default
False retains the checkpoint's native behavior and effective mode is unknown;
requesting True fails clearly. Generated reasoning is not silently discarded.

On the Qwen path, frame pixels, order, hashes, temporal pairing, timestamps, actual
visual tokens and effective resolution are checked against processor output. Native
trace provenance records the installed implementation rather than claiming a fixed
sampling algorithm across all versions. Model class, model type, Transformers
version, precision and thinking settings are exported with measurements.

Automatic routing and the Qwen adapter accept Transformers >=4.57.6,<6, with Qwen3.5
requiring >=5.2 and its native class. The exact profiles above are the intended
reproducible configurations; accepting a version is not a claim that every release
in that range has been tested. If a future processor changes its input contract,
validation fails explicitly. Update the profile in `pyproject.toml`, verify the
processor and rerun regression checks before relying on new measurements.

For model comparisons, hold source frames, question, resolution and output budget
fixed, and record representation, precision and thinking mode. Rerun Qwen3-VL on
the modern runtime to distinguish a model change from a library change. A model-loading smoke test
does not establish improved accuracy or speed; see [validation](validation.md).

The existing opt-in real-model tests check the Qwen native-video contract, including
visual tokens and odd-frame padding. Their model override is for compatible Qwen
checkpoints, not arbitrary image-chat models. For Qwen3.5, after installing the
`dev` extra and caching the model:

```powershell
$env:VIDFIT_RUN_MODEL_TEST = "1"
$env:VIDFIT_TEST_MODEL = "Qwen/Qwen3.5-2B"
$env:VIDFIT_TEST_DTYPE = "bfloat16"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest -q -m model
```

Optional `VIDFIT_TEST_REVISION` pins the tested checkpoint revision. Without those
model overrides the tests use the original Qwen3-VL-2B-Instruct / float16 baseline.

The separate generic regression test runs one- and two-frame selections through
one loaded model. In the modern environment, with the `dev` extra and SmolVLM2
500M already cached:

```powershell
$env:VIDFIT_RUN_VLM_TEST = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest -q tests/test_real_transformers_vlm.py
```

Its defaults are SmolVLM2 500M, BF16 and `cuda:0`. Optional
`VIDFIT_TEST_VLM_MODEL`, `VIDFIT_TEST_VLM_DTYPE`, `VIDFIT_TEST_VLM_REVISION` and
`VIDFIT_TEST_VLM_MAX_PIXELS` select another compatible generic checkpoint and its
settings. These checks validate inference, inputs and accounting, not accuracy.
