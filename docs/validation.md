# Validation record

This file records actual validation of the proof of concept. Generated reports,
videos, model weights, and caches remain local and are excluded from Git.

## Reference environment

- Windows, x86-64; Python 3.12.14 in the project-local virtual environment.
- PyAV 16.0.1, Pillow 12.0.0, NumPy 2.2.6.
- Transformers 4.57.6; PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128.
- NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB; driver 621.01.

## Checks performed

- Editable installation and public import succeeded.
- 41 ordinary tests passed; the opt-in real-model test was skipped in that run.
- Ruff lint and format checks passed.
- Generated the 60-second 480-frame synthetic video twice; bytes matched in the
  reference environment.
- Actual decoded pixels and timestamps confirmed absent event evidence for 4
  and 8 frames and present evidence for 16 frames.
- Default inspection comparison exported self-contained HTML and structured JSON
  with every trial and explicit unavailable metrics.
- Browser preview was attempted, but the available browser's URL policy rejected
  local `file:` URLs. Browser visual QA remains unverified.

## Real Qwen comparison

Run `4954f0aa-66f3-4a3a-bf2e-0c3b77797464` at `2026-10-30T05:50:28.297543+00:00`. Model and processor resolved
to `89644892e4d85e24eaac8bacfd4f463576704203`. Video SHA-256:
`b2e21e305a045592a52a345a26c60705c36d05d73b3944718ba2c271b9d29e33`.

Ran `cuda:0`, float16, SDPA, greedy decoding, 128-new-token limit,
262144-pixel per-frame cap, one warmup and three measured trials per budget,
plus one separately recorded first invocation (13 invocations total).
All configurations completed with matching trial/gallery hashes and validated
native temporal metadata. Effective processing was 640×352 per frame,
with no temporal padding for these even budgets.

| Frames | Event evidence | Visual tokens | Median generation (s) | Median pipeline (s) | Peak allocated (MiB) | Peak reserved (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | Absent | 440 | 4.945 | 5.440 | 4193.9 | 4260.0 |
| 8 | Absent | 880 | 3.474 | 4.177 | 4376.5 | 4516.0 |
| 16 | Present | 1760 | 2.938 | 3.824 | 4921.4 | 5164.0 |

All three measured answers agreed within each configuration:

- **4 frames:** There was no error message. The screen showed a progress bar indicating that the data synchronization process was in progress.
- **8 frames:** The error message that appeared was "Processing records...".
- **16 frames:** Error E104: Connection timed out.

These are observations from one synthetic fixture, not accuracy or speedup
claims. Output lengths differ (23, 12, and 11 generated tokens), so generation
latency does not isolate visual-input cost. Peak figures are PyTorch allocator
measurements, not total GPU use or minimum hardware requirements. The model was
loaded from the local cache in 7.299 seconds; this is not a cold-start benchmark.

The actual native processor was also exercised on 3 selected frames, which
produced 4 input frames through explicit final-frame padding, at 672×384 and
504 verified visual tokens. The first real attempt exposed metadata tensor
conversion incompatibility; it was fixed without changing the native video
representation. Failed-attempt reports remain local under `reports/attempts/`.

Downloads succeeded with `hf-xet==1.7.1`; the initial regular HTTP download
stalled. The optional Qwen extra includes that tested downloader. The final
comparison ran with Hugging Face offline mode enabled.

## Installation and integration checks

- Source and wheel distributions built successfully, with only deliberate
  source/documentation/test files in the archives. No generated data or weights
  were included.
- The installed wheel passed a public-API import smoke check outside the editable
  source path.
- The opt-in real-model pytest integration test passed on all 4/8/16 budgets
  (`1 passed, 41 deselected`), using cached weights in offline mode.

## Clean CPU environment

A second, project-local virtual environment installed the wheel without model
dependencies: Python 3.12.14, PyAV 16.1.0, Pillow 12.3.0, NumPy 2.5.4,
pytest 9.2.0, and Ruff 0.17.3. PyTorch and Transformers were confirmed absent.
All 41 ordinary tests passed (one opt-in test skipped); lint and formatting
checks passed. Its installed CLI generated the default inspection comparison
for all three budgets successfully.

The FFmpeg libraries in both environments were libavutil 60.8.100,
libavcodec 62.11.100, libavformat 62.3.100, libavdevice 62.1.100,
libavfilter 11.4.100, libswscale 9.1.100, and libswresample 6.1.100.

## Publication status

Published to the public repository [xyhsy11/vidfit](https://github.com/xyhsy11/vidfit)
on branch `main`, with branch tracking configured. The pushed branch was
verified against the intended local commit. Only deliberate source,
documentation, and tests were committed; generated videos, reports, model
weights, caches, and local environments remain excluded. No force push or
unrelated repository update was performed.

[GitHub Actions run 41011710756](https://github.com/xyhsy11/vidfit/actions/runs/41011710756)
passed for source commit `a458bc66301d58752a45252c59c11804d648833c` across
Windows and Linux with Python 3.10 and 3.12. All four jobs passed installation,
lint, formatting, ordinary tests, distribution builds, and installed-wheel
API/CLI smoke checks without model downloads or GPU access. This publication
record update changes documentation only.


## v0.2 local interface release (2026-10-30)

The v0.1 observations above are historical. v0.2 was developed from clean `main`
`dfb5180` on `codex/vidfit-v0.2`. The reference GPU and Qwen revision are unchanged.
Gradio 6.0.2 / Gradio Client 2.0.1 / Hugging Face Hub 0.36.2 were installed together
with the pinned Qwen extra; `pip check` passed. Gradio 5 conflicted with Pillow 12,
and a newer Gradio release required a Hub version incompatible with Transformers
4.57.6, so the UI extra explicitly constrains the verified combination.

- 76 ordinary tests passed; two real-model tests and one optional UI component test
  were skipped in that CPU run. The suite includes a spawned inspection worker.
- The separate opt-in real Qwen checks passed: **2 passed**. These exercise legacy
  4/8/16-frame benchmarking, manual single mode, trial/gallery identity, native
  temporal accounting and odd-count final-frame padding. No correctness assertion
  was imposed on generated answers.
- The optional real Gradio component check passed and confirms uploads reach the
  core without invoking a system FFmpeg executable. Gradio remains optional for
  the CPU-only installation.
- Ruff lint and format checks passed, including README Python examples.
- Five evaluation clips were prepared and checksum-verified; ten inspection cases
  completed with null model answers/costs. References have agent visual review,
  not human review. Sources, licenses and reviewed timestamps are recorded.
- The controlled uniform/manual Qwen comparison and the existing downloaded sample
  completed. [Actual answers and measurements](v0.2-results.md) separate evidence
  coverage, answer judgment, tokens, allocator memory, and latency/output length.

The managed shell could not access pytest's existing default temporary/cache
folders, so checks used `-p no:cacheprovider --basetemp .local/<unique-run-name>`.
The initial permissions errors were rerun successfully with those workspace paths.

Browser QA found and fixed an unwanted Gradio audio-stripping conversion that
required a system FFmpeg executable. Callback decoding/imports also suffered
severe contention with the UI event-loop process. Jobs now use one persistent
local process, with runtime imports before request handling. No warmup inference
is hidden in startup. Two diagnostic UI attempts were interrupted during setup;
they produced no model answer and are not reported as successful inference runs.


Browser verification used the in-app browser against loopback HTTP. It exercised
video upload and playback, uniform preview, real Qwen Answer once, invalid manual
timestamps and error text, stale-preview invalidation, corrected manual preview,
real manual rerun, previous/current answers and timestamped galleries. The
[README screenshot](assets/interface.jpg) is the actual running interface with
synthetic inputs and real model outputs, not a mockup. Those UI answers remain
Not evaluated; the explicit agent judgments in the controlled comparison are
separate. The offline report was also rendered and visually reviewed, including
enlarging the event frame, source PTS/hash metadata and advanced measurements.


Compare settings also completed in the browser: one uniform and one manual
configuration, single mode, exactly one measured trial each, native processor
validation true, coverage false/true and answers unreviewed in the downloaded
payload. Both HTML/JSON export URLs returned HTTP 200 with complete content;
the HTML retained embedded images and no external assets.

Validation limits: the in-app browser did not expose a completed download event,
and its file-download capability timed out. The native Chrome fallback was stopped
by Computer Use because it could not reliably determine the current browser URL.
Native download completion was therefore **not verified**. Browser automation ended
at that point. The final responsive-width/card-minimum and timestamp-caption polish
passed component/tests but was not visually rechecked. The README screenshot shows
the real functional UI immediately before that cosmetic polish. No claim is made
that those blocked browser checks passed. Local UI/GPU checks were on Windows;
Linux/macOS UI and other GPUs were not exercised locally.


Final package checks: `python -m build` built the source distribution and wheel.
Archive inspection found no videos, weights, caches or generated reports; the sdist
includes the annotation collection, preparation script and actual UI screenshot.
The installed v0.2 wheel passed API and CLI single/manual inspection smoke checks
outside the source path in the clean CPU environment, with Torch, Transformers
and Gradio confirmed absent. UI help remains available there without UI imports.
`pip check` passed. The final source UI restarted successfully on loopback with
its width setting verified through its configuration endpoint; no inference is
performed on launch.


v0.2 publication: implementation commit
`25b4525c251618d52f332e6b77d9dac36e214fad` was pushed to
`codex/vidfit-v0.2` and the remote branch was verified against that exact commit.
[PR #1](https://github.com/xyhsy11/vidfit/pull/1) targets `main`; it was opened through
the configured GitHub CLI after the connector returned an integration-permission
403. No default-branch merge, force push or unrelated history rewrite was performed.
The remaining checklist/publication record is a separate documentation commit.
Remote CI results, if still running, are not counted as passed local validation.

## Native sampling trace (2026-10-31)

Added an independent sampling trace to the UI, API and CLI. Local validation used
Windows / Python 3.12, Transformers 4.57.6 and the cached Qwen3-VL-2B-Instruct
configuration, with Hugging Face offline mode enabled. No model weights or GPU
inference were needed for these checks.

- 117 tests passed with UI and native-processor checks enabled; two GPU model
  tests were deselected. This includes ordinary sampling/processor regression
  tests, optional-dependency isolation, trace exports, UI invalidation and capture
  bounds. Ruff lint and formatting checks passed.
- Actual native sampling on the 480-frame, 8-FPS demo returned indices
  `[0, 160, 319, 479]` at an explicit four-frame count. VidFit returned
  `[60, 180, 300, 420]`. Default 2-FPS sampling returned 120 frames; explicit
  0.5 FPS returned 30. Native short-source bounds, maximum-frame limits and
  ties-to-even index rounding were checked against the installed processor.
- An independent review found a decimal-rounding edge case in temporal-pair
  arithmetic. The trace and existing inference timestamp check now divide each
  index by FPS before averaging, matching upstream operation order. Regression
  cases include 30 FPS and 30000/1001 FPS.
- In the live local Gradio interface, the real processor produced matched-count
  and custom-FPS traces with thumbnails, source-time tables and separate export
  links. Changing native settings cleared the old trace and downloads. The
  narrow in-app browser layout was visually inspected.
- The browser's download helper timed out, so browser-managed download completion
  is not claimed. The displayed JSON download URL returned HTTP 200 and its actual
  payload was checked for the 30-frame native / four-frame VidFit result. CLI
  HTML/JSON exports were also generated under `reports/native-trace`.
- Source distribution and wheel builds passed using Hatchling in an isolated
  project-local tools directory. The initial environment lacked Hatchling;
  downloaded build tools were used only for packaging, not the Gradio runtime.

To repeat the optional processor-only tests with an existing cache:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:VIDFIT_RUN_NATIVE_TRACE_TEST = "1"
.\.venv\Scripts\python.exe -m pytest -q -m native_trace
```

## Visual-change selector and real inference experiment (2026-10-31)

Added a fixed RGB-change selector to the API, CLI and UI, plus an experiment runner
for visual-change 4 versus uniform midpoint 4/16. The new runner measures complete
warm-model requests including metadata probing, all selector passes, source capture,
Qwen preprocessing, transfer, generation and final synchronization. Existing
`compare` measurements retain their documented boundaries.

- **150 tests passed**, with optional Gradio and actual native-processor checks
  enabled; two separate model pytest cases were deselected. New checks cover static
  bins, flashes, ties, exact source identity, bounded scan behavior, CLI/UI controls,
  timer accounting, reference isolation, explicit judgment binding, failed-trial
  denominators, optional reference fields and report timing definitions.
- The real GPU experiment completed **60 measured Qwen inferences plus two recorded
  warmups**, without failures. All 30 repeat pairs had identical selected evidence,
  answers and token counts. Native processor validation, source identities, embedded
  PNG hashes and timing aggregation were independently audited.
- A separate method-blinded agent reviewed all 21 unique question/answer pairs.
  Reviewed summaries preserve original unreviewed JSON and distinguish strict
  success, partial credit, unsupported-question abstention and evidence coverage.
  [Results and limitations](visual-change-results.md) include actual accuracy,
  preprocessing and complete request times, model revision and reproduction steps.
- Ruff lint, formatting, diff whitespace and source/wheel packaging checks passed.
  The new UI control passed component tests; no additional live browser validation
  of this selector is claimed. All model work used the existing offline cache on
  the reference RTX 3080 Laptop GPU.

Generated run data remains local under `reports/visual-change-20261031/`; measured
source snapshots match the manifest fingerprints. The subsequent optional-reference
default fix changes neither the selected inputs nor inference/timing behavior for
this collection. No selector tuning followed answer review. This is a small known
starter-set pilot with agent judgments, not an independent human accuracy benchmark.

## Configurable Qwen runtimes and model switching (2026-10-31)

Added the `qwen35` dependency profile (Transformers 5.5.4) alongside the original
`qwen` profile (4.57.6). The modern runtime selects dense Qwen3-VL or dense Qwen3.5
from the checkpoint configuration. `--backend qwen3vl` remains an alias for
`--backend qwen`. Existing frame/hash, pixel, temporal, token and timestamp checks
remain enforced. Thinking requests and effective template control are recorded.

- Original environment: **195 tests passed**, including optional Gradio and native
  processor checks; two model tests were deselected in that run. The separate
  original-runtime GPU regression run also passed: **2 model tests**, covering
  4/8/16-frame inference and manual/odd-count padding.
- Modern environment: **90 focused tests passed**, including actual cached native
  processors, UI components, model routing, missing/unsupported model errors,
  thinking control and existing processor integrity checks.
- The modern runtime used an isolated project-local dependency directory with the
  existing PyTorch 2.8.0+cu128, torchvision 0.23.0 and Gradio 6.0.2. Its Transformers
  version was 5.5.4 and Hugging Face Hub was 1.33.0. `pip check` passed. The documented
  `pip install -e ".[ui,qwen35]"` command resolved successfully in a normal isolated-build
  dry run. This was not a clean, separately installed modern virtual environment;
  the original `.venv` remains on Transformers 4.57.6.
- Actual Qwen3.5-2B weights were downloaded and **four real inference requests passed**
  on the reference 16 GB GPU: visual-change 4, uniform 4, uniform 16, and manual 3.
  Revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`; BF16, SDPA, 64 output-token
  limit, thinking requested/effective false, processor controls verified. All inputs
  passed validation; the three-frame input padded to four as expected. Peak allocated
  memory across these requests was approximately 4.22-4.57 GiB. Qwen3.5 used the
  available PyTorch implementation because optional fast linear-attention/convolution
  extensions were absent. No speed comparison is claimed from these smoke checks.
- The original Qwen3-VL-2B-Instruct checkpoint also completed **three real requests
  under the modern runtime**, using uniform 4/16 and manual 3 in float16. Inputs,
  padding and output accounting passed validation. Its fixed chat template records
  effective thinking mode as unknown rather than claiming the flag changed it.
- Qwen3.5's processor-only CLI trace exported successfully in offline mode, using
  the actual installed sampler, revision and source provenance. It loaded no weights.
- Ruff lint, formatting, whitespace and source/wheel build checks passed. UI checks
  here were component checks; no new browser interaction claim is made.

Smoke reports are local under `reports/model-switching/qwen35-2b/`,
`reports/model-switching/qwen3vl-modern/` and `reports/model-switching/qwen35-trace/`.
In the single error-screen smoke example, Qwen3.5 answered the full error message
correctly for visual-change 4, uniform 16 and manual 3, while uniform 4 denied an
error. The exports remain unreviewed; these are model integration observations,
not a rerun of the ten-question accuracy study. See the
[model switching guide](model-switching.md) to reproduce setup and model selection.

## Multiple model families and UI selection (2026-10-31)

Added the `vlm` dependency profile, configuration-based `auto` routing, an explicit
Transformers image-chat adapter, a dependency-free `vidfit models` catalog, and
per-request model controls in the interface. Native dense Qwen video behavior is
preserved. Other compatible checkpoints receive chronological timestamped images;
this does not claim native-video equivalence or support for every Hugging Face model.

- **242 tests passed and 2 were skipped** in the full CPU/UI/native-processor suite
  on the original runtime. Checks cover optional-dependency isolation, routing,
  generic image budgets/layouts, model-setting propagation, stale-preview rejection,
  preset defaults and actual Gradio component wiring. A subsequently added generic
  real-model test is opt-in and skips by default; it was not included in that count.
- The final **modern-runtime suite passed 243 tests with 2 skipped**, with
  `VIDFIT_RUN_UI_TEST=1`, `VIDFIT_RUN_NATIVE_TRACE_TEST=1` and
  `VIDFIT_RUN_VLM_TEST=1`. This includes actual Gradio/native-processor checks and
  real SmolVLM2 500M inference on one- and two-frame selections using one loaded
  model. Source identity, image budgets, null visual-token reasons, generation
  outputs and timing accounting passed. The two Qwen-specific GPU pytest cases
  were skipped in this run; their earlier validation is recorded above.
- Real CLI inference on Transformers 5.5.4, PyTorch 2.8.0+cu128 and the reference
  Windows GPU completed **four requests across two non-Qwen model families**:
  uniform 4 and uniform 16 for each model. Both configurations used single-run
  mode, with one measured invocation and no warmups. All inputs passed source
  identity, image-layout and processed-pixel checks; all answers remain unreviewed.
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` loaded
  `SmolVLMForConditionalGeneration` at revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, using BF16 and SDPA. The processor
  produced one 512×512 view per source frame with image splitting disabled.
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` loaded
  `LlavaOnevisionForConditionalGeneration` at revision
  `74dd0bf867a4cda7950c17663794267c60cf4b40`, using FP16 and SDPA. Nested multi-image
  inputs produced one 384×384 view per source frame. The 16-frame request had
  11967 input tokens, approximately 143.35 seconds pipeline latency and a reported
  peak CUDA allocation of 20,900,664,320 bytes (about 19.47 GiB), exceeding the
  device's nominal 16 GB capacity. Completion is not evidence that this setting
  fits entirely in physical GPU memory or is practical for interactive use.
- Both generic models leave visual tokens null with a reason. Their fixed chat
  templates record requested thinking off and effective mode unknown. Reports
  preserve timestamped-image representation, actual tensor/view sizes, revisions,
  precision, timings and allocator peaks.
- Live browser checks on the local interface verified Qwen-to-Smol preset changes,
  automatic routing/BF16 defaults, disabled non-Qwen native tracing and a four-frame
  preview. Switching back to Qwen3.5 cleared that preview and re-enabled native
  tracing. These checks establish the controls and invalidation behavior; they do
  not establish answer accuracy.
- A real spawned-worker check switched from SmolVLM2 500M to Qwen3-VL 2B in the
  same local worker. Both four-frame requests completed with a 24-token output
  limit. A stale preview from the other model was rejected, source frames matched,
  the previous result's model card was preserved, and request settings did not
  mutate the baseline analyzer. The offline summary is under
  `reports/multi-model/ui-worker-switch/worker-switch-summary.json`.
- `pip check` found no broken requirements in the modern runtime. The documented
  `pip install -e ".[ui,vlm]"` resolved in a dry run, source/wheel packaging passed,
  and Ruff lint/format checks passed for 57 Python files. As above, the modern
  runtime uses a project-local dependency directory alongside the existing
  PyTorch environment; a fresh separate virtual-environment installation is not
  claimed.

Reports are local under `reports/multi-model/smolvlm2-500m/` and
`reports/multi-model/llava-onevision-05b/`. The SmolVLM2 2.2B preset and arbitrary
custom checkpoints have not been verified. These are integration smoke checks on
one synthetic clip, not an accuracy evaluation, a fair speed comparison or a GPU
capacity recommendation. The new integration checks ran eight real inference
requests in total: four CLI requests, two worker-switch requests and two requests
inside the generic pytest case. They do not rerun the ten-question accuracy study.

## Four-model selection comparison and LLaVA diagnosis (2026-10-31)

A subsequent campaign completed the same five-clip, ten-question starter evaluation
with Qwen3-VL 2B, Qwen3.5 2B, SmolVLM2 500M and LLaVA-OneVision 0.5B. Each model
used visual-change 4, uniform midpoint 4 and uniform midpoint 16, with two measured
repeats: **240 measured requests plus eight recorded warmups, with no failures**.
Each selection method supplied identical source indices, timestamps and RGB hashes
across models and repeats. Answers were identical across repeats within each
model/question/method. A separate agent reviewed 77 deduplicated question/answer
pairs with model, selection method and timing hidden; explicit judgments covered
all 240 measured requests. The saved-artifact audit passed without issues.

Each table cell gives **strict successes per ten questions; mean complete request
time**. Success requires a fully correct supported answer or appropriate abstention
on an unsupported question; partial answers receive no full credit. There are seven
supported and three unsupported questions. Repeats are timing observations, not
additional independent questions.

| Model | Visual changes, 4 frames | Uniform midpoints, 4 frames | Uniform midpoints, 16 frames |
| --- | --- | --- | --- |
| Qwen3-VL 2B | 6/10; 1.927 s | 6/10; 2.072 s | 7/10; 2.777 s |
| Qwen3.5 2B | 8/10; 3.796 s | 6/10; 4.123 s | 7/10; 4.101 s |
| SmolVLM2 500M | 4/10; 2.058 s | 3/10; 1.853 s | 4/10; 2.654 s |
| LLaVA-OneVision 0.5B | 4/10; 2.590 s | 4/10; 2.359 s | 6/10; 116.719 s |

All models used Transformers 5.5.4, PyTorch 2.8.0+cu128 and the reference Windows
RTX 3080 Laptop GPU. Qwen3-VL and LLaVA used FP16; Qwen3.5 and SmolVLM2 used BF16.
Checkpoint revisions are the same pinned revisions recorded in the integration
sections above. Generation was greedy with a 64-token limit, thinking requested off,
and a requested per-frame pixel cap of 262144. Six Qwen3.5 outputs reached the token
limit, indicating possible truncation. Complete-request timing includes source
probing/hashing, decoding, selection, evidence capture, model preprocessing,
transfer, generation, text decoding and synchronization. Model loading and warmups
were recorded separately and excluded from these warm-model request times.

The Qwen models used native video, while SmolVLM2 and LLaVA used timestamped images.
Actual processed resolutions, visual compression and prompts therefore differ;
equal source-frame counts and pixel caps do not mean equal model workloads. This
is an exploratory comparison on five known clips with agent judgments, not a
held-out benchmark, independent human evaluation or a general model ranking. The
earlier Qwen-only pilot used Transformers 4.57.6 and remains a separate experiment.

### Why the LLaVA 16-frame path was slow

Across the campaign, LLaVA's mean generation time grew from 1.905 seconds with four
uniform frames to 115.761 seconds with sixteen. Mean input preparation grew only
from 0.452 to 0.956 seconds. The large delay was inside model inference.

The current generic adapter sends separate images, bypassing LLaVA's native video
pooling. An actual four-image model trace also confirmed the memory-heavy math
attention fallback with grouped-query attention; the installed PyTorch build had
no Flash attention support. Small kernel probes showed that explicitly repeated
key/value heads allowed the memory-efficient attention implementation instead.

A bounded diagnostic replayed the saved error-message example with the same sixteen
source frames and cached checkpoint. Results below compare **generation time**, not
complete requests. The original baseline comes from its two earlier campaign calls;
each alternative had two new diagnostic calls.

| Path | Mean generation | Peak allocated GPU memory | Input tokens |
| --- | --- | --- | --- |
| Original timestamped images | 108.796 s | 19.465 GiB | 11967 |
| Same images, explicit key/value repetition | 2.333 s | 2.625 GiB | 11967 |
| Native video, original attention dispatch | 3.228 s | 3.238 GiB | 3416 |

The dispatch-only alternative preserved the exact prompt, source frames and answer.
The native-video alternative retained all sixteen frames without resampling but
changed pooling and prompt structure; it also answered the error-message question
correctly. Its processor produced 3137 video placeholders, compared with 11680 image
placeholders in the original path. This is checkpoint-specific accounting, not a
new cross-model visual-token metric.

These results show avoidable integration/runtime costs, not a hard sixteen-frame
model limit. They do not establish benchmark-wide speed or accuracy after a fix.
The actual sixteen-frame dispatch-only kernel was not separately profiled, Windows
memory paging was not traced, and the alternatives were not a randomized rerun of
the full campaign. Diagnostic overrides existed only in their own process:
**the application and installed runtime remain unchanged**. Native LLaVA video
handling and verified efficient attention dispatch still need implementation and
a full matched-frame accuracy/timing rerun.

## Current interface screenshots (2026-10-31)

The README images were refreshed from the running application at commit
`79e31d243e8d938aa058f05fb896400fb07a8135`. The captures show the current model
controls, visual-change selector, real answer cards and generated native sampling
report. They are browser screenshots, with no generated or substituted answers.

- [Interface overview](assets/interface-current.jpg): two real Qwen3-VL 2B requests
  on the MIT synthetic `brief-error.mp4`, first uniform four frames, then
  visual-change four frames. The question was "What error message appeared?".
  Both used the native Qwen adapter via `auto`, FP16, a 262144-pixel cap,
  a 64-token output limit and single execution mode. The resolved checkpoint was
  `89644892e4d85e24eaac8bacfd4f463576704203`, with Transformers 5.5.4.
- [Model menu](assets/model-selection.jpg): the current preset list and custom
  model option. Listing a preset does not validate it; SmolVLM2 2.2B remains untested.
- [Native sampling export](assets/native-sampling.jpg): the actual HTML generated
  by the UI, opened separately for readability. The same-count trace compares
  VidFit source indices `[64, 192, 250, 376]` with Qwen's `[0, 160, 319, 479]`.
  The sampling-rule disclosure was collapsed for the capture. This trace loads
  processor configuration and runs sampling, with no model inference.

The demo SHA-256 is
`b2e21e305a045592a52a345a26c60705c36d05d73b3944718ba2c271b9d29e33`.
The answer cards retain "Not evaluated (unreviewed)"; these captures demonstrate
the interface and are not an additional accuracy or latency benchmark. The older
`interface.jpg` remains linked in the historical v0.2 validation record.
