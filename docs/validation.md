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
