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

The public repository [xyhsy11/vidfit](https://github.com/xyhsy11/vidfit) was
created and verified. Source is committed locally. The initial HTTPS push
was rejected because the GitHub CLI OAuth token lacks the `workflow` scope
needed to publish `.github/workflows/ci.yml`. The existing SSH key also needs
an unlocked agent. No force push or unrelated repository update was attempted.

The repository has no published `main` branch yet, so GitHub Actions has not
run. Publication requires the account holder to run
`gh auth refresh -h github.com -s workflow`, complete GitHub authentication,
and resume the push. No token should be pasted into chat. The Windows/Linux
CPU workflow is implemented but remote CI remains unverified until that push.
