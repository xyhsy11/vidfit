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

Real inference and publication validation are recorded below after their runs.
