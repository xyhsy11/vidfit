# Contributing

Use a project-local virtual environment with Python 3.10–3.13. Python 3.12 is the
reference environment. Install `python -m pip install -e ".[dev]"`, then run:

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

Ordinary tests must work without PyTorch, CUDA, network access, or downloaded
weights. Processor doubles are bookkeeping tests, never model-validation evidence.
The real-model test is opt-in; see the README. Include exact model/processor
revisions and dependency versions when reporting integration failures.

Keep changes small and test observable behavior. Preserve actual PTS, source
frame identity, temporal metadata, and measurement boundaries. Never substitute
simulated answers or costs. Explain unavailable measurements with null and a reason.

Do not commit private videos, generated reports, model weights, credentials, or
machine-specific caches. Synthetic demo source is welcome; attach large outputs
only when deliberately requested. Describe what changed and how it was checked
in a pull request. Contributions are under the project's MIT license; model
weights retain their separate upstream license.


For UI development install `.[ui,dev]` and launch `vidfit ui --backend inspect`.
Keep Gradio optional: base package imports, the CLI and ordinary tests must work
without it. Browser QA must include upload, playback, preview invalidation,
uniform/manual runs, previous/current results and both downloads. Use synthetic
footage for screenshots. GPU callbacks must share the UI lock/queue.

Evaluation references belong in separate annotation files. New cases need source
permission, checksums and recorded visual-review provenance. Never mark agent or
automated work as human-reviewed. Test reference isolation at the backend boundary.
A successful run, answer agreement, and event overlap must never be labeled accuracy.

Selection and execution changes must preserve the measurement contract. Test actual
source frame identity, timestamp ties and duplicate rejection, not just target
arithmetic. Single mode must have exactly one invocation per selected configuration.
Keep native CFR validation, temporal padding and processor token checks intact.
