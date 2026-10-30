# Synthetic example

Generate assets locally; generated videos, truth manifests, and reports are ignored:

```sh
vidfit demo --output examples/generated/brief-error.mp4
vidfit compare --video examples/generated/brief-error.mp4 --question "What error message appeared?" --frame-budgets 4 8 16 --backend inspect --fixture-manifest examples/generated/brief-error.truth.json --output-dir reports/inspection
```

The ground-truth manifest is optional for analysis. It is used only to annotate
fixture coverage in reports and is never passed to a model. The filename and
question contain no answer. The synthetic recording is 60 seconds, 8 fps, 640×360;
the event appears only during `[24,26)` seconds. Source code is in `vidfit.demo`.
