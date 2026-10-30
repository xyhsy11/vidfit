# Examples

Generate the brief-event demo (60 s, 8 fps, 640x360):

```sh
vidfit demo --output examples/generated/brief-error.mp4
vidfit ui --video examples/generated/brief-error.mp4
```

Use `--backend inspect` on the UI for hardware-free input inspection. The separate
synthetic `.truth.json` manifest annotates overlap only and never enters model input.
The event appears in `[24,26)`; manual timestamps `7.5,24.375,37.5,52.5` include it
with four frames. Uniform four-frame sampling misses it.

Prepare the five-clip, ten-question starter evaluation set:

```sh
python examples/prepare_evaluation.py
vidfit evaluate --collection examples/evaluation/starter.json --video-dir examples/generated/evaluation --backend inspect --output-dir reports/starter
```

See [the evaluation guide](../docs/evaluation.md) for licenses, attribution,
checksums, normalization, agent visual-review provenance and judgment commands.
Only code and small annotation metadata are versioned. Videos, downloaded
originals, caches and generated reports stay local and are ignored by Git.
