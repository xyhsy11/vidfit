"""Offline HTML with escaped text, inline styling, and lossless evidence previews."""

import html
import json

from vidfit.backends.inspection import INSPECTION_NOTICE


def escape(value):
    return html.escape(str(value), quote=True)


def pretty(value):
    return escape(json.dumps(value, indent=2, ensure_ascii=False))


def metric(value, key):
    if value is None:
        return "Unavailable"
    if key.endswith("_bytes"):
        return f"{value / 1024**2:,.1f} MiB"
    if key.endswith("_s"):
        return f"{value:.4f} s"
    return f"{value:,.0f}"


LABELS = {
    "end_to_end_s": "Complete request time",
    "input_preparation_s": "Input preparation (included in total)",
    "probe_s": "Metadata probe + hash",
    "pipeline_s": "Pipeline latency",
    "decode_sampling_s": "Decode + sample",
    "visual_change_scan_s": "Visual-change scan (within decode + sample)",
    "visual_change_capture_s": "Visual-change frame capture (within decode + sample)",
    "preprocessing_s": "Model preprocessing",
    "device_transfer_s": "Device transfer",
    "generation_s": "Generation",
    "output_decode_s": "Output text decoding",
    "peak_cuda_allocated_bytes": "Peak CUDA allocated",
    "peak_cuda_reserved_bytes": "Peak CUDA reserved",
    "visual_tokens": "Verified visual tokens",
    "input_tokens": "Input tokens",
    "generated_tokens": "Generated tokens",
    "decoded_frames": "Decoded source frames",
    "model_input_frames": "Model-input frames (including padding)",
}

OPTIONAL_METRICS = {
    "end_to_end_s",
    "input_preparation_s",
    "probe_s",
    "visual_change_scan_s",
    "visual_change_capture_s",
}


def render_html(result):
    header_cells = "".join(
        f"<th>{escape(c.selection.get('method', 'uniform'))} · {c.requested_frames} frames</th>"
        for c in result.configurations
    )
    rows = []
    for key, label in LABELS.items():
        if key in OPTIONAL_METRICS and not any(
            config.summary.get("metrics", {}).get(key) for config in result.configurations
        ):
            continue
        cells = []
        for config in result.configurations:
            stats = config.summary.get("metrics", {}).get(key)
            if stats:
                content = metric(stats["median"], key)
                content += (
                    f"<small>range {metric(stats['min'], key)} – {metric(stats['max'], key)}"
                    f" · n={stats['n']}</small>"
                )
            else:
                content = "<span class='muted'>Unavailable</span>"
            cells.append(f"<td>{content}</td>")
        rows.append(f"<tr><th>{escape(label)}</th>{''.join(cells)}</tr>")
    cards = []
    for config in result.configurations:
        measured = [t for t in config.trials if t.phase == "measured"]
        answers = list(dict.fromkeys(t.answer for t in measured if t.answer is not None))
        answer = (
            "\n\n".join(answers)
            if answers
            else (
                INSPECTION_NOTICE
                if result.backend == "inspect"
                else "No model answer is available."
            )
        )
        coverage = config.validation.get("annotation_evidence_coverage") or config.validation.get(
            "fixture_evidence_coverage"
        )
        coverage_html = "<p>Evidence coverage: Not annotated.</p>"
        if coverage and coverage["overlaps_event"] is not None:
            text = (
                "Overlaps annotated event"
                if coverage["overlaps_event"]
                else "No overlap with annotated event"
            )
            coverage_html = f"<p class='coverage'>Evidence coverage: {text}. This does not evaluate the answer.</p>"
        evaluations = []
        for trial in measured:
            judgment = trial.evaluation
            if judgment["verdict"] == "not_evaluated":
                evaluations.append("Not evaluated (unreviewed)")
            else:
                evaluations.append(
                    f"{judgment['verdict'].replace('_', ' ')} ({judgment['method']}; "
                    f"{judgment.get('reviewer', '')}): {judgment.get('rationale', '')}"
                )
        evaluation_text = "; ".join(dict.fromkeys(evaluations)) or "Not evaluated"
        gallery = []
        for frame in config.evidence:
            event = bool(coverage and frame.frame_index in coverage["overlapping_frame_indices"])
            gallery.append(
                f"<details class='frame {'event' if event else ''}'><summary>"
                f"<img alt='Source frame {frame.frame_index} at {frame.timestamp_s:.3f} seconds' "
                f"src='data:image/png;base64,{frame.image_png_base64}'>"
                f"<span>{frame.timestamp_s:.3f} s · frame #{frame.frame_index}"
                f"{' · EVENT' if event else ''}</span></summary>"
                f"<p>Target {frame.target_timestamp_s:.3f} s · Actual relative PTS {frame.timestamp_s:.3f} s"
                f"<br>Source PTS {frame.pts} × {escape(frame.time_base)} = "
                f"{frame.presentation_timestamp_s:.6f} s<br>Source {frame.width} × {frame.height}"
                f"<br>RGB SHA-256 <code>{frame.sha256_rgb}</code></p></details>"
            )
        trials = []
        for trial in config.trials:
            table = "".join(
                f"<tr><th>{escape(LABELS.get(k, k))}</th><td>{metric(v, k)}"
                + (
                    f"<small>{escape(trial.unavailable_reasons.get(k, ''))}</small>"
                    if v is None
                    else ""
                )
                + "</td></tr>"
                for k, v in trial.metrics.items()
            )
            trials.append(
                f"<details><summary>{escape(trial.phase)} #{trial.index} · {escape(trial.status)}</summary>"
                f"<p class='answer'>{escape(trial.answer) if trial.answer is not None else escape(trial.unavailable_reasons.get('answer', 'Answer unavailable'))}</p>"
                f"<p class='failure'>{escape(trial.error or '')}</p><p>Answer evaluation: {pretty(trial.evaluation)}</p><table>{table}</table>"
                f"<h4>Processor settings and effective inputs</h4><pre>{pretty(trial.processing)}</pre></details>"
            )
        cards.append(
            f"<article><h2>{escape(config.selection.get('method', 'uniform'))} · {config.requested_frames} frames <span class='status'>{'completed' if config.status == 'ok' else 'failed'}</span></h2>"
            f"<h3>Model answer</h3><div class='answer'>{escape(answer)}</div>"
            f"<p>Run status: {'completed' if config.status == 'ok' else 'failed'}. A completed run does not establish correctness.</p>{coverage_html}"
            f"<p>Answer correctness: {escape(evaluation_text)}</p>"
            f"<p class='failure'>{escape(config.error or '')}</p>"
            f"<h3>Inputs provided to the model</h3><p class='muted'>Click a frame to enlarge. "
            f"These are the selected RGB source frames. They are not model-verified citations or justification. The adapter's input format, resizing, processor dimensions and padding are recorded in provenance below.</p>"
            f"<div class='gallery'>{''.join(gallery)}</div><h3>Every trial</h3>{''.join(trials)}"
            f"<details><summary>Summary statistics and validation</summary>"
            f"<pre>{pretty({'summary': config.summary, 'validation': config.validation})}</pre></details></article>"
        )
    notice = (
        INSPECTION_NOTICE
        if result.backend == "inspect"
        else f"{result.model.get('identifier', result.settings.get('model', 'Local model'))} · "
        f"{result.model.get('input_representation', result.backend).replace('_', ' ')} · "
        "sequential configurations · inspect trial status below"
    )
    if result.settings.get("execution_mode") == "selection_experiment":
        statistics_notice = (
            "Medians and min–max ranges of successful measured requests. "
            "The experiment's two global warmups are recorded separately and excluded."
        )
        timing_boundaries = """<p>Complete request time includes the metadata probe and hash,
all source decoding, frame selection, lossless PNG evidence capture and hashes,
PNG-to-RGB conversion, adapter resizing and processor work, input validation, device transfer,
generation, output text decoding, and final CUDA synchronization.
Input preparation includes the metadata probe, decode + sample, model preprocessing,
and device transfer. Visual-change scan and frame capture are included within
decode + sample; do not add these subtotals again.
Generation includes visual encoding, prefill, and autoregressive decoding.
Model/runtime startup, the experiment's two global warmups, reference review,
and report export are outside complete request time.</p>
<p>Every measured request repeats the metadata probe and frame selection.
The model and processor are reused, filesystem caches may be warm, and the CUDA allocator cache is retained.
Peak allocated and reserved memory are PyTorch allocator measurements, include resident model weights,
and are not total device memory usage. Peaks are reset for each request and CUDA timing boundaries are synchronized.</p>"""
    else:
        statistics_notice = (
            "Medians and min–max ranges of successful measured trials. "
            "Warmups and the first invocation are excluded."
        )
        timing_boundaries = """<p>Each trial decodes the video again. Decode + sample includes lossless PNG evidence capture and hashes.
Pipeline latency starts before decoding and ends after output text decoding and CUDA synchronization.
Preprocessing includes PNG-to-RGB conversion, adapter resizing and processor work, and input validation.
Device transfer is timed separately. Generation includes visual encoding, prefill, and autoregressive decoding.
Model loading, the initial metadata probe, initial gallery extraction, and report export are outside trial latency.</p>
<p>In benchmark mode, one first invocation per comparison is recorded separately; it is not a fresh-process cold-start measurement.
Single mode runs exactly once per configuration with no warmup. The model and processor are reused, filesystem caches may be warm, and the CUDA allocator cache is retained.
Peak allocated and reserved memory are PyTorch allocator measurements, include resident model weights,
and are not total device memory usage. Peaks are reset for each trial and CUDA timing boundaries are synchronized.</p>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VidFit · Video evidence comparison</title><style>
:root{{color-scheme:dark;--bg:#101722;--panel:#182334;--line:#304258;--text:#e8eff9;--muted:#afbed1;--accent:#69e3c3}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:32px}}h1{{font-size:44px;letter-spacing:-2px;margin:8px 0}}h2{{font-size:25px}}
.eyebrow{{text-transform:uppercase;letter-spacing:3px;color:var(--accent);font-size:12px}}.banner{{padding:16px 20px;border-left:3px solid var(--accent);background:var(--panel)}}
.muted,small{{color:var(--muted)}}small{{display:block;font-size:12px;margin-top:5px}}.question{{font-size:24px;overflow-wrap:anywhere}}
.configs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:20px;margin-top:28px}}
article{{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px}}
.answer{{white-space:pre-wrap;overflow-wrap:anywhere;background:#101a29;padding:16px;border-radius:8px}}
.status{{font-size:12px;color:var(--accent)}}.coverage{{color:var(--accent);font-size:14px}}.failure{{color:#ffadba;overflow-wrap:anywhere}}
.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}img{{display:block;width:100%;border-radius:6px}}
.frame{{margin:0;padding:0;border:1px solid var(--line);border-radius:8px;overflow:hidden}}.frame summary{{list-style:none;padding:5px}}
.frame summary span{{font-size:12px;display:block;padding:5px}}.frame[open]{{grid-column:1/-1}}.frame p{{padding:8px;font-size:12px}}
.event{{border:2px solid var(--accent)}}summary{{cursor:pointer;padding:10px 0;font-weight:600}}details{{border-top:1px solid var(--line);margin-top:10px}}
pre,code{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}pre{{padding:12px;background:#101a29;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:13px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
.table-scroll{{overflow:auto;margin-top:20px}}.table-scroll table{{min-width:650px}}.definitions{{margin:36px 0;color:var(--muted)}}
@media(max-width:600px){{main{{padding:16px}}h1{{font-size:36px}}article{{padding:16px}}.question{{font-size:20px}}}}
</style></head><body><main><div class="eyebrow">Local video-language experiments</div><h1>VidFit</h1>
<p class="banner">{escape(notice)} · {escape(result.settings.get("execution_mode", "benchmark"))} mode</p><p class="question">{escape(result.question)}</p>
<p>{escape(result.video["filename"])} · {result.video["duration_s"]:.3f} s · {result.video["width"]} × {result.video["height"]}
 · {result.video["frame_count"]} source frames</p>
<details><summary>Video identity, requested settings, model, and environment</summary>
<p>Run {escape(result.run_id)} · {escape(result.created_at_utc)}</p><pre>{pretty({"video": result.video, "settings": result.settings, "model": result.model, "environment": result.environment, "fixture": result.fixture, "annotation": result.annotation, "schema_version": result.schema_version})}</pre></details>
<section class="configs">{"".join(cards)}</section>
<details><summary>Advanced measurements</summary>
<p>Latency is elapsed time to process input and generate an answer. Longer answers can take more time.
Visual tokens measure the processed visual input size, not answer quality.
GPU memory is PyTorch allocator use including model weights, not total device usage or a hardware minimum.</p>
<p class="muted">{escape(statistics_notice)}</p>
<div class="table-scroll"><table><thead><tr><th>Metric</th>{header_cells}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>
<section class="definitions"><h2>Measurement boundaries</h2>
{timing_boundaries}
<p>For the native Qwen video adapter, visual tokens are counted from processed input IDs and checked against the video grid.
For the generic image adapter, visual tokens are unavailable; actual image views, tensor shapes and processed pixel budgets are recorded instead.
Unavailable values are null in JSON with per-trial reasons. Inspection mode runs no model and has no answer.
Evidence coverage describes source inputs, not answer correctness. Agreement between answers is not accuracy. Reports contain the selected video frames.</p>
</section></main></body></html>"""
