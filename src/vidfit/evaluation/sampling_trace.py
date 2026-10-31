"""Offline, escaped native-sampling comparisons, independent of inference reports."""

import html
import json
from pathlib import Path


def _escape(value):
    return html.escape(str(value), quote=True)


def _seconds(value):
    return "Unavailable" if value is None else f"{value:.4f} s"


def _pretty(value):
    return _escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _selection_panel(label, selection, shared):
    images, rows = [], []
    policy = selection["selection"]
    count_explanation = ""
    if policy.get("method") == "qwen_native":
        rate = policy.get("effective_sampling_fps")
        requested = policy.get("requested_count", policy.get("arguments", {}).get("num_frames"))
        count_explanation = (
            f"<p>Requested count: {_escape(requested)} → returned "
            f"{selection['selected_frames']}.</p>"
            if policy.get("mode") == "matched"
            else f"<p>Configured sampling rate: {_escape(rate)} fps → returned "
            f"{selection['selected_frames']}.</p>"
        )
        if policy.get("count_note"):
            count_explanation += f"<p>{_escape(policy['count_note'])}</p>"
    for frame in selection["frames"]:
        index = frame["frame_index"]
        common = index in shared
        timestamp = _seconds(frame["timestamp_s"])
        if frame.get("image_png_base64"):
            images.append(
                "<figure>"
                f'<img src="data:image/png;base64,{_escape(frame["image_png_base64"])}" '
                f'alt="{_escape(label)} source frame {index} at {timestamp}" loading="lazy">'
                f"<figcaption>#{index} · {timestamp}<br>"
                f"{'Shared frame' if common else 'Only in this selection'}"
                "</figcaption></figure>"
            )
        requested = _seconds(frame.get("target_timestamp_s"))
        rows.append(
            f'<tr class="{"st-shared" if common else ""}">'
            f"<td>{frame['position'] + 1}</td><td>#{index}</td>"
            f"<td>{timestamp}</td><td>{_seconds(frame.get('fps_timestamp_s'))}</td>"
            f"<td>{requested if label == 'VidFit selection' else '—'}</td>"
            f"<td>{'Yes' if common else 'No'}</td></tr>"
        )
    gallery = selection["gallery"]
    pairs = selection.get("temporal_pairs", [])
    pair_rows = "".join(
        "<tr>"
        f"<td>{i + 1}</td><td>{_escape(pair['frame_indices'])}</td>"
        f"<td>{_seconds(pair.get('fps_timestamp_s'))}</td>"
        f"<td>{_escape(pair.get('prompt_timestamp_s', 'Unavailable'))}</td>"
        f"<td>{_escape(pair.get('padding_positions', []))}</td></tr>"
        for i, pair in enumerate(pairs)
    )
    return (
        '<section class="st-panel">'
        f"<h3>{label}</h3>"
        f"<p><b>{selection['selected_frames']} selected</b> · "
        f"{selection['distinct_frames']} distinct source frames</p>"
        f"{count_explanation}"
        f"<p>Thumbnails: {gallery['shown']} of {gallery['total']} selected positions. "
        "All selected positions appear in the table and JSON.</p>"
        f'<div class="st-gallery">{"".join(images)}</div>'
        '<div class="st-table-scroll" tabindex="0" aria-label="Selected source frames">'
        "<table><thead><tr><th>Position</th><th>Source index</th>"
        "<th>Decoded time</th><th>Index / FPS time</th><th>Requested time</th>"
        f"<th>Shared</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<details><summary>Derived temporal groups</summary>"
        "<p>Projected from the processor's grouping rule and source FPS. "
        "The video encoder and language model were not run. "
        "Padding positions are zero-based; repeated final frames are padding, "
        "not additional source evidence.</p>"
        '<div class="st-table-scroll" tabindex="0"><table><thead><tr>'
        "<th>Group</th><th>Source indices</th><th>Mean index / FPS</th>"
        "<th>Prompt time (seconds)</th><th>Padding positions</th></tr></thead>"
        f"<tbody>{pair_rows}</tbody></table></div></details>"
        "<details><summary>Selection settings and duplicates</summary>"
        f"<pre>{_pretty(selection['selection'])}</pre>"
        f"<p>Duplicate source indices: {_escape(selection['duplicate_frame_indices'])}</p>"
        "</details></section>"
    )


def render_sampling_trace(trace, *, standalone=True):
    """Render only observed sampling plus clearly labeled derived grouping, never answers."""
    processor = trace["processor"]
    settings = processor["settings"]
    video = trace["video"]
    comparison = trace["comparison"]
    shared = set(comparison["shared_frame_indices"])
    mode = {
        "matched": "Same frame count as VidFit",
        "default": "Processor default FPS",
        "fps": "Custom native FPS",
    }.get(trace["request"]["native_mode"], trace["request"]["native_mode"])
    notes = "".join(f"<li>{_escape(note)}</li>" for note in trace.get("notes", []))
    algorithm = processor.get(
        "algorithm",
        "The installed native sampler is called directly; "
        "returned source indices are authoritative.",
    )
    body = (
        '<div class="sampling-trace">'
        "<h2>How Qwen picks frames</h2>"
        "<p class='st-lead'>Native processor sampling compared with your VidFit selection.</p>"
        "<p>No model weights loaded and no inference run. The native sampler receives video "
        "metadata and sampling settings, with no question or pixels. This trace does not "
        "change the frames used by Answer once or Compare settings.</p>"
        '<div class="st-summary">'
        f"<span><b>Video</b><br>{_escape(video['filename'])}</span>"
        f"<span><b>Source</b><br>{video['frame_count']} frames · "
        f"{_seconds(video['duration_s'])}</span>"
        f"<span><b>Native policy</b><br>{_escape(mode)}</span>"
        f"<span><b>Shared source frames</b><br>{len(shared)}</span></div>"
        "<details open><summary>Sampling rule and effective defaults</summary>"
        f"<p>{_escape(algorithm)}</p>"
        f"<p>Processor defaults: {_escape(settings.get('fps'))} fps · "
        f"minimum {_escape(settings.get('min_frames'))} · "
        f"maximum {_escape(settings.get('max_frames'))} frames. "
        f"Source FPS: {_escape(video.get('nominal_fps'))}.</p>"
        "<p>VidFit uniform selection targets time-bin midpoints; manual selection uses "
        "your requested times. Both map to the nearest decoded frame, choosing the "
        "earlier frame on a tie. Visual-change selection keeps the strongest pixel "
        "change in each time bin, using a midpoint when the bin is static. "
        "Source indices are zero-based. Decoded times are "
        "relative to the first source frame; index / FPS times are shown separately.</p>"
        "</details>"
        f'<div class="st-columns">{_selection_panel("VidFit selection", trace["vidfit"], shared)}'
        f"{_selection_panel('Native Qwen selection', trace['native'], shared)}</div>"
        f"<ul>{notes}</ul>"
        "<details><summary>Reproducibility and processor provenance</summary>"
        f"<pre>{_pretty(processor)}</pre>"
        f"<p>Video SHA-256: {_escape(video['sha256'])}</p>"
        f"<p>Trace: {_escape(trace['run_id'])} · {_escape(trace['created_at_utc'])}</p>"
        "</details></div>"
    )
    styled = f"<style>{_CSS}</style>{body}"
    if not standalone:
        return styled
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>VidFit · Native sampling trace</title></head><body>{styled}</body></html>"
    )


def export_sampling_trace(trace, output_dir):
    """Return paths to independent offline HTML and complete structured trace JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "sampling-trace.html"
    json_path = output_dir / "sampling-trace.json"
    markup = render_sampling_trace(trace)
    data = json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False)
    html_path.write_text(markup, encoding="utf-8")
    json_path.write_text(data, encoding="utf-8")
    return html_path, json_path


_CSS = """
.sampling-trace {font:14px/1.5 system-ui,sans-serif;color:#193b39;background:#f5faf9;
 padding:22px;border:1px solid #b8cfca;border-radius:12px;max-width:1500px;margin:auto;}
.sampling-trace * {box-sizing:border-box;}
.sampling-trace h2 {font-size:25px;margin:0 0 6px;color:#123633;}
.sampling-trace h3 {font-size:19px;margin:0 0 8px;color:#123633;}
.sampling-trace p {margin:8px 0 12px;}
.sampling-trace .st-lead {font-size:16px;}
.sampling-trace .st-summary {display:flex;flex-wrap:wrap;gap:16px;background:#e3efec;
 padding:14px;border-radius:8px;margin:16px 0;}
.sampling-trace .st-summary span {flex:1;min-width:130px;overflow-wrap:anywhere;}
.sampling-trace .st-columns {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
 gap:16px;margin-top:16px;}
.sampling-trace .st-panel {min-width:0;background:#fff;border:1px solid #c6d9d4;
 border-radius:8px;padding:14px;}
.sampling-trace .st-gallery {display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
 gap:10px;max-height:400px;overflow:auto;margin-bottom:12px;}
.sampling-trace figure {margin:0;border:1px solid #d1dfdb;border-radius:6px;overflow:hidden;}
.sampling-trace figure img {width:100%;height:110px;object-fit:contain;background:#eaf0ee;}
.sampling-trace figcaption {font-size:12px;padding:6px;}
.sampling-trace .st-table-scroll {max-height:360px;overflow:auto;border:1px solid #d1dfdb;}
.sampling-trace table {border-collapse:collapse;width:100%;font-size:12px;text-align:left;}
.sampling-trace th,.sampling-trace td {padding:7px;border-bottom:1px solid #dce7e3;
 white-space:nowrap;}
.sampling-trace th {position:sticky;top:0;background:#e3efec;}
.sampling-trace .st-shared {background:#eff8f1;}
.sampling-trace details {margin-top:12px;}
.sampling-trace summary {cursor:pointer;font-weight:650;}
.sampling-trace pre {max-height:340px;overflow:auto;background:#edf3f1;padding:12px;
 border-radius:6px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;}
@media(max-width:760px) {.sampling-trace .st-columns {grid-template-columns:1fr;}
 .sampling-trace {padding:12px;}}
"""
