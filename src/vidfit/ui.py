"""Optional loopback-only Gradio UI over VideoAnalyzer; no model dependency at import."""

import html
import json
import tempfile
import threading
from pathlib import Path

from vidfit import SelectionSpec
from vidfit.core.video import evidence_image, file_sha256
from vidfit.evaluation.reports import metric

_GPU_LOCK = threading.Lock()


def selection_from_controls(method, budget, timestamps):
    from vidfit.core.sampling import parse_timestamps

    if method == "manual":
        return parse_timestamps(timestamps)
    if method != "uniform" or int(budget) != budget:
        raise ValueError("choose uniform or manual selection with an integer frame budget")
    return SelectionSpec.uniform(int(budget))


def choose_selections(mode, method_a, budget_a, times_a, method_b, budget_b, times_b):
    selections = [selection_from_controls(method_a, budget_a, times_a)]
    if mode == "Compare settings":
        selections.append(selection_from_controls(method_b, budget_b, times_b))
    elif mode != "Answer once":
        raise ValueError("choose Answer once or Compare settings")
    return selections


def identity(frames):
    return [(f.frame_index, f.timestamp_s, f.target_timestamp_s, f.sha256_rgb) for f in frames]


def gallery(result):
    return [
        (
            evidence_image(f),
            f"{f.timestamp_s:.3f} s · #{f.frame_index} · "
            f"requested {f.target_timestamp_s:.3f} s · {c.selection['method']}",
        )
        for c in result.configurations
        for f in c.evidence
    ]


def result_card(result):
    if result is None:
        return "<p class='empty-result'>Your previous result will appear here after a rerun.</p>"
    cards = []
    for config in result.configurations:
        measured = [t for t in config.trials if t.phase == "measured"]
        answers = list(dict.fromkeys(t.answer for t in measured if t.answer is not None))
        answer = "\n\n".join(answers) or (
            "Inspection only — model inference was not run."
            if result.backend == "inspect"
            else "No model answer available."
        )
        coverage = config.validation.get("annotation_evidence_coverage") or config.validation.get(
            "fixture_evidence_coverage", {}
        )
        overlap = coverage.get("overlaps_event")
        covered = (
            "Not annotated"
            if overlap is None
            else ("Overlaps annotated event" if overlap else "No overlap with annotated event")
        )
        evaluations = [t.evaluation for t in measured]
        verdict = (
            "; ".join(
                dict.fromkeys(
                    "Not evaluated (unreviewed)"
                    if e["verdict"] == "not_evaluated"
                    else f"{e['verdict']} ({e['method']})"
                    for e in evaluations
                )
            )
            or "Not evaluated"
        )
        costs = []
        for key, label in (
            ("pipeline_s", "Pipeline"),
            ("generation_s", "Generation"),
            ("visual_tokens", "Visual tokens"),
            ("generated_tokens", "Output tokens"),
            ("peak_cuda_allocated_bytes", "GPU allocated"),
            ("peak_cuda_reserved_bytes", "GPU reserved"),
        ):
            stats = config.summary.get("metrics", {}).get(key)
            costs.append(f"{label}: {metric(stats['median'] if stats else None, key)}")
        cards.append(
            f"<section class='result-card'><h3>"
            f"{html.escape(config.selection.get('method', 'uniform'))}"
            f" · {config.requested_frames} frames</h3>"
            f"<h2>Model answer</h2>"
            f"<div class='model-answer'>{html.escape(answer)}</div>"
            f"<p><b>Run status:</b> "
            f"{'completed' if config.status == 'ok' else 'failed'}</p>"
            f"<p><b>Evidence coverage:</b> {covered}</p>"
            f"<p><b>Answer correctness:</b> {html.escape(verdict)}</p>"
            f"<p class='run-error'>{html.escape(config.error or '')}</p>"
            f"<details><summary>Measurements "
            f"({html.escape(result.settings['execution_mode'])})</summary>"
            f"<p>{'<br>'.join(costs)}</p></details></section>"
        )
    return (
        f"<p><b>Question:</b> {html.escape(result.question)}</p>"
        f"<p>{html.escape(result.video['filename'])} · {result.video['sha256'][:12]} · "
        f"{html.escape(result.created_at_utc)}</p>" + "".join(cards)
    )


class UIController:
    """Framework-independent callbacks, also exercised by CPU tests."""

    def __init__(self, analyzer, output_root=None):
        self.analyzer = analyzer
        self.output_root = (
            Path(output_root) if output_root else Path(tempfile.mkdtemp(prefix="vidfit-exports-"))
        )

    def preview(self, video, mode, *controls):
        if not video:
            raise ValueError("Choose a video first.")
        selections = choose_selections(mode, *controls)
        frame_sets, metadata = [], None
        for selection in selections:
            metadata, frames, _ = self.analyzer.preview(video, selection)
            frame_sets.append(frames)
        snapshot = {
            "video_sha256": metadata["sha256"],
            "selections": selections,
            "identities": [identity(frames) for frames in frame_sets],
        }
        images = [
            (
                evidence_image(f),
                f"{f.timestamp_s:.3f} s · #{f.frame_index} · "
                f"requested {f.target_timestamp_s:.3f} s · {s.method}",
            )
            for s, frames in zip(selections, frame_sets, strict=True)
            for f in frames
        ]
        return (
            snapshot,
            images,
            (
                f"Preview ready · {metadata['duration_s']:.3f}s · "
                f"{metadata['frame_count']} source frames. "
                "Preview and metadata setup are outside measured inference."
            ),
        )

    def run(
        self,
        video,
        question,
        mode,
        controls,
        snapshot,
        previous=None,
        *,
        benchmark=False,
        warmup=1,
        repeats=3,
        reference=None,
        annotation_id=None,
        progress=None,
    ):
        selections = choose_selections(mode, *controls)
        if not snapshot or selections != snapshot["selections"]:
            raise ValueError("Selection changed or has no preview. Click Preview selected frames.")
        if not video or file_sha256(Path(video)) != snapshot["video_sha256"]:
            raise ValueError("Video changed. Preview the new video before running.")
        # Lock also covers direct callback/API calls that could bypass Gradio's queue.
        with _GPU_LOCK:
            refs = {}
            if reference:
                schema = json.loads(Path(reference).read_text(encoding="utf-8"))
                if not isinstance(schema, dict):
                    raise ValueError("Reference must be a JSON annotation object or collection")
                refs = (
                    {"fixture_manifest": reference}
                    if schema.get("fixture_schema")
                    else {"annotation": reference, "annotation_id": annotation_id or None}
                )
            result = self.analyzer.compare(
                video,
                question,
                selections=selections,
                execution_mode="benchmark" if benchmark else "single",
                warmup=int(warmup),
                repeats=int(repeats),
                progress=progress,
                **refs,
            )
            for index, config in enumerate(result.configurations):
                if config.evidence and identity(config.evidence) != snapshot["identities"][index]:
                    raise ValueError("Input frames changed since preview. Preview again.")
            folder = self.output_root / result.run_id
            html_file = result.export_html(folder / "comparison.html")
            json_file = result.export_json(folder / "comparison.json")
        return (
            result,
            result_card(result),
            gallery(result),
            result_card(previous),
            gallery(previous) if previous else [],
            str(html_file),
            str(json_file),
            "Completed. Run completion does not establish answer correctness."
            if result.complete
            else "Run failed. Review the error and export; no fallback was applied.",
        )


def build_app(analyzer, *, video=None, output_root=None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            'UI dependencies are missing. Install: python -m pip install ".[ui]"'
        ) from exc
    from vidfit.ui_worker import LocalWorker

    controller = UIController(analyzer, output_root)
    worker = LocalWorker(controller)
    with gr.Blocks(
        title="VidFit · local video questions", analytics_enabled=False, fill_width=True
    ) as app:
        gr.Markdown("# VidFit\n### Ask a video. Inspect the inputs. Compare the result.")
        gr.Markdown(
            "Video processing and inference stay on this computer. "
            "**A completed run is not a correctness verdict.** "
            "Frames below are model inputs, not verified citations."
        )
        if analyzer.backend_name == "inspect":
            gr.Markdown(
                "**Inspection only — model inference was not run.** No answers or GPU costs."
            )
        with gr.Row():
            with gr.Column(scale=4):
                clip = gr.Video(
                    value=str(video) if video else None,
                    label="1 · Choose your video",
                    sources=["upload"],
                    include_audio=True,
                    height=250,
                )
                question = gr.Textbox(
                    label="2 · Ask a question", value="What error message appeared?", lines=2
                )
                mode = gr.Radio(
                    ["Answer once", "Compare settings"], value="Answer once", label="Mode"
                )
                gr.Markdown("**3 · Select input frames** · Seconds start at the first video frame.")
                method_a = gr.Radio(["uniform", "manual"], value="uniform", label="Selection A")
                budget_a = gr.Slider(2, 32, value=4, step=1, label="A · Uniform frame budget")
                times_a = gr.Textbox(
                    label="A · Manual timestamps (seconds, commas or spaces)",
                    value="7.5, 24.375, 37.5, 52.5",
                    visible=False,
                )
                with gr.Group(visible=False) as second:
                    method_b = gr.Radio(["uniform", "manual"], value="manual", label="Selection B")
                    budget_b = gr.Slider(2, 32, value=4, step=1, label="B · Uniform frame budget")
                    times_b = gr.Textbox(
                        label="B · Manual timestamps (seconds)", value="7.5, 24.375, 37.5, 52.5"
                    )
                preview_btn = gr.Button("Preview selected frames")
                with gr.Accordion("Evaluation and benchmark options", open=False):
                    reference = gr.File(
                        label="Optional annotation / collection / synthetic truth JSON",
                        file_types=[".json"],
                        type="filepath",
                    )
                    annotation_id = gr.Textbox(label="Annotation ID (for collections)")
                    benchmark = gr.Checkbox(
                        label="Benchmark: include first invocation, warmups and repeats",
                        value=False,
                    )
                    warmup = gr.Number(
                        value=1, minimum=0, precision=0, label="Warmups per selection"
                    )
                    repeats = gr.Number(
                        value=3, minimum=1, precision=0, label="Measured repetitions"
                    )
                    gr.Markdown(
                        "Unchecked: exactly one inference per configuration. "
                        "Checked: one separate "
                        "first invocation, then warmups and measured repeats per configuration."
                    )
                run_btn = gr.Button("Run selected settings", variant="primary")
                status = gr.Textbox(
                    label="Progress / status",
                    interactive=False,
                    value="Choose a video, then preview.",
                )
                gr.Markdown(
                    f"Backend: **{analyzer.backend_name}** · {analyzer.options['device']} · "
                    f"{analyzer.options['max_pixels']} pixels/frame · "
                    f"{analyzer.options['max_new_tokens']} output-token limit"
                )
            with gr.Column(scale=7):
                previews = gr.Gallery(
                    label="Preview · inputs provided to the model",
                    columns=4,
                    height=280,
                    object_fit="contain",
                    preview=False,
                )
                with gr.Row():
                    with gr.Column(min_width=230):
                        gr.Markdown("## Previous result")
                        previous_html = gr.HTML(result_card(None))
                        previous_gallery = gr.Gallery(
                            label="Previous input frames",
                            columns=2,
                            height=260,
                            object_fit="contain",
                        )
                    with gr.Column(min_width=230):
                        gr.Markdown("## Current result")
                        current_html = gr.HTML("<p>Preview the frames, then run your question.</p>")
                        current_gallery = gr.Gallery(
                            label="Current input frames",
                            columns=2,
                            height=260,
                            object_fit="contain",
                        )
                with gr.Row():
                    html_file = gr.File(label="Offline HTML report", interactive=False)
                    json_file = gr.File(label="Structured JSON", interactive=False)
                gr.Markdown(
                    "**Read the costs separately:** latency is elapsed work; "
                    "answer length matters. "
                    "Visual tokens measure processed input size. GPU memory is allocator use, "
                    "including weights. Coverage and agreement are not accuracy."
                )
        snapshot = gr.State(None)
        current = gr.State(None)
        controls = [method_a, budget_a, times_a, method_b, budget_b, times_b]

        def invalidate():
            return None, [], "Preview is stale. Click Preview selected frames before running."

        for component in [clip, mode, *controls]:
            component.change(invalidate, outputs=[snapshot, previews, status], queue=False)
        mode.change(
            lambda value: gr.update(visible=value == "Compare settings"), mode, second, queue=False
        )

        def preview_callback(*args):
            try:
                return worker.preview(*args)
            except (ValueError, OSError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc

        for method, budget, times in ((method_a, budget_a, times_a), (method_b, budget_b, times_b)):
            method.change(
                lambda value: (
                    gr.update(visible=value == "uniform"),
                    gr.update(visible=value == "manual"),
                ),
                method,
                [budget, times],
                queue=False,
            )

        preview_btn.click(
            preview_callback,
            [clip, mode, *controls],
            [snapshot, previews, status],
            api_name="preview",
            concurrency_limit=1,
        )

        progress_tracker = gr.Progress()

        def run_callback(
            video,
            question,
            mode,
            method_a,
            budget_a,
            times_a,
            method_b,
            budget_b,
            times_b,
            preview,
            prev,
            bench,
            warm,
            repeat,
            ref,
            ref_id,
            progress=progress_tracker,
        ):
            control_values = [method_a, budget_a, times_a, method_b, budget_b, times_b]
            try:
                return worker.run(
                    video,
                    question,
                    mode,
                    control_values,
                    preview,
                    prev,
                    benchmark=bench,
                    warmup=warm,
                    repeats=repeat,
                    reference=ref,
                    annotation_id=ref_id,
                    progress=lambda message: progress(None, desc=message),
                )
            except (ValueError, OSError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc

        run_btn.click(
            run_callback,
            [
                clip,
                question,
                mode,
                *controls,
                snapshot,
                current,
                benchmark,
                warmup,
                repeats,
                reference,
                annotation_id,
            ],
            [
                current,
                current_html,
                current_gallery,
                previous_html,
                previous_gallery,
                html_file,
                json_file,
                status,
            ],
            api_name="run",
            concurrency_id="gpu",
            concurrency_limit=1,
        )
    app.vidfit_worker = worker
    return app.queue(default_concurrency_limit=1, max_size=8)


CSS = """
.gradio-container {max-width:1500px !important; margin:auto;}
.result-card {border:1px solid #b8cfca; border-radius:12px; padding:16px; margin:12px 0;}
.result-card h2 {font-size:23px; margin:14px 0 8px;}
.result-card p {font-size:13px; margin:6px 0;}
.model-answer {white-space:pre-wrap; overflow-wrap:anywhere; font-size:17px; line-height:1.6;
 background:rgba(40,170,140,.09); padding:14px; border-radius:8px;}
.run-error {color:#be3346; overflow-wrap:anywhere;}
.empty-result {padding:20px; opacity:.7;}
"""


def launch(analyzer, *, video=None, port=7860):
    import os

    # Gradio's version checks and analytics are disabled; share tunnels are never enabled.
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        os.environ[key] = ",".join(filter(None, [existing, "127.0.0.1", "localhost", "::1"]))
    app = build_app(analyzer, video=video)
    print("Preparing local worker (runtime imports only; no weights or inference)...", flush=True)
    app.vidfit_worker.start()
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=False,
        css=CSS,
        theme="default",
        show_error=True,
    )
