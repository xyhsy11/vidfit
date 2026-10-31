"""Optional loopback-only Gradio UI over VideoAnalyzer; no model dependency at import."""

import html
import json
import math
import tempfile
import threading
from pathlib import Path

from vidfit import SelectionSpec
from vidfit.core.video import evidence_image, file_sha256
from vidfit.evaluation.reports import metric

_GPU_LOCK = threading.Lock()
CUSTOM_MODEL = "Custom Hugging Face model"


def model_settings_from_controls(model, backend, revision, dtype, thinking, max_pixels):
    if not isinstance(model, str) or not isinstance(revision, str):
        raise ValueError("Enter a model ID and revision.")
    if (
        isinstance(max_pixels, bool)
        or not isinstance(max_pixels, (int, float))
        or not math.isfinite(max_pixels)
        or int(max_pixels) != max_pixels
    ):
        raise ValueError("Pixel limit must be an integer.")
    return {
        "model": model.strip(),
        "backend": backend,
        "revision": revision.strip(),
        "dtype": dtype,
        "enable_thinking": thinking,
        "max_pixels": int(max_pixels),
    }


def native_trace_capability(model, backend):
    """Only known presets advertise capability; custom checkpoints are checked on request."""
    from vidfit.backends.registry import MODEL_PRESETS

    preset = next((item for item in MODEL_PRESETS if item["model"] == model), None)
    if backend == "transformers" or (preset and not preset["supports_native_trace"]):
        return False, "Native sampling trace is unavailable for this model/backend."
    if (preset and preset["supports_native_trace"]) or backend in ("qwen", "qwen3vl"):
        return True, "Choose a video and Selection A, then trace the native Qwen sampler."
    return True, (
        "Custom checkpoint: native trace supports dense Qwen3-VL / Qwen3.5 only. "
        "Trace checks checkpoint compatibility before sampling; other models are unavailable."
    )


def model_notice(backend):
    if backend == "inspect":
        return "**Inspection only.** Frames can be previewed; no model answer will be generated."
    return (
        "Model files are loaded only when you run an answer. "
        "The generic Transformers adapter sends selected frames as timestamped images. "
        "Automatic mode uses the Qwen native video adapter for supported Qwen checkpoints."
    )


def selection_from_controls(method, budget, timestamps):
    from vidfit.core.sampling import parse_timestamps

    if method == "manual":
        return parse_timestamps(timestamps)
    if method not in ("uniform", "visual_change") or int(budget) != budget:
        raise ValueError("choose uniform, visual changes, or manual with an integer frame budget")
    constructor = (
        SelectionSpec.visual_change if method == "visual_change" else SelectionSpec.uniform
    )
    return constructor(int(budget))


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
    model = result.model.get("identifier") or result.settings.get("model", "Unknown checkpoint")
    model_label = (
        f"Inspection only; requested {model}" if result.backend == "inspect" else str(model)
    )
    revision = result.model.get("resolved_revision") or result.settings.get("revision", "Unknown")
    return (
        f"<p><b>Model:</b> {html.escape(model_label)} · "
        f"<b>Backend:</b> {html.escape(result.backend)}<br>"
        f"Revision: {html.escape(str(revision))} · "
        f"dtype: {html.escape(str(result.settings.get('dtype', 'Unknown')))} · "
        f"thinking requested: {'on' if result.settings.get('enable_thinking') else 'off'}</p>"
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

    def _request_analyzer(self, model_settings):
        if model_settings is None:
            return self.analyzer
        from vidfit import VideoAnalyzer

        allowed = {*self.analyzer.options, "backend"}
        if not isinstance(model_settings, dict) or set(model_settings) - allowed:
            raise ValueError("Model settings must contain supported analyzer options.")
        options = {**self.analyzer.options, **model_settings}
        backend = options.pop("backend", self.analyzer.backend_name)
        return VideoAnalyzer(backend=backend, **options)

    @staticmethod
    def _settings(analyzer):
        return {"backend": analyzer.backend_name, **analyzer.options}

    def preview(self, video, mode, *controls, model_settings=None):
        if not video:
            raise ValueError("Choose a video first.")
        analyzer = self._request_analyzer(model_settings)
        selections = choose_selections(mode, *controls)
        frame_sets, metadata = [], None
        for selection in selections:
            metadata, frames, _ = analyzer.preview(video, selection)
            frame_sets.append(frames)
        snapshot = {
            "video_sha256": metadata["sha256"],
            "selections": selections,
            "identities": [identity(frames) for frames in frame_sets],
            "model_settings": self._settings(analyzer),
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

    def trace(
        self,
        video,
        method_a,
        budget_a,
        times_a,
        native_mode="matched",
        native_fps=None,
        *,
        model_settings=None,
    ):
        if not video:
            raise ValueError("Choose a video first.")
        selection = selection_from_controls(method_a, budget_a, times_a)
        analyzer = self._request_analyzer(model_settings)
        if model_settings is not None:
            available, message = native_trace_capability(
                analyzer.options["model"], analyzer.backend_name
            )
            if not available:
                raise ValueError(message)
        trace = analyzer.trace_sampling(
            video,
            selection,
            native_mode=native_mode,
            native_fps=native_fps if native_mode == "fps" else None,
        )
        from vidfit.evaluation.sampling_trace import render_sampling_trace

        markup = render_sampling_trace(trace, standalone=False)
        report = render_sampling_trace(trace, standalone=True)
        payload = json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False)
        folder = self.output_root / trace["run_id"]
        folder.mkdir(parents=True, exist_ok=True)
        html_file = folder / "sampling-trace.html"
        json_file = folder / "sampling-trace.json"
        html_file.write_text(report, encoding="utf-8")
        json_file.write_text(payload, encoding="utf-8")
        return (
            markup,
            str(html_file),
            str(json_file),
            "Trace complete. No model weights or inference were used.",
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
        model_settings=None,
    ):
        analyzer = self._request_analyzer(model_settings)
        selections = choose_selections(mode, *controls)
        if not snapshot or selections != snapshot["selections"]:
            raise ValueError("Selection changed or has no preview. Click Preview selected frames.")
        if snapshot.get("model_settings", self._settings(self.analyzer)) != self._settings(
            analyzer
        ):
            raise ValueError("Model settings changed. Preview selected frames before running.")
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
            result = analyzer.compare(
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
    from vidfit.backends.registry import BACKEND_NAMES, MODEL_PRESETS
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
        with gr.Row():
            with gr.Column(scale=4):
                preset_models = {item["model"] for item in MODEL_PRESETS}
                model_preset = gr.Dropdown(
                    choices=[(item["name"], item["model"]) for item in MODEL_PRESETS]
                    + [(CUSTOM_MODEL, CUSTOM_MODEL)],
                    value=analyzer.options["model"]
                    if analyzer.options["model"] in preset_models
                    else CUSTOM_MODEL,
                    label="Model preset",
                )
                model_id = gr.Textbox(
                    label="Hugging Face model ID",
                    value=analyzer.options["model"],
                )
                with gr.Accordion("Model runtime settings", open=False):
                    backend = gr.Dropdown(
                        choices=list(BACKEND_NAMES),
                        value=analyzer.backend_name,
                        label="Backend",
                    )
                    revision = gr.Textbox(
                        label="Model revision", value=analyzer.options["revision"]
                    )
                    dtype = gr.Dropdown(
                        choices=["float16", "bfloat16", "float32"],
                        value=analyzer.options["dtype"],
                        label="Model precision",
                    )
                    thinking = gr.Checkbox(
                        label="Request thinking (when supported)",
                        value=analyzer.options["enable_thinking"],
                    )
                    max_pixels = gr.Number(
                        label="Maximum pixels per frame",
                        minimum=4096,
                        precision=0,
                        value=analyzer.options["max_pixels"],
                    )
                    gr.Markdown(
                        f"Device: **{html.escape(analyzer.options['device'])}** · "
                        f"Output-token limit: **{analyzer.options['max_new_tokens']}** "
                        "(set when starting the app)."
                    )
                model_status = gr.Markdown(model_notice(analyzer.backend_name))
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
                selection_choices = [
                    ("uniform", "uniform"),
                    ("manual", "manual"),
                    ("Visual changes", "visual_change"),
                ]
                method_a = gr.Radio(selection_choices, value="uniform", label="Selection A")
                budget_a = gr.Slider(2, 32, value=4, step=1, label="A · Frame budget")
                times_a = gr.Textbox(
                    label="A · Manual timestamps (seconds, commas or spaces)",
                    value="7.5, 24.375, 37.5, 52.5",
                    visible=False,
                )
                with gr.Group(visible=False) as second:
                    method_b = gr.Radio(selection_choices, value="manual", label="Selection B")
                    budget_b = gr.Slider(2, 32, value=4, step=1, label="B · Frame budget")
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
            with gr.Column(scale=7):
                previews = gr.Gallery(
                    label="Preview · inputs provided to the model",
                    columns=4,
                    height=280,
                    object_fit="contain",
                    preview=False,
                )
                with gr.Accordion("How Qwen picks frames", open=True):
                    gr.Markdown(
                        "Compare **Selection A** with the native Qwen video sampler. "
                        "The sampler uses video timing and sampling settings; it does not "
                        "read your question or inspect frame content. "
                        "This trace uses no model weights or inference. "
                        "Your answer settings stay unchanged. "
                        "Requires the optional Qwen dependencies; first use may download "
                        "processor configuration files."
                    )
                    native_mode = gr.Radio(
                        choices=[
                            ("Same frame count as Selection A", "matched"),
                            ("Processor default FPS", "default"),
                            ("Custom FPS", "fps"),
                        ],
                        value="matched",
                        label="Native sampler settings",
                    )
                    native_fps = gr.Number(
                        value=2,
                        minimum=0.001,
                        label="Native sampling frames per second",
                        visible=False,
                    )
                    trace_available, trace_message = native_trace_capability(
                        analyzer.options["model"], analyzer.backend_name
                    )
                    trace_btn = gr.Button("Trace native sampling", interactive=trace_available)
                    trace_status = gr.Textbox(
                        label="Sampling trace status",
                        interactive=False,
                        value=trace_message,
                    )
                    trace_html = gr.HTML("")
                    with gr.Row():
                        trace_html_file = gr.File(label="Sampling trace HTML", interactive=False)
                        trace_json_file = gr.File(label="Sampling trace JSON", interactive=False)
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
        model_controls = [model_id, backend, revision, dtype, thinking, max_pixels]
        trace_outputs = [trace_html, trace_html_file, trace_json_file, trace_status]

        def select_preset(value):
            preset = next((item for item in MODEL_PRESETS if item["model"] == value), None)
            if preset is None:
                return [gr.update()] * 6
            return (
                preset["model"],
                preset["backend"],
                preset["dtype"],
                preset["max_pixels"],
                "main",
                False,
            )

        model_preset.change(
            select_preset,
            model_preset,
            [model_id, backend, dtype, max_pixels, revision, thinking],
            queue=False,
        )
        model_id.input(lambda: CUSTOM_MODEL, outputs=model_preset, queue=False)
        backend.change(model_notice, backend, model_status, queue=False)

        def update_trace_capability(model, backend):
            available, message = native_trace_capability(model, backend)
            return gr.update(interactive=available), message

        for component in (model_id, backend):
            component.change(
                update_trace_capability,
                [model_id, backend],
                [trace_btn, trace_status],
                queue=False,
            )

        def invalidate_trace():
            return "", None, None, "Settings changed. Trace native sampling again."

        def invalidate_model_trace(model, backend):
            _, message = native_trace_capability(model, backend)
            return "", None, None, message

        native_mode.change(
            lambda value: gr.update(visible=value == "fps"),
            native_mode,
            native_fps,
            queue=False,
        )

        def trace_callback(*args):
            yield "", None, None, "Tracing native sampling; no model inference..."
            try:
                request = (
                    {"model_settings": model_settings_from_controls(*args[6:])}
                    if len(args) > 6
                    else {}
                )
                result = worker.trace(*args[:6], **request)
            except (ImportError, ValueError, OSError, RuntimeError) as exc:
                result = "", None, None, f"Trace failed: {exc}"
            yield result

        trace_event = trace_btn.click(
            lambda: ("", None, None, "Tracing native sampling; no model inference..."),
            outputs=trace_outputs,
            queue=False,
        ).then(
            trace_callback,
            [clip, method_a, budget_a, times_a, native_mode, native_fps, *model_controls],
            trace_outputs,
            api_name="trace_sampling",
            concurrency_limit=1,
        )
        for component in [clip, method_a, budget_a, times_a, native_mode, native_fps]:
            # Cancel the generator's delivery if settings change during a trace.
            # The worker may finish its CPU job, but its stale result is discarded.
            component.change(
                invalidate_trace,
                outputs=trace_outputs,
                cancels=[trace_event],
                queue=False,
            )

        def invalidate():
            return None, [], "Preview is stale. Click Preview selected frames before running."

        for component in [clip, mode, *controls]:
            component.change(invalidate, outputs=[snapshot, previews, status], queue=False)
        mode.change(
            lambda value: gr.update(visible=value == "Compare settings"), mode, second, queue=False
        )

        def preview_callback(*args):
            try:
                request = (
                    {"model_settings": model_settings_from_controls(*args[8:])}
                    if len(args) > 8
                    else {}
                )
                yield worker.preview(*args[:8], **request)
            except (ValueError, OSError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc

        for method, budget, times in ((method_a, budget_a, times_a), (method_b, budget_b, times_b)):
            method.change(
                lambda value: (
                    gr.update(visible=value in ("uniform", "visual_change")),
                    gr.update(visible=value == "manual"),
                ),
                method,
                [budget, times],
                queue=False,
            )

        preview_event = preview_btn.click(
            preview_callback,
            [clip, mode, *controls, *model_controls],
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
            model=None,
            backend=None,
            revision=None,
            dtype=None,
            thinking=None,
            max_pixels=None,
            progress=progress_tracker,
        ):
            control_values = [method_a, budget_a, times_a, method_b, budget_b, times_b]
            try:
                request = (
                    {
                        "model_settings": model_settings_from_controls(
                            model, backend, revision, dtype, thinking, max_pixels
                        )
                    }
                    if model is not None
                    else {}
                )
                yield worker.run(
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
                    **request,
                )
            except (ValueError, OSError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc

        run_event = run_btn.click(
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
                *model_controls,
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

        def invalidate_model():
            return (
                None,
                [],
                "Model settings changed. Preview selected frames before running.",
                "",
                None,
                None,
                None,
                None,
            )

        for component in [model_preset, *model_controls]:
            component.change(
                invalidate_model,
                outputs=[
                    snapshot,
                    previews,
                    status,
                    trace_html,
                    trace_html_file,
                    trace_json_file,
                    html_file,
                    json_file,
                ],
                cancels=[preview_event, trace_event, run_event],
                queue=False,
            )
        for component in model_controls:
            component.change(
                invalidate_model_trace,
                [model_id, backend],
                outputs=trace_outputs,
                cancels=[trace_event],
                queue=False,
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
    print("Preparing local worker (no model loading or inference)...", flush=True)
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
