"""Dependency-light command line entry point. UI and model imports remain optional."""

import argparse
import json
import sys
from pathlib import Path

from vidfit import SelectionSpec, VideoAnalyzer, __version__
from vidfit.backends.inspection import INSPECTION_NOTICE
from vidfit.backends.registry import MODEL_PRESETS, backend_names
from vidfit.core.schemas import Comparison
from vidfit.demo import generate_demo


def model_options(command):
    command.add_argument(
        "--backend",
        choices=backend_names(),
        default="inspect",
        help=(
            "auto routes by model config; transformers uses timestamped images; "
            "inspect loads no model"
        ),
    )
    command.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    command.add_argument("--revision", default="main", help="HF revision; resolved SHA recorded")
    command.add_argument("--device", default="cuda:0", help="Explicit cpu or cuda:N; no fallback")
    command.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    command.add_argument("--max-pixels", type=int, default=262144, help="Pixel cap per frame")
    command.add_argument("--max-new-tokens", type=int, default=128)
    command.add_argument(
        "--thinking",
        dest="enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking explicitly (off by default)",
    )


def make_analyzer(args):
    return VideoAnalyzer(
        **{
            k: getattr(args, k)
            for k in (
                "backend",
                "model",
                "revision",
                "device",
                "dtype",
                "max_pixels",
                "max_new_tokens",
                "enable_thinking",
            )
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Inspect video inputs and compare local VLM answers."
    )
    root.add_argument("--version", action="version", version=f"vidfit {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="List local model presets (no downloads)")
    models.add_argument("--json", action="store_true", help="Print the preset catalog as JSON")
    demo = commands.add_parser("demo", help="Generate the synthetic brief-error video")
    demo.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser(
        "compare", help="Compare selections; export offline HTML and JSON"
    )
    compare.add_argument("--video", type=Path, required=True)
    compare.add_argument("--question", required=True)
    compare.add_argument("--frame-budgets", type=int, nargs="+", default=None)
    compare.add_argument(
        "--sampling", choices=["uniform", "manual", "visual_change"], default="uniform"
    )
    compare.add_argument(
        "--timestamps",
        type=float,
        nargs="+",
        action="append",
        help="Manual seconds; repeat option for multiple configurations",
    )
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument(
        "--fixture-manifest", type=Path, help="Separate synthetic truth; hash checked"
    )
    compare.add_argument(
        "--annotation", type=Path, help="Separate evaluation reference; never input"
    )
    compare.add_argument("--annotation-id", help="Question ID when using a collection")
    compare.add_argument("--execution-mode", choices=["single", "benchmark"], default="benchmark")
    compare.add_argument("--warmup", type=int, default=1, help="Benchmark warmups per selection")
    compare.add_argument("--repeats", type=int, default=3, help="Benchmark measured repetitions")
    model_options(compare)
    ui = commands.add_parser("ui", help="Launch the optional local Gradio interface")
    ui.add_argument("--video", type=Path, help="Optional initial local video")
    ui.add_argument("--port", type=int, default=7860)
    model_options(ui)
    ui.set_defaults(backend="auto")
    trace = commands.add_parser(
        "trace", help="Trace native Qwen sampling against VidFit; no model weights or inference"
    )
    trace.add_argument("--video", type=Path, required=True)
    selection = trace.add_mutually_exclusive_group()
    selection.add_argument("--frames", type=int, default=4, help="VidFit uniform frame budget")
    selection.add_argument("--timestamps", type=float, nargs="+", help="VidFit manual seconds")
    trace.add_argument(
        "--native-mode",
        choices=["matched", "default", "fps"],
        default="matched",
        help="Match VidFit's frame count, use processor defaults, or set native FPS",
    )
    trace.add_argument("--native-fps", type=float, help="Required only with --native-mode fps")
    trace.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    trace.add_argument(
        "--revision", default="main", help="Processor revision; resolved SHA recorded"
    )
    trace.add_argument("--output-dir", type=Path, required=True)
    judge = commands.add_parser("judge", help="Record an explicit judgment of one exported answer")
    judge.add_argument("--report", type=Path, required=True)
    judge.add_argument("--configuration", type=int, required=True, help="Zero-based index")
    judge.add_argument("--trial", type=int, required=True, help="Zero-based trial index, not phase")
    judge.add_argument(
        "--verdict",
        choices=["correct", "partially_correct", "incorrect", "unanswerable"],
        required=True,
    )
    judge.add_argument("--method", choices=["human", "automated", "agent"], required=True)
    judge.add_argument("--reviewer", required=True, help="Person, agent or check identifier")
    judge.add_argument("--rationale", required=True)
    judge.add_argument("--output-dir", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate", help="Run a starter collection; leaves answers unreviewed"
    )
    evaluate.add_argument("--collection", type=Path, required=True)
    evaluate.add_argument("--video-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--frames", type=int, default=4)
    model_options(evaluate)
    experiment = commands.add_parser(
        "experiment", help="Compare visual-change 4 with uniform 4/16; time full requests"
    )
    experiment.add_argument("--collection", type=Path, required=True)
    experiment.add_argument("--video-dir", type=Path, required=True)
    experiment.add_argument("--output-dir", type=Path, required=True)
    experiment.add_argument("--repeats", type=int, default=2)
    model_options(experiment)
    experiment.set_defaults(backend="auto", max_new_tokens=64)
    review = commands.add_parser("review-experiment", help="Apply explicit answer judgments")
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument("--judgments", type=Path, required=True)
    return root


def export(result, output):
    markup = result.export_html(output / "comparison.html")
    data = result.export_json(output / "comparison.json")
    for config in result.configurations:
        print(
            f"{config.selection.get('method', 'uniform')} / {config.requested_frames} frames: "
            f"{config.status}" + (f" — {config.error}" if config.error else "")
        )
    print(f"HTML: {markup}\nJSON: {data}")
    return 0 if result.complete else 1


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "models":
            if args.json:
                print(json.dumps(MODEL_PRESETS, indent=2, ensure_ascii=False))
            else:
                for preset in MODEL_PRESETS:
                    print(
                        f"{preset['name']}: {preset['model']} "
                        f"(dtype={preset['dtype']}, max_pixels={preset['max_pixels']})"
                    )
                print(
                    "Use --backend auto --model MODEL_ID. Custom native HF image-text models "
                    "are accepted when their processor satisfies the adapter's checks."
                )
            return 0
        if args.command == "demo":
            video, manifest = generate_demo(args.output)
            print(f"Synthetic video: {video}\nSeparate ground truth: {manifest}")
            return 0
        if args.command == "ui":
            from vidfit.ui import launch

            launch(make_analyzer(args), video=args.video, port=args.port)
            return 0
        if args.command == "trace":
            from vidfit.evaluation.sampling_trace import export_sampling_trace

            selection = (
                SelectionSpec.manual(args.timestamps)
                if args.timestamps is not None
                else SelectionSpec.uniform(args.frames)
            )
            trace = VideoAnalyzer(
                backend="inspect", model=args.model, revision=args.revision
            ).trace_sampling(
                args.video,
                selection,
                native_mode=args.native_mode,
                native_fps=args.native_fps,
            )
            markup, data = export_sampling_trace(trace, args.output_dir)
            print("Native Qwen sampling trace. No model weights loaded or inference run.")
            print(
                f"VidFit: {trace['vidfit']['selected_frames']} frames; "
                f"Qwen: {trace['native']['selected_frames']} frames; "
                f"shared source frames: {len(trace['comparison']['shared_frame_indices'])}"
            )
            print(f"HTML: {markup}\nJSON: {data}")
            return 0
        if args.command == "judge":
            from vidfit.evaluation.annotations import record_judgment

            result = Comparison.from_dict(json.loads(args.report.read_text(encoding="utf-8")))
            record_judgment(
                result,
                args.configuration,
                args.trial,
                args.verdict,
                args.method,
                args.rationale,
                args.reviewer,
            )
            return export(result, args.output_dir)
        if args.command == "review-experiment":
            from vidfit.experiment import apply_experiment_judgments

            apply_experiment_judgments(
                args.output_dir, json.loads(args.judgments.read_text(encoding="utf-8"))
            )
            print(f"Reviewed summary: {args.output_dir / 'summary.html'}")
            return 0
        analyzer = make_analyzer(args)
        if args.command == "experiment":
            from vidfit.experiment import run_selection_experiment

            summary = run_selection_experiment(
                args.collection,
                args.video_dir,
                args.output_dir,
                analyzer=analyzer,
                repeats=args.repeats,
            )
            print(f"Experiment summary: {args.output_dir / 'summary.html'}")
            print("Answers remain unreviewed until explicit judgments are recorded.")
            return 0 if summary["experiment"]["status"] == "complete" else 1
        if args.command == "evaluate":
            from vidfit.evaluation.annotations import read_collection

            failed = False
            for item in read_collection(args.collection):
                # Only basenames resolve under this explicit media directory.
                name = item["video"]["filename"]
                if Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError("collection video filename must be a basename")
                case_dir = args.output_dir / f"case-{len(list(args.output_dir.glob('case-*'))):03d}"
                result = analyzer.compare(
                    args.video_dir / name,
                    item["question"],
                    selections=[SelectionSpec.uniform(args.frames)],
                    execution_mode="single",
                    annotation=args.collection,
                    annotation_id=item["id"],
                )
                failed |= export(result, case_dir) != 0
            return int(failed)
        if args.sampling == "manual" and not args.timestamps:
            raise ValueError("--sampling manual requires --timestamps; no uniform fallback")
        if args.sampling == "visual_change" and args.timestamps:
            raise ValueError("--sampling visual_change cannot be combined with --timestamps")
        budget_selection = (
            SelectionSpec.visual_change
            if args.sampling == "visual_change"
            else SelectionSpec.uniform
        )
        selections = []
        if args.frame_budgets is not None or not args.timestamps:
            selections += [
                budget_selection(b)
                for b in (args.frame_budgets if args.frame_budgets is not None else [4, 8, 16])
            ]
        selections += [SelectionSpec.manual(t) for t in (args.timestamps or [])]
        # Preserve duplicate-budget validation for the original CLI.
        if args.frame_budgets and len(set(args.frame_budgets)) != len(args.frame_budgets):
            raise ValueError("duplicate frame budgets are invalid")
        result = analyzer.compare(
            video=args.video,
            question=args.question,
            selections=selections,
            warmup=args.warmup,
            repeats=args.repeats,
            execution_mode=args.execution_mode,
            fixture_manifest=args.fixture_manifest,
            annotation=args.annotation,
            annotation_id=args.annotation_id,
        )
        if args.backend == "inspect":
            print(INSPECTION_NOTICE)
        return export(result, args.output_dir)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"vidfit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
