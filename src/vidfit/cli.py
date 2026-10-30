"""Dependency-light command line entry point."""

import argparse
import sys
from pathlib import Path

from vidfit import VideoAnalyzer
from vidfit.backends.inspection import INSPECTION_NOTICE
from vidfit.demo import generate_demo


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Inspect video evidence and compare local VLM costs."
    )
    root.add_argument("--version", action="version", version="vidfit 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Generate a synthetic 60-second video and truth manifest"
    )
    demo.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Compare frame budgets and export HTML plus JSON")
    compare.add_argument("--video", type=Path, required=True)
    compare.add_argument("--question", required=True)
    compare.add_argument("--frame-budgets", type=int, nargs="+", default=[4, 8, 16])
    compare.add_argument("--sampling", choices=["uniform"], default="uniform")
    compare.add_argument("--backend", choices=["inspect", "qwen3vl"], default="inspect")
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument(
        "--fixture-manifest",
        type=Path,
        help="Optional separate synthetic truth manifest; SHA-256 verified",
    )
    compare.add_argument("--warmup", type=int, default=1, help="Warmups per budget (default: 1)")
    compare.add_argument(
        "--repeats", type=int, default=3, help="Measured trials per budget (default: 3)"
    )
    compare.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    compare.add_argument(
        "--revision", default="main", help="Hugging Face revision; resolved SHA recorded"
    )
    compare.add_argument("--device", default="cuda:0", help="Explicit cpu or cuda:N; no fallback")
    compare.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    compare.add_argument(
        "--max-pixels",
        type=int,
        default=262144,
        help="Maximum processed pixels per frame (default: 262144)",
    )
    compare.add_argument("--max-new-tokens", type=int, default=128)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            video, manifest = generate_demo(args.output)
            print(f"Synthetic video: {video}\nSeparate ground truth: {manifest}")
            return 0
        analyzer = VideoAnalyzer(
            backend=args.backend,
            model=args.model,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
        )
        result = analyzer.compare(
            video=args.video,
            question=args.question,
            frame_budgets=args.frame_budgets,
            sampling=args.sampling,
            warmup=args.warmup,
            repeats=args.repeats,
            fixture_manifest=args.fixture_manifest,
        )
        html = result.export_html(args.output_dir / "comparison.html")
        data = result.export_json(args.output_dir / "comparison.json")
        if args.backend == "inspect":
            print(INSPECTION_NOTICE)
        for config in result.configurations:
            print(
                f"{config.requested_frames} frames: {config.status}"
                + (f" — {config.error}" if config.error else "")
            )
        print(f"HTML: {html}\nJSON: {data}")
        return 0 if result.complete else 1
    except (ValueError, OSError) as exc:
        print(f"vidfit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
