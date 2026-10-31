"""Paired visual-change experiment with real answers and complete per-request timing."""

import hashlib
import itertools
import json
import random
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from vidfit.analyzer import VideoAnalyzer
from vidfit.core.profiling import describe_error, environment, summarize
from vidfit.core.sampling import SelectionSpec
from vidfit.core.schemas import Comparison, Configuration, Trial
from vidfit.core.video import file_sha256, probe_video, sample_video
from vidfit.evaluation.annotations import evidence_coverage, read_collection, record_judgment

METHODS = ("visual_change_4", "uniform_4", "uniform_16")
LABELS = {
    "visual_change_4": "Visual changes · 4 frames",
    "uniform_4": "Uniform midpoints · 4 frames",
    "uniform_16": "Uniform midpoints · 16 frames",
}


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _answer_id(question_id, answer):
    return hashlib.sha256(f"{question_id}\0{answer}".encode()).hexdigest()


def _run_trial(analyzer, backend, video, question, selection, phase, index):
    """No annotation, rubric, event interval or prior answer is available at this boundary."""
    trial = Trial(phase, index)
    frames, metadata, selection_trace = [], None, None
    backend.synchronize()
    backend.reset_peak_memory()
    started = perf_counter()
    try:
        probe_started = perf_counter()
        metadata = probe_video(video)
        trial.metrics["probe_s"] = perf_counter() - probe_started
        frames, metrics = sample_video(video, metadata, selection)
        selection_trace = metrics.pop("selection_trace", None)
        trial.metrics.update(metrics)
        analyzer._validate_backend_selection(metadata, frames)
        trial.evidence_hashes = [f.sha256_rgb for f in frames]
        outcome = backend.run(frames, metadata, question)
        trial.answer = outcome["answer"]
        trial.processing = outcome["processing"]
        if selection_trace is not None:
            trial.processing["selection_trace"] = selection_trace
        trial.metrics.update(outcome["metrics"])
        trial.unavailable_reasons.update(outcome.get("unavailable_reasons", {}))
        backend.synchronize()
    except Exception as exc:
        trial.status, trial.error = "failed", describe_error(exc)
        trial.unavailable_reasons["answer"] = "trial failed; no evaluated answer"
    finally:
        trial.metrics["end_to_end_s"] = perf_counter() - started
        trial.metrics["input_preparation_s"] = sum(
            trial.metrics.get(k, 0) or 0
            for k in ("probe_s", "decode_sampling_s", "preprocessing_s", "device_transfer_s")
        )
    return trial, metadata, frames


def _load_backend(analyzer):
    from vidfit.backends.registry import create_backend

    return create_backend(analyzer.backend_name, **analyzer.options)


def run_selection_experiment(
    collection, video_dir, output_dir, *, analyzer=None, repeats=2, progress=print
):
    """Run all three fixed methods; references attach only after each inference completes.

    An output directory must be new, so a later run cannot overwrite prior evidence.
    Real answers remain unreviewed until explicit judgments are applied.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("experiment repeats must be a positive integer")
    analyzer = analyzer or VideoAnalyzer(max_new_tokens=64)
    items = read_collection(collection)
    video_dir, output_dir = Path(video_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError("experiment output directory already exists; choose a new directory")
    paths = {}
    for item in items:
        name = item["video"]["filename"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("collection video filename must be a basename")
        path = video_dir / name
        if not path.is_file() or file_sha256(path) != item["video"]["sha256"]:
            raise ValueError(f"missing or mismatched evaluation video: {name}")
        paths[item["id"]] = path
    selections = [
        SelectionSpec.visual_change(4),
        SelectionSpec.uniform(4),
        SelectionSpec.uniform(16),
    ]
    output_dir.mkdir(parents=True)
    run_started = perf_counter()
    manifest = {
        "schema": "vidfit.selection_experiment.v1",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "backend": analyzer.backend_name,
        "settings": analyzer.options,
        "repeats": repeats,
        "method_order": list(METHODS),
        "selection_specs": [s.to_dict() for s in selections],
        "implementation_sha256": {
            "experiment.py": file_sha256(Path(__file__)),
            "core/visual_change.py": file_sha256(Path(__file__).parent / "core/visual_change.py"),
            "core/sampling.py": file_sha256(Path(__file__).parent / "core/sampling.py"),
            "backends/qwen3vl.py": file_sha256(Path(__file__).parent / "backends/qwen3vl.py"),
            "backends/qwen_runtime.py": file_sha256(
                Path(__file__).parent / "backends/qwen_runtime.py"
            ),
            "backends/registry.py": file_sha256(Path(__file__).parent / "backends/registry.py"),
            "backends/transformers_vlm.py": file_sha256(
                Path(__file__).parent / "backends/transformers_vlm.py"
            ),
        },
        "uniform_definition": (
            "VidFit uniform time-bin midpoints, not native Qwen endpoint sampling"
        ),
        "timing_contract": {
            "end_to_end_s": (
                "Warm-model request: metadata probe/hash through final answer synchronization; "
                "includes selector scan, source decode/capture/PNG hashing, model preprocessing, "
                "transfer, generation and text decoding."
            ),
            "input_preparation_s": (
                "probe_s + decode_sampling_s + preprocessing_s + device_transfer_s; "
                "visual_change_scan_s is nested inside decode_sampling_s, never added twice."
            ),
            "excluded": (
                "One-time runtime/model loading, recorded warmups, reference review, file exports "
                "and UI transport. Model loading and whole-run wall time are reported separately."
            ),
        },
        "evaluation_notice": (
            "Small exploratory starter set; known clips, agent-reviewed references, "
            "not a held-out or representative accuracy benchmark."
        ),
        "environment": environment(),
        "cases": [],
        "schedule": [],
        "warmups": [],
    }
    manifest_path = output_dir / "experiment.json"
    _write_json(manifest_path, manifest)
    progress("Loading one backend for the complete experiment")
    startup = perf_counter()
    try:
        backend = _load_backend(analyzer)
    except Exception as exc:
        manifest.update(status="failed", loading_error=describe_error(exc))
        _write_json(manifest_path, manifest)
        raise
    manifest["startup_s"] = perf_counter() - startup
    manifest["model"] = backend.info
    # Warm each distinct input shape explicitly. Warmups are retained, not accuracy observations.
    for count in (4, 16):
        progress(f"Recorded warmup: uniform {count} frames")
        trial, _, _ = _run_trial(
            analyzer,
            backend,
            paths[items[0]["id"]],
            items[0]["question"],
            SelectionSpec.uniform(count),
            "warmup",
            count,
        )
        from dataclasses import asdict

        manifest["warmups"].append(asdict(trial))
        _write_json(manifest_path, manifest)
        if trial.status != "ok":
            manifest.update(status="failed", run_elapsed_s=perf_counter() - run_started)
            _write_json(manifest_path, manifest)
            raise RuntimeError(f"experiment warmup failed: {trial.error}")
    orders = list(itertools.permutations(range(3)))
    results = []
    for case_index, item in enumerate(items):
        relative = f"case-{case_index:03d}/comparison.json"
        case_path = output_dir / relative
        case_path.parent.mkdir()
        result = Comparison(
            run_id=str(uuid.uuid4()),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            backend=analyzer.backend_name,
            video={},
            question=item["question"],
            settings={
                **analyzer.options,
                "execution_mode": "selection_experiment",
                "repeats": repeats,
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": analyzer.options["max_new_tokens"],
                    "enable_thinking": analyzer.options["enable_thinking"],
                },
            },
            environment=manifest["environment"],
            model=backend.info,
            configurations=[Configuration(s.count, selection=s.to_dict()) for s in selections],
        )
        manifest["cases"].append({"question_id": item["id"], "report": relative})
        for repetition in range(repeats):
            order = orders[(case_index * repeats + repetition) % len(orders)]
            for method_index in order:
                selection = selections[method_index]
                method = METHODS[method_index]
                progress(f"{item['id']} · {method} · repeat {repetition + 1}/{repeats}")
                trial, metadata, frames = _run_trial(
                    analyzer,
                    backend,
                    paths[item["id"]],
                    item["question"],
                    selection,
                    "measured",
                    repetition,
                )
                config = result.configurations[method_index]
                if metadata is not None:
                    if metadata["sha256"] != item["video"]["sha256"]:
                        trial.status, trial.error = (
                            "failed",
                            "video identity changed during experiment",
                        )
                        trial.answer = None
                    result.video = metadata
                if frames:
                    actual = [(f.frame_index, f.timestamp_s, f.sha256_rgb) for f in frames]
                    previous = [
                        (f.frame_index, f.timestamp_s, f.sha256_rgb) for f in config.evidence
                    ]
                    if config.evidence and actual != previous:
                        trial.status, trial.error = (
                            "failed",
                            "selected evidence changed across repeats",
                        )
                        trial.answer = None
                    if not config.evidence:
                        config.evidence = frames
                        config.selected_frames = len(frames)
                    config.validation["annotation_evidence_coverage"] = evidence_coverage(
                        item, frames
                    )
                if trial.status != "ok":
                    config.status, config.error = "failed", trial.error
                config.trials.append(trial)
                config.summary = summarize(config.trials)
                result.annotation = item
                manifest["schedule"].append(
                    {
                        "question_id": item["id"],
                        "method": method,
                        "repetition": repetition,
                        "status": trial.status,
                    }
                )
                result.export_json(case_path)
                _write_json(manifest_path, manifest)
        result.export_json(case_path.with_name("comparison.raw.json"))
        result.export_html(case_path.with_suffix(".html"))
        results.append(result)
    manifest.update(
        status="complete" if all(r.complete for r in results) else "failed",
        run_elapsed_s=perf_counter() - run_started,
    )
    _write_json(manifest_path, manifest)
    _write_review_packet(output_dir, manifest)
    return summarize_experiment(output_dir)


def _read_results(output_dir, manifest):
    return [
        (
            case["question_id"],
            Comparison.from_dict(
                json.loads((output_dir / case["report"]).read_text(encoding="utf-8"))
            ),
        )
        for case in manifest["cases"]
    ]


def _write_review_packet(output_dir, manifest):
    unique = {}
    for question_id, result in _read_results(output_dir, manifest):
        for config in result.configurations:
            for trial in config.trials:
                if trial.status != "ok" or trial.answer is None:
                    continue
                key = _answer_id(question_id, trial.answer)
                unique[key] = {
                    "answer_id": key,
                    "question_id": question_id,
                    "question": result.question,
                    "answerability": result.annotation.get("answerability", "supported"),
                    "accepted_answers": result.annotation.get("accepted_answers", []),
                    "rubric": result.annotation["rubric"],
                    "answer": trial.answer,
                }
    packet = list(unique.values())
    random.Random(20261031).shuffle(packet)
    _write_json(output_dir / "answers-for-review.json", packet)


def apply_experiment_judgments(output_dir, judgments):
    """Apply explicit method-blinded judgments by question/answer hash, never inferred scores."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "experiment.json").read_text(encoding="utf-8"))
    if not isinstance(judgments, list):
        raise ValueError("judgments must be a list")
    by_id = {j["answer_id"]: j for j in judgments}
    if len(by_id) != len(judgments):
        raise ValueError("duplicate judgment answer IDs")
    prepared = []
    used = set()
    for case, (question_id, result) in zip(
        manifest["cases"], _read_results(output_dir, manifest), strict=True
    ):
        for config_index, config in enumerate(result.configurations):
            for trial_index, trial in enumerate(config.trials):
                if trial.status != "ok" or trial.answer is None:
                    continue
                key = _answer_id(question_id, trial.answer)
                if key not in by_id:
                    raise ValueError(f"missing explicit judgment for answer {key}")
                judgment = by_id[key]
                record_judgment(
                    result,
                    config_index,
                    trial_index,
                    judgment["verdict"],
                    judgment["method"],
                    judgment["rationale"],
                    judgment["reviewer"],
                )
                used.add(key)
        prepared.append((output_dir / case["report"], result))
    if used != set(by_id):
        raise ValueError("judgments contain answers not present in this experiment")
    for path, result in prepared:
        result.export_json(path)
        result.export_html(path.with_suffix(".html"))
    _write_json(output_dir / "judgments.json", judgments)
    return summarize_experiment(output_dir)


def summarize_experiment(output_dir):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "experiment.json").read_text(encoding="utf-8"))
    results = _read_results(output_dir, manifest)
    methods = []
    for index, method in enumerate(METHODS):
        counts = dict(correct=0, partially_correct=0, incorrect=0, unanswerable=0, not_evaluated=0)
        supported, unsupported, correct, abstentions, failures = 0, 0, 0, 0, 0
        trials, judgment_methods, cases = [], set(), []
        for question_id, result in results:
            config = result.configurations[index]
            is_supported = result.annotation.get("answerability", "supported") == "supported"
            case_trials = []
            for trial in config.trials:
                supported += int(is_supported)
                unsupported += int(not is_supported)
                if trial.status != "ok":
                    failures += 1
                    continue
                trials.append(trial)
                case_trials.append(trial)
                verdict = trial.evaluation["verdict"]
                counts[verdict] += 1
                judgment_methods.add(trial.evaluation["method"])
                correct += int(is_supported and verdict == "correct")
                abstentions += int(not is_supported and verdict == "unanswerable")
            cases.append(
                {
                    "question_id": question_id,
                    "question": result.question,
                    "answerability": result.annotation.get("answerability", "supported"),
                    "selected_timestamps_s": [f.timestamp_s for f in config.evidence],
                    "event_coverage": config.validation.get("annotation_evidence_coverage"),
                    "answers": [t.answer for t in config.trials],
                    "statuses": [t.status for t in config.trials],
                    "errors": [t.error for t in config.trials],
                    "end_to_end_s": [t.metrics.get("end_to_end_s") for t in config.trials],
                    "verdicts": [t.evaluation for t in config.trials],
                    "median_end_to_end_s": statistics.median(
                        t.metrics["end_to_end_s"] for t in case_trials
                    )
                    if case_trials
                    else None,
                }
            )
        stats = {}
        for key in (
            "end_to_end_s",
            "input_preparation_s",
            "probe_s",
            "decode_sampling_s",
            "visual_change_scan_s",
            "preprocessing_s",
            "generation_s",
            "generated_tokens",
            "visual_tokens",
            "peak_cuda_allocated_bytes",
        ):
            values = [t.metrics[key] for t in trials if t.metrics.get(key) is not None]
            stats[key] = (
                {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }
                if values
                else None
            )
        judged = counts["not_evaluated"] == 0 and bool(trials)
        methods.append(
            {
                "method": method,
                "label": LABELS[method],
                "judgment_methods": sorted(judgment_methods),
                "verdict_counts": counts,
                "failures": failures,
                "supported_trials": supported,
                "unsupported_trials": unsupported,
                "supported_correct": correct if judged else None,
                "unsupported_abstentions": abstentions if judged else None,
                "strict_success_rate": (correct + abstentions) / (supported + unsupported)
                if judged and supported + unsupported
                else None,
                "metrics": stats,
                "cases": cases,
            }
        )
    summary = {
        "schema": "vidfit.selection_experiment_summary.v1",
        "experiment": manifest,
        "unique_questions": len(results),
        "methods": methods,
        "accuracy_definition": (
            "Strict success: correct supported answer or appropriate abstention on an unsupported "
            "question. Partial answers receive no full credit. Repeats are repeated observations "
            "of the same questions, not new independent cases."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    from vidfit.evaluation.experiment_report import render_experiment

    (output_dir / "summary.html").write_text(render_experiment(summary), encoding="utf-8")
    return summary
