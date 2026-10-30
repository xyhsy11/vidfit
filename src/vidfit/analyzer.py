"""Sequential comparisons with a reused model and independently decoded trials."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from vidfit.backends.inspection import InspectionBackend
from vidfit.core.profiling import INFERENCE_METRICS, describe_error, environment, summarize
from vidfit.core.sampling import SelectionSpec
from vidfit.core.schemas import Comparison, Configuration, Trial
from vidfit.core.video import probe_video, sample_video


class VideoAnalyzer:
    def __init__(
        self,
        model="Qwen/Qwen3-VL-2B-Instruct",
        backend="qwen3vl",
        *,
        device="cuda:0",
        dtype="float16",
        max_pixels=262144,
        max_new_tokens=128,
        revision="main",
    ):
        if backend not in ("inspect", "qwen3vl"):
            raise ValueError("backend must be inspect or qwen3vl")
        if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("device must be cpu or cuda:N")
        if dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels < 4096:
            raise ValueError("max_pixels must be an integer of at least 4096")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if not model.strip() or not revision.strip():
            raise ValueError("model and revision must not be empty")
        self.backend_name = backend
        self.options = dict(
            model=model,
            device=device,
            dtype=dtype,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            revision=revision,
        )

    def preview(self, video, selection):
        """Decode the exact selection without constructing a model. Outside trial timing."""
        metadata = probe_video(video)
        frames, metrics = sample_video(video, metadata, selection)
        self._validate_backend_selection(metadata, frames)
        return metadata, frames, metrics

    def _validate_backend_selection(self, metadata, frames):
        if self.backend_name == "qwen3vl":
            if len(frames) < 2:
                raise ValueError("Qwen3-VL requires at least 2 distinct selected frames")
            if not metadata["constant_frame_rate"]:
                raise ValueError("Qwen3-VL requires constant-frame-rate video; use inspect for VFR")

    def compare(
        self,
        video,
        question,
        frame_budgets=(4, 8, 16),
        sampling="uniform",
        warmup=1,
        repeats=3,
        fixture_manifest=None,
        *,
        selections=None,
        execution_mode="benchmark",
        annotation=None,
        annotation_id=None,
        progress=None,
    ) -> Comparison:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if sampling != "uniform":
            raise ValueError("use selections=[SelectionSpec.manual(...)] for manual sampling")
        if execution_mode not in ("single", "benchmark"):
            raise ValueError("execution_mode must be single or benchmark")
        if execution_mode == "single":
            warmup, repeats = 0, 1
        notify = progress or (lambda message: None)
        for name, value, minimum in (("warmup", warmup, 0), ("repeats", repeats, 1)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if selections is None:
            budgets = list(frame_budgets)
            if not budgets or len(set(budgets)) != len(budgets):
                raise ValueError(
                    "provide at least one distinct frame budget; duplicates are invalid"
                )
            selections = [SelectionSpec.uniform(b) for b in budgets]
        else:
            selections = list(selections)
            if not selections or any(not isinstance(s, SelectionSpec) for s in selections):
                raise ValueError("selections must contain at least one SelectionSpec")
        notify("Probing video (setup; outside measured inference)")
        metadata = probe_video(video)
        evaluation_annotation = None
        if annotation is not None:
            from vidfit.evaluation.annotations import load_annotation

            evaluation_annotation = load_annotation(annotation, metadata, question, annotation_id)
        fixture = None
        if fixture_manifest is not None:
            fixture = json.loads(Path(fixture_manifest).read_text(encoding="utf-8"))
            interval = fixture.get("event_interval_s", [])
            if (
                fixture.get("fixture_schema") != "vidfit.synthetic.v1"
                or fixture.get("synthetic") is not True
                or fixture.get("video_sha256") != metadata["sha256"]
                or len(interval) != 2
                or not 0 <= interval[0] < interval[1] <= metadata["duration_s"]
            ):
                raise ValueError(
                    "fixture manifest must be synthetic, valid, and match the video SHA-256"
                )
        result = Comparison(
            run_id=str(uuid.uuid4()),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            backend=self.backend_name,
            video=metadata,
            question=question,
            settings={
                **self.options,
                "sampling": "uniform"
                if all(s.method == "uniform" for s in selections)
                else "specified",
                "selections": [s.to_dict() for s in selections],
                "execution_mode": execution_mode,
                "warmup": warmup,
                "repeats": repeats,
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": self.options["max_new_tokens"],
                    "use_cache": True,
                },
                "trial_policy": (
                    "exactly one measured invocation per selection; no warmup or hidden calls"
                    if execution_mode == "single"
                    else "one first invocation, then warmup and measured trials per selection"
                ),
                "sampling_mapping": "nearest actual PTS; ties choose earlier frame",
            },
            environment=environment(),
            model={},
            fixture=fixture,
        )
        backend, load_error, first_invocation = None, None, True
        for selection in selections:
            config = Configuration(selection.count, selection=selection.to_dict())
            result.configurations.append(config)
            try:
                notify(f"Previewing {selection.method} / {selection.count} frames (setup)")
                config.evidence, config.setup = sample_video(video, metadata, selection)
                self._validate_backend_selection(metadata, config.evidence)
                config.selected_frames = len(config.evidence)
                config.selection["target_timestamps_s"] = [
                    e.target_timestamp_s for e in config.evidence
                ]
                config.validation = {
                    "distinct_source_frames": True,
                    "trial_evidence_consistent": True,
                }
                if fixture:
                    left, right = fixture["event_interval_s"]
                    hits = [e.frame_index for e in config.evidence if left <= e.timestamp_s < right]
                    config.validation["fixture_evidence_coverage"] = {
                        "overlaps_event": bool(hits),
                        "overlapping_frame_indices": hits,
                        "definition": (
                            "selected actual PTS inside fixture interval; not model accuracy"
                        ),
                    }
                if backend is None and load_error is None:
                    try:
                        notify("Loading backend (setup; outside measured inference)")
                        if self.backend_name == "inspect":
                            backend = InspectionBackend()
                        else:
                            from vidfit.backends.qwen3vl import Qwen3VLBackend

                            backend = Qwen3VLBackend(**self.options)
                        result.model = backend.info
                    except Exception as exc:
                        load_error = describe_error(exc)
                        result.model = {
                            "identifier": self.options["model"],
                            "loading_error": load_error,
                        }
                phases = (
                    ["first_invocation"]
                    if first_invocation and execution_mode == "benchmark"
                    else []
                )
                phases += ["warmup"] * warmup + ["measured"] * repeats
                first_invocation = False
                for index, phase in enumerate(phases):
                    notify(f"{selection.method} / {selection.count} frames: {phase} {index + 1}")
                    trial = Trial(phase, index)
                    config.trials.append(trial)
                    trial.metrics = dict.fromkeys(
                        (*INFERENCE_METRICS, "pipeline_s", "decode_sampling_s", "decoded_frames")
                    )
                    started = None
                    try:
                        if load_error:
                            raise RuntimeError(load_error)
                        backend.synchronize()
                        backend.reset_peak_memory()
                        started = perf_counter()
                        frames, decode_metrics = sample_video(video, metadata, selection)
                        trial.metrics.update(decode_metrics)
                        trial.evidence_hashes = [e.sha256_rgb for e in frames]

                        def identity(e):
                            return (
                                e.frame_index,
                                e.timestamp_s,
                                e.target_timestamp_s,
                                e.sha256_rgb,
                            )

                        if [identity(e) for e in frames] != [identity(e) for e in config.evidence]:
                            config.validation["trial_evidence_consistent"] = False
                            raise ValueError("decoded evidence changed between trials")
                        outcome = backend.run(frames, metadata, question)
                        trial.answer = outcome["answer"]
                        trial.processing = outcome["processing"]
                        trial.metrics.update(outcome["metrics"])
                        trial.unavailable_reasons.update(outcome.get("unavailable_reasons", {}))
                        backend.synchronize()
                    except Exception as exc:
                        trial.status, trial.error = "failed", describe_error(exc)
                        config.status, config.error = "failed", trial.error
                        trial.unavailable_reasons["answer"] = "trial failed; no model answer"
                        for key, value in trial.metrics.items():
                            if value is None:
                                trial.unavailable_reasons[key] = (
                                    "trial failed before measurement completed"
                                )
                    finally:
                        if started is not None:
                            trial.metrics["pipeline_s"] = perf_counter() - started
                config.summary = summarize(config.trials)
                config.validation["measured_trials_complete"] = (
                    config.summary["measured_trials"] == repeats
                )
            except Exception as exc:
                config.status, config.error = "failed", describe_error(exc)
        # References and rubrics are attached only AFTER inference. They are never backend inputs.
        result.annotation = evaluation_annotation
        if evaluation_annotation is not None:
            from vidfit.evaluation.annotations import evidence_coverage

            for config in result.configurations:
                config.validation["annotation_evidence_coverage"] = evidence_coverage(
                    evaluation_annotation, config.evidence
                )
        notify("Finished; exporting is outside measured inference")
        return result
