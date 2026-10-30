"""Sequential comparisons with a reused model and independently decoded trials."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from vidfit.backends.inspection import InspectionBackend
from vidfit.core.profiling import INFERENCE_METRICS, environment, summarize
from vidfit.core.sampling import midpoint_targets
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

    def compare(
        self,
        video,
        question,
        frame_budgets=(4, 8, 16),
        sampling="uniform",
        warmup=1,
        repeats=3,
        fixture_manifest=None,
    ) -> Comparison:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if sampling != "uniform":
            raise ValueError("only uniform midpoint sampling is supported")
        for name, value, minimum in (("warmup", warmup, 0), ("repeats", repeats, 1)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        budgets = list(frame_budgets)
        if not budgets or len(set(budgets)) != len(budgets):
            raise ValueError("provide at least one distinct frame budget; duplicates are invalid")
        for budget in budgets:
            midpoint_targets(1, budget)
        metadata = probe_video(video)
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
                "sampling": sampling,
                "warmup": warmup,
                "repeats": repeats,
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": self.options["max_new_tokens"],
                    "use_cache": True,
                },
                "trial_policy": "one first invocation, then warmup and measured trials per budget",
                "sampling_mapping": "nearest actual PTS to midpoint; ties choose earlier frame",
            },
            environment=environment(),
            model={},
            fixture=fixture,
        )
        backend, load_error, first_invocation = None, None, True
        for budget in budgets:
            config = Configuration(budget)
            result.configurations.append(config)
            try:
                config.evidence, _ = sample_video(video, metadata, budget)
                config.selected_frames = len(config.evidence)
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
                        if self.backend_name == "inspect":
                            backend = InspectionBackend()
                        else:
                            from vidfit.backends.qwen3vl import Qwen3VLBackend

                            backend = Qwen3VLBackend(**self.options)
                        result.model = backend.info
                    except Exception as exc:
                        load_error = f"{type(exc).__name__}: {exc}"
                        result.model = {
                            "identifier": self.options["model"],
                            "loading_error": load_error,
                        }
                phases = ["first_invocation"] if first_invocation else []
                phases += ["warmup"] * warmup + ["measured"] * repeats
                first_invocation = False
                for index, phase in enumerate(phases):
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
                        frames, decode_metrics = sample_video(video, metadata, budget)
                        trial.metrics.update(decode_metrics)
                        trial.evidence_hashes = [e.sha256_rgb for e in frames]
                        if trial.evidence_hashes != [e.sha256_rgb for e in config.evidence]:
                            config.validation["trial_evidence_consistent"] = False
                            raise ValueError("decoded evidence changed between trials")
                        outcome = backend.run(frames, metadata, question)
                        trial.answer = outcome["answer"]
                        trial.processing = outcome["processing"]
                        trial.metrics.update(outcome["metrics"])
                        trial.unavailable_reasons.update(outcome.get("unavailable_reasons", {}))
                        backend.synchronize()
                    except Exception as exc:
                        trial.status, trial.error = "failed", f"{type(exc).__name__}: {exc}"
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
                config.status, config.error = "failed", f"{type(exc).__name__}: {exc}"
        return result
