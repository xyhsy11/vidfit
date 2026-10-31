"""Experiment accounting and explicit judgments with CPU-only backend doubles."""

import hashlib
import itertools
import json
import sys
from types import ModuleType

import pytest

from vidfit import SelectionSpec, VideoAnalyzer, experiment
from vidfit.backends.inspection import InspectionBackend
from vidfit.core.video import probe_video, sample_video


@pytest.fixture
def experiment_fixture(demo, tmp_path, monkeypatch):
    metadata = probe_video(demo[0])
    supported_question = "What appeared?"
    unsupported_question = "What is the operator's name?"
    annotations = []
    for question_id, question, answerability in (
        ("supported", supported_question, "supported"),
        ("unsupported", unsupported_question, "unsupported"),
    ):
        annotations.append(
            {
                "schema": "vidfit.annotation.v1",
                "id": question_id,
                "video": {"filename": demo[0].name, "sha256": metadata["sha256"]},
                "source": {
                    "uri": "synthetic://unit-test",
                    "license": "MIT",
                    "attribution": "tests",
                },
                "question": question,
                "answerability": answerability,
                "accepted_answers": ["SECRET_REFERENCE"],
                "rubric": "SECRET_RUBRIC",
                "intervals_s": [[0, 1]] if answerability == "supported" else [],
                "review": {"status": "unreviewed"},
            }
        )
    collection = tmp_path / "collection.json"
    collection.write_text(
        json.dumps(
            {
                "schema": "vidfit.collection.v1",
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    # The presentation layer has its own tests; keep this fixture focused on execution.
    renderer = ModuleType("vidfit.evaluation.experiment_report")
    renderer.render_experiment = lambda summary: "<html>Experiment unit-test summary</html>"
    monkeypatch.setitem(sys.modules, renderer.__name__, renderer)
    calls, loads = [], []

    class BackendDouble(InspectionBackend):
        info = {"identifier": "inspection-test-double", "loading_s": 0}
        fail_once = False

        def run(self, frames, source_metadata, question):
            calls.append((len(frames), source_metadata, question))
            if self.fail_once and question == unsupported_question and len(frames) == 16:
                self.fail_once = False
                raise RuntimeError("injected measured trial failure")
            return {
                "answer": "visible event" if question == supported_question else "not shown",
                "processing": {"test_double": True},
                "metrics": {
                    "preprocessing_s": 0.01,
                    "generation_s": 0.02,
                    "generated_tokens": 2,
                    "visual_tokens": len(frames) * 4,
                },
            }

    backend = BackendDouble()

    def load(analyzer):
        loads.append(analyzer)
        return backend

    monkeypatch.setattr(experiment, "_load_backend", load)
    return {
        "collection": collection,
        "video_dir": demo[0].parent,
        "output_dir": tmp_path / "experiment",
        "backend": backend,
        "calls": calls,
        "loads": loads,
    }


def run_fixture(fixture):
    return experiment.run_selection_experiment(
        fixture["collection"],
        fixture["video_dir"],
        fixture["output_dir"],
        analyzer=VideoAnalyzer(backend="inspect"),
        repeats=2,
        progress=lambda message: None,
    )


def explicit_judgments(output_dir):
    packet = json.loads((output_dir / "answers-for-review.json").read_text(encoding="utf-8"))
    return packet, [
        {
            "answer_id": item["answer_id"],
            "verdict": "correct" if item["answerability"] == "supported" else "unanswerable",
            "method": "agent",
            "reviewer": "unit-test reviewer",
            "rationale": "Explicit fixture judgment; not a real model-quality observation.",
        }
        for item in packet
    ]


def test_end_to_end_clock_includes_probe_selector_and_preprocessing(demo, monkeypatch):
    metadata = probe_video(demo[0])
    frames, _ = sample_video(demo[0], metadata, 4)
    clock = [0.0]
    monkeypatch.setattr(experiment, "perf_counter", lambda: clock[0])

    def probe(video):
        clock[0] += 2
        return metadata

    def sample(video, source_metadata, selection):
        assert selection == SelectionSpec.visual_change(4)
        clock[0] += 3
        return frames, {
            "decode_sampling_s": 3,
            "visual_change_scan_s": 2,
            "selection_trace": {"selected": ["fixture"]},
        }

    class TimedBackend(InspectionBackend):
        def run(self, supplied_frames, source_metadata, question):
            clock[0] += 7
            return {
                "answer": "fixture answer",
                "processing": {},
                "metrics": {
                    "preprocessing_s": 4,
                    "device_transfer_s": 1,
                    "generation_s": 1,
                    "output_decode_s": 1,
                },
            }

    monkeypatch.setattr(experiment, "probe_video", probe)
    monkeypatch.setattr(experiment, "sample_video", sample)
    trial, _, _ = experiment._run_trial(
        VideoAnalyzer(backend="inspect"),
        TimedBackend(),
        demo[0],
        "Question",
        SelectionSpec.visual_change(4),
        "measured",
        0,
    )
    assert trial.status == "ok"
    assert trial.metrics["end_to_end_s"] == 12
    assert trial.metrics["input_preparation_s"] == 10
    assert trial.metrics["generation_s"] == 1
    assert "selection_trace" not in trial.metrics
    assert trial.processing["selection_trace"] == {"selected": ["fixture"]}


def test_one_backend_recorded_warmups_rotated_order_and_reference_isolation(experiment_fixture):
    fixture = experiment_fixture
    summary = run_fixture(fixture)
    manifest = summary["experiment"]
    assert len(fixture["loads"]) == 1
    assert len(fixture["calls"]) == 2 + 2 * 3 * 2
    assert len(manifest["warmups"]) == 2
    assert [trial["phase"] for trial in manifest["warmups"]] == ["warmup", "warmup"]
    assert [count for count, _, _ in fixture["calls"][:2]] == [4, 16]
    orders = list(itertools.permutations(experiment.METHODS))
    assert [entry["method"] for entry in manifest["schedule"]] == [
        method for order in orders[:4] for method in order
    ]
    for _, metadata, question in fixture["calls"]:
        assert question in ("What appeared?", "What is the operator's name?")
        text = json.dumps(metadata)
        assert "SECRET" not in text and "accepted_answers" not in text and "intervals_s" not in text
    for method in summary["methods"]:
        assert method["strict_success_rate"] is None
        assert method["supported_correct"] is None
        assert method["unsupported_abstentions"] is None
        assert method["verdict_counts"]["not_evaluated"] == 4
    assert len(list(fixture["output_dir"].glob("case-*/comparison.raw.json"))) == 2


def test_explicit_blinded_judgments_score_supported_answers_and_abstentions(experiment_fixture):
    fixture = experiment_fixture
    run_fixture(fixture)
    packet, judgments = explicit_judgments(fixture["output_dir"])
    assert len(packet) == 2  # Identical repeated answers are reviewed once per question.
    assert all("method" not in item and "configuration" not in item for item in packet)
    summary = experiment.apply_experiment_judgments(fixture["output_dir"], judgments)
    for method in summary["methods"]:
        assert method["supported_trials"] == method["supported_correct"] == 2
        assert method["unsupported_trials"] == method["unsupported_abstentions"] == 2
        assert method["strict_success_rate"] == 1
        assert method["judgment_methods"] == ["agent"]
    report_path = fixture["output_dir"] / "case-000/comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for config in report["configurations"]:
        for trial in config["trials"]:
            assert (
                trial["evaluation"]["answer_sha256"]
                == hashlib.sha256(trial["answer"].encode("utf-8")).hexdigest()
            )
    raw = json.loads(report_path.with_name("comparison.raw.json").read_text(encoding="utf-8"))
    assert raw["configurations"][0]["trials"][0]["evaluation"]["verdict"] == "not_evaluated"


def test_answer_hash_mismatch_rejects_judgments_without_partial_writes(experiment_fixture):
    fixture = experiment_fixture
    run_fixture(fixture)
    _, judgments = explicit_judgments(fixture["output_dir"])
    judgments[0]["answer_id"] = "0" * 64
    paths = list(fixture["output_dir"].glob("case-*/comparison.json"))
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="missing explicit judgment"):
        experiment.apply_experiment_judgments(fixture["output_dir"], judgments)
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not (fixture["output_dir"] / "judgments.json").exists()


def test_failed_trials_remain_in_accuracy_denominator(experiment_fixture):
    fixture = experiment_fixture
    fixture["backend"].fail_once = True
    run_fixture(fixture)
    _, judgments = explicit_judgments(fixture["output_dir"])
    summary = experiment.apply_experiment_judgments(fixture["output_dir"], judgments)
    assert summary["experiment"]["status"] == "failed"
    method = next(item for item in summary["methods"] if item["method"] == "uniform_16")
    assert method["failures"] == 1
    assert method["supported_trials"] + method["unsupported_trials"] == 4
    assert method["supported_correct"] == 2 and method["unsupported_abstentions"] == 1
    assert method["strict_success_rate"] == 0.75
    assert method["metrics"]["end_to_end_s"]["n"] == 3


def test_existing_output_is_not_overwritten_or_model_loaded(experiment_fixture):
    fixture = experiment_fixture
    fixture["output_dir"].mkdir()
    marker = fixture["output_dir"] / "keep.txt"
    marker.write_text("Existing evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        run_fixture(fixture)
    assert marker.read_text(encoding="utf-8") == "Existing evidence"
    assert not fixture["loads"] and not fixture["calls"]


def test_collection_optional_reference_fields_use_schema_defaults(experiment_fixture):
    fixture = experiment_fixture
    path = fixture["collection"]
    collection = json.loads(path.read_text(encoding="utf-8"))
    del collection["annotations"][0]["answerability"]
    del collection["annotations"][0]["accepted_answers"]
    path.write_text(json.dumps(collection), encoding="utf-8")
    summary = run_fixture(fixture)
    packet = json.loads(
        (fixture["output_dir"] / "answers-for-review.json").read_text(encoding="utf-8")
    )
    supported = next(item for item in packet if item["question_id"] == "supported")
    assert supported["answerability"] == "supported" and supported["accepted_answers"] == []
    assert all(method["supported_trials"] == 2 for method in summary["methods"])
