import json
import math
from pathlib import Path

import pytest

from vidfit import SelectionSpec, VideoAnalyzer
from vidfit.backends.inspection import InspectionBackend
from vidfit.core.video import probe_video, sample_video
from vidfit.evaluation.annotations import load_annotation, record_judgment
from vidfit.ui import UIController, identity


@pytest.mark.parametrize("times", [[], [math.nan], [math.inf], [True], ["1"], [0, 0]])
def test_invalid_manual_spec(times):
    with pytest.raises(ValueError):
        SelectionSpec.manual(times)


@pytest.mark.parametrize("times", [[-0.1, 0.5], [0, 1], [2, 3]])
def test_manual_outside_duration(short_video, times):
    with pytest.raises(ValueError, match="within"):
        sample_video(short_video, probe_video(short_video), SelectionSpec.manual(times))


def test_manual_nearest_tie_chronology_and_duplicate_frame(short_video):
    metadata = probe_video(short_video)
    frames, _ = sample_video(short_video, metadata, SelectionSpec.manual([0.9, 0.25]))
    assert [(f.target_timestamp_s, f.timestamp_s, f.frame_index) for f in frames] == [
        (0.25, 0, 0),
        (0.9, 0.5, 1),
    ]
    with pytest.raises(ValueError, match="distinct frames"):
        sample_video(short_video, metadata, SelectionSpec.manual([0.01, 0.1]))


def test_manual_error_selection_and_exports(demo, tmp_path):
    result = VideoAnalyzer(backend="inspect").compare(
        demo[0],
        "What error message appeared?",
        execution_mode="single",
        selections=[SelectionSpec.uniform(4), SelectionSpec.manual([7.5, 24.375, 37.5, 52.5])],
        fixture_manifest=demo[1],
    )
    assert result.complete
    a, b = result.configurations
    assert not a.validation["fixture_evidence_coverage"]["overlaps_event"]
    assert b.validation["fixture_evidence_coverage"]["overlaps_event"]
    data = json.loads(result.export_json(tmp_path / "result.json").read_text(encoding="utf-8"))
    manual = data["configurations"][1]
    assert manual["selection"]["method"] == "manual"
    assert manual["selection"]["requested_timestamps_s"] == [7.5, 24.375, 37.5, 52.5]
    assert [e["frame_index"] for e in manual["evidence"]] == [60, 195, 300, 420]
    assert all(e["sha256_rgb"] for e in manual["evidence"])
    assert manual["trials"][0]["evaluation"]["method"] == "unreviewed"
    markup = result.export_html(tmp_path / "result.html").read_text(encoding="utf-8")
    assert markup.index("Model answer") < markup.index("Advanced measurements")
    assert "Run status: completed" in markup
    assert "Answer correctness: Not evaluated" in markup
    assert "Inputs provided to the model" in markup


def fake_backend(monkeypatch):
    calls = []

    def run(self, frames, metadata, question):
        calls.append((frames, dict(metadata), question))
        return {"answer": "red then blue", "metrics": {}, "processing": {}}

    monkeypatch.setattr(InspectionBackend, "run", run)
    return calls


@pytest.mark.parametrize("mode,expected", [("single", 2), ("benchmark", 9)])
def test_exact_invocations_and_preview_identity(short_video, monkeypatch, tmp_path, mode, expected):
    calls = fake_backend(monkeypatch)
    analyzer = VideoAnalyzer(backend="inspect")
    spec = SelectionSpec.manual([0, 0.9])
    _, preview, _ = analyzer.preview(short_video, spec)
    assert not calls
    result = analyzer.compare(
        short_video, "Describe", selections=[spec, spec], execution_mode=mode, warmup=1, repeats=3
    )
    assert result.complete
    assert len(calls) == expected
    for config in result.configurations:
        assert identity(config.evidence) == identity(preview)
        assert all(t.evidence_hashes == [f.sha256_rgb for f in preview] for t in config.trials)
    if mode == "single":
        assert all([t.phase for t in c.trials] == ["measured"] for c in result.configurations)
        assert result.settings["warmup"] == 0 and result.settings["repeats"] == 1


def annotation_file(short_video, tmp_path):
    metadata = probe_video(short_video)
    item = {
        "schema": "vidfit.annotation.v1",
        "id": "short-colors",
        "video": {"filename": short_video.name, "sha256": metadata["sha256"]},
        "source": {"uri": "generated:test", "license": "MIT", "attribution": "VidFit tests"},
        "question": "Describe",
        "accepted_answers": ["SECRET_REFERENCE"],
        "rubric": "SECRET_RUBRIC",
        "intervals_s": [[0, 0.4]],
        "review": {"status": "unreviewed"},
    }
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(item), encoding="utf-8")
    return path, metadata


def test_annotation_isolation_and_coverage(short_video, tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    path, _ = annotation_file(short_video, tmp_path)
    result = VideoAnalyzer(backend="inspect").compare(
        short_video,
        "Describe",
        [2],
        execution_mode="single",
        annotation=path,
    )
    assert len(calls) == 1
    _, metadata, question = calls[0]
    assert "SECRET" not in json.dumps(metadata) + question
    assert "annotation" not in metadata and "rubric" not in metadata
    assert result.annotation["rubric"] == "SECRET_RUBRIC"
    assert result.configurations[0].validation["annotation_evidence_coverage"]["overlaps_event"]
    assert result.configurations[0].trials[0].evaluation["verdict"] == "not_evaluated"


def test_annotation_identity_and_intervals(short_video, tmp_path):
    path, metadata = annotation_file(short_video, tmp_path)
    with pytest.raises(ValueError, match="question"):
        load_annotation(path, metadata, "Other question")
    with pytest.raises(ValueError, match="SHA-256"):
        load_annotation(path, {**metadata, "sha256": "0" * 64}, "Describe")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["intervals_s"] = [[0, 2]]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duration"):
        load_annotation(path, metadata, "Describe")


@pytest.mark.parametrize("verdict", ["correct", "partially_correct", "incorrect", "unanswerable"])
@pytest.mark.parametrize("method", ["agent", "automated", "human"])
def test_explicit_judgments(short_video, monkeypatch, verdict, method):
    fake_backend(monkeypatch)
    result = VideoAnalyzer(backend="inspect").compare(
        short_video, "Describe", [2], execution_mode="single"
    )
    record_judgment(result, 0, 0, verdict, method, "Test rationale", "test-check")
    evaluation = result.to_dict()["configurations"][0]["trials"][0]["evaluation"]
    assert evaluation["verdict"] == verdict and evaluation["method"] == method
    assert evaluation["answer_sha256"]


def test_cannot_judge_no_answer(short_video):
    result = VideoAnalyzer(backend="inspect").compare(
        short_video, "Describe", [2], execution_mode="single"
    )
    with pytest.raises(ValueError, match="no model answer"):
        record_judgment(result, 0, 0, "correct", "agent", "No answer", "test")


def test_ui_stale_preview_and_previous_result(short_video, tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    ui = UIController(VideoAnalyzer(backend="inspect"), tmp_path)
    controls = ["manual", 4, "0, 0.9", "uniform", 2, ""]
    snapshot, images, _ = ui.preview(short_video, "Answer once", *controls)
    assert len(images) == 2 and not calls
    with pytest.raises(ValueError, match="Preview"):
        ui.run(short_video, "Describe", "Answer once", controls, None)
    changed = controls.copy()
    changed[2] = "0, 0.5"
    with pytest.raises(ValueError, match="Selection changed"):
        ui.run(short_video, "Describe", "Answer once", changed, snapshot)
    first = ui.run(short_video, "Describe", "Answer once", controls, snapshot)
    second = ui.run(short_video, "Describe", "Answer once", controls, snapshot, first[0])
    assert len(calls) == 2
    assert "red then blue" in second[1] and "red then blue" in second[3]
    assert Path(first[5]).is_file() and Path(first[6]).is_file()


def test_qwen_two_frame_minimum_before_model_load(short_video):
    result = VideoAnalyzer().compare(
        short_video, "Describe", selections=[SelectionSpec.manual([0])], execution_mode="single"
    )
    assert not result.complete and "at least 2" in result.configurations[0].error
    assert not result.model


def test_manual_cli_no_uniform_fallback(short_video, tmp_path):
    from vidfit.cli import main

    args = [
        "compare",
        "--video",
        str(short_video),
        "--question",
        "Describe",
        "--output-dir",
        str(tmp_path),
    ]
    assert main([*args, "--sampling", "manual"]) == 2
    assert main([*args, "--timestamps", "0", "0.9", "--execution-mode", "single"]) == 0
    data = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert len(data["configurations"]) == 1
    assert data["configurations"][0]["selection"]["method"] == "manual"


def test_starter_metadata_is_valid_and_has_real_and_synthetic_clips():
    from vidfit.evaluation.annotations import read_collection

    items = read_collection(Path(__file__).parents[1] / "examples/evaluation/starter.json")
    assert len(items) >= 10
    assert len({i["video"]["sha256"] for i in items}) >= 5
    assert {i["video"]["synthetic"] for i in items} == {True, False}
    assert all(i["review"]["status"] == "agent_visual_review" for i in items)


def test_failed_single_run_is_exportable(short_video, monkeypatch, tmp_path):
    def out_of_memory(*args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(InspectionBackend, "run", out_of_memory)
    result = VideoAnalyzer(backend="inspect").compare(
        short_video, "Describe", [2], execution_mode="single"
    )
    assert not result.complete
    trial = result.configurations[0].trials[0]
    assert trial.status == "failed" and trial.answer is None
    assert "Reduce the frame budget" in trial.error
    assert trial.evaluation["verdict"] == "not_evaluated"
    assert result.export_json(tmp_path / "failed.json").is_file()


def test_local_worker_preserves_selection_and_single_policy(short_video, tmp_path):
    from vidfit.ui_worker import LocalWorker

    worker = LocalWorker(UIController(VideoAnalyzer(backend="inspect"), tmp_path))
    controls = ["manual", 4, "0, 0.9", "uniform", 2, ""]
    try:
        assert worker.start()
        snapshot, images, _ = worker.preview(short_video, "Answer once", *controls)
        result, *_ = worker.run(short_video, "Describe", "Answer once", controls, snapshot)
        assert result.complete and len(images) == 2
        config = result.configurations[0]
        assert config.selection["method"] == "manual"
        assert [t.phase for t in config.trials] == ["measured"]
        assert identity(config.evidence) == snapshot["identities"][0]
    finally:
        worker.close()


def test_ui_rejects_non_object_reference_before_backend(short_video, tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    ui = UIController(VideoAnalyzer(backend="inspect"), tmp_path)
    controls = ["manual", 4, "0, 0.9", "uniform", 2, ""]
    snapshot, *_ = ui.preview(short_video, "Answer once", *controls)
    path = tmp_path / "bad-reference.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON annotation object"):
        ui.run(short_video, "Describe", "Answer once", controls, snapshot, reference=path)
    assert not calls
