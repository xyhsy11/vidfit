import json
import subprocess
import sys

import pytest

from vidfit import VideoAnalyzer
from vidfit.core.profiling import summarize
from vidfit.core.schemas import Trial


def test_inspection_api_and_export(demo, tmp_path):
    result = VideoAnalyzer(backend="inspect").compare(
        demo[0],
        "What error message appeared?",
        [4, 16],
        fixture_manifest=demo[1],
    )
    assert result.complete
    a, b = result.configurations
    assert not a.validation["fixture_evidence_coverage"]["overlaps_event"]
    assert b.validation["fixture_evidence_coverage"]["overlaps_event"]
    assert [t.phase for t in a.trials] == ["first_invocation", "warmup"] + ["measured"] * 3
    assert [t.phase for t in b.trials] == ["warmup"] + ["measured"] * 3
    for config in result.configurations:
        assert config.summary["measured_trials"] == 3
        assert config.summary["metrics"]["generation_s"] is None
        for trial in config.trials:
            assert trial.answer is None
            assert trial.metrics["visual_tokens"] is None
            assert "not run" in trial.unavailable_reasons["visual_tokens"]
            assert trial.metrics["pipeline_s"] >= trial.metrics["decode_sampling_s"] > 0
            assert trial.evidence_hashes == [f.sha256_rgb for f in config.evidence]
    data = json.loads(result.export_json(tmp_path / "comparison.json").read_text())
    assert data["schema_version"] == "1.0.0"
    assert data["video"]["filename"] == "brief-error.mp4"
    assert data["model"]["identifier"] is None
    markup = result.export_html(tmp_path / "comparison.html").read_text(encoding="utf-8")
    assert "Inspection only" in markup
    assert "data:image/png;base64," in markup
    assert "https://" not in markup


def test_warmups_and_failures_excluded_from_statistics():
    trials = [
        Trial("first_invocation", 0, metrics={"pipeline_s": 100}),
        Trial("warmup", 1, metrics={"pipeline_s": 80}),
        Trial("measured", 2, metrics={"pipeline_s": 1}),
        Trial("measured", 3, metrics={"pipeline_s": 3}),
        Trial("measured", 4, status="failed", metrics={"pipeline_s": 90}),
    ]
    stats = summarize(trials)["metrics"]["pipeline_s"]
    assert stats == {"n": 2, "median": 2, "min": 1, "max": 3, "stdev": 2**0.5}


def test_partial_failure_preserves_success(short_video):
    result = VideoAnalyzer(backend="inspect").compare(
        short_video,
        "Describe",
        [1, 3, 2],
        warmup=0,
        repeats=1,
    )
    assert not result.complete
    assert [c.status for c in result.configurations] == ["ok", "failed", "ok"]
    assert result.configurations[-1].trials[-1].status == "ok"


@pytest.mark.parametrize(
    "options",
    [
        {"question": ""},
        {"question": "  "},
        {"frame_budgets": []},
        {"frame_budgets": [4, 4]},
        {"frame_budgets": [0]},
        {"warmup": -1},
        {"repeats": 0},
        {"repeats": True},
        {"sampling": "scene"},
    ],
)
def test_invalid_comparison(short_video, options):
    kwargs = {"question": "Describe", **options}
    with pytest.raises(ValueError):
        VideoAnalyzer(backend="inspect").compare(short_video, **kwargs)


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "fake"},
        {"device": "auto"},
        {"dtype": "auto"},
        {"max_pixels": 0},
        {"max_pixels": True},
        {"max_new_tokens": 0},
        {"revision": ""},
    ],
)
def test_invalid_analyzer(options):
    with pytest.raises(ValueError):
        VideoAnalyzer(**options)


def test_manifest_mismatch(demo, short_video):
    with pytest.raises(ValueError, match="match the video"):
        VideoAnalyzer(backend="inspect").compare(short_video, "Describe", fixture_manifest=demo[1])


def test_html_escaping(short_video):
    result = VideoAnalyzer(backend="inspect").compare(
        short_video,
        '<script>alert("question")</script>',
        [1],
        warmup=0,
        repeats=1,
    )
    result.video["filename"] = '<img src=x onerror="evil()">'
    result.configurations[0].trials[-1].answer = '<svg onload="answer()">' + "long answer " * 2000
    from vidfit.evaluation.reports import render_html

    markup = render_html(result)
    assert "<script>alert" not in markup
    assert "<svg onload" not in markup
    assert "<img src=x" not in markup
    assert "&lt;script&gt;" in markup
    assert "&lt;svg onload=" in markup


def test_cli_inspect_without_torch(short_video, tmp_path):
    code = """
import sys
class BlockModels:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('Inspection imported model dependencies')
sys.meta_path.insert(0, BlockModels())
from vidfit.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "compare",
            "--video",
            str(short_video),
            "--question",
            "Describe",
            "--frame-budgets",
            "2",
            "--backend",
            "inspect",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.returncode == 0, process.stderr
    assert "Inspection only" in process.stdout
    assert (tmp_path / "comparison.html").is_file()


def test_cli_incomplete_exit_code(short_video, tmp_path):
    from vidfit.cli import main

    assert (
        main(
            [
                "compare",
                "--video",
                str(short_video),
                "--question",
                "Describe",
                "--frame-budgets",
                "2",
                "3",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
