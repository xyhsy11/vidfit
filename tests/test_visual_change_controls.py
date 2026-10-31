"""Visual-change selections use the same CLI and UI preview/run paths."""

import json
import os

import pytest

from vidfit import SelectionSpec, VideoAnalyzer, cli
from vidfit.ui import choose_selections, selection_from_controls


def test_visual_change_controls_preserve_method_and_budget():
    assert selection_from_controls("visual_change", 4.0, "unused manual field") == (
        SelectionSpec.visual_change(4)
    )
    assert choose_selections("Compare settings", "visual_change", 4, "", "uniform", 16, "") == [
        SelectionSpec.visual_change(4),
        SelectionSpec.uniform(16),
    ]
    with pytest.raises(ValueError, match="integer frame budget"):
        selection_from_controls("visual_change", 4.5, "")


@pytest.mark.parametrize("budgets,expected", [([], [4, 8, 16]), (["--frame-budgets", "4"], [4])])
def test_visual_change_cli_forwards_only_explicit_selection(budgets, expected, monkeypatch):
    calls = []

    def compare(self, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(VideoAnalyzer, "compare", compare)
    monkeypatch.setattr(cli, "export", lambda result, output: 0)
    assert (
        cli.main(
            [
                "compare",
                "--video",
                "example.mp4",
                "--question",
                "What changed?",
                "--sampling",
                "visual_change",
                "--output-dir",
                "unused",
                *budgets,
            ]
        )
        == 0
    )
    assert calls[0]["selections"] == [SelectionSpec.visual_change(n) for n in expected]


def test_visual_change_cli_rejects_ambiguous_manual_timestamps(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("ambiguous selections must fail before inference")

    monkeypatch.setattr(VideoAnalyzer, "compare", forbidden)
    assert (
        cli.main(
            [
                "compare",
                "--video",
                "example.mp4",
                "--question",
                "What changed?",
                "--sampling",
                "visual_change",
                "--timestamps",
                "0",
                "1",
                "--output-dir",
                "unused",
            ]
        )
        == 2
    )
    assert "cannot be combined with --timestamps" in capsys.readouterr().err


def test_visual_change_cli_inspection_exports_real_frames(short_video, tmp_path):
    assert (
        cli.main(
            [
                "compare",
                "--video",
                str(short_video),
                "--question",
                "What changed?",
                "--sampling",
                "visual_change",
                "--frame-budgets",
                "2",
                "--execution-mode",
                "single",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    data = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    config = data["configurations"][0]
    assert config["selection"]["method"] == "visual_change"
    assert [frame["frame_index"] for frame in config["evidence"]] == [0, 1]
    assert len(config["trials"]) == 1
    assert config["trials"][0]["answer"] is None


@pytest.mark.ui
@pytest.mark.skipif(os.getenv("VIDFIT_RUN_UI_TEST") != "1", reason="opt-in Gradio component check")
def test_visual_change_ui_choices_keep_frame_budget_visible(tmp_path):
    import gradio as gr

    from vidfit.ui import build_app

    app = build_app(VideoAnalyzer(backend="inspect"), output_root=tmp_path)
    try:
        for label, prefix in [("Selection A", "A"), ("Selection B", "B")]:
            method = next(
                block
                for block in app.blocks.values()
                if isinstance(block, gr.Radio) and block.label == label
            )
            budget = next(
                block
                for block in app.blocks.values()
                if isinstance(block, gr.Slider) and block.label == f"{prefix} · Frame budget"
            )
            assert "visual_change" in [value for _, value in method.choices]
            update = next(
                fn
                for fn in app.fns.values()
                if fn.inputs == [method] and len(fn.outputs) == 2 and fn.outputs[0] is budget
            )
            show_budget, show_times = update.fn("visual_change")
            assert show_budget["visible"] is True
            assert show_times["visible"] is False
    finally:
        app.close()
        app.vidfit_worker.close()
