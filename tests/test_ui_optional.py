"""Real Gradio component check, separate from dependency-light CPU tests."""

import os

import pytest


@pytest.mark.ui
@pytest.mark.skipif(os.getenv("VIDFIT_RUN_UI_TEST") != "1", reason="opt-in Gradio component check")
def test_upload_preprocess_does_not_transcode(short_video, monkeypatch, tmp_path):
    import gradio as gr
    import gradio.components.video as video_module
    from gradio.data_classes import FileData

    from vidfit import VideoAnalyzer
    from vidfit.ui import build_app

    def forbidden(*args, **kwargs):
        raise AssertionError("upload must not invoke an external FFmpeg executable")

    monkeypatch.setattr(video_module, "FFmpeg", forbidden)
    app = build_app(VideoAnalyzer(backend="inspect"), output_root=tmp_path)
    videos = [block for block in app.blocks.values() if isinstance(block, gr.Video)]
    assert len(videos) == 1
    assert videos[0].preprocess(FileData(path=str(short_video))) == str(short_video)
    assert videos[0].include_audio and videos[0].format is None
    assert app.analytics_enabled is False
    app.close()
    app.vidfit_worker.close()
