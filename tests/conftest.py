import av
import pytest
from PIL import Image

from vidfit.demo import generate_demo


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    return generate_demo(tmp_path_factory.mktemp("fixture") / "brief-error.mp4")


@pytest.fixture
def short_video(tmp_path):
    path = tmp_path / "short.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for color in ("red", "blue"):
            for packet in stream.encode(
                av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
