
import base64

import pytest

import frames
from frames import FrameError, decode_frame


def test_reads_plain_jpeg(frame):


    image = decode_frame(frame(width=8, height=6))

    assert image.shape == (6, 8, 3)
    assert image.dtype.name == "uint8"
    assert image.ndim == 3


def test_reads_browser_data_url(frame):


   

    image = decode_frame(frame(data_url=True))
    assert image.shape == (6, 8, 3)
    assert image[0, 0].tolist() != [0, 0, 0]
    assert image.size == 6 * 8 * 3


def test_reads_png(frame):
    """A PNG comes back with three colour channels."""

    image = decode_frame(frame(fmt="PNG"))

    assert image.shape[2] == 3
    assert image.shape[:2] == (6, 8)
    assert image[0, 0].tolist() == [200, 120, 40]


def test_rejects_empty_or_wrong_type():
    """Empty values and non-strings are rejected."""

    for bad in ("", "   ", None, 123, b"abc"):
        with pytest.raises(FrameError):
            decode_frame(bad)


def test_rejects_bad_base64():
    """Text that is not base64 is rejected."""

    for bad in ("not base64!!", "data:image/jpeg;base64,@@@"):
        with pytest.raises(FrameError, match="base64"):
            decode_frame(bad)


def test_rejects_empty_data_url():
    """A data URL with nothing in it is rejected."""

    with pytest.raises(FrameError, match="empty"):
        decode_frame("data:image/jpeg;base64,")

    with pytest.raises(FrameError, match="empty"):
        decode_frame("data:")


def test_rejects_text_that_is_not_a_picture():
    """Base64 that is not a picture is rejected."""

    text = base64.b64encode(b"hello, this is not an image").decode()

    with pytest.raises(FrameError, match="readable picture"):
        decode_frame(text)


def test_rejects_large_images(monkeypatch, frame):
    """Images over the size limit are rejected."""

    monkeypatch.setattr(frames, "MAX_FRAME_BYTES", 10)

    with pytest.raises(FrameError, match="too large"):
        decode_frame(frame())

    assert frames.MAX_FRAME_BYTES == 10
