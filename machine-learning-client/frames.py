"""reads and relays information the camera images the web app sends."""

import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_FRAME_BYTES = 2_000_000


class FrameError(ValueError):
    """the image couldnot be read."""


def decode_frame(data):
 

    if not isinstance(data, str) or not data.strip():
        raise FrameError("image must be a non-empty base64 string")

    if data.startswith("data:"):
        data = data.split(",", 1)[1] if "," in data else ""

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameError("image is not valid base64") from exc

    if not raw:
        raise FrameError("image is empty")

    if len(raw) > MAX_FRAME_BYTES:
        raise FrameError("image is too large")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise FrameError("image is not a readable picture") from exc
