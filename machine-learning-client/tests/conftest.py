
import base64
import io

import pytest
from PIL import Image

from app import create_app


def make_frame(width=8, height=6, fmt="JPEG", data_url=False):
    """makes a small test image as base64."""

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode()

    if data_url:
        return f"data:image/{fmt.lower()};base64,{encoded}"

    return encoded


@pytest.fixture(name="frame")
def fixture_frame():
    """Give tests the image maker."""
    return make_frame


@pytest.fixture(name="readings")
def fixture_readings():
    """Fake detector results, used one per frame."""
    return []


@pytest.fixture(name="fake_analyzer")
def fixture_fake_analyzer(readings):


  
    calls = []

    def analyzer(image):
        calls.append(image)

        if readings:
            return readings.pop(0)

        return {"face_found": True, "eyes_open": True}

    analyzer.calls = calls
    return analyzer


@pytest.fixture(name="client")
def fixture_client(fake_analyzer):

 

    app = create_app(analyzer=fake_analyzer)
    app.config["TESTING"] = True

    return app.test_client()
