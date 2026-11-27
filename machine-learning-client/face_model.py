"""Finds and downloads the face landmark model the blink detector needs.

The model is a single file of about 4 MB that is not stored in git.
Download it once before starting the ML client:

    python face_model.py

It is saved to models/face_landmarker.task next to this file. Set the
FACE_MODEL_PATH environment variable to keep it somewhere else.
"""

import os
import urllib.error
import urllib.request
from pathlib import Path

MODEL_URLS = (
    "https://huggingface.co/litert-community/MediaPipe-FaceLandmarker"
    "/resolve/main/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker"
    "/face_landmarker/float16/1/face_landmarker.task",
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
# A real model is millions of bytes. Anything tiny is an error page, not a model.
MIN_MODEL_BYTES = 100_000
DOWNLOAD_TIMEOUT_SECONDS = 60


def model_path():
    """Return where the model file lives, from FACE_MODEL_PATH or the default."""
    configured = os.environ.get("FACE_MODEL_PATH", "").strip()
    return Path(configured) if configured else DEFAULT_MODEL_PATH


def fetch(url, opener=urllib.request.urlopen):
    """Return the bytes at url, or raise OSError if it is not a usable model."""
    with opener(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        data = response.read()
    if len(data) < MIN_MODEL_BYTES:
        raise OSError(f"{url} returned only {len(data)} bytes")
    return data


def download(destination=None, urls=MODEL_URLS, opener=urllib.request.urlopen):
    """Download the model to destination unless it is already there.

    Each address in urls is tried in order until one works.
    Returns the path of the model file. Raises OSError if every address fails.
    """
    destination = Path(destination) if destination else model_path()
    if destination.is_file():
        return destination

    problems = []
    for url in urls:
        try:
            data = fetch(url, opener)
        except (OSError, urllib.error.URLError) as exc:
            problems.append(f"{url}: {exc}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    raise OSError("could not download the face model. " + "; ".join(problems))


def main():
    """Download the model and say where it went."""
    path = download()
    print(f"face model ready at {path}")


if __name__ == "__main__":
    main()
