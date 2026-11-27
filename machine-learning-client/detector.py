"""Blink detector: finds a face in a camera frame and says if the eyes are open.

How it works:

1. MediaPipe's face landmarker finds 478 points on the face.
2. Six of those points outline each eye.
3. The eye aspect ratio (EAR) compares each eye's height to its width.
   An open eye gives roughly 0.25 to 0.35, and the ratio falls towards 0
   as the eye closes. It does not change when the face moves closer to or
   further from the camera, because height and width scale together.
4. If the average ratio of both eyes is under the threshold, the eyes are closed.

The face model is a separate file. Run `python face_model.py` once to get it.

To try the detector on a photo of a face:

    python detector.py photo.jpg
"""

import sys
import threading
from functools import lru_cache
from math import dist

import mediapipe as mp
import numpy as np
from PIL import Image

from face_model import model_path

POINTS_PER_EYE = 6
# Where the six points of each eye sit in MediaPipe's list of face points,
# in the order eye_aspect_ratio() expects:
# corner, upper lid, upper lid, opposite corner, lower lid, lower lid.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
# Below this average ratio the eyes count as closed.
CLOSED_EAR_THRESHOLD = 0.2

# One landmarker is shared by every request, so only one thread may use it at a time.
_DETECT_LOCK = threading.Lock()


def eye_aspect_ratio(eye):
    """Return the eye aspect ratio for one eye.

    eye is a sequence of six (x, y) points in this order:
    corner, upper lid, upper lid, opposite corner, lower lid, lower lid.
    The second point sits above the sixth and the third above the fifth.

    Raises ValueError if eye does not hold six (x, y) points,
    or if the two corners are the same point.
    """
    if len(eye) != POINTS_PER_EYE:
        raise ValueError(f"an eye needs {POINTS_PER_EYE} points, got {len(eye)}")
    for point in eye:
        if len(point) != 2:
            raise ValueError(f"each point must be an (x, y) pair, got {point!r}")

    corner, upper_a, upper_b, far_corner, lower_b, lower_a = eye
    width = dist(corner, far_corner)
    if width == 0:
        raise ValueError("the eye corners are the same point, so it has no width")

    height = dist(upper_a, lower_a) + dist(upper_b, lower_b)
    return height / (2.0 * width)


def average_eye_aspect_ratio(left_eye, right_eye):
    """Return the mean eye aspect ratio of both eyes.

    Averaging the two eyes smooths out noise in a single eye's landmarks.
    """
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0


@lru_cache(maxsize=1)
def get_landmarker():
    """Load the MediaPipe face landmarker once and reuse it.

    Raises FileNotFoundError if the model file has not been downloaded.
    """
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"face model not found at {path}. Run 'python face_model.py' to download it."
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def find_face_landmarks(image):
    """Return the face points MediaPipe finds in image, or None if there is no face.

    image is an RGB picture as a numpy array of shape (height, width, 3).
    Each point is an (x, y) pair between 0 and 1, as a fraction of the
    picture's width and height.
    """
    landmarker = get_landmarker()
    frame = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image, dtype=np.uint8),
    )
    with _DETECT_LOCK:
        result = landmarker.detect(frame)

    if not result.face_landmarks:
        return None

    return [(point.x, point.y) for point in result.face_landmarks[0]]


def eye_points(landmarks, indexes, width, height):
    """Return one eye's six points in pixels.

    MediaPipe gives each point as a fraction of the picture's size. The
    eye ratio needs real distances, so a wide picture must not stretch
    the eye sideways. Multiplying by width and height puts both
    directions back on the same scale.
    """
    return [(landmarks[i][0] * width, landmarks[i][1] * height) for i in indexes]


def no_face():
    """Return the result for a frame where no usable face was found."""
    return {"face_found": False, "eyes_open": True, "ear": None}


def analyze_eyes(image, find_landmarks=None, threshold=CLOSED_EAR_THRESHOLD):
    """Say whether a face is in the frame and whether its eyes are open.

    image is an RGB picture as a numpy array of shape (height, width, 3).
    find_landmarks replaces the MediaPipe step, which lets tests run without
    the model. threshold is the eye ratio below which the eyes count as closed.

    Returns a dict with:
      face_found  True if a face was found
      eyes_open   False only when a face was found and its eyes are closed
      ear         the average eye aspect ratio, or None if no face was found
    """
    finder = find_landmarks or find_face_landmarks
    landmarks = finder(image)
    if not landmarks or len(landmarks) <= max(LEFT_EYE + RIGHT_EYE):
        return no_face()

    height, width = image.shape[:2]
    left_eye = eye_points(landmarks, LEFT_EYE, width, height)
    right_eye = eye_points(landmarks, RIGHT_EYE, width, height)

    try:
        ear = average_eye_aspect_ratio(left_eye, right_eye)
    except ValueError:
        return no_face()

    return {"face_found": True, "eyes_open": ear >= threshold, "ear": round(ear, 4)}


def main(argv=None):
    """Print what the detector sees in each photo named on the command line.

    Returns 0, or 2 if no photo was named.
    """
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: python detector.py photo.jpg [more photos]")
        return 2

    for path in paths:
        with Image.open(path) as photo:
            image = np.asarray(photo.convert("RGB"))
        print(f"{path}: {analyze_eyes(image)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
