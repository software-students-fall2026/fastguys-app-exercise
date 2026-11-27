"""Blink detector: measures how open an eye is from its landmark points.

The eye aspect ratio (EAR) compares the height of an eye to its width.
An open eye gives a ratio of roughly 0.25 to 0.35, and the ratio falls
towards 0 as the eye closes. It does not change when the face moves
closer to or further from the camera, because height and width scale together.
"""

from math import dist

POINTS_PER_EYE = 6


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
