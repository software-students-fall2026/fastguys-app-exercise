"""Tests for the blink detector in detector.py."""

import pytest

import detector

# An open eye 6 units wide and 2 units tall: ratio (2 + 2) / (2 * 6) = 1/3.
OPEN_EYE = [(0, 0), (2, 1), (4, 1), (6, 0), (4, -1), (2, -1)]
# The same eye with both lids on the centre line.
CLOSED_EYE = [(0, 0), (2, 0), (4, 0), (6, 0), (4, 0), (2, 0)]
# Half as tall as OPEN_EYE: ratio (1 + 1) / (2 * 6) = 1/6.
HALF_OPEN_EYE = [(0, 0), (2, 0.5), (4, 0.5), (6, 0), (4, -0.5), (2, -0.5)]


def move(eye, shift_x, shift_y):
    """Return the eye shifted by (shift_x, shift_y)."""
    return [(x + shift_x, y + shift_y) for x, y in eye]


def scale(eye, factor):
    """Return the eye with every coordinate multiplied by factor."""
    return [(x * factor, y * factor) for x, y in eye]


class TestEyeAspectRatio:
    """eye_aspect_ratio() turns six eye points into one number."""

    def test_open_eye_gives_expected_ratio(self):
        """A 6 by 2 eye has a ratio of one third."""
        assert detector.eye_aspect_ratio(OPEN_EYE) == pytest.approx(1 / 3)

    def test_closed_eye_gives_zero(self):
        """An eye with no height has a ratio of zero."""
        assert detector.eye_aspect_ratio(CLOSED_EYE) == 0

    def test_ratio_falls_as_the_eye_closes(self):
        """Open is larger than half open, which is larger than closed."""
        open_ratio = detector.eye_aspect_ratio(OPEN_EYE)
        half_ratio = detector.eye_aspect_ratio(HALF_OPEN_EYE)
        closed_ratio = detector.eye_aspect_ratio(CLOSED_EYE)
        assert open_ratio > half_ratio > closed_ratio
        assert half_ratio == pytest.approx(1 / 6)

    def test_ratio_ignores_distance_from_camera(self):
        """A bigger or smaller copy of the same eye gives the same ratio."""
        expected = detector.eye_aspect_ratio(OPEN_EYE)
        bigger = detector.eye_aspect_ratio(scale(OPEN_EYE, 10))
        smaller = detector.eye_aspect_ratio(scale(OPEN_EYE, 0.01))
        assert bigger == pytest.approx(expected)
        assert smaller == pytest.approx(expected)

    def test_ratio_ignores_position_in_the_frame(self):
        """Moving the eye around the image does not change the ratio."""
        expected = detector.eye_aspect_ratio(OPEN_EYE)
        moved = move(OPEN_EYE, 250, -40)
        assert detector.eye_aspect_ratio(moved) == pytest.approx(expected)

    def test_accepts_tuples_and_lists(self):
        """Points can be tuples or lists, and so can the eye itself."""
        as_lists = [list(point) for point in OPEN_EYE]
        as_tuple = tuple(OPEN_EYE)
        assert detector.eye_aspect_ratio(as_lists) == pytest.approx(1 / 3)
        assert detector.eye_aspect_ratio(as_tuple) == pytest.approx(1 / 3)

    @pytest.mark.parametrize("count", [0, 1, 5, 7])
    def test_rejects_wrong_number_of_points(self, count):
        """Anything other than six points is an error."""
        with pytest.raises(ValueError, match="6 points"):
            detector.eye_aspect_ratio([(0, 0)] * count)

    def test_rejects_points_that_are_not_pairs(self):
        """A point with three coordinates is an error."""
        eye = list(OPEN_EYE)
        eye[2] = (4, 1, 0)
        with pytest.raises(ValueError, match="pair"):
            detector.eye_aspect_ratio(eye)

    def test_rejects_eye_with_no_width(self):
        """Two corners on the same point would mean dividing by zero."""
        eye = list(OPEN_EYE)
        eye[3] = eye[0]
        with pytest.raises(ValueError, match="no width"):
            detector.eye_aspect_ratio(eye)


class TestAverageEyeAspectRatio:
    """average_eye_aspect_ratio() combines the left and right eye."""

    def test_two_identical_eyes_give_the_same_ratio(self):
        """Averaging an eye with itself changes nothing."""
        result = detector.average_eye_aspect_ratio(OPEN_EYE, OPEN_EYE)
        assert result == pytest.approx(1 / 3)

    def test_one_open_and_one_closed_eye_give_half(self):
        """A wink averages an open eye (1/3) and a closed eye (0)."""
        result = detector.average_eye_aspect_ratio(OPEN_EYE, CLOSED_EYE)
        assert result == pytest.approx(1 / 6)

    def test_order_of_eyes_does_not_matter(self):
        """Left and right can be swapped."""
        one_way = detector.average_eye_aspect_ratio(OPEN_EYE, HALF_OPEN_EYE)
        other_way = detector.average_eye_aspect_ratio(HALF_OPEN_EYE, OPEN_EYE)
        assert one_way == pytest.approx(other_way)

    def test_bad_eye_is_still_an_error(self):
        """A broken eye on either side raises instead of returning a number."""
        with pytest.raises(ValueError):
            detector.average_eye_aspect_ratio(OPEN_EYE, OPEN_EYE[:5])
