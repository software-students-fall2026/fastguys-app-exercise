"""Tests for the blink detector in detector.py."""

from types import SimpleNamespace

import numpy as np
import pytest

import detector
from app import create_app

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


FACE_POINT_COUNT = 478
# A camera frame that is wider than it is tall, like a real webcam picture.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def blank_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Return an all-black RGB picture."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def face_with_eyes(left_eye, right_eye, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Return 478 face points, as fractions of the frame, with the given eyes.

    left_eye and right_eye are in pixels. Every other point sits at the centre.
    """
    points = [(0.5, 0.5)] * FACE_POINT_COUNT
    for indexes, eye in (
        (detector.LEFT_EYE, left_eye),
        (detector.RIGHT_EYE, right_eye),
    ):
        for index, (x, y) in zip(indexes, eye):
            points[index] = (x / width, y / height)
    return points


def eye_in_frame(eye, left, top, size=10):
    """Return eye made size times bigger and moved to (left, top) in pixels."""
    return move(scale(eye, size), left, top)


def finder_returning(landmarks):
    """Return a stand-in for MediaPipe that always finds these landmarks."""
    return lambda image: landmarks


class TestEyePoints:
    """eye_points() turns fractions of the frame into pixels."""

    def test_scales_by_width_and_height(self):
        """x is multiplied by the width and y by the height."""
        landmarks = [(0.5, 0.25), (0.1, 0.5)]
        assert detector.eye_points(landmarks, (1, 0), 200, 100) == [
            (pytest.approx(20), pytest.approx(50)),
            (pytest.approx(100), pytest.approx(25)),
        ]

    def test_keeps_the_order_of_the_indexes(self):
        """Points come back in the order asked for, not list order."""
        landmarks = [(0.0, 0.0), (1.0, 1.0)]
        assert detector.eye_points(landmarks, (1, 0), 10, 10)[0] == (10.0, 10.0)


class TestAnalyzeEyes:
    """analyze_eyes() is what the ML service calls for every frame."""

    def test_open_eyes(self):
        """Two open eyes are reported as a face with open eyes."""
        face = face_with_eyes(
            eye_in_frame(OPEN_EYE, 350, 200), eye_in_frame(OPEN_EYE, 200, 200)
        )
        result = detector.analyze_eyes(blank_frame(), finder_returning(face))
        assert result == {"face_found": True, "eyes_open": True, "ear": 0.3333}

    def test_closed_eyes(self):
        """Two closed eyes are reported as a face with closed eyes."""
        face = face_with_eyes(
            eye_in_frame(CLOSED_EYE, 350, 200), eye_in_frame(CLOSED_EYE, 200, 200)
        )
        result = detector.analyze_eyes(blank_frame(), finder_returning(face))
        assert result == {"face_found": True, "eyes_open": False, "ear": 0.0}

    def test_half_open_eyes_count_as_closed(self):
        """A ratio of 1/6 is under the default threshold of 0.2."""
        face = face_with_eyes(
            eye_in_frame(HALF_OPEN_EYE, 350, 200),
            eye_in_frame(HALF_OPEN_EYE, 200, 200),
        )
        result = detector.analyze_eyes(blank_frame(), finder_returning(face))
        assert result["face_found"] is True
        assert result["eyes_open"] is False
        assert result["ear"] == pytest.approx(1 / 6, abs=1e-4)

    def test_wink_uses_the_average_of_both_eyes(self):
        """One open and one closed eye average to 1/6, which is closed."""
        face = face_with_eyes(
            eye_in_frame(OPEN_EYE, 350, 200), eye_in_frame(CLOSED_EYE, 200, 200)
        )
        result = detector.analyze_eyes(blank_frame(), finder_returning(face))
        assert result["eyes_open"] is False

    def test_threshold_can_be_changed(self):
        """The same half-open eyes count as open with a lower threshold."""
        face = face_with_eyes(
            eye_in_frame(HALF_OPEN_EYE, 350, 200),
            eye_in_frame(HALF_OPEN_EYE, 200, 200),
        )
        result = detector.analyze_eyes(
            blank_frame(), finder_returning(face), threshold=0.1
        )
        assert result["eyes_open"] is True

    def test_ratio_exactly_on_the_threshold_is_open(self):
        """Only a ratio below the threshold counts as closed."""
        face = face_with_eyes(
            eye_in_frame(OPEN_EYE, 350, 200), eye_in_frame(OPEN_EYE, 200, 200)
        )
        result = detector.analyze_eyes(
            blank_frame(), finder_returning(face), threshold=1 / 3
        )
        assert result["eyes_open"] is True

    @pytest.mark.parametrize("width,height", [(640, 480), (1280, 720), (480, 640)])
    def test_frame_shape_does_not_change_the_ratio(self, width, height):
        """A wide or tall frame must not stretch the eye."""
        face = face_with_eyes(
            eye_in_frame(OPEN_EYE, 250, 200),
            eye_in_frame(OPEN_EYE, 100, 200),
            width,
            height,
        )
        result = detector.analyze_eyes(
            blank_frame(width, height), finder_returning(face)
        )
        assert result["ear"] == pytest.approx(1 / 3, abs=1e-4)

    def test_no_face(self):
        """With no face in the frame nothing is counted as a blink."""
        result = detector.analyze_eyes(blank_frame(), finder_returning(None))
        assert result == {"face_found": False, "eyes_open": True, "ear": None}

    def test_too_few_points_is_no_face(self):
        """A short list of points cannot hold both eyes, so it is not a face."""
        result = detector.analyze_eyes(
            blank_frame(), finder_returning([(0.5, 0.5)] * 100)
        )
        assert result["face_found"] is False

    def test_eye_with_no_width_is_no_face(self):
        """If every point lands on one spot the eye cannot be measured."""
        face = [(0.5, 0.5)] * FACE_POINT_COUNT
        result = detector.analyze_eyes(blank_frame(), finder_returning(face))
        assert result == {"face_found": False, "eyes_open": True, "ear": None}

    def test_finder_gets_the_image(self):
        """The picture is handed to the landmark finder unchanged."""
        seen = []
        image = blank_frame()

        def finder(picture):
            seen.append(picture)

        detector.analyze_eyes(image, finder)
        assert seen[0] is image

    def test_uses_mediapipe_by_default(self, monkeypatch):
        """Without a finder, find_face_landmarks() is used."""
        face = face_with_eyes(
            eye_in_frame(OPEN_EYE, 350, 200), eye_in_frame(OPEN_EYE, 200, 200)
        )
        monkeypatch.setattr(detector, "find_face_landmarks", finder_returning(face))
        assert detector.analyze_eyes(blank_frame())["face_found"] is True


def fake_landmarker(faces):
    """Return a stand-in for MediaPipe's landmarker that remembers its frames."""
    fake = SimpleNamespace(frames=[])

    def detect(frame):
        fake.frames.append(frame)
        return SimpleNamespace(face_landmarks=faces)

    fake.detect = detect
    return fake


def landmark(x, y):
    """Return one face point the way MediaPipe gives it."""
    return SimpleNamespace(x=x, y=y, z=0.0)


class TestFindFaceLandmarks:
    """find_face_landmarks() wraps the MediaPipe model."""

    def test_returns_points_of_the_first_face(self, monkeypatch):
        """Each landmark becomes an (x, y) pair."""
        first = [landmark(0.1, 0.2), landmark(0.3, 0.4)]
        second = [landmark(0.9, 0.9)]
        fake = fake_landmarker([first, second])
        monkeypatch.setattr(detector, "get_landmarker", lambda: fake)

        assert detector.find_face_landmarks(blank_frame(8, 6)) == [
            (0.1, 0.2),
            (0.3, 0.4),
        ]

    def test_no_face_returns_none(self, monkeypatch):
        """An empty result from MediaPipe means no face."""
        monkeypatch.setattr(detector, "get_landmarker", lambda: fake_landmarker([]))
        assert detector.find_face_landmarks(blank_frame(8, 6)) is None

    def test_passes_the_frame_size_to_mediapipe(self, monkeypatch):
        """The picture reaches MediaPipe with its width and height intact."""
        fake = fake_landmarker([])
        monkeypatch.setattr(detector, "get_landmarker", lambda: fake)

        detector.find_face_landmarks(blank_frame(8, 6))

        assert (fake.frames[0].width, fake.frames[0].height) == (8, 6)

    def test_accepts_a_read_only_image(self, monkeypatch):
        """Pictures decoded by Pillow are read-only and must still work."""
        image = blank_frame(8, 6)
        image.flags.writeable = False
        fake = fake_landmarker([])
        monkeypatch.setattr(detector, "get_landmarker", lambda: fake)

        assert detector.find_face_landmarks(image) is None
        assert len(fake.frames) == 1


class TestGetLandmarker:
    """get_landmarker() loads the model file once."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Forget any landmarker loaded by another test."""
        detector.get_landmarker.cache_clear()
        yield
        detector.get_landmarker.cache_clear()

    def test_missing_model_says_how_to_get_it(self, monkeypatch, tmp_path):
        """Without the model file the error names the download command."""
        monkeypatch.setenv("FACE_MODEL_PATH", str(tmp_path / "missing.task"))

        with pytest.raises(FileNotFoundError, match="python face_model.py"):
            detector.get_landmarker()

    def test_loads_the_model_from_the_configured_path(self, monkeypatch, tmp_path):
        """The model path is handed to MediaPipe and one face is asked for."""
        model = tmp_path / "face.task"
        model.write_bytes(b"model")
        monkeypatch.setenv("FACE_MODEL_PATH", str(model))
        created = []

        def fake_create(options):
            created.append(options)
            return "landmarker"

        monkeypatch.setattr(
            detector.mp.tasks.vision.FaceLandmarker, "create_from_options", fake_create
        )

        assert detector.get_landmarker() == "landmarker"
        assert created[0].base_options.model_asset_path == str(model)
        assert created[0].num_faces == 1

    def test_model_is_loaded_only_once(self, monkeypatch, tmp_path):
        """A second call reuses the first landmarker."""
        model = tmp_path / "face.task"
        model.write_bytes(b"model")
        monkeypatch.setenv("FACE_MODEL_PATH", str(model))
        created = []

        def fake_create(options):
            created.append(options)
            return object()

        monkeypatch.setattr(
            detector.mp.tasks.vision.FaceLandmarker, "create_from_options", fake_create
        )

        assert detector.get_landmarker() is detector.get_landmarker()
        assert len(created) == 1


class TestMain:
    """main() is what `python detector.py photo.jpg` runs."""

    def test_prints_the_result_for_each_photo(self, monkeypatch, tmp_path, capsys):
        """Every photo named gets one line with its result."""
        monkeypatch.setattr(detector, "find_face_landmarks", finder_returning(None))
        photos = []
        for name in ("one.png", "two.png"):
            detector.Image.new("RGB", (8, 6)).save(tmp_path / name)
            photos.append(str(tmp_path / name))

        assert detector.main(photos) == 0

        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith(photos[0])
        assert "'face_found': False" in lines[0]

    def test_no_photo_prints_how_to_use_it(self, capsys):
        """Without a photo the usage line is printed and the exit code is 2."""
        assert detector.main([]) == 2
        assert "usage" in capsys.readouterr().out

    def test_reads_the_command_line_by_default(self, monkeypatch, capsys):
        """With no argument list, sys.argv is used."""
        monkeypatch.setattr(detector.sys, "argv", ["detector.py"])
        assert detector.main() == 2
        assert "usage" in capsys.readouterr().out


class TestServiceUsesDetector:
    """The ML service counts blinks with the real analyze_eyes()."""

    def test_open_closed_open_is_one_blink(self, monkeypatch, frame):
        """Three frames through /process_frame add up to one blink."""
        open_face = face_with_eyes(
            eye_in_frame(OPEN_EYE, 350, 200), eye_in_frame(OPEN_EYE, 200, 200)
        )
        closed_face = face_with_eyes(
            eye_in_frame(CLOSED_EYE, 350, 200), eye_in_frame(CLOSED_EYE, 200, 200)
        )
        faces = iter([open_face, closed_face, open_face])
        monkeypatch.setattr(detector, "find_face_landmarks", lambda image: next(faces))
        client = create_app().test_client()

        replies = [
            client.post(
                "/process_frame",
                json={"game_id": "game-1", "image": frame(FRAME_WIDTH, FRAME_HEIGHT)},
            ).get_json()
            for _ in range(3)
        ]

        assert [reply["eyes_open"] for reply in replies] == [True, False, True]
        assert [reply["blink_count"] for reply in replies] == [0, 1, 1]

    def test_no_face_does_not_count_a_blink(self, monkeypatch, frame):
        """A frame with nobody in it leaves the count alone."""
        monkeypatch.setattr(detector, "find_face_landmarks", finder_returning(None))
        client = create_app().test_client()

        reply = client.post(
            "/process_frame", json={"game_id": "game-2", "image": frame()}
        ).get_json()

        assert reply["face_found"] is False
        assert reply["blink_count"] == 0
