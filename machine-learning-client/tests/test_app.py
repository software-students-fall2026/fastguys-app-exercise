
import pytest

import detector
from app import check_game_id, create_app, default_analyzer


def send(client, game_id, image):



    return client.post("/process_frame", json={"game_id": game_id, "image": image})


def test_health_check(client):



    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert response.is_json


def test_frame_returns_blink_count(client, frame):



    response = send(client, "g1", frame())

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == {
        "game_id": "g1",
        "blink_count": 0,
        "face_found": True,
        "eyes_open": True,
    }


def test_count_goes_up_over_frames(client, readings, frame):




    for state in (True, False, True, False):
        readings.append({"face_found": True, "eyes_open": state})

    counts = [send(client, "g1", frame()).get_json()["blink_count"] for _ in range(4)]

    assert counts == [0, 1, 1, 2]
    assert counts[-1] == 2
    assert not readings


def test_games_counted_separately(client, readings, frame):


#games dont collide

    readings.append({"face_found": True, "eyes_open": False})
    readings.append({"face_found": True, "eyes_open": False})

    first = send(client, "a", frame()).get_json()
    second = send(client, "b", frame()).get_json()

    assert first["blink_count"] == 1
    assert second["blink_count"] == 1
    assert first["game_id"] != second["game_id"]


def test_no_face_keeps_the_count(client, readings, frame):
  

    readings.append({"face_found": True, "eyes_open": False})
    readings.append({"face_found": False, "eyes_open": False})
    readings.append({"face_found": False})

    results = [send(client, "g1", frame()).get_json() for _ in range(3)]

    assert [result["blink_count"] for result in results] == [1, 1, 1]
    assert results[1]["face_found"] is False
    assert results[2]["eyes_open"] is True


def test_detector_gets_the_image(client, fake_analyzer, frame):
    """The detector receives the decoded image."""

    send(client, "g1", frame(width=10, height=4, data_url=True))

    image = fake_analyzer.calls[0]

    assert image.shape == (4, 10, 3)
    assert image.dtype.name == "uint8"
    assert len(fake_analyzer.calls) == 1


def test_rejects_body_that_is_not_json(client, fake_analyzer):
    """A body that is not a JSON object is rejected."""

    for kwargs in ({"json": ["image"]}, {"data": "image=abc"}, {"json": "text"}):
        response = client.post("/process_frame", **kwargs)

        assert response.status_code == 400
        assert "game_id" in response.get_json()["error"]

    assert not fake_analyzer.calls


def test_rejects_bad_game_id(client, fake_analyzer, frame):
    """Missing, blank or very long game ids are rejected."""

    response = client.post("/process_frame", json={"image": frame()})
    assert response.status_code == 400

    for game_id in ("", "   ", 42, "x" * 101):
        response = send(client, game_id, frame())

        assert response.status_code == 400
        assert "game_id" in response.get_json()["error"]

    assert not fake_analyzer.calls


def test_check_game_id():
#message for bad id/missing id

    assert check_game_id("g1") is None
    assert check_game_id("") == "missing game_id"
    assert check_game_id("x" * 101) == "game_id is too long"


def test_rejects_bad_images(client, fake_analyzer):
  

    response = client.post("/process_frame", json={"game_id": "g1"})
    assert response.status_code == 400

    for image in ("", "not base64!!", 42):
        response = send(client, "g1", image)

        assert response.status_code == 400
        assert "image" in response.get_json()["error"]

    assert not fake_analyzer.calls


def test_not_ready_until_detector_exists(monkeypatch, frame):




    monkeypatch.delattr(detector, "analyze_eyes", raising=False)

    response = send(create_app().test_client(), "g1", frame())

    assert response.status_code == 503
    assert "not ready" in response.get_json()["error"]


    with pytest.raises(NotImplementedError):
        default_analyzer(None)


def test_uses_detector_once_it_exists(monkeypatch, frame):




    def fake_detector(image):
        return {"face_found": True, "eyes_open": image is None}

    monkeypatch.setattr(detector, "analyze_eyes", fake_detector, raising=False)

    response = send(create_app().test_client(), "g1", frame())

    assert response.status_code == 200
    assert response.get_json()["blink_count"] == 1
    assert default_analyzer(None) == {"face_found": True, "eyes_open": True}


def test_only_post_is_allowed(client, frame):


    assert client.get("/process_frame").status_code == 405
    assert client.put("/process_frame", json={"image": frame()}).status_code == 405
    assert client.post("/health").status_code == 405
