"""Unit tests for the Blink Speed Test web app."""

from unittest.mock import MagicMock, patch
import mongomock
import pytest
import app as app_module


@pytest.fixture(name="client")
def fixture_client(monkeypatch):
    """Provide a Flask test client with a mocked database and ML URL."""
    mock_db = mongomock.MongoClient().blink_test
    monkeypatch.setattr(app_module, "db", mock_db)
    monkeypatch.setattr(
        app_module, "ML_SERVICE_URL", "http://ml-test:5001/process_frame"
    )
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize(
    "blinks,expected",
    [
        (0, "Sloth"),
        (9, "Sloth"),
        (10, "Tortoise"),
        (19, "Tortoise"),
        (20, "Cat"),
        (29, "Cat"),
        (30, "Rabbit"),
        (39, "Rabbit"),
        (40, "Cheetah"),
        (55, "Cheetah"),
    ],
)
def test_get_animal(blinks, expected):
    """Test blink counts map to the right animal."""
    assert app_module.get_animal(blinks) == expected


def test_home_page(client):
    """Test the index route loads."""
    res = client.get("/")
    assert res.status_code == 200


@patch("app.requests.post")
def test_game_flow(mock_post, client):
    """Test starting, sending a frame, finishing, and fetching the leaderboard."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"blink_count": 25}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    start_res = client.post("/api/game/start")
    assert start_res.status_code == 201
    game_id = start_res.get_json()["game_id"]

    frame_res = client.post(
        f"/api/game/{game_id}/frame", json={"image": "base64_sample"}
    )
    assert frame_res.status_code == 200
    assert frame_res.get_json()["blink_count"] == 25

    finish_res = client.post(f"/api/game/{game_id}/finish")
    assert finish_res.status_code == 200
    assert finish_res.get_json()["animal"] == "Cat"

    board_res = client.get("/api/leaderboard")
    assert board_res.status_code == 200
    assert len(board_res.get_json()) == 1


def test_bad_requests(client):
    """Test missing image and unknown game ID error responses."""
    assert client.post("/api/game/123/frame", json={}).status_code == 400
    assert client.post("/api/game/123/finish").status_code == 404

def test_leaderboard_counts_all_completed_games(client):
    games = app_module.db.games
    games.insert_many([
        {"game_id": f"round-{i}", "status": "completed",
         "animal": "Cat", "blink_count": 20}
        for i in range(10)
    ])
    games.insert_one({
        "game_id": "last-place", "status": "completed",
        "animal": "Sloth", "blink_count": 2,
    })
    games.insert_one({"game_id": "unfinished", "status": "in_progress"})

    page = client.get("/api/leaderboard").get_data(as_text=True)

    assert "Cat: 10" in page
    assert "Sloth: 1" in page
    assert "Cheetah: 0" in page
    assert "last-place" not in page