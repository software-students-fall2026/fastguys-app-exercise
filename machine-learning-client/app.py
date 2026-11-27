"""Flask service that checks each frame and counts blinks."""

from flask import Flask, jsonify, request

import detector
from blinks import BlinkTracker
from frames import FrameError, decode_frame

MAX_GAME_ID_LENGTH = 100


def default_analyzer(image):
    """Use the eye detector once it exists."""

    try:
        analyze = getattr(detector, "analyze_eyes")
    except AttributeError as exc:
        raise NotImplementedError("eye detector not built yet") from exc

    return analyze(image)


def check_game_id(game_id):
    """Return an error message if the game id is bad."""

    if not isinstance(game_id, str) or not game_id.strip():
        return "missing game_id"

    if len(game_id) > MAX_GAME_ID_LENGTH:
        return "game_id is too long"

    return None


def create_app(analyzer=None, tracker=None):
    """Build the Flask app."""

    app = Flask(__name__)
    analyze = analyzer or default_analyzer
    blinks = tracker or BlinkTracker()

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/process_frame")
    def process_frame():
        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify(error="send JSON with 'game_id' and 'image'"), 400

        game_id = body.get("game_id")
        problem = check_game_id(game_id)

        if problem:
            return jsonify(error=problem), 400

        try:
            image = decode_frame(body.get("image"))
        except FrameError as exc:
            return jsonify(error=str(exc)), 400

        try:
            result = analyze(image)
        except NotImplementedError:
            return jsonify(error="eye detector not ready"), 503

        face_found = bool(result.get("face_found", False))
        eyes_open = bool(result.get("eyes_open", True))

        if face_found:
            blink_count = blinks.update(game_id, eyes_open)
        else:
            blink_count = blinks.count(game_id)

        return jsonify(
            game_id=game_id,
            blink_count=blink_count,
            face_found=face_found,
            eyes_open=eyes_open,
        )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5001)
