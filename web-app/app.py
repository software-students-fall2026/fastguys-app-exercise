"""Flask backend for the Blink Speed Test web app."""

import os
import uuid
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
import requests

app = Flask(__name__)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL")

client = MongoClient(MONGO_URI)
db = client[DB_NAME] if DB_NAME else client["blink_test"]


def get_animal(blink_count):
    """Return the animal tier for a given 10-second blink count."""
    if blink_count < 10:
        return "Sloth"
    elif blink_count < 20:
        return "Tortoise"
    elif blink_count < 30:
        return "Cat"
    elif blink_count < 40:
        return "Rabbit"
    else:
        return "Cheetah"


@app.route("/")
def index():
    """Render the home page."""
    return render_template("index.html")


@app.route("/api/game/start", methods=["POST"])
def start_game():
    """Create a new game record in MongoDB and return its game_id."""
    game_id = str(uuid.uuid4())
    db.games.insert_one(
        {
            "game_id": game_id,
            "blink_count": 0,
            "status": "in_progress",
            "created_at": datetime.now(timezone.utc),
        }
    )
    return jsonify({"game_id": game_id}), 201


@app.route("/api/game/<game_id>/frame", methods=["POST"])
def send_frame(game_id):
    """Send a webcam frame to the ML client and save the updated blink count."""
    data = request.get_json(silent=True) or {}
    image = data.get("image")
    if not image:
        return jsonify({"error": "Missing image"}), 400

    try:
        response = requests.post(
            ML_SERVICE_URL,
            json={"game_id": game_id, "image": image},
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as err:
        return jsonify({"error": str(err)}), 502

    blink_count = result.get("blink_count", 0)
    db.games.update_one({"game_id": game_id}, {"$set": {"blink_count": blink_count}})
    return jsonify({"game_id": game_id, "blink_count": blink_count}), 200


@app.route("/api/game/<game_id>/finish", methods=["POST"])
def finish_game(game_id):
    """Finish the game, calculate the animal, and update MongoDB."""
    game = db.games.find_one({"game_id": game_id})
    if not game:
        return jsonify({"error": "Game not found"}), 404

    blink_count = game.get("blink_count", 0)
    animal = get_animal(blink_count)

    db.games.update_one(
        {"game_id": game_id},
        {"$set": {"status": "completed", "animal": animal}},
    )
    return (
        jsonify({"game_id": game_id, "blink_count": blink_count, "animal": animal}),
        200,
    )


@app.route("/api/leaderboard", methods=["GET"])
def leaderboard():
    """Return the top 10 completed scores."""
    scores = list(
        db.games.find({"status": "completed"}, {"_id": 0})
        .sort("blink_count", -1)
        .limit(10)
    )
    return jsonify(scores), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
