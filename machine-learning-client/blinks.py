"""Keeps a blink counter for each game."""
import threading #threading allows smoother gameplay

class BlinkTracker:
    """Counts a blink every time the eyes go from open to closed."""

    def __init__(self):
        self.games = {}
        self.lock = threading.Lock()

    def update(self, game_id, eyes_open):
        """Add one reading for a game and return its blink count."""

        with self.lock:
            game = self.games.setdefault(game_id, {"count": 0, "was_open": True})

            if game["was_open"] and not eyes_open:
                game["count"] += 1

            game["was_open"] = eyes_open

            return game["count"]

    def count(self, game_id):
        """Return the blink count for a game."""

        with self.lock:
            game = self.games.get(game_id)

            if game is None:
                return 0

            return game["count"]

    def reset(self, game_id):
        """Remove a game."""

        with self.lock:
            self.games.pop(game_id, None)
