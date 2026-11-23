"""rtests for the blink tracker."""

from blinks import BlinkTracker


def test_new_game_starts_at_zero():
    """ new game has 0 blinks."""

    tracker = BlinkTracker()

    assert tracker.count("g1") == 0
    assert tracker.update("g1", True) == 0
    assert tracker.count("g1") == 0


def test_counts_each_blink_once():
  

    tracker = BlinkTracker()
    readings = [True, False, False, True, False, True, True, False]

    counts = [tracker.update("g1", reading) for reading in readings]

    assert counts == [0, 1, 1, 1, 2, 2, 2, 3]
    assert tracker.count("g1") == 3
    assert tracker.update("g1", False) == 3


def test_closed_at_start_counts():

  

    tracker = BlinkTracker()

    assert tracker.update("g1", False) == 1
    assert tracker.update("g1", False) == 1
    assert tracker.update("g1", True) == 1


def test_games_are_separate():




    tracker = BlinkTracker()

    tracker.update("g1", False)
    tracker.update("g1", True)
    tracker.update("g1", False)
    tracker.update("g2", False)

    assert tracker.count("g1") == 2
    assert tracker.count("g2") == 1
    assert tracker.count("g3") == 0



#resets the game score
def test_reset_removes_a_game():


    tracker = BlinkTracker()

    tracker.update("g1", False)
    tracker.reset("g1")
    tracker.reset("never-seen")

    assert tracker.count("g1") == 0
    assert tracker.update("g1", True) == 0
    assert tracker.count("never-seen") == 0
