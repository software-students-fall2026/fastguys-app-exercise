"""Tests for face_model.py, which finds and downloads the face model."""

import urllib.error

import pytest

import face_model

MODEL_BYTES = b"m" * face_model.MIN_MODEL_BYTES


class FakeResponse:
    """Stands in for the response urlopen() gives back."""

    def __init__(self, data):
        self.data = data

    def read(self):
        """Return the prepared bytes."""
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def opener_serving(answers):
    """Return a stand-in for urlopen() that serves bytes or errors per address.

    The addresses it was asked for are kept in its `asked` list.
    """

    def opener(url, timeout=None):
        assert timeout
        opener.asked.append(url)
        answer = answers[url]
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)

    opener.asked = []
    return opener


class TestModelPath:
    """model_path() says where the model file lives."""

    def test_default_is_the_models_folder(self, monkeypatch):
        """Without the variable the model sits in models/ beside the code."""
        monkeypatch.delenv("FACE_MODEL_PATH", raising=False)
        path = face_model.model_path()
        assert path == face_model.DEFAULT_MODEL_PATH
        assert path.parent.name == "models"
        assert path.name == "face_landmarker.task"

    def test_environment_variable_wins(self, monkeypatch, tmp_path):
        """FACE_MODEL_PATH moves the model somewhere else."""
        monkeypatch.setenv("FACE_MODEL_PATH", str(tmp_path / "other.task"))
        assert face_model.model_path() == tmp_path / "other.task"

    def test_blank_variable_is_ignored(self, monkeypatch):
        """An empty FACE_MODEL_PATH falls back to the default."""
        monkeypatch.setenv("FACE_MODEL_PATH", "  ")
        assert face_model.model_path() == face_model.DEFAULT_MODEL_PATH


class TestFetch:
    """fetch() reads one address."""

    def test_returns_the_bytes(self):
        """A full-size answer is returned as it is."""
        opener = opener_serving({"http://a": MODEL_BYTES})
        assert face_model.fetch("http://a", opener) == MODEL_BYTES

    def test_tiny_answer_is_rejected(self):
        """An error page is far too small to be the model."""
        opener = opener_serving({"http://a": b"not found"})
        with pytest.raises(OSError, match="only 9 bytes"):
            face_model.fetch("http://a", opener)


class TestDownload:
    """download() saves the model if it is not there yet."""

    def test_saves_the_model(self, tmp_path):
        """The bytes land in the destination, and folders are created."""
        destination = tmp_path / "models" / "face.task"
        opener = opener_serving({"http://a": MODEL_BYTES})

        result = face_model.download(destination, ["http://a"], opener)

        assert result == destination
        assert destination.read_bytes() == MODEL_BYTES

    def test_existing_model_is_not_downloaded_again(self, tmp_path):
        """A model that is already there is left alone."""
        destination = tmp_path / "face.task"
        destination.write_bytes(b"already here")
        opener = opener_serving({})

        assert face_model.download(destination, ["http://a"], opener) == destination
        assert not opener.asked
        assert destination.read_bytes() == b"already here"

    def test_second_address_is_used_when_the_first_fails(self, tmp_path):
        """A broken first address does not stop the download."""
        destination = tmp_path / "face.task"
        opener = opener_serving(
            {
                "http://a": urllib.error.URLError("no route"),
                "http://b": MODEL_BYTES,
            }
        )

        face_model.download(destination, ["http://a", "http://b"], opener)

        assert opener.asked == ["http://a", "http://b"]
        assert destination.read_bytes() == MODEL_BYTES

    def test_tiny_answer_moves_on_to_the_next_address(self, tmp_path):
        """An error page from the first address is not saved as the model."""
        destination = tmp_path / "face.task"
        opener = opener_serving({"http://a": b"oops", "http://b": MODEL_BYTES})

        face_model.download(destination, ["http://a", "http://b"], opener)

        assert destination.read_bytes() == MODEL_BYTES

    def test_all_addresses_failing_is_an_error(self, tmp_path):
        """The error lists every address that was tried and saves nothing."""
        destination = tmp_path / "face.task"
        opener = opener_serving(
            {"http://a": urllib.error.URLError("down"), "http://b": b"oops"}
        )

        with pytest.raises(OSError) as error:
            face_model.download(destination, ["http://a", "http://b"], opener)

        assert "http://a" in str(error.value)
        assert "http://b" in str(error.value)
        assert not destination.exists()

    def test_default_destination_is_the_model_path(self, monkeypatch, tmp_path):
        """With no destination the model goes where model_path() points."""
        monkeypatch.setenv("FACE_MODEL_PATH", str(tmp_path / "face.task"))
        opener = opener_serving({"http://a": MODEL_BYTES})

        assert face_model.download(urls=["http://a"], opener=opener) == (
            tmp_path / "face.task"
        )


def test_main_prints_where_the_model_is(monkeypatch, tmp_path, capsys):
    """Running `python face_model.py` prints the path of the model."""
    model = tmp_path / "face.task"
    model.write_bytes(MODEL_BYTES)
    monkeypatch.setenv("FACE_MODEL_PATH", str(model))

    face_model.main()

    assert str(model) in capsys.readouterr().out
