import io
import pathlib

import pytest

from excitertools import Iter


def test_open():
    fname = pathlib.Path(__file__)
    x = Iter.open(fname).map(str.strip).filter(None).map(str.upper).collect()
    assert x[0] == "IMPORT IO"


def test_close_closes_wrapped_stream():
    stream = io.StringIO("one\ntwo\n")
    lines = Iter.read_lines(stream, rewind=False)

    assert lines.next() == "one\n"
    lines.close()

    assert stream.closed


def test_context_manager_closes_wrapped_stream():
    stream = io.StringIO("one\ntwo\n")

    with Iter.read_lines(stream, rewind=False) as lines:
        assert lines.next() == "one\n"

    assert stream.closed


def test_open_context_manager_stops_generator_after_partial_read(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("one\ntwo\n")

    with Iter.open(path) as lines:
        assert lines.next() == "one\n"

    with pytest.raises(StopIteration):
        lines.next()
