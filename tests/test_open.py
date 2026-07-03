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


def test_read_lines_starts_at_current_position_by_default():
    stream = io.StringIO("one\ntwo\nthree\n")

    assert stream.readline() == "one\n"

    assert Iter.read_lines(stream).collect() == ["two\n", "three\n"]


def test_read_lines_can_rewind_seekable_streams():
    stream = io.StringIO("one\ntwo\n")

    assert stream.readline() == "one\n"

    assert Iter.read_lines(stream, rewind=True).collect() == ["one\n", "two\n"]


def test_read_bytes_starts_at_current_position_and_chunks_by_default():
    stream = io.BytesIO((b"a" * 8192) + b"bc")

    assert stream.read(1) == b"a"

    assert Iter.read_bytes(stream).map(len).collect() == [8192, 1]


def test_read_bytes_can_rewind_and_read_whole_stream():
    stream = io.BytesIO(b"abcdef")

    assert stream.read(2) == b"ab"

    assert Iter.read_bytes(stream, size=-1, rewind=True).collect() == [b"abcdef"]


def test_open_context_manager_stops_generator_after_partial_read(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("one\ntwo\n")

    with Iter.open(path) as lines:
        assert lines.next() == "one\n"

    with pytest.raises(StopIteration):
        lines.next()
