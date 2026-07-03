import io
import pathlib

from excitertools import Iter


def test_open():
    fname = pathlib.Path(__file__)
    x = Iter.open(fname).map(str.strip).filter(None).map(str.upper).collect()
    assert x[0] == "IMPORT IO"


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


def test_read_file_reads_text_lines(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("one\ntwo\n")

    assert Iter.read_file(path).map(str.strip).collect() == ["one", "two"]


def test_read_file_bytes_reads_binary_chunks(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdef")

    assert Iter.read_file_bytes(path, size=2).collect() == [b"ab", b"cd", b"ef"]


def test_write_file_writes_text(tmp_path):
    path = tmp_path / "data.txt"

    Iter(["one", "two"]).intersperse("\n").write_file(path)

    assert path.read_text() == "one\ntwo"


def test_write_file_writes_bytes(tmp_path):
    path = tmp_path / "data.bin"

    Iter([b"one", b"two"]).write_file(path, mode="wb")

    assert path.read_bytes() == b"onetwo"
