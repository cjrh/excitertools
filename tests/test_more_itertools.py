import pytest

from excitertools import Iter


def test_chunked():
    it = Iter(range(12))
    assert it.chunked(3).collect() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]


def test_ichunked():
    it = Iter(range(12))
    chunks = it.ichunked(3).collect()
    assert list(chunks[3]) == [9, 10, 11]
    assert list(chunks[2]) == [6, 7, 8]
    assert list(chunks[1]) == [3, 4, 5]
    assert list(chunks[0]) == [0, 1, 2]


def test_sliced():
    seq = list(range(4))
    assert Iter.sliced(seq, 2).collect() == [[0, 1], [2, 3]]
    assert Iter.sliced(seq, 3).collect() == [[0, 1, 2], [3]]


def test_distribute():
    g1, g2 = Iter.range(6).distribute(2)
    assert g1.collect() == [0, 2, 4]
    assert g2.collect() == [1, 3, 5]


def test_divide():
    g1, g2 = Iter.range(6).divide(2)
    assert g1.collect() == [0, 1, 2]
    assert g2.collect() == [3, 4, 5]


def test_split_at():
    it = Iter("abcdcba").split_at(lambda x: x == "b")
    assert it.collect() == [
        ["a"],
        ["c", "d", "c"],
        ["a"],
    ]


def test_split_before():
    it = Iter("OneTwo").split_before(lambda s: s.isupper())
    assert it.collect() == [
        ["O", "n", "e"],
        ["T", "w", "o"],
    ]


def test_split_after():
    it = Iter("one1two2").split_after(lambda s: s.isdigit())
    assert it.collect() == [
        ["o", "n", "e", "1"],
        ["t", "w", "o", "2"],
    ]


def test_bucket():
    iterable = ["a1", "b1", "c1", "a2", "b2", "c2", "b3"]
    bucket = Iter(iterable).bucket(key=lambda x: x[0])
    assert bucket["a"].collect() == ["a1", "a2"]
    assert bucket["b"].collect() == ["b1", "b2", "b3"]

    bucket = Iter(iterable).bucket(key=lambda x: x[0])
    a_iterable = bucket["a"]
    for x in a_iterable:
        print(x)

    bucket = Iter(iterable).bucket(key=lambda x: x[0])
    a_iterable = bucket["a"]
    assert next(a_iterable) == "a1"
    assert next(a_iterable) == "a2"

    bucket = Iter(iterable).bucket(key=lambda x: x[0])
    a_iterable = bucket["a"]
    result = a_iterable.collect()
    assert result == ["a1", "a2"]

    bucket = Iter(iterable).bucket(key=lambda x: x[0])
    result = Iter(bucket).collect()
    assert result == ["a", "b", "c"]

    bucket = Iter(iterable).bucket(key=lambda x: x[0])
    for k, expected in zip(bucket, ["a", "b", "c"]):
        assert k == expected

    result = Iter(iterable).bucket(key=lambda x: x[0])["a"].collect()
    assert result == ["a1", "a2"]


def test_unzip():
    iterable = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
    it = Iter(iterable).unzip()
    parts = it.collect()
    assert parts[0].collect() == ["a", "b", "c", "d"]
    assert parts[1].collect() == [1, 2, 3, 4]


def test_grouper():
    assert Iter("ABCDEFG").grouper(3, "x").collect() == [
        ("A", "B", "C"),
        ("D", "E", "F"),
        ("G", "x", "x"),
    ]


def test_partition():
    left, right = Iter(range(10)).partition(lambda x: x % 2 != 0)
    assert left.collect() == [0, 2, 4, 6, 8]
    assert right.collect() == [1, 3, 5, 7, 9]

    def odd(x):
        return x % 2 != 0

    left = Iter(range(10)).partition(odd).nth(0).collect()

    right = Iter(range(10)).partition(odd).nth(1).collect()
    assert left == [0, 2, 4, 6, 8]
    assert right == [1, 3, 5, 7, 9]


def test_spy():
    head, iterable = Iter("abcdefg").spy()
    assert head.collect() == ["a"]
    assert iterable.collect() == ["a", "b", "c", "d", "e", "f", "g"]


def test_peekable():
    p = Iter(["a", "b"]).peekable()
    assert p.peek() == "a"
    assert next(p) == "a"

    assert Iter([]).peekable().peek("hi") == "hi"

    p = Iter([1, 2, 3]).peekable()
    p.prepend(10, 11, 12)
    assert next(p) == 10
    assert p.peek() == 11
    # TODO: is there a clean way to keep p an Iter instance?
    assert Iter(p).collect() == [11, 12, 1, 2, 3]

    p = Iter([1, 2, 3]).peekable()
    p.prepend(10, 11, 12)
    assert p.peek() == 10
    assert p[2] == 12
    assert p[-1] == 3
    assert next(p) == 10
    assert p[2] == 1


def test_seekable():
    it = Iter.count().map(str).seekable()
    assert (next(it), next(it), next(it)) == ("0", "1", "2")

    it.seek(0)
    assert (next(it), next(it), next(it)) == ("0", "1", "2")

    assert next(it) == "3"

    it.seek(10)
    assert next(it) == "10"


def test_map_reduce_keyfunc():
    x = Iter("abbccc").map_reduce(lambda x: x.upper())
    assert sorted(x.items()) == [
        ("A", ["a"]),
        ("B", ["b", "b"]),
        ("C", ["c", "c", "c"]),
    ]


def test_map_reduce_valuefunc():
    x = Iter("abbccc").map_reduce(lambda x: x.upper(), lambda x: 1)
    assert sorted(x.items()) == [("A", [1]), ("B", [1, 1]), ("C", [1, 1, 1])]


def test_map_reduce_reducefunc():
    x = Iter("abbccc").map_reduce(
        lambda x: x.upper(),
        lambda x: 1,
        sum,
    )
    assert sorted(x.items()) == [("A", 1), ("B", 2), ("C", 3)]


def test_interleave():
    result = Iter("caleb").interleave(Iter.repeat("x")).collect()
    assert result == list("cxaxlxexbx")

    result = Iter("caleb").interleave(Iter.repeat("x")).concat("")
    assert result == "cxaxlxexbx"


def test_split_into():
    result = Iter("caleb").split_into([2, 1, 2]).collect()
    assert result == [["c", "a"], ["l"], ["e", "b"]]


def test_split_when():
    result = (
        Iter([1, 2, 3, 1, 4, 100, 4, 5, 1, 2]).split_when(lambda x, y: y < x).collect()
    )
    assert result == [[1, 2, 3], [1, 4, 100], [4, 5], [1, 2]]


@pytest.mark.parametrize(
    "iterable,n,fillvalue,step,expected",
    [
        ([1, 2, 3, 4, 5], 3, None, 1, [(1, 2, 3), (2, 3, 4), (3, 4, 5)]),
        ([1, 2, 3], 4, None, 1, [(1, 2, 3, None)]),
        ([1, 2, 3, 4, 5, 6], 3, "!", 2, [(1, 2, 3), (3, 4, 5), (5, 6, "!")]),
    ],
)
def test_windowed(iterable, n, fillvalue, step, expected):
    result = Iter(iterable).windowed(n, fillvalue, step).collect()
    assert result == expected


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ("more", ["m", "o", "r", "e", "mo", "or", "re", "mor", "ore", "more"]),
    ],
)
def test_substrings(iterable, expected):
    result = Iter(iterable).substrings().map(lambda s: "".join(s)).collect()
    assert result == expected


@pytest.mark.parametrize(
    "iterable,expected",
    [
        (
            "more",
            [
                ("m", 0, 1),
                ("o", 1, 2),
                ("r", 2, 3),
                ("e", 3, 4),
                ("mo", 0, 2),
                ("or", 1, 3),
                ("re", 2, 4),
                ("mor", 0, 3),
                ("ore", 1, 4),
                ("more", 0, 4),
            ],
        ),
    ],
)
def test_substrings_indexes(iterable, expected):
    result = (
        Iter(iterable)
        .substrings_indexes()
        .map(lambda tup: ("".join(tup[0]), *tup[1:]))
        .collect()
    )
    assert result == expected
