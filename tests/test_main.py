import pytest

from excitertools import Iter, insert_separator, concat


def test_basic():
    it = Iter(range(5))
    assert list(it) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "container,it,expected",
    [
        (list, range(5), [0, 1, 2, 3, 4]),
        (set, range(5), {0, 1, 2, 3, 4}),
        (dict, zip("abc", range(3)), dict(a=0, b=1, c=2)),
        (dict.fromkeys, "abc", dict(a=None, b=None, c=None)),
        (frozenset, range(5), frozenset([0, 1, 2, 3, 4])),
        (str, ["a", "b", "c"], "abc"),
        # (bytes, [b'a', b'b', b'c'], b'abc'),
    ],
)
def test_collect(container, it, expected):
    it = Iter(it)
    assert it.collect(container=container) == expected


def test_islice():
    it = Iter(range(5))
    assert it.islice(3).collect() == [0, 1, 2]


def test_cycle_islice():
    it = Iter(range(5))
    assert it.cycle().islice(6, 9, 2).collect() == [1, 3]


def test_count():
    it = Iter.count().islice(3, 6)
    assert it.collect() == [3, 4, 5]


def test_map():
    it = Iter.range(5).map(lambda x: x * 2)
    assert it.collect() == [0, 2, 4, 6, 8]

    it = Iter.range(5).map(lambda x: x * 2).filter(lambda x: x > 5)
    assert it.collect() == [6, 8]

    result = (
        Iter.range(5)
        .map(lambda x: x * 2)
        .filter(lambda x: x > 5)
        .reduce(lambda x, y: x + y)
    )
    assert result == 14


def test_piecemeal_results():
    it = Iter.range(5).map(lambda x: x * 2).filter(lambda x: x > 5)

    x = iter(it)
    assert next(x) == 6
    assert next(x) == 8


@pytest.mark.parametrize(
    "v,expected",
    [
        (8, True),
        (9, False),
    ],
)
def test_any(v, expected):
    result = (
        Iter.range(10)
        .map(lambda x: x * 2)
        .filter(lambda x: x > 5)
        .map(lambda x: x == v)
        .any()
    )
    assert result == expected


@pytest.mark.parametrize(
    "v,expected",
    [
        (5, True),
        (6, False),
    ],
)
def test_all(v, expected):
    assert (
        len(
            Iter.range(0, 100, 5)
            .filter(lambda x: x > 5 and x < 90)
            .map(lambda x: x % v == 0)
            .collect()
        )
        > 0
    )
    result = (
        Iter.range(0, 100, 5)
        .filter(lambda x: x > 5 and x < 90)
        .map(lambda x: x % v == 0)
        .all()
    )
    assert result == expected


def test_sum():
    result = Iter.range(5).sum()
    assert result == 10


@pytest.mark.parametrize(
    "input,glue,output",
    [
        ("abc", "-", "a-b-c"),
        (iter("abc"), "-", "a-b-c"),
        (b"abc", b"-", b"a-b-c"),
        (b"abc", b"--", b"a--b--c"),
        (bytearray(b"abc"), b"--", b"a--b--c"),
        (b"abc", bytearray(b"--"), b"a--b--c"),
    ],
)
def test_concat(input, glue, output):
    assert concat(input, glue) == output


def test_concat_error():
    with pytest.raises(ValueError):
        assert concat("caleb", None) == "c-a-l-e-b"


@pytest.mark.parametrize(
    "input,glue,output",
    [
        ([1, 2, 3], None, [1, 2, 3]),
        ([1, 2, 3], 0, [1, 0, 2, 0, 3]),
        (iter([1, 2, 3]), 0, [1, 0, 2, 0, 3]),
        ([1, 2, 3], [8, 9], [1, [8, 9], 2, [8, 9], 3]),
        ("abc", "-", list("a-b-c")),
        pytest.param(
            b"abc",
            b"-",
            list(b"a-b-c"),
            marks=pytest.mark.xfail(reason="bytes are dumb"),
        ),
        # (b'abc', b'--', list(b'a--b--c')),
        # (bytearray(b'abc'), b'--', list(b'a--b--c')),
        # (b'abc', bytearray(b'--'), list(b'a--b--c')),
    ],
)
def test_insert_separator(input, glue, output):
    assert list(insert_separator(input, glue)) == output


def test_insert():
    result = Iter("caleb").insert("x").collect(container=str)
    # Note the difference as compare to the output of interleave
    assert result == "cxaxlxexb"


@pytest.mark.parametrize(
    "elem,times,expected",
    [
        ("x", 3, ["x"] * 3),
    ],
)
def test_repeat_finite(elem, times, expected):
    result = Iter.repeat(elem, times=times).collect()
    assert result == expected


@pytest.mark.parametrize(
    "elem,taken,expected",
    [
        ("x", 0, []),
        ("x", 3, ["x"] * 3),
    ],
)
def test_repeat_infinite(elem, taken, expected):
    result = Iter.repeat(elem).take(taken).collect()
    assert result == expected


@pytest.mark.parametrize(
    "arg,func,expected",
    [
        ([1, 2, 3, 4, 5], None, [1, 3, 6, 10, 15]),
        ([1, 2, 3, 4, 5], lambda x, y: x * y, [1, 2, 6, 24, 120]),
    ],
)
def test_accumulate(arg, func, expected):
    result = Iter(arg).accumulate(func).collect()
    assert result == expected


@pytest.mark.parametrize(
    "arg,iterables,expected",
    [
        ("ABC", ["DEF"], ["A", "B", "C", "D", "E", "F"]),
    ],
)
def test_chain(arg, iterables, expected):
    result = Iter(arg).chain(*iterables).collect()
    assert result == expected


@pytest.mark.parametrize(
    "args,expected",
    [
        (["ABC", "DEF"], ["A", "B", "C", "D", "E", "F"]),
    ],
)
def test_from_iterable(args, expected):
    result = Iter(args).chain_from_iterable().collect()
    assert result == expected


@pytest.mark.parametrize(
    "arg,selectors,expected",
    [
        ("ABCDEF", [1, 0, 1, 0, 1, 1], ["A", "C", "E", "F"]),
    ],
)
def test_compress(arg, selectors, expected):
    result = Iter(arg).compress(selectors).collect()
    assert result == expected


@pytest.mark.parametrize(
    "arg,pred,expected",
    [
        ([1, 4, 6, 4, 1], lambda x: x < 5, [6, 4, 1]),
    ],
)
def test_dropwhile(arg, pred, expected):
    result = Iter(arg).dropwhile(pred).collect()
    assert result == expected


@pytest.mark.parametrize(
    "arg,pred,expected",
    [
        (range(10), lambda x: x % 2, [0, 2, 4, 6, 8]),
    ],
)
def test_filterfalse(arg, pred, expected):
    result = Iter(arg).filterfalse(pred).collect()
    assert result == expected


@pytest.mark.parametrize(
    "arg,key,expected",
    [
        ("AABCBA", None, [["A", "A", "A"], ["B", "B"], ["C"]]),
    ],
)
def test_groupby(arg, key, expected):
    # NOTE: this doesn't work because the underlying iterator is shared,
    # see the docs for more info. You have to do something with the groups
    # while the iterator is in-flight.
    result = Iter(sorted(arg)).groupby(key).map(lambda tup: list(tup[1])).collect()
    assert result == expected


@pytest.mark.parametrize(
    "arg,func,expected",
    [
        (
            [("a", 1), ("b", 2), ("c", 3)],
            lambda c, n: f"{c}={n}",
            ["a=1", "b=2", "c=3"],
        ),
    ],
)
def test_starmap(arg, func, expected):
    result = Iter(arg).starmap(func).collect()
    assert result == expected


def test_takewhile():
    result = Iter([1, 4, 6, 4, 1]).takewhile(lambda x: x < 5).collect()
    assert result == [1, 4]


def test_tee():
    x, y = Iter([1, 4, 6, 4, 1]).tee()
    assert x.take(2).collect() == [1, 4]
    assert y.take(3).collect() == [1, 4, 6]

    result = Iter([1, 4, 6, 4, 1]).tee(3).nth(2).take(2).collect()
    assert result == [1, 4]


def test_zip():
    result = Iter([1, 4, 6, 4, 1]).zip("abc").collect()
    assert result == [(1, "a"), (4, "b"), (6, "c")]


def test_zip_longest():
    result = Iter([1, 4, 6, 4, 1]).zip_longest("abc").collect()
    assert result == [(1, "a"), (4, "b"), (6, "c"), (4, None), (1, None)]


def test_non_iterable_error():
    """
    Test the error message we give if a user accidentally passes a
    generator function without evaluating it. This is only to give
    a friendlier error message for a fairly common error. I still do
    it sometimes so this helps me as much as anyone.
    """

    def gen():
        for x in range(5):
            yield x

    with pytest.raises(TypeError, match="seems you passed"):
        Iter(gen)

    with pytest.raises(TypeError, match="'object' object is not iterable"):
        Iter(object())
