import pytest

from excitertools import Iter, insert_separator, concat


def test_basic():
    it = Iter(range(5))
    assert list(it) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize('container,it,expected', [
    (list, range(5), [0, 1, 2, 3, 4]),
    (set, range(5), {0, 1, 2, 3, 4}),
    (dict, zip('abc', range(3)), dict(a=0, b=1, c=2)),
    (dict.fromkeys, 'abc', dict(a=None, b=None, c=None)),
    (frozenset, range(5), frozenset([0, 1, 2, 3, 4])),
    (str, ['a', 'b', 'c'], 'abc'),
    # (bytes, [b'a', b'b', b'c'], b'abc'),
])
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
        Iter
            .range(5)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 5)
            .reduce(lambda x, y: x + y)
    )
    assert result == 14


def func(x):
    return x * 2


def test_piecemeal_results():
    it = (
        Iter
            .range(5)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 5)
    )

    x = iter(it)
    assert next(x) == 6
    assert next(x) == 8


@pytest.mark.parametrize(
    'v,expected', [
        (8, True),
        (9, False),
    ]
)
def test_any(v, expected):
    result = (
        Iter
            .range(10)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 5)
            .map(lambda x: x == v)
            .any()
    )
    assert result == expected


@pytest.mark.parametrize(
    'v,expected', [
        (5, True),
        (6, False),
    ]
)
def test_all(v, expected):
    assert len(
        Iter
            .range(0, 100, 5)
            .filter(lambda x: x > 5 and x < 90)
            .map(lambda x: x % v == 0)
            .collect()
    ) > 0
    result = (
        Iter
            .range(0, 100, 5)
            .filter(lambda x: x > 5 and x < 90)
            .map(lambda x: x % v == 0)
            .all()
    )
    assert result == expected

def test_sum():
    result = Iter.range(5).sum()
    assert result == 10


@pytest.mark.parametrize('input,glue,output', [
    ('abc', '-', 'a-b-c'),
    (iter('abc'), '-', 'a-b-c'),
    (b'abc', b'-', b'a-b-c'),
    (b'abc', b'--', b'a--b--c'),
    (bytearray(b'abc'), b'--', b'a--b--c'),
    (b'abc', bytearray(b'--'), b'a--b--c'),
])
def test_concat(input, glue, output):
    assert concat(input, glue) == output


def test_concat_error():
    with pytest.raises(ValueError):
        assert concat('caleb', None) == 'c-a-l-e-b'


@pytest.mark.parametrize('input,glue,output', [
    ([1, 2, 3], None, [1, 2, 3]),
    ([1, 2, 3], 0, [1, 0, 2, 0, 3]),
    (iter([1, 2, 3]), 0, [1, 0, 2, 0, 3]),
    ([1, 2, 3], [8, 9], [1, [8, 9], 2, [8, 9], 3]),
    ('abc', '-', list('a-b-c')),
    pytest.param(b'abc', b'-', list(b'a-b-c'),
                 marks=pytest.mark.xfail(reason='bytes are dumb')),
    # (b'abc', b'--', list(b'a--b--c')),
    # (bytearray(b'abc'), b'--', list(b'a--b--c')),
    # (b'abc', bytearray(b'--'), list(b'a--b--c')),
])
def test_insert(input, glue, output):
    assert list(insert_separator(input, glue)) == output
