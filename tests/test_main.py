import pytest

from excitertools import Iter


def test_basic():
    it = Iter(range(5))
    assert list(it) == [0, 1, 2, 3, 4]


def test_collect():
    it = Iter(range(5))
    assert it.collect() == [0, 1, 2, 3, 4]


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
