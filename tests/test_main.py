import excitertools


def test_basic():
    it = excitertools.iter_(range(5))
    assert list(it) == [0, 1, 2, 3, 4]


def test_collect():
    it = excitertools.iter_(range(5))
    assert it.collect() == [0, 1, 2, 3, 4]


def test_islice():
    it = excitertools.iter_(range(5))
    assert it.islice(3).collect() == [0, 1, 2]


def test_cycle_islice():
    it = excitertools.iter_(range(5))
    assert it.cycle().islice(6, 9, 2).collect() == [1, 3]


def test_count():
    it = excitertools.Iter.count().islice(3, 6)
    assert it.collect() == [3, 4, 5]


def test_map():
    it = excitertools.Iter.range(5).map(lambda x: x * 2)
    assert it.collect() == [0, 2, 4, 6, 8]

    it = excitertools.Iter.range(5).map(lambda x: x * 2).filter(lambda x: x > 5)
    assert it.collect() == [6, 8]

    result = (
        excitertools.Iter
            .range(5)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 5)
            .reduce(lambda x, y: x + y)
    )
    assert result == 14
