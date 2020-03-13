import pytest
from excitertools import Iter


def test_basic():
    it = Iter(range(5))
    assert list(it) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize('val,wargs,jargs,result', [
    ('caleb', [], [], '(caleb)'),
    ('caleb', ['""'], [], '"caleb"'),
    ('caleb', [('"""', '"""')], [], '"""caleb"""'),
    ('caleb', [], [','], '(,c,a,l,e,b,)'),
])
def test_wrap(val, wargs, jargs, result):
    assert Iter(val).wrap(*wargs).join_str(*jargs) == result


@pytest.mark.parametrize('val,wargs,jargs,result', [
    (iter('caleb'), [(iter('abc'), iter('def'))], [], 'abccalebdef'),
])
def test_wrap_iterators(val, wargs, jargs, result):
    assert Iter(val).wrap(*wargs).join_str(*jargs) == result


@pytest.mark.parametrize('val,wargs,result', [
    ([1,2,3], [([0, 0], [9, 9])], [0, 0, 1, 2, 3, 9, 9]),
])
def test_wrap_iterators2(val, wargs, result):
    assert Iter(val).wrap(*wargs).collect() == result


def test_wrap_error():
    with pytest.raises(ValueError):
        Iter('caleb').wrap(ends='x').join()


def test_enumerate():
    result = Iter('caleb').enumerate().collect()
    assert result == [
        (0, 'c'),
        (1, 'a'),
        (2, 'l'),
        (3, 'e'),
        (4, 'b'),
    ]

    result = Iter('caleb').enumerate().filter(
        lambda x: x[1] in 'ab'
    ).collect()
    assert result == [
        (1, 'a'),
        (4, 'b'),
    ]
