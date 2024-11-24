import pytest
from excitertools import Iter, IterDict


def test_basic():
    it = Iter(range(5))
    assert list(it) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "val,wargs,jargs,result",
    [
        ("caleb", [], [], "(caleb)"),
        ("caleb", ['""'], [], '"caleb"'),
        ("caleb", [('"""', '"""')], [], '"""caleb"""'),
        ("caleb", [], [","], "(,c,a,l,e,b,)"),
    ],
)
def test_wrap(val, wargs, jargs, result):
    assert Iter(val).wrap(*wargs).concat(*(jargs or [""])) == result


@pytest.mark.parametrize(
    "val,wargs,jargs,result",
    [
        (iter("caleb"), [(iter("abc"), iter("def"))], [""], "abccalebdef"),
    ],
)
def test_wrap_iterators(val, wargs, jargs, result):
    assert Iter(val).wrap(*wargs).concat(*jargs) == result


@pytest.mark.parametrize(
    "val,wargs,result",
    [
        ([1, 2, 3], [([0, 0], [9, 9])], [0, 0, 1, 2, 3, 9, 9]),
    ],
)
def test_wrap_iterators2(val, wargs, result):
    assert Iter(val).wrap(*wargs).collect() == result


def test_wrap_error():
    with pytest.raises(ValueError):
        Iter("caleb").wrap(ends="x").join()


def test_enumerate():
    result = Iter("caleb").enumerate().collect()
    assert result == [
        (0, "c"),
        (1, "a"),
        (2, "l"),
        (3, "e"),
        (4, "b"),
    ]

    result = Iter("caleb").enumerate().filter(lambda x: x[1] in "ab").collect()
    assert result == [
        (1, "a"),
        (4, "b"),
    ]


def test_dict():
    result = Iter("caleb").enumerate().dict()
    assert result == {0: "c", 1: "a", 2: "l", 3: "e", 4: "b"}

    result = Iter("caleb").map(lambda x: (x, ord(x))).dict()
    assert result == {"a": 97, "b": 98, "c": 99, "e": 101, "l": 108}

    result = Iter("caleb").map("x, ord(x)").dict()
    assert result == {"a": 97, "b": 98, "c": 99, "e": 101, "l": 108}


def test_iterdict():
    result = IterDict().__iter__().collect()
    assert result == []

    result = IterDict(dict(a=1, b=2)).__iter__().collect()
    assert result == ["a", "b"]

    result = IterDict([("a", 1), ("b", 2)]).__iter__().collect()
    assert result == ["a", "b"]

    result = IterDict(a=1, b=2, c=3).__iter__().collect()
    assert result == ["a", "b", "c"]

    result = [x for x in IterDict(a=1, b=2, c=3)]
    assert result == ["a", "b", "c"]

    result = IterDict(a=1, b=2, c=3).keys().collect()
    assert result == ["a", "b", "c"]

    result = IterDict(a=1, b=2, c=3).values().collect()
    assert result == [1, 2, 3]

    result = IterDict(a=1, b=2, c=3).items().collect()
    assert result == [("a", 1), ("b", 2), ("c", 3)]

    result = IterDict(a=1, b=2, c=3).items().filter(lambda x: x[1] % 2).collect()
    assert result == [("a", 1), ("c", 3)]

    result = IterDict(a=1, b=2, c=3).items().filter(lambda x: x[0] == "b").collect()
    assert result == [("b", 2)]

    result = (
        IterDict(a=1, b=2, c=3)
        .update(d=4, e=5)
        .items()
        .filter(lambda x: x[1] % 2)
        .collect()
    )
    assert result == [("a", 1), ("c", 3), ("e", 5)]
