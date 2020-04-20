"""

This is the documentation.

Here is a heading:

Heading
-------

That was the heading.


"""
from __future__ import annotations
import itertools
import functools
import operator
from collections import UserDict
import typing
from typing import (
    Iterable,
    Tuple,
    Any,
    TypeVar,
    List,
    Iterator,
    Sequence,
    Dict,
    AnyStr,
    Set,
    FrozenSet,
    Callable,
    Union,
    Generic,
    MutableSequence,
    Sized,
    Collection,
    Type,
    Optional,
)
import collections.abc

import more_itertools

__all__ = [
    "Iter",
    "insert_separator",
    "concat",
]

T = TypeVar("T")
C = TypeVar("C")
K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")


class class_or_instancemethod(classmethod):
    """From: https://stackoverflow.com/a/28238047/170656"""

    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


class Iter(Generic[T]):
    """
    This is the docstring for the ``Iter`` class.

    Each of the following methods of ``Iter`` describe how they work.
    """
    x: Iterator[T]

    def __init__(self, x: Iterable[T]):
        if isinstance(x, collections.abc.Iterator):
            self.x = x
        else:
            self.x = iter(x)

    def __iter__(self) -> Iterator[T]:
        return self.x

    def __next__(self) -> T:
        return next(self.x)

    def collect(self, container=list) -> List[T]:
        """
        .. code-block:: python

            >>> Iter('abc').collect()
            ['a', 'b', 'c']
            >>> Iter('abc').collect(str)
            'abc'
            >>> Iter('abcaaaabbbbccc').collect(set) == {'a', 'b', 'c'}
            True

        """
        if container == str:
            return self.concat("")
        elif container == bytes:
            return self.concat(b"")
        else:
            return container(self)

    # File operations

    @classmethod
    def open(
        cls,
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ) -> Iter:
        """ Docstring TBD """
        def inner():
            with open(
                file=file,
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
                closefd=closefd,
                opener=opener,
            ) as f:
                yield from f

        return cls(inner())

    # Standard utilities

    @classmethod
    def range(cls, *args) -> Iter[int]:
        """ Docstring TBD """
        return cls(range(*args))

    def zip(self, *iterables: Any) -> Iter[Tuple[T, ...]]:
        """ Docstring TBD """
        return Iter(zip(self.x, *iterables))

    def any(self) -> bool:
        """ Docstring TBD """
        return any(self.x)

    def all(self) -> bool:
        """ Docstring TBD """
        return all(self.x)

    def enumerate(self) -> Iter[Tuple[int, T]]:
        """ Docstring TBD """
        return Iter(enumerate(self.x))

    def dict(self) -> Dict:
        """ Docstring TBD """
        return dict(self.x)

    ###

    def map(self, func: Union[Callable[..., C], str]) -> Iter[C]:
        """
        >>> result = Iter('caleb').map(lambda x: (x, ord(x))).dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

        >>> result = Iter('caleb').map('x, ord(x)').dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}
        """
        if isinstance(func, str):
            return Iter(map(lambda x: eval(func), self.x))
        else:
            return Iter(map(func, self.x))

    def filter(self, *args) -> Iter[T]:
        """ Docstring TBD """
        return Iter(filter(*args, self.x))

    def reduce(self, func: Callable[..., T], *args) -> T:
        """ Docstring TBD """
        return functools.reduce(func, self.x, *args)

    def sum(self):
        """ Docstring TBD """
        return sum(self.x)

    def concat(self, glue: AnyStr) -> AnyStr:
        """ Docstring TBD """
        return concat(self.x, glue)

    def insert(self, glue: C) -> Iter[Union[C, T]]:
        """ Docstring TBD """
        return Iter(insert_separator(self, glue))

    # standard library
    # =================

    # Infinite iterators

    @classmethod
    def count(cls, *args) -> Iter[int]:
        """ Docstring TBD """
        return cls(itertools.count(*args))

    def cycle(self) -> Iter[T]:
        """ Docstring TBD """
        return Iter(itertools.cycle(self.x))

    @classmethod
    def repeat(cls, elem: C, times=None) -> Iter[C]:
        """ Docstring TBD """
        # TODO: does it really work like this? Wow.
        if times:
            return Iter(itertools.repeat(elem, times=times))
        else:
            return Iter(itertools.repeat(elem))

    # Iterators terminating on the shortest input sequence
    def accumulate(self, func):
        """ Docstring TBD """
        return Iter(itertools.accumulate(self.x, func))

    def chain(self, *iterables: Iterable[T]) -> Iter[T]:
        """ Docstring TBD """
        return Iter(itertools.chain(self.x, *iterables))

    def chain_from_iterable(self) -> Iter[T]:
        """ Docstring TBD """
        return Iter(itertools.chain.from_iterable(self.x))

    def compress(self, selectors):
        """ Docstring TBD """
        return Iter(itertools.compress(self.x, selectors))

    def dropwhile(self, pred):
        """ Docstring TBD """
        return Iter(itertools.dropwhile(pred, self.x))

    def filterfalse(self, pred):
        """ Docstring TBD """
        return Iter(itertools.filterfalse(pred, self.x))

    def groupby(self, key=None):
        """ Docstring TBD """
        return Iter(itertools.groupby(self.x, key=key))

    def islice(self, *args) -> Iter:
        """ Docstring TBD """
        return Iter(itertools.islice(self.x, *args))

    def starmap(self, func):
        """ Docstring TBD """
        return Iter(itertools.starmap(func, self.x))

    def takewhile(self, pred):
        """ Docstring TBD """
        return Iter(itertools.takewhile(pred, self.x))

    def tee(self, n=2):
        """ Docstring TBD """
        # Pay attention
        return Iter(Iter(_) for _ in itertools.tee(self.x, n))

    def zip_longest(self, *iterables, fillvalue=None):
        """ Docstring TBD """
        return Iter(itertools.zip_longest(self.x, *iterables, fillvalue=fillvalue))

    # more-itertools
    # ===============

    # Grouping

    def chunked(self, n: int) -> Iter:
        """ Docstring TBD """
        return Iter(more_itertools.chunked(self.x, n))

    def ichunked(self, n: int) -> Iter:
        """ Docstring TBD """
        return Iter(Iter(it) for it in more_itertools.ichunked(self.x, n))

    @classmethod
    def sliced(cls, seq: Sequence, n: int) -> Iter:
        """ Docstring TBD """
        return Iter(more_itertools.sliced(seq, n))

    def distribute(self, n: int) -> Iter:
        """ Docstring TBD """
        return Iter((Iter(x) for x in Iter(more_itertools.distribute(n, self.x))))

    def divide(self, n: int) -> Iter:
        """ Docstring TBD """
        return Iter(Iter(x) for x in more_itertools.divide(n, self.x))

    def split_at(self, pred):
        """ Docstring TBD """
        return Iter(more_itertools.split_at(self.x, pred))

    def split_before(self, pred):
        """ Docstring TBD """
        return Iter(more_itertools.split_before(self.x, pred))

    def split_after(self, pred):
        """ Docstring TBD """
        return Iter(more_itertools.split_after(self.x, pred))

    def split_into(self, sizes):
        """ Docstring TBD """
        return Iter(more_itertools.split_into(self.x, sizes))

    def split_when(self, pred):
        """ Docstring TBD """
        return Iter(more_itertools.split_when(self.x, pred))

    def bucket(self, key, validator=None):
        """ Docstring TBD """
        class _bucket(more_itertools.bucket):
            def __iter__(self):
                return Iter(super().__iter__())

            def __getitem__(self, item):
                return Iter(super().__getitem__(item))

        return _bucket(self.x, key, validator=validator)

    def unzip(self):
        """ Docstring TBD """
        return Iter(Iter(x) for x in more_itertools.unzip(self.x))

    def grouper(self, n: int, fillvalue=None) -> Iter:
        """ Docstring TBD """
        return Iter(more_itertools.grouper(self.x, n, fillvalue=fillvalue))

    def partition(self, pred) -> Iter:
        """ Docstring TBD """
        left, right = more_itertools.partition(pred, self.x)
        return Iter((Iter(left), Iter(right)))

    # Lookahead and lookback

    def spy(self, n=1) -> Tuple[Iter, Iter]:
        """ Docstring TBD """
        head, iterable = more_itertools.spy(self.x, n)
        return Iter(head), Iter(iterable)

    def peekable(self) -> more_itertools.peekable:
        """ Docstring TBD """
        class _peekable(more_itertools.peekable):
            def __iter__(self):
                return Iter(super().__iter__())

            def __getitem__(self, item):
                return super().__getitem__(item)

            # TODO: need to somehow combine peekable and Iter
            # def prepend(self, *items) -> Iter:
            #     super().prepend(*items)
            #     return Iter(self)

        return _peekable(self.x)

    def seekable(self) -> more_itertools.seekable:
        """ Docstring TBD """
        class _seekable(more_itertools.seekable):
            def __iter__(self):
                return Iter(super().__iter__())

        return _seekable(self.x)

    # Windowing

    def windowed(self, n, fillvalue=None, step=1) -> Iter:
        """ Docstring TBD """
        return Iter(more_itertools.windowed(self.x, n, fillvalue=fillvalue, step=step))

    def substrings(self):
        """ Docstring TBD """
        return Iter(more_itertools.substrings(self.x))

    def substrings_indexes(self, reverse=False):
        """ Docstring TBD """
        return Iter(more_itertools.substrings_indexes(list(self.x), reverse=reverse))

    def stagger(self, offsets=(-1, 0, 1), longest=False, fillvalue=None):
        """
        .. code-block:: python

            >>> Iter([0, 1, 2, 3]).stagger().collect()
            [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
            >>> Iter(range(8)).stagger(offsets=(0, 2, 4)).collect()
            [(0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7)]
            >>> Iter([0, 1, 2, 3]).stagger(longest=True).collect()
            [(None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]

        """
        return Iter(
            more_itertools.stagger(
                self.x, offsets=offsets, longest=longest, fillvalue=fillvalue,
            )
        )

    def pairwise(self):
        """
        See https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise

        .. code-block:: python

            >>> Iter.count().pairwise().take(4).collect()
            [(0, 1), (1, 2), (2, 3), (3, 4)]
        """
        return Iter(more_itertools.pairwise(self.x))

    # Augmenting

    def count_cycle(self, n=None) -> Iter:
        """

        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle

        .. code-block:: python

            >>> Iter('AB').count_cycle(3).collect()
            [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]

        """
        return Iter(more_itertools.count_cycle(self.x, n=n))

    def intersperse(self, e, n=1) -> Iter:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse

        .. code-block:: python

            >>> Iter([1, 2, 3, 4, 5]).intersperse('!').collect()
            [1, '!', 2, '!', 3, '!', 4, '!', 5]

            >>> Iter([1, 2, 3, 4, 5]).intersperse(None, n=2).collect()
            [1, 2, None, 3, 4, None, 5]

        """
        return Iter(more_itertools.intersperse(e, self.x, n=n))

    def padded(
        self,
        fillvalue: Optional[C] = None,
        n: Optional[int] = None,
        next_multiple: bool = False,
    ) -> Iter[Union[T, C]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded

        .. code-block:: python

            >>> Iter([1, 2, 3]).padded('?', 5).collect()
            [1, 2, 3, '?', '?']

            >>> Iter([1, 2, 3, 4]).padded(n=3, next_multiple=True).collect()
            [1, 2, 3, 4, None, None]

        """
        return Iter(
            more_itertools.padded(
                self.x, fillvalue=fillvalue, n=n, next_multiple=next_multiple,
            )
        )

    # repeat from upstream

    def repeat_last(self, default=None) -> Iter[T]:
        """
        https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeat_last

        .. code-block:: python

            >>> Iter(range(3)).repeat_last().islice(5).collect()
            [0, 1, 2, 2, 2]

            >>> Iter(range(0)).repeat_last(42).islice(5).collect()
            [42, 42, 42, 42, 42]

        """
        return Iter(more_itertools.repeat_last(self.x, default=default))

    def adjacent(self, pred, distance=1) -> Iter[Tuple[bool, T]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent

        .. code-block:: python

            >>> Iter(range(6)).adjacent(lambda x: x == 3).collect()
            [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]

            >>> Iter(range(6)).adjacent(lambda x: x == 3, distance=2).collect()
            [(False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)]


        """
        return Iter(more_itertools.adjacent(pred, self.x, distance=distance))

    def groupby_transform(
        self,
        keyfunc: Optional[Callable[..., K]] = None,
        valuefunc: Optional[Callable[..., V]] = None,
    ) -> Iter[Tuple[K, Iterable[V]]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.groupby_transform

        This example has been modified somewhat from the original. We're using
        ``starmap`` here to "unzip" the tuples produced by the group
        transform.

        .. code-block:: python

            >>> iterable = 'AaaABbBCcA'
            >>> keyfunc = lambda x: x.upper()
            >>> valuefunc = lambda x: x.lower()
            >>> (
            ...    Iter(iterable)
            ...        .groupby_transform(keyfunc, valuefunc)
            ...        .starmap(lambda k, g: (k, ''.join(g)))
            ...        .collect()
            ... )
            [('A', 'aaaa'), ('B', 'bbb'), ('C', 'cc'), ('A', 'a')]

            >>> from operator import itemgetter
            >>> keys = [0, 0, 1, 1, 1, 2, 2, 2, 3]
            >>> values = 'abcdefghi'
            >>> iterable = zip(keys, values)
            >>> (
            ...     Iter(iterable)
            ...        .groupby_transform(itemgetter(0), itemgetter(1))
            ...        .starmap(lambda k, g: (k, ''.join(g)))
            ...        .collect()
            ... )
            [(0, 'ab'), (1, 'cde'), (2, 'fgh'), (3, 'i')]

        """
        return Iter(
            more_itertools.groupby_transform(
                self.x, keyfunc=keyfunc, valuefunc=valuefunc,
            )
        )

    def padnone(self) -> Iter[Union[T, None]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone

        .. code-block:: python

            >>> Iter(range(3)).padnone().take(5).collect()
            [0, 1, 2, None, None]

        """
        return Iter(more_itertools.padnone(self.x))

    def ncycles(self, n) -> Iter[T]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles

        .. code-block:: python

            >>> Iter(['a', 'b']).ncycles(3).collect()
            ['a', 'b', 'a', 'b', 'a', 'b']

        """
        return Iter(more_itertools.ncycles(self.x, n))

    # Combining

    def collapse(self, base_type=None, levels=None) -> Iter:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.collapse

        .. code-block:: python

            >>> iterable = [(1, 2), ([3, 4], [[5], [6]])]
            >>> Iter(iterable).collapse().collect()
            [1, 2, 3, 4, 5, 6]

            >>> iterable = ['ab', ('cd', 'ef'), ['gh', 'ij']]
            >>> Iter(iterable).collapse(base_type=tuple).collect()
            ['ab', ('cd', 'ef'), 'gh', 'ij']

            >>> iterable = [('a', ['b']), ('c', ['d'])]
            >>> Iter(iterable).collapse().collect() # Fully flattened
            ['a', 'b', 'c', 'd']
            >>> Iter(iterable).collapse(levels=1).collect() # Only one level flattened
            ['a', ['b'], 'c', ['d']]

        """
        return Iter(more_itertools.collapse(self.x, base_type=base_type, levels=levels))

    @class_or_instancemethod
    def sort_together(self_or_cls, iterables, key_list=(0,), reverse=False):
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sort_together

        This can be called either as an instance method or a class method.
        The classmethod form is more convenient if all the iterables are
        already available. The instancemethod form is more convenient if
        one of the iterables already goes through some transformation.

        Here are examples from the classmethod form, which mirror the
        examples in the *more-itertools* documentation:

        .. code-block:: python

            >>> iterables = [(4, 3, 2, 1), ('a', 'b', 'c', 'd')]
            >>> Iter.sort_together(iterables).collect()
            [(1, 2, 3, 4), ('d', 'c', 'b', 'a')]

            >>> iterables = [(3, 1, 2), (0, 1, 0), ('c', 'b', 'a')]
            >>> Iter.sort_together(iterables, key_list=(1, 2)).collect()
            [(2, 3, 1), (0, 0, 1), ('a', 'c', 'b')]

            >>> Iter.sort_together([(1, 2, 3), ('c', 'b', 'a')], reverse=True).collect()
            [(3, 2, 1), ('a', 'b', 'c')]

        Here is an examples using the instancemethod form:

            >>> iterables = [('a', 'b', 'c', 'd')]
            >>> Iter([4, 3, 2, 1]).sort_together(iterables).collect()
            [(1, 2, 3, 4), ('d', 'c', 'b', 'a')]

        """
        if isinstance(self_or_cls, type):
            return self_or_cls(
                more_itertools.sort_together(
                    iterables, key_list=key_list, reverse=reverse,
                )
            )
        else:

            def _temp():
                """Temporary generator function as a way of prepending the
                iterator of this instance."""
                yield self_or_cls.x
                yield from iterables

            return Iter(
                more_itertools.sort_together(
                    _temp(), key_list=key_list, reverse=reverse,
                )
            )

    @class_or_instancemethod
    def interleave(self_or_cls, *iterables) -> Iter:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave

        Classmethod form:

        .. code-block:: python

            >>> Iter.interleave([1, 2, 3], [4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7]

        Instancemethod form:

        .. code-block:: python

            >>> Iter([1, 2, 3]).interleave([4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7]

        """
        if isinstance(self_or_cls, type):
            return Iter(more_itertools.interleave(*iterables))
        else:
            return Iter(more_itertools.interleave(self_or_cls, *iterables))

    @class_or_instancemethod
    def interleave_longest(self_or_cls, *iterables) -> Iter:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave_longest

        Classmethod form:

        .. code-block:: python

            >>> Iter.interleave_longest([1, 2, 3], [4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7, 3, 8]

        Instancemethod form:

        .. code-block:: python

            >>> Iter([1, 2, 3]).interleave_longest([4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7, 3, 8]

        """
        if isinstance(self_or_cls, type):
            return Iter(more_itertools.interleave_longest(*iterables))
        else:
            return Iter(more_itertools.interleave_longest(self_or_cls, *iterables))

    @classmethod
    def zip_offset(cls, *iterables, offsets, longest=False, fillvalue=None) -> Iter:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_offset

        .. code-block:: python

            >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1)).collect()
            [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e')]

            >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True).collect()
            [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e'), (None, 'f')]
        """
        # TODO: also make an instancemethod
        return cls(
            more_itertools.zip_offset(
                *iterables, offsets=offsets, longest=longest, fillvalue=fillvalue,
            )
        )

    def dotproduct(self, vec2: Iterable):
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.dotproduct

        .. code-block:: python

            >>> Iter([10, 10]).dotproduct([20, 20])
            400
        """
        return more_itertools.dotproduct(self.x, vec2)

    def flatten(self) -> Iter[T]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten

        .. code-block:: python

            >>> Iter([[0, 1], [2, 3]]).flatten().collect()
            [0, 1, 2, 3]

        """
        return Iter(more_itertools.flatten(self))

    @class_or_instancemethod
    def roundrobin(self_or_cls: Union[Type[T], T], *iterables: C) -> Iter[Union[T, C]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.roundrobin

        Classmethod form:

        .. code-block:: python

            >>> Iter.roundrobin('ABC', 'D', 'EF').collect()
            ['A', 'D', 'E', 'B', 'F', 'C']

        Instancemethod form:

        .. code-block:: python

            >>> Iter('ABC').roundrobin('D', 'EF').collect()
            ['A', 'D', 'E', 'B', 'F', 'C']

        """
        if isinstance(self_or_cls, type):
            return self_or_cls(more_itertools.roundrobin(*iterables))
        else:
            return Iter(more_itertools.roundrobin(self_or_cls, *iterables))

    def prepend(self, value: C) -> Iter[Union[T, C]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.prepend

        .. code-block:: python

            >>> value = '0'
            >>> iterator = ['1', '2', '3']
            >>> Iter(iterator).prepend(value).collect()
            ['0', '1', '2', '3']

        """
        return Iter(more_itertools.prepend(value, self))

    # Summarizing

    def ilen(self) -> int:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ilen

        .. code-block:: python

            >>> Iter(x for x in range(1000000) if x % 3 == 0).ilen()
            333334

        """
        return more_itertools.ilen(self)

    def unique_to_each(self) -> Iter[T]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each

        .. code-block:: python

            >>> Iter([{'A', 'B'}, {'B', 'C'}, {'B', 'D'}]).unique_to_each().collect()
            [['A'], ['C'], ['D']]

            >>> Iter(["mississippi", "missouri"]).unique_to_each().collect()
            [['p', 'p'], ['o', 'u', 'r']]
        """
        # TODO: this operation is a little different to the others that take
        #  a *iterables parameter. Not sure if this is what we want.
        return Iter(more_itertools.unique_to_each(*self))

    def sample(self, k=1, weights=None) -> Iter:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sample

        .. code-block:: python

            >>> iterable = range(100)
            >>> Iter(iterable).sample(5).collect()  # doctest: +SKIP
            [81, 60, 96, 16, 4]

            >>> iterable = range(100)
            >>> weights = (i * i + 1 for i in range(100))
            >>> Iter(iterable).sample(5, weights=weights)  # doctest: +SKIP
            [79, 67, 74, 66, 78]

            >>> data = "abcdefgh"
            >>> weights = range(1, len(data) + 1)
            >>> Iter(data).sample(k=len(data), weights=weights)  # doctest: +SKIP
            ['c', 'a', 'b', 'e', 'g', 'd', 'h', 'f']


            >>> # This one just to let the doctest run
            >>> iterable = range(100)
            >>> Iter(iterable).sample(5).map(lambda x: 0 <= x < 100).all()
            True

        """
        return Iter(more_itertools.sample(self.x, k=k, weights=weights))

    def consecutive_groups(self, ordering=lambda x: x):
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consecutive_groups

        .. code-block:: python

            >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
            >>> Iter(iterable).consecutive_groups().map(lambda g: list(g)).print('{v}').consume()
            [1]
            [10, 11, 12]
            [20]
            [30, 31, 32, 33]
            [40]

        """

        return Iter(more_itertools.consecutive_groups(self.x, ordering=ordering))

    def run_length_encode(self) -> Iter[Tuple[T, int]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length

        .. code-block:: python

            >>> uncompressed = 'abbcccdddd'
            >>> Iter(uncompressed).run_length_encode().collect()
            [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

        """
        return Iter(more_itertools.run_length.encode(self))

    def run_length_decode(self) -> Iter:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length

        .. code-block:: python

            >>> compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
            >>> Iter(compressed).run_length_decode().collect()
            ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

        """
        return Iter(more_itertools.run_length.decode(self))

    def map_reduce(self, keyfunc, valuefunc=None, reducefunc=None) -> Dict:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce

        This interface mirrors what *more-itertools* does in that it returns
        a dict. See ``map_reduce_it()`` for a slightly-modified interface
        that returns the dict items as another iterator.

        .. code-block:: python

            >>> keyfunc = lambda x: x.upper()
            >>> d = Iter('abbccc').map_reduce(keyfunc)
            >>> sorted(d.items())
            [('A', ['a']), ('B', ['b', 'b']), ('C', ['c', 'c', 'c'])]

            >>> keyfunc = lambda x: x.upper()
            >>> valuefunc = lambda x: 1
            >>> d = Iter('abbccc').map_reduce(keyfunc, valuefunc)
            >>> sorted(d.items())
            [('A', [1]), ('B', [1, 1]), ('C', [1, 1, 1])]

            >>> keyfunc = lambda x: x.upper()
            >>> valuefunc = lambda x: 1
            >>> reducefunc = sum
            >>> d = Iter('abbccc').map_reduce(keyfunc, valuefunc, reducefunc)
            >>> sorted(d.items())
            [('A', 1), ('B', 2), ('C', 3)]

        Note the warning given in the *more-itertools* docs about how
        lists are created before the reduce step. This means you always want
        to filter *before* applying map_reduce, not after.

        .. code-block:: python

            >>> all_items = range(30)
            >>> keyfunc = lambda x: x % 2  # Evens map to 0; odds to 1
            >>> categories = Iter(all_items).filter(lambda x: 10<=x<=20).map_reduce(keyfunc=keyfunc)
            >>> sorted(categories.items())
            [(0, [10, 12, 14, 16, 18, 20]), (1, [11, 13, 15, 17, 19])]
            >>> summaries = Iter(all_items).filter(lambda x: 10<=x<=20).map_reduce(keyfunc=keyfunc, reducefunc=sum)
            >>> sorted(summaries.items())
            [(0, 90), (1, 75)]

        """
        return more_itertools.map_reduce(self.x, keyfunc, valuefunc, reducefunc,)

    def map_reduce_it(
        self,
        keyfunc: Callable[..., K],
        valuefunc: Optional[Callable[..., V]] = None,
        reducefunc: Optional[Callable[..., R]] = None
    ) -> Iter[Tuple[K, R]]:
        """
        See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce

        .. code-block:: python

            >>> keyfunc = lambda x: x.upper()
            >>> Iter('abbccc').map_reduce_it(keyfunc).collect()
            [('A', ['a']), ('B', ['b', 'b']), ('C', ['c', 'c', 'c'])]

            >>> keyfunc = lambda x: x.upper()
            >>> valuefunc = lambda x: 1
            >>> Iter('abbccc').map_reduce_it(keyfunc, valuefunc).collect()
            [('A', [1]), ('B', [1, 1]), ('C', [1, 1, 1])]

            >>> keyfunc = lambda x: x.upper()
            >>> valuefunc = lambda x: 1
            >>> reducefunc = sum
            >>> Iter('abbccc').map_reduce_it(keyfunc, valuefunc, reducefunc).collect()
            [('A', 1), ('B', 2), ('C', 3)]

        """
        d = more_itertools.map_reduce(self.x, keyfunc, valuefunc, reducefunc,)
        return Iter(d.items())

    def exactly_n(self, n, predicate=bool) -> Iter:
        return Iter(more_itertools.exactly_n(self.x, n=n, predicate=predicate))

    def all_equal(self):
        raise NotImplementedError

    def first_true(self):
        raise NotImplementedError

    def quantify(self):
        raise NotImplementedError

    # Selecting

    def islice_extended(self, *args):
        return Iter(more_itertools.islice_extended(self.x, *args))

    def first(self):
        raise NotImplementedError

    def last(self):
        raise NotImplementedError

    def one(self):
        raise NotImplementedError

    def only(self, default=None, too_long=None) -> Any:
        return more_itertools.only(self.x, default=default, too_long=too_long)

    def strip(self, pred):
        return Iter(more_itertools.strip(self.x, pred))

    def lstrip(self, pred):
        return Iter(more_itertools.lstrip(self.x, pred))

    def rstrip(self, pred):
        return Iter(more_itertools.rstrip(self.x, pred))

    def filter_except(self, validator, *exceptions):
        return Iter(more_itertools.filter_except(validator, self.x, *exceptions))

    def map_except(self, function, *exceptions):
        return Iter(more_itertools.map_except(function, self.x, *exceptions))

    def nth_or_last(self):
        raise NotImplementedError

    def nth(self, n, default=None):
        return next(self.islice(n, None), default)

    def take(self, n: int) -> Iter:
        return Iter(more_itertools.take(n, self.x))

    def tail(self):
        raise NotImplementedError

    def unique_everseen(self):
        raise NotImplementedError

    def unique_justseen(self):
        raise NotImplementedError

    # Combinatorics

    def distinct_permutations(self):
        return Iter(more_itertools.distinct_permutations(self.x))

    def distinct_combinations(self, r):
        return Iter(more_itertools.distinct_combinations(self.x, r))

    def circular_shifts(self) -> Iter:
        return Iter(more_itertools.circular_shifts(self.x))

    def partitions(self) -> Iter:
        return Iter(more_itertools.partitions(self.x))

    def set_partitions(self, k=None) -> Iter:
        return Iter(more_itertools.set_partitions(self.x, k=k))

    def powerset(self):
        raise NotImplementedError

    def random_product(self):
        raise NotImplementedError

    def random_permutation(self):
        raise NotImplementedError

    def random_combination(self):
        raise NotImplementedError

    def random_combination_with_replacement(self):
        raise NotImplementedError

    def nth_combination(self):
        raise NotImplementedError

    # Wrapping

    def always_iterable(self):
        raise NotImplementedError

    def always_reversible(self):
        return Iter(more_itertools.always_reversible(self.x))

    def consumer(self):
        raise NotImplementedError

    def with_iter(self):
        raise NotImplementedError

    def iter_except(self):
        raise NotImplementedError

    # Others

    def locate(self, pred=bool, window_size=None) -> Iter:
        return Iter(more_itertools.locate(self.x, pred=pred, window_size=window_size))

    def rlocate(self, pred=bool, window_size=None) -> Iter:
        return Iter(more_itertools.rlocate(self.x, pred, window_size))

    def replace(self, pred, substitutes, count=None, window_size=1) -> Iter:
        return Iter(
            more_itertools.replace(
                self.x, pred, substitutes, count=count, window_size=window_size
            )
        )

    @classmethod
    def numeric_range(cls, *args) -> Iter:
        return Iter(more_itertools.numeric_range(*args))

    def side_effect(self, func, chunk_size=None, before=None, after=None):
        return Iter(
            more_itertools.side_effect(
                func, self.x, chunk_size=chunk_size, before=before, after=after
            )
        )

    def iterate(self):
        raise NotImplementedError

    def difference(self, func=operator.sub, *, initial=None):
        return Iter(more_itertools.difference(self.x, func=func, initial=initial,))

    def make_decorator(self):
        raise NotImplementedError

    def SequenceView(self):
        raise NotImplementedError

    def time_limited(self, limit_seconds) -> Iter:
        return Iter(more_itertools.time_limited(limit_seconds, self.x))

    def consume(self, n: Optional[int] = None) -> Optional[Iter[T]]:
        """ If n is not provided, the entire iterator is consumed and
        ``None`` is returned. Otherwise, an iterator will always be
        returned, even if n is greater than the number of items left in
        the iterator."""
        more_itertools.consume(self, n=n)
        if n is not None:
            return self

    def tabulate(self):
        raise NotImplementedError

    def repeatfunc(self):
        raise NotImplementedError

    # New

    def wrap(self, ends: Sequence[T, T] = "()"):
        """ Other examples for ends: '"' * 2, or '`' * 2, or '[]' etc. """
        if len(ends) != 2:
            raise ValueError("The ends must be a 2-length sequence")

        return Iter(itertools.chain.from_iterable([ends[0], self, ends[1]]))

    def print(self, template="{i}: {v}") -> Iter[T]:
        """
        Printing during the execution of an iterator. Mostly useful
        for debugging. Returns another iterator instance through which
        the original data is passed unchanged. This means you can include
        a `print()` step as necessary to observe data during iteration.

        .. code-block:: python

            >>> Iter('abc').print().collect()
            0: a
            1: b
            2: c
            ['a', 'b', 'c']

            >>> (
            ...    Iter(range(5))
            ...        .print('before filter {i}: {v}')
            ...        .filter(lambda x: x > 2)
            ...        .print('after filter {i}: {v}')
            ...        .collect()
            ... )
            before filter 0: 0
            before filter 1: 1
            before filter 2: 2
            before filter 3: 3
            after filter 0: 3
            before filter 4: 4
            after filter 1: 4
            [3, 4]

        """

        def _print(elem):
            i, v = elem
            print(template.format(**locals()))

        return self.enumerate().side_effect(_print).starmap(lambda i, elem: elem)


class IterDict(UserDict):
    def __iter__(self) -> Iter:
        return Iter(self.data.keys())

    def keys(self) -> Iter:
        return Iter(self.data.keys())

    def values(self) -> Iter:
        return Iter(self.data.values())

    def items(self) -> Iter:
        return Iter(self.data.items())

    def update(self, *args, **kwargs) -> IterDict:
        self.data.update(*args, **kwargs)
        return self


def insert_separator(iterable: Iterable[Any], glue: Any) -> Iterable[Any]:
    """ Similar functionality can be obtained with, e.g.,
    interleave, as in

    >>> result = Iter('caleb').interleave(Iter.repeat('x')).collect()
    >>> result == list('cxaxlxexbx')
    True

    But you'll see a trailing "x" there, which join avoids. join
    makes sure to only add the glue separator if another element
    has arrived.

    It can handle strings without any special considerations, but it doesn't
    do any special handling for bytes and bytearrays. For that, rather
    look at `concat()`.
    """
    if not isinstance(iterable, Iterator):
        iterable = iter(iterable)

    yield next(iterable)
    for item in iterable:
        if glue is not None:
            yield glue
        yield item


def concat(iterable: Iterable[AnyStr], glue: AnyStr) -> AnyStr:
    """Concatenate strings, bytes and bytearrays. It is careful to avoid the
     problem with single bytes becoming integers, and it looks at the value
     of `glue` to know whether to handle bytes or strings."""
    if isinstance(glue, (bytes, bytearray)):
        return glue.join(iterable[i : i + 1] for i, _ in enumerate(iterable))

    elif isinstance(glue, str):
        return glue.join(iterable)

    else:
        raise ValueError("Must be called with bytes, bytearray or str")


"""

Blah blah blah here is some more text

"""


