from __future__ import annotations
import itertools
import functools
from typing import Iterable, Tuple, Any, TypeVar, List, Iterator, Sequence
import collections.abc
from types import MethodType
from more_itertools.more import bucket as bucket_class

import more_itertools

__all__ = [
    'Iter',
]

T = TypeVar('T')


class Iter:
    def __init__(self, x):
        if isinstance(x, collections.abc.Iterator):
            self.x = x
        else:
            self.x = iter(x)

    # def __getattr__(self, name):
    #     func = getattr(itertools, name)
    #     new_func = lambda *args, **kwargs: Iter(func(self.x, *args, **kwargs))
    #     return new_func
    #
    # def __getattr__(self, name):
    #     return globals()[name]

    def __iter__(self) -> Iterator:
        return self.x

    def __next__(self):
        return next(self.x)

    def collect(self) -> List:
        return list(self)

    @classmethod
    def range(cls, *args) -> Iter:
        return cls(range(*args))

    def any(self) -> bool:
        return any(self.x)

    def all(self) -> bool:
        return all(self.x)

    def enumerate(self) -> Iter:
        return Iter(enumerate(self.x))

    def dict(self) -> Iter:
        return Iter(dict(self.x).items)

    ###

    def map(self, *args) -> Iter:
        return Iter(map(*args, self.x))

    def filter(self, *args) -> Iter:
        return Iter(filter(*args, self.x))

    def reduce(self, func, *args):
        return functools.reduce(func, self.x, *args)

    def sum(self):
        return sum(self.x)

    ###

    @classmethod
    def count(cls, *args) -> Iter:
        return cls(itertools.count(*args))

    def cycle(self) -> Iter:
        return Iter(itertools.cycle(self.x))

    @classmethod
    def repeat(cls, elem, n=None) -> Iter:
        return Iter(itertools.repeat(elem, n))

    def chain(self, *args):
        return Iter(itertools.chain(self.x, *args))

    def islice(self, *args) -> Iter:
        return Iter(itertools.islice(self.x, *args))



    # more-itertools

    def chunked(self, n: int) -> Iter:
        return Iter(more_itertools.chunked(self.x, n))

    def ichunked(self, n: int) -> List[Iter]:
        return [
            Iter(it) for it in more_itertools.ichunked(self.x, n)
        ]

    @classmethod
    def sliced(cls, seq: Sequence, n: int) -> Iter:
        return Iter(more_itertools.sliced(seq, n))

    def distribute(self, n: int) -> Iter:
        return Iter((Iter(x) for x in Iter(more_itertools.distribute(n, self.x))))

    def divide(self, n: int) -> Iter:
        return Iter(Iter(x) for x in more_itertools.divide(n, self.x))

    def split_at(self, pred):
        return Iter(more_itertools.split_at(self.x, pred))

    def split_before(self, pred):
        return Iter(more_itertools.split_before(self.x, pred))

    def split_after(self, pred):
        return Iter(more_itertools.split_after(self.x, pred))

    def split_into(self, sizes):
        return Iter(more_itertools.split_into(self.x, sizes))

    def split_when(self, pred):
        return Iter(more_itertools.split_when(self.x, pred))

    def bucket(self, key, validator=None):

        class _bucket(more_itertools.bucket):
            def __iter__(self):
                return Iter(super().__iter__())

            def __getitem__(self, item):
                return Iter(super().__getitem__(item))

        return _bucket(self.x, key, validator=validator)

    def unzip(self):
        return Iter(Iter(x) for x in more_itertools.unzip(self.x))

    def grouper(self, n: int, fillvalue=None) -> Iter:
        return Iter(more_itertools.grouper(self.x, n, fillvalue=fillvalue))

    def partition(self, pred) -> Iter:
        left, right = more_itertools.partition(pred, self.x)
        return Iter((Iter(left), Iter(right)))

    # Lookahead and lookback

    def spy(self, n=1) -> Tuple[Iter, Iter]:
        head, iterable = more_itertools.spy(self.x, n)
        return Iter(head), Iter(iterable)

    def peekable(self) -> more_itertools.peekable:

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

        class _seekable(more_itertools.seekable):
            def __iter__(self):
                return Iter(super().__iter__())

        return _seekable(self.x)

    # Windowing

    '''
    more_itertools.windowed
    more_itertools.substrings
    more_itertools.substrings_indexes
    more_itertools.stagger
    more_itertools.pairwise
    '''

    # Augmenting



    # Other

    def nth(self, n, default=None):
        return next(self.islice(n, None), default)
