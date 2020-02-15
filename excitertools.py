from __future__ import annotations
import itertools
import operator
import functools

__all__ = [
    'Iter',
]

from typing import Iterable, Tuple, Any, TypeVar

T = TypeVar('T')

class Iter:
    def __init__(self, x):
        self.x = iter(x)

    # def __getattr__(self, name):
    #     func = getattr(itertools, name)
    #     new_func = lambda *args, **kwargs: Iter(func(self.x, *args, **kwargs))
    #     return new_func
    #
    # def __getattr__(self, name):
    #     return globals()[name]

    def __iter__(self):
        return self.x

    @classmethod
    def range(cls, *args) -> Iter:
        return cls(range(*args))

    def any(self) -> bool:
        return any(self.x)

    def all(self) -> bool:
        return all(self.x)

    def enumerate(self) -> Iter:
        return Iter(enumerate(self.x))

    def dict(self):
        return Iter(dict(self.x).items)

    ###

    def map(self, *args) -> Iter:
        print('map')
        return Iter(map(*args, self.x))

    def filter(self, *args):
        print('filter')
        return Iter(filter(*args, self.x))

    def reduce(self, func, *args):
        print(self.x)
        return functools.reduce(func, self.x, *args)

    ###

    @classmethod
    def count(cls, *args):
        return cls(itertools.count(*args))

    def cycle(self):
        return Iter(itertools.cycle(self.x))

    @classmethod
    def repeat(cls, elem, n=None):
        return Iter(itertools.repeat(elem, n))

    def collect(self):
        return list(self)

    def islice(self, *args):
        return Iter(itertools.islice(self.x, *args))
