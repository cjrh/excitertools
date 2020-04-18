from __future__ import annotations
import itertools
import functools
import operator
from collections import UserDict
from typing import Iterable, Tuple, Any, TypeVar, List, Iterator, Sequence, Dict, AnyStr, Set, FrozenSet, Callable, \
    Union
import collections.abc

from types import MethodType
from more_itertools.more import bucket as bucket_class

import more_itertools

__all__ = [
    'Iter',
    'insert_separator',
    'concat',
]

T = TypeVar('T')


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
        return glue.join(iterable[i:i + 1] for i, _ in enumerate(iterable))

    elif isinstance(glue, str):
        return glue.join(iterable)

    else:
        raise ValueError("Must be called with bytes, bytearray or str")


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

    def collect(self, container: T = list) -> T:
        if container == str:
            return self.concat('')
        elif container == bytes:
            return self.concat(b'')
        else:
            return container(self)

    # File operations

    @classmethod
    def open(cls, file, mode='r', buffering=-1, encoding=None, errors=None,
             newline=None, closefd=True, opener=None) -> Iter:
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
    def range(cls, *args) -> Iter:
        return cls(range(*args))

    def any(self) -> bool:
        return any(self.x)

    def all(self) -> bool:
        return all(self.x)

    def enumerate(self) -> Iter:
        return Iter(enumerate(self.x))

    def dict(self) -> Dict:
        return dict(self.x)

    ###

    def map(self, func: Union[Callable, str]) -> Iter:
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

    def filter(self, *args) -> Iter:
        return Iter(filter(*args, self.x))

    def reduce(self, func, *args):
        return functools.reduce(func, self.x, *args)

    def sum(self):
        return sum(self.x)

    def concat(self, glue: AnyStr) -> AnyStr:
        return concat(self.x, glue)

    def insert(self, glue: Any) -> Iter:
        return Iter(insert_separator(self, glue))

    ###

    @classmethod
    def count(cls, *args) -> Iter:
        return cls(itertools.count(*args))

    def cycle(self) -> Iter:
        return Iter(itertools.cycle(self.x))

    @classmethod
    def repeat(cls, elem, times=None) -> Iter:
        # TODO: does it really work like this? Wow.
        if times:
            return Iter(itertools.repeat(elem, times=times))
        else:
            return Iter(itertools.repeat(elem))

    def chain(self, *args):
        return Iter(itertools.chain(self.x, *args))

    def islice(self, *args) -> Iter:
        return Iter(itertools.islice(self.x, *args))

    # more-itertools
    #===============

    # Grouping

    def chunked(self, n: int) -> Iter:
        return Iter(more_itertools.chunked(self.x, n))

    def ichunked(self, n: int) -> Iter:
        return Iter(
            Iter(it) for it in more_itertools.ichunked(self.x, n)
        )

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

    def windowed(self, n, fillvalue=None, step=1) -> Iter:
        return Iter(more_itertools.windowed(self.x, n, fillvalue=fillvalue, step=step))

    def substrings(self):
        return Iter(more_itertools.substrings(self.x))

    def substrings_indexes(self, reverse=False):
        return Iter(more_itertools.substrings_indexes(self.x, reverse=reverse))

    def stagger(self, offsets=(-1, 0, 1), longest=False, fillvalue=None):
        return Iter(
            more_itertools.stagger(
                self.x,
                offsets=offsets,
                longest=longest,
                fillvalue=fillvalue,
            )
        )

    def pairwise(self):
        raise NotImplementedError

    # Augmenting

    def count_cycle(self, n=None) -> Iter:
        return Iter(more_itertools.count_cycle(self.x, n=n))

    def intersperse(self):
        raise NotImplementedError

    def padded(self, fillvalue=None, n=None, next_multiple=False):
        return Iter(
            more_itertools.padded(
                self.x,
                fillvalue=fillvalue,
                n=n,
                next_multiple=next_multiple,
            )
        )

    # repeat from upstream

    def repeat_last(self, default=None):
        return Iter(more_itertools.repeat_last(self.x, default=default))

    def adjacent(self, predicate, distance=1):
        return Iter(more_itertools.adjacent(predicate, self.x, distance=distance))

    def groupby_transform(self, keyfunc=None, valuefunc=None) -> Iter:
        return Iter(
            more_itertools.groupby_transform(
                self.x,
                keyfunc=keyfunc,
                valuefunc=valuefunc,
            )
        )

    def padnone(self):
        raise NotImplementedError

    def ncycles(self):
        raise NotImplementedError

    # Combining

    def collapse(self, base_type=None, levels=None):
        return Iter(more_itertools.collapse(self.x, base_type=base_type, levels=None))

    @classmethod
    def sort_together(cls, iterables, key_list=(0,), reverse=False):
        return cls(
            more_itertools.sort_together(
                iterables,
                key_list=key_list,
                reverse=reverse,
            )
        )

    def interleave(self, *iterables) -> Iter:
        return Iter(more_itertools.interleave(self, *iterables))

    def interleave_longest(self, *iterables) -> Iter:
        return Iter(more_itertools.interleave_longest(self, *iterables))

    @classmethod
    def zip_offset(cls, *iterables, offsets, longest=False, fillvalue=None):
        return cls(
            more_itertools.zip_offset(
                *iterables,
                offsets=offsets,
                longest=longest,
                fillvalue=fillvalue,
            )
        )

    def dotproduct(self):
        raise NotImplementedError

    def flatten(self):
        raise NotImplementedError

    def roundrobin(self):
        raise NotImplementedError

    def prepend(self):
        raise NotImplementedError

    # Summarizing

    def ilen(self):
        raise NotImplementedError

    def unique_to_each(self):
        raise NotImplementedError

    def sample(self, k, weights=None) -> Iter:
        return Iter(more_itertools.sample(self.x, k, weights=weights))

    def consecutive_groups(self, ordering=lambda x: x):
        return Iter(more_itertools.consecutive_groups(self.x, ordering=ordering))

    def run_length_encode(self) -> Iter:
        return Iter(more_itertools.run_length.encode(self.x))

    def run_length_decode(self) -> Iter:
        return Iter(more_itertools.run_length.decode(self.x))


    def map_reduce(self, keyfunc, valuefunc=None, reducefunc=None) -> Dict:
        return more_itertools.map_reduce(
            self.x,
            keyfunc,
            valuefunc,
            reducefunc,
        )

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
        return Iter(
            more_itertools.filter_except(validator, self.x, *exceptions))

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
        return Iter(
            more_itertools.locate(self.x, pred=pred, window_size=window_size)
        )

    def rlocate(self, pred=bool, window_size=None) -> Iter:
        return Iter(more_itertools.rlocate(self.x, pred, window_size))

    def replace(self, pred, substitutes, count=None, window_size=1) -> Iter:
        return Iter(more_itertools.replace(
            self.x, pred, substitutes, count=count, window_size=window_size
        ))

    @classmethod
    def numeric_range(cls, *args) -> Iter:
        return Iter(
            more_itertools.numeric_range(
                *args
            )
        )

    def side_effect(self, func, chunk_size=None, before=None, after=None):
        return Iter(
            more_itertools.side_effect(
                func, self.x, chunk_size=chunk_size, before=before, after=after
            )
        )

    def iterate(self):
        raise NotImplementedError

    def difference(self, func=operator.sub, *, initial=None):
        return Iter(
            more_itertools.difference(
                self.x,
                func=func,
                initial=initial,
            )
        )

    def make_decorator(self):
        raise NotImplementedError

    def SequenceView(self):
        raise NotImplementedError

    def time_limited(self, limit_seconds) -> Iter:
        return Iter(more_itertools.time_limited(limit_seconds, self.x))

    def consume(self):
        raise NotImplementedError

    def tabulate(self):
        raise NotImplementedError

    def repeatfunc(self):
        raise NotImplementedError

    # New

    def wrap(self, ends: Sequence[T, T] = '()'):
        """ Other examples for ends: '"' * 2, or '`' * 2, or '[]' etc. """
        if len(ends) != 2:
            raise ValueError("The ends must be a 2-length sequence")

        return Iter(
            itertools.chain.from_iterable(
                [ends[0], self, ends[1]]
            )
        )


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
