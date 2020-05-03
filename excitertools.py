"""

.. image:: https://github.com/cjrh/excitertools/workflows/Python%20application/badge.svg
    :target: https://github.com/cjrh/excitertools/actions

.. image:: https://coveralls.io/repos/github/cjrh/excitertools/badge.svg?branch=master
    :target: https://coveralls.io/github/cjrh/excitertools?branch=master

.. image:: https://img.shields.io/pypi/pyversions/excitertools.svg
    :target: https://pypi.python.org/pypi/excitertools

.. image:: https://img.shields.io/github/tag/cjrh/excitertools.svg
    :target: https://img.shields.io/github/tag/cjrh/excitertools.svg

.. image:: https://img.shields.io/badge/install-pip%20install%20excitertools-ff69b4.svg
    :target: https://img.shields.io/badge/install-pip%20install%20excitertools-ff69b4.svg

.. image:: https://img.shields.io/pypi/v/excitertools.svg
    :target: https://img.shields.io/pypi/v/excitertools.svg

.. image:: https://img.shields.io/badge/calver-YYYY.MM.MINOR-22bfda.svg
    :target: http://calver.org/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black


excitertools
############

itertools in the form of function call chaining

.. contents::
    :depth: 1

Related projects
****************

* My fork of a now-missing library: `chained-iterable <https://github.com/cjrh/chained-iterable>`_.

* `https://github.com/olirice/flupy <https://github.com/olirice/flupy>`_

* `https://github.com/ddstte/chiter <https://github.com/ddstte/chiter>`_

* `https://github.com/neverendingqs/pyiterable <https://github.com/neverendingqs/pyiterable>`_

* `https://github.com/alliefitter/iterable_collections <https://github.com/alliefitter/iterable_collections>`_

* `https://github.com/halprin/iterator-chain <https://github.com/halprin/iterator-chain>`_

* `https://github.com/jagill/python-chainz <https://github.com/jagill/python-chainz>`_

* `https://github.com/ZianVW/IterPipe <https://github.com/ZianVW/IterPipe>`_

* `https://github.com/Evelyn-H/iterchain <https://github.com/Evelyn-H/iterchain>`_

* `https://github.com/EntilZha/PyFunctional <https://github.com/EntilZha/PyFunctional>`_

Tangentially related:

* `https://github.com/jreese/aioitertools <https://github.com/jreese/aioitertools>`_


.. |warning| unicode:: U+26A0
.. |cool| unicode:: U+2728
.. |flux| unicode:: U+1F6E0
.. |source| unicode:: U+1F3A4
.. |sink| unicode:: U+1F3A7


API Documentation
#################

.. contents::
    :local:

"""
from __future__ import annotations
import string
import itertools
import functools
import operator
from collections import UserDict
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
    Callable,
    Union,
    Generic,
    Type,
    Optional,
)
import collections.abc
import queue

import more_itertools

__all__ = [
    "Iter",
    'range',
    'zip',
    'enumerate',
    'map',
    'filter',
    'count',
    'cycle',
    'repeat',
    'accumulate',
    'chain',
    'chain_from_iterable',
    'compress',
    'dropwhile',
    'filterfalse',
    'groupby',
    'islice',
    'starmap',
    'takewhile',
    'tee',
    'zip_longest',
    "insert_separator",
    "concat",
    "IterDict",
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


# First save the builtins
_range = __builtins__['range']
_zip = __builtins__['zip']
_enumerate = __builtins__['enumerate']
_map = __builtins__['map']
_filter = __builtins__['filter']

"""

-----

The following module-level functions, like range_, zip_ and so on, are 
intended to be used as replacements for their homonymous builtins. The
only difference between these and the builtin versions is that these 
return instances of the Iter_ class. Note that because Iter_ is itself
iterable, it means that the functions here can be used as drop-in 
replacements.

"""

def range(*args) -> Iter[int]:
    """ Replacement for the builtin ``range`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    All the same calling variations work because this function merely wraps
    the original function.

    .. code-block:: python

        >>> range(3).collect()
        [0, 1, 2]
        >>> range(1, 4).collect()
        [1, 2, 3]
        >>> range(1, 6, 2).collect()
        [1, 3, 5]
        >>> range(1, 101, 3).filter(lambda x: x % 7 == 0).collect()
        [7, 28, 49, 70, 91]

    This example multiples, element by element, the series [0:5] with the
    series [1:6]. Two things to note: Firstly, Iter.zip_ is used to emit
    the tuples from each series. Secondly, Iter.starmap_ is used to receive
    those tuples into separate arguments in the lambda.

    .. code-block:: python

        >>> range(5).zip(range(1, 6)).starmap(lambda x, y: x * y).collect()
        [0, 2, 6, 12, 20]

    When written in a single line as above, it can get difficult to follow
    the chain of logic if there are many processing steps. Parentheses in
    Python allow grouping such that expressions can be spread over multiple
    lines.

    This is the same example as the prior one, but formatted to be spread
    over several lines. This is much clearer:

    .. code-block:: python

        >>> # Written out differently
        >>> (
        ...     range(5)
        ...         .zip(range(1, 6))
        ...         .starmap(lambda x, y: x * y)
        ...         .collect()
        ... )
        [0, 2, 6, 12, 20]

    If you wanted the sum instead, it isn't necessary to do the collection
    at all:

    .. code-block:: python

        >>> (
        ...     range(5)
        ...         .zip(range(1, 6))
        ...         .starmap(lambda x, y: x * y)
        ...         .sum()
        ... )
        40

    """
    return Iter(_range(*args))


def zip(*iterables: Any) -> Iter[Tuple[T, ...]]:
    """ Replacement for the builtin ``zip`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining."""
    return Iter(__builtins__['zip'](*iterables))


def enumerate(iterable) -> Iter[Tuple[int, T]]:
    """ Replacement for the builtin ``enumerate`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> enumerate(string.ascii_lowercase).take(3).collect()
        [(0, 'a'), (1, 'b'), (2, 'c')]


    """
    return Iter(__builtins__['enumerate'](iterable))


def map(func: Union[Callable[..., C], str], iterable) -> Iter[C]:
    """ Replacement for the builtin ``map`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> result = map(lambda x: (x, ord(x)), 'caleb').dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

        >>> result = map('x, ord(x)', 'caleb').dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}
    """
    if isinstance(func, str):
        return Iter(_map(lambda x: eval(func), iterable))
    else:
        return Iter(_map(func, iterable))


def filter(function: Callable[[Any], ...], iterable: Iterable) -> Iter[T]:
    """ Replacement for the builtin ``filter`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> filter(lambda x: x % 3 == 0, range(10)).collect()
        [0, 3, 6, 9]


    """
    return Iter(_filter(function, iterable))


# standard library
# =================

# Infinite iterators

def count(start, step: int = 1) -> Iter[int]:
    """ Replacement for the itertools ``count`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> count(0).take(5).collect()
        [0, 1, 2, 3, 4]
        >>> count(0).take(0).collect()
        []
        >>> count(10).take(0).collect()
        []
        >>> count(10).take(5).collect()
        [10, 11, 12, 13, 14]
        >>> count(1).filter(lambda x: x > 10).take(5).collect()
        [11, 12, 13, 14, 15]

    """
    return Iter(itertools.count(start, step))


def cycle(iterable) -> Iter[T]:
    """ Replacement for the itertools ``count`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> cycle(range(3)).take(6).collect()
        [0, 1, 2, 0, 1, 2]
        >>> cycle([]).take(6).collect()
        []
        >>> cycle(range(3)).take(0).collect()
        []

    """
    return Iter(itertools.cycle(iterable))


def repeat(object: C, times=None) -> Iter[C]:
    """ Replacement for the itertools ``count`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> repeat('a').take(3).collect()
        ['a', 'a', 'a']
        >>> repeat([1, 2]).take(3).collect()
        [[1, 2], [1, 2], [1, 2]]
        >>> repeat([1, 2]).take(3).collapse().collect()
        [1, 2, 1, 2, 1, 2]
        >>> repeat([1, 2]).collapse().take(3).collect()
        [1, 2, 1]
        >>> repeat('a', times=3).collect()
        ['a', 'a', 'a']

    """
    # TODO: does it really work like this? Wow. https://bugs.python.org/issue34169
    if times:
        return Iter(itertools.repeat(object, times=times))
    else:
        return Iter(itertools.repeat(object))


"""

This next set of functions return iterators that terminate on the shortest 
input sequence.

"""

def accumulate(iterable, func=None, *, initial=None):
    """ Replacement for the itertools ``accumulate`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> accumulate([1, 2, 3, 4, 5]).collect()
        [1, 3, 6, 10, 15]
        >>> accumulate([1, 2, 3, 4, 5], initial=100).collect()
        [100, 101, 103, 106, 110, 115]
        >>> accumulate([1, 2, 3, 4, 5], operator.mul).collect()
        [1, 2, 6, 24, 120]
        >>> accumulate([]).collect()
        []
        >>> accumulate('abc').collect()
        ['a', 'ab', 'abc']
        >>> accumulate(b'abc').collect()
        [97, 195, 294]
        >>> accumulate(bytearray(b'abc')).collect()
        [97, 195, 294]

    """
    return Iter(itertools.accumulate(iterable, func, initial=initial))


def chain(*iterables: Iterable[T]) -> Iter[T]:
    """ Replacement for the itertools ``chain`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> chain('ABC', 'DEF').collect()
        ['A', 'B', 'C', 'D', 'E', 'F']
        >>> chain().collect()
        []

    """
    return Iter(itertools.chain(*iterables))


def chain_from_iterable(iterable) -> Iter[T]:
    """ Replacement for the itertools ``chain.from_iterable`` method.
    This version returns an instance of ``excitertools.Iter`` to allow
    further iterable chaining.

    .. code-block:: python

        >>> chain_from_iterable(['ABC', 'DEF']).collect()
        ['A', 'B', 'C', 'D', 'E', 'F']
        >>> chain_from_iterable([]).collect()
        []

    """
    return Iter(itertools.chain.from_iterable(iterable))


def compress(data, selectors):
    """ Replacement for the itertools ``compress`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> compress('ABCDEF', [1, 0, 1, 0, 1, 1]).collect()
        ['A', 'C', 'E', 'F']


    """
    return Iter(itertools.compress(data, selectors))


def dropwhile(pred, iterable):
    """ Replacement for the itertools ``dropwhile`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> dropwhile(lambda x: x < 4, range(6)).collect()
        [4, 5]

    """
    return Iter(itertools.dropwhile(pred, iterable))


def filterfalse(pred, iterable):
    """ Replacement for the itertools ``filterfalse`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> filterfalse(None, [2, 0, 3, None, 4, 0]).collect()
        [0, None, 0]

    """
    return Iter(itertools.filterfalse(pred, iterable))


def groupby(iterable, key=None):
    """ Replacement for the itertools ``groupby`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    groupby_ returns an iterator of a key and "grouper" iterable. In the
    example below, we use Iter.starmap_ to collect each grouper iterable
    into a list, as this makes it neater for display here in the docstring.

    .. code-block:: python

        >>> (
        ...     groupby(['john', 'jill', 'anne', 'jack'], key=lambda x: x[0])
        ...         .starmap(lambda k, g: (k, list(g)))
        ...         .collect()
        ... )
        [('j', ['john', 'jill']), ('a', ['anne']), ('j', ['jack'])]


    """
    return Iter(itertools.groupby(iterable, key=key))


def islice(iterable, *args) -> Iter:
    """ Replacement for the itertools ``islice`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> islice('ABCDEFG', 2).collect()
        ['A', 'B']
        >>> islice('ABCDEFG', 2, 4).collect()
        ['C', 'D']
        >>> islice('ABCDEFG', 2, None).collect()
        ['C', 'D', 'E', 'F', 'G']
        >>> islice('ABCDEFG', 0, None, 2).collect()
        ['A', 'C', 'E', 'G']

    """
    return Iter(itertools.islice(iterable, *args))


def starmap(func, iterable):
    """ Replacement for the itertools ``starmap`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> starmap(pow, [(2, 5), (3, 2), (10, 3)]).collect()
        [32, 9, 1000]

    """
    return Iter(itertools.starmap(func, iterable))


def takewhile(pred, iterable):
    """ Replacement for the itertools ``takewhile`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> takewhile(lambda x: x < 5, [1, 4, 6, 4, 1]).collect()
        [1, 4]

    """
    return Iter(itertools.takewhile(pred, iterable))


def tee(iterable, n=2):
    """ Replacement for the itertools ``tee`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> a, b = tee(range(5))
        >>> a.collect()
        [0, 1, 2, 3, 4]
        >>> b.sum()
        10

    It is also possible to operate on the returned iterators in the chain
    but it gets quite difficult to understand:

    .. code-block:: python

        >>> tee(range(5)).map(lambda it: it.sum()).collect()
        [10, 10]

    In the example above we passed in range_, but with *excitertools*
    it's usually more natural to push data sources further left:

    .. code-block:: python

        >>> range(5).tee().map(lambda it: it.sum()).collect()
        [10, 10]

    Pay close attention to the above. The map_ is acting on each of the
    copied iterators.

    """
    # Pay attention!
    return Iter(Iter(_) for _ in itertools.tee(iterable, n))


def zip_longest(*iterables, fillvalue=None):
    """ Replacement for the itertools ``zip_longest`` function.  This version returns
    an instance of ``excitertools.Iter`` to allow further iterable chaining.

    .. code-block:: python

        >>> zip_longest('ABCD', 'xy', fillvalue='-').collect()
        [('A', 'x'), ('B', 'y'), ('C', '-'), ('D', '-')]
        >>> (
        ...     zip_longest('ABCD', 'xy', fillvalue='-')
        ...         .map(lambda tup: concat(tup, ''))
        ...         .collect()
        ... )
        ['Ax', 'By', 'C-', 'D-']
        >>> (
        ...     zip_longest('ABCD', 'xy', fillvalue='-')
        ...         .starmap(operator.add)
        ...         .collect()
        ... )
        ['Ax', 'By', 'C-', 'D-']

    """
    return Iter(itertools.zip_longest(*iterables, fillvalue=fillvalue))


class Iter(Generic[T]):
    """
    This is the docstring for the ``Iter`` class.

    Each of the following methods of ``Iter`` describe how they work.

    Test warning: |warning|

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
        """
        |cool|

        Wrap the ``open()`` builtin precisely, but return an ``Iter``
        instance to allow function chaining on the result.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as td:
        ...     open(td + 'text.txt', 'w').writelines(['abc\\n', 'def\\n', 'ghi\\n'])
        ...     Iter.open(td + 'text.txt').filter(lambda line: 'def' in line).collect()
        ['def\\n']

        Note that this is a convenience method for *reading* from a file,
        not for writing. The function signature includes the ``mode``
        parameter for parity with the builtin ``open()`` function, but
        only reading is supported.

        """

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
        return Iter(_zip(self.x, *iterables))

    def any(self) -> bool:
        """ Docstring TBD """
        return any(self.x)

    def all(self) -> bool:
        """ Docstring TBD """
        return all(self.x)

    def enumerate(self) -> Iter[Tuple[int, T]]:
        """ Docstring TBD """
        return Iter(_enumerate(self.x))

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
            return Iter(_map(lambda x: eval(func), self.x))
        else:
            return Iter(_map(func, self.x))

    def filter(self, *args) -> Iter[T]:
        """ Docstring TBD """
        return Iter(_filter(*args, self.x))

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
    def accumulate(self, func=None, *, initial=None):
        """ Docstring TBD

        .. code-block:: python

            >>> Iter([1, 2, 3, 4, 5]).accumulate().collect()
            [1, 3, 6, 10, 15]
            >>> Iter([1, 2, 3, 4, 5]).accumulate(initial=100).collect()
            [100, 101, 103, 106, 110, 115]
            >>> Iter([1, 2, 3, 4, 5]).accumulate(operator.mul).collect()
            [1, 2, 6, 24, 120]

        """
        return Iter(itertools.accumulate(self.x, func, initial=initial))

    def chain(self, *iterables: Iterable[T]) -> Iter[T]:
        """ Docstring TBD

        .. code-block:: python

            >>> Iter('ABC').chain('DEF').collect()
            ['A', 'B', 'C', 'D', 'E', 'F']
            >>> Iter('ABC').chain().collect()
            ['A', 'B', 'C']

         """
        return Iter(itertools.chain(self.x, *iterables))

    def chain_from_iterable(self) -> Iter[T]:
        """ Docstring TBD

        .. code-block:: python

            >>> Iter(['ABC', 'DEF']).chain_from_iterable().collect()
            ['A', 'B', 'C', 'D', 'E', 'F']

         """
        return Iter(itertools.chain.from_iterable(self.x))

    def compress(self, selectors):
        """ Replacement for the itertools ``compress`` function.  This version returns
        an instance of ``excitertools.Iter`` to allow further iterable chaining.

        .. code-block:: python

            >>> Iter('ABCDEF').compress([1, 0, 1, 0, 1, 1]).collect()
            ['A', 'C', 'E', 'F']

        """
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

            >>> all_items = _range(30)
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

    def exactly_n(self, n, predicate=bool) -> bool:
        """Docstring TBD

        .. code-block:: python

            >>> Iter([True, True, False]).exactly_n(2)
            True

        """
        return more_itertools.exactly_n(self.x, n=n, predicate=predicate)

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

    @classmethod
    def repeatfunc(cls, func, *args, times=None):
        """Docstring TBD

        .. code-block:: python

            >>> Iter.repeatfunc(operator.add, 3, 5, times=4).collect()
            [8, 8, 8, 8]

        """
        return Iter(more_itertools.repeatfunc(func, times, *args))

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

    @classmethod
    def from_queue(cls, q: queue.Queue, timeout=None, sentinel=None):
        """
        |source|
        Wrap a queue with an iterator interface. This allows it to participate
        in chaining operations. The iterator will block while waiting for
        new values to appear on the queue. This is useful: it allows you
        to easily and safely pass data between threads or processes, and
        feed the incoming data into a pipeline.

        The sentinel value, default ``None``, will terminate the iterator.

        .. code-block:: python

            >>> q = queue.Queue()
            >>> # This line puts stuff onto a queue
            >>> range(10).chain([None]).map(q.put).consume()
            >>> # This is where we consume data from the queue:
            >>> Iter.from_queue(q).filter(lambda x: 2 < x < 9).collect()
            [3, 4, 5, 6, 7, 8]

        If ``None`` had not been chained onto the data, the iterator would
        have waited in Iter.collect_ forever.

        """
        return Iter.repeatfunc(q.get, timeout).takewhile(lambda v: v is not None)

    def into_queue(self, q: queue.Queue):
        """
        |sink|
        This is a sink, like Iter.collect_, that consumes data from
        an iterator chain and puts the data into the given queue.

        .. code-block:: python

            >>> q = queue.Queue()
            >>> # This demonstrates the queue sink
            >>> range(5).into_queue(q)
            >>> # Code below is only for verification
            >>> out = []
            >>> finished = False
            >>> while not finished:
            ...     try:
            ...         out.append(q.get_nowait())
            ...     except queue.Empty:
            ...         finished = True
            >>> out
            [0, 1, 2, 3, 4]

        """
        self.map(q.put).consume()


class IterDict(UserDict):
    """
    |flux|

    The idea here was to make a custom dict where several of
    the standard dict methods return ``Iter`` instances, which can then
    be chained. I'm not sure if this will be kept yet.
    """
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
    """Similar functionality can be obtained with, e.g.,
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
        return glue.join(iterable[i : i + 1] for i, _ in _enumerate(iterable))

    elif isinstance(glue, str):
        return glue.join(iterable)

    else:
        """ 
        This function can raise ``ValueError`` if called with something
        other than ``bytes``, ``bytearray`` or ``str``."""
        raise ValueError("Must be called with bytes, bytearray or str")


def from_queue(q: queue.Queue, timeout=None, sentinel=None) -> Iter:
    """
    |source|
    Wrap a queue with an iterator interface. This allows it to participate
    in chaining operations. The iterator will block while waiting for
    new values to appear on the queue. This is useful: it allows you
    to easily and safely pass data between threads or processes, and
    feed the incoming data into a pipeline.

    The sentinel value, default ``None``, will terminate the iterator.

    .. code-block:: python

        >>> q = queue.Queue()
        >>> # This line puts stuff onto a queue
        >>> range(10).chain([None]).map(q.put).consume()
        >>> from_queue(q).filter(lambda x: 2 < x < 9).collect()
        [3, 4, 5, 6, 7, 8]

    """
    return Iter.from_queue(
        q,
        timeout=timeout,
        sentinel=sentinel
    )


"""

Dev Instructions
################

For general dev:

.. code-block:: shell

    $ python -m venv venv
    $ source venv/bin/activate
    (venv) $ pip install -e .[dev,test]

To run the tests:

.. code-block:: shell

    (venv) $ pytest

To regenerate the file ``README.rst``:

.. code-block:: shell

    (venv) $ python regenerate_readme.py -m excitertools.py > README.rst

"""
