"""

.. image:: https://github.com/cjrh/excitertools/workflows/Python%20application/badge.svg
    :target: https://github.com/cjrh/excitertools/actions

.. image:: https://coveralls.io/repos/github/cjrh/excitertools/badge.svg?branch=master
    :target: https://coveralls.io/github/cjrh/excitertools?branch=master

.. image:: https://img.shields.io/pypi/pyversions/excitertools.svg
    :target: https://pypi.python.org/pypi/excitertools

.. image:: https://img.shields.io/pypi/implementation/excitertools.svg
    :target: https://pypi.python.org/pypi/excitertools

.. image:: https://img.shields.io/github/tag/cjrh/excitertools.svg
    :target: https://github.com/cjrh/excitertools

.. image:: https://img.shields.io/badge/install-pip%20install%20excitertools-ff69b4.svg
    :target: https://img.shields.io/badge/install-pip%20install%20excitertools-ff69b4.svg

.. image:: https://img.shields.io/badge/dependencies-more--itertools-4488ff.svg
    :target: https://more-itertools.readthedocs.io/en/stable/

.. image:: https://img.shields.io/pypi/v/excitertools.svg
    :target: https://img.shields.io/pypi/v/excitertools.svg

.. image:: https://img.shields.io/badge/calver-YYYY.MM.MINOR-22bfda.svg
    :target: http://calver.org/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black


.. _more-itertools: https://more-itertools.readthedocs.io/en/stable/index.html

.. _excitertools:

excitertools
############

``itertools`` reimagined as a `fluent interface <https://en.wikipedia.org/wiki/Fluent_interface>`_.

    In software engineering, a fluent interface is an object-oriented API whose design
    relies extensively on method chaining. Its goal is to increase code legibility by
    creating a domain-specific language (DSL). The term was coined in 2005 by Eric
    Evans and Martin Fowler.

    `*Wikipedia - Fluent Interface* <https://en.wikipedia.org/wiki/Fluent_interface>`_

Note that nearly all of the ``more-itertools`` extension library is included.

Demo
****

.. code-block:: python

    >>> from excitertools import *
    >>> import builtins, operator, queue
    >>> range(10).map(lambda x: x*7).filter(lambda x: x % 3 == 0).collect()
    [0, 21, 42, 63]
    >>> range(10).map(lambda x: x*7).filter(lambda x: x > 0 and x % 3 == 0).collect()
    [21, 42, 63]

When the lines get long, parens can be used to split up each instruction:

.. code-block:: python

    >>> (
    ...     range(10)
    ...         .map(lambda x: x*7)
    ...         .filter(lambda x: x % 3 == 0)
    ...         .collect()
    ... )
    [0, 21, 42, 63]

What's also interesting about that is how lambda's can easily contain these
processing chains, since an entire chain is a single expression. For
example:

.. code-block:: python

    >>> names = ['caleb', 'james', 'gina']
    >>> Iter(names).map(
    ...     lambda name: (
    ...         Iter(name)
    ...             .map(lambda c: c.upper() if c in 'aeiouy' else c)
    ...             .collect(str)
    ...     )
    ... ).collect()
    ['cAlEb', 'jAmEs', 'gInA']

Something I've noticed is that ``reduce`` seems easier to use and reason
about with this fluent interface, as compared to the conventional usage
as a standalone function; also, the operator module makes ``reduce`` quite
useful for simple cases:

.. code-block:: python

    >>> from operator import add, mul
    >>> (
    ...     range(10)
    ...     .map(lambda x: x*7)
    ...     .filter(lambda x: x > 0 and x % 3 == 0)
    ...     .reduce(add)
    ... )
    126
    >>> (
    ...     range(10)
    ...     .map(lambda x: x*7)
    ...     .filter(lambda x: x > 0 and x % 3 == 0)
    ...     .reduce(mul)
    ... )
    55566

.. contents::
    :depth: 1


.. |warning| unicode:: U+26A0
.. |cool| unicode:: U+2728
.. |flux| unicode:: U+1F6E0
.. |source| unicode:: U+1F3A4
.. |sink| unicode:: U+1F3A7
.. |inf| unicode:: U+267E


How to Understand the API Documentation
#######################################

Several symbols are used to indicate things about parts of the API:

- |source| This function is a *source*, meaning that it produces data
  that will be processed in an iterator chain.
- |sink| This function is a *sink*, meaning that it consumes data that
  was processed in an iterator chain.
- |inf| This function returns an infinite iterable
- |warning| Warning - pay attention
- |flux| This API is still in flux, and might be changed or
  removed in the future
- |cool| Noteworthy; could be especially useful in many situations.

The API is arranged roughly with the module-level functions first, and
thereafter the Iter_ class itself. It is the Iter_ class that does
the work to allow these iterators to be chained together. However, the
module-level functions are more likely to be used directly and that's
why they're presented first.

The API includes wrappers for the stdlib *itertools* module, including
the "recipes" given in the *itertools* docs, as well as wrappers for
the iterators from the more-itertools_ 3rd-party package.

Module-level Replacements for Builtins
######################################

The following module-level functions, like range_, zip_ and so on, are
intended to be used as replacements for their homonymous builtins. The
only difference between these and the builtin versions is that these
return instances of the Iter_ class. Note that because Iter_ is itself
iterable, it means that the functions here can be used as drop-in
replacements.

Once you have an Iter_ instance, all of its methods become available
via function call chaining, so these toplevel functions are really only
a convenience to "get started" using the chaining syntax with minimal
upfront cost in your own code.

.. contents::
    :local:

"""

from __future__ import annotations
import itertools
import functools
import operator
import inspect
import asyncio
from collections import UserDict, deque
import builtins
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
    Generator,
    IO,
    Sized,
)
from typing_extensions import Self
import collections.abc
import queue
import re
import fileinput as _fileinput

import more_itertools

__all__ = [
    "Iter",
    "AIter",
    "range",
    "zip",
    "enumerate",
    "map",
    "filter",
    "count",
    "cycle",
    "repeat",
    "accumulate",
    "chain",
    "chain_from_iterable",
    "compress",
    "dropwhile",
    "fileinput",
    "filterfalse",
    "groupby",
    "islice",
    "starmap",
    "takewhile",
    "tee",
    "zip_longest",
    "insert_separator",
    "concat",
    "from_queue",
    "IterDict",
    "finditer_regex",
    "splititer_regex",
]

T = TypeVar("T")
C = TypeVar("C")
K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")

DEFAULT_BYTE_CHUNK_SIZE = 8192


class class_or_instancemethod(classmethod):
    """From: https://stackoverflow.com/a/28238047/170656"""

    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


_marker = object()


# First save the builtins
_range = builtins.range
_zip = builtins.zip
_enumerate = builtins.enumerate
_map = builtins.map
_filter = builtins.filter


def _eval_mapper(expression: str) -> Callable[..., Any]:
    """Return a mapper for the provisional strings-as-code API.

    The single-argument case keeps the historical ``x`` local. Multiple
    iterables can use ``args`` to access the full tuple of mapped values.
    """

    def mapper(*values):
        namespace = {"args": values}
        if values:
            namespace["x"] = values[0]
        return eval(expression, globals(), namespace)

    return mapper


def range(*args) -> "Iter[int]":
    """
    |source|
    Replacement for the builtin ``range`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

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

    This example multiples, element by element, the series ``[0:5]`` with the
    series ``[1:6]``. Two things to note: Firstly, Iter.zip_ is used to emit
    the tuples from each series. Secondly, Iter.starmap_ is used to receive
    those tuples into separate arguments in the ``lambda``.

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


def zip(*iterables: Any, strict: bool = False) -> "Iter[Tuple[T, ...]]":
    """Replacement for the builtin ``zip`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining."""
    return Iter(_zip(*iterables, strict=strict))


def enumerate(iterable, start: int = 0) -> "Iter[Tuple[int, T]]":
    """Replacement for the builtin ``enumerate`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> import string
        >>> enumerate(string.ascii_lowercase).take(3).collect()
        [(0, 'a'), (1, 'b'), (2, 'c')]


    """
    return Iter(_enumerate(iterable, start=start))


def map(func: Union[Callable[..., C], str], *iterables: Iterable[Any]) -> "Iter[C]":
    """Replacement for the builtin ``map`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> result = map(lambda x: (x, ord(x)), 'caleb').dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

        >>> result = map('x, ord(x)', 'caleb').dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}
    """
    if isinstance(func, str):
        func = _eval_mapper(func)
    return Iter(_map(func, *iterables))


def filter(function: "Callable[[Any], bool]", iterable: "Iterable[T]") -> "Iter[T]":
    """Replacement for the builtin ``filter`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> filter(lambda x: x % 3 == 0, range(10)).collect()
        [0, 3, 6, 9]


    """
    return Iter(_filter(function, iterable))


def apply(function: "Callable[[Any], Iterable[T]]", value: Any) -> "Iter[T]":
    return Iter(function(value))


# standard library
# =================

# Infinite iterators


def count(start=0, step: int = 1) -> "Iter[int]":
    """
    |source|
    Replacement for the itertools ``count`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> count().take(5).collect()
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


def cycle(iterable) -> "Iter[T]":
    """Replacement for the itertools ``count`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> cycle(range(3)).take(6).collect()
        [0, 1, 2, 0, 1, 2]
        >>> cycle([]).take(6).collect()
        []
        >>> cycle(range(3)).take(0).collect()
        []

    """
    return Iter(itertools.cycle(iterable))


def repeat(object: C, times=None) -> "Iter[C]":
    """
    |source|
    Replacement for the itertools ``count`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

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
    if times is None:
        return Iter(itertools.repeat(object))
    else:
        return Iter(itertools.repeat(object, times=times))


"""

This next set of functions return iterators that terminate on the shortest 
input sequence.

"""


def accumulate(iterable, func=None, *, initial=None):
    """Replacement for the itertools ``accumulate`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

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
    return Iter(iterable).accumulate(func, initial=initial)


def chain(*iterables: Iterable[T]) -> "Iter[T]":
    """Replacement for the itertools ``chain`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> chain('ABC', 'DEF').collect()
        ['A', 'B', 'C', 'D', 'E', 'F']
        >>> chain().collect()
        []

    """
    return Iter(itertools.chain(*iterables))


def chain_from_iterable(iterable) -> "Iter[T]":
    """Replacement for the itertools ``chain.from_iterable`` method.
    This version returns an instance of Iter_ to allow
    further iterable chaining.

    .. code-block:: python

        >>> chain_from_iterable(['ABC', 'DEF']).collect()
        ['A', 'B', 'C', 'D', 'E', 'F']
        >>> chain_from_iterable([]).collect()
        []

    """
    return Iter(itertools.chain.from_iterable(iterable))


def compress(data, selectors):
    """Replacement for the itertools ``compress`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> compress('ABCDEF', [1, 0, 1, 0, 1, 1]).collect()
        ['A', 'C', 'E', 'F']


    """
    return Iter(itertools.compress(data, selectors))


def dropwhile(pred, iterable):
    """Replacement for the itertools ``dropwhile`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> dropwhile(lambda x: x < 4, range(6)).collect()
        [4, 5]

    """
    return Iter(itertools.dropwhile(pred, iterable))


def filterfalse(pred, iterable):
    """Replacement for the itertools ``filterfalse`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> filterfalse(None, [2, 0, 3, None, 4, 0]).collect()
        [0, None, 0]

    """
    return Iter(itertools.filterfalse(pred, iterable))


def groupby(iterable, key=None):
    """Replacement for the itertools ``groupby`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

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


def islice(iterable, *args) -> "Iter":
    """Replacement for the itertools ``islice`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

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
    """Replacement for the itertools ``starmap`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> starmap(pow, [(2, 5), (3, 2), (10, 3)]).collect()
        [32, 9, 1000]

    """
    return Iter(itertools.starmap(func, iterable))


def takewhile(pred, iterable):
    """Replacement for the itertools ``takewhile`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> takewhile(lambda x: x < 5, [1, 4, 6, 4, 1]).collect()
        [1, 4]

    """
    return Iter(itertools.takewhile(pred, iterable))


def tee(iterable, n=2):
    """Replacement for the itertools ``tee`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

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

    In the example above we passed in range_, but with excitertools_
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
    """Replacement for the itertools ``zip_longest`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

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


def finditer_regex(
    pat: "re.Pattern[AnyStr]", s: AnyStr, flags: Union[int, re.RegexFlag] = 0
) -> "Iter[AnyStr]":
    """
    Wrapper for ``re.finditer``. Returns an instance of Iter_ to allow
    chaining.

    .. code-block:: python

        >>> pat = r"\\w+"
        >>> text = "Well hello there! How ya doin!"
        >>> finditer_regex(pat, text).map(str.lower).filter(lambda w: 'o' in w).collect()
        ['hello', 'how', 'doin']
        >>> finditer_regex(r"[A-Za-z']+", "A programmer's RegEx test.").collect()
        ['A', "programmer's", 'RegEx', 'test']
        >>> finditer_regex(r"[A-Za-z']+", "").collect()
        []
        >>> finditer_regex("", "").collect()
        ['']
        >>> finditer_regex("", "").filter(None).collect()
        []

    """
    return Iter(m.group(0) for m in re.finditer(pat, s, flags))


def splititer_regex(
    pat: "re.Pattern[AnyStr]", s: AnyStr, flags: Union[int, re.RegexFlag] = 0
) -> "Iter[AnyStr]":
    """
    Lazy string splitting using regular expressions.

    Most of the time you want ``str.split``. Really! That will almost
    always be fastest. You might think that ``str.split`` is inefficient
    because it always has to build a list, but it can do this very, very
    quickly.

    The lazy splitting shown here is more about supporting a particular
    kind of programming model, rather than performance.

    See more discussion `here <https://stackoverflow.com/questions/3862010/is-there-a-generator-version-of-string-split-in-python>`_.

    .. code-block:: python

        >>> splititer_regex(r"\\s", "A programmer's RegEx test.").collect()
        ['A', "programmer's", 'RegEx', 'test.']

    Note that splitting at a single whitespace character will return blanks
    for each found. This is different to how ``str.split()`` works.

    .. code-block:: python

        >>> splititer_regex(r"\\s", "aaa     bbb  \\n  ccc\\nddd\\teee").collect()
        ['aaa', '', '', '', '', 'bbb', '', '', '', '', 'ccc', 'ddd', 'eee']

    To match ``str.split()``, specify a sequence of whitespace as the
    regex pattern.

    .. code-block:: python

        >>> splititer_regex(r"\\s+", "aaa     bbb  \\n  ccc\\nddd\\teee").collect()
        ['aaa', 'bbb', 'ccc', 'ddd', 'eee']

    Counting the whitespace

    .. code-block:: python

        >>> from collections import Counter
        >>> splititer_regex(r"\\s", "aaa     bbb  \\n  ccc\\nddd\\teee").collect(Counter)
        Counter({'': 8, 'aaa': 1, 'bbb': 1, 'ccc': 1, 'ddd': 1, 'eee': 1})

    Lazy splitting at newlines

    .. code-block:: python

        >>> splititer_regex(r"\\n", "aaa     bbb  \\n  ccc\\nddd\\teee").collect()
        ['aaa     bbb  ', '  ccc', 'ddd\\teee']

    A few more examples:

    .. code-block:: python

        >>> splititer_regex(r"", "aaa").collect()
        ['', 'a', 'a', 'a', '']
        >>> splititer_regex(r"", "").collect()
        ['', '']
        >>> splititer_regex(r"\\s", "").collect()
        ['']
        >>> splititer_regex(r"a", "").collect()
        ['']
        >>> splititer_regex(r"\\s", "aaa").collect()
        ['aaa']

    """

    def inner():
        prev = 0
        for m in re.finditer(pat, s, flags):
            yield s[prev : m.start()]
            prev = m.end()

        yield s[prev:]

    return Iter(inner())


def concat(iterable: Iterable[AnyStr], glue: AnyStr) -> "AnyStr":
    """Concatenate strings, bytes and bytearrays. It is careful to avoid the
    problem with single bytes becoming integers, and it looks at the value
    of `glue` to know whether to handle bytes or strings."""
    if isinstance(glue, (bytes, bytearray)):
        return glue.join(
            # TODO: this seems like a really inefficient way to do this.
            #  it might be better to use interleave/intersperse and just
            #  call bytes() on the result
            int.to_bytes(v, 1, "little") if isinstance(v, int) else v
            for v in iterable
        )

    elif isinstance(glue, str):
        return glue.join(iterable)

    else:
        """ 
        This function can raise ``ValueError`` if called with something
        other than ``bytes``, ``bytearray`` or ``str``."""
        raise ValueError("Must be called with bytes, bytearray or str")


def from_queue(q: queue.Queue, timeout=None, sentinel=None) -> "Iter":
    """
    |cool|
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

    This can be used in the same way you would normally use a queue, in
    that it will block while waiting for future input. This makes it
    convenient to run in a thread and wait for work. Below is a rough
    sketch of how one might cobble together a thread pool using this
    feature. Note the use of Iter.side_effect_ to call ``task_done()``
    on the queue.

    .. code-block:: python

        import queue
        from threading import Thread
        import logging
        from excitertools import from_queue

        logger = logging.getLogger(__name__)

        def process_job(job):
            result = ...
            return result

        def worker(inputs: Queue, results: Queue):
            (
                from_queue(inputs)
                .side_effect(lambda job: logger.info(f"Received job {job}")
                .map(process_job)
                .side_effect(lambda result: logger.info(f"Got result {job}")
                .into_queue(results)
                # Note that we only mark the task as done after the result
                # is added to the results queue.
                .side_effect(lambda _: inputs.task_done()
            )

        def create_job_pool(n: int) -> Tuple[Queue, Queue, Callable]:
            \"\"\"Returns two queues, and a pool shutdown method. The
            shutdown function can be called to shut down the pool and
            the ``inputs`` queue. Caller is responsible for draining
            the ``results`` queue.\"\"\"

            # Independent control of the sizes of the input and output
            # queues is interesting: it lets you decide how to bias
            # backpressure depending on the details of your workload.
            inputs, results = Queue(maxsize=100), Queue(maxsize=3)

            kwargs = dict(target=worker, args=(inputs, results), daemon=True)
            threads = repeat(Thread).map(lambda T: T(**kwargs)).take(n).collect()

            def shutdown():
                # Shut down each thread
                repeat(None).map(inputs.put).take(n).consume()
                inputs.join()
                Iter(threads).map(lambda t: t.join()).consume()

            return inputs, results, shutdown

    Now the two queues ``inputs`` and ``results`` can be used in various
    other threads to supply and consume data.

    """
    return Iter.from_queue(q, timeout=timeout, sentinel=sentinel)


def fileinput(
    files=None,
    inplace=False,
    backup="",
    mode="r",
    openhook=None,
    encoding=None,
    errors=None
) -> Iter:
    """
    |source|
    A wrapper around fileinput.input that returns an Excitertools Iter_ instance.

    The documentation for the stdlib fileinput module is here:
        https://docs.python.org/3/library/fileinput.html

    Here is an example of use:

    .. code-block:: python

        >>> from excitertools import fileinput
        >>> # Read from a file
        >>> fileinput(['data.txt']).take(3).collect()  # doctest: +SKIP
        ['1', '2', '3']
        >>> # Read from stdin OR files listed on the command line (sys.argv[1:])
        >>> fileinput().take(3).collect()  # doctest: +SKIP
        ['1', '2', '3']

    :param files: A list of filenames or '-' for stdin (default: sys.argv[1:]).
    :param inplace: Whether to allow in-place editing (default: False).
    :param backup: Backup extension for in-place editing (default: "").
    :param mode: File mode, e.g., 'r' or 'rb' (default: 'r').
    :param openhook: Optional hook to customize file opening.
    :param encoding: File encoding (default: None).
    :param errors: Error handling mode (default: None).
    """

    def yielder():
        stream = _fileinput.input(
            files=files,
            inplace=inplace,
            backup=backup,
            mode=mode,
            openhook=openhook,
            encoding=encoding,
            errors=errors,
        )
        with stream:
            yield from stream

    return Iter(yielder())


"""

The ``Iter`` Class
##################

.. contents::
    :backlinks: entry
    :local:

"""


class Iter(Generic[T], Iterator[T]):
    """
    |cool|
    This class is what allows chaining. Many of the methods in this class
    return an instance of Iter_, which allows further chaining. There
    are two exceptions to this: *sources* and *sinks*.

    A "source" is usually a ``classmethod`` which can be used as an
    initializer to produce data via an iterable. For example, the Iter.range_
    classmethod can be used to get a sequence of numbers:

    .. code-block:: python

        >>> Iter.range(1_000_000).take(3).collect()
        [0, 1, 2]

    Even though our range was a million elements, the iterator chaining
    took only 3 of those elements before collecting.

    A "sink" is a method that is usually the last component of a processing
    chain and often (but not always!) consumes the entire iterator. In the
    example above, the call to Iter.collect_ was a sink. Note that we still
    call it a sink even though it did not consume the entire iterator.

    We're using the term "source" to refer to a classmethod of Iter_ that
    produces data; but, the most typical source is going to be data that
    you provide. Iter_ can be called with anything that is iterable, including
    sequences, iterators, mappings, sets, generators and so on.

    Examples:

    .. code-block:: python

        List
        >>> Iter([1, 2, 3]).map(lambda x: x * 2).sum()
        12

        Generator
        >>> Iter((1, 2, 3)).map(lambda x: x * 2).sum()
        12
        >>> def g():
        ...     for i in [1, 2, 3]:
        ...         yield i
        >>> Iter(g()).map(lambda x: x * 2).sum()
        12

        Iterator
        >>> Iter(iter([1, 2, 3])).map(lambda x: x * 2).sum()
        12

        Dict
        >>> Iter(dict(a=1, b=2)).map(lambda x: x.upper()).collect()
        ['A', 'B']
        >>> d = dict(a=1, b=2, c=3)
        >>> Iter(d.items()).starmap(lambda k, v: v).map(lambda x: x * 2).sum()
        12

    A common error with generators is forgetting to actually evaluate, i.e.,
    call a generator function. If you do this there's a friendly error
    pointing out the mistake:

    .. code-block:: python

        >>> def mygen(): yield 123
        >>> Iter(mygen).collect()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: It seems you passed a generator function, but you ...
        >>> Iter(mygen()).collect()
        [123]

    Instance of Iter_ are resumable. Once an instance it created, it can
    be partially iterated in successive calls, like the following example
    shows:

    .. code-block:: python

        >>> it = Iter.range(1_000_000)
        >>> it.take(3).collect()
        [0, 1, 2]
        >>> it.take(4).collect()
        [3, 4, 5, 6]
        >>> # Consume most of the stream, collect the last few
        >>> it.consume(999_990).collect()
        [999997, 999998, 999999]

    This class implements the chaining. However, the module-level functions
    in excitertools_, such as range_, zip_ and so on, also return
    instances of Iter_, so they allow the chaining to continue. These are
    equivalent:

    .. code-block:: python

        >>> Iter.range(10).filter(lambda x: x > 7).collect()
        [8, 9]
        >>> range(10).filter(lambda x: x > 7).collect()
        [8, 9]

    It is intended that the module-level functions can act as drop-in
    replacements for the builtins they wrap:

    >>> import builtins
    >>> list(builtins.range(3))
    [0, 1, 2]
    >>> list(range(3))  # This is excitertools.range!
    [0, 1, 2]
    >>> list(Iter.range(3))
    [0, 1, 2]

    In your own code where you might like to use the excitertools_ version of
    range_ and the other functions, you can just import it and use it to access all the other
    cool stuff:

    .. code-block:: python

        # mymodule.py
        from excitertools import (
            range,
            map,
            filter,
            repeat,
            count,
            enumerate,
            zip,
            ...
        )

        def func(inputs):
            data = (
                map(lambda x: x + 2, inputs)
                    .enumerate()
                    .filter(lambda x: x[1] > 10)
                    ...
                    .collect()

            )

    Alternatively, if you don't want to hide the builtins you can do just
    fine with importing this class only, or even importing the module only:

    .. code-block:: python

        # mymodule.py - same example as before
        import excitertools

        def func(inputs):
            data = (
                excitertools.Iter(inputs)
                    .map(lambda x: x + 2)
                    .enumerate()
                    .filter(lambda x: x[1] > 10)
                    ...
                    .collect()
            )

            # Do something with data

    There are several valuable additions to the standard *itertools* and
    more-itertools_ functions. These usually involve sources and sinks,
    which are ways of getting data into an iterator pipeline, and then
    getting results out again. In the majority of documentation examples
    shown here, the Iter.collect_ method is used to collect all the
    remaining data on a stream into a list; but in practice this is not
    useful because large lists consume memory.

    In practice it is more useful to send iterator data to one of these
    common sinks:

    - files
    - sockets
    - queues
    - HTTP APIs
    - Cloud storage buckets
    - (Ideas for more to add here?)

    Iter_ has support for these use-cases, both for reading and for writing.

    """

    __slots__ = "x"
    x: Iterator[T]

    def __init__(self, x: Iterable[T]):
        if isinstance(x, collections.abc.Iterator):
            self.x = x
        else:
            try:
                self.x = iter(x)
            except TypeError:
                if inspect.isgeneratorfunction(x):
                    raise TypeError(
                        "It seems you passed a generator function, but you "
                        "probably intended to pass a generator. Remember to "
                        "evaluate the function to obtain a generator "
                        "instance:\n"
                        "\n"
                        "def mygen():\n"
                        "    yield 123\n"
                        "\n"
                        "Iter(mygen)    # ERROR - a generator function object "
                        "is not iterable\n"
                        "Iter(mygen())  # CORRECT - a generator instance is "
                        "iterable."
                    ) from None

                # Raise the original exception
                raise

    def __iter__(self) -> "Iterator[T]":
        return self.x

    def __next__(self) -> "T":
        return next(self.x)

    def next(self) -> "T":
        """Convenience function to avoid having to wrap an interator with
        the `next()` builtin, just to advance it by one step (and return
        the value). Typical use cases for this might be for tutorials
        and explainers, where you want to show the next value in a
        sequence.

        .. code-block:: python

            >>> it = Iter(range(5))
            >>> it.next()
            0
            >>> it.next()
            1
            >>> it.collect()
            [2, 3, 4]

        """
        return next(self.x)

    @classmethod
    def register(cls, *func):
        """
        Add a new method to Iter_. Sure, you could subclass Iter_ to get
        new chaining features, but it would be neat to let all existing
        Iter_ instance just immediately have the new registered function
        available.

        The new function must take ``iterable`` as the first parameter.

        .. code-block:: python

            >>> def up(iterable):
            ...     for v in iterable:
            ...         yield v.upper()
            >>> Iter.register(up)
            >>> Iter('abc').up().collect()
            ['A', 'B', 'C']
            >>> def poly(iterable, a, b, c):
            ...     # Polynomials a.x^2 + b.x + c
            ...     for x in iterable:
            ...         yield a*x**2 + b*x + c
            >>> Iter.register(poly)
            >>> Iter(range(-5, 5, 1)).poly(1, -5, 6).collect()
            [56, 42, 30, 20, 12, 6, 2, 0, 0, 2]

        Here's a math round-trip rollercoaster.

        .. code-block:: python

            >>> import math
            >>> def log(iterable):
            ...     for x in iterable:
            ...         yield math.log(x)
            >>> def exp(iterable):
            ...     for x in iterable:
            ...         yield math.exp(x)
            >>> def rnd(iterable):
            ...     for x in iterable:
            ...         yield round(x)
            >>> Iter.register(log, exp, rnd)
            >>> Iter(range(5)).exp().log().rnd().collect()
            [0, 1, 2, 3, 4]

        These are silly examples, but hopefully you get the idea.

        """
        def make_method(registered_func):
            def inner(self, *args, **kwargs):
                return type(self)(registered_func(self, *args, **kwargs))

            inner.__name__ = registered_func.__name__
            inner.__doc__ = registered_func.__doc__
            return inner

        for f in func:
            setattr(cls, f.__name__, make_method(f))

    def collect(self, container=list) -> "List[T]":
        """
        |sink|

        This is the most common way of "realizing" an interable chain
        into a concrete data structure. It should be the case that this
        is where most of the memory allocation occurs.

        The default container is a list and you'll see throughout this
        documentation that most examples produce lists. However,
        any container, and indeed any function, can be used as the sink.

        The basic example:

        .. code-block:: python

            >>> Iter(range(3)).collect()
            [0, 1, 2]
            >>> Iter(range(3)).collect(tuple)
            (0, 1, 2)

        You must pay attention to some things. For example, if your
        iterable is a string, the characters of the string are what
        get iterated over, and when you collect you'll get a collection
        of those atoms. You can however use ``str`` as your "container
        function" and that will give you back a string. It's like a join
        with blank joiner.

        .. code-block:: python

            >>> Iter('abc').collect()
            ['a', 'b', 'c']
            >>> Iter('abc').collect(str)
            'abc'

        With some types, things get a little more tricky. Take ``bytes``
        for example:

        .. code-block:: python

            >>> Iter(b'abc').collect()
            [97, 98, 99]

        You probably didn't expect to get the integers back right? Anyhow,
        you can use ``bytes`` as the "collection container", just like
        we did with strings and that will work:

        .. code-block:: python

            >>> Iter(b'abc').collect(bytes)
            b'abc'
            >>> Iter(b'abc').collect(bytearray)
            bytearray(b'abc')

        The other standard collections also work, here's a set for
        completeness.

        .. code-block:: python

            >>> Iter('abcaaaabbbbccc').collect(set) == {'a', 'b', 'c'}
            True

        String subclasses also work.

        .. code-block:: python

            >>> class MyString(str): pass
            >>> out = Iter(MyString('abc')).collect(MyString)
            >>> out
            'abc'
            >>> type(out).__name__
            'MyString'

        """
        isstr = isinstance(container, type) and container is str
        isstrsubclass = isinstance(container, type) and issubclass(container, str)
        if isstr or isstrsubclass:
            return container(self.concat(""))
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
    ) -> "Iter":
        """
        |cool|
        |source|

        Wrap the ``open()`` builtin precisely, but return an ``Iter``
        instance to allow function chaining on the result.

        I know you're thinking that we should always use a context
        manager for files. Don't worry, there is one being used
        internally. When the iterator chain is terminated the underlying
        file will be closed.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as td:
        ...     # Put some random text into a temporary file
        ...     with open(td + 'text.txt', 'w') as f:
        ...         f.writelines(['abc\\n', 'def\\n', 'ghi\\n'])
        ...
        ...     # Open the file, filter some lines, collect the result
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

    @classmethod
    def read_file(
        cls,
        file,
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ) -> Self:
        """
        |cool|
        |source|

        Read a text file lazily, one line at a time.

        This is the file-path counterpart to Iter.read_lines_. Unlike
        Iter.open_, it exposes only the text-reading options that matter for
        this source, so binary chunking and write modes do not obscure the
        common case.

        .. code-block:: python

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as td:
            ...     filename = td + 'text.txt'
            ...     with open(filename, 'w') as f:
            ...         _ = f.write('one\\ntwo\\n')
            ...
            ...     Iter.read_file(filename).map(str.strip).collect()
            ['one', 'two']

        """
        return cls.open(
            file=file,
            mode="r",
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    @classmethod
    def read_lines(cls, stream: IO[str], rewind=False):
        """
        |source|

        Read lines from a file-like object.

        First, let's put some data in a file. We'll be using that
        file in the examples that follow.

        .. code-block:: python

            >>> import tempfile
            >>> td = tempfile.TemporaryDirectory()
            ... # Put some random text into a temporary file
            >>> with open(td.name + 'text.txt', 'w') as f:
            ...     f.writelines(['abc\\n', 'def\\n', 'ghi\\n'])
            ...

        Use read_lines to process the file data

        .. code-block:: python

            >>> with open(td.name + 'text.txt') as f:
            ...     Iter.read_lines(f).filter(lambda line: 'def' in line).collect()
            ['def\\n']

        Reading starts at the stream's current position. The ``rewind``
        parameter can be used to seek back to the beginning first, if the
        stream is seekable.

        .. code-block:: python

            >>> with open(td.name + 'text.txt') as f:
            ...     part1 = Iter.read_lines(f).take(1).collect()
            ...     part2 = Iter.read_lines(f, rewind=False).collect()
            >>> part1
            ['abc\\n']
            >>> part2
            ['def\\n', 'ghi\\n']
            >>> td.cleanup()

        """
        if rewind:
            stream.seek(0)
        return cls(stream)

    @classmethod
    def read_bytes(
        cls,
        stream: IO[bytes],
        size: Union[Callable[[], int], int] = DEFAULT_BYTE_CHUNK_SIZE,
        rewind=False,
    ):
        """
        |source|

        The ``size`` parameter controls how many bytes are read for each
        advancement of the iterator chain. By default, bytes are read in
        chunks of ``DEFAULT_BYTE_CHUNK_SIZE``. Here we set ``size=1``, which
        means we'll get back one byte at a time.

        .. code-block:: python

            >>> import tempfile
            >>> td = tempfile.TemporaryDirectory()
            >>> filename = td.name + 'bytes.bin'

        Put some random text into a temporary file:

        .. code-block:: python

            >>> with open(filename, 'wb') as f:
            ...     x = f.write(b'\\x00' * 100)
            ...
            >>> with open(filename, 'rb') as f:
            ...     data = Iter.read_bytes(f, size=1).collect()
            ...     len(data)
            100
            >>> with open(filename, 'rb') as f:
            ...     data = Iter.read_bytes(f).collect()
            ...     len(data)
            1

        A little more ambitious. Because ``size`` is a callable, we can use
        a ``deque`` and a ``side_effect`` to pass information back into
        the reader to control how many bytes are read in each chunk.

        In this example we're reading 1 byte at a time. In a real example
        you might have a sequence of headers and bodies, where headers
        give size information about how many bytes are in the body
        corresponding to that header. Then you can precisely read
        each body in sequence.

        .. code-block:: python

            >>> from collections import deque
            >>> read_sizes = deque([1])
            >>> with open(filename, 'rb') as f:
            ...     data = (
            ...         Iter
            ...             .read_bytes(f, size=lambda: read_sizes.popleft())
            ...             .side_effect(lambda bytes: read_sizes.append(1))
            ...             .collect()
            ...     )
            ...     len(data)
            100

        Reading starts at the stream's current position. The ``rewind``
        parameter can be used to seek back to the beginning first, if the
        stream is seekable.

        .. code-block:: python

            >>> with open(filename, 'rb') as f:
            ...     part1 = Iter.read_bytes(f, size=10).take(1).collect()
            ...     part2 = Iter.read_bytes(f, rewind=False).collect()
            >>> part1
            [b'\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00']
            >>> len(part2[0])
            90
            >>> td.cleanup()

        """
        if rewind:
            stream.seek(0)

        def inner():
            n = size if callable(size) else lambda: size
            data = stream.read(n())
            while data:
                yield data
                data = stream.read(n())

        return cls(inner())

    @classmethod
    def read_file_bytes(
        cls,
        file,
        size: Union[Callable[[], int], int] = DEFAULT_BYTE_CHUNK_SIZE,
        buffering=-1,
        closefd=True,
        opener=None,
    ) -> Self:
        """
        |cool|
        |source|

        Read a binary file lazily in byte chunks.

        This is the file-path counterpart to Iter.read_bytes_. The default
        chunk size is ``DEFAULT_BYTE_CHUNK_SIZE``; pass ``size=-1`` to read
        the whole file as a single chunk.

        .. code-block:: python

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as td:
            ...     filename = td + 'bytes.bin'
            ...     with open(filename, 'wb') as f:
            ...         _ = f.write(b'abcdef')
            ...
            ...     Iter.read_file_bytes(filename, size=2).collect()
            [b'ab', b'cd', b'ef']

        """

        def inner():
            with open(
                file=file,
                mode="rb",
                buffering=buffering,
                closefd=closefd,
                opener=opener,
            ) as f:
                yield from cls.read_bytes(f, size=size, rewind=False)

        return cls(inner())

    def write_text_to_stream(self, stream: IO[str], insert_newlines=True, flush=True):
        """
        |sink|

        .. code-block:: python

            >>> import tempfile
            >>> td = tempfile.TemporaryDirectory()
            >>> filename = td.name + 'text.txt'

            >>> data = ['a', 'b', 'c']
            >>> with open(filename, 'w') as f:
            ...     Iter(data).map(str.upper).write_text_to_stream(f)
            ...     with open(filename) as f2:
            ...         Iter.read_lines(f2).concat()
            'A\\nB\\nC'

        If some prior step adds newlines, or more commonly, newlines
        originate with a data source and are simply carried through the
        processing chain unaltered, disable the insertion of newlines:

        .. code-block:: python

            >>> with open(filename, 'w') as f:
            ...     Iter(data).map(str.upper).write_text_to_stream(f, insert_newlines=False)
            ...     with open(filename) as f2:
            ...         Iter.read_lines(f2).concat()
            'ABC'

        Multiple successive writes may be slowed down by the default
        ``flush=True`` parameter. In this case you can delay flushing until
        everything has been written.

        .. code-block:: python

            >>> with open(filename, 'w') as f:
            ...     Iter(data).map(str.upper).write_text_to_stream(f, flush=False)
            ...     Iter(data).map(str.upper).write_text_to_stream(f, flush=False)
            ...     Iter(data).map(str.upper).write_text_to_stream(f, flush=True)
            ...     with open(filename) as f2:
            ...         Iter.read_lines(f2).concat()
            'A\\nB\\nCA\\nB\\nCA\\nB\\nC'
            >>> td.cleanup()

        """
        if insert_newlines:
            stream.writelines(self.intersperse("\n"))
        else:
            stream.writelines(self)

        if flush:
            stream.flush()

    def write_bytes_to_stream(self, stream: IO[bytes], flush=True):
        """
        |sink|

        .. code-block:: python

            >>> import tempfile
            >>> td = tempfile.TemporaryDirectory()
            >>> filename = td.name + 'bytes.bin'
            >>> data = [b'a', b'b', b'c']
            >>> with open(filename, 'wb') as f:
            ...     Iter(data).map(lambda x: x * 2 ).write_bytes_to_stream(f)
            ...     with open(filename, 'rb') as f2:
            ...         Iter.read_bytes(f2).collect()
            [b'aabbcc']
            >>> with open(filename, 'wb') as f:
            ...     Iter(data).map(lambda x: x * 2 ).write_bytes_to_stream(f)
            ...     with open(filename, 'rb') as f2:
            ...         Iter.read_bytes(f2).concat(b'')
            b'aabbcc'
            >>> with open(filename, 'wb') as f:
            ...     Iter(data).map(lambda x: x * 2 ).write_bytes_to_stream(f)
            ...     with open(filename, 'rb') as f2:
            ...         Iter.read_bytes(f2, size=1).collect()
            [b'a', b'a', b'b', b'b', b'c', b'c']
            >>> with open(filename, 'wb') as f:
            ...     Iter(data).map(lambda x: x * 2 ).write_bytes_to_stream(f)
            ...     with open(filename, 'rb') as f2:
            ...         Iter.read_bytes(f2, size=2).map(bytes.decode).collect()
            ['aa', 'bb', 'cc']

        Flushing can be delayed if multiple parts are to be written.

        .. code-block:: python

            >>> with open(filename, 'wb') as f:
            ...     it = Iter(data)
            ...     it.map(lambda x: x * 2 ).take(2).write_bytes_to_stream(f, flush=False)
            ...     it.map(lambda x: x * 2 ).write_bytes_to_stream(f, flush=True)
            ...     with open(filename, 'rb') as f2:
            ...         Iter.read_bytes(f2, size=2).map(bytes.decode).collect()
            ['aa', 'bb', 'cc']
            >>> td.cleanup()

        """
        stream.writelines(self)
        if flush:
            stream.flush()

    def write_file(
        self,
        file,
        mode="w",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        """
        |cool|
        |sink|

        Write items from the chain to a file path.

        Text and binary output use the same method; choose an appropriate
        file ``mode`` for the item type being written.

        .. code-block:: python

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as td:
            ...     # Put some random text into a temporary file
            ...     with open(td + 'text.txt', 'w') as f:
            ...         f.writelines(['abc\\n', 'def\\n', 'ghi\\n'])
            ...
            ...     # Open the file, transform, write out to new file.
            ...     Iter.read_file(td + 'text.txt').map(str.upper).write_file(td + 'test2.txt')
            ...     # Read the new file, for the test
            ...     Iter.read_file(td + 'test2.txt').collect()
            ['ABC\\n', 'DEF\\n', 'GHI\\n']
            >>> with tempfile.TemporaryDirectory() as td:
            ...     Iter([b'a', b'b']).write_file(td + 'bytes.bin', mode='wb')
            ...     Iter.read_file_bytes(td + 'bytes.bin', size=-1).collect()
            [b'ab']

        """
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
            for piece in self:
                f.write(piece)

    # Database operations

    def executemany(
        self,
        cursor: Any,
        sql: str,
        *,
        batch_size: int = 1000,
        commit: Optional[Callable[[], Any]] = None,
        rollback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        |sink|

        Execute a DB-API 2.0-style ``executemany`` call over batches from
        this iterator.

        The SQL string is supplied by the caller so driver-specific
        placeholder styles stay outside this method. Each item in the
        iterator can be any parameter object accepted by the driver, such as
        a tuple/list for positional placeholders or a dict for named
        placeholders.

        .. code-block:: python

            >>> import sqlite3
            >>> conn = sqlite3.connect(":memory:")
            >>> cursor = conn.cursor()
            >>> _ = cursor.execute("CREATE TABLE customers (id INTEGER, email TEXT, active INTEGER)")
            >>> (
            ...     Iter([(1, "A@EXAMPLE.COM", True), (2, "B@EXAMPLE.COM", False)])
            ...         .map(lambda row: (row[0], row[1].lower(), row[2]))
            ...         .executemany(
            ...             cursor,
            ...             "INSERT INTO customers VALUES (?, ?, ?)",
            ...             batch_size=1,
            ...             commit=conn.commit,
            ...         )
            ... )
            >>> cursor.execute("SELECT id, email, active FROM customers ORDER BY id").fetchall()
            [(1, 'a@example.com', 1), (2, 'b@example.com', 0)]
            >>> conn.close()

        By default this method does not commit or roll back; callers can use
        their driver's transaction or context-manager conventions. If
        ``commit`` is supplied, it is called after all batches finish
        successfully. If ``rollback`` is supplied, it is called before
        re-raising an exception from batching, execution, or commit.

        Here's a somewhat realistic ETL example:

        .. code-block:: python

            from excitertools import Iter
            import psycopg

            with psycopg.connect(src_dsn) as src:
                with psycopg.connect(dst_dsn) as dst:
                    src_cur = src.cursor()
                    dst_cur = dst.cursor()

                    sql_read = ''' SELECT id, email, signup_date FROM customers ORDER BY id '''
                    sql_write = '''
                        INSERT INTO customers (id, email, signup_date)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                    '''

                    (
                        Iter(src_cur.execute(sql_read))
                            # Only records with email
                            .filter(lambda r: r.email is not None)
                            # Lowercase the email address
                            .starmap(
                                lambda id, email, signup_date:
                                (id, email.lower(), signup_date)
                            )
                            # Batch insert into the destination database
                            .executemany(
                                dst_cur,
                                '''
                                INSERT INTO customers (id, email, signup_date)
                                VALUES (%s, %s, %s)
                                ON CONFLICT DO NOTHING
                                ''',
                                batch_size=1000,
                                commit=dst.commit,
                            )
                    )

        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        execute_batch = cursor.executemany
        try:
            for batch in self.chunked(batch_size):
                execute_batch(sql, batch)

            if commit is not None:
                commit()
        except Exception:
            if rollback is not None:
                rollback()
            raise

    # Standard utilities

    @classmethod
    def range(cls, *args) -> "Iter[int]":
        """
        |source|

        The ``range`` function you all know and love.

        .. code-block:: python

            >>> Iter.range(3).collect()
            [0, 1, 2]
            >>> Iter.range(0).collect()
            []

        """
        return cls(range(*args))

    def zip(self, *iterables: Any, strict: bool = False) -> "Iter[Tuple[T, ...]]":
        """

        The ``zip`` function you all know and love. The only thing to
        note here is that the first iterable is really what the Iter_
        instance is wrapping. The Iter.zip_ invocation brings in the
        other iterables.

        Make an Iter_ instance, then call ``zip`` on that.

        .. code-block:: python

            >>> Iter('caleb').zip(range(10)).collect()
            [('c', 0), ('a', 1), ('l', 2), ('e', 3), ('b', 4)]

        Use a classmethod to get an infinite stream using Iter.count_
        and zip against that with more finite iterators.

        .. code-block:: python

            >>> Iter.count().zip(range(5), range(3, 100, 2)).collect()
            [(0, 0, 3), (1, 1, 5), (2, 2, 7), (3, 3, 9), (4, 4, 11)]

        It takes a few minutes to get used to that but feels comfortable
        pretty quickly.

        Iter.take_ can be used to stop infinite zip sequences:

        .. code-block:: python

            >>> Iter('caleb').cycle().enumerate().take(8).collect()
            [(0, 'c'), (1, 'a'), (2, 'l'), (3, 'e'), (4, 'b'), (5, 'c'), (6, 'a'), (7, 'l')]

        While we're here (assuming you worked through the previous
        example), note the difference if you switch the order of the
        Iter.cycle_ and Iter.enumerate_ calls:

        .. code-block:: python

            >>> Iter('caleb').enumerate().cycle().take(8).collect()
            [(0, 'c'), (1, 'a'), (2, 'l'), (3, 'e'), (4, 'b'), (0, 'c'), (1, 'a'), (2, 'l')]

        If you understand how this works, everything else in _excitertools_
        will be intuitive to use.

        """
        return type(self)(_zip(self, *iterables, strict=strict))

    def any(self) -> "bool":
        """
        |sink|

        .. code-block:: python

            >>> Iter([0, 0, 0]).any()
            False
            >>> Iter([0, 0, 1]).any()
            True
            >>> Iter([]).any()
            False

        """
        return any(self)

    def all(self) -> "bool":
        """
        |sink|


        .. code-block:: python

            >>> Iter([0, 0, 0]).all()
            False
            >>> Iter([0, 0, 1]).all()
            False
            >>> Iter([1, 1, 1]).all()
            True

        Now pay attention:

        .. code-block:: python

            >>> Iter([]).all()
            True

        This behaviour has some controversy around it, but that's how the
        ``all()`` builtin works so that's what we do too. The way to
        think about what ``all()`` does is this: it returns False if there
        is at least one element that is falsy.  Thus, if there are no elements
        it follows that there are no elements that are falsy and that's why
        ``all([]) == True``.

        """
        return all(self)

    def enumerate(self, start: int = 0) -> "Iter[Tuple[int, T]]":
        """

        .. code-block:: python

            >>> Iter('abc').enumerate().collect()
            [(0, 'a'), (1, 'b'), (2, 'c')]
            >>> Iter([]).enumerate().collect()
            []

        """
        return type(self)(_enumerate(self, start=start))

    def dict(self) -> "Dict":
        """
        In regular Python a dict can be constructed through an iterable
        of tuples:

        .. code-block:: python

            >>> dict([('a', 0), ('b', 1)])  # doctest: +SKIP
            {'a': 0, 'b': 1}

        In *excitertools* we prefer chaining so this method is a shortcut
        for that:

        .. code-block:: python

            >>> d = Iter('abc').zip(count()).dict()
            >>> assert d == {'a': 0, 'b': 1, 'c': 2}

        """
        return dict(self)

    ###

    def map(self, func: Union[Callable[..., C], str], *iterables: Iterable[Any]) -> "Iter[C]":
        """
        The ``map`` function you all know and love.

        .. code-block:: python

            >>> Iter('abc').map(str.upper).collect()
            ['A', 'B', 'C']
            >>> Iter(['abc', 'def']).map(str.upper).collect()
            ['ABC', 'DEF']

        Using lambdas might seem convenient but in practice it turns
        out that they make code difficult to read:

        .. code-block:: python

            >>> result = Iter('caleb').map(lambda x: (x, ord(x))).dict()
            >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

        It's recommended that you make a separate function instead:

        .. code-block:: python

            >>> def f(x):
            ...     return x, ord(x)
            >>> result = Iter('caleb').map(f).dict()
            >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

        I know many people prefer anonymous functions (often on
        philosphical grounds) but in practice it's just easier to make
        a separate, named function.

        I've experimented with passing a string into the map, and using
        ``eval()`` to make a lambda internally. This simplifies the code
        very slightly, at the cost of using strings-as-code. I'm pretty
        sure this feature will be removed so don't use it.

        .. code-block:: python

            >>> result = Iter('caleb').map('x, ord(x)').dict()
            >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

        """
        if isinstance(func, str):
            func = _eval_mapper(func)
        return type(self)(_map(func, self, *iterables))

    def amap(self, func: Callable[[T], Any], *, concurrency: int = 1, ordered: bool = True) -> "AIter":
        """
        |flux|

        The asynchronous counterpart to map_. This is the *boundary*
        method: it is the point at which a synchronous chain becomes an
        asynchronous one. It always returns an AIter_, so every method
        after it in the chain is asynchronous too, all the way to an
        async sink such as acollect_.

        ``func`` may be either a coroutine function (``async def``) or an
        ordinary function; each result is awaited if it is awaitable. Up
        to ``concurrency`` calls are kept in flight at once. Results are
        yielded in input order by default; pass ``ordered=False`` to
        yield them as they complete.

        Note that ``amap`` itself is *not* a coroutine and is not awaited.
        Like every other combinator it merely builds a lazy pipeline;
        nothing runs until the pipeline is driven by an async sink.

        .. code-block:: python

            >>> import asyncio
            >>> async def double(x):
            ...     return x * 2
            >>> async def main():
            ...     return await Iter(range(5)).amap(double).acollect()
            >>> asyncio.run(main())
            [0, 2, 4, 6, 8]

        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        def call(item):
            return _maybe_await(func, item)

        return AIter(_abounded(_as_async_iter(self), call, concurrency, ordered))

    def astarmap(self, func: Callable[..., Any], *, concurrency: int = 1, ordered: bool = True) -> "AIter":
        """
        |flux|

        The concurrent, asynchronous counterpart to starmap_. Each item is
        an argument tuple, unpacked into ``func(*item)``; ``func`` may be a
        coroutine function or an ordinary one. Like amap_ this is a
        boundary method that returns an AIter_. See amap_ for the meaning
        of ``concurrency`` and ``ordered``.

        .. code-block:: python

            >>> import asyncio
            >>> async def add(a, b):
            ...     return a + b
            >>> async def main():
            ...     return await Iter([(1, 2), (3, 4)]).astarmap(add).acollect()
            >>> asyncio.run(main())
            [3, 7]

        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        async def call(item):
            return await _awaited(func(*item))

        return AIter(_abounded(_as_async_iter(self), call, concurrency, ordered))

    def afilter(self, pred: Callable[[T], Any], *, concurrency: int = 1, ordered: bool = True) -> "AIter":
        """
        |flux|

        The concurrent, asynchronous counterpart to filter_, for when the
        *predicate* itself is asynchronous (or merely benefits from
        concurrent evaluation). ``pred`` may be a coroutine function or an
        ordinary one; up to ``concurrency`` predicate evaluations run at
        once. A boundary method: it returns an AIter_.

        Note that ``ordered`` controls the order of the surviving items;
        predicate evaluation is always concurrent regardless.

        .. code-block:: python

            >>> import asyncio
            >>> async def is_even(x):
            ...     return x % 2 == 0
            >>> async def main():
            ...     return await Iter(range(6)).afilter(is_even).acollect()
            >>> asyncio.run(main())
            [0, 2, 4]

        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        async def call(item):
            return item, await _awaited(pred(item))

        async def agen():
            engine = _abounded(_as_async_iter(self), call, concurrency, ordered)
            async for item, keep in engine:
                if keep:
                    yield item

        return AIter(agen())

    def to_async(self) -> "AIter":
        """
        |flux|

        Explicitly lift a synchronous chain into the asynchronous world
        without introducing any async callable. Useful when the source is
        synchronous but you want to attach an async sink, or use AIter_
        combinators, further down the chain.

        .. code-block:: python

            >>> import asyncio
            >>> async def main():
            ...     return await Iter(range(3)).to_async().acollect()
            >>> asyncio.run(main())
            [0, 1, 2]

        """
        return AIter(self)

    def filter(self, function: "Optional[Callable[[T], bool]]" = None) -> "Iter[T]":
        """
        The ``map`` function you all know and love.

        .. code-block:: python

            >>> Iter('caleb').filter(lambda x: x in 'aeiou').collect()
            ['a', 'e']

        There is a slight difference between this method signature and
        the builtin ``filter``:  how the identity function is handled.
        This is a consquence of chaining. In the function signature above
        it is possible for us to give the ``function`` parameter a
        default value of ``None`` because the parameter appears towards
        the end of the parameter list. Last, in fact.  In the
        `builtin filter signature <https://docs.python.org/3/library/functions.html#filter>`_
        it doesn't allow for this because the predicate parameter appears
        first.

        This is a long way of saying: if you just want to filter out
        falsy values, no parameter is needed:

        .. code-block:: python

            >>> Iter([0, 1, 0, 0, 0, 1, 1, 1, 0, 0]).filter().collect()
            [1, 1, 1, 1]

        Using the builtin, you'd have to do ``filter(None, iterable)``.

        You'll find that Iter.map_ and Iter.filter_
        (and Iter.reduce_, up next) work together very nicely:

        .. code-block:: python

            >>> def not_eve(x):
            ...    return x != 'eve'
            >>> Iter(['bob', 'eve', 'alice']).filter(not_eve).map(str.upper).collect()
            ['BOB', 'ALICE']

        The long chains get unwieldy so let's rewrite that:

        .. code-block:: python

            >>> (
            ...     Iter(['bob', 'eve', 'alice'])
            ...         .filter(not_eve)
            ...         .map(str.upper)
            ...         .collect()
            ... )
            ['BOB', 'ALICE']

        """
        return Iter(_filter(function, self))

    def starfilter(
        self, function: "Optional[Callable[[T, ...], bool]]" = None
    ) -> "Iter[T]":
        """
        |cool|
        Like Iter.filter_, but arg unpacking in lambdas will work.

        With the normal ``filter``, this fails:

        .. code-block:: python

            >>> Iter('caleb').enumerate().filter(lambda i, x: i > 2).collect()
            Traceback (most recent call last):
                ...
            TypeError: <lambda>() missing 1 required positional argument: 'x'

        This is a real buzzkill. ``starfilter`` is very similar to
        ``starmap`` in that tuples are unpacked when calling the function:

        .. code-block:: python

            >>> Iter('caleb').enumerate().starfilter(lambda i, x: i > 2).collect()
            [(3, 'e'), (4, 'b')]

        """
        return Iter(v for v in self if function(*v))

    def filter_gt(self, value) -> "Iter[T]":
        """
        Convenience method

        .. code-block:: python

            >>> Iter([1,2,3]).filter_gt(1).collect()
            [2, 3]

        """
        return self.filter(lambda x: x > value)

    def filter_ge(self, value) -> "Iter[T]":
        """
        Convenience method

        .. code-block:: python

            >>> Iter([1,2,3]).filter_ge(2).collect()
            [2, 3]

        """
        return self.filter(lambda x: x >= value)

    def filter_lt(self, value) -> "Iter[T]":
        """
        Convenience method

        .. code-block:: python

            >>> Iter([1,2,3]).filter_lt(3).collect()
            [1, 2]
        """
        return self.filter(lambda x: x < value)

    def filter_le(self, value) -> "Iter[T]":
        """
        Convenience method

        .. code-block:: python

            >>> Iter([1,2,3]).filter_le(2).collect()
            [1, 2]
        """
        return self.filter(lambda x: x <= value)

    def filter_eq(self, value) -> "Iter[T]":
        """
        Convenience method

        .. code-block:: python

            >>> Iter([1,2,3]).filter_eq(2).collect()
            [2]
        """
        return self.filter(lambda x: x == value)

    def filter_ne(self, value) -> "Iter[T]":
        """
        Convenience method

        .. code-block:: python

            >>> Iter([1,2,3]).filter_ne(2).collect()
            [1, 3]
        """
        return self.filter(lambda x: x != value)

    def filter_in(self, value: Sized) -> "Iter[T]":
        """
        Convenience method for membership testing. Note that the value
        parameter must be at least ``Sized`` because it gets reused
        over and over for each pass of the iterator chain. For example,
        passing in things like ``range()`` will not work properly because
        it will become progressively exhausted.

        .. code-block:: python

            >>> Iter([1,2,3]).filter_in([2, 3, 4, 5]).collect()
            [2, 3]
            >>> Iter([1,2,3]).filter_in(range(2, 8).collect()).collect()
            [2, 3]
            >>> Iter([1,2,3]).filter_in({2, 3, 4, 5}).collect()
            [2, 3]
            >>> Iter([1,2,3]).filter_in(dict.fromkeys({2, 3, 4, 5})).collect()
            [2, 3]
        """
        return self.filter(lambda x: x in value)

    def filter_ni(self, value) -> "Iter[T]":
        """
        Convenience method for membership testing. Note that the value
        parameter must be at least ``Sized`` because it gets reused
        over and over for each pass of the iterator chain. For example,
        passing in things like ``range()`` will not work properly because
        it will become progressively exhausted.

        .. code-block:: python

            >>> Iter([1,2,3]).filter_ni([2, 3, 4, 5]).collect()
            [1]
            >>> Iter([1,2,3]).filter_ni(range(2, 8).collect()).collect()
            [1]
            >>> Iter([1,2,3]).filter_ni({2, 3, 4, 5}).collect()
            [1]
            >>> Iter([1,2,3]).filter_ni(dict.fromkeys({2, 3, 4, 5})).collect()
            [1]
        """
        return self.filter(lambda x: x not in value)

    def reduce(self, func: Callable[..., T], *args) -> "T":
        """
        |sink|
        The ``reduce`` function you all know and...hang on, actually
        ``reduce`` is rather unloved. In the past I've found it very complex
        to reason about, when looking at a bunch of nested function calls
        in typical ``itertools`` code. Hopefully iterable chaining makes
        it easier to read code that uses ``reduce``?

        Let's check, does this make sense?

        .. code-block:: python

            >>> payments = [
            ...     ('bob', 100),
            ...     ('alice', 50),
            ...     ('eve', -100),
            ...     ('bob', 19.95),
            ...     ('bob', -5.50),
            ...     ('eve', 11.95),
            ...     ('eve', 200),
            ...     ('alice', -45),
            ...     ('alice', -67),
            ...     ('bob', 1.99),
            ...     ('alice', 89),
            ... ]
            >>> (
            ...     Iter(payments)
            ...         .filter(lambda entry: entry[0] == 'bob')
            ...         .map(lambda entry: entry[1])
            ...         .reduce(lambda total, value: total + value, 0)
            ... )
            116.44

        I intentionally omitted comments above so that you can try the
        "readability experiment", but in practice you would definitely
        want to add some comments on these chains:

        .. code-block:: python

            >>> (
            ...     # Iterate over all payments
            ...     Iter(payments)
            ...         # Only look at bob's payments
            ...         .filter(lambda entry: entry[0] == 'bob')
            ...         # Extract the value of the payment
            ...         .map(lambda entry: entry[1])
            ...         # Add all those payments together
            ...         .reduce(lambda total, value: total + value, 0)
            ... )
            116.44

        ``reduce`` is a quite crude low-level tool. In many cases you'll
        find that there are other functions and methods better suited
        to the situations you'll encounter most often. For example,
        there is already Iter.sum_ if you just want to add up numbers,
        and it's much easier to use Iter.groupby_ for grouping than
        to try to make that work with Iter.reduce_. You *can* make it
        work but it'll be easier to use Iter.groupby_.

        """
        return functools.reduce(func, self, *args)

    def starreduce(self, function: Callable[..., T], initializer=0) -> "T":
        """
        |sink|
        Iter.starreduce_ is the same as Iter.reduce_ except that args are
        star-unpacked when passed into ``function``. This is frequently
        more convenient than the default behaviour.

        We can see this using the same example shown for Iter.reduce_.
        The star unpacking makes it easier to just do the filtering
        directly inside the reducer function.

        .. code-block:: python

            >>> payments = [
            ...     ('bob', 100),
            ...     ('alice', 50),
            ...     ('eve', -100),
            ...     ('bob', 19.95),
            ...     ('bob', -5.50),
            ...     ('eve', 11.95),
            ...     ('eve', 200),
            ...     ('alice', -45),
            ...     ('alice', -67),
            ...     ('bob', 1.99),
            ...     ('alice', 89),
            ... ]
            >>> (
            ...     Iter(payments)
            ...         .starreduce(
            ...             lambda tot, name, value: tot + value if name == 'bob' else tot,
            ...             0
            ...         )
            ... )
            116.44

        This is how that looks if you avoid a lambda:

        .. code-block:: python

            >>> def f(tot, name, value):
            ...     if name == 'bob':
            ...         return tot + value
            ...     else:
            ...         return tot
            >>> Iter(payments).starreduce(f)
            116.44

        """
        for v in self:
            initializer = function(initializer, *v)

        return initializer

    def sum(self):
        """
        |sink|
        Exactly what you expect:

        .. code-block:: python

            >>> Iter(range(10)).sum()
            45

        """
        return sum(self.x)

    def concat(self, glue: AnyStr = "") -> "AnyStr":
        """
        |sink|

        Joining strings (and bytes).

        .. code-block:: python

            >>> Iter(['hello', 'there']).concat()
            'hellothere'
            >>> Iter(['hello', 'there']).concat(' ')
            'hello there'
            >>> Iter(['hello', 'there']).concat(',')
            'hello,there'
            >>> Iter([b'hello', b'there']).concat(b',')
            b'hello,there'

        """
        return concat(self.x, glue)

    def insert(self, glue: C) -> "Iter[Union[C, T]]":
        """

        .. code-block:: python

            >>> Iter('abc').insert('x').collect()
            ['a', 'x', 'b', 'x', 'c']
            >>> Iter('abc').insert('x').concat('')
            'axbxc'
            >>> Iter([]).insert('x').collect()
            []

        """
        cls = type(self)
        return cls(insert_separator(self, glue))

    # standard library
    # =================

    # Infinite iterators

    @classmethod
    def count(cls, *args) -> "Iter[int]":
        """
        |source|

        .. code-block:: python

            >>> Iter.count().take(3).collect()
            [0, 1, 2]
            >>> Iter.count(100).take(3).collect()
            [100, 101, 102]
            >>> Iter.count(100, 2).take(3).collect()
            [100, 102, 104]

        """
        return cls(itertools.count(*args))

    def cycle(self) -> Self:
        """
        |inf|

        .. code-block:: python

            >>> Iter('abc').cycle().take(8).collect()
            ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']
            >>> Iter('abc').cycle().take(8).concat('')
            'abcabcab'

        """
        return type(self)(itertools.cycle(self.x))

    @classmethod
    def repeat(cls, elem: C, times=None) -> "Iter[C]":
        """
        |source|
        |inf|

        .. code-block:: python

            >>> Iter.repeat('c', times=3).collect()
            ['c', 'c', 'c']

        """
        # TODO: does it really work like this? Wow.
        if times is None:
            return cls(itertools.repeat(elem))
        else:
            return cls(itertools.repeat(elem, times=times))

    # Iterators terminating on the shortest input sequence
    def accumulate(self, func=None, *, initial=None):
        """ Reference `itertools.accumulate <https://docs.python.org/3/library/itertools.html#itertools.accumulate>`_

        .. code-block:: python

            >>> Iter([1, 2, 3, 4, 5]).accumulate().collect()
            [1, 3, 6, 10, 15]
            >>> Iter([1, 2, 3, 4, 5]).accumulate(initial=100).collect()
            [100, 101, 103, 106, 110, 115]
            >>> Iter([1, 2, 3, 4, 5]).accumulate(operator.mul).collect()
            [1, 2, 6, 24, 120]

            Example from the itertools docs:
            Amortize a 5% loan of 1000 with 10 annual payments of 90
            >>> update = lambda balance, payment: round(balance * 1.05) - payment

            This is written in the itertools docs:
            >>> list(accumulate(repeat(90, 10), update, initial=1_000)) # doctest: +SKIP

            This is using excitertools:
            >>> repeat(90, 10).accumulate(update, initial=1000).collect()
            [1000, 960, 918, 874, 828, 779, 728, 674, 618, 559, 497]

        """
        return type(self)(itertools.accumulate(self.x, func, initial=initial))

    def chain(self, *iterables: Iterable[T]) -> Self:
        """Chain together multiple iterables. This is a replacement for the
        itertools ``chain`` function.  This version returns an instance of
        Iter_ to allow further iterable chaining.

        .. code-block:: python

            >>> Iter('ABC').chain('DEF').collect()
            ['A', 'B', 'C', 'D', 'E', 'F']
            >>> Iter('AB').chain('CD', 'EF').collect()
            ['A', 'B', 'C', 'D', 'E', 'F']
            >>> Iter('ABC').chain().collect()
            ['A', 'B', 'C']

        """
        return type(self)(itertools.chain(self.x, *iterables))

    def chain_from_iterable(self) -> Self:
        """This is similar to Iter.chain_ but it takes a single iterable
        of iterables. This is a replacement for the itertools
        ``chain.from_iterable`` function.  This version returns an
        instance of Iter_ to allow further iterable chaining.

        .. code-block:: python

            >>> Iter(['ABC', 'DEF']).chain_from_iterable().collect()
            ['A', 'B', 'C', 'D', 'E', 'F']
            >>> Iter([range(3), range(4)]).chain_from_iterable().collect()
            [0, 1, 2, 0, 1, 2, 3]

        """
        return type(self)(itertools.chain.from_iterable(self.x))

    def compress(self, selectors) -> Self:
        """Replacement for the itertools ``compress`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        .. code-block:: python

            >>> Iter('ABCDEF').compress([1, 0, 1, 0, 1, 1]).collect()
            ['A', 'C', 'E', 'F']

        """
        return type(self)(itertools.compress(self.x, selectors))

    def dropwhile(self, pred) -> Self:
        """
        Replacement for the itertools ``dropwhile`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        .. code-block:: python

            >>> Iter('abc').dropwhile(lambda x: x < 'c').collect()
            ['c']

        """
        return type(self)(itertools.dropwhile(pred, self.x))

    def filterfalse(self, pred) -> Self:
        """
        Replacement for the itertools ``filterfalse`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        .. code-block:: python

            >>> Iter('abc').filterfalse(lambda x: x < 'c').collect()
            ['c']

        `filterfalse` is useful when you want to exclude elements based
        on a membership test.

        .. code-block:: python

            >>> stopwords = {'the', 'and', 'or', 'but'}
            >>> text = 'the quick brown fox jumps over the lazy dog'.split()
            >>> Iter(text).filterfalse(stopwords.__contains__).collect()
            ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']

        """
        return type(self)(itertools.filterfalse(pred, self.x))

    def groupby(self, key=None) -> "Iter[Tuple[Any, Iter[T]]]":
        """
        Replacement for the itertools ``groupby`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        The grouper is also an instance of Iter_, mainly because that
        allows many different operations to be performed on the group
        as well as realizations like `.collect(...)` or `.ilen()`.

        .. code-block:: python

            >>> from collections import Counter
            >>> (
            ...   Iter('AAAABBBCCDAABBB')
            ...   .groupby()
            ...   .starmap(lambda key, grouper: (key, grouper.ilen()))
            ...   .collect()
            ... )
            [('A', 4), ('B', 3), ('C', 2), ('D', 1), ('A', 2), ('B', 3)]

        Note that it doesn't do a groupby in the sense that you would
        normally expect from a database query. It's more like a
        "consecutive groupby".

        To group up everything, it needs a bit more thinking:

        .. code-block:: python

            >>> def add_group_counts(d: dict, k, v):
            ...     d[k] = d.get(k, 0) + v
            ...     return d
            >>> (
            ...   Iter('AAAABBBCCDAABBB')
            ...   .groupby()
            ...   .starmap(lambda key, grouper: (key, grouper.ilen()))
            ...   .starreduce(add_group_counts, {})
            ... )
            {'A': 6, 'B': 6, 'C': 2, 'D': 1}

        In this specific example, we have merely reimplemented `collections.Counter`.

        """
        it = map(
            lambda x: (x[0], Iter(x[1])),
            itertools.groupby(self.x, key=key)
        )
        return type(self)(it)

    def islice(self, *args) -> Self:
        """
        Replacement for the itertools ``islice`` function.

        .. code-block:: python

            >>> Iter('abcdef').islice(2).collect()
            ['a', 'b']
            >>> Iter('abcdef').islice(2, 4).collect()
            ['c', 'd']

        """
        return type(self)(itertools.islice(self.x, *args))

    def starmap(self, func) -> Self:
        """
        Replacement for the itertools ``starmap`` function.

        .. code-block:: python

            >>> Iter([(0, 1), (2, 3)]).starmap(operator.add).collect()
            [1, 5]
            >>> Iter([(0, 1), (2, 3)]).starmap(lambda x, y: x + y).collect()
            [1, 5]

        """
        return type(self)(itertools.starmap(func, self.x))

    def takewhile(self, pred) -> Self:
        """
        Replacement for the itertools ``takewhile`` function.

        .. code-block:: python

            >>> Iter('abc').takewhile(lambda x: x < 'c').collect()
            ['a', 'b']

        """
        return type(self)(itertools.takewhile(pred, self.x))

    def tee(self, n=2) -> "Tuple[Self, ...]":
        """
        Replacement for the itertools ``tee`` function.  This version returns
        instances of Iter_ to allow further iterable chaining.

        .. code-block:: python

            >>> a, b = Iter('abc').tee()
            >>> a.collect()
            ['a', 'b', 'c']
            >>> b.collect()
            ['a', 'b', 'c']

        """
        # Pay attention
        cls = type(self)
        return cls(cls(_) for _ in itertools.tee(self.x, n))

    def zip_longest(self, *iterables, fillvalue=None) -> Self:
        """
        Replacement for the itertools ``zip_longest`` function.

        .. code-block:: python

            >>> Iter('abc').zip_longest('123').collect()
            [('a', '1'), ('b', '2'), ('c', '3')]
            >>> Iter('abcdef').zip_longest('123', fillvalue='x').collect()
            [('a', '1'), ('b', '2'), ('c', '3'), ('d', 'x'), ('e', 'x'), ('f', 'x')]

        """
        return type(self)(itertools.zip_longest(self.x, *iterables, fillvalue=fillvalue))

    # more-itertools
    # ===============

    # Grouping

    def chunked(self, n: int, strict: bool = False) -> Self:
        """
        Replacement for the more-itertools ``chunked`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.chunked <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.chunked>`_

        .. code-block:: python

            >>> Iter('abcdef').chunked(3).collect()
            [['a', 'b', 'c'], ['d', 'e', 'f']]
            >>> Iter('abcde').chunked(3).collect()
            [['a', 'b', 'c'], ['d', 'e']]

        """
        cls = type(self)
        return cls(more_itertools.chunked(self.x, n, strict=strict))

    def ichunked(self, n: int) -> Self:
        """
        Replacement for the more-itertools ``ichunked`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        This version differs from Iter.chunked_ in that it returns
        an iterator of iterators rather than an iterator of lists.

        Reference: `more_itertools.ichunked <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ichunked>`_

        .. code-block:: python

            >>> Iter('aabbcc').ichunked(3).map(list).collect()
            [['a', 'a', 'b'], ['b', 'c', 'c']]
            >>> Iter('aabbcc').ichunked(3).map(tuple).collect()
            [('a', 'a', 'b'), ('b', 'c', 'c')]
            >>> out = Iter('aabbcc').ichunked(3).map(set).collect()
            >>> out == [{'a', 'b'}, {'b', 'c'}]
            True

        """
        cls = type(self)
        return cls(cls(it) for it in more_itertools.ichunked(self.x, n))

    @classmethod
    def sliced(cls, seq: Sequence, n: int, strict: bool = False) -> Self:
        """
        |source|

        Replacement for the more-itertools ``sliced`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.sliced <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sliced>`_

        .. code-block:: python

            >>> Iter.sliced('abcdef', 3).collect()
            ['abc', 'def']

        """
        return cls(more_itertools.sliced(seq, n, strict=strict))

    def constrained_batches(self, max_size: int, max_count=None, get_len=len, strict=True) -> Self:
        """
        Replacement for the more-itertools ``constrained_batches`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.constrained_batches <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.constrained_batches>`_

        .. code-block:: python

            >>> iterable = [b'12345', b'123', b'12345678', b'1', b'1', b'12', b'1']
            >>> Iter(iterable).constrained_batches(10).collect()
            [(b'12345', b'123'), (b'12345678', b'1', b'1'), (b'12', b'1')]
            >>> Iter(iterable).constrained_batches(10, max_count=2).collect()
            [(b'12345', b'123'), (b'12345678', b'1'), (b'1', b'12'), (b'1',)]

        """
        return type(self)(more_itertools.constrained_batches(
            self.x, max_size, max_count=max_count, get_len=get_len, strict=strict
        ))

    def distribute(self, n: int) -> Self:
        """
        Replacement for the more-itertools ``distribute`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.distribute <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute>`_

        .. code-block:: python

            >>> group_1, group_2, group_3 = Iter('abcdef').distribute(3).collect()
            >>> group_1.collect()
            ['a', 'd']
            >>> group_2.collect()
            ['b', 'e']
            >>> group_3.collect()
            ['c', 'f']

        Note that each of the returned iterables is an instance of Iter_, so chaining works.

        .. code-block:: python

            >>> groups = Iter('abcdef').distribute(3).collect()
            >>> groups[0].map(str.upper).collect()
            ['A', 'D']

        """
        cls = type(self)
        return cls(
            (cls(x) for x in more_itertools.distribute(n, self.x))
        )

    def divide(self, n: int) -> Self:
        """
        Replacement for the more-itertools ``divide`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.divide <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.divide>`_

        .. code-block:: python

            >>> Iter('abcdef').divide(3).map(list).collect()
            [['a', 'b'], ['c', 'd'], ['e', 'f']]
            >>> Iter('abc').divide(5).map(list).collect()
            [['a'], ['b'], ['c'], [], []]

        """
        cls = type(self)
        return cls(cls(x) for x in more_itertools.divide(n, self.x))

    def split_at(self, pred: Callable[[T], bool], maxsplit=-1, keep_separator=False) -> Self:
        """
        Replacement for the more-itertools ``split_at`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.split_at <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_at>`_

        .. code-block:: python

            >>> Iter('abcdef').split_at(lambda char: char == 'd').collect()
            [['a', 'b', 'c'], ['e', 'f']]
            >>> Iter.range(10).split_at(lambda x: x % 2 == 1).collect()
            [[0], [2], [4], [6], [8], []]
            >>> Iter.range(10).split_at(lambda x: x % 2 == 1, maxsplit=2).collect()
            [[0], [2], [4, 5, 6, 7, 8, 9]]
            >>> Iter("abcdcba").split_at(lambda x: x == 'b', keep_separator=True).collect()
            [['a'], ['b'], ['c', 'd', 'c'], ['b'], ['a']]

        """
        return type(self)(more_itertools.split_at(self.x, pred, maxsplit=maxsplit, keep_separator=keep_separator))

    def split_before(self, pred: Callable[[T], bool], maxsplit=-1) -> Self:
        """
        Replacement for the more-itertools ``split_before`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.split_before <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_before>`_

        .. code-block:: python

            >>> Iter('OneTwo').split_before(str.isupper).collect()
            [['O', 'n', 'e'], ['T', 'w', 'o']]
            >>> Iter(range(10)).split_before(lambda x: x % 3 == 0, maxsplit=2).collect()
            [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

        """
        return type(self)(more_itertools.split_before(self.x, pred, maxsplit=maxsplit))

    def split_after(self, pred: Callable[[T], bool], maxsplit=-1) -> Self:
        """
        Replacement for the more-itertools ``split_after`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.split_after <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_after>`_

        .. code-block:: python

            >>> Iter('one1two2').split_after(str.isdigit).collect()
            [['o', 'n', 'e', '1'], ['t', 'w', 'o', '2']]
            >>> Iter(range(10)).split_after(lambda x: x % 3 == 0, maxsplit=2).collect()
            [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9]]

        """
        return type(self)(more_itertools.split_after(self.x, pred, maxsplit=maxsplit))

    def split_into(self, sizes: Iterable[int]) -> Self:
        """
        Replacement for the more-itertools ``split_into`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.split_into <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_into>`_

        .. code-block:: python

            >>> Iter('abcdef').split_into([2, 3, 1]).collect()
            [['a', 'b'], ['c', 'd', 'e'], ['f']]
            >>> Iter('abcdef').split_into(iter([2, 3, 1])).collect()
            [['a', 'b'], ['c', 'd', 'e'], ['f']]
            >>> Iter('abcdef').split_into([2, 3]).collect()
            [['a', 'b'], ['c', 'd', 'e']]

        """
        return type(self)(more_itertools.split_into(self.x, sizes))

    def split_when(self, pred: Callable[[T], bool], maxsplit=-1) -> Self:
        """
        Replacement for the more-itertools ``split_when`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.split_when <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_when>`_

        .. code-block:: python

            >>> Iter([1, 2, 3, 3, 2, 5, 2, 4, 2]).split_when(lambda x, y: x > y).collect()
            [[1, 2, 3, 3], [2, 5], [2, 4], [2]]

        """
        return type(self)(more_itertools.split_when(self.x, pred, maxsplit=maxsplit))

    def bucket(self, key, validator=None):
        """
        This is the basic example, copied from the more-itertools
        docs:

        .. code-block:: python

            >>> iterable = ['a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'b3']
            >>> b = Iter(iterable).bucket(key=lambda x: x[0])
            >>> sorted(b)
            ['a', 'b', 'c']
            >>> list(b['a'])
            ['a1', 'a2']

        Note that once consumed, you can't iterate over the contents
        of a group again.
        """
        cls = type(self)

        class _bucket(more_itertools.bucket):
            def __iter__(self):
                return cls(super().__iter__())

            def __getitem__(self, item):
                return cls(super().__getitem__(item))

        return _bucket(self.x, key, validator=validator)

    def unzip(self):
        """Docstring TODO"""
        return type(self)(type(self)(x) for x in more_itertools.unzip(self.x))

    def grouper(self, n: int, fillvalue=None, incomplete="fill") -> "Iter":
        """Docstring TODO"""
        return type(self)(
            more_itertools.grouper(
                self.x,
                n,
                incomplete=incomplete,
                fillvalue=fillvalue,
            )
        )

    def partition(self, pred) -> "Iter":
        """Docstring TODO"""
        left, right = more_itertools.partition(pred, self.x)
        return type(self)((type(self)(left), type(self)(right)))

    # Lookahead and lookback

    def spy(self, n=1) -> "Tuple[Iter, Iter]":
        """
        Replacement for the more-itertools ``spy`` function.  This version returns
        instances of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.spy <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.spy>`_

        .. code-block:: python

            >>> head, iterable = Iter('abcdefg').spy()
            >>> head.collect()
            ['a']
            >>> iterable.collect()
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> (head,), iterable = Iter('abcdefg').spy()
            >>> head
            'a'
            >>> (first, second), iterable = Iter('abcdefg').spy(2)
            >>> first
            'a'
            >>> second
            'b'

        """
        head, iterable = more_itertools.spy(self.x, n)
        return type(self)(head), type(self)(iterable)

    def peekable(self) -> "more_itertools.peekable":
        """
        Reference: `more_itertools.peekable <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.peekable>`_

        .. code-block:: python

            >>> p = Iter(['a', 'b']).peekable()
            >>> p.peek()
            'a'
            >>> next(p)
            'a'

        The peekable can be used to inspect what will be coming up.
        But if you then want to resume iterator chaining, pass the
        peekable back into an Iter_ instance.

        .. code-block:: python

            >>> p = Iter(range(10)).peekable()
            >>> p.peek()
            0
            >>> Iter(p).take(3).collect()
            [0, 1, 2]

        A peekable is not an Iter_ instance so it doesn't provide
        the iterator chaining methods. But if you want to get into
        chaining, use the ``iter()`` method.

        .. code-block:: python

            >>> p = Iter(range(5)).peekable()
            >>> p.peek()
            0
            >>> p[1]
            1
            >>> p.iter().take(3).collect()
            [0, 1, 2]

        Peekables can be prepended. But then you usually want to go
        right back to iterator chaining. Thus, the ``prepend`` method
        (on the returned ``peekable`` instance) returns an Iter_ instance.

        .. code-block:: python

            >>> p = Iter(range(3)).peekable()
            >>> p.peek()
            0
            >>> p.prepend('a', 'b').take(4).collect()
            ['a', 'b', 0, 1]

        """

        class _peekable(more_itertools.peekable):
            def __iter__(self):
                return Iter(super().__iter__())

            def iter(self) -> "Iter":
                return self.__iter__()

            def __getitem__(self, item):
                return super().__getitem__(item)

            def prepend(self, *items) -> "Iter":
                super().prepend(*items)
                return self.__iter__()

        return _peekable(self.x)

    def seekable(self, maxlen=None) -> "IterSeekable[T]":
        """
        Reference: `more_itertools.seekable <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.seekable>`_

        Allow for seeking forward and backward.

        .. code-block:: python

            >>> it = count().map(str).seekable()
            >>> next(it), next(it), next(it)
            ('0', '1', '2')
            >>> it.seek(0).take(3).collect(tuple)
            ('0', '1', '2')

        Seeking forward:

        .. code-block:: python

            >>> it = range(20).map(str).seekable()
            >>> it.seek(10).next()
            '10'
            >>> it.seek(20).collect()
            []
            >>> it.seek(0).next()
            '0'

        Call `relative_seek()` to seek relative to the current position:

        .. code-block:: python

            >>> it = range(20).map(str).seekable()
            >>> it.take(3).collect(tuple)
            ('0', '1', '2')
            >>> it.relative_seek(2).next()
            '5'
            >>> it.relative_seek(-3).next()
            '3'
            >>> it.relative_seek(-3).next()
            '1'

        Call peek() to look ahead one item without advancing the iterator:

        .. code-block:: python

            >>> it = Iter('1234').seekable()
            >>> it.peek()
            '1'
            >>> it.collect()
            ['1', '2', '3', '4']
            >>> it.peek(default='empty')
            'empty'

        Before the iterator is at its end, calling bool() on it will
        return ``True``. After it will return ``False``.

        .. code-block:: python

            >>> it = Iter('5678').seekable()
            >>> it.bool()
            True
            >>> it.collect()
            ['5', '6', '7', '8']
            >>> it.bool()
            False

        Use ``maxlen`` to limit the size of the internal cache used
        for seeking. This is useful to prevent memory issues when
        seeking through a very large iterator.

        .. code-block:: python

            >>> it = count().map(str).seekable(maxlen=2)
            >>> it.take(4).collect(tuple)
            ('0', '1', '2', '3')
            >>> it.seek(0).take(4).collect(tuple)
            ('2', '3', '4', '5')

        """

        class _seekable(more_itertools.seekable):
            def __iter__(self):
                return Iter(super().__iter__())

        return IterSeekable(_seekable(self.x, maxlen=maxlen))

    # Windowing

    def windowed(self, n, fillvalue=None, step=1) -> "Iter":
        """
        Replacement for the more-itertools ``windowed`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.windowed <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.windowed>`_

        .. code-block:: python

            >>> Iter([1, 2, 3, 4, 5]).windowed(3).collect()
            [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
            >>> Iter([1, 2, 3]).windowed(4).collect()
            [(1, 2, 3, None)]
            >>> Iter([1, 2, 3, 4, 5, 6]).windowed(3, fillvalue='!', step=2).collect()
            [(1, 2, 3), (3, 4, 5), (5, 6, '!')]

        """
        return type(self)(more_itertools.windowed(self.x, n, fillvalue=fillvalue, step=step))

    def substrings(self) -> "Iter":
        """
        Replacement for the more-itertools ``substrings`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.substrings <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.substrings>`_

        .. code-block:: python

            >>> Iter('more').substrings().map("".join).collect()
            ['m', 'o', 'r', 'e', 'mo', 'or', 're', 'mor', 'ore', 'more']
            >>> Iter([0, 1, 2]).substrings().collect()
            [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)]

        """
        return type(self)(more_itertools.substrings(self.x))

    def substrings_indexes(self, reverse=False) -> "Iter":
        """
        |warning| This function is not lazy and will consume the iterable fully before returning
        another iterable.

        Replacement for the more-itertools ``substrings_indexes`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

        Reference: `more_itertools.substrings_indexes <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.substrings_indexes>`_

        .. code-block:: python

            >>> (
            ...     Iter('more')
            ...     .substrings_indexes()
            ...     .starmap(lambda sub, start, end: ("".join(sub), start, end))
            ...     .side_effect(print)
            ...     .consume()
            ... )
            ('m', 0, 1)
            ('o', 1, 2)
            ('r', 2, 3)
            ('e', 3, 4)
            ('mo', 0, 2)
            ('or', 1, 3)
            ('re', 2, 4)
            ('mor', 0, 3)
            ('ore', 1, 4)
            ('more', 0, 4)

        """
        return type(self)(more_itertools.substrings_indexes(list(self.x), reverse=reverse))

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
        return type(self)(
            more_itertools.stagger(
                self.x,
                offsets=offsets,
                longest=longest,
                fillvalue=fillvalue,
            )
        )

    def pairwise(self):
        """
        Reference `more_itertools.pairwise <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise>`_

        .. code-block:: python

            >>> Iter.count().pairwise().take(4).collect()
            [(0, 1), (1, 2), (2, 3), (3, 4)]
        """
        return type(self)(more_itertools.pairwise(self.x))

    # Augmenting

    def count_cycle(self, n=None) -> "Iter":
        """
        Reference: `more_itertools.count_cycle <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle>`_

        .. code-block:: python

            >>> Iter('AB').count_cycle(3).collect()
            [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]

        """
        return type(self)(more_itertools.count_cycle(self.x, n=n))

    def intersperse(self, e, n=1) -> "Iter":
        """
        Reference: `more_itertools.intersperse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse>`_

        .. code-block:: python

            >>> Iter([1, 2, 3, 4, 5]).intersperse('!').collect()
            [1, '!', 2, '!', 3, '!', 4, '!', 5]

            >>> Iter([1, 2, 3, 4, 5]).intersperse(None, n=2).collect()
            [1, 2, None, 3, 4, None, 5]

        """
        return type(self)(more_itertools.intersperse(e, self.x, n=n))

    def padded(
        self,
        fillvalue: Optional[C] = None,
        n: Optional[int] = None,
        next_multiple: bool = False,
    ) -> "Iter[Union[T, C]]":
        """
        Reference: `more_itertools.padded <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).padded('?', 5).collect()
            [1, 2, 3, '?', '?']

            >>> Iter([1, 2, 3, 4]).padded(n=3, next_multiple=True).collect()
            [1, 2, 3, 4, None, None]

        """
        return type(self)(
            more_itertools.padded(
                self.x,
                fillvalue=fillvalue,
                n=n,
                next_multiple=next_multiple,
            )
        )

    # repeat from upstream

    def repeat_last(self, default=None) -> "Iter[T]":
        """
        Reference: `more_itertools.repeat_last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeat_last>`_

        .. code-block:: python

            >>> Iter(range(3)).repeat_last().islice(5).collect()
            [0, 1, 2, 2, 2]

            >>> Iter(range(0)).repeat_last(42).islice(5).collect()
            [42, 42, 42, 42, 42]

        """
        return type(self)(more_itertools.repeat_last(self.x, default=default))

    def adjacent(self, pred, distance=1) -> "Iter[Tuple[bool, T]]":
        """
        Reference: `more_itertools.adjacent <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent>`_

        .. code-block:: python

            >>> Iter(range(6)).adjacent(lambda x: x == 3).collect()
            [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]

            >>> Iter(range(6)).adjacent(lambda x: x == 3, distance=2).collect()
            [(False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)]


        """
        return type(self)(more_itertools.adjacent(pred, self.x, distance=distance))

    def groupby_transform(
        self,
        keyfunc: Optional[Callable[..., K]] = None,
        valuefunc: Optional[Callable[..., V]] = None,
        reducefunc: Optional[Callable[..., R]] = None,
    ) -> "Iter[Tuple[K, Iterable[V]]]":
        """
        Reference: `more_itertools.groupby_transform <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.groupby_transform>`_

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
        return type(self)(
            more_itertools.groupby_transform(
                self.x,
                keyfunc=keyfunc,
                valuefunc=valuefunc,
                reducefunc=reducefunc,
            )
        )

    def padnone(self) -> "Iter[Union[T, None]]":
        """
        Reference: `more_itertools.padnone <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone>`_

        .. code-block:: python

            >>> Iter(range(3)).padnone().take(5).collect()
            [0, 1, 2, None, None]

        """
        return type(self)(more_itertools.padnone(self.x))

    def ncycles(self, n) -> Self:
        """
        Reference: `more_itertools.ncycles <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles>`_

        .. code-block:: python

            >>> Iter(['a', 'b']).ncycles(3).collect()
            ['a', 'b', 'a', 'b', 'a', 'b']

        """
        return type(self)(more_itertools.ncycles(self.x, n))

    # Combining

    def collapse(self, base_type=None, levels=None) -> Self:
        """
        Reference: `more_itertools.collapse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.collapse>`_

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
        return type(self)(more_itertools.collapse(self.x, base_type=base_type, levels=levels))

    @class_or_instancemethod
    def sort_together(
        self_or_cls,
        iterables,
        key_list=(0,),
        key=None,
        reverse=False,
        strict=False,
    ):
        """
        Reference: `more_itertools.sort_together <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sort_together>`_

        This can be called either as an instance method or a class method.
        The classmethod form is more convenient if all the iterables are
        already available. The instancemethod form is more convenient if
        one of the iterables already goes through some transformation.

        Here are examples from the classmethod form, which mirror the
        examples in the more-itertools_ documentation:

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

        .. code-block:: python

            >>> iterables = [('a', 'b', 'c', 'd')]
            >>> Iter([4, 3, 2, 1]).sort_together(iterables).collect()
            [(1, 2, 3, 4), ('d', 'c', 'b', 'a')]

        """
        if isinstance(self_or_cls, type):
            return self_or_cls(
                more_itertools.sort_together(
                    iterables,
                    key_list=key_list,
                    key=key,
                    reverse=reverse,
                    strict=strict,
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
                    _temp(),
                    key_list=key_list,
                    key=key,
                    reverse=reverse,
                    strict=strict,
                )
            )

    @class_or_instancemethod
    def interleave(self_or_cls, *iterables) -> Self:
        """
        Reference: `more_itertools.interleave <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave>`_

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
    def interleave_longest(self_or_cls, *iterables) -> Self:
        """
        Reference: `more_itertools.interleave_longest <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave_longest>`_

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
    def zip_offset(cls, *iterables, offsets, longest=False, fillvalue=None) -> Self:
        """
        Reference: `more_itertools.zip_offset <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_offset>`_

        .. code-block:: python

            >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1)).collect()
            [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e')]

            >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True).collect()
            [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e'), (None, 'f')]
        """
        # TODO: also make an instancemethod
        return cls(
            more_itertools.zip_offset(
                *iterables,
                offsets=offsets,
                longest=longest,
                fillvalue=fillvalue,
            )
        )

    def dotproduct(self, vec2: Iterable):
        """
        Reference: `more_itertools.dotproduct <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.dotproduct>`_

        .. code-block:: python

            >>> Iter([10, 10]).dotproduct([20, 20])
            400
        """
        return more_itertools.dotproduct(self.x, vec2)

    def flatten(self) -> Self:
        """
        Reference: `more_itertools.flatten <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten>`_

        .. code-block:: python

            >>> Iter([[0, 1], [2, 3]]).flatten().collect()
            [0, 1, 2, 3]

        """
        return type(self)(more_itertools.flatten(self))

    @class_or_instancemethod
    def roundrobin(
        self_or_cls: Union[Type[T], T], *iterables: C
    ) -> "Iter[Union[T, C]]":
        """
        Reference: `more_itertools.roundrobin <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.roundrobin>`_

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

    def prepend(self, value: C) -> "Iter[Union[T, C]]":
        """
        Reference: `more_itertools.prepend <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.prepend>`_

        .. code-block:: python

            >>> value = '0'
            >>> iterator = ['1', '2', '3']
            >>> Iter(iterator).prepend(value).collect()
            ['0', '1', '2', '3']

        """
        return type(self)(more_itertools.prepend(value, self))

    # Summarizing

    def ilen(self) -> "int":
        """
        |sink|

        Reference: `more_itertools.ilen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ilen>`_

        .. code-block:: python

            >>> Iter(x for x in range(1000000) if x % 3 == 0).ilen()
            333334

        """
        return more_itertools.ilen(self)

    def unique_to_each(self) -> Self:
        """
        Reference: `more_itertools.unique_to_each <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each>`_


        .. code-block:: python

            >>> Iter([{'A', 'B'}, {'B', 'C'}, {'B', 'D'}]).unique_to_each().collect()
            [['A'], ['C'], ['D']]

            >>> Iter(["mississippi", "missouri"]).unique_to_each().collect()
            [['p', 'p'], ['o', 'u', 'r']]

        Note that this will internally construct the full list of the uniques for each group.

        """
        # TODO: this operation is a little different to the others that take
        #  a *iterables parameter. Not sure if this is what we want.
        return Iter(more_itertools.unique_to_each(*self))

    def sample(self, k=1, weights=None, *, counts=None, strict=False) -> "Iter":
        """
        Reference: `more_itertools.sample <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sample>`_

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
        return type(self)(
            more_itertools.sample(
                self.x,
                k=k,
                weights=weights,
                counts=counts,
                strict=strict,
            )
        )

    def consecutive_groups(self, ordering=lambda x: x):
        """
        Reference: `more_itertools.consecutive_groups <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consecutive_groups>`_

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

    def run_length_encode(self) -> "Iter[Tuple[T, int]]":
        """
        Reference: `more_itertools.run_length <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length>`_

        .. code-block:: python

            >>> uncompressed = 'abbcccdddd'
            >>> Iter(uncompressed).run_length_encode().collect()
            [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

        """
        return Iter(more_itertools.run_length.encode(self))

    def run_length_decode(self) -> "Iter":
        """
        Reference: `more_itertools.run_length <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length>`_

        .. code-block:: python

            >>> compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
            >>> Iter(compressed).run_length_decode().collect()
            ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

        """
        return Iter(more_itertools.run_length.decode(self))

    def map_reduce(self, keyfunc, valuefunc=None, reducefunc=None) -> "Dict":
        """
        Reference: `more_itertools.map_reduce <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce>`_

        This interface mirrors what more-itertools_ does in that it returns
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

        Note the warning given in the more-itertools_ docs about how
        lists are created before the reduce step. This means you always want
        to filter *before* applying map_reduce, not after.

        .. code-block:: python

            >>> all_items = builtins.range(30)
            >>> keyfunc = lambda x: x % 2  # Evens map to 0; odds to 1
            >>> categories = Iter(all_items).filter(lambda x: 10<=x<=20).map_reduce(keyfunc=keyfunc)
            >>> sorted(categories.items())
            [(0, [10, 12, 14, 16, 18, 20]), (1, [11, 13, 15, 17, 19])]
            >>> summaries = Iter(all_items).filter(lambda x: 10<=x<=20).map_reduce(keyfunc=keyfunc, reducefunc=sum)
            >>> sorted(summaries.items())
            [(0, 90), (1, 75)]

        """
        return more_itertools.map_reduce(
            self.x,
            keyfunc,
            valuefunc,
            reducefunc,
        )

    def map_reduce_it(
        self,
        keyfunc: Callable[..., K],
        valuefunc: Optional[Callable[..., V]] = None,
        reducefunc: Optional[Callable[..., R]] = None,
    ) -> "Iter[Tuple[K, R]]":
        """
        Reference: `more_itertools.map_reduce <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce>`_

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
        d = more_itertools.map_reduce(
            self.x,
            keyfunc,
            valuefunc,
            reducefunc,
        )
        return Iter(d.items())

    def exactly_n(self, n, predicate=bool) -> "bool":
        """
        |sink|

        Reference: `more_itertools.exactly_n <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.exactly_n>`_

        .. code-block:: python

            >>> Iter([True, True, False]).exactly_n(2)
            True

        """
        return more_itertools.exactly_n(self.x, n=n, predicate=predicate)

    def is_sorted(self, key=None, reverse=False, strict=False) -> "bool":
        """
        |sink|

        Reference: `more_itertools.is_sorted <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.is_sorted>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).is_sorted()
            True

        """
        return more_itertools.is_sorted(
            self.x,
            key=key,
            reverse=reverse,
            strict=strict,
        )

    def all_unique(self, key=None) -> "bool":
        """
        |sink|

        Reference: `more_itertools.all_unique <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.all_unique>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).all_unique()
            True

        """
        return more_itertools.all_unique(self.x, key=key)

    def minmax(
        self,
        key: Optional[Callable[[T], Any]] = None,
        default: Any = _marker,
    ) -> "Tuple[T, T]":
        """
        |sink|

        Reference: `more_itertools.minmax <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.minmax>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).minmax()
            (1, 3)
            >>> Iter([5, 30]).minmax(key=str)
            (30, 5)
            >>> Iter([]).minmax(default=(0, 0))
            (0, 0)

        """
        if default is _marker:
            return more_itertools.minmax(self.x, key=key)
        return more_itertools.minmax(self.x, key=key, default=default)

    def all_equal(self, key: Optional[Callable[[T], Any]] = None) -> bool:
        """
        |sink|

        Reference: `more_itertools.all_equal <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.all_equal>`_

        .. code-block:: python

            >>> Iter([1, 1, 1]).all_equal()
            True
            >>> Iter([1, 2, 1]).all_equal()
            False
            >>> Iter("aaaa").all_equal()
            True

        """
        return more_itertools.all_equal(self.x, key=key)

    def first_true(self, default=None, pred=None) -> Optional[T]:
        """
        |sink|

        Reference: `more_itertools.first_true <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.first_true>`_

        .. code-block:: python

            >>> Iter([False, False, True, False]).first_true()
            True
            >>> Iter(range(10)).first_true()
            1
            >>> Iter(range(10)).first_true(pred=lambda x: x > 5)
            6
            >>> Iter(range(10)).first_true(default="missing", pred=lambda x: x > 9)
            'missing'

        """
        return more_itertools.first_true(self.x, default=default, pred=pred)

    def quantify(self, pred=bool):
        """
        |sink|

        Reference: `more_itertools.quantify <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.quantify>`_

        .. code-block:: python

            >>> Iter([True, False, True, False, True]).quantify()
            3

        """
        return more_itertools.quantify(self.x, pred=pred)

    # Selecting

    def islice_extended(self, *args) -> "Iter":
        """
        Reference: `more_itertools.islice_extended <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.islice_extended>`_

        .. code-block:: python

            >>> Iter('abcdefgh').islice_extended(-4, -1).collect()
            ['e', 'f', 'g']

        .. code-block:: python

            >>> Iter.count().islice_extended(110, 99, -2).collect()
            [110, 108, 106, 104, 102, 100]

        """
        return Iter(more_itertools.islice_extended(self.x, *args))

    def first(self, default: "V" = _marker) -> "T | V":
        """
        Reference: `more_itertools.first <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.first>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).first()
            1
            >>> Iter([]).first(default='missing')
            'missing'

        """
        if default is _marker:
            return more_itertools.first(self.x)
        return more_itertools.first(self.x, default)

    def last(self, default: "V" = _marker) -> "T | V":
        """
        Reference: `more_itertools.last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.last>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).last()
            3
            >>> Iter([]).last(default='missing')
            'missing'
        """
        if default is _marker:
            return more_itertools.last(self.x)
        return more_itertools.last(self.x, default)

    def one(self, too_short=None, too_long=None) -> "T":
        """
        Return the first item from the iterable, raising an exception if there
        is not exactly one item.

        Note that one() attempts to advance iterable twice to ensure there is
        only one item. See Iter.spy_ or Iter.peekable_ to check iterable
        contents less destructively.

        Reference: `more_itertools.one <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.one>`_

        .. code-block:: python

            >>> Iter([]).one()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ...
            >>> Iter([42]).one()
            42

        """
        return more_itertools.one(self.x, too_short=too_short, too_long=too_long)

    def only(self, default=None, too_long=None) -> "T":
        """
        Reference: `more_itertools.one <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.one>`_

        .. code-block:: python

            >>> Iter([]).only(default='missing')
            'missing'
            >>> Iter([42]).only(default='missing')
            42
            >>> Iter([1, 2]).only()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ...

        """
        return more_itertools.only(self.x, default=default, too_long=too_long)

    def strictly_n(self, n, too_short=None, too_long=None) -> "Self":
        """
        Make sure you understand how the underlying function works. Check
        the more-itertools_ documentation for more information.

        Reference: `more_itertools.strictly_n <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.strictly_n>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).strictly_n(3).collect()
            [1, 2, 3]
            >>> Iter([1, 2, 3]).strictly_n(2).collect()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ...
            >>> Iter([1, 2, 3]).strictly_n(4).collect()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ...

        """
        return type(self)(more_itertools.strictly_n(self.x, n, too_short=too_short, too_long=too_long))

    def strip(self, pred) -> "Iter[T]":
        """
        Reference: `more_itertools.strip <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.strip>`_

        .. code-block:: python

            >>> iterable = (None, False, None, 1, 2, None, 3, False, None)
            >>> pred = lambda x: x in {None, False, ''}
            >>> Iter(iterable).strip(pred).collect()
            [1, 2, None, 3]

        """
        return Iter(more_itertools.strip(self.x, pred))

    def lstrip(self, pred) -> "Iter[T]":
        """
        Reference: `more_itertools.lstrip <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.lstrip>`_

        .. code-block:: python

            >>> iterable = (None, False, None, 1, 2, None, 3, False, None)
            >>> pred = lambda x: x in {None, False, ''}
            >>> Iter(iterable).lstrip(pred).collect()
            [1, 2, None, 3, False, None]

        """
        return Iter(more_itertools.lstrip(self.x, pred))

    def rstrip(self, pred) -> "Iter[T]":
        """
        Reference: `more_itertools.rstrip <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.rstrip>`_

        .. code-block:: python

            >>> iterable = (None, False, None, 1, 2, None, 3, False, None)
            >>> pred = lambda x: x in {None, False, ''}
            >>> Iter(iterable).rstrip(pred).collect()
            [None, False, None, 1, 2, None, 3]

        """
        return Iter(more_itertools.rstrip(self.x, pred))

    def filter_except(self, validator, *exceptions) -> "Iter[T]":
        """
        Reference: `more_itertools.filter_except <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.filter_except>`_

        .. code-block:: python

            >>> iterable = ['1', '2', 'three', '4', None]
            >>> Iter(iterable).filter_except(int, ValueError, TypeError).collect()
            ['1', '2', '4']

        """
        return Iter(more_itertools.filter_except(validator, self.x, *exceptions))

    def map_except(self, function, *exceptions) -> "Iter":
        """
        Reference: `more_itertools.map_except <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_except>`_

        .. code-block:: python

            >>> iterable = ['1', '2', 'three', '4', None]
            >>> Iter(iterable).map_except(int, ValueError, TypeError).collect()
            [1, 2, 4]

        """
        return Iter(more_itertools.map_except(function, self.x, *exceptions))

    def nth_or_last(self, n, default=_marker) -> "T":
        """
        Reference: `more_itertools.nth_or_last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.nth_or_last>`_

        .. code-block:: python

            >>> Iter([0, 1, 2, 3]).nth_or_last(2)
            2
            >>> Iter([0, 1]).nth_or_last(2)
            1
            >>> Iter([]).nth_or_last(0, 'some default')
            'some default'

        """
        if default is _marker:
            return more_itertools.nth_or_last(self, n)
        else:
            return more_itertools.nth_or_last(self, n, default)

    def nth(self, n, default=None):
        """
        Reference: `more_itertools.nth <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.nth>`_
        """
        return next(self.islice(n, None), default)

    def take(self, n: int) -> "Iter":
        """
        Reference: `more_itertools.take <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.take>`_
        """
        return Iter(more_itertools.take(n, self.x))

    def tail(self, n) -> "Iter[T]":
        """
        Reference: `more_itertools.tail <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.tail>`_

        .. code-block:: python

            >>> Iter('ABCDEFG').tail(3).collect()
            ['E', 'F', 'G']

        """
        return Iter(more_itertools.tail(n, self))

    def unique_everseen(self, key=None) -> "Iter[T]":
        """
        Reference: `more_itertools.unique_everseen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_everseen>`_

        .. code-block:: python

            >>> Iter('AAAABBBCCDAABBB').unique_everseen().collect()
            ['A', 'B', 'C', 'D']
            >>> Iter('ABBCcAD').unique_everseen(key=str.lower).collect()
            ['A', 'B', 'C', 'D']

        Be sure to read the *more-itertools* docs when using unhashable
        items; provide a key that converts each item to a hashable value.

        .. code-block:: python

            >>> iterable = ([1, 2], [2, 3], [1, 2])
            >>> Iter(iterable).unique_everseen(key=tuple).collect()
            [[1, 2], [2, 3]]

        """
        return Iter(more_itertools.unique_everseen(self, key=key))

    def unique_justseen(self, key=None) -> "Iter[T]":
        """
        Reference: `more_itertools.unique_justseen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_justseen>`_

        .. code-block:: python

            >>> Iter('AAAABBBCCDAABBB').unique_justseen().collect()
            ['A', 'B', 'C', 'D', 'A', 'B']
            >>> Iter('ABBCcAD').unique_justseen(key=str.lower).collect()
            ['A', 'B', 'C', 'A', 'D']

        """
        return Iter(more_itertools.unique_justseen(self, key=key))

    # Combinatorics

    def distinct_permutations(self, r=None):
        """
        Reference: `more_itertools.distinct_permutations <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distinct_permutations>`_

        .. code-block:: python

            >>> Iter([1, 0, 1]).distinct_permutations().sorted().collect()
            [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

        """
        return type(self)(more_itertools.distinct_permutations(self.x, r=r))

    def distinct_combinations(self, r) -> "Iter[T]":
        """
        Reference: `more_itertools.distinct_combinations <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distinct_combinations>`_

        .. code-block:: python

            >>> Iter([0, 0, 1]).distinct_combinations(2).collect()
            [(0, 0), (0, 1)]

        """
        return Iter(more_itertools.distinct_combinations(self, r))

    def circular_shifts(self, steps=1) -> "Iter[T]":
        """
        Reference: `more_itertools.circular_shifts <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.circular_shifts>`_

        .. code-block:: python

            >>> Iter(range(4)).circular_shifts().collect()
            [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]

        """
        return type(self)(more_itertools.circular_shifts(self, steps=steps))

    def partitions(self) -> "Iter[T]":
        """
        Reference: `more_itertools.partitions <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.partitions>`_

        .. code-block:: python

            >>> Iter('abc').partitions().collect()
            [[['a', 'b', 'c']], [['a'], ['b', 'c']], [['a', 'b'], ['c']], [['a'], ['b'], ['c']]]
            >>> Iter('abc').partitions().print('{v}').consume()
            [['a', 'b', 'c']]
            [['a'], ['b', 'c']]
            [['a', 'b'], ['c']]
            [['a'], ['b'], ['c']]
            >>> Iter('abc').partitions().map(lambda v: [''.join(p) for p in v]).print('{v}').consume()
            ['abc']
            ['a', 'bc']
            ['ab', 'c']
            ['a', 'b', 'c']

        """
        return Iter(more_itertools.partitions(self.x))

    def set_partitions(self, k=None, min_size=None, max_size=None) -> "Iter[T]":
        """
        Reference: `more_itertools.set_partitions <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.set_partitions>`_

        .. code-block:: python

            >>> Iter('abc').set_partitions(2).collect()
            [[['a'], ['b', 'c']], [['a', 'b'], ['c']], [['b'], ['a', 'c']]]

        """
        return type(self)(
            more_itertools.set_partitions(
                self,
                k=k,
                min_size=min_size,
                max_size=max_size,
            )
        )

    def powerset(self):
        """
        Reference: `more_itertools.powerset <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.powerset>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).powerset().collect()
            [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

        """
        return Iter(more_itertools.powerset(self))

    @class_or_instancemethod
    def random_product(self_or_cls, *args, repeat=1):
        """
        Reference: `more_itertools.random_product <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_product>`_

        .. code-block:: python

            >>> Iter('abc').random_product(range(4), 'XYZ').collect()  # doctest: +SKIP
            ['c', 3, 'X']
            >>> Iter.random_product('abc', range(4), 'XYZ').collect()  # doctest: +SKIP
            ['c', 0, 'Z']
            >>> Iter('abc').random_product(range(0)).collect()
            Traceback (most recent call last):
                ...
            IndexError: Cannot choose from an empty sequence
            >>> Iter.random_product(range(0)).collect()
            Traceback (most recent call last):
                ...
            IndexError: Cannot choose from an empty sequence

        """
        if isinstance(self_or_cls, type):
            return Iter(more_itertools.random_product(*args, repeat=repeat))
        else:
            return Iter(
                more_itertools.random_product(self_or_cls, *args, repeat=repeat)
            )

    def random_permutation(self, r=None):
        """
        Reference: `more_itertools.random_permutation <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_permutation>`_

        .. code-block:: python

            >>> Iter(range(5)).random_permutation().collect()  # doctest: +SKIP
            [2, 0, 4, 3, 1]
            >>> Iter(range(0)).random_permutation().collect()
            []

        """
        return Iter(more_itertools.random_permutation(self, r=r))

    def random_combination(self, r):
        """
        Reference: `more_itertools.random_combination <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_combination>`_

        .. code-block:: python

            >>> Iter(range(5)).random_combination(3).collect()  # doctest: +SKIP
            [0, 1, 4]
            >>> Iter(range(5)).random_combination(0).collect()
            []

        """
        return Iter(more_itertools.random_combination(self, r))

    def random_combination_with_replacement(self, r):
        """
        Reference: `more_itertools.random_combination_with_replacement <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_combination_with_replacement>`_

        .. code-block:: python

            >>> Iter(range(3)).random_combination_with_replacement(5).collect()  # doctest: +SKIP
            [0, 0, 1, 2, 2]
            >>> Iter(range(3)).random_combination_with_replacement(0).collect()
            []

        """
        return Iter(more_itertools.random_combination_with_replacement(self, r))

    def nth_combination(self, r, index):
        """
        Reference: `more_itertools.nth_combination <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.nth_combination>`_

        .. code-block:: python

            >>> Iter(range(9)).nth_combination(3, 1).collect()
            [0, 1, 3]
            >>> Iter(range(9)).nth_combination(3, 2).collect()
            [0, 1, 4]
            >>> Iter(range(9)).nth_combination(3, 3).collect()
            [0, 1, 5]
            >>> Iter(range(9)).nth_combination(4, 3).collect()
            [0, 1, 2, 6]
            >>> Iter(range(9)).nth_combination(3, 7).collect()
            [0, 2, 3]

        """
        return Iter(more_itertools.nth_combination(self, r, index))

    # Wrapping

    @classmethod
    def always_iterable(cls, obj, base_type=(str, bytes)) -> "Iter":
        """
        Reference: `more_itertools.always_iterable <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.always_iterable>`_

        .. code-block: python

        .. code-block:: python

            >>> Iter.always_iterable([1, 2, 3]).collect()
            [1, 2, 3]
            >>> Iter.always_iterable(1).collect()
            [1]
            >>> Iter.always_iterable(None).collect()
            []
            >>> Iter.always_iterable('foo').collect()
            ['foo']
            >>> Iter.always_iterable(dict(a=1), base_type=dict).collect()
            [{'a': 1}]

        """
        return Iter(more_itertools.always_iterable(obj, base_type=base_type))

    def always_reversible(self):
        """
        Reference: `more_itertools.always_reversible <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.always_reversible>`_

        This is like ``reversed()`` but it also operates on things that
        wouldn't normally be reversible, like generators. It does this with
        internal caching, so be careful with memory use.

        .. code-block: python

            >>> Iter('abc').always_reversible().collect()
            ['c', 'b', 'a']
            >>> Iter(x for x in 'abc').always_reversible().collect()
            ['c', 'b', 'a']

        """
        return Iter(more_itertools.always_reversible(self.x))

    # TODO: do we need this? See Iter.send_ further down.
    # def consumer(self):
    #     """
    #     Reference: `more_itertools.consumer <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.consumer>`_
    #     """
    #     raise NotImplementedError

    @classmethod
    def with_iter(cls, context_manager):
        """
        Reference: `more_itertools.with_iter <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.with_iter>`_

        Note: Any context manager which returns an iterable is a candidate for
        Iter.with_iter_.

        .. code-block:: python

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as td:
            ...     with open(td + 'text.txt', 'w') as f:
            ...         f.writelines(['abc\\n', 'def\\n', 'ghi\\n'])
            ...     Iter.with_iter(open(td + 'text.txt')).map(lambda x: x.upper()).collect()
            ['ABC\\n', 'DEF\\n', 'GHI\\n']

        See also: Iter.open_

        |flux| TODO: perhaps we should get rid of Iter.open_ and just use this?

        """
        return Iter(more_itertools.with_iter(context_manager))

    @classmethod
    def iter_except(cls, func, exception, first=None) -> "Iter":
        """
        Reference: `more_itertools.iter_except <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.iter_except>`_

        .. code-block:: python

            >>> l = [0, 1, 2]
            >>> Iter.iter_except(l.pop, IndexError).collect()
            [2, 1, 0]

        """
        return Iter(more_itertools.iter_except(func, exception, first=first))

    # Others

    def locate(self, pred=bool, window_size=None) -> "Iter":
        """
        Reference: `more_itertools.locate <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.locate>`_

        .. code-block:: python

            >>> Iter([0, 1, 1, 0, 1, 0, 0]).locate().collect()
            [1, 2, 4]

        .. code-block:: python

            >>> Iter(['a', 'b', 'c', 'b']).locate(lambda x: x == 'b').collect()
            [1, 3]

        .. code-block:: python

            >>> iterable = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
            >>> pred = lambda *args: args == (1, 2, 3)
            >>> Iter(iterable).locate(pred=pred, window_size=3).collect()
            [1, 5, 9]

        .. code-block:: python

            >>> from itertools import count
            >>> from more_itertools import seekable
            >>> source = (3 * n + 1 if (n % 2) else n // 2 for n in count())
            >>> it = Iter(source).seekable()
            >>> pred = lambda x: x > 100
            >>> # TODO: can we avoid making two instances?
            >>> indexes = it.locate(pred=pred)
            >>> i = next(indexes)
            >>> it.seek(i).next()
            106

        """
        return Iter(more_itertools.locate(self.x, pred=pred, window_size=window_size))

    def rlocate(self, pred=bool, window_size=None) -> "Iter":
        """
        Reference: `more_itertools.rlocate <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.rlocate>`_

        .. code-block:: python

            >>> Iter([0, 1, 1, 0, 1, 0, 0]).rlocate().collect()  # Truthy at 1, 2, and 4
            [4, 2, 1]

        .. code-block:: python

            >>> pred = lambda x: x == 'b'
            >>> Iter('abcb').rlocate(pred).collect()
            [3, 1]

        .. code-block:: python

            >>> iterable = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
            >>> pred = lambda *args: args == (1, 2, 3)
            >>> Iter(iterable).rlocate(pred=pred, window_size=3).collect()
            [9, 5, 1]

        """
        return Iter(more_itertools.rlocate(self.x, pred, window_size))

    def replace(self, pred, substitutes, count=None, window_size=1) -> "Iter":
        """
        Reference: `more_itertools.replace <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.replace>`_

        .. code-block:: python

            >>> iterable = [1, 1, 0, 1, 1, 0, 1, 1]
            >>> pred = lambda x: x == 0
            >>> substitutes = (2, 3)
            >>> Iter(iterable).replace(pred, substitutes).collect()
            [1, 1, 2, 3, 1, 1, 2, 3, 1, 1]

        .. code-block:: python

            >>> iterable = [1, 1, 0, 1, 1, 0, 1, 1, 0]
            >>> pred = lambda x: x == 0
            >>> substitutes = [None]
            >>> Iter(iterable).replace(pred, substitutes, count=2).collect()
            [1, 1, None, 1, 1, None, 1, 1, 0]

        .. code-block:: python

            >>> iterable = [0, 1, 2, 5, 0, 1, 2, 5]
            >>> window_size = 3
            >>> pred = lambda *args: args == (0, 1, 2)  # 3 items passed to pred
            >>> substitutes = [3, 4] # Splice in these items
            >>> Iter(iterable).replace(
            ...     pred, substitutes, window_size=window_size
            ... ).collect()
            [3, 4, 5, 3, 4, 5]

        """
        return Iter(
            more_itertools.replace(
                self.x, pred, substitutes, count=count, window_size=window_size
            )
        )

    @classmethod
    def numeric_range(cls, *args) -> "Iter":
        """
        Reference: `more_itertools.numeric_range <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.numeric_range>`_

        .. code-block:: python

            >>> Iter.numeric_range(3.5).collect()
            [0.0, 1.0, 2.0, 3.0]

        .. code-block:: python

            >>> from decimal import Decimal
            >>> start = Decimal('2.1')
            >>> stop = Decimal('5.1')
            >>> Iter.numeric_range(start, stop).collect()
            [Decimal('2.1'), Decimal('3.1'), Decimal('4.1')]

        .. code-block:: python

            >>> from fractions import Fraction
            >>> start = Fraction(1, 2)  # Start at 1/2
            >>> stop = Fraction(5, 2)  # End at 5/2
            >>> step = Fraction(1, 2)  # Count by 1/2
            >>> Iter.numeric_range(start, stop, step).collect()
            [Fraction(1, 2), Fraction(1, 1), Fraction(3, 2), Fraction(2, 1)]

        .. code-block:: python

            >>> Iter.numeric_range(3, -1, -1.0).collect()
            [3.0, 2.0, 1.0, 0.0]

        """
        return Iter(more_itertools.numeric_range(*args))

    def side_effect(self, func, *args, chunk_size=None, before=None, after=None):
        """
        Reference: `more_itertools.side_effect <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.side_effect>`_

        .. code-block:: python

            >>> def f(item):
            ...     if item == 3:
            ...         raise Exception('got 3')
            >>> Iter.range(5).side_effect(f).consume()
            Traceback (most recent call last):
                ...
            Exception: got 3

        .. code-block:: python

            >>> func = lambda item: print('Received {}'.format(item))
            >>> Iter.range(2).side_effect(func).consume()
            Received 0
            Received 1

        This version of ``side_effect`` also allows extra args:

        .. code-block:: python

            >>> func = lambda item, format_str='Received {}': print(format_str.format(item))
            >>> Iter.range(2).side_effect(func).consume()
            Received 0
            Received 1
            >>> func = lambda item, format_str='Received {}': print(format_str.format(item))
            >>> Iter.range(2).side_effect(func, 'Got {}').consume()
            Got 0
            Got 1


        """
        f = lambda *value: func(*value, *args)
        return Iter(
            more_itertools.side_effect(
                f, self.x, chunk_size=chunk_size, before=before, after=after
            )
        )

    @classmethod
    def iterate(cls, func, start) -> "Iter":
        """
        |source|

        Reference: `more_itertools.iterate <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.iterate>`_

        .. code-block:: python

            >>> def f(x):
            ...     return x * 2
            >>> Iter.iterate(f, 1).take(4).collect()
            [1, 2, 4, 8]
        """
        return Iter(more_itertools.iterate(func, start))

    def difference(self, func=operator.sub, *, initial=None) -> "Iter":
        """
        Reference: `more_itertools.difference <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=difference#more_itertools.difference>`_

        .. code-block:: python

            >>> iterable = [0, 1, 3, 6, 10]
            >>> Iter(iterable).difference().collect()
            [0, 1, 2, 3, 4]

        .. code-block:: python

            >>> iterable = [1, 2, 6, 24, 120]  # Factorial sequence
            >>> func = lambda x, y: x // y
            >>> Iter(iterable).difference(func).collect()
            [1, 2, 3, 4, 5]

        """
        return Iter(
            more_itertools.difference(
                self.x,
                func=func,
                initial=initial,
            )
        )

    @classmethod
    def make_decorator(cls, wrapping_func, result_index=0):
        """
        |flux|

        Reference: `more_itertools.make_decorator <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=difference#more_itertools.make_decorator>`_

        .. code-block:: python

            >>> chunker = Iter.make_decorator(Iter.chunked, result_index=0)
            >>> @chunker(3)
            ... def iter_range(n):
            ...     return range(n)
            >>> list(iter_range(9))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            >>> peekable_function = Iter.make_decorator(Iter.peekable)
            >>> @peekable_function()
            ... def str_range(*args):
            ...     return Iter(str(x) for x in range(*args))
            >>> it = str_range(1, 20, 2)
            >>> it.peek()
            '1'
            >>> next(it)
            '1'
            >>> it.peek()
            '3'

        """
        # TODO: a bit unfortunate that we need the `Iter` in there in the peekable
        #  doctest.
        return more_itertools.make_decorator(wrapping_func, result_index=result_index)

    # def SequenceView(self):
    #     raise NotImplementedError

    def time_limited(self, limit_seconds) -> "Iter":
        """
        Reference: `more_itertools.time_limited <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=time_limited#more_itertools.time_limited>`_

        .. code-block:: python

            >>> from time import sleep
            >>> def generator():
            ...     yield 1
            ...     yield 2
            ...     sleep(0.2)
            ...     yield 3
            >>> Iter(generator()).time_limited(0.1).collect()
            [1, 2]

        """
        return Iter(more_itertools.time_limited(limit_seconds, self.x))

    def consume(self, n: Optional[int] = None) -> "Optional[Iter[T]]":
        """
        |sink|
        If n is not provided, the entire iterator is consumed and
        ``None`` is returned. Otherwise, an iterator will *always* be
        returned, even if n is greater than the number of items left in
        the iterator.

        In this example, the source has more elements than what we consume,
        so there will still be data available on the chain:

        .. code-block:: python

            >>> range(10).consume(5).collect()
            [5, 6, 7, 8, 9]

        We can bump up the count of how many items can be consumed. Note that
        even though ``n`` is greater than the number of items in the source,
        it is still required to call Iter.collect_ to consume the remaining
        items.

        .. code-block:: python

            >>> range(10).consume(50).collect()
            []

        Finally, if ``n`` is not provided, the entire stream is consumed.
        In this scenario, Iter.collect_ would fail since nothing is being
        returned from the consume call.

        .. code-block:: python

            >>> assert range(10).consume() is None

        """
        more_itertools.consume(self, n=n)
        if n is not None:
            return self

    @classmethod
    def tabulate(cls, function, start=0) -> "Iter":
        """
        |source|

        Reference: `more_itertools.tabulate <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.tabulate>`_

        .. code-block:: python

            >>> Iter.tabulate(lambda x: x ** 2, start=-3).take(4).collect()
            [9, 4, 1, 0]
        """
        return cls(more_itertools.tabulate(function, start=start))

    @classmethod
    def repeatfunc(cls, func, *args, times=None):
        """
        |source|
        Docstring TODO

        .. code-block:: python

            >>> Iter.repeatfunc(operator.add, 3, 5, times=4).collect()
            [8, 8, 8, 8]

        """
        return Iter(more_itertools.repeatfunc(func, times, *args))

    # New

    def wrap(self, ends: "Sequence[T, T]" = "()"):
        """
        Wrap the iterator with a start and end value. This is useful for
        adding brackets, parens, or other delimiters around the data.

        .. code-block:: python

            >>> Iter('abc').wrap('()').concat()
            '(abc)'
            >>> Iter('abc').wrap('[]').concat()
            '[abc]'
            >>> Iter('abc').wrap(['(((', ')))']).concat()
            '(((abc)))'
            >>> Iter(range(5)).wrap([123, 123]).collect()
            [123, 0, 1, 2, 3, 4, 123]
            >>> Iter(range(20, 25)).wrap([range(3), range(3)]).collect()
            [0, 1, 2, 20, 21, 22, 23, 24, 0, 1, 2]

        """
        if len(ends) != 2:
            raise ValueError("The ends must be a 2-length sequence")

        start, end = ends
        if not isinstance(start, Iterable):
            start = [start]
        if not isinstance(end, Iterable):
            end = [end]

        return Iter(itertools.chain.from_iterable([start, self, end]))

    def print(self, template="{i}: {v}") -> "Iter[T]":
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
        def values():
            while True:
                value = q.get(timeout=timeout)
                if value == sentinel:
                    return
                yield value

        return cls(values())

    def into_queue(self, q: queue.Queue) -> "Iter[T]":
        """
        This is a sink, like Iter.collect_, that consumes data from
        an iterator chain and puts the data into the given queue.

        .. code-block:: python

            >>> q = queue.Queue()
            >>> # This demonstrates the queue sink
            >>> range(5).into_queue(q).consume()
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
        return self.map(q.put)

    def send(self, collector: Generator, close_collector_when_done=False) -> "None":
        """
        |sink|
        See also: `more_itertools.consumer <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.consumer>`_

        Send data into a generator. You do not have to first call ``next()``
        on the generator. Iter.send_ will do this for you.

        |warning| Look carefully at the examples below; you'll see that the
        ``yield`` keyword is wrapped in a second set of parens, e.g.
        ``output.append((yield))``. This is required!

        Simple case:

        .. code-block:: python

            >>> output = []
            >>> def collector():
            ...     while True:
            ...         output.append((yield))
            >>> Iter.range(3).send(collector())
            >>> output
            [0, 1, 2]

        Note that the generator is **not** closed by default after the iterable is
        exhausted. But this can be changed. If you choose to close the
        generator, use the parameter:

        .. code-block:: python

            >>> output = []
            >>> def collector():
            ...     while True:
            ...         output.append((yield))
            >>> g = collector()
            >>> Iter.range(3).send(g, close_collector_when_done=True)
            >>> Iter.range(3).send(g)
            Traceback (most recent call last):
                ...
            StopIteration

        The default behaviour is that the generator is left open which means you
        can keep using it for other iterators:

        .. code-block:: python

            >>> output = []
            >>> def collector():
            ...     while True:
            ...         output.append((yield))
            >>> g = collector()
            >>> Iter.range(3).send(g)
            >>> Iter.range(10, 13).send(g)
            >>> Iter.range(100, 103).send(g)
            >>> output
            [0, 1, 2, 10, 11, 12, 100, 101, 102]


        If the generator is closed before the iteration is complete,
        you'll get a ``StopIteration`` exception:

        .. code-block:: python

            >>> output = []
            >>> def collector():
            ...   for i in range(3):
            ...       output.append((yield))
            >>> Iter.range(5).send(collector())
            Traceback (most recent call last):
                ...
            StopIteration

        Note that Iter.send_ is a sink, so no further chaining is allowed.

        """
        if inspect.getgeneratorstate(collector) == "GEN_CREATED":
            next(collector)

        for v in self:
            collector.send(v)

        if close_collector_when_done:
            collector.close()

    def send_also(self, collector: Generator) -> "Iter":
        """
        Reference: `more_itertools.consumer <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.consumer>`_

        Some ideas around a reverse iterator as a sink. Usually you have
        first to "send" a ``None`` into a generator if you want to send
        more values into it (or call ``next()`` on it), but we handle
        that automatically.

        Simple case:

        .. code-block:: python

            >>> output = []
            >>> def collector():
            ...     while True:
            ...         output.append((yield))
            >>> Iter.range(3).send_also(collector()).collect()
            [0, 1, 2]
            >>> output
            [0, 1, 2]

        However, if the caller already started the generator, that
        works too:

        .. code-block:: python

            >>> output = []
            >>> def collector():
            ...     while True:
            ...         output.append((yield))
            >>> g = collector()
            >>> next(g)  # This "starts" the generator
            >>> Iter.range(3).send_also(g).collect()
            [0, 1, 2]
            >>> output
            [0, 1, 2]

        If the generator is closed before the iteration is complete,
        you'll get an exception (Python 3.7+):

        .. code-block:: python

            >>> output = []
            >>> def collector():
            ...   for i in builtins.range(3):
            ...       output.append((yield))
            >>> Iter.range(50).send_also(collector()).collect()  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            RuntimeError

        Note that the above doesn't happen in Python < 3.7 (which includes
        pypy 7.3.1 that matches Python 3.6.9 compatibility). Instead, you
        collect out the items up to until the point that the collector
        returns; in this case, you'd get [0, 1, 2]. This change was made
        as part of `PEP 479 <https://www.python.org/dev/peps/pep-0479/>`_.

        Regardless, for any Python it's recommended that your generator
        live at least as long as the iterator feeding it.

        """
        if inspect.getgeneratorstate(collector) == "GEN_CREATED":
            next(collector)

        def func(v):
            collector.send(v)

        return self.side_effect(func)

    def sorted(self, key=None, reverse=False) -> "Iter[T]":
        """
        |sink|

        Simple wrapper for the ``sorted`` builtin.

        |warning|
        Calling this will read the entire stream before producing
        results.

        .. code-block:: python

            >>> Iter('bac').sorted().collect()
            ['a', 'b', 'c']
            >>> Iter('bac').sorted(reverse=True).collect()
            ['c', 'b', 'a']
            >>> Iter('bac').zip([2, 1, 0]).sorted(key=lambda tup: tup[1]).collect()
            [('c', 0), ('a', 1), ('b', 2)]

        """
        return Iter(sorted(self, key=key, reverse=reverse))

    def reversed(self) -> "Iter[T]":
        """
        |sink|

        Simple wrapper for the ``reversed`` builtin.

        |warning|
        Calling this will read the entire stream before producing
        results.

        .. code-block:: python

            >>> Iter(range(4)).reversed().collect()
            [3, 2, 1, 0]

        """
        return Iter(reversed(self.collect()))


class IterSeekable(Iter[T]):
    """ See the docstring for Iter.seekable_ """

    def __init__(self, x: more_itertools.seekable):
        self.x: more_itertools.seekable = x

    def seek(self, index: int) -> "Iter[T]":
        """ Seek forwards or backwards to a position in the iterator."""
        self.x.seek(index)
        return self

    def relative_seek(self, offset: int) -> "Iter[T]":
        """ Seek forwards or backwards relative to the current position."""
        self.x.relative_seek(offset)
        return self

    def peek(self, default=more_itertools.recipes._marker) -> "T":
        """ Peek ahead n items in the iterator."""
        return self.x.peek(default=default)

    def bool(self) -> bool:
        """ Return True if the iterator has more items."""
        return bool(self.x)


"""

Experiments and Provisional Ideas
#################################

"""


class IterDict(UserDict):
    """
    |flux|

    The idea here was to make a custom dict where several of
    the standard dict methods return ``Iter`` instances, which can then
    be chained. I'm not sure if this will be kept yet.
    """

    def __iter__(self) -> "Iter":
        return Iter(self.data.keys())

    def keys(self) -> "Iter":
        return Iter(self.data.keys())

    def values(self) -> "Iter":
        return Iter(self.data.values())

    def items(self) -> "Iter":
        return Iter(self.data.items())

    def update(self, *args, **kwargs) -> "IterDict":
        self.data.update(*args, **kwargs)
        return self


def insert_separator(iterable: Iterable[Any], glue: Any) -> "Iterable[Any]":
    """Similar functionality can be obtained with, e.g.,
    interleave, as in

    .. code-block:: python

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

    try:
        # Empty iterator
        yield next(iterable)
    except StopIteration:
        return

    for item in iterable:
        if glue is not None:
            yield glue
        yield item


async def _sync_to_async(iterable):
    """Adapt any synchronous iterable into an async iterator.

    This is the seam between the synchronous and asynchronous worlds:
    it lets the async machinery pull from an ordinary ``Iter`` (or list,
    generator, ...) using ``async for`` without the caller having to know
    which side of the boundary the source came from.
    """
    for item in iterable:
        yield item


def _as_async_iter(source):
    """Return an async *iterator* for ``source``, sync or async.

    Callers can then uniformly drive ``source`` with ``__anext__``
    regardless of whether it was a synchronous ``Iter`` upstream of the
    boundary or an ``AIter`` produced by an earlier async step.
    """
    if hasattr(source, "__aiter__"):
        return source.__aiter__()
    return _sync_to_async(source)


async def _awaited(result):
    """Await ``result`` if it is awaitable, otherwise return it as-is.

    The single primitive that lets every async combinator accept a plain
    value or a coroutine interchangeably -- the colour of a callable no
    longer matters to the pipeline, only whether its result needs
    awaiting.
    """
    if inspect.isawaitable(result):
        return await result
    return result


async def _maybe_await(func, item):
    """Call ``func(item)`` and await the result if it is awaitable."""
    return await _awaited(func(item))


async def _cancel_all(tasks):
    """Cancel and drain any still-outstanding tasks.

    Called from the ``finally`` of the concurrency engine so that whenever
    the pipeline stops early -- a scheduled task raised, or a downstream
    combinator such as ``take`` closed the generator before the source was
    drained -- the in-flight tasks are cancelled and awaited rather than
    orphaned. Without this the event loop emits "Task was destroyed but it
    is pending" warnings and the abandoned work keeps running. Draining
    with ``return_exceptions=True`` also retrieves any sibling exceptions,
    suppressing "exception was never retrieved" noise.
    """
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _abounded(source, call, concurrency, ordered):
    """Drive ``call`` over an async ``source`` with bounded concurrency.

    ``source`` is an async iterator; ``call`` is a coroutine function
    taking one item. At most ``concurrency`` calls are kept in flight.
    When ``ordered`` is true, results are yielded in the order their
    inputs arrived (a sliding window whose left edge is awaited); when
    false, they are yielded as they complete.

    If a scheduled call raises, the exception propagates from the awaiting
    ``yield`` point and the remaining in-flight tasks are cancelled on the
    way out (see _cancel_all). For ``ordered`` the surfaced exception is
    the first in input order; for unordered it is the first to complete.

    This is the single place the sync->async concurrency machinery
    lives, shared by amap_, astarmap_ and afilter_.
    """
    exhausted = [False]

    async def pump(add, size):
        # Pull upstream and schedule tasks until the window is full or the
        # source is exhausted. Upstream is pulled one item at a time
        # (cheap); the expensive ``call`` work is what runs concurrently.
        while not exhausted[0] and size() < concurrency:
            try:
                item = await source.__anext__()
            except StopAsyncIteration:
                exhausted[0] = True
                return
            add(asyncio.ensure_future(call(item)))

    if ordered:
        window = deque()
        try:
            await pump(window.append, lambda: len(window))
            while window:
                result = await window.popleft()
                yield result
                await pump(window.append, lambda: len(window))
        finally:
            await _cancel_all(window)
    else:
        pending = set()
        try:
            await pump(pending.add, lambda: len(pending))
            while pending:
                done, remaining = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                pending.clear()
                pending.update(remaining)
                for task in done:
                    yield task.result()
                await pump(pending.add, lambda: len(pending))
        finally:
            await _cancel_all(pending)


def _islice_bounds(args):
    """Normalise ``islice``-style positional args into (start, stop, step).

    Mirrors the argument handling of ``itertools.islice`` so the async
    ``AIter.islice`` accepts the same call shapes.
    """
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    else:
        start = args[0] or 0
        stop = args[1]
        step = args[2] if len(args) > 2 and args[2] is not None else 1
    if step < 1:
        raise ValueError("Step for islice() must be a positive integer or None.")
    return start, stop, step


def _sync_sink_error(sync_name, async_name):
    """Build the error raised when a synchronous sink is used on an AIter.

    Kept in one place so every guard gives the same guidance: name the
    synchronous sink that cannot run, and point at its async twin.
    """
    return TypeError(
        f"{sync_name}() is a synchronous sink and cannot drive an "
        f"asynchronous pipeline (this chain is an AIter). Use "
        f"`await ...{async_name}()` instead."
    )


class AIter(Iter[T]):
    """
    |flux|

    The asynchronous sibling of Iter_. You never construct one directly
    in normal use; instead a chain *becomes* an AIter the moment it
    passes through an async-introducing method such as amap_. From that
    point on the ``type(self)`` propagation used throughout Iter_ keeps
    every subsequent method returning an AIter, so the whole tail of the
    chain is asynchronous.

    An AIter wraps an *async* iterator (``__aiter__``/``__anext__``)
    rather than a synchronous one. Its constructor normalises whatever it
    is given -- an async iterator, an async iterable, or even a plain
    synchronous iterable -- into a single async iterator, so the rest of
    the class can be written uniformly against ``async for``.

    The transition is one-way: because you cannot pull from an async
    iterator without ``await``, there is no way back to a synchronous
    Iter_. Synchronous sinks such as collect_ therefore raise, and you
    must finish the chain with an async sink such as acollect_.
    """

    def __init__(self, x):
        if hasattr(x, "__anext__"):
            self.x = x
        elif hasattr(x, "__aiter__"):
            self.x = x.__aiter__()
        else:
            self.x = _sync_to_async(x)

    def __aiter__(self):
        return self.x

    async def __anext__(self):
        return await self.x.__anext__()

    def __iter__(self):
        raise TypeError(
            "AIter is asynchronous and cannot be driven by a synchronous "
            "`for` loop. Use `async for`, or an async sink such as "
            "`await ...acollect()`."
        )

    # -- Transformations (lazy; each returns a new AIter) ---------------
    #
    # These re-implement the corresponding Iter_ combinators with an
    # ``async for`` body, because a synchronous ``for`` cannot drive an
    # async iterator. Callables passed in (predicates, mappers) may be
    # either ordinary functions or coroutine functions; their results are
    # awaited only if awaitable (see _awaited).

    def map(self, func: Callable[[T], Any], *iterables: Iterable[Any]) -> "AIter":
        """Asynchronous map. ``func`` may be sync or a coroutine function.

        Unlike amap_ this runs one item at a time (no concurrency); it is
        the plain per-item transform that a chain already in the async
        world uses. Use amap_ when you want concurrent calls.
        """
        if iterables:
            raise NotImplementedError(
                "AIter.map over multiple iterables is not yet supported."
            )

        async def agen():
            async for v in self:
                yield await _awaited(func(v))

        return type(self)(agen())

    def filter(self, function: "Optional[Callable[[T], Any]]" = None) -> "AIter":
        """Asynchronous filter. ``function`` may be sync or a coroutine function."""

        async def agen():
            async for v in self:
                keep = bool(v) if function is None else await _awaited(function(v))
                if keep:
                    yield v

        return type(self)(agen())

    def enumerate(self, start: int = 0) -> "AIter":
        """Asynchronous enumerate_."""

        async def agen():
            i = start
            async for v in self:
                yield (i, v)
                i += 1

        return type(self)(agen())

    def starmap(self, func: Callable[..., Any]) -> "AIter":
        """Asynchronous starmap_ (one item at a time). See astarmap_ for a
        concurrent version."""

        async def agen():
            async for args in self:
                yield await _awaited(func(*args))

        return type(self)(agen())

    def takewhile(self, pred) -> "AIter":
        """Asynchronous takewhile_. ``pred`` may be sync or async."""

        async def agen():
            async for v in self:
                if not await _awaited(pred(v)):
                    break
                yield v

        return type(self)(agen())

    def dropwhile(self, pred) -> "AIter":
        """Asynchronous dropwhile_. ``pred`` may be sync or async."""

        async def agen():
            dropping = True
            async for v in self:
                if dropping:
                    if await _awaited(pred(v)):
                        continue
                    dropping = False
                yield v

        return type(self)(agen())

    def filterfalse(self, pred) -> "AIter":
        """Asynchronous filterfalse_. ``pred`` may be sync or async."""

        async def agen():
            async for v in self:
                keep = (not v) if pred is None else (not await _awaited(pred(v)))
                if keep:
                    yield v

        return type(self)(agen())

    def islice(self, *args) -> "AIter":
        """Asynchronous islice_. Accepts the same argument shapes as the
        stdlib ``itertools.islice``."""
        start, stop, step = _islice_bounds(args)

        async def agen():
            idx = 0
            target = start
            async for v in self:
                if stop is not None and idx >= stop:
                    break
                if idx == target:
                    yield v
                    target += step
                idx += 1

        return type(self)(agen())

    def take(self, n: int) -> "AIter":
        """Asynchronous take_: the first ``n`` items."""
        return self.islice(n)

    def tail(self, n: int) -> "AIter":
        """Asynchronous tail_: the last ``n`` items.

        Necessarily buffers up to ``n`` items, since the last items are
        only known once the source is exhausted.
        """

        async def agen():
            buf = deque(maxlen=n)
            async for v in self:
                buf.append(v)
            for v in buf:
                yield v

        return type(self)(agen())

    def chunked(self, n: int, strict: bool = False) -> "AIter":
        """Asynchronous chunked_: batch items into lists of length ``n``.

        Especially useful in async pipelines for amortising per-item cost
        at a sink -- e.g. batching results before a single offloaded write.
        """

        async def agen():
            batch = []
            async for v in self:
                batch.append(v)
                if len(batch) == n:
                    yield batch
                    batch = []
            if batch:
                if strict and len(batch) != n:
                    raise ValueError("iterable is not divisible by n.")
                yield batch

        return type(self)(agen())

    def flatten(self) -> "AIter":
        """Asynchronous flatten_ (one level). Sub-iterables may themselves
        be synchronous or asynchronous."""

        async def agen():
            async for sub in self:
                if hasattr(sub, "__aiter__"):
                    async for v in sub:
                        yield v
                else:
                    for v in sub:
                        yield v

        return type(self)(agen())

    def chain(self, *iterables: Iterable[Any]) -> "AIter":
        """Asynchronous chain_: append further iterables (sync or async)
        after this one."""

        async def agen():
            async for v in self:
                yield v
            for it in iterables:
                if hasattr(it, "__aiter__"):
                    async for v in it:
                        yield v
                else:
                    for v in it:
                        yield v

        return type(self)(agen())

    def prepend(self, value) -> "AIter":
        """Asynchronous prepend_: yield ``value`` before the rest."""

        async def agen():
            yield value
            async for v in self:
                yield v

        return type(self)(agen())

    def accumulate(self, func=None, *, initial=None) -> "AIter":
        """Asynchronous accumulate_. ``func`` may be sync or async."""
        op = func if func is not None else operator.add

        async def agen():
            if initial is not None:
                total = initial
                yield total
                async for v in self:
                    total = await _awaited(op(total, v))
                    yield total
            else:
                have = False
                total = None
                async for v in self:
                    total = v if not have else await _awaited(op(total, v))
                    have = True
                    yield total

        return type(self)(agen())

    # -- Materialising bridge -------------------------------------------
    #
    # Many Iter_ combinators are thin wrappers over a synchronous
    # more_itertools generator: ``type(self)(more_itertools.foo(self.x,
    # ...))``. Such a generator cannot be driven by an async source, and
    # some (windowing, sorting, combinatorics) inherently need the whole
    # input before they can yield anyway. For those we buffer the async
    # source into a list, run the *identical* synchronous function on it
    # -- guaranteeing exact semantic parity with the sync method -- and
    # lift the result back into the async world. The cost is
    # materialisation: unlike the streaming combinators these do not
    # support infinite async sources, and any callables they take must be
    # ordinary (synchronous) functions.

    def _abuffer_then(self, fn: Callable[[list], Iterable[Any]]) -> "AIter":
        """Buffer the async source, apply the synchronous ``fn`` to the
        list, and re-wrap its (sync-iterable) result as an AIter."""

        async def agen():
            buf = [x async for x in self]
            for v in fn(buf):
                yield v

        return type(self)(agen())

    # -- Streaming combinators (lazy; re-implemented with ``async for``) --

    def starfilter(self, function=None) -> "AIter":
        """Asynchronous starfilter_: like filter_ but tuple args are
        star-unpacked into ``function`` (which may be sync or async)."""

        async def agen():
            async for v in self:
                if await _awaited(function(*v)):
                    yield v

        return type(self)(agen())

    def filter_except(self, validator, *exceptions) -> "AIter":
        """Asynchronous filter_except_: keep items for which ``validator``
        does not raise one of ``exceptions``. ``validator`` may be async."""

        async def agen():
            async for v in self:
                try:
                    await _awaited(validator(v))
                except exceptions:
                    continue
                yield v

        return type(self)(agen())

    def map_except(self, function, *exceptions) -> "AIter":
        """Asynchronous map_except_: apply ``function`` (sync or async),
        dropping items for which it raises one of ``exceptions``."""

        async def agen():
            async for v in self:
                try:
                    result = await _awaited(function(v))
                except exceptions:
                    continue
                yield result

        return type(self)(agen())

    def compress(self, selectors) -> "AIter":
        """Asynchronous compress_: yield items where the paired selector is
        truthy. ``selectors`` is a synchronous iterable; iteration stops
        when either the source or the selectors are exhausted."""

        async def agen():
            sel = iter(selectors)
            async for v in self:
                try:
                    keep = next(sel)
                except StopIteration:
                    break
                if keep:
                    yield v

        return type(self)(agen())

    def chain_from_iterable(self) -> "AIter":
        """Asynchronous chain_from_iterable_: flatten one level of a
        stream of iterables. Identical to flatten_ for an async chain."""
        return self.flatten()

    def intersperse(self, e, n: int = 1) -> "AIter":
        """Asynchronous intersperse_: insert ``e`` between every ``n``
        items. For ``n > 1`` the source is buffered (see _abuffer_then)."""
        if n <= 0:
            raise ValueError("n must be > 0")
        if n != 1:
            return self._abuffer_then(
                lambda buf: more_itertools.intersperse(e, buf, n=n)
            )

        async def agen():
            first = True
            async for v in self:
                if not first:
                    yield e
                first = False
                yield v

        return type(self)(agen())

    def pairwise(self) -> "AIter":
        """Asynchronous pairwise_: yield overlapping ``(prev, cur)`` pairs."""

        async def agen():
            prev = _marker
            async for v in self:
                if prev is not _marker:
                    yield (prev, v)
                prev = v

        return type(self)(agen())

    def unique_everseen(self, key=None) -> "AIter":
        """Asynchronous unique_everseen_: drop items seen before (globally).
        ``key`` may be sync or async. Unhashable keys fall back to a list."""

        async def agen():
            seen_set = set()
            seen_list = []
            async for v in self:
                k = await _awaited(key(v)) if key is not None else v
                try:
                    if k not in seen_set:
                        seen_set.add(k)
                        yield v
                except TypeError:
                    if k not in seen_list:
                        seen_list.append(k)
                        yield v

        return type(self)(agen())

    def unique_justseen(self, key=None) -> "AIter":
        """Asynchronous unique_justseen_: drop *consecutive* duplicates.
        ``key`` may be sync or async."""

        async def agen():
            prev_key = _marker
            async for v in self:
                k = await _awaited(key(v)) if key is not None else v
                if k != prev_key:
                    prev_key = k
                    yield v

        return type(self)(agen())

    def side_effect(
        self, func, *args, chunk_size=None, before=None, after=None
    ) -> "AIter":
        """Asynchronous side_effect_: invoke ``func`` (sync or async) on
        each item -- or each chunk of ``chunk_size`` items -- purely for
        its effect, yielding the items unchanged. ``before``/``after`` run
        once around the stream; ``after`` runs even if the stream errors."""
        f = lambda *value: func(*value, *args)

        async def agen():
            if before is not None:
                await _awaited(before())
            try:
                if chunk_size is None:
                    async for v in self:
                        await _awaited(f(v))
                        yield v
                else:
                    chunk = []
                    async for v in self:
                        chunk.append(v)
                        if len(chunk) == chunk_size:
                            await _awaited(f(chunk))
                            for x in chunk:
                                yield x
                            chunk = []
                    if chunk:
                        await _awaited(f(chunk))
                        for x in chunk:
                            yield x
            finally:
                if after is not None:
                    await _awaited(after())

        return type(self)(agen())

    def padnone(self) -> "AIter":
        """Asynchronous padnone_: yield the items, then ``None`` forever."""

        async def agen():
            async for v in self:
                yield v
            while True:
                yield None

        return type(self)(agen())

    def repeat_last(self, default=None) -> "AIter":
        """Asynchronous repeat_last_: after the source is exhausted, repeat
        its final item forever (or ``default`` if the source was empty)."""

        async def agen():
            last = _marker
            async for v in self:
                last = v
                yield v
            fill = default if last is _marker else last
            while True:
                yield fill

        return type(self)(agen())

    def cycle(self) -> "AIter":
        """Asynchronous cycle_: yield the items, then repeat them forever.
        Buffers items as they are first seen."""

        async def agen():
            saved = []
            async for v in self:
                saved.append(v)
                yield v
            if not saved:
                return
            while True:
                for v in saved:
                    yield v

        return type(self)(agen())

    def ncycles(self, n: int) -> "AIter":
        """Asynchronous ncycles_: yield the whole source ``n`` times.
        Buffers the source."""

        async def agen():
            saved = [x async for x in self]
            for _ in builtins.range(n):
                for v in saved:
                    yield v

        return type(self)(agen())

    def count_cycle(self, n=None) -> "AIter":
        """Asynchronous count_cycle_: cycle the source, pairing each item
        with its 0-based cycle number. Buffers the source."""

        async def agen():
            saved = [x async for x in self]
            if not saved:
                return
            i = 0
            while n is None or i < n:
                for v in saved:
                    yield (i, v)
                i += 1

        return type(self)(agen())

    def lstrip(self, pred) -> "AIter":
        """Asynchronous lstrip_: drop leading items while ``pred`` (sync or
        async) is truthy."""

        async def agen():
            stripping = True
            async for v in self:
                if stripping and await _awaited(pred(v)):
                    continue
                stripping = False
                yield v

        return type(self)(agen())

    def rstrip(self, pred) -> "AIter":
        """Asynchronous rstrip_: drop trailing items for which ``pred``
        (sync or async) is truthy."""

        async def agen():
            cache = []
            async for v in self:
                if await _awaited(pred(v)):
                    cache.append(v)
                else:
                    for c in cache:
                        yield c
                    cache.clear()
                    yield v

        return type(self)(agen())

    def strip(self, pred) -> "AIter":
        """Asynchronous strip_: drop both leading and trailing items for
        which ``pred`` is truthy."""
        return self.lstrip(pred).rstrip(pred)

    def run_length_encode(self) -> "AIter":
        """Asynchronous run_length_encode_: collapse runs of equal items
        into ``(item, count)`` pairs."""

        async def agen():
            cur = _marker
            count = 0
            async for v in self:
                if cur is _marker:
                    cur, count = v, 1
                elif v == cur:
                    count += 1
                else:
                    yield (cur, count)
                    cur, count = v, 1
            if cur is not _marker:
                yield (cur, count)

        return type(self)(agen())

    def run_length_decode(self) -> "AIter":
        """Asynchronous run_length_decode_: expand ``(item, count)`` pairs
        back into a flat run of items."""

        async def agen():
            async for item, count in self:
                for _ in builtins.range(count):
                    yield item

        return type(self)(agen())

    def consecutive_groups(self, ordering=lambda x: x) -> "AIter":
        """Asynchronous consecutive_groups_: group runs of consecutive
        values (by ``ordering``). Each group is yielded as a list."""

        async def agen():
            group = []
            prev = _marker
            async for v in self:
                o = ordering(v)
                if prev is _marker or o == prev + 1:
                    group.append(v)
                else:
                    yield group
                    group = [v]
                prev = o
            if group:
                yield group

        return type(self)(agen())

    def locate(self, pred=bool, window_size=None) -> "AIter":
        """Asynchronous locate_: yield the indices of items for which
        ``pred`` is truthy. With ``window_size`` the source is buffered and
        ``pred`` must be synchronous; otherwise it streams and ``pred`` may
        be async."""
        if window_size is not None:
            return self._abuffer_then(
                lambda buf: more_itertools.locate(
                    buf, pred=pred, window_size=window_size
                )
            )

        async def agen():
            i = 0
            async for v in self:
                if await _awaited(pred(v)):
                    yield i
                i += 1

        return type(self)(agen())

    def difference(self, func=operator.sub, *, initial=None) -> "AIter":
        """Asynchronous difference_: inverse of accumulate_ -- yield the
        first item, then ``func(cur, prev)`` for each subsequent pair.
        ``func`` may be sync or async."""

        async def agen():
            prev = _marker
            first = True
            async for v in self:
                if first and initial is None:
                    yield v
                elif prev is _marker:
                    yield v
                else:
                    yield await _awaited(func(v, prev))
                prev = v
                first = False

        return type(self)(agen())

    def grouper(self, n: int, fillvalue=None, incomplete="fill") -> "AIter":
        """Asynchronous grouper_: batch into length-``n`` tuples. The final
        short group is padded (``fill``), dropped (``ignore``) or raises
        (``strict``), matching more_itertools."""

        async def agen():
            batch = []
            async for v in self:
                batch.append(v)
                if len(batch) == n:
                    yield tuple(batch)
                    batch = []
            if batch:
                if incomplete == "fill":
                    yield tuple(batch) + (fillvalue,) * (n - len(batch))
                elif incomplete == "strict":
                    raise ValueError("iterable is not divisible by n.")
                elif incomplete == "ignore":
                    return
                else:
                    raise ValueError("Expected fill, strict, or ignore")

        return type(self)(agen())

    def ichunked(self, n: int) -> "AIter":
        """Asynchronous ichunked_: batch into consecutive chunks of length
        ``n``, each chunk itself an AIter. Chunks are materialised eagerly
        (an async chunk cannot be lazily re-entered after the next is
        pulled), so this is chunked_ with AIter-wrapped groups."""
        cls = type(self)

        async def agen():
            batch = []
            async for v in self:
                batch.append(v)
                if len(batch) == n:
                    yield cls(batch)
                    batch = []
            if batch:
                yield cls(batch)

        return cls(agen())

    def groupby(self, key=None) -> "AIter":
        """Asynchronous groupby_: group *consecutive* items sharing a key
        into ``(key, group)`` pairs, each group an AIter. ``key`` may be
        sync or async. Unlike itertools each group is materialised, so
        groups stay valid after the outer iterator advances."""
        cls = type(self)

        async def agen():
            cur_key = _marker
            group = []
            async for v in self:
                k = await _awaited(key(v)) if key is not None else v
                if cur_key is _marker:
                    cur_key, group = k, [v]
                elif k == cur_key:
                    group.append(v)
                else:
                    yield (cur_key, cls(group))
                    cur_key, group = k, [v]
            if cur_key is not _marker:
                yield (cur_key, cls(group))

        return cls(agen())

    def split_at(self, pred, maxsplit=-1, keep_separator=False) -> "AIter":
        """Asynchronous split_at_: split into lists at items where ``pred``
        (sync or async) is truthy. ``keep_separator`` keeps the matched
        item as its own single-element list."""

        async def agen():
            buf = []
            splits = 0
            async for v in self:
                if maxsplit != splits and await _awaited(pred(v)):
                    yield buf
                    if keep_separator:
                        yield [v]
                    buf = []
                    splits += 1
                else:
                    buf.append(v)
            yield buf

        return type(self)(agen())

    def split_before(self, pred, maxsplit=-1) -> "AIter":
        """Asynchronous split_before_: start a new list before each item
        where ``pred`` (sync or async) is truthy."""

        async def agen():
            buf = []
            splits = 0
            async for v in self:
                if maxsplit != splits and buf and await _awaited(pred(v)):
                    yield buf
                    buf = []
                    splits += 1
                buf.append(v)
            if buf:
                yield buf

        return type(self)(agen())

    def split_after(self, pred, maxsplit=-1) -> "AIter":
        """Asynchronous split_after_: end the current list after each item
        where ``pred`` (sync or async) is truthy."""

        async def agen():
            buf = []
            splits = 0
            async for v in self:
                buf.append(v)
                if maxsplit != splits and await _awaited(pred(v)):
                    yield buf
                    buf = []
                    splits += 1
            if buf:
                yield buf

        return type(self)(agen())

    def split_when(self, pred, maxsplit=-1) -> "AIter":
        """Asynchronous split_when_: split between two adjacent items
        whenever ``pred(prev, cur)`` (sync or async) is truthy."""

        async def agen():
            src = self.__aiter__()
            try:
                cur = await src.__anext__()
            except StopAsyncIteration:
                return
            buf = [cur]
            splits = 0
            while True:
                try:
                    nxt = await src.__anext__()
                except StopAsyncIteration:
                    break
                if maxsplit != splits and await _awaited(pred(cur, nxt)):
                    yield buf
                    buf = []
                    splits += 1
                buf.append(nxt)
                cur = nxt
            yield buf

        return type(self)(agen())

    # -- Buffered combinators (materialise, then delegate) --------------
    #
    # These forward to the identical more_itertools function used by the
    # synchronous Iter_ method, run against the buffered source, so their
    # behaviour matches the sync version exactly. See _abuffer_then.

    def windowed(self, n, fillvalue=None, step=1) -> "AIter":
        """Asynchronous windowed_ (buffered): sliding windows of length ``n``."""
        return self._abuffer_then(
            lambda buf: more_itertools.windowed(buf, n, fillvalue=fillvalue, step=step)
        )

    def substrings(self) -> "AIter":
        """Asynchronous substrings_ (buffered)."""
        return self._abuffer_then(more_itertools.substrings)

    def substrings_indexes(self, reverse=False) -> "AIter":
        """Asynchronous substrings_indexes_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.substrings_indexes(buf, reverse=reverse)
        )

    def stagger(self, offsets=(-1, 0, 1), longest=False, fillvalue=None) -> "AIter":
        """Asynchronous stagger_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.stagger(
                buf, offsets=offsets, longest=longest, fillvalue=fillvalue
            )
        )

    def padded(self, fillvalue=None, n=None, next_multiple=False) -> "AIter":
        """Asynchronous padded_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.padded(
                buf, fillvalue=fillvalue, n=n, next_multiple=next_multiple
            )
        )

    def adjacent(self, pred, distance=1) -> "AIter":
        """Asynchronous adjacent_ (buffered; ``pred`` must be synchronous)."""
        return self._abuffer_then(
            lambda buf: more_itertools.adjacent(pred, buf, distance=distance)
        )

    def collapse(self, base_type=None, levels=None) -> "AIter":
        """Asynchronous collapse_ (buffered): recursively flatten nested
        (synchronous) iterables."""
        return self._abuffer_then(
            lambda buf: more_itertools.collapse(buf, base_type=base_type, levels=levels)
        )

    def constrained_batches(
        self, max_size, max_count=None, get_len=len, strict=True
    ) -> "AIter":
        """Asynchronous constrained_batches_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.constrained_batches(
                buf, max_size, max_count=max_count, get_len=get_len, strict=strict
            )
        )

    def sort_together(
        self, iterables, key_list=(0,), key=None, reverse=False, strict=False
    ) -> "AIter":
        """Asynchronous sort_together_ (buffered): sort this stream together
        with ``iterables`` (synchronous), keyed on the ``key_list`` columns."""
        return self._abuffer_then(
            lambda buf: more_itertools.sort_together(
                [buf, *iterables],
                key_list=key_list,
                key=key,
                reverse=reverse,
                strict=strict,
            )
        )

    def islice_extended(self, *args) -> "AIter":
        """Asynchronous islice_extended_ (buffered): slicing with support
        for negative indices."""
        return self._abuffer_then(
            lambda buf: more_itertools.islice_extended(buf, *args)
        )

    def rlocate(self, pred=bool, window_size=None) -> "AIter":
        """Asynchronous rlocate_ (buffered): indices of matches, from the
        right."""
        return self._abuffer_then(
            lambda buf: more_itertools.rlocate(buf, pred, window_size)
        )

    def replace(self, pred, substitutes, count=None, window_size=1) -> "AIter":
        """Asynchronous replace_ (buffered): replace runs matching ``pred``
        with ``substitutes``."""
        return self._abuffer_then(
            lambda buf: more_itertools.replace(
                buf, pred, substitutes, count=count, window_size=window_size
            )
        )

    def distinct_permutations(self, r=None) -> "AIter":
        """Asynchronous distinct_permutations_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.distinct_permutations(buf, r=r)
        )

    def distinct_combinations(self, r) -> "AIter":
        """Asynchronous distinct_combinations_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.distinct_combinations(buf, r)
        )

    def circular_shifts(self, steps=1) -> "AIter":
        """Asynchronous circular_shifts_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.circular_shifts(buf, steps=steps)
        )

    def partitions(self) -> "AIter":
        """Asynchronous partitions_ (buffered)."""
        return self._abuffer_then(more_itertools.partitions)

    def set_partitions(self, k=None, min_size=None, max_size=None) -> "AIter":
        """Asynchronous set_partitions_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.set_partitions(
                buf, k=k, min_size=min_size, max_size=max_size
            )
        )

    def powerset(self) -> "AIter":
        """Asynchronous powerset_ (buffered)."""
        return self._abuffer_then(more_itertools.powerset)

    def always_reversible(self) -> "AIter":
        """Asynchronous always_reversible_ (buffered)."""
        return self._abuffer_then(more_itertools.always_reversible)

    def sample(self, k=1, weights=None, *, counts=None, strict=False) -> "AIter":
        """Asynchronous sample_ (buffered): reservoir sample of ``k`` items."""
        return self._abuffer_then(
            lambda buf: more_itertools.sample(
                buf, k=k, weights=weights, counts=counts, strict=strict
            )
        )

    def sorted(self, key=None, reverse=False) -> "AIter":
        """Asynchronous sorted_ (buffered)."""
        return self._abuffer_then(
            lambda buf: sorted(buf, key=key, reverse=reverse)
        )

    def reversed(self) -> "AIter":
        """Asynchronous reversed_ (buffered)."""
        return self._abuffer_then(lambda buf: buf[::-1])

    def insert(self, glue) -> "AIter":
        """Asynchronous insert_ (buffered): place ``glue`` between items."""
        return self._abuffer_then(lambda buf: insert_separator(buf, glue))

    # -- Buffered multi-output combinators ------------------------------
    #
    # These fan the source out into several sub-iterators. Because an async
    # sub-iterator cannot be replayed once a sibling has advanced, each sub
    # is materialised into its own AIter.

    def groupby_transform(
        self, keyfunc=None, valuefunc=None, reducefunc=None
    ) -> "AIter":
        """Asynchronous groupby_transform_ (buffered). Without a
        ``reducefunc`` each group is materialised into an AIter; with one,
        the reduced value is yielded as-is."""
        cls = type(self)

        async def agen():
            buf = [x async for x in self]
            for k, g in more_itertools.groupby_transform(
                buf, keyfunc=keyfunc, valuefunc=valuefunc, reducefunc=reducefunc
            ):
                yield (k, g if reducefunc is not None else cls(list(g)))

        return cls(agen())

    def distribute(self, n: int) -> "AIter":
        """Asynchronous distribute_ (buffered): deal items round-robin into
        ``n`` child AIters."""
        cls = type(self)

        async def agen():
            buf = [x async for x in self]
            for sub in more_itertools.distribute(n, buf):
                yield cls(list(sub))

        return cls(agen())

    def divide(self, n: int) -> "AIter":
        """Asynchronous divide_ (buffered): split into ``n`` contiguous
        child AIters."""
        cls = type(self)

        async def agen():
            buf = [x async for x in self]
            for sub in more_itertools.divide(n, buf):
                yield cls(list(sub))

        return cls(agen())

    def unzip(self) -> "AIter":
        """Asynchronous unzip_ (buffered): transpose a stream of tuples into
        a tuple of child AIters."""
        cls = type(self)

        async def agen():
            buf = [x async for x in self]
            for sub in more_itertools.unzip(buf):
                yield cls(list(sub))

        return cls(agen())

    def partition(self, pred) -> "AIter":
        """Asynchronous partition_ (buffered): yields two child AIters --
        items for which ``pred`` (sync or async) is false, then true."""
        cls = type(self)

        async def agen():
            false_, true_ = [], []
            async for v in self:
                (true_ if await _awaited(pred(v)) else false_).append(v)
            yield cls(false_)
            yield cls(true_)

        return cls(agen())

    # -- Multi-iterable combinators -------------------------------------
    #
    # ``self`` is the first iterable; the ``*iterables`` arguments (sync or
    # async) are normalised to async iterators via the AIter constructor.

    def zip(self, *iterables, strict: bool = False) -> "AIter":
        """Asynchronous zip_: pair items from this stream with ``iterables``
        (sync or async), stopping at the shortest. ``strict=True`` raises if
        the inputs are of unequal length."""
        cls = type(self)

        async def agen():
            sources = [self.__aiter__()] + [cls(it).__aiter__() for it in iterables]
            gap = object()
            while True:
                row = []
                for s in sources:
                    try:
                        row.append(await s.__anext__())
                    except StopAsyncIteration:
                        row.append(gap)
                if all(v is gap for v in row):
                    return
                if any(v is gap for v in row):
                    if strict:
                        raise ValueError(
                            "zip() arguments have unequal lengths (strict)"
                        )
                    return
                yield tuple(row)

        return cls(agen())

    def zip_longest(self, *iterables, fillvalue=None) -> "AIter":
        """Asynchronous zip_longest_: pair items from this stream with
        ``iterables`` (sync or async), padding short inputs with
        ``fillvalue`` until the longest is exhausted."""
        cls = type(self)

        async def agen():
            sources = [self.__aiter__()] + [cls(it).__aiter__() for it in iterables]
            while True:
                row = []
                alive = False
                for s in sources:
                    try:
                        row.append(await s.__anext__())
                        alive = True
                    except StopAsyncIteration:
                        row.append(fillvalue)
                if not alive:
                    return
                yield tuple(row)

        return cls(agen())

    def split_into(self, sizes: Iterable[int]) -> "AIter":
        """Asynchronous split_into_: yield lists of the given ``sizes`` from
        the stream. A ``None`` size takes all remaining items; a short
        source ends the last (partial) list early."""
        cls = type(self)

        async def agen():
            src = self.__aiter__()
            for size in sizes:
                if size is None:
                    yield [x async for x in cls(src)]
                    return
                group = []
                for _ in builtins.range(size):
                    try:
                        group.append(await src.__anext__())
                    except StopAsyncIteration:
                        yield group
                        return
                yield group

        return cls(agen())

    def print(self, template="{i}: {v}") -> "AIter":
        """Asynchronous print_: a debugging pass-through that prints each
        item (formatted with ``{i}``/``{v}``) and yields it unchanged."""
        cls = type(self)

        async def agen():
            i = 0
            async for v in self:
                builtins.print(template.format(i=i, v=v))
                yield v
                i += 1

        return cls(agen())

    def wrap(self, ends: "Sequence[T, T]" = "()") -> "AIter":
        """Asynchronous wrap_: bracket the stream with a start and end value
        (each either a scalar or an iterable of items)."""
        if len(ends) != 2:
            raise ValueError("The ends must be a 2-length sequence")

        start, end = ends
        if not isinstance(start, Iterable):
            start = [start]
        if not isinstance(end, Iterable):
            end = [end]

        cls = type(self)

        async def agen():
            for v in start:
                yield v
            async for v in self:
                yield v
            for v in end:
                yield v

        return cls(agen())

    # -- Buffered combinatoric selectors --------------------------------

    def nth_combination(self, r, index) -> "AIter":
        """Asynchronous nth_combination_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.nth_combination(buf, r, index)
        )

    def random_combination(self, r) -> "AIter":
        """Asynchronous random_combination_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.random_combination(buf, r)
        )

    def random_combination_with_replacement(self, r) -> "AIter":
        """Asynchronous random_combination_with_replacement_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.random_combination_with_replacement(buf, r)
        )

    def random_permutation(self, r=None) -> "AIter":
        """Asynchronous random_permutation_ (buffered)."""
        return self._abuffer_then(
            lambda buf: more_itertools.random_permutation(buf, r=r)
        )

    def unique_to_each(self) -> "AIter":
        """Asynchronous unique_to_each_ (buffered): from a stream of
        iterables, keep the items unique to each."""
        return self._abuffer_then(
            lambda buf: more_itertools.unique_to_each(*buf)
        )

    # -- Not (yet) supported on an async chain --------------------------
    #
    # These return stateful helper objects (or rely on wall-clock pulls)
    # that pull synchronously and have no coherent async form yet. They
    # fail loudly rather than silently mis-behaving; collect the chain
    # first (``await ...acollect()``) and use the synchronous Iter_ method.

    def _aunsupported(name):
        def guard(self, *args, **kwargs):
            raise NotImplementedError(
                f"{name}() is not available on an asynchronous chain. Collect "
                f"the chain first with `await ...acollect()` and call "
                f"Iter(...).{name}() synchronously."
            )

        return guard

    peekable = _aunsupported("peekable")
    seekable = _aunsupported("seekable")
    spy = _aunsupported("spy")
    tee = _aunsupported("tee")
    bucket = _aunsupported("bucket")
    time_limited = _aunsupported("time_limited")
    del _aunsupported

    # -- Sinks (asynchronous; each is a coroutine to be awaited) ---------

    async def acollect(self, container=list) -> "List[T]":
        """
        |sink|

        The asynchronous counterpart to collect_. Drives the whole async
        pipeline to completion and gathers the results into ``container``.
        This is a coroutine, so it must be awaited.

        .. code-block:: python

            >>> import asyncio
            >>> async def main():
            ...     return await Iter(range(3)).amap(lambda x: x).acollect()
            >>> asyncio.run(main())
            [0, 1, 2]

        """
        return container([x async for x in self])

    async def aforeach(self, func) -> None:
        """
        |sink|

        Drive the pipeline purely for side effects, calling ``func`` (sync
        or async) on each item and discarding the results. Returns None.
        The async analogue of a plain ``for`` loop over the chain.
        """
        async for v in self:
            await _awaited(func(v))

    async def afirst(self, default=_marker):
        """
        |sink|

        The asynchronous counterpart to first_. Returns the first item,
        or ``default`` if the pipeline is empty; with no default an empty
        pipeline raises ``ValueError``.
        """
        async for v in self:
            return v
        if default is _marker:
            raise ValueError("afirst() called on an empty async iterator")
        return default

    async def asum(self, start=0):
        """|sink| Asynchronous sum_."""
        total = start
        async for v in self:
            total = total + v
        return total

    async def amax(self, *, key=None, default=_marker):
        """|sink| Asynchronous max over the pipeline."""
        best = _marker
        best_key = None
        async for v in self:
            k = key(v) if key is not None else v
            if best is _marker or k > best_key:
                best, best_key = v, k
        if best is _marker:
            if default is _marker:
                raise ValueError("amax() called on an empty async iterator")
            return default
        return best

    async def amin(self, *, key=None, default=_marker):
        """|sink| Asynchronous min over the pipeline."""
        best = _marker
        best_key = None
        async for v in self:
            k = key(v) if key is not None else v
            if best is _marker or k < best_key:
                best, best_key = v, k
        if best is _marker:
            if default is _marker:
                raise ValueError("amin() called on an empty async iterator")
            return default
        return best

    async def areduce(self, func, initial=_marker):
        """|sink| Asynchronous reduce. ``func`` may be sync or async."""
        source = self.__aiter__()
        if initial is _marker:
            try:
                acc = await source.__anext__()
            except StopAsyncIteration:
                raise TypeError(
                    "areduce() of empty async iterator with no initial value"
                ) from None
        else:
            acc = initial
        async for v in self:
            acc = await _awaited(func(acc, v))
        return acc

    async def adict(self) -> "Dict":
        """|sink| Asynchronous dict_: consume (key, value) pairs into a dict."""
        return {k: v async for k, v in self}

    async def acount(self) -> int:
        """|sink| Consume the pipeline and return the number of items."""
        n = 0
        async for _ in self:
            n += 1
        return n

    async def aany(self, pred=None) -> bool:
        """|sink| True if any item is truthy (or satisfies ``pred``).
        Short-circuits."""
        async for v in self:
            if (await _awaited(pred(v))) if pred is not None else v:
                return True
        return False

    async def aall(self, pred=None) -> bool:
        """|sink| True if every item is truthy (or satisfies ``pred``).
        Short-circuits."""
        async for v in self:
            ok = (await _awaited(pred(v))) if pred is not None else v
            if not ok:
                return False
        return True

    async def aconcat(self, glue: AnyStr = "") -> "AnyStr":
        """|sink| Asynchronous concat_: join the items into a string."""
        return glue.join([x async for x in self])

    async def astarreduce(self, function, initializer=0):
        """|sink| Asynchronous starreduce_: like areduce_ but each item is
        star-unpacked into ``function`` (which may be sync or async)."""
        acc = initializer
        async for v in self:
            acc = await _awaited(function(acc, *v))
        return acc

    async def anth(self, n, default=None):
        """|sink| The ``n``-th item (0-indexed), or ``default`` if the
        pipeline has fewer than ``n + 1`` items."""
        i = 0
        async for v in self:
            if i == n:
                return v
            i += 1
        return default

    async def anth_or_last(self, n, default=_marker):
        """|sink| The ``n``-th item, or the last item if the pipeline is
        shorter. Empty with no ``default`` raises ``ValueError``."""
        last = _marker
        i = 0
        async for v in self:
            last = v
            if i == n:
                return v
            i += 1
        if last is not _marker:
            return last
        if default is _marker:
            raise ValueError("anth_or_last() called on an empty async iterator")
        return default

    async def alast(self, default=_marker):
        """|sink| The last item, or ``default``. Empty with no ``default``
        raises ``ValueError``."""
        last = _marker
        async for v in self:
            last = v
        if last is not _marker:
            return last
        if default is _marker:
            raise ValueError("alast() called on an empty async iterator")
        return default

    async def aone(self, too_short=None, too_long=None):
        """|sink| The single item of a one-item pipeline. Raises (or the
        supplied exception) if there are zero or more than one items."""
        src = self.__aiter__()
        try:
            first = await src.__anext__()
        except StopAsyncIteration:
            raise (
                too_short
                or ValueError("too few items in async iterator (expected 1)")
            ) from None
        try:
            await src.__anext__()
        except StopAsyncIteration:
            return first
        raise too_long or ValueError(
            "too many items in async iterator (expected 1)"
        )

    async def aonly(self, default=None, too_long=None):
        """|sink| The single item, ``default`` if empty, or raise (or the
        supplied exception) if there is more than one item."""
        src = self.__aiter__()
        try:
            first = await src.__anext__()
        except StopAsyncIteration:
            return default
        try:
            await src.__anext__()
        except StopAsyncIteration:
            return first
        raise too_long or ValueError(
            "too many items in async iterator (expected 1)"
        )

    async def ailen(self) -> int:
        """|sink| The number of items in the pipeline. Synonym for acount_."""
        return await self.acount()

    async def adotproduct(self, vec2: Iterable):
        """|sink| The dot product of this pipeline with the synchronous
        ``vec2``; stops at the shorter of the two."""
        it = iter(vec2)
        total = 0
        async for v in self:
            try:
                w = next(it)
            except StopIteration:
                break
            total += v * w
        return total

    async def ais_sorted(self, key=None, reverse=False, strict=False) -> bool:
        """|sink| Whether the pipeline is sorted (short-circuits)."""
        prev = _marker
        prev_key = None
        async for v in self:
            k = key(v) if key is not None else v
            if prev is not _marker:
                if reverse:
                    ok = prev_key > k if strict else prev_key >= k
                else:
                    ok = prev_key < k if strict else prev_key <= k
                if not ok:
                    return False
            prev, prev_key = v, k
        return True

    async def aall_unique(self, key=None) -> bool:
        """|sink| Whether every item (or ``key(item)``) is unique."""
        seen_set = set()
        seen_list = []
        async for v in self:
            k = key(v) if key is not None else v
            try:
                if k in seen_set:
                    return False
                seen_set.add(k)
            except TypeError:
                if k in seen_list:
                    return False
                seen_list.append(k)
        return True

    async def aall_equal(self, key=None) -> bool:
        """|sink| Whether all items (or all ``key(item)``) are equal."""
        first = _marker
        first_key = None
        async for v in self:
            k = key(v) if key is not None else v
            if first is _marker:
                first, first_key = v, k
            elif k != first_key:
                return False
        return True

    async def aminmax(self, key=None, default=_marker):
        """|sink| The ``(min, max)`` pair in a single pass. Empty with no
        ``default`` raises ``ValueError``."""
        lo = hi = _marker
        lo_key = hi_key = None
        async for v in self:
            k = key(v) if key is not None else v
            if lo is _marker:
                lo = hi = v
                lo_key = hi_key = k
            else:
                if k < lo_key:
                    lo, lo_key = v, k
                if k > hi_key:
                    hi, hi_key = v, k
        if lo is _marker:
            if default is _marker:
                raise ValueError("aminmax() called on an empty async iterator")
            return default
        return (lo, hi)

    async def aquantify(self, pred=bool) -> int:
        """|sink| The count of items for which ``pred`` (sync or async) is
        truthy."""
        n = 0
        async for v in self:
            if await _awaited(pred(v)):
                n += 1
        return n

    async def afirst_true(self, default=None, pred=None):
        """|sink| The first truthy item (or first satisfying ``pred``), or
        ``default``. Short-circuits."""
        async for v in self:
            if (await _awaited(pred(v))) if pred is not None else v:
                return v
        return default

    async def aexactly_n(self, n, predicate=bool) -> bool:
        """|sink| Whether exactly ``n`` items satisfy ``predicate`` (sync or
        async). Short-circuits after the ``n + 1``-th match."""
        count = 0
        async for v in self:
            if await _awaited(predicate(v)):
                count += 1
                if count > n:
                    return False
        return count == n

    async def amap_reduce(self, keyfunc, valuefunc=None, reducefunc=None) -> "Dict":
        """|sink| Asynchronous map_reduce_: bucket items by ``keyfunc``,
        transform each by ``valuefunc``, then optionally fold each bucket
        with ``reducefunc``. All three callables may be sync or async."""
        ret = collections.defaultdict(list)
        async for v in self:
            k = await _awaited(keyfunc(v))
            value = await _awaited(valuefunc(v)) if valuefunc is not None else v
            ret[k].append(value)
        if reducefunc is not None:
            for k in list(ret):
                ret[k] = await _awaited(reducefunc(ret[k]))
        ret.default_factory = None
        return ret

    async def aconsume(self, n=None):
        """|sink| Advance the pipeline, discarding items. With ``n`` given,
        skip ``n`` items and return ``self`` (so the chain can continue);
        with ``n`` None, exhaust it entirely and return ``None``."""
        if n is None:
            async for _ in self:
                pass
            return None
        src = self.__aiter__()
        for _ in builtins.range(n):
            try:
                await src.__anext__()
            except StopAsyncIteration:
                break
        return self

    # -- I/O sinks -------------------------------------------------------
    #
    # These drain the async chain into a blocking destination (a file
    # stream or a DB cursor). The stream/cursor stays synchronous -- we
    # deliberately do *not* pull in an async-file dependency -- so writes
    # are handed to a worker thread in batches (``batch_size``) to keep the
    # event loop responsive without paying a thread hop per item. A DB
    # cursor whose ``executemany``/``commit`` are themselves coroutines is
    # awaited natively instead (see ``aexecutemany``).

    async def _adrain_writelines(self, stream, batch_size):
        """Consume the chain and feed it to ``stream.writelines`` in
        thread-offloaded batches of ``batch_size`` items. A pending partial
        batch is flushed even if the source errors mid-stream, so every item
        that was consumed is persisted (matching the synchronous sink).
        ``batch`` is cleared *before* each write, so a failing write is never
        retried by the finally-flush."""
        batch = []

        async def flush():
            nonlocal batch
            if batch:
                pending, batch = batch, []
                await asyncio.to_thread(stream.writelines, pending)

        try:
            async for v in self:
                batch.append(v)
                if len(batch) >= batch_size:
                    await flush()
        finally:
            await flush()

    async def awrite_text_to_stream(
        self, stream, insert_newlines=True, flush=True, batch_size=1000
    ):
        """|sink| Asynchronous write_text_to_stream_: write items to an open
        text ``stream``, batching writes onto a worker thread. Set
        ``insert_newlines=False`` if the items already carry newlines."""
        src = self.intersperse("\n") if insert_newlines else self
        await src._adrain_writelines(stream, batch_size)
        if flush:
            await asyncio.to_thread(stream.flush)

    async def awrite_bytes_to_stream(self, stream, flush=True, batch_size=1000):
        """|sink| Asynchronous write_bytes_to_stream_: write items to an open
        binary ``stream``, batching writes onto a worker thread."""
        await self._adrain_writelines(stream, batch_size)
        if flush:
            await asyncio.to_thread(stream.flush)

    async def awrite_file(
        self,
        file,
        mode="w",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
        batch_size=1000,
    ):
        """|sink| Asynchronous write_file_: open ``file``, write every item
        to it (batched onto a worker thread), and close it -- even if the
        chain errors mid-stream. Text or binary is selected by ``mode``."""
        f = await asyncio.to_thread(
            open,
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )
        try:
            await self._adrain_writelines(f, batch_size)
        finally:
            await asyncio.to_thread(f.close)

    async def aexecutemany(
        self,
        cursor: Any,
        sql: str,
        *,
        batch_size: int = 1000,
        commit: Optional[Callable[[], Any]] = None,
        rollback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """|sink| Asynchronous executemany_: run a DB-API ``executemany`` over
        ``batch_size`` chunks of this chain. ``cursor.executemany`` (and the
        optional ``commit``/``rollback``) may be sync or coroutine -- each is
        awaited iff it returns an awaitable, so async drivers (asyncpg,
        psycopg async, aiosqlite) work natively."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        try:
            async for batch in self.chunked(batch_size):
                await _awaited(cursor.executemany(sql, batch))
            if commit is not None:
                await _awaited(commit())
        except Exception:
            if rollback is not None:
                await _awaited(rollback())
            raise

    async def asend(self, collector, close_collector_when_done=False) -> None:
        """|sink| Asynchronous send_: drive an async-generator ``collector``
        (one that does ``(yield)``) by ``asend``-ing each item into it. The
        generator is primed automatically. It is left open unless
        ``close_collector_when_done`` is set."""
        if inspect.getasyncgenstate(collector) == inspect.AGEN_CREATED:
            await collector.asend(None)
        async for v in self:
            await collector.asend(v)
        if close_collector_when_done:
            await collector.aclose()

    # -- Count-validating combinators (return an AIter) ------------------

    def strictly_n(self, n, too_short=None, too_long=None) -> "AIter":
        """Asynchronous strictly_n_ (buffered): yield the items only if
        there are exactly ``n`` of them, else raise."""
        return self._abuffer_then(
            lambda buf: more_itertools.strictly_n(
                buf, n, too_short=too_short, too_long=too_long
            )
        )

    def map_reduce_it(self, keyfunc, valuefunc=None, reducefunc=None) -> "AIter":
        """Asynchronous map_reduce_it_: like amap_reduce_ but streamed as an
        AIter of ``(key, reduced)`` pairs rather than a dict."""
        cls = type(self)

        async def agen():
            d = await self.amap_reduce(keyfunc, valuefunc, reducefunc)
            for item in d.items():
                yield item

        return cls(agen())

    # -- Fan-out combinators (yield items unchanged after a side-effect) -

    def into_queue(self, q) -> "AIter":
        """Asynchronous into_queue_: put each item onto ``q`` and yield it
        unchanged. ``q`` may be a ``queue.Queue`` or an ``asyncio.Queue``
        (its coroutine ``put`` is awaited). Drive it with ``aconsume``."""
        cls = type(self)

        async def agen():
            async for v in self:
                await _awaited(q.put(v))
                yield v

        return cls(agen())

    def send_also(self, collector) -> "AIter":
        """Asynchronous send_also_: ``asend`` each item into an
        async-generator ``collector`` (primed automatically) while yielding
        the items onward unchanged."""

        async def prime():
            if inspect.getasyncgenstate(collector) == inspect.AGEN_CREATED:
                await collector.asend(None)

        async def func(v):
            await collector.asend(v)

        return self.side_effect(func, before=prime)

    # -- Guards: synchronous sinks cannot drive an async pipeline --------

    def collect(self, container=list):
        raise _sync_sink_error("collect", "acollect")

    def sum(self):
        raise _sync_sink_error("sum", "asum")

    def first(self, default=_marker):
        raise _sync_sink_error("first", "afirst")

    def dict(self):
        raise _sync_sink_error("dict", "adict")

    def concat(self, glue: AnyStr = ""):
        raise _sync_sink_error("concat", "aconcat")

    # The remaining synchronous scalar sinks likewise cannot drive an
    # async pipeline; each points at its ``a``-prefixed coroutine twin.
    # (*args/**kwargs so the guard accepts each sink's real signature.)

    def reduce(self, *args, **kwargs):
        raise _sync_sink_error("reduce", "areduce")

    def starreduce(self, *args, **kwargs):
        raise _sync_sink_error("starreduce", "astarreduce")

    def any(self, *args, **kwargs):
        raise _sync_sink_error("any", "aany")

    def all(self, *args, **kwargs):
        raise _sync_sink_error("all", "aall")

    def nth(self, *args, **kwargs):
        raise _sync_sink_error("nth", "anth")

    def nth_or_last(self, *args, **kwargs):
        raise _sync_sink_error("nth_or_last", "anth_or_last")

    def last(self, *args, **kwargs):
        raise _sync_sink_error("last", "alast")

    def one(self, *args, **kwargs):
        raise _sync_sink_error("one", "aone")

    def only(self, *args, **kwargs):
        raise _sync_sink_error("only", "aonly")

    def ilen(self, *args, **kwargs):
        raise _sync_sink_error("ilen", "ailen")

    def dotproduct(self, *args, **kwargs):
        raise _sync_sink_error("dotproduct", "adotproduct")

    def is_sorted(self, *args, **kwargs):
        raise _sync_sink_error("is_sorted", "ais_sorted")

    def all_unique(self, *args, **kwargs):
        raise _sync_sink_error("all_unique", "aall_unique")

    def all_equal(self, *args, **kwargs):
        raise _sync_sink_error("all_equal", "aall_equal")

    def minmax(self, *args, **kwargs):
        raise _sync_sink_error("minmax", "aminmax")

    def quantify(self, *args, **kwargs):
        raise _sync_sink_error("quantify", "aquantify")

    def first_true(self, *args, **kwargs):
        raise _sync_sink_error("first_true", "afirst_true")

    def exactly_n(self, *args, **kwargs):
        raise _sync_sink_error("exactly_n", "aexactly_n")

    def map_reduce(self, *args, **kwargs):
        raise _sync_sink_error("map_reduce", "amap_reduce")

    def consume(self, *args, **kwargs):
        raise _sync_sink_error("consume", "aconsume")

    def next(self, *args, **kwargs):
        raise _sync_sink_error("next", "afirst")

    def write_text_to_stream(self, *args, **kwargs):
        raise _sync_sink_error("write_text_to_stream", "awrite_text_to_stream")

    def write_bytes_to_stream(self, *args, **kwargs):
        raise _sync_sink_error("write_bytes_to_stream", "awrite_bytes_to_stream")

    def write_file(self, *args, **kwargs):
        raise _sync_sink_error("write_file", "awrite_file")

    def executemany(self, *args, **kwargs):
        raise _sync_sink_error("executemany", "aexecutemany")

    def send(self, *args, **kwargs):
        raise _sync_sink_error("send", "asend")


"""

Related projects
################

It turns out the idea of chaining iterators is not new. There are many
libraries that offer similar features:

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

* `https://github.com/dwt/fluent <https://github.com/dwt/fluent>`_

Somewhat related:

* `https://github.com/jreese/aioitertools <https://github.com/jreese/aioitertools>`_


Dev Instructions
################

Setup
*****

This project uses `uv <https://docs.astral.sh/uv/>`_ for dependency
management and packaging, with common local commands captured in the
``justfile``.

.. code-block:: shell

    $ just sync

Testing
*******

.. code-block:: shell

    $ just test
    $ just coverage

Documentation
*************

To regenerate the documentation file ``README.rst``:

.. code-block:: shell

    $ just docs

Releasing
*********

Releases are built and published by GitHub Actions when a ``v*`` tag is
pushed. PyPI publishing uses trusted publishing (OIDC), so there are no PyPI
API tokens in GitHub secrets. See ``RELEASING.md`` for the one-time PyPI and
GitHub environment setup.

.. code-block:: shell

    $ just release         # patch release by default
    $ just release minor   # or major

|
|

-----

|
|

    Work is a necessary evil to be avoided. 
    *Mark Twain*


"""
