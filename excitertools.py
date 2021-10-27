"""

.. image:: https://github.com/cjrh/excitertools/workflows/Python%20application/badge.svg
    :target: https://github.com/cjrh/excitertools/actions

.. image:: https://coveralls.io/repos/github/cjrh/excitertools/badge.svg?branch=master
    :target: https://coveralls.io/github/cjrh/excitertools?branch=master

.. image:: https://img.shields.io/pypi/pyversions/excitertools.svg
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
# This cannot be enabled because we still support 3.6 and pypy
#from __future__ import annotations
import sys
import string
import itertools
import functools
import operator
import inspect
from collections import UserDict, Counter, deque
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
import collections.abc
import queue
import re

import more_itertools

__all__ = [
    "Iter",
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


def zip(*iterables: Any) -> "Iter[Tuple[T, ...]]":
    """ Replacement for the builtin ``zip`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining."""
    return Iter(_zip(*iterables))


def enumerate(iterable) -> "Iter[Tuple[int, T]]":
    """ Replacement for the builtin ``enumerate`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> enumerate(string.ascii_lowercase).take(3).collect()
        [(0, 'a'), (1, 'b'), (2, 'c')]


    """
    return Iter(_enumerate(iterable))


def map(func: Union[Callable[..., C], str], iterable) -> "Iter[C]":
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


def filter(function: "Callable[[Any], bool]", iterable: "Iterable[T]") -> "Iter[T]":
    """ Replacement for the builtin ``filter`` function.  This version returns
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
    """ Replacement for the itertools ``count`` function.  This version returns
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
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> accumulate([1, 2, 3, 4, 5]).collect()
        [1, 3, 6, 10, 15]
        >>> if sys.version_info >= (3, 8):
        ...     output = accumulate([1, 2, 3, 4, 5], initial=100).collect()
        ...     assert output == [100, 101, 103, 106, 110, 115]
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
    """ Replacement for the itertools ``chain`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> chain('ABC', 'DEF').collect()
        ['A', 'B', 'C', 'D', 'E', 'F']
        >>> chain().collect()
        []

    """
    return Iter(itertools.chain(*iterables))


def chain_from_iterable(iterable) -> "Iter[T]":
    """ Replacement for the itertools ``chain.from_iterable`` method.
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
    """ Replacement for the itertools ``compress`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> compress('ABCDEF', [1, 0, 1, 0, 1, 1]).collect()
        ['A', 'C', 'E', 'F']


    """
    return Iter(itertools.compress(data, selectors))


def dropwhile(pred, iterable):
    """ Replacement for the itertools ``dropwhile`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> dropwhile(lambda x: x < 4, range(6)).collect()
        [4, 5]

    """
    return Iter(itertools.dropwhile(pred, iterable))


def filterfalse(pred, iterable):
    """ Replacement for the itertools ``filterfalse`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> filterfalse(None, [2, 0, 3, None, 4, 0]).collect()
        [0, None, 0]

    """
    return Iter(itertools.filterfalse(pred, iterable))


def groupby(iterable, key=None):
    """ Replacement for the itertools ``groupby`` function.  This version returns
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
    """ Replacement for the itertools ``islice`` function.  This version returns
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
    """ Replacement for the itertools ``starmap`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> starmap(pow, [(2, 5), (3, 2), (10, 3)]).collect()
        [32, 9, 1000]

    """
    return Iter(itertools.starmap(func, iterable))


def takewhile(pred, iterable):
    """ Replacement for the itertools ``takewhile`` function.  This version returns
    an instance of Iter_ to allow further iterable chaining.

    .. code-block:: python

        >>> takewhile(lambda x: x < 5, [1, 4, 6, 4, 1]).collect()
        [1, 4]

    """
    return Iter(itertools.takewhile(pred, iterable))


def tee(iterable, n=2):
    """ Replacement for the itertools ``tee`` function.  This version returns
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
    """ Replacement for the itertools ``zip_longest`` function.  This version returns
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
        pat: "re.Pattern[AnyStr]",
        s: AnyStr,
        flags: Union[int, re.RegexFlag] = 0
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
        pat: "re.Pattern[AnyStr]",
        s: AnyStr,
        flags: Union[int, re.RegexFlag] = 0
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
            yield s[prev:m.start()]
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
            int.to_bytes(v, 1, 'little') if isinstance(v, int) else v
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

"""

The ``Iter`` Class
##################

.. contents::
    :backlinks: entry
    :local:

"""

class Iter(Generic[T]):
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
        >>> Iter(mygen).collect()
        Traceback (most recent call last):
            ...
        TypeError: It seems you passed a generator function, but you
        probably intended to pass a generator. Remember to evaluate the
        function to obtain a generator instance:
        <BLANKLINE>
        def mygen():
            yield 123
        <BLANKLINE>
        Iter(mygen)    # ERROR - a generator function object is not iterable
        Iter(mygen())  # CORRECT - a generator instance is iterable.
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
            reduce,
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
                    .map(lambda x: x + 2, inputs)
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

    x: Iterator[T]

    def __init__(self, x: Iterable[T]):
        if isinstance(x, collections.abc.Iterator):
            self.x = x
        else:
            try:
                self.x = iter(x)
            except TypeError as e:
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
        for f in func:
            def inner(self, *args, **kwargs):
                return Iter(f(self, *args, **kwargs))

            setattr(cls, f.__name__, inner)

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

        """
        if container == str:
            return self.concat("")
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
    def read_lines(cls, stream: IO[str], rewind=True):
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

        The ``rewind`` parameter can be used to read sections of a file.

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
    def read_bytes(cls, stream: IO[bytes], size: Union[Callable[[], int], int] = -1, rewind=True):
        """
        |source|

        The ``size`` parameter can be used to control how many bytes are
        read for each advancement of the iterator chain. Here we set ``size=1``
        which means we'll get back one byte at a time.

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

        The ``rewind`` parameter can be used to read sections of a file.

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
            stream.writelines(self.intersperse('\n'))
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

    def write_to_file(
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

        .. code-block:: python

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as td:
            ...     # Put some random text into a temporary file
            ...     with open(td + 'text.txt', 'w') as f:
            ...         f.writelines(['abc\\n', 'def\\n', 'ghi\\n'])
            ...
            ...     # Open the file, transform, write out to new file.
            ...     Iter.open(td + 'text.txt').map(str.upper).write_to_file(td + 'test2.txt')
            ...     # Read the new file, for the test
            ...     Iter.open(td + 'test2.txt').collect()
            ['ABC\\n', 'DEF\\n', 'GHI\\n']

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

    def zip(self, *iterables: Any) -> "Iter[Tuple[T, ...]]":
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
        return Iter(_zip(self, *iterables))

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

    def enumerate(self) -> "Iter[Tuple[int, T]]":
        """

        .. code-block:: python

            >>> Iter('abc').enumerate().collect()
            [(0, 'a'), (1, 'b'), (2, 'c')]
            >>> Iter([]).enumerate().collect()
            []

        """
        return Iter(_enumerate(self))

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

    def map(self, func: Union[Callable[..., C], str]) -> "Iter[C]":
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
            return Iter(_map(lambda x: eval(func), self))
        else:
            return Iter(_map(func, self))

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

    def starfilter(self, function: "Optional[Callable[[T, ...], bool]]" = None) -> "Iter[T]":
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

    def concat(self, glue: AnyStr = '') -> "AnyStr":
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
        """ Docstring TBD """
        return Iter(insert_separator(self, glue))

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

    def cycle(self) -> "Iter[T]":
        """
        |inf|

        .. code-block:: python

            >>> Iter('abc').cycle().take(8).collect()
            ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']
            >>> Iter('abc').cycle().take(8).concat('')
            'abcabcab'

        """
        return Iter(itertools.cycle(self.x))

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
            >>> if sys.version_info >= (3, 8):
            ...     out = Iter([1, 2, 3, 4, 5]).accumulate(initial=100).collect()
            ...     assert out == [100, 101, 103, 106, 110, 115]
            >>> Iter([1, 2, 3, 4, 5]).accumulate(operator.mul).collect()
            [1, 2, 6, 24, 120]

        """
        if sys.version_info < (3, 8):  # pragma: no cover
            if initial:
                raise RuntimeError(
                    f'The "initial" kwarg was added in Python 3.8 and is not'
                    f'available on this version of Python which is '
                    f'{sys.version_info} '
                )
            return Iter(itertools.accumulate(self.x, func))

        return Iter(itertools.accumulate(self.x, func, initial=initial))

    def chain(self, *iterables: Iterable[T]) -> "Iter[T]":
        """ Docstring TBD

        .. code-block:: python

            >>> Iter('ABC').chain('DEF').collect()
            ['A', 'B', 'C', 'D', 'E', 'F']
            >>> Iter('ABC').chain().collect()
            ['A', 'B', 'C']

         """
        return Iter(itertools.chain(self.x, *iterables))

    def chain_from_iterable(self) -> "Iter[T]":
        """ Docstring TBD

        .. code-block:: python

            >>> Iter(['ABC', 'DEF']).chain_from_iterable().collect()
            ['A', 'B', 'C', 'D', 'E', 'F']

         """
        return Iter(itertools.chain.from_iterable(self.x))

    def compress(self, selectors):
        """ Replacement for the itertools ``compress`` function.  This version returns
        an instance of Iter_ to allow further iterable chaining.

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

    def islice(self, *args) -> "Iter":
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

    def chunked(self, n: int) -> "Iter":
        """ Docstring TBD """
        return Iter(more_itertools.chunked(self.x, n))

    def ichunked(self, n: int) -> "Iter":
        """ Docstring TBD """
        return Iter(Iter(it) for it in more_itertools.ichunked(self.x, n))

    @classmethod
    def sliced(cls, seq: Sequence, n: int) -> "Iter":
        """ Docstring TBD """
        return Iter(more_itertools.sliced(seq, n))

    def distribute(self, n: int) -> "Iter":
        """ Docstring TBD """
        return Iter((Iter(x) for x in Iter(more_itertools.distribute(n, self.x))))

    def divide(self, n: int) -> "Iter":
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

        class _bucket(more_itertools.bucket):
            def __iter__(self):
                return Iter(super().__iter__())

            def __getitem__(self, item):
                return Iter(super().__getitem__(item))

        return _bucket(self.x, key, validator=validator)

    def unzip(self):
        """ Docstring TBD """
        return Iter(Iter(x) for x in more_itertools.unzip(self.x))

    def grouper(self, n: int, fillvalue=None) -> "Iter":
        """ Docstring TBD """
        return Iter(more_itertools.grouper(self.x, n, fillvalue=fillvalue))

    def partition(self, pred) -> "Iter":
        """ Docstring TBD """
        left, right = more_itertools.partition(pred, self.x)
        return Iter((Iter(left), Iter(right)))

    # Lookahead and lookback

    def spy(self, n=1) -> "Tuple[Iter, Iter]":
        """ Docstring TBD """
        head, iterable = more_itertools.spy(self.x, n)
        return Iter(head), Iter(iterable)

    def peekable(self) -> "more_itertools.peekable":
        """
        Docstring TBD

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

    def seekable(self) -> "more_itertools.seekable":
        """ Docstring TBD """

        class _seekable(more_itertools.seekable):
            def __iter__(self):
                return Iter(super().__iter__())

        return _seekable(self.x)

    # Windowing

    def windowed(self, n, fillvalue=None, step=1) -> "Iter":
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
        Reference `more_itertools.pairwise <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise>`_

        .. code-block:: python

            >>> Iter.count().pairwise().take(4).collect()
            [(0, 1), (1, 2), (2, 3), (3, 4)]
        """
        return Iter(more_itertools.pairwise(self.x))

    # Augmenting

    def count_cycle(self, n=None) -> "Iter":
        """

        Reference: `more_itertools.count_cycle <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle>`_

        .. code-block:: python

            >>> Iter('AB').count_cycle(3).collect()
            [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]

        """
        return Iter(more_itertools.count_cycle(self.x, n=n))

    def intersperse(self, e, n=1) -> "Iter":
        """
        Reference: `more_itertools.intersperse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse>`_

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
    ) -> "Iter[Union[T, C]]":
        """
        Reference: `more_itertools.padded <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded>`_

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

    def repeat_last(self, default=None) -> "Iter[T]":
        """
        Reference: `more_itertools.repeat_last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeat_last>`_

        .. code-block:: python

            >>> Iter(range(3)).repeat_last().islice(5).collect()
            [0, 1, 2, 2, 2]

            >>> Iter(range(0)).repeat_last(42).islice(5).collect()
            [42, 42, 42, 42, 42]

        """
        return Iter(more_itertools.repeat_last(self.x, default=default))

    def adjacent(self, pred, distance=1) -> "Iter[Tuple[bool, T]]":
        """
        Reference: `more_itertools.adjacent <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent>`_

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
        return Iter(
            more_itertools.groupby_transform(
                self.x, keyfunc=keyfunc, valuefunc=valuefunc,
            )
        )

    def padnone(self) -> "Iter[Union[T, None]]":
        """
        Reference: `more_itertools.padnone <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone>`_

        .. code-block:: python

            >>> Iter(range(3)).padnone().take(5).collect()
            [0, 1, 2, None, None]

        """
        return Iter(more_itertools.padnone(self.x))

    def ncycles(self, n) -> "Iter[T]":
        """
        Reference: `more_itertools.ncycles <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles>`_

        .. code-block:: python

            >>> Iter(['a', 'b']).ncycles(3).collect()
            ['a', 'b', 'a', 'b', 'a', 'b']

        """
        return Iter(more_itertools.ncycles(self.x, n))

    # Combining

    def collapse(self, base_type=None, levels=None) -> "Iter":
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
        return Iter(more_itertools.collapse(self.x, base_type=base_type, levels=levels))

    @class_or_instancemethod
    def sort_together(self_or_cls, iterables, key_list=(0,), reverse=False):
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
    def interleave(self_or_cls, *iterables) -> "Iter":
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
    def interleave_longest(self_or_cls, *iterables) -> "Iter":
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
    def zip_offset(cls, *iterables, offsets, longest=False, fillvalue=None) -> "Iter":
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
                *iterables, offsets=offsets, longest=longest, fillvalue=fillvalue,
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

    def flatten(self) -> "Iter[T]":
        """
        Reference: `more_itertools.flatten <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten>`_

        .. code-block:: python

            >>> Iter([[0, 1], [2, 3]]).flatten().collect()
            [0, 1, 2, 3]

        """
        return Iter(more_itertools.flatten(self))

    @class_or_instancemethod
    def roundrobin(self_or_cls: Union[Type[T], T], *iterables: C) -> "Iter[Union[T, C]]":
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
        return Iter(more_itertools.prepend(value, self))

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

    def unique_to_each(self) -> "Iter[T]":
        """
        Reference: `more_itertools.unique_to_each <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each>`_

        .. code-block:: python

            >>> Iter([{'A', 'B'}, {'B', 'C'}, {'B', 'D'}]).unique_to_each().collect()
            [['A'], ['C'], ['D']]

            >>> Iter(["mississippi", "missouri"]).unique_to_each().collect()
            [['p', 'p'], ['o', 'u', 'r']]
        """
        # TODO: this operation is a little different to the others that take
        #  a *iterables parameter. Not sure if this is what we want.
        return Iter(more_itertools.unique_to_each(*self))

    def sample(self, k=1, weights=None) -> "Iter":
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
        return Iter(more_itertools.sample(self.x, k=k, weights=weights))

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
        d = more_itertools.map_reduce(self.x, keyfunc, valuefunc, reducefunc,)
        return Iter(d.items())

    def exactly_n(self, n, predicate=bool) -> "bool":
        """
        |sink|

        Docstring TBD

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
        """
        Reference: `more_itertools.islice_extended <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.islice_extended>`_

        .. code-block:: python

            >>> Iter('abcdefgh').islice_extended(-4, -1).collect()
            ['e', 'f', 'g']

        .. code-block:: python

            >>> Iter.count().islice_extended( 110, 99, -2).collect()
            [110, 108, 106, 104, 102, 100]

        """
        return Iter(more_itertools.islice_extended(self.x, *args))

    def first(self):
        """
        Reference: `more_itertools.first <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.first>`_
        """
        raise NotImplementedError

    def last(self):
        """
        Reference: `more_itertools.last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.last>`_
        """
        raise NotImplementedError

    def one(self):
        """
        Reference: `more_itertools.one <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.one>`_

        """
        raise NotImplementedError

    def only(self, default=None, too_long=ValueError) -> "T":
        """
        Reference: `more_itertools.one <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.one>`_

        .. code-block:: python

            >>> Iter([]).only(default='missing')
            'missing'
            >>> Iter([42]).only(default='missing')
            42
            >>> Iter([1, 2]).only()
            Traceback (most recent call last):
                ...
            ValueError: ...

        """
        return more_itertools.only(self.x, default=default, too_long=too_long)

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

        Be sure to read the *more-itertools* docs whne using unhashable
        items.

        .. code-block:: python

            >>> iterable = ([1, 2], [2, 3], [1, 2])
            >>> Iter(iterable).unique_everseen().collect()  # Slow
            [[1, 2], [2, 3]]
            >>> Iter(iterable).unique_everseen(key=tuple).collect()  # Faster
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

    def distinct_permutations(self):
        """
        Reference: `more_itertools.distinct_permutations <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distinct_permutations>`_

        .. code-block:: python

            >>> Iter([1, 0, 1]).distinct_permutations().sorted().collect()
            [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

        """
        return Iter(more_itertools.distinct_permutations(self.x))

    def distinct_combinations(self, r) -> "Iter[T]":
        """
        Reference: `more_itertools.distinct_combinations <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distinct_combinations>`_

        .. code-block:: python

            >>> Iter([0, 0, 1]).distinct_combinations(2).collect()
            [(0, 0), (0, 1)]

        """
        return Iter(more_itertools.distinct_combinations(self, r))

    def circular_shifts(self) -> "Iter[T]":
        """
        Reference: `more_itertools.circular_shifts <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.circular_shifts>`_

        .. code-block:: python

            >>> Iter(range(4)).circular_shifts().collect()
            [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]

        """
        return Iter(more_itertools.circular_shifts(self))

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

    def set_partitions(self, k=None) -> "Iter[T]":
        """
        Reference: `more_itertools.set_partitions <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.set_partitions>`_

        .. code-block:: python

            >>> Iter('abc').set_partitions(2).collect()
            [[['a'], ['b', 'c']], [['a', 'b'], ['c']], [['b'], ['a', 'c']]]

        """
        return Iter(more_itertools.set_partitions(self, k=k))

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
            return Iter(more_itertools.random_product(self_or_cls, *args, repeat=repeat))

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
        return Iter(more_itertools.random_combination_with_replacement(
            self, r
        ))

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
    def always_iterable(cls, obj, base_type=(str, bytes)) -> 'Iter':
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
            >>> indexes = Iter(it).locate(pred=pred)
            >>> i = next(indexes)
            >>> it.seek(i)
            >>> next(it)
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

    def iterate(self):
        raise NotImplementedError

    def difference(self, func=operator.sub, *, initial=None):
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
        return Iter(more_itertools.difference(self.x, func=func, initial=initial,))

    def make_decorator(self):
        raise NotImplementedError

    def SequenceView(self):
        raise NotImplementedError

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

    def tabulate(self):
        raise NotImplementedError

    @classmethod
    def repeatfunc(cls, func, *args, times=None):
        """
        |source|
        Docstring TBD

        .. code-block:: python

            >>> Iter.repeatfunc(operator.add, 3, 5, times=4).collect()
            [8, 8, 8, 8]

        """
        return Iter(more_itertools.repeatfunc(func, times, *args))

    # New

    def wrap(self, ends: "Sequence[T, T]" = "()"):
        """ Other examples for ends: '"' * 2, or '`' * 2, or '[]' etc. """
        if len(ends) != 2:
            raise ValueError("The ends must be a 2-length sequence")

        return Iter(itertools.chain.from_iterable([ends[0], self, ends[1]]))

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
        return Iter.repeatfunc(q.get, timeout).takewhile(lambda v: v is not None)

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
        if inspect.getgeneratorstate(collector) == 'GEN_CREATED':
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
        if inspect.getgeneratorstate(collector) == 'GEN_CREATED':
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

    yield next(iterable)
    for item in iterable:
        if glue is not None:
            yield glue
        yield item


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

.. code-block:: shell

    $ python -m venv venv
    $ source venv/bin/activate
    (venv) $ pip install -e .[dev,test]

Testing
*******

.. code-block:: shell

    (venv) $ pytest --cov

Documentation
*************

To regenerate the documentation, file ``README.rst``:

.. code-block:: shell

    (venv) $ python regenerate_readme.py -m excitertools.py > README.rst

Releasing
*********

To do a release, we're using `bumpymcbumpface <https://pypi.org/project/bumpymcbumpface/>`_.
Make sure that is set up correctly according to its own documentation. I 
like to use `pipx <https://github.com/pipxproject/pipx>`_ to install and 
manage these kinds of tools.

.. code-block:: shell

    $ bumpymcbumpface --push-git --push-pypi

|
|

-----

|
|

    Work is a necessary evil to be avoided. 
    *Mark Twain*


"""
