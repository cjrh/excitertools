

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


.. _more-itertools: https://more-itertools.readthedocs.io/en/stable/index.html

.. _excitertools:

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

Several emoji are used to indicate things about parts of the API:

- |source| This API method is a *source*, meaning that it produces data
  that will be processed in an iterator chain.
- |sink| This API method is a *sink*, meaning that it consumes data that
  was processed in an iterator chain.
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

.. contents::
    :local:




-----

The following module-level functions, like range_, zip_ and so on, are 
intended to be used as replacements for their homonymous builtins. The
only difference between these and the builtin versions is that these 
return instances of the Iter_ class. Note that because Iter_ is itself
iterable, it means that the functions here can be used as drop-in 
replacements.



.. _range:


|source| ``range(*args) -> "Iter[int]"``
****************************************


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



.. _zip:


``zip(*iterables: Any) -> "Iter[Tuple[T, ...]]"``
*************************************************
Replacement for the builtin ``zip`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. _enumerate:


``enumerate(iterable) -> "Iter[Tuple[int, T]]"``
************************************************
Replacement for the builtin ``enumerate`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> enumerate(string.ascii_lowercase).take(3).collect()
    [(0, 'a'), (1, 'b'), (2, 'c')]




.. _map:


``map(func: Union[Callable[..., C], str], iterable) -> "Iter[C]"``
******************************************************************
Replacement for the builtin ``map`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> result = map(lambda x: (x, ord(x)), 'caleb').dict()
    >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

    >>> result = map('x, ord(x)', 'caleb').dict()
    >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}


.. _filter:


``filter(function: "Callable[[Any], ...]", iterable: Iterable) -> "Iter[T]"``
*****************************************************************************
Replacement for the builtin ``filter`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> filter(lambda x: x % 3 == 0, range(10)).collect()
    [0, 3, 6, 9]




.. _count:


|source| ``count(start, step: int = 1) -> "Iter[int]"``
*******************************************************


Replacement for the itertools ``count`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

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



.. _cycle:


``cycle(iterable) -> "Iter[T]"``
********************************
Replacement for the itertools ``count`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> cycle(range(3)).take(6).collect()
    [0, 1, 2, 0, 1, 2]
    >>> cycle([]).take(6).collect()
    []
    >>> cycle(range(3)).take(0).collect()
    []



.. _repeat:


|source| ``repeat(object: C, times=None) -> "Iter[C]"``
*******************************************************


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




This next set of functions return iterators that terminate on the shortest 
input sequence.



.. _accumulate:


``accumulate(iterable, func=None, *, initial=None)``
****************************************************
Replacement for the itertools ``accumulate`` function.  This version returns
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



.. _chain:


``chain(*iterables: Iterable[T]) -> "Iter[T]"``
***********************************************
Replacement for the itertools ``chain`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> chain('ABC', 'DEF').collect()
    ['A', 'B', 'C', 'D', 'E', 'F']
    >>> chain().collect()
    []



.. _chain_from_iterable:


``chain_from_iterable(iterable) -> "Iter[T]"``
**********************************************
Replacement for the itertools ``chain.from_iterable`` method.
This version returns an instance of Iter_ to allow
further iterable chaining.

.. code-block:: python

    >>> chain_from_iterable(['ABC', 'DEF']).collect()
    ['A', 'B', 'C', 'D', 'E', 'F']
    >>> chain_from_iterable([]).collect()
    []



.. _compress:


``compress(data, selectors)``
*****************************
Replacement for the itertools ``compress`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> compress('ABCDEF', [1, 0, 1, 0, 1, 1]).collect()
    ['A', 'C', 'E', 'F']




.. _dropwhile:


``dropwhile(pred, iterable)``
*****************************
Replacement for the itertools ``dropwhile`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> dropwhile(lambda x: x < 4, range(6)).collect()
    [4, 5]



.. _filterfalse:


``filterfalse(pred, iterable)``
*******************************
Replacement for the itertools ``filterfalse`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> filterfalse(None, [2, 0, 3, None, 4, 0]).collect()
    [0, None, 0]



.. _groupby:


``groupby(iterable, key=None)``
*******************************
Replacement for the itertools ``groupby`` function.  This version returns
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




.. _islice:


``islice(iterable, *args) -> "Iter"``
*************************************
Replacement for the itertools ``islice`` function.  This version returns
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



.. _starmap:


``starmap(func, iterable)``
***************************
Replacement for the itertools ``starmap`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> starmap(pow, [(2, 5), (3, 2), (10, 3)]).collect()
    [32, 9, 1000]



.. _takewhile:


``takewhile(pred, iterable)``
*****************************
Replacement for the itertools ``takewhile`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> takewhile(lambda x: x < 5, [1, 4, 6, 4, 1]).collect()
    [1, 4]



.. _tee:


``tee(iterable, n=2)``
**********************
Replacement for the itertools ``tee`` function.  This version returns
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



.. _zip_longest:


``zip_longest(*iterables, fillvalue=None)``
*******************************************
Replacement for the itertools ``zip_longest`` function.  This version returns
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



.. _Iter:


|cool| ``class Iter(Generic[T])``
*********************************


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





.. _Iter.collect:


|sink| ``Iter.collect(self, container=list) -> "List[T]"``
==========================================================



.. code-block:: python

    >>> Iter('abc').collect()
    ['a', 'b', 'c']
    >>> Iter('abc').collect(str)
    'abc'
    >>> Iter('abcaaaabbbbccc').collect(set) == {'a', 'b', 'c'}
    True



.. _Iter.open:


|cool| |source| ``@classmethod Iter.open(cls, file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None, ) -> "Iter"``
==============================================================================================================================================================




Wrap the ``open()`` builtin precisely, but return an ``Iter``
instance to allow function chaining on the result.

>>> import tempfile
>>> with tempfile.TemporaryDirectory() as td:
...     with open(td + 'text.txt', 'w') as f:
...         f.writelines(['abc\n', 'def\n', 'ghi\n'])
...     Iter.open(td + 'text.txt').filter(lambda line: 'def' in line).collect()
['def\n']

Note that this is a convenience method for *reading* from a file,
not for writing. The function signature includes the ``mode``
parameter for parity with the builtin ``open()`` function, but
only reading is supported.



.. _Iter.range:


|source| ``@classmethod Iter.range(cls, *args) -> "Iter[int]"``
===============================================================


Docstring TBD


.. _Iter.zip:


``Iter.zip(self, *iterables: Any) -> "Iter[Tuple[T, ...]]"``
============================================================
Docstring TBD


.. _Iter.any:


|sink| ``Iter.any(self) -> "bool"``
===================================


Docstring TBD


.. _Iter.all:


|sink| ``Iter.all(self) -> "bool"``
===================================


Docstring TBD


.. _Iter.enumerate:


``Iter.enumerate(self) -> "Iter[Tuple[int, T]]"``
=================================================
Docstring TBD


.. _Iter.dict:


``Iter.dict(self) -> "Dict"``
=============================
Docstring TBD


.. _Iter.map:


``Iter.map(self, func: Union[Callable[..., C], str]) -> "Iter[C]"``
===================================================================

>>> result = Iter('caleb').map(lambda x: (x, ord(x))).dict()
>>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

>>> result = Iter('caleb').map('x, ord(x)').dict()
>>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}


.. _Iter.filter:


``Iter.filter(self, *args) -> "Iter[T]"``
=========================================
Docstring TBD


.. _Iter.reduce:


``Iter.reduce(self, func: Callable[..., T], *args) -> "T"``
===========================================================
Docstring TBD


.. _Iter.sum:


|sink| ``Iter.sum(self)``
=========================


Docstring TBD 

.. _Iter.concat:


|sink| ``Iter.concat(self, glue: AnyStr) -> "AnyStr"``
======================================================


Docstring TBD


.. _Iter.insert:


``Iter.insert(self, glue: C) -> "Iter[Union[C, T]]"``
=====================================================
Docstring TBD


.. _Iter.count:


|source| ``@classmethod Iter.count(cls, *args) -> "Iter[int]"``
===============================================================


Docstring TBD 

.. _Iter.cycle:


``Iter.cycle(self) -> "Iter[T]"``
=================================
Docstring TBD


.. _Iter.repeat:


|source| ``@classmethod Iter.repeat(cls, elem: C, times=None) -> "Iter[C]"``
============================================================================


Docstring TBD 

.. _Iter.accumulate:


``Iter.accumulate(self, func=None, *, initial=None)``
=====================================================
Docstring TBD

.. code-block:: python

    >>> Iter([1, 2, 3, 4, 5]).accumulate().collect()
    [1, 3, 6, 10, 15]
    >>> if sys.version_info >= (3, 8):
    ...     out = Iter([1, 2, 3, 4, 5]).accumulate(initial=100).collect()
    ...     assert out == [100, 101, 103, 106, 110, 115]
    >>> Iter([1, 2, 3, 4, 5]).accumulate(operator.mul).collect()
    [1, 2, 6, 24, 120]



.. _Iter.chain:


``Iter.chain(self, *iterables: Iterable[T]) -> "Iter[T]"``
==========================================================
Docstring TBD

.. code-block:: python

    >>> Iter('ABC').chain('DEF').collect()
    ['A', 'B', 'C', 'D', 'E', 'F']
    >>> Iter('ABC').chain().collect()
    ['A', 'B', 'C']



.. _Iter.chain_from_iterable:


``Iter.chain_from_iterable(self) -> "Iter[T]"``
===============================================
Docstring TBD

.. code-block:: python

    >>> Iter(['ABC', 'DEF']).chain_from_iterable().collect()
    ['A', 'B', 'C', 'D', 'E', 'F']



.. _Iter.compress:


``Iter.compress(self, selectors)``
==================================
Replacement for the itertools ``compress`` function.  This version returns
an instance of Iter_ to allow further iterable chaining.

.. code-block:: python

    >>> Iter('ABCDEF').compress([1, 0, 1, 0, 1, 1]).collect()
    ['A', 'C', 'E', 'F']



.. _Iter.dropwhile:


``Iter.dropwhile(self, pred)``
==============================
Docstring TBD


.. _Iter.filterfalse:


``Iter.filterfalse(self, pred)``
================================
Docstring TBD


.. _Iter.groupby:


``Iter.groupby(self, key=None)``
================================
Docstring TBD


.. _Iter.islice:


``Iter.islice(self, *args) -> "Iter"``
======================================
Docstring TBD


.. _Iter.starmap:


``Iter.starmap(self, func)``
============================
Docstring TBD


.. _Iter.takewhile:


``Iter.takewhile(self, pred)``
==============================
Docstring TBD


.. _Iter.tee:


``Iter.tee(self, n=2)``
=======================
Docstring TBD


.. _Iter.zip_longest:


``Iter.zip_longest(self, *iterables, fillvalue=None)``
======================================================
Docstring TBD


.. _Iter.chunked:


``Iter.chunked(self, n: int) -> "Iter"``
========================================
Docstring TBD


.. _Iter.ichunked:


``Iter.ichunked(self, n: int) -> "Iter"``
=========================================
Docstring TBD


.. _Iter.sliced:


``@classmethod Iter.sliced(cls, seq: Sequence, n: int) -> "Iter"``
==================================================================
Docstring TBD


.. _Iter.distribute:


``Iter.distribute(self, n: int) -> "Iter"``
===========================================
Docstring TBD


.. _Iter.divide:


``Iter.divide(self, n: int) -> "Iter"``
=======================================
Docstring TBD


.. _Iter.split_at:


``Iter.split_at(self, pred)``
=============================
Docstring TBD


.. _Iter.split_before:


``Iter.split_before(self, pred)``
=================================
Docstring TBD


.. _Iter.split_after:


``Iter.split_after(self, pred)``
================================
Docstring TBD


.. _Iter.split_into:


``Iter.split_into(self, sizes)``
================================
Docstring TBD


.. _Iter.split_when:


``Iter.split_when(self, pred)``
===============================
Docstring TBD


.. _Iter.bucket:


``Iter.bucket(self, key, validator=None)``
==========================================
Docstring TBD


.. _Iter.unzip:


``Iter.unzip(self)``
====================
Docstring TBD


.. _Iter.grouper:


``Iter.grouper(self, n: int, fillvalue=None) -> "Iter"``
========================================================
Docstring TBD


.. _Iter.partition:


``Iter.partition(self, pred) -> "Iter"``
========================================
Docstring TBD


.. _Iter.spy:


``Iter.spy(self, n=1) -> "Tuple[Iter, Iter]"``
==============================================
Docstring TBD


.. _Iter.peekable:


``Iter.peekable(self) -> "more_itertools.peekable"``
====================================================
Docstring TBD


.. _Iter.seekable:


``Iter.seekable(self) -> "more_itertools.seekable"``
====================================================
Docstring TBD


.. _Iter.windowed:


``Iter.windowed(self, n, fillvalue=None, step=1) -> "Iter"``
============================================================
Docstring TBD


.. _Iter.substrings:


``Iter.substrings(self)``
=========================
Docstring TBD


.. _Iter.substrings_indexes:


``Iter.substrings_indexes(self, reverse=False)``
================================================
Docstring TBD


.. _Iter.stagger:


``Iter.stagger(self, offsets=(-1, 0, 1), longest=False, fillvalue=None)``
=========================================================================

.. code-block:: python

    >>> Iter([0, 1, 2, 3]).stagger().collect()
    [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
    >>> Iter(range(8)).stagger(offsets=(0, 2, 4)).collect()
    [(0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7)]
    >>> Iter([0, 1, 2, 3]).stagger(longest=True).collect()
    [(None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]



.. _Iter.pairwise:


``Iter.pairwise(self)``
=======================

Reference `more_itertools.pairwise <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise>`_

.. code-block:: python

    >>> Iter.count().pairwise().take(4).collect()
    [(0, 1), (1, 2), (2, 3), (3, 4)]


.. _Iter.count_cycle:


``Iter.count_cycle(self, n=None) -> "Iter"``
============================================


Reference: `more_itertools.count_cycle <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle>`_

.. code-block:: python

    >>> Iter('AB').count_cycle(3).collect()
    [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]



.. _Iter.intersperse:


``Iter.intersperse(self, e, n=1) -> "Iter"``
============================================

Reference: `more_itertools.intersperse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse>`_

.. code-block:: python

    >>> Iter([1, 2, 3, 4, 5]).intersperse('!').collect()
    [1, '!', 2, '!', 3, '!', 4, '!', 5]

    >>> Iter([1, 2, 3, 4, 5]).intersperse(None, n=2).collect()
    [1, 2, None, 3, 4, None, 5]



.. _Iter.padded:


``Iter.padded(self, fillvalue: Optional[C] = None, n: Optional[int] = None, next_multiple: bool = False, ) -> "Iter[Union[T, C]]"``
===================================================================================================================================

Reference: `more_itertools.padded <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded>`_

.. code-block:: python

    >>> Iter([1, 2, 3]).padded('?', 5).collect()
    [1, 2, 3, '?', '?']

    >>> Iter([1, 2, 3, 4]).padded(n=3, next_multiple=True).collect()
    [1, 2, 3, 4, None, None]



.. _Iter.repeat_last:


``Iter.repeat_last(self, default=None) -> "Iter[T]"``
=====================================================

Reference: `more_itertools.repeat_last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeat_last>`_

.. code-block:: python

    >>> Iter(range(3)).repeat_last().islice(5).collect()
    [0, 1, 2, 2, 2]

    >>> Iter(range(0)).repeat_last(42).islice(5).collect()
    [42, 42, 42, 42, 42]



.. _Iter.adjacent:


``Iter.adjacent(self, pred, distance=1) -> "Iter[Tuple[bool, T]]"``
===================================================================

Reference: `more_itertools.adjacent <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent>`_

.. code-block:: python

    >>> Iter(range(6)).adjacent(lambda x: x == 3).collect()
    [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]

    >>> Iter(range(6)).adjacent(lambda x: x == 3, distance=2).collect()
    [(False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)]




.. _Iter.groupby_transform:


``Iter.groupby_transform(self, keyfunc: Optional[Callable[..., K]] = None, valuefunc: Optional[Callable[..., V]] = None, ) -> "Iter[Tuple[K, Iterable[V]]]"``
=============================================================================================================================================================

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



.. _Iter.padnone:


``Iter.padnone(self) -> "Iter[Union[T, None]]"``
================================================

Reference: `more_itertools.padnone <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone>`_

.. code-block:: python

    >>> Iter(range(3)).padnone().take(5).collect()
    [0, 1, 2, None, None]



.. _Iter.ncycles:


``Iter.ncycles(self, n) -> "Iter[T]"``
======================================

Reference: `more_itertools.ncycles <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles>`_

.. code-block:: python

    >>> Iter(['a', 'b']).ncycles(3).collect()
    ['a', 'b', 'a', 'b', 'a', 'b']



.. _Iter.collapse:


``Iter.collapse(self, base_type=None, levels=None) -> "Iter"``
==============================================================

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



.. _Iter.sort_together:


``@class_or_instancemethod Iter.sort_together(self_or_cls, iterables, key_list=(0,), reverse=False)``
=====================================================================================================

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

    >>> iterables = [('a', 'b', 'c', 'd')]
    >>> Iter([4, 3, 2, 1]).sort_together(iterables).collect()
    [(1, 2, 3, 4), ('d', 'c', 'b', 'a')]



.. _Iter.interleave:


``@class_or_instancemethod Iter.interleave(self_or_cls, *iterables) -> "Iter"``
===============================================================================

Reference: `more_itertools.interleave <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave>`_

Classmethod form:

.. code-block:: python

    >>> Iter.interleave([1, 2, 3], [4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7]

Instancemethod form:

.. code-block:: python

    >>> Iter([1, 2, 3]).interleave([4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7]



.. _Iter.interleave_longest:


``@class_or_instancemethod Iter.interleave_longest(self_or_cls, *iterables) -> "Iter"``
=======================================================================================

Reference: `more_itertools.interleave_longest <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave_longest>`_

Classmethod form:

.. code-block:: python

    >>> Iter.interleave_longest([1, 2, 3], [4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7, 3, 8]

Instancemethod form:

.. code-block:: python

    >>> Iter([1, 2, 3]).interleave_longest([4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7, 3, 8]



.. _Iter.zip_offset:


``@classmethod Iter.zip_offset(cls, *iterables, offsets, longest=False, fillvalue=None) -> "Iter"``
===================================================================================================

Reference: `more_itertools.zip_offset <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_offset>`_

.. code-block:: python

    >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1)).collect()
    [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e')]

    >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True).collect()
    [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e'), (None, 'f')]


.. _Iter.dotproduct:


``Iter.dotproduct(self, vec2: Iterable)``
=========================================

Reference: `more_itertools.dotproduct <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.dotproduct>`_

.. code-block:: python

    >>> Iter([10, 10]).dotproduct([20, 20])
    400


.. _Iter.flatten:


``Iter.flatten(self) -> "Iter[T]"``
===================================

Reference: `more_itertools.flatten <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten>`_

.. code-block:: python

    >>> Iter([[0, 1], [2, 3]]).flatten().collect()
    [0, 1, 2, 3]



.. _Iter.roundrobin:


``@class_or_instancemethod Iter.roundrobin(self_or_cls: Union[Type[T], T], *iterables: C) -> "Iter[Union[T, C]]"``
==================================================================================================================

Reference: `more_itertools.roundrobin <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.roundrobin>`_

Classmethod form:

.. code-block:: python

    >>> Iter.roundrobin('ABC', 'D', 'EF').collect()
    ['A', 'D', 'E', 'B', 'F', 'C']

Instancemethod form:

.. code-block:: python

    >>> Iter('ABC').roundrobin('D', 'EF').collect()
    ['A', 'D', 'E', 'B', 'F', 'C']



.. _Iter.prepend:


``Iter.prepend(self, value: C) -> "Iter[Union[T, C]]"``
=======================================================

Reference: `more_itertools.prepend <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.prepend>`_

.. code-block:: python

    >>> value = '0'
    >>> iterator = ['1', '2', '3']
    >>> Iter(iterator).prepend(value).collect()
    ['0', '1', '2', '3']



.. _Iter.ilen:


|sink| ``Iter.ilen(self) -> "int"``
===================================



Reference: `more_itertools.ilen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ilen>`_

.. code-block:: python

    >>> Iter(x for x in range(1000000) if x % 3 == 0).ilen()
    333334



.. _Iter.unique_to_each:


``Iter.unique_to_each(self) -> "Iter[T]"``
==========================================

Reference: `more_itertools.unique_to_each <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each>`_

.. code-block:: python

    >>> Iter([{'A', 'B'}, {'B', 'C'}, {'B', 'D'}]).unique_to_each().collect()
    [['A'], ['C'], ['D']]

    >>> Iter(["mississippi", "missouri"]).unique_to_each().collect()
    [['p', 'p'], ['o', 'u', 'r']]


.. _Iter.sample:


``Iter.sample(self, k=1, weights=None) -> "Iter"``
==================================================

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



.. _Iter.consecutive_groups:


``Iter.consecutive_groups(self, ordering=lambda x: x)``
=======================================================

Reference: `more_itertools.consecutive_groups <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consecutive_groups>`_

.. code-block:: python

    >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
    >>> Iter(iterable).consecutive_groups().map(lambda g: list(g)).print('{v}').consume()
    [1]
    [10, 11, 12]
    [20]
    [30, 31, 32, 33]
    [40]



.. _Iter.run_length_encode:


``Iter.run_length_encode(self) -> "Iter[Tuple[T, int]]"``
=========================================================

Reference: `more_itertools.run_length <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length>`_

.. code-block:: python

    >>> uncompressed = 'abbcccdddd'
    >>> Iter(uncompressed).run_length_encode().collect()
    [('a', 1), ('b', 2), ('c', 3), ('d', 4)]



.. _Iter.run_length_decode:


``Iter.run_length_decode(self) -> "Iter"``
==========================================

Reference: `more_itertools.run_length <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length>`_

.. code-block:: python

    >>> compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    >>> Iter(compressed).run_length_decode().collect()
    ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']



.. _Iter.map_reduce:


``Iter.map_reduce(self, keyfunc, valuefunc=None, reducefunc=None) -> "Dict"``
=============================================================================

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



.. _Iter.map_reduce_it:


``Iter.map_reduce_it(self, keyfunc: Callable[..., K], valuefunc: Optional[Callable[..., V]] = None, reducefunc: Optional[Callable[..., R]] = None, ) -> "Iter[Tuple[K, R]]"``
=============================================================================================================================================================================

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



.. _Iter.exactly_n:


|sink| ``Iter.exactly_n(self, n, predicate=bool) -> "bool"``
============================================================



Docstring TBD

.. code-block:: python

    >>> Iter([True, True, False]).exactly_n(2)
    True



.. _Iter.all_equal:


``Iter.all_equal(self)``
========================

.. _Iter.first_true:


``Iter.first_true(self)``
=========================

.. _Iter.quantify:


``Iter.quantify(self)``
=======================

.. _Iter.islice_extended:


``Iter.islice_extended(self, *args)``
=====================================

.. _Iter.first:


``Iter.first(self)``
====================

.. _Iter.last:


``Iter.last(self)``
===================

.. _Iter.one:


``Iter.one(self)``
==================

.. _Iter.only:


``Iter.only(self, default=None, too_long=None) -> "Any"``
=========================================================

.. _Iter.strip:


``Iter.strip(self, pred)``
==========================

.. _Iter.lstrip:


``Iter.lstrip(self, pred)``
===========================

.. _Iter.rstrip:


``Iter.rstrip(self, pred)``
===========================

.. _Iter.filter_except:


``Iter.filter_except(self, validator, *exceptions)``
====================================================

.. _Iter.map_except:


``Iter.map_except(self, function, *exceptions)``
================================================

.. _Iter.nth_or_last:


``Iter.nth_or_last(self)``
==========================

.. _Iter.nth:


``Iter.nth(self, n, default=None)``
===================================

.. _Iter.take:


``Iter.take(self, n: int) -> "Iter"``
=====================================

.. _Iter.tail:


``Iter.tail(self)``
===================

.. _Iter.unique_everseen:


``Iter.unique_everseen(self)``
==============================

.. _Iter.unique_justseen:


``Iter.unique_justseen(self)``
==============================

.. _Iter.distinct_permutations:


``Iter.distinct_permutations(self)``
====================================

.. _Iter.distinct_combinations:


``Iter.distinct_combinations(self, r)``
=======================================

.. _Iter.circular_shifts:


``Iter.circular_shifts(self) -> "Iter"``
========================================

.. _Iter.partitions:


``Iter.partitions(self) -> "Iter"``
===================================

.. _Iter.set_partitions:


``Iter.set_partitions(self, k=None) -> "Iter"``
===============================================

.. _Iter.powerset:


``Iter.powerset(self)``
=======================

.. _Iter.random_product:


``Iter.random_product(self)``
=============================

.. _Iter.random_permutation:


``Iter.random_permutation(self)``
=================================

.. _Iter.random_combination:


``Iter.random_combination(self)``
=================================

.. _Iter.random_combination_with_replacement:


``Iter.random_combination_with_replacement(self)``
==================================================

.. _Iter.nth_combination:


``Iter.nth_combination(self)``
==============================

.. _Iter.always_iterable:


``Iter.always_iterable(self)``
==============================

.. _Iter.always_reversible:


``Iter.always_reversible(self)``
================================

.. _Iter.with_iter:


``@classmethod Iter.with_iter(cls, context_manager)``
=====================================================

Reference: `more_itertools.with_iter <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.with_iter>`_

Note: Any context manager which returns an iterable is a candidate for
Iter.with_iter_.

.. code-block:: python

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as td:
    ...     with open(td + 'text.txt', 'w') as f:
    ...         f.writelines(['abc\n', 'def\n', 'ghi\n'])
    ...     Iter.with_iter(open(td + 'text.txt')).map(lambda x: x.upper()).collect()
    ['ABC\n', 'DEF\n', 'GHI\n']

See also: Iter.open_

|flux| TODO: perhaps we should get rid of Iter.open_ and just use this?



.. _Iter.iter_except:


``@classmethod Iter.iter_except(cls, func, exception, first=None) -> "Iter"``
=============================================================================

Reference: `more_itertools.iter_except <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.iter_except>`_

.. code-block:: python

    >>> l = [0, 1, 2]
    >>> Iter.iter_except(l.pop, IndexError).collect()
    [2, 1, 0]



.. _Iter.locate:


``Iter.locate(self, pred=bool, window_size=None) -> "Iter"``
============================================================

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



.. _Iter.rlocate:


``Iter.rlocate(self, pred=bool, window_size=None) -> "Iter"``
=============================================================

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



.. _Iter.replace:


``Iter.replace(self, pred, substitutes, count=None, window_size=1) -> "Iter"``
==============================================================================

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



.. _Iter.numeric_range:


``@classmethod Iter.numeric_range(cls, *args) -> "Iter"``
=========================================================

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



.. _Iter.side_effect:


``Iter.side_effect(self, func, chunk_size=None, before=None, after=None)``
==========================================================================

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




.. _Iter.iterate:


``Iter.iterate(self)``
======================

.. _Iter.difference:


``Iter.difference(self, func=operator.sub, *, initial=None)``
=============================================================

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



.. _Iter.make_decorator:


``Iter.make_decorator(self)``
=============================

.. _Iter.SequenceView:


``Iter.SequenceView(self)``
===========================

.. _Iter.time_limited:


``Iter.time_limited(self, limit_seconds) -> "Iter"``
====================================================

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



.. _Iter.consume:


|sink| ``Iter.consume(self, n: Optional[int] = None) -> "Optional[Iter[T]]"``
=============================================================================


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



.. _Iter.tabulate:


``Iter.tabulate(self)``
=======================

.. _Iter.repeatfunc:


|source| ``@classmethod Iter.repeatfunc(cls, func, *args, times=None)``
=======================================================================


Docstring TBD

.. code-block:: python

    >>> Iter.repeatfunc(operator.add, 3, 5, times=4).collect()
    [8, 8, 8, 8]



.. _Iter.wrap:


``Iter.wrap(self, ends: "Sequence[T, T]" = "()")``
==================================================
Other examples for ends: '"' * 2, or '`' * 2, or '[]' etc.


.. _Iter.print:


``Iter.print(self, template="{i}: {v}") -> "Iter[T]"``
======================================================

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



.. _Iter.from_queue:


|source| ``@classmethod Iter.from_queue(cls, q: queue.Queue, timeout=None, sentinel=None)``
===========================================================================================


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



.. _Iter.into_queue:


|sink| ``Iter.into_queue(self, q: queue.Queue)``
================================================


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



.. _Iter.send:


|sink| ``Iter.send(self, collector: Generator, close_collector_when_done=False) -> "None"``
===========================================================================================


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



.. _Iter.send_also:


``Iter.send_also(self, collector: Generator) -> "Iter"``
========================================================

Reference: `more_itertools.consumer <https://more-itertools.readthedocs.io/en/stable/api.html?highlight=numeric_range#more_itertools.consumer>`_

Some ideas around a reverse iterator as a sink. The requirement to
first "next" a just-started generator before you can send values
into it is irritating, but not insurmountable. This method will
automatically detect the "just-started generator" situation, do the
``next()``, and then send in the first value as necessary.

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



.. _IterDict:


|flux| ``class IterDict(UserDict)``
***********************************



The idea here was to make a custom dict where several of
the standard dict methods return ``Iter`` instances, which can then
be chained. I'm not sure if this will be kept yet.


.. _IterDict.keys:


``IterDict.keys(self) -> "Iter"``
=================================

.. _IterDict.values:


``IterDict.values(self) -> "Iter"``
===================================

.. _IterDict.items:


``IterDict.items(self) -> "Iter"``
==================================

.. _IterDict.update:


``IterDict.update(self, *args, **kwargs) -> "IterDict"``
========================================================

.. _insert_separator:


``insert_separator(iterable: Iterable[Any], glue: Any) -> "Iterable[Any]"``
***************************************************************************
Similar functionality can be obtained with, e.g.,
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


.. _concat:


``concat(iterable: Iterable[AnyStr], glue: AnyStr) -> "AnyStr"``
****************************************************************
Concatenate strings, bytes and bytearrays. It is careful to avoid the
problem with single bytes becoming integers, and it looks at the value
of `glue` to know whether to handle bytes or strings.

This function can raise ``ValueError`` if called with something
other than ``bytes``, ``bytearray`` or ``str``.

.. _from_queue:


|source| ``from_queue(q: queue.Queue, timeout=None, sentinel=None) -> "Iter"``
******************************************************************************


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



