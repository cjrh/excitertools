

.. image:: https://travis-ci.org/cjrh/excitertools.svg?branch=master
    :target: https://travis-ci.org/cjrh/excitertools

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


API Documentation
#################

.. contents::
    :local:



.. _range:


``range(*args) -> Iter[int]``
*****************************
Replacement for the builtin ``range`` function.  This version returns
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



.. _zip:


``zip(*iterables: Any) -> Iter[Tuple[T, ...]]``
***********************************************
Replacement for the builtin ``zip`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _enumerate:


``enumerate(iterable) -> Iter[Tuple[int, T]]``
**********************************************
Replacement for the builtin ``enumerate`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _map:


``map(func: Union[Callable[..., C], str], iterable) -> Iter[C]``
****************************************************************

.. code-block:: python

    >>> result = Iter('caleb').map(lambda x: (x, ord(x))).dict()
    >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

    >>> result = Iter('caleb').map('x, ord(x)').dict()
    >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}


.. _filter:


``filter(*args, iterable) -> Iter[T]``
**************************************
Replacement for the builtin ``filter`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _count:


``count(*args) -> Iter[int]``
*****************************
Replacement for the itertools ``count`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _cycle:


``cycle(iterable) -> Iter[T]``
******************************
Replacement for the itertools ``count`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _repeat:


``repeat(elem: C, times=None) -> Iter[C]``
******************************************
Replacement for the itertools ``count`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _accumulate:


``accumulate(func, iterable)``
******************************
Replacement for the itertools ``accumulate`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _chain:


``chain(*iterables: Iterable[T]) -> Iter[T]``
*********************************************
Replacement for the itertools ``chain`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _chain_from_iterable:


``chain_from_iterable(iterable) -> Iter[T]``
********************************************
Replacement for the itertools ``chain.from_iterable`` method.
This version returns an instance of ``excitertools.Iter`` to allow
further iterable chaining.

.. _compress:


``compress(selectors, iterable)``
*********************************
Replacement for the itertools ``compress`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _dropwhile:


``dropwhile(pred, iterable)``
*****************************
Replacement for the itertools ``dropwhile`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _filterfalse:


``filterfalse(pred, iterable)``
*******************************
Replacement for the itertools ``filterfalse`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _groupby:


``groupby(iterable, key=None)``
*******************************
Replacement for the itertools ``groupby`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _islice:


``islice(*args, iterable) -> Iter``
***********************************
Replacement for the itertools ``islice`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _starmap:


``starmap(func, iterable)``
***************************
Replacement for the itertools ``starmap`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _takewhile:


``takewhile(pred, iterable)``
*****************************
Replacement for the itertools ``takewhile`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _tee:


``tee(iterable, n=2)``
**********************
Replacement for the itertools ``tee`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.

.. _zip_longest:


``zip_longest(*iterables, fillvalue=None)``
*******************************************
Replacement for the itertools ``zip_longest`` function.  This version returns
an instance of ``excitertools.Iter`` to allow further iterable chaining.
.. _Iter:


``class Iter(Generic[T])``
**************************

This is the docstring for the ``Iter`` class.

Each of the following methods of ``Iter`` describe how they work.

Test warning: |warning|



.. _Iter.collect:


``Iter.collect(self, container=list) -> List[T]``
=================================================

.. code-block:: python

    >>> Iter('abc').collect()
    ['a', 'b', 'c']
    >>> Iter('abc').collect(str)
    'abc'
    >>> Iter('abcaaaabbbbccc').collect(set) == {'a', 'b', 'c'}
    True



.. _Iter.open:


|cool| ``@classmethod Iter.open(cls, file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None, ) -> Iter``
===================================================================================================================================================



Wrap the ``open()`` builtin precisely, but return an ``Iter``
instance to allow function chaining on the result.

>>> import tempfile
>>> with tempfile.TemporaryDirectory() as td:
...     open('text.txt', 'w').writelines(['abc\n', 'def\n', 'ghi\n'])
...     Iter.open('text.txt').filter(lambda line: 'def' in line).collect()
['def\n']

Note that this is a convenience method for *reading* from a file,
not for writing. The function signature includes the ``mode``
parameter for parity with the builtin ``open()`` function, but
only reading is supported.



.. _Iter.range:


``@classmethod Iter.range(cls, *args) -> Iter[int]``
====================================================
Docstring TBD


.. _Iter.zip:


``Iter.zip(self, *iterables: Any) -> Iter[Tuple[T, ...]]``
==========================================================
Docstring TBD


.. _Iter.any:


``Iter.any(self) -> bool``
==========================
Docstring TBD


.. _Iter.all:


``Iter.all(self) -> bool``
==========================
Docstring TBD


.. _Iter.enumerate:


``Iter.enumerate(self) -> Iter[Tuple[int, T]]``
===============================================
Docstring TBD


.. _Iter.dict:


``Iter.dict(self) -> Dict``
===========================
Docstring TBD


.. _Iter.map:


``Iter.map(self, func: Union[Callable[..., C], str]) -> Iter[C]``
=================================================================

>>> result = Iter('caleb').map(lambda x: (x, ord(x))).dict()
>>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

>>> result = Iter('caleb').map('x, ord(x)').dict()
>>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}


.. _Iter.filter:


``Iter.filter(self, *args) -> Iter[T]``
=======================================
Docstring TBD


.. _Iter.reduce:


``Iter.reduce(self, func: Callable[..., T], *args) -> T``
=========================================================
Docstring TBD


.. _Iter.sum:


``Iter.sum(self)``
==================
Docstring TBD


.. _Iter.concat:


``Iter.concat(self, glue: AnyStr) -> AnyStr``
=============================================
Docstring TBD


.. _Iter.insert:


``Iter.insert(self, glue: C) -> Iter[Union[C, T]]``
===================================================
Docstring TBD


.. _Iter.count:


``@classmethod Iter.count(cls, *args) -> Iter[int]``
====================================================
Docstring TBD


.. _Iter.cycle:


``Iter.cycle(self) -> Iter[T]``
===============================
Docstring TBD


.. _Iter.repeat:


``@classmethod Iter.repeat(cls, elem: C, times=None) -> Iter[C]``
=================================================================
Docstring TBD


.. _Iter.accumulate:


``Iter.accumulate(self, func)``
===============================
Docstring TBD


.. _Iter.chain:


``Iter.chain(self, *iterables: Iterable[T]) -> Iter[T]``
========================================================
Docstring TBD


.. _Iter.chain_from_iterable:


``Iter.chain_from_iterable(self) -> Iter[T]``
=============================================
Docstring TBD


.. _Iter.compress:


``Iter.compress(self, selectors)``
==================================
Docstring TBD


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


``Iter.islice(self, *args) -> Iter``
====================================
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


``Iter.chunked(self, n: int) -> Iter``
======================================
Docstring TBD


.. _Iter.ichunked:


``Iter.ichunked(self, n: int) -> Iter``
=======================================
Docstring TBD


.. _Iter.sliced:


``@classmethod Iter.sliced(cls, seq: Sequence, n: int) -> Iter``
================================================================
Docstring TBD


.. _Iter.distribute:


``Iter.distribute(self, n: int) -> Iter``
=========================================
Docstring TBD


.. _Iter.divide:


``Iter.divide(self, n: int) -> Iter``
=====================================
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


``Iter.grouper(self, n: int, fillvalue=None) -> Iter``
======================================================
Docstring TBD


.. _Iter.partition:


``Iter.partition(self, pred) -> Iter``
======================================
Docstring TBD


.. _Iter.spy:


``Iter.spy(self, n=1) -> Tuple[Iter, Iter]``
============================================
Docstring TBD


.. _Iter.peekable:


``Iter.peekable(self) -> more_itertools.peekable``
==================================================
Docstring TBD


.. _Iter.seekable:


``Iter.seekable(self) -> more_itertools.seekable``
==================================================
Docstring TBD


.. _Iter.windowed:


``Iter.windowed(self, n, fillvalue=None, step=1) -> Iter``
==========================================================
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

See https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise

.. code-block:: python

    >>> Iter.count().pairwise().take(4).collect()
    [(0, 1), (1, 2), (2, 3), (3, 4)]


.. _Iter.count_cycle:


``Iter.count_cycle(self, n=None) -> Iter``
==========================================


See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle

.. code-block:: python

    >>> Iter('AB').count_cycle(3).collect()
    [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]



.. _Iter.intersperse:


``Iter.intersperse(self, e, n=1) -> Iter``
==========================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse

.. code-block:: python

    >>> Iter([1, 2, 3, 4, 5]).intersperse('!').collect()
    [1, '!', 2, '!', 3, '!', 4, '!', 5]

    >>> Iter([1, 2, 3, 4, 5]).intersperse(None, n=2).collect()
    [1, 2, None, 3, 4, None, 5]



.. _Iter.padded:


``Iter.padded(self, fillvalue: Optional[C] = None, n: Optional[int] = None, next_multiple: bool = False, ) -> Iter[Union[T, C]]``
=================================================================================================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded

.. code-block:: python

    >>> Iter([1, 2, 3]).padded('?', 5).collect()
    [1, 2, 3, '?', '?']

    >>> Iter([1, 2, 3, 4]).padded(n=3, next_multiple=True).collect()
    [1, 2, 3, 4, None, None]



.. _Iter.repeat_last:


``Iter.repeat_last(self, default=None) -> Iter[T]``
===================================================

https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeat_last

.. code-block:: python

    >>> Iter(range(3)).repeat_last().islice(5).collect()
    [0, 1, 2, 2, 2]

    >>> Iter(range(0)).repeat_last(42).islice(5).collect()
    [42, 42, 42, 42, 42]



.. _Iter.adjacent:


``Iter.adjacent(self, pred, distance=1) -> Iter[Tuple[bool, T]]``
=================================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent

.. code-block:: python

    >>> Iter(range(6)).adjacent(lambda x: x == 3).collect()
    [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]

    >>> Iter(range(6)).adjacent(lambda x: x == 3, distance=2).collect()
    [(False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)]




.. _Iter.groupby_transform:


``Iter.groupby_transform(self, keyfunc: Optional[Callable[..., K]] = None, valuefunc: Optional[Callable[..., V]] = None, ) -> Iter[Tuple[K, Iterable[V]]]``
===========================================================================================================================================================

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



.. _Iter.padnone:


``Iter.padnone(self) -> Iter[Union[T, None]]``
==============================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone

.. code-block:: python

    >>> Iter(range(3)).padnone().take(5).collect()
    [0, 1, 2, None, None]



.. _Iter.ncycles:


``Iter.ncycles(self, n) -> Iter[T]``
====================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles

.. code-block:: python

    >>> Iter(['a', 'b']).ncycles(3).collect()
    ['a', 'b', 'a', 'b', 'a', 'b']



.. _Iter.collapse:


``Iter.collapse(self, base_type=None, levels=None) -> Iter``
============================================================

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



.. _Iter.sort_together:


``@class_or_instancemethod Iter.sort_together(self_or_cls, iterables, key_list=(0,), reverse=False)``
=====================================================================================================

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



.. _Iter.interleave:


``@class_or_instancemethod Iter.interleave(self_or_cls, *iterables) -> Iter``
=============================================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave

Classmethod form:

.. code-block:: python

    >>> Iter.interleave([1, 2, 3], [4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7]

Instancemethod form:

.. code-block:: python

    >>> Iter([1, 2, 3]).interleave([4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7]



.. _Iter.interleave_longest:


``@class_or_instancemethod Iter.interleave_longest(self_or_cls, *iterables) -> Iter``
=====================================================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave_longest

Classmethod form:

.. code-block:: python

    >>> Iter.interleave_longest([1, 2, 3], [4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7, 3, 8]

Instancemethod form:

.. code-block:: python

    >>> Iter([1, 2, 3]).interleave_longest([4, 5], [6, 7, 8]).collect()
    [1, 4, 6, 2, 5, 7, 3, 8]



.. _Iter.zip_offset:


``@classmethod Iter.zip_offset(cls, *iterables, offsets, longest=False, fillvalue=None) -> Iter``
=================================================================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_offset

.. code-block:: python

    >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1)).collect()
    [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e')]

    >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True).collect()
    [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e'), (None, 'f')]


.. _Iter.dotproduct:


``Iter.dotproduct(self, vec2: Iterable)``
=========================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.dotproduct

.. code-block:: python

    >>> Iter([10, 10]).dotproduct([20, 20])
    400


.. _Iter.flatten:


``Iter.flatten(self) -> Iter[T]``
=================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten

.. code-block:: python

    >>> Iter([[0, 1], [2, 3]]).flatten().collect()
    [0, 1, 2, 3]



.. _Iter.roundrobin:


``@class_or_instancemethod Iter.roundrobin(self_or_cls: Union[Type[T], T], *iterables: C) -> Iter[Union[T, C]]``
================================================================================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.roundrobin

Classmethod form:

.. code-block:: python

    >>> Iter.roundrobin('ABC', 'D', 'EF').collect()
    ['A', 'D', 'E', 'B', 'F', 'C']

Instancemethod form:

.. code-block:: python

    >>> Iter('ABC').roundrobin('D', 'EF').collect()
    ['A', 'D', 'E', 'B', 'F', 'C']



.. _Iter.prepend:


``Iter.prepend(self, value: C) -> Iter[Union[T, C]]``
=====================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.prepend

.. code-block:: python

    >>> value = '0'
    >>> iterator = ['1', '2', '3']
    >>> Iter(iterator).prepend(value).collect()
    ['0', '1', '2', '3']



.. _Iter.ilen:


``Iter.ilen(self) -> int``
==========================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ilen

.. code-block:: python

    >>> Iter(x for x in range(1000000) if x % 3 == 0).ilen()
    333334



.. _Iter.unique_to_each:


``Iter.unique_to_each(self) -> Iter[T]``
========================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each

.. code-block:: python

    >>> Iter([{'A', 'B'}, {'B', 'C'}, {'B', 'D'}]).unique_to_each().collect()
    [['A'], ['C'], ['D']]

    >>> Iter(["mississippi", "missouri"]).unique_to_each().collect()
    [['p', 'p'], ['o', 'u', 'r']]


.. _Iter.sample:


``Iter.sample(self, k=1, weights=None) -> Iter``
================================================

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



.. _Iter.consecutive_groups:


``Iter.consecutive_groups(self, ordering=lambda x: x)``
=======================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consecutive_groups

.. code-block:: python

    >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
    >>> Iter(iterable).consecutive_groups().map(lambda g: list(g)).print('{v}').consume()
    [1]
    [10, 11, 12]
    [20]
    [30, 31, 32, 33]
    [40]



.. _Iter.run_length_encode:


``Iter.run_length_encode(self) -> Iter[Tuple[T, int]]``
=======================================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length

.. code-block:: python

    >>> uncompressed = 'abbcccdddd'
    >>> Iter(uncompressed).run_length_encode().collect()
    [('a', 1), ('b', 2), ('c', 3), ('d', 4)]



.. _Iter.run_length_decode:


``Iter.run_length_decode(self) -> Iter``
========================================

See: https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length

.. code-block:: python

    >>> compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    >>> Iter(compressed).run_length_decode().collect()
    ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']



.. _Iter.map_reduce:


``Iter.map_reduce(self, keyfunc, valuefunc=None, reducefunc=None) -> Dict``
===========================================================================

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



.. _Iter.map_reduce_it:


``Iter.map_reduce_it(self, keyfunc: Callable[..., K], valuefunc: Optional[Callable[..., V]] = None, reducefunc: Optional[Callable[..., R]] = None) -> Iter[Tuple[K, R]]``
=========================================================================================================================================================================

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



.. _Iter.exactly_n:


``Iter.exactly_n(self, n, predicate=bool) -> Iter``
===================================================

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


``Iter.only(self, default=None, too_long=None) -> Any``
=======================================================

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


``Iter.take(self, n: int) -> Iter``
===================================

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


``Iter.circular_shifts(self) -> Iter``
======================================

.. _Iter.partitions:


``Iter.partitions(self) -> Iter``
=================================

.. _Iter.set_partitions:


``Iter.set_partitions(self, k=None) -> Iter``
=============================================

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

.. _Iter.consumer:


``Iter.consumer(self)``
=======================

.. _Iter.with_iter:


``Iter.with_iter(self)``
========================

.. _Iter.iter_except:


``Iter.iter_except(self)``
==========================

.. _Iter.locate:


``Iter.locate(self, pred=bool, window_size=None) -> Iter``
==========================================================

.. _Iter.rlocate:


``Iter.rlocate(self, pred=bool, window_size=None) -> Iter``
===========================================================

.. _Iter.replace:


``Iter.replace(self, pred, substitutes, count=None, window_size=1) -> Iter``
============================================================================

.. _Iter.numeric_range:


``@classmethod Iter.numeric_range(cls, *args) -> Iter``
=======================================================

.. _Iter.side_effect:


``Iter.side_effect(self, func, chunk_size=None, before=None, after=None)``
==========================================================================

.. _Iter.iterate:


``Iter.iterate(self)``
======================

.. _Iter.difference:


``Iter.difference(self, func=operator.sub, *, initial=None)``
=============================================================

.. _Iter.make_decorator:


``Iter.make_decorator(self)``
=============================

.. _Iter.SequenceView:


``Iter.SequenceView(self)``
===========================

.. _Iter.time_limited:


``Iter.time_limited(self, limit_seconds) -> Iter``
==================================================

.. _Iter.consume:


``Iter.consume(self, n: Optional[int] = None) -> Optional[Iter[T]]``
====================================================================
If n is not provided, the entire iterator is consumed and
``None`` is returned. Otherwise, an iterator will always be
returned, even if n is greater than the number of items left in
the iterator.

.. _Iter.tabulate:


``Iter.tabulate(self)``
=======================

.. _Iter.repeatfunc:


``Iter.repeatfunc(self)``
=========================

.. _Iter.wrap:


``Iter.wrap(self, ends: Sequence[T, T] = "()")``
================================================
Other examples for ends: '"' * 2, or '`' * 2, or '[]' etc.


.. _Iter.print:


``Iter.print(self, template="{i}: {v}") -> Iter[T]``
====================================================

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


.. _IterDict:


|flux| ``class IterDict(UserDict)``
***********************************



The idea here was to make a custom dict where several of
the standard dict methods return ``Iter`` instances, which can then
be chained. I'm not sure if this will be kept yet.


.. _IterDict.keys:


``IterDict.keys(self) -> Iter``
===============================

.. _IterDict.values:


``IterDict.values(self) -> Iter``
=================================

.. _IterDict.items:


``IterDict.items(self) -> Iter``
================================

.. _IterDict.update:


``IterDict.update(self, *args, **kwargs) -> IterDict``
======================================================

.. _insert_separator:


``insert_separator(iterable: Iterable[Any], glue: Any) -> Iterable[Any]``
*************************************************************************
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


``concat(iterable: Iterable[AnyStr], glue: AnyStr) -> AnyStr``
**************************************************************
Concatenate strings, bytes and bytearrays. It is careful to avoid the
problem with single bytes becoming integers, and it looks at the value
of `glue` to know whether to handle bytes or strings.

This function can raise ``ValueError`` if called with something
other than ``bytes``, ``bytearray`` or ``str``.


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


