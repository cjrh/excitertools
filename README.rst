
excitertools
************

itertools in the form of function call chaining

TOC

* `Documentation <excitertools.rst>`_
  * `Heading <excitertools.rst#heading>`_

Related projects
================

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


Documentation
=============

This is the documentation.

Here is a heading:


Heading
-------

That was the heading.

``class excitertools.Iter(x: Iterable[T])``

    Bases: ``typing.Generic``

    This is the docstring for the ``Iter`` class.

    Each of the following methods of ``Iter`` describe how they work.

    ``collect(container=<class 'list'>) -> List[T]``

        .. code-block:: python

            >>> Iter('abc').collect()
            ['a', 'b', 'c']
            >>> Iter('abc').collect(str)
            'abc'
            >>> Iter('abcaaaabbbbccc').collect(set) == {'a', 'b', 'c'}
            True

    ``classmethod open(file, mode='r', buffering=- 1, encoding=None, errors=None, newline=None, closefd=True, opener=None) -> excitertools.Iter``

        Docstring TBD

    ``classmethod range(*args) -> excitertools.Iter[int][int]``

        Docstring TBD

    ``zip(*iterables: Any) -> excitertools.Iter[typing.Tuple[~T, …]][Tuple[T, …]]``

        Docstring TBD

    ``any() -> bool``

        Docstring TBD

    ``all() -> bool``

        Docstring TBD

    ``enumerate() -> excitertools.Iter[typing.Tuple[int, ~T]][Tuple[int, T]]``

        Docstring TBD

    ``dict() -> Dict``

        Docstring TBD

    ``map(func: Union[Callable[[…], C], str]) -> excitertools.Iter[~C][C]``

        >>> result = Iter('caleb').map(lambda x: (x, ord(x))).dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

        >>> result = Iter('caleb').map('x, ord(x)').dict()
        >>> assert result == {'a': 97, 'b': 98, 'c': 99, 'e': 101, 'l': 108}

    ``filter(*args) -> excitertools.Iter[~T][T]``

        Docstring TBD

    ``reduce(func: Callable[[…], T], *args) -> T``

        Docstring TBD

    ``sum()``

        Docstring TBD

    ``concat(glue: AnyStr) -> AnyStr``

        Docstring TBD

    ``insert(glue: C) -> excitertools.Iter[typing.Union[~C, ~T]][Union[C, T]]``

        Docstring TBD

    ``classmethod count(*args) -> excitertools.Iter[int][int]``

        Docstring TBD

    ``cycle() -> excitertools.Iter[~T][T]``

        Docstring TBD

    ``classmethod repeat(elem: C, times=None) -> excitertools.Iter[~C][C]``

        Docstring TBD

    ``accumulate(func)``

        Docstring TBD

    ``chain(*iterables: Iterable[T]) -> excitertools.Iter[~T][T]``

        Docstring TBD

    ``chain_from_iterable() -> excitertools.Iter[~T][T]``

        Docstring TBD

    ``compress(selectors)``

        Docstring TBD

    ``dropwhile(pred)``

        Docstring TBD

    ``filterfalse(pred)``

        Docstring TBD

    ``groupby(key=None)``

        Docstring TBD

    ``islice(*args) -> excitertools.Iter``

        Docstring TBD

    ``starmap(func)``

        Docstring TBD

    ``takewhile(pred)``

        Docstring TBD

    ``tee(n=2)``

        Docstring TBD

    ``zip_longest(*iterables, fillvalue=None)``

        Docstring TBD

    ``chunked(n: int) -> excitertools.Iter``

        Docstring TBD

    ``ichunked(n: int) -> excitertools.Iter``

        Docstring TBD

    ``classmethod sliced(seq: Sequence, n: int) -> excitertools.Iter``

        Docstring TBD

    ``distribute(n: int) -> excitertools.Iter``

        Docstring TBD

    ``divide(n: int) -> excitertools.Iter``

        Docstring TBD

    ``split_at(pred)``

        Docstring TBD

    ``split_before(pred)``

        Docstring TBD

    ``split_after(pred)``

        Docstring TBD

    ``split_into(sizes)``

        Docstring TBD

    ``split_when(pred)``

        Docstring TBD

    ``bucket(key, validator=None)``

        Docstring TBD

    ``unzip()``

        Docstring TBD

    ``grouper(n: int, fillvalue=None) -> excitertools.Iter``

        Docstring TBD

    ``partition(pred) -> excitertools.Iter``

        Docstring TBD

    ``spy(n=1) -> Tuple[excitertools.Iter, excitertools.Iter]``

        Docstring TBD

    ``peekable() -> more_itertools.more.peekable``

        Docstring TBD

    ``seekable() -> more_itertools.more.seekable``

        Docstring TBD

    ``windowed(n, fillvalue=None, step=1) -> excitertools.Iter``

        Docstring TBD

    ``substrings()``

        Docstring TBD

    ``substrings_indexes(reverse=False)``

        Docstring TBD

    ``stagger(offsets=- 1, 0, 1, longest=False, fillvalue=None)``

        .. code-block:: python

            >>> Iter([0, 1, 2, 3]).stagger().collect()
            [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
            >>> Iter(range(8)).stagger(offsets=(0, 2, 4)).collect()
            [(0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7)]
            >>> Iter([0, 1, 2, 3]).stagger(longest=True).collect()
            [(None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]

    ``pairwise()``

        See `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise>`_

        .. code-block:: python

            >>> Iter.count().pairwise().take(4).collect()
            [(0, 1), (1, 2), (2, 3), (3, 4)]

    ``count_cycle(n=None) -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle>`_

        .. code-block:: python

            >>> Iter('AB').count_cycle(3).collect()
            [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]

    ``intersperse(e, n=1) -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse>`_

        .. code-block:: python

            >>> Iter([1, 2, 3, 4, 5]).intersperse('!').collect()
            [1, '!', 2, '!', 3, '!', 4, '!', 5]

            >>> Iter([1, 2, 3, 4, 5]).intersperse(None, n=2).collect()
            [1, 2, None, 3, 4, None, 5]

    ``padded(fillvalue: Optional[C] = None, n: Optional[int] = None, next_multiple: bool = False) -> excitertools.Iter[typing.Union[~C, ~T]][Union[C, T]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded>`_

        .. code-block:: python

            >>> Iter([1, 2, 3]).padded('?', 5).collect()
            [1, 2, 3, '?', '?']

            >>> Iter([1, 2, 3, 4]).padded(n=3, next_multiple=True).collect()
            [1, 2, 3, 4, None, None]

    ``repeat_last(default=None) -> excitertools.Iter[~T][T]``

        `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeat_last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeat_last>`_

        .. code-block:: python

            >>> Iter(range(3)).repeat_last().islice(5).collect()
            [0, 1, 2, 2, 2]

            >>> Iter(range(0)).repeat_last(42).islice(5).collect()
            [42, 42, 42, 42, 42]

    ``adjacent(pred, distance=1) -> excitertools.Iter[typing.Tuple[bool, ~T]][Tuple[bool, T]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent>`_

        .. code-block:: python

            >>> Iter(range(6)).adjacent(lambda x: x == 3).collect()
            [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]

            >>> Iter(range(6)).adjacent(lambda x: x == 3, distance=2).collect()
            [(False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)]

    ``groupby_transform(keyfunc: Optional[Callable[[…], K]] = None, valuefunc: Optional[Callable[[…], V]] = None) -> excitertools.Iter[typing.Tuple[~K, typing.Iterable[~V]]][Tuple[K, Iterable[V]]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.groupby_transform <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.groupby_transform>`_

        This example has been modified somewhat from the original. We’re using
        ``starmap`` here to “unzip” the tuples produced by the group
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

    ``padnone() -> excitertools.Iter[typing.Union[~T, NoneType]][Optional[T]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone>`_

        .. code-block:: python

            >>> Iter(range(3)).padnone().take(5).collect()
            [0, 1, 2, None, None]

    ``ncycles(n) -> excitertools.Iter[~T][T]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles>`_

        .. code-block:: python

            >>> Iter(['a', 'b']).ncycles(3).collect()
            ['a', 'b', 'a', 'b', 'a', 'b']

    ``collapse(base_type=None, levels=None) -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.collapse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.collapse>`_

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

    ``classmethod sort_together(iterables, key_list=0, reverse=False)``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sort_together <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sort_together>`_

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

    ``classmethod interleave(*iterables) -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave>`_

        Classmethod form:

        .. code-block:: python

            >>> Iter.interleave([1, 2, 3], [4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7]

        Instancemethod form:

        .. code-block:: python

            >>> Iter([1, 2, 3]).interleave([4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7]

    ``classmethod interleave_longest(*iterables) -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave_longest <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave_longest>`_

        Classmethod form:

        .. code-block:: python

            >>> Iter.interleave_longest([1, 2, 3], [4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7, 3, 8]

        Instancemethod form:

        .. code-block:: python

            >>> Iter([1, 2, 3]).interleave_longest([4, 5], [6, 7, 8]).collect()
            [1, 4, 6, 2, 5, 7, 3, 8]

    ``classmethod zip_offset(*iterables, offsets=None, longest=False, fillvalue=None) -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_offset <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_offset>`_

        .. code-block:: python

            >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1)).collect()
            [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e')]

            >>> Iter.zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True).collect()
            [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e'), (None, 'f')]

    ``dotproduct(vec2: Iterable)``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.dotproduct <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.dotproduct>`_

        .. code-block:: python

            >>> Iter([10, 10]).dotproduct([20, 20])
            400

    ``flatten() -> excitertools.Iter[~T][T]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten>`_

        .. code-block:: python

            >>> Iter([[0, 1], [2, 3]]).flatten().collect()
            [0, 1, 2, 3]

    ``classmethod roundrobin(*iterables: C) -> excitertools.Iter[typing.Union[~C, ~T]][Union[C, T]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.roundrobin <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.roundrobin>`_

        Classmethod form:

        .. code-block:: python

            >>> Iter.roundrobin('ABC', 'D', 'EF').collect()
            ['A', 'D', 'E', 'B', 'F', 'C']

        Instancemethod form:

        .. code-block:: python

            >>> Iter('ABC').roundrobin('D', 'EF').collect()
            ['A', 'D', 'E', 'B', 'F', 'C']

    ``prepend(value: C) -> excitertools.Iter[typing.Union[~C, ~T]][Union[C, T]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.prepend <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.prepend>`_

        .. code-block:: python

            >>> value = '0'
            >>> iterator = ['1', '2', '3']
            >>> Iter(iterator).prepend(value).collect()
            ['0', '1', '2', '3']

    ``ilen() -> int``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ilen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ilen>`_

        .. code-block:: python

            >>> Iter(x for x in range(1000000) if x % 3 == 0).ilen()
            333334

    ``unique_to_each() -> excitertools.Iter[~T][T]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each>`_

        .. code-block:: python

            >>> Iter([{'A', 'B'}, {'B', 'C'}, {'B', 'D'}]).unique_to_each().collect()
            [['A'], ['C'], ['D']]

            >>> Iter(["mississippi", "missouri"]).unique_to_each().collect()
            [['p', 'p'], ['o', 'u', 'r']]

    ``sample(k=1, weights=None) -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sample <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sample>`_

        .. code-block:: python

            >>> iterable = range(100)
            >>> Iter(iterable).sample(5).collect()  
            [81, 60, 96, 16, 4]

            >>> iterable = range(100)
            >>> weights = (i * i + 1 for i in range(100))
            >>> Iter(iterable).sample(5, weights=weights)  
            [79, 67, 74, 66, 78]

            >>> data = "abcdefgh"
            >>> weights = range(1, len(data) + 1)
            >>> Iter(data).sample(k=len(data), weights=weights)  
            ['c', 'a', 'b', 'e', 'g', 'd', 'h', 'f']


            >>> # This one just to let the doctest run
            >>> iterable = range(100)
            >>> Iter(iterable).sample(5).map(lambda x: 0 <= x < 100).all()
            True

    ``consecutive_groups(ordering=<function Iter.<lambda>>)``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consecutive_groups <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consecutive_groups>`_

        .. code-block:: python

            >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
            >>> Iter(iterable).consecutive_groups().map(lambda g: list(g)).print('{v}').consume()
            [1]
            [10, 11, 12]
            [20]
            [30, 31, 32, 33]
            [40]

    ``run_length_encode() -> excitertools.Iter[typing.Tuple[~T, int]][Tuple[T, int]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length>`_

        .. code-block:: python

            >>> uncompressed = 'abbcccdddd'
            >>> Iter(uncompressed).run_length_encode().collect()
            [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

    ``run_length_decode() -> excitertools.Iter``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length>`_

        .. code-block:: python

            >>> compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
            >>> Iter(compressed).run_length_decode().collect()
            ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

    ``map_reduce(keyfunc, valuefunc=None, reducefunc=None) -> Dict``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce>`_

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

    ``map_reduce_it(keyfunc: Callable[[…], K], valuefunc: Optional[Callable[[…], V]] = None, reducefunc: Optional[Callable[[…], R]] = None) -> excitertools.Iter[typing.Tuple[~K, ~R]][Tuple[K, R]]``

        See: `https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce>`_

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

    ``exactly_n(n, predicate=<class 'bool'>) -> excitertools.Iter``

    ``all_equal()``

    ``first_true()``

    ``quantify()``

    ``islice_extended(*args)``

    ``first()``

    ``last()``

    ``one()``

    ``only(default=None, too_long=None) -> Any``

    ``strip(pred)``

    ``lstrip(pred)``

    ``rstrip(pred)``

    ``filter_except(validator, *exceptions)``

    ``map_except(function, *exceptions)``

    ``nth_or_last()``

    ``nth(n, default=None)``

    ``take(n: int) -> excitertools.Iter``

    ``tail()``

    ``unique_everseen()``

    ``unique_justseen()``

    ``distinct_permutations()``

    ``distinct_combinations(r)``

    ``circular_shifts() -> excitertools.Iter``

    ``partitions() -> excitertools.Iter``

    ``set_partitions(k=None) -> excitertools.Iter``

    ``powerset()``

    ``random_product()``

    ``random_permutation()``

    ``random_combination()``

    ``random_combination_with_replacement()``

    ``nth_combination()``

    ``always_iterable()``

    ``always_reversible()``

    ``consumer()``

    ``with_iter()``

    ``iter_except()``

    ``locate(pred=<class 'bool'>, window_size=None) -> excitertools.Iter``

    ``rlocate(pred=<class 'bool'>, window_size=None) -> excitertools.Iter``

    ``replace(pred, substitutes, count=None, window_size=1) -> excitertools.Iter``

    ``classmethod numeric_range(*args) -> excitertools.Iter``

    ``side_effect(func, chunk_size=None, before=None, after=None)``

    ``iterate()``

    ``difference(func=<built-in function sub>, *, initial=None)``

    ``make_decorator()``

    ``SequenceView()``

    ``time_limited(limit_seconds) -> excitertools.Iter``

    ``consume(n: Optional[int] = None) -> Optional[excitertools.Iter[~T][T]]``

        If n is not provided, the entire iterator is consumed and
        ``None`` is returned. Otherwise, an iterator will always be
        returned, even if n is greater than the number of items left in
        the iterator.

    ``tabulate()``

    ``repeatfunc()``

    ``wrap(ends: Sequence[T, T] = '()')``

        Other examples for ends: ‘”’ * 2, or ‘`’ * 2, or ‘[]’ etc.

    ``print(template='{i}: {v}') -> excitertools.Iter[~T][T]``

        Printing during the execution of an iterator. Mostly useful
        for debugging. Returns another iterator instance through which
        the original data is passed unchanged. This means you can include
        a *print()* step as necessary to observe data during iteration.

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

``excitertools.insert_separator(iterable: Iterable[Any], glue: Any) -> Iterable[Any]``

    Similar functionality can be obtained with, e.g.,
    interleave, as in

    >>> result = Iter('caleb').interleave(Iter.repeat('x')).collect()
    >>> result == list('cxaxlxexbx')
    True

    But you’ll see a trailing “x” there, which join avoids. join
    makes sure to only add the glue separator if another element
    has arrived.

    It can handle strings without any special considerations, but it doesn’t
    do any special handling for bytes and bytearrays. For that, rather
    look at *concat()*.

``excitertools.concat(iterable: Iterable[AnyStr], glue: AnyStr) -> AnyStr``

    Concatenate strings, bytes and bytearrays. It is careful to avoid the
    problem with single bytes becoming integers, and it looks at the value
    of *glue* to know whether to handle bytes or strings.
