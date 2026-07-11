"""Vertical-slice tests for the async fluent interface (AIter / amap).

These exercise the sync->async "ratchet": a chain starts synchronous,
crosses the boundary at ``amap`` into an ``AIter``, and finishes at the
async sink ``acollect``. The synchronous prefix is pulled lazily by the
async engine; the post-boundary ``filter``/``map`` are async combinators
over ordinary (sync) callables.

No pytest-asyncio dependency: each test is a plain sync function that
drives its coroutine with ``asyncio.run``.
"""

import asyncio
import time
import tempfile
import os

from excitertools import Iter, AIter, fileinput


# ---- fake async "network" ----
async def url_fetch(url):
    await asyncio.sleep(0.1)              # pretend network latency
    return f"CONTENT[{url}]"


def is_valid(content):
    return "example.org" not in content  # drop one host


def process_each_result(content):
    return content.lower()


async def _run_example():
    # Build two temp files of URLs, like urls1.txt / urls2.txt
    d = tempfile.mkdtemp()
    f1 = os.path.join(d, "urls1.txt")
    f2 = os.path.join(d, "urls2.txt")
    with open(f1, "w") as fh:
        fh.write("  http://example.com  \n http://example.org \n")
    with open(f2, "w") as fh:
        fh.write("http://example.net\nhttp://example.io\n")

    # ---- THE EXAMPLE ----
    t0 = time.perf_counter()
    results = await (
        fileinput([f1, f2])
        .map(str.strip)                          # sync zone
        .amap(url_fetch, concurrency=20)         # boundary -> AIter
        .filter(is_valid)                        # async filter, sync predicate
        .map(process_each_result)                # async map, sync callable
        .acollect()                              # async sink
    )
    dt = time.perf_counter() - t0
    return results, dt, f1


def test_fileinput_async_example():
    results, dt, _ = asyncio.run(_run_example())

    # order preserved (com, net, io -- org filtered out)
    assert results == [
        "content[http://example.com]",
        "content[http://example.net]",
        "content[http://example.io]",
    ], results

    # concurrency is real: 4 x 0.1s complete in ~0.1s, not ~0.4s
    assert dt < 0.25, f"expected concurrent execution, took {dt:.3f}s"


def test_type_ratchet():
    async def go():
        d = tempfile.mkdtemp()
        f1 = os.path.join(d, "urls.txt")
        with open(f1, "w") as fh:
            fh.write("http://example.com\n")

        chain = fileinput([f1]).map(str.strip).amap(url_fetch)
        assert isinstance(chain, AIter), type(chain)
        # sync-named combinators stay AIter via type(self) propagation
        assert isinstance(chain.filter(is_valid), AIter)
        assert isinstance(chain.filter(is_valid).map(process_each_result), AIter)

    asyncio.run(go())


def test_collect_guard_on_aiter():
    import pytest

    with pytest.raises(TypeError, match="acollect"):
        Iter(range(3)).amap(url_fetch).collect()


def test_sync_for_guard_on_aiter():
    import pytest

    with pytest.raises(TypeError, match="async for"):
        for _ in Iter(range(3)).amap(url_fetch):
            pass


def test_unordered_returns_all_items():
    async def go():
        return await Iter(range(6)).amap(
            url_fetch, concurrency=6, ordered=False
        ).acollect()

    unordered = asyncio.run(go())
    assert len(unordered) == 6
    assert sorted(unordered) == sorted(f"CONTENT[{i}]" for i in range(6))


def test_pure_sync_chain_unaffected():
    # The async machinery must not change existing synchronous behaviour.
    assert Iter(range(5)).map(lambda x: x * 2).filter(lambda x: x > 3).collect() == [4, 6, 8]


# ---------------------------------------------------------------------------
# Combinators: each Iter combinator re-implemented on AIter with async bodies.
# ---------------------------------------------------------------------------

async def _adouble(x):
    return x * 2


def _run(coro_factory):
    return asyncio.run(coro_factory())


def test_to_async_lift_then_combinators():
    async def go():
        return await (
            Iter(range(10))
            .to_async()               # explicit lift, no async callable
            .filter(lambda x: x % 2 == 0)
            .map(_adouble)            # async mapper on AIter
            .acollect()
        )
    assert _run(go) == [0, 4, 8, 12, 16]


def test_enumerate_take_tail():
    async def go():
        e = await Iter("abcde").to_async().enumerate(start=1).acollect()
        t = await Iter(range(10)).to_async().take(3).acollect()
        tl = await Iter(range(10)).to_async().tail(3).acollect()
        return e, t, tl
    e, t, tl = _run(go)
    assert e == [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
    assert t == [0, 1, 2]
    assert tl == [7, 8, 9]


def test_islice_full_form():
    async def go():
        return await Iter(range(20)).to_async().islice(2, 12, 3).acollect()
    assert _run(go) == [2, 5, 8, 11]


def test_takewhile_dropwhile_filterfalse():
    async def go():
        tw = await Iter([1, 2, 3, 1]).to_async().takewhile(lambda x: x < 3).acollect()
        dw = await Iter([1, 2, 3, 1]).to_async().dropwhile(lambda x: x < 3).acollect()
        ff = await Iter(range(6)).to_async().filterfalse(lambda x: x % 2).acollect()
        return tw, dw, ff
    tw, dw, ff = _run(go)
    assert tw == [1, 2]
    assert dw == [3, 1]
    assert ff == [0, 2, 4]


def test_chunked_flatten_chain_prepend():
    async def go():
        ch = await Iter(range(7)).to_async().chunked(3).acollect()
        fl = await Iter([[1, 2], [3], [4, 5]]).to_async().flatten().acollect()
        cn = await Iter([1, 2]).to_async().chain([3, 4], Iter([5]).to_async()).acollect()
        pp = await Iter([2, 3]).to_async().prepend(1).acollect()
        return ch, fl, cn, pp
    ch, fl, cn, pp = _run(go)
    assert ch == [[0, 1, 2], [3, 4, 5], [6]]
    assert fl == [1, 2, 3, 4, 5]
    assert cn == [1, 2, 3, 4, 5]
    assert pp == [1, 2, 3]


def test_starmap_and_astarmap():
    async def add(a, b):
        return a + b

    async def go():
        sm = await Iter([(1, 2), (3, 4)]).to_async().starmap(add).acollect()
        asm = await Iter([(1, 2), (3, 4), (5, 6)]).astarmap(add, concurrency=3).acollect()
        return sm, asm
    sm, asm = _run(go)
    assert sm == [3, 7]
    assert asm == [3, 7, 11]


def test_afilter_concurrent_predicate():
    async def is_valid(content):
        await asyncio.sleep(0.05)
        return "example.org" not in content

    async def go():
        return await (
            Iter(["http://example.com", "http://example.org", "http://example.net"])
            .afilter(is_valid, concurrency=10)
            .acollect()
        )
    assert _run(go) == ["http://example.com", "http://example.net"]


def test_accumulate():
    async def go():
        a = await Iter([1, 2, 3, 4]).to_async().accumulate().acollect()
        b = await Iter([1, 2, 3]).to_async().accumulate(initial=10).acollect()
        return a, b
    a, b = _run(go)
    assert a == [1, 3, 6, 10]
    assert b == [10, 11, 13, 16]      # initial yielded first, matching itertools


def test_reducing_sinks():
    async def go():
        return {
            "sum": await Iter(range(5)).to_async().asum(),
            "max": await Iter([3, 1, 4, 1, 5]).to_async().amax(),
            "min": await Iter([3, 1, 4, 1, 5]).to_async().amin(),
            "reduce": await Iter([1, 2, 3, 4]).to_async().areduce(lambda a, b: a * b),
            "count": await Iter(range(7)).to_async().acount(),
            "dict": await Iter([("a", 1), ("b", 2)]).to_async().adict(),
            "concat": await Iter(["a", "b", "c"]).to_async().aconcat("-"),
            "first": await Iter(range(5)).to_async().afirst(),
            "any": await Iter([0, 0, 3]).to_async().aany(),
            "all": await Iter([1, 2, 3]).to_async().aall(),
        }
    r = _run(go)
    assert r == {
        "sum": 10, "max": 5, "min": 1, "reduce": 24, "count": 7,
        "dict": {"a": 1, "b": 2}, "concat": "a-b-c", "first": 0,
        "any": True, "all": True,
    }


def test_aforeach_side_effect():
    seen = []

    async def go():
        await Iter(range(4)).to_async().aforeach(seen.append)
    _run(go)
    assert seen == [0, 1, 2, 3]


def test_more_sync_sink_guards():
    import pytest
    for method, twin in [("sum", "asum"), ("first", "afirst"),
                         ("dict", "adict"), ("concat", "aconcat")]:
        with pytest.raises(TypeError, match=twin):
            getattr(Iter(range(3)).to_async(), method)()


# ---------------------------------------------------------------------------
# Exception handling in the bounded-concurrency engine.
#
# When a scheduled call raises, the exception must surface to the awaiter and
# every still-in-flight task must be cancelled and drained -- not orphaned
# (which leaks work and emits "Task was destroyed but it is pending"). These
# assert the cancellation *mechanism*, which is the reliable proxy for "no
# leak": if the siblings were left pending they would run to completion and
# populate ``completed`` instead of ``cancelled``.
# ---------------------------------------------------------------------------

def _explode_after(fail_index):
    """Return (work, started, completed, cancelled) for a failing map.

    ``work(i)`` raises on ``i == fail_index`` after a short delay; every
    other item sleeps long enough that it can only finish if it was *not*
    cancelled. The three lists record the fate of each scheduled call.
    """
    import asyncio
    started, completed, cancelled = [], [], []

    async def work(i):
        started.append(i)
        try:
            if i == fail_index:
                await asyncio.sleep(0.01)
                raise ValueError(f"boom-{i}")
            await asyncio.sleep(1.0)      # long: only completes if never cancelled
            completed.append(i)
            return i
        except asyncio.CancelledError:
            cancelled.append(i)
            raise

    return work, started, completed, cancelled


def test_amap_exception_cancels_inflight_ordered():
    import pytest
    work, started, completed, cancelled = _explode_after(0)

    async def go():
        with pytest.raises(ValueError, match="boom-0"):
            await Iter(range(5)).amap(work, concurrency=5).acollect()

    _run(lambda: go())
    # the failing item surfaced; every other scheduled task was cancelled,
    # none ran to completion (they would have needed a full second)
    assert set(started) == {0, 1, 2, 3, 4}
    assert completed == []
    assert set(cancelled) == {1, 2, 3, 4}


def test_amap_exception_cancels_inflight_unordered():
    import pytest
    work, started, completed, cancelled = _explode_after(0)

    async def go():
        with pytest.raises(ValueError, match="boom-0"):
            await Iter(range(5)).amap(
                work, concurrency=5, ordered=False
            ).acollect()

    _run(lambda: go())
    assert completed == []
    assert set(cancelled) == {1, 2, 3, 4}


def test_ordered_surfaces_first_input_error_not_first_to_fail():
    # Two calls fail; ordered mode must raise the one earliest in INPUT
    # order (index 1), even though index 3 fails sooner in wall-clock.
    import asyncio
    import pytest

    async def work(i):
        if i == 1:
            await asyncio.sleep(0.05)
            raise ValueError("slow-early")
        if i == 3:
            await asyncio.sleep(0.01)
            raise ValueError("fast-late")
        await asyncio.sleep(0.02)
        return i

    async def go():
        with pytest.raises(ValueError, match="slow-early"):
            await Iter(range(5)).amap(work, concurrency=5).acollect()

    _run(lambda: go())


def test_explicit_aclose_cancels_inflight():
    # A consumer that stops early and *closes* the engine (aclosing, or any
    # explicit ``aclose``) must not leave upstream concurrent work running:
    # the generator's finally cancels every task still in flight. This is
    # the deterministic cleanup contract; a bare ``break`` defers cleanup to
    # GC / loop-shutdown (see test_bare_break_defers_cleanup).
    import asyncio
    from contextlib import aclosing
    started, completed, cancelled = [], [], []

    async def work(i):
        started.append(i)
        try:
            await asyncio.sleep(0.05 * (i + 1))   # later items finish later
            completed.append(i)
            return i
        except asyncio.CancelledError:
            cancelled.append(i)
            raise

    async def go():
        chain = Iter(range(6)).amap(work, concurrency=6)
        out = []
        async with aclosing(chain.__aiter__()) as g:
            async for x in g:
                out.append(x)
                break                             # stop after the first result
        return out

    got = _run(lambda: go())
    assert got == [0]
    # only item 0 was awaited to a result; the rest were cancelled promptly
    assert set(started) == {0, 1, 2, 3, 4, 5}
    assert completed == [0]
    assert set(cancelled) == {1, 2, 3, 4, 5}


def test_concurrency_ceiling_is_bounded():
    # The central promise of amap: never more than ``concurrency`` calls in
    # flight at once. The fileinput test proves work IS concurrent; this
    # proves it is BOUNDED -- a bug that ignored the limit and scheduled the
    # whole source at once would pass the timing test but fail here.
    import asyncio
    inflight = 0
    peak = 0

    async def work(i):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.01)
        inflight -= 1
        return i

    async def go():
        return await Iter(range(20)).amap(work, concurrency=3).acollect()

    got = _run(lambda: go())
    assert got == list(range(20))          # bounded, still order-preserving
    assert peak == 3, f"expected to reach and not exceed 3, peak={peak}"


def test_pipeline_is_lazy_not_materialised():
    # The pipeline must pull incrementally. A large source paired with
    # take(2) must NOT be drained in full -- if any combinator buffered the
    # whole source internally, ``produced`` would reach 1000.
    import asyncio
    produced = []

    async def source():
        for i in range(1000):
            produced.append(i)
            yield i
            await asyncio.sleep(0)

    async def go():
        return await AIter(source()).map(_adouble).take(2).acollect()

    got = _run(lambda: go())
    assert got == [0, 2]
    assert len(produced) < 10, f"pipeline over-pulled source: {len(produced)} items"


def test_ordered_reassembles_under_inverse_latency():
    # First input is the SLOWEST to resolve; ordered mode must still yield
    # it first. A naive as-completed implementation would surface the fast
    # last item first -- this is what distinguishes ordered from unordered.
    import asyncio

    async def work(i):
        await asyncio.sleep(0.02 * (5 - i))    # item 0 slowest ... item 4 fastest
        return i

    async def go():
        return await Iter(range(5)).amap(work, concurrency=5).acollect()

    assert _run(lambda: go()) == [0, 1, 2, 3, 4]


def test_amap_accepts_sync_callable():
    # The boundary is colour-agnostic: amap lifts a plain sync function just
    # as it lifts a coroutine function (via _maybe_await).
    async def go():
        return await Iter(range(4)).amap(lambda x: x * 10, concurrency=2).acollect()

    assert _run(lambda: go()) == [0, 10, 20, 30]


def test_empty_source_edges():
    import pytest

    async def go():
        return {
            "amap": await Iter([]).amap(_adouble).acollect(),
            "map": await Iter([]).to_async().map(_adouble).acollect(),
            "sum": await Iter([]).to_async().asum(),
            "count": await Iter([]).to_async().acount(),
            "first_default": await Iter([]).to_async().afirst("none"),
            "any": await Iter([]).to_async().aany(),
            "all": await Iter([]).to_async().aall(),
            "concat": await Iter([]).to_async().aconcat("-"),
        }

    r = _run(go)
    assert r == {
        "amap": [], "map": [], "sum": 0, "count": 0, "first_default": "none",
        "any": False, "all": True, "concat": "",
    }

    # errors preserved on empty: max/min/first/reduce with no default
    for coro_factory in [
        lambda: Iter([]).to_async().amax(),
        lambda: Iter([]).to_async().amin(),
        lambda: Iter([]).to_async().afirst(),
    ]:
        with pytest.raises(ValueError):
            _run(coro_factory)
    with pytest.raises(TypeError):
        _run(lambda: Iter([]).to_async().areduce(lambda a, b: a + b))


def test_all_overridden_combinators_stay_aiter():
    # The one-way ratchet: every combinator re-implemented on AIter must
    # return an AIter (via ``type(self)`` propagation), never drop back to a
    # plain Iter. This is the guard against a port accidentally writing
    # ``Iter(...)`` instead of ``type(self)(...)``. Each lambda builds one
    # combinator on a suitably-shaped async source; add a line as each new
    # combinator is ported.
    def A(src):
        return Iter(src).to_async()

    builders = {
        # -- core (first tranche) --
        "map": lambda: A(range(4)).map(lambda x: x),
        "filter": lambda: A(range(4)).filter(lambda x: True),
        "enumerate": lambda: A(range(4)).enumerate(),
        "takewhile": lambda: A(range(4)).takewhile(lambda x: True),
        "dropwhile": lambda: A(range(4)).dropwhile(lambda x: False),
        "filterfalse": lambda: A(range(4)).filterfalse(lambda x: False),
        "islice": lambda: A(range(4)).islice(2),
        "take": lambda: A(range(4)).take(2),
        "tail": lambda: A(range(4)).tail(2),
        "chunked": lambda: A(range(4)).chunked(2),
        "chain": lambda: A(range(4)).chain([9]),
        "prepend": lambda: A(range(4)).prepend(-1),
        "accumulate": lambda: A(range(4)).accumulate(),
        "starmap": lambda: A([(1, 2)]).starmap(lambda a, b: a + b),
        "flatten": lambda: A([[1], [2]]).flatten(),
        # -- free delegators (inherited via self.filter) --
        "filter_gt": lambda: A(range(4)).filter_gt(1),
        "filter_ge": lambda: A(range(4)).filter_ge(1),
        "filter_lt": lambda: A(range(4)).filter_lt(3),
        "filter_le": lambda: A(range(4)).filter_le(3),
        "filter_eq": lambda: A(range(4)).filter_eq(2),
        "filter_ne": lambda: A(range(4)).filter_ne(2),
        "filter_in": lambda: A(range(4)).filter_in([1, 2]),
        "filter_ni": lambda: A(range(4)).filter_ni([1, 2]),
        # -- streaming combinators (tranche 1) --
        "starfilter": lambda: A([(1, 2)]).starfilter(lambda a, b: True),
        "filter_except": lambda: A(["1", "x"]).filter_except(int, ValueError),
        "map_except": lambda: A(["1", "x"]).map_except(int, ValueError),
        "compress": lambda: A(range(4)).compress([1, 0, 1, 0]),
        "chain_from_iterable": lambda: A([[1], [2]]).chain_from_iterable(),
        "intersperse": lambda: A(range(4)).intersperse("x"),
        "pairwise": lambda: A(range(4)).pairwise(),
        "unique_everseen": lambda: A("aabb").unique_everseen(),
        "unique_justseen": lambda: A("aabb").unique_justseen(),
        "side_effect": lambda: A(range(4)).side_effect(lambda x: None),
        "padnone": lambda: A(range(2)).padnone().take(3),
        "repeat_last": lambda: A(range(2)).repeat_last().take(3),
        "cycle": lambda: A(range(2)).cycle().take(3),
        "ncycles": lambda: A(range(2)).ncycles(2),
        "count_cycle": lambda: A("ab").count_cycle(2),
        "lstrip": lambda: A([0, 1]).lstrip(lambda x: x == 0),
        "rstrip": lambda: A([1, 0]).rstrip(lambda x: x == 0),
        "strip": lambda: A([0, 1, 0]).strip(lambda x: x == 0),
        "run_length_encode": lambda: A("aabb").run_length_encode(),
        "run_length_decode": lambda: A([("a", 2)]).run_length_decode(),
        "consecutive_groups": lambda: A([1, 2, 4]).consecutive_groups(),
        "locate": lambda: A([0, 1]).locate(),
        "difference": lambda: A([1, 3, 6]).difference(),
        "grouper": lambda: A("abcd").grouper(2),
        "ichunked": lambda: A(range(4)).ichunked(2),
        "groupby": lambda: A("aabb").groupby(),
        "split_at": lambda: A(range(4)).split_at(lambda x: x == 2),
        "split_before": lambda: A(range(4)).split_before(lambda x: x == 2),
        "split_after": lambda: A(range(4)).split_after(lambda x: x == 2),
        "split_when": lambda: A(range(4)).split_when(lambda a, b: b < a),
        # -- buffered combinators (tranche 2) --
        "windowed": lambda: A(range(4)).windowed(2),
        "substrings": lambda: A("abc").substrings(),
        "substrings_indexes": lambda: A("abc").substrings_indexes(),
        "stagger": lambda: A(range(4)).stagger(),
        "padded": lambda: A(range(2)).padded("x", 4),
        "adjacent": lambda: A(range(4)).adjacent(lambda x: x == 2),
        "collapse": lambda: A([1, [2, [3]]]).collapse(),
        "constrained_batches": lambda: A(range(4)).constrained_batches(2, get_len=lambda x: 1),
        "sort_together": lambda: A([3, 1, 2]).sort_together([["c", "a", "b"]]),
        "islice_extended": lambda: A(range(6)).islice_extended(-2, None),
        "rlocate": lambda: A([0, 1]).rlocate(),
        "replace": lambda: A([0, 1]).replace(lambda x: x == 0, [9]),
        "distinct_permutations": lambda: A("aab").distinct_permutations(),
        "distinct_combinations": lambda: A(range(4)).distinct_combinations(2),
        "circular_shifts": lambda: A(range(3)).circular_shifts(),
        "partitions": lambda: A(range(3)).partitions(),
        "set_partitions": lambda: A(range(3)).set_partitions(2),
        "powerset": lambda: A(range(3)).powerset(),
        "always_reversible": lambda: A(range(3)).always_reversible(),
        "sample": lambda: A(range(5)).sample(2),
        "sorted": lambda: A([3, 1, 2]).sorted(),
        "reversed": lambda: A(range(3)).reversed(),
        "insert": lambda: A("abc").insert("-"),
        "strictly_n": lambda: A(range(3)).strictly_n(3),
        "map_reduce_it": lambda: A("aAbB").map_reduce_it(str.lower),
        # -- multi-output combinators (tranche 2) --
        "groupby_transform": lambda: A("aabb").groupby_transform(),
        "distribute": lambda: A(range(4)).distribute(2),
        "divide": lambda: A(range(4)).divide(2),
        "unzip": lambda: A([(1, 2), (3, 4)]).unzip(),
        "partition": lambda: A(range(4)).partition(lambda x: x % 2 == 0),
        # -- multi-iterable & buffered selectors (tranche 4) --
        "zip": lambda: A(range(4)).zip(range(4)),
        "zip_longest": lambda: A(range(4)).zip_longest(range(6)),
        "split_into": lambda: A(range(6)).split_into([2, 2, 2]),
        "print": lambda: A(range(3)).print(),
        "nth_combination": lambda: A(range(5)).nth_combination(2, 3),
        "random_combination": lambda: A(range(5)).random_combination(2),
        "random_combination_with_replacement": lambda: A(range(5)).random_combination_with_replacement(2),
        "random_permutation": lambda: A(range(5)).random_permutation(),
        "unique_to_each": lambda: A([{1, 2}, {2, 3}]).unique_to_each(),
        "wrap": lambda: A("abc").wrap("()"),
    }
    for name, build in builders.items():
        c = build()
        assert type(c) is AIter, f"{name} dropped to {type(c).__name__}"


def test_take_over_concurrent_source_overpulls_by_one():
    # KNOWN BEHAVIOUR (documented, not a bug): islice/take detects its stop
    # boundary by pulling one item *past* the last it yields. Over a
    # concurrent source that means the boundary-probe task is awaited to
    # completion rather than cancelled -- take(1) completes items 0 AND 1.
    # Sync islice hides this (pure); concurrent islice cannot. Guarding it
    # here so a future islice rewrite that removes the over-pull trips this
    # test and forces a conscious decision.
    import asyncio
    completed = []

    async def work(i):
        await asyncio.sleep(0.02 * (i + 1))
        completed.append(i)
        return i

    async def go():
        return await Iter(range(6)).amap(work, concurrency=6).take(1).acollect()

    got = _run(lambda: go())
    assert got == [0]
    assert 1 in completed          # the boundary-probe ran to completion
    assert 5 not in completed      # deep-tail items were still cancelled


def test_ported_combinators_match_more_itertools():
    # Parity: each ported combinator must produce exactly what the
    # synchronous more_itertools function produces on the same input.
    import more_itertools as mit
    import itertools

    def A(src):
        return Iter(src).to_async()

    cases = [
        (lambda: A(range(5)).pairwise(), list(mit.pairwise(range(5)))),
        (lambda: A(range(5)).compress([1, 0, 1, 0, 1]),
         list(itertools.compress(range(5), [1, 0, 1, 0, 1]))),
        (lambda: A("AAAABBBCCD").unique_everseen(),
         list(mit.unique_everseen("AAAABBBCCD"))),
        (lambda: A("AAAABBBCCD").unique_justseen(),
         list(mit.unique_justseen("AAAABBBCCD"))),
        (lambda: A(range(4)).intersperse("x"),
         list(mit.intersperse("x", range(4)))),
        (lambda: A("aaabbc").run_length_encode(),
         list(mit.run_length.encode("aaabbc"))),
        (lambda: A(range(6)).split_when(lambda a, b: a > b),
         list(mit.split_when(range(6), lambda a, b: a > b))),
        (lambda: A(range(6)).windowed(3),
         list(mit.windowed(range(6), 3))),
        (lambda: A(range(4)).stagger(),
         list(mit.stagger(range(4)))),
        (lambda: A([1, [2, [3, 4]], 5]).collapse(),
         list(mit.collapse([1, [2, [3, 4]], 5]))),
        (lambda: A("aab").distinct_permutations(),
         list(mit.distinct_permutations("aab"))),
        (lambda: A(range(3)).powerset(),
         list(mit.powerset(range(3)))),
        (lambda: A(range(10)).islice_extended(-3, None),
         list(mit.islice_extended(range(10), -3, None))),
        (lambda: A(range(4)).zip(range(6)),
         list(zip(range(4), range(6)))),
        (lambda: A(range(4)).zip("abcd", range(10)),
         list(zip(range(4), "abcd", range(10)))),
        (lambda: A(range(3)).zip_longest(range(5), fillvalue=-1),
         list(itertools.zip_longest(range(3), range(5), fillvalue=-1))),
        (lambda: A(range(9)).split_into([2, 3, None]),
         list(mit.split_into(range(9), [2, 3, None]))),
        (lambda: A(range(5)).nth_combination(2, 3),
         list(mit.nth_combination(range(5), 2, 3))),
        (lambda: A([{1, 2}, {2, 3}, {1, 3}]).unique_to_each(),
         list(mit.unique_to_each({1, 2}, {2, 3}, {1, 3}))),
        (lambda: A("abc").wrap("()"), list("(abc)")),
        (lambda: A(range(5)).wrap([range(3), range(3)]),
         [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2]),
    ]
    for i, (build, expected) in enumerate(cases):
        got = _run(lambda b=build: b().acollect())
        assert got == expected, f"case {i}: {got!r} != {expected!r}"


def test_streaming_combinators_stay_lazy_over_infinite_source():
    # Streaming combinators must not materialise: they compose over an
    # unbounded async source and terminate under a downstream take().
    async def counter():
        i = 0
        while True:
            yield i
            i += 1

    async def go():
        pw = await AIter(counter()).pairwise().take(3).acollect()
        ue = await AIter(counter()).map(lambda x: x % 2).unique_justseen().take(3).acollect()
        rl = await AIter(counter()).repeat_last().take(2).acollect()  # never repeats
        return pw, ue, rl

    pw, ue, rl = _run(go)
    assert pw == [(0, 1), (1, 2), (2, 3)]
    assert ue == [0, 1, 0]
    assert rl == [0, 1]


def test_ported_sinks_match_semantics():
    def A(src):
        return Iter(src).to_async()

    assert _run(lambda: A(range(5)).anth(2)) == 2
    assert _run(lambda: A(range(2)).anth(9, default=-1)) == -1
    assert _run(lambda: A(range(4)).alast()) == 3
    assert _run(lambda: A([42]).aone()) == 42
    assert _run(lambda: A([]).aonly(default="x")) == "x"
    assert _run(lambda: A(range(7)).ailen()) == 7
    assert _run(lambda: A([1, 2, 3]).adotproduct([4, 5, 6])) == 32
    assert _run(lambda: A([1, 2, 3]).ais_sorted()) is True
    assert _run(lambda: A([1, 3, 2]).ais_sorted()) is False
    assert _run(lambda: A([1, 1]).aall_unique()) is False
    assert _run(lambda: A([2, 2, 2]).aall_equal()) is True
    assert _run(lambda: A([3, 1, 4, 1, 5]).aminmax()) == (1, 5)
    assert _run(lambda: A(range(6)).aquantify(lambda x: x % 2 == 0)) == 3
    assert _run(lambda: A([0, 0, 5, 0]).afirst_true()) == 5
    assert _run(lambda: A([1, 0, 1, 0]).aexactly_n(2)) is True
    assert _run(lambda: A([(1, 2), (3, 4)]).astarreduce(lambda t, a, b: t + a + b, 0)) == 10


def test_sync_scalar_sink_guards_point_to_async_twin():
    # Every synchronous scalar sink on an AIter must raise a TypeError that
    # names its ``a``-prefixed twin -- never silently iterate or hang.
    import pytest

    def A(src):
        return Iter(src).to_async()

    twins = {
        "reduce": "areduce", "starreduce": "astarreduce", "any": "aany",
        "all": "aall", "nth": "anth", "nth_or_last": "anth_or_last",
        "last": "alast", "one": "aone", "only": "aonly", "ilen": "ailen",
        "dotproduct": "adotproduct", "is_sorted": "ais_sorted",
        "all_unique": "aall_unique", "all_equal": "aall_equal",
        "minmax": "aminmax", "quantify": "aquantify",
        "first_true": "afirst_true", "exactly_n": "aexactly_n",
        "map_reduce": "amap_reduce", "consume": "aconsume",
        "next": "afirst",
    }
    for sync_name, async_name in twins.items():
        with pytest.raises(TypeError) as exc:
            getattr(A(range(3)), sync_name)()
        assert async_name in str(exc.value), sync_name


def test_zip_strict_raises_on_unequal_lengths():
    import pytest

    def A(src):
        return Iter(src).to_async()

    # Non-strict truncates to the shortest; strict rejects unequal inputs.
    assert _run(lambda: A(range(5)).zip(range(3)).acollect()) == [
        (0, 0), (1, 1), (2, 2)
    ]
    with pytest.raises(ValueError):
        _run(lambda: A(range(5)).zip(range(3), strict=True).acollect())


def test_stateful_helpers_refuse_on_async_chain():
    # peekable/seekable/spy/tee/bucket/time_limited have no coherent async
    # form; each must fail loudly (pointing at the collect-then-Iter escape)
    # rather than silently pulling the underlying sync-only machinery.
    import pytest

    def A(src):
        return Iter(src).to_async()

    for name in ("peekable", "seekable", "spy", "tee", "bucket", "time_limited"):
        with pytest.raises(NotImplementedError) as exc:
            getattr(A(range(3)), name)()
        assert "acollect" in str(exc.value), name


def A_(src):
    return Iter(src).to_async()


def test_awrite_text_and_bytes_to_stream():
    # Text stream: newlines interspersed by default; suppressed on request;
    # batching (batch_size < len) must not change the bytes written.
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "t.txt")
        with open(fn, "w") as f:
            _run(lambda: A_(["a", "b", "c"]).map(str.upper).awrite_text_to_stream(f))
        assert open(fn).read() == "A\nB\nC"

        with open(fn, "w") as f:
            _run(lambda: A_("abcde").awrite_text_to_stream(
                f, insert_newlines=False, batch_size=2))
        assert open(fn).read() == "abcde"

        bn = os.path.join(td, "b.bin")
        with open(bn, "wb") as f:
            _run(lambda: A_([b"a", b"b", b"c"]).map(lambda x: x * 2)
                 .awrite_bytes_to_stream(f))
        assert open(bn, "rb").read() == b"aabbcc"


def test_awrite_file_text_bytes_and_error_persists_consumed_items():
    with tempfile.TemporaryDirectory() as td:
        wf = os.path.join(td, "w.txt")
        _run(lambda: A_(["ABC\n", "DEF\n"]).awrite_file(wf))
        assert open(wf).read() == "ABC\nDEF\n"

        wfb = os.path.join(td, "w.bin")
        _run(lambda: A_([b"a", b"b"]).awrite_file(wfb, mode="wb"))
        assert open(wfb, "rb").read() == b"ab"

        # A mid-stream source error still persists everything consumed so
        # far and closes the file (finally-flush), matching the sync sink.
        import pytest

        async def boom():
            yield "ok\n"
            raise RuntimeError("mid-stream")

        ef = os.path.join(td, "e.txt")
        with pytest.raises(RuntimeError):
            _run(lambda: AIter(boom()).awrite_file(ef))
        assert open(ef).read() == "ok\n"


def test_aexecutemany_commits_and_rolls_back():
    import sqlite3
    import pytest

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE c(id INTEGER, email TEXT)")
    _run(lambda: A_([(1, "A@X.COM"), (2, "B@X.COM")])
         .starmap(lambda i, e: (i, e.lower()))
         .aexecutemany(cur, "INSERT INTO c VALUES (?,?)",
                       batch_size=1, commit=conn.commit))
    assert cur.execute("SELECT * FROM c ORDER BY id").fetchall() == [
        (1, "a@x.com"), (2, "b@x.com")
    ]
    conn.close()

    # rollback fires (and the error re-raises) when executemany blows up.
    rolled = []

    class FakeCursor:
        def executemany(self, sql, batch):
            raise ValueError("boom")

    with pytest.raises(ValueError):
        _run(lambda: A_([(1,)]).aexecutemany(
            FakeCursor(), "x", rollback=lambda: rolled.append(1)))
    assert rolled == [1]


def test_into_queue_sync_and_async_queue():
    import queue

    q = queue.Queue()
    _run(lambda: A_(range(5)).into_queue(q).aconsume())
    drained = []
    while not q.empty():
        drained.append(q.get_nowait())
    assert drained == [0, 1, 2, 3, 4]

    async def go():
        aq = asyncio.Queue()               # coroutine .put is awaited
        passed = await A_(range(3)).into_queue(aq).acollect()
        return passed, [aq.get_nowait() for _ in range(3)]

    passed, in_queue = _run(go)
    assert passed == [0, 1, 2]             # items still flow downstream
    assert in_queue == [0, 1, 2]


def test_asend_and_send_also_drive_async_generator():
    # asend is a sink: it primes and feeds an async-generator collector.
    async def go_send():
        output = []

        async def collector():
            while True:
                output.append((yield))

        await A_(range(3)).asend(collector())
        return output

    assert _run(go_send) == [0, 1, 2]

    # close_collector_when_done then reuse -> StopAsyncIteration.
    import pytest

    async def go_close():
        async def collector():
            while True:
                (yield)

        g = collector()
        await A_(range(3)).asend(g, close_collector_when_done=True)
        await A_(range(3)).asend(g)         # closed -> raises

    with pytest.raises(StopAsyncIteration):
        _run(go_close)

    # send_also tees into the collector AND passes items downstream.
    async def go_also():
        seen = []

        async def collector():
            while True:
                seen.append((yield))

        passed = await A_(range(3)).send_also(collector()).acollect()
        return passed, seen

    passed, seen = _run(go_also)
    assert passed == [0, 1, 2] and seen == [0, 1, 2]


def test_io_sink_sync_guards_point_to_async_twin():
    import pytest

    twins = {
        "write_text_to_stream": "awrite_text_to_stream",
        "write_bytes_to_stream": "awrite_bytes_to_stream",
        "write_file": "awrite_file",
        "executemany": "aexecutemany",
        "send": "asend",
    }
    for sync_name, async_name in twins.items():
        with pytest.raises(TypeError) as exc:
            getattr(A_(range(3)), sync_name)()
        assert async_name in str(exc.value), sync_name
