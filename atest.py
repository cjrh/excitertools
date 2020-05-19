from typing import Callable, Awaitable, AsyncIterable
import collections.abc
import builtins
import inspect
import asyncio


class Xiter:
    def __init__(self, seq):
        self.isasyncgen = inspect.isasyncgen(seq)
        if self.isasyncgen:
            self.it = seq
        else:
            self.it = iter(seq)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.it)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.isasyncgen:
            return await self.it.__anext__()
        else:
            try:
                return next(self.it)
            except StopIteration:
                raise StopAsyncIteration


def xiter(seq):
    if isinstance(seq, Xiter):
        return seq

    return Xiter(seq)


def xfunc(func) -> Callable[..., Awaitable]:
    async def actual_func(v):
        if inspect.iscoroutinefunction(func):
            return await func(v)
        else:
            return func(v)

    return actual_func


class amap:
    def __init__(self, func, iterable):
        self.func = func
        self.iterable = xiter(iterable)

    def __iter__(self):
        return builtins.map(self.func, self.iterable)

    def __aiter__(self):
        async def inner():
            async for v in self.iterable:
                if inspect.iscoroutinefunction(self.func):
                    yield await self.func(v)
                else:
                    yield self.func(v)

        return inner()


async def tests(map_func):
    map = map_func

    def abc(v):
        return v + 1

    async def aabc(v):
        return v + 1

    ####################################################################
    #
    # sync iteration: func and iterable must also both be sync
    #
    ####################################################################

    items = [x for x in map(abc, range(10))]
    print(items)

    ####################################################################
    #
    # async iteration: there are permutations
    #
    #     - sync func, sync iterable (async for not necessary)
    #     - async func, sync iterable
    #     - sync func, async iterable
    #     - async func, async iterable
    #
    ####################################################################

    ####################################################################
    # async for: map - Sync function, sync iterable
    items = [x async for x in map(abc, range(10))]
    print(items)

    ####################################################################
    # async for: map - Async function, sync iterable
    items = [x async for x in map(aabc, range(10))]
    print(items)

    ####################################################################
    # async for: map sync function, async iterable
    async def arange(start):
        await asyncio.sleep(0.01)
        for x in range(10):
            yield x

    items = [x async for x in map(abc, arange(10))]
    print(items)

    ####################################################################
    # async for: map async function, async iterable
    async def arange(start):
        await asyncio.sleep(0.01)
        for x in range(10):
            yield x

    items = [x async for x in map(aabc, arange(10))]
    print(items)


async def main():
    await tests(map_func=amap)


asyncio.run(main())
