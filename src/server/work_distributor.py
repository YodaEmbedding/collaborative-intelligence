import queue
from itertools import count
from typing import (
    Awaitable,
    Dict,
    Generator,
    Generic,
    Tuple,
    TypeVar,
)


import janus

R = TypeVar("R")
T = TypeVar("T")


class WorkDistributor(Generic[T, R]):
    """Process async items synchronously.

    Queues asynchronous requests for synchronous processing. Once
    processor is ready, it reads item from request queue, then puts the
    result into the result queue.
    """

    _requests: janus.Queue
    _results: Dict[int, janus.Queue]

    def __init__(self):
        self._guid = 0
        self._requests = janus.Queue()
        self._results = {}

    def register(self):
        """Register client for processing.

        Returns:
            request_callback: Asynchronously push request.
            result_callback: Asynchronously receive result.
        """
        guid = self._guid
        self._guid += 1
        self._results[guid] = janus.Queue()

        async def put_request(item: T):
            await self._requests.async_q.put((guid, item))

        async def get_result() -> Awaitable[R]:
            return await self._results[guid].async_q.get()

        return put_request, get_result

    def get(self) -> Tuple[int, T]:
        """Synchronously retrieve request for processing."""
        return self._requests.sync_q.get()

    def get_many(
        self, min_items=1, max_items=None
    ) -> Generator[Tuple[int, T], None, None]:
        """Synchronously retrieve requests for processing.

        Retrieve at least min_items, blocking if necessary. Retreive up
        to max_items if possible without blocking.
        """
        for _ in range(min_items):
            yield self.get()
        it = count() if max_items is None else range(max_items - min_items)
        for _ in it:
            if self._requests.sync_q.empty():
                break
            yield self.get()

    def empty(self) -> bool:
        """Check if process queue is empty."""
        return self._requests.sync_q.empty()

    def put(self, guid: int, item: R):
        """Synchronously push processed result."""
        self._results[guid].sync_q.put(item)


class SmartProcessor(Generic[T, R]):
    """Looks ahead to determine if work items should be cancelled."""

    def __init__(self, work_distributor: WorkDistributor[T, R]):
        self.work_distributor = work_distributor
        self.buffer = queue.Queue()

    def get(self) -> Tuple[int, T]:
        self._refresh_buffer()
        return self.buffer.get()

    def _refresh_buffer(self):
        min_items = 1 if self.buffer.empty() else 0
        items = list(self.work_distributor.get_many(min_items=min_items))
        idxs = (
            len(items) - i - 1
            for i, (_, (_, request_type, _)) in enumerate(reversed(items))
            if request_type == "release"
        )
        idx = next(idxs, None)
        if idx is None:
            for x in items:
                self.buffer.put(x)
            return
        self.buffer = queue.Queue()
        for x in items[idx:]:
            self.buffer.put(x)
