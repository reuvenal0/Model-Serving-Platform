"""Bounded HTTP work admission; native work owns capacity until completion."""

import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Callable, TypeVar

from app.core.exceptions import InferenceUnavailableError

T = TypeVar("T")


class AdmissionRejectedError(InferenceUnavailableError):
    """Rejected before the inference observation boundary."""


class InferenceAdmission:
    """One event-loop owner, a dedicated bounded pool, and a FIFO waiting queue.

    Only admitted work is submitted. Cancelling an HTTP waiter never cancels its
    concurrent future or releases capacity while native work continues. Shutdown
    rejects queued requests and drains submitted work before services close.
    """

    def __init__(self, capacity: int, waiting_capacity: int, wait_timeout: float) -> None:
        if type(capacity) is not int or capacity <= 0 or type(waiting_capacity) is not int or waiting_capacity < 0:
            raise ValueError("Invalid admission capacities")
        if type(wait_timeout) not in (int, float) or not math.isfinite(wait_timeout) or wait_timeout <= 0:
            raise ValueError("Invalid admission timeout")
        self.capacity = capacity
        self.waiting_capacity = waiting_capacity
        self.wait_timeout = wait_timeout
        self._executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="inference")
        self._waiting: deque[asyncio.Future] = deque()
        self._work: set[asyncio.Future] = set()
        self._active = self._rejections = self._timeouts = self._admitted = 0
        self._wait_total = self._wait_max = 0.0
        self._closed = False

    def snapshot(self) -> dict:
        """Call on the owning application event loop; counters are process-local."""
        return {"active": self._active, "waiting": len(self._waiting),
                "capacity": self.capacity, "waiting_capacity": self.waiting_capacity,
                "capacity_rejections": self._rejections, "wait_timeouts": self._timeouts,
                "admitted_requests": self._admitted, "total_admission_wait_seconds": self._wait_total,
                "max_admission_wait_seconds": self._wait_max, "wait_timeout_seconds": self.wait_timeout}

    def _release(self) -> None:
        self._active -= 1
        if self._waiting and not self._closed:
            waiter = self._waiting.popleft()
            self._active += 1
            waiter.set_result(None)

    async def run(self, operation: Callable[[], T]) -> T:
        started = perf_counter()
        if self._closed:
            raise AdmissionRejectedError("Inference admission is closed")
        if self._active < self.capacity:
            self._active += 1
        else:
            if len(self._waiting) >= self.waiting_capacity:
                self._rejections += 1
                raise AdmissionRejectedError("Inference capacity is exhausted")
            waiter = asyncio.get_running_loop().create_future()
            self._waiting.append(waiter)
            try:
                await asyncio.wait_for(asyncio.shield(waiter), self.wait_timeout)
            except (asyncio.CancelledError, TimeoutError) as error:
                if waiter in self._waiting:
                    self._waiting.remove(waiter)
                elif not waiter.cancelled() and waiter.exception() is None:
                    # Capacity was transferred just before cancellation/timeout.
                    self._release()
                if isinstance(error, asyncio.CancelledError):
                    raise
                self._timeouts += 1
                raise AdmissionRejectedError("Inference admission waiting expired") from None
        if self._closed:
            # Shutdown can overlap a queued permit transfer before this task
            # resumes. Never submit new work to a drained/closed executor.
            self._release()
            raise AdmissionRejectedError("Inference admission is closed")
        waited = perf_counter() - started
        self._wait_total += waited
        self._wait_max = max(self._wait_max, waited)
        self._admitted += 1
        loop = asyncio.get_running_loop()
        try:
            work = asyncio.wrap_future(self._executor.submit(operation), loop=loop)
        except Exception:
            self._release()
            raise
        self._work.add(work)
        released = False

        def completed(future):
            nonlocal released
            if released:
                return
            released = True
            self._work.discard(future)
            future.exception()  # Retrieve detached failures without logging private context.
            self._release()

        work.add_done_callback(completed)
        try:
            return await asyncio.shield(work)
        finally:
            # A very fast worker may finish before add_done_callback schedules
            # its callback. Complete bookkeeping before returning to the caller.
            if work.done():
                completed(work)

    async def close(self) -> None:
        self._closed = True
        while self._waiting:
            self._waiting.popleft().set_exception(AdmissionRejectedError("Inference admission is closed"))
        interrupted = False
        if self._work:
            draining = asyncio.gather(*self._work, return_exceptions=True)
            while not draining.done():
                try:
                    await asyncio.shield(draining)
                except asyncio.CancelledError:
                    # Even cancelled shutdown must not close the database while
                    # a synchronous prediction/observation is still using it.
                    interrupted = True
        self._executor.shutdown(wait=True)
        if interrupted:
            raise asyncio.CancelledError
