"""Admission tests synchronize actual worker completion, not HTTP cancellation."""
import asyncio
from threading import Event, Lock

import pytest

from app.services.inference_admission import AdmissionRejectedError, InferenceAdmission


def test_capacity_queue_fifo_and_rejection():
    async def scenario():
        gate = InferenceAdmission(1, 1, 5)
        started, release = Event(), Event()
        order = []
        def slow():
            started.set()
            assert release.wait(5)
            order.append(1)
            return 1
        first = asyncio.create_task(gate.run(slow))
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(gate.run(lambda: order.append(2)))
        await asyncio.sleep(0)  # Let the explicitly created task enqueue.
        assert gate.snapshot()['active'] == gate.snapshot()['waiting'] == 1
        with pytest.raises(AdmissionRejectedError):
            await gate.run(lambda: pytest.fail('Rejected work ran'))
        assert gate.snapshot()['capacity_rejections'] == 1
        release.set()
        assert await first == 1
        await second
        assert order == [1, 2]
        assert gate.snapshot()['active'] == gate.snapshot()['waiting'] == 0
        assert gate.snapshot()['admitted_requests'] == 2
        assert gate.snapshot()['total_admission_wait_seconds'] >= 0
        await gate.close()
    asyncio.run(scenario())


def test_cancelled_http_waiter_retains_native_execution_slot():
    async def scenario():
        gate = InferenceAdmission(1, 0, 5)
        started, release = Event(), Event()
        def slow():
            started.set()
            assert release.wait(5)
            raise RuntimeError('private backend detail')
        first = asyncio.create_task(gate.run(slow))
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert gate.snapshot()['active'] == 1
        with pytest.raises(AdmissionRejectedError):
            await gate.run(lambda: None)
        release.set()
        await gate.close()
        assert gate.snapshot()['active'] == 0
    asyncio.run(scenario())


def test_queued_cancellation_removes_waiter_without_releasing_active_slot():
    async def scenario():
        gate = InferenceAdmission(1, 1, 5)
        started, release = Event(), Event()
        def slow():
            started.set()
            assert release.wait(5)
        first = asyncio.create_task(gate.run(slow))
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(gate.run(lambda: pytest.fail('Cancelled work ran')))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert gate.snapshot()['waiting'] == 0 and gate.snapshot()['active'] == 1
        release.set()
        await first
        await gate.close()
    asyncio.run(scenario())


def test_wait_timeout_does_not_terminate_or_release_running_work():
    async def scenario():
        gate = InferenceAdmission(1, 1, 0.001)
        started, release = Event(), Event()
        def slow():
            started.set()
            assert release.wait(5)
        first = asyncio.create_task(gate.run(slow))
        assert await asyncio.to_thread(started.wait, 5)
        with pytest.raises(AdmissionRejectedError):
            await gate.run(lambda: pytest.fail('Expired work ran'))
        assert gate.snapshot()['wait_timeouts'] == 1
        assert gate.snapshot()['active'] == 1 and gate.snapshot()['waiting'] == 0
        release.set()
        await first
        assert await gate.run(lambda: 42) == 42
        await gate.close()
    asyncio.run(scenario())


def test_worker_exception_releases_capacity():
    async def scenario():
        gate = InferenceAdmission(1, 0, 5)
        def fail():
            raise ValueError('failure')
        with pytest.raises(ValueError):
            await gate.run(fail)
        assert await gate.run(lambda: 42) == 42
        assert gate.snapshot()['active'] == 0
        await gate.close()
    asyncio.run(scenario())


def test_simultaneous_admission_never_exceeds_capacity():
    async def scenario():
        gate = InferenceAdmission(3, 0, 5)
        all_started, release, lock = Event(), Event(), Lock()
        running = peak = 0
        def work():
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
                if running == 3:
                    all_started.set()
            assert release.wait(5)
            with lock:
                running -= 1
        tasks = [asyncio.create_task(gate.run(work)) for _ in range(30)]
        assert await asyncio.to_thread(all_started.wait, 5)
        assert gate.snapshot()['active'] == 3
        assert gate.snapshot()['capacity_rejections'] == 27
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert sum(isinstance(item, AdmissionRejectedError) for item in results) == 27
        assert peak == 3
        await gate.close()
    asyncio.run(scenario())


def test_close_rejects_queue_and_drains_active_work():
    async def scenario():
        gate = InferenceAdmission(1, 1, 5)
        started, release = Event(), Event()
        def work():
            started.set()
            assert release.wait(5)
        first = asyncio.create_task(gate.run(work))
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(gate.run(lambda: None))
        await asyncio.sleep(0)
        closing = asyncio.create_task(gate.close())
        with pytest.raises(AdmissionRejectedError):
            await second
        assert not closing.done()
        release.set()
        await closing
        await first
        assert gate.snapshot()['active'] == 0
        with pytest.raises(AdmissionRejectedError):
            await gate.run(lambda: None)
    asyncio.run(scenario())


@pytest.mark.parametrize('capacity,waiting,timeout', [(0,1,1),(True,1,1),(1,-1,1),(1,True,1),(1,1,0),(1,1,float('inf')),(1,1,float('nan'))])
def test_invalid_admission_settings(capacity, waiting, timeout):
    with pytest.raises(ValueError):
        InferenceAdmission(capacity, waiting, timeout)


def test_cancellation_during_capacity_transfer_releases_exactly_once():
    async def scenario():
        gate = InferenceAdmission(1, 1, 5)
        started, release = Event(), Event()
        def work():
            started.set()
            assert release.wait(5)
        first = asyncio.create_task(gate.run(work))
        assert await asyncio.to_thread(started.wait,5)
        second = asyncio.create_task(gate.run(lambda: pytest.fail('Cancelled transfer ran')))
        await asyncio.sleep(0)
        original = gate._release
        transferred = False
        def release_and_cancel():
            nonlocal transferred
            original()
            if not transferred:
                transferred = True
                second.cancel()
        gate._release = release_and_cancel
        release.set()
        await first
        with pytest.raises(asyncio.CancelledError):
            await second
        assert gate.snapshot()['active'] == gate.snapshot()['waiting'] == 0
        assert await gate.run(lambda: 7) == 7
        await gate.close()
    asyncio.run(scenario())


def test_cancelled_shutdown_still_drains_native_work():
    async def scenario():
        gate = InferenceAdmission(1,0,5)
        started, release = Event(), Event()
        def work():
            started.set()
            assert release.wait(5)
        request = asyncio.create_task(gate.run(work))
        assert await asyncio.to_thread(started.wait,5)
        closing = asyncio.create_task(gate.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done() and gate.snapshot()['active'] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        await request
        assert gate.snapshot()['active'] == 0
    asyncio.run(scenario())


def test_shutdown_overlapping_granted_waiter_rejects_without_submission():
    async def scenario():
        gate = InferenceAdmission(1,1,5)
        started, release = Event(), Event()
        def work():
            started.set()
            assert release.wait(5)
        first = asyncio.create_task(gate.run(work))
        assert await asyncio.to_thread(started.wait,5)
        second = asyncio.create_task(gate.run(lambda: pytest.fail('Shutdown work was submitted')))
        await asyncio.sleep(0)
        original = gate._release
        closing = None
        def transfer_then_close():
            nonlocal closing
            original()
            if closing is None:
                closing = asyncio.create_task(gate.close())
        gate._release = transfer_then_close
        release.set()
        await first
        with pytest.raises(AdmissionRejectedError):
            await second
        await closing
        assert gate.snapshot()['active'] == 0
    asyncio.run(scenario())
