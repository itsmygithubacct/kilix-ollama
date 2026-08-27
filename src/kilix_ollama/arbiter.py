"""Bounded, cancellable arbitration for one shared accelerator.

The arbiter schedules execution, not model residency.  Ollama may keep one LLM
loaded while a small speech model remains resident under the externally
qualified VRAM budget, but only one inference workload owns the device at a
time.  TTS gets bounded priority at an utterance boundary; a burst limit keeps
continuous speech requests from starving conversation turns.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable, Iterator

from .errors import (
    AcceleratorQueueFull,
    AcceleratorRequestCancelled,
    AcceleratorRequestTimeout,
    AcceleratorUnavailable,
    RuntimeContractError,
)


LLM_TURN = "llm-turn"
TTS_UTTERANCE = "tts-utterance"
WORKLOADS = frozenset((LLM_TURN, TTS_UTTERANCE))
DEFAULT_MAX_QUEUE = 8
DEFAULT_TTS_BURST = 2
_CANCEL_POLL_SECONDS = 0.05


def _positive_integer(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeContractError(f"{label} must be a positive integer")
    return value


def _timeout(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise RuntimeContractError("accelerator wait timeout must be finite and non-negative")
    return float(value)


@dataclass(eq=False)
class AcceleratorLease:
    """An identity-bound grant; only the arbiter that issued it can release it."""

    ticket: int
    workload: str
    granted_monotonic: float
    _arbiter: "AcceleratorArbiter" = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def release(self) -> bool:
        """Release once.  Repeated cleanup is harmless and returns ``False``."""
        return self._arbiter.release(self)

    def __enter__(self) -> "AcceleratorLease":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()

    def record(self) -> dict[str, object]:
        return {"ticket": self.ticket, "workload": self.workload}


@dataclass(eq=False)
class LeaseRequest:
    """A queued request that can be waited for or cancelled by its owner."""

    ticket: int
    workload: str
    _arbiter: "AcceleratorArbiter" = field(repr=False)
    _state: str = field(default="queued", init=False, repr=False)
    _lease: AcceleratorLease | None = field(default=None, init=False, repr=False)

    @property
    def state(self) -> str:
        with self._arbiter._condition:
            return self._state

    def wait(
        self,
        *,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AcceleratorLease:
        return self._arbiter.wait(self, timeout=timeout, cancelled=cancelled)

    def cancel(self) -> bool:
        return self._arbiter.cancel(self)

    def record(self) -> dict[str, object]:
        return {"ticket": self.ticket, "workload": self.workload, "state": self.state}


class AcceleratorArbiter:
    """Serialize LLM turns and TTS utterances with bounded TTS priority."""

    def __init__(
        self,
        *,
        max_queue: int = DEFAULT_MAX_QUEUE,
        tts_burst_limit: int = DEFAULT_TTS_BURST,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_queue = _positive_integer(max_queue, "max_queue")
        self.tts_burst_limit = _positive_integer(tts_burst_limit, "tts_burst_limit")
        if not callable(clock):
            raise RuntimeContractError("accelerator clock must be callable")
        self._clock = clock
        self._condition = threading.Condition(threading.Lock())
        self._queues: dict[str, deque[LeaseRequest]] = {
            LLM_TURN: deque(),
            TTS_UTTERANCE: deque(),
        }
        self._active: AcceleratorLease | None = None
        self._next_ticket = 1
        self._tts_burst = 0
        self._closed = False
        self._granted = {LLM_TURN: 0, TTS_UTTERANCE: 0}
        self._cancelled = 0
        self._timed_out = 0

    @staticmethod
    def _workload(value: str) -> str:
        if not isinstance(value, str) or value not in WORKLOADS:
            raise RuntimeContractError(
                f"unknown accelerator workload {value!r}; expected one of: "
                f"{', '.join(sorted(WORKLOADS))}"
            )
        return value

    def submit(self, workload: str) -> LeaseRequest:
        kind = self._workload(workload)
        with self._condition:
            if self._closed:
                raise AcceleratorUnavailable("accelerator arbiter is closed")
            queued = sum(len(queue) for queue in self._queues.values())
            if queued >= self.max_queue:
                raise AcceleratorQueueFull(
                    f"accelerator queue is full ({self.max_queue} waiting requests)"
                )
            request = LeaseRequest(self._next_ticket, kind, self)
            self._next_ticket += 1
            self._queues[kind].append(request)
            self._condition.notify_all()
            return request

    def _selected_queue(self) -> str | None:
        llm_waiting = bool(self._queues[LLM_TURN])
        tts_waiting = bool(self._queues[TTS_UTTERANCE])
        if not llm_waiting and not tts_waiting:
            return None
        if tts_waiting and (
            not llm_waiting or self._tts_burst < self.tts_burst_limit
        ):
            return TTS_UTTERANCE
        return LLM_TURN

    def _can_grant(self, request: LeaseRequest) -> bool:
        selected = self._selected_queue()
        return (
            self._active is None
            and selected == request.workload
            and bool(self._queues[selected])
            and self._queues[selected][0] is request
        )

    def _remove_queued(self, request: LeaseRequest) -> bool:
        if request._arbiter is not self or request._state != "queued":
            return False
        queue = self._queues[request.workload]
        try:
            queue.remove(request)
        except ValueError:
            return False
        return True

    def wait(
        self,
        request: LeaseRequest,
        *,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AcceleratorLease:
        if not isinstance(request, LeaseRequest) or request._arbiter is not self:
            raise RuntimeContractError("accelerator request belongs to another arbiter")
        wait_seconds = _timeout(timeout)
        if cancelled is not None and not callable(cancelled):
            raise RuntimeContractError("accelerator cancellation probe must be callable")
        deadline = None if wait_seconds is None else self._clock() + wait_seconds

        with self._condition:
            while True:
                if request._state == "granted" and request._lease is not None:
                    return request._lease
                if request._state == "cancelled":
                    raise AcceleratorRequestCancelled(
                        f"accelerator request {request.ticket} was cancelled"
                    )
                if request._state == "timed-out":
                    raise AcceleratorRequestTimeout(
                        f"accelerator request {request.ticket} timed out"
                    )
                if request._state == "closed" or self._closed:
                    if self._remove_queued(request):
                        request._state = "closed"
                    raise AcceleratorUnavailable("accelerator arbiter is closed")
                if cancelled is not None and cancelled():
                    if self._remove_queued(request):
                        request._state = "cancelled"
                        self._cancelled += 1
                        self._condition.notify_all()
                    raise AcceleratorRequestCancelled(
                        f"accelerator request {request.ticket} was cancelled"
                    )
                if self._can_grant(request):
                    queued = self._queues[request.workload].popleft()
                    if queued is not request:
                        raise RuntimeError("accelerator queue invariant violated")
                    lease = AcceleratorLease(
                        request.ticket,
                        request.workload,
                        self._clock(),
                        self,
                    )
                    request._state = "granted"
                    request._lease = lease
                    self._active = lease
                    self._granted[request.workload] += 1
                    return lease

                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    if self._remove_queued(request):
                        request._state = "timed-out"
                        self._timed_out += 1
                        self._condition.notify_all()
                    raise AcceleratorRequestTimeout(
                        f"accelerator request {request.ticket} timed out"
                    )
                interval = remaining
                if cancelled is not None:
                    interval = (
                        _CANCEL_POLL_SECONDS
                        if remaining is None
                        else min(remaining, _CANCEL_POLL_SECONDS)
                    )
                self._condition.wait(interval)

    def acquire(
        self,
        workload: str,
        *,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AcceleratorLease:
        return self.submit(workload).wait(timeout=timeout, cancelled=cancelled)

    @contextmanager
    def hold(
        self,
        workload: str,
        *,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[AcceleratorLease]:
        lease = self.acquire(workload, timeout=timeout, cancelled=cancelled)
        try:
            yield lease
        finally:
            lease.release()

    def cancel(self, request: LeaseRequest) -> bool:
        if not isinstance(request, LeaseRequest) or request._arbiter is not self:
            raise RuntimeContractError("accelerator request belongs to another arbiter")
        with self._condition:
            if not self._remove_queued(request):
                return False
            request._state = "cancelled"
            self._cancelled += 1
            self._condition.notify_all()
            return True

    def release(self, lease: AcceleratorLease) -> bool:
        if not isinstance(lease, AcceleratorLease) or lease._arbiter is not self:
            raise RuntimeContractError("accelerator lease belongs to another arbiter")
        with self._condition:
            if lease._released:
                return False
            if self._active is not lease:
                raise RuntimeContractError("accelerator lease is not the active grant")
            lease._released = True
            self._active = None
            if lease.workload == TTS_UTTERANCE:
                self._tts_burst += 1
            else:
                self._tts_burst = 0
            self._condition.notify_all()
            return True

    def close(self) -> None:
        """Refuse new work and wake every queued waiter; active work may clean up."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for queue in self._queues.values():
                for request in queue:
                    request._state = "closed"
                queue.clear()
            self._condition.notify_all()

    def record(self) -> dict[str, object]:
        with self._condition:
            return {
                "schema": "kilix.ollama.accelerator-arbiter/v0-unfrozen",
                "schema_status": "development-unfrozen",
                "policy": "finish-active-turn-then-bounded-tts-priority",
                "active": self._active.record() if self._active is not None else None,
                "queued": {
                    workload: len(self._queues[workload])
                    for workload in sorted(WORKLOADS)
                },
                "limits": {
                    "max_queue": self.max_queue,
                    "tts_burst": self.tts_burst_limit,
                },
                "counters": {
                    "granted": dict(sorted(self._granted.items())),
                    "cancelled": self._cancelled,
                    "timed_out": self._timed_out,
                },
                "closed": self._closed,
                "request_content_recorded": False,
                "weight_access": False,
            }


__all__ = [
    "DEFAULT_MAX_QUEUE",
    "DEFAULT_TTS_BURST",
    "LLM_TURN",
    "TTS_UTTERANCE",
    "WORKLOADS",
    "AcceleratorArbiter",
    "AcceleratorLease",
    "LeaseRequest",
]
