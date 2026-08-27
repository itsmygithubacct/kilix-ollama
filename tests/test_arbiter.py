from __future__ import annotations

import threading
import unittest

from kilix_ollama.arbiter import (
    LLM_TURN,
    TTS_UTTERANCE,
    AcceleratorArbiter,
)
from kilix_ollama.errors import (
    AcceleratorQueueFull,
    AcceleratorRequestCancelled,
    AcceleratorRequestTimeout,
    AcceleratorUnavailable,
    RuntimeContractError,
)


class AcceleratorArbiterTests(unittest.TestCase):
    def test_active_turn_finishes_then_tts_gets_bounded_priority(self) -> None:
        arbiter = AcceleratorArbiter(max_queue=8, tts_burst_limit=2)
        active_llm = arbiter.acquire(LLM_TURN)
        speech_one = arbiter.submit(TTS_UTTERANCE)
        speech_two = arbiter.submit(TTS_UTTERANCE)
        speech_three = arbiter.submit(TTS_UTTERANCE)
        conversation = arbiter.submit(LLM_TURN)

        active_llm.release()
        first = speech_one.wait(timeout=0)
        first.release()
        second = speech_two.wait(timeout=0)
        second.release()

        # Two speech utterances are enough to restore conversation fairness.
        llm = conversation.wait(timeout=0)
        llm.release()
        third = speech_three.wait(timeout=0)
        third.release()

        self.assertEqual(
            arbiter.record()["counters"]["granted"],
            {LLM_TURN: 2, TTS_UTTERANCE: 3},
        )

    def test_fifo_order_is_preserved_within_each_workload(self) -> None:
        arbiter = AcceleratorArbiter()
        active = arbiter.acquire(LLM_TURN)
        first = arbiter.submit(TTS_UTTERANCE)
        second = arbiter.submit(TTS_UTTERANCE)
        active.release()
        with self.assertRaises(AcceleratorRequestTimeout):
            second.wait(timeout=0)
        lease = first.wait(timeout=0)
        lease.release()

    def test_queue_is_bounded_and_does_not_count_active_work(self) -> None:
        arbiter = AcceleratorArbiter(max_queue=2)
        active = arbiter.acquire(LLM_TURN)
        arbiter.submit(LLM_TURN)
        arbiter.submit(TTS_UTTERANCE)
        with self.assertRaises(AcceleratorQueueFull):
            arbiter.submit(TTS_UTTERANCE)
        active.release()

    def test_queued_request_can_be_cancelled_without_disturbing_active_turn(self) -> None:
        arbiter = AcceleratorArbiter()
        active = arbiter.acquire(LLM_TURN)
        request = arbiter.submit(TTS_UTTERANCE)
        self.assertTrue(request.cancel())
        self.assertFalse(request.cancel())
        with self.assertRaises(AcceleratorRequestCancelled):
            request.wait(timeout=0)
        self.assertEqual(arbiter.record()["active"]["ticket"], active.ticket)
        active.release()

    def test_external_cancel_probe_wakes_a_waiter(self) -> None:
        arbiter = AcceleratorArbiter()
        active = arbiter.acquire(LLM_TURN)
        cancelled = threading.Event()
        request = arbiter.submit(TTS_UTTERANCE)
        outcome: list[type[BaseException]] = []

        def wait() -> None:
            try:
                request.wait(cancelled=cancelled.is_set)
            except BaseException as error:  # captured for the test thread
                outcome.append(type(error))

        thread = threading.Thread(target=wait)
        thread.start()
        cancelled.set()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, [AcceleratorRequestCancelled])
        active.release()

    def test_timeout_removes_request_and_double_release_is_harmless(self) -> None:
        arbiter = AcceleratorArbiter()
        active = arbiter.acquire(LLM_TURN)
        waiting = arbiter.submit(TTS_UTTERANCE)
        with self.assertRaises(AcceleratorRequestTimeout):
            waiting.wait(timeout=0)
        self.assertEqual(arbiter.record()["queued"][TTS_UTTERANCE], 0)
        self.assertTrue(active.release())
        self.assertFalse(active.release())

    def test_close_wakes_waiters_and_refuses_new_work(self) -> None:
        arbiter = AcceleratorArbiter()
        active = arbiter.acquire(LLM_TURN)
        waiting = arbiter.submit(TTS_UTTERANCE)
        arbiter.close()
        with self.assertRaises(AcceleratorUnavailable):
            waiting.wait(timeout=0)
        with self.assertRaises(AcceleratorUnavailable):
            arbiter.submit(LLM_TURN)
        active.release()

    def test_unknown_workloads_and_invalid_limits_fail_closed(self) -> None:
        with self.assertRaises(RuntimeContractError):
            AcceleratorArbiter(max_queue=0)
        arbiter = AcceleratorArbiter()
        with self.assertRaises(RuntimeContractError):
            arbiter.submit("arbitrary-command")
        with self.assertRaises(RuntimeContractError):
            arbiter.acquire(LLM_TURN, timeout=float("nan"))

    def test_status_contains_no_workload_content_or_weights(self) -> None:
        record = AcceleratorArbiter().record()
        self.assertFalse(record["request_content_recorded"])
        self.assertFalse(record["weight_access"])
        self.assertEqual(record["schema_status"], "development-unfrozen")


if __name__ == "__main__":
    unittest.main()
