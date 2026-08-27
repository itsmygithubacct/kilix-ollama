from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from kilix_ollama.arbiter import LLM_TURN, TTS_UTTERANCE, AcceleratorArbiter
from kilix_ollama.errors import SynthesisCancelled, SynthesisError
from kilix_ollama.paths import RuntimePaths
from kilix_ollama.tts_runtime import (
    MAX_TEXT_BYTES,
    PERSISTENT_TTS_RUNTIME_CLOSURE,
    PersistentVulkanTtsWorker,
    TtsRuntimeAdmission,
    VulkanTtsRunner,
)


def private_paths(root: Path) -> RuntimePaths:
    return RuntimePaths.from_environment(
        {
            "HOME": str(root / "home"),
            "XDG_RUNTIME_DIR": str(root / "run"),
            "XDG_STATE_HOME": str(root / "state"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
        },
        uid=os.getuid(),
    )


class VulkanTtsRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trusted = self.root / "trusted"
        self.trusted.mkdir(mode=0o700)
        self.launcher = self.trusted / "pocket-tts-current"
        self.launcher.write_text(
            "#!/usr/bin/python3\n"
            "import pathlib, sys, time, wave\n"
            "def value(flag):\n"
            "    return sys.argv[sys.argv.index(flag) + 1]\n"
            "prompt = pathlib.Path(value('--file')).read_text(encoding='utf-8')\n"
            "if any(prompt in argument for argument in sys.argv[1:]):\n"
            "    raise SystemExit(12)\n"
            "if prompt == 'SLOW':\n"
            "    time.sleep(5)\n"
            "if prompt == 'FAIL':\n"
            "    raise SystemExit(9)\n"
            "output = pathlib.Path(value('--output'))\n"
            "if prompt == 'BAD':\n"
            "    output.write_bytes(b'not a wav')\n"
            "else:\n"
            "    with wave.open(str(output), 'wb') as audio:\n"
            "        audio.setnchannels(1)\n"
            "        audio.setsampwidth(2)\n"
            "        audio.setframerate(24000)\n"
            "        audio.writeframes(b'\\x00\\x00' * 2400)\n",
            encoding="utf-8",
        )
        self.launcher.chmod(0o700)
        self.model = self.root / "model.gguf"
        self.projector = self.root / "projector.gguf"
        self.model.write_bytes(b"fixture model")
        self.projector.write_bytes(b"fixture projector")
        self.model.chmod(0o600)
        self.projector.chmod(0o600)
        self.paths = private_paths(self.root)
        self.arbiter = AcceleratorArbiter()
        self.admission = TtsRuntimeAdmission(
            launcher_sha256=hashlib.sha256(self.launcher.read_bytes()).hexdigest(),
            vulkan_qualification_verified=True,
        )
        self.resolver_calls = 0

        def resolve(_model: str, _paths: RuntimePaths) -> tuple[Path, Path]:
            self.resolver_calls += 1
            return self.model, self.projector

        self.runner = VulkanTtsRunner(
            self.paths,
            self.arbiter,
            self.admission,
            launcher=self.launcher,
            launcher_uid=os.getuid(),
            trusted_root=self.trusted,
            artifact_resolver=resolve,
        )

    def test_valid_pcm_is_returned_and_text_never_enters_argv(self) -> None:
        result = self.runner.synthesize("private words")
        self.assertEqual(result.sample_rate, 24_000)
        self.assertEqual(result.frames, 2400)
        self.assertAlmostEqual(result.audio_seconds, 0.1)
        self.assertTrue(result.wav.startswith(b"RIFF"))
        self.assertEqual(list(self.paths.temp_dir.glob(".tts-request-*")), [])
        self.assertFalse(result.record()["text_recorded"])

    def test_llm_finishes_before_queued_speech_starts(self) -> None:
        llm = self.arbiter.acquire(LLM_TURN)
        results = []

        def synthesize() -> None:
            results.append(self.runner.synthesize("queued speech", queue_timeout=2))

        thread = threading.Thread(target=synthesize)
        thread.start()
        deadline = time.monotonic() + 1
        while (
            self.arbiter.record()["queued"][TTS_UTTERANCE] != 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertEqual(self.arbiter.record()["active"]["workload"], LLM_TURN)
        self.assertEqual(self.arbiter.record()["queued"][TTS_UTTERANCE], 1)
        llm.release()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)

    def test_cancellation_kills_process_group_and_releases_lease(self) -> None:
        cancel = threading.Event()
        errors: list[BaseException] = []

        def synthesize() -> None:
            try:
                self.runner.synthesize("SLOW", run_timeout=10, cancelled=cancel.is_set)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=synthesize)
        thread.start()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            active = self.arbiter.record()["active"]
            if active is not None and active["workload"] == TTS_UTTERANCE:
                break
            time.sleep(0.01)
        cancel.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], SynthesisCancelled)
        self.assertIsNone(self.arbiter.record()["active"])
        self.assertEqual(list(self.paths.temp_dir.glob(".tts-request-*")), [])

    def test_runtime_timeout_and_nonzero_exit_are_safe_errors(self) -> None:
        started = time.monotonic()
        with self.assertRaises(SynthesisError):
            self.runner.synthesize("SLOW", run_timeout=0.05)
        self.assertLess(time.monotonic() - started, 2)
        with self.assertRaises(SynthesisError) as caught:
            self.runner.synthesize("FAIL")
        self.assertIn("status 9", str(caught.exception))

    def test_invalid_wav_is_rejected(self) -> None:
        with self.assertRaises(SynthesisError):
            self.runner.synthesize("BAD")
        self.assertEqual(list(self.paths.temp_dir.glob(".tts-request-*")), [])

    def test_admission_fails_before_model_resolution(self) -> None:
        wrong_hash = replace(self.admission, launcher_sha256="0" * 64)
        runner = VulkanTtsRunner(
            self.paths,
            self.arbiter,
            wrong_hash,
            launcher=self.launcher,
            launcher_uid=os.getuid(),
            trusted_root=self.trusted,
            artifact_resolver=lambda *_arguments: (_ for _ in ()).throw(
                AssertionError("artifact resolver called")
            ),
        )
        with self.assertRaises(SynthesisError):
            runner.synthesize("hello")
        unqualified = replace(
            self.admission,
            vulkan_qualification_verified=False,
        )
        with self.assertRaises(SynthesisError):
            VulkanTtsRunner(
                self.paths,
                self.arbiter,
                unqualified,
                launcher=self.launcher,
                launcher_uid=os.getuid(),
                trusted_root=self.trusted,
                artifact_resolver=lambda *_arguments: (self.model, self.projector),
            ).synthesize("hello")

    def test_text_is_bounded_before_runtime_or_model_access(self) -> None:
        for text in ("", "\x00", "x" * (MAX_TEXT_BYTES + 1)):
            with self.subTest(length=len(text)), self.assertRaises(SynthesisError):
                self.runner.synthesize(text)
        self.assertEqual(self.resolver_calls, 0)

    def test_runner_record_is_explicit_about_local_weights_and_no_network(self) -> None:
        record = self.runner.record()
        self.assertTrue(record["local_verified_weights_required"])
        self.assertFalse(record["model_publication"])
        self.assertFalse(record["network_access"])
        self.assertFalse(record["text_in_argv"])


class PersistentVulkanTtsWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trusted = self.root / "trusted"
        self.trusted.mkdir(mode=0o700)
        self.launcher = self.trusted / "kilix-tts-worker-current"
        self.launcher.write_text(
            "#!/usr/bin/python3\n"
            "import io, struct, sys, time, wave\n"
            "def exact(length):\n"
            "    output = bytearray()\n"
            "    while len(output) < length:\n"
            "        part = sys.stdin.buffer.read(length - len(output))\n"
            "        if not part:\n"
            "            return None\n"
            "        output.extend(part)\n"
            "    return bytes(output)\n"
            "def reply(status, payload=b''):\n"
            "    sys.stdout.buffer.write(struct.pack('<4sIQ', b'KXR1', status, len(payload)))\n"
            "    sys.stdout.buffer.write(payload)\n"
            "    sys.stdout.buffer.flush()\n"
            "sys.stdout.buffer.write(b'KILIXT1\\n')\n"
            "sys.stdout.buffer.flush()\n"
            "while True:\n"
            "    header = exact(8)\n"
            "    if header is None:\n"
            "        raise SystemExit(0)\n"
            "    magic, length = struct.unpack('<4sI', header)\n"
            "    if magic != b'KXQ1' or length == 0:\n"
            "        raise SystemExit(6)\n"
            "    raw = exact(length)\n"
            "    if raw is None:\n"
            "        raise SystemExit(7)\n"
            "    text = raw.decode('utf-8')\n"
            "    if any(text in argument for argument in sys.argv[1:]):\n"
            "        reply(90)\n"
            "        raise SystemExit(9)\n"
            "    if text == 'SLOW':\n"
            "        time.sleep(5)\n"
            "    if text == 'FAIL':\n"
            "        reply(23)\n"
            "        raise SystemExit(9)\n"
            "    if text == 'DIE':\n"
            "        raise SystemExit(17)\n"
            "    if text == 'HUGE':\n"
            "        sys.stdout.buffer.write(struct.pack('<4sIQ', b'KXR1', 0, 64 * 1024 * 1024 + 1))\n"
            "        sys.stdout.buffer.flush()\n"
            "        continue\n"
            "    if text == 'BAD':\n"
            "        reply(0, b'not a wav')\n"
            "        continue\n"
            "    target = io.BytesIO()\n"
            "    with wave.open(target, 'wb') as audio:\n"
            "        audio.setnchannels(1)\n"
            "        audio.setsampwidth(2)\n"
            "        audio.setframerate(24000)\n"
            "        audio.writeframes(b'\\x00\\x00' * 2400)\n"
            "    reply(0, target.getvalue())\n",
            encoding="utf-8",
        )
        self.launcher.chmod(0o700)
        self.model = self.root / "model.gguf"
        self.projector = self.root / "projector.gguf"
        self.model.write_bytes(b"fixture model")
        self.projector.write_bytes(b"fixture projector")
        self.model.chmod(0o600)
        self.projector.chmod(0o600)
        self.paths = private_paths(self.root)
        self.arbiter = AcceleratorArbiter()
        self.admission = TtsRuntimeAdmission(
            launcher_sha256=hashlib.sha256(self.launcher.read_bytes()).hexdigest(),
            closure=PERSISTENT_TTS_RUNTIME_CLOSURE,
            vulkan_qualification_verified=True,
        )
        self.resolver_calls = 0

        def resolve(_model: str, _paths: RuntimePaths) -> tuple[Path, Path]:
            self.resolver_calls += 1
            return self.model, self.projector

        self.worker = PersistentVulkanTtsWorker(
            self.paths,
            self.arbiter,
            self.admission,
            launcher=self.launcher,
            launcher_uid=os.getuid(),
            trusted_root=self.trusted,
            artifact_resolver=resolve,
        )
        self.addCleanup(self.worker.close)

    def test_worker_is_reused_and_text_stays_off_argv(self) -> None:
        first = self.worker.synthesize("private words")
        initial = self.worker.record()
        second = self.worker.synthesize("private words")
        record = self.worker.record()
        self.assertEqual(first.wav, second.wav)
        self.assertEqual(first.frames, 2400)
        self.assertEqual(record["worker_pid"], initial["worker_pid"])
        self.assertEqual(record["counters"]["launches"], 1)
        self.assertEqual(record["counters"]["completed"], 2)
        self.assertTrue(record["worker_running"])
        self.assertFalse(record["text_in_argv"])
        self.assertFalse(record["text_logged"])
        self.assertGreaterEqual(first.queue_seconds, 0)
        self.assertGreater(first.run_seconds, 0)

    def test_worker_starts_only_after_llm_releases_accelerator(self) -> None:
        llm = self.arbiter.acquire(LLM_TURN)
        results = []

        def synthesize() -> None:
            results.append(self.worker.synthesize("queued speech", queue_timeout=2))

        thread = threading.Thread(target=synthesize)
        thread.start()
        deadline = time.monotonic() + 1
        while (
            self.arbiter.record()["queued"][TTS_UTTERANCE] != 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertFalse(self.worker.record()["worker_running"])
        llm.release()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].queue_seconds, 0)

    def test_cancellation_kills_worker_and_next_request_restarts_it(self) -> None:
        cancel = threading.Event()
        errors: list[BaseException] = []

        def synthesize() -> None:
            try:
                self.worker.synthesize(
                    "SLOW", run_timeout=10, cancelled=cancel.is_set
                )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=synthesize)
        thread.start()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            active = self.arbiter.record()["active"]
            if active is not None and active["workload"] == TTS_UTTERANCE:
                break
            time.sleep(0.01)
        cancel.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], SynthesisCancelled)
        self.assertFalse(self.worker.record()["worker_running"])
        self.assertIsNone(self.arbiter.record()["active"])
        self.worker.synthesize("recovered")
        self.assertEqual(self.worker.record()["counters"]["launches"], 2)

    def test_timeout_and_protocol_failures_drop_worker(self) -> None:
        with self.assertRaisesRegex(SynthesisError, "runtime limit"):
            self.worker.synthesize("SLOW", run_timeout=0.05)
        self.assertFalse(self.worker.record()["worker_running"])
        self.worker.synthesize("recovered once")
        with self.assertRaisesRegex(SynthesisError, "status 23"):
            self.worker.synthesize("FAIL")
        self.assertFalse(self.worker.record()["worker_running"])
        self.worker.synthesize("recovered twice")
        with self.assertRaisesRegex(SynthesisError, "response size"):
            self.worker.synthesize("BAD")
        self.assertFalse(self.worker.record()["worker_running"])
        with self.assertRaisesRegex(SynthesisError, "response size"):
            self.worker.synthesize("HUGE")
        self.assertFalse(self.worker.record()["worker_running"])
        self.assertEqual(self.worker.record()["counters"]["failures"], 4)

    def test_unexpected_exit_is_safe_and_restartable(self) -> None:
        with self.assertRaisesRegex(SynthesisError, "status 17"):
            self.worker.synthesize("DIE")
        self.assertFalse(self.worker.record()["worker_running"])
        result = self.worker.synthesize("back again")
        self.assertEqual(result.frames, 2400)

    def test_invalid_input_and_admission_fail_before_model_access(self) -> None:
        for text in ("", "\x00", "x" * (MAX_TEXT_BYTES + 1)):
            with self.subTest(length=len(text)), self.assertRaises(SynthesisError):
                self.worker.synthesize(text)
        self.assertEqual(self.resolver_calls, 0)
        invalid = replace(self.admission, closure="candidate:unknown")
        with self.assertRaises(SynthesisError):
            PersistentVulkanTtsWorker(
                self.paths,
                self.arbiter,
                invalid,
                launcher=self.launcher,
                launcher_uid=os.getuid(),
                trusted_root=self.trusted,
                artifact_resolver=lambda *_arguments: (_ for _ in ()).throw(
                    AssertionError("artifact resolver called")
                ),
            ).synthesize("hello")

    def test_close_is_idempotent_and_prevents_relaunch(self) -> None:
        self.worker.synthesize("hello")
        self.worker.close()
        self.worker.close()
        record = self.worker.record()
        self.assertTrue(record["closed"])
        self.assertFalse(record["worker_running"])
        with self.assertRaisesRegex(SynthesisError, "closed"):
            self.worker.synthesize("hello again")


if __name__ == "__main__":
    unittest.main()
