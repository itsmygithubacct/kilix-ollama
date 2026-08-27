"""Vulkan Pocket TTS execution behind the shared accelerator lease.

The launcher and model pair are fixed and independently verified.  Caller text
never enters argv or logs.  The preferred worker keeps weights resident and
uses a bounded binary pipe protocol; the one-shot file path remains available
as an independently testable correctness fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import re
import select
import signal
import stat
import struct
import subprocess
import tempfile
import threading
import time
import wave

from .arbiter import TTS_UTTERANCE, AcceleratorArbiter
from .errors import (
    AcceleratorRequestCancelled,
    AcceleratorRequestTimeout,
    RuntimeContractError,
    SynthesisCancelled,
    SynthesisError,
)
from .paths import RuntimePaths
from .supervisor import _ensure_private_directory
from .trust import trusted_executable_sha256
from .tts_artifacts import MODEL_ID, verified_runtime_artifacts


DEFAULT_TTS_LAUNCHER = Path("/usr/lib/plebian-os/ollama/pocket-tts-current")
DEFAULT_PERSISTENT_TTS_LAUNCHER = Path(
    "/usr/lib/plebian-os/ollama/kilix-tts-worker-current"
)
TTS_RUNTIME_CLOSURE = "candidate:plebian-os-llama.cpp-b10434-pocket-vulkan-r1"
PERSISTENT_TTS_RUNTIME_CLOSURE = (
    "candidate:plebian-os-llama.cpp-b10434-pocket-vulkan-worker-r1"
)
MAX_TEXT_BYTES = 4096
MAX_WAV_BYTES = 64 * 1024 * 1024
MAX_AUDIO_SECONDS = 120.0
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2
DEFAULT_QUEUE_TIMEOUT = 30.0
DEFAULT_RUN_TIMEOUT = 30.0
_POLL_SECONDS = 0.02
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_READY_MAGIC = b"KILIXT1\n"
_REQUEST_MAGIC = b"KXQ1"
_RESPONSE_MAGIC = b"KXR1"
_RESPONSE_HEADER = struct.Struct("<4sIQ")
_REQUEST_HEADER = struct.Struct("<4sI")


@dataclass(frozen=True)
class TtsRuntimeAdmission:
    launcher_sha256: str
    closure: str = TTS_RUNTIME_CLOSURE
    backend: str = "vulkan"
    vulkan_qualification_verified: bool = False
    evidence_status: str = "development-local-qualification"

    def validate(self, *, expected_closure: str = TTS_RUNTIME_CLOSURE) -> None:
        if not isinstance(self.launcher_sha256, str) or not _SHA256.fullmatch(
            self.launcher_sha256
        ):
            raise SynthesisError("TTS admission has an invalid launcher SHA-256")
        if self.closure != expected_closure:
            raise SynthesisError("TTS admission names an unknown runtime closure")
        if self.backend != "vulkan" or self.vulkan_qualification_verified is not True:
            raise SynthesisError("the Vulkan TTS runtime is not externally qualified")
        if self.evidence_status != "development-local-qualification":
            raise SynthesisError("TTS admission has an unknown evidence status")

    def record(self) -> dict[str, object]:
        return {
            "launcher_sha256": self.launcher_sha256,
            "closure": self.closure,
            "backend": self.backend,
            "vulkan_qualification_verified": self.vulkan_qualification_verified,
            "evidence_status": self.evidence_status,
            "release_admission": False,
        }


@dataclass(frozen=True)
class SynthesisResult:
    wav: bytes
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    elapsed_seconds: float
    queue_seconds: float
    run_seconds: float

    @property
    def audio_seconds(self) -> float:
        return self.frames / self.sample_rate

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.tts-result/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "wav_bytes": len(self.wav),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "frames": self.frames,
            "audio_seconds": self.audio_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "queue_seconds": self.queue_seconds,
            "run_seconds": self.run_seconds,
            "text_recorded": False,
            "local_verified_weights_used": True,
        }


def _positive_timeout(value: float, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuntimeContractError(f"{label} must be finite and positive")
    return float(value)


def _text_bytes(text: str) -> bytes:
    if not isinstance(text, str) or not text.strip():
        raise SynthesisError("speech text must be a non-empty string")
    if "\x00" in text:
        raise SynthesisError("speech text contains a NUL byte")
    try:
        payload = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise SynthesisError("speech text is not valid Unicode") from error
    if len(payload) > MAX_TEXT_BYTES:
        raise SynthesisError(f"speech text exceeds the {MAX_TEXT_BYTES}-byte utterance limit")
    return payload


def _write_private(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise SynthesisError(f"cannot stage private speech input: {error}") from error


def _terminate(process: subprocess.Popen[bytes], *, timeout: float = 1.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise SynthesisError("TTS process group did not stop after SIGKILL") from error


def _read_wav(path: Path, *, uid: int) -> tuple[bytes, int]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise SynthesisError(f"TTS did not produce a readable WAV: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != uid:
            raise SynthesisError("TTS output is not an owner-controlled regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise SynthesisError("TTS output must have mode 0600")
        if metadata.st_size <= 44 or metadata.st_size > MAX_WAV_BYTES:
            raise SynthesisError("TTS output size is outside the WAV limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            try:
                with wave.open(stream, "rb") as audio:
                    if (
                        audio.getnchannels() != CHANNELS
                        or audio.getsampwidth() != SAMPLE_WIDTH
                        or audio.getframerate() != SAMPLE_RATE
                        or audio.getcomptype() != "NONE"
                    ):
                        raise SynthesisError(
                            "TTS output is not 24 kHz mono 16-bit PCM"
                        )
                    frames = audio.getnframes()
                    if frames <= 0 or frames / SAMPLE_RATE > MAX_AUDIO_SECONDS:
                        raise SynthesisError("TTS output duration is outside the limit")
            except (EOFError, wave.Error) as error:
                raise SynthesisError(f"TTS output is not a valid WAV: {error}") from error
            stream.seek(0)
            payload = stream.read(MAX_WAV_BYTES + 1)
            if len(payload) != metadata.st_size:
                raise SynthesisError("TTS output changed while it was being read")
            return payload, frames
    finally:
        os.close(descriptor)


def _validate_wav_payload(payload: bytes) -> int:
    if not isinstance(payload, bytes) or len(payload) <= 44 or len(payload) > MAX_WAV_BYTES:
        raise SynthesisError("TTS output size is outside the WAV limit")
    try:
        with wave.open(io.BytesIO(payload), "rb") as audio:
            if (
                audio.getnchannels() != CHANNELS
                or audio.getsampwidth() != SAMPLE_WIDTH
                or audio.getframerate() != SAMPLE_RATE
                or audio.getcomptype() != "NONE"
            ):
                raise SynthesisError("TTS output is not 24 kHz mono 16-bit PCM")
            frames = audio.getnframes()
            if frames <= 0 or frames / SAMPLE_RATE > MAX_AUDIO_SECONDS:
                raise SynthesisError("TTS output duration is outside the limit")
            expected_pcm_bytes = frames * CHANNELS * SAMPLE_WIDTH
            if len(audio.readframes(frames + 1)) != expected_pcm_bytes:
                raise SynthesisError("TTS WAV contains an incomplete PCM payload")
    except (EOFError, wave.Error) as error:
        raise SynthesisError(f"TTS output is not a valid WAV: {error}") from error
    return frames


def _cleanup_stage(stage: Path) -> None:
    # Unlink only the two names this runner owns.  An unexpected child-created
    # entry leaves the mode-0700 stage behind for diagnosis, never recursively
    # deletes an unverified tree, and never retains the prompt.
    for name in ("prompt.txt", "output.wav"):
        path = stage / name
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        stage.rmdir()
    except OSError:
        pass


class VulkanTtsRunner:
    """Run the exact Pocket Q8 pair as one bounded TTS utterance."""

    def __init__(
        self,
        paths: RuntimePaths,
        arbiter: AcceleratorArbiter,
        admission: TtsRuntimeAdmission,
        *,
        launcher: Path = DEFAULT_TTS_LAUNCHER,
        launcher_uid: int = 0,
        trusted_root: Path = Path("/"),
        owner_uid: int | None = None,
        artifact_resolver: Callable[[str, RuntimePaths], tuple[Path, Path]]
        | None = None,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self.paths = paths
        self.arbiter = arbiter
        self.admission = admission
        self.launcher = Path(launcher)
        self.launcher_uid = launcher_uid
        self.trusted_root = Path(trusted_root)
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        self._resolve = (
            (lambda model, runtime_paths: verified_runtime_artifacts(model, paths=runtime_paths))
            if artifact_resolver is None
            else artifact_resolver
        )
        if not callable(self._resolve) or not callable(popen_factory):
            raise RuntimeContractError("TTS runtime dependencies must be callable")
        self._popen = popen_factory

    def _validate_runtime(self) -> None:
        self.admission.validate()
        try:
            actual = trusted_executable_sha256(
                self.launcher,
                expected_uid=self.launcher_uid,
                trusted_root=self.trusted_root,
            )
        except RuntimeContractError as error:
            raise SynthesisError(str(error)) from error
        if actual != self.admission.launcher_sha256:
            raise SynthesisError("TTS launcher does not match the admitted SHA-256")

    def _environment(self, stage: Path) -> dict[str, str]:
        home = self.paths.state_dir / "tts-home"
        cache = self.paths.cache_dir / "vulkan-tts"
        _ensure_private_directory(home, uid=self.owner_uid)
        _ensure_private_directory(cache, uid=self.owner_uid)
        return {
            "GGML_VK_FORCE_MMVQ": "1",
            "HF_HUB_OFFLINE": "1",
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "NO_PROXY": "*",
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(stage),
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": str(cache),
            "no_proxy": "*",
        }

    def synthesize(
        self,
        text: str,
        *,
        model: str = MODEL_ID,
        queue_timeout: float = DEFAULT_QUEUE_TIMEOUT,
        run_timeout: float = DEFAULT_RUN_TIMEOUT,
        cancelled: Callable[[], bool] | None = None,
    ) -> SynthesisResult:
        payload = _text_bytes(text)
        wait_limit = _positive_timeout(queue_timeout, "TTS queue timeout")
        run_limit = _positive_timeout(run_timeout, "TTS run timeout")
        if cancelled is not None and not callable(cancelled):
            raise RuntimeContractError("TTS cancellation probe must be callable")
        self._validate_runtime()
        try:
            model_path, projector_path = self._resolve(model, self.paths)
        except Exception as error:
            if isinstance(error, SynthesisError):
                raise
            raise SynthesisError(f"TTS model pair is unavailable: {error}") from error
        if not isinstance(model_path, Path) or not isinstance(projector_path, Path):
            raise SynthesisError("TTS artifact resolver returned an invalid model pair")

        _ensure_private_directory(self.paths.runtime_dir, uid=self.owner_uid)
        _ensure_private_directory(self.paths.temp_dir, uid=self.owner_uid)
        stage = Path(tempfile.mkdtemp(prefix=".tts-request-", dir=self.paths.temp_dir))
        stage.chmod(0o700)
        prompt = stage / "prompt.txt"
        output = stage / "output.wav"
        _write_private(prompt, payload)
        lease = None
        process: subprocess.Popen[bytes] | None = None
        started = time.monotonic()
        queue_started = started
        try:
            try:
                lease = self.arbiter.acquire(
                    TTS_UTTERANCE,
                    timeout=wait_limit,
                    cancelled=cancelled,
                )
            except AcceleratorRequestCancelled as error:
                raise SynthesisCancelled("speech synthesis was cancelled while queued") from error
            except AcceleratorRequestTimeout as error:
                raise SynthesisError("speech synthesis timed out in the accelerator queue") from error
            run_started = time.monotonic()
            queue_seconds = run_started - queue_started
            arguments = (
                str(self.launcher),
                "--model",
                str(model_path),
                "--mmproj",
                str(projector_path),
                "--file",
                str(prompt),
                "--output",
                str(output),
                "--ctx-size",
                "512",
                "--batch-size",
                "128",
                "--ubatch-size",
                "128",
                "--gpu-layers",
                "all",
                "--offline",
                "--no-escape",
                "--log-disable",
            )
            try:
                process = self._popen(
                    arguments,
                    executable=str(self.launcher),
                    cwd=str(stage),
                    env=self._environment(stage),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                    umask=0o077,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                raise SynthesisError(f"could not start local TTS: {error}") from error
            deadline = time.monotonic() + run_limit
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    _terminate(process)
                    raise SynthesisCancelled("speech synthesis was cancelled")
                if time.monotonic() >= deadline:
                    _terminate(process)
                    raise SynthesisError("speech synthesis exceeded its runtime limit")
                time.sleep(_POLL_SECONDS)
            if process.returncode != 0:
                raise SynthesisError(
                    f"local TTS exited with status {process.returncode}"
                )
            wav, frames = _read_wav(output, uid=self.owner_uid)
            finished = time.monotonic()
            return SynthesisResult(
                wav=wav,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                sample_width=SAMPLE_WIDTH,
                frames=frames,
                elapsed_seconds=finished - started,
                queue_seconds=queue_seconds,
                run_seconds=finished - run_started,
            )
        finally:
            if process is not None and process.poll() is None:
                _terminate(process)
            if lease is not None:
                lease.release()
            _cleanup_stage(stage)

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.vulkan-tts-runner/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "launcher": str(self.launcher),
            "admission": self.admission.record(),
            "model": MODEL_ID,
            "shape": {"context": 512, "batch": 128, "ubatch": 128},
            "shared_arbiter": True,
            "text_in_argv": False,
            "text_logged": False,
            "network_access": False,
            "model_publication": False,
            "local_verified_weights_required": True,
        }


def _read_worker_bytes(
    process: subprocess.Popen[bytes],
    length: int,
    *,
    deadline: float,
    cancelled: Callable[[], bool] | None,
    phase: str,
) -> bytes:
    if process.stdout is None:
        raise SynthesisError("persistent TTS worker has no protocol output pipe")
    output = bytearray()
    descriptor = process.stdout.fileno()
    while len(output) < length:
        if cancelled is not None and cancelled():
            raise SynthesisCancelled("speech synthesis was cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SynthesisError(f"persistent TTS {phase} exceeded its runtime limit")
        try:
            readable, _, _ = select.select(
                (descriptor,), (), (), min(remaining, _POLL_SECONDS)
            )
        except (OSError, ValueError) as error:
            raise SynthesisError(
                f"persistent TTS {phase} pipe is unavailable: {error}"
            ) from error
        if not readable:
            continue
        try:
            chunk = os.read(descriptor, length - len(output))
        except OSError as error:
            raise SynthesisError(
                f"persistent TTS {phase} pipe read failed: {error}"
            ) from error
        if not chunk:
            status = process.poll()
            detail = "" if status is None else f" with status {status}"
            raise SynthesisError(
                f"persistent TTS worker ended during {phase}{detail}"
            )
        output.extend(chunk)
    return bytes(output)


def _write_worker_bytes(process: subprocess.Popen[bytes], payload: bytes) -> None:
    if process.stdin is None:
        raise SynthesisError("persistent TTS worker has no protocol input pipe")
    view = memoryview(payload)
    offset = 0
    descriptor = process.stdin.fileno()
    while offset < len(view):
        try:
            written = os.write(descriptor, view[offset:])
        except (BrokenPipeError, OSError) as error:
            raise SynthesisError(
                f"persistent TTS request pipe write failed: {error}"
            ) from error
        if written <= 0:
            raise SynthesisError("persistent TTS request pipe made no progress")
        offset += written


class PersistentVulkanTtsWorker:
    """Keep the admitted Pocket model pair resident behind a framed pipe."""

    def __init__(
        self,
        paths: RuntimePaths,
        arbiter: AcceleratorArbiter,
        admission: TtsRuntimeAdmission,
        *,
        launcher: Path = DEFAULT_PERSISTENT_TTS_LAUNCHER,
        launcher_uid: int = 0,
        trusted_root: Path = Path("/"),
        owner_uid: int | None = None,
        artifact_resolver: Callable[[str, RuntimePaths], tuple[Path, Path]]
        | None = None,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self.paths = paths
        self.arbiter = arbiter
        self.admission = admission
        self.launcher = Path(launcher)
        self.launcher_uid = launcher_uid
        self.trusted_root = Path(trusted_root)
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        self._resolve = (
            (lambda model, runtime_paths: verified_runtime_artifacts(model, paths=runtime_paths))
            if artifact_resolver is None
            else artifact_resolver
        )
        if not callable(self._resolve) or not callable(popen_factory):
            raise RuntimeContractError("TTS runtime dependencies must be callable")
        self._popen = popen_factory
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._worker_artifacts: tuple[Path, Path] | None = None
        self._closed = False
        self._launches = 0
        self._completed = 0
        self._failures = 0
        self._unexpected_exits = 0

    def _validate_runtime(self) -> None:
        self.admission.validate(expected_closure=PERSISTENT_TTS_RUNTIME_CLOSURE)
        try:
            actual = trusted_executable_sha256(
                self.launcher,
                expected_uid=self.launcher_uid,
                trusted_root=self.trusted_root,
            )
        except RuntimeContractError as error:
            raise SynthesisError(str(error)) from error
        if actual != self.admission.launcher_sha256:
            raise SynthesisError(
                "persistent TTS worker does not match the admitted SHA-256"
            )

    def _environment(self) -> dict[str, str]:
        home = self.paths.state_dir / "tts-home"
        cache = self.paths.cache_dir / "vulkan-tts"
        _ensure_private_directory(home, uid=self.owner_uid)
        _ensure_private_directory(cache, uid=self.owner_uid)
        _ensure_private_directory(self.paths.temp_dir, uid=self.owner_uid)
        return {
            "GGML_VK_FORCE_MMVQ": "1",
            "HF_HUB_OFFLINE": "1",
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "NO_PROXY": "*",
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(self.paths.temp_dir),
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": str(cache),
            "no_proxy": "*",
        }

    def _discard_worker(self, *, graceful: bool = False) -> None:
        process = self._process
        self._process = None
        self._worker_artifacts = None
        if process is None:
            return
        if graceful and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.close()
                process.wait(timeout=1.0)
            except (OSError, subprocess.SubprocessError):
                pass
        if process.poll() is None:
            try:
                _terminate(process)
            except SynthesisError:
                pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _start_worker(
        self,
        artifacts: tuple[Path, Path],
        *,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> subprocess.Popen[bytes]:
        model_path, projector_path = artifacts
        _ensure_private_directory(self.paths.runtime_dir, uid=self.owner_uid)
        arguments = (
            str(self.launcher),
            "--model",
            str(model_path),
            "--mmproj",
            str(projector_path),
            "--ctx-size",
            "512",
            "--batch-size",
            "128",
            "--ubatch-size",
            "128",
            "--gpu-layers",
            "all",
            "--seed",
            "1",
            "--offline",
            "--no-escape",
            "--log-disable",
        )
        try:
            process = self._popen(
                arguments,
                executable=str(self.launcher),
                cwd=str(self.paths.runtime_dir),
                env=self._environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                umask=0o077,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise SynthesisError(
                f"could not start persistent local TTS: {error}"
            ) from error
        self._process = process
        self._launches += 1
        ready = _read_worker_bytes(
            process,
            len(_READY_MAGIC),
            deadline=deadline,
            cancelled=cancelled,
            phase="startup",
        )
        if ready != _READY_MAGIC:
            raise SynthesisError("persistent TTS worker sent invalid readiness data")
        self._worker_artifacts = artifacts
        return process

    def _ensure_worker(
        self,
        artifacts: tuple[Path, Path],
        *,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> subprocess.Popen[bytes]:
        if self._closed:
            raise SynthesisError("persistent TTS worker is closed")
        process = self._process
        if process is not None and process.poll() is not None:
            self._unexpected_exits += 1
            self._discard_worker()
            process = None
        if process is not None and self._worker_artifacts != artifacts:
            self._discard_worker(graceful=True)
            process = None
        if process is None:
            process = self._start_worker(
                artifacts,
                deadline=deadline,
                cancelled=cancelled,
            )
        return process

    def synthesize(
        self,
        text: str,
        *,
        model: str = MODEL_ID,
        queue_timeout: float = DEFAULT_QUEUE_TIMEOUT,
        run_timeout: float = DEFAULT_RUN_TIMEOUT,
        cancelled: Callable[[], bool] | None = None,
    ) -> SynthesisResult:
        payload = _text_bytes(text)
        wait_limit = _positive_timeout(queue_timeout, "TTS queue timeout")
        run_limit = _positive_timeout(run_timeout, "TTS run timeout")
        if cancelled is not None and not callable(cancelled):
            raise RuntimeContractError("TTS cancellation probe must be callable")
        self._validate_runtime()
        try:
            model_path, projector_path = self._resolve(model, self.paths)
        except Exception as error:
            if isinstance(error, SynthesisError):
                raise
            raise SynthesisError(f"TTS model pair is unavailable: {error}") from error
        if not isinstance(model_path, Path) or not isinstance(projector_path, Path):
            raise SynthesisError("TTS artifact resolver returned an invalid model pair")
        artifacts = (model_path, projector_path)

        started = time.monotonic()
        queue_started = started
        lease = None
        try:
            try:
                lease = self.arbiter.acquire(
                    TTS_UTTERANCE,
                    timeout=wait_limit,
                    cancelled=cancelled,
                )
            except AcceleratorRequestCancelled as error:
                raise SynthesisCancelled(
                    "speech synthesis was cancelled while queued"
                ) from error
            except AcceleratorRequestTimeout as error:
                raise SynthesisError(
                    "speech synthesis timed out in the accelerator queue"
                ) from error
            queue_finished = time.monotonic()

            with self._lock:
                run_started = time.monotonic()
                deadline = run_started + run_limit
                try:
                    if self._closed:
                        raise SynthesisError("persistent TTS worker is closed")
                    if cancelled is not None and cancelled():
                        raise SynthesisCancelled("speech synthesis was cancelled")
                    process = self._ensure_worker(
                        artifacts,
                        deadline=deadline,
                        cancelled=cancelled,
                    )
                    request = _REQUEST_HEADER.pack(_REQUEST_MAGIC, len(payload)) + payload
                    _write_worker_bytes(process, request)
                    header = _read_worker_bytes(
                        process,
                        _RESPONSE_HEADER.size,
                        deadline=deadline,
                        cancelled=cancelled,
                        phase="response",
                    )
                    magic, status, length = _RESPONSE_HEADER.unpack(header)
                    if magic != _RESPONSE_MAGIC:
                        raise SynthesisError(
                            "persistent TTS worker sent an invalid response magic"
                        )
                    if status != 0:
                        if length != 0:
                            raise SynthesisError(
                                "persistent TTS worker sent an invalid error response"
                            )
                        raise SynthesisError(
                            f"persistent TTS worker rejected synthesis with status {status}"
                        )
                    if length <= 44 or length > MAX_WAV_BYTES:
                        raise SynthesisError(
                            "persistent TTS response size is outside the WAV limit"
                        )
                    wav = _read_worker_bytes(
                        process,
                        length,
                        deadline=deadline,
                        cancelled=cancelled,
                        phase="response",
                    )
                    frames = _validate_wav_payload(wav)
                except (SynthesisCancelled, SynthesisError):
                    self._failures += 1
                    self._discard_worker()
                    raise
                finished = time.monotonic()
                self._completed += 1
                return SynthesisResult(
                    wav=wav,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    sample_width=SAMPLE_WIDTH,
                    frames=frames,
                    elapsed_seconds=finished - started,
                    queue_seconds=queue_finished - queue_started,
                    run_seconds=finished - run_started,
                )
        finally:
            if lease is not None:
                lease.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._discard_worker(graceful=True)

    def __enter__(self) -> "PersistentVulkanTtsWorker":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def record(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            return {
                "schema": "kilix.ollama.persistent-vulkan-tts/v0-unfrozen",
                "schema_status": "development-unfrozen",
                "launcher": str(self.launcher),
                "admission": self.admission.record(),
                "model": MODEL_ID,
                "shape": {"context": 512, "batch": 128, "ubatch": 128},
                "protocol": "kilix-tts-pipe/v1",
                "persistent_model": True,
                "worker_running": running,
                "worker_pid": process.pid if running else None,
                "closed": self._closed,
                "counters": {
                    "launches": self._launches,
                    "completed": self._completed,
                    "failures": self._failures,
                    "unexpected_exits": self._unexpected_exits,
                },
                "shared_arbiter": True,
                "text_in_argv": False,
                "text_logged": False,
                "network_access": False,
                "model_publication": False,
                "local_verified_weights_required": True,
            }


__all__ = [
    "CHANNELS",
    "DEFAULT_PERSISTENT_TTS_LAUNCHER",
    "DEFAULT_QUEUE_TIMEOUT",
    "DEFAULT_RUN_TIMEOUT",
    "DEFAULT_TTS_LAUNCHER",
    "MAX_AUDIO_SECONDS",
    "MAX_TEXT_BYTES",
    "MAX_WAV_BYTES",
    "PERSISTENT_TTS_RUNTIME_CLOSURE",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "TTS_RUNTIME_CLOSURE",
    "SynthesisResult",
    "PersistentVulkanTtsWorker",
    "TtsRuntimeAdmission",
    "VulkanTtsRunner",
]
