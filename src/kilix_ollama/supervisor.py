"""Fail-closed lifecycle supervision for the fixed Ollama runtime.

This module can start a process, but it cannot admit a transport or runtime by
itself.  The caller must provide exact external evidence binding the launcher
hash to either a verified Unix-listener delta or an owner-isolated TCP path.
Only the Unix path is executable in this development revision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import threading
import time
from typing import Callable

from .adapter import parse_version
from .backend import BackendSelection
from .errors import RuntimeContractError, SupervisorError
from .paths import RuntimePaths
from .service import DEFAULT_RUNTIME_LAUNCHER, RuntimeLaunchSpec, build_launch_spec
from .trust import trusted_executable_sha256


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_READINESS_REQUEST = (
    b"GET /api/version HTTP/1.1\r\n"
    b"Host: localhost\r\n"
    b"Accept: application/json\r\n"
    b"Connection: close\r\n\r\n"
)
MAX_READINESS_RESPONSE = 64 * 1024
DEFAULT_START_TIMEOUT = 30.0
DEFAULT_STOP_TIMEOUT = 5.0


@dataclass(frozen=True)
class TransportAdmission:
    """Exact local evidence; this is deliberately not a release schema pin."""

    transport: str
    runtime_sha256: str
    unix_listener_delta_verified: bool = False
    owner_isolation_verified: bool = False
    evidence_status: str = "development-local-qualification"

    def validate(self) -> None:
        if self.transport not in ("unix", "netns-tcp"):
            raise SupervisorError("transport admission names an unknown transport")
        if not isinstance(self.runtime_sha256, str) or not _SHA256.fullmatch(
            self.runtime_sha256
        ):
            raise SupervisorError("transport admission has an invalid runtime SHA-256")
        if self.evidence_status != "development-local-qualification":
            raise SupervisorError("transport admission has an unknown evidence status")
        if not isinstance(self.unix_listener_delta_verified, bool) or not isinstance(
            self.owner_isolation_verified, bool
        ):
            raise SupervisorError("transport admission flags must be booleans")

    def record(self) -> dict[str, object]:
        return {
            "transport": self.transport,
            "runtime_sha256": self.runtime_sha256,
            "unix_listener_delta_verified": self.unix_listener_delta_verified,
            "owner_isolation_verified": self.owner_isolation_verified,
            "evidence_status": self.evidence_status,
            "release_admission": False,
        }


def _ensure_private_directory(path: Path, *, uid: int) -> None:
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise SupervisorError(f"cannot find an existing ancestor for {path}")
        cursor = parent
    try:
        anchor = cursor.lstat()
    except OSError as error:
        raise SupervisorError(f"cannot inspect private path anchor {cursor}: {error}") from error
    if stat.S_ISLNK(anchor.st_mode) or not stat.S_ISDIR(anchor.st_mode):
        raise SupervisorError(f"private path anchor is not a real directory: {cursor}")
    if anchor.st_uid not in (0, uid) or anchor.st_mode & 0o022:
        raise SupervisorError(
            f"private path anchor is not owner-controlled and non-writable: {cursor}"
        )

    for candidate in reversed(missing):
        try:
            candidate.mkdir(mode=0o700)
        except OSError as error:
            raise SupervisorError(
                f"cannot create private runtime path {candidate}: {error}"
            ) from error
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise SupervisorError(
                f"cannot inspect new private runtime path {candidate}: {error}"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SupervisorError(f"new private runtime path is unsafe: {candidate}")

    try:
        metadata = path.lstat()
    except OSError as error:
        raise SupervisorError(f"cannot inspect private runtime path {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SupervisorError(f"private runtime path is not a real directory: {path}")
    if metadata.st_uid != uid:
        raise SupervisorError(
            f"private runtime path {path} is owned by uid {metadata.st_uid}, expected {uid}"
        )
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise SupervisorError(f"private runtime path must have mode 0700: {path}")


def _socket_identity(path: Path, *, uid: int) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SupervisorError(f"upstream Unix socket is unavailable: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISSOCK(metadata.st_mode):
        raise SupervisorError("upstream listener path is not a Unix socket")
    if metadata.st_uid != uid or metadata.st_mode & 0o077:
        raise SupervisorError("upstream Unix socket is not owner-private")
    return metadata.st_dev, metadata.st_ino


def _decode_http_response(payload: bytes) -> bytes:
    if len(payload) > MAX_READINESS_RESPONSE:
        raise SupervisorError("upstream readiness response exceeds its limit")
    try:
        header, body = payload.split(b"\r\n\r\n", 1)
        lines = header.split(b"\r\n")
        version, status, _reason = lines[0].split(b" ", 2)
        status_code = int(status)
    except (ValueError, IndexError) as error:
        raise SupervisorError("upstream readiness response is malformed") from error
    if version not in (b"HTTP/1.0", b"HTTP/1.1") or status_code != 200:
        raise SupervisorError("upstream readiness probe did not return HTTP 200")
    content_length: int | None = None
    for raw in lines[1:]:
        if b":" not in raw:
            raise SupervisorError("upstream readiness header is malformed")
        name, value = raw.split(b":", 1)
        if name.lower() == b"content-length":
            if content_length is not None:
                raise SupervisorError("upstream readiness response repeats Content-Length")
            try:
                content_length = int(value.strip())
            except ValueError as error:
                raise SupervisorError("upstream readiness Content-Length is invalid") from error
    if content_length is not None and content_length != len(body):
        raise SupervisorError("upstream readiness body length is inconsistent")
    return body


class RuntimeSupervisor:
    """Own one process group and expose prompt-free lifecycle status."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        launcher: Path | None = None,
        launcher_uid: int = 0,
        trusted_root: Path = Path("/"),
        owner_uid: int | None = None,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self.paths = paths
        self.launcher = (
            Path(launcher) if launcher is not None else DEFAULT_RUNTIME_LAUNCHER
        )
        self.launcher_uid = launcher_uid
        self.trusted_root = Path(trusted_root)
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        if (
            not isinstance(self.owner_uid, int)
            or isinstance(self.owner_uid, bool)
            or self.owner_uid < 0
        ):
            raise RuntimeContractError("supervisor owner uid must be non-negative")
        if not callable(popen_factory):
            raise RuntimeContractError("popen_factory must be callable")
        self._popen = popen_factory
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._state = "stopped"
        self._last_exit: int | None = None
        self._failure: str | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._monitor: threading.Thread | None = None

    def _expected_spec(self, spec: RuntimeLaunchSpec, launcher: Path) -> RuntimeLaunchSpec:
        selection = BackendSelection(
            requested=spec.backend,
            selected=spec.backend,
            reason="supervisor structural validation",
            qualified_backends=(spec.backend,),
        )
        expected = build_launch_spec(
            self.paths,
            selection,
            transport=spec.transport,
            runtime_profile=spec.runtime_profile,
        )
        return replace(
            expected,
            executable=launcher,
            argv=(str(launcher), "serve"),
        )

    def _validate_spec(self, spec: RuntimeLaunchSpec, admission: TransportAdmission) -> Path:
        if not isinstance(spec, RuntimeLaunchSpec):
            raise SupervisorError("runtime launch specification has an invalid type")
        admission.validate()
        launcher = self.launcher
        if spec != self._expected_spec(spec, launcher):
            raise SupervisorError("runtime launch specification is not the exact fixed contract")
        if admission.transport != spec.transport:
            raise SupervisorError("transport evidence does not match the launch specification")
        try:
            actual_sha256 = trusted_executable_sha256(
                launcher,
                expected_uid=self.launcher_uid,
                trusted_root=self.trusted_root,
            )
        except RuntimeContractError as error:
            raise SupervisorError(str(error)) from error
        if actual_sha256 != admission.runtime_sha256:
            raise SupervisorError("runtime launcher does not match the admitted SHA-256")
        if spec.requires_listener_delta and not admission.unix_listener_delta_verified:
            raise SupervisorError("the Unix-listener delta has not been verified")
        if spec.requires_isolation_proof and not admission.owner_isolation_verified:
            raise SupervisorError("owner-specific TCP isolation has not been verified")
        if spec.transport != "unix":
            raise SupervisorError(
                "owner-isolated TCP supervision is not implemented; use a verified Unix listener"
            )
        return launcher

    def _prepare_paths(self) -> None:
        for path in (
            self.paths.runtime_dir,
            self.paths.temp_dir,
            self.paths.state_dir,
            self.paths.upstream_home,
            self.paths.data_dir,
            self.paths.model_store,
            self.paths.cache_dir,
        ):
            _ensure_private_directory(path, uid=self.owner_uid)
        if os.path.lexists(self.paths.upstream_socket):
            raise SupervisorError(
                f"refusing an occupied upstream socket path: {self.paths.upstream_socket}"
            )

    def _default_readiness(
        self,
        spec: RuntimeLaunchSpec,
        process: subprocess.Popen[bytes],
        *,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "listener has not appeared"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise SupervisorError(
                    f"runtime exited during startup with status {process.returncode}"
                )
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.settimeout(min(0.5, max(0.01, deadline - time.monotonic())))
                connection.connect(str(self.paths.upstream_socket))
                identity = _socket_identity(
                    self.paths.upstream_socket,
                    uid=self.owner_uid,
                )
                if (
                    self._socket_identity is not None
                    and self._socket_identity != identity
                ):
                    raise SupervisorError("upstream Unix socket changed during readiness")
                self._socket_identity = identity
                connection.sendall(_READINESS_REQUEST)
                chunks: list[bytes] = []
                size = 0
                while True:
                    block = connection.recv(4096)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_READINESS_RESPONSE:
                        raise SupervisorError(
                            "upstream readiness response exceeds its limit"
                        )
                    chunks.append(block)
                body = _decode_http_response(b"".join(chunks))
                parse_version(body)
                return
            except (OSError, SupervisorError, RuntimeContractError) as error:
                last_error = str(error)
            finally:
                connection.close()
            time.sleep(0.05)
        raise SupervisorError(f"runtime readiness timed out: {last_error}")

    def _watch(self, process: subprocess.Popen[bytes]) -> None:
        returncode = process.wait()
        with self._lock:
            if self._process is not process:
                return
            self._last_exit = returncode
            self._process = None
            if self._state == "stopping":
                self._state = "stopped"
                self._failure = None
            else:
                self._state = "failed"
                self._failure = f"runtime exited with status {returncode}"

    def start(
        self,
        spec: RuntimeLaunchSpec,
        admission: TransportAdmission,
        *,
        timeout: float = DEFAULT_START_TIMEOUT,
        readiness: Callable[[RuntimeLaunchSpec, subprocess.Popen[bytes]], None]
        | None = None,
    ) -> int:
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise RuntimeContractError("supervisor start timeout must be positive")
        with self._lock:
            if self._process is not None or self._state in ("starting", "running", "stopping"):
                raise SupervisorError("runtime is already active")
            launcher = self._validate_spec(spec, admission)
            self._prepare_paths()
            self._state = "starting"
            self._failure = None
            self._last_exit = None
            self._socket_identity = None
            environment = spec.environ
            try:
                process = self._popen(
                    spec.argv,
                    executable=str(launcher),
                    cwd=str(self.paths.runtime_dir),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                    umask=0o077,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self._state = "failed"
                self._failure = f"runtime launch failed: {error}"
                raise SupervisorError(self._failure) from error
            self._process = process

        try:
            if readiness is None:
                self._default_readiness(spec, process, timeout=float(timeout))
            else:
                readiness(spec, process)
            if process.poll() is not None:
                raise SupervisorError(
                    f"runtime exited during startup with status {process.returncode}"
                )
        except BaseException as error:
            self._terminate_process(process, timeout=min(float(timeout), DEFAULT_STOP_TIMEOUT))
            with self._lock:
                if self._process is process:
                    self._process = None
                self._last_exit = process.returncode
                self._state = "failed"
                self._failure = str(error)
            self._remove_own_socket()
            if isinstance(error, SupervisorError):
                raise
            raise SupervisorError(f"runtime readiness check failed: {error}") from error

        with self._lock:
            if self._process is not process:
                raise SupervisorError("runtime changed during startup")
            self._state = "running"
            monitor = threading.Thread(
                target=self._watch,
                args=(process,),
                name="kilix-ollama-runtime-watch",
                daemon=True,
            )
            self._monitor = monitor
            monitor.start()
            return process.pid

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes], *, timeout: float) -> None:
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
            raise SupervisorError("runtime process group did not stop after SIGKILL") from error

    def _remove_own_socket(self) -> None:
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            metadata = self.paths.upstream_socket.lstat()
        except FileNotFoundError:
            return
        except OSError:
            return
        if (
            stat.S_ISSOCK(metadata.st_mode)
            and metadata.st_uid == self.owner_uid
            and (metadata.st_dev, metadata.st_ino) == identity
        ):
            try:
                self.paths.upstream_socket.unlink()
            except OSError:
                pass

    def stop(self, *, timeout: float = DEFAULT_STOP_TIMEOUT) -> bool:
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise RuntimeContractError("supervisor stop timeout must be positive")
        with self._lock:
            process = self._process
            if process is None:
                self._remove_own_socket()
                if self._state != "failed":
                    self._state = "stopped"
                return False
            self._state = "stopping"
        self._terminate_process(process, timeout=float(timeout))
        with self._lock:
            if self._process is process:
                self._process = None
            self._last_exit = process.returncode
            self._state = "stopped"
            self._failure = None
            self._remove_own_socket()
        return True

    def record(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            return {
                "schema": "kilix.ollama.runtime-supervisor/v0-unfrozen",
                "schema_status": "development-unfrozen",
                "state": self._state,
                "pid": process.pid if process is not None else None,
                "last_exit_status": self._last_exit,
                "failure": self._failure,
                "transport": "unix-only",
                "arbitrary_argv": False,
                "inherited_environment": False,
                "request_content_logged": False,
                "weight_access": False,
            }


__all__ = [
    "DEFAULT_START_TIMEOUT",
    "DEFAULT_STOP_TIMEOUT",
    "MAX_READINESS_RESPONSE",
    "RuntimeSupervisor",
    "TransportAdmission",
]
