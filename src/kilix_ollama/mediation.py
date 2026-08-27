"""Private same-UID HTTP mediation for an upstream Ollama Unix listener."""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import re
import select
import socket
import socketserver
import stat
import struct
import threading

from .arbiter import LLM_TURN, AcceleratorArbiter
from .errors import (
    AcceleratorQueueFull,
    AcceleratorRequestCancelled,
    AcceleratorRequestTimeout,
    AcceleratorUnavailable,
    PolicyDenied,
    RuntimeContractError,
    UpstreamProtocolError,
)
from .paths import RuntimePaths, private_directory_status
from .policy import (
    MAX_REQUEST_BODY,
    MediationRequest,
    forward_response_headers,
    verify_peer,
)


MAX_REQUEST_HEADERS = 64 * 1024
MAX_HEADER_LINE = 8192
MAX_HEADER_COUNT = 64
MAX_RESPONSE_BODY = 64 * 1024 * 1024
MAX_CONNECTIONS = 16
QUEUE_WAIT_SECONDS = 300.0
UPSTREAM_TIMEOUT_SECONDS = 300.0
INFERENCE_ROUTES = frozenset(("/api/chat", "/api/generate", "/api/embed"))
_HEADER_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}\Z")


class _RequestFailure(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _ResponseAborted(Exception):
    """The response already started, so the connection must simply close."""


class UnixHTTPConnection(http.client.HTTPConnection):
    """An HTTP/1.1 connection whose only destination is one AF_UNIX path."""

    def __init__(self, path: Path, *, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = str(path)

    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(self.timeout)
            connection.connect(self._path)
        except BaseException:
            connection.close()
            raise
        self.sock = connection


def _error_response(status: int, message: str) -> bytes:
    reason = http.client.responses.get(status, "Error")
    payload = json.dumps(
        {"error": message},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: application/json\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + payload


def _read_request(connection: socket.socket) -> MediationRequest:
    payload = bytearray()
    delimiter = -1
    while delimiter < 0:
        if len(payload) >= MAX_REQUEST_HEADERS:
            raise _RequestFailure(431, "request headers exceed the mediation limit")
        try:
            block = connection.recv(min(4096, MAX_REQUEST_HEADERS - len(payload)))
        except socket.timeout as error:
            raise _RequestFailure(408, "request header timed out") from error
        if not block:
            raise _RequestFailure(400, "request ended before its headers")
        payload.extend(block)
        delimiter = payload.find(b"\r\n\r\n")
    header_bytes = bytes(payload[:delimiter])
    remainder = bytes(payload[delimiter + 4 :])
    lines = header_bytes.split(b"\r\n")
    if not lines or len(lines[0]) > MAX_HEADER_LINE:
        raise _RequestFailure(400, "request line is missing or too long")
    try:
        method_bytes, target_bytes, version = lines[0].split(b" ")
        method = method_bytes.decode("ascii")
        target = target_bytes.decode("utf-8", errors="strict")
    except (ValueError, UnicodeError) as error:
        raise _RequestFailure(400, "request line is malformed") from error
    if version != b"HTTP/1.1":
        raise _RequestFailure(505, "only HTTP/1.1 is accepted")
    if len(lines) - 1 > MAX_HEADER_COUNT:
        raise _RequestFailure(431, "request has too many headers")

    headers: dict[str, str] = {}
    lowered_names: set[str] = set()
    for line in lines[1:]:
        if not line or len(line) > MAX_HEADER_LINE or line[:1] in (b" ", b"\t"):
            raise _RequestFailure(400, "request header is malformed")
        if b":" not in line:
            raise _RequestFailure(400, "request header has no colon")
        raw_name, raw_value = line.split(b":", 1)
        if not _HEADER_NAME.fullmatch(raw_name):
            raise _RequestFailure(400, "request header name is invalid")
        try:
            name = raw_name.decode("ascii")
            value = raw_value.strip(b" ").decode("ascii")
        except UnicodeError as error:
            raise _RequestFailure(400, "request header is not ASCII") from error
        if any(byte < 0x20 or byte > 0x7E for byte in raw_value):
            raise _RequestFailure(400, "request header value contains a control byte")
        lowered = name.casefold()
        if lowered in lowered_names:
            raise _RequestFailure(400, f"duplicate request header: {lowered}")
        lowered_names.add(lowered)
        headers[name] = value

    normalized = {name.casefold(): value for name, value in headers.items()}
    if "transfer-encoding" in normalized:
        raise _RequestFailure(400, "Transfer-Encoding requests are not accepted")
    if "expect" in normalized:
        raise _RequestFailure(417, "Expect requests are not accepted")
    raw_length = normalized.get("content-length", "0")
    if (
        not raw_length.isascii()
        or not raw_length.isdecimal()
        or len(raw_length) > 20
    ):
        raise _RequestFailure(400, "Content-Length is invalid")
    content_length = int(raw_length)
    if content_length > MAX_REQUEST_BODY:
        raise _RequestFailure(413, "request body exceeds the mediation limit")
    if len(remainder) > content_length:
        raise _RequestFailure(400, "pipelined or trailing request data is not accepted")
    body = bytearray(remainder)
    while len(body) < content_length:
        try:
            block = connection.recv(min(64 * 1024, content_length - len(body)))
        except socket.timeout as error:
            raise _RequestFailure(408, "request body timed out") from error
        if not block:
            raise _RequestFailure(400, "request body ended before Content-Length")
        body.extend(block)
    try:
        return MediationRequest(method, target, headers, bytes(body)).prepared()
    except PolicyDenied as error:
        raise _RequestFailure(403, str(error)) from error


def _peer_disconnected(connection: socket.socket) -> bool:
    try:
        readable, _writable, exceptional = select.select([connection], [], [connection], 0)
        if exceptional:
            return True
        if not readable:
            return False
        pending = connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        # After the exact admitted body, any further byte is pipelined input and
        # is just as terminal for this one-request connection as EOF.
        return pending == b"" or bool(pending)
    except (OSError, ValueError):
        return True


def _response_headers(values: list[tuple[str, str]]) -> dict[str, str]:
    candidates: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in values:
        lowered = name.casefold()
        if lowered in seen and lowered == "content-type":
            raise UpstreamProtocolError("upstream repeated Content-Type")
        seen.add(lowered)
        if lowered == "content-type":
            candidates[name] = value
    try:
        return forward_response_headers(candidates)
    except PolicyDenied as error:
        raise UpstreamProtocolError(str(error)) from error


class _BoundedThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = False

    def __init__(self, *args, max_connections: int, **kwargs) -> None:
        self.max_connections = max_connections
        self._slots = threading.BoundedSemaphore(max_connections)
        self._connections_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._upstreams: set[UnixHTTPConnection] = set()
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(_error_response(503, "mediation connection limit reached"))
            except OSError:
                pass
            self.shutdown_request(request)
            return
        with self._connections_lock:
            self._connections.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._connections_lock:
                self._connections.discard(request)
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._connections_lock:
                self._connections.discard(request)
            self._slots.release()

    def track_upstream(self, upstream: UnixHTTPConnection) -> None:
        with self._connections_lock:
            self._upstreams.add(upstream)

    def untrack_upstream(self, upstream: UnixHTTPConnection) -> None:
        with self._connections_lock:
            self._upstreams.discard(upstream)

    def abort_connections(self) -> None:
        with self._connections_lock:
            clients = tuple(self._connections)
            upstreams = tuple(self._upstreams)
        for connection in clients:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for upstream in upstreams:
            upstream.close()


class _MediationHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: _ProxyServer = self.server  # type: ignore[assignment]
        connection: socket.socket = self.request
        connection.settimeout(server.client_timeout)
        try:
            request = _read_request(connection)
            server.proxy_request(connection, request)
        except _RequestFailure as error:
            server.note_failure("request")
            try:
                connection.sendall(_error_response(error.status, error.message))
            except OSError:
                pass
        except (OSError, http.client.HTTPException, UpstreamProtocolError) as error:
            server.note_failure("upstream")
            try:
                connection.sendall(_error_response(502, "local Ollama runtime unavailable"))
            except OSError:
                pass
        except (
            AcceleratorQueueFull,
            AcceleratorRequestTimeout,
            AcceleratorUnavailable,
        ) as error:
            server.note_failure("queue")
            try:
                connection.sendall(_error_response(503, str(error)))
            except OSError:
                pass
        except AcceleratorRequestCancelled:
            server.note_failure("cancelled")
        except _ResponseAborted:
            pass
        except Exception:
            server.note_failure("internal")
            try:
                connection.sendall(_error_response(500, "internal mediation failure"))
            except OSError:
                pass


class _ProxyServer(_BoundedThreadingUnixServer):
    def __init__(
        self,
        path: Path,
        paths: RuntimePaths,
        arbiter: AcceleratorArbiter,
        *,
        owner_uid: int,
        max_connections: int,
        max_response_bytes: int,
        queue_timeout: float,
        upstream_timeout: float,
        client_timeout: float,
    ) -> None:
        self.paths = paths
        self.arbiter = arbiter
        self.owner_uid = owner_uid
        self.max_response_bytes = max_response_bytes
        self.queue_timeout = queue_timeout
        self.upstream_timeout = upstream_timeout
        self.client_timeout = client_timeout
        self._counter_lock = threading.Lock()
        self._counters = {
            "completed": 0,
            "cancelled": 0,
            "queue_failures": 0,
            "request_failures": 0,
            "response_limit_failures": 0,
            "upstream_failures": 0,
            "internal_failures": 0,
        }
        super().__init__(
            str(path),
            _MediationHandler,
            bind_and_activate=False,
            max_connections=max_connections,
        )

    def verify_request(self, request: socket.socket, _client_address) -> bool:
        if not hasattr(socket, "SO_PEERCRED"):
            return False
        try:
            size = struct.calcsize("3i")
            credentials = request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, size)
            _pid, peer_uid, _gid = struct.unpack("3i", credentials)
            verify_peer(peer_uid, owner_uid=self.owner_uid)
            return True
        except (OSError, struct.error, PolicyDenied):
            return False

    def note_failure(self, kind: str) -> None:
        key = {
            "cancelled": "cancelled",
            "queue": "queue_failures",
            "request": "request_failures",
            "response-limit": "response_limit_failures",
            "upstream": "upstream_failures",
            "internal": "internal_failures",
        }[kind]
        with self._counter_lock:
            self._counters[key] += 1

    def counters(self) -> dict[str, int]:
        with self._counter_lock:
            return dict(self._counters)

    def handle_error(self, _request, _client_address) -> None:
        # Never print a traceback from a request-handling thread: exceptions
        # may carry local paths, and request content is never diagnostic data.
        self.note_failure("internal")

    def _begin_upstream(
        self,
        request: MediationRequest,
    ) -> tuple[UnixHTTPConnection, http.client.HTTPResponse]:
        upstream = UnixHTTPConnection(
            self.paths.upstream_socket,
            timeout=self.upstream_timeout,
        )
        self.track_upstream(upstream)
        try:
            upstream.putrequest(
                request.method,
                request.path,
                skip_host=True,
                skip_accept_encoding=True,
            )
            upstream.putheader("Host", "localhost")
            upstream.putheader("Connection", "close")
            for name, value in request.headers.items():
                upstream.putheader(name, value)
            upstream.putheader("Content-Length", str(len(request.body)))
            upstream.endheaders(request.body)
            response = upstream.getresponse()
            if not 100 <= response.status <= 599:
                raise UpstreamProtocolError("upstream returned an invalid HTTP status")
            return upstream, response
        except BaseException:
            self.untrack_upstream(upstream)
            upstream.close()
            raise

    def proxy_request(self, connection: socket.socket, request: MediationRequest) -> None:
        lease = None
        upstream: UnixHTTPConnection | None = None
        response_started = False
        try:
            if request.path in INFERENCE_ROUTES:
                lease = self.arbiter.acquire(
                    LLM_TURN,
                    timeout=self.queue_timeout,
                    cancelled=lambda: _peer_disconnected(connection),
                )
            upstream, response = self._begin_upstream(request)
            headers = _response_headers(response.getheaders())
            reason = http.client.responses.get(response.status, "Response")
            response_head = [f"HTTP/1.1 {response.status} {reason}\r\n"]
            for name, value in headers.items():
                response_head.append(f"{name}: {value}\r\n")
            response_head.extend(
                (
                    "Transfer-Encoding: chunked\r\n",
                    "X-Content-Type-Options: nosniff\r\n",
                    "Connection: close\r\n\r\n",
                )
            )
            connection.sendall("".join(response_head).encode("ascii"))
            response_started = True
            total = 0
            while True:
                block = response.read1(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > self.max_response_bytes:
                    self.note_failure("response-limit")
                    raise _ResponseAborted()
                connection.sendall(f"{len(block):x}\r\n".encode("ascii"))
                connection.sendall(block)
                connection.sendall(b"\r\n")
            connection.sendall(b"0\r\n\r\n")
            with self._counter_lock:
                self._counters["completed"] += 1
        except (OSError, http.client.HTTPException, UpstreamProtocolError) as error:
            if response_started:
                # A second status line after a streamed response head would be
                # response smuggling.  Closing also cancels upstream inference.
                if isinstance(error, OSError):
                    self.note_failure("cancelled")
                else:
                    self.note_failure("upstream")
                raise _ResponseAborted() from error
            raise
        finally:
            if upstream is not None:
                self.untrack_upstream(upstream)
                upstream.close()
            if lease is not None:
                lease.release()


class MediationProxy:
    """Own the private socket and mediate only exact local Ollama routes."""

    def __init__(
        self,
        paths: RuntimePaths,
        arbiter: AcceleratorArbiter,
        *,
        owner_uid: int | None = None,
        max_connections: int = MAX_CONNECTIONS,
        max_response_bytes: int = MAX_RESPONSE_BODY,
        queue_timeout: float = QUEUE_WAIT_SECONDS,
        upstream_timeout: float = UPSTREAM_TIMEOUT_SECONDS,
        client_timeout: float = 30.0,
    ) -> None:
        uid = os.getuid() if owner_uid is None else owner_uid
        integer_values = ((max_connections, "max_connections"), (max_response_bytes, "max_response_bytes"))
        for value, label in integer_values:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RuntimeContractError(f"{label} must be a positive integer")
        for value, label in (
            (queue_timeout, "queue_timeout"),
            (upstream_timeout, "upstream_timeout"),
            (client_timeout, "client_timeout"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise RuntimeContractError(f"{label} must be positive")
        private, detail = private_directory_status(paths.runtime_dir, uid=uid)
        if not private:
            raise RuntimeContractError(
                f"mediation requires a private runtime directory: {detail}"
            )
        if len(os.fsencode(paths.mediation_socket)) > 107:
            raise RuntimeContractError("mediation Unix socket path exceeds the Linux limit")
        if os.path.lexists(paths.mediation_socket):
            raise RuntimeContractError(
                f"refusing an occupied mediation socket path: {paths.mediation_socket}"
            )
        self.paths = paths
        self.arbiter = arbiter
        self.owner_uid = uid
        self._server = _ProxyServer(
            paths.mediation_socket,
            paths,
            arbiter,
            owner_uid=uid,
            max_connections=max_connections,
            max_response_bytes=max_response_bytes,
            queue_timeout=float(queue_timeout),
            upstream_timeout=float(upstream_timeout),
            client_timeout=float(client_timeout),
        )
        self._thread: threading.Thread | None = None
        self._closed = False
        self._socket_identity: tuple[int, int] | None = None
        try:
            self._server.server_bind()
            metadata = paths.mediation_socket.lstat()
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != uid:
                raise RuntimeContractError("mediation listener path is not an owned socket")
            self._socket_identity = (metadata.st_dev, metadata.st_ino)
            paths.mediation_socket.chmod(0o600)
            metadata = paths.mediation_socket.lstat()
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or metadata.st_uid != uid
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise RuntimeContractError("mediation socket is not owner-private")
            self._server.server_activate()
        except BaseException:
            self._server.server_close()
            self._remove_own_socket()
            raise

    def _remove_own_socket(self) -> None:
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            metadata = self.paths.mediation_socket.lstat()
        except OSError:
            return
        if (
            stat.S_ISSOCK(metadata.st_mode)
            and metadata.st_uid == self.owner_uid
            and (metadata.st_dev, metadata.st_ino) == identity
        ):
            try:
                self.paths.mediation_socket.unlink()
            except OSError:
                pass

    def start(self) -> None:
        if self._closed:
            raise RuntimeContractError("mediation proxy is closed")
        if self._thread is not None:
            raise RuntimeContractError("mediation proxy is already running")
        thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="kilix-ollama-mediation",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        thread, self._thread = self._thread, None
        if thread is not None:
            self._server.shutdown()
            self._server.abort_connections()
            thread.join(timeout=5)
        self._server.server_close()
        self._remove_own_socket()

    def __enter__(self) -> "MediationProxy":
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.live-mediation/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "socket": str(self.paths.mediation_socket),
            "same_uid_required": True,
            "upstream_transport": "unix",
            "one_request_per_connection": True,
            "limits": {
                "connections": self._server.max_connections,
                "request_headers": MAX_REQUEST_HEADERS,
                "request_body": MAX_REQUEST_BODY,
                "response_body": self._server.max_response_bytes,
            },
            "inference_routes": sorted(INFERENCE_ROUTES),
            "counters": self._server.counters(),
            "arbiter": self.arbiter.record(),
            "request_content_logged": False,
            "cloud_fallback": False,
            "weight_access": False,
        }


__all__ = [
    "INFERENCE_ROUTES",
    "MAX_CONNECTIONS",
    "MAX_REQUEST_HEADERS",
    "MAX_RESPONSE_BODY",
    "MediationProxy",
    "UnixHTTPConnection",
]
