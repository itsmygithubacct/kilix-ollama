"""Private, atomic conversation storage for the local chat surface."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Iterable, Iterator, Mapping
import uuid

from .attachments import validate_attachment_records
from .errors import ConversationError
from .paths import RuntimePaths
from .sanitizer import sanitize_text


CONVERSATION_SCHEMA = "kilix.ollama.conversation/v0-unfrozen"
MAX_CONVERSATION_BYTES = 8 * 1024 * 1024
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_MESSAGES = 4096
MAX_TITLE_BYTES = 256
_IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")
_ROLES = frozenset(("assistant", "system", "tool", "user"))
_MESSAGE_KEYS = frozenset(
    ("attachments", "content", "role", "thinking", "tool_calls", "tool_name")
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConversationError(f"conversation JSON repeats member {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ConversationError(f"conversation JSON contains non-finite number {value}")


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise ConversationError("conversation cannot be encoded as strict JSON") from error


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ConversationError(f"conversation {label} is not text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise ConversationError(f"conversation {label} contains invalid Unicode") from error
    if size > maximum:
        raise ConversationError(f"conversation {label} exceeds its byte limit")
    return value


def _timestamp(value: object, label: str) -> str:
    text = _bounded_text(value, label, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ConversationError(f"conversation {label} is not an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise ConversationError(f"conversation {label} has no timezone")
    return text


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ConversationError("conversation id is invalid")
    return value


def _ensure_private_directory(path: Path, *, uid: int) -> None:
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise ConversationError("no safe anchor exists for conversation storage")
        cursor = parent
    try:
        anchor = cursor.lstat()
    except OSError as error:
        raise ConversationError("conversation storage anchor cannot be inspected") from error
    if stat.S_ISLNK(anchor.st_mode) or not stat.S_ISDIR(anchor.st_mode):
        raise ConversationError("conversation storage anchor is not a real directory")
    if anchor.st_uid not in (0, uid) or anchor.st_mode & 0o022:
        raise ConversationError("conversation storage anchor is not owner-controlled")
    for candidate in reversed(missing):
        try:
            candidate.mkdir(mode=0o700)
        except OSError as error:
            raise ConversationError("conversation directory cannot be created") from error
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise ConversationError("new conversation directory cannot be inspected") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ConversationError("new conversation directory is unsafe")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ConversationError("conversation directory cannot be inspected") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ConversationError("conversation directory must be owner-only mode 0700")


def _validate_tool_calls(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) > 16:
        raise ConversationError("conversation tool_calls is not a bounded array")
    encoded = _canonical(value)
    if len(encoded) > 256 * 1024:
        raise ConversationError("conversation tool_calls exceeds its byte limit")
    for index, call in enumerate(value):
        if not isinstance(call, dict):
            raise ConversationError(f"conversation tool call {index} is not an object")
    return value


def validate_message(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConversationError("conversation message is not an object")
    if set(value) - _MESSAGE_KEYS:
        raise ConversationError("conversation message has unknown members")
    role = value.get("role")
    if role not in _ROLES:
        raise ConversationError("conversation message role is invalid")
    content = _bounded_text(value.get("content", ""), "message content", maximum=MAX_MESSAGE_BYTES)
    result: dict[str, object] = {"role": role, "content": content}
    thinking = value.get("thinking")
    if thinking is not None:
        result["thinking"] = _bounded_text(
            thinking, "message thinking", maximum=MAX_MESSAGE_BYTES
        )
    tool_calls = value.get("tool_calls")
    if tool_calls is not None:
        if role != "assistant":
            raise ConversationError("only an assistant message may contain tool calls")
        result["tool_calls"] = _validate_tool_calls(tool_calls)
    tool_name = value.get("tool_name")
    if tool_name is not None:
        if role != "tool":
            raise ConversationError("only a tool message may name a tool")
        name = _bounded_text(tool_name, "tool name", maximum=256)
        if not name or any(ord(character) < 0x21 or ord(character) > 0x7E for character in name):
            raise ConversationError("conversation tool name is invalid")
        result["tool_name"] = name
    if role == "tool" and "tool_name" not in result:
        raise ConversationError("tool message has no tool name")
    if "attachments" in value:
        attachments = value["attachments"]
        if role != "user":
            raise ConversationError("only a user message may contain attachments")
        result["attachments"] = validate_attachment_records(attachments)
    return result


@dataclass(frozen=True)
class Conversation:
    identifier: str
    title: str
    model: str
    created_at: str
    updated_at: str
    messages: tuple[dict[str, object], ...]

    @classmethod
    def from_record(cls, value: object) -> "Conversation":
        if not isinstance(value, dict):
            raise ConversationError("conversation root is not an object")
        expected = {
            "schema",
            "schema_status",
            "id",
            "title",
            "model",
            "created_at",
            "updated_at",
            "messages",
        }
        if set(value) != expected:
            raise ConversationError("conversation root members do not match the schema")
        if value.get("schema") != CONVERSATION_SCHEMA:
            raise ConversationError("conversation schema is unsupported")
        if value.get("schema_status") != "development-unfrozen":
            raise ConversationError("conversation schema status is invalid")
        identifier = _identifier(value.get("id"))
        title = _bounded_text(value.get("title"), "title", maximum=MAX_TITLE_BYTES)
        model = _bounded_text(value.get("model"), "model", maximum=1024)
        if not model:
            raise ConversationError("conversation model is empty")
        created_at = _timestamp(value.get("created_at"), "created_at")
        updated_at = _timestamp(value.get("updated_at"), "updated_at")
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list) or len(raw_messages) > MAX_MESSAGES:
            raise ConversationError("conversation messages is not a bounded array")
        messages = tuple(validate_message(message) for message in raw_messages)
        conversation = cls(identifier, title, model, created_at, updated_at, messages)
        if len(_canonical(conversation.storage_record())) > MAX_CONVERSATION_BYTES:
            raise ConversationError("conversation exceeds its total byte limit")
        return conversation

    def storage_record(self) -> dict[str, object]:
        return {
            "schema": CONVERSATION_SCHEMA,
            "schema_status": "development-unfrozen",
            "id": self.identifier,
            "title": self.title,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": list(self.messages),
        }

    def record(self, *, include_thinking: bool = False) -> dict[str, object]:
        record = self.storage_record()
        if not include_thinking:
            record["messages"] = [
                {key: value for key, value in message.items() if key != "thinking"}
                for message in self.messages
            ]
        return record

    def api_messages(self) -> list[dict[str, object]]:
        return [dict(message) for message in self.messages]


class ConversationStore:
    def __init__(
        self,
        paths: RuntimePaths,
        *,
        uid: int | None = None,
        clock: Callable[[], datetime] | None = None,
        identifier_factory: Callable[[], str] | None = None,
    ) -> None:
        self.root = paths.data_dir / "conversations"
        self.uid = os.getuid() if uid is None else uid
        if not isinstance(self.uid, int) or isinstance(self.uid, bool) or self.uid < 0:
            raise ConversationError("conversation owner uid is invalid")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._identifier_factory = identifier_factory or (lambda: uuid.uuid4().hex)

    def _now(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ConversationError("conversation clock did not return an aware datetime")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _path(self, identifier: str) -> Path:
        return self.root / f"{_identifier(identifier)}.json"

    def _prepare(self) -> None:
        _ensure_private_directory(self.root, uid=self.uid)

    def _allocate_identifier_unlocked(self) -> str:
        for _attempt in range(32):
            identifier = _identifier(self._identifier_factory())
            if not os.path.lexists(self._path(identifier)):
                return identifier
        raise ConversationError("conversation id allocation exhausted")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._prepare()
        lock_path = self.root / ".lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise ConversationError("conversation lock cannot be opened safely") from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != self.uid
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise ConversationError("conversation lock is not owner-private")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _read_unlocked(self, identifier: str) -> Conversation:
        path = self._path(identifier)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as error:
            raise ConversationError("conversation does not exist") from error
        except OSError as error:
            raise ConversationError("conversation cannot be opened safely") from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != self.uid
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size > MAX_CONVERSATION_BYTES
            ):
                raise ConversationError("conversation file is not an owner-private bounded file")
            payload = bytearray()
            while len(payload) <= MAX_CONVERSATION_BYTES:
                block = os.read(descriptor, min(64 * 1024, MAX_CONVERSATION_BYTES + 1 - len(payload)))
                if not block:
                    break
                payload.extend(block)
            if len(payload) > MAX_CONVERSATION_BYTES:
                raise ConversationError("conversation file exceeds its byte limit")
        finally:
            os.close(descriptor)
        try:
            value = json.loads(
                bytes(payload).decode("utf-8", errors="strict"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except ConversationError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ConversationError("conversation file is not strict JSON") from error
        conversation = Conversation.from_record(value)
        if conversation.identifier != identifier:
            raise ConversationError("conversation filename and embedded id disagree")
        return conversation

    def _write_unlocked(self, conversation: Conversation, *, replace: bool) -> None:
        payload = _canonical(conversation.storage_record())
        if len(payload) > MAX_CONVERSATION_BYTES:
            raise ConversationError("conversation exceeds its total byte limit")
        target = self._path(conversation.identifier)
        if not replace and os.path.lexists(target):
            raise ConversationError("conversation id already exists")
        temporary = self.root / f".{conversation.identifier}.{uuid.uuid4().hex}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = -1
        try:
            descriptor = os.open(temporary, flags, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ConversationError("conversation write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            if replace:
                try:
                    metadata = target.lstat()
                except FileNotFoundError:
                    raise ConversationError("conversation disappeared during update")
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise ConversationError("conversation target was substituted")
                os.replace(temporary, target)
            else:
                os.link(temporary, target, follow_symlinks=False)
                temporary.unlink()
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except ConversationError:
            raise
        except OSError as error:
            raise ConversationError("conversation cannot be committed atomically") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def create(self, *, model: str, title: str, messages: Iterable[Mapping[str, object]] = ()) -> Conversation:
        checked_model = _bounded_text(model, "model", maximum=1024)
        checked_title = _bounded_text(title.strip(), "title", maximum=MAX_TITLE_BYTES)
        if not checked_model or not checked_title:
            raise ConversationError("conversation model and title must not be empty")
        checked_messages = tuple(validate_message(message) for message in messages)
        if len(checked_messages) > MAX_MESSAGES:
            raise ConversationError("conversation message count exceeds its limit")
        with self._locked():
            identifier = self._allocate_identifier_unlocked()
            now = self._now()
            conversation = Conversation(
                identifier,
                checked_title,
                checked_model,
                now,
                now,
                checked_messages,
            )
            Conversation.from_record(conversation.storage_record())
            self._write_unlocked(conversation, replace=False)
            return conversation

    def load(self, identifier: str) -> Conversation:
        with self._locked():
            return self._read_unlocked(identifier)

    def append(self, identifier: str, messages: Iterable[Mapping[str, object]]) -> Conversation:
        additions = tuple(validate_message(message) for message in messages)
        if not additions:
            raise ConversationError("conversation append is empty")
        with self._locked():
            current = self._read_unlocked(identifier)
            combined = current.messages + additions
            if len(combined) > MAX_MESSAGES:
                raise ConversationError("conversation message count exceeds its limit")
            updated = Conversation(
                current.identifier,
                current.title,
                current.model,
                current.created_at,
                self._now(),
                combined,
            )
            Conversation.from_record(updated.storage_record())
            self._write_unlocked(updated, replace=True)
            return updated

    def replace_messages(
        self,
        expected: Conversation,
        messages: Iterable[Mapping[str, object]],
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> Conversation:
        if not isinstance(expected, Conversation):
            raise ConversationError("expected conversation state is invalid")
        if before_commit is not None and not callable(before_commit):
            raise ConversationError("conversation pre-commit action is invalid")
        Conversation.from_record(expected.storage_record())
        replacements = tuple(validate_message(message) for message in messages)
        if len(replacements) > MAX_MESSAGES:
            raise ConversationError("conversation message count exceeds its limit")
        with self._locked():
            current = self._read_unlocked(expected.identifier)
            if current != expected:
                raise ConversationError("conversation changed during the operation")
            updated = Conversation(
                current.identifier,
                current.title,
                current.model,
                current.created_at,
                self._now(),
                replacements,
            )
            Conversation.from_record(updated.storage_record())
            if before_commit is not None:
                before_commit()
            self._write_unlocked(updated, replace=True)
            return updated

    def replace_model(self, expected: Conversation, model: str) -> Conversation:
        if not isinstance(expected, Conversation):
            raise ConversationError("expected conversation state is invalid")
        Conversation.from_record(expected.storage_record())
        checked_model = _bounded_text(model, "model", maximum=1024)
        if not checked_model:
            raise ConversationError("conversation model is empty")
        with self._locked():
            current = self._read_unlocked(expected.identifier)
            if current != expected:
                raise ConversationError("conversation changed during the operation")
            if current.model == checked_model:
                return current
            updated = Conversation(
                current.identifier,
                current.title,
                checked_model,
                current.created_at,
                self._now(),
                current.messages,
            )
            Conversation.from_record(updated.storage_record())
            self._write_unlocked(updated, replace=True)
            return updated

    def fork(
        self,
        identifier: str,
        *,
        message_count: int | None = None,
        title: str | None = None,
    ) -> Conversation:
        if message_count is not None and (
            not isinstance(message_count, int) or isinstance(message_count, bool)
        ):
            raise ConversationError("fork message count is invalid")
        with self._locked():
            source = self._read_unlocked(identifier)
            count = len(source.messages) if message_count is None else message_count
            if not 0 <= count <= len(source.messages):
                raise ConversationError("fork message count is outside the conversation")
            messages = source.messages[:count]
            if messages and (
                messages[-1]["role"] == "tool" or messages[-1].get("tool_calls")
            ):
                raise ConversationError("fork point splits a tool exchange")
            checked_title = source.title
            if title is not None:
                checked_title = _bounded_text(
                    title.strip(),
                    "title",
                    maximum=MAX_TITLE_BYTES,
                )
                if not checked_title:
                    raise ConversationError("conversation title must not be empty")
            new_identifier = self._allocate_identifier_unlocked()
            now = self._now()
            forked = Conversation(
                new_identifier,
                checked_title,
                source.model,
                now,
                now,
                messages,
            )
            Conversation.from_record(forked.storage_record())
            self._write_unlocked(forked, replace=False)
            return forked

    def list(self) -> tuple[dict[str, object], ...]:
        if not os.path.lexists(self.root):
            return ()
        with self._locked():
            summaries: list[dict[str, object]] = []
            try:
                candidates = sorted(self.root.iterdir(), key=lambda path: path.name)
            except OSError as error:
                raise ConversationError("conversation directory cannot be listed") from error
            for path in candidates:
                match = re.fullmatch(r"([0-9a-f]{32})\.json", path.name)
                if match is None:
                    if path.name == ".lock":
                        continue
                    raise ConversationError("conversation directory contains an unexpected entry")
                conversation = self._read_unlocked(match.group(1))
                summaries.append(
                    {
                        "id": conversation.identifier,
                        "title": conversation.title,
                        "model": conversation.model,
                        "updated_at": conversation.updated_at,
                        "message_count": len(conversation.messages),
                    }
                )
            summaries.sort(key=lambda value: (str(value["updated_at"]), str(value["id"])), reverse=True)
            return tuple(summaries)

    def delete(self, identifier: str) -> None:
        with self._locked():
            self._read_unlocked(identifier)
            try:
                self._path(identifier).unlink()
                directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError as error:
                raise ConversationError("conversation cannot be deleted safely") from error

    def export_markdown(self, identifier: str, *, include_thinking: bool = False) -> str:
        conversation = self.load(identifier)
        lines = [f"# {sanitize_text(conversation.title)}", "", f"Model: `{sanitize_text(conversation.model)}`", ""]
        for message in conversation.messages:
            role = str(message["role"]).title()
            lines.extend((f"## {role}", "", sanitize_text(str(message["content"])), ""))
            if include_thinking and message.get("thinking"):
                lines.extend(("### Thinking (explicitly included)", "", sanitize_text(str(message["thinking"])), ""))
            if message.get("tool_calls"):
                lines.extend(("Tool-call metadata omitted from Markdown export.", ""))
            if message.get("attachments"):
                count = len(message["attachments"])  # type: ignore[arg-type]
                lines.extend((f"Attachments omitted from Markdown export ({count} image(s)).", ""))
        return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CONVERSATION_SCHEMA",
    "Conversation",
    "ConversationStore",
    "MAX_CONVERSATION_BYTES",
    "MAX_MESSAGE_BYTES",
    "MAX_MESSAGES",
    "validate_message",
]
