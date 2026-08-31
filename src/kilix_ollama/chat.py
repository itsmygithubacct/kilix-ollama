"""Persistent local chat orchestration over the mediated client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .client import OllamaClient, validate_model_name
from .conversations import Conversation, ConversationStore, validate_message
from .errors import ConversationError, ToolError
from .sanitizer import sanitize_text
from .streaming import StreamDelta, StreamResult
from .tools import (
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOOL_ROUNDS,
    ToolPreview,
    ToolRegistry,
)


def conversation_title(prompt: str) -> str:
    line = sanitize_text(prompt).strip().splitlines()[0] if prompt.strip() else "New chat"
    if len(line) > 72:
        line = line[:71].rstrip() + "…"
    return line or "New chat"


@dataclass(frozen=True)
class ChatTurn:
    conversation: Conversation
    result: StreamResult
    tool_rounds: int
    tool_calls: int
    tool_denials: int
    operation: str = "turn"

    def __post_init__(self) -> None:
        if self.operation not in ("regenerate", "turn"):
            raise ToolError("chat operation is invalid")

    def record(self, *, include_thinking: bool = False) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.chat-turn/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "conversation_id": self.conversation.identifier,
            "message_count": len(self.conversation.messages),
            "result": self.result.record(include_thinking=include_thinking),
            "tool_rounds": self.tool_rounds,
            "tool_calls": self.tool_calls,
            "tool_denials": self.tool_denials,
            "persistent_tool_approval": False,
            "operation": self.operation,
        }


class ChatEngine:
    def __init__(
        self,
        client: OllamaClient,
        store: ConversationStore,
    ) -> None:
        self.client = client
        self.store = store

    def create(self, *, model: str, title: str | None = None) -> Conversation:
        checked = validate_model_name(model)
        return self.store.create(model=checked, title=title or "New chat")

    def _complete(
        self,
        conversation: Conversation,
        messages: list[dict[str, object]],
        *,
        operation: str,
        registry: ToolRegistry | None,
        approve: Callable[[ToolPreview], bool] | None,
        on_delta: Callable[[StreamDelta], None] | None,
        temperature: float | None,
        seed: int | None,
        context: int | None,
    ) -> ChatTurn:
        declarations = registry.declarations() if registry is not None else ()
        rounds = 0
        calls = 0
        denials = 0
        final: StreamResult | None = None
        while True:
            result = self.client.chat(
                conversation.model,
                messages,
                tools=declarations,
                temperature=temperature,
                seed=seed,
                context=context,
                on_delta=on_delta,
                check_installed=False,
            )
            final = result
            assistant: dict[str, object] = {
                "role": "assistant",
                "content": result.content,
            }
            if result.thinking:
                assistant["thinking"] = result.thinking
            if result.tool_calls:
                assistant["tool_calls"] = list(result.tool_calls)
            messages.append(assistant)
            if not result.tool_calls:
                break
            if registry is None:
                raise ToolError("runtime emitted tool calls while tools are disabled")
            rounds += 1
            calls += len(result.tool_calls)
            if rounds > MAX_TOOL_ROUNDS or calls > MAX_TOOL_CALLS_PER_TURN:
                raise ToolError("tool-call loop exceeded its total budget")
            for call in result.tool_calls:
                outcome = registry.execute(call, approve=approve)
                if not outcome.approved:
                    denials += 1
                messages.append(outcome.message())
        if final is None:
            raise ToolError("chat turn produced no runtime result")
        committed = self.store.replace_messages(conversation, messages)
        return ChatTurn(
            committed,
            final,
            rounds,
            calls,
            denials,
            operation,
        )

    def turn(
        self,
        conversation_id: str,
        prompt: str,
        *,
        registry: ToolRegistry | None = None,
        approve: Callable[[ToolPreview], bool] | None = None,
        on_delta: Callable[[StreamDelta], None] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        context: int | None = None,
    ) -> ChatTurn:
        if not isinstance(prompt, str) or not prompt:
            raise ToolError("chat prompt must not be empty")
        user_message = validate_message({"role": "user", "content": prompt})
        conversation = self.store.load(conversation_id)
        self.client.require_installed(conversation.model)
        messages = conversation.api_messages()
        messages.append(user_message)
        return self._complete(
            conversation,
            messages,
            operation="turn",
            registry=registry,
            approve=approve,
            on_delta=on_delta,
            temperature=temperature,
            seed=seed,
            context=context,
        )

    def regenerate(
        self,
        conversation_id: str,
        *,
        registry: ToolRegistry | None = None,
        approve: Callable[[ToolPreview], bool] | None = None,
        on_delta: Callable[[StreamDelta], None] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        context: int | None = None,
    ) -> ChatTurn:
        conversation = self.store.load(conversation_id)
        self.client.require_installed(conversation.model)
        last_user = next(
            (
                index
                for index in range(len(conversation.messages) - 1, -1, -1)
                if conversation.messages[index]["role"] == "user"
            ),
            -1,
        )
        suffix = conversation.messages[last_user + 1 :] if last_user >= 0 else ()
        if last_user < 0 or not any(message["role"] == "assistant" for message in suffix):
            raise ConversationError("conversation has no completed response to regenerate")
        messages = [dict(message) for message in conversation.messages[: last_user + 1]]
        return self._complete(
            conversation,
            messages,
            operation="regenerate",
            registry=registry,
            approve=approve,
            on_delta=on_delta,
            temperature=temperature,
            seed=seed,
            context=context,
        )

    def switch_model(self, conversation_id: str, model: str) -> Conversation:
        checked = validate_model_name(model)
        conversation = self.store.load(conversation_id)
        self.client.require_installed(checked)
        return self.store.replace_model(conversation, checked)

    def fork(
        self,
        conversation_id: str,
        *,
        message_count: int | None = None,
        title: str | None = None,
    ) -> Conversation:
        return self.store.fork(
            conversation_id,
            message_count=message_count,
            title=title,
        )


__all__ = ["ChatEngine", "ChatTurn", "conversation_title"]
