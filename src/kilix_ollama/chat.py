"""Persistent local chat orchestration over the mediated client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .client import OllamaClient, validate_model_name
from .conversations import Conversation, ConversationStore
from .errors import ToolError
from .sanitizer import sanitize_text
from .streaming import StreamDelta, StreamResult
from .tools import (
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOOL_ROUNDS,
    ToolPreview,
    ToolRegistry,
    ToolResult,
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
        conversation = self.store.load(conversation_id)
        self.client.require_installed(conversation.model)
        conversation = self.store.append(
            conversation.identifier,
            ({"role": "user", "content": prompt},),
        )
        declarations = registry.declarations() if registry is not None else ()
        rounds = 0
        calls = 0
        denials = 0
        final: StreamResult | None = None
        while True:
            result = self.client.chat(
                conversation.model,
                conversation.api_messages(),
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
            conversation = self.store.append(conversation.identifier, (assistant,))
            if not result.tool_calls:
                break
            if registry is None:
                raise ToolError("runtime emitted tool calls while tools are disabled")
            rounds += 1
            calls += len(result.tool_calls)
            if rounds > MAX_TOOL_ROUNDS or calls > MAX_TOOL_CALLS_PER_TURN:
                raise ToolError("tool-call loop exceeded its total budget")
            tool_messages: list[dict[str, object]] = []
            for call in result.tool_calls:
                outcome = registry.execute(call, approve=approve)
                if not outcome.approved:
                    denials += 1
                tool_messages.append(outcome.message())
            conversation = self.store.append(conversation.identifier, tool_messages)
        if final is None:
            raise ToolError("chat turn produced no runtime result")
        return ChatTurn(conversation, final, rounds, calls, denials)


__all__ = ["ChatEngine", "ChatTurn", "conversation_title"]
