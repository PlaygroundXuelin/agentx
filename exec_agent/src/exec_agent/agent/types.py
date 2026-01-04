"""Shared agent message and tool-call types."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

JsonValue = Any


@dataclass(slots=True)
class ToolCall:
    """Normalized tool call emitted by the model."""

    id: str
    name: str
    arguments: Mapping[str, JsonValue]
    arguments_raw: str | None = None

    def to_openai(self) -> dict[str, Any]:
        arguments = self.arguments_raw or json.dumps(self.arguments, ensure_ascii=True)
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments},
        }


@dataclass(slots=True)
class ChatMessage:
    """Normalized chat message exchanged with the model."""

    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return payload


@dataclass(slots=True)
class ModelResponse:
    """Model response normalized for the agent loop."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any | None = None


@dataclass(slots=True)
class AgentResult:
    """Final response returned by the agent runner."""

    output: str
    steps: int
    tool_calls_executed: int = 0
