"""Base interfaces for tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

JsonValue = Any


@dataclass(slots=True)
class ToolSpec:
    """Machine-readable description of a tool."""

    name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    scopes: list[str] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


@dataclass(slots=True)
class ToolResult:
    """Normalized tool result returned to the agent."""

    name: str
    ok: bool
    content: str
    data: JsonValue | None = None
    error: str | None = None

    @classmethod
    def from_data(
        cls,
        name: str,
        data: JsonValue,
        *,
        content: str | None = None,
    ) -> ToolResult:
        text = content or json.dumps(data, ensure_ascii=True)
        return cls(name=name, ok=True, content=text, data=data)

    @classmethod
    def failure(cls, name: str, error: str) -> ToolResult:
        payload = {"error": error}
        return cls(name=name, ok=False, content=json.dumps(payload, ensure_ascii=True), error=error)


class Tool(Protocol):
    """Tool interface for the executor."""

    spec: ToolSpec

    async def run(self, arguments: Mapping[str, JsonValue]) -> ToolResult: ...
