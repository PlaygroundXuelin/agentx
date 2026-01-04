"""Tool registry and discovery helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from exec_agent.tools.base import Tool, ToolSpec
from exec_agent.tools.policies import ToolAuthContext


@dataclass(slots=True)
class ToolRegistry:
    """Registry for available tools with optional scope filtering."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_specs(self, auth: ToolAuthContext | None = None) -> list[ToolSpec]:
        specs = [tool.spec for tool in self._tools.values()]
        if auth is None or not auth.scopes:
            return [spec for spec in specs if not spec.scopes]
        return [spec for spec in specs if not spec.scopes or set(spec.scopes) & auth.scopes]
