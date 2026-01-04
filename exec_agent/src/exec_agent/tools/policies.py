"""Policies for tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from exec_agent.tools.base import ToolSpec


@dataclass(slots=True)
class ToolAuthContext:
    """Authorization context for filtering tools."""

    scopes: set[str] = field(default_factory=set)
    user_id: str | None = None


@dataclass(slots=True)
class ToolPolicyDecision:
    """Decision returned from a tool policy check."""

    allow: bool
    reason: str | None = None


@dataclass(slots=True)
class ToolPolicy:
    """Simple allow/deny policy with call limits and scope checks."""

    allowed_tools: set[str] | None = None
    denied_tools: set[str] = field(default_factory=set)
    max_calls: int = 8

    def evaluate(
        self,
        spec: ToolSpec,
        auth: ToolAuthContext | None,
        *,
        call_count: int,
    ) -> ToolPolicyDecision:
        if call_count >= self.max_calls:
            return ToolPolicyDecision(allow=False, reason="tool_call_limit_reached")
        if spec.name in self.denied_tools:
            return ToolPolicyDecision(allow=False, reason="tool_denied")
        if self.allowed_tools is not None and spec.name not in self.allowed_tools:
            return ToolPolicyDecision(allow=False, reason="tool_not_allowed")
        if spec.scopes:
            if auth is None:
                return ToolPolicyDecision(allow=False, reason="missing_auth_context")
            if not (set(spec.scopes) & auth.scopes):
                return ToolPolicyDecision(allow=False, reason="missing_required_scope")
        return ToolPolicyDecision(allow=True)
