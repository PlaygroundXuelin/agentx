"""Tool execution helpers with policy checks and tracing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import structlog

from exec_agent.agent.types import ToolCall
from exec_agent.infra.tracing import tool_span
from exec_agent.tools.base import ToolResult
from exec_agent.tools.policies import ToolAuthContext, ToolPolicy
from exec_agent.tools.registry import ToolRegistry

LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class ToolExecutionSummary:
    results: list[ToolResult]
    call_count: int


class ToolExecutor:
    """Execute tool calls and normalize failures."""

    def __init__(self, registry: ToolRegistry, policy: ToolPolicy) -> None:
        self._registry = registry
        self._policy = policy

    async def execute(
        self,
        calls: Iterable[ToolCall],
        *,
        auth: ToolAuthContext | None,
        call_count: int = 0,
    ) -> ToolExecutionSummary:
        results: list[ToolResult] = []
        for call in calls:
            tool = self._registry.get(call.name)
            if tool is None:
                results.append(ToolResult.failure(call.name, "tool_not_found"))
                call_count += 1
                continue

            decision = self._policy.evaluate(tool.spec, auth, call_count=call_count)
            if not decision.allow:
                results.append(ToolResult.failure(call.name, decision.reason or "tool_not_allowed"))
                call_count += 1
                continue

            call_count += 1
            with tool_span("tool.execute", tool_name=call.name):
                try:
                    LOGGER.debug(
                        "tool.execute_start",
                        tool=call.name,
                        call_id=call.id,
                        arguments=call.arguments,
                        call_count=call_count,
                    )
                    result = await tool.run(call.arguments)
                    LOGGER.debug(
                        "tool.execute_end",
                        tool=call.name,
                        call_id=call.id,
                        ok=result.ok,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.exception("tool.execute_failed", tool=call.name, error=str(exc))
                    result = ToolResult.failure(call.name, "tool_execution_failed")
            results.append(result)

        return ToolExecutionSummary(results=results, call_count=call_count)
