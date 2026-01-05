from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from exec_agent.agent.runner import AgentRunner
from exec_agent.agent.types import ModelResponse, ToolCall
from exec_agent.tools.base import ToolResult, ToolSpec
from exec_agent.tools.executor import ToolExecutor
from exec_agent.tools.policies import ToolPolicy
from exec_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class RecordingTool:
    spec: ToolSpec
    seen_args: list[Mapping[str, Any]] = field(default_factory=list)

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        self.seen_args.append(arguments)
        return ToolResult.from_data(self.spec.name, {"ok": True})


@dataclass(slots=True)
class FakeClient:
    responses: list[ModelResponse]
    calls: int = 0

    async def complete(
        self,
        messages: Sequence[object],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        assert tools, "Expected tool specs to be provided to the model"
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_agent_run_with_tool_call() -> None:
    tool_spec = ToolSpec(
        name="retrieve",
        description="Test tool",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    tool = RecordingTool(spec=tool_spec)
    registry = ToolRegistry()
    registry.register(tool)

    executor = ToolExecutor(registry, ToolPolicy(allowed_tools={"retrieve"}, max_calls=2))
    client = FakeClient(
        responses=[
            ModelResponse(
                text="",
                tool_calls=[ToolCall(id="call-1", name="retrieve", arguments={"query": "hello"})],
            ),
            ModelResponse(text="done", tool_calls=[]),
        ]
    )

    runner = AgentRunner(client=client, registry=registry, executor=executor, max_steps=4)
    result = asyncio.run(runner.run("hi"))

    assert result.output == "done"
    assert result.steps == 2
    assert result.tool_calls_executed == 1
    assert tool.seen_args == [{"query": "hello"}]
