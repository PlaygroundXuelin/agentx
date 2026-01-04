"""Agent runner with tool-call handling."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from litellm import aresponses

from exec_agent.agent.types import AgentResult, ChatMessage, ModelResponse, ToolCall
from exec_agent.tools.base import ToolSpec
from exec_agent.tools.executor import ToolExecutor
from exec_agent.tools.policies import ToolAuthContext
from exec_agent.tools.registry import ToolRegistry


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_tool_calls(tool_calls: Iterable[Any]) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    for idx, raw_call in enumerate(tool_calls):
        call_id = _get_attr(raw_call, "id", f"call_{idx}")
        function = _get_attr(raw_call, "function", None)
        if function is None:
            name = _get_attr(raw_call, "name", "")
            arguments = _get_attr(raw_call, "arguments", "")
        else:
            name = _get_attr(function, "name", "")
            arguments = _get_attr(function, "arguments", "")
        arguments_raw = arguments if isinstance(arguments, str) else None
        if isinstance(arguments, str):
            try:
                arguments_obj = json.loads(arguments)
            except json.JSONDecodeError:
                arguments_obj = {}
        elif isinstance(arguments, dict):
            arguments_obj = arguments
        else:
            arguments_obj = {}
        parsed.append(
            ToolCall(
                id=str(call_id),
                name=str(name),
                arguments=arguments_obj,
                arguments_raw=arguments_raw,
            )
        )
    return parsed


def _extract_response_payload(response: Any) -> tuple[str, list[ToolCall]]:
    output = _get_attr(response, "output", []) or []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in output:
        item_type = _get_attr(item, "type", "")
        if item_type in {"message", "output_message"}:
            content = _get_attr(item, "content", []) or []
            for chunk in content:
                chunk_type = _get_attr(chunk, "type", "")
                if chunk_type in {"output_text", "text"}:
                    text = _get_attr(chunk, "text", "")
                    if text:
                        text_parts.append(str(text))
            raw_tool_calls = _get_attr(item, "tool_calls", None)
            if raw_tool_calls:
                tool_calls.extend(_parse_tool_calls(raw_tool_calls))
        elif item_type == "tool_call":
            tool_calls.extend(_parse_tool_calls([item]))
        elif item_type in {"output_text", "text"}:
            text = _get_attr(item, "text", "")
            if text:
                text_parts.append(str(text))

    return "".join(text_parts).strip(), tool_calls


@dataclass(slots=True)
class LiteLLMChatClient:
    model: str
    temperature: float
    timeout_seconds: int

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "input": [message.to_openai() for message in messages],
        }
        if tools:
            payload["tools"] = [tool.to_openai() for tool in tools]
            payload["tool_choice"] = "auto"

        response = await aresponses(**payload)
        text, tool_calls = _extract_response_payload(response)
        return ModelResponse(text=text, tool_calls=tool_calls, raw=response)


@dataclass(slots=True)
class AgentRunner:
    """Run a tool-enabled agent loop until completion."""

    client: LiteLLMChatClient
    registry: ToolRegistry
    executor: ToolExecutor
    max_steps: int = 6

    async def run(
        self,
        user_input: str,
        *,
        auth: ToolAuthContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentResult:
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_input))

        call_count = 0
        for step in range(self.max_steps):
            response = await self.client.complete(messages, self.registry.list_specs(auth))
            if response.tool_calls:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )
                summary = await self.executor.execute(
                    response.tool_calls,
                    auth=auth,
                    call_count=call_count,
                )
                call_count = summary.call_count
                for call, result in zip(response.tool_calls, summary.results, strict=False):
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=result.content,
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
                continue

            return AgentResult(
                output=response.text.strip(),
                steps=step + 1,
                tool_calls_executed=call_count,
            )

        return AgentResult(output="", steps=self.max_steps, tool_calls_executed=call_count)
