"""Agent runner with tool-call handling."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from litellm import acompletion, aresponses

from exec_agent.agent.types import AgentResult, ChatMessage, ModelResponse, ToolCall
from exec_agent.tools.base import ToolSpec
from exec_agent.tools.executor import ToolExecutor
from exec_agent.tools.policies import ToolAuthContext
from exec_agent.tools.registry import ToolRegistry

LOGGER = structlog.get_logger(__name__)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_tool_calls(tool_calls: Iterable[Any]) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    for idx, raw_call in enumerate(tool_calls):
        call_id = _get_attr(raw_call, "call_id", None) or _get_attr(raw_call, "id", f"call_{idx}")
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
                    text_parts.extend(_coerce_text_chunks(_get_attr(chunk, "text", None) or chunk))
                else:
                    text_parts.extend(_coerce_text_chunks(chunk))
            raw_tool_calls = _get_attr(item, "tool_calls", None)
            if raw_tool_calls:
                tool_calls.extend(_parse_tool_calls(raw_tool_calls))
        elif item_type == "tool_call":
            tool_calls.extend(_parse_tool_calls([item]))
        elif item_type in {"output_text", "text"}:
            text = _get_attr(item, "text", "")
            if text:
                text_parts.append(str(text))
        else:
            if _looks_like_tool_call(item):
                tool_calls.extend(_parse_tool_calls([item]))
                continue
            text_parts.extend(_coerce_text_chunks(_get_attr(item, "text", None)))
            content = _get_attr(item, "content", None)
            if content:
                text_parts.extend(_coerce_text_chunks(content))

    if not text_parts:
        output_text = _get_attr(response, "output_text", "")
        if isinstance(output_text, str) and output_text:
            text_parts.append(output_text)

    if not text_parts:
        message_text = _get_attr(response, "text", "")
        text_parts.extend(_coerce_text_chunks(message_text))

    if not text_parts:
        choices = _get_attr(response, "choices", []) or []
        message = _get_attr(choices[0], "message", {}) if choices else {}
        choice_text = _get_attr(message, "content", "")
        text_parts.extend(_coerce_text_chunks(choice_text))
        choice_tool_calls = _get_attr(message, "tool_calls", []) or []
        if choice_tool_calls:
            tool_calls.extend(_parse_tool_calls(choice_tool_calls))

    return "".join(text_parts).strip(), tool_calls


def _coerce_text_chunks(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, dict):
            text = text.get("value") or text.get("text")
        if text:
            return [str(text)]
        direct = value.get("value")
        if direct:
            return [str(direct)]
        content = value.get("content")
        if content:
            return _coerce_text_chunks(content)
        return []
    if hasattr(value, "text"):
        text = value.text
        if isinstance(text, dict):
            text = text.get("value") or text.get("text")
        return [str(text)] if text else []
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            chunks.extend(_coerce_text_chunks(item))
        return chunks
    return []


def _looks_like_tool_call(value: Any) -> bool:
    return bool(_get_attr(value, "name", None)) or _get_attr(value, "arguments", None) is not None


def _summarize_response(response: Any) -> dict[str, Any]:
    output = _get_attr(response, "output", None)
    output_len = len(output) if isinstance(output, list) else None
    output_types = None
    output_preview = None
    if isinstance(output, list):
        output_types = [type(item).__name__ for item in output]
        if output:
            output_preview = _preview_value(output[0])
    text_value = _get_attr(response, "text", None)
    text_keys = list(text_value.keys()) if isinstance(text_value, dict) else None
    return {
        "type": type(response).__name__,
        "has_output": output is not None,
        "output_len": output_len,
        "output_types": output_types,
        "output_preview": output_preview,
        "has_output_text": bool(_get_attr(response, "output_text", None)),
        "has_text": bool(_get_attr(response, "text", None)),
        "text_type": type(_get_attr(response, "text", None)).__name__,
        "text_keys": text_keys,
        "has_choices": bool(_get_attr(response, "choices", None)),
    }


def _preview_value(value: Any, limit: int = 800) -> str:
    data = value
    if hasattr(value, "model_dump"):
        try:
            data = value.model_dump()
        except Exception:
            data = value
    elif hasattr(value, "dict"):
        try:
            data = value.dict()
        except Exception:
            data = value
    elif hasattr(value, "__dict__"):
        data = value.__dict__

    try:
        text = json.dumps(data, ensure_ascii=True, default=str)
    except TypeError:
        text = repr(data)
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def _to_chat_tool_spec(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


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
            "input": [item for message in messages for item in _to_responses_inputs(message)],
        }
        if tools:
            payload["tools"] = [tool.to_openai() for tool in tools]
            payload["tool_choice"] = "auto"

        response = await aresponses(**payload)
        text, tool_calls = _extract_response_payload(response)
        if not text and not tool_calls:
            LOGGER.debug(
                "llm.empty_response",
                summary=_summarize_response(response),
                model=self.model,
                fallback="acompletion",
            )
            completion_payload = {
                "model": self.model,
                "temperature": self.temperature,
                "timeout": self.timeout_seconds,
                "messages": [message.to_openai() for message in messages],
            }
            if tools:
                completion_payload["tools"] = [_to_chat_tool_spec(tool) for tool in tools]
                completion_payload["tool_choice"] = "auto"
            response = await acompletion(**completion_payload)
            text, tool_calls = _extract_response_payload(response)
            if not text and not tool_calls:
                LOGGER.debug(
                    "llm.empty_response",
                    summary=_summarize_response(response),
                    model=self.model,
                    fallback="none",
                )
        return ModelResponse(text=text, tool_calls=tool_calls, raw=response)


def _to_responses_inputs(message: ChatMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if message.role == "tool":
        items.append(
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id or "",
                "output": message.content or "",
            }
        )
        return items

    if message.content:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        items.append(payload)

    if message.tool_calls:
        for call in message.tool_calls:
            arguments = call.arguments_raw or json.dumps(call.arguments, ensure_ascii=True)
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": arguments,
                }
            )

    return items


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
                assistant_text = response.text.strip()
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=assistant_text or None,
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
