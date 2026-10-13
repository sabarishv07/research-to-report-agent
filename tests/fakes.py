"""Test doubles: a scripted LLM and helpers to build its responses."""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.llm import LLMClient, LLMResponse, ToolUse


def text_response(text: Any, input_tokens: int = 100, output_tokens: int = 50) -> LLMResponse:
    if not isinstance(text, str):
        text = json.dumps(text)
    return LLMResponse(
        text=text,
        content_blocks=[{"type": "text", "text": text}],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def tool_response(name: str, args: dict, tool_id: str = "tu_1") -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[ToolUse(id=tool_id, name=name, input=args)],
        content_blocks=[{"type": "tool_use", "id": tool_id, "name": name, "input": args}],
        stop_reason="tool_use",
        input_tokens=100,
        output_tokens=50,
    )


class FakeLLM(LLMClient):
    """LLMClient whose network call is replaced by `handler(system, messages, tools)`."""

    def __init__(self, settings, tracer, guardrails, handler: Callable[..., LLMResponse]) -> None:
        super().__init__(settings, tracer, guardrails)
        self.handler = handler
        self.calls = 0

    def _create(self, system, messages, tools, max_tokens):
        self.calls += 1
        return self.handler(system, messages, tools)
