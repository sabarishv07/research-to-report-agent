"""Thin wrapper around the Anthropic API.

Responsibilities: retries on transient errors, token/cost accounting (via Guardrails),
tracing, and a validated-JSON helper that re-asks the model when its output is malformed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from .config import Settings
from .exceptions import BadJSONError
from .guardrails import Guardrails
from .tracing import Tracer

T = TypeVar("T", bound=BaseModel)


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolUse] = field(default_factory=list)
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0


def extract_json(text: str) -> Any:
    """Pull a JSON object out of model text (tolerates code fences and surrounding prose)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


def _is_transient(exc: BaseException) -> bool:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return False
    return isinstance(
        exc,
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError),
    )


class LLMClient:
    def __init__(self, settings: Settings, tracer: Tracer, guardrails: Guardrails) -> None:
        self.settings = settings
        self.tracer = tracer
        self.guardrails = guardrails
        self._client: Any = None

    # ---- the only method that touches the network (override in tests) ---------
    def _create(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> LLMResponse:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key or None, timeout=60.0, max_retries=0
            )
        kwargs: dict[str, Any] = dict(
            model=self.settings.llm_model, max_tokens=max_tokens, system=system, messages=messages
        )
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        blocks: list[dict[str, Any]] = []
        calls: list[ToolUse] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(ToolUse(id=block.id, name=block.name, input=dict(block.input)))
                blocks.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
                )
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=calls,
            content_blocks=blocks,
            stop_reason=resp.stop_reason or "end_turn",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    # ---- public API -----------------------------------------------------------
    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        label: str = "chat",
    ) -> LLMResponse:
        max_tokens = max_tokens or self.settings.llm_max_tokens
        started = time.perf_counter()
        resp: LLMResponse | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, max=20),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        ):
            with attempt:
                resp = self._create(system, messages, tools, max_tokens)
        assert resp is not None
        latency = time.perf_counter() - started
        self.tracer.event(
            "llm_call",
            label=label,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=round(self.guardrails.cost_of(resp.input_tokens, resp.output_tokens), 6),
            latency_s=round(latency, 2),
            stop_reason=resp.stop_reason,
            tool_calls=[c.name for c in resp.tool_calls],
            text_preview=resp.text[:200],
        )
        self.guardrails.record_usage(resp.input_tokens, resp.output_tokens)  # may raise
        return resp

    def json_call(
        self,
        system: str,
        prompt: str,
        model_cls: type[T],
        label: str = "json",
        max_tokens: int | None = None,
    ) -> T:
        """Ask for JSON matching `model_cls`. Malformed output is fed back to the model for a fix."""
        schema = json.dumps(model_cls.model_json_schema())
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"{prompt}\n\nRespond with ONLY a JSON object matching this JSON schema "
                f"(no prose, no code fences):\n{schema}",
            }
        ]
        last_error = ""
        for _ in range(self.settings.json_max_retries + 1):
            resp = self.chat(system, messages, max_tokens=max_tokens, label=label)
            try:
                return model_cls.model_validate(extract_json(resp.text))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:300]
                self.tracer.failure("bad_json", f"{label}: {last_error}")
                messages.append({"role": "assistant", "content": resp.text or "(empty)"})
                messages.append(
                    {
                        "role": "user",
                        "content": f"That reply was not valid. Error: {last_error}\n"
                        "Reply again with ONLY the corrected JSON object.",
                    }
                )
        raise BadJSONError(f"{label}: model did not return valid JSON ({last_error})")
