"""Executor loop: for each sub-question, the model picks tools, reads sources, and a
self-check step decides whether the question is actually answered."""

from __future__ import annotations

import time
from typing import Any

from .config import Settings
from .exceptions import StuckLoopError
from .guardrails import Guardrails
from .llm import LLMClient, ToolUse
from .models import Finding, SelfCheck, SubQuestion
from .tools import SourceStore, ToolRegistry
from .tracing import Tracer

RESEARCHER_SYSTEM = """You are a careful research assistant with web tools.

Rules:
- Use web_search to find candidate pages, then web_fetch to read the best ones. Search snippets are not enough to cite.
- Only state facts that appear in pages you fetched. If sources disagree or a fact is missing, say so plainly.
- Never repeat an identical tool call. Change the query or pick a different URL.
- Use calculator for arithmetic instead of doing it in your head.
- When you have enough evidence (usually 1-3 pages), stop calling tools and write concise research notes, listing the URL that supports each fact."""

SELF_CHECK_SYSTEM = """You are the self-check step of a research agent. Given a research question, the agent's notes and the list of URLs it actually fetched, decide whether the notes answer the question and extract atomic claims.

Rules:
- Each claim is one self-contained factual statement taken from the notes.
- source_url must be exactly one of the fetched URLs. If a fact has no fetched source, leave it out.
- Set answered=true only if the claims substantially answer the question. Otherwise set answered=false and describe in 'missing' what is still needed."""


class Executor:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        registry: ToolRegistry,
        guardrails: Guardrails,
        tracer: Tracer,
        sources: SourceStore,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.registry = registry
        self.guardrails = guardrails
        self.tracer = tracer
        self.sources = sources

    def research(self, topic: str, sq: SubQuestion) -> Finding:
        """Research one sub-question. May raise BudgetExceededError / IterationLimitError
        (handled by the pipeline); a stuck loop is handled here."""
        self.guardrails.begin_subquestion()
        self.tracer.event("subquestion_start", id=sq.id, question=sq.question)
        try:
            finding = self._research(topic, sq)
        except StuckLoopError:
            finding = Finding(
                sub_question_id=sq.id,
                question=sq.question,
                answered=False,
                notes="Stopped: the agent kept repeating the same tool call.",
            )
        self.tracer.event(
            "subquestion_end", id=sq.id, answered=finding.answered, claims=len(finding.claims)
        )
        return finding

    # ------------------------------------------------------------------------------
    def _research(self, topic: str, sq: SubQuestion) -> Finding:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Overall topic: {topic}\nResearch question: {sq.question}\n\n"
                "Search the web, read the best sources, then write concise research notes "
                "with the URL that supports each fact.",
            }
        ]
        fetched: list[str] = []
        tools = self.registry.schemas()
        limit = self.settings.max_iterations_per_subquestion

        for _ in range(limit):
            self.guardrails.tick()
            resp = self.llm.chat(RESEARCHER_SYSTEM, messages, tools=tools, label="researcher")
            messages.append(
                {"role": "assistant", "content": resp.content_blocks or [{"type": "text", "text": "(no output)"}]}
            )

            if resp.tool_calls:
                results = [self._run_tool(call, fetched) for call in resp.tool_calls]
                messages.append({"role": "user", "content": results})
                continue

            if not fetched:  # cost guard: no point self-checking notes with no sources behind them
                messages.append(
                    {
                        "role": "user",
                        "content": "You have not read any page yet. Use web_search, then web_fetch "
                        "a relevant result, before writing notes.",
                    }
                )
                continue

            check = self.llm.json_call(
                SELF_CHECK_SYSTEM, self._self_check_prompt(sq, resp.text, fetched), SelfCheck, label="self_check"
            )
            self.tracer.event(
                "self_check", id=sq.id, answered=check.answered, claims=len(check.claims), missing=check.missing
            )
            if check.answered and check.claims:
                return Finding(
                    sub_question_id=sq.id,
                    question=sq.question,
                    answered=True,
                    answer=check.answer,
                    claims=check.claims,
                )
            messages.append(
                {
                    "role": "user",
                    "content": "Self-check: the question is not fully answered yet. Missing: "
                    f"{check.missing or 'claims backed by a fetched source'}. "
                    "Try a different query or source.",
                }
            )

        self.tracer.failure("iteration_limit", f"sub-question {sq.id} unanswered after {limit} steps")
        return Finding(
            sub_question_id=sq.id,
            question=sq.question,
            answered=False,
            notes=f"Not answered within {limit} steps.",
        )

    @staticmethod
    def _self_check_prompt(sq: SubQuestion, notes: str, fetched: list[str]) -> str:
        urls = "\n".join(f"- {u}" for u in fetched)
        return f"Question: {sq.question}\n\nFetched URLs:\n{urls}\n\nAgent notes:\n{notes}"

    def _run_tool(self, call: ToolUse, fetched: list[str]) -> dict[str, Any]:
        self.tracer.event("tool_call", tool=call.name, args=call.input)
        if self.guardrails.register_call(call.name, call.input):  # may raise StuckLoopError
            return {
                "type": "tool_result",
                "tool_use_id": call.id,
                "is_error": True,
                "content": "You already made this exact call. Do not repeat it: change the query, "
                "pick a different URL, or finish by writing your notes.",
            }
        started = time.perf_counter()
        result = self.registry.execute(call.name, call.input)
        self.tracer.event(
            "tool_result",
            tool=call.name,
            ok=result.ok,
            chars=len(result.content),
            latency_s=round(time.perf_counter() - started, 2),
            preview=result.content[:200],
        )
        if call.name == "web_fetch":
            if result.ok:
                url = result.data.get("url")
                if url and url not in fetched:
                    fetched.append(url)
            else:
                self.tracer.failure("fetch_failed", result.content[:200])
        elif call.name == "web_search" and result.ok and result.data.get("empty"):
            self.tracer.failure("search_empty", str(call.input)[:200])
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": result.content,
            "is_error": not result.ok,
        }
