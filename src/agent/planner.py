"""Planner: topic -> focused, searchable sub-questions."""

from __future__ import annotations

from .config import Settings
from .llm import LLMClient
from .models import Plan, SubQuestion

PLANNER_SYSTEM = (
    "You are a research planner. Break a research topic into focused, independently answerable "
    "sub-questions that together cover it. Prefer concrete, searchable questions (who, what, how "
    "much, when) over vague ones. Never include questions that need private or paywalled data."
)


class Planner:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def plan(self, topic: str) -> Plan:
        limit = self.settings.max_subquestions
        prompt = (
            f"Topic: {topic}\n\nProduce between 3 and {limit} sub-questions, numbered from 1. "
            f"Set the topic field to the topic text."
        )
        plan = self.llm.json_call(PLANNER_SYSTEM, prompt, Plan, label="planner")
        questions = [q.question.strip() for q in plan.sub_questions if q.question.strip()][:limit]
        # Re-number so ids are always 1..n regardless of what the model returned.
        return Plan(
            topic=topic,
            sub_questions=[SubQuestion(id=i, question=q) for i, q in enumerate(questions, 1)],
        )
