"""Guardrails: cost ceiling, iteration cap, stuck-loop detector."""

from __future__ import annotations

import json

from .config import Settings
from .exceptions import BudgetExceededError, IterationLimitError, StuckLoopError
from .tracing import Tracer


class Guardrails:
    def __init__(self, settings: Settings, tracer: Tracer) -> None:
        self.settings = settings
        self.tracer = tracer
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.total_iterations = 0
        self.budget_exceeded = False
        self._seen_calls: set[str] = set()
        self._repeats = 0

    # ---- cost -----------------------------------------------------------
    def cost_of(self, input_tokens: int, output_tokens: int) -> float:
        s = self.settings
        return (
            input_tokens / 1_000_000 * s.price_input_per_mtok
            + output_tokens / 1_000_000 * s.price_output_per_mtok
        )

    def record_usage(self, input_tokens: int, output_tokens: int) -> float:
        """Add one LLM call to the running total. Raises once the ceiling is crossed."""
        cost = self.cost_of(input_tokens, output_tokens)
        self.cost_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls += 1
        if self.cost_usd > self.settings.max_cost_usd:
            self.budget_exceeded = True
            detail = f"${self.cost_usd:.4f} spent, ceiling ${self.settings.max_cost_usd:.2f}"
            self.tracer.failure("budget_exceeded", detail)
            raise BudgetExceededError(detail)
        return cost

    # ---- iterations -----------------------------------------------------
    def tick(self) -> None:
        """Count one agent step against the run-wide cap."""
        self.total_iterations += 1
        if self.total_iterations > self.settings.max_total_iterations:
            detail = f"more than {self.settings.max_total_iterations} agent steps"
            self.tracer.failure("iteration_limit", detail)
            raise IterationLimitError(detail)

    # ---- stuck detector ---------------------------------------------------
    def begin_subquestion(self) -> None:
        """Repeat tracking is scoped to one sub-question."""
        self._seen_calls.clear()
        self._repeats = 0

    def register_call(self, tool_name: str, args: dict) -> bool:
        """Return True if this exact call was already made (caller should refuse it and
        tell the model to change strategy). Raises StuckLoopError after repeated repeats."""
        signature = tool_name + json.dumps(args, sort_keys=True, default=str)
        if signature not in self._seen_calls:
            self._seen_calls.add(signature)
            return False
        self._repeats += 1
        if self._repeats >= self.settings.max_stuck_repeats:
            self.tracer.failure("stuck_loop", f"{tool_name} repeated with identical arguments")
            raise StuckLoopError(f"{tool_name} called repeatedly with identical arguments")
        return True
