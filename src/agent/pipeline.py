"""End-to-end orchestration: plan -> research -> verify -> report -> deliver.

Design goal: `run()` never raises. Whatever goes wrong, the caller gets a RunResult whose
status ("success" / "partial" / "failed") and failure counts explain what happened.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

from .config import Settings, get_settings
from .exceptions import BadJSONError, BudgetExceededError, IterationLimitError
from .executor import Executor
from .guardrails import Guardrails
from .llm import LLMClient
from .models import Finding, Plan, RunResult, RunStatus, VerifiedClaim
from .planner import Planner
from .reporter import Reporter, compute_unanswered
from .tools import EmailTool, SendEmailInput, SourceStore, ToolRegistry, default_registry
from .tracing import Tracer
from .verifier import Verifier

LLMFactory = Callable[[Settings, Tracer, Guardrails], LLMClient]
RegistryFactory = Callable[[Settings, SourceStore], ToolRegistry]


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        llm_factory: LLMFactory | None = None,
        registry_factory: RegistryFactory | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_factory = llm_factory or (lambda s, t, g: LLMClient(s, t, g))
        self.registry_factory = registry_factory or default_registry

    # ------------------------------------------------------------------------------
    def run(self, topic: str, email_to: str | None = None, run_id: str | None = None) -> RunResult:
        s = self.settings
        run_id = run_id or uuid.uuid4().hex[:8]
        tracer = Tracer(run_id, Path(s.traces_dir))
        guardrails = Guardrails(s, tracer)
        sources = SourceStore(s.source_store_chars)
        llm = self.llm_factory(s, tracer, guardrails)
        registry = self.registry_factory(s, sources)
        started = time.perf_counter()
        tracer.event("run_start", topic=topic, model=s.llm_model, max_cost_usd=s.max_cost_usd)

        findings: list[Finding] = []
        verified: list[VerifiedClaim] = []
        unanswered: list[str] = []
        notes: list[str] = []
        markdown = ""
        status: RunStatus = "failed"
        error: str | None = None
        emailed = False
        report_path: Path | None = None

        try:
            plan = Planner(llm, s).plan(topic)
            tracer.event("plan", sub_questions=[q.model_dump() for q in plan.sub_questions])

            executor = Executor(s, llm, registry, guardrails, tracer, sources)
            findings, stop_reason = self._research(topic, plan, executor, tracer)
            if stop_reason:
                notes.append(f"Research stopped early: {stop_reason}. Some questions are unanswered.")

            verifier = Verifier(s, llm, sources, tracer)
            if guardrails.budget_exceeded:
                verified = verifier.mark_unverified(findings)
            else:
                try:
                    verified = verifier.verify(findings)
                except BudgetExceededError:
                    verified = verifier.mark_unverified(findings)
            if guardrails.budget_exceeded:
                notes.append(
                    f"The cost ceiling (${s.max_cost_usd:.2f}) was reached, so claims marked "
                    "'unverified' were not fact-checked."
                )

            unanswered = compute_unanswered(findings, verified)
            usable = [v for v in verified if v.usable]
            reporter = Reporter(s, llm, tracer)

            drafted_by_llm = False
            if usable and not guardrails.budget_exceeded:
                try:
                    markdown = reporter.write(topic, findings, verified, unanswered, notes)
                    drafted_by_llm = True
                except BudgetExceededError:
                    notes.append("The cost ceiling was reached while drafting; showing a plain summary.")
                except BadJSONError:
                    notes.append("Report drafting failed; showing a plain summary of verified claims.")
            if not drafted_by_llm:
                markdown = reporter.fallback(topic, findings, verified, unanswered, notes)

            if not usable:
                error = "No verifiable findings were produced."
                status = "failed"
            elif drafted_by_llm and not unanswered and not stop_reason and not guardrails.budget_exceeded:
                status = "success"
            else:
                status = "partial"

            report_path = Path(s.reports_dir) / f"{run_id}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(markdown, encoding="utf-8")

            if email_to and usable:
                emailed = self._deliver(email_to, topic, markdown, run_id, tracer)
        except Exception as exc:  # noqa: BLE001 - run() must always return a result
            error = f"{type(exc).__name__}: {exc}"
            status = "failed"
            tracer.event("error", error=error)
            if not markdown:
                markdown = f"# Research failed\n\n{error}\n"

        latency = time.perf_counter() - started
        tracer.event(
            "run_end",
            status=status,
            cost_usd=round(guardrails.cost_usd, 5),
            latency_s=round(latency, 2),
            llm_calls=guardrails.llm_calls,
            failures=dict(tracer.failures),
        )
        return RunResult(
            run_id=run_id,
            topic=topic,
            status=status,
            report_markdown=markdown,
            report_path=str(report_path) if report_path else None,
            trace_path=str(tracer.path) if tracer.path else None,
            cost_usd=round(guardrails.cost_usd, 5),
            latency_s=round(latency, 2),
            llm_calls=guardrails.llm_calls,
            failures=dict(tracer.failures),
            unanswered=unanswered,
            emailed=emailed,
            error=error,
        )

    # ------------------------------------------------------------------------------
    def _research(
        self, topic: str, plan: Plan, executor: Executor, tracer: Tracer
    ) -> tuple[list[Finding], str | None]:
        """Run every sub-question. Budget / iteration limits end research early but keep
        whatever was already found (graceful degradation)."""
        findings: list[Finding] = []
        stop_reason: str | None = None

        def unanswered(sq_id: int, question: str, why: str) -> Finding:
            return Finding(sub_question_id=sq_id, question=question, answered=False, notes=why)

        for sq in plan.sub_questions:
            if stop_reason:
                findings.append(unanswered(sq.id, sq.question, f"Skipped: {stop_reason}."))
                continue
            try:
                findings.append(executor.research(topic, sq))
            except BudgetExceededError:
                stop_reason = "cost budget exceeded"
                findings.append(unanswered(sq.id, sq.question, "Stopped: cost budget exceeded."))
            except IterationLimitError:
                stop_reason = "total iteration limit reached"
                findings.append(unanswered(sq.id, sq.question, "Stopped: iteration limit reached."))
            except BadJSONError:
                findings.append(unanswered(sq.id, sq.question, "Model returned malformed output."))
            except Exception as exc:  # noqa: BLE001 - one bad sub-question must not sink the run
                tracer.event("error", sub_question=sq.id, error=f"{type(exc).__name__}: {exc}")
                findings.append(unanswered(sq.id, sq.question, "Unexpected error."))
        return findings, stop_reason

    def _deliver(self, to: str, topic: str, markdown: str, run_id: str, tracer: Tracer) -> bool:
        tool = EmailTool(self.settings)
        result = tool.execute_safely(
            SendEmailInput(
                to=to,
                subject=f"Research report: {topic}"[:200],
                body=markdown,
                attachment_name=f"report_{run_id}.md",
            )
        )
        if result.ok:
            tracer.event("email", to=to, ok=True)
            return True
        tracer.failure("email_failed", result.content[:200])
        return False
