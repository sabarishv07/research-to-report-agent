"""Typed data that crosses module boundaries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: int
    question: str


class Plan(BaseModel):
    topic: str
    sub_questions: list[SubQuestion] = Field(min_length=1)


class Claim(BaseModel):
    """One atomic factual statement and the fetched URL that supports it."""

    text: str
    source_url: str


class SelfCheck(BaseModel):
    """Executor self-check output: is the question answered, and which claims came out of it."""

    answered: bool
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    missing: str = ""


class Finding(BaseModel):
    sub_question_id: int
    question: str
    answered: bool
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    notes: str = ""


ClaimStatus = Literal["supported", "unsupported", "unverified"]


class VerifiedClaim(BaseModel):
    sub_question_id: int
    claim: Claim
    status: ClaimStatus
    reason: str = ""

    @property
    def usable(self) -> bool:
        """Supported claims, and unverified ones (verifier could not run), may appear in a report."""
        return self.status in ("supported", "unverified")


class VerdictItem(BaseModel):
    index: int
    supported: bool
    reason: str = ""


class VerdictBatch(BaseModel):
    verdicts: list[VerdictItem]


class ReportSection(BaseModel):
    heading: str
    body: str


class ReportDraft(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection]


RunStatus = Literal["success", "partial", "failed"]


class RunResult(BaseModel):
    run_id: str
    topic: str
    status: RunStatus
    report_markdown: str = ""
    report_path: str | None = None
    trace_path: str | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    llm_calls: int = 0
    failures: dict[str, int] = Field(default_factory=dict)
    unanswered: list[str] = Field(default_factory=list)
    emailed: bool = False
    error: str | None = None
