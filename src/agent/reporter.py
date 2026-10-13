"""Report generator: structured, cited markdown built only from verified claims."""

from __future__ import annotations

import re
from datetime import date

from .config import Settings
from .llm import LLMClient
from .models import Finding, ReportDraft, VerifiedClaim
from .tools.sources import normalize_url
from .tracing import Tracer

REPORTER_SYSTEM = (
    "You are a report writer. Write a clear, well-structured research report using ONLY the "
    "verified claims provided. Do not add facts, numbers or sources of your own. Cite with the "
    "bracketed source numbers given next to each claim, like [1] or [2][3]. Group related claims "
    "into sections with descriptive headings. Be concise and neutral."
)


def compute_unanswered(findings: list[Finding], verified: list[VerifiedClaim]) -> list[str]:
    """Sub-questions with no usable (supported/unverified) claim."""
    answered_ids = {v.sub_question_id for v in verified if v.usable}
    return [f.question for f in findings if f.sub_question_id not in answered_ids]


def build_source_index(verified: list[VerifiedClaim]) -> list[str]:
    """Ordered, de-duplicated URLs of claims that may appear in the report."""
    urls: list[str] = []
    seen: set[str] = set()
    for v in verified:
        if v.usable and normalize_url(v.claim.source_url) not in seen:
            seen.add(normalize_url(v.claim.source_url))
            urls.append(v.claim.source_url)
    return urls


def strip_invalid_citations(text: str, max_n: int) -> str:
    """Remove [n] markers that don't point at a real source number."""
    return re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= max_n else "", text)


def render_markdown(
    topic: str,
    title: str,
    summary: str,
    sections: list[tuple[str, str]],
    unanswered: list[str],
    sources: list[str],
    notes: list[str],
) -> str:
    out = [f"# {title}", "", f"*Topic: {topic} · Generated {date.today().isoformat()}*", ""]
    for note in notes:
        out += [f"> **Note:** {note}", ""]
    if summary:
        out += ["## Executive summary", "", summary, ""]
    for heading, body in sections:
        out += [f"## {heading}", "", body, ""]
    if unanswered:
        out += ["## Unanswered questions", "", "No verifiable answer was found for:", ""]
        out += [f"- {q}" for q in unanswered] + [""]
    if sources:
        out += ["## Sources", ""] + [f"{i}. {u}" for i, u in enumerate(sources, 1)] + [""]
    return "\n".join(out)


def render_fallback(
    topic: str,
    findings: list[Finding],
    verified: list[VerifiedClaim],
    unanswered: list[str],
    notes: list[str],
) -> str:
    """Deterministic report (no LLM). Used when drafting fails or the budget is spent."""
    sources = build_source_index(verified)
    index = {normalize_url(u): i for i, u in enumerate(sources, 1)}
    sections: list[tuple[str, str]] = []
    for f in findings:
        lines = []
        for v in verified:
            if v.sub_question_id == f.sub_question_id and v.usable:
                tag = " *(unverified)*" if v.status == "unverified" else ""
                lines.append(f"- {v.claim.text} [{index[normalize_url(v.claim.source_url)]}]{tag}")
        if lines:
            sections.append((f.question, "\n".join(lines)))
    return render_markdown(
        topic, f"Research notes: {topic}", "", sections, unanswered, sources, notes
    )


class Reporter:
    def __init__(self, settings: Settings, llm: LLMClient, tracer: Tracer) -> None:
        self.settings = settings
        self.llm = llm
        self.tracer = tracer

    def write(
        self,
        topic: str,
        findings: list[Finding],
        verified: list[VerifiedClaim],
        unanswered: list[str],
        notes: list[str],
    ) -> str:
        """Draft with the LLM, then sanitise. May raise BadJSONError / BudgetExceededError."""
        sources = build_source_index(verified)
        index = {normalize_url(u): i for i, u in enumerate(sources, 1)}

        blocks = []
        for f in findings:
            claims = [v for v in verified if v.sub_question_id == f.sub_question_id and v.usable]
            if not claims:
                continue
            lines = "\n".join(f"- {v.claim.text} [{index[normalize_url(v.claim.source_url)]}]" for v in claims)
            blocks.append(f"Sub-question: {f.question}\n{lines}")
        prompt = f"Topic: {topic}\n\nVerified claims:\n\n" + "\n\n".join(blocks)

        draft = self.llm.json_call(REPORTER_SYSTEM, prompt, ReportDraft, label="reporter", max_tokens=4000)
        n = len(sources)
        markdown = render_markdown(
            topic,
            draft.title.strip() or f"Research report: {topic}",
            strip_invalid_citations(draft.executive_summary, n),
            [(s.heading, strip_invalid_citations(s.body, n)) for s in draft.sections],
            unanswered,
            sources,
            notes,
        )
        self.tracer.event("report", sections=len(draft.sections), sources=n, chars=len(markdown))
        return markdown

    def fallback(
        self,
        topic: str,
        findings: list[Finding],
        verified: list[VerifiedClaim],
        unanswered: list[str],
        notes: list[str],
    ) -> str:
        markdown = render_fallback(topic, findings, verified, unanswered, notes)
        self.tracer.event("report", fallback=True, chars=len(markdown))
        return markdown
