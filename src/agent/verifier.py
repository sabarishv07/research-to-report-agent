"""Verifier: every claim must be backed by the text of the page it cites.

Two layers:
1. Cheap, deterministic: the cited URL must have been fetched in this run (otherwise it is
   a hallucinated citation).
2. LLM fact-check of the claim against the stored page text, batched per source.
"""

from __future__ import annotations

from .config import Settings
from .exceptions import BadJSONError
from .llm import LLMClient
from .models import Claim, Finding, VerdictBatch, VerifiedClaim
from .tools import SourceStore
from .tracing import Tracer

VERIFIER_SYSTEM = (
    "You are a strict fact-checker. You are given the text of one web page and a list of claims "
    "that cite it. For each claim decide whether the page text explicitly supports it. Numbers, "
    "dates and names must match. If the page does not clearly say it, mark supported=false. "
    "Return one verdict per claim, using the claim's index."
)


class Verifier:
    def __init__(self, settings: Settings, llm: LLMClient, sources: SourceStore, tracer: Tracer) -> None:
        self.settings = settings
        self.llm = llm
        self.sources = sources
        self.tracer = tracer

    @staticmethod
    def _flatten(findings: list[Finding]) -> list[tuple[int, Claim]]:
        return [(f.sub_question_id, c) for f in findings for c in f.claims]

    def mark_unverified(self, findings: list[Finding]) -> list[VerifiedClaim]:
        """Used when the LLM can't be called (budget spent). Still rejects never-fetched URLs."""
        out = []
        for sid, claim in self._flatten(findings):
            if self.sources.has(claim.source_url):
                out.append(VerifiedClaim(sub_question_id=sid, claim=claim, status="unverified",
                                         reason="not verified (budget exceeded)"))
            else:
                self.tracer.failure("hallucinated_citation", claim.source_url)
                out.append(VerifiedClaim(sub_question_id=sid, claim=claim, status="unsupported",
                                         reason="URL was not fetched during this run"))
        return out

    def verify(self, findings: list[Finding]) -> list[VerifiedClaim]:
        """May raise BudgetExceededError (handled by the pipeline)."""
        items = self._flatten(findings)
        results: list[VerifiedClaim | None] = [None] * len(items)
        groups: dict[str, list[int]] = {}

        for idx, (sid, claim) in enumerate(items):
            source = self.sources.get(claim.source_url)
            if source is None:
                self.tracer.failure("hallucinated_citation", claim.source_url)
                results[idx] = VerifiedClaim(
                    sub_question_id=sid, claim=claim, status="unsupported",
                    reason="URL was not fetched during this run",
                )
            else:
                groups.setdefault(source.url, []).append(idx)

        for url, indices in groups.items():
            source = self.sources.get(url)
            assert source is not None
            numbered = "\n".join(f"[{n}] {items[i][1].text}" for n, i in enumerate(indices))
            prompt = (
                f"Page URL: {url}\n\nPage text:\n{source.text[: self.settings.verify_chars]}\n\n"
                f"Claims to check:\n{numbered}"
            )
            try:
                batch = self.llm.json_call(VERIFIER_SYSTEM, prompt, VerdictBatch, label="verifier")
            except BadJSONError:
                for i in indices:
                    sid, claim = items[i]
                    results[i] = VerifiedClaim(
                        sub_question_id=sid, claim=claim, status="unverified",
                        reason="verifier returned malformed output",
                    )
                continue
            verdicts = {v.index: v for v in batch.verdicts}
            for n, i in enumerate(indices):
                sid, claim = items[i]
                verdict = verdicts.get(n)
                if verdict is not None and verdict.supported:
                    results[i] = VerifiedClaim(sub_question_id=sid, claim=claim, status="supported",
                                               reason=verdict.reason)
                else:
                    reason = verdict.reason if verdict else "no verdict returned"
                    self.tracer.failure("hallucinated_citation", f"{claim.source_url}: {claim.text[:80]}")
                    results[i] = VerifiedClaim(sub_question_id=sid, claim=claim, status="unsupported",
                                               reason=reason)

        verified = [r for r in results if r is not None]
        self.tracer.event(
            "verify",
            total=len(verified),
            supported=sum(v.status == "supported" for v in verified),
            unsupported=sum(v.status == "unsupported" for v in verified),
            unverified=sum(v.status == "unverified" for v in verified),
        )
        return verified
