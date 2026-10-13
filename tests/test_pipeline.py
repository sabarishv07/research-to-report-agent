"""End-to-end tests: real pipeline, executor, verifier, reporter and tools, with the LLM
scripted and the network patched."""

from __future__ import annotations

import json

import pytest

from agent.pipeline import Pipeline
from agent.tools import EmailTool, FetchTool, SearchTool
from tests.fakes import FakeLLM, text_response, tool_response

URL = "https://example.com/article"
PAGE = "<html><title>Article</title><body><p>" + "The market grew by 12 percent in 2025. " * 10 + "</p></body></html>"


@pytest.fixture(autouse=True)
def fake_network(monkeypatch):
    monkeypatch.setattr("agent.tools.fetch.is_safe_url", lambda url: True)
    monkeypatch.setattr(FetchTool, "_download", lambda self, url: (url, "text/html", PAGE))
    monkeypatch.setattr(
        SearchTool, "_search_ddgs", lambda self, q, n: [{"title": "Article", "url": URL, "snippet": "s"}]
    )


def make_handler(*, claims=None, verifier_tokens=100, researcher=None, plan_questions=2):
    """Routes each LLM call by the role named in its system prompt."""
    claims = claims if claims is not None else [{"text": "The market grew 12% in 2025.", "source_url": URL}]
    questions = [{"id": i, "question": f"Question {i}?"} for i in range(1, plan_questions + 1)]

    def handler(system, messages, tools):
        role = system.lower()
        if "research planner" in role:
            return text_response({"topic": "t", "sub_questions": questions})
        if "research assistant" in role:
            if researcher:
                return researcher(messages)
            n = sum(1 for m in messages if m["role"] == "assistant")
            if n == 0:
                return tool_response("web_search", {"query": "market growth"}, "a")
            if n == 1:
                return tool_response("web_fetch", {"url": URL}, "b")
            return text_response(f"Notes: market grew 12% in 2025 ({URL})")
        if "self-check" in role:
            return text_response({"answered": True, "answer": "It grew.", "claims": claims, "missing": ""})
        if "fact-checker" in role:
            verdicts = [{"index": i, "supported": True, "reason": "stated on page"} for i in range(10)]
            return text_response({"verdicts": verdicts}, input_tokens=verifier_tokens)
        if "report writer" in role:
            return text_response(
                {
                    "title": "Market Report",
                    "executive_summary": "The market grew [1]. Bogus citation [9].",
                    "sections": [{"heading": "Growth", "body": "Grew 12% in 2025 [1]."}],
                }
            )
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    return handler


def pipeline(settings, handler):
    return Pipeline(settings, llm_factory=lambda s, t, g: FakeLLM(s, t, g, handler))


def test_happy_path_produces_cited_report(settings):
    result = pipeline(settings, make_handler()).run("market growth")

    assert result.status == "success", result.error
    md = result.report_markdown
    assert "# Market Report" in md and "## Sources" in md and URL in md
    assert "[9]" not in md  # invalid citation stripped
    assert result.failures == {}
    assert result.unanswered == []
    assert result.cost_usd > 0 and result.llm_calls >= 6
    assert open(result.report_path, encoding="utf-8").read() == md
    # trace has the key step types
    types = {json.loads(line)["type"] for line in open(result.trace_path, encoding="utf-8")}
    assert {"run_start", "plan", "tool_call", "tool_result", "self_check", "verify", "report", "run_end"} <= types


def test_hallucinated_citation_is_rejected(settings):
    claims = [
        {"text": "Real claim.", "source_url": URL},
        {"text": "Invented claim.", "source_url": "https://never-fetched.example/x"},
    ]
    result = pipeline(settings, make_handler(claims=claims)).run("topic")

    assert result.failures["hallucinated_citation"] == 2  # once per sub-question
    assert "never-fetched" not in result.report_markdown
    assert result.status == "success"


def test_only_hallucinated_claims_means_failed_run(settings):
    claims = [{"text": "Invented.", "source_url": "https://never-fetched.example/x"}]
    result = pipeline(settings, make_handler(claims=claims)).run("topic")
    assert result.status == "failed"
    assert result.error == "No verifiable findings were produced."
    assert "Unanswered questions" in result.report_markdown


def test_budget_exceeded_during_verification_degrades_gracefully(settings):
    s = settings.model_copy(update={"max_cost_usd": 1.0})
    # 1M input tokens in the verifier call costs $3 -> over the $1 ceiling.
    result = pipeline(s, make_handler(verifier_tokens=1_000_000)).run("topic")

    assert result.status == "partial"
    assert result.failures["budget_exceeded"] == 1
    assert "unverified" in result.report_markdown
    assert "cost ceiling" in result.report_markdown
    assert URL in result.report_markdown


def test_budget_exceeded_during_research_returns_failed_with_report(settings):
    # each call costs ~$0.00105; ceiling lets planner + one researcher call through
    s = settings.model_copy(update={"max_cost_usd": 0.0025})
    result = pipeline(s, make_handler()).run("topic")

    assert result.failures["budget_exceeded"] >= 1
    assert result.status == "failed"
    assert result.report_markdown  # never empty
    assert len(result.unanswered) == 2


def test_stuck_loop_is_stopped(settings):
    handler = make_handler(researcher=lambda messages: tool_response("web_search", {"query": "same"}, "x"))
    result = pipeline(settings, handler).run("topic")
    assert result.failures["stuck_loop"] == 2  # both sub-questions get stuck
    assert result.status == "failed"


def test_iteration_limit_per_subquestion(settings):
    s = settings.model_copy(update={"max_iterations_per_subquestion": 2})

    def never_finishes(messages):
        n = sum(1 for m in messages if m["role"] == "assistant")
        return tool_response("web_search", {"query": f"q{n}"}, f"t{n}")

    result = pipeline(s, make_handler(researcher=never_finishes, plan_questions=1)).run("topic")
    assert result.failures["iteration_limit"] == 1
    assert result.status == "failed"


def test_total_iteration_cap_stops_research_but_keeps_findings(settings):
    # 3 steps per sub-question (search, fetch, notes); cap of 4 lets sub-question 1 finish
    # and cuts sub-question 2 off.
    s = settings.model_copy(update={"max_total_iterations": 4})
    result = pipeline(s, make_handler()).run("topic")

    assert result.status == "partial"
    assert result.failures["iteration_limit"] == 1
    assert len(result.unanswered) == 1
    assert "Research stopped early" in result.report_markdown


def test_search_empty_and_fetch_failure_are_categorised(settings, monkeypatch):
    monkeypatch.setattr(SearchTool, "_search_ddgs", lambda self, q, n: [])

    def researcher(messages):
        n = sum(1 for m in messages if m["role"] == "assistant")
        if n == 0:
            return tool_response("web_search", {"query": "nothing"}, "a")
        if n == 1:
            return tool_response("web_fetch", {"url": "http://127.0.0.1/blocked"}, "b")
        return text_response("I could not find anything.")

    monkeypatch.setattr("agent.tools.fetch.is_safe_url", lambda url: "127.0.0.1" not in url)
    result = pipeline(settings, make_handler(researcher=researcher, plan_questions=1)).run("topic")
    assert result.failures["search_empty"] == 1
    assert result.failures["fetch_failed"] == 1


def test_planner_bad_json_fails_cleanly(settings):
    result = pipeline(settings, lambda *_: text_response("I refuse to output JSON")).run("topic")
    assert result.status == "failed"
    assert result.failures["bad_json"] == settings.json_max_retries + 1
    assert "BadJSONError" in result.error
    assert result.report_markdown.startswith("# Research failed")


def test_report_drafting_failure_falls_back_to_plain_summary(settings):
    base = make_handler()

    def handler(system, messages, tools):
        if "report writer" in system.lower():
            return text_response("not json")
        return base(system, messages, tools)

    result = pipeline(settings, handler).run("topic")
    assert result.status == "partial"
    assert "plain summary" in result.report_markdown
    assert "The market grew 12% in 2025." in result.report_markdown


def test_email_delivery_success_and_failure(settings, monkeypatch):
    s = settings.model_copy(update={"smtp_host": "smtp.test", "smtp_from": "bot@test.com"})
    sent = []
    monkeypatch.setattr(EmailTool, "_send", lambda self, msg: sent.append(msg))

    ok = pipeline(s, make_handler()).run("topic", email_to="me@example.com")
    assert ok.emailed is True and len(sent) == 1
    assert sent[0]["To"] == "me@example.com"

    unconfigured = pipeline(settings, make_handler()).run("topic", email_to="me@example.com")
    assert unconfigured.emailed is False
    assert unconfigured.failures["email_failed"] == 1
    assert unconfigured.status == "success"  # a failed email doesn't invalidate the report


def test_run_never_raises_on_unexpected_errors(settings):
    def explode(*_):
        raise RuntimeError("provider exploded")

    result = pipeline(settings, explode).run("topic")
    assert result.status == "failed"
    assert "provider exploded" in result.error
