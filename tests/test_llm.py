import pytest

from agent.exceptions import BadJSONError
from agent.guardrails import Guardrails
from agent.llm import extract_json
from agent.models import Plan
from agent.tracing import Tracer
from tests.fakes import FakeLLM, text_response

GOOD = {"topic": "t", "sub_questions": [{"id": 1, "question": "Q?"}]}


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_code_fence_and_prose():
    text = 'Sure! Here you go:\n```json\n{"a": [1, 2]}\n```\nHope that helps.'
    assert extract_json(text) == {"a": [1, 2]}


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here")


def _llm(settings, replies):
    tracer = Tracer("t")
    it = iter(replies)
    llm = FakeLLM(settings, tracer, Guardrails(settings, tracer), lambda *_: next(it))
    return llm, tracer


def test_json_call_recovers_from_malformed_output(settings):
    llm, tracer = _llm(settings, [text_response("not json"), text_response(GOOD)])
    plan = llm.json_call("sys", "prompt", Plan)
    assert plan.sub_questions[0].question == "Q?"
    assert llm.calls == 2
    assert tracer.failures["bad_json"] == 1


def test_json_call_gives_up_after_retries(settings):
    llm, tracer = _llm(settings, [text_response("nope")] * 10)
    with pytest.raises(BadJSONError):
        llm.json_call("sys", "prompt", Plan)
    assert llm.calls == settings.json_max_retries + 1
    assert tracer.failures["bad_json"] == settings.json_max_retries + 1


def test_json_call_rejects_schema_violations(settings):
    bad = {"topic": "t", "sub_questions": []}  # min_length=1 violated
    llm, _ = _llm(settings, [text_response(bad), text_response(GOOD)])
    assert llm.json_call("sys", "prompt", Plan).topic == "t"
