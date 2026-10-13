import pytest

from agent.exceptions import BudgetExceededError, IterationLimitError, StuckLoopError
from agent.guardrails import Guardrails
from agent.tracing import Tracer


def make(settings, **overrides):
    s = settings.model_copy(update=overrides)
    tracer = Tracer("t1")
    return Guardrails(s, tracer), tracer


def test_cost_is_accumulated(settings):
    g, _ = make(settings, price_input_per_mtok=1.0, price_output_per_mtok=2.0)
    cost = g.record_usage(1_000_000, 500_000)
    assert cost == pytest.approx(2.0)
    assert g.cost_usd == pytest.approx(2.0)
    assert g.llm_calls == 1


def test_budget_exceeded_raises_and_is_recorded(settings):
    g, tracer = make(settings, max_cost_usd=0.001)
    with pytest.raises(BudgetExceededError):
        g.record_usage(1_000_000, 0)
    assert g.budget_exceeded
    assert tracer.failures["budget_exceeded"] == 1


def test_iteration_cap(settings):
    g, tracer = make(settings, max_total_iterations=2)
    g.tick()
    g.tick()
    with pytest.raises(IterationLimitError):
        g.tick()
    assert tracer.failures["iteration_limit"] == 1


def test_stuck_detector_refuses_then_aborts(settings):
    g, tracer = make(settings, max_stuck_repeats=2)
    args = {"query": "same"}
    assert g.register_call("web_search", args) is False  # first call is fine
    assert g.register_call("web_search", args) is True  # first repeat: refused, model is told
    with pytest.raises(StuckLoopError):  # second repeat: abort
        g.register_call("web_search", args)
    assert tracer.failures["stuck_loop"] == 1


def test_different_args_are_not_repeats(settings):
    g, _ = make(settings)
    assert g.register_call("web_search", {"query": "a"}) is False
    assert g.register_call("web_search", {"query": "b"}) is False
    assert g.register_call("web_fetch", {"query": "a"}) is False


def test_repeat_tracking_resets_per_subquestion(settings):
    g, _ = make(settings)
    assert g.register_call("web_search", {"query": "a"}) is False
    g.begin_subquestion()
    assert g.register_call("web_search", {"query": "a"}) is False
