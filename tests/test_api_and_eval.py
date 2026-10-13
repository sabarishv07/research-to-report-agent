import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.models import RunResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from run_eval import summarize  # noqa: E402


def result(status="success", cost=0.1, latency=10.0, failures=None):
    return RunResult(
        run_id="abcd1234", topic="t", status=status, cost_usd=cost, latency_s=latency, failures=failures or {}
    )


def test_eval_summary_math():
    summary = summarize(
        [
            result("success", 0.10, 10, {}),
            result("success", 0.20, 20, {"fetch_failed": 1}),
            result("partial", 0.30, 30, {"fetch_failed": 2, "stuck_loop": 1}),
            result("failed", 0.40, 40, {"bad_json": 1}),
        ]
    )
    assert summary["runs"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["partial_rate"] == 0.25
    assert summary["failed_rate"] == 0.25
    assert summary["avg_cost_usd"] == pytest.approx(0.25)
    assert summary["avg_latency_s"] == pytest.approx(25.0)
    assert summary["failure_categories"]["fetch_failed"] == 3


def test_eval_summary_empty():
    assert summarize([]) == {"runs": 0}


@pytest.fixture
def client(settings, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(main.app), main, settings


def test_health(client):
    c, _, _ = client
    assert c.get("/health").json() == {"status": "ok"}


def test_create_run_uses_pipeline(client, monkeypatch):
    c, main, _ = client

    class StubPipeline:
        def __init__(self, settings):
            pass

        def run(self, topic, email_to=None):
            return result()

    monkeypatch.setattr(main, "Pipeline", StubPipeline)
    resp = c.post("/runs", json={"topic": "AI chips"})
    assert resp.status_code == 200 and resp.json()["status"] == "success"


def test_create_run_validation(client):
    c, _, _ = client
    assert c.post("/runs", json={"topic": "x"}).status_code == 422
    assert c.post("/runs", json={"topic": "valid topic", "email_to": "nope"}).status_code == 422


def test_create_run_requires_api_key(client, monkeypatch):
    c, main, settings = client
    monkeypatch.setattr(main, "get_settings", lambda: settings.model_copy(update={"anthropic_api_key": ""}))
    assert c.post("/runs", json={"topic": "AI chips"}).status_code == 503


def test_report_and_trace_endpoints_and_id_validation(client):
    c, _, settings = client
    Path(settings.reports_dir).mkdir(parents=True)
    Path(settings.traces_dir).mkdir(parents=True)
    (Path(settings.reports_dir) / "abcd1234.md").write_text("# Hello", encoding="utf-8")
    (Path(settings.traces_dir) / "20260101-000000_abcd1234.jsonl").write_text(
        json.dumps({"type": "run_start", "t": 0}) + "\n", encoding="utf-8"
    )

    assert c.get("/runs/abcd1234/report").text == "# Hello"
    assert c.get("/runs/abcd1234/trace").json()[0]["type"] == "run_start"
    assert c.get("/runs/ffffffff/report").status_code == 404
    assert c.get("/runs/..%2F..%2Fetc/report").status_code in (400, 404)
    assert c.get("/runs/NOT-AN-ID/trace").status_code == 400
