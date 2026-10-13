"""Structured, per-run tracing. One JSON object per line (JSONL)."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FAILURE_CATEGORIES = (
    "search_empty",
    "fetch_failed",
    "bad_json",
    "budget_exceeded",
    "stuck_loop",
    "hallucinated_citation",
    "iteration_limit",
    "email_failed",
)


class Tracer:
    """Records every step of a run: plan, tool calls, results, checks, failures, LLM usage."""

    def __init__(self, run_id: str, directory: Path | None = None) -> None:
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.failures: Counter[str] = Counter()
        self.path: Path | None = None
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = directory / f"{stamp}_{run_id}.jsonl"

    def event(self, type: str, **data: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t": round(time.perf_counter() - self._t0, 3),
            "run_id": self.run_id,
            "type": type,
            **data,
        }
        with self._lock:
            self.events.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def failure(self, category: str, detail: str = "") -> None:
        """Record a categorised failure (used by the evaluation harness)."""
        with self._lock:
            self.failures[category] += 1
        self.event("failure", category=category, detail=detail)


def load_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def format_trace(events: list[dict[str, Any]], width: int = 220) -> str:
    """Human-readable one-line-per-event view of a trace."""
    lines = []
    for e in events:
        rest = {k: v for k, v in e.items() if k not in ("ts", "t", "run_id", "type")}
        detail = json.dumps(rest, ensure_ascii=False, default=str)
        if len(detail) > width:
            detail = detail[: width - 3] + "..."
        lines.append(f"[{e.get('t', 0.0):8.2f}s] {e.get('type', '?'):<16} {detail}")
    return "\n".join(lines)
