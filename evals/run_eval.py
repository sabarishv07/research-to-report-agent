"""Evaluation harness: run the agent on a topic set and report reliability, cost and latency.

    python evals/run_eval.py                    # all topics
    python evals/run_eval.py --limit 3          # quick smoke run
    python evals/run_eval.py --max-cost 0.40

Results are saved to evals/results/<timestamp>.json. Paste the summary table into the README.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.config import get_settings  # noqa: E402
from agent.models import RunResult  # noqa: E402
from agent.pipeline import Pipeline  # noqa: E402


def summarize(results: list[RunResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"runs": 0}
    failures: Counter[str] = Counter()
    for r in results:
        failures.update(r.failures)
    by_status = Counter(r.status for r in results)
    return {
        "runs": n,
        "success_rate": round(by_status["success"] / n, 3),
        "partial_rate": round(by_status["partial"] / n, 3),
        "failed_rate": round(by_status["failed"] / n, 3),
        "avg_cost_usd": round(statistics.fmean(r.cost_usd for r in results), 4),
        "avg_latency_s": round(statistics.fmean(r.latency_s for r in results), 1),
        "p95_latency_s": round(sorted(r.latency_s for r in results)[max(0, int(n * 0.95) - 1)], 1),
        "failure_categories": dict(failures.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topics", default=str(ROOT / "evals" / "topics.json"))
    parser.add_argument("--limit", type=int, default=None, help="only run the first N topics")
    parser.add_argument("--max-cost", type=float, default=None, help="per-run cost ceiling in USD")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.anthropic_api_key:
        sys.exit("ANTHROPIC_API_KEY is not set (see .env.example).")
    if args.max_cost is not None:
        settings = settings.model_copy(update={"max_cost_usd": args.max_cost})

    topics: list[str] = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    if args.limit:
        topics = topics[: args.limit]

    pipeline = Pipeline(settings)
    results: list[RunResult] = []
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] {topic}", flush=True)
        result = pipeline.run(topic)
        results.append(result)
        print(
            f"    {result.status:<8} ${result.cost_usd:.4f}  {result.latency_s:>6.1f}s  "
            f"failures={result.failures or '-'}",
            flush=True,
        )

    summary = summarize(results)
    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": settings.llm_model,
                "summary": summary,
                "runs": [r.model_dump(exclude={"report_markdown"}) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key:>20}: {value}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
