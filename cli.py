"""Command line interface.

    python cli.py run "AI chip market in 2026" --email you@example.com
    python cli.py trace traces/20261011-120000_ab12cd34.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from agent.config import get_settings
from agent.pipeline import Pipeline
from agent.tracing import format_trace, load_trace

app = typer.Typer(add_completion=False, help="Research-to-Report Agent")


@app.command()
def run(
    topic: str = typer.Argument(..., help="Topic to research"),
    email: Optional[str] = typer.Option(None, "--email", help="Email the report to this address"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Cost ceiling in USD for this run"),
) -> None:
    """Research a topic and print the report."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        typer.echo("ANTHROPIC_API_KEY is not set (see .env.example).", err=True)
        raise typer.Exit(code=2)
    if max_cost is not None:
        settings = settings.model_copy(update={"max_cost_usd": max_cost})

    result = Pipeline(settings).run(topic, email_to=email)

    typer.echo(result.report_markdown)
    typer.echo("\n---", err=True)
    typer.echo(
        f"status={result.status} cost=${result.cost_usd:.4f} latency={result.latency_s}s "
        f"llm_calls={result.llm_calls} failures={result.failures or '{}'}",
        err=True,
    )
    if result.error:
        typer.echo(f"error: {result.error}", err=True)
    typer.echo(f"report: {result.report_path}\ntrace:  {result.trace_path}", err=True)
    if email:
        typer.echo(f"emailed: {result.emailed}", err=True)
    if result.status == "failed":
        raise typer.Exit(code=1)


@app.command()
def trace(path: Path = typer.Argument(..., exists=True, readable=True, help="A .jsonl trace file")) -> None:
    """Print a readable view of a saved trace."""
    typer.echo(format_trace(load_trace(path)))


if __name__ == "__main__":
    app()
