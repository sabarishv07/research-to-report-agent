"""Web search tool: Tavily first (if a key is set), DuckDuckGo as the fallback."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..config import Settings
from ..exceptions import ToolError
from ._http import http_retry
from .base import Tool, ToolResult


class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="Search query")
    max_results: int = Field(default=5, ge=1, le=10)


class SearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web. Returns titles, URLs and short snippets. Snippets are not citable: "
        "use web_fetch to read a page before relying on it."
    )
    input_model = SearchInput

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ---- providers (each returns [{"title","url","snippet"}]) -------------------
    @http_retry
    def _search_tavily(self, query: str, n: int) -> list[dict[str, str]]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.settings.tavily_api_key}"},
            json={"query": query, "max_results": n},
            timeout=self.settings.tool_timeout_s,
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in resp.json().get("results", [])
            if r.get("url")
        ]

    def _search_ddgs(self, query: str, n: int) -> list[dict[str, str]]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise ToolError("ddgs package is not installed") from exc
        raw = DDGS(timeout=int(self.settings.tool_timeout_s)).text(query, max_results=n)
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in (raw or [])
            if r.get("href")
        ]

    def _providers(self) -> list[tuple[str, Any]]:
        providers: list[tuple[str, Any]] = []
        if self.settings.tavily_api_key:
            providers.append(("tavily", self._search_tavily))
        providers.append(("ddgs", self._search_ddgs))
        return providers

    # ---- tool entry point ---------------------------------------------------------
    def run(self, args: SearchInput) -> ToolResult:
        providers = self._providers()
        errors: list[str] = []
        for name, search in providers:
            try:
                results = search(args.query, args.max_results)
            except Exception as exc:  # noqa: BLE001 - fall through to the next provider
                errors.append(f"{name}: {exc}")
                continue
            if results:
                lines = [
                    f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet'][:300]}"
                    for i, r in enumerate(results, 1)
                ]
                return ToolResult(
                    content="\n".join(lines),
                    data={
                        "provider": name,
                        "results": [{"title": r["title"], "url": r["url"]} for r in results],
                    },
                )
        if len(errors) == len(providers):
            raise ToolError("all search providers failed (" + "; ".join(errors) + ")")
        return ToolResult(
            content="No results found. Try a different or broader query.", data={"empty": True}
        )
