"""Store of every page fetched during a run.

The verifier only accepts citations whose URL is in this store, which is what stops the
model from citing pages it never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """Canonical form for comparing URLs: lowercase scheme/host, no fragment, no trailing slash."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


@dataclass
class Source:
    url: str
    title: str
    text: str


class SourceStore:
    def __init__(self, max_chars: int = 40_000) -> None:
        self.max_chars = max_chars
        self._items: dict[str, Source] = {}

    def add(self, url: str, title: str, text: str, *aliases: str) -> Source:
        source = Source(url=url, title=title, text=text[: self.max_chars])
        for u in (url, *aliases):
            self._items[normalize_url(u)] = source
        return source

    def get(self, url: str) -> Source | None:
        return self._items.get(normalize_url(url))

    def has(self, url: str) -> bool:
        return normalize_url(url) in self._items

    def urls(self) -> list[str]:
        seen: list[str] = []
        for src in self._items.values():
            if src.url not in seen:
                seen.append(src.url)
        return seen
