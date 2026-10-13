"""Page fetch tool: downloads a page, extracts readable text, stores it for verification.

Includes SSRF protection: only http(s), and never private/loopback/link-local addresses
(checked on every redirect hop).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from ..config import Settings
from ..exceptions import ToolError
from ._http import http_retry
from .base import Tool, ToolResult
from .sources import SourceStore

MAX_REDIRECTS = 5
_NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]


def is_safe_url(url: str) -> bool:
    """True only for http(s) URLs whose host resolves exclusively to public addresses.

    Note: DNS can change between this check and the request (rebinding). For a hardened
    deployment, pin the resolved IP or run the fetcher in an egress-restricted network.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, readable_text) from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return title, text


class FetchInput(BaseModel):
    url: str = Field(min_length=8, max_length=2000, description="Full http(s) URL to read")


class FetchTool(Tool):
    name = "web_fetch"
    description = (
        "Download a web page and return its readable text. Only pages fetched with this tool "
        "can be cited in the final report."
    )
    input_model = FetchInput

    def __init__(self, settings: Settings, sources: SourceStore) -> None:
        self.settings = settings
        self.sources = sources

    @http_retry
    def _download(self, url: str) -> tuple[str, str, str]:
        """Return (final_url, content_type, text). Follows redirects manually so each hop is checked."""
        s = self.settings
        headers = {"User-Agent": s.user_agent, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"}
        with httpx.Client(timeout=s.tool_timeout_s, headers=headers, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                if not is_safe_url(url):
                    raise ToolError("URL blocked: not http(s) or points to a private/internal address")
                with client.stream("GET", url) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise ToolError("redirect without a Location header")
                        url = urljoin(url, location)
                        continue
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "").lower()
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in resp.iter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size >= s.fetch_max_bytes:
                            break
                    raw = b"".join(chunks)[: s.fetch_max_bytes]
                    return str(resp.url), ctype, raw.decode(resp.encoding or "utf-8", errors="replace")
        raise ToolError("too many redirects")

    def run(self, args: FetchInput) -> ToolResult:
        cached = self.sources.get(args.url)
        if cached is not None:  # same page requested again (e.g. another sub-question)
            return self._result(cached.url, cached.title, cached.text, cached=True)

        if not is_safe_url(args.url):
            raise ToolError("URL blocked: not http(s) or points to a private/internal address")
        try:
            final_url, ctype, body = self._download(args.url)
        except httpx.HTTPStatusError as exc:
            raise ToolError(f"HTTP {exc.response.status_code} for {args.url}") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"network error: {type(exc).__name__}: {exc}") from exc

        if "html" in ctype or ctype == "":
            title, text = extract_text(body)
        elif ctype.startswith("text/"):
            title, text = "", body.strip()
        else:
            raise ToolError(f"unsupported content type '{ctype}' (only HTML/text pages are readable)")
        if len(text) < 100:
            raise ToolError("page has no readable text (likely JavaScript-rendered or blocked)")

        source = self.sources.add(final_url, title, text, args.url)
        return self._result(source.url, source.title, source.text, cached=False)

    def _result(self, url: str, title: str, text: str, cached: bool) -> ToolResult:
        limit = self.settings.fetch_context_chars
        shown = text[:limit]
        suffix = "\n\n[truncated]" if len(text) > limit else ""
        return ToolResult(
            content=f"Source: {url}\nTitle: {title}\n\n{shown}{suffix}",
            data={"url": url, "chars": len(text), "cached": cached},
        )
