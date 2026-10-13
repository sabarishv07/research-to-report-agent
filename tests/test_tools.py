import pytest

from agent.exceptions import ToolError
from agent.tools import (
    CalculatorTool,
    EmailTool,
    FetchTool,
    SearchTool,
    SendEmailInput,
    SourceStore,
    ToolRegistry,
    default_registry,
)
from agent.tools.calculator import safe_eval
from agent.tools.fetch import extract_text, is_safe_url
from agent.tools.sources import normalize_url


# ---- calculator ------------------------------------------------------------------
def test_calculator_basic():
    assert safe_eval("(120 - 90) / 90 * 100") == pytest.approx(33.3333, rel=1e-3)
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("-3 + 5") == 2


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "abs(-1)", "1 +", "2 ** 1000", "1/0", "True + 1"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        safe_eval(expr)


# ---- registry --------------------------------------------------------------------
def test_registry_validation_errors_are_returned_not_raised(settings):
    registry = default_registry(settings, SourceStore())
    result = registry.execute("web_search", {"query": ""})
    assert not result.ok
    assert "Invalid arguments" in result.content


def test_registry_unknown_tool():
    result = ToolRegistry().execute("nope", {})
    assert not result.ok and "Unknown tool" in result.content


def test_registry_exposes_schemas_without_email(settings):
    registry = default_registry(settings, SourceStore())
    assert set(registry.names()) == {"web_search", "web_fetch", "calculator"}
    for schema in registry.schemas():
        assert {"name", "description", "input_schema"} <= schema.keys()


def test_registry_converts_tool_bugs_to_failed_results(settings):
    class Boom(CalculatorTool):
        name = "boom"

        def run(self, args):
            raise RuntimeError("kaput")

    registry = ToolRegistry()
    registry.register(Boom())
    result = registry.execute("boom", {"expression": "1+1"})
    assert not result.ok and "kaput" in result.content


# ---- sources ---------------------------------------------------------------------
def test_normalize_url():
    assert normalize_url("HTTPS://Example.com/a/#frag") == "https://example.com/a"
    assert normalize_url("https://example.com/a?x=1") == "https://example.com/a?x=1"


def test_source_store_lookup_with_aliases():
    store = SourceStore(max_chars=10)
    store.add("https://example.com/final", "T", "x" * 50, "https://short.link/abc")
    assert store.has("https://short.link/abc/")
    assert store.get("https://example.com/final").text == "x" * 10  # truncated
    assert store.urls() == ["https://example.com/final"]
    assert not store.has("https://other.com")


# ---- fetch -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "not a url",
    ],
)
def test_is_safe_url_blocks_internal_and_non_http(url):
    assert is_safe_url(url) is False


def test_is_safe_url_allows_public_ip_literal():
    assert is_safe_url("http://93.184.216.34/") is True


def test_extract_text_strips_noise():
    html = """<html><head><title>My Page</title><style>.x{}</style></head>
    <body><nav>MENU</nav><script>var secret=1;</script>
    <p>Real   content here.</p><footer>FOOT</footer></body></html>"""
    title, text = extract_text(html)
    assert title == "My Page"
    assert "Real content here." in text
    for junk in ("MENU", "secret", "FOOT", ".x{}"):
        assert junk not in text


def test_fetch_tool_stores_source_and_caches(settings, monkeypatch):
    monkeypatch.setattr("agent.tools.fetch.is_safe_url", lambda url: True)
    calls = []
    body = "<html><title>T</title><body><p>" + "useful sentence. " * 20 + "</p></body></html>"

    def fake_download(self, url):
        calls.append(url)
        return "https://example.com/final", "text/html; charset=utf-8", body

    monkeypatch.setattr(FetchTool, "_download", fake_download)
    store = SourceStore()
    registry = ToolRegistry()
    registry.register(FetchTool(settings, store))

    first = registry.execute("web_fetch", {"url": "https://example.com/start"})
    assert first.ok and first.data["url"] == "https://example.com/final"
    assert store.has("https://example.com/final") and store.has("https://example.com/start")

    second = registry.execute("web_fetch", {"url": "https://example.com/start"})
    assert second.ok and second.data["cached"] is True
    assert len(calls) == 1


def test_fetch_tool_rejects_blocked_url_and_bad_content(settings, monkeypatch):
    registry = ToolRegistry()
    registry.register(FetchTool(settings, SourceStore()))
    blocked = registry.execute("web_fetch", {"url": "http://127.0.0.1:8080/secret"})
    assert not blocked.ok and "blocked" in blocked.content

    monkeypatch.setattr("agent.tools.fetch.is_safe_url", lambda url: True)
    monkeypatch.setattr(FetchTool, "_download", lambda self, url: (url, "application/pdf", "%PDF"))
    pdf = registry.execute("web_fetch", {"url": "https://example.com/a.pdf"})
    assert not pdf.ok and "unsupported content type" in pdf.content

    monkeypatch.setattr(FetchTool, "_download", lambda self, url: (url, "text/html", "<p>tiny</p>"))
    tiny = registry.execute("web_fetch", {"url": "https://example.com/tiny"})
    assert not tiny.ok and "no readable text" in tiny.content


# ---- search ----------------------------------------------------------------------
RESULT = [{"title": "T", "url": "https://example.com/a", "snippet": "s"}]


def test_search_falls_back_when_primary_provider_fails(settings, monkeypatch):
    s = settings.model_copy(update={"tavily_api_key": "key"})

    def tavily_down(self, q, n):
        raise ToolError("tavily down")

    monkeypatch.setattr(SearchTool, "_search_tavily", tavily_down)
    monkeypatch.setattr(SearchTool, "_search_ddgs", lambda self, q, n: RESULT)
    result = SearchTool(s).execute_safely(SearchTool.input_model(query="hello"))
    assert result.ok and result.data["provider"] == "ddgs"


def test_search_empty_results_are_flagged(settings, monkeypatch):
    monkeypatch.setattr(SearchTool, "_search_ddgs", lambda self, q, n: [])
    result = SearchTool(settings).execute_safely(SearchTool.input_model(query="hello"))
    assert result.ok and result.data == {"empty": True}


def test_search_all_providers_failing_is_an_error(settings, monkeypatch):
    def down(self, q, n):
        raise ToolError("down")

    monkeypatch.setattr(SearchTool, "_search_ddgs", down)
    result = SearchTool(settings).execute_safely(SearchTool.input_model(query="hello"))
    assert not result.ok and "all search providers failed" in result.content


# ---- email -----------------------------------------------------------------------
def email_settings(settings):
    return settings.model_copy(update={"smtp_host": "smtp.test", "smtp_from": "bot@test.com"})


def test_email_message_building_and_header_injection_guard(settings):
    tool = EmailTool(email_settings(settings))
    msg = tool.build_message(
        SendEmailInput(to="a@b.com", subject="Hi\nBcc: evil@x.com", body="# Report", attachment_name="r.md")
    )
    assert msg["To"] == "a@b.com" and msg["From"] == "bot@test.com"
    assert "\n" not in msg["Subject"]
    assert msg["Bcc"] is None
    assert any(part.get_filename() == "r.md" for part in msg.iter_attachments())


def test_email_sends_via_smtp(settings, monkeypatch):
    sent = []
    monkeypatch.setattr(EmailTool, "_send", lambda self, msg: sent.append(msg))
    result = EmailTool(email_settings(settings)).execute_safely(
        SendEmailInput(to="a@b.com", subject="S", body="B")
    )
    assert result.ok and len(sent) == 1


def test_email_requires_configuration_and_valid_address(settings):
    unconfigured = EmailTool(settings).execute_safely(SendEmailInput(to="a@b.com", subject="S", body="B"))
    assert not unconfigured.ok and "not configured" in unconfigured.content
    invalid = EmailTool(email_settings(settings)).execute_safely(
        SendEmailInput(to="not-an-email", subject="S", body="B")
    )
    assert not invalid.ok and "valid email" in invalid.content
