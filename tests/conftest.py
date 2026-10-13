import pytest

from agent.config import Settings


@pytest.fixture
def settings(tmp_path):
    """Isolated settings: no .env, temp output folders, generous budget."""
    return Settings(
        _env_file=None,
        anthropic_api_key="test-key",
        tavily_api_key="",
        traces_dir=str(tmp_path / "traces"),
        reports_dir=str(tmp_path / "reports"),
        max_cost_usd=5.0,
    )
