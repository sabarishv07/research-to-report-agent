"""Settings, loaded from environment variables / .env. No secrets in code."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    # Prices are used only for the cost ceiling. Check current pricing and override via env.
    price_input_per_mtok: float = 3.0
    price_output_per_mtok: float = 15.0
    llm_max_tokens: int = 2000
    llm_max_retries: int = 3  # transient API errors (rate limit, 5xx, network)
    json_max_retries: int = 2  # re-asks when the model returns malformed JSON

    # --- Search ---
    tavily_api_key: str = ""

    # --- Email ---
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_starttls: bool = True

    # --- Guardrails ---
    max_subquestions: int = 5
    max_iterations_per_subquestion: int = 6
    max_total_iterations: int = 30
    max_cost_usd: float = 0.75
    max_stuck_repeats: int = 2

    # --- Tools ---
    tool_timeout_s: float = 20.0
    fetch_context_chars: int = 8000  # how much page text the model sees
    fetch_max_bytes: int = 1_500_000
    source_store_chars: int = 40_000  # how much page text is kept for verification
    verify_chars: int = 12_000  # how much page text the verifier sees
    user_agent: str = "ResearchReportAgent/0.1 (portfolio project)"

    # --- Output ---
    traces_dir: str = "traces"
    reports_dir: str = "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
