from ..config import Settings
from .base import Tool, ToolRegistry, ToolResult
from .calculator import CalculatorTool
from .email_tool import EmailTool, SendEmailInput
from .fetch import FetchTool
from .search import SearchTool
from .sources import SourceStore


def default_registry(settings: Settings, sources: SourceStore) -> ToolRegistry:
    """The tools the research model may call. (Email is intentionally absent.)"""
    registry = ToolRegistry()
    registry.register(SearchTool(settings))
    registry.register(FetchTool(settings, sources))
    registry.register(CalculatorTool())
    return registry


__all__ = [
    "CalculatorTool",
    "EmailTool",
    "FetchTool",
    "SearchTool",
    "SendEmailInput",
    "SourceStore",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
]
