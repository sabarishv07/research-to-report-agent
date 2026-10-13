"""Tool interface and registry. Every tool the model can call goes through here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationError

from ..exceptions import ToolError


class ToolResult(BaseModel):
    ok: bool = True
    content: str
    data: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    def schema(self) -> dict[str, Any]:
        """Tool definition in the Anthropic tool-use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }

    @abstractmethod
    def run(self, args: Any) -> ToolResult:
        """Do the work. Raise ToolError for expected failures."""

    def execute_safely(self, args: Any) -> ToolResult:
        """Run the tool and turn any failure into ToolResult(ok=False) instead of raising."""
        try:
            return self.run(args)
        except ToolError as exc:
            return ToolResult(ok=False, content=f"{self.name} failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the run
            return ToolResult(
                ok=False, content=f"{self.name} failed unexpectedly: {type(exc).__name__}: {exc}"
            )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, raw_args: dict[str, Any]) -> ToolResult:
        """Validate arguments and run a tool. Never raises: failures come back as
        ToolResult(ok=False) so the model can read the error and adapt."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False, content=f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}"
            )
        try:
            args = tool.input_model.model_validate(raw_args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            return ToolResult(ok=False, content=f"Invalid arguments for {name}: {problems}")
        return tool.execute_safely(args)
