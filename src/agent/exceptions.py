"""Custom exceptions. Explicit errors beat silent failures."""


class AgentError(Exception):
    """Base class for all agent errors."""


class ToolError(AgentError):
    """A tool could not complete its job (bad URL, provider down, ...)."""


class BudgetExceededError(AgentError):
    """The run's cost ceiling was exceeded."""


class IterationLimitError(AgentError):
    """The run's total iteration cap was reached."""


class StuckLoopError(AgentError):
    """The agent kept repeating the same tool call with the same arguments."""


class BadJSONError(AgentError):
    """The model failed to return valid JSON even after retries."""
