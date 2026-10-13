"""Safe arithmetic evaluator (AST based; no eval)."""

from __future__ import annotations

import ast
import operator
from typing import Any

from pydantic import BaseModel, Field

from ..exceptions import ToolError
from .base import Tool, ToolResult

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent too large")
            try:
                return _BINARY[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ToolError("division by zero") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ToolError("only numbers and + - * / // % ** ( ) are allowed")

    return ev(tree)


class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=200, description="e.g. (120 - 90) / 90 * 100")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression exactly. Use it instead of doing maths mentally."
    input_model = CalculatorInput

    def run(self, args: CalculatorInput) -> ToolResult:
        value = safe_eval(args.expression)
        return ToolResult(content=f"{args.expression} = {value}", data={"value": value})
