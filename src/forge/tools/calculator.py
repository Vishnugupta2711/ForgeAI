"""Safe AST-based arithmetic calculator tool (no eval/exec)."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from forge.tools.base import BaseTool, ToolContext, ToolResult

_MAX_EXPR_LENGTH = 256
_MAX_EXPONENT = 1000.0
_MAX_MAGNITUDE = 1e100

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}

_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a pure arithmetic expression using a strict AST allowlist."""
    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("Expression cannot be empty.")
    if len(cleaned) > _MAX_EXPR_LENGTH:
        raise ValueError(f"Expression exceeds max length of {_MAX_EXPR_LENGTH} characters.")

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic syntax: {exc.msg}") from exc

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported literal type: {type(node.value).__name__}")
            val = float(node.value)
            if not math.isfinite(val):
                raise ValueError("Non-finite literal is not allowed.")
            return val

        if isinstance(node, ast.Name):
            if node.id in _SAFE_CONSTANTS:
                return _SAFE_CONSTANTS[node.id]
            raise ValueError(f"Unknown or disallowed identifier: '{node.id}'")

        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
            return float(op_fn(_eval_node(node.operand)))

        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0.0:
                raise ValueError("Division or modulo by zero.")
            if isinstance(node.op, ast.Pow):
                if abs(right) > _MAX_EXPONENT:
                    raise ValueError("Exponent exceeds safe limit.")
                if abs(left) > 1e10 and abs(right) > 10:
                    raise ValueError("Power operands exceed safe magnitude.")
            result = float(op_fn(left, right))
            if not math.isfinite(result) or abs(result) > _MAX_MAGNITUDE:
                raise ValueError("Arithmetic result out of safe numeric bounds.")
            return result

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise ValueError("Function call not in arithmetic allowlist.")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported in calculator.")
            args = [_eval_node(arg) for arg in node.args]
            if not args:
                raise ValueError("Function call requires at least one argument.")
            fn = _SAFE_FUNCS[node.func.id]
            if node.func.id == "round":
                if len(args) == 1:
                    return float(round(args[0]))
                if len(args) == 2:
                    return float(round(args[0], int(args[1])))
                raise ValueError("round() accepts 1 or 2 arguments.")
            if node.func.id in {"abs", "sqrt"} and len(args) != 1:
                raise ValueError(f"{node.func.id}() requires exactly 1 argument.")
            if node.func.id == "sqrt" and args[0] < 0:
                raise ValueError("sqrt() domain error for negative input.")
            res = float(fn(*args) if node.func.id in {"min", "max"} else fn(args[0]))
            if not math.isfinite(res):
                raise ValueError("Function result is not finite.")
            return res

        raise ValueError(f"Disallowed syntax element: {type(node).__name__}")

    return _eval_node(tree)


class CalculatorArgs(BaseModel):
    """Input arguments for the AST-safe calculator tool."""

    expression: str = Field(
        min_length=1,
        description=(
            "Arithmetic expression using numbers, +, -, *, /, //, %, **, "
            "and abs/round/min/max/sqrt."
        ),
    )


class CalculatorTool(BaseTool):
    """Deterministic AST-based arithmetic calculator."""

    name = "calculator"
    description = (
        "Evaluate pure arithmetic expressions (+, -, *, /, //, %, **, abs, round, min, max, sqrt) "
        "without executing arbitrary code."
    )
    capability_tags = ["math", "arithmetic", "numeric"]
    args_model = CalculatorArgs

    def _execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, CalculatorArgs)
        value = safe_eval_arithmetic(args.expression)
        formatted: int | float = int(value) if value.is_integer() else round(value, 10)
        payload: dict[str, Any] = {"expression": args.expression, "result": formatted}
        return ToolResult(ok=True, data=payload)
