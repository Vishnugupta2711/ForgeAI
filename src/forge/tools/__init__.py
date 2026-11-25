"""Tool registry and built-in declarative tools."""

from forge.tools.base import BaseTool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from forge.tools.calculator import CalculatorArgs, CalculatorTool, safe_eval_arithmetic
from forge.tools.csv_lookup import CsvFilterOp, CsvLookupArgs, CsvLookupTool
from forge.tools.current_date import CurrentDateArgs, CurrentDateTool
from forge.tools.http_api import ALLOWED_HTTP_HOSTS, HttpApiArgs, HttpApiTool


def default_registry() -> ToolRegistry:
    """Create a ToolRegistry populated with the 4 standard ForgeAI tools."""
    registry = ToolRegistry()
    registry.register(CsvLookupTool())
    registry.register(CalculatorTool())
    registry.register(HttpApiTool())
    registry.register(CurrentDateTool())
    return registry


__all__ = [
    "ALLOWED_HTTP_HOSTS",
    "BaseTool",
    "CalculatorArgs",
    "CalculatorTool",
    "CsvFilterOp",
    "CsvLookupArgs",
    "CsvLookupTool",
    "CurrentDateArgs",
    "CurrentDateTool",
    "HttpApiArgs",
    "HttpApiTool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_registry",
    "safe_eval_arithmetic",
]
