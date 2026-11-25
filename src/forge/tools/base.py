"""Base classes, context, and registry for ForgeAI declarative tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from forge.ingest.models import Document


class ToolSpec(BaseModel):
    """Declarative specification describing a registered tool."""

    name: str
    description: str
    args_schema: dict[str, Any]
    capability_tags: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Uniform response returned by every tool invocation."""

    ok: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolContext(BaseModel):
    """Runtime execution context passed to tools (tables, cache, test hooks)."""

    tables: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    clock_now: datetime | None = None
    http_fixtures: dict[str, Any] = Field(default_factory=dict)
    cache_dir: Path | None = None
    offline_only: bool = False
    tool_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        *,
        cache_dir: Path | None = None,
        offline_only: bool = False,
        tool_configs: dict[str, dict[str, Any]] | None = None,
    ) -> ToolContext:
        """Build a ToolContext populated with CSV tables from ingested Documents."""
        tables: dict[str, list[dict[str, str]]] = {}
        for doc in documents:
            if doc.type == "csv" and isinstance(doc.metadata.get("rows"), list):
                tables[doc.doc_id] = list(doc.metadata["rows"])
        return cls(
            tables=tables,
            cache_dir=cache_dir,
            offline_only=offline_only,
            tool_configs=tool_configs or {},
        )


class BaseTool(ABC):
    """Abstract base class for safely typed ForgeAI tools."""

    name: str
    description: str
    capability_tags: list[str]
    args_model: type[BaseModel]

    @property
    def spec(self) -> ToolSpec:
        """Return the declarative ToolSpec for this tool."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema=self.args_model.model_json_schema(),
            capability_tags=list(self.capability_tags),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext | None = None) -> ToolResult:
        """Validate arguments against args_model and invoke _execute safely."""
        context = ctx or ToolContext()
        try:
            parsed = self.args_model.model_validate(args or {})
        except ValidationError as exc:
            return ToolResult(ok=False, error=f"Invalid arguments for '{self.name}': {exc}")

        try:
            return self._execute(parsed, context)
        except Exception as exc:
            return ToolResult(ok=False, error=f"Tool '{self.name}' failed: {exc}")

    @abstractmethod
    def _execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """Execute validated tool logic."""


class ToolRegistry:
    """Registry holding available tools and enforcing attachment allowlists."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance by its unique name."""
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        """Return True if a tool is registered."""
        return name in self._tools

    def get(self, name: str) -> BaseTool:
        """Look up a registered tool or raise KeyError."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}'")
        return self._tools[name]

    def list_specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        """Return ToolSpecs for all registered tools or a requested subset."""
        if names is None:
            return [t.spec for t in self._tools.values()]
        return [self._tools[n].spec for n in names if n in self._tools]

    def call(
        self,
        name: str,
        args: dict[str, Any],
        ctx: ToolContext | None = None,
        *,
        allowed_tools: set[str] | None = None,
    ) -> ToolResult:
        """Execute a tool if registered and permitted by allowed_tools."""
        if allowed_tools is not None and name not in allowed_tools:
            return ToolResult(
                ok=False,
                error=f"Tool '{name}' is not attached in the agent definition.",
            )
        if name not in self._tools:
            return ToolResult(ok=False, error=f"Unknown tool '{name}'.")
        return self._tools[name].run(args, ctx)
