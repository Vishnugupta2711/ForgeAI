"""Current date and time tool using stdlib zoneinfo."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from forge.tools.base import BaseTool, ToolContext, ToolResult


class CurrentDateArgs(BaseModel):
    """Input arguments for current_date."""

    timezone: str = Field(
        default="UTC",
        description="IANA timezone name (e.g. 'UTC', 'Asia/Kolkata').",
    )


class CurrentDateTool(BaseTool):
    """Return the current date, ISO timestamp, and day of week."""

    name = "current_date"
    description = "Get the current calendar date, ISO timestamp, and weekday in a timezone."
    capability_tags = ["time", "date", "calendar"]
    args_model = CurrentDateArgs

    def _execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, CurrentDateArgs)
        try:
            tz = ZoneInfo(args.timezone)
        except ZoneInfoNotFoundError:
            return ToolResult(ok=False, error=f"Unknown IANA timezone: '{args.timezone}'.")

        base_now = ctx.clock_now or datetime.now(UTC)
        if base_now.tzinfo is None:
            base_now = base_now.replace(tzinfo=UTC)
        localized = base_now.astimezone(tz)
        return ToolResult(
            ok=True,
            data={
                "date": localized.date().isoformat(),
                "datetime_iso": localized.isoformat(),
                "weekday": localized.strftime("%A"),
                "timezone": args.timezone,
            },
        )
