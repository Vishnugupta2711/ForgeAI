"""Keyless public HTTP API tool (Open-Meteo) with host allowlist, cache, and offline fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, model_validator

from forge.tools.base import BaseTool, ToolContext, ToolResult

ALLOWED_HTTP_HOSTS = frozenset({"api.open-meteo.com", "geocoding-api.open-meteo.com"})

_ENDPOINT_URLS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "geocoding": "https://geocoding-api.open-meteo.com/v1/search",
}


class HttpApiArgs(BaseModel):
    """Arguments for querying the keyless Open-Meteo weather/geocoding API."""

    endpoint: Literal["forecast", "geocoding"] = "forecast"
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    city: str | None = Field(default=None, min_length=1, max_length=100)
    forecast_days: int = Field(default=3, ge=1, le=7)
    url_override: str | None = Field(
        default=None,
        description="Optional full URL (must still match the strict host allowlist).",
    )

    @model_validator(mode="after")
    def _validate_params(self) -> HttpApiArgs:
        if self.endpoint == "forecast" and (self.latitude is None or self.longitude is None):
            if not self.city:
                raise ValueError(
                    "Forecast requires latitude and longitude (or city for geocoding)."
                )
        if self.endpoint == "geocoding" and not self.city:
            raise ValueError("Geocoding requires 'city'.")
        return self


def _validate_allowed_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS requests are allowed (got '{parsed.scheme}').")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HTTP_HOSTS:
        raise ValueError(
            f"Host '{host}' is not in the allowlist {sorted(ALLOWED_HTTP_HOSTS)}."
        )
    return url


def _cache_file(cache_dir: Path, url: str, params: dict[str, Any]) -> Path:
    canonical = json.dumps({"url": url, "params": params}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


class HttpApiTool(BaseTool):
    """Keyless Open-Meteo weather & geocoding lookup with allowlist, caching, and offline mode."""

    name = "http_api"
    description = (
        "Fetch live weather forecast or geocoding data from the keyless Open-Meteo API "
        "(host-allowlisted, cached, and offline-fixture testable)."
    )
    capability_tags = ["weather", "http", "forecast", "external_api"]
    args_model = HttpApiArgs

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def _execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, HttpApiArgs)
        tool_cfg = ctx.tool_configs.get(self.name, {})
        offline_only = bool(ctx.offline_only or tool_cfg.get("offline_only", False))

        url = args.url_override or _ENDPOINT_URLS[args.endpoint]
        try:
            _validate_allowed_url(url)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))

        if args.endpoint == "geocoding":
            params: dict[str, Any] = {"name": args.city, "count": 3, "language": "en"}
        else:
            if args.latitude is not None and args.longitude is not None:
                params = {
                    "latitude": args.latitude,
                    "longitude": args.longitude,
                    "current_weather": "true",
                    "forecast_days": args.forecast_days,
                }
            else:
                params = {
                    "city": args.city,
                    "current_weather": "true",
                    "forecast_days": args.forecast_days,
                }

        # 1. Offline fixture mode (in ToolContext or tool_configs)
        fixture_key = (
            f"geocoding:{args.city}"
            if args.endpoint == "geocoding"
            else (
                f"forecast:{args.latitude},{args.longitude}"
                if args.latitude is not None and args.longitude is not None
                else f"forecast:{args.city}"
            )
        )
        fixtures = dict(ctx.http_fixtures)
        if isinstance(tool_cfg.get("fixtures"), dict):
            fixtures.update(tool_cfg["fixtures"])

        if fixture_key in fixtures:
            return ToolResult(
                ok=True,
                data=fixtures[fixture_key],
                metadata={"source": "offline_fixture", "key": fixture_key},
            )
        if args.endpoint in fixtures:
            return ToolResult(
                ok=True,
                data=fixtures[args.endpoint],
                metadata={"source": "offline_fixture", "key": args.endpoint},
            )

        # 2. Disk response cache
        cache_dir = ctx.cache_dir or Path(".cache/tools/http_api")
        c_path = _cache_file(cache_dir, url, params)
        if c_path.is_file():
            try:
                cached = json.loads(c_path.read_text(encoding="utf-8"))
                return ToolResult(
                    ok=True,
                    data=cached,
                    metadata={"source": "cache", "url": url},
                )
            except (OSError, json.JSONDecodeError):
                pass

        if offline_only:
            return ToolResult(
                ok=False,
                error=f"http_api is in offline_only mode and no fixture matched '{fixture_key}'.",
            )

        # 3. Live HTTP GET
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"HTTP request failed: {exc}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        c_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return ToolResult(
            ok=True,
            data=payload,
            metadata={"source": "live", "url": url},
        )
