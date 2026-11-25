"""Unit tests for forge.tools registry and all 4 built-in tools."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from forge.tools import (
    HttpApiTool,
    ToolContext,
    default_registry,
)


def test_registry_has_four_tools_with_specs() -> None:
    registry = default_registry()
    specs = registry.list_specs()
    names = {s.name for s in specs}
    assert names == {"csv_lookup", "calculator", "http_api", "current_date"}
    for spec in specs:
        assert spec.description
        assert "properties" in spec.args_schema
        assert spec.capability_tags


def test_registry_rejects_unattached_tool() -> None:
    registry = default_registry()
    res = registry.call(
        "http_api",
        {"endpoint": "geocoding", "city": "Oslo"},
        allowed_tools={"csv_lookup", "calculator"},
    )
    assert res.ok is False
    assert "not attached" in (res.error or "")


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(45 - 15) / 2", 15),
        ("sqrt(16) + round(10 / 3, 2)", 7.33),
        ("max(10, 25, 3) - min(2, 5)", 23),
    ],
)
def test_calculator_valid_expressions(expr: str, expected: float) -> None:
    registry = default_registry()
    res = registry.call("calculator", {"expression": expr})
    assert res.ok is True
    assert res.data["result"] == expected


@pytest.mark.parametrize(
    "adversarial_expr",
    [
        "__import__('os').system('echo pwned')",
        "eval('1 + 1')",
        "exec('import sys')",
        "open('/etc/passwd').read()",
        "(1).__class__.__bases__",
        "[x for x in range(10)]",
        "lambda x: x + 1",
        "9 ** 999999",
        "10 / 0",
        "sqrt(-4)",
        "'hello' + 'world'",
    ],
)
def test_calculator_rejects_adversarial_and_invalid_inputs(adversarial_expr: str) -> None:
    registry = default_registry()
    res = registry.call("calculator", {"expression": adversarial_expr})
    assert res.ok is False
    assert res.error


def test_csv_lookup_operations_and_adversarial_inputs() -> None:
    registry = default_registry()
    ctx = ToolContext(
        tables={
            "sensors.csv": [
                {"sensor_id": "SNS-101", "depth_m": "45", "active": "true"},
                {"sensor_id": "SNS-102", "depth_m": "0", "active": "true"},
                {"sensor_id": "SNS-103", "depth_m": "120", "active": "false"},
            ]
        }
    )

    # Filter + sort + select
    res = registry.call(
        "csv_lookup",
        {
            "filters": [{"column": "depth_m", "op": "gte", "value": 45}],
            "select": ["sensor_id", "depth_m"],
            "sort_by": "depth_m",
            "descending": True,
            "limit": 5,
        },
        ctx=ctx,
    )
    assert res.ok is True
    assert res.data["total_matched"] == 2
    assert res.data["rows"][0]["sensor_id"] == "SNS-103"

    # Dict-style shorthand filter coercion for small models
    short_res = registry.call(
        "csv_lookup",
        {"filters": {"active": "false"}},
        ctx=ctx,
    )
    assert short_res.ok is True
    assert len(short_res.data["rows"]) == 1

    # Adversarial inputs
    assert registry.call("csv_lookup", {"table": "../../etc/passwd"}, ctx=ctx).ok is False
    assert (
        registry.call(
            "csv_lookup",
            {"filters": [{"column": "id; DROP TABLE", "op": "eq", "value": "1"}]},
            ctx=ctx,
        ).ok
        is False
    )
    assert (
        registry.call(
            "csv_lookup",
            {"filters": [{"column": "sensor_id", "op": "exec", "value": "1"}]},
            ctx=ctx,
        ).ok
        is False
    )


def test_http_api_offline_fixture_cache_mock_and_allowlist(tmp_path: Path) -> None:
    registry = default_registry()

    # 1. Offline fixture mode (no network)
    ctx = ToolContext(
        offline_only=True,
        http_fixtures={"forecast:60.0,10.0": {"current_weather": {"temperature": 12.5}}},
        cache_dir=tmp_path / "http_cache",
    )
    ok_res = registry.call(
        "http_api",
        {"endpoint": "forecast", "latitude": 60.0, "longitude": 10.0},
        ctx=ctx,
    )
    assert ok_res.ok is True
    assert ok_res.data["current_weather"]["temperature"] == 12.5
    assert ok_res.metadata["source"] == "offline_fixture"

    # 2. Disallowed host / SSRF / scheme rejected
    bad_host = registry.call(
        "http_api",
        {
            "endpoint": "forecast",
            "latitude": 1.0,
            "longitude": 1.0,
            "url_override": "http://127.0.0.1:8080/secret",
        },
        ctx=ctx,
    )
    assert bad_host.ok is False
    assert "HTTPS" in (bad_host.error or "")

    bad_domain = registry.call(
        "http_api",
        {
            "endpoint": "forecast",
            "latitude": 1.0,
            "longitude": 1.0,
            "url_override": "https://evil.example.com/steal",
        },
        ctx=ctx,
    )
    assert bad_domain.ok is False
    assert "allowlist" in (bad_domain.error or "")

    # 3. MockTransport + cache hit
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"results": [{"name": "Bergen"}]})

    tool = HttpApiTool(transport=httpx.MockTransport(handler))
    live_ctx = ToolContext(cache_dir=tmp_path / "live_cache")
    r1 = tool.run({"endpoint": "geocoding", "city": "Bergen"}, live_ctx)
    r2 = tool.run({"endpoint": "geocoding", "city": "Bergen"}, live_ctx)
    assert r1.ok and r2.ok
    assert calls == 1
    assert r2.metadata["source"] == "cache"


def test_current_date_tool() -> None:
    registry = default_registry()
    fixed = datetime(2026, 6, 15, 10, 30, tzinfo=UTC)
    ctx = ToolContext(clock_now=fixed)

    utc_res = registry.call("current_date", {"timezone": "UTC"}, ctx=ctx)
    assert utc_res.ok is True
    assert utc_res.data["date"] == "2026-06-15"
    assert utc_res.data["weekday"] == "Monday"

    bad_tz = registry.call("current_date", {"timezone": "Invalid/Zone"}, ctx=ctx)
    assert bad_tz.ok is False
