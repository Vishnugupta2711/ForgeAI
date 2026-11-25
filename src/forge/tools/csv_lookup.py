"""Safe tabular CSV lookup tool (filter, select, sort, limit; no eval/exec)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from forge.tools.base import BaseTool, ToolContext, ToolResult


class CsvFilterOp(BaseModel):
    """Declarative single-column filter predicate."""

    column: str = Field(min_length=1)
    op: Literal["eq", "ne", "contains", "gt", "gte", "lt", "lte"] = "eq"
    value: str | int | float | bool


class CsvLookupArgs(BaseModel):
    """Arguments for querying a loaded CSV table."""

    table: str | None = Field(
        default=None,
        description="CSV doc_id or filename; optional if only one table is loaded.",
    )
    filters: list[CsvFilterOp] = Field(
        default_factory=list,
        description="List of column filter predicates.",
    )
    select: list[str] = Field(
        default_factory=list,
        description="Columns to include in output rows (empty means all columns).",
    )
    sort_by: str | None = Field(default=None, description="Optional column name to sort by.")
    descending: bool = Field(default=False, description="Sort descending when sort_by is set.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum rows to return.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict_filters(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("filters"), dict):
            coerced = [
                {"column": k, "op": "eq", "value": v}
                for k, v in data["filters"].items()
            ]
            data = {**data, "filters": coerced}
        return data


def _try_float(val: Any) -> float | None:
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def _matches_filter(cell: str, f: CsvFilterOp) -> bool:
    target = str(f.value)
    if f.op == "contains":
        return target.lower() in cell.lower()

    left_num = _try_float(cell)
    right_num = _try_float(f.value)
    if left_num is not None and right_num is not None:
        if f.op == "eq":
            return left_num == right_num
        if f.op == "ne":
            return left_num != right_num
        if f.op == "gt":
            return left_num > right_num
        if f.op == "gte":
            return left_num >= right_num
        if f.op == "lt":
            return left_num < right_num
        if f.op == "lte":
            return left_num <= right_num

    left_s = cell.strip().lower()
    right_s = target.strip().lower()
    if f.op == "eq":
        return left_s == right_s
    if f.op == "ne":
        return left_s != right_s
    if f.op == "gt":
        return left_s > right_s
    if f.op == "gte":
        return left_s >= right_s
    if f.op == "lt":
        return left_s < right_s
    if f.op == "lte":
        return left_s <= right_s
    return False


def _resolve_table(
    tables: dict[str, list[dict[str, str]]], requested: str | None
) -> tuple[str | None, list[dict[str, str]] | None, str | None]:
    if not tables:
        return None, None, "No CSV tables are loaded in the current document context."
    if requested is None:
        if len(tables) == 1:
            name, rows = next(iter(tables.items()))
            return name, rows, None
        return (
            None,
            None,
            f"Multiple CSV tables available ({sorted(tables.keys())}); specify 'table'.",
        )

    cleaned = requested.strip()
    if ".." in cleaned or cleaned.startswith("/"):
        return None, None, "Invalid table identifier."
    if cleaned in tables:
        return cleaned, tables[cleaned], None

    by_basename = [name for name in tables if name.rsplit("/", 1)[-1] == cleaned]
    if len(by_basename) == 1:
        name = by_basename[0]
        return name, tables[name], None

    return (
        None,
        None,
        f"CSV table '{requested}' not found. Available tables: {sorted(tables.keys())}",
    )


class CsvLookupTool(BaseTool):
    """Safe tabular lookup over ingested CSV tables."""

    name = "csv_lookup"
    description = (
        "Filter, select columns, sort, and limit rows from loaded CSV tables "
        "using declarative predicates."
    )
    capability_tags = ["csv", "tabular", "lookup", "structured_data"]
    args_model = CsvLookupArgs

    def _execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, CsvLookupArgs)
        table_name, rows, err = _resolve_table(ctx.tables, args.table)
        if err is not None or rows is None or table_name is None:
            return ToolResult(ok=False, error=err or "Table resolution failed.")

        columns = list(rows[0].keys()) if rows else []
        col_set = set(columns)

        for flt in args.filters:
            if flt.column not in col_set:
                return ToolResult(
                    ok=False,
                    error=f"Unknown filter column '{flt.column}'. Available: {columns}",
                )
        for col in args.select:
            if col not in col_set:
                return ToolResult(
                    ok=False,
                    error=f"Unknown select column '{col}'. Available: {columns}",
                )
        if args.sort_by is not None and args.sort_by not in col_set:
            return ToolResult(
                ok=False,
                error=f"Unknown sort_by column '{args.sort_by}'. Available: {columns}",
            )

        matched = [
            row
            for row in rows
            if all(_matches_filter(row.get(flt.column, ""), flt) for flt in args.filters)
        ]

        if args.sort_by is not None:
            sort_col = args.sort_by

            def _sort_key(row: dict[str, str]) -> tuple[int, float | str]:
                raw = row.get(sort_col, "")
                num = _try_float(raw)
                return (0, num) if num is not None else (1, raw.lower())

            matched.sort(key=_sort_key, reverse=args.descending)

        selected_cols = args.select or columns
        limited = [
            {col: row.get(col, "") for col in selected_cols}
            for row in matched[: args.limit]
        ]

        return ToolResult(
            ok=True,
            data={
                "table": table_name,
                "total_matched": len(matched),
                "returned_count": len(limited),
                "rows": limited,
            },
        )
