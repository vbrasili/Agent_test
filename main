import asyncio
import uuid
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

MAX_ROWS = 200_000
MAX_CELLS = 2_000_000

_DATASETS: dict[str, dict[str, Any]] = {}


def _get(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in _DATASETS:
        raise agents.ToolError(
            f"Unknown dataset '{dataset_id}'. Load a CSV first with load_csv, "
            "or list loaded datasets with list_datasets."
        )
    return _DATASETS[dataset_id]


def _store(frame: pd.DataFrame, name: str, source: str) -> str:
    dataset_id = uuid.uuid4().hex[:12]
    _DATASETS[dataset_id] = {"frame": frame, "name": name, "source": source}
    return dataset_id


class Filter(BaseModel):
    column: str = Field(description="Column name to filter on, exactly as in the dataset.")
    operator: Literal[
        "eq", "ne", "gt", "ge", "lt", "le", "contains", "starts_with", "ends_with", "in", "is_null", "not_null"
    ] = Field(description="Comparison to apply to the column's value.")
    value: Any | None = Field(
        default=None,
        description="Value to compare against. A list for 'in', a string for text operators, a number for comparisons. Unused for 'is_null'/'not_null'.",
    )


class Aggregation(BaseModel):
    column: str = Field(
        description="Column to aggregate. Use '*' with function 'count' to count rows."
    )
    function: Literal["count", "sum", "mean", "min", "max", "median", "std", "nunique"] = Field(
        description="Aggregate function to compute."
    )
    alias: str | None = Field(default=None, description="Optional output name for the result.")


@agents.tool(
    name="load_csv",
    description="Parse raw CSV text into an in-memory dataset and return its dataset_id, row count and columns. Use this after reading a CSV file's content from Google Drive, or when the user pastes CSV text directly.",
    model_access="direct",
)
def load_csv(
    csv_content: Annotated[str, Field(description="The full raw CSV text, with a header row.")],
    name: Annotated[str, Field(description="Short name for the dataset, e.g. 'ventes_2024'.")],
) -> dict[str, Any]:
    from io import StringIO

    try:
        frame = pd.read_csv(StringIO(csv_content))
    except pd.errors.EmptyDataError as exc:
        raise agents.ToolError("The CSV content is empty. Read the file again from Google Drive.") from exc
    except ValueError as exc:
        raise agents.ToolError(f"Could not parse the CSV: {exc}") from exc

    if len(frame) > MAX_ROWS or frame.size > MAX_CELLS:
        raise agents.ToolError(
            f"Dataset too large ({len(frame)} rows, {frame.size} cells). Limits: {MAX_ROWS} rows, {MAX_CELLS} cells."
        )
    dataset_id = _store(frame, name, "csv text")
    return {
        "dataset_id": dataset_id,
        "name": name,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
    }


@agents.tool(
    name="describe_columns",
    description="Describe every column of a loaded dataset: inferred dtype, non-null count, missing values, and for numeric columns min/max/mean, or the top unique values for text columns.",
    model_access="direct",
)
def describe_columns(
    dataset_id: Annotated[str, Field(description="Identifier returned by load_csv.")],
) -> list[dict[str, Any]]:
    data = _get(dataset_id)
    frame: pd.DataFrame = data["frame"]
    out: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        info: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null": int(series.count()),
            "missing": int(series.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(series):
            info["min"] = series.min()
            info["max"] = series.max()
            info["mean"] = series.mean()
        else:
            info["unique_count"] = int(series.nunique())
            info["top_values"] = series.value_counts().head(5).to_dict()
        out.append(info)
    return out


@agents.tool(
    name="filter_rows",
    description="Filter the rows of a loaded dataset with a list of conditions (AND-ed together) and store the result as a new dataset. Returns the new dataset_id and row count.",
    model_access="direct",
)
def filter_rows(
    dataset_id: Annotated[str, Field(description="Identifier of the source dataset.")],
    filters: Annotated[list[Filter], Field(description="Conditions to apply; all must hold.")],
    output_name: Annotated[str, Field(description="Name for the filtered dataset.")],
) -> dict[str, Any]:
    data = _get(dataset_id)
    frame: pd.DataFrame = data["frame"].copy()

    mask = pd.Series(True, index=frame.index)
    for condition in filters:
        if condition.column not in frame.columns:
            raise agents.ToolError(
                f"Unknown column '{condition.column}'. Available: {list(frame.columns)}"
            )
        series = frame[condition.column]
        op = condition.operator
        if op == "is_null":
            mask &= series.isna()
        elif op == "not_null":
            mask &= series.notna()
        else:
            value = condition.value
            if value is None:
                raise agents.ToolError(f"Operator '{op}' needs a value.")
            if op in {"gt", "ge", "lt", "le"}:
                numeric = pd.to_numeric(series, errors="coerce")
                value_num = pd.to_numeric(value, errors="coerce")
                if pd.isna(value_num):
                    raise agents.ToolError(f"Value '{value}' is not numeric; '{op}' needs a number.")
                mask &= {"gt": numeric > value_num, "ge": numeric >= value_num,
                         "lt": numeric < value_num, "le": numeric <= value_num}[op]
            elif op == "eq":
                mask &= series.astype(str) == str(value)
            elif op == "ne":
                mask &= series.astype(str) != str(value)
            elif op == "contains":
                mask &= series.astype(str).str.contains(str(value), case=False, na=False)
            elif op == "starts_with":
                mask &= series.astype(str).str.lower().str.startswith(str(value).lower(), na=False)
            elif op == "ends_with":
                mask &= series.astype(str).str.lower().str.endswith(str(value).lower(), na=False)
            elif op == "in":
                values = value if isinstance(value, list) else [value]
                mask &= series.astype(str).isin([str(v) for v in values])

    filtered = frame[mask]
    new_id = _store(filtered, output_name, f"filter of {dataset_id}")
    return {
        "dataset_id": new_id,
        "name": output_name,
        "row_count": int(len(filtered)),
        "source_row_count": int(len(frame)),
    }


@agents.tool(
    name="aggregate",
    description="Compute grouped aggregations on a loaded dataset: group rows by one or more columns and apply functions (count, sum, mean, min, max, median, std, nunique) to other columns. Returns real numbers computed from the data.",
    model_access="direct",
)
def aggregate(
    dataset_id: Annotated[str, Field(description="Identifier of the dataset to aggregate.")],
    group_by: Annotated[list[str], Field(description="Column(s) to group by. Use [] for a total over all rows.")],
    aggregations: Annotated[list[Aggregation], Field(description="Aggregations to compute per group.")],
    sort_by: Annotated[str | None, Field(description="Optional result column to sort by, descending.")] = None,
    limit: Annotated[int, Field(description="Maximum number of groups to return.")] = 20,
) -> list[dict[str, Any]]:
    data = _get(dataset_id)
    frame: pd.DataFrame = data["frame"]

    for column in group_by:
        if column not in frame.columns:
            raise agents.ToolError(f"Unknown group-by column '{column}'. Available: {list(frame.columns)}")

    ops = {
        "count": "count", "sum": "sum", "mean": "mean", "min": "min",
        "max": "max", "median": "median", "std": "std", "nunique": "nunique",
    }
    spec: dict[str, tuple[str, str]] = {}
    for agg in aggregations:
        if agg.column == "*":
            if agg.function != "count":
                raise agents.ToolError("'*' is only valid with function 'count'.")
            column = "__row__"
        elif agg.column not in frame.columns:
            raise agents.ToolError(f"Unknown column '{agg.column}'. Available: {list(frame.columns)}")
        else:
            column = agg.column
            if agg.function in {"sum", "mean", "median", "std"} and not pd.api.types.is_numeric_dtype(frame[agg.column]):
                raise agents.ToolError(f"Column '{agg.column}' is not numeric; '{agg.function}' needs numbers.")
        label = agg.alias or f"{agg.function}_{agg.column}"
        spec[label] = (column, ops[agg.function])

    frame = frame.assign(__row__=1)
    if group_by:
        result = frame.groupby(group_by, dropna=False).agg(**{
            label: pd.NamedAgg(column=column, aggfunc=function)
            for label, (column, function) in spec.items()
        }).reset_index()
    else:
        result = pd.DataFrame(
            {
                label: (
                    [float(len(frame))] if function == "count"
                    else [float(getattr(frame[column], function)())]
                )
                for label, (column, function) in spec.items()
            }
        )

    if sort_by:
        if sort_by not in result.columns:
            raise agents.ToolError(f"sort_by '{sort_by}' is not in the result columns: {list(result.columns)}")
        result = result.sort_values(sort_by, ascending=False)

    result = result.head(limit)
    records = result.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, "item"):
                record[key] = value.item()
    return records


@agents.tool(
    name="list_datasets",
    description="List the datasets loaded in this session, with their dataset_id, name, row count and columns.",
    model_access="direct",
)
def list_datasets() -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": dataset_id,
            "name": data["name"],
            "row_count": int(len(data["frame"])),
            "columns": list(data["frame"].columns),
        }
        for dataset_id, data in _DATASETS.items()
    ]


INSTRUCTIONS = """You are a conversational data analyst. The user gives you a link to a CSV file in
their Google Drive and asks questions in natural language; you answer with real numbers computed
from their data.

Workflow:
1. When the user gives a Google Drive link or file name, use the Google Drive connector's tools
   (google_drive_mcp) to find the file and read its raw CSV content. If the connector asks for a
   sign-in, tell the user to connect Google Drive and wait.
2. Load the content with load_csv. Remember the dataset_id: the session keeps the conversation, so
   for follow-up questions reuse the already-loaded dataset instead of reloading it.
3. Answer questions using describe_columns, filter_rows, aggregate and list_datasets. Chain them
   when needed, e.g. filter_rows then aggregate on the filtered dataset_id.

Rules:
- Only use the fixed tools. Never invent, estimate or guess a number: every figure in your answers
  comes from a tool result. If a question needs something the tools cannot compute, say so.
- There is no code execution: do not write or ask to run Python, SQL or shell code.
- Answer in the user's language, concisely, and show the actual figures (with units and column
  names). State which columns you grouped or filtered on so the user can verify.
- If a column name or filter value does not exist, use describe_columns to check and ask the user
  to clarify.
"""

agent = agents.Agent(
    model="mistral-large-4",
    name="data-analyst",
    instructions=INSTRUCTIONS,
    tools=[load_csv, describe_columns, filter_rows, aggregate, list_datasets],
    connectors=[agents.connector("google_drive_mcp")],
)


async def main() -> None:
    result = await agent.run("Bonjour !")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
