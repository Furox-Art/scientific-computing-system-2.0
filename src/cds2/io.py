"""I/O — convenience re-export of pandas read/write; native value is summarize/iter_csv."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = [
    "read_csv",
    "write_csv",
    "read_json",
    "write_json",
    "read_excel",
    "write_excel",
    "read_parquet",
    "write_parquet",
    "summarize",
    "iter_csv",
]


def read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return cast("pd.DataFrame", pd.read_csv(path, **kwargs))


def write_csv(df: pd.DataFrame, path: str, index: bool = False, **kwargs: Any) -> str:
    """Write a DataFrame to CSV."""
    df.to_csv(path, index=index, **kwargs)
    return path


def read_json(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read records-oriented JSON into a DataFrame."""
    return cast("pd.DataFrame", pd.read_json(path, **kwargs))


def write_json(df: pd.DataFrame, path: str, orient: str = "records", **kwargs: Any) -> str:
    """Write a DataFrame to JSON."""
    df.to_json(path, orient=cast(Any, orient), **kwargs)
    return path


def read_excel(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read the first sheet of an Excel workbook (requires openpyxl)."""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        msg = "Excel support requires the optional 'openpyxl' package"
        raise RuntimeError(msg) from exc
    return cast("pd.DataFrame", pd.read_excel(path, **kwargs))


def write_excel(df: pd.DataFrame, path: str, index: bool = False, **kwargs: Any) -> str:
    """Write a DataFrame to an Excel workbook (requires openpyxl)."""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        msg = "Excel support requires the optional 'openpyxl' package"
        raise RuntimeError(msg) from exc
    df.to_excel(path, index=index, **kwargs)
    return path


def read_parquet(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file (requires pyarrow or fastparquet)."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str, index: bool = False, **kwargs: Any) -> str:
    """Write a Parquet file (requires pyarrow or fastparquet)."""
    df.to_parquet(path, index=index, **kwargs)
    return path


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level summary: dtypes, nulls, uniques and numeric moments."""
    non_null = df.notna().sum()
    nulls = len(df) - non_null
    unique_counts = df.nunique(dropna=True)
    numeric = df.select_dtypes(include=[np.number])
    moments = (
        numeric.agg(["mean", "std", "min", "max"])
        if len(numeric.columns)
        else pd.DataFrame(index=["mean", "std", "min", "max"])
    )
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        mean_value: float | None = None
        std_value: float | None = None
        min_value: float | None = None
        max_value: float | None = None
        if column in moments.columns:
            mean_value, std_value, min_value, max_value = (float(v) for v in moments[column])
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "non_null": int(non_null[column]),
                "nulls": int(nulls[column]),
                "unique": int(unique_counts[column]),
                "mean": mean_value,
                "std": std_value,
                "min": min_value,
                "max": max_value,
            }
        )
    result: pd.DataFrame = pd.DataFrame(rows)
    return result


def iter_csv(path: str, chunksize: int = 100_000, **kwargs: Any) -> Any:
    """Yield successive DataFrame chunks - process datasets larger than RAM."""
    reader = pd.read_csv(path, chunksize=chunksize, **kwargs)
    for chunk in reader:
        yield cast("pd.DataFrame", chunk)
