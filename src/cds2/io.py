"""Pandas-backed data I/O with optional Excel/Parquet bridges."""

from __future__ import annotations

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
]


def read_csv(path: str, **kwargs: object) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str, index: bool = False, **kwargs: object) -> str:
    """Write a DataFrame to CSV."""
    df.to_csv(path, index=index, **kwargs)
    return path


def read_json(path: str, **kwargs: object) -> pd.DataFrame:
    """Read records-oriented JSON into a DataFrame."""
    return pd.read_json(path, **kwargs)


def write_json(df: pd.DataFrame, path: str, orient: str = "records", **kwargs: object) -> str:
    """Write a DataFrame to JSON."""
    df.to_json(path, orient=orient, **kwargs)
    return path


def read_excel(path: str, **kwargs: object) -> pd.DataFrame:
    """Read the first sheet of an Excel workbook (requires openpyxl)."""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        msg = "Excel support requires the optional 'openpyxl' package"
        raise RuntimeError(msg) from exc
    return pd.read_excel(path, **kwargs)


def write_excel(df: pd.DataFrame, path: str, index: bool = False, **kwargs: object) -> str:
    """Write a DataFrame to an Excel workbook (requires openpyxl)."""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        msg = "Excel support requires the optional 'openpyxl' package"
        raise RuntimeError(msg) from exc
    df.to_excel(path, index=index, **kwargs)
    return path


def read_parquet(path: str, **kwargs: object) -> pd.DataFrame:
    """Read a Parquet file (requires pyarrow or fastparquet)."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str, index: bool = False, **kwargs: object) -> str:
    """Write a Parquet file (requires pyarrow or fastparquet)."""
    df.to_parquet(path, index=index, **kwargs)
    return path


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level summary: dtypes, nulls, uniques and numeric moments."""
    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        entry: dict[str, object] = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric):
            entry["mean"] = float(numeric.mean())
            entry["std"] = float(numeric.std())
            entry["min"] = float(numeric.min())
            entry["max"] = float(numeric.max())
        else:
            entry.update({"mean": None, "std": None, "min": None, "max": None})
        rows.append(entry)
    return pd.DataFrame(rows)
