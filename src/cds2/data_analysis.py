"""Pandas-backed data analysis — DataSet and DataFrame bridge.

A :class:`DataSet` is the v2 successor to the pure-Python ``cds.data_analysis.DataSet``
from v1. The public API is intentionally compatible (``columns``, ``shape``,
``column``, ``filter``, ``head``/``tail``, ``select``, ``group_by``,
``to_list``) so existing tutorials port without rewrites, but the internal
storage is a :class:`pandas.DataFrame` for vectorised speed and for seamless
interoperability with the pandas ecosystem.

Design choices mirror the v1 ``pandas_io`` bridge:

* ``None`` round-trips as ``NaN`` in numeric columns and back to ``None``.
* Column order is preserved in both directions.
* The module can be used either through instance methods
  (``ds.to_dataframe()`` / ``DataSet.from_dataframe(df)``) or the
  free functions :func:`to_dataframe` / :func:`from_dataframe`.
* Summary statistics are exposed via :meth:`DataSet.describe` (a thin
  wrapper around :meth:`pandas.DataFrame.describe`) and
  :meth:`DataSet.summarize` (column-level null / unique / numeric moments
  compatible with :func:`cds2.io.summarize`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd

Scalar: TypeAlias = int | float | str | bool | None
Row: TypeAlias = dict[str, Scalar]

__all__: list[str] = [
    "Scalar",
    "Row",
    "DataSet",
    "DataGroup",
    "to_dataframe",
    "from_dataframe",
]


def _is_na(value: object) -> bool:
    """Return ``True`` for pandas NA values (``NaN``, ``None``, ``NaT``)."""
    try:
        return bool(cast(Any, pd).isna(cast(Any, value)))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return value is None


def _normalize_row(record: dict[str, object]) -> Row:
    """Convert a raw mapping (possibly containing ``NaN``) to a :class:`Row`."""
    row: Row = {}
    for key, value in record.items():
        row[str(key)] = None if _is_na(value) else cast(Scalar, value)
    return row


class DataSet:
    """A lightweight DataFrame-like container backed by :class:`pandas.DataFrame`.

    Data is stored internally as a :class:`pandas.DataFrame` while the public
    API stays compatible with the v1 pure-Python implementation (list-of-dicts
    construction, ``column``/``filter``/``select``/``group_by`` helpers).
    """

    def __init__(self, data: list[Row] | pd.DataFrame | None = None) -> None:
        """Create a dataset.

        Args:
            data: either a list of row dictionaries (``Row``) or a
                :class:`pandas.DataFrame`. ``None`` or an empty list creates
                an empty dataset with no columns.
        """
        if data is None:
            self._df: pd.DataFrame = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            self._df = data.copy()
        else:
            rows: list[Row] = data
            if not rows:
                self._df = pd.DataFrame()
            else:
                columns = list(rows[0].keys())
                self._df = pd.DataFrame(rows, columns=columns if columns else None)

    @property
    def columns(self) -> list[str]:
        """Return the ordered list of column names."""
        return [str(col) for col in self._df.columns.tolist()]

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``."""
        return int(self._df.shape[0]), int(self._df.shape[1])

    def __len__(self) -> int:
        """Return the number of rows."""
        return int(len(self._df))

    def __getitem__(self, idx: int) -> Row:
        """Return the row at integer position ``idx`` as a :class:`Row`."""
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("DataSet index out of range")
        series = self._df.iloc[idx]
        raw: dict[str, object] = cast(dict[str, object], series.to_dict())
        return _normalize_row(raw)

    def column(self, name: str) -> list[Scalar]:
        """Extract a single column as a list, mapping ``NaN`` to ``None``."""
        if name not in self._df.columns:
            raise ValueError(f"Column '{name}' not found. Available: {self.columns}")
        series = self._df[name]
        result: list[Scalar] = []
        for value in series.tolist():
            result.append(None if _is_na(value) else cast(Scalar, value))
        return result

    def filter(self, predicate: Callable[[Row], bool]) -> DataSet:
        """Filter rows by a predicate applied to each :class:`Row`."""
        filtered: list[Row] = [row for row in self.to_list() if predicate(row)]
        return DataSet(filtered)

    def head(self, n: int = 5) -> DataSet:
        """Return the first ``n`` rows."""
        return DataSet(cast(pd.DataFrame, self._df.head(n)))

    def tail(self, n: int = 5) -> DataSet:
        """Return the last ``n`` rows."""
        return DataSet(cast(pd.DataFrame, self._df.tail(n)))

    def select(self, *names: str) -> DataSet:
        """Select a subset of columns by name, preserving order."""
        for name in names:
            if name not in self._df.columns:
                raise ValueError(f"Column '{name}' not found.")
        subset: pd.DataFrame = self._df[list(names)].copy()
        return DataSet(subset)

    def group_by(self, column_name: str) -> DataGroup:
        """Group rows by ``column_name`` for aggregation."""
        if column_name not in self._df.columns:
            raise ValueError(f"Column '{column_name}' not found.")
        return DataGroup(self._df, column_name)

    def to_list(self) -> list[Row]:
        """Export data as a list of dictionaries, mapping ``NaN`` to ``None``."""
        raw_records: list[dict[str, object]] = cast(
            list[dict[str, object]], self._df.to_dict(orient="records")
        )
        return [_normalize_row(record) for record in raw_records]

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying :class:`pandas.DataFrame`."""
        return cast(pd.DataFrame, self._df.copy())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> DataSet:
        """Create a :class:`DataSet` from a :class:`pandas.DataFrame`."""
        return cls(df)

    def describe(self) -> pd.DataFrame:
        """Descriptive statistics via :meth:`pandas.DataFrame.describe`.

        Includes all columns (numeric and non-numeric) when data is present;
        returns an empty :class:`DataFrame` for an empty dataset.
        """
        if self._df.empty:
            return cast(pd.DataFrame, pd.DataFrame())
        result = cast(pd.DataFrame, self._df.describe(include="all"))
        return result

    def summarize(self) -> pd.DataFrame:
        """Column-level summary: dtypes, nulls, uniques and numeric moments.

        Mirrors :func:`cds2.io.summarize` but operates directly on a
        :class:`DataSet`. Returns a :class:`pandas.DataFrame` with one row per
        column and fields ``column``, ``dtype``, ``non_null``, ``nulls``,
        ``unique``, ``mean``, ``std``, ``min``, ``max``.
        """
        if self._df.empty and len(self._df.columns) == 0:
            return cast(
                pd.DataFrame,
                pd.DataFrame(
                    columns=[
                        "column",
                        "dtype",
                        "non_null",
                        "nulls",
                        "unique",
                        "mean",
                        "std",
                        "min",
                        "max",
                    ]
                ),
            )
        df = self._df
        non_null = df.notna().sum()
        nulls = len(df) - non_null
        unique_counts = df.nunique(dropna=True)
        numeric = df.select_dtypes(include=[np.number])
        moments: pd.DataFrame
        if len(numeric.columns):
            moments = cast(pd.DataFrame, numeric.agg(["mean", "std", "min", "max"]))
        else:
            moments = pd.DataFrame(index=["mean", "std", "min", "max"])
        rows: list[dict[str, object]] = []
        for column in df.columns:
            mean_value: float | None = None
            std_value: float | None = None
            min_value: float | None = None
            max_value: float | None = None
            if column in moments.columns:
                col_moments = moments[column]
                mean_value = None if _is_na(col_moments["mean"]) else float(col_moments["mean"])
                std_value = None if _is_na(col_moments["std"]) else float(col_moments["std"])
                min_value = None if _is_na(col_moments["min"]) else float(col_moments["min"])
                max_value = None if _is_na(col_moments["max"]) else float(col_moments["max"])
            rows.append(
                {
                    "column": str(column),
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
        return cast(pd.DataFrame, pd.DataFrame(rows))

    def __repr__(self) -> str:
        """Return a compact summary."""
        if self._df.empty and len(self._df.columns) == 0:
            return "DataSet(empty)"
        return f"DataSet(rows={len(self._df)}, cols={len(self._df.columns)})"


class DataGroup:
    """Helper for grouped aggregations, backed by ``pandas.groupby``."""

    def __init__(self, df: pd.DataFrame, group_col: str) -> None:
        """Store the source frame and grouping column."""
        self._df: pd.DataFrame = df
        self.group_col: str = group_col

    def mean(self, numeric_col: str) -> dict[Scalar, float]:
        """Calculate the mean of ``numeric_col`` for each group.

        Non-numeric cells are ignored; groups with no numeric values map to
        ``0.0`` (matching the v1 behaviour).
        """
        result: dict[Scalar, float] = {}
        if numeric_col not in self._df.columns:
            raise ValueError(f"Column '{numeric_col}' not found.")
        grouped = self._df.groupby(self.group_col, dropna=False, observed=True)
        for key, group in grouped:
            py_key: Scalar = None if _is_na(key) else cast(Scalar, key)
            values: list[float] = []
            for value in group[numeric_col].tolist():
                if _is_na(value):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    values.append(float(value))
                else:
                    try:
                        values.append(float(cast(float, value)))
                    except (TypeError, ValueError):
                        continue
            result[py_key] = sum(values) / len(values) if values else 0.0
        return result

    def count(self) -> dict[Scalar, int]:
        """Count the number of rows in each group."""
        result: dict[Scalar, int] = {}
        grouped = self._df.groupby(self.group_col, dropna=False, observed=True)
        for key, group in grouped:
            py_key: Scalar = None if _is_na(key) else cast(Scalar, key)
            result[py_key] = int(len(group))
        return result


def to_dataframe(ds: DataSet) -> pd.DataFrame:
    """Convert a :class:`DataSet` to a :class:`pandas.DataFrame`.

    Thin wrapper around :meth:`DataSet.to_dataframe` for compatibility with
    the v1 ``cds.data_analysis.pandas_io.to_dataframe`` bridge.
    """
    return ds.to_dataframe()


def from_dataframe(df: pd.DataFrame) -> DataSet:
    """Convert a :class:`pandas.DataFrame` to a :class:`DataSet`.

    Thin wrapper around :meth:`DataSet.from_dataframe` for compatibility with
    the v1 ``cds.data_analysis.pandas_io.from_dataframe`` bridge. ``NaN``
    values are exposed as ``None`` through the :class:`DataSet` API, matching
    the v1 scalar contract.
    """
    return DataSet.from_dataframe(df)
