"""Tests for cds2.data_analysis — pandas-backed DataSet."""

from __future__ import annotations

import pandas as pd
import pytest

from cds2.data_analysis import DataSet, from_dataframe, to_dataframe


def test_dataset_basic_properties() -> None:
    data = [
        {"name": "Alice", "age": 25, "score": 88},
        {"name": "Bob", "age": 30, "score": 92},
        {"name": "Charlie", "age": 25, "score": 70},
    ]
    ds = DataSet(data)
    assert ds.shape == (3, 3)
    assert len(ds) == 3
    assert ds.columns == ["name", "age", "score"]
    assert ds[0] == {"name": "Alice", "age": 25, "score": 88}
    assert "DataSet(rows=3" in repr(ds)


def test_dataset_column_and_select() -> None:
    ds = DataSet([{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}])
    assert ds.column("a") == [1, 4]
    assert ds.column("b") == [2, 5]
    selected = ds.select("a", "c")
    assert selected.columns == ["a", "c"]
    assert selected.to_list() == [{"a": 1, "c": 3}, {"a": 4, "c": 6}]
    with pytest.raises(ValueError):
        ds.column("missing")
    with pytest.raises(ValueError):
        ds.select("a", "missing")


def test_dataset_filter() -> None:
    data = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
        {"name": "Charlie", "age": 22},
    ]
    ds = DataSet(data)
    filtered = ds.filter(lambda r: isinstance(r["age"], (int, float)) and r["age"] >= 25)
    assert len(filtered) == 2
    assert "Bob" in filtered.column("name")
    assert "Charlie" not in filtered.column("name")


def test_dataset_head_tail() -> None:
    ds = DataSet([{"a": i} for i in range(10)])
    assert len(ds.head(3)) == 3
    assert len(ds.tail(3)) == 3
    assert ds.head(3)[0]["a"] == 0
    assert ds.tail(3)[-1]["a"] == 9
    assert len(ds.head()) == 5
    assert len(ds.tail()) == 5


def test_dataset_group_by() -> None:
    data = [
        {"city": "NYC", "temp": 20},
        {"city": "NYC", "temp": 22},
        {"city": "LA", "temp": 30},
        {"city": "LA", "temp": 32},
    ]
    ds = DataSet(data)
    grouped = ds.group_by("city")
    means = grouped.mean("temp")
    assert means["NYC"] == pytest.approx(21.0)
    assert means["LA"] == pytest.approx(31.0)
    counts = grouped.count()
    assert counts["NYC"] == 2
    assert counts["LA"] == 2
    # non-numeric column should map to 0.0
    data2 = [{"cat": "A", "val": "not_a_number"}]
    assert DataSet(data2).group_by("cat").mean("val")["A"] == 0.0


def test_to_dataframe_conversion() -> None:
    ds = DataSet([{"z": 1, "a": 2, "m": 3}, {"z": 4, "a": 5, "m": 6}])
    df = to_dataframe(ds)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["z", "a", "m"]
    assert len(df) == 2
    # instance method should give the same result
    df2 = ds.to_dataframe()
    pd.testing.assert_frame_equal(df, df2)
    # empty dataset
    empty = DataSet([])
    assert len(to_dataframe(empty)) == 0
    assert "empty" in repr(empty)


def test_from_dataframe_nan_handling() -> None:
    df = pd.DataFrame({"x": [1.0, float("nan"), 3.0], "label": ["a", "b", "c"]})
    ds = from_dataframe(df)
    col = ds.column("x")
    assert col[0] == 1.0
    assert col[1] is None
    assert col[2] == 3.0
    # classmethod and instance round-trip
    ds2 = DataSet.from_dataframe(pd.DataFrame({"a": [1, 2]}))
    assert ds2.column("a") == [1, 2]
    # bool preservation through round-trip
    bool_ds = DataSet([{"flag": True}, {"flag": False}])
    df_bool = to_dataframe(bool_ds)
    back = from_dataframe(df_bool)
    assert back.column("flag") == [True, False]


def test_roundtrip_and_summary() -> None:
    original = DataSet(
        [
            {"id": 1, "temp": 36.5, "city": "Istanbul"},
            {"id": 2, "temp": 22.0, "city": "Ankara"},
        ]
    )
    df = to_dataframe(original)
    restored = from_dataframe(df)
    assert restored.columns == original.columns
    assert restored.column("id") == original.column("id")
    assert restored.column("city") == original.column("city")
    # summary statistics
    desc = original.describe()
    assert not desc.empty
    assert "temp" in desc.columns or "temp" in str(desc)
    summ = original.summarize()
    assert isinstance(summ, pd.DataFrame)
    assert "column" in summ.columns
    assert len(summ) == 3
    # numeric moments are populated for numeric columns
    temp_row = summ[summ["column"] == "temp"].iloc[0]
    assert temp_row["mean"] == pytest.approx(29.25)
    # empty summarize still has correct schema
    empty_summ = DataSet([]).summarize()
    assert list(empty_summ.columns) == [
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
