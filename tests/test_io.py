"""Tests for cds2.io."""

from __future__ import annotations

import pandas as pd
import pytest

from cds2 import io


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})


class TestCsvRoundtrip:
    def test_roundtrip(self, tmp_path, sample_frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        io.write_csv(sample_frame, str(path))
        loaded = io.read_csv(str(path))
        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 3


class TestJsonRoundtrip:
    def test_roundtrip(self, tmp_path, sample_frame: pd.DataFrame) -> None:
        path = tmp_path / "data.json"
        io.write_json(sample_frame, str(path))
        loaded = io.read_json(str(path))
        assert loaded["b"].tolist() == ["x", "y", "z"]


class TestOptionalFormats:
    def test_excel_roundtrip(self, tmp_path, sample_frame: pd.DataFrame) -> None:
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        path = tmp_path / "data.xlsx"
        io.write_excel(sample_frame, str(path))
        loaded = io.read_excel(str(path))
        assert len(loaded) == 3

    def test_parquet_roundtrip(self, tmp_path, sample_frame: pd.DataFrame) -> None:
        pytest.importorskip("pyarrow", reason="pyarrow not installed")
        path = tmp_path / "data.parquet"
        io.write_parquet(sample_frame, str(path))
        loaded = io.read_parquet(str(path))
        assert loaded["a"].sum() == pytest.approx(6.0)


class TestSummarize:
    def test_summary_columns_and_stats(self, sample_frame: pd.DataFrame) -> None:
        summary = io.summarize(sample_frame)
        assert set(summary["column"]) == {"a", "b"}
        numeric_row = summary.loc[summary["column"] == "a"].iloc[0]
        assert numeric_row["mean"] == pytest.approx(2.0)
        assert numeric_row["min"] == pytest.approx(1.0)
        text_row = summary.loc[summary["column"] == "b"].iloc[0]
        assert pd.isna(text_row["mean"])

    def test_nulls_counted(self) -> None:
        frame = pd.DataFrame({"x": [1.0, None, 3.0]})
        summary = io.summarize(frame)
        assert summary.iloc[0]["nulls"] == 1
        assert summary.iloc[0]["non_null"] == 2
