"""Tests for v3.1.0 industrial additions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cds2 import io, stats
from cds2.montecarlo import parallel_mc_integrate
from cds2.stats import StreamingStats


def square_integrand(x):
    return x * x


class TestParallelMonteCarlo:
    def test_matches_single_worker_estimate(self) -> None:
        estimate = parallel_mc_integrate(
            square_integrand, 0.0, 1.0, n_total=400_000, workers=4, seed=7
        )
        assert estimate == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_seeded_reproducible(self) -> None:
        first = parallel_mc_integrate(square_integrand, -1.0, 2.0, n_total=200_000, seed=11)
        second = parallel_mc_integrate(square_integrand, -1.0, 2.0, n_total=200_000, seed=11)
        assert first == second

    def test_reversed_bounds_raise(self) -> None:
        with pytest.raises(ValueError, match="greater"):
            parallel_mc_integrate(square_integrand, 1.0, 0.0)


class TestStreamingStats:
    def test_matches_batch_statistics(self) -> None:
        data = np.random.default_rng(0).normal(loc=-4.0, scale=6.0, size=5000)
        stream = StreamingStats()
        for chunk in np.array_split(data, 13):
            stream.push(chunk)
        assert stream.mean == pytest.approx(float(np.mean(data)), abs=1e-10)
        assert stream.standard_deviation == pytest.approx(float(np.std(data, ddof=1)), rel=1e-10)

    def test_merge_equals_joint(self) -> None:
        data = np.random.default_rng(1).uniform(size=800)
        left = StreamingStats().push(data[:300])
        right = StreamingStats().push(data[300:])
        merged = left.merge(right)
        assert merged.count_value == 800
        assert merged.mean == pytest.approx(float(np.mean(data)))
        assert merged.standard_deviation == pytest.approx(float(np.std(data, ddof=1)), rel=1e-9)

    def test_empty_push_is_noop(self) -> None:
        stream = StreamingStats().push([])
        assert stream.count_value == 0

    def test_mean_before_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="no observations"):
            StreamingStats().mean

    def test_variance_needs_two(self) -> None:
        stream = StreamingStats().push([42.0])
        with pytest.raises(ValueError, match="two observations"):
            _ = stream.variance


class TestIterCsv:
    def test_chunked_reading_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        frame = pd.DataFrame({"value": np.arange(250.0), "tag": ["a"] * 250})
        path = tmp_path / "big.csv"
        io.write_csv(frame, str(path))
        chunks = list(io.iter_csv(str(path), chunksize=100))
        total_rows = sum(len(chunk) for chunk in chunks)
        assert len(chunks) == 3
        assert total_rows == 250

    def test_chunks_preserve_columns(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        path = tmp_path / "small.csv"
        frame.to_csv(path, index=False)
        first = next(iter(io.iter_csv(str(path), chunksize=1)))
        assert list(first.columns) == ["x", "y"]


class TestStatsMatrixAdditionsV31:
    def test_covariance_symmetry(self) -> None:
        data = np.random.default_rng(2).normal(size=(40, 3))
        matrix = stats.covariance_matrix(data)
        assert np.allclose(matrix, matrix.T)

    def test_correlation_values_bounded(self) -> None:
        data = np.random.default_rng(3).normal(size=(60, 3))
        off_diagonal = stats.correlation_matrix(data)[np.triu_indices(3, k=1)]
        assert np.all(np.abs(off_diagonal) <= 1.0)

    def test_multivariate_normal_logpdf_zero_vector(self) -> None:
        value = stats.multivariate_normal_logpdf([0.0], [0.0], [[4.0]])
        expected = -0.5 * (np.log(2 * np.pi * 4.0))
        assert value == pytest.approx(expected)

    def test_merge_two_empty_streams(self) -> None:
        merged = StreamingStats().merge(StreamingStats())
        assert merged.count_value == 0

    def test_merge_empty_self_adopts_other(self) -> None:
        data = np.random.default_rng(4).normal(size=500)
        merged = StreamingStats().merge(StreamingStats().push(data))
        assert merged.count_value == 500
        assert merged.mean == pytest.approx(float(np.mean(data)))
        assert merged.standard_deviation == pytest.approx(
            float(np.std(data, ddof=1)), rel=1e-9
        )
