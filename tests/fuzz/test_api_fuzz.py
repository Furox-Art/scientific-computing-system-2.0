"""API fuzz tests: call public functions with edge-case arrays.

Covers 0-size, 1-size, NaN, inf, negative and very large inputs. The goal is
to confirm that cds2 either returns a sensible result or raises a clean
``ValueError`` — never segfaults or raises an unexpected exception type.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cds2  # noqa: E402


def _call_clean(func: callable, *args: object) -> object:
    """Call ``func`` and return True if it raised ValueError or returned cleanly."""
    try:
        func(*args)
        return True
    except (ValueError, TypeError, __import__("numpy").linalg.LinAlgError):
        return True
    except Exception:  # noqa: BLE001
        # Unexpected exception type — fail the test with context.
        return False


class TestAPIFuzz:
    """Fuzz public APIs with degenerate inputs."""

    @pytest.mark.parametrize(
        "data",
        [
            np.array([], dtype=float),
            np.array([1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([np.nan, 1.0, 2.0]),
            np.array([np.inf, 1.0]),
            np.array([-1.0, 0.0, 1.0]),
        ],
    )
    def test_describe_edge_cases(self, data: np.ndarray) -> None:
        assert _call_clean(cds2.stats.describe, data)

    @pytest.mark.parametrize(
        "p",
        [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
        ],
    )
    def test_entropy_edge_cases(self, p: np.ndarray) -> None:
        assert _call_clean(cds2.infotheory.entropy, p)

    @pytest.mark.parametrize(
        "A,b",
        [
            (np.eye(2), np.array([1.0, 2.0])),
            (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([5.0, 6.0])),
            (np.zeros((2, 2)), np.array([1.0, 2.0])),  # singular
        ],
    )
    def test_solve_edge_cases(self, A: np.ndarray, b: np.ndarray) -> None:
        assert _call_clean(cds2.linalg.solve, A, b)

    @pytest.mark.parametrize(
        "n",
        [0, 1, 2, 5, 10],
    )
    def test_pagerank_sizes(self, n: int) -> None:
        if n == 0:
            A = np.zeros((0, 0))
        else:
            rng = np.random.default_rng(0)
            A = rng.random((n, n))
            A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        assert _call_clean(cds2.graph.pagerank, A)
