"""Worker-count invariance for ``parallel_mc_integrate``.

The estimate must be a pure function of ``(func, a, b, n_total, seed)``. Before
the fix the interval was split into ``workers`` chunks and seeded per worker, so
the same seed produced a different number for every worker count -- and, because
the default worker count comes from ``os.cpu_count()``, on every machine. These
tests pin the public guarantee: only the ``workers`` argument changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cds2.montecarlo import parallel_mc_integrate


def _square(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(x * x, dtype=np.float64)


def test_same_seed_is_identical_for_every_worker_count() -> None:
    one = parallel_mc_integrate(_square, 0.0, 1.0, n_total=50_000, workers=1, seed=7)
    three = parallel_mc_integrate(_square, 0.0, 1.0, n_total=50_000, workers=3, seed=7)
    eight = parallel_mc_integrate(_square, 0.0, 1.0, n_total=50_000, workers=8, seed=7)
    assert one == three == eight


def test_default_worker_count_matches_explicit_workers() -> None:
    explicit = parallel_mc_integrate(_square, -1.0, 2.0, n_total=20_000, workers=1, seed=11)
    default = parallel_mc_integrate(_square, -1.0, 2.0, n_total=20_000, seed=11)
    assert default == explicit


def test_estimate_stays_accurate_after_fixed_chunking() -> None:
    estimate = parallel_mc_integrate(_square, 0.0, 1.0, n_total=200_000, workers=4, seed=13)
    assert estimate == pytest.approx(1.0 / 3.0, abs=0.01)
