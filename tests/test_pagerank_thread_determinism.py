"""The compiled PageRank kernel must not depend on the OpenMP thread count."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# 2,000 nodes with 800 dangling ones: the dangling-mass sum is large enough
# that a thread-partitioned reduction changes the floating-point association
# order, which previously shifted every rank for the whole run.
PROBE = """
import hashlib

import numpy as np

from cds2 import _fast_pagerank as kernel

N, LIVE = 2000, 1200
rng = np.random.default_rng(7)
rows, cols, vals = [], [], []
for node in range(LIVE):
    targets = rng.integers(0, N, size=2)
    weights = rng.random(2) + 0.05
    weights /= weights.sum()
    rows.extend([node, node])
    cols.extend(int(target) for target in targets)
    vals.extend(float(weight) for weight in weights)
indptr = np.zeros(N + 1, dtype=np.int64)
indptr[1:] = np.cumsum(np.bincount(np.asarray(rows, dtype=np.int64), minlength=N))
indices = np.ascontiguousarray(np.asarray(cols, dtype=np.int64))
data = np.ascontiguousarray(np.asarray(vals, dtype=np.float64))
dangling = np.ascontiguousarray(np.arange(LIVE, N, dtype=np.int64))
rank_bytes, _ = kernel.iterate(indptr, indices, data, N, 0.85, dangling, 100, 1e-14)
print(hashlib.sha256(rank_bytes).hexdigest())
"""


def _rank_digest(threads: int) -> str:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    # Mirror the parent's import resolution: an editable or in-place build
    # keeps the compiled kernel in a checkout rather than site-packages.
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_pagerank_kernel_ignores_the_thread_count() -> None:
    pytest.importorskip("cds2._fast_pagerank")
    assert _rank_digest(1) == _rank_digest(4)
