"""The compiled Lloyd kernel must not depend on the OpenMP thread count."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# 80 points on an integer grid: every squared distance and coordinate sum is
# exactly representable, so a digest difference between the runs below can
# only come from a thread-partitioned update, not from floating-point noise.
PROBE = """
import hashlib

import numpy as np

from cds2 import _fast_kmeans as kernel

points = np.ascontiguousarray(
    np.array(
        [[index % 7, (index * 3) % 5] for index in range(42)]
        + [[index % 5 + 9, (index * 2) % 6 + 7] for index in range(38)],
        dtype=np.float64,
    )
)
centers = np.ascontiguousarray(
    np.array([[1.0, 1.0], [4.0, 2.0], [10.0, 8.0]], dtype=np.float64)
)
labels, fitted, _iterations = kernel.lloyd(points, centers, 200, 1e-10)
print(hashlib.sha256(labels + fitted).hexdigest())
"""


def _digest(threads: int) -> str:
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


def test_kmeans_kernel_ignores_the_thread_count() -> None:
    pytest.importorskip("cds2._fast_kmeans")
    assert _digest(1) == _digest(4)
