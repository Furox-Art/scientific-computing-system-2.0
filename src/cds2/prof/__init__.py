"""cds2.prof — persistent profiling, benchmark history and regression gates.

Public API::

    from cds2.prof import profile, BenchHistory, RegressionGate

    # One-shot profiling
    result = profile(np.linalg.solve, A, b, repeats=5)
    print(result.median, result.peak_rss_mb)

    # Persistent history
    hist = BenchHistory()
    hist.append(entries=[{"name": "solve", "ratio": 1.05}])
    df = hist.load_range("2026-01-01", "2026-12-31")

    # Regression gate (used by pytest --regression)
    gate = RegressionGate(tolerance=0.10)
    gate.check_latest(df, name="solve")
"""

from __future__ import annotations

from .gates import RegressionGate
from .history import BenchHistory
from .profiler import ProfileResult, profile, timed

__all__ = [
    "BenchHistory",
    "ProfileResult",
    "RegressionGate",
    "profile",
    "timed",
]
