"""Wall-clock, memory and CPU-time profiling for cds2 callables.

ProfileResult holds the timing statistics for a profiled run; ``profile`` is
the single entry point. ``@cds2.prof.timed`` decorates a function so every call
is profiled and the result is attached as ``call.last_profile``.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import statistics
import time
import tracemalloc
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class ProfileResult:
    """Statistics from a profiled callable run.

    Attributes:
        name: the callable name (or ``<lambda>``).
        repeats: number of timed repetitions.
        wall_min / wall_median / wall_max: wall-clock seconds.
        cpu_min / cpu_median / cpu_max: process CPU seconds (user + sys).
        peak_rss_mb: peak resident-set size growth during the run, in MiB.
    """

    name: str
    repeats: int
    wall_min: float
    wall_median: float
    wall_max: float
    cpu_min: float
    cpu_median: float
    cpu_max: float
    peak_rss_mb: float

    def summary(self) -> str:
        """Single-line human summary for log output."""
        return (
            f"{self.name}: wall {self.wall_median:.4f}s "
            f"(min {self.wall_min:.4f}, max {self.wall_max:.4f}) | "
            f"cpu {self.cpu_median:.4f}s | peak RSS {self.peak_rss_mb:.2f} MiB"
        )


def _repeats_times(func: Callable[[], object], repeats: int) -> tuple[list[float], list[float]]:
    """Run ``func`` *repeats* times, collecting wall and CPU times.

    The callable is called with no arguments; use ``functools.partial`` or a
    ``lambda`` to bind arguments. CPU time is taken from ``os.times`` so it
    reflects the process's own user+system time, not wall-clock.
    """
    walls: list[float] = []
    cpus: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        c0 = os.times()
        func()
        t1 = time.perf_counter()
        c1 = os.times()
        walls.append(t1 - t0)
        cpus.append((c1.user + c1.system) - (c0.user + c0.system))
    return walls, cpus


def profile(
    fn: Callable[..., object],
    *args: Any,
    repeats: int = 5,
    **kw: Any,
) -> ProfileResult:
    """Profile ``fn(*args, **kw)`` over *repeats* repetitions.

    Memory tracking is active only while the callable runs so imports and
    module-level setup are not counted. The function is always executed at
    least once even if ``repeats`` is 1.

    Example::

        A = np.random.randn(100, 100)
        b = np.random.randn(100)
        r = cds2.prof.profile(np.linalg.solve, A, b, repeats=3)
        print(r.wall_median)
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", repr(fn))
    bound = functools.partial(fn, *args, **kw)  # type: ignore[arg-type]

    tracemalloc.start()
    try:
        walls, cpus = _repeats_times(bound, repeats)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return ProfileResult(
        name=name,
        repeats=repeats,
        wall_min=min(walls),
        wall_median=statistics.median(walls),
        wall_max=max(walls),
        cpu_min=min(cpus),
        cpu_median=statistics.median(cpus),
        cpu_max=max(cpus),
        peak_rss_mb=peak / (1024 * 1024),
    )


def timed(func: Callable[..., object]) -> Callable[..., object]:
    """Decorator that attaches a ``last_profile`` attribute to each call.

    The decorated function runs exactly as before; after the call returns,
    ``call.last_profile`` holds the :class:`ProfileResult`. Repeats is fixed at
    1 for the decorator path — use :func:`profile` directly for multi-repeat
    measurements.

    Example::

        @cds2.prof.timed
        def my_solve(A, b):
            return cds2.linalg.solve(A, b)

        x = my_solve(A, b)
        print(my_solve.last_profile.wall_median)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kw: Any) -> Any:
        result = func(*args, **kw)
        wrapper.last_profile = profile(func, *args, repeats=1, **kw)  # type: ignore[attr-defined]
        return result

    return wrapper
