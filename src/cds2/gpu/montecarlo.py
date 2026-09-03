"""GPU-accelerated Monte Carlo via CuPy.

Each function mirrors its CPU counterpart in ``cds2.montecarlo``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _ensure_cupy

__all__ = ["metropolis_hastings", "mc_integrate", "pi_estimate"]


def pi_estimate(n_samples: int = 1_000_000, seed: int | None = None) -> float:
    """Estimate pi via GPU-accelerated hit-or-miss."""
    cp = _ensure_cupy()
    rng = cp.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    inside = cp.sum(xy[:, 0] ** 2 + xy[:, 1] ** 2 <= 1.0)
    return float(4.0 * inside / n_samples)


def mc_integrate(
    f: Callable[..., Any],
    a: float,
    b: float,
    n_samples: int = 1_000_000,
    seed: int | None = None,
) -> float:
    """1-D Monte Carlo integration on the GPU.

    The integrand ``f`` must accept and return CuPy arrays; the wrapper handles
    the host↔device transfer. For a NumPy-callable integrand, see the CPU
    fallback in ``cds2.montecarlo.mc_integrate``.
    """
    cp = _ensure_cupy()
    rng = cp.random.default_rng(seed)
    x = rng.uniform(a, b, size=n_samples)
    fx = f(x)
    return float((b - a) * cp.mean(fx))


def metropolis_hastings(
    log_pdf: Callable[..., Any],
    x0: float,
    n_samples: int = 10_000,
    proposal_scale: float = 1.0,
    seed: int | None = None,
) -> Any:
    """Metropolis-Hastings sampler on the GPU.

    ``log_pdf`` must accept a CuPy array and return a CuPy array of log-density
    values. Returns a NumPy array of samples.
    """
    cp = _ensure_cupy()
    rng = cp.random.default_rng(seed)
    samples = cp.empty(n_samples)
    x = cp.float64(x0)
    log_p_x = log_pdf(x)
    for i in range(n_samples):
        x_prop = x + rng.normal(0.0, proposal_scale)
        log_p_prop = log_pdf(x_prop)
        if cp.log(rng.uniform()) < log_p_prop - log_p_x:
            x, log_p_x = x_prop, log_p_prop
        samples[i] = x
    return cp.asnumpy(samples)
