"""cds2.gpu — optional GPU-accelerated backend via CuPy.

This submodule is a soft dependency: importing it raises ``RuntimeError`` with
an install hint when CuPy is absent. Install with the ``gpu`` extra::

    pip install cds2[gpu]            # picks the CuPy build matching your driver
    pip install cupy-cuda12x         # or cupy-cuda11x for older toolkits

Each function mirrors its CPU counterpart in ``cds2.linalg``, ``cds2.signals``
and ``cds2.montecarlo`` but runs on the GPU and returns a NumPy array (data is
synced back to host on return).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "cupy",
    "is_available",
    "synchronize",
]

# Lazily import CuPy. We do NOT import it at module load time so that
# ``import cds2`` stays free of the CuPy dependency.
_cupy: Any = None


def _ensure_cupy() -> Any:
    """Return the CuPy module, raising a helpful error if not installed."""
    global _cupy
    if _cupy is None:
        try:
            import cupy as cp  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "cds2.gpu requires CuPy. Install it with one of:\n"
                "  pip install cupy-cuda12x\n"
                "  pip install cupy-cuda11x\n"
                "  pip install cds2[gpu]\n"
                f"(original error: {exc})"
            ) from exc
        _cupy = cp  # pragma: no cover - requires CuPy
    return _cupy  # pragma: no cover - requires CuPy


def is_available() -> bool:
    """Return True if CuPy is importable (does not check GPU visibility)."""
    try:
        _ensure_cupy()
        return True  # pragma: no cover - requires CuPy
    except RuntimeError:
        return False


def cupy() -> Any:
    """Return the CuPy module, raising if not installed."""
    return _ensure_cupy()


def synchronize() -> None:
    """Synchronize the default CuPy stream (no-op if CuPy is unavailable)."""
    try:
        cp = _ensure_cupy()
        cp.cuda.Stream.null.synchronize()  # pragma: no cover - requires CuPy
    except RuntimeError:
        pass
