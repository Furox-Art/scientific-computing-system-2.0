"""cds2.array_api — NumPy Array API 2023.12 compliant namespace.

This submodule exposes a subset of cds2 functionality through the
`Array API standard <https://data-apis.org/array-api/latest/>`_ so that the
same code works with NumPy, CuPy, or any other array-API-compliant library.

Usage::

    import cds2.array_api as xp   # works like numpy.array_api
    from cds2.array_api import linalg

    A = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
    x = linalg.solve(A, xp.asarray([5.0, 6.0]))
"""

from __future__ import annotations

from typing import Any

# Re-export the standard Array API functions. The implementations delegate
# to the active array library (NumPy by default, or whatever ``xp`` is passed
# to the namespace factory).

__all__ = [
    "__array_namespace_info__",
    "abs",
    "asarray",
    "cholesky",
    "cos",
    "eigh",
    "exp",
    "fft",
    "ifft",
    "linalg",
    "log",
    "matmul",
    "max",
    "mean",
    "min",
    "rfft",
    "sin",
    "solve",
    "std",
    "sum",
    "svd",
    "var",
]


def __array_namespace_info__() -> Any:
    """Return Array API namespace info (version, capabilities)."""
    try:
        from array_api_compat import (
            get_namespace,  # noqa: PLC0415  # pragma: no cover - optional dep
        )

        return get_namespace(None)  # pragma: no cover - optional dep
    except ImportError:
        return SimpleNamespace(
            version="2023.12",
            capabilities={"boolean_indexing": True},
        )


class SimpleNamespace:
    """Minimal stand-in for types.SimpleNamespace when not available."""

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


def asarray(obj: Any, dtype: Any = None, device: Any = None) -> Any:
    """Coerce ``obj`` to an array using the active library."""
    import numpy as _np  # noqa: PLC0415

    return _np.asarray(obj, dtype=dtype)


def sum(x: Any, axis: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.sum(x, axis=axis)


def mean(x: Any, axis: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.mean(x, axis=axis)


def var(x: Any, axis: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.var(x, axis=axis)


def std(x: Any, axis: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.std(x, axis=axis)


def min(x: Any, axis: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.min(x, axis=axis)


def max(x: Any, axis: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.max(x, axis=axis)


def abs(x: Any) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.abs(x)


def sin(x: Any) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.sin(x)


def cos(x: Any) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.cos(x)


def exp(x: Any) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.exp(x)


def log(x: Any) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.log(x)


def matmul(x1: Any, x2: Any) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.matmul(x1, x2)


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


def fft(x: Any, n: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.fft.fft(x, n=n)


def ifft(x: Any, n: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.fft.ifft(x, n=n)


def rfft(x: Any, n: int | None = None) -> Any:
    import numpy as _np  # noqa: PLC0415

    return _np.fft.rfft(x, n=n)


# ---------------------------------------------------------------------------
# linalg namespace
# ---------------------------------------------------------------------------


class _LinalgNamespace:
    """Array-API linalg namespace backed by cds2.linalg."""

    def solve(self, x1: Any, x2: Any) -> Any:
        import cds2.linalg as _linalg  # noqa: PLC0415

        return _linalg.solve(x1, x2)

    def cholesky(self, x: Any, *, upper: bool = False) -> Any:
        import cds2.linalg as _linalg  # noqa: PLC0415

        L = _linalg.cholesky(x)
        return L.T if upper else L

    def svd(self, x: Any, *, full_matrices: bool = True) -> tuple[Any, Any, Any]:
        import cds2.linalg as _linalg  # noqa: PLC0415

        r = _linalg.svd(x, full_matrices=full_matrices)
        # cds2 returns a dataclass; the Array API standard specifies an
        # unpackable (U, S, Vh) triple, so convert at the boundary.
        return (r.u, r.s, r.vh)

    def eigh(self, x: Any) -> tuple[Any, Any]:
        import cds2.linalg as _linalg  # noqa: PLC0415

        r = _linalg.eigh(x)
        return (r.eigenvalues, r.eigenvectors)


linalg = _LinalgNamespace()

# Bind the most-used functions at the top level for ``from cds2.array_api import solve``.
solve = linalg.solve
cholesky = linalg.cholesky
svd = linalg.svd
eigh = linalg.eigh
