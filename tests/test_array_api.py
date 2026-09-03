"""Tests for cds2.array_api Array API compatibility namespace."""

import numpy as np

from cds2 import array_api


class TestReductions:
    def test_asarray_and_scalars(self) -> None:
        x = array_api.asarray([1.0, 2.0, 3.0])
        assert array_api.sum(x) == 6.0
        assert array_api.mean(x) == 2.0
        assert array_api.var(x) == np.var([1.0, 2.0, 3.0])
        assert array_api.std(x) == np.std([1.0, 2.0, 3.0])
        assert array_api.min(x) == 1.0
        assert array_api.max(x) == 3.0

    def test_axis_reduction(self) -> None:
        x = array_api.asarray([[1.0, 2.0], [3.0, 4.0]])
        assert list(array_api.sum(x, axis=0)) == [4.0, 6.0]


class TestElementwise:
    def test_math_functions(self) -> None:
        x = array_api.asarray([0.0, 1.0, -2.0])
        assert list(array_api.abs(x)) == [0.0, 1.0, 2.0]
        assert list(array_api.sin(x)) == list(np.sin([0.0, 1.0, -2.0]))
        assert list(array_api.cos(x)) == list(np.cos([0.0, 1.0, -2.0]))
        assert list(array_api.exp(x)) == list(np.exp([0.0, 1.0, -2.0]))
        assert list(array_api.log(array_api.asarray([1.0, 2.0]))) == list(np.log([1.0, 2.0]))

    def test_matmul(self) -> None:
        a = array_api.asarray([[1.0, 2.0], [3.0, 4.0]])
        b = array_api.asarray([[1.0, 0.0], [0.0, 1.0]])
        assert (array_api.matmul(a, b) == a).all()


class TestFFT:
    def test_roundtrip(self) -> None:
        x = array_api.asarray([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(array_api.ifft(array_api.fft(x)), x)
        assert len(array_api.rfft(x)) == 3


class TestLinalgNamespace:
    def test_solve(self) -> None:
        a = array_api.asarray([[2.0, 0.0], [0.0, 2.0]])
        b = array_api.asarray([4.0, 6.0])
        assert list(array_api.solve(a, b)) == [2.0, 3.0]
        assert list(array_api.linalg.solve(a, b)) == [2.0, 3.0]

    def test_cholesky(self) -> None:
        a = array_api.asarray([[4.0, 2.0], [2.0, 3.0]])
        L = array_api.cholesky(a)
        assert np.allclose(L @ L.T, a)
        U = array_api.linalg.cholesky(a, upper=True)
        assert np.allclose(U.T @ U, a)

    def test_svd_reconstructs(self) -> None:
        rng = np.random.default_rng(0)
        a = array_api.asarray(rng.normal(size=(4, 3)))
        U, s, Vt = array_api.svd(a, full_matrices=False)
        assert np.allclose(U * s @ Vt, a, atol=1e-10)

    def test_eigh(self) -> None:
        a = array_api.asarray([[2.0, 1.0], [1.0, 2.0]])
        w, _v = array_api.eigh(a)
        assert sorted(w.tolist()) == [1.0, 3.0]


class TestNamespaceInfo:
    def test_fallback_info_without_compat_package(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import sys

        # Force the `from array_api_compat import ...` to raise ImportError
        # even if the optional package is installed.
        monkeypatch.setitem(sys.modules, "array_api_compat", None)
        info = array_api.__array_namespace_info__()
        assert info.version == "2023.12"
        assert info.capabilities["boolean_indexing"] is True
