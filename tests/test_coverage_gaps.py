"""Gap-closing tests pushing blended coverage to 100%."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from scipy import interpolate as spi

import cds2.ml as ml_module
from cds2 import (
    cli,
    graph,
    interpolate,
    io,
    linalg,
    ml,
    montecarlo,
    optimize,
    signals,
    sparse,
    special,
    spectral,
    stats,
    timeseries,
    viz,
)


# ---------------------------------------------------------------- cli ----
class TestCliGaps:
    def test_parse_numbers_invalid_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit, match="could not parse"):
            cli._parse_numbers("1,abc,3")

    @pytest.mark.parametrize("name", ["sin", "cos", "exp", "x2", "unit"])
    def test_every_builtin_function_runs(self, name: str) -> None:
        assert cli.main(["integrate", name, "--a", "0", "--b", "1"]) == 0


# --------------------------------------------------------------- graph ----
class TestGraphGaps:
    def test_from_edges_nonpositive_nodes(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            graph.from_edges(0, [(0, 1)])

    def test_pagerank_empty_graph(self) -> None:
        from scipy import sparse as sparse_module

        empty = sparse_module.csr_matrix((0, 0))
        scores = graph.pagerank(empty)
        assert scores.size == 0

    def test_pagerank_fallback_with_dangling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(graph, "_HAS_PR_KERNEL", False)
        adj = graph.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)], directed=True)
        scores = graph.pagerank(adj)
        assert scores.sum() == pytest.approx(1.0)

    def test_pagerank_fallback_dangling_only_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(graph, "_HAS_PR_KERNEL", False)
        adj = graph.from_edges(3, [], directed=True)
        scores = graph.pagerank(adj)
        assert np.allclose(scores, 1.0 / 3.0)

    def test_topological_order_nonpositive_nodes(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            graph.topological_order(0, [])


# ---------------------------------------------------------- interpolate ----
class TestInterpolateGaps:
    def test_lagrange_unexpected_array_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(spi, "lagrange", lambda x, y: np.array([1.0, 2.0]))
        with pytest.raises(TypeError, match="unexpected array"):
            interpolate.lagrange_poly([0.0, 1.0], [1.0, 2.0])


# ------------------------------------------------------------------ io ----
class TestIoExcelGuards:
    def test_read_excel_without_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "openpyxl", None)
        with pytest.raises(RuntimeError, match="openpyxl"):
            io.read_excel("whatever.xlsx")

    def test_write_excel_without_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "openpyxl", None)
        frame = pd_frame()
        with pytest.raises(RuntimeError, match="openpyxl"):
            io.write_excel(frame, "whatever.xlsx")

    def test_excel_roundtrip_with_real_engine(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        openpyxl = pytest.importorskip("openpyxl")
        assert openpyxl is not None
        path = tmp_path / "data.xlsx"
        frame = pd_frame()
        io.write_excel(frame, str(path))
        loaded = io.read_excel(str(path))
        assert len(loaded) == len(frame)


def pd_frame():
    import pandas as pd

    return pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})


# -------------------------------------------------------------- linalg ----
class TestLinalgGaps:
    def test_lstsq_rejects_1d_matrix(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            linalg.lstsq([1.0, 2.0], [1.0])

    def test_expm_zero_is_identity(self) -> None:
        result = linalg.expm([[0.0, 0.0], [0.0, 0.0]])
        assert np.allclose(result, np.eye(2))

    def test_logm_inverts_expm(self) -> None:
        a = np.array([[2.0, 0.5], [0.3, 1.5]])
        assert np.allclose(linalg.logm(linalg.expm(a)), a, atol=1e-8)

    def test_sqrtm_squared(self) -> None:
        a = np.array([[4.0, 1.0], [0.2, 3.0]])
        root = linalg.sqrtm(a)
        assert np.allclose(root @ root, a)


# ------------------------------------------------------------------ ml ----
class TestMlFallbackAndGuards:
    def test_import_error_branch_and_numpy_lloyd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cds2 as cds_package

        monkeypatch.setitem(sys.modules, "cds2._fast_kmeans", None)
        monkeypatch.delattr(cds_package, "_fast_kmeans", raising=False)
        reloaded = importlib.reload(ml_module)
        try:
            assert reloaded._HAS_C_KERNEL is False
            rng_values = np.random.default_rng(0)
            points = np.vstack(
                [rng_values.normal(-5, 0.5, (30, 2)), rng_values.normal(5, 0.5, (30, 2))]
            )
            model = reloaded.KMeans(n_clusters=2, seed=3).fit(points)
            assert model.inertia_ > 0
            assert set(np.unique(model.labels_)) == {0, 1}
        finally:
            monkeypatch.undo()
            importlib.reload(ml_module)

    def test_scaler_transform_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            ml.StandardScaler().transform([1.0])

    def test_split_mismatched_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="first dimension"):
            ml.train_test_split(np.zeros((5, 1)), np.zeros((4, 1)))

    def test_split_without_shuffle(self) -> None:
        x = np.arange(20.0).reshape(10, 2)
        first_train, first_test = ml.train_test_split(x, shuffle=False, test_size=0.4)
        assert first_test.ravel().tolist() == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        assert len(first_train) == 6

    def test_logistic_fit_accepts_1d_features(self) -> None:
        model = ml.LogisticRegression(max_iter=200).fit([1.0, -1.0], [1, 0])
        assert model.predict_proba([0.5]).shape == (1,)

    def test_logistic_proba_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            ml.LogisticRegression().predict_proba([1.0])

    def test_kmeans_rejects_zero_clusters(self) -> None:
        with pytest.raises(ValueError, match="n_clusters"):
            ml.KMeans(n_clusters=0).fit(np.ones((4, 2)))

    def test_kmeans_plus_plus_handles_duplicate_points(self) -> None:
        model = ml.KMeans(n_clusters=3, seed=0).fit(np.ones((6, 2)))
        assert model.cluster_centers_ is not None

    def test_knn_predict_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            ml.KNeighborsClassifier().predict([1.0])

    def test_knn_invalid_neighbor_count(self) -> None:
        with pytest.raises(ValueError, match="n_neighbors"):
            ml.KNeighborsClassifier(n_neighbors=50).fit(np.zeros((3, 1)), [0, 1, 0])

    def test_pca_transform_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            ml.PCA().transform(np.ones((3, 2)))

    def test_pca_invalid_components(self) -> None:
        with pytest.raises(ValueError, match="n_components"):
            ml.PCA(n_components=9).fit(np.ones((5, 2)))


class TestMlMacroMetrics:
    def test_precision_recall_f1_macro(self) -> None:
        truth = [0, 1, 2, 2]
        predicted = [0, 2, 2, 2]
        precision = ml.precision_score(truth, predicted, average="macro")
        recall = ml.recall_score(truth, predicted, average="macro")
        f1 = ml.f1_score(truth, predicted, average="macro")
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert f1 == pytest.approx(2 * precision * recall / (precision + recall))

    def test_macro_zero_denominator_guard(self) -> None:
        truth = [0, 1]
        predicted = [1, 1]
        assert ml.precision_score(truth, predicted, average="macro") == pytest.approx(0.25)
        assert ml.recall_score(truth, predicted, average="macro") == pytest.approx(0.5)
        assert ml.f1_score(truth, predicted, average="macro") >= 0.0


# --------------------------------------------------------- montecarlo ----
class TestMontecarloFallback:
    def test_hit_or_miss_scalar_function_uses_loop_path(self) -> None:
        area = montecarlo.hit_or_miss(lambda _x: 0.5, 0.0, 1.0, 1.0, n=500, seed=1)
        assert area == pytest.approx(0.5, abs=0.15)


# ------------------------------------------------------------- optimize ----
class TestOptimizeScalarPaths:
    def test_bracket_mode(self) -> None:
        result = optimize.minimize_scalar(lambda x: (x - 1.5) ** 2, bracket=(0.0, 2.0))
        assert result.x == pytest.approx(1.5, abs=1e-3)

    def test_explicit_brent_without_bounds(self) -> None:
        result = optimize.minimize_scalar(lambda x: x**2, method="brent")
        assert result.x == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------- signals ----
class TestSignalsGuards:
    def test_moving_average_window_validation(self) -> None:
        with pytest.raises(ValueError, match="window"):
            signals.moving_average([1.0, 2.0], window=0)

    def test_find_peaks_height_only(self) -> None:
        y = np.array([0.0, 2.0, 0.0])
        peaks = signals.find_peaks(y, height=1.0)
        assert peaks.indices.tolist() == [1]

    def test_find_peaks_distance_only(self) -> None:
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        peaks = signals.find_peaks(y, distance=3)
        assert len(peaks.indices) == 1

    def test_find_peaks_prominence_only(self) -> None:
        y = np.array([0.0, 3.0, 0.5, 3.0, 0.0])
        peaks = signals.find_peaks(y, prominence=2.0)
        assert len(peaks.indices) >= 1


# --------------------------------------------------------------- sparse ----
class TestSparseLinearOperator:
    def test_solve_cg_accepts_linear_operator(self) -> None:
        from scipy.sparse.linalg import LinearOperator

        n = 40
        matrix = (
            np.diag(2.0 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
        )
        rhs = np.arange(1.0, n + 1.0)

        def matvec(vector):
            return matrix @ vector

        operator = LinearOperator((n, n), matvec=matvec)
        result = sparse.solve_cg(operator, rhs)
        reference = np.linalg.solve(matrix, rhs)
        assert result.converged
        assert np.allclose(result.x, reference, atol=1e-5)


# ------------------------------------------------------------- special ----
class TestZetaHurwitz:
    def test_hurwitz_shift_identity(self) -> None:
        assert special.zeta(2.0, q=1.0) == pytest.approx(special.zeta(2.0))

    def test_hurwitz_known_value(self) -> None:
        assert special.zeta(2.0, q=2.0) == pytest.approx(special.zeta(2.0) - 1.0)


# ------------------------------------------------------------ spectral ----
class TestSpectralGuards:
    def test_fiedler_needs_three_nodes(self) -> None:
        tiny = graph.from_edges(2, [(0, 1)], directed=False)
        with pytest.raises(ValueError, match="at least 3"):
            spectral.fiedler_vector(tiny)

    def test_connectivity_needs_two_nodes(self) -> None:
        lonely = graph.from_edges(1, [], directed=False)
        with pytest.raises(ValueError, match="at least 2"):
            spectral.algebraic_connectivity(lonely)


# --------------------------------------------------------------- stats ----
class TestStatsValidation:
    def test_describe_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            stats.describe([])

    def test_kruskal_single_group_raises(self) -> None:
        with pytest.raises(ValueError, match="two groups"):
            stats.kruskal_wallis([1.0, 2.0])

    def test_levene_single_group_raises(self) -> None:
        with pytest.raises(ValueError, match="two groups"):
            stats.levene_test([1.0, 2.0])

    def test_chi_square_rejects_vector_table(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            stats.chi_square_independence([1, 2, 3])

    def test_bootstrap_custom_statistic_falls_back_to_loop(self) -> None:
        def first_value(sample):  # no axis kwarg -> forces TypeError path
            return float(sample.flat[0])

        data = np.arange(10.0, dtype=float)
        result = stats.bootstrap_ci(data, statistic=first_value, n_resamples=300, seed=4)
        assert 0.0 <= result.ci_low < result.ci_high <= 9.0


# ---------------------------------------------------------- timeseries ----
class TestTimeseriesValidation:
    def test_difference_invalid_lag(self) -> None:
        with pytest.raises(ValueError, match="lag"):
            timeseries.difference([1.0, 2.0], lag=5)

    def test_decompose_period_too_small(self) -> None:
        with pytest.raises(ValueError, match="period must be at least 2"):
            timeseries.seasonal_decompose(list(range(20)), period=1)

    def test_decompose_unknown_model(self) -> None:
        series = list(range(30))
        with pytest.raises(ValueError, match="unsupported decomposition"):
            timeseries.seasonal_decompose(series, period=5, model="trig")

    def test_acf_nlags_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="nlags"):
            timeseries.acf([1.0, 2.0], nlags=99)

    def test_pacf_nlags_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="nlags"):
            timeseries.pacf([1.0] * 20, nlags=99)

    def test_pacf_single_lag_skips_recursion(self) -> None:
        values = np.random.default_rng(0).normal(size=60)
        partial = timeseries.pacf(values, nlags=1)
        assert partial[0] == 1.0

    def test_ljung_box_invalid_lags(self) -> None:
        with pytest.raises(ValueError, match="lags"):
            timeseries.ljung_box(np.arange(30.0), lags=99)


# ----------------------------------------------------------------- viz ----
class TestVizBranches:
    def test_heatmap_without_colorbar(self) -> None:
        import matplotlib.pyplot as plt

        figure = viz.plot_heatmap(np.eye(3), colorbar=False)
        assert len(figure.axes) == 1
        plt.close(figure)

    def test_save_figure_without_close(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import matplotlib.pyplot as plt

        target = tmp_path / "kept.png"
        figure = viz.plot_series([1.0, 2.0])
        saved = viz.save_figure(figure, target, close=False)
        assert saved.exists()
        plt.close(figure)


class TestCliInvalidRange:
    def test_integrate_reversed_bounds(self) -> None:
        assert cli.main(["integrate", "sin", "--a", "1", "--b", "0"]) == 1


class TestScalerOneDimensional:
    def test_fit_and_transform_1d(self) -> None:
        scaler = ml.StandardScaler().fit([1.0, 2.0, 3.0])
        scaled = scaler.transform([3.0])
        assert scaled[0][0] == pytest.approx(1.0 / np.sqrt(2.0 / 3.0))

    def test_fit_transform(self) -> None:
        scaled = ml.StandardScaler().fit_transform([[1.0], [2.0], [3.0]])
        assert scaled.shape == (3, 1)


class TestKMeansGuards:
    def test_fit_accepts_1d_points(self) -> None:
        model = ml.KMeans(n_clusters=2, seed=0).fit([1.0, 1.1, 9.0, 9.1])
        assert model.labels_ is not None

    def test_predict_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            ml.KMeans().predict([[1.0]])

    def test_predict_1d_query_reshaped(self) -> None:
        model = ml.KMeans(n_clusters=2, seed=0).fit([1.0, 1.1, 9.0, 9.1])
        prediction = model.predict([9.05])
        assert prediction[0] in (0, 1)


class TestKMeansNumpyLoopBranches:
    def test_max_iter_exhaustion_and_empty_cluster_rescue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ml_module, "_HAS_C_KERNEL", False)
        identical = np.ones((8, 1))
        model = ml_module.KMeans(n_clusters=3, max_iter=1, seed=0).fit(identical)
        assert model.inertia_ >= 0.0

    def test_empty_cluster_rescue_with_degenerate_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ml_module, "_HAS_C_KERNEL", False)
        data = np.vstack([np.zeros((4, 1)), np.full((4, 1), 10.0)])

        def fixed_init(points, n_clusters, rng):
            return np.array([[0.0], [0.0], [10.0]], dtype=float)

        monkeypatch.setattr(ml_module.KMeans, "_kmeans_pp_init", staticmethod(fixed_init))
        model = ml_module.KMeans(n_clusters=3, max_iter=20, seed=0).fit(data)
        assert model.cluster_centers_ is not None
        assert len(np.unique(model.labels_)) >= 1


class TestVizSavePaths:
    def test_every_chart_supports_save(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import matplotlib.pyplot as plt

        signal_values = np.sin(np.linspace(0.0, 10.0, 128))
        targets = {
            "series": lambda path: viz.plot_series([1.0, 2.0], save=path),
            "histogram": lambda path: viz.plot_histogram([1.0, 2.0, 2.0], save=path),
            "scatter": lambda path: viz.plot_scatter([1, 2], [2, 4], save=path),
            "heatmap": lambda path: viz.plot_heatmap(np.eye(3), save=path),
            "spectrum": lambda path: viz.plot_spectrum(signal_values, fs=64.0, save=path),
            "regression": lambda path: viz.plot_regression([1, 2, 3], [2, 4, 6], save=path),
            "confusion": lambda path: viz.plot_confusion_matrix([[1, 0], [0, 1]], save=path),
        }
        for name, factory in targets.items():
            target = tmp_path / f"{name}.png"
            factory(target)
            assert target.exists(), name
            plt.close("all")


class TestSparseSeedParam:
    def test_eigensolvers_accept_seed_argument(self) -> None:
        symmetric = np.diag([1.0, 4.0, 9.0, 16.0, 25.0])
        top = sparse.largest_eigenpairs(symmetric, k=2, seed=1)
        bottom = sparse.smallest_eigenpairs(symmetric, k=2, seed=1)
        assert top.eigenvalues[0] == pytest.approx(25.0, rel=1e-6)
        assert bottom.eigenvalues[0] == pytest.approx(1.0, rel=1e-6)


@pytest.fixture()
def poisson_system_sparse():
    n = 200
    main_diag = 2.0 * np.ones(n)
    off_diag = -np.ones(n - 1)
    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    rhs = np.arange(1.0, n + 1.0)
    return matrix, rhs


class TestFinalArcs:
    def test_pagerank_fallback_max_iter_exhaustion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(graph, "_HAS_PR_KERNEL", False)
        adj = graph.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)], directed=True)
        scores = graph.pagerank(adj, max_iter=1)
        assert scores.sum() == pytest.approx(1.0)

    def test_numpy_lloyd_exhausts_iterations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ml_module, "_HAS_C_KERNEL", False)
        separated = np.vstack([np.zeros((5, 1)), np.full((5, 1), 10.0)])
        model = ml_module.KMeans(n_clusters=2, max_iter=1, seed=0).fit(separated)
        assert model.inertia_ >= 0.0
        assert len(np.unique(model.labels_)) == 2

    def test_pca_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            ml.PCA().fit([1.0, 2.0, 3.0])

    def test_knn_fit_accepts_1d_features(self) -> None:
        knn = ml.KNeighborsClassifier(n_neighbors=3).fit([0.0, 0.2, 9.0, 9.2], [0, 0, 1, 1])
        assert knn.predict([[0.1]]).tolist() == [0]

    def test_knn_predict_accepts_1d_query(self) -> None:
        knn = ml.KNeighborsClassifier(n_neighbors=3).fit([[0.0], [0.2], [9.0], [9.2]], [0, 0, 1, 1])
        assert knn.predict([0.15]).tolist() == [0]

    def test_hit_or_miss_vectorized_success_path(self) -> None:
        area = montecarlo.hit_or_miss(
            lambda grid: np.full_like(np.asarray(grid, dtype=float), 0.5),
            0.0,
            1.0,
            1.0,
            n=400,
            seed=9,
        )
        assert area == pytest.approx(0.5, abs=0.1)

    def test_solve_gmres_without_restart_argument(self, poisson_system_sparse) -> None:
        matrix, rhs = poisson_system_sparse
        result = sparse.solve_gmres(matrix, rhs, rtol=1e-12)
        reference = np.linalg.solve(matrix, rhs)
        assert np.max(np.abs(result.x - reference)) < 1e-5

    def test_pacf_two_lags_runs_recursion(self) -> None:
        values = np.random.default_rng(1).normal(size=80)
        partial = timeseries.pacf(values, nlags=2)
        assert partial.shape == (3,)

    def test_numpy_lloyd_zero_iterations_returns_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ml_module, "_HAS_C_KERNEL", False)
        separated = np.vstack([np.zeros((5, 1)), np.full((5, 1), 10.0)])
        model = ml_module.KMeans(n_clusters=2, max_iter=0, seed=0).fit(separated)
        assert model.labels_ is not None
        assert model.cluster_centers_ is not None
