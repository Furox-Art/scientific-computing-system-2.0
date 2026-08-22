"""Tests for cds2.spectral graph analysis."""

import numpy as np
import pytest

from cds2 import spectral
from cds2.graph import from_edges


@pytest.fixture()
def path_graph():
    return from_edges(6, [(i, i + 1) for i in range(5)], directed=False)


class TestLaplacian:
    def test_row_sums_zero(self, path_graph) -> None:
        matrix = spectral.laplacian(path_graph).toarray()
        assert np.allclose(matrix.sum(axis=1), 0.0)

    def test_normalized_has_unit_diagonal(self, path_graph) -> None:
        matrix = spectral.laplacian(path_graph, normalized=True).toarray()
        assert np.allclose(np.diag(matrix), 1.0)

    def test_known_combinatorial_values(self, path_graph) -> None:
        matrix = spectral.laplacian(path_graph).toarray()
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[0, 1] == pytest.approx(-1.0)
        assert matrix[2, 2] == pytest.approx(2.0)


class TestConnectivity:
    def test_algebraic_connectivity_of_path(self, path_graph) -> None:
        expected = 2.0 * (1.0 - np.cos(np.pi / 6))
        assert spectral.algebraic_connectivity(path_graph) == pytest.approx(expected, abs=1e-6)

    def test_disconnected_graph_has_zero_connectivity(self) -> None:
        adj = from_edges(4, [(0, 1), (2, 3)], directed=False)
        value = spectral.algebraic_connectivity(adj)
        assert value < 1e-8

    def test_fiedler_vector_bipartition_signs(self, path_graph) -> None:
        fiedler = spectral.fiedler_vector(path_graph)
        signs = np.sign(fiedler)
        assert np.all(signs[:3] == signs[0])
        assert np.all(signs[3:] == -signs[0])


class TestSpectralClustering:
    def test_recovers_two_blobs(self) -> None:
        edges = [(i, i + 1) for i in range(4)] + [(5, 6), (6, 7), (7, 8), (8, 5)]
        adj = from_edges(9, edges, directed=False)
        labels = spectral.spectral_cluster(adj, n_clusters=2, seed=42)
        first_blob = set(labels[:4])
        second_blob = set(labels[5:])
        assert len(first_blob) == 1 and len(second_blob) == 1
        assert first_blob != second_blob

    def test_invalid_cluster_count_raises(self, path_graph) -> None:
        with pytest.raises(ValueError, match="n_clusters"):
            spectral.spectral_cluster(path_graph, n_clusters=99)
