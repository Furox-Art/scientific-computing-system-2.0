"""Tests for cds2.graph."""

import numpy as np
import pytest
from scipy import sparse

from cds2 import graph


@pytest.fixture()
def path_graph() -> sparse.csr_matrix:
    return graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=True)


class TestFromEdges:
    def test_shape_and_weight(self) -> None:
        adj = graph.from_edges(3, [(0, 1), (2, 0)])
        assert adj.shape == (3, 3)
        assert adj[0, 1] == 1.0

    def test_undirected_mirror(self) -> None:
        adj = graph.from_edges(3, [(0, 2)], directed=False)
        assert adj[0, 2] == adj[2, 0] == 1.0

    def test_weighted_edges(self) -> None:
        adj = graph.from_edges(2, [(0, 1, 5.0)], weighted=True)
        assert adj[0, 1] == pytest.approx(5.0)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            graph.from_edges(2, [(0, 9)])

    def test_empty(self) -> None:
        assert graph.from_edges(3, []).nnz == 0


class TestDegree:
    def test_directed_degrees(self) -> None:
        adj = graph.from_edges(3, [(0, 1), (0, 2), (1, 2)])
        result = graph.degree(adj, directed=True)
        assert list(result.out_degree) == [2, 1, 0]
        assert list(result.in_degree) == [0, 1, 2]
        assert list(result.total) == [2, 2, 2]

    def test_undirected_symmetry(self) -> None:
        adj = graph.from_edges(3, [(0, 1)], directed=False)
        result = graph.degree(adj, directed=False)
        assert list(result.total) == [1, 1, 0]


class TestComponents:
    def test_two_components(self) -> None:
        adj = graph.from_edges(4, [(0, 1), (2, 3)], directed=False)
        result = graph.connected_components(adj)
        assert result.count == 2
        assert result.labels[0] == result.labels[1]
        assert result.labels[2] == result.labels[3]


class TestShortestPaths:
    def test_dijkstra_path_distances(self) -> None:
        edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)]
        dist = graph.single_source_shortest_paths(
            graph.from_edges(3, edges, weighted=True), source=0
        )
        assert dist[2] == pytest.approx(3.0)

    def test_bellman_ford_negative_edge(self) -> None:
        edges = [(0, 1, 4.0), (0, 2, 5.0), (1, 2, -3.0)]
        result = graph.bellman_ford_paths(
            graph.from_edges(3, edges, weighted=True), indices=np.array([0])
        )
        assert result.distances[0][2] == pytest.approx(1.0)

    def test_floyd_warshall_all_pairs(self) -> None:
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        result = graph.floyd_warshall_paths(graph.from_edges(3, edges, weighted=True))
        assert result.distances[0][2] == pytest.approx(2.0)

    def test_unreachable_is_inf(self) -> None:
        adj = graph.from_edges(3, [(0, 1)], directed=True)
        dist = graph.single_source_shortest_paths(adj, source=0)
        assert np.isinf(dist[2])

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            graph.shortest_paths(path_graph_adj(), method="astar")


def path_graph_adj() -> sparse.csr_matrix:
    return graph.from_edges(3, [(0, 1), (1, 2)])


class TestMST:
    def test_spanning_tree_weight(self) -> None:
        edges = [(0, 1, 4.0), (1, 2, 1.0), (0, 2, 3.0)]
        tree = graph.minimum_spanning_forest(
            graph.from_edges(3, edges, weighted=True, directed=False)
        )
        total = float(tree.sum())
        assert total == pytest.approx(4.0)


class TestPageRank:
    def test_sum_to_one(self) -> None:
        scores = graph.pagerank(graph.from_edges(5, [(0, 1), (1, 2), (2, 0), (3, 4)]))
        assert scores.sum() == pytest.approx(1.0)

    def test_hub_ranks_higher(self) -> None:
        star = graph.from_edges(4, [(1, 0), (2, 0), (3, 0)], directed=True)
        scores = graph.pagerank(star)
        assert scores[0] > scores[1]

    def test_regular_cycle_uniform(self) -> None:
        cycle = graph.from_edges(3, [(0, 1), (1, 2), (2, 0)])
        scores = graph.pagerank(cycle)
        assert np.allclose(scores, 1.0 / 3.0, atol=1e-8)

    def test_invalid_damping_raises(self) -> None:
        with pytest.raises(ValueError, match="damping"):
            graph.pagerank(graph.from_edges(2, [(0, 1)]), damping=1.5)


class TestTopologicalOrder:
    def test_valid_order(self) -> None:
        order = graph.topological_order(4, [(0, 2), (1, 2), (2, 3)])
        assert order.index(0) < order.index(2) < order.index(3)
        assert order.index(1) < order.index(2)

    def test_cycle_raises(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_order(3, [(0, 1), (1, 2), (2, 0)])
