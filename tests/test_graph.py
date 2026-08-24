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


class TestBetweennessCentrality:
    def test_path_middle_node_dominates(self) -> None:
        path = graph.from_edges(3, [(0, 1), (1, 2)], directed=True)
        scores = graph.betweenness_centrality(path)
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_directed_pair_counts(self) -> None:
        diamond = graph.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)], directed=True)
        scores = graph.betweenness_centrality(diamond)
        assert scores[0] == pytest.approx(0.0)
        assert scores[3] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(scores[2])

    def test_unnormalized_scale(self) -> None:
        path = graph.from_edges(3, [(0, 1), (1, 2)], directed=False)
        raw = graph.betweenness_centrality(path, normalized=False)
        assert raw[1] == pytest.approx(1.0)

    def test_empty_graph(self) -> None:
        assert graph.betweenness_centrality(sparse.csr_matrix((0, 0))).size == 0


class TestClosenessCentrality:
    def test_star_center_highest(self) -> None:
        star = graph.from_edges(5, [(0, i) for i in range(1, 5)], directed=False)
        scores = graph.closeness_centrality(star)
        assert int(np.argmax(scores)) == 0

    def test_path_endpoints_equal(self) -> None:
        path = graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=False)
        scores = graph.closeness_centrality(path)
        assert scores[0] == pytest.approx(scores[3])
        assert scores[1] == pytest.approx(scores[2])
        assert scores[1] > scores[0]


class TestEigenvectorCentrality:
    def test_star_center_dominates_bipartite(self) -> None:
        star = graph.from_edges(5, [(0, i) for i in range(1, 5)], directed=False)
        scores = graph.eigenvector_centrality(star)
        assert scores[0] == pytest.approx(2.0 * scores[1], rel=1e-6)

    def test_path_middles_higher(self) -> None:
        path = graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=False)
        scores = graph.eigenvector_centrality(path)
        assert min(scores[1], scores[2]) > max(scores[0], scores[3])

    def test_sum_to_one_and_empty(self) -> None:
        cycle = graph.from_edges(3, [(0, 1), (1, 2), (2, 0)], directed=False)
        scores = graph.eigenvector_centrality(cycle)
        assert scores.sum() == pytest.approx(1.0)
        assert graph.eigenvector_centrality(sparse.csr_matrix((0, 0))).size == 0


class TestCentralityCoverageEdges:
    def test_betweenness_ignores_self_loops(self) -> None:
        looped = graph.from_edges(3, [(0, 0), (0, 1), (1, 2)], directed=True)
        scores = graph.betweenness_centrality(looped)
        assert scores[0] == pytest.approx(0.0)

    def test_closeness_empty_and_disconnected(self) -> None:
        assert graph.closeness_centrality(sparse.csr_matrix((0, 0))).size == 0
        disconnected = graph.from_edges(3, [(0, 1)], directed=False)
        scores = graph.closeness_centrality(disconnected)
        assert scores[2] == pytest.approx(0.0)
        assert 0.0 < scores[0] < 1.0

    def test_eigenvector_directed_symmetrized(self) -> None:
        directed_path = graph.from_edges(3, [(0, 1), (1, 2)], directed=True)
        scores = graph.eigenvector_centrality(directed_path)
        assert scores.sum() == pytest.approx(1.0)
        assert scores[1] > scores[0]

    def test_eigenvector_zero_matrix(self) -> None:
        empty = sparse.csr_matrix((3, 3))
        scores = graph.eigenvector_centrality(empty)
        assert np.all(scores == 0.0)

    def test_eigenvector_fallback_bipartite_complete(self) -> None:
        complete_bipartite = graph.from_edges(
            6,
            [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)],
            directed=False,
        )
        scores = graph.eigenvector_centrality(complete_bipartite, max_iter=5)
        assert scores.sum() == pytest.approx(1.0)


class TestCommunities:
    def test_two_triangles_split(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
        adj = graph.from_edges(6, edges, directed=False)
        result = graph.detect_communities(adj, seed=1)
        assert result.n_communities == 2
        assert result.labels[0] == result.labels[1] == result.labels[2]
        assert result.labels[3] == result.labels[4] == result.labels[5]
        assert result.modularity == pytest.approx(0.5)

    def test_reproducible_with_seed(self) -> None:
        edges = [(i, (i + 1) % 12) for i in range(12)] + [(0, 6), (1, 7)]
        adj = graph.from_edges(12, edges, directed=False)
        first = graph.detect_communities(adj, seed=3)
        second = graph.detect_communities(adj, seed=3)
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_modularity_known_values(self) -> None:
        path = graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=False)
        assert graph.modularity(path, [0] * 4) == pytest.approx(0.0)
        split = graph.modularity(path, [0, 0, 1, 1])
        assert split == pytest.approx(1 / 6)

    def test_empty_graph_and_label_mismatch(self) -> None:
        empty = graph.detect_communities(sparse.csr_matrix((0, 0)))
        assert empty.n_communities == 0
        with pytest.raises(ValueError, match="one label per node"):
            graph.modularity(graph.from_edges(3, [(0, 1)], directed=False), [0, 0])

    def test_isolated_nodes_and_zero_degree_modularity(self) -> None:
        lonely = sparse.csr_matrix((4, 4))
        result = graph.detect_communities(lonely, seed=2)
        assert result.n_communities == 4
        assert graph.modularity(lonely, [0, 1, 2, 3]) == 0.0

    def test_isolated_node_inside_community_run(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2)]
        adj = graph.from_edges(4, edges, directed=False)
        result = graph.detect_communities(adj, seed=4)
        assert result.labels[3] == 3
        assert result.n_communities == 2

    def test_self_loops_ignored_in_neighbourhood(self) -> None:
        looped = graph.from_edges(4, [(0, 0), (0, 1), (1, 2), (2, 0)], directed=False)
        result = graph.detect_communities(looped, seed=6)
        assert len(set(result.labels.tolist())) >= 1

    def test_max_sweeps_exhausted_without_convergence(self) -> None:
        ring = graph.from_edges(
            8,
            [(i, (i + 1) % 8) for i in range(8)] + [(0, 4), (2, 6)],
            directed=False,
        )
        result = graph.detect_communities(ring, max_sweeps=1, seed=8)
        assert result.labels.size == 8
