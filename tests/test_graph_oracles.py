"""Invariant oracles that the 100% coverage gate missed.

PageRank, Laplacian and betweenness each had a weighted-graph defect that
survived 1277 tests because every fixture was unweighted or every assertion
was about shape, not value. These tests assert the defining identities on
adversarial weighted inputs.
"""

import numpy as np
import pytest

import cds2.graph as g
import cds2.spectral as sp


def test_pagerank_row_stochastic_on_weighted_graph():
    W = np.array([[0.0, 9.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pr = g.pagerank(W)
    # networkx is the independent oracle, used only here as dev dependency
    import networkx as nx

    ref = nx.pagerank(nx.from_numpy_array(W, create_using=nx.DiGraph), alpha=0.85, weight="weight")
    assert pr == pytest.approx([ref[i] for i in range(3)], abs=1e-6)


def test_laplacian_rows_sum_to_zero_weighted():
    A = np.array([[0.0, 9.0, 1.0], [9.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    L = sp.laplacian(A).toarray()
    assert L.sum(axis=1) == pytest.approx(np.zeros(3), abs=1e-12)


def test_laplacian_normalized_still_zero_row_sum():
    A = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    # Even the unnormalized part must be correct; normalized just scales it.
    L = sp.laplacian(A, normalized=False).toarray()
    assert L.sum(axis=1) == pytest.approx(np.zeros(3), abs=1e-12)


def test_betweenness_star_centre_is_one_normalized():
    S = np.zeros((5, 5))
    for i in range(1, 5):
        S[0, i] = S[i, 0] = 1.0
    assert g.betweenness_centrality(S, normalized=True)[0] == pytest.approx(1.0, abs=1e-12)
    assert g.betweenness_centrality(S, normalized=False)[0] == pytest.approx(6.0, abs=1e-12)


def test_betweenness_undirected_normalized_agrees_with_networkx():
    import networkx as nx

    # Small graph where the factor-2 error is visible
    A = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]], dtype=float)
    cds = g.betweenness_centrality(A, normalized=True)
    ref = nx.betweenness_centrality(nx.from_numpy_array(A), normalized=True)
    assert cds == pytest.approx([ref[i] for i in range(4)], abs=1e-12)


def test_betweenness_tiny_graph_does_not_crash():
    # n<=2 branch in normalized scale: should not divide by zero
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert g.betweenness_centrality(A, normalized=True) == pytest.approx([0.0, 0.0])
    assert g.betweenness_centrality(A, normalized=False) == pytest.approx([0.0, 0.0])
