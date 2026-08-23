"""Tests for cds2.knowledge graph, notebook and retrieval."""

from __future__ import annotations

import pytest

from cds2.knowledge import Concept, KnowledgeGraph, Notebook, Relation, search


@pytest.fixture()
def graph() -> KnowledgeGraph:
    built = KnowledgeGraph()
    built.add_concept(Concept("Laplace", "donusum matematigi"))
    built.add_concept(Concept("Olasilik", "olasilik teorisi"))
    built.add_concept(Concept("Istatistik", "veri bilimi temeli"))
    built.add_concept(Concept("Izole"))
    built.add_relation(Relation("Olasilik", "Istatistik", "temel-olur"))
    built.add_relation(Relation("Laplace", "Olasilik", "arac-olur"))
    return built


class TestGraphBuilding:
    def test_unknown_endpoint_raises(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="unknown concept"):
            graph.add_relation(Relation("Olasilik", "Gizli", "link"))

    def test_neighbors_directions(self, graph: KnowledgeGraph) -> None:
        assert graph.neighbors("Olasilik", "outgoing") == ["Istatistik"]
        assert graph.neighbors("Olasilik", "incoming") == ["Laplace"]
        assert graph.neighbors("Olasilik", "both") == ["Istatistik", "Laplace"]

    def test_invalid_direction_raises(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="direction"):
            graph.neighbors("Olasilik", "sideways")

    def test_unknown_concept_neighbors_raises(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="unknown"):
            graph.neighbors("Yok")


class TestTraversal:
    def test_shortest_path_two_hops(self, graph: KnowledgeGraph) -> None:
        assert graph.shortest_path("Laplace", "Istatistik") == [
            "Laplace",
            "Olasilik",
            "Istatistik",
        ]

    def test_same_start_and_goal(self, graph: KnowledgeGraph) -> None:
        assert graph.shortest_path("Laplace", "Laplace") == ["Laplace"]

    def test_unreachable_returns_empty(self, graph: KnowledgeGraph) -> None:
        assert graph.shortest_path("Izole", "Laplace") == []

    def test_transitive_closure(self, graph: KnowledgeGraph) -> None:
        assert graph.transitive_closure("Laplace") == {"Olasilik", "Istatistik"}

    def test_closure_unknown_raises(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="unknown"):
            graph.transitive_closure("Yok")

    def test_cycle_detection_true(self) -> None:
        cyclic = KnowledgeGraph()
        for name in ("a", "b", "c"):
            cyclic.add_concept(Concept(name))
        cyclic.add_relation(Relation("a", "b"))
        cyclic.add_relation(Relation("b", "c"))
        cyclic.add_relation(Relation("c", "a"))
        assert cyclic.has_cycle()

    def test_acyclic_graph_false(self, graph: KnowledgeGraph) -> None:
        assert not graph.has_cycle()


class TestNotebookAndSearch:
    def test_tag_and_concept_lookup(self, graph: KnowledgeGraph) -> None:
        notebook = Notebook()
        notebook.add("Not1", "laplace donusumu", tags=["analiz"], linked_concepts=["Laplace"])
        notebook.add("Not2", "istatistiksel cikarim", tags=["analiz"])
        assert len(notebook.find_by_tag("analiz")) == 2
        assert [note.title for note in notebook.find_by_concept("laplace")] == ["Not1"]

    def test_search_ranks_concept_above_note(self, graph: KnowledgeGraph) -> None:
        notebook = Notebook()
        notebook.add("Not1", "olasilik yogunlugu")
        results = search("olasilik", graph, notebook)
        assert results
        assert results[0][1].startswith("concept:")

    def test_search_no_hits(self, graph: KnowledgeGraph) -> None:
        assert search("kuantum", graph) == []

    def test_search_relation_labels(self, graph: KnowledgeGraph) -> None:
        results = search("arac", graph)
        assert any("relation:" in description for _score, description in results)
