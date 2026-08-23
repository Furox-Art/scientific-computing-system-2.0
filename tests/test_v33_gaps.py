"""Final coverage sweep for the v3.3.0 heritage modules."""

from __future__ import annotations

import numpy as np
import pytest

from cds2 import scientific
from cds2.hypothesis import HypothesisEngine
from cds2.knowledge import Concept, KnowledgeGraph, Notebook, Relation, search
from cds2.modeling import (
    Constant,
    Expression,
    MathModel,
    Negate,
    diff,
    evaluate,
    iter_postorder,
    simplify,
    substitute,
    symbol,
    to_latex,
    to_string,
)
from cds2.nlp.autograd import Value
from cds2.nlp.tokenizer import BPETokenizer
from cds2.quantum import QuantumCircuit


@pytest.fixture()
def graph() -> KnowledgeGraph:
    built = KnowledgeGraph()
    for name in ("Laplace", "Olasilik", "Istatistik", "Izole"):
        built.add_concept(Concept(name))
    built.add_relation(Relation("Olasilik", "Istatistik", "temel-olur"))
    built.add_relation(Relation("Laplace", "Olasilik", "arac-olur"))
    return built


class TestHypothesisArcs:
    def test_from_pair_below_accumulation_threshold(self) -> None:
        engine = HypothesisEngine(minimum_confidence=0.6)
        first = np.linspace(0, 10, 50)
        noise = np.random.default_rng(0).normal(scale=4.0, size=50)
        found = engine.from_pair("a", first, "b", first * 0.45 + noise)
        if found and found[0].confidence >= 0.6:
            assert len(engine.generated) == 1
        else:
            assert engine.generated == []

    def test_periodicity_short_circuit_on_tiny_series(self) -> None:
        engine = HypothesisEngine()
        four_points = np.array([1.0, -1.0, 1.0, -1.0])
        found = engine.from_series("tiny", four_points)
        assert isinstance(found, list)


class TestKnowledgeArcs:
    def test_shortest_path_unknown_endpoint(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="known concepts"):
            graph.shortest_path("Laplace", "Yok")

    def test_bfs_revisits_visited_neighbor(self) -> None:
        diamond = KnowledgeGraph()
        for name in ("A", "B", "C", "D"):
            diamond.add_concept(Concept(name))
        diamond.add_relation(Relation("A", "B"))
        diamond.add_relation(Relation("A", "C"))
        diamond.add_relation(Relation("B", "D"))
        diamond.add_relation(Relation("C", "D"))
        assert diamond.shortest_path("A", "D") == ["A", "B", "D"]

    def test_closure_with_back_edge_to_start(self) -> None:
        looped = KnowledgeGraph()
        for name in ("A", "B"):
            looped.add_concept(Concept(name))
        looped.add_relation(Relation("A", "B"))
        looped.add_relation(Relation("B", "A"))
        assert looped.transitive_closure("A") == {"B"}

    def test_has_cycle_partial_decrement_arc(self, graph: KnowledgeGraph) -> None:
        assert graph.has_cycle() is False

    def test_search_skips_notes_without_terms(self, graph: KnowledgeGraph) -> None:
        notebook = Notebook()
        notebook.add("ilgisiz", "hicbir anahtar kelime yok burada")
        results = search("olasilik", graph, notebook)
        assert all(not description.startswith("note:") for _score, description in results)


class TestModelingArcs:
    def test_wrap_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="boolean"):
            symbol("x") + True

    def test_wrap_text_rejected(self) -> None:
        with pytest.raises(TypeError, match="cannot wrap"):
            symbol("x") * "metin"

    def test_divide_latex(self) -> None:
        latex_text = to_latex(symbol("a") / symbol("b"))
        assert "\\frac{a}{b}" in latex_text

    def test_power_latex_parenthesizes_sums(self) -> None:
        latex_text = to_latex((symbol("a") + symbol("b")) ** 2)
        assert "(a + b)^{2}" in latex_text

    def test_negate_latex_parenthesizes_sums(self) -> None:
        latex_text = to_latex(Negate(symbol("a") + symbol("b")))
        assert "-(a + b)" in latex_text

    def test_diff_subtract(self) -> None:
        x = symbol("x")
        derivative = simplify(diff(x - 3 * x**2, "x"))
        assert evaluate(derivative, {"x": 1.0}) == pytest.approx(-5.0)

    def test_simplify_negate_constant_folds(self) -> None:
        folded = simplify(Negate(Constant(4.0)))
        assert folded.value == pytest.approx(-4.0)

    def test_divide_zero_numerator(self) -> None:
        x = symbol("x")
        result = simplify(Constant(0.0) / x)
        assert isinstance(result, Constant)

    def test_evaluate_negation(self) -> None:
        assert evaluate(Negate(symbol("x")), {"x": 2.5}) == pytest.approx(-2.5)

    def test_postorder_handles_negation(self) -> None:
        expression = Negate(symbol("q"))
        visited = [type(node).__name__ for node in iter_postorder(expression)]
        assert visited == ["Variable", "Negate"]

    def test_solve_equation_zero_derivative_breaks(self) -> None:
        model = MathModel.from_formula(Constant(3.0), Constant(3.0))
        solution = model.solve_equation(
            "missing_var", target_value=0.0, known={}, initial_guess=2.0
        )
        assert solution == pytest.approx(2.0)

    def test_to_string_unknown_node_raises(self) -> None:
        class Mystery(Expression):
            def __init__(self) -> None:
                pass

            def variables(self) -> set[str]:
                return set()

        with pytest.raises(NotImplementedError):
            to_string(Mystery())

    def test_substitute_walk_unknown_node_raises(self) -> None:
        class Mystery(Expression):
            def __init__(self) -> None:
                pass

            def variables(self) -> set[str]:
                return set()

        with pytest.raises(NotImplementedError):
            substitute(Mystery(), {})

    def test_evaluate_unknown_node_raises(self) -> None:
        class Mystery(Expression):
            def __init__(self) -> None:
                pass

            def variables(self) -> set[str]:
                return set()

        with pytest.raises(NotImplementedError):
            evaluate(Mystery(), {})


class TestAutogradArcs:
    def test_right_add_and_multiply(self) -> None:
        x = Value(2.0)
        y = 3 + x * 2
        y.backward()
        assert y.data == pytest.approx(7.0)
        assert x.grad == pytest.approx(2.0)

    def test_right_subtract(self) -> None:
        x = Value(4.0)
        difference = 10 - x
        difference.backward()
        assert difference.data == pytest.approx(6.0)
        assert x.grad == pytest.approx(-1.0)


class TestTokenizerArcs:
    def test_words_without_alphanumerics_skipped(self) -> None:
        tokenizer = BPETokenizer(merges=3).train("!!! ??? ...")
        assert tokenizer.vocabulary == set()

    def test_unique_pairs_stop_early(self) -> None:
        tokenizer = BPETokenizer(merges=50).train("a b c d e f g")
        assert tokenizer.merge_rules == []

    def test_encode_empty_word(self) -> None:
        tokenizer = BPETokenizer(merges=5).train("abc abc")
        assert tokenizer.encode("!!!") == []


class TestQuantumArcs:
    def test_y_gate(self) -> None:
        circuit = QuantumCircuit(1).h(0).y(0)
        assert circuit.probabilities()[0] == pytest.approx(0.5)

    def test_rz_gate_phase_only(self) -> None:
        circuit = QuantumCircuit(1).h(0).rz(0, np.pi)
        assert np.allclose(circuit.probabilities(), [0.5, 0.5])

    def test_cz_same_qubit_raises(self) -> None:
        with pytest.raises(ValueError, match="differ"):
            QuantumCircuit(2).cz(0, 0)

    def test_statevector_after_manual_none(self) -> None:
        circuit = QuantumCircuit(1)
        circuit._state = None
        assert circuit.statevector()[0] == pytest.approx(1.0)

    def test_ry_matrix_values(self) -> None:
        circuit = QuantumCircuit(1).ry(0, np.pi)
        assert abs(circuit.statevector()[0]) == pytest.approx(0.0, abs=1e-12)


class TestScientificValidation:
    def test_coulomb_distance_validation(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            scientific.coulomb_force(1.0, 1.0, -1.0)

    def test_photon_wavelength_validation(self) -> None:
        with pytest.raises(ValueError, match="wavelength"):
            scientific.photon_energy(0.0)

    def test_de_broglie_mass_validation(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            scientific.de_broglie_wavelength(0.0, 1.0)

    def test_escape_velocity_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            scientific.escape_velocity(0.0, 1.0)
        with pytest.raises(ValueError, match="positive"):
            scientific.escape_velocity(1.0, 0.0)
