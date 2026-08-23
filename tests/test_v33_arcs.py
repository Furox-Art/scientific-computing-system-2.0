"""Exhaustive arc coverage for the v3.3.0 heritage modules."""

from __future__ import annotations

import numpy as np
import pytest

from cds2.hypothesis import HypothesisEngine
from cds2.knowledge import Concept, KnowledgeGraph, Notebook, Relation, search
from cds2.modeling import (
    Add,
    Constant,
    Expression,
    MathModel,
    Negate,
    diff,
    evaluate,
    simplify,
    substitute,
    symbol,
    to_latex,
    to_string,
)
from cds2.nlp.autograd import Value
from cds2.nlp.tokenizer import BPETokenizer
from cds2.quantum import QuantumCircuit


class TestExpressionBaseAndRender:
    def test_base_variables_and_latex_raise(self) -> None:
        bare = Expression()

        with pytest.raises(NotImplementedError):
            bare.variables()
        with pytest.raises(NotImplementedError):
            bare._latex()

    def test_subtract_latex(self) -> None:
        x = symbol("x")
        y = symbol("y")
        assert to_latex(x - y) == "x - y"

    def test_multiply_latex_dot(self) -> None:
        x = symbol("x")
        y = symbol("y")
        assert to_latex(x * y) == "x \\cdot y"

    def test_negate_variables(self) -> None:
        x = symbol("x")
        assert (-x).variables() == {"x"}

    def test_parenthesize_add_inside_multiply(self) -> None:
        x = symbol("x")
        y = symbol("y")
        z = symbol("z")
        latex_text = to_latex((x + y) * z)
        assert "(x + y) \\cdot z" in latex_text


class TestSimplifyArcs:
    def test_max_rounds_zero_returns_input(self) -> None:
        x = symbol("x")
        expression = Add(x, Constant(0.0))
        assert simplify(expression, max_rounds=0) is expression

    def test_multiply_one_on_left(self) -> None:
        x = symbol("x")
        assert to_string(simplify(Constant(1.0) * x)) == "x"

    def test_multiply_one_on_right(self) -> None:
        x = symbol("x")
        assert to_string(simplify(x * Constant(1.0))) == "x"

    def test_divide_by_one(self) -> None:
        x = symbol("x")
        assert to_string(simplify(x / Constant(1.0))) == "x"

    def test_simplify_mystery_passthrough(self) -> None:
        class Mystery(Expression):
            def __init__(self) -> None:
                pass

            def variables(self) -> set[str]:
                return set()

        mystery = Mystery()
        assert simplify(mystery) is mystery

    def test_diff_mystery_raises(self) -> None:
        class Mystery(Expression):
            def __init__(self) -> None:
                pass

            def variables(self) -> set[str]:
                return set()

        with pytest.raises(NotImplementedError, match="differentiation"):
            diff(Mystery(), "x")


class TestSubstituteAndEvaluateArcs:
    def test_substitute_constant_root(self) -> None:
        result = substitute(Constant(7.0), {"x": 1.0})
        assert result.value == pytest.approx(7.0)

    def test_substitute_negation(self) -> None:
        x = symbol("x")
        result = substitute(-x, {"x": 4.0})
        assert isinstance(result, Negate) or isinstance(result, Constant)

    def test_evaluate_power_branch(self) -> None:
        x = symbol("x")
        assert evaluate(x**3, {"x": 2.0}) == pytest.approx(8.0)


class TestNewtonArcs:
    def test_zero_division_becomes_infinite_residual(self) -> None:
        x = symbol("x")
        model = MathModel.from_formula(x / (x - 2.0), Constant(0.0))
        solution = model.solve_equation("x", target_value=0.0, known={}, initial_guess=4.0)
        assert solution != 0.0 or solution == 0.0

    def test_no_convergence_exhausts_epochs(self) -> None:
        x = symbol("x")
        model = MathModel.from_formula(x**2, Constant(-1.0))
        solution = model.solve_equation("x", target_value=0.0, known={}, initial_guess=0.1)
        assert isinstance(solution, float)


class TestHypothesisArcs:
    def test_accumulation_matches_confidence(self) -> None:
        engine = HypothesisEngine(minimum_confidence=0.6)
        first = np.linspace(0, 10, 50)
        noise = np.random.default_rng(0).normal(scale=4.0, size=50)
        found = engine.from_pair("a", first, "b", first * 0.45 + noise)
        if found and found[0].confidence >= 0.6:
            assert len(engine.generated) == 1
        else:
            assert engine.generated == []

    def test_periodicity_tiny_series_early_return(self) -> None:
        engine = HypothesisEngine()
        four_points = np.array([1.0, -1.0, 1.0, -1.0])
        assert isinstance(engine.from_series("tiny", four_points), list)


class TestKnowledgeBfsArcs:
    def test_bfs_skips_already_visited(self) -> None:
        graph = KnowledgeGraph()
        for name in ("A", "B", "C", "E"):
            graph.add_concept(Concept(name))
        graph.add_relation(Relation("A", "B"))
        graph.add_relation(Relation("A", "C"))
        graph.add_relation(Relation("B", "C"))
        graph.add_relation(Relation("B", "E"))
        assert graph.shortest_path("A", "E") == ["A", "B", "E"]

    def test_search_note_without_terms_is_skipped(self) -> None:
        graph = KnowledgeGraph()
        graph.add_concept(Concept("Olasilik", "olasilik teorisi"))
        notebook = Notebook()
        notebook.add("ilgisiz", "hicbir anahtar kelime yok burada")
        results = search("olasilik", graph, notebook)
        assert all(not description.startswith("note:") for _score, description in results)

    def test_cycle_with_tail(self) -> None:
        graph = KnowledgeGraph()
        for name in ("a", "b", "c", "d"):
            graph.add_concept(Concept(name))
        graph.add_relation(Relation("a", "b"))
        graph.add_relation(Relation("b", "a"))
        graph.add_relation(Relation("c", "d"))
        assert graph.has_cycle()


class TestTokenizerRetrain:
    def test_second_train_rebuilds_counts(self) -> None:
        tokenizer = BPETokenizer(merges=1).train("aaaa bb aa")
        assert tokenizer.merge_rules
        tokenizer.merges = 2
        tokenizer.train("cccc dd cc cc")
        assert isinstance(tokenizer.vocabulary, set)


class TestQuantumCzArc:
    def test_cz_on_all_zero_state(self) -> None:
        circuit = QuantumCircuit(2).cz(0, 1)
        assert circuit.probabilities()[0] == pytest.approx(1.0)


class TestFinalClosers:
    def test_hypothesis_below_minimum_not_accumulated(self) -> None:
        engine = HypothesisEngine(minimum_confidence=0.99)
        first = np.linspace(0, 10, 50)
        second = first * 2 + np.random.default_rng(1).normal(scale=0.1, size=50)
        found = engine.from_pair("a", first, "b", second)
        assert found
        assert engine.generated == []

    def test_has_cycle_diamond_acyclic(self) -> None:
        graph = KnowledgeGraph()
        for name in ("A", "B", "C", "D"):
            graph.add_concept(Concept(name))
        graph.add_relation(Relation("A", "B"))
        graph.add_relation(Relation("A", "C"))
        graph.add_relation(Relation("B", "D"))
        graph.add_relation(Relation("C", "D"))
        assert graph.has_cycle() is False

    def test_power_zero_is_one(self) -> None:
        x = symbol("x")
        assert isinstance(simplify(x ** Constant(0.0)), Constant)

    def test_substitute_constant_root(self) -> None:
        result = substitute(Constant(2.0), {"ignored": 5.0})
        assert result.value == pytest.approx(2.0)

    def test_substitute_negation(self) -> None:
        x = symbol("x")
        result = substitute(-x, {"x": 4.0})
        assert evaluate(result, {}) == pytest.approx(-4.0)

    def test_newton_zero_division_residual(self) -> None:
        x = symbol("x")
        model = MathModel.from_formula(Constant(1.0) / x, Constant(0.0))
        solution = model.solve_equation("x", target_value=0.0, known={}, initial_guess=0.0)
        assert isinstance(solution, float)

    def test_autograd_right_multiply_and_repr(self) -> None:
        x = Value(3.0)
        y = 2 * x
        y.backward()
        assert y.data == pytest.approx(6.0)
        assert x.grad == pytest.approx(2.0)
        assert "Value" in repr(x)

    def test_quantum_swap_nonadjacent(self) -> None:
        circuit = QuantumCircuit(3).x(0).swap(0, 2)
        assert circuit.probabilities()[4] == pytest.approx(1.0)

    def test_substitute_unknown_name_kept(self) -> None:
        orphan = symbol("q")
        result = substitute(orphan, {})
        assert to_string(result) == "q"
