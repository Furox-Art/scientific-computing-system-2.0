"""Tests for cds2.text."""

import math

import numpy as np
import pytest

from cds2 import text as txt


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self) -> None:
        assert txt.tokenize("Hello, World! It's 2026.") == [
            "hello",
            "world",
            "it",
            "s",
            "2026",
        ]

    def test_empty_string(self) -> None:
        assert txt.tokenize("") == []

    def test_whitespace_only(self) -> None:
        assert txt.tokenize("   \t\n ") == []


class TestTermFrequencies:
    def test_counts_distinct_tokens(self) -> None:
        assert txt.term_frequencies(["a", "b", "a", "c", "a"]) == {"a": 3, "b": 1, "c": 1}

    def test_empty_token_list(self) -> None:
        assert txt.term_frequencies([]) == {}


class TestTfidfMatrix:
    def test_hand_checked_two_document_corpus(self) -> None:
        result = txt.tfidf_matrix(["apple banana apple", "banana cherry"])
        assert result.vocabulary == ["apple", "banana", "cherry"]
        assert result.matrix.shape == (2, 3)
        idf_rare = math.log((1 + 2) / (1 + 1)) + 1
        expected = np.array(
            [
                [(2 / 3) * idf_rare, 1 / 3, 0.0],
                [0.0, (1 / 2) * 1.0, (1 / 2) * idf_rare],
            ]
        )
        np.testing.assert_allclose(result.matrix, expected)

    def test_vocabulary_is_sorted_and_unique(self) -> None:
        result = txt.tfidf_matrix(["zebra apple zebra"])
        assert result.vocabulary == ["apple", "zebra"]

    def test_document_without_tokens_gets_zero_row(self) -> None:
        result = txt.tfidf_matrix(["apple", "   "])
        assert result.matrix[0, 0] > 0
        assert result.matrix[1, 0] == 0.0

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            txt.tfidf_matrix([])

    def test_whitespace_only_documents_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            txt.tfidf_matrix(["   ", "\n\t"])

    def test_result_is_frozen_dataclass(self) -> None:
        result = txt.tfidf_matrix(["word"])
        with pytest.raises(AttributeError):
            result.vocabulary = []  # type: ignore[misc]


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert txt.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert txt.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert txt.cosine_similarity([0.0, 0.0], [3.0, 4.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            txt.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


class TestJaccardSimilarity:
    def test_overlap_math(self) -> None:
        assert txt.jaccard_similarity("a b c", "b c d") == pytest.approx(2 / 4)

    def test_duplicates_collapse_to_sets(self) -> None:
        assert txt.jaccard_similarity("x x y", "x y y") == 1.0

    def test_disjoint_texts_zero(self) -> None:
        assert txt.jaccard_similarity("cat", "dog") == 0.0

    def test_both_empty_union_scores_zero_by_convention(self) -> None:
        assert txt.jaccard_similarity("!!!", "???") == 0.0


class TestSummarizeTerms:
    def test_mean_tfidf_ordering_descending(self) -> None:
        summary = txt.summarize_terms(["banana banana apple", "cherry apple"], top_k=3)
        idf_rare = math.log((1 + 2) / (1 + 1)) + 1
        expected = [
            ("banana", ((2 / 3) * idf_rare + 0.0) / 2),
            ("apple", ((1 / 3) * 1.0 + (1 / 2) * 1.0) / 2),
            ("cherry", ((1 / 2) * idf_rare) / 2),
        ]
        for (term, score), (expected_term, expected_score) in zip(
            summary.top_terms, expected, strict=True
        ):
            assert term == expected_term
            assert score == pytest.approx(expected_score)

    def test_ties_broken_alphabetically(self) -> None:
        summary = txt.summarize_terms(["beta alpha"])
        assert [term for term, _ in summary.top_terms] == ["alpha", "beta"]

    def test_top_k_clamps_to_vocabulary_size(self) -> None:
        summary = txt.summarize_terms(["alpha beta"], top_k=99)
        assert len(summary.top_terms) == 2

    def test_default_top_k_caps_at_ten(self) -> None:
        corpus = " ".join(f"term{i}" for i in range(12))
        summary = txt.summarize_terms([corpus])
        assert len(summary.top_terms) == 10

    def test_top_k_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            txt.summarize_terms(["word"], top_k=0)

    def test_result_is_frozen_dataclass(self) -> None:
        summary = txt.summarize_terms(["word"])
        with pytest.raises(AttributeError):
            summary.top_terms = []  # type: ignore[misc]
