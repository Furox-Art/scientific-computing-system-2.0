"""Lightweight text analysis: tokenization, TF-IDF weighting and similarities."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FloatArray",
    "SummaryResult",
    "TfidfResult",
    "cosine_similarity",
    "jaccard_similarity",
    "summarize_terms",
    "term_frequencies",
    "tfidf_matrix",
    "tokenize",
]

FloatArray = NDArray[np.float64]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens of ``text`` (punctuation dropped)."""
    return _TOKEN_PATTERN.findall(text.lower())


def term_frequencies(tokens: Sequence[str]) -> dict[str, int]:
    """Occurrence count of every distinct token."""
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


@dataclass(frozen=True)
class TfidfResult:
    """TF-IDF matrix together with the vocabulary indexing its columns."""

    matrix: FloatArray
    vocabulary: list[str]


def tfidf_matrix(documents: Sequence[str]) -> TfidfResult:
    """Smoothed TF-IDF matrix (rows = documents, columns = sorted vocabulary).

    Term frequency is ``count / len(doc_tokens)`` (zero for token-less
    documents); inverse document frequency uses the smoothed form
    ``ln((1 + N) / (1 + df)) + 1``.
    """
    doc_tokens = [tokenize(document) for document in documents]
    vocabulary = sorted({token for tokens in doc_tokens for token in tokens})
    if not vocabulary:
        msg = "documents must contain at least one token"
        raise ValueError(msg)
    column_of = {term: column for column, term in enumerate(vocabulary)}
    counts = np.zeros((len(doc_tokens), len(vocabulary)), dtype=float)
    for row, tokens in enumerate(doc_tokens):
        for term, count in term_frequencies(tokens).items():
            counts[row, column_of[term]] = count
    lengths = np.asarray([len(tokens) for tokens in doc_tokens], dtype=float)
    present = lengths > 0
    tf = np.zeros_like(counts)
    tf[present] = counts[present] / lengths[present, None]
    document_frequency = np.count_nonzero(counts, axis=0).astype(float)
    idf = np.log((1.0 + len(doc_tokens)) / (1.0 + document_frequency)) + 1.0
    return TfidfResult(matrix=tf * idf, vocabulary=vocabulary)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors; 0.0 when a norm vanishes."""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        msg = "vectors must have equal length"
        raise ValueError(msg)
    norm_product = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norm_product == 0.0:
        return 0.0
    return float((left @ right) / norm_product)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity of the token sets; empty unions score 0.0."""
    set_a = set(tokenize(text_a))
    set_b = set(tokenize(text_b))
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


@dataclass(frozen=True)
class SummaryResult:
    """Highest scoring terms according to mean TF-IDF weight."""

    top_terms: list[tuple[str, float]]


def summarize_terms(documents: Sequence[str], top_k: int = 10) -> SummaryResult:
    """Mean TF-IDF score per term across documents, best ``top_k`` first."""
    if top_k < 1:
        msg = "top_k must be at least 1"
        raise ValueError(msg)
    result = tfidf_matrix(documents)
    scores = [float(score) for score in result.matrix.mean(axis=0)]
    ranked = sorted(
        zip(result.vocabulary, scores, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    return SummaryResult(top_terms=[(term, score) for term, score in ranked[:top_k]])
