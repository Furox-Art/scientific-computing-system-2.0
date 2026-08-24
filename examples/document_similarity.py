"""Case study 19: Document similarity search.

Five abstracts are converted into a smoothed TF-IDF space; pairwise cosine
similarities reveal topical clusters and a Jaccard check contrasts set-based
overlap with weighted similarity.
"""

from __future__ import annotations

import numpy as np

import cds2

ABSTRACTS = [
    "deep learning models improve image classification accuracy on large datasets",
    "convolutional networks detect objects in satellite images with high precision",
    "bayesian inference estimates parameter uncertainty in statistical models",
    "monte carlo methods approximate posterior distributions for bayesian models",
    "wavelet denoising removes gaussian noise from biomedical signals",
]


def main() -> None:
    result = cds2.text.tfidf_matrix(ABSTRACTS)
    print("== TF-IDF space ==")
    print(f"documents: {result.matrix.shape[0]}, vocabulary: {result.matrix.shape[1]}")

    matrix = result.matrix
    n_docs = matrix.shape[0]
    similarities = np.zeros((n_docs, n_docs))
    for i in range(n_docs):
        for j in range(n_docs):
            similarities[i, j] = cds2.text.cosine_similarity(matrix[i].tolist(), matrix[j].tolist())

    print("\n== Pairwise cosine similarity ==")
    header = "      " + "".join(f"doc{j + 1:<7d}" for j in range(n_docs))
    print(header)
    for i in range(n_docs):
        row = "".join(f"{similarities[i, j]:<8.3f}" for j in range(n_docs))
        print(f"doc{i + 1}  {row}")

    pairs = sorted(
        ((i, j) for i in range(n_docs) for j in range(i + 1, n_docs)),
        key=lambda pair: -similarities[pair[0], pair[1]],
    )
    best = pairs[0]
    print(
        f"\nmost similar pair : doc{best[0] + 1} & doc{best[1] + 1} "
        f"({similarities[best[0], best[1]]:.3f})"
    )

    jaccard = cds2.text.jaccard_similarity(ABSTRACTS[0], ABSTRACTS[1])
    cosine_pair = similarities[0, 1]
    print(f"doc1 vs doc2      : cosine {cosine_pair:.3f} vs jaccard {jaccard:.3f}")

    summary = cds2.text.summarize_terms(ABSTRACTS, top_k=6)
    print("\n== Corpus-defining terms ==")
    for term, score in summary.top_terms:
        bar = "#" * int(score * 60)
        print(f"{term:<14s} {score:.3f} |{bar}")


if __name__ == "__main__":
    main()
