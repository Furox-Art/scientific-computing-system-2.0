"""DNA sequence analysis: composition, k-mers, global alignment and ORF finding."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AlignmentResult",
    "gc_content",
    "hamming_distance",
    "reverse_complement",
    "kmer_counts",
    "global_align",
    "find_orfs",
]

_VALID_NUCLEOTIDES: frozenset[str] = frozenset("ACGTN")
_STOP_CODONS: frozenset[str] = frozenset({"TAA", "TAG", "TGA"})
_COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
_CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def _clean(seq: str) -> str:
    """Uppercase a sequence and reject any character outside ``ACGTN``."""
    sequence = seq.upper()
    for ch in sequence:
        if ch not in _VALID_NUCLEOTIDES:
            msg = f"invalid nucleotide {ch!r}"
            raise ValueError(msg)
    return sequence


def gc_content(seq: str) -> float:
    """Fraction of G/C bases among non-N bases; 0.0 when empty or all-N."""
    sequence = _clean(seq)
    counted = [ch for ch in sequence if ch != "N"]
    if not counted:
        return 0.0
    gc = sum(1 for ch in counted if ch in "GC")
    return gc / len(counted)


def hamming_distance(a: str, b: str) -> int:
    """Number of differing positions between two equal-length sequences."""
    seq_a = _clean(a)
    seq_b = _clean(b)
    if len(seq_a) != len(seq_b):
        msg = "sequences must have equal length"
        raise ValueError(msg)
    return sum(ca != cb for ca, cb in zip(seq_a, seq_b))


def reverse_complement(seq: str) -> str:
    """Reverse complement of the sequence (A<->T, C<->G, N preserved)."""
    return "".join(_COMPLEMENT[ch] for ch in reversed(_clean(seq)))


def kmer_counts(seq: str, k: int) -> dict[str, int]:
    """Histogram of all length-``k`` sliding-window substrings."""
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)
    sequence = _clean(seq)
    counts: dict[str, int] = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


@dataclass(frozen=True)
class AlignmentResult:
    """Optimal global alignment of two sequences.

    ``identity`` is the fraction of aligned columns whose residues match;
    gap characters ``'-'`` always count as mismatches.
    """

    score: int
    identity: float
    aligned_a: str
    aligned_b: str


def global_align(
    a: str, b: str, match_score: int = 1, mismatch_score: int = -1, gap_penalty: int = -1
) -> AlignmentResult:
    """Needleman-Wunsch global alignment with full dynamic-programming traceback.

    Traceback ties prefer the diagonal (substitution) move; gaps are rendered
    as ``'-'`` and count as mismatches toward ``identity``.
    """
    seq_a = _clean(a)
    seq_b = _clean(b)
    if not seq_a or not seq_b:
        msg = "sequences must be non-empty"
        raise ValueError(msg)
    rows, cols = len(seq_a), len(seq_b)
    dp: list[list[int]] = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        dp[i][0] = i * gap_penalty
    for j in range(1, cols + 1):
        dp[0][j] = j * gap_penalty
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            substitution = match_score if seq_a[i - 1] == seq_b[j - 1] else mismatch_score
            dp[i][j] = max(
                dp[i - 1][j - 1] + substitution,
                dp[i - 1][j] + gap_penalty,
                dp[i][j - 1] + gap_penalty,
            )

    aligned_a_chars: list[str] = []
    aligned_b_chars: list[str] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution = match_score if seq_a[i - 1] == seq_b[j - 1] else mismatch_score
            if dp[i][j] == dp[i - 1][j - 1] + substitution:
                aligned_a_chars.append(seq_a[i - 1])
                aligned_b_chars.append(seq_b[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
            aligned_a_chars.append(seq_a[i - 1])
            aligned_b_chars.append("-")
            i -= 1
        else:
            aligned_a_chars.append("-")
            aligned_b_chars.append(seq_b[j - 1])
            j -= 1
    aligned_a = "".join(reversed(aligned_a_chars))
    aligned_b = "".join(reversed(aligned_b_chars))
    matches = sum(1 for ca, cb in zip(aligned_a, aligned_b) if ca == cb)
    return AlignmentResult(
        score=dp[rows][cols],
        identity=matches / len(aligned_a),
        aligned_a=aligned_a,
        aligned_b=aligned_b,
    )


def find_orfs(seq: str, min_length: int = 6) -> list[tuple[int, int, str]]:
    """Forward-strand open reading frames from ``ATG`` to the first in-frame stop.

    Returns ``(start_zero_based, end_excluding_stop, protein)`` tuples where
    ``protein`` translates the codons in ``[start, end)`` via the standard
    table (unknown/N-containing codons become ``X``).  ``min_length`` compares
    the coding nucleotide length excluding the stop codon.
    """
    if min_length < 6:
        msg = "min_length must be at least 6"
        raise ValueError(msg)
    sequence = _clean(seq)
    orfs: list[tuple[int, int, str]] = []
    for start in range(len(sequence) - 2):
        if sequence[start : start + 3] != "ATG":
            continue
        for pos in range(start + 3, len(sequence) - 2, 3):
            if sequence[pos : pos + 3] not in _STOP_CODONS:
                continue
            protein = "".join(
                _CODON_TABLE.get(sequence[i : i + 3], "X") for i in range(start, pos, 3)
            )
            if pos - start >= min_length:
                orfs.append((start, pos, protein))
            break
    return orfs
