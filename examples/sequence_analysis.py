"""Case study 13: DNA sequence analysis with cds2.genetics.

A parent sequence is mutated into a child strain; alignment quantifies the
edit pattern while k-mer and ORF scans summarise composition and coding
potential.
"""

from __future__ import annotations

import cds2

PARENT = "ATGCGTACGTTAGCCGATTACAGGTACCGGATTCAGCATGCCTAAGGTCAGTTACGGATCCTAG"
MUTATION_SITES = {7: "A", 22: "T", 41: "G"}


def mutate(sequence: str) -> str:
    chars = list(sequence)
    for site, base in MUTATION_SITES.items():
        chars[site] = base
    return "".join(chars)


def main() -> None:
    child = mutate(PARENT)

    print("== Sequence composition ==")
    print(f"parent GC content : {cds2.genetics.gc_content(PARENT):.3f}")
    print(f"child  GC content : {cds2.genetics.gc_content(child):.3f}")

    prefix = PARENT[: len(PARENT) // 2]
    distance = cds2.genetics.hamming_distance(prefix, child[: len(prefix)])
    print(f"\nhamming distance on first {len(prefix)} bases: {distance}")

    print("\n== Global alignment (child vs parent) ==")
    result = cds2.genetics.global_align(child, PARENT)
    print(f"score    : {result.score}")
    print(f"identity : {result.identity:.3f}")
    for start in range(0, len(result.aligned_a), 40):
        print(f"  {result.aligned_a[start : start + 40]}")
        print(f"  {result.aligned_b[start : start + 40]}")
        print()

    counts = cds2.genetics.kmer_counts(PARENT, k=3)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    print("top trimers:", ", ".join(f"{mer}x{count}" for mer, count in top))

    orfs = cds2.genetics.find_orfs(PARENT, min_length=12)
    print(f"\nforward ORFs (>=12 nt): {len(orfs)}")
    if orfs:
        start, end, protein = orfs[0]
        print(f"first ORF [{start}:{end}] protein: {protein}")

    print(f"\nrev-comp of first 21 nt: {cds2.genetics.reverse_complement(PARENT[:21])}")
    print(f"original first 21 nt   : {PARENT[:21]}")


if __name__ == "__main__":
    main()
