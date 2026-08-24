"""Tests for cds2.genetics."""

import dataclasses

import pytest

from cds2 import genetics as gen


class TestGcContent:
    def test_known_fractions(self) -> None:
        assert gen.gc_content("GGCC") == 1.0
        assert gen.gc_content("AAAA") == 0.0
        assert gen.gc_content("GCAT") == 0.5

    def test_lowercase_folded(self) -> None:
        assert gen.gc_content("gcAt") == 0.5

    def test_n_bases_excluded(self) -> None:
        assert gen.gc_content("NGCN") == 1.0
        assert gen.gc_content("NGNT") == 0.5
        assert gen.gc_content("NNAT") == 0.0

    def test_empty_and_all_n_are_zero(self) -> None:
        assert gen.gc_content("") == 0.0
        assert gen.gc_content("NNNN") == 0.0

    def test_invalid_nucleotide_reports_offender(self) -> None:
        with pytest.raises(ValueError, match=r"invalid nucleotide 'X'"):
            gen.gc_content("ACXT")


class TestHammingDistance:
    def test_known_differences(self) -> None:
        assert gen.hamming_distance("AAAAAA", "AAATAA") == 1
        assert gen.hamming_distance("ACGT", "TGCA") == 4

    def test_lowercase_folded(self) -> None:
        assert gen.hamming_distance("gattaca", "GACTATA") == 2

    def test_identical_zero(self) -> None:
        assert gen.hamming_distance("ATG", "ATG") == 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            gen.hamming_distance("ATG", "AT")


class TestReverseComplement:
    def test_case_folded_translation(self) -> None:
        assert gen.reverse_complement("atGCn") == "NGCAT"

    def test_palindrome_roundtrip(self) -> None:
        for seq in ("ACGTN", "TTTT", "GANTC"):
            assert gen.reverse_complement(gen.reverse_complement(seq)) == seq


class TestKmerCounts:
    def test_exact_dict(self) -> None:
        assert gen.kmer_counts("ATATA", 2) == {"AT": 2, "TA": 2}

    def test_lowercase_matches_upper(self) -> None:
        assert gen.kmer_counts("atata", 2) == {"AT": 2, "TA": 2}

    def test_single_base_counts(self) -> None:
        assert gen.kmer_counts("AAA", 1) == {"A": 3}

    def test_k_larger_than_sequence_is_empty(self) -> None:
        assert gen.kmer_counts("ATG", 5) == {}

    def test_k_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            gen.kmer_counts("ATG", 0)


class TestGlobalAlign:
    def test_hand_checked_gap_in_b(self) -> None:
        result = gen.global_align("CAT", "CT")
        assert result.score == 1
        assert result.aligned_a == "CAT"
        assert result.aligned_b == "C-T"
        assert result.identity == pytest.approx(2 / 3)

    def test_hand_checked_gap_in_a(self) -> None:
        result = gen.global_align("CT", "CAT")
        assert result.score == 1
        assert result.aligned_a == "C-T"
        assert result.aligned_b == "CAT"
        assert result.identity == pytest.approx(2 / 3)

    def test_ties_prefer_diagonal_with_leading_gap(self) -> None:
        first = gen.global_align("A", "AA")
        assert (first.aligned_a, first.aligned_b) == ("-A", "AA")
        second = gen.global_align("AA", "A")
        assert (second.aligned_a, second.aligned_b) == ("AA", "-A")

    def test_identical_sequences_full_identity(self) -> None:
        result = gen.global_align("GATTACA", "GATTACA")
        assert result.score == 7
        assert result.identity == 1.0
        assert result.aligned_a == "GATTACA"
        assert result.aligned_b == "GATTACA"

    def test_all_mismatches(self) -> None:
        result = gen.global_align("GT", "AC")
        assert result.score == -2
        assert result.identity == 0.0
        assert (result.aligned_a, result.aligned_b) == ("GT", "AC")

    def test_custom_scores(self) -> None:
        result = gen.global_align("ACG", "AGG", match_score=2, mismatch_score=-2, gap_penalty=-2)
        assert result.score == 2
        assert (result.aligned_a, result.aligned_b) == ("ACG", "AGG")
        assert result.identity == pytest.approx(2 / 3)

    def test_result_is_frozen(self) -> None:
        result = gen.global_align("A", "A")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.score = 99

    def test_empty_first_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            gen.global_align("", "ACG")

    def test_empty_second_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            gen.global_align("ACG", "")


class TestFindOrfs:
    def test_single_orf_protein(self) -> None:
        assert gen.find_orfs("ATGAAATAA") == [(0, 6, "MK")]

    def test_lowercase_input(self) -> None:
        assert gen.find_orfs("atgaaataa") == [(0, 6, "MK")]

    def test_no_stop_codon_yields_nothing(self) -> None:
        assert gen.find_orfs("ATGAAAGGGCCC") == []

    def test_too_short_for_stop(self) -> None:
        assert gen.find_orfs("ATG") == []

    def test_nested_start_codons_overlap(self) -> None:
        assert gen.find_orfs("ATGATGTAA") == [(0, 6, "MM")]

    def test_unknown_codon_translates_as_x(self) -> None:
        assert gen.find_orfs("ATGANNTAA") == [(0, 6, "MX")]

    def test_min_length_filters_short_orfs(self) -> None:
        seq = "ATGAAATAA" + "ATGGGGCAATAA"
        assert gen.find_orfs(seq) == [(0, 6, "MK"), (9, 18, "MGQ")]
        assert gen.find_orfs(seq, min_length=9) == [(9, 18, "MGQ")]

    def test_min_length_below_six_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 6"):
            gen.find_orfs("ATGAAATAA", min_length=5)
