"""Byte-pair encoding tokenizer trained on a small corpus."""

from __future__ import annotations

__all__ = ["BPETokenizer"]


class BPETokenizer:
    """Trainable sub-word tokenizer using greedy byte-pair merges."""

    END_OF_WORD = "</w>"

    def __init__(self, merges: int = 64) -> None:
        if merges < 1:
            msg = "merges must be at least 1"
            raise ValueError(msg)
        self.merges = merges
        self.merge_rules: list[tuple[str, str]] = []
        self.vocabulary: set[str] = set()

    def train(self, corpus: str) -> BPETokenizer:
        """Learn ``merges`` pair rules from whitespace-split words."""
        words: list[tuple[str, ...]] = []
        for raw_word in corpus.lower().split():
            cleaned = "".join(character for character in raw_word if character.isalnum())
            if cleaned:
                symbols = tuple(cleaned) + (self.END_OF_WORD,)
                words.append(symbols)

        pair_counts: dict[tuple[str, str], int] = {}
        for word in words:
            for index in range(len(word) - 1):
                pair_counts[word[index], word[index + 1]] = (
                    pair_counts.get((word[index], word[index + 1]), 0) + 1
                )

        for _ in range(self.merges):
            if not pair_counts:
                break
            best_pair, best_count = max(pair_counts.items(), key=lambda item: item[1])
            if best_count < 2:
                break
            self.merge_rules.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            self.vocabulary.add(merged)

            words = [self._merge_in_word(word, best_pair) for word in words]
            pair_counts = {}
            for word in words:
                for index in range(len(word) - 1):
                    pair_counts[word[index], word[index + 1]] = (
                        pair_counts.get((word[index], word[index + 1]), 0) + 1
                    )

        for word in words:
            for symbol in word:
                self.vocabulary.add(symbol)
        return self

    @staticmethod
    def _merge_in_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        merged_symbol = pair[0] + pair[1]
        output: list[str] = []
        index = 0
        while index < len(word):
            if index < len(word) - 1 and word[index] == pair[0] and word[index + 1] == pair[1]:
                output.append(merged_symbol)
                index += 2
            else:
                output.append(word[index])
                index += 1
        return tuple(output)

    def encode(self, word: str) -> list[str]:
        """Split ``word`` into learned sub-word symbols."""
        symbols = tuple("".join(character for character in word.lower() if character.isalnum()))
        if not symbols:
            return []
        symbols = symbols + (self.END_OF_WORD,)
        for pair in self.merge_rules:
            symbols = self._merge_in_word(symbols, pair)
        return list(symbols)

    def decode(self, tokens: list[str]) -> str:
        """Reconstruct text by joining symbols and dropping end markers."""
        text = "".join(tokens)
        return text.replace(self.END_OF_WORD, " ").strip()
