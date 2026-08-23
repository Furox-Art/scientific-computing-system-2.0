"""Tests for the educational NLP toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from cds2.nlp import (
    BPETokenizer,
    TinyGPT,
    Value,
    multi_head_attention,
    scaled_dot_product_attention,
    softmax,
)


class TestAutograd:
    def test_gradient_of_product_plus_tanh(self) -> None:
        a = Value(2.0)
        b = Value(-3.0)
        c = a * b + a.tanh()
        c.backward()
        expected = b.data + (1 - np.tanh(a.data) ** 2)
        assert a.grad == pytest.approx(expected)
        assert b.grad == pytest.approx(a.data)

    def test_division_and_subtraction(self) -> None:
        x = Value(6.0)
        y = Value(2.0)
        z = (x - y) / y
        z.backward()
        assert z.data == pytest.approx(2.0)
        assert x.grad == pytest.approx(0.5)
        assert y.grad == pytest.approx(-1.5)

    def test_power_and_exp(self) -> None:
        x = Value(1.5)
        y = x.exp() * x**3
        y.backward()
        expected = float(np.exp(1.5) * (1.5**3 + 3 * 1.5**2))
        assert x.grad == pytest.approx(expected, rel=1e-9)

    def test_relu_zeroes_negative(self) -> None:
        x = Value(-2.0)
        z = x.relu()
        z.backward()
        assert z.data == 0.0
        assert x.grad == 0.0

    def test_constant_power_value_raises(self) -> None:
        with pytest.raises(TypeError, match="constant exponents"):
            Value(2.0) ** Value(3.0)

    def test_unsupported_operand(self) -> None:
        with pytest.raises(TypeError, match="unsupported"):
            Value(1.0) + "text"

    def test_second_order_branch_accumulation(self) -> None:
        x = Value(3.0)
        y = x * x + x
        y.backward()
        assert x.grad == pytest.approx(7.0)


class TestBPETokenizer:
    def test_train_and_encode(self) -> None:
        tokenizer = BPETokenizer(merges=30).train("the cat sat on the mat the cat ate the rat")
        tokens = tokenizer.encode("the cat")
        assert tokens
        assert tokens and tokenizer.decode(tokens)

    def test_decode_roundtrip_letters(self) -> None:
        tokenizer = BPETokenizer(merges=10).train("hello world hello")
        tokens = tokenizer.encode("hello")
        assert "".join(tokens).replace(tokenizer.END_OF_WORD, "") == "hello"

    def test_non_alphanumeric_stripped(self) -> None:
        tokenizer = BPETokenizer(merges=5).train("abc")
        assert tokenizer.encode("a!b?c") == list("abc") + [tokenizer.END_OF_WORD]

    def test_invalid_merges_raise(self) -> None:
        with pytest.raises(ValueError, match="merges"):
            BPETokenizer(merges=0)


class TestAttention:
    def test_softmax_normalizes(self) -> None:
        weights = softmax(np.array([[1.0, 2.0, 3.0]]))
        assert weights.sum() == pytest.approx(1.0)
        assert weights[0, -1] > weights[0, 0]

    def test_softmax_numerical_stability(self) -> None:
        weights = softmax(np.array([[1000.0, 1001.0]]))
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_scaled_dot_product_shapes_and_normalization(self) -> None:
        queries = np.random.default_rng(0).normal(size=(2, 5, 8))
        keys = np.random.default_rng(1).normal(size=(2, 5, 8))
        values = np.random.default_rng(2).normal(size=(2, 5, 8))
        result = scaled_dot_product_attention(queries, keys, values)
        assert result.output.shape == (2, 5, 8)
        assert np.allclose(result.weights.sum(axis=-1), 1.0)

    def test_mask_blocks_attention(self) -> None:
        queries = np.random.default_rng(3).normal(size=(1, 3, 4))
        mask = np.tril(np.ones((3, 3)))
        result = scaled_dot_product_attention(queries, queries, queries, mask=mask)
        assert result.weights[0, 0, 1] == pytest.approx(0.0)

    def test_multi_head_splits_dimension(self) -> None:
        queries = np.random.default_rng(4).normal(size=(2, 6, 8))
        keys = np.random.default_rng(5).normal(size=(2, 6, 8))
        values = np.random.default_rng(6).normal(size=(2, 6, 8))
        result = multi_head_attention(queries, keys, values, heads=4)
        assert result.output.shape == (2, 6, 8)

    def test_indivisible_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            multi_head_attention(
                np.ones((1, 2, 7)), np.ones((1, 2, 7)), np.ones((1, 2, 7)), heads=2
            )


class TestTinyGPT:
    def test_forward_logits_shape(self) -> None:
        model = TinyGPT(vocab_size=50, seed=7)
        logits = model.forward([1, 2, 3, 4])
        assert logits.shape == (4, 50)

    def test_causal_mask_makes_prefix_stable(self) -> None:
        model = TinyGPT(vocab_size=30, seed=8)
        short = model.forward([5, 6])
        longer = model.forward([5, 6, 7])
        assert np.allclose(short[0], longer[0])
        assert np.allclose(short[1], longer[1])

    def test_sample_length_and_valid_tokens(self) -> None:
        model = TinyGPT(vocab_size=20, seed=9)
        sampled = model.sample([0], max_new_tokens=6, seed=1)
        assert len(sampled) == 7
        assert all(0 <= token < 20 for token in sampled)

    def test_empty_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TinyGPT(vocab_size=10).forward([])

    def test_out_of_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="vocabulary"):
            TinyGPT(vocab_size=10).forward([99])

    def test_sequence_too_long_raises(self) -> None:
        model = TinyGPT(vocab_size=10, context_length=4)
        with pytest.raises(ValueError, match="context"):
            model.forward([1, 2, 3, 4, 5])

    def test_indivisible_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            TinyGPT(vocab_size=10, embedding_dim=7, heads=3)

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            TinyGPT(vocab_size=10, seed=2).sample([1], temperature=0.0)
