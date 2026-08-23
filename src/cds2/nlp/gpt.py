"""A tiny GPT-style transformer: forward pass and sampling (educational)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .attention import multi_head_attention

__all__ = ["TinyGPT"]

FloatArray = NDArray[np.float64]


def _layer_norm(values: FloatArray, epsilon: float = 1e-9) -> FloatArray:
    mean: FloatArray = values.mean(axis=-1, keepdims=True)
    variance: FloatArray = values.var(axis=-1, keepdims=True)
    normalized: FloatArray = (values - mean) / np.sqrt(variance + epsilon)
    return normalized


def _gelu(values: FloatArray) -> FloatArray:
    inner: FloatArray = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)
    activated: FloatArray = 0.5 * values * (1.0 + np.tanh(inner))
    return activated


def _softmax(values: FloatArray) -> FloatArray:
    shifted: FloatArray = values - values.max(axis=-1, keepdims=True)
    exponentials: FloatArray = np.exp(shifted)
    normalized: FloatArray = exponentials / exponentials.sum(axis=-1, keepdims=True)
    return normalized


class TinyGPT:
    """Untrained-but-deterministic mini GPT for studying the forward pass."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        heads: int = 4,
        layers: int = 2,
        context_length: int = 64,
        seed: int | None = None,
    ) -> None:
        if embedding_dim % heads != 0:
            msg = "embedding_dim must be divisible by heads"
            raise ValueError(msg)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.layers = layers
        self.context_length = context_length
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(embedding_dim)
        self.token_embeddings = rng.normal(scale=scale, size=(vocab_size, embedding_dim))
        self.position_embeddings = rng.normal(scale=scale, size=(context_length, embedding_dim))
        self.blocks: list[dict[str, FloatArray]] = []
        for _ in range(layers):
            self.blocks.append(
                {
                    "w_q": rng.normal(scale=scale, size=(embedding_dim, embedding_dim)),
                    "w_k": rng.normal(scale=scale, size=(embedding_dim, embedding_dim)),
                    "w_v": rng.normal(scale=scale, size=(embedding_dim, embedding_dim)),
                    "w_o": rng.normal(scale=scale, size=(embedding_dim, embedding_dim)),
                    "w_in": rng.normal(scale=scale, size=(embedding_dim, 4 * embedding_dim)),
                    "w_out": rng.normal(
                        scale=1.0 / (4 * embedding_dim), size=(4 * embedding_dim, embedding_dim)
                    ),
                }
            )
        self.lm_head = rng.normal(scale=scale, size=(embedding_dim, vocab_size))

    def forward(self, token_ids: list[int]) -> FloatArray:
        """Return logits of shape (len(token_ids), vocab_size)."""
        if not token_ids:
            msg = "at least one token is required"
            raise ValueError(msg)
        if any(not 0 <= token < self.vocab_size for token in token_ids):
            msg = "token id outside the vocabulary"
            raise ValueError(msg)
        if len(token_ids) > self.context_length:
            msg = f"sequence longer than context ({self.context_length})"
            raise ValueError(msg)
        positions = np.arange(len(token_ids))
        hidden = self.token_embeddings[np.asarray(token_ids)] + self.position_embeddings[positions]
        causal_mask = np.tril(np.ones((len(token_ids), len(token_ids))))
        for block in self.blocks:
            queries = hidden @ block["w_q"]
            keys = hidden @ block["w_k"]
            values = hidden @ block["w_v"]
            attended = multi_head_attention(
                queries, keys, values, self.heads, mask=causal_mask
            ).output
            hidden = _layer_norm(hidden + attended @ block["w_o"])
            feed_forward = _gelu(hidden @ block["w_in"]) @ block["w_out"]
            hidden = _layer_norm(hidden + feed_forward)
        return np.asarray(hidden @ self.lm_head, dtype=float)

    def sample(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[int]:
        """Autoregressively extend ``prompt_tokens`` by sampling the logits."""
        if temperature <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        rng_values = np.random.default_rng(seed)
        tokens = list(prompt_tokens)
        for _ in range(max_new_tokens):
            window = tokens[-self.context_length :]
            logits = self.forward(window)[-1] / temperature
            probabilities = _softmax(logits)
            next_token = int(rng_values.choice(self.vocab_size, p=probabilities))
            tokens.append(next_token)
        return tokens
