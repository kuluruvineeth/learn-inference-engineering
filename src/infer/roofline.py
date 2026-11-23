import json
from dataclasses import dataclass
from pathlib import Path

from infer.model import Config


@dataclass
class Cost:
    flops: float
    weight_bytes: float
    kv_bytes: float = 0.0
    activation_bytes: float = 0.0

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.kv_bytes + self.activation_bytes

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes

    def time(self, bandwidth: float, peak: float) -> float:
        return max(self.bytes / bandwidth, self.flops / peak)


@dataclass
class DenseParams:
    matmul: int
    embedding: int
    vectors: int

    @property
    def total(self) -> int:
        return self.matmul + self.embedding + self.vectors


def dense_params(config: Config) -> DenseParams:
    h, i, d = config.hidden_size, config.intermediate_size, config.head_dim
    q, kv = config.num_attention_heads * d, config.num_key_value_heads * d
    layer = 2 * h * q + 2 * h * kv + 3 * h * i
    head = config.vocab_size * h
    embedding = 0 if config.tie_word_embeddings else head
    vectors = config.num_hidden_layers * (q + 2 * kv + 2 * h) + h
    return DenseParams(config.num_hidden_layers * layer + head, embedding, vectors)


def kv_bytes_per_token(config: Config, value_bytes: int = 4) -> int:
    return 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * value_bytes


def matmul_values_per_row(config: Config) -> int:
    h, i, d = config.hidden_size, config.intermediate_size, config.head_dim
    q, kv = config.num_attention_heads * d, config.num_key_value_heads * d
    per_layer = (h + q) + 2 * (h + kv) + (q + h) + 2 * (h + i) + (i + h)
    return config.num_hidden_layers * per_layer + config.hidden_size + config.vocab_size


def dense_pass(
    config: Config, tokens: int, context: int = 0, sequences: int = 1, value_bytes: int = 4
) -> Cost:
    params = dense_params(config)
    rows = tokens * sequences
    keys = sequences * (tokens * context + tokens * (tokens + 1) // 2)
    width = config.num_attention_heads * config.head_dim
    attention = 4 * config.num_hidden_layers * width * keys
    embedding_rows = rows * config.hidden_size if params.embedding else 0
    return Cost(
        flops=2 * params.matmul * rows + attention,
        weight_bytes=(params.matmul + params.vectors + embedding_rows) * value_bytes,
        kv_bytes=sequences * (context + tokens) * kv_bytes_per_token(config, value_bytes),
        activation_bytes=rows * matmul_values_per_row(config) * value_bytes,
    )
