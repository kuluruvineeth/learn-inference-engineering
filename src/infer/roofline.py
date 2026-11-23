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


@dataclass
class MoEParams:
    total: int
    active: int
    active_matmul: int
    dense_matmul: int
    expert: int
    moe_layers: int
    routed: int
    shared: int
    top_k: int


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def mla_attention_params(c: dict) -> int:
    h, heads = c["hidden_size"], c["num_attention_heads"]
    q_rank, kv_rank = c["q_lora_rank"], c["kv_lora_rank"]
    nope, rope, v = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    return (
        h * q_rank
        + q_rank * heads * (nope + rope)
        + h * (kv_rank + rope)
        + kv_rank * heads * (nope + v)
        + heads * v * h
    )


def deepseek_v3_params(c: dict) -> MoEParams:
    h, layers, dense_layers = c["hidden_size"], c["num_hidden_layers"], c["first_k_dense_replace"]
    routed, shared, top_k = c["n_routed_experts"], c["n_shared_experts"], c["num_experts_per_tok"]
    moe_layers = layers - dense_layers
    attention = mla_attention_params(c)
    norms = c["q_lora_rank"] + c["kv_lora_rank"] + 2 * h
    expert = 3 * h * c["moe_intermediate_size"]
    dense_mlp = 3 * h * c["intermediate_size"]
    gate = routed * h + routed
    table = c["vocab_size"] * h
    common = layers * (attention + norms) + dense_layers * dense_mlp + moe_layers * gate + h
    dense_matmul = layers * attention + dense_layers * dense_mlp + moe_layers * routed * h + table
    return MoEParams(
        total=common + moe_layers * (routed + shared) * expert + 2 * table,
        active=common + moe_layers * (top_k + shared) * expert + 2 * table,
        active_matmul=dense_matmul + moe_layers * (top_k + shared) * expert,
        dense_matmul=dense_matmul,
        expert=expert,
        moe_layers=moe_layers,
        routed=routed,
        shared=shared,
        top_k=top_k,
    )


def experts_touched(routed: int, top_k: int, tokens: int) -> float:
    return routed * (1 - (1 - top_k / routed) ** tokens)


FP8_BLOCK_SCALED = 1 + 4 / (128 * 128)


def deepseek_v3_decode(
    c: dict, context: int, batch: int = 1, weight_bytes: float = FP8_BLOCK_SCALED, kv_value_bytes=2
) -> Cost:
    p = deepseek_v3_params(c)
    latent = c["kv_lora_rank"] + c["qk_rope_head_dim"]
    heads, layers = c["num_attention_heads"], c["num_hidden_layers"]
    attention_per_key = 2 * heads * latent + 2 * heads * c["kv_lora_rank"]
    experts = p.shared + experts_touched(p.routed, p.top_k, batch)
    return Cost(
        flops=batch * (2 * p.active_matmul + attention_per_key * layers * context),
        weight_bytes=(p.dense_matmul + p.moe_layers * experts * p.expert) * weight_bytes,
        kv_bytes=batch * context * latent * layers * kv_value_bytes,
    )


def deepseek_v3_prefill(c: dict, tokens: int) -> Cost:
    p = deepseek_v3_params(c)
    heads, layers = c["num_attention_heads"], c["num_hidden_layers"]
    width = c["qk_nope_head_dim"] + c["qk_rope_head_dim"] + c["v_head_dim"]
    attention = layers * 2 * heads * width * tokens * (tokens + 1) // 2
    experts = p.shared + experts_touched(p.routed, p.top_k, tokens)
    return Cost(
        flops=2 * p.active_matmul * tokens + attention,
        weight_bytes=(p.dense_matmul + p.moe_layers * experts * p.expert) * FP8_BLOCK_SCALED,
    )


FP4_GROUP_SCALED = 0.5 + 1 / 32


@dataclass
class ExpertParams:
    expert: int
    total: int
    active: int

    @property
    def active_bytes(self) -> float:
        return self.active * FP4_GROUP_SCALED

    @property
    def total_bytes(self) -> float:
        return self.total * FP4_GROUP_SCALED


def deepseek_v41_experts(c: dict) -> ExpertParams:
    t = c["text_config"]
    expert = 3 * t["hidden_size"] * t["moe_intermediate_size"]
    layers, shared = t["num_hidden_layers"], t["n_shared_experts"]
    return ExpertParams(
        expert=expert,
        total=layers * (t["n_routed_experts"] + shared) * expert,
        active=layers * (t["num_experts_per_tok"] + shared) * expert,
    )
