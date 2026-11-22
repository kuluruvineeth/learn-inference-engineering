from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dim = x.size(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device) / dim))
    angles = positions.float()[:, None] * freqs
    cos = torch.cos(angles).repeat(1, 2)[:, None, :]
    sin = torch.sin(angles).repeat(1, 2)[:, None, :]
    half = dim // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
    v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True
    )
    return out.transpose(0, 1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class Attention(nn.Module):
    def __init__(self, config: Config, layer: int):
        super().__init__()
        self.layer = layer
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache=None) -> torch.Tensor:
        tokens = x.size(0)
        q = self.q_proj(x).view(tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(tokens, self.num_kv_heads, self.head_dim)
        q, k = rope(q, positions, self.theta), rope(k, positions, self.theta)
        out = causal_attention(q, k, v) if cache is None else cache.attend(self.layer, q, k, v)
        return self.o_proj(out.reshape(tokens, -1))


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, config: Config, layer: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache=None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Layer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def hidden(self, tokens: torch.Tensor, positions: torch.Tensor, cache=None) -> torch.Tensor:
        x = self.embed_tokens(tokens)
        for layer in self.layers:
            x = layer(x, positions, cache)
        return self.norm(x)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, cache=None) -> torch.Tensor:
        return self.lm_head(self.hidden(tokens, positions, cache))
