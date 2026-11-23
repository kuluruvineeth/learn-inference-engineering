import torch
import torch.nn as nn
import torch.nn.functional as F

from infer.model import MLP, Attention, Model, causal_attention, rope


def _stack(*linears: nn.Linear) -> nn.Linear:
    first = linears[0]
    out = sum(linear.out_features for linear in linears)
    fused = nn.Linear(first.in_features, out, bias=first.bias is not None)
    fused = fused.to(first.weight.device, first.weight.dtype)
    with torch.no_grad():
        fused.weight.copy_(torch.cat([linear.weight for linear in linears]))
        if first.bias is not None:
            fused.bias.copy_(torch.cat([linear.bias for linear in linears]))
    return fused


class FusedAttention(nn.Module):
    def __init__(self, attention: Attention):
        super().__init__()
        self.layer, self.theta = attention.layer, attention.theta
        self.num_heads, self.num_kv_heads = attention.num_heads, attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.qkv_proj = _stack(attention.q_proj, attention.k_proj, attention.v_proj)
        self.o_proj = attention.o_proj
        q, kv = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.sizes = [q, kv, kv]

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache=None) -> torch.Tensor:
        tokens = x.size(0)
        q, k, v = self.qkv_proj(x).split(self.sizes, dim=-1)
        q = q.view(tokens, self.num_heads, self.head_dim)
        k = k.view(tokens, self.num_kv_heads, self.head_dim)
        v = v.view(tokens, self.num_kv_heads, self.head_dim)
        q, k = rope(q, positions, self.theta), rope(k, positions, self.theta)
        out = causal_attention(q, k, v) if cache is None else cache.attend(self.layer, q, k, v)
        return self.o_proj(out.reshape(tokens, -1))


class FusedMLP(nn.Module):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.gate_up_proj = _stack(mlp.gate_proj, mlp.up_proj)
        self.down_proj = mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def fuse_projections(model: Model) -> Model:
    for layer in model.layers:
        layer.self_attn = FusedAttention(layer.self_attn)
        layer.mlp = FusedMLP(layer.mlp)
    return model
