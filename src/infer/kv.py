import torch
import torch.nn.functional as F

from infer.model import Config


class ContiguousCache:
    def __init__(self, config: Config, max_tokens: int, dtype=torch.float32, device="cpu"):
        shape = (config.num_hidden_layers, max_tokens, config.num_key_value_heads, config.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0

    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        start, end = self.length, self.length + q.size(0)
        self.k[layer, start:end] = k
        self.v[layer, start:end] = v
        keys = self.k[layer, :end].repeat_interleave(q.size(1) // k.size(1), dim=1)
        values = self.v[layer, :end].repeat_interleave(q.size(1) // v.size(1), dim=1)
        rows = torch.arange(start, end, device=q.device)
        mask = torch.arange(end, device=q.device)[None, :] <= rows[:, None]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1), attn_mask=mask
        )
        return out.transpose(0, 1)

    def advance(self, tokens: int) -> None:
        self.length += tokens
