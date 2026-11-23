import torch
import torch.nn.functional as F

from infer.kv import ContiguousCache


class StaticCache(ContiguousCache):
    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        start, end = self.length, self.length + q.size(0)
        self.k[layer, start:end] = k
        self.v[layer, start:end] = v
        keys = self.k[layer].repeat_interleave(q.size(1) // k.size(1), dim=1)
        values = self.v[layer].repeat_interleave(q.size(1) // v.size(1), dim=1)
        rows = torch.arange(start, end, device=q.device)
        mask = torch.arange(self.k.size(1), device=q.device)[None, :] <= rows[:, None]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1), attn_mask=mask
        )
        return out.transpose(0, 1)
