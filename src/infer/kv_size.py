from dataclasses import dataclass
from itertools import cycle

import torch

from infer.model import Config
from infer.paged import BlockPool, blocks_for


def bytes_per_token(config: Config, dtype: torch.dtype, kv_heads: int | None = None) -> int:
    heads = kv_heads or config.num_key_value_heads
    return 2 * heads * config.head_dim * dtype.itemsize * config.num_hidden_layers


def per_head_bytes_per_token(hf: dict, element_bytes: int = 2) -> int:
    key = hf["qk_nope_head_dim"] + hf["qk_rope_head_dim"]
    per_layer = hf["num_attention_heads"] * (key + hf["v_head_dim"]) * element_bytes
    return per_layer * hf["num_hidden_layers"]


def latent_bytes_per_token(hf: dict, element_bytes: int = 2) -> int:
    per_layer = (hf["kv_lora_rank"] + hf["qk_rope_head_dim"]) * element_bytes
    return per_layer * hf["num_hidden_layers"]


def cache_bytes_per_slot(cache) -> int:
    slots = cache.k[0].size(0) * cache.block_size
    return sum(t.numel() * t.element_size() for t in cache.k + cache.v) // slots


@dataclass
class Fit:
    sessions: int = 0
    held_tokens: int = 0
    live_tokens: int = 0

    @property
    def wasted_tokens(self) -> int:
        return self.held_tokens - self.live_tokens


def fit_contiguous(requests: list[tuple[int, int]], max_tokens: int, budget_tokens: int) -> Fit:
    fit = Fit()
    for prompt, generated in cycle(requests):
        if fit.held_tokens + prompt + max_tokens > budget_tokens:
            return fit
        fit.sessions += 1
        fit.held_tokens += prompt + max_tokens
        fit.live_tokens += prompt + generated


def fit_paged(requests: list[tuple[int, int]], block_size: int, budget_tokens: int) -> Fit:
    pool, fit = BlockPool(budget_tokens // block_size), Fit()
    for prompt, generated in cycle(requests):
        blocks = pool.allocate(blocks_for(prompt + generated, block_size))
        if blocks is None:
            return fit
        fit.sessions += 1
        fit.held_tokens += len(blocks) * block_size
        fit.live_tokens += prompt + generated
