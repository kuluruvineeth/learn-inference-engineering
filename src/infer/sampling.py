from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 64
    seed: int = 0


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    if params.temperature == 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.cpu().float() / params.temperature, dim=-1)
    sorted_probs, order = probs.sort(descending=True)
    keep = sorted_probs.cumsum(-1) - sorted_probs < params.top_p
    sorted_probs = sorted_probs * keep
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(), 1, generator=generator)
    return int(order[choice])
