import time
from dataclasses import dataclass, field

import torch

from infer.model import Model
from infer.sampling import SamplingParams, sample


@dataclass
class Result:
    tokens: list[int] = field(default_factory=list)
    ttft: float = 0.0
    gaps: list[float] = field(default_factory=list)

    @property
    def itl(self) -> float:
        return sum(self.gaps) / len(self.gaps) if self.gaps else 0.0


def _loop(step, prompt: list[int], params: SamplingParams, stop_ids: set[int]) -> Result:
    generator = torch.Generator().manual_seed(params.seed)
    result = Result()
    start = last = time.perf_counter()
    logits = step(prompt, 0)
    while True:
        token = sample(logits, params, generator)
        now = time.perf_counter()
        if result.tokens:
            result.gaps.append(now - last)
        else:
            result.ttft = now - start
        last = now
        result.tokens.append(token)
        if token in stop_ids or len(result.tokens) == params.max_tokens:
            return result
        logits = step([token], len(prompt) + len(result.tokens) - 1)


@torch.inference_mode()
def generate_naive(
    model: Model, prompt: list[int], params: SamplingParams, stop_ids: set[int] = frozenset()
) -> Result:
    seen, device = list(prompt), model.lm_head.weight.device

    def step(new: list[int], position: int) -> torch.Tensor:
        if position:
            seen.extend(new)
        tokens = torch.tensor(seen, device=device)
        return model(tokens, torch.arange(len(seen), device=device))[-1]

    return _loop(step, prompt, params, stop_ids)
