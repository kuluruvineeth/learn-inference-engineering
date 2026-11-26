import time
from collections import deque

import torch

from infer.model import Model
from infer.scheduler import Entry, Scheduler


def to_device(values: list[int], device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long).to(device, non_blocking=True)


class Engine:
    def __init__(self, model: Model, scheduler: Scheduler, depth: int = 0):
        self.model, self.scheduler, self.depth = model, scheduler, depth
        self.device = model.lm_head.weight.device
        self.in_flight: deque[tuple[list[Entry], torch.Tensor]] = deque()

    @property
    def busy(self) -> bool:
        return self.scheduler.busy or bool(self.in_flight)

    def step(self) -> list[Entry]:
        plan = self.scheduler.schedule()
        if plan:
            self.in_flight.append((plan, self.launch(plan)))
        while len(self.in_flight) > (self.depth if plan else 0):
            self.land(*self.in_flight.popleft())
        return plan

    def run(self) -> None:
        while self.busy:
            self.step()

    @torch.inference_mode()
    def launch(self, plan: list[Entry]) -> torch.Tensor:
        cache = self.scheduler.cache
        ids, relayed, sources, last, rows = [], [], [], [], []
        for entry in plan:
            known = entry.request.tokens[entry.start : entry.start + entry.count]
            if len(known) < entry.count:
                relayed.append(len(ids) + len(known))
                sources.append(entry.request.future)
            ids += known + [0] * (entry.count - len(known))
            if entry.sample:
                rows.append(entry)
                last.append(len(ids) - 1)
        tokens = to_device(ids, self.device)
        if relayed:
            tokens[to_device(relayed, self.device)] = torch.stack([t[i] for t, i in sources])
        batch = cache.begin([(entry.request.id, entry.count) for entry in plan])
        hidden = self.model.hidden(tokens, batch.positions, cache)
        cache.commit()
        sampled = self.sample(self.model.lm_head(hidden[to_device(last, self.device)]), rows)
        for row, entry in enumerate(rows):
            entry.request.future = (sampled, row)
        return sampled

    def sample(self, logits: torch.Tensor, rows: list[Entry]) -> torch.Tensor:
        return logits.argmax(-1)

    def land(self, plan: list[Entry], sampled: torch.Tensor) -> None:
        self.scheduler.update(plan, sampled.tolist(), time.perf_counter())
