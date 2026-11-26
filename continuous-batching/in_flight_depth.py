import statistics
import time
from collections import Counter

import torch

from infer.engine import Engine, to_device
from infer.hf import download, load_model
from infer.paged import PagedCache, generate_paged
from infer.sampling import SamplingParams
from infer.scheduler import Request, Scheduler
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session, turns

MAX_TOKENS, BUDGET, BLOCKS, REPEATS = 64, 8192, 600, 3
PHASES = ("schedule", "launch", "wait", "update")
DEPTHS = (0, 1, 2)

model = load_model()
weight = model.lm_head.weight
tokenizer = Tokenizer(download(weights=False))
stop = frozenset(tokenizer.stop_ids)
prompts = [tokenizer.encode(chat(session(t), TOOLS)) for t in range(turns())]
alone = [
    generate_paged(model, p, SamplingParams(max_tokens=MAX_TOKENS), stop).tokens for p in prompts
]


class Timed(Engine):
    def step(self):
        start = time.perf_counter()
        plan = self.scheduler.schedule()
        self.phases["schedule"] += time.perf_counter() - start
        if plan:
            self.in_flight.append((plan, self.launch(plan)))
        while len(self.in_flight) > (self.depth if plan else 0):
            self.land(*self.in_flight.popleft())
        self.steps += bool(plan)
        return plan

    def launch(self, plan):
        start = time.perf_counter()
        sampled = super().launch(plan)
        self.phases["launch"] += time.perf_counter() - start
        return sampled

    def land(self, plan, sampled):
        start = time.perf_counter()
        tokens = sampled.tolist()
        landed = time.perf_counter()
        self.scheduler.update(plan, tokens, landed)
        self.phases["wait"] += landed - start
        self.phases["update"] += time.perf_counter() - landed


def serve(depth: int):
    cache = PagedCache(model.config, BLOCKS, 16, weight.dtype, weight.device)
    scheduler = Scheduler(cache, BUDGET)
    for turn, prompt in enumerate(prompts):
        scheduler.add(Request(turn, prompt, MAX_TOKENS, stop))
    engine = Timed(model, scheduler, depth)
    engine.phases, engine.steps = Counter(), 0
    start = time.perf_counter()
    engine.run()
    return time.perf_counter() - start, engine, sorted(scheduler.finished, key=lambda r: r.id)


def sync() -> None:
    if weight.device.type == "mps":
        torch.mps.synchronize()


matrix = torch.randn(2048, 2048, device=weight.device)


def forward() -> torch.Tensor:
    product = matrix
    for _ in range(30):
        product = (product @ matrix) * 1e-3
    return product[0, :1]


def read_before_forward() -> None:
    tiny = matrix[0, :1] + 1
    forward()
    tiny.tolist()


def probe_ms(action, queued: int) -> float:
    def once() -> float:
        sync()
        for _ in range(queued):
            forward()
        start = time.perf_counter()
        action()
        elapsed = time.perf_counter() - start
        sync()
        return elapsed * 1e3

    once()
    return statistics.median(once() for _ in range(REPEATS))


print(f"the {turns()} turns at once, budget {BUDGET}, greedy; each step plans the next ", end="")
print("before the previous one's tokens are on the host")
print(f"each depth's row is its run with the median wall time of {REPEATS} after a warm-up,")
print("the depths taking turns\n")
print("  depth   steps   tokens kept   tokens sampled   wasted   equal   wall time   tokens/s")
runs = {depth: [] for depth in DEPTHS}
for depth in DEPTHS:
    serve(depth)
for _ in range(REPEATS):
    for depth in DEPTHS:
        runs[depth].append(serve(depth))
median = {depth: sorted(runs[depth], key=lambda run: run[0])[REPEATS // 2] for depth in DEPTHS}
for depth, (seconds, engine, finished) in median.items():
    kept = sum(len(r.outputs) for r in finished)
    wasted = sum(r.wasted for r in finished)
    equal = sum(r.outputs == alone[r.id] for r in finished)
    print(
        f"  {depth:5}   {engine.steps:5}   {kept:11}   {kept + wasted:14}   {wasted:6}   "
        f"{equal}/{len(finished)}   {seconds:7.1f} s   {kept / seconds:8.1f}"
    )
print("  wasted per turn at depth 1:", [r.wasted for r in median[1][2]])

print("\nwhere a step's time goes in the same runs, ms per step")
print("  depth   " + "   ".join(f"{p:>8}" for p in (*PHASES, "step")) + "   x steps   wall time")
per_step = {}
for depth in (0, 1):
    seconds, engine, _ = median[depth]
    per_step[depth] = {p: engine.phases[p] / engine.steps * 1e3 for p in PHASES}
    step = sum(per_step[depth].values())
    print(
        f"  {depth:5}   "
        + "   ".join(f"{per_step[depth][p]:8.1f}" for p in PHASES)
        + f"   {step:8.1f}   {step * engine.steps / 1e3:5.1f} s   {seconds:7.1f} s"
    )
zero = per_step[0]
step = sum(zero.values())
hidden = zero["launch"] + max(zero["wait"], zero["schedule"] + zero["update"])
print("\nsimulated depth 1 on a device whose host can read step N while step N+1 runs:")
print(f"  launch + max(wait, schedule + update) = {hidden:.1f} ms per step vs {step:.1f} ", end="")
print(f"measured at depth 0 ({1 - hidden / step:.1%} less)")

device = weight.device
probes = (
    ("launch one", forward, 0),
    ("launch one, read its result", lambda: forward().tolist(), 0),
    ("a tiny op, then one, read the tiny op's result", read_before_forward, 0),
    (
        "one queued, then torch.tensor(list, device=...)",
        lambda: torch.tensor([1], device=device),
        1,
    ),
    ("one queued, then .to(device, non_blocking=True)", lambda: to_device([1], device), 1),
)
print(
    f"\nwhy depth 1 gains nothing here ({device.type}), a stand-in forward of 30 chained ", end=""
)
print(f"2048 x 2048 matmuls, ms, median of {REPEATS}")
for name, action, queued in probes:
    print(f"  {name:50} {probe_ms(action, queued):7.1f}")
