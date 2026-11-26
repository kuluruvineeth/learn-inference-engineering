import statistics
import time
from itertools import pairwise

from infer.engine import Engine
from infer.hf import download, load_model
from infer.paged import PagedCache, generate_paged
from infer.sampling import SamplingParams
from infer.scheduler import Request, Scheduler
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session, turns

MAX_TOKENS, BLOCKS, REPEATS, WARM = 64, 600, 5, 3
BUDGETS = (2048, 512, 256, 128)

model = load_model()
weight = model.lm_head.weight
tokenizer = Tokenizer(download(weights=False))
stop = frozenset(tokenizer.stop_ids)
prompts = [tokenizer.encode(chat(session(t), TOOLS)) for t in range(turns())]
alone = [
    generate_paged(model, p, SamplingParams(max_tokens=MAX_TOKENS), stop).tokens for p in prompts
]
decoders, late = range(turns() - 1), turns() - 1


def serve(budget: int):
    cache = PagedCache(model.config, BLOCKS, 16, weight.dtype, weight.device)
    scheduler = Scheduler(cache, budget)
    requests = [Request(t, prompts[t], MAX_TOKENS, stop) for t in range(turns())]
    for t in decoders:
        scheduler.add(requests[t])
    engine, steps, arrived = Engine(model, scheduler), [], None
    while engine.busy:
        if arrived is None and all(len(requests[t].outputs) >= WARM for t in decoders):
            requests[late].arrival = arrived = time.perf_counter()
            scheduler.add(requests[late])
        plan = engine.step()
        if arrived is not None:
            steps.append(plan)
    gaps = [b - a for t in decoders for a, b in pairwise(requests[t].times) if b > arrived]
    prefill = [
        p for p in steps if any(e.request.id == late and e.start < len(prompts[late]) for e in p)
    ]
    return requests, gaps, prefill


print(
    f"{len(decoders)} turns decoding; turn {late} ({len(prompts[late])} prompt tokens) arrives ",
    end="",
)
print(f"once each has {WARM} tokens")
print("decode-first: running decodes take the budget, the new prompt gets what is left")
print(f"timings are medians of {REPEATS} runs after a warm-up, the budgets taking turns\n")
print(
    "  budget   prefill   largest step   decode gap   decode gap   max / p50   TTFT turn 4   equal"
)
print("           steps     (tokens)       p50 (ms)     max (ms)                 (ms)")
for budget in BUDGETS:
    serve(budget)
runs = {budget: [] for budget in BUDGETS}
for _ in range(REPEATS):
    for budget in BUDGETS:
        runs[budget].append(serve(budget))
for budget in BUDGETS:
    prefill = runs[budget][0][2]
    equal = sum(r.outputs == alone[r.id] for run in runs[budget] for r in run[0])
    p50 = statistics.median(statistics.median(gaps) for _, gaps, _ in runs[budget])
    worst = statistics.median(max(gaps) for _, gaps, _ in runs[budget])
    ratio = statistics.median(max(gaps) / statistics.median(gaps) for _, gaps, _ in runs[budget])
    ttft = statistics.median(run[0][late].result().ttft for run in runs[budget])
    largest = max(sum(e.count for e in plan) for plan in prefill)
    chunks = [e.count for plan in prefill for e in plan if e.request.id == late]
    print(
        f"  {budget:6}   {len(prefill):7}   {largest:12}   {p50 * 1e3:10.0f}   "
        f"{worst * 1e3:10.0f}   {ratio:9.1f}   {ttft * 1e3:11.0f}   {equal}/{turns() * REPEATS}"
    )
    print(f"           chunks of turn {late}: {chunks}")
print("\nequal: replies equal to one request at a time, over every run")
