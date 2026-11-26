import statistics
import time

from infer.engine import Engine
from infer.hf import download, load_model
from infer.paged import PagedCache, generate_paged
from infer.sampling import SamplingParams
from infer.scheduler import Request, Scheduler, StaticScheduler
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session, turns

MAX_TOKENS, SLOTS, BUDGET, BLOCKS, REPEATS = 64, 4, 8192, 600, 3

model = load_model()
weight = model.lm_head.weight
tokenizer = Tokenizer(download(weights=False))
stop = frozenset(tokenizer.stop_ids)
prompts = [tokenizer.encode(chat(session(t), TOOLS)) for t in range(turns())]
alone = [
    generate_paged(model, p, SamplingParams(max_tokens=MAX_TOKENS), stop).tokens for p in prompts
]
order = list(range(turns())) * 2


def serve(kind):
    cache = PagedCache(model.config, BLOCKS, 16, weight.dtype, weight.device)
    scheduler = kind(cache, BUDGET, SLOTS)
    for i, turn in enumerate(order):
        scheduler.add(Request(i, prompts[turn], MAX_TOKENS, stop))
    engine, plans, times = Engine(model, scheduler), [], []
    start = time.perf_counter()
    while engine.busy:
        began = time.perf_counter()
        plans.append(engine.step())
        times.append(time.perf_counter() - began)
    finished = sorted(scheduler.finished, key=lambda r: r.id)
    return time.perf_counter() - start, plans, times, finished


def split(plans, times) -> tuple[int, float, int, float]:
    prompt = [any(e.start < len(e.request.prompt) for e in plan) for plan in plans]
    with_prompt = sum(t for t, p in zip(times, prompt, strict=True) if p)
    return sum(prompt), with_prompt, len(plans) - sum(prompt), sum(times) - with_prompt


def log(plans) -> None:
    previous = set()
    for step, plan in enumerate(plans):
        ids = {e.request.id for e in plan}
        joined = [f"+{i} ({e.count})" for e in plan if (i := e.request.id) not in previous]
        left = [f"-{i}" for i in sorted(previous - ids)]
        if joined or left:
            print(
                f"    step {step:3}  {' '.join(left + joined):48} "
                f"running {len(plan)}  tokens {sum(e.count for e in plan):5}"
            )
        previous = ids


print(f"device={weight.device.type}")
print(f"{len(order)} requests: the {turns()} agent turns, twice (request i is turn i % {turns()})")
print(f"replies one at a time (greedy): {[len(r) for r in alone]} tokens")
print(f"at most {SLOTS} running, budget {BUDGET} tokens per step, block 16, {BLOCKS} blocks")
results = {}
kinds = (StaticScheduler, Scheduler)
for kind in kinds:
    serve(kind)
repeats = {kind: [] for kind in kinds}
for _ in range(REPEATS):
    for kind in kinds:
        repeats[kind].append(serve(kind))
for kind in kinds:
    runs = repeats[kind]
    seconds = statistics.median(run[0] for run in runs)
    _, plans, _, finished = runs[0]
    splits = [split(run[1], run[2]) for run in runs]
    prompt_steps, _, decode_steps, _ = splits[0]
    prompt_s = statistics.median(s[1] for s in splits)
    decode_s = statistics.median(s[3] for s in splits)
    same = sum(r.outputs == alone[order[r.id]] for r in finished)
    generated = sum(len(r.outputs) for r in finished)
    busy = sum(len(plan) for plan in plans)
    slots = SLOTS * len(plans)
    results[kind] = (len(plans), seconds)
    name = "static batch" if kind is StaticScheduler else "continuous (iteration-level)"
    print(f"\n{name}")
    log(plans)
    print(f"  forward passes {len(plans)}   slot-steps {slots}   busy {busy}   ", end="")
    print(f"idle {slots - busy} ({(slots - busy) / slots:.0%})")
    print(f"  replies equal to one at a time: {same} of {len(finished)}   generated {generated}")
    print(f"  wall time {seconds:.1f} s (median of {REPEATS})   {generated / seconds:.1f} tokens/s")
    print(f"    {prompt_steps:3} steps with a prompt   {prompt_s:5.1f} s   ", end="")
    print(f"{decode_steps:3} decode-only steps   {decode_s:5.1f} s ", end="")
    print(f"({decode_s / decode_steps * 1e3:.0f} ms each)")

(static_steps, static_time), (steps, seconds) = results[StaticScheduler], results[Scheduler]
print(f"\nstatic / continuous: forward passes {static_steps / steps:.2f}x, ", end="")
print(f"wall time {static_time / seconds:.2f}x")
