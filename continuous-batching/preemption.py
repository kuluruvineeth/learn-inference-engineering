from infer.engine import Engine
from infer.hf import download, load_model
from infer.paged import PagedCache, generate_paged
from infer.sampling import SamplingParams
from infer.scheduler import Request, Scheduler, fewest_output, most_output, newest
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session, turns

MAX_TOKENS, BUDGET, BLOCKS = 64, 512, 250

model = load_model()
weight = model.lm_head.weight
tokenizer = Tokenizer(download(weights=False))
stop = frozenset(tokenizer.stop_ids)
prompts = [tokenizer.encode(chat(session(t), TOOLS)) for t in range(turns())]
alone = [
    generate_paged(model, p, SamplingParams(max_tokens=MAX_TOKENS), stop).tokens for p in prompts
]


class Logged(Scheduler):
    def preempt(self, victim: Request) -> None:
        blocks = len(self.cache.sequences[victim.id].blocks)
        self.victims.append((self.step, victim.id, len(victim.outputs), victim.computed, blocks))
        super().preempt(victim)


class FirstChunk(Logged):
    def admit(self, request: Request) -> bool:
        chunk = min(request.work, self.left)
        return self.cache.blocks_needed(request.id, chunk) <= self.cache.pool.num_free


def serve(kind, policy):
    cache = PagedCache(model.config, BLOCKS, 16, weight.dtype, weight.device)
    scheduler = kind(cache, BUDGET, policy=policy)
    scheduler.victims, scheduler.step = [], 0
    for turn, prompt in enumerate(prompts):
        scheduler.add(Request(turn, prompt, MAX_TOKENS, stop))
    engine = Engine(model, scheduler)
    while engine.busy:
        engine.step()
        scheduler.step += 1
    return scheduler


usable = BLOCKS - 1
print(f"the {turns()} turns at once: prompts {[len(p) for p in prompts]} tokens, ", end="")
print(f"replies {[len(r) for r in alone]}")
print(f"KV pool {usable} blocks of 16 = {usable * 16} slots; the prompts alone need ", end="")
print(f"{sum(-(-len(p) // 16) for p in prompts)} blocks")
print(f"budget {BUDGET} tokens per step, decode-first, preemption by recompute\n")
print("  admission        victim policy    preemptions   recomputed tokens   steps   equal")
traces = {}
for kind, rule in ((Logged, "whole sequence"), (FirstChunk, "first chunk")):
    for policy in (newest, fewest_output, most_output):
        scheduler = serve(kind, policy)
        finished = scheduler.finished
        equal = sum(r.outputs == alone[r.id] for r in finished)
        recomputed = sum(r.recomputed for r in finished)
        print(
            f"  {rule:15}  {policy.__name__:15}  {scheduler.preemptions:11}   "
            f"{recomputed:17}   {scheduler.step:5}   {equal}/{len(finished)}"
        )
        traces[rule, policy.__name__] = scheduler.victims
        assert scheduler.cache.pool.num_free == usable

for key in (
    ("whole sequence", "newest"),
    ("first chunk", "newest"),
    ("first chunk", "most_output"),
):
    print(f"\nvictims, {key[0]} admission, {key[1]}:")
    for step, turn, outputs, computed, blocks in traces[key][:6]:
        print(
            f"  step {step:3}: turn {turn} evicted, {outputs:2} tokens generated, "
            f"KV of {computed:4} tokens dropped, {blocks} blocks freed"
        )
    if len(traces[key]) > 6:
        print(f"  ... {len(traces[key]) - 6} more")
