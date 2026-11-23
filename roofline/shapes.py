import statistics
import time

import torch

from infer.generate import generate
from infer.hf import download, load_model
from infer.kv import ContiguousCache
from infer.measure import sync
from infer.sampling import SamplingParams
from infer.static import StaticCache
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session

STEPS = 32

model = load_model()
device, config = model.lm_head.weight.device.type, model.config
prompt = Tokenizer(download(weights=False)).encode(chat(session(0), TOOLS))
ids = torch.tensor(prompt, device=device)


def prefilled(kind):
    cache = kind(config, len(prompt) + 2 * STEPS + 1, model.lm_head.weight.dtype, device)
    model(ids, torch.arange(len(prompt), device=device), cache)
    cache.advance(len(prompt))
    return cache


def timed(fn) -> tuple[float, float]:
    sync(device)
    start, host = time.perf_counter(), time.thread_time()
    fn()
    host = time.thread_time() - host
    sync(device)
    return time.perf_counter() - start, host


def decode_at(cache, length: int):
    def step():
        cache.length = length
        model(ids[-1:], torch.tensor([length], device=device), cache)

    return step


def forward(length: int):
    tokens = torch.arange(length, device=device) % config.vocab_size
    return lambda: model(tokens, torch.arange(length, device=device))


with torch.inference_mode():
    grows, static = prefilled(ContiguousCache), prefilled(StaticCache)
    for length in range(len(prompt) + STEPS, len(prompt) + 2 * STEPS):
        timed(decode_at(grows, length))
        timed(decode_at(static, length))
    times = {"new": [], "again": [], "static": []}
    for length in range(len(prompt), len(prompt) + STEPS):
        times["new"].append(timed(decode_at(grows, length)))
        times["again"].append(timed(decode_at(grows, length)))
        times["static"].append(timed(decode_at(static, length)))

    timed(forward(len(prompt)))
    full = {"new": [], "again": []}
    for length in range(len(prompt) + 1, len(prompt) + 6):
        full["new"].append(timed(forward(length)))
        full["again"].append(timed(forward(length)))

    generate(model, prompt, SamplingParams(max_tokens=2))
    runs = [generate(model, prompt, SamplingParams(max_tokens=STEPS)) for _ in range(3)]


def ms(values) -> str:
    return f"{statistics.median(values) * 1000:7.1f}"


def row(name: str, values) -> None:
    wall, host = zip(*values, strict=True)
    print(f"  {name:43}  {ms(wall)}  {ms(host):>12}")


def ratio(name: str, top, bottom) -> None:
    wall = statistics.median(t[0] / b[0] for t, b in zip(top, bottom, strict=True))
    host = statistics.median(t[1] / b[1] for t, b in zip(top, bottom, strict=True))
    print(f"    {name:41}  {wall:7.2f}x  {host:11.2f}x")


print(f"one decode step after the {len(prompt)}-token prompt, median of {STEPS} lengths")
print("                                                wall ms   host CPU ms")
row("cache, KV length never seen before", times["new"])
row("cache, same KV length a second time", times["again"])
row("whole buffer + mask, one shape every step", times["static"])
print(f"  ratio to the same length a second time, median of {STEPS} interleaved pairs")
ratio("new KV length", times["new"], times["again"])
ratio("whole buffer + mask", times["static"], times["again"])
print("\nno cache, the whole sequence, median of 5 lengths")
row("length never seen before", full["new"])
row("same length a second time", full["again"])
print(f"\ngenerate(), greedy, {STEPS} tokens, 3 runs")
for run in runs:
    print(f"  inter-token latency mean {run.itl * 1000:6.1f} ms   median {ms(run.gaps)} ms")
