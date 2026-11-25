import json
import statistics
from pathlib import Path

import torch

from infer.generate import generate
from infer.hf import download, load_model
from infer.kv import ContiguousCache
from infer.kv_size import (
    bytes_per_token,
    cache_bytes_per_slot,
    fit_contiguous,
    fit_paged,
    latent_bytes_per_token,
    per_head_bytes_per_token,
)
from infer.paged import PagedCache, blocks_for, generate_paged
from infer.sampling import SamplingParams
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session, turns

MAX_TOKENS = 4096
BUDGET = 1 << 30
BLOCK = 16

model = load_model()
config, device = model.config, model.lm_head.weight.device
tokenizer = Tokenizer(download(weights=False))
v3 = json.loads((Path(__file__).parents[1] / "configs" / "deepseek-v3.json").read_text())


def row(label: str, value: int, note: str = "") -> None:
    print(f"  {label:66} {value:>10,} B{note}")


print("bytes per token, K and V at every layer")
heads, kv_heads, layers = (
    config.num_attention_heads,
    config.num_key_value_heads,
    config.num_hidden_layers,
)
for dtype in (torch.float32, torch.bfloat16):
    allocated = cache_bytes_per_slot(PagedCache(config, 2, BLOCK, dtype, device))
    row(
        f"Qwen2.5-0.5B GQA, {kv_heads} KV heads x {config.head_dim} x {layers} layers, {dtype}"[
            :66
        ],
        bytes_per_token(config, dtype),
        f"   allocated tensor: {allocated:,} B per slot",
    )
row(
    f"Qwen2.5-0.5B if all {heads} heads kept K and V, torch.bfloat16",
    bytes_per_token(config, torch.bfloat16, heads),
)
per_head, latent = per_head_bytes_per_token(v3), latent_bytes_per_token(v3)
key = f"{v3['qk_nope_head_dim']} + {v3['qk_rope_head_dim']}"
v3_layers = f"{v3['num_hidden_layers']} layers"
row(
    f"DeepSeek-V3 per-head K ({key}) and V ({v3['v_head_dim']}), "
    f"{v3['num_attention_heads']} heads x {v3_layers}, bf16",
    per_head,
)
row(
    f"DeepSeek-V3 MLA latent {v3['kv_lora_rank']} + {v3['qk_rope_head_dim']}, {v3_layers}, bf16",
    latent,
    f"   {per_head / latent:.1f}x smaller",
)

prompts = [tokenizer.encode(chat(session(t), TOOLS)) for t in range(turns())]
params = SamplingParams(max_tokens=MAX_TOKENS)
print(f"\none request at a time on the paged cache, block {BLOCK}, greedy, max_tokens {MAX_TOKENS}")
replies = []
for turn, prompt in enumerate(prompts):
    paged = generate_paged(model, prompt, params, tokenizer.stop_ids)
    contiguous = generate(model, prompt, params, tokenizer.stop_ids)
    replies.append(paged.tokens)
    length = len(prompt) + len(paged.tokens)
    print(
        f"  turn {turn}: prompt {len(prompt):5}  reply {len(paged.tokens):4}  "
        f"blocks {blocks_for(length, BLOCK):3}  same tokens as the contiguous cache: "
        f"{paged.tokens == contiguous.tokens}"
    )

print("\nall five turns as one batch: one flat forward for the five prompts, then decoded together")
stop = tokenizer.stop_ids
with torch.inference_mode():
    blocks = sum(blocks_for(len(p) + len(r), BLOCK) for p, r in zip(prompts, replies, strict=True))
    cache = PagedCache(config, 1 + blocks, BLOCK, model.lm_head.weight.dtype, device)
    for seq_id, prompt in enumerate(prompts):
        assert cache.reserve(seq_id, len(prompt))
    batch = cache.begin([(seq_id, len(prompt)) for seq_id, prompt in enumerate(prompts)])
    flat = torch.tensor([t for prompt in prompts for t in prompt], device=device)
    first = model.lm_head(model.hidden(flat, batch.positions, cache)[batch.last])
    cache.commit()
    print(f"  prefill: {flat.numel()} tokens in one forward, {cache.pool.num_free} blocks free")
    gaps = []
    for seq_id, prompt in enumerate(prompts):
        alone = ContiguousCache(config, len(prompt), model.lm_head.weight.dtype, device)
        positions = torch.arange(len(prompt), device=device)
        hidden = model.hidden(torch.tensor(prompt, device=device), positions, alone)[-1:]
        gaps.append(float((first[seq_id] - model.lm_head(hidden)[0]).abs().max()))
    print(f"  last-token logits vs each prompt alone: max abs difference {max(gaps):.2e}")
    tokens = {seq_id: [int(t)] for seq_id, t in enumerate(first.argmax(-1).tolist())}
    active = list(tokens)
    while active:
        finished = [s for s in active if tokens[s][-1] in stop or len(tokens[s]) == len(replies[s])]
        for seq_id in finished:
            cache.release(seq_id)
        active = [s for s in active if s not in finished]
        if not active:
            break
        for seq_id in active:
            assert cache.reserve(seq_id, 1)
        batch = cache.begin([(seq_id, 1) for seq_id in active])
        last = torch.tensor([tokens[s][-1] for s in active], device=device)
        step = model(last, batch.positions, cache).argmax(-1).tolist()
        cache.commit()
        for seq_id, token in zip(active, step, strict=True):
            tokens[seq_id].append(token)
    same = sum(tokens[s] == replies[s] for s in tokens)
    print(f"  replies equal to one request at a time: {same} of {len(prompts)}")
    print(f"  after every sequence finished: {cache.pool.num_free} of {blocks} blocks free")

per_token = bytes_per_token(config, model.lm_head.weight.dtype)
budget = BUDGET // per_token
requests = [(len(p), len(r)) for p, r in zip(prompts, replies, strict=True)]
mib = per_token / (1 << 20)
print(f"\nsessions in a {BUDGET >> 30} GiB KV budget: {budget:,} tokens at {per_token:,} B each")
print("  every session holds one of the five turns, in rotation")
fits = {
    f"contiguous, prompt + max_tokens {MAX_TOKENS}": fit_contiguous(requests, MAX_TOKENS, budget),
    f"contiguous, whole {config.max_position_embeddings:,}-token context": fit_contiguous(
        [(0, p + r) for p, r in requests], config.max_position_embeddings, budget
    ),
    f"paged, block {BLOCK}": fit_paged(requests, BLOCK, budget),
}
for name, fit in fits.items():
    print(
        f"  {name:44} {fit.sessions:3} sessions   live {fit.live_tokens:6,} tokens   "
        f"held {fit.held_tokens:6,}   wasted {fit.wasted_tokens * mib:7.1f} MiB "
        f"({fit.wasted_tokens / max(fit.held_tokens, 1):.1%} of held)"
    )

print("\nblock size: waste in the last block vs block table size")
print("  block   wasted tokens per sequence   wasted KiB   table entries, longest turn / 32,768")
lengths = [p + r for p, r in requests]
for size in (1, 4, 8, 16, 32, 64, 128, 256):
    waste = statistics.mean(blocks_for(n, size) * size - n for n in lengths)
    entries = blocks_for(max(lengths), size), blocks_for(config.max_position_embeddings, size)
    print(
        f"  {size:5}   {waste:26.1f}   {waste * per_token / 1024:10.1f}   "
        f"{entries[0]:13,} / {entries[1]:,}"
    )
