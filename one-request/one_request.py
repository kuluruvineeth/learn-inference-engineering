from infer.generate import generate, generate_naive
from infer.hf import download, load_model
from infer.sampling import SamplingParams
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session

model = load_model()
tokenizer = Tokenizer(download(weights=False))
prompt = tokenizer.encode(chat(session(0), TOOLS))
print(f"device={model.lm_head.weight.device.type} prompt={len(prompt)} tokens")
print(f"last prompt tokens: {tokenizer.decode(prompt[-12:])!r}")

params = SamplingParams(max_tokens=48)
generate(model, prompt, SamplingParams(max_tokens=2), tokenizer.stop_ids)
cached = generate(model, prompt, params, tokenizer.stop_ids)
print(f"\nwith a KV cache: {len(cached.tokens)} tokens")
print(f"  time to first token {cached.ttft * 1000:7.0f} ms")
print(f"  inter-token latency {cached.itl * 1000:7.1f} ms   ({1 / cached.itl:.1f} tokens/s)")
print(f"  reply: {tokenizer.decode(cached.tokens)!r}")

naive = generate_naive(model, prompt, SamplingParams(max_tokens=8), tokenizer.stop_ids)
print("\nrecomputing the whole sequence every step: 8 tokens")
print(f"  time to first token {naive.ttft * 1000:7.0f} ms")
print(f"  inter-token latency {naive.itl * 1000:7.1f} ms")
print(f"  same tokens: {naive.tokens == cached.tokens[:8]}")

sampled = generate(model, prompt, SamplingParams(0.7, 0.8, 48, seed=1), tokenizer.stop_ids)
print(f"\ntemperature 0.7, top-p 0.8, seed 1: {tokenizer.decode(sampled.tokens)!r}")
