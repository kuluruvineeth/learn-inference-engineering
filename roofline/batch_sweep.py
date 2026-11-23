import torch

from infer.hf import download, load_model
from infer.kv import ContiguousCache
from infer.measure import copy_bandwidth, matmul_peak, median_time
from infer.roofline import dense_params, dense_pass
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session

model = load_model()
device, config = model.lm_head.weight.device.type, model.config
bandwidth, peak = copy_bandwidth(device), matmul_peak(device, torch.float32)
print(f"Qwen2.5-0.5B, fp32 on {device}: {dense_params(config).total:,} parameters")
print(
    f"measured (median of 20): {bandwidth / 1e9:.1f} GB/s copy,"
    f" {peak / 1e12:.2f} TFLOP/s fp32 matmul"
)
print(
    f"predicted time = max(bytes / {bandwidth / 1e9:.1f} GB/s, FLOPs / {peak / 1e12:.2f} TFLOP/s)"
)

print("\none forward pass of B tokens, no cache (median of 5)")
print("    B   measured ms   tokens/s   predicted ms   measured/predicted   FLOP/byte")
with torch.inference_mode():
    for batch in [2**i for i in range(10)]:
        tokens = torch.randint(0, config.vocab_size, (batch,), device=device)
        positions = torch.arange(batch, device=device)
        seconds = median_time(lambda t=tokens, p=positions: model(t, p), device, repeats=5)
        cost = dense_pass(config, batch)
        predicted = cost.time(bandwidth, peak)
        print(
            f"  {batch:3}   {seconds * 1000:11.1f}   {batch / seconds:8.0f}"
            f"   {predicted * 1000:12.1f}   {seconds / predicted:18.1f}   {cost.intensity:9.1f}"
        )

    tokenizer = Tokenizer(download(weights=False))
    prompt = tokenizer.encode(chat(session(0), TOOLS))
    ids = torch.tensor(prompt, device=device)
    positions = torch.arange(len(prompt), device=device)
    cache = ContiguousCache(config, len(prompt) + 1, model.lm_head.weight.dtype, device)
    prefill_seconds = median_time(lambda: model(ids, positions, cache), device, repeats=5)
    cache.advance(len(prompt))
    last, position = ids[-1:], positions[-1:] + 1
    decode_seconds = median_time(lambda: model(last, position, cache), device, repeats=10)

print(f"\nthe agent prompt, {len(prompt)} tokens (prefill median of 5, decode median of 10)")
print("            FLOP/byte    GFLOP   GB moved   measured ms   achieved")
for name, cost, seconds in [
    ("prefill", dense_pass(config, len(prompt)), prefill_seconds),
    ("decode", dense_pass(config, 1, len(prompt)), decode_seconds),
]:
    print(
        f"  {name:8}  {cost.intensity:9.1f}   {cost.flops / 1e9:6.1f}   {cost.bytes / 1e9:8.2f}"
        f"   {seconds * 1000:11.1f}   {cost.flops / seconds / 1e12:.2f} TFLOP/s,"
        f" {cost.bytes / seconds / 1e9:.1f} GB/s"
    )
