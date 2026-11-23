from pathlib import Path

from infer.roofline import (
    deepseek_v3_decode,
    deepseek_v3_params,
    deepseek_v3_prefill,
    deepseek_v41_experts,
    experts_touched,
    load_json,
)

CONFIGS = Path(__file__).parent.parent / "configs"
v3 = load_json(CONFIGS / "deepseek-v3.json")
params = deepseek_v3_params(v3)
print("DeepSeek-V3 from its config.json")
print(f"  total parameters           {params.total:>18,}")
print(f"  active per token           {params.active:>18,}  ({params.active / params.total:.1%})")
print(f"  matmul parameters / token  {params.active_matmul:>18,}")

print("\ndecode, batch 1, FP8 weights with 128x128 scales, BF16 latent KV")
print("  context   weight GB   KV GB    GFLOP   FLOP/byte")
for context in (4096, 4989, 32768):
    cost = deepseek_v3_decode(v3, context)
    print(
        f"  {context:7,}   {cost.weight_bytes / 1e9:9.2f}   {cost.kv_bytes / 1e9:5.3f}"
        f"   {cost.flops / 1e9:6.1f}   {cost.intensity:9.1f}"
    )

print("\ndecode at 4,096 context, B sequences per step, uniform routing")
print("      B   experts touched   weight GB    KV GB   FLOP/weight byte   FLOP/byte")
for batch in (1, 8, 64, 256, 1024, 8192):
    cost = deepseek_v3_decode(v3, 4096, batch)
    touched = experts_touched(params.routed, params.top_k, batch) / params.routed
    print(
        f"  {batch:5,}   {touched:15.1%}   {cost.weight_bytes / 1e9:9.1f}"
        f"   {cost.kv_bytes / 1e9:6.1f}   {cost.flops / cost.weight_bytes:16.1f}"
        f"   {cost.intensity:9.1f}"
    )

print("\nprefill, one prompt of N tokens")
print("        N    TFLOP   FLOP/weight byte")
for tokens in (4096, 32768):
    cost = deepseek_v3_prefill(v3, tokens)
    print(f"  {tokens:7,}   {cost.flops / 1e12:6.0f}   {cost.intensity:16,.0f}")

v41 = deepseek_v41_experts(load_json(CONFIGS / "deepseek-v4.1-flash.json"))
print("\nDeepSeek-V4.1-Flash experts, FP4 with one scale byte per 32")
print(f"  all experts      {v41.total:>16,} params   {v41.total_bytes / 1e9:6.1f} GB")
print(f"  active / token   {v41.active:>16,} params   {v41.active_bytes / 1e9:6.2f} GB")
