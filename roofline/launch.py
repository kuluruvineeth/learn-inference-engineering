import torch

from infer.fused import fuse_projections
from infer.hf import download, load_model
from infer.kv import ContiguousCache
from infer.measure import copy_bandwidth, count_ops, median_time, op_time
from infer.roofline import dense_pass
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session

prompt = Tokenizer(download(weights=False)).encode(chat(session(0), TOOLS))
bandwidth, per_op = copy_bandwidth("mps"), op_time("mps")
print(
    f"measured: {bandwidth / 1e9:.1f} GB/s copy (median of 20),"
    f" {per_op * 1e6:.1f} us per tiny op on MPS (median of 10 x 1000 ops)"
)


def decode_step(model, forward=None):
    forward = model if forward is None else forward
    weight = model.lm_head.weight
    cache = ContiguousCache(model.config, len(prompt) + 1, weight.dtype, weight.device)
    ids = torch.tensor(prompt, device=weight.device)
    model(ids, torch.arange(len(prompt), device=weight.device), cache)
    cache.advance(len(prompt))
    position = torch.tensor([len(prompt)], device=weight.device)
    return lambda: forward(ids[-1:], position, cache)


def matmuls_only(model):
    weights = [m.weight for m in model.modules() if isinstance(m, torch.nn.Linear)]
    inputs = {w.size(1): torch.randn(1, w.size(1), device=w.device, dtype=w.dtype) for w in weights}
    seconds = median_time(
        lambda: [torch.nn.functional.linear(inputs[w.size(1)], w) for w in weights], "mps", 10
    )
    return len(weights), seconds, sum(w.numel() * w.element_size() for w in weights)


rows = []
with torch.inference_mode():
    for dtype, size in ((torch.float32, 4), (torch.float16, 2)):
        model = load_model(dtype=dtype)
        floor = dense_pass(model.config, 1, len(prompt), value_bytes=size).bytes / bandwidth
        step = decode_step(model)
        ops, seconds = count_ops(step).total, median_time(step, "mps", 20)
        count, matmul_seconds, matmul_bytes = matmuls_only(model)
        rows.append((str(dtype)[6:], "", floor, ops, seconds, count, matmul_seconds, matmul_bytes))
        step = decode_step(fuse_projections(model))
        ops, seconds = count_ops(step).total, median_time(step, "mps", 20)
        count, matmul_seconds, matmul_bytes = matmuls_only(model)
        rows.append(
            (str(dtype)[6:], "fused", floor, ops, seconds, count, matmul_seconds, matmul_bytes)
        )
        step = decode_step(model, torch.compile(model))
        seconds = median_time(step, "mps", 20)
        rows.append((str(dtype)[6:], "compiled", floor, None, seconds, None, None, None))
        del model, step

print(f"\none decode step after the {len(prompt)}-token prompt (median of 20, matmuls alone 10)")
print("                      bytes/bandwidth   ops   ops x tiny op   step      matmuls alone")
for dtype, kind, floor, ops, seconds, count, matmul_seconds, matmul_bytes in rows:
    line = f"  {dtype:7} {kind:8}   {floor * 1000:12.1f} ms"
    if ops is None:
        print(f"{line}      -               -  {seconds * 1000:6.1f} ms")
        continue
    print(
        f"{line}  {ops:5}  {ops * per_op * 1000:10.1f} ms  {seconds * 1000:6.1f} ms"
        f"  {count} in {matmul_seconds * 1000:4.1f} ms"
        f" = {matmul_bytes / matmul_seconds / 1e9:4.1f} GB/s"
    )
