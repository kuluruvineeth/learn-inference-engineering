import statistics
import time
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def median_time(fn, device: str, repeats: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def copy_bandwidth(device: str, megabytes: int = 512, repeats: int = 20) -> float:
    source = torch.ones(megabytes * 2**20 // 4, device=device)
    target = torch.empty_like(source)
    seconds = median_time(lambda: target.copy_(source), device, repeats)
    return 2 * source.numel() * source.element_size() / seconds


def matmul_peak(device: str, dtype: torch.dtype, n: int = 2048, repeats: int = 20) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    return 2 * n**3 / median_time(lambda: a @ b, device, repeats)


def op_time(device: str, ops: int = 1000, repeats: int = 10) -> float:
    x = torch.zeros(1, device=device)

    def chain():
        for _ in range(ops):
            x.add_(1)

    return median_time(chain, device, repeats) / ops


def _storages(values) -> set[int]:
    flat = values if isinstance(values, (list, tuple)) else [values]
    return {v.untyped_storage().data_ptr() for v in flat if isinstance(v, torch.Tensor)}


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        inputs = _storages(list(args) + list((kwargs or {}).values()))
        if not func.is_view and (func._schema.is_mutable or not _storages(out) <= inputs):
            self.ops[func.overloadpacket.__name__] += 1
        return out

    @property
    def total(self) -> int:
        return sum(self.ops.values())


def count_ops(fn) -> OpCounter:
    counter = OpCounter()
    with counter:
        fn()
    return counter
