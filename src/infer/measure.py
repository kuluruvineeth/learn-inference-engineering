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
