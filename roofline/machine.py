import torch

from infer.measure import copy_bandwidth, matmul_peak, op_time

print("this Mac, measured (median of 20)")
mps, cpu = copy_bandwidth("mps"), copy_bandwidth("cpu")
print(f"  copy bandwidth, GPU (MPS)   {mps / 1e9:6.1f} GB/s   (bytes read + written)")
print(f"  copy bandwidth, CPU         {cpu / 1e9:6.1f} GB/s")
peaks = {dtype: matmul_peak("mps", dtype) for dtype in (torch.float32, torch.float16)}
for dtype, peak in peaks.items():
    print(f"  matmul 2048x2048, {str(dtype)[6:]:8s}  {peak / 1e12:6.2f} TFLOP/s")
print(f"  one tiny op on MPS          {op_time('mps') * 1e6:6.1f} us   (median of 10 x 1000 ops)")

print("\nridge point = peak / bandwidth, FLOPs per byte")
print(f"  fp32   {peaks[torch.float32] / mps:5.1f}")
print(f"  fp16   {peaks[torch.float16] / mps:5.1f}")
