# Why decode is slow

A decode step makes one token from about a gigaflop of maths. So why does it take 65 ms on a GPU that can do more than a teraflop per second?

To make that token, the step reads every weight of the model from memory. Qwen2.5-0.5B in fp32 is 2 GB of weights, and the step does half a FLOP for each byte it reads. The GPU spends most of the step waiting on memory.

```bash
uv run python roofline/machine.py
uv run python roofline/batch_sweep.py
uv run python roofline/launch.py
uv run python roofline/shapes.py
uv run python roofline/deepseek.py
```

The outputs below are one run on an M1 Mac with 16 GB. Times move from run to run.

## Two speeds

```
this Mac, measured (median of 20)
  copy bandwidth, GPU (MPS)     48.6 GB/s   (bytes read + written)
  copy bandwidth, CPU           35.8 GB/s
  matmul 2048x2048, float32     1.27 TFLOP/s
  matmul 2048x2048, float16     1.84 TFLOP/s
  one tiny op on MPS             8.5 us   (median of 10 x 1000 ops)

ridge point = peak / bandwidth, FLOPs per byte
  fp32    26.2
  fp16    37.8
```

A large copy measures how fast memory moves bytes, and a large matmul measures how fast the GPU does maths. The ridge point is one divided by the other. Work that does fewer FLOPs per byte than the ridge waits on memory, and work that does more waits on the maths. The time is the slower of the two, `max(bytes / bandwidth, FLOPs / peak)`.

## Decode against the roofline

```
one forward pass of B tokens, no cache (median of 5)
    B   measured ms   tokens/s   predicted ms   measured/predicted   FLOP/byte
    1          53.7         19           41.0                  1.3         0.5
    2          54.7         37           41.0                  1.3         1.0
    4          55.4         72           41.1                  1.3         2.0
    8          57.7        139           41.4                  1.4         4.0
   16          93.8        171           41.9                  2.2         7.8
   32          97.3        329           42.8                  2.3        15.3
   64         102.5        625           48.5                  2.1        29.4
  128         150.4        851           97.2                  1.5        54.4
  256         279.5        916          195.5                  1.4        94.6
  512         486.3       1053          395.4                  1.2       150.9

the agent prompt, 870 tokens (prefill median of 5, decode median of 10)
            FLOP/byte    GFLOP   GB moved   measured ms   achieved
  prefill       200.8    892.1       4.44         905.4   0.99 TFLOP/s, 4.9 GB/s
  decode          0.5      1.1       2.00          65.9   0.02 TFLOP/s, 30.3 GB/s
```

- One decode step does 1.1 GFLOP over 2 GB, half a FLOP per byte and about fifty times below the ridge. Its time is set by bytes over bandwidth.
- A pass of B tokens still reads every weight once, and each weight now serves B rows. Up to 8 tokens the step costs about the same, so tokens per second go from 19 to 139. From 128 tokens the step grows with B, because it is past the ridge.
- Prefill of the 870-token prompt does 200 FLOPs per byte. It waits on the maths, not on memory.

## What else is in a step

```
one decode step after the 870-token prompt (median of 20, matmuls alone 10)
                      bytes/bandwidth   ops   ops x tiny op   step      matmuls alone
  float32                    41.7 ms   1472         6.7 ms    60.8 ms  169 in 52.0 ms = 38.0 GB/s
  float32 fused              41.7 ms   1400         6.3 ms    57.1 ms  97 in 45.8 ms = 43.1 GB/s
  float32 compiled           41.7 ms      -               -    65.6 ms
  float16                    20.9 ms   1472         6.7 ms    35.1 ms  169 in 29.4 ms = 33.6 GB/s
  float16 fused              20.9 ms   1400         6.3 ms    33.3 ms  97 in 25.8 ms = 38.3 GB/s
  float16 compiled           20.9 ms      -               -    35.6 ms
```

- The 169 weight matmuls, run back to back, take 52 ms of the 61 ms fp32 step. The other 1,303 ops (norms, RoPE, cache writes, attention, activations) are small, and each one is launched on its own.
- Fusing q, k and v into one matmul, and gate and up into another, removes 72 ops. It made no reliable difference across runs, because the fused matmuls read the same bytes.

## Shapes

```
one decode step after the 870-token prompt, median of 32 lengths
                                                wall ms   host CPU ms
  cache, KV length never seen before              68.3          30.3
  cache, same KV length a second time             63.1          24.4
  whole buffer + mask, one shape every step       62.8          24.6
  ratio to the same length a second time, median of 32 interleaved pairs
    new KV length                                 1.08x         1.22x
    whole buffer + mask                           0.99x         0.99x
```

- The KV length grows by one every step, so every step hands the GPU a shape it has not seen. The first time a shape runs costs about 8% more than the second.
- `StaticCache` attends over the whole preallocated buffer with a mask, so every step has the same shapes and gets that time back.

## DeepSeek-V3 from its config

```
DeepSeek-V3 from its config.json
  total parameters              671,026,419,200
  active per token               37,552,297,472  (5.6%)
  matmul parameters / token      36,624,596,992

decode, batch 1, FP8 weights with 128x128 scales, BF16 latent KV
  context   weight GB   KV GB    GFLOP   FLOP/byte
    4,096       36.63   0.288    142.8         3.9
    4,989       36.63   0.351    158.0         4.3
   32,768       36.63   2.303    630.0        16.2

decode at 4,096 context, B sequences per step, uniform routing
      B   experts touched   weight GB    KV GB   FLOP/weight byte   FLOP/byte
      1              3.1%        36.6      0.3                3.9         3.9
      8             22.4%       162.9      2.3                7.0         6.9
     64             86.9%       584.5     18.4               15.6        15.2
    256            100.0%       670.1     73.7               54.6        49.2
  1,024            100.0%       670.3    294.7              218.2       151.6
  8,192            100.0%       670.3   2357.9             1745.8       386.4

prefill, one prompt of N tokens
        N    TFLOP   FLOP/weight byte
    4,096      342                510
   32,768     5083              7,584

DeepSeek-V4.1-Flash experts, FP4 with one scale byte per 32
  all experts       544,997,376,000 params    289.5 GB
  active / token      9,909,043,200 params     5.26 GB
```

- Of DeepSeek-V3's 671 B parameters, 37.6 B are used per token. At batch 1 a decode step reads 36.6 GB of FP8 weights for 143 GFLOP, about 4 FLOPs per byte.
- A bigger batch reads more experts: 22% at 8 sequences, 87% at 64. Past 256 every expert is read, and each sequence's own KV cache keeps the step below the ridge.

## Code

- `src/infer/roofline.py`: `Cost` (FLOPs, weight, KV and activation bytes, intensity, roofline time), `dense_params` and `dense_pass` for our model, and DeepSeek parameter, decode and prefill costs from their configs.
- `src/infer/measure.py`: `median_time`, `copy_bandwidth`, `matmul_peak`, `op_time`, and `OpCounter`, which counts the ops a forward launches.
- `src/infer/fused.py`: `fuse_projections`, one matmul for q, k and v and one for gate and up.
- `src/infer/static.py`: `StaticCache`, attention over the whole buffer with a mask so every decode step has the same shapes.
