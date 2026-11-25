# The KV cache

In our coding agent session, the first turn is 893 tokens long. Before its first token came out, the cache had set aside 4,966 slots for it. So where does all that empty memory go, and what does it cost a server?

The cache from one request reserves room for the prompt plus `max_tokens` before the model writes anything. An agent reply is usually a short tool call, so almost all of that room stays empty. On a server that holds many sessions at once, it is the cache, not the weights, that decides how many fit.

```bash
uv run python kv-cache/kv_cache.py
```

The output below is one run on an M1 Mac with 16 GB.

## What one token costs

```
bytes per token, K and V at every layer
  Qwen2.5-0.5B GQA, 2 KV heads x 64 x 24 layers, torch.float32           24,576 B   allocated tensor: 24,576 B per slot
  Qwen2.5-0.5B GQA, 2 KV heads x 64 x 24 layers, torch.bfloat16          12,288 B   allocated tensor: 12,288 B per slot
  Qwen2.5-0.5B if all 14 heads kept K and V, torch.bfloat16              86,016 B
  DeepSeek-V3 per-head K (128 + 64) and V (128), 128 heads x 61 layers, bf16  4,997,120 B
  DeepSeek-V3 MLA latent 512 + 64, 61 layers, bf16                       70,272 B   71.1x smaller
```

- Every layer keeps a key and a value for each KV head and every token it has seen. Qwen2.5-0.5B shares 2 KV heads among its 14 query heads, so a token costs 2 x 2 x 64 x 24 = 6,144 values, 24,576 bytes in fp32.
- DeepSeek-V3 stores one latent of 512 + 64 values per layer instead of a K and V per head, 71 times smaller.

## The same replies on blocks

```
one request at a time on the paged cache, block 16, greedy, max_tokens 4096
  turn 0: prompt   870  reply   23  blocks  56  same tokens as the contiguous cache: True
  turn 1: prompt   972  reply   34  blocks  63  same tokens as the contiguous cache: True
  turn 2: prompt  1020  reply   22  blocks  66  same tokens as the contiguous cache: True
  turn 3: prompt  1089  reply   48  blocks  72  same tokens as the contiguous cache: True
  turn 4: prompt  1132  reply   16  blocks  72  same tokens as the contiguous cache: True

all five turns as one batch: one flat forward for the five prompts, then decoded together
  prefill: 5083 tokens in one forward, 9 blocks free
  last-token logits vs each prompt alone: max abs difference 0.00e+00
  replies equal to one request at a time: 5 of 5
  after every sequence finished: 329 of 329 blocks free
```

- The paged cache cuts each layer's storage into blocks of 16 slots and gives a request a block only when it fills one. Each request keeps a block table, and the token at position p lives at slot `table[p // 16] * 16 + p % 16`.
- Every layer writes the new keys and values into their slots, gathers each sequence's blocks back into order through its table, and attends over that sequence's tokens only. Five prompts go through one forward of 5,083 tokens and give the same logits as each prompt alone.

## Sessions in 1 GiB

```
sessions in a 1 GiB KV budget: 43,690 tokens at 24,576 B each
  every session holds one of the five turns, in rotation
  contiguous, prompt + max_tokens 4096           8 sessions   live  8,167 tokens   held 40,713   wasted   762.8 MiB (79.9% of held)
  contiguous, whole 32,768-token context         1 sessions   live    893 tokens   held 32,768   wasted   747.1 MiB (97.3% of held)
  paged, block 16                               41 sessions   live 42,701 tokens   held 43,008   wasted     7.2 MiB (0.7% of held)

block size: waste in the last block vs block table size
  block   wasted tokens per sequence   wasted KiB   table entries, longest turn / 32,768
      1                          0.0          0.0           1,148 / 32,768
      4                          2.0         48.0             287 / 8,192
      8                          4.4        105.6             144 / 4,096
     16                          7.6        182.4              72 / 2,048
     32                         10.8        259.2              36 / 1,024
     64                         17.2        412.8              18 / 512
    128                         30.0        720.0               9 / 256
    256                        132.4       3177.6               5 / 128
```

- Reserving prompt + 4,096 tokens fits 8 sessions, and 80% of what they hold stays empty. On blocks, the only waste is the unused end of each request's last block, so the same 1 GiB holds 41.
- Smaller blocks waste less in the last block but need longer tables, one entry per block, read on every step.

## Code

- `src/infer/paged.py`: `BlockPool` (free queue, null block, reference counts, all-or-nothing allocation), `PagedCache` (block storage, block tables, `reserve`, `begin`, `attend`, `commit`, `release`) and `generate_paged`.
- `src/infer/kv_size.py`: bytes per token for grouped KV heads, per-head K and V and an MLA latent, and how many sessions fit a budget with contiguous reservations or with blocks.
