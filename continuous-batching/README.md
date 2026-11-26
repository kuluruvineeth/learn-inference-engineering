# Continuous batching

Ten requests share our engine, at most four at a time. Run them as fixed batches and 45% of the batch slots carry nobody. So who should go into each forward pass?

A static batch admits only when nothing is running and then waits for its longest reply. Every request here keeps one token list and a count of the tokens already computed, and the difference is its work: 1 for a decode, the prompt length for a new request. Before every forward pass the scheduler hands out a token budget, running requests first and decodes before prompts, then lets waiting requests in with what is left.

The output below is one run on an M1 Mac with 16 GB. The steps, tokens and replies are the same on every run. The times move from run to run.

## Static and continuous batching

```bash
uv run python continuous-batching/static_vs_continuous.py
```

```
device=mps
10 requests: the 5 agent turns, twice (request i is turn i % 5)
replies one at a time (greedy): [23, 34, 22, 48, 16] tokens
at most 4 running, budget 8192 tokens per step, block 16, 600 blocks

static batch
    step   0  +0 (870) +1 (972) +2 (1020) +3 (1089)            running 4  tokens  3951
    step  22  -2                                               running 3  tokens     3
    step  23  -0                                               running 2  tokens     2
    step  34  -1                                               running 1  tokens     1
    step  48  -3 +4 (1132) +5 (870) +6 (972) +7 (1020)         running 4  tokens  3994
    step  64  -4                                               running 3  tokens     3
    step  70  -7                                               running 2  tokens     2
    step  71  -5                                               running 1  tokens     1
    step  82  -6 +8 (1089) +9 (1132)                           running 2  tokens  2221
    step  98  -9                                               running 1  tokens     1
  forward passes 130   slot-steps 520   busy 286   idle 234 (45%)
  replies equal to one at a time: 10 of 10   generated 286
  wall time 16.4 s (median of 3)   17.4 tokens/s
      3 steps with a prompt     8.4 s   127 decode-only steps     7.9 s (62 ms each)

continuous (iteration-level)
    step   0  +0 (870) +1 (972) +2 (1020) +3 (1089)            running 4  tokens  3951
    step  22  -2 +4 (1132)                                     running 4  tokens  1135
    step  23  -0 +5 (870)                                      running 4  tokens   873
    step  34  -1 +6 (972)                                      running 4  tokens   975
    step  38  -4 +7 (1020)                                     running 4  tokens  1023
    step  46  -5 +8 (1089)                                     running 4  tokens  1092
    step  48  -3 +9 (1132)                                     running 4  tokens  1135
    step  60  -7                                               running 3  tokens     3
    step  64  -9                                               running 2  tokens     2
    step  68  -6                                               running 1  tokens     1
  forward passes 94   slot-steps 376   busy 286   idle 90 (24%)
  replies equal to one at a time: 10 of 10   generated 286
  wall time 14.3 s (median of 3)   20.0 tokens/s
      7 steps with a prompt     8.5 s    87 decode-only steps     5.8 s (67 ms each)

static / continuous: forward passes 1.38x, wall time 1.15x
```

- In the static batch, turn 2 finishes at step 22, but its slot stays empty until turn 3 ends at step 48. Of 520 slot-steps, 234 carry nobody.
- The continuous scheduler plans again before every step, so a finished request leaves and the next one joins in the same step (`-2 +4` at step 22). It takes 94 forward passes instead of 130, and the only idle slots are at the end, when nobody is left to join.
- Both give the same 10 replies as one request at a time. Wall time gains less than the passes, 1.15x against 1.38x, because a decode step that carries four requests costs more than one that carries one.

## Chunked prefill

```bash
uv run python continuous-batching/chunked_prefill.py
```

```
4 turns decoding; turn 4 (1132 prompt tokens) arrives once each has 3 tokens
decode-first: running decodes take the budget, the new prompt gets what is left
timings are medians of 5 runs after a warm-up, the budgets taking turns

  budget   prefill   largest step   decode gap   decode gap   max / p50   TTFT turn 4   equal
           steps     (tokens)       p50 (ms)     max (ms)                 (ms)
    2048         1           1136           74          999        12.9           999   25/25
           chunks of turn 4: [1132]
     512         3            512           74          447         6.0          1011   25/25
           chunks of turn 4: [508, 508, 116]
     256         5            256           71          269         3.8          1152   25/25
           chunks of turn 4: [252, 252, 252, 252, 124]
     128        10            128           68          153         2.3          1354   25/25
           chunks of turn 4: [125, 125, 125, 125, 125, 125, 125, 125, 125, 7]

equal: replies equal to one request at a time, over every run
```

- Four turns are decoding when turn 4 arrives with 1,132 prompt tokens. Each step gives every decode its one token first, then cuts the prompt to fit what is left of the budget: 508 of 512.
- At budget 2048 the whole prompt goes into one step and every decoding turn waits 999 ms for its next token, 12.9 times its usual gap. At 128 the prompt goes in ten chunks and the worst gap is 153 ms.
- The new turn pays for it. Its first token comes after 1,354 ms instead of 999 ms, because its prompt now takes ten forward passes. The budget is a dial between the two.

## Preemption by recompute

```bash
uv run python continuous-batching/preemption.py
```

```
the 5 turns at once: prompts [870, 972, 1020, 1089, 1132] tokens, replies [23, 34, 22, 48, 16]
KV pool 249 blocks of 16 = 3984 slots; the prompts alone need 320 blocks
budget 512 tokens per step, decode-first, preemption by recompute

  admission        victim policy    preemptions   recomputed tokens   steps   equal
  whole sequence   newest                     1                1089      73   5/5
  whole sequence   fewest_output              1                1089      73   5/5
  whole sequence   most_output                1                 876      55   5/5
  first chunk      newest                     6                6179      73   5/5
  first chunk      fewest_output              6                6179      73   5/5
  first chunk      most_output               12               11519      58   5/5

victims, whole sequence admission, newest:
  step   8: turn 3 evicted,  1 tokens generated, KV of 1089 tokens dropped, 69 blocks freed

victims, first chunk admission, newest:
  step   8: turn 3 evicted,  1 tokens generated, KV of 1089 tokens dropped, 69 blocks freed
  step  11: turn 3 evicted,  1 tokens generated, KV of 1018 tokens dropped, 64 blocks freed
  step  14: turn 3 evicted,  1 tokens generated, KV of 1018 tokens dropped, 64 blocks freed
  step  17: turn 3 evicted,  1 tokens generated, KV of 1018 tokens dropped, 64 blocks freed
  step  20: turn 3 evicted,  1 tokens generated, KV of 1018 tokens dropped, 64 blocks freed
  step  23: turn 3 evicted,  1 tokens generated, KV of 1018 tokens dropped, 64 blocks freed

victims, first chunk admission, most_output:
  step   8: turn 0 evicted,  7 tokens generated, KV of  877 tokens dropped, 55 blocks freed
  step  10: turn 1 evicted,  7 tokens generated, KV of  979 tokens dropped, 62 blocks freed
  step  12: turn 0 evicted,  9 tokens generated, KV of  879 tokens dropped, 55 blocks freed
  step  14: turn 2 evicted,  9 tokens generated, KV of 1029 tokens dropped, 65 blocks freed
  step  16: turn 1 evicted, 11 tokens generated, KV of  983 tokens dropped, 62 blocks freed
  step  18: turn 0 evicted, 13 tokens generated, KV of  883 tokens dropped, 56 blocks freed
  ... 6 more
```

- The five prompts alone need 320 blocks and the pool has 249. A request is admitted only if blocks for its whole sequence fit, but its reply is not reserved, so a running request can still find no block for its next token.
- Then the scheduler evicts a victim: it frees the victim's blocks, sets its computed count to zero and puts it at the front of the queue. Its tokens are kept, so when it comes back one prefill covers the prompt and what it had generated. Every policy gives the same replies.
- Admitting a request when only its first chunk fits evicts turn 3 every three steps. Under `most_output` it goes to 12 evictions and 11,519 recomputed tokens, more than twice all five prompts.

## In-flight depth

```bash
uv run python continuous-batching/in_flight_depth.py
```

```
the 5 turns at once, budget 8192, greedy; each step plans the next before the previous one's tokens are on the host
each depth's row is its run with the median wall time of 3 after a warm-up,
the depths taking turns

  depth   steps   tokens kept   tokens sampled   wasted   equal   wall time   tokens/s
      0      48           143              143        0   5/5       7.8 s       18.3
      1      49           143              148        5   5/5       7.9 s       18.1
      2      50           143              153       10   5/5       8.2 s       17.5
  wasted per turn at depth 1: [1, 1, 1, 1, 1]

where a step's time goes in the same runs, ms per step
  depth   schedule     launch       wait     update       step   x steps   wall time
      0        0.0       46.0      116.7        0.0      162.7     7.8 s       7.8 s
      1        0.1      132.3       28.7        0.0      161.1     7.9 s       7.9 s

simulated depth 1 on a device whose host can read step N while step N+1 runs:
  launch + max(wait, schedule + update) = 162.6 ms per step vs 162.7 measured at depth 0 (0.1% less)

why depth 1 gains nothing here (mps), a stand-in forward of 30 chained 2048 x 2048 matmuls, ms, median of 3
  launch one                                             4.4
  launch one, read its result                          454.0
  a tiny op, then one, read the tiny op's result       484.4
  one queued, then torch.tensor(list, device=...)      447.7
  one queued, then .to(device, non_blocking=True)        0.2
```

- At depth 1 the engine launches step N+1 before it reads the tokens of step N. The scheduler counts a token when it plans it, so the next decode reads its input from the previous step's sampled tensor on the device.
- The price is one wasted token per turn: when a turn stops, one more decode is already launched. 148 tokens sampled for 143 kept, with the same replies.
- The gain would be the host's work hidden behind the forward. Here that work is a fraction of a millisecond of a 163 ms step, 0.1%, and on this Mac reading or building a tensor waits for the forward queued before it anyway. Depth 1 is not faster here.

## Code

- `src/infer/scheduler.py`: `Request` (one token list, `computed`, states waiting, running, finishing, finished), `Entry`, `Scheduler` (`schedule`, `admit`, `grow`, `preempt`, `update`, `accept`, `retire`), the victim policies `newest`, `fewest_output` and `most_output`, and `StaticScheduler`.
- `src/infer/engine.py`: `Engine`, one flat forward per step through the paged cache, decode inputs read on the device from the previous step's samples, and `depth` launches kept in flight.
