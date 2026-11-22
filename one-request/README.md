# One request

A model writes its answer one token at a time. So what does it do with your prompt, and what does each token cost?

One turn of a coding agent: a system prompt with the repository and three tools, then the user's message. The chat template turns it into 870 tokens, and the model reads all of them before it writes anything.

```bash
uv run python one-request/one_request.py
```

```
device=mps prompt=870 tokens
last prompt tokens: ' tests fail. Please fix them.<|im_end|>\n<|im_start|>assistant\n'

with a KV cache: 23 tokens
  time to first token     947 ms
  inter-token latency    65.5 ms   (15.3 tokens/s)
  reply: 'Sure, I can help with that. Please provide the failing test cases so I can review and fix them.<|im_end|>'

recomputing the whole sequence every step: 8 tokens
  time to first token     839 ms
  inter-token latency   940.3 ms
  same tokens: True

temperature 0.7, top-p 0.8, seed 1: 'Sure, I can help with that. Please provide the failing test case so I can review and fix it.<|im_end|>'
```

- **Prefill** runs all 870 prompt tokens through the model in one forward pass. Its last row of scores gives the first token, so this pass sets the time to first token.
- **Decode** runs one token per forward pass. The keys and values of every earlier token are already in the KV cache, so a step computes only the new token's row. The gap between tokens is the inter-token latency.
- Without the cache, every step runs the whole sequence again and each token takes about as long as the first one. Greedy decoding gives the same tokens either way.
- **Sampling**: temperature divides the scores before the softmax, top-p keeps the smallest set of tokens whose probabilities add up to p, and a seed makes a sampled run repeatable.

Times move from run to run.

## Code

- `src/infer/model.py`: Qwen2 in PyTorch. Attention hands q, k and v to whatever cache it is given; with none it attends causally over the tokens it was called with.
- `src/infer/kv.py`: `ContiguousCache`, one preallocated tensor per layer for one sequence.
- `src/infer/generate.py`: `generate` (prefill, then decode with the cache) and `generate_naive`.
- `src/infer/sampling.py`: greedy picks, temperature and top-p.
- `src/infer/tokenizer.py`: the chat template, checked against the official one for every turn of the session.
- `src/infer/hf.py`: downloads Qwen2.5-0.5B-Instruct and loads its weights.
- `src/infer/workload.py`: the coding agent session.
