# Learn inference engineering

What happens between sending a prompt and the tokens that come back? And why can the same model run 30 times faster with a different engine around it?

- [What is inference engineering?](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/inference-engineering/inference-engineering.excalidraw) looks at what an engine does with one agent turn, who waits and who pays, and why a decode step spends most of its time reading weights rather than doing maths.
- [One request](one-request/) follows one coding agent turn, 870 tokens, through Qwen2.5-0.5B in plain PyTorch, and measures what each token costs with and without a KV cache.
- [Why decode is slow](roofline/) measures the memory bandwidth and compute peak of the machine, counts the bytes and FLOPs of one decode step, and shows why the GPU spends that step waiting on memory. Then it does the same sums for DeepSeek-V3 from its config.
- [The KV cache](kv-cache/) measures what one token costs in the cache, cuts the cache into 16-token blocks with a block pool and block tables, and runs attention through the table. The same 1 GiB holds 41 agent sessions instead of 8.

## Run it

```bash
uv sync
uv run pytest -q
uv run python one-request/one_request.py
uv run python roofline/batch_sweep.py
uv run python kv-cache/kv_cache.py
```

The first run downloads Qwen2.5-0.5B-Instruct from Hugging Face. The code runs on an Apple Silicon GPU or a CPU.

## Diagrams

Each board is attached to its release and kept in `boards/`. Open it in [Excalidraw](https://excalidraw.com).

- [inference-engineering.excalidraw](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/inference-engineering/inference-engineering.excalidraw)
- [one-request.excalidraw](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/one-request/one-request.excalidraw)
- [roofline.excalidraw](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/roofline/roofline.excalidraw)
- [kv-cache.excalidraw](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/kv-cache/kv-cache.excalidraw)

## License

[Apache-2.0](LICENSE).
