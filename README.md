# Learn inference engineering

What happens between sending a prompt and the tokens that come back? And why can the same model run 30 times faster with a different engine around it?

- [What is inference engineering?](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/inference-engineering/inference-engineering.excalidraw) looks at what an engine does with one agent turn, who waits and who pays, and why a decode step spends most of its time reading weights rather than doing maths.
- [One request](one-request/) follows one coding agent turn, 870 tokens, through Qwen2.5-0.5B in plain PyTorch, and measures what each token costs with and without a KV cache.

## Run it

```bash
uv sync
uv run pytest -q
uv run python one-request/one_request.py
```

The first run downloads Qwen2.5-0.5B-Instruct from Hugging Face. The code runs on an Apple Silicon GPU or a CPU.

## Diagrams

Each board is attached to its release and kept in `boards/`. Open it in [Excalidraw](https://excalidraw.com).

- [inference-engineering.excalidraw](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/inference-engineering/inference-engineering.excalidraw)
- [one-request.excalidraw](https://github.com/kuluruvineeth/learn-inference-engineering/releases/download/one-request/one-request.excalidraw)

## License

[Apache-2.0](LICENSE).
