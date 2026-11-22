import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from infer.model import Config, Model

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]


def download(name: str = QWEN, weights: bool = True) -> Path:
    patterns = FILES + (["*.safetensors"] if weights else [])
    return Path(snapshot_download(name, allow_patterns=patterns))


def load_config(path: Path) -> Config:
    hf = json.loads((Path(path) / "config.json").read_text())
    return Config(
        vocab_size=hf["vocab_size"],
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["intermediate_size"],
        num_hidden_layers=hf["num_hidden_layers"],
        num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf["num_key_value_heads"],
        max_position_embeddings=hf["max_position_embeddings"],
        rms_norm_eps=hf["rms_norm_eps"],
        rope_theta=hf["rope_theta"],
        tie_word_embeddings=hf["tie_word_embeddings"],
    )


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(
    name: str = QWEN, dtype: torch.dtype = torch.float32, device: str | None = None
) -> Model:
    path = download(name)
    model = Model(load_config(path))
    weights = {
        k.removeprefix("model."): v for k, v in load_file(path / "model.safetensors").items()
    }
    weights.setdefault("lm_head.weight", weights["embed_tokens.weight"])
    model.load_state_dict(weights)
    return model.to(device or default_device(), dtype).eval()
