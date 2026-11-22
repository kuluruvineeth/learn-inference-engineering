import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from infer.model import Config, Model

CONFIG = Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2)


def reference_pair() -> tuple[Model, Qwen2ForCausalLM]:
    torch.manual_seed(0)
    reference = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=CONFIG.vocab_size,
            hidden_size=CONFIG.hidden_size,
            intermediate_size=CONFIG.intermediate_size,
            num_hidden_layers=CONFIG.num_hidden_layers,
            num_attention_heads=CONFIG.num_attention_heads,
            num_key_value_heads=CONFIG.num_key_value_heads,
            rms_norm_eps=CONFIG.rms_norm_eps,
            rope_theta=CONFIG.rope_theta,
            tie_word_embeddings=True,
        )
    ).eval()
    model = Model(CONFIG).eval()
    weights = {k.removeprefix("model."): v for k, v in reference.state_dict().items()}
    model.load_state_dict(weights)
    return model, reference


def test_logits_match_reference():
    model, reference = reference_pair()
    tokens = torch.randint(0, CONFIG.vocab_size, (12,))
    with torch.no_grad():
        ours = model(tokens, torch.arange(12))
        theirs = reference(tokens[None]).logits[0]
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)
