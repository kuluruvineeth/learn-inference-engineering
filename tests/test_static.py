import torch

from infer.kv import ContiguousCache
from infer.model import Config, Model
from infer.static import StaticCache

CONFIG = Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2)


def test_whole_buffer_attention_matches_the_contiguous_cache():
    torch.manual_seed(0)
    model, tokens = Model(CONFIG).eval(), torch.randint(0, 97, (12,))
    outputs = []
    with torch.no_grad():
        for kind in (ContiguousCache, StaticCache):
            cache = kind(CONFIG, 32)
            logits = [model(tokens[:8], torch.arange(8), cache)]
            cache.advance(8)
            for position in range(8, 12):
                logits.append(
                    model(tokens[position : position + 1], torch.tensor([position]), cache)
                )
                cache.advance(1)
            outputs.append(torch.cat(logits))
    torch.testing.assert_close(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)
