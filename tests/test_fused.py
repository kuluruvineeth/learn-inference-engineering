import torch

from infer.fused import fuse_projections
from infer.kv import ContiguousCache
from infer.measure import count_ops
from infer.model import Config, Model

CONFIG = Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2)


def pair() -> tuple[Model, Model]:
    torch.manual_seed(0)
    model = Model(CONFIG).eval()
    fused = Model(CONFIG).eval()
    fused.load_state_dict(model.state_dict())
    return model, fuse_projections(fused)


def test_fused_projections_give_the_same_logits():
    model, fused = pair()
    tokens = torch.randint(0, 97, (9,))
    with torch.no_grad():
        torch.testing.assert_close(
            fused(tokens, torch.arange(9)), model(tokens, torch.arange(9)), atol=1e-5, rtol=1e-5
        )


def test_fused_decode_matches_with_a_cache():
    model, fused = pair()
    tokens = torch.randint(0, 97, (7,))
    outputs = []
    with torch.no_grad():
        for m in (model, fused):
            cache = ContiguousCache(CONFIG, 7)
            m(tokens[:6], torch.arange(6), cache)
            cache.advance(6)
            outputs.append(m(tokens[6:], torch.tensor([6]), cache))
    torch.testing.assert_close(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)


def test_fusing_removes_three_matmuls_per_layer():
    model, fused = pair()
    tokens = torch.randint(0, 97, (1,))
    with torch.no_grad():
        before = count_ops(lambda: model(tokens, torch.arange(1)))
        after = count_ops(lambda: fused(tokens, torch.arange(1)))
    matmuls = ("mm", "addmm")
    assert sum(before.ops[k] for k in matmuls) - sum(after.ops[k] for k in matmuls) == 3 * 2
