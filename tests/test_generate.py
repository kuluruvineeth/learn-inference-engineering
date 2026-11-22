import torch

from infer.generate import generate, generate_naive
from infer.kv import ContiguousCache
from infer.model import Config, Model
from infer.sampling import SamplingParams

CONFIG = Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2)


def tiny() -> Model:
    torch.manual_seed(0)
    return Model(CONFIG).eval()


def test_cache_prefill_matches_no_cache():
    model, tokens = tiny(), torch.randint(0, 97, (10,))
    cache = ContiguousCache(CONFIG, 10)
    with torch.no_grad():
        torch.testing.assert_close(
            model(tokens, torch.arange(10), cache), model(tokens, torch.arange(10))
        )


def test_cache_decode_matches_recompute():
    model, tokens = tiny(), torch.randint(0, 97, (10,))
    cache = ContiguousCache(CONFIG, 10)
    with torch.no_grad():
        model(tokens[:6], torch.arange(6), cache)
        cache.advance(6)
        step = model(tokens[6:7], torch.tensor([6]), cache)[-1]
        full = model(tokens[:7], torch.arange(7))[-1]
    torch.testing.assert_close(step, full, atol=1e-5, rtol=1e-5)


def test_generate_matches_naive_greedy():
    model, prompt = tiny(), [3, 14, 15, 92, 65]
    params = SamplingParams(max_tokens=12)
    assert generate(model, prompt, params).tokens == generate_naive(model, prompt, params).tokens


def test_generate_matches_naive_sampled():
    model, prompt = tiny(), [3, 14, 15, 92, 65]
    params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=12, seed=7)
    assert generate(model, prompt, params).tokens == generate_naive(model, prompt, params).tokens


def test_stops_on_stop_id():
    model, prompt = tiny(), [3, 14, 15]
    first = generate(model, prompt, SamplingParams(max_tokens=1)).tokens[0]
    assert generate(model, prompt, SamplingParams(max_tokens=9), {first}).tokens == [first]
