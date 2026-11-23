from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from infer.kv import ContiguousCache
from infer.model import Config, Model
from infer.roofline import (
    Cost,
    deepseek_v3_decode,
    deepseek_v3_params,
    deepseek_v41_experts,
    dense_params,
    dense_pass,
    experts_touched,
    kv_bytes_per_token,
    load_json,
    matmul_values_per_row,
)

CONFIGS = Path(__file__).parent.parent / "configs"
TIED = Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2)
UNTIED = Config(
    vocab_size=97,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=3,
    tie_word_embeddings=False,
)


def counted_flops(model: Model, tokens: int, context: int) -> int:
    cache = ContiguousCache(model.config, context + tokens)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        if context:
            model(torch.arange(context), torch.arange(context), cache)
            cache.advance(context)
        with counter:
            model(torch.arange(tokens), torch.arange(context, context + tokens), cache)
    return counter.get_total_flops()


def test_parameter_count_matches_the_model():
    for config in (TIED, UNTIED):
        counted = sum(p.numel() for p in Model(config).parameters())
        assert dense_params(config).total == counted


def test_weight_bytes_are_every_parameter_read_once():
    for config in (TIED, UNTIED):
        model, rows = Model(config), 3
        table = 0 if config.tie_word_embeddings else model.embed_tokens.weight.numel()
        expected = (sum(p.numel() for p in model.parameters()) - table) * 4
        expected += 0 if config.tie_word_embeddings else rows * config.hidden_size * 4
        assert dense_pass(config, rows).weight_bytes == expected


def test_decode_flops_match_a_flop_counter():
    torch.manual_seed(0)
    model = Model(TIED).eval()
    for context in (0, 7, 40):
        assert counted_flops(model, 1, context) == dense_pass(TIED, 1, context).flops


def test_matmul_flops_scale_with_tokens():
    torch.manual_seed(0)
    model = Model(UNTIED).eval()
    rows = 5
    attention = 4 * UNTIED.num_hidden_layers * UNTIED.hidden_size * rows * rows
    assert counted_flops(model, rows, 0) - attention == 2 * dense_params(UNTIED).matmul * rows


def test_kv_bytes_match_the_cache():
    cache = ContiguousCache(TIED, 10)
    allocated = (cache.k.numel() + cache.v.numel()) * cache.k.element_size()
    assert kv_bytes_per_token(TIED) * 10 == allocated


def test_activation_values_are_matmul_inputs_and_outputs():
    model, seen = Model(TIED).eval(), []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(lambda m, i, o: seen.append(i[0].numel() + o.numel()))
    with torch.no_grad():
        model(torch.arange(4), torch.arange(4))
    assert sum(seen) == 4 * matmul_values_per_row(TIED)


def test_intensity_grows_with_tokens():
    values = [dense_pass(TIED, tokens).intensity for tokens in (1, 4, 16, 64)]
    assert values == sorted(values)


def test_time_is_the_slower_of_bytes_and_flops():
    cost = Cost(flops=1000.0, weight_bytes=100.0)
    assert cost.time(bandwidth=10.0, peak=1000.0) == 10.0
    assert cost.time(bandwidth=1000.0, peak=10.0) == 100.0


def v3_checkpoint_shapes(c: dict) -> list[tuple[int, ...]]:
    h, heads = c["hidden_size"], c["num_attention_heads"]
    nope, rope, v = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    q_rank, kv_rank = c["q_lora_rank"], c["kv_lora_rank"]
    shapes = [(c["vocab_size"], h), (h,), (c["vocab_size"], h)]
    for layer in range(c["num_hidden_layers"]):
        shapes += [(h,), (h,), (q_rank, h), (q_rank,), (heads * (nope + rope), q_rank)]
        shapes += [(kv_rank + rope, h), (kv_rank,), (heads * (nope + v), kv_rank), (h, heads * v)]
        if layer < c["first_k_dense_replace"]:
            shapes += [(c["intermediate_size"], h)] * 2 + [(h, c["intermediate_size"])]
            continue
        shapes += [(c["n_routed_experts"], h), (c["n_routed_experts"],)]
        experts = c["n_routed_experts"] + c["n_shared_experts"]
        moe = c["moe_intermediate_size"]
        shapes += ([(moe, h)] * 2 + [(h, moe)]) * experts
    return shapes


def test_v3_total_matches_its_checkpoint_shapes():
    config = load_json(CONFIGS / "deepseek-v3.json")
    total = sum(torch.Size(shape).numel() for shape in v3_checkpoint_shapes(config))
    assert deepseek_v3_params(config).total == total == 671_026_419_200


def test_v3_active_parameters():
    params = deepseek_v3_params(load_json(CONFIGS / "deepseek-v3.json"))
    assert params.active == 37_552_297_472
    assert params.active_matmul == 36_624_596_992


def test_v3_decode_reads_the_active_weights_at_batch_one():
    config = load_json(CONFIGS / "deepseek-v3.json")
    params = deepseek_v3_params(config)
    cost = deepseek_v3_decode(config, 4096, weight_bytes=1)
    assert round(cost.weight_bytes) == params.active_matmul
    assert cost.kv_bytes == 4096 * 576 * 61 * 2


def test_experts_touched_matches_sampling():
    torch.manual_seed(0)
    picks = torch.rand(500, 64, 256).argsort(-1)[..., :8]
    touched = torch.zeros(500, 256).scatter_(1, picks.flatten(1), 1).sum(1)
    assert abs(touched.mean().item() - experts_touched(256, 8, 64)) < 0.5
    assert experts_touched(256, 8, 1) == 8


def test_v41_expert_parameters():
    experts = deepseek_v41_experts(load_json(CONFIGS / "deepseek-v4.1-flash.json"))
    assert experts.total == 544_997_376_000
    assert experts.active == 9_909_043_200
