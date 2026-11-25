import pytest
import torch

from infer.generate import generate
from infer.kv import ContiguousCache
from infer.kv_size import (
    bytes_per_token,
    cache_bytes_per_slot,
    fit_contiguous,
    fit_paged,
    latent_bytes_per_token,
    per_head_bytes_per_token,
)
from infer.model import Config, Model
from infer.paged import NULL_BLOCK, BlockPool, PagedCache, generate_paged
from infer.sampling import SamplingParams

CONFIG = Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2)
STEPS = [{1: 11, 2: 5}, {3: 9}, {1: 1, 3: 8, 2: 1}, {2: 4}, {1: 1, 2: 1, 3: 1}]
LENGTHS = {s: sum(step.get(s, 0) for step in STEPS) for s in (1, 2, 3)}


def tiny() -> Model:
    torch.manual_seed(0)
    return Model(CONFIG).eval()


def interleaved(block_size: int = 4, num_blocks: int = 40):
    model, cache = tiny(), PagedCache(CONFIG, num_blocks, block_size)
    tokens = {s: torch.randint(0, 97, (n,)) for s, n in LENGTHS.items()}
    done = dict.fromkeys(LENGTHS, 0)
    with torch.no_grad():
        for step in STEPS:
            for seq_id, count in step.items():
                assert cache.reserve(seq_id, count)
            batch = cache.begin(list(step.items()))
            flat = torch.cat([tokens[s][done[s] : done[s] + n] for s, n in step.items()])
            logits = model(flat, batch.positions, cache)
            cache.commit()
            start = 0
            for seq_id, count in step.items():
                yield model, seq_id, tokens[seq_id], done[seq_id], logits[start : start + count]
                start, done[seq_id] = start + count, done[seq_id] + count


def test_pool_never_hands_out_null_block():
    pool = BlockPool(8)
    blocks = pool.allocate(7)
    assert NULL_BLOCK not in blocks and sorted(blocks) == list(range(1, 8))
    assert pool.allocate(1) is None and pool.usage == 1.0


def test_pool_allocation_is_all_or_nothing():
    pool = BlockPool(6)
    pool.allocate(2)
    assert pool.allocate(4) is None and pool.num_free == 3


def test_pool_release_returns_blocks_and_rejects_double_free():
    pool = BlockPool(6)
    blocks = pool.allocate(5)
    pool.release(reversed(blocks))
    assert pool.num_free == 5 and pool.allocate(5) == blocks
    pool.release(blocks)
    with pytest.raises(ValueError):
        pool.release(blocks[:1])
    with pytest.raises(ValueError):
        pool.release([NULL_BLOCK])


def test_reserve_grows_one_block_at_a_time():
    cache = PagedCache(CONFIG, 10, 4)
    assert cache.reserve(7, 9) and len(cache.sequences[7].blocks) == 3
    cache.sequences[7].length = 9
    assert cache.reserve(7, 3) and len(cache.sequences[7].blocks) == 3
    assert cache.reserve(7, 4) and len(cache.sequences[7].blocks) == 4
    assert not cache.reserve(8, 24) and cache.pool.num_free == 5
    cache.release(7)
    assert cache.pool.num_free == 9 and 7 not in cache.sequences


def test_slot_mapping():
    cache = PagedCache(CONFIG, 10, 4)
    cache.reserve(1, 6)
    cache.reserve(2, 3)
    cache.sequences[1].length = 6
    cache.reserve(1, 3)
    table = cache.sequences[1].blocks
    batch = cache.begin([(1, 3), (2, 3)])
    expected = [table[p // 4] * 4 + p % 4 for p in (6, 7, 8)]
    expected += [cache.sequences[2].blocks[0] * 4 + p for p in (0, 1, 2)]
    assert batch.slots.tolist() == expected
    assert batch.positions.tolist() == [6, 7, 8, 0, 1, 2]
    assert batch.tables[1].tolist() == cache.sequences[2].blocks + [NULL_BLOCK] * 2


def test_attend_writes_new_kv_into_its_slots():
    cache = PagedCache(CONFIG, 10, 4)
    cache.reserve(1, 6)
    batch = cache.begin([(1, 6)])
    q, k, v = torch.randn(6, 4, 16), torch.randn(6, 2, 16), torch.randn(6, 2, 16)
    cache.attend(1, q, k, v)
    torch.testing.assert_close(cache.k[1].view(-1, 2, 16)[batch.slots], k)
    torch.testing.assert_close(cache.v[1].view(-1, 2, 16)[batch.slots], v)


def test_paged_batch_matches_no_cache():
    for model, _, tokens, done, logits in interleaved():
        count = logits.size(0)
        with torch.no_grad():
            full = model(tokens[: done + count], torch.arange(done + count))[done:]
        torch.testing.assert_close(logits, full, atol=1e-5, rtol=1e-5)


def test_paged_batch_matches_contiguous_cache():
    caches = {s: ContiguousCache(CONFIG, n) for s, n in LENGTHS.items()}
    for model, seq_id, tokens, done, logits in interleaved(block_size=3):
        count = logits.size(0)
        with torch.no_grad():
            ours = model(
                tokens[done : done + count], torch.arange(done, done + count), caches[seq_id]
            )
        caches[seq_id].advance(count)
        torch.testing.assert_close(logits, ours, atol=1e-5, rtol=1e-5)


def test_mixed_step_matches_each_sequence_alone():
    model, cache = tiny(), PagedCache(CONFIG, 40, 4)
    history = {seq_id: torch.randint(0, 97, (n,)) for seq_id, n in ((1, 9), (2, 2), (3, 14))}
    step = [(1, 1), (4, 11), (2, 1), (3, 1)]
    new = {seq_id: torch.randint(0, 97, (count,)) for seq_id, count in step}
    with torch.no_grad():
        for seq_id, tokens in history.items():
            cache.reserve(seq_id, tokens.numel())
            batch = cache.begin([(seq_id, tokens.numel())])
            model(tokens, batch.positions, cache)
            cache.commit()
        for seq_id, count in step:
            assert cache.reserve(seq_id, count)
        batch = cache.begin(step)
        logits = model(torch.cat([new[seq_id] for seq_id, _ in step]), batch.positions, cache)
        alone = []
        for seq_id, count in step:
            tokens = torch.cat([history.get(seq_id, new[seq_id][:0]), new[seq_id]])
            alone.append(model(tokens, torch.arange(tokens.numel()))[-count:])
    assert sorted(tuple(group.rows.shape) for group in batch.groups) == [(1, 11), (3, 1)]
    torch.testing.assert_close(logits, torch.cat(alone), atol=1e-5, rtol=1e-5)


def test_freed_blocks_are_reused_without_leaking_old_kv():
    model, cache = tiny(), PagedCache(CONFIG, 5, 4)
    old, new = torch.randint(0, 97, (16,)), torch.randint(0, 97, (7,))
    with torch.no_grad():
        for seq_id, tokens in ((1, old), (2, new)):
            assert cache.reserve(seq_id, tokens.numel())
            batch = cache.begin([(seq_id, tokens.numel())])
            logits = model(tokens, batch.positions, cache)
            cache.commit()
            cache.release(seq_id)
        torch.testing.assert_close(logits, model(new, torch.arange(7)), atol=1e-5, rtol=1e-5)


def test_generate_paged_matches_generate_greedy():
    model, prompt = tiny(), [3, 14, 15, 92, 65]
    params = SamplingParams(max_tokens=40)
    assert generate_paged(model, prompt, params).tokens == generate(model, prompt, params).tokens


def test_generate_paged_matches_generate_sampled():
    model, prompt = tiny(), [3, 14, 15, 92, 65]
    params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=40, seed=7)
    cache = PagedCache(CONFIG, 12, 4)
    assert generate_paged(model, prompt, params, cache=cache).tokens == (
        generate(model, prompt, params).tokens
    )
    assert cache.pool.num_free == 11 and not cache.sequences


def test_bytes_per_token_matches_allocated_storage():
    for dtype in (torch.float32, torch.bfloat16):
        cache = PagedCache(CONFIG, 6, 4, dtype)
        assert cache_bytes_per_slot(cache) == bytes_per_token(CONFIG, dtype)
        assert bytes_per_token(CONFIG, dtype) == 2 * 2 * 16 * dtype.itemsize * 2


def test_deepseek_formulas():
    hf = {
        "num_attention_heads": 4,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 2,
        "v_head_dim": 8,
        "kv_lora_rank": 16,
        "num_hidden_layers": 3,
    }
    assert per_head_bytes_per_token(hf) == 4 * (8 + 2 + 8) * 2 * 3
    assert latent_bytes_per_token(hf) == (16 + 2) * 2 * 3


def test_fit_counts_reservations_and_blocks():
    requests = [(10, 3), (20, 1)]
    contiguous = fit_contiguous(requests, max_tokens=30, budget_tokens=100)
    assert (contiguous.sessions, contiguous.held_tokens, contiguous.live_tokens) == (2, 90, 34)
    paged = fit_paged(requests, block_size=8, budget_tokens=100)
    assert (paged.sessions, paged.held_tokens, paged.live_tokens) == (4, 80, 68)
    assert paged.wasted_tokens == 12
