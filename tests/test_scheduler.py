import pytest
import torch

from infer.engine import Engine
from infer.model import Config, Model
from infer.paged import PagedCache, generate_paged
from infer.sampling import SamplingParams
from infer.scheduler import (
    Request,
    Scheduler,
    State,
    StaticScheduler,
    fewest_output,
    most_output,
    newest,
)

CONFIG = Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2)
LENGTHS = [(13, 9), (5, 3), (22, 12), (9, 6), (17, 2), (30, 8)]


def tiny() -> Model:
    torch.manual_seed(0)
    return Model(CONFIG).eval()


def prompts() -> list[list[int]]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(0, 97, (n,), generator=generator).tolist() for n, _ in LENGTHS]


def reference(model: Model, stop_ids=frozenset()) -> list[list[int]]:
    return [
        generate_paged(model, prompt, SamplingParams(max_tokens=m), stop_ids).tokens
        for prompt, (_, m) in zip(prompts(), LENGTHS, strict=True)
    ]


def requests(stop_ids=frozenset()) -> list[Request]:
    pairs = zip(prompts(), LENGTHS, strict=True)
    return [Request(i, p, m, frozenset(stop_ids)) for i, (p, (_, m)) in enumerate(pairs)]


def serve(model, scheduler, depth=0):
    for request in requests():
        scheduler.add(request)
    engine, plans = Engine(model, scheduler, depth), []
    while engine.busy:
        plans.append(engine.step())
    return sorted(scheduler.finished, key=lambda r: r.id), plans


def scheduler(blocks=60, block_size=4, **kwargs) -> Scheduler:
    return Scheduler(PagedCache(CONFIG, blocks, block_size), **kwargs)


def test_continuous_batching_matches_one_request_at_a_time():
    model = tiny()
    finished, plans = serve(model, scheduler(max_running=3))
    assert [r.outputs for r in finished] == reference(model)
    assert max(len(plan) for plan in plans) == 3


def test_requests_join_and_leave_between_steps():
    model = tiny()
    finished, plans = serve(model, scheduler(max_running=2))
    assert [r.outputs for r in finished] == reference(model)
    assert any({0, 2} <= {e.request.id for e in plan} for plan in plans)


def test_static_batch_waits_for_its_longest_request():
    model, sched = tiny(), StaticScheduler(PagedCache(CONFIG, 60, 4), max_running=3)
    finished, plans = serve(model, sched)
    assert [r.outputs for r in finished] == reference(model)
    batches = [{e.request.id for e in plan} for plan in plans]
    first = next(i for i, ids in enumerate(batches) if ids & {3, 4, 5})
    assert all(ids <= {0, 1, 2} for ids in batches[:first])
    assert batches[first] == {3, 4, 5} and len(plans) == 12 + 8


def test_budget_caps_every_step_and_decodes_go_first():
    model = tiny()
    finished, plans = serve(model, scheduler(budget=7, max_running=4))
    assert [r.outputs for r in finished] == reference(model)
    for plan in plans:
        assert sum(e.count for e in plan) <= 7
        prefills = [e.start < len(e.request.prompt) for e in plan]
        assert prefills == sorted(prefills)
        assert sum(not e.sample for e in plan) <= 1
    assert max(sum(not e.sample for e in plan) for plan in plans) == 1


def test_chunked_prefill_splits_a_long_prompt():
    model = tiny()
    finished, plans = serve(model, scheduler(budget=4, max_running=1))
    longest = finished[5]
    chunks = [e.count for plan in plans for e in plan if e.request is longest][:8]
    assert chunks == [4, 4, 4, 4, 4, 4, 4, 2]
    assert [r.outputs for r in finished] == reference(model)


def test_admission_needs_the_whole_sequence_to_fit():
    cache = PagedCache(CONFIG, 10, 4)
    sched = Scheduler(cache, budget=4)
    assert cache.reserve(99, 20)
    sched.add(Request(1, list(range(20)), 1))
    sched.add(Request(2, [1, 2], 1))
    assert sched.schedule() == [] and cache.pool.num_free == 4
    cache.release(99)
    plan = sched.schedule()
    assert [(e.request.id, e.count) for e in plan] == [(1, 4)]


def test_rejects_a_request_that_can_never_fit():
    sched = scheduler(blocks=5)
    with pytest.raises(ValueError):
        sched.add(Request(1, list(range(10)), 7))


def test_victim_policies():
    def made(i, prompt, outputs):
        request = Request(i, list(range(prompt)), 50, admitted=i)
        request.tokens += [0] * outputs
        return request

    candidates = [made(1, 8, 5), made(2, 30, 2), made(3, 10, 9), made(4, 4, 2)]
    assert newest(candidates).id == 4
    assert fewest_output(candidates).id == 2
    assert most_output(candidates).id == 3


@pytest.mark.parametrize("policy", [newest, fewest_output, most_output])
def test_preemption_by_recompute_keeps_the_tokens(policy):
    model = tiny()
    sched = scheduler(blocks=15, budget=64, policy=policy)
    finished, _ = serve(model, sched)
    assert [r.outputs for r in finished] == reference(model)
    assert sched.preemptions > 0 and sum(r.recomputed for r in finished) > 0
    assert sched.cache.pool.num_free == 14 and not sched.cache.sequences


def test_a_preempted_request_goes_to_the_front_and_restarts_from_zero():
    sched = scheduler()
    first, second = Request(1, [1, 2, 3], 4), Request(2, [4, 5], 4)
    sched.add(first)
    sched.add(second)
    sched.update(sched.schedule(), [7, 8])
    sched.preempt(second)
    assert sched.waiting[0] is second and sched.running == [first]
    assert (second.computed, second.recomputed, second.work) == (0, 2, 3)
    assert second.state is State.WAITING and 2 not in sched.cache.sequences


def test_in_flight_depth_keeps_the_tokens_and_wastes_one_per_step_of_depth():
    model = tiny()
    stops = {r[-1] for r in reference(model)[::2]}
    ref = reference(model, stops)
    results = {}
    for depth in (0, 1, 2):
        sched = Scheduler(PagedCache(CONFIG, 60, 4), max_running=4)
        for request in requests(stops):
            sched.add(request)
        engine = Engine(model, sched, depth)
        engine.run()
        results[depth] = sorted(sched.finished, key=lambda r: r.id)
        assert sched.cache.pool.num_free == 59
    early = [len(r.outputs) < r.max_tokens for r in results[0]]
    assert any(early) and not all(early)
    for depth, finished in results.items():
        assert [r.outputs for r in finished] == ref
        assert [r.wasted for r in finished] == [
            min(depth, r.max_tokens - len(r.outputs)) for r in finished
        ]


def test_a_victim_planned_this_step_is_unplanned_and_its_budget_returned():
    sched = Scheduler(PagedCache(CONFIG, 5, 4), budget=5, policy=most_output)
    decoder, prefill = Request(1, [1, 2, 3, 4], 8), Request(2, list(range(12)), 2)
    sched.add(decoder)
    sched.add(prefill)
    for _ in range(2):
        plan = sched.schedule()
        sched.cache.begin([(e.request.id, e.count) for e in plan])
        sched.cache.commit()
        sched.update(plan, [7] * sum(e.sample for e in plan))
    plan = sched.schedule()
    assert [(e.request.id, e.count) for e in plan] == [(2, 4)]
    assert sched.waiting[0] is decoder and decoder.recomputed == 5 and decoder.in_flight == 0
    assert sched.left == 1
