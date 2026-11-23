import torch

from infer.measure import count_ops, median_time


def test_views_and_no_op_casts_launch_nothing():
    x = torch.ones(4, 4)
    counted = count_ops(lambda: (x.view(16).float() + 1).view(4, 4).t().contiguous())
    assert dict(counted.ops) == {"add": 1, "clone": 1}


def test_in_place_ops_count():
    x = torch.zeros(3)
    assert count_ops(lambda: x.add_(1).mul_(2)).total == 2


def test_median_time_runs_warmup_and_repeats():
    calls = []
    assert median_time(lambda: calls.append(1), "cpu", repeats=5, warmup=2) >= 0
    assert len(calls) == 7
