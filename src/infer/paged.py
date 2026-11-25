from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F

from infer.generate import Result, _loop
from infer.model import Config, Model
from infer.sampling import SamplingParams

NULL_BLOCK = 0


def blocks_for(tokens: int, block_size: int) -> int:
    return -(-tokens // block_size)


class BlockPool:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.refs = [0] * num_blocks
        self.free = OrderedDict.fromkeys(range(1, num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free)

    @property
    def usage(self) -> float:
        return 1 - len(self.free) / (self.num_blocks - 1)

    def allocate(self, count: int) -> list[int] | None:
        if count > len(self.free):
            return None
        return [self._take() for _ in range(count)]

    def release(self, blocks) -> None:
        for block in blocks:
            if self.refs[block] == 0:
                raise ValueError(f"block {block} is not allocated")
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self._give_back(block)

    def _take(self) -> int:
        block, _ = self.free.popitem(last=False)
        self.refs[block] = 1
        return block

    def _give_back(self, block: int) -> None:
        self.free[block] = None
        self.free.move_to_end(block, last=False)
