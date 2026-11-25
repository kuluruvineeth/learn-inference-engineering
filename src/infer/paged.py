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


@dataclass
class Sequence:
    blocks: list[int] = field(default_factory=list)
    length: int = 0


@dataclass
class Group:
    rows: torch.Tensor
    tables: torch.Tensor
    mask: torch.Tensor


@dataclass
class Batch:
    entries: list[tuple[int, int]]
    positions: torch.Tensor
    slots: torch.Tensor
    tables: torch.Tensor
    groups: list[Group]
    last: torch.Tensor


class PagedCache:
    def __init__(
        self,
        config: Config,
        num_blocks: int,
        block_size: int = 16,
        dtype=torch.float32,
        device="cpu",
        pool: BlockPool | None = None,
    ):
        shape = (num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        layers = range(config.num_hidden_layers)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in layers]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in layers]
        self.block_size = block_size
        self.heads_per_kv = config.num_attention_heads // config.num_key_value_heads
        self.device = device
        self.pool = pool or BlockPool(num_blocks)
        self.sequences: dict[int, Sequence] = {}
        self.batch: Batch | None = None

    def blocks_needed(self, seq_id: int, tokens: int) -> int:
        seq = self.sequences.get(seq_id, Sequence())
        return max(0, blocks_for(seq.length + tokens, self.block_size) - len(seq.blocks))

    def reserve(self, seq_id: int, tokens: int) -> bool:
        blocks = self.pool.allocate(self.blocks_needed(seq_id, tokens))
        if blocks is None:
            return False
        self.sequences.setdefault(seq_id, Sequence()).blocks.extend(blocks)
        return True

    def release(self, seq_id: int) -> None:
        self.pool.release(reversed(self.sequences.pop(seq_id).blocks))

    def begin(self, entries: list[tuple[int, int]]) -> Batch:
        sequences, counts = [self.sequences[s] for s, _ in entries], [n for _, n in entries]
        for seq_id, count in entries:
            seq = self.sequences[seq_id]
            if seq.length + count > len(seq.blocks) * self.block_size:
                raise ValueError(f"sequence {seq_id} has no slots for {count} tokens")
        size, width = self.block_size, max(len(s.blocks) for s in sequences)
        arange = partial(torch.arange, device=self.device)
        tensor = partial(torch.tensor, device=self.device)
        tables = tensor([s.blocks + [NULL_BLOCK] * (width - len(s.blocks)) for s in sequences])
        starts, count_t = tensor([s.length for s in sequences]), tensor(counts)
        owner = torch.repeat_interleave(arange(len(entries)), count_t)
        ends = count_t.cumsum(0)
        offsets = arange(sum(counts)) - (ends - count_t)[owner]
        positions = starts[owner] + offsets
        groups = []
        for count in dict.fromkeys(counts):
            members = [i for i, n in enumerate(counts) if n == count]
            used = blocks_for(max(sequences[i].length for i in members) + count, size)
            rows = (ends - count)[members, None] + arange(count)
            visible = arange(used * size) <= positions[rows].repeat(1, self.heads_per_kv)[..., None]
            groups.append(Group(rows, tables[members, :used], visible[:, None]))
        self.batch = Batch(
            entries=list(entries),
            positions=positions,
            slots=tables[owner, positions // size] * size + positions % size,
            tables=tables,
            groups=groups,
            last=ends - 1,
        )
        return self.batch
