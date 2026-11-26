from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

from infer.generate import Result
from infer.paged import PagedCache


class State(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHING = "finishing"
    FINISHED = "finished"


@dataclass(eq=False)
class Request:
    id: int
    prompt: list[int]
    max_tokens: int
    stop_ids: frozenset[int] = frozenset()
    tokens: list[int] = field(default_factory=list)
    computed: int = 0
    pending: int = 0
    in_flight: int = 0
    state: State = State.WAITING
    admitted: int = 0
    preemptions: int = 0
    recomputed: int = 0
    wasted: int = 0
    arrival: float = 0.0
    times: list[float] = field(default_factory=list)
    future: tuple | None = None

    def __post_init__(self):
        self.tokens = self.tokens or list(self.prompt)

    @property
    def outputs(self) -> list[int]:
        return self.tokens[len(self.prompt) :]

    @property
    def length(self) -> int:
        return len(self.tokens) + self.pending

    @property
    def work(self) -> int:
        return self.length - self.computed

    @property
    def done_by_length(self) -> bool:
        return self.length - len(self.prompt) >= self.max_tokens

    def result(self) -> Result:
        gaps = [b - a for a, b in pairwise(self.times)]
        return Result(self.outputs, self.times[0] - self.arrival if self.times else 0.0, gaps)
