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


@dataclass
class Entry:
    request: Request
    start: int
    count: int
    sample: bool


Policy = Callable[[list[Request]], Request]


def newest(candidates: list[Request]) -> Request:
    return max(candidates, key=lambda r: r.admitted)


def fewest_output(candidates: list[Request]) -> Request:
    return min(candidates, key=lambda r: (len(r.outputs), -len(r.prompt)))


def most_output(candidates: list[Request]) -> Request:
    return max(candidates, key=lambda r: len(r.outputs))


class Scheduler:
    def __init__(
        self,
        cache: PagedCache,
        budget: int = 2048,
        max_running: int = 64,
        policy: Policy = newest,
    ):
        self.cache, self.budget, self.max_running, self.policy = cache, budget, max_running, policy
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.admissions = self.preemptions = 0
        self.plan: list[Entry] = []
        self.left = budget

    @property
    def busy(self) -> bool:
        return bool(self.waiting or self.running)

    def add(self, request: Request) -> None:
        capacity = (self.cache.pool.num_blocks - 1) * self.cache.block_size
        if len(request.prompt) + request.max_tokens > capacity:
            raise ValueError(f"request {request.id} can never fit in {capacity} slots")
        self.waiting.append(request)

    def schedule(self) -> list[Entry]:
        self.plan, self.left, preemptions = [], self.budget, self.preemptions
        for request in sorted(self.running, key=self.priority):
            if self.left == 0:
                break
            if request.state is not State.RUNNING or request.done_by_length:
                continue
            count = min(request.work, self.left)
            if self.grow(request, count):
                self.take(request, count)
            elif request.state is State.WAITING:
                break
        while self.left and self.waiting and self.preemptions == preemptions:
            request = self.waiting[0]
            if len(self.running) == self.max_running or not self.admit(request):
                break
            self.waiting.popleft()
            self.running.append(request)
            self.admissions += 1
            request.state, request.admitted = State.RUNNING, self.admissions
            count = min(request.work, self.left)
            if not self.allocate(request, count):
                raise RuntimeError(f"admitted request {request.id} has no blocks")
            self.take(request, count)
        plan, self.plan = self.plan, []
        return plan

    def priority(self, request: Request) -> bool:
        return request.work > 1

    def admit(self, request: Request) -> bool:
        return self.cache.blocks_needed(request.id, request.work) <= self.cache.pool.num_free

    def allocate(self, request: Request, count: int) -> bool:
        return self.cache.reserve(request.id, count)

    def release(self, request: Request) -> None:
        self.cache.release(request.id)

    def grow(self, request: Request, count: int) -> bool:
        while not self.allocate(request, count):
            planned = {entry.request.id for entry in self.plan}
            quiet = [r for r in self.running if r.in_flight == int(r.id in planned)]
            if not quiet:
                return False
            victim = self.policy(quiet)
            self.preempt(victim)
            if victim is request:
                return False
        return True

    def preempt(self, victim: Request) -> None:
        for entry in [e for e in self.plan if e.request is victim]:
            self.plan.remove(entry)
            self.left += entry.count
            victim.computed -= entry.count
            victim.in_flight -= 1
            victim.pending -= entry.sample
        self.release(victim)
        self.running.remove(victim)
        self.waiting.appendleft(victim)
        self.preemptions += 1
        victim.preemptions += 1
        victim.recomputed += victim.computed
        victim.computed, victim.state = 0, State.WAITING

    def take(self, request: Request, count: int) -> None:
        entry = Entry(request, request.computed, count, request.computed + count == request.length)
        self.plan.append(entry)
        self.left -= count
        request.computed += count
        request.in_flight += 1
        request.pending += entry.sample

    def update(self, plan: list[Entry], sampled: list[int], now: float = 0.0) -> None:
        tokens = iter(sampled)
        for entry in plan:
            request = entry.request
            request.in_flight -= 1
            if entry.sample:
                request.pending -= 1
                self.accept(request, next(tokens), now)
            if request.state is State.FINISHING and request.in_flight == 0:
                self.retire(request)

    def accept(self, request: Request, token: int, now: float) -> None:
        if request.state is not State.RUNNING:
            request.wasted += 1
            return
        request.tokens.append(token)
        request.times.append(now)
        if token in request.stop_ids or len(request.outputs) == request.max_tokens:
            self.running.remove(request)
            request.state = State.FINISHING

    def retire(self, request: Request) -> None:
        self.release(request)
        self.finished.append(request)
        request.state = State.FINISHED


class StaticScheduler(Scheduler):
    def schedule(self) -> list[Entry]:
        self.forming = not self.running
        return super().schedule()

    def admit(self, request: Request) -> bool:
        return self.forming and super().admit(request)
