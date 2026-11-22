import json

SYSTEM = """You are a coding agent working inside a Python repository. You fix bugs by reading
files, running the tests and editing code. Make the smallest change that makes the failing test
pass. Never edit a test to make it pass. After every edit, run the tests again. When all tests
pass, reply with a one-paragraph summary of the change.

The repository has two files.

durations.py
```python
import re

UNITS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
PATTERN = re.compile(r"(\\d+(?:\\.\\d+)?)(m|ms|s|h|d)")


def parse_duration(text: str) -> float:
    total = 0.0
    position = 0
    for match in PATTERN.finditer(text.replace(" ", "")):
        if match.start() != position:
            raise ValueError(f"unexpected text at {position}: {text!r}")
        value, unit = match.groups()
        total += float(value) * UNITS[unit]
        position = match.end()
    if position == 0:
        raise ValueError(f"no duration in {text!r}")
    return total


def format_duration(seconds: float) -> str:
    parts = []
    for unit in ("d", "h", "m", "s"):
        size = UNITS[unit]
        if seconds >= size:
            count = int(seconds // size)
            parts.append(f"{count}{unit}")
            seconds -= count * size
    if seconds:
        parts.append(f"{round(seconds * 1000)}ms")
    return "".join(parts) or "0s"
```

test_durations.py
```python
import pytest

from durations import format_duration, parse_duration


def test_parse_simple():
    assert parse_duration("90s") == 90


def test_parse_compound():
    assert parse_duration("1h 30m") == 5400


def test_parse_milliseconds():
    assert parse_duration("1500ms") == 1.5


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("ten minutes")


def test_round_trip():
    for text, expected in [("1d2h", "1d2h"), ("2h0m30s", "2h30s"), ("250ms", "250ms")]:
        assert format_duration(parse_duration(text)) == expected
```"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path from the root."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "Optional -k filter."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact snippet in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to replace."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]


def _call(name: str, **arguments) -> dict:
    payload = json.dumps({"name": name, "arguments": arguments})
    return {"role": "assistant", "content": f"<tool_call>\n{payload}\n</tool_call>"}


def _result(text: str) -> dict:
    return {"role": "user", "content": f"<tool_response>\n{text}\n</tool_response>"}


TEST_OUTPUT = "\n".join(
    [
        "FAILED test_durations.py::test_parse_milliseconds - AssertionError: assert 90000.0 == 1.5",
        "FAILED test_durations.py::test_round_trip - AssertionError: assert '4h10m' == '250ms'",
        "2 failed, 3 passed in 0.09s",
    ]
)

TURNS = [
    {"role": "user", "content": "Two tests fail. Please fix them."},
    _call("run_tests"),
    _result(TEST_OUTPUT),
    _call("read_file", path="durations.py"),
    _result("(same as the copy in your instructions)"),
    _call(
        "edit_file",
        path="durations.py",
        old="(m|ms|s|h|d)",
        new="(ms|s|m|h|d)",
    ),
    _result("edited durations.py"),
    _call("run_tests"),
    _result("5 passed in 0.01s"),
]


def session(turn: int) -> list[dict]:
    head = [{"role": "system", "content": SYSTEM}]
    taken = 1 + 2 * turn
    return head + TURNS[:taken]


def turns() -> int:
    return (len(TURNS) + 1) // 2
