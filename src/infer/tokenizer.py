import json
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

TOOLS_HEADER = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    'within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, '
    '"arguments": <args-json-object>}\n</tool_call>'
)


def chat(messages: list[dict], tools: list[dict] | None = None) -> str:
    system, rest = messages[0]["content"], messages[1:]
    if tools:
        system += TOOLS_HEADER + "".join("\n" + json.dumps(t) for t in tools) + TOOLS_FOOTER
    text = f"<|im_start|>system\n{system}<|im_end|>\n"
    for message in rest:
        text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


class Tokenizer:
    def __init__(self, path: Path):
        self.inner = HFTokenizer.from_file(str(Path(path) / "tokenizer.json"))
        generation = json.loads((Path(path) / "generation_config.json").read_text())
        self.stop_ids = set(generation["eos_token_id"])

    def encode(self, text: str) -> list[int]:
        return self.inner.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.inner.decode(ids, skip_special_tokens=False)
