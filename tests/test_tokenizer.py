from transformers import AutoTokenizer

from infer.hf import QWEN, download
from infer.tokenizer import Tokenizer, chat
from infer.workload import TOOLS, session, turns


def test_chat_matches_official_template():
    reference = AutoTokenizer.from_pretrained(QWEN)
    for turn in range(turns()):
        messages = session(turn)
        expected = reference.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
        )
        assert chat(messages, TOOLS) == expected


def test_encode_round_trip():
    tokenizer = Tokenizer(download(weights=False))
    text = chat(session(0), TOOLS)
    assert tokenizer.decode(tokenizer.encode(text)) == text
