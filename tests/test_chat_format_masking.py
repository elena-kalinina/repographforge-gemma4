#!/usr/bin/env python3
"""
Offline regression test for the F1 (stop-token starvation) fix in `src/chat_format.py`.

Uses a mock tokenizer that reproduces the REAL Gemma 4 template structure measured in
Phase 0.1, so the masking logic can be validated without the 31B model or a GPU:

    <bos><|turn>system\nSYS<turn|>\n<|turn>user\nUSR<turn|>\n<|turn>model\n
    <|tool_call>call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|>
    <|tool_response>response:read_file{value:<|"|>OBS<|"|>}<tool_response|>THINK<turn|>\nDONE<turn|>\n

Three properties that must hold, each corresponding to a failure in the taxonomy:
  1. The turn-terminating token `<turn|>` IS supervised on model turns       (F1, part 2)
  2. System and user turns are NOT supervised                                (F1, part 1)
  3. Folded `<|tool_response>` observations are NOT supervised               (F1, part 1)

Run:  python3 -I tests/test_chat_format_masking.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from chat_format import _mask_rendered_token_ids  # noqa: E402

SPECIALS = [
    "<bos>", "<eos>", "<unk>", "<pad>",
    "<|turn>", "<turn|>",
    "<|tool_call>", "<tool_call|>",
    "<|tool_response>", "<tool_response|>",
    '<|"|>',
]
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


class MockGemma4Tokenizer:
    """Minimal tokenizer reproducing the measured Gemma 4 special-token inventory."""

    def __init__(self):
        self._vocab = {}
        self._inv = {}
        for t in SPECIALS:
            self._add(t)
        for t in ["system", "user", "model", "\n", "SYS", "USR", "THINK", "DONE",
                  "call", ":", "read", "_", "file", "{", "}", "filepath", "a", ".", "py",
                  "response", "value", "OBS", "status", "ok", "File"]:
            self._add(t)

    def _add(self, tok):
        if tok not in self._vocab:
            i = len(self._vocab)
            self._vocab[tok] = i
            self._inv[i] = tok
        return self._vocab[tok]

    @property
    def unk_token_id(self):
        return self._vocab["<unk>"]

    @property
    def bos_token_id(self):
        return self._vocab["<bos>"]

    @property
    def eos_token_id(self):
        return self._vocab["<eos>"]

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for chunk in _SPECIAL_RE.split(text):
            if not chunk:
                continue
            if chunk in self._vocab and chunk in SPECIALS:
                ids.append(self._vocab[chunk])
            else:
                for piece in re.findall(r"\n|[A-Za-z]+|\d+|.", chunk):
                    ids.append(self._add(piece))
        return ids

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            t = self._inv.get(i, "<unk>")
            if skip_special_tokens and t in SPECIALS:
                continue
            out.append(t)
        return "".join(out)


# The exact string measured in Phase 0.1.
RENDERED = (
    '<bos><|turn>system\nSYS<turn|>\n'
    '<|turn>user\nUSR<turn|>\n'
    '<|turn>model\n'
    '<|tool_call>call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|>'
    '<|tool_response>response:read_file{value:<|"|>OBS<|"|>}<tool_response|>'
    'THINK<turn|>\nDONE<turn|>\n'
)


def main():
    tok = MockGemma4Tokenizer()
    ids = tok.encode(RENDERED)
    labels = _mask_rendered_token_ids(ids, tok)

    supervised = [i for i, l in zip(ids, labels) if l != -100]
    text = tok.decode(supervised)
    eot = tok.convert_tokens_to_ids("<turn|>")

    print(f"tokens={len(ids)} supervised={len(supervised)} ({len(supervised)/len(ids):.1%})")
    print(f"supervised text: {text!r}\n")

    failures = []

    # (1) F1 part 2 — the stop token must carry gradient.
    if eot not in supervised:
        failures.append("FAIL [F1a]: <turn|> stop token is NOT supervised")
    else:
        print("✓ [F1a] <turn|> stop token IS supervised")

    # (2) F1 part 1 — user/system content must not.
    for marker in ("SYS", "USR"):
        if marker in text:
            failures.append(f"FAIL [F1b]: system/user content {marker!r} is supervised")
    if not any("F1b" in f for f in failures):
        print("✓ [F1b] system and user turns are NOT supervised")

    # (3) F1 part 1 — folded tool observations must not.
    if "OBS" in text or "<|tool_response>" in text:
        failures.append("FAIL [F1c]: folded tool observation is supervised")
    else:
        print("✓ [F1c] folded <|tool_response> observation is NOT supervised")

    # (4) The assistant's own content and its tool call must carry gradient.
    for marker in ("THINK", "DONE"):
        if marker not in text:
            failures.append(f"FAIL [F1d]: assistant content {marker!r} is NOT supervised")
    if "<|tool_call>" not in text:
        failures.append("FAIL [F1d]: the tool call itself is NOT supervised")
    if not any("F1d" in f for f in failures):
        print("✓ [F1d] assistant content and tool call ARE supervised")

    # (5) Supervised fraction must land in the gate's accepted band.
    frac = len(supervised) / len(ids)
    if not (0.03 < frac < 0.45):
        failures.append(f"FAIL [gate]: supervised fraction {frac:.1%} outside 3-45% band")
    else:
        print(f"✓ [gate] supervised fraction {frac:.1%} within band")

    print()
    if failures:
        for f in failures:
            print(f)
        print(f"\n{len(failures)} CHECK(S) FAILED")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
