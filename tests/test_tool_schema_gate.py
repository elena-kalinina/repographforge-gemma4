#!/usr/bin/env python3
"""Offline regression test for the Phase 0.2 tool-schema training gate.

The gate is the only thing standing between a wrong COMPETITION_TOOLS transcription
and ~3 GPU hours producing a run that converges, completes, submits patches, and
scores zero. It therefore needs a test that does not require a GPU, the ADK, or Kaggle.

Run:  python3 tests/test_tool_schema_gate.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chat_format as cf


def _receipt(**over):
    base = {
        "chat_format_version": cf.CHAT_FORMAT_VERSION,
        "tool_schema_fingerprint": cf.tool_schema_fingerprint(),
        "tool_names": ["read_file"],
        "outcome": "clean",
        "f2_ok": True,
        "f2_preamble_tokens": 1830,
        "discovery": "test",
        "mismatches": [],
        "source_evidence": {},
        "written_at": "2026-11-10T00:00:00",
    }
    base.update(over)
    return base


def _write(tmp, data):
    p = Path(tmp) / "receipt.json"
    p.write_text(json.dumps(data))
    return p


def _expect_refusal(tmp, data, label, **kw):
    p = _write(tmp, data)
    try:
        cf.assert_tool_schema_verified(path=p, **kw)
    except RuntimeError as e:
        assert "TRAINING REFUSED" in str(e), str(e)
        print(f"✓ {label}")
        return
    raise AssertionError(f"gate did NOT refuse: {label}")


def main():
    os.environ.pop("TOOL_SCHEMA_ACK", None)
    with tempfile.TemporaryDirectory() as tmp:
        # 1. no receipt at all
        try:
            cf.assert_tool_schema_verified(path=Path(tmp) / "absent.json")
        except RuntimeError as e:
            assert "no verification receipt" in str(e)
            print("✓ [gate] missing receipt refuses training")
        else:
            raise AssertionError("missing receipt did not refuse")

        # 2. the happy path
        p = _write(tmp, _receipt())
        got = cf.assert_tool_schema_verified(path=p)
        assert got["outcome"] == "clean"
        print("✓ [gate] clean, current receipt passes")

        # 3. receipt written against a different chat_format version
        _expect_refusal(
            tmp, _receipt(chat_format_version="2026-01-01.ancient"),
            "[gate] stale chat_format version refuses training",
        )

        # 4. COMPETITION_TOOLS edited after verification
        _expect_refusal(
            tmp, _receipt(tool_schema_fingerprint="deadbeefdeadbeef"),
            "[gate] changed tool schemas invalidate the receipt",
        )

        # 5. Cell 0C found real mismatches
        _expect_refusal(
            tmp, _receipt(outcome="mismatch", mismatches=["PARAMS edit_file: ..."]),
            "[gate] recorded mismatch refuses training",
        )

        # 6. F2 live
        _expect_refusal(
            tmp, _receipt(f2_ok=False),
            "[gate] failed F2 probe refuses training",
        )

        # 7. inconclusive needs an explicit acknowledgement
        _expect_refusal(
            tmp, _receipt(outcome="inconclusive"),
            "[gate] inconclusive refuses training by default",
        )
        p = _write(tmp, _receipt(outcome="inconclusive"))
        got = cf.assert_tool_schema_verified(path=p, allow_inconclusive=True)
        assert got["outcome"] == "inconclusive"
        print("✓ [gate] inconclusive passes only with explicit acknowledgement")

        os.environ["TOOL_SCHEMA_ACK"] = "1"
        try:
            got = cf.assert_tool_schema_verified(path=p)
            assert got["outcome"] == "inconclusive"
            print("✓ [gate] TOOL_SCHEMA_ACK=1 env var is honoured")
        finally:
            os.environ.pop("TOOL_SCHEMA_ACK", None)

        # 8. an acknowledgement must NOT rescue a real mismatch
        _expect_refusal(
            tmp, _receipt(outcome="mismatch", mismatches=["x"]),
            "[gate] acknowledgement cannot override a real mismatch",
            allow_inconclusive=True,
        )

        # 9. round-trip through the writer Cell 0C actually calls
        out = cf.write_tool_schema_receipt(
            "clean", f2_ok=True, f2_preamble_tokens=1830,
            discovery="unit-test", path=Path(tmp) / "rt.json",
        )
        rt = cf.assert_tool_schema_verified(path=out)
        assert rt["tool_schema_fingerprint"] == cf.tool_schema_fingerprint()
        assert rt["tool_names"][0] == "get_code_subgraph", rt["tool_names"][:1]
        print("✓ [gate] write_tool_schema_receipt round-trips through the gate")

        # 10. the writer rejects a bogus outcome
        try:
            cf.write_tool_schema_receipt("probably_fine", f2_ok=True, path=Path(tmp) / "b.json")
        except ValueError:
            print("✓ [gate] writer rejects an unknown outcome")
        else:
            raise AssertionError("writer accepted an unknown outcome")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
